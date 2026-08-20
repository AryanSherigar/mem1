from __future__ import annotations

import os
import time
import asyncio
from datetime import datetime, timezone
from typing import Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel

from context_memory.engine import MemoryEngine
from context_memory.client.hydradb_http import HydraHttpTransport
from context_memory.core.config import Config
from context_memory.ingestion.embedding import SentenceTransformerEmbedder
from context_memory.ingestion.extraction import ExtractionService
from context_memory.ingestion.graph_plan_builder import GraphPlanBuilder
from context_memory.ingestion.graph_writer import GraphWriter
from context_memory.ingestion.fact_lookup import HydraFactLookup
from context_memory.ingestion.model_adapters import LLMEntityResolutionModel, LLMExtractor, LLMTemporalUpdateModel
from context_memory.ingestion.orchestrator import IngestionOrchestrator
from context_memory.ingestion.resolution import EntityRegistry, TemporalUpdateClassifier
from context_memory.persistence.postgres import (
    PostgresChunkStore,
    PostgresEmbeddingStore,
    PostgresExtractionStore,
    PostgresGraphManifestStore,
    PostgresJobStore,
    PostgresSearchIndexStore,
)
from context_memory.retrieval import HybridRetrievalEngine
from context_memory.hydration import HydrationManager

_GLOBAL_ENGINE: MemoryEngine | None = None


def get_engine() -> MemoryEngine:
    """Every tunable value here comes from one `Config` instance — see
    `core/config.py`. This function only wires objects together; it does not
    read `os.environ` itself."""
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is not None:
        return _GLOBAL_ENGINE

    config = Config()

    hydra_url = config.hydradb_url
    if not hydra_url.startswith("http://") and not hydra_url.startswith("https://"):
        hydra_url = f"http://{hydra_url}"

    import psycopg
    from pathlib import Path
    from context_memory.persistence.migrations import apply_migrations

    pg_conn = psycopg.connect(config.database_url, autocommit=True)
    migrations_dir = Path(__file__).resolve().parents[2] / "db" / "migrations"
    if migrations_dir.exists():
        apply_migrations(pg_conn, migrations_dir)

    hydra_transport = HydraHttpTransport(
        base_url=hydra_url,
        bearer_token=config.hydradb_token or None,
        database=config.hydradb_database,
        timeout_seconds=config.hydradb_request_timeout_seconds,
    )
    embedder = SentenceTransformerEmbedder(model_name=config.embedding_model_name)

    chunk_store = PostgresChunkStore(pg_conn)
    job_store = PostgresJobStore(pg_conn)
    manifest_store = PostgresGraphManifestStore(pg_conn)
    embedding_store = PostgresEmbeddingStore(pg_conn)
    search_index_store = PostgresSearchIndexStore(pg_conn)
    extraction_store = PostgresExtractionStore(pg_conn)

    extractor = LLMExtractor(config.get_extractor_client(), config)
    extraction_service = ExtractionService(extractor, extraction_store)
    entity_resolution_model = LLMEntityResolutionModel(config.get_entity_resolution_client(), config)
    entity_registry = EntityRegistry(allocator=chunk_store, model=entity_resolution_model)
    plan_builder = GraphPlanBuilder(allocator=chunk_store)
    graph_writer = GraphWriter(manifest_store=manifest_store, transport=hydra_transport)

    temporal_model = LLMTemporalUpdateModel(config.get_temporal_update_client(), config)
    update_classifier = TemporalUpdateClassifier(temporal_model)
    fact_lookup = HydraFactLookup(hydra_transport)

    orchestrator = IngestionOrchestrator(
        chunk_store=chunk_store,
        job_store=job_store,
        extraction_service=extraction_service,
        graph_plan_builder=plan_builder,
        graph_writer=graph_writer,
        resolve_entity=entity_registry.resolve_entity,
        embedder=embedder,
        embedding_store=embedding_store,
        search_index_store=search_index_store,
        update_classifier=update_classifier,
        find_existing_facts=fact_lookup.find_existing,
    )

    retrieval_engine = HybridRetrievalEngine(
        llm_client=config.get_reader_client(),
        embedder=embedder,
        pg_connection=pg_conn,
        hydra_client=hydra_transport,
        config=config,
        temporal_resolver_client=config.get_temporal_resolver_client(),
        query_rewriter_client=config.get_query_rewriter_client(),
    )

    hydration_manager = HydrationManager(
        pg_connection=pg_conn,
        hydra_client=hydra_transport,
        embedder=embedder,
    )

    _GLOBAL_ENGINE = MemoryEngine(
        orchestrator=orchestrator,
        retrieval_engine=retrieval_engine,
        llm_client=config.get_reader_client(),
        pg_connection=pg_conn,
        hydration_manager=hydration_manager,
        config=config,
    )
    return _GLOBAL_ENGINE


router = APIRouter()


class IngestRequest(BaseModel):
    context_id: str
    session_id: str
    role: str
    content: str
    timestamp: Optional[datetime] = None


class SearchRequest(BaseModel):
    context_id: str
    query: str
    question_date: Optional[datetime] = None


class ChatRequest(BaseModel):
    context_id: str
    session_id: str
    user_message: str


@router.post("/v1/memory/ingest")
def ingest_memory(req: IngestRequest, engine: MemoryEngine = Depends(get_engine)):
    ts = req.timestamp or datetime.now(timezone.utc)
    engine.add_turn_async(req.context_id, req.session_id, req.role, req.content, ts)
    return {"status": "success", "ingested_at": ts}


@router.post("/v1/memory/search")
def search_memory(req: SearchRequest, engine: MemoryEngine = Depends(get_engine)):
    q_date = req.question_date or datetime.now(timezone.utc)
    answer = engine.search_memories(req.context_id, req.query, q_date)
    return {"answer": answer}


@router.post("/v1/chat")
def chat_turn(req: ChatRequest, engine: MemoryEngine = Depends(get_engine)):
    reply = engine.generate_reply(req.context_id, req.session_id, req.user_message)
    return {"reply": reply}


@router.get("/v1/health")
def health_check():
    db_url = os.environ.get("CONTEXT_MEMORY_DATABASE_URL", os.environ.get("DATABASE_URL", "postgresql://context_memory@127.0.0.1:54329/context_memory"))
    hydra_url = os.environ.get("CONTEXT_MEMORY_HYDRADB_URL", os.environ.get("HYDRA_DB_HOST", "http://127.0.0.1:8080"))
    if not hydra_url.startswith("http://") and not hydra_url.startswith("https://"):
        hydra_url = f"http://{hydra_url}"

    postgres_status = "unknown"
    hydradb_status = "unknown"

    try:
        import psycopg
        with psycopg.connect(db_url, connect_timeout=2) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                postgres_status = "up"
    except Exception as e:
        postgres_status = f"down ({type(e).__name__})"

    try:
        import httpx
        r = httpx.get(f"{hydra_url}/health", timeout=2.0)
        hydradb_status = "up" if r.status_code < 500 else f"unhealthy ({r.status_code})"
    except Exception as e:
        hydradb_status = f"down ({type(e).__name__})"

    overall_status = "ok" if postgres_status == "up" else "degraded"
    return {"status": "ok", "postgres": postgres_status, "hydradb": hydradb_status}


@router.post("/v1/demo/clear")
def clear_demo():
    from api.stream import streamer
    streamer.push_event({"type": "graph_clear"})
    return {"status": "cleared"}


@router.post("/v1/demo/simulate")
async def simulate_demo(background_tasks: BackgroundTasks):
    from api.stream import streamer
    async def run_simulation():
        streamer.push_event({"type": "graph_clear"})
        await asyncio.sleep(0.3)

        streamer.push_event({
            "type": "chat_message",
            "message": {
                "id": f"demo-{int(time.time()*1000)}-1",
                "role": "user",
                "content": "Hi! My name is Alice, and I am a Principal AI Engineer at TechCorp.",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
        await asyncio.sleep(0.6)

        streamer.push_event({
            "type": "graph_update",
            "nodes": [
                {"id": "node-alice", "label": "Entity", "properties": {"name": "Alice"}},
                {"id": "node-techcorp", "label": "Entity", "properties": {"name": "TechCorp"}},
                {"id": "node-role", "label": "Fact", "properties": {"name": "Principal AI Engineer"}}
            ],
            "edges": [
                {"id": "edge-1", "source_id": "node-alice", "target_id": "node-techcorp", "type": "WORKS_AT"},
                {"id": "edge-2", "source_id": "node-alice", "target_id": "node-role", "type": "HAS_ROLE"}
            ]
        })
        await asyncio.sleep(1.0)

        streamer.push_event({
            "type": "chat_message",
            "message": {
                "id": f"demo-{int(time.time()*1000)}-2",
                "role": "agent",
                "content": "Hello Alice! Great to meet you. I've stored your role at TechCorp in long-term memory.",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
        await asyncio.sleep(1.0)

        streamer.push_event({
            "type": "chat_message",
            "message": {
                "id": f"demo-{int(time.time()*1000)}-3",
                "role": "user",
                "content": "I prefer dark mode UI and love drinking Matcha Latte during code reviews.",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
        await asyncio.sleep(0.6)

        streamer.push_event({
            "type": "graph_update",
            "nodes": [
                {"id": "node-alias-alice", "label": "Alias", "properties": {"name": "Alice_Alias"}},
                {"id": "node-darkmode", "label": "Fact", "properties": {"name": "Prefers Dark Mode"}},
                {"id": "node-matcha", "label": "Fact", "properties": {"name": "Loves Matcha Latte"}}
            ],
            "edges": [
                {"id": "edge-3", "source_id": "node-alice", "target_id": "node-alias-alice", "type": "HAS_ALIAS"},
                {"id": "edge-4", "source_id": "node-alice", "target_id": "node-darkmode", "type": "PREFERS"},
                {"id": "edge-5", "source_id": "node-alice", "target_id": "node-matcha", "type": "LIKES"}
            ]
        })
        await asyncio.sleep(1.0)

        streamer.push_event({
            "type": "chat_message",
            "message": {
                "id": f"demo-{int(time.time()*1000)}-4",
                "role": "agent",
                "content": "Noted! Preferences for Dark Mode and Matcha Latte saved to your memory profile.",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
        await asyncio.sleep(1.0)

        streamer.push_event({
            "type": "chat_message",
            "message": {
                "id": f"demo-{int(time.time()*1000)}-5",
                "role": "user",
                "content": "Recently moved from San Francisco to Neo-Tokyo.",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
        await asyncio.sleep(0.6)

        streamer.push_event({
            "type": "graph_update",
            "nodes": [
                {"id": "node-neotokyo", "label": "Entity", "properties": {"name": "Neo-Tokyo"}},
                {"id": "node-turn-3", "label": "Turn", "properties": {"name": "Session Turn 3"}}
            ],
            "edges": [
                {"id": "edge-6", "source_id": "node-alice", "target_id": "node-neotokyo", "type": "LIVES_IN"},
                {"id": "edge-7", "source_id": "node-neotokyo", "target_id": "node-turn-3", "type": "LOCATED_AT"}
            ]
        })

    background_tasks.add_task(run_simulation)
    return {"status": "simulation_started"}

