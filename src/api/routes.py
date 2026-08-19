from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from context_memory.engine import MemoryEngine
from context_memory.client.hydradb_http import HydraHttpTransport
from context_memory.core.llm_client import LLMClient
from context_memory.ingestion.embedding import SentenceTransformerEmbedder
from context_memory.ingestion.extraction import ExtractionService
from context_memory.ingestion.graph_plan_builder import GraphPlanBuilder
from context_memory.ingestion.graph_writer import GraphWriter
from context_memory.ingestion.model_adapters import LLMExtractor
from context_memory.ingestion.orchestrator import IngestionOrchestrator
from context_memory.ingestion.resolution import EntityRegistry
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
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is not None:
        return _GLOBAL_ENGINE

    db_url = os.environ.get("CONTEXT_MEMORY_DATABASE_URL", os.environ.get("DATABASE_URL", "postgresql://context_memory@127.0.0.1:54329/context_memory"))
    hydra_url = os.environ.get("CONTEXT_MEMORY_HYDRADB_URL", os.environ.get("HYDRA_DB_HOST", "http://127.0.0.1:8080"))
    if not hydra_url.startswith("http://") and not hydra_url.startswith("https://"):
        hydra_url = f"http://{hydra_url}"
    hydra_token = os.environ.get("CONTEXT_MEMORY_HYDRADB_TOKEN", os.environ.get("HYDRA_DB_API_KEY", ""))
    hydra_db = os.environ.get("CONTEXT_MEMORY_HYDRADB_DATABASE", os.environ.get("HYDRA_DB_GRAPH_ID", "default"))

    llm_base_url = os.environ.get("FIREWORKS_BASE_URL", os.environ.get("OPENAI_BASE_URL", "https://api.fireworks.ai/inference/v1"))
    llm_api_key = os.environ.get("FIREWORKS_API_KEY", os.environ.get("OPENAI_API_KEY", "fake-key"))
    llm_model = os.environ.get("EXTRACTOR_MODEL", "accounts/fireworks/models/deepseek-v4-flash-0731")

    import psycopg
    pg_conn = psycopg.connect(db_url, autocommit=True)
    hydra_transport = HydraHttpTransport(base_url=hydra_url, bearer_token=hydra_token or None, database=hydra_db)
    llm_client = LLMClient(base_url=llm_base_url, api_key=llm_api_key, model_name=llm_model)
    embedder = SentenceTransformerEmbedder()

    chunk_store = PostgresChunkStore(pg_conn)
    job_store = PostgresJobStore(pg_conn)
    manifest_store = PostgresGraphManifestStore(pg_conn)
    embedding_store = PostgresEmbeddingStore(pg_conn)
    search_index_store = PostgresSearchIndexStore(pg_conn)
    extraction_store = PostgresExtractionStore(pg_conn)

    extractor = LLMExtractor(llm_client)
    extraction_service = ExtractionService(extractor, extraction_store)
    entity_registry = EntityRegistry(allocator=chunk_store, model=None)
    plan_builder = GraphPlanBuilder(allocator=chunk_store)
    graph_writer = GraphWriter(manifest_store=manifest_store, transport=hydra_transport)

    from context_memory.ingestion.model_adapters import LLMTemporalUpdateModel
    from context_memory.ingestion.resolution import TemporalUpdateClassifier
    temporal_model = LLMTemporalUpdateModel(llm_client)
    update_classifier = TemporalUpdateClassifier(temporal_model)

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
    )

    retrieval_engine = HybridRetrievalEngine(
        llm_client=llm_client,
        embedder=embedder,
        pg_connection=pg_conn,
        hydra_client=hydra_transport,
    )

    hydration_manager = HydrationManager(
        pg_connection=pg_conn,
        hydra_client=hydra_transport,
        embedder=embedder,
    )

    _GLOBAL_ENGINE = MemoryEngine(
        orchestrator=orchestrator,
        retrieval_engine=retrieval_engine,
        llm_client=llm_client,
        pg_connection=pg_conn,
        hydration_manager=hydration_manager,
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
    return {"status": overall_status, "postgres": postgres_status, "hydradb": hydradb_status}
