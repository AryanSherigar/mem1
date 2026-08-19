"""Full-stack live M8 smoke: real PostgreSQL + a real local HydraDB graph-node.

Mirrors test_hydradb_live.py's gating. No LLM provider required — entity
resolution here only needs exact/new-entity matching, so this stays gated on
just the two stores that ADR-031's orchestrator actually touches.
"""

from __future__ import annotations

import os
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

try:
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None

from context_memory.core.models import ContextBatch, ContextRecord, EntityCandidate, ExtractionDraft, SourceDescriptor
from context_memory.client.hydradb_http import HydraHttpTransport
from context_memory.ingestion.extraction import ExtractionService
from context_memory.ingestion.fakes import DeterministicExtractor
from context_memory.ingestion.graph_plan_builder import GraphPlanBuilder
from context_memory.ingestion.graph_writer import GraphWriter
from context_memory.ingestion.orchestrator import IngestionOrchestrator
from context_memory.ingestion.embedding import SentenceTransformerEmbedder
from context_memory.ingestion.resolution import EntityRegistry
from context_memory.persistence.migrations import apply_migrations
from context_memory.persistence.postgres import (
    PostgresChunkStore,
    PostgresEmbeddingStore,
    PostgresExtractionStore,
    PostgresGraphManifestStore,
    PostgresJobStore,
)

ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = ROOT / "db" / "migrations"
DATABASE_URL = os.environ.get("CONTEXT_MEMORY_TEST_DATABASE_URL")
HYDRA_URL = os.environ.get("CONTEXT_MEMORY_HYDRADB_URL")
HYDRA_TOKEN = os.environ.get("CONTEXT_MEMORY_HYDRADB_TOKEN")
HYDRA_DATABASE = os.environ.get("CONTEXT_MEMORY_HYDRADB_DATABASE", "default")


@unittest.skipUnless(
    psycopg is not None and DATABASE_URL and HYDRA_URL and HYDRA_TOKEN,
    "requires CONTEXT_MEMORY_TEST_DATABASE_URL, CONTEXT_MEMORY_HYDRADB_URL, CONTEXT_MEMORY_HYDRADB_TOKEN",
)
class OrchestratorLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.connection = psycopg.connect(DATABASE_URL)
        apply_migrations(cls.connection, MIGRATIONS)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.connection.close()

    def test_full_stack_chunk_reaches_completed_and_is_readable_in_hydradb(self) -> None:
        context_id = f"live-orchestrator-{uuid4().hex}"
        record = ContextRecord(
            record_id="record-001", session_id="session-001",
            occurred_at=datetime(2026, 1, 10, 9, tzinfo=timezone.utc), content="Max the dog likes walks",
        )
        batch = ContextBatch(f"ingestion-{uuid4().hex}", context_id, SourceDescriptor("live-test", "live-fixture"), (record,))

        chunk_store = PostgresChunkStore(self.connection)
        job_store = PostgresJobStore(self.connection)
        extractor = DeterministicExtractor({
            "record-001": (
                ExtractionDraft("fact-001", "Max likes walks", 0, 3, 0.9, entities=(EntityCandidate("Max", "pet"),)),
            )
        })
        extraction_service = ExtractionService(extractor, PostgresExtractionStore(self.connection))
        transport = HydraHttpTransport(HYDRA_URL, HYDRA_TOKEN, graph_id=HYDRA_DATABASE)
        graph_writer = GraphWriter(PostgresGraphManifestStore(self.connection), transport)
        plan_builder = GraphPlanBuilder(chunk_store)  # PostgresChunkStore implements GraphIdAllocator
        registry = EntityRegistry(chunk_store)
        embedding_store = PostgresEmbeddingStore(self.connection)

        def resolve_entity(cid: str, surface: str, entity_type: str):
            return registry.resolve(context_id=cid, surface=surface, entity_type=entity_type).entity

        orchestrator = IngestionOrchestrator(
            chunk_store=chunk_store, job_store=job_store, extraction_service=extraction_service,
            graph_plan_builder=plan_builder, graph_writer=graph_writer, resolve_entity=resolve_entity,
            embedder=SentenceTransformerEmbedder(), embedding_store=embedding_store,
        )

        result = orchestrator.run_record(batch, record)
        self.assertEqual(result.state.value, "completed", result.error)
        self.assertEqual(result.accepted_fact_count, 1)

        # Independent re-read directly against HydraDB, not same-process evidence.
        job = job_store.get(result.chunk_id)
        rows = transport.read(
            "MATCH (f:Fact {context_id: $cid})-[:ABOUT]->(e:Entity {context_id: $cid}) RETURN e.canonical_name AS name",
            {"cid": context_id}, None,
        )
        self.assertEqual(rows, [{"name": "max"}])


if __name__ == "__main__":
    unittest.main()
