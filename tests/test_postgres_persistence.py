from __future__ import annotations

import os
import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

try:
    import psycopg
except ImportError:  # pragma: no cover - exercised only before optional dependency setup
    psycopg = None

from context_memory.ingestion.fakes import DeterministicExtractor
from context_memory.persistence.postgres import PostgresChunkStore, PostgresExtractionStore, PostgresGraphManifestStore
from context_memory.ingestion.sources.longmemeval import adapt_longmemeval_instance
from context_memory.core.errors import GraphPayloadConflictError, ImmutableRecordConflictError
from context_memory.ingestion.extraction import ExtractionService
from context_memory.ingestion.service import IngestionService
from context_memory.ingestion.resolution import EntityRegistry
from context_memory.persistence.migrations import apply_migrations
from context_memory.core.models import Chunk, ContextBatch, ContextRecord, ExtractionDraft, SourceDescriptor
from context_memory.core.graph import GraphNode, GraphWritePlan
from context_memory.core.validation import chunk_from_record, content_hash

ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = ROOT / "db" / "migrations"
FIXTURES = ROOT / "docs" / "fixtures"
DATABASE_URL = os.environ.get("CONTEXT_MEMORY_TEST_DATABASE_URL")


@unittest.skipUnless(psycopg is not None and DATABASE_URL, "requires CONTEXT_MEMORY_TEST_DATABASE_URL and psycopg")
class PostgresPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.connection = psycopg.connect(DATABASE_URL)
        apply_migrations(cls.connection, MIGRATIONS)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.connection.close()

    def chunk(self, record_id: str, raw_text: str) -> Chunk:
        unique = uuid4().hex
        return Chunk(
            chunk_id=f"chunk:{unique}:{record_id}",
            context_id=f"context:{unique}",
            source=SourceDescriptor("integration_test", unique),
            source_record_id=record_id,
            raw_text=raw_text,
            content_hash=content_hash(raw_text),
            occurred_at=datetime(2026, 1, 10, tzinfo=timezone.utc),
        )

    def test_idempotent_insert_and_changed_payload_conflict(self) -> None:
        store = PostgresChunkStore(self.connection)
        chunk = self.chunk("record-001", "alpha")
        self.assertEqual(store.put(chunk), chunk)
        self.assertEqual(store.put(chunk), chunk)
        changed = Chunk(
            chunk_id=chunk.chunk_id,
            context_id=chunk.context_id,
            source=chunk.source,
            source_record_id=chunk.source_record_id,
            raw_text="beta",
            content_hash=content_hash("beta"),
            occurred_at=chunk.occurred_at,
        )
        with self.assertRaises(ImmutableRecordConflictError):
            store.put(changed)

    def test_duplicate_text_from_distinct_records_is_preserved(self) -> None:
        store = PostgresChunkStore(self.connection)
        first = self.chunk("record-001", "same text")
        second = self.chunk("record-002", "same text")
        store.put(first)
        store.put(second)
        self.assertEqual(store.get(first.context_id, first.chunk_id), first)
        self.assertEqual(store.get(second.context_id, second.chunk_id), second)

    def test_graph_id_is_stable_and_unique(self) -> None:
        store = PostgresChunkStore(self.connection)
        context_id = f"context:{uuid4().hex}"
        first = store.allocate_graph_id("fact", context_id, "fact:001")
        again = store.allocate_graph_id("fact", context_id, "fact:001")
        second = store.allocate_graph_id("fact", context_id, "fact:002")
        self.assertEqual(first, again)
        self.assertNotEqual(first, second)
        self.assertGreaterEqual(first, 0)

    def test_longmemeval_adapter_uses_generic_service_and_replays(self) -> None:
        payload = json.loads((FIXTURES / "longmemeval_adapter_input_v1.json").read_text())[0]
        batch = adapt_longmemeval_instance(payload, "integration-longmemeval-run")
        service = IngestionService(PostgresChunkStore(self.connection))
        first = service.ingest(batch)
        replay = service.ingest(batch)
        self.assertEqual(first.chunk_ids, replay.chunk_ids)
        self.assertEqual(first.accepted_record_count, 2)

    def test_extraction_baseline_persists_accepts_and_rejections(self) -> None:
        unique = uuid4().hex
        record = ContextRecord("record-001", datetime(2026, 1, 10, tzinfo=timezone.utc), "alpha 🐶")
        batch = ContextBatch(
            f"ingestion:{unique}", f"context:{unique}", SourceDescriptor("integration_test", unique), (record,)
        )
        chunk = chunk_from_record(batch, record)
        PostgresChunkStore(self.connection).put(chunk)
        extractor = DeterministicExtractor({
            record.record_id: (
                ExtractionDraft("accepted", "Dog", 6, 7, 0.9),
                ExtractionDraft("rejected", "bad", 7, 9, 0.5),
            )
        })
        result = ExtractionService(extractor, PostgresExtractionStore(self.connection)).extract(batch, record, chunk)
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT extractor_kind, quality_status, accepted_count, rejected_count FROM extraction_attempts WHERE attempt_id = %s", (result.attempt_id,))
            attempt = cursor.fetchone()
            cursor.execute("SELECT count(*) FROM extracted_memory_candidates WHERE attempt_id = %s", (result.attempt_id,))
            accepted = cursor.fetchone()
            cursor.execute("SELECT count(*) FROM rejected_extraction_candidates WHERE attempt_id = %s", (result.attempt_id,))
            rejected = cursor.fetchone()
        self.assertEqual(attempt, ("deterministic_fixture", "baseline_only", 1, 1))
        self.assertEqual(accepted, (1,))
        self.assertEqual(rejected, (1,))

    def test_entity_registry_uses_stable_postgres_graph_ids(self) -> None:
        store = PostgresChunkStore(self.connection)
        context_id = f"context:{uuid4().hex}"
        first = EntityRegistry(store).resolve(context_id=context_id, surface="Max")
        replay = EntityRegistry(store).resolve(context_id=context_id, surface=" max ")
        other_context = EntityRegistry(store).resolve(context_id=f"{context_id}:other", surface="Max")
        self.assertEqual(first.entity.graph_id, replay.entity.graph_id)
        self.assertNotEqual(first.entity.graph_id, other_context.entity.graph_id)

    def test_graph_manifest_replay_and_changed_payload_conflict(self) -> None:
        store = PostgresChunkStore(self.connection)
        context_id = f"context:{uuid4().hex}"
        graph_id = store.allocate_graph_id("session", context_id, "session:001")
        manifest = PostgresGraphManifestStore(self.connection)
        plan = GraphWritePlan(context_id, "plan:001", (GraphNode(graph_id, "Session", "session:001", {"context_id": context_id, "session_id": "session-001"}),), ())
        manifest.register(plan)
        manifest.register(plan)
        changed = GraphWritePlan(context_id, "plan:001", (GraphNode(graph_id, "Session", "session:001", {"context_id": context_id, "session_id": "changed"}),), ())
        with self.assertRaises(GraphPayloadConflictError):
            manifest.register(changed)
