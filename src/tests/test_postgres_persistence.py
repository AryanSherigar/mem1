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

from context_memory.core.enums import IngestionJobState
from context_memory.ingestion.fakes import DeterministicExtractor
from context_memory.persistence.postgres import (
    PostgresChunkStore,
    PostgresEmbeddingStore,
    PostgresExtractionStore,
    PostgresGraphManifestStore,
    PostgresJobStore,
)
from context_memory.ingestion.sources.longmemeval import adapt_longmemeval_instance
from context_memory.core.errors import GraphPayloadConflictError, IllegalJobTransitionError, ImmutableRecordConflictError
from context_memory.ingestion.extraction import ExtractionService
from context_memory.ingestion.service import IngestionService
from context_memory.ingestion.resolution import EntityRegistry
from context_memory.persistence.migrations import apply_migrations
from context_memory.core.models import Chunk, ContextBatch, ContextRecord, Embedding, ExtractionDraft, SourceDescriptor
from context_memory.core.graph import GraphNode, GraphWritePlan
from context_memory.core.validation import chunk_from_record, content_hash

ROOT = Path(__file__).resolve().parents[2]
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

    def test_embedding_put_is_idempotent_and_rejects_changed_hash(self) -> None:
        chunk_store = PostgresChunkStore(self.connection)
        chunk = self.chunk("record-001", "Max is a golden retriever")
        chunk_store.put(chunk)
        store = PostgresEmbeddingStore(self.connection)
        embedding = Embedding(
            context_id=chunk.context_id, subject_kind="fact", subject_id=f"fact:{uuid4().hex}",
            source_chunk_id=chunk.chunk_id, model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_version="1", values=(0.1, 0.2, 0.3), embedded_content_hash="sha256:abc",
        )
        self.assertEqual(store.put(embedding), embedding)
        self.assertEqual(store.put(embedding), embedding)
        changed = Embedding(
            context_id=embedding.context_id, subject_kind=embedding.subject_kind, subject_id=embedding.subject_id,
            source_chunk_id=embedding.source_chunk_id, model_name=embedding.model_name,
            model_version=embedding.model_version, values=(0.9, 0.9, 0.9), embedded_content_hash="sha256:different",
        )
        with self.assertRaises(ImmutableRecordConflictError):
            store.put(changed)

    def test_embedding_deactivate_marks_inactive_without_deleting(self) -> None:
        chunk_store = PostgresChunkStore(self.connection)
        chunk = self.chunk("record-001", "Max moved to Seattle")
        chunk_store.put(chunk)
        store = PostgresEmbeddingStore(self.connection)
        subject_id = f"fact:{uuid4().hex}"
        embedding = Embedding(
            context_id=chunk.context_id, subject_kind="fact", subject_id=subject_id,
            source_chunk_id=chunk.chunk_id, model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_version="1", values=(0.1, 0.2, 0.3), embedded_content_hash="sha256:abc",
        )
        store.put(embedding)
        store.deactivate(chunk.context_id, "fact", subject_id)
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT is_active FROM memory_embeddings WHERE context_id = %s AND subject_id = %s",
                (chunk.context_id, subject_id),
            )
            row = cursor.fetchone()
        self.assertEqual(row, (False,))

    def test_job_lifecycle_reaches_completed(self) -> None:
        chunk_store = PostgresChunkStore(self.connection)
        chunk = self.chunk("record-001", "Max likes walks")
        chunk_store.put(chunk)
        jobs = PostgresJobStore(self.connection)
        job = jobs.get(chunk.chunk_id)
        self.assertEqual(job.state, IngestionJobState.PENDING_GRAPH)
        job = jobs.transition(chunk.chunk_id, IngestionJobState.PENDING_EMBEDDINGS)
        self.assertEqual(job.state, IngestionJobState.PENDING_EMBEDDINGS)
        job = jobs.transition(chunk.chunk_id, IngestionJobState.VERIFYING)
        job = jobs.transition(chunk.chunk_id, IngestionJobState.COMPLETED)
        self.assertEqual(job.state, IngestionJobState.COMPLETED)

    def test_seed_is_idempotent_and_does_not_reset_progress(self) -> None:
        chunk_store = PostgresChunkStore(self.connection)
        chunk = self.chunk("record-001", "Max likes walks")
        chunk_store.put(chunk)  # already seeds pending_graph
        jobs = PostgresJobStore(self.connection)
        jobs.transition(chunk.chunk_id, IngestionJobState.PENDING_EMBEDDINGS)
        reseeded = jobs.seed(chunk.chunk_id, chunk.context_id)
        self.assertEqual(reseeded.state, IngestionJobState.PENDING_EMBEDDINGS)  # not reset to pending_graph

    def test_illegal_transition_is_rejected(self) -> None:
        chunk_store = PostgresChunkStore(self.connection)
        chunk = self.chunk("record-001", "Max likes walks")
        chunk_store.put(chunk)
        jobs = PostgresJobStore(self.connection)
        # COMPLETED is terminal with no next state at all.
        jobs.transition(chunk.chunk_id, IngestionJobState.PENDING_EMBEDDINGS)
        jobs.transition(chunk.chunk_id, IngestionJobState.VERIFYING)
        jobs.transition(chunk.chunk_id, IngestionJobState.COMPLETED)
        with self.assertRaises(IllegalJobTransitionError):
            jobs.transition(chunk.chunk_id, IngestionJobState.PENDING_GRAPH)

    def test_retryable_failure_increments_attempt_count(self) -> None:
        chunk_store = PostgresChunkStore(self.connection)
        chunk = self.chunk("record-001", "Max likes walks")
        chunk_store.put(chunk)
        jobs = PostgresJobStore(self.connection)
        job = jobs.transition(chunk.chunk_id, IngestionJobState.RETRYABLE_FAILED, error="simulated")
        self.assertEqual(job.attempt_count, 1)
        self.assertEqual(job.last_error, "simulated")
        job = jobs.transition(chunk.chunk_id, IngestionJobState.PENDING_GRAPH)
        job = jobs.transition(chunk.chunk_id, IngestionJobState.RETRYABLE_FAILED, error="simulated again")
        self.assertEqual(job.attempt_count, 2)

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

    def test_graph_manifest_allows_supersession_mutable_fields_only(self) -> None:
        """SUPERSEDES (ADR-030) needs to flip is_current/superseded_at/valid_to on
        a Fact node that was already fully registered in an earlier turn's plan.
        Without this allowance, the second write below would hit the exact same
        GraphPayloadConflictError the Session-node bug did (different partial
        payload, same logical_key) -- the bug this test guards against was never
        hypothetical, it's the same shape, just for a different node kind."""
        store = PostgresChunkStore(self.connection)
        context_id = f"context:{uuid4().hex}"
        graph_id = store.allocate_graph_id("fact", context_id, "fact:cand-001")
        manifest = PostgresGraphManifestStore(self.connection)

        full_properties = {
            "context_id": context_id, "logical_key": "fact:cand-001", "text": "Max lives in Boston",
            "speaker": "user", "predicate_key": "lives_in", "is_current": True,
            "superseded_at": 9999999999, "valid_to": 9999999999,
        }
        original = GraphWritePlan(context_id, "plan:001", (GraphNode(graph_id, "Fact", "fact:cand-001", full_properties),), ())
        manifest.register(original)

        # A later turn's SUPERSEDES handling only has these 3 fields to send --
        # not the fact's full original property set (graph_plan_builder.py only
        # has a FactState, not the original GraphNode, at that point).
        partial_update = {
            "context_id": context_id, "logical_key": "fact:cand-001",
            "is_current": False, "superseded_at": 1750000000, "valid_to": 1750000000,
        }
        superseded = GraphWritePlan(context_id, "plan:002", (GraphNode(graph_id, "Fact", "fact:cand-001", partial_update),), ())
        manifest.register(superseded)  # must NOT raise

        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT payload FROM graph_write_manifests WHERE record_kind = 'node' AND context_id = %s AND logical_key = %s",
                (context_id, "fact:cand-001"),
            )
            stored = cursor.fetchone()[0]

        # The merge must be additive: text/speaker/predicate_key (never resent)
        # survive, while the 3 mutable fields reflect the update.
        self.assertEqual(stored["properties"]["text"], "Max lives in Boston")
        self.assertEqual(stored["properties"]["speaker"], "user")
        self.assertEqual(stored["properties"]["predicate_key"], "lives_in")
        self.assertEqual(stored["properties"]["is_current"], False)
        self.assertEqual(stored["properties"]["superseded_at"], 1750000000)

        # A genuine conflict (a non-mutable field actually changing) must still
        # raise -- this allowance must not become a general escape hatch.
        real_conflict = {**partial_update, "text": "Max lives in Seattle"}
        conflicting = GraphWritePlan(context_id, "plan:003", (GraphNode(graph_id, "Fact", "fact:cand-001", real_conflict),), ())
        with self.assertRaises(GraphPayloadConflictError):
            manifest.register(conflicting)
