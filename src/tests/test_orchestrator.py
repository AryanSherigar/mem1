from __future__ import annotations

import unittest
from datetime import datetime, timezone

from context_memory.core.enums import IngestionJobState
from context_memory.core.models import ContextBatch, ContextRecord, EntityCandidate, ExtractionDraft, SourceDescriptor
from context_memory.core.validation import chunk_from_record
from context_memory.ingestion.extraction import ExtractionService, InMemoryExtractionStore
from context_memory.ingestion.fakes import (
    DeterministicEmbedder,
    DeterministicExtractor,
    InMemoryChunkStore,
    InMemoryEmbeddingStore,
    InMemoryGraphIdAllocator,
    InMemoryGraphManifestStore,
    InMemoryJobStore,
    RecordingGraphTransport,
)
from context_memory.ingestion.graph_plan_builder import GraphPlanBuilder
from context_memory.ingestion.graph_writer import GraphWriter
from context_memory.ingestion.orchestrator import IngestionOrchestrator
from context_memory.ingestion.resolution import EntityRegistry


class _FailNTimesTransport:
    """Fails its first `fail_count` writes, then behaves like RecordingGraphTransport."""

    def __init__(self, fail_count: int) -> None:
        self._remaining_failures = fail_count
        self._inner = RecordingGraphTransport()

    def write(self, cypher, rows, idempotency_key):
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise ConnectionError("simulated transient graph-node network failure")
        return self._inner.write(cypher, rows, idempotency_key)

    def read(self, cypher, parameters, bookmark):
        return self._inner.read(cypher, parameters, bookmark)


class OrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunk_store = InMemoryChunkStore()
        self.job_store = InMemoryJobStore()
        self.allocator = InMemoryGraphIdAllocator()
        self.manifest_store = InMemoryGraphManifestStore()
        self.embedding_store = InMemoryEmbeddingStore()
        self.entity_registry = EntityRegistry(self.allocator)

    def batch_and_record(self, content: str = "Max the dog likes walks", session_id: str = "session-001"):
        record = ContextRecord(
            record_id="record-001", session_id=session_id,
            occurred_at=datetime(2026, 1, 10, 9, tzinfo=timezone.utc), content=content,
        )
        batch = ContextBatch("ingestion-001", "context-001", SourceDescriptor("fixture", "fixture-001"), (record,))
        return batch, record

    def orchestrator(self, transport) -> IngestionOrchestrator:
        extractor = DeterministicExtractor({
            "record-001": (
                ExtractionDraft(
                    "fact-001", "Max likes walks", 0, 3, 0.9,
                    entities=(EntityCandidate("Max", "pet"),),
                ),
            )
        })
        extraction_service = ExtractionService(extractor, InMemoryExtractionStore())
        graph_writer = GraphWriter(self.manifest_store, transport)
        plan_builder = GraphPlanBuilder(self.allocator)

        def resolve_entity(context_id: str, surface: str, entity_type: str):
            return self.entity_registry.resolve(context_id=context_id, surface=surface, entity_type=entity_type).entity

        return IngestionOrchestrator(
            chunk_store=self.chunk_store, job_store=self.job_store, extraction_service=extraction_service,
            graph_plan_builder=plan_builder, graph_writer=graph_writer, resolve_entity=resolve_entity,
            embedder=DeterministicEmbedder(), embedding_store=self.embedding_store,
        )

    def test_happy_path_reaches_completed(self) -> None:
        batch, record = self.batch_and_record()
        orchestrator = self.orchestrator(RecordingGraphTransport())
        result = orchestrator.run_record(batch, record)
        self.assertEqual(result.state, IngestionJobState.COMPLETED)
        self.assertEqual(result.accepted_fact_count, 1)
        job = self.job_store.get(result.chunk_id)
        self.assertEqual(job.state, IngestionJobState.COMPLETED)

    def test_happy_path_writes_graph_and_embedding(self) -> None:
        batch, record = self.batch_and_record()
        transport = RecordingGraphTransport()
        orchestrator = self.orchestrator(transport)
        orchestrator.run_record(batch, record)
        self.assertTrue(any("Session" in cypher for cypher, _, _ in transport.writes))
        self.assertTrue(any("Turn" in cypher for cypher, _, _ in transport.writes))
        self.assertTrue(any("Fact" in cypher for cypher, _, _ in transport.writes))
        self.assertTrue(any("Entity" in cypher for cypher, _, _ in transport.writes))
        self.assertEqual(len(self.embedding_store._rows), 1)

    def test_no_entities_no_facts_still_completes(self) -> None:
        record = ContextRecord(
            record_id="record-001", session_id="session-001",
            occurred_at=datetime(2026, 1, 10, 9, tzinfo=timezone.utc), content="unrelated text",
        )
        batch = ContextBatch("ingestion-001", "context-001", SourceDescriptor("fixture", "fixture-001"), (record,))
        extractor = DeterministicExtractor({"record-001": ()})  # nothing extracted
        extraction_service = ExtractionService(extractor, InMemoryExtractionStore())
        transport = RecordingGraphTransport()
        orchestrator = IngestionOrchestrator(
            chunk_store=self.chunk_store, job_store=self.job_store, extraction_service=extraction_service,
            graph_plan_builder=GraphPlanBuilder(self.allocator), graph_writer=GraphWriter(self.manifest_store, transport),
            resolve_entity=lambda *a: None, embedder=DeterministicEmbedder(), embedding_store=self.embedding_store,
        )
        result = orchestrator.run_record(batch, record)
        self.assertEqual(result.state, IngestionJobState.COMPLETED)
        self.assertEqual(result.accepted_fact_count, 0)
        # Session + Turn still written even with zero facts.
        self.assertTrue(any("Session" in cypher for cypher, _, _ in transport.writes))

    def test_transient_failure_is_retryable_and_retry_succeeds(self) -> None:
        batch, record = self.batch_and_record()
        transport = _FailNTimesTransport(fail_count=1)
        orchestrator = self.orchestrator(transport)
        first = orchestrator.run_record(batch, record)
        self.assertEqual(first.state, IngestionJobState.RETRYABLE_FAILED)
        self.assertIsNotNone(first.error)

        chunk = self.chunk_store.get(batch.context_id, first.chunk_id)
        job = self.job_store.get(first.chunk_id)
        retry = orchestrator.run_chunk(batch, record, chunk, job.state)
        self.assertEqual(retry.state, IngestionJobState.COMPLETED)

    def test_completed_job_is_not_reprocessed(self) -> None:
        batch, record = self.batch_and_record()
        orchestrator = self.orchestrator(RecordingGraphTransport())
        first = orchestrator.run_record(batch, record)
        chunk = self.chunk_store.get(batch.context_id, first.chunk_id)
        job = self.job_store.get(first.chunk_id)
        self.assertEqual(job.state, IngestionJobState.COMPLETED)
        result_again = orchestrator.run_chunk(batch, record, chunk, job.state)
        self.assertEqual(result_again.state, IngestionJobState.COMPLETED)
        self.assertEqual(result_again.accepted_fact_count, 0)  # short-circuited, extraction never re-ran

    def test_terminal_failed_blocks_auto_retry(self) -> None:
        batch, record = self.batch_and_record()
        orchestrator = self.orchestrator(RecordingGraphTransport())
        chunk = self.chunk_store.put(chunk_from_record(batch, record))
        self.job_store.seed(chunk.chunk_id, chunk.context_id)
        self.job_store.transition(chunk.chunk_id, IngestionJobState.RETRYABLE_FAILED, error="forced")
        self.job_store.transition(chunk.chunk_id, IngestionJobState.TERMINAL_FAILED, error="forced terminal")
        job = self.job_store.get(chunk.chunk_id)
        result = orchestrator.run_chunk(batch, record, chunk, job.state)
        self.assertEqual(result.state, IngestionJobState.TERMINAL_FAILED)
        self.assertIn("blocked", result.error)


if __name__ == "__main__":
    unittest.main()
