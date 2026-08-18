from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_memory.ingestion.fakes import InMemoryChunkStore
from context_memory.ingestion.service import IngestionPartialFailure, IngestionService
from context_memory.core.models import ContextBatch

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "docs" / "fixtures"


class FailingChunkStore(InMemoryChunkStore):
    def __init__(self, failure_after: int) -> None:
        super().__init__()
        self.failure_after = failure_after
        self.calls = 0

    def put(self, chunk):
        if self.calls == self.failure_after:
            raise RuntimeError("simulated persistence failure")
        self.calls += 1
        return super().put(chunk)


class IngestionServiceTests(unittest.TestCase):
    def fixture_batch(self) -> ContextBatch:
        payload = json.loads((FIXTURES / "generic_context_batch_v1.json").read_text())
        return ContextBatch.from_mapping(payload)

    def test_generic_fixture_creates_one_deterministic_chunk_per_record(self) -> None:
        batch = self.fixture_batch()
        store = InMemoryChunkStore()
        service = IngestionService(store)
        result = service.ingest(batch)
        replay = service.ingest(batch)
        self.assertEqual(result.accepted_record_count, 7)
        self.assertEqual(replay.chunk_ids, result.chunk_ids)
        for chunk_id in result.chunk_ids:
            self.assertIsNotNone(store.get(batch.context_id, chunk_id))

    def test_partial_failure_reports_only_durable_chunks(self) -> None:
        batch = self.fixture_batch()
        service = IngestionService(FailingChunkStore(failure_after=2))
        with self.assertRaises(IngestionPartialFailure) as raised:
            service.ingest(batch)
        self.assertEqual(len(raised.exception.persisted_chunk_ids), 2)
