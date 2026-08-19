"""Unit tests for the LongMemEval benchmark runner."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from context_memory.core.models import ContextBatch, ContextRecord, SourceDescriptor
from context_memory.ingestion.fakes import (
    DeterministicEmbedder,
    DeterministicExtractor,
    InMemoryChunkStore,
    InMemoryEmbeddingStore,
    InMemoryGraphIdAllocator,
    InMemoryGraphManifestStore,
    InMemoryJobStore,
    InMemorySearchIndexStore,
    RecordingGraphTransport,
)
from context_memory.ingestion.extraction import ExtractionService, InMemoryExtractionStore
from context_memory.ingestion.graph_plan_builder import GraphPlanBuilder
from context_memory.ingestion.graph_writer import GraphWriter
from context_memory.ingestion.orchestrator import IngestionOrchestrator
from context_memory.ingestion.resolution import EntityRegistry
from evaluation.benchmark_runner import evaluate_dataset, evaluate_instance


class BenchmarkRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunk_store = InMemoryChunkStore()
        self.job_store = InMemoryJobStore()
        self.manifest_store = InMemoryGraphManifestStore()
        self.embedding_store = InMemoryEmbeddingStore()
        self.search_index_store = InMemorySearchIndexStore()
        self.allocator = InMemoryGraphIdAllocator()
        self.transport = RecordingGraphTransport()
        self.extractor = DeterministicExtractor()
        self.extraction_store = InMemoryExtractionStore()
        self.extraction_service = ExtractionService(self.extractor, self.extraction_store)
        self.entity_registry = EntityRegistry(allocator=self.allocator)
        self.plan_builder = GraphPlanBuilder(allocator=self.allocator)
        self.graph_writer = GraphWriter(manifest_store=self.manifest_store, transport=self.transport)
        self.embedder = DeterministicEmbedder()

        self.orchestrator = IngestionOrchestrator(
            chunk_store=self.chunk_store,
            job_store=self.job_store,
            extraction_service=self.extraction_service,
            graph_plan_builder=self.plan_builder,
            graph_writer=self.graph_writer,
            resolve_entity=self.entity_registry.resolve_entity,
            embedder=self.embedder,
            embedding_store=self.embedding_store,
            search_index_store=self.search_index_store,
        )

        self.mock_retrieval = MagicMock()
        self.mock_retrieval.retrieve_and_answer.return_value = "Max is the dog's name."

        self.sample_instance = {
            "question_id": "test_q_001",
            "haystack_session_ids": ["session-001"],
            "haystack_dates": ["2022/01/01 (Sat) 09:00"],
            "haystack_sessions": [
                [{"role": "user", "content": "I adopted a dog named Max."}]
            ],
            "question": "What is the dog's name?",
            "question_type": "single-session-user",
            "question_date": "2022/01/03 (Mon) 10:00",
            "answer": "Max",
        }

    def test_evaluate_instance(self) -> None:
        res = evaluate_instance(self.sample_instance, self.orchestrator, self.mock_retrieval)
        self.assertEqual(res["question_id"], "test_q_001")
        self.assertEqual(res["hypothesis"], "Max is the dog's name.")
        self.mock_retrieval.retrieve_and_answer.assert_called_once()

    def test_evaluate_dataset_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "predictions.jsonl"
            instances = [
                self.sample_instance,
                {
                    "question_id": "test_q_002",
                    "haystack_session_ids": ["session-002"],
                    "haystack_dates": ["2022/01/02 (Sun) 10:00"],
                    "haystack_sessions": [
                        [{"role": "user", "content": "My favorite fruit is apple."}]
                    ],
                    "question": "What is my favorite fruit?",
                    "question_type": "single-session-user",
                    "question_date": "2022/01/03 (Mon) 10:00",
                    "answer": "apple",
                },
            ]

            results = evaluate_dataset(
                instances=instances,
                orchestrator=self.orchestrator,
                retrieval_engine=self.mock_retrieval,
                output_path=out_path,
            )

            self.assertEqual(len(results), 2)
            self.assertTrue(out_path.exists())

            lines = out_path.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 2)

            rec1 = json.loads(lines[0])
            self.assertEqual(rec1["question_id"], "test_q_001")
            self.assertEqual(rec1["hypothesis"], "Max is the dog's name.")

            rec2 = json.loads(lines[1])
            self.assertEqual(rec2["question_id"], "test_q_002")
            self.assertEqual(rec2["hypothesis"], "Max is the dog's name.")


if __name__ == "__main__":
    unittest.main()
