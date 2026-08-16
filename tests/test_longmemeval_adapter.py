from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from context_memory.ingestion.sources.longmemeval import adapt_longmemeval_instance, parse_longmemeval_timestamp
from context_memory.core.errors import ContractValidationError
from context_memory.core.models import ContextBatch

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "docs" / "fixtures"


class LongMemEvalAdapterTests(unittest.TestCase):
    def input_instance(self) -> dict[str, object]:
        return json.loads((FIXTURES / "longmemeval_adapter_input_v1.json").read_text())[0]

    def test_adapter_matches_canonical_fixture_and_sorts_sessions(self) -> None:
        batch = adapt_longmemeval_instance(self.input_instance(), "fixture-longmemeval-v1-run-001")
        expected_payload = json.loads(
            (FIXTURES / "longmemeval_adapter_expected_context_batch_v1.json").read_text()
        )
        expected_payload.pop("adapter_fixture_note")
        self.assertEqual(batch, ContextBatch.from_mapping(expected_payload))
        self.assertEqual(
            [record.session_id for record in batch.records],
            ["session-earlier:source-0001", "session-later:source-0000"],
        )
        self.assertEqual(batch.records[0].metadata["source_index"], 1)

    def test_custom_minute_timestamp_parses_to_ordering_basis(self) -> None:
        normalized = parse_longmemeval_timestamp("2021/05/03 (Mon) 09:02", "fixture.date")
        self.assertEqual(normalized.isoformat(), "2021-05-03T09:02:00+00:00")

    def test_parallel_array_mismatch_is_rejected_before_core_batch(self) -> None:
        payload = self.input_instance()
        payload["haystack_dates"] = []
        with self.assertRaisesRegex(ContractValidationError, "equal lengths"):
            adapt_longmemeval_instance(payload, "fixture-run")

    def test_evaluation_labels_are_not_canonical_metadata(self) -> None:
        batch = adapt_longmemeval_instance(self.input_instance(), "fixture-run")
        forbidden = {"question_id", "question_type", "has_answer", "_abs", "answer", "answer_session_ids"}
        self.assertFalse(forbidden.intersection(batch.metadata))
        self.assertTrue(all(not forbidden.intersection(record.metadata) for record in batch.records))

    def test_empty_turns_are_skipped_with_audit_count(self) -> None:
        payload = self.input_instance()
        sessions = payload["haystack_sessions"]
        self.assertIsInstance(sessions, list)
        sessions[0].append({"role": "assistant", "content": ""})
        batch = adapt_longmemeval_instance(payload, "fixture-run")
        self.assertEqual(len(batch.records), 2)
        self.assertEqual(batch.metadata["source_empty_turn_count"], 1)

    def test_cli_dry_run_does_not_require_database(self) -> None:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT / "src")
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "ingest_longmemeval.py"),
                "--input",
                str(FIXTURES / "longmemeval_adapter_input_v1.json"),
                "--dry-run",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(completed.stdout.strip(), "validated_batches=1 validated_records=2")
