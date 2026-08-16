from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from context_memory.core.errors import ContractValidationError
from context_memory.core.models import ContextBatch

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "docs" / "fixtures"


class ContractV1Tests(unittest.TestCase):
    def load_fixture(self, name: str) -> dict[str, object]:
        return json.loads((FIXTURES / name).read_text())

    def test_generic_fixture_parses(self) -> None:
        batch = ContextBatch.from_mapping(self.load_fixture("generic_context_batch_v1.json"))
        self.assertEqual(batch.contract_version, "v1")
        self.assertEqual(len(batch.records), 7)
        self.assertEqual(batch.records[0].record_id, "session-a-turn-001")

    def test_longmemeval_output_has_generic_shape(self) -> None:
        payload = self.load_fixture("longmemeval_adapter_expected_context_batch_v1.json")
        payload.pop("adapter_fixture_note")
        batch = ContextBatch.from_mapping(payload)
        self.assertEqual(batch.source.source_type, "longmemeval")
        self.assertEqual(len(batch.records), 2)

    def test_duplicate_record_id_is_rejected(self) -> None:
        payload = self.load_fixture("generic_context_batch_v1.json")
        records = payload["records"]
        self.assertIsInstance(records, list)
        records.append(dict(records[0]))
        with self.assertRaisesRegex(ContractValidationError, "duplicate record_id"):
            ContextBatch.from_mapping(payload)

    def test_evaluation_label_in_metadata_is_rejected(self) -> None:
        payload = self.load_fixture("generic_context_batch_v1.json")
        records = payload["records"]
        self.assertIsInstance(records, list)
        records[0]["metadata"] = {"has_answer": True}
        with self.assertRaisesRegex(ContractValidationError, "evaluation-only"):
            ContextBatch.from_mapping(payload)

    def test_timestamp_requires_offset(self) -> None:
        payload = self.load_fixture("generic_context_batch_v1.json")
        records = payload["records"]
        self.assertIsInstance(records, list)
        records[0]["occurred_at"] = "2026-01-10T09:00:00"
        with self.assertRaisesRegex(ContractValidationError, "UTC offset"):
            ContextBatch.from_mapping(payload)
