from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from context_memory.ingestion.fakes import DeterministicEmbedder, InMemoryChunkStore
from context_memory.core.errors import ContractValidationError, ImmutableRecordConflictError
from context_memory.core.enums import MemoryScope, MemoryType
from context_memory.core.models import (
    Chunk,
    ContextRecord,
    ExtractedMemoryCandidate,
    SourceDescriptor,
    SourceSpan,
    TemporalBounds,
)
from context_memory.core.validation import content_hash, validate_candidate


class DomainValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.occurred_at = datetime(2026, 1, 10, 9, 0, tzinfo=timezone.utc)
        self.record = ContextRecord(
            record_id="unicode-001",
            occurred_at=self.occurred_at,
            content="café 🐶",
        )

    def candidate(self, **overrides: object) -> ExtractedMemoryCandidate:
        values: dict[str, object] = {
            "candidate_id": "candidate-001",
            "text": "User mentioned a dog.",
            "memory_type": MemoryType.SEMANTIC,
            "scope_type": MemoryScope.USER,
            "scope_id": "fixture-user-001",
            "source_span": SourceSpan("unicode-001", 5, 6),
            "confidence": 0.9,
            "temporal": TemporalBounds(observed_at=self.occurred_at),
        }
        values.update(overrides)
        return ExtractedMemoryCandidate(**values)

    def test_unicode_code_point_span_is_valid(self) -> None:
        candidate = self.candidate()
        validate_candidate(candidate, [self.record])
        self.assertEqual(self.record.content[candidate.source_span.source_start:candidate.source_span.source_end], "🐶")

    def test_invalid_span_is_rejected(self) -> None:
        candidate = self.candidate(source_span=SourceSpan("unicode-001", 6, 7))
        with self.assertRaisesRegex(ContractValidationError, "Unicode code-point span"):
            validate_candidate(candidate, [self.record])

    def test_organization_scope_is_rejected(self) -> None:
        with self.assertRaisesRegex(ContractValidationError, "organization scope"):
            self.candidate(scope_type="organization")

    def test_observed_time_must_match_source(self) -> None:
        candidate = self.candidate(
            temporal=TemporalBounds(observed_at=datetime(2026, 1, 11, tzinfo=timezone.utc))
        )
        with self.assertRaisesRegex(ContractValidationError, "must equal source"):
            validate_candidate(candidate, [self.record])

    def test_content_hash_is_stable(self) -> None:
        self.assertEqual(content_hash("café 🐶"), content_hash("café 🐶"))
        self.assertNotEqual(content_hash("café 🐶"), content_hash("cafe dog"))

    def test_fake_embedder_is_deterministic(self) -> None:
        embedder = DeterministicEmbedder(dimensions=4)
        self.assertEqual(embedder.embed("fact"), embedder.embed("fact"))
        self.assertEqual(len(embedder.embed("fact")), 4)

    def test_chunk_store_rejects_mutation(self) -> None:
        store = InMemoryChunkStore()
        source = SourceDescriptor("fixture", "fixture-001")
        first = Chunk(
            "chunk-001", "context-001", source, "record-001", "alpha", content_hash("alpha"), self.occurred_at
        )
        changed = Chunk(
            "chunk-001", "context-001", source, "record-001", "beta", content_hash("beta"), self.occurred_at
        )
        store.put(first)
        self.assertEqual(store.put(first), first)
        with self.assertRaises(ImmutableRecordConflictError):
            store.put(changed)
