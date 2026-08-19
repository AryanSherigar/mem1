from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_memory.ingestion.fakes import DeterministicExtractor
from context_memory.ingestion.extraction import ExtractionService, InMemoryExtractionStore
from context_memory.core.enums import MemoryScope, MemoryType
from context_memory.core.models import ContextBatch, ContextRecord, ExtractionDraft, SourceDescriptor
from context_memory.core.validation import chunk_from_record


class ExtractionServiceTests(unittest.TestCase):
    def batch(self, *, session_id: str | None = "session-001") -> tuple[ContextBatch, ContextRecord]:
        record = ContextRecord(
            record_id="record-001", session_id=session_id,
            occurred_at=datetime(2026, 1, 10, 9, tzinfo=timezone.utc), content="café 🐶 likes walks"
        )
        return ContextBatch("ingestion-001", "context-001", SourceDescriptor("fixture", "fixture-001"), (record,)), record

    def service(self, drafts: list[ExtractionDraft]) -> tuple[ExtractionService, InMemoryExtractionStore]:
        store = InMemoryExtractionStore()
        return ExtractionService(DeterministicExtractor({"record-001": drafts}), store), store

    def test_defaults_to_semantic_session_and_preserves_unicode_span(self) -> None:
        batch, record = self.batch()
        service, store = self.service([ExtractionDraft("fact-001", "Dog preference", 5, 6, 0.9)])
        result = service.extract(batch, record, chunk_from_record(batch, record))
        candidate = result.accepted[0]
        self.assertEqual((candidate.memory_type, candidate.scope_type, candidate.scope_id),
                         (MemoryType.SEMANTIC, MemoryScope.SESSION, "session-001"))
        self.assertEqual(record.content[5:6], "🐶")
        self.assertEqual(len(store.attempts), 1)

    def test_defaults_to_chat_without_session(self) -> None:
        batch, record = self.batch(session_id=None)
        service, _ = self.service([ExtractionDraft("fact-001", "Dog preference", 5, 6, 0.9)])
        candidate = service.extract(batch, record, chunk_from_record(batch, record)).accepted[0]
        self.assertEqual((candidate.scope_type, candidate.scope_id), (MemoryScope.CHAT, "context-001"))

    def test_procedural_requires_explicit_fixture_marking(self) -> None:
        batch, record = self.batch()
        service, _ = self.service([
            ExtractionDraft("ordinary", "User likes walks", 7, 18, 0.7),
            ExtractionDraft("procedure", "Walk dog every morning", 7, 18, 0.8, MemoryType.PROCEDURAL),
        ])
        result = service.extract(batch, record, chunk_from_record(batch, record))
        self.assertEqual([item.memory_type for item in result.accepted], [MemoryType.SEMANTIC, MemoryType.PROCEDURAL])

    def test_malformed_and_episodic_drafts_are_preserved_as_rejections(self) -> None:
        batch, record = self.batch()
        service, store = self.service([
            ExtractionDraft("bad-span", "bad", 19, 21, 0.5),
            ExtractionDraft("episodic", "raw", 0, 4, 0.5, MemoryType.EPISODIC),
        ])
        result = service.extract(batch, record, chunk_from_record(batch, record))
        self.assertEqual(len(result.accepted), 0)
        self.assertEqual(len(result.rejected), 2)
        self.assertIn("Unicode code-point span", result.rejected[0].reason)
        self.assertIn("reserved for the immutable raw chunk", result.rejected[1].reason)
        self.assertEqual(len(store.attempts[0]["rejected"]), 2)

    def test_same_input_replays_same_deterministic_attempt_id(self) -> None:
        batch, record = self.batch()
        service, _ = self.service([ExtractionDraft("fact-001", "Dog preference", 5, 6, 0.9)])
        chunk = chunk_from_record(batch, record)
        self.assertEqual(service.extract(batch, record, chunk).attempt_id, service.extract(batch, record, chunk).attempt_id)

    def test_changed_fixture_output_gets_a_distinct_audit_attempt(self) -> None:
        batch, record = self.batch()
        chunk = chunk_from_record(batch, record)
        first, _ = self.service([ExtractionDraft("fact-001", "Dog preference", 5, 6, 0.9)])
        second, _ = self.service([ExtractionDraft("fact-002", "Dog preference", 5, 6, 0.9)])
        self.assertNotEqual(first.extract(batch, record, chunk).attempt_id, second.extract(batch, record, chunk).attempt_id)

    def test_non_fixture_extractor_is_blocked(self) -> None:
        class NonFixture:
            def extract(self, record: ContextRecord):
                return ()
        batch, record = self.batch()
        with self.assertRaisesRegex(ValueError, "only deterministic-fixture"):
            ExtractionService(NonFixture(), InMemoryExtractionStore()).extract(batch, record, chunk_from_record(batch, record))

from context_memory.ingestion.extraction import LLMExtractionService, ExtractionResponse, ExtractedDraft

class FakeLLMClient:
    def __init__(self, response):
        self.response = response
    def structured_completion(self, system, user, schema):
        return self.response

class FakeContextRepo:
    def get_conversation_buffer(self, context_id, session_id, limit):
        return [{"role": "user", "content": "I have a dog named Max."}]
    def get_top_candidate_facts(self, context_id, content, limit):
        return []

class LLMExtractionServiceTests(unittest.TestCase):
    def batch(self, *, session_id: str | None = "session-001") -> tuple[ContextBatch, ContextRecord]:
        record = ContextRecord(
            record_id="record-001", session_id=session_id, actor_role="user",
            occurred_at=datetime(2026, 1, 10, 9, tzinfo=timezone.utc), content="café 🐶 likes walks"
        )
        return ContextBatch("ingestion-001", "context-001", SourceDescriptor("fixture", "fixture-001"), (record,)), record
    
    def test_llm_extraction(self):
        batch, record = self.batch()
        llm = FakeLLMClient(ExtractionResponse(drafts=[ExtractedDraft(text="Dog likes walks", confidence=0.9, action="UPDATE", predicate_key="pet_preference")]))
        repo = FakeContextRepo()
        store = InMemoryExtractionStore()
        service = LLMExtractionService(llm, store, repo)
        
        result = service.extract(batch, record, chunk_from_record(batch, record))
        self.assertEqual(len(result.accepted), 1)
        self.assertEqual(result.accepted[0].action, "UPDATE")
        self.assertEqual(result.accepted[0].predicate_key, "pet_preference")
