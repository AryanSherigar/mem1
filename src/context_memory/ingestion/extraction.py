"""Deterministic extraction baseline; deliberately not a model-quality claim."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import uuid
from pydantic import BaseModel, Field

from context_memory.core.errors import ContractValidationError
from context_memory.ingestion.ports import ExtractionStore, Extractor
from context_memory.core.enums import MemoryScope, MemoryType
from context_memory.core.models import (
    Chunk,
    ContextBatch,
    ContextRecord,
    ExtractedMemoryCandidate,
    ExtractionDraft,
    SourceSpan,
    TemporalBounds,
)
from context_memory.core.validation import validate_candidate

EXTRACTOR_KIND = "deterministic_fixture"
QUALITY_STATUS = "baseline_only"


@dataclass(frozen=True)
class RejectedExtraction:
    candidate_id: str
    reason: str
    draft: ExtractionDraft


@dataclass(frozen=True)
class ExtractionResult:
    attempt_id: str
    accepted: tuple[ExtractedMemoryCandidate, ...]
    rejected: tuple[RejectedExtraction, ...]


class InMemoryExtractionStore:
    def __init__(self) -> None:
        self.attempts: list[dict[str, object]] = []

    def record(self, **kwargs: object) -> None:
        self.attempts.append(kwargs)


class ExtractionService:
    """Converts untrusted fixture drafts into validated, source-linked candidates."""

    def __init__(self, extractor: Extractor, store: ExtractionStore) -> None:
        self._extractor = extractor
        self._store = store

    def extract(self, batch: ContextBatch, record: ContextRecord, chunk: Chunk) -> ExtractionResult:
        if chunk.context_id != batch.context_id or chunk.source_record_id != record.record_id:
            raise ValueError("chunk must be the immutable chunk for the supplied record and context")
        name = getattr(self._extractor, "extractor_name", None)
        version = getattr(self._extractor, "extractor_version", None)
        if (name, version) not in {("deterministic-fixture", "v1"), ("llm-extractor", "v1")}:
            raise ValueError("extraction accepts only deterministic-fixture or llm-extractor v1 output")
        drafts = tuple(self._extractor.extract(record))
        attempt_id = self._attempt_id(chunk, name, version, drafts)
        accepted: list[ExtractedMemoryCandidate] = []
        rejected: list[RejectedExtraction] = []
        for draft in drafts:
            try:
                candidate = self._resolve_draft(batch, record, draft)
                validate_candidate(candidate, batch.records)
            except (ContractValidationError, ValueError) as error:
                rejected.append(RejectedExtraction(draft.candidate_id, str(error), draft))
            else:
                accepted.append(candidate)
        self._store.record(
            attempt_id=attempt_id,
            chunk=chunk,
            extractor_name=name,
            extractor_version=version,
            accepted=tuple(accepted),
            rejected=tuple(rejected),
        )
        return ExtractionResult(attempt_id, tuple(accepted), tuple(rejected))

    @staticmethod
    def _attempt_id(chunk: Chunk, name: str, version: str, drafts: tuple[ExtractionDraft, ...]) -> str:
        logical = f"{chunk.chunk_id}\x00{chunk.content_hash}\x00{name}\x00{version}\x00{drafts!r}".encode()
        return f"extract:{sha256(logical).hexdigest()}"

    @staticmethod
    def _resolve_draft(
        batch: ContextBatch, record: ContextRecord, draft: ExtractionDraft
    ) -> ExtractedMemoryCandidate:
        if draft.memory_type is MemoryType.EPISODIC:
            raise ValueError("episodic is reserved for the immutable raw chunk, not an extracted candidate")
        scope_type = draft.scope_type or (MemoryScope.SESSION if record.session_id else MemoryScope.CHAT)
        scope_id = draft.scope_id or (record.session_id if scope_type is MemoryScope.SESSION else batch.context_id)
        if scope_id is None:
            raise ValueError("session scope requires a session_id")
        return ExtractedMemoryCandidate(
            candidate_id=draft.candidate_id,
            text=draft.text,
            memory_type=draft.memory_type or MemoryType.SEMANTIC,
            scope_type=scope_type,
            scope_id=scope_id,
            source_span=SourceSpan(record.record_id, draft.source_start, draft.source_end),
            confidence=float(draft.confidence),
            temporal=TemporalBounds(record.occurred_at, draft.valid_from, draft.valid_to),
            entities=draft.entities,
            action=draft.action,
            predicate_key=draft.predicate_key,
        )


class ExtractedEntity(BaseModel):
    surface: str
    entity_type: str = "other"

class ExtractedDraft(BaseModel):
    text: str = Field(description="The atomic factual claim")
    confidence: float = Field(description="Confidence from 0.0 to 1.0")
    memory_type: str = Field(description="episodic, semantic, or procedural", default="semantic")
    scope_type: str = Field(description="chat, session, or user", default="session")
    entities: list[ExtractedEntity] = Field(default_factory=list)
    action: str = Field(description="ADD, UPDATE, or DELETE", default="ADD")
    predicate_key: str | None = Field(description="If updating a state, the property/predicate being updated.", default=None)

class ExtractionResponse(BaseModel):
    drafts: list[ExtractedDraft]

class LLMExtractionService:
    def __init__(self, llm_client, store: ExtractionStore, context_repo) -> None:
        self._llm = llm_client
        self._store = store
        self._context_repo = context_repo

    def extract(self, batch: ContextBatch, record: ContextRecord, chunk: Chunk) -> ExtractionResult:
        if chunk.context_id != batch.context_id or chunk.source_record_id != record.record_id:
            raise ValueError("chunk must be the immutable chunk for the supplied record and context")

        # Phase 1: Context Assembly
        buffer_messages = self._context_repo.get_conversation_buffer(batch.context_id, record.session_id, limit=10)
        candidate_facts = self._context_repo.get_top_candidate_facts(batch.context_id, record.content, limit=5)
        
        system_prompt = (
            "You are an expert extraction AI. Extract atomic facts from the current turn.\n"
            "Identify the appropriate action (ADD, UPDATE, DELETE).\n"
            "If updating an existing state or property, specify a predicate_key (e.g. 'pet_breed').\n"
        )
        if buffer_messages:
            system_prompt += "\nRecent conversation:\n" + "\n".join([f"{m['role']}: {m['content']}" for m in buffer_messages])
        if candidate_facts:
            system_prompt += "\nExisting candidate facts:\n" + "\n".join([f"- {f.text}" for f in candidate_facts])

        user_prompt = f"Extract facts from this turn:\n{record.actor_role}: {record.content}"
        
        response = self._llm.structured_completion(system_prompt, user_prompt, ExtractionResponse)
        
        drafts = []
        for d in response.drafts:
            drafts.append(ExtractionDraft(
                candidate_id=str(uuid.uuid4()),
                text=d.text,
                source_start=0,
                source_end=len(record.content),
                confidence=d.confidence,
                memory_type=MemoryType(d.memory_type),
                scope_type=MemoryScope(d.scope_type),
                entities=tuple(EntityCandidate(e.surface, e.entity_type) for e in d.entities),
                action=d.action,
                predicate_key=d.predicate_key
            ))

        name = "llm-extractor"
        version = "v1"
        attempt_id = ExtractionService._attempt_id(chunk, name, version, tuple(drafts))
        accepted: list[ExtractedMemoryCandidate] = []
        rejected: list[RejectedExtraction] = []
        
        for draft in drafts:
            try:
                candidate = ExtractionService._resolve_draft(batch, record, draft)
                validate_candidate(candidate, batch.records)
                accepted.append(candidate)
            except Exception as error:
                rejected.append(RejectedExtraction(draft.candidate_id, str(error), draft))

        self._store.record(
            attempt_id=attempt_id,
            chunk=chunk,
            extractor_name=name,
            extractor_version=version,
            accepted=tuple(accepted),
            rejected=tuple(rejected),
        )
        return ExtractionResult(attempt_id, tuple(accepted), tuple(rejected))
