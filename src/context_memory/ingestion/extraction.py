"""Deterministic extraction baseline; deliberately not a model-quality claim."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

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
        if (name, version) != ("deterministic-fixture", "v1"):
            raise ValueError("MVP extraction accepts only deterministic-fixture v1 output")
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
        )
