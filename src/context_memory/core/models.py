"""Immutable, dependency-free domain models for Generic Context Ingestion v1."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from context_memory.core.errors import ContractValidationError
from context_memory.core.enums import IngestionJobState, MemoryScope, MemoryType

CONTRACT_VERSION = "v1"
TEXT_PLAIN = "text/plain"
EVALUATION_ONLY_FIELDS = frozenset(
    {"question_id", "question_type", "has_answer", "_abs", "answer", "answer_session_ids"}
)


def parse_rfc3339(value: object, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise ContractValidationError(field_name, "must be an RFC 3339 timestamp string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ContractValidationError(field_name, "must be a valid RFC 3339 timestamp") from error
    if parsed.tzinfo is None:
        raise ContractValidationError(field_name, "must include a UTC offset")
    return parsed


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractValidationError(field_name, "must be a non-empty string")
    return value


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name)


def _metadata(value: object, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ContractValidationError(field_name, "must be an object")
    forbidden = EVALUATION_ONLY_FIELDS.intersection(value.keys())
    if forbidden:
        names = ", ".join(sorted(forbidden))
        raise ContractValidationError(field_name, f"contains evaluation-only field(s): {names}")
    return dict(value)


@dataclass(frozen=True)
class SourceDescriptor:
    source_type: str
    source_external_id: str

    def __post_init__(self) -> None:
        _required_text(self.source_type, "batch.source_type")
        _required_text(self.source_external_id, "batch.source_external_id")


@dataclass(frozen=True)
class ContextRecord:
    record_id: str
    occurred_at: datetime
    content: str
    content_type: str = TEXT_PLAIN
    session_id: str | None = None
    actor_id: str | None = None
    actor_role: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required_text(self.record_id, "record.record_id")
        if not isinstance(self.occurred_at, datetime) or self.occurred_at.tzinfo is None:
            raise ContractValidationError("record.occurred_at", "must include a UTC offset")
        if self.content_type != TEXT_PLAIN:
            raise ContractValidationError("record.content_type", "must equal text/plain in v1")
        _required_text(self.content, "record.content")
        _optional_text(self.session_id, "record.session_id")
        _optional_text(self.actor_id, "record.actor_id")
        _optional_text(self.actor_role, "record.actor_role")
        _metadata(self.metadata, "record.metadata")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ContextRecord:
        allowed = {
            "record_id",
            "session_id",
            "actor_id",
            "actor_role",
            "occurred_at",
            "content_type",
            "content",
            "metadata",
        }
        unknown = set(payload).difference(allowed)
        if unknown:
            raise ContractValidationError("record", f"contains unknown field(s): {', '.join(sorted(unknown))}")
        content_type = _required_text(payload.get("content_type"), "record.content_type")
        if content_type != TEXT_PLAIN:
            raise ContractValidationError("record.content_type", "must equal text/plain in v1")
        return cls(
            record_id=_required_text(payload.get("record_id"), "record.record_id"),
            session_id=_optional_text(payload.get("session_id"), "record.session_id"),
            actor_id=_optional_text(payload.get("actor_id"), "record.actor_id"),
            actor_role=_optional_text(payload.get("actor_role"), "record.actor_role"),
            occurred_at=parse_rfc3339(payload.get("occurred_at"), "record.occurred_at"),
            content_type=content_type,
            content=_required_text(payload.get("content"), "record.content"),
            metadata=_metadata(payload.get("metadata"), "record.metadata"),
        )


@dataclass(frozen=True)
class ContextBatch:
    ingestion_id: str
    context_id: str
    source: SourceDescriptor
    records: tuple[ContextRecord, ...]
    contract_version: str = CONTRACT_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.contract_version != CONTRACT_VERSION:
            raise ContractValidationError("batch.contract_version", "must equal v1")
        _required_text(self.ingestion_id, "batch.ingestion_id")
        _required_text(self.context_id, "batch.context_id")
        if not self.records:
            raise ContractValidationError("batch.records", "must be a non-empty array")
        if not all(isinstance(record, ContextRecord) for record in self.records):
            raise ContractValidationError("batch.records", "must contain ContextRecord values")
        record_ids = [record.record_id for record in self.records]
        if len(record_ids) != len(set(record_ids)):
            raise ContractValidationError("batch.records", "contains duplicate record_id")
        _metadata(self.metadata, "batch.metadata")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ContextBatch:
        allowed = {
            "contract_version",
            "ingestion_id",
            "context_id",
            "source_type",
            "source_external_id",
            "records",
            "metadata",
        }
        unknown = set(payload).difference(allowed)
        if unknown:
            raise ContractValidationError("batch", f"contains unknown field(s): {', '.join(sorted(unknown))}")
        version = _required_text(payload.get("contract_version"), "batch.contract_version")
        if version != CONTRACT_VERSION:
            raise ContractValidationError("batch.contract_version", "must equal v1")
        raw_records = payload.get("records")
        if not isinstance(raw_records, list) or not raw_records:
            raise ContractValidationError("batch.records", "must be a non-empty array")
        records = tuple(
            ContextRecord.from_mapping(record)
            if isinstance(record, Mapping)
            else _invalid_record(index)
            for index, record in enumerate(raw_records)
        )
        record_ids = [record.record_id for record in records]
        if len(record_ids) != len(set(record_ids)):
            raise ContractValidationError("batch.records", "contains duplicate record_id")
        return cls(
            contract_version=version,
            ingestion_id=_required_text(payload.get("ingestion_id"), "batch.ingestion_id"),
            context_id=_required_text(payload.get("context_id"), "batch.context_id"),
            source=SourceDescriptor(
                source_type=_required_text(payload.get("source_type"), "batch.source_type"),
                source_external_id=_required_text(
                    payload.get("source_external_id"), "batch.source_external_id"
                ),
            ),
            records=records,
            metadata=_metadata(payload.get("metadata"), "batch.metadata"),
        )


def _invalid_record(index: int) -> ContextRecord:
    raise ContractValidationError(f"batch.records[{index}]", "must be an object")


@dataclass(frozen=True)
class SourceSpan:
    source_record_id: str
    source_start: int
    source_end: int

    def __post_init__(self) -> None:
        _required_text(self.source_record_id, "candidate.source_record_id")
        if not isinstance(self.source_start, int) or not isinstance(self.source_end, int):
            raise ContractValidationError("candidate.source_span", "offsets must be integers")


@dataclass(frozen=True)
class TemporalBounds:
    observed_at: datetime
    valid_from: datetime | None = None
    valid_to: datetime | None = None

    def __post_init__(self) -> None:
        if self.observed_at.tzinfo is None:
            raise ContractValidationError("candidate.observed_at", "must include a UTC offset")
        if self.valid_from is not None and self.valid_from.tzinfo is None:
            raise ContractValidationError("candidate.valid_from", "must include a UTC offset")
        if self.valid_to is not None and self.valid_to.tzinfo is None:
            raise ContractValidationError("candidate.valid_to", "must include a UTC offset")


@dataclass(frozen=True)
class EntityCandidate:
    surface: str
    entity_type: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.surface, "candidate.entities.surface")


@dataclass(frozen=True)
class ExtractedMemoryCandidate:
    candidate_id: str
    text: str
    memory_type: MemoryType
    scope_type: MemoryScope
    scope_id: str
    source_span: SourceSpan
    confidence: float
    temporal: TemporalBounds
    entities: tuple[EntityCandidate, ...] = ()
    action: str = "ADD"
    predicate_key: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.candidate_id, "candidate.candidate_id")
        _required_text(self.text, "candidate.text")
        if not isinstance(self.memory_type, MemoryType):
            raise ContractValidationError("candidate.memory_type", "must be a supported MemoryType")
        if not isinstance(self.scope_type, MemoryScope):
            raise ContractValidationError("candidate.scope_type", "organization scope is not supported in MVP")
        _required_text(self.scope_id, "candidate.scope_id")
        if not isinstance(self.source_span, SourceSpan):
            raise ContractValidationError("candidate.source_span", "must be a SourceSpan")
        if not isinstance(self.temporal, TemporalBounds):
            raise ContractValidationError("candidate.temporal", "must be TemporalBounds")
        if not isinstance(self.confidence, (float, int)) or not 0.0 <= self.confidence <= 1.0:
            raise ContractValidationError("candidate.confidence", "must be within 0.0..1.0")
        if not all(isinstance(entity, EntityCandidate) for entity in self.entities):
            raise ContractValidationError("candidate.entities", "must contain EntityCandidate values")
        if self.action not in ("ADD", "UPDATE", "DELETE"):
            raise ContractValidationError("candidate.action", "must be ADD, UPDATE, or DELETE")
        if self.predicate_key is not None and not isinstance(self.predicate_key, str):
            raise ContractValidationError("candidate.predicate_key", "must be a string")


@dataclass(frozen=True)
class ExtractionDraft:
    """Untrusted deterministic-fixture output; service applies MVP defaults."""

    candidate_id: str
    text: str
    source_start: int
    source_end: int
    confidence: float
    memory_type: MemoryType | None = None
    scope_type: MemoryScope | None = None
    scope_id: str | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    entities: tuple[EntityCandidate, ...] = ()
    action: str = "ADD"
    predicate_key: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.candidate_id, "draft.candidate_id")
        _required_text(self.text, "draft.text")
        if not isinstance(self.source_start, int) or not isinstance(self.source_end, int):
            raise ContractValidationError("draft.source_span", "offsets must be integers")
        if not isinstance(self.confidence, (float, int)) or not 0.0 <= self.confidence <= 1.0:
            raise ContractValidationError("draft.confidence", "must be within 0.0..1.0")
        if self.memory_type is not None and not isinstance(self.memory_type, MemoryType):
            raise ContractValidationError("draft.memory_type", "must be a supported MemoryType")
        if self.scope_type is not None and not isinstance(self.scope_type, MemoryScope):
            raise ContractValidationError("draft.scope_type", "organization scope is not supported in MVP")
        _optional_text(self.scope_id, "draft.scope_id")
        if not all(isinstance(entity, EntityCandidate) for entity in self.entities):
            raise ContractValidationError("draft.entities", "must contain EntityCandidate values")
        if self.action not in ("ADD", "UPDATE", "DELETE"):
            raise ContractValidationError("draft.action", "must be ADD, UPDATE, or DELETE")
        if self.predicate_key is not None and not isinstance(self.predicate_key, str):
            raise ContractValidationError("draft.predicate_key", "must be a string")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    context_id: str
    source: SourceDescriptor
    source_record_id: str
    raw_text: str
    content_hash: str
    occurred_at: datetime
    session_id: str | None = None
    actor_id: str | None = None
    actor_role: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required_text(self.chunk_id, "chunk.chunk_id")
        _required_text(self.context_id, "chunk.context_id")
        if not isinstance(self.source, SourceDescriptor):
            raise ContractValidationError("chunk.source", "must be a SourceDescriptor")
        _required_text(self.source_record_id, "chunk.source_record_id")
        _required_text(self.raw_text, "chunk.raw_text")
        _required_text(self.content_hash, "chunk.content_hash")
        if self.occurred_at.tzinfo is None:
            raise ContractValidationError("chunk.occurred_at", "must include a UTC offset")
        _optional_text(self.session_id, "chunk.session_id")
        _optional_text(self.actor_id, "chunk.actor_id")
        _optional_text(self.actor_role, "chunk.actor_role")
        _metadata(self.metadata, "chunk.metadata")


SUBJECT_KINDS = frozenset({"fact", "chunk"})


@dataclass(frozen=True)
class Embedding:
    """Versioned semantic embedding for a fact or chunk (Milestone 7, ADR-013/ADR-027)."""

    context_id: str
    subject_kind: str
    subject_id: str
    source_chunk_id: str
    model_name: str
    model_version: str
    values: tuple[float, ...]
    embedded_content_hash: str
    is_active: bool = True

    def __post_init__(self) -> None:
        _required_text(self.context_id, "embedding.context_id")
        if self.subject_kind not in SUBJECT_KINDS:
            raise ContractValidationError("embedding.subject_kind", "must be 'fact' or 'chunk'")
        _required_text(self.subject_id, "embedding.subject_id")
        _required_text(self.source_chunk_id, "embedding.source_chunk_id")
        _required_text(self.model_name, "embedding.model_name")
        _required_text(self.model_version, "embedding.model_version")
        if not self.values or not all(isinstance(value, (float, int)) for value in self.values):
            raise ContractValidationError("embedding.values", "must be a non-empty vector of numbers")
        _required_text(self.embedded_content_hash, "embedding.embedded_content_hash")


@dataclass(frozen=True)
class IngestionJob:
    job_id: str
    chunk_id: str
    context_id: str
    state: IngestionJobState
    attempt_count: int = 0
    last_verified_state: IngestionJobState | None = None
    last_error: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.job_id, "job.job_id")
        _required_text(self.chunk_id, "job.chunk_id")
        _required_text(self.context_id, "job.context_id")
        if not isinstance(self.state, IngestionJobState):
            raise ContractValidationError("job.state", "must be a supported IngestionJobState")
        if not isinstance(self.attempt_count, int) or self.attempt_count < 0:
            raise ContractValidationError("job.attempt_count", "must be a non-negative integer")
        if self.last_verified_state is not None and not isinstance(self.last_verified_state, IngestionJobState):
            raise ContractValidationError("job.last_verified_state", "must be a supported IngestionJobState")
