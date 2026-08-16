"""Validation for derived memory candidates and immutable record helpers."""

from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256

from context_memory.core.errors import ContractValidationError
from context_memory.core.models import Chunk, ContextBatch, ContextRecord, ExtractedMemoryCandidate


def content_hash(content: str) -> str:
    """Stable SHA-256 hash of exact UTF-8 source content."""
    return f"sha256:{sha256(content.encode('utf-8')).hexdigest()}"


def chunk_id_for(context_id: str, record_id: str) -> str:
    """Stable logical chunk identity; content changes remain detectable by hash."""
    logical_key = f"{context_id}\x00{record_id}".encode("utf-8")
    return f"chunk:{sha256(logical_key).hexdigest()}"


def chunk_from_record(batch: ContextBatch, record: ContextRecord) -> Chunk:
    """Create the one-record immutable MVP chunk approved by ADR-013."""
    return Chunk(
        chunk_id=chunk_id_for(batch.context_id, record.record_id),
        context_id=batch.context_id,
        source=batch.source,
        source_record_id=record.record_id,
        raw_text=record.content,
        content_hash=content_hash(record.content),
        occurred_at=record.occurred_at,
        session_id=record.session_id,
        actor_id=record.actor_id,
        actor_role=record.actor_role,
        metadata=record.metadata,
    )


def validate_candidate(candidate: ExtractedMemoryCandidate, records: Iterable[ContextRecord]) -> None:
    """Validate spans, confidence, source time, and world-validity ordering."""
    by_id = {record.record_id: record for record in records}
    source = by_id.get(candidate.source_span.source_record_id)
    if source is None:
        raise ContractValidationError("candidate.source_record_id", "does not identify a record in this batch")
    if not candidate.text:
        raise ContractValidationError("candidate.text", "must be non-empty")
    if not candidate.scope_id:
        raise ContractValidationError("candidate.scope_id", "must be non-empty")
    if not 0.0 <= candidate.confidence <= 1.0:
        raise ContractValidationError("candidate.confidence", "must be within 0.0..1.0")
    span = candidate.source_span
    if not 0 <= span.source_start < span.source_end <= len(source.content):
        raise ContractValidationError("candidate.source_span", "must be a non-empty Unicode code-point span")
    if candidate.temporal.observed_at != source.occurred_at:
        raise ContractValidationError("candidate.observed_at", "must equal source record occurred_at")
    valid_from = candidate.temporal.valid_from
    valid_to = candidate.temporal.valid_to
    if valid_from is not None and valid_to is not None and valid_from > valid_to:
        raise ContractValidationError("candidate.valid_from", "must not be later than valid_to")
