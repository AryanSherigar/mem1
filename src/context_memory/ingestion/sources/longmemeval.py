"""LongMemEval source adapter. Benchmark structure never reaches core models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from context_memory.core.errors import ContractValidationError
from context_memory.core.models import ContextBatch, ContextRecord, SourceDescriptor

LONGMEMEVAL_TIMESTAMP_FORMAT = "%Y/%m/%d (%a) %H:%M"
SOURCE_TYPE = "longmemeval"


def parse_longmemeval_timestamp(value: object, field_name: str) -> datetime:
    """Normalize known timezone-free benchmark timestamps onto one UTC ordering basis."""
    if not isinstance(value, str):
        raise ContractValidationError(field_name, "must be a LongMemEval timestamp string")
    try:
        parsed = datetime.strptime(value, LONGMEMEVAL_TIMESTAMP_FORMAT)
    except ValueError as error:
        raise ContractValidationError(
            field_name, f"must match {LONGMEMEVAL_TIMESTAMP_FORMAT!r}"
        ) from error
    return parsed.replace(tzinfo=timezone.utc)


def adapt_longmemeval_instance(payload: Mapping[str, Any], ingestion_id: str) -> ContextBatch:
    """Map one benchmark instance into one canonical, chronologically ordered batch."""
    question_id = _text(payload.get("question_id"), "longmemeval.question_id")
    session_ids = _list(payload.get("haystack_session_ids"), "longmemeval.haystack_session_ids")
    session_dates = _list(payload.get("haystack_dates"), "longmemeval.haystack_dates")
    sessions = _list(payload.get("haystack_sessions"), "longmemeval.haystack_sessions")
    if not len(session_ids) == len(session_dates) == len(sessions):
        raise ContractValidationError("longmemeval", "session IDs, dates, and sessions must have equal lengths")
    parsed_sessions: list[tuple[datetime, int, str, str, str, list[Any]]] = []
    for index, (raw_session_id, raw_date, raw_session) in enumerate(zip(session_ids, session_dates, sessions)):
        source_session_id = _text(raw_session_id, f"longmemeval.haystack_session_ids[{index}]")
        session_id = f"{source_session_id}:source-{index:04d}"
        parsed_sessions.append(
            (
                parse_longmemeval_timestamp(raw_date, f"longmemeval.haystack_dates[{index}]"),
                index,
                session_id,
                source_session_id,
                _text(raw_date, f"longmemeval.haystack_dates[{index}]"),
                _list(raw_session, f"longmemeval.haystack_sessions[{index}]"),
            )
        )

    records: list[ContextRecord] = []
    skipped_empty_turn_count = 0
    for occurred_at, source_index, session_id, source_session_id, raw_date, turns in sorted(parsed_sessions):
        for turn_index, turn in enumerate(turns):
            if not isinstance(turn, Mapping):
                raise ContractValidationError(
                    f"longmemeval.haystack_sessions[{source_index}][{turn_index}]", "must be an object"
                )
            role = _text(turn.get("role"), f"longmemeval.turn[{source_index}][{turn_index}].role")
            if role not in {"user", "assistant"}:
                raise ContractValidationError(
                    f"longmemeval.turn[{source_index}][{turn_index}].role", "must be user or assistant"
                )
            content = turn.get("content")
            if content == "":
                skipped_empty_turn_count += 1
                continue
            records.append(
                ContextRecord(
                    record_id=f"{session_id}:turn-{turn_index:04d}",
                    session_id=session_id,
                    actor_role=role,
                    occurred_at=occurred_at,
                    content_type="text/plain",
                    content=_text(content, f"longmemeval.turn[{source_index}][{turn_index}].content"),
                    metadata={
                        "source_session_id": source_session_id,
                        "source_occurred_at": raw_date,
                        "source_time_precision": "minute",
                        "source_timezone": "unknown",
                        "timestamp_normalization": "assumed_utc_for_benchmark_ordering_only",
                        "source_index": source_index,
                        "turn_index": turn_index,
                    },
                )
            )
    if not records:
        raise ContractValidationError("longmemeval.haystack_sessions", "must contain at least one turn")
    return ContextBatch(
        ingestion_id=_text(ingestion_id, "ingestion_id"),
        context_id=f"longmemeval:{question_id}",
        source=SourceDescriptor(source_type=SOURCE_TYPE, source_external_id=question_id),
        records=tuple(records),
        metadata={"source_empty_turn_count": skipped_empty_turn_count},
    )


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractValidationError(field_name, "must be a non-empty string")
    return value


def _list(value: object, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContractValidationError(field_name, "must be an array")
    return value
