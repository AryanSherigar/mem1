"""Interactive chat source adapter. Maps single turns to the generic ingestion contract."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Mapping

from context_memory.core.errors import ContractValidationError
from context_memory.core.models import ContextBatch, ContextRecord, SourceDescriptor

SOURCE_TYPE = "chat"

def adapt_chat_turn(payload: Mapping[str, Any], ingestion_id: str) -> ContextBatch:
    """Map a single interactive chat turn into a ContextBatch."""
    context_id = _text(payload.get("context_id"), "chat.context_id")
    session_id = _text(payload.get("session_id"), "chat.session_id")
    role = _text(payload.get("role"), "chat.role")
    
    if role not in {"user", "assistant"}:
        raise ContractValidationError("chat.role", "must be user or assistant")
        
    content = _text(payload.get("content"), "chat.content")
    
    timestamp = payload.get("timestamp")
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    elif not isinstance(timestamp, datetime):
        raise ContractValidationError("chat.timestamp", "must be a datetime object")
        
    record_id = f"rec-{uuid.uuid4()}"
    record = ContextRecord(
        record_id=record_id,
        session_id=session_id,
        actor_role=role,
        actor_id=role,
        occurred_at=timestamp,
        content_type="text/plain",
        content=content,
        metadata={
            "source_type": "interactive_chat"
        },
    )
    
    return ContextBatch(
        ingestion_id=_text(ingestion_id, "ingestion_id"),
        context_id=context_id,
        source=SourceDescriptor(source_type=SOURCE_TYPE, source_external_id="live"),
        records=(record,),
        metadata={},
    )


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractValidationError(field_name, "must be a non-empty string")
    return value
