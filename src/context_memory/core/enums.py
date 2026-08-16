"""Closed vocabularies approved in Generic Context Ingestion Contract v1."""

from enum import StrEnum


class MemoryType(StrEnum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryScope(StrEnum):
    CHAT = "chat"
    SESSION = "session"
    USER = "user"


class IngestionJobState(StrEnum):
    PENDING_GRAPH = "pending_graph"
    PENDING_EMBEDDINGS = "pending_embeddings"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    RETRYABLE_FAILED = "retryable_failed"
    TERMINAL_FAILED = "terminal_failed"
    MANUAL_REPAIR = "manual_repair"
