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


# Happy-path ordering, per docs/ingestion_contract_v1.md §5. A transition to an
# index >= the current one is legal — not just the single documented next step —
# so an orchestrator that always replays a chunk's whole pipeline from the top
# (idempotent at every stage, see ingestion/orchestrator.py) can re-issue a
# state it already reached without that being treated as an illegal regression.
_FORWARD_ORDER: tuple[IngestionJobState, ...] = (
    IngestionJobState.PENDING_GRAPH,
    IngestionJobState.PENDING_EMBEDDINGS,
    IngestionJobState.VERIFYING,
    IngestionJobState.COMPLETED,
)

# Every non-terminal state may fail into one of these; COMPLETED cannot (it is terminal).
_FAILURE_TRANSITIONS: frozenset[IngestionJobState] = frozenset(
    {IngestionJobState.RETRYABLE_FAILED, IngestionJobState.TERMINAL_FAILED, IngestionJobState.MANUAL_REPAIR}
)

_NON_TERMINAL_STATES: frozenset[IngestionJobState] = frozenset(
    {IngestionJobState.PENDING_GRAPH, IngestionJobState.PENDING_EMBEDDINGS, IngestionJobState.VERIFYING}
)


def is_legal_job_transition(
    current: IngestionJobState, new: IngestionJobState, last_verified_state: IngestionJobState | None
) -> bool:
    """Table-driven legality check for `docs/ingestion_contract_v1.md` §5's state contract.

    `last_verified_state` is the non-terminal state a job last successfully reached; it is
    what `retryable_failed`/`manual_repair` resume into, since the contract's "prior
    non-terminal state" / "approved non-terminal state" targets are dynamic, not fixed.
    """
    if current is IngestionJobState.COMPLETED:
        return False  # terminal, no next state
    if current in _NON_TERMINAL_STATES:
        if new in _FAILURE_TRANSITIONS:
            return True
        if new in _FORWARD_ORDER:
            return _FORWARD_ORDER.index(new) >= _FORWARD_ORDER.index(current)
        return False
    if current is IngestionJobState.RETRYABLE_FAILED:
        if new in (IngestionJobState.TERMINAL_FAILED, IngestionJobState.MANUAL_REPAIR):
            return True
        resume_from = last_verified_state or IngestionJobState.PENDING_GRAPH
        return new in _FORWARD_ORDER and _FORWARD_ORDER.index(new) >= _FORWARD_ORDER.index(resume_from)
    if current is IngestionJobState.TERMINAL_FAILED:
        return new is IngestionJobState.MANUAL_REPAIR
    if current is IngestionJobState.MANUAL_REPAIR:
        return new in _NON_TERMINAL_STATES or new is IngestionJobState.TERMINAL_FAILED
    return False
