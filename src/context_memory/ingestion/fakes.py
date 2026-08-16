"""Deterministic fakes for unit tests; no provider, database, or graph calls."""

from collections.abc import Mapping, Sequence
from hashlib import sha256

from context_memory.core.errors import GraphPayloadConflictError, ImmutableRecordConflictError
from context_memory.core.graph import GraphNode, GraphWritePlan
from context_memory.core.models import Chunk, ContextRecord, ExtractionDraft
from context_memory.core.resolution import EntityProfile, FactState, TemporalRelation


class DeterministicExtractor:
    extractor_name = "deterministic-fixture"
    extractor_version = "v1"

    def __init__(self, candidates_by_record: Mapping[str, Sequence[ExtractionDraft]] | None = None) -> None:
        self._candidates_by_record = dict(candidates_by_record or {})

    def extract(self, record: ContextRecord) -> tuple[ExtractionDraft, ...]:
        return tuple(self._candidates_by_record.get(record.record_id, ()))


class DeterministicEmbedder:
    def __init__(self, dimensions: int = 8) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed(self, text: str) -> tuple[float, ...]:
        digest = sha256(text.encode("utf-8")).digest()
        return tuple(byte / 255.0 for byte in digest[: self.dimensions])


class InMemoryChunkStore:
    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}

    def put(self, chunk: Chunk) -> Chunk:
        existing = self._chunks.get(chunk.chunk_id)
        if existing is not None and existing != chunk:
            raise ImmutableRecordConflictError(f"chunk_id {chunk.chunk_id} has different immutable content")
        self._chunks[chunk.chunk_id] = chunk
        return chunk

    def get(self, context_id: str, chunk_id: str) -> Chunk | None:
        chunk = self._chunks.get(chunk_id)
        return chunk if chunk is not None and chunk.context_id == context_id else None


class InMemoryGraphIdAllocator:
    def __init__(self) -> None:
        self._ids: dict[tuple[str, str, str], int] = {}

    def allocate_graph_id(self, node_kind: str, context_id: str, logical_key: str) -> int:
        key = (node_kind, context_id, logical_key)
        if key not in self._ids:
            self._ids[key] = len(self._ids)
        return self._ids[key]


class DeterministicEntityResolutionModel:
    """Fixture-only bounded selection. Never calls a provider."""

    def __init__(self, selections: Mapping[str, int | None] | None = None) -> None:
        self._selections = dict(selections or {})
        self.calls: list[tuple[str, str, tuple[int, ...]]] = []

    def resolve_entity(self, *, context_id: str, surface: str, candidates: Sequence[EntityProfile]) -> int | None:
        self.calls.append((context_id, surface, tuple(profile.graph_id for profile in candidates)))
        return self._selections.get(surface)


class DeterministicTemporalUpdateModel:
    """Fixture-only update judgment keyed by new/prior fact IDs."""

    def __init__(self, relations: Mapping[tuple[str, str], TemporalRelation] | None = None) -> None:
        self._relations = dict(relations or {})
        self.calls: list[tuple[str, str]] = []

    def classify_update(self, *, new_fact: FactState, prior_fact: FactState) -> TemporalRelation:
        key = (new_fact.fact_id, prior_fact.fact_id)
        self.calls.append(key)
        return self._relations.get(key, TemporalRelation.UNRESOLVED)


class InMemoryGraphManifestStore:
    def __init__(self) -> None:
        self._records: dict[tuple[str, str, str], tuple[int, str]] = {}

    def register(self, plan: GraphWritePlan) -> None:
        for record in plan.records():
            kind = "node" if isinstance(record, GraphNode) else "relationship"
            key = (kind, plan.context_id, record.logical_key)
            value = (record.graph_id, plan.payload_hash(record))
            existing = self._records.get(key)
            if existing is not None and existing != value:
                raise GraphPayloadConflictError(f"{kind} {record.logical_key} has a different immutable graph payload")
            self._records[key] = value


class RecordingGraphTransport:
    def __init__(self) -> None:
        self.writes: list[tuple[str, list[dict[str, object]], str]] = []
        self.reads: list[tuple[str, dict[str, object], str | None]] = []
        self.read_rows: list[dict[str, object]] = []

    def write(self, cypher: str, rows: Sequence[dict[str, object]], idempotency_key: str) -> str:
        self.writes.append((cypher, list(rows), idempotency_key))
        return f"bookmark-{len(self.writes)}"

    def read(self, cypher: str, parameters: dict[str, object], bookmark: str | None) -> Sequence[dict[str, object]]:
        self.reads.append((cypher, parameters, bookmark))
        return self.read_rows
