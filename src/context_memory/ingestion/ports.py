"""Provider and persistence boundaries. Implementations stay outside domain."""

from collections.abc import Sequence
from typing import Protocol

from context_memory.core.models import Chunk, ContextRecord, ExtractionDraft
from context_memory.core.resolution import EntityProfile, FactState, TemporalRelation
from context_memory.core.graph import GraphWritePlan


class Extractor(Protocol):
    """Returns candidates without writing graph or SQL state."""

    def extract(self, record: ContextRecord) -> Sequence[ExtractionDraft]: ...


class Embedder(Protocol):
    """Embeds validated fact text only; model selection is deferred."""

    def embed(self, text: str) -> tuple[float, ...]: ...


class ChunkStore(Protocol):
    """Future SQL-backed immutable chunk store."""

    def put(self, chunk: Chunk) -> Chunk: ...

    def get(self, context_id: str, chunk_id: str) -> Chunk | None: ...


class ExtractionStore(Protocol):
    """Append-only audit trail for deterministic extraction baseline output."""

    def record(
        self,
        *,
        attempt_id: str,
        chunk: Chunk,
        extractor_name: str,
        extractor_version: str,
        accepted: Sequence[object],
        rejected: Sequence[object],
    ) -> None: ...


class GraphIdAllocator(Protocol):
    """Stable graph IDs are allocated in PostgreSQL before HydraDB writes."""

    def allocate_graph_id(self, node_kind: str, context_id: str, logical_key: str) -> int: ...


class EntityResolutionModel(Protocol):
    """Bounded LLM judgment over application-supplied candidate entities."""

    def resolve_entity(
        self, *, context_id: str, surface: str, candidates: Sequence[EntityProfile]
    ) -> int | None: ...


class TemporalUpdateModel(Protocol):
    """LLM classification after deterministic subject/predicate gating."""

    def classify_update(self, *, new_fact: FactState, prior_fact: FactState) -> TemporalRelation: ...


class GraphManifestStore(Protocol):
    """Durable immutable-payload guard before non-atomic graph writes."""

    def register(self, plan: GraphWritePlan) -> None: ...


class GraphTransport(Protocol):
    """Local HydraDB HTTP transport; write identity is diagnostic, not durable."""

    def write(self, cypher: str, rows: Sequence[dict[str, object]], idempotency_key: str) -> str | None: ...

    def read(self, cypher: str, parameters: dict[str, object], bookmark: str | None) -> Sequence[dict[str, object]]: ...
