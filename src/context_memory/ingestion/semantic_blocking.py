"""Non-authoritative Tier-2 semantic-blocking cache for entity resolution (ADR-027).

Feeds `candidate_ids` into `ingestion.resolution.EntityRegistry.resolve`. This
cache never resolves anything by itself: `EntityRegistry.resolve` still
requires an exact/alias hit or a bounded-LLM confirmation before returning a
match (ADR-020). Losing this cache (process restart, empty state) only means
Tier 2 finds nothing to offer the LLM tier — never a correctness loss, only a
recall loss. It is rebuilt from PostgreSQL-backed `EntityProfile` records, not
persisted itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from context_memory.core.resolution import EntityProfile, canonicalize_entity_surface


class TextEmbedder(Protocol):
    def embed(self, text: str) -> tuple[float, ...]: ...


class SemanticBlockingCache:
    """In-process cosine-similarity index over canonical entity names, scoped by context."""

    def __init__(self, embedder: TextEmbedder, threshold: float = 0.75) -> None:
        self._embedder = embedder
        self._threshold = threshold
        self._graph_ids: list[int] = []
        self._context_ids: list[str] = []
        self._vectors: np.ndarray | None = None

    def rebuild(self, profiles: Sequence[EntityProfile]) -> None:
        """Recompute the index from the current active entity set (e.g. per context batch)."""
        self._graph_ids = [profile.graph_id for profile in profiles]
        self._context_ids = [profile.context_id for profile in profiles]
        if not profiles:
            self._vectors = np.empty((0, 0), dtype=np.float32)
            return
        vectors = [np.asarray(self._embedder.embed(profile.canonical_name), dtype=np.float32) for profile in profiles]
        self._vectors = np.stack(vectors)

    def candidate_ids(self, surface: str, context_id: str, top_k: int = 5) -> tuple[int, ...]:
        """Return up to `top_k` graph IDs whose name is similar to `surface`, scoped to `context_id`."""
        if self._vectors is None or self._vectors.size == 0:
            return ()
        canonical = canonicalize_entity_surface(surface)
        query = np.asarray(self._embedder.embed(canonical), dtype=np.float32)
        scores = self._vectors @ query
        ranked = np.argsort(scores)[::-1]
        results: list[int] = []
        for index in ranked:
            if self._context_ids[index] != context_id:
                continue
            if scores[index] < self._threshold:
                break
            results.append(self._graph_ids[index])
            if len(results) >= top_k:
                break
        return tuple(results)
