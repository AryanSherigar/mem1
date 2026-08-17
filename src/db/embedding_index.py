import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sentence_transformers import SentenceTransformer


class EmbeddingIndexError(Exception):
    """Base exception for EmbeddingIndex errors."""
    pass


class EmbeddingIndex:
    """
    In-process vector store for Fact embeddings.
    Maps fact_id (str UUID) -> L2-normalized numpy embedding (384-dim).
    Partitioned by haystack_id for multi-tenant isolation.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        self._model = model
        self.model_name = model_name
        self.device = device

        self._embeddings: Dict[str, np.ndarray] = {}
        self._haystacks: Dict[str, str] = {}

        self.fact_ids: List[str] = []
        self.fact_haystacks: List[str] = []
        self.matrix: Optional[np.ndarray] = None
        self._dirty: bool = True

    def _get_model(self) -> Any:
        """Lazy loader for sentence-transformer encoder model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text into an L2-normalized 1D float32 numpy vector."""
        if not text or not isinstance(text, str):
            raise ValueError("text must be a non-empty string")
        model = self._get_model()
        vector = model.encode(text, normalize_embeddings=True)
        return np.asarray(vector, dtype=np.float32)

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Ensure vector is 1D float32 and L2-normalized."""
        vec = np.asarray(vector, dtype=np.float32).flatten()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def add(self, fact_id: str, text: str, haystack_id: str) -> None:
        """Encode and add a Fact embedding keyed by fact_id and haystack_id."""
        if not fact_id or not isinstance(fact_id, str):
            raise ValueError("fact_id must be a non-empty string")
        if not haystack_id or not isinstance(haystack_id, str):
            raise ValueError("haystack_id must be a non-empty string")

        vec = self.encode_text(text)
        self._embeddings[fact_id] = vec
        self._haystacks[fact_id] = haystack_id
        self._dirty = True

    def add_vector(self, fact_id: str, vector: np.ndarray, haystack_id: str) -> None:
        """Add a pre-computed embedding vector for a Fact."""
        if not fact_id or not isinstance(fact_id, str):
            raise ValueError("fact_id must be a non-empty string")
        if not haystack_id or not isinstance(haystack_id, str):
            raise ValueError("haystack_id must be a non-empty string")

        norm_vec = self._normalize_vector(vector)
        self._embeddings[fact_id] = norm_vec
        self._haystacks[fact_id] = haystack_id
        self._dirty = True

    def remove(self, fact_id: str) -> None:
        """Remove a fact embedding (e.g., when superseded)."""
        existed = (self._embeddings.pop(fact_id, None) is not None)
        self._haystacks.pop(fact_id, None)
        if existed:
            self._dirty = True

    def _rebuild(self) -> None:
        """Rebuild the cached numpy matrix for batch matrix multiplication."""
        self.fact_ids = list(self._embeddings.keys())
        self.fact_haystacks = [self._haystacks[fid] for fid in self.fact_ids]
        if self.fact_ids:
            self.matrix = np.stack([self._embeddings[fid] for fid in self.fact_ids])
        else:
            self.matrix = np.empty((0, 384), dtype=np.float32)
        self._dirty = False

    def search(
        self,
        query_text: str,
        haystack_id: str,
        top_k: int = 10,
        include_non_current: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Cosine similarity search over facts strictly filtered to haystack_id.
        Returns list of (fact_id, score) sorted descending by similarity score.
        """
        if not haystack_id or not isinstance(haystack_id, str):
            raise ValueError("haystack_id must be a non-empty string")

        if self._dirty:
            self._rebuild()

        if self.matrix is None or len(self.fact_ids) == 0:
            return []

        mask = [h == haystack_id for h in self.fact_haystacks]
        if not any(mask):
            return []

        scoped_ids = [self.fact_ids[i] for i, m in enumerate(mask) if m]
        scoped_matrix = self.matrix[mask]

        query_vec = self.encode_text(query_text)
        scores = scoped_matrix @ query_vec

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(scoped_ids[i], float(scores[i])) for i in top_indices]

    def clear(self) -> None:
        """Clear all stored embeddings and reset state."""
        self._embeddings.clear()
        self._haystacks.clear()
        self.fact_ids.clear()
        self.fact_haystacks.clear()
        self.matrix = None
        self._dirty = True

    def size(self, haystack_id: Optional[str] = None) -> int:
        """Return count of indexed facts, optionally filtered by haystack_id."""
        if haystack_id is not None:
            return sum(1 for h in self._haystacks.values() if h == haystack_id)
        return len(self._embeddings)


class EntityNameIndex:
    """
    In-process vector index over entity display names for Tier 2 semantic blocking.
    Partitioned by haystack_id.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        self._model = model
        self.model_name = model_name
        self.device = device

        self._embeddings: Dict[str, np.ndarray] = {}
        self._haystacks: Dict[str, str] = {}
        self._types: Dict[str, str] = {}
        self._names: Dict[str, str] = {}

    def _get_model(self) -> Any:
        """Lazy loader for sentence-transformer encoder model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text into an L2-normalized 1D float32 numpy vector."""
        if not text or not isinstance(text, str):
            raise ValueError("text must be a non-empty string")
        model = self._get_model()
        vector = model.encode(text, normalize_embeddings=True)
        return np.asarray(vector, dtype=np.float32)

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Ensure vector is 1D float32 and L2-normalized."""
        vec = np.asarray(vector, dtype=np.float32).flatten()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def add(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        haystack_id: str,
    ) -> None:
        """Index an entity display name for semantic blocking."""
        if not entity_id or not isinstance(entity_id, str):
            raise ValueError("entity_id must be a non-empty string")
        if not haystack_id or not isinstance(haystack_id, str):
            raise ValueError("haystack_id must be a non-empty string")
        if not name or not isinstance(name, str):
            raise ValueError("name must be a non-empty string")

        vec = self.encode_text(name)
        self._embeddings[entity_id] = vec
        self._haystacks[entity_id] = haystack_id
        self._types[entity_id] = entity_type
        self._names[entity_id] = name

    def add_vector(
        self,
        entity_id: str,
        vector: np.ndarray,
        name: str,
        entity_type: str,
        haystack_id: str,
    ) -> None:
        """Add pre-computed embedding vector for an entity."""
        if not entity_id or not isinstance(entity_id, str):
            raise ValueError("entity_id must be a non-empty string")
        if not haystack_id or not isinstance(haystack_id, str):
            raise ValueError("haystack_id must be a non-empty string")

        norm_vec = self._normalize_vector(vector)
        self._embeddings[entity_id] = norm_vec
        self._haystacks[entity_id] = haystack_id
        self._types[entity_id] = entity_type
        self._names[entity_id] = name

    def remove(self, entity_id: str) -> None:
        """Remove an entity from the index (e.g. when merged or deleted)."""
        self._embeddings.pop(entity_id, None)
        self._haystacks.pop(entity_id, None)
        self._types.pop(entity_id, None)
        self._names.pop(entity_id, None)

    def find_candidates(
        self,
        query_name: str,
        entity_type: str,
        haystack_id: str,
        top_k: int = 5,
        threshold: float = 0.75,
    ) -> List[Dict[str, Any]]:
        """
        Return candidate entities with similarity >= threshold, scoped to haystack_id.
        Sorted by type_match (True first), then similarity descending.
        """
        if not haystack_id or not isinstance(haystack_id, str):
            raise ValueError("haystack_id must be a non-empty string")

        query_vec = self.encode_text(query_name)

        candidates = []
        for eid, vec in self._embeddings.items():
            if self._haystacks[eid] != haystack_id:
                continue

            sim = float(np.dot(query_vec, vec))
            if sim >= threshold:
                candidates.append({
                    "entity_id": eid,
                    "name": self._names[eid],
                    "entity_type": self._types[eid],
                    "similarity": sim,
                    "type_match": self._types[eid] == entity_type,
                })

        candidates.sort(key=lambda c: (-c["type_match"], -c["similarity"]))
        return candidates[:top_k]

    def clear(self) -> None:
        """Clear all stored entity name embeddings."""
        self._embeddings.clear()
        self._haystacks.clear()
        self._types.clear()
        self._names.clear()

    def size(self, haystack_id: Optional[str] = None) -> int:
        """Return count of indexed entities, optionally filtered by haystack_id."""
        if haystack_id is not None:
            return sum(1 for h in self._haystacks.values() if h == haystack_id)
        return len(self._embeddings)
