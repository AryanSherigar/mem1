"""Real (non-fake) `Embedder` port implementation (ADR-027, Milestone 7).

Local `sentence-transformers` model — the model named in
`retrieval_architecture.md` §3 — so this path needs no provider credential and
is verifiable without live-model gating, unlike `model_adapters.py`.
"""

from __future__ import annotations

from typing import Any

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DIMENSIONS = 384


class EmbeddingError(RuntimeError):
    """Raised when text cannot be embedded."""


class SentenceTransformerEmbedder:
    """`ingestion.ports.Embedder` backed by a local sentence-transformers model."""

    def __init__(self, model: Any | None = None, model_name: str = DEFAULT_MODEL_NAME, device: str = "cpu") -> None:
        self._model = model
        self.model_name = model_name
        self.model_version = "1"
        self.device = device

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, text: str) -> tuple[float, ...]:
        if not text or not isinstance(text, str):
            raise EmbeddingError("text must be a non-empty string")
        model = self._get_model()
        vector = model.encode(text, normalize_embeddings=True)
        return tuple(float(value) for value in vector)
