from __future__ import annotations

import os
import unittest

from context_memory.ingestion.embedding import EmbeddingError, SentenceTransformerEmbedder


class _FakeModel:
    """Deterministic stand-in for SentenceTransformer; no network/model download."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode(self, text: str, normalize_embeddings: bool = True):
        self.calls.append(text)
        # Distinct, stable 4-dim vectors keyed by first character.
        base = float(ord(text[0]) % 7) if text else 0.0
        vector = [base, base / 2, base / 3, 1.0]
        if normalize_embeddings:
            norm = sum(v * v for v in vector) ** 0.5
            vector = [v / norm for v in vector] if norm else vector
        return vector


class SentenceTransformerEmbedderTests(unittest.TestCase):
    def test_embed_returns_float_tuple(self) -> None:
        embedder = SentenceTransformerEmbedder(model=_FakeModel())
        vector = embedder.embed("hello world")
        self.assertIsInstance(vector, tuple)
        self.assertTrue(all(isinstance(v, float) for v in vector))
        self.assertEqual(len(vector), 4)

    def test_model_is_lazy_and_reused(self) -> None:
        fake = _FakeModel()
        embedder = SentenceTransformerEmbedder(model=fake)
        embedder.embed("first")
        embedder.embed("second")
        self.assertEqual(fake.calls, ["first", "second"])

    def test_empty_text_is_rejected(self) -> None:
        embedder = SentenceTransformerEmbedder(model=_FakeModel())
        with self.assertRaises(EmbeddingError):
            embedder.embed("")

    def test_non_string_text_is_rejected(self) -> None:
        embedder = SentenceTransformerEmbedder(model=_FakeModel())
        with self.assertRaises(EmbeddingError):
            embedder.embed(None)  # type: ignore[arg-type]


@unittest.skipUnless(
    os.environ.get("CONTEXT_MEMORY_RUN_EMBEDDING_SMOKE"),
    "opt-in: downloads/loads the real sentence-transformers model locally",
)
class RealSentenceTransformerSmokeTest(unittest.TestCase):
    """No provider credential required, unlike the LLM-backed adapters."""

    def test_real_model_embeds_text(self) -> None:
        embedder = SentenceTransformerEmbedder()
        vector = embedder.embed("Parth joined Clinsta Labs on 2024-01-15.")
        self.assertEqual(len(vector), 384)
        norm = sum(v * v for v in vector) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
