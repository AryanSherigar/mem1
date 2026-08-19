from __future__ import annotations

import unittest

from context_memory.core.resolution import EntityProfile
from context_memory.ingestion.semantic_blocking import SemanticBlockingCache

# Hand-built 2D unit vectors so cosine similarity is exact and legible.
_VECTORS = {
    "max the dog": (1.0, 0.0),
    "the golden retriever": (0.99, 0.14),  # near "max the dog"
    "max the colleague": (0.0, 1.0),  # orthogonal, unrelated
    "sam": (-1.0, 0.0),  # opposite
}


class _FakeEmbedder:
    def embed(self, text: str) -> tuple[float, ...]:
        return _VECTORS.get(text, (0.0, 0.0))


class SemanticBlockingCacheTests(unittest.TestCase):
    def profiles(self) -> tuple[EntityProfile, ...]:
        return (
            EntityProfile(1, "context:1", "max the dog", "pet"),
            EntityProfile(2, "context:1", "max the colleague", "person"),
            EntityProfile(3, "context:2", "max the dog", "pet"),  # same name, other context
        )

    def test_finds_similar_name_above_threshold(self) -> None:
        cache = SemanticBlockingCache(_FakeEmbedder(), threshold=0.75)
        cache.rebuild(self.profiles())
        candidates = cache.candidate_ids("the golden retriever", context_id="context:1")
        self.assertIn(1, candidates)
        self.assertNotIn(2, candidates)

    def test_context_isolation_excludes_other_context_matches(self) -> None:
        cache = SemanticBlockingCache(_FakeEmbedder(), threshold=0.75)
        cache.rebuild(self.profiles())
        candidates = cache.candidate_ids("max the dog", context_id="context:1")
        self.assertIn(1, candidates)
        self.assertNotIn(3, candidates)

    def test_below_threshold_returns_nothing(self) -> None:
        cache = SemanticBlockingCache(_FakeEmbedder(), threshold=0.75)
        cache.rebuild(self.profiles())
        candidates = cache.candidate_ids("sam", context_id="context:1")
        self.assertEqual(candidates, ())

    def test_empty_index_returns_nothing(self) -> None:
        cache = SemanticBlockingCache(_FakeEmbedder(), threshold=0.75)
        cache.rebuild(())
        self.assertEqual(cache.candidate_ids("max the dog", context_id="context:1"), ())

    def test_top_k_is_respected(self) -> None:
        cache = SemanticBlockingCache(_FakeEmbedder(), threshold=0.0)
        cache.rebuild(self.profiles())
        candidates = cache.candidate_ids("max the dog", context_id="context:1", top_k=1)
        self.assertEqual(len(candidates), 1)


if __name__ == "__main__":
    unittest.main()
