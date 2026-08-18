from __future__ import annotations

import unittest

from context_memory.core.errors import ContractValidationError, ImmutableRecordConflictError
from context_memory.core.models import Embedding
from context_memory.ingestion.fakes import InMemoryEmbeddingStore


def _embedding(**overrides: object) -> Embedding:
    fields = dict(
        context_id="context:1",
        subject_kind="fact",
        subject_id="fact:001",
        source_chunk_id="chunk:001",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_version="1",
        values=(0.1, 0.2, 0.3),
        embedded_content_hash="sha256:abc",
    )
    fields.update(overrides)
    return Embedding(**fields)


class EmbeddingModelValidationTests(unittest.TestCase):
    def test_invalid_subject_kind_is_rejected(self) -> None:
        with self.assertRaises(ContractValidationError):
            _embedding(subject_kind="turn")

    def test_empty_values_are_rejected(self) -> None:
        with self.assertRaises(ContractValidationError):
            _embedding(values=())


class InMemoryEmbeddingStoreTests(unittest.TestCase):
    def test_put_is_idempotent_for_same_hash(self) -> None:
        store = InMemoryEmbeddingStore()
        embedding = _embedding()
        self.assertEqual(store.put(embedding), embedding)
        self.assertEqual(store.put(embedding), embedding)

    def test_changed_hash_under_same_version_conflicts(self) -> None:
        store = InMemoryEmbeddingStore()
        store.put(_embedding())
        with self.assertRaises(ImmutableRecordConflictError):
            store.put(_embedding(embedded_content_hash="sha256:different"))

    def test_new_model_version_does_not_conflict(self) -> None:
        store = InMemoryEmbeddingStore()
        store.put(_embedding())
        store.put(_embedding(model_version="2", embedded_content_hash="sha256:different"))

    def test_deactivate_marks_all_versions_inactive(self) -> None:
        store = InMemoryEmbeddingStore()
        store.put(_embedding())
        store.put(_embedding(model_version="2"))
        store.deactivate("context:1", "fact", "fact:001")
        for embedding in store._rows.values():
            self.assertFalse(embedding.is_active)


if __name__ == "__main__":
    unittest.main()
