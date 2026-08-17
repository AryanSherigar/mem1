import pytest
import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.id_generator import IdGenerator
from src.db.embedding_index import EmbeddingIndex, EntityNameIndex


@pytest.fixture(scope="module")
def shared_model():
    """Module-scoped shared SentenceTransformer model instance for test speed."""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")


@pytest.fixture
def id_gen():
    """IdGenerator instance."""
    return IdGenerator()


class TestEmbeddingIndex:
    """Test suite for EmbeddingIndex."""

    def test_add_and_search_basic(self, shared_model, id_gen):
        index = EmbeddingIndex(model=shared_model)
        hid = "haystack_001"
        sid = "sess_1"

        fid1 = id_gen.fact_id(hid, sid, 0, 0, "User lives in New York City.")
        fid2 = id_gen.fact_id(hid, sid, 0, 1, "User loves eating pepperoni pizza.")
        fid3 = id_gen.fact_id(hid, sid, 1, 0, "User owns a golden retriever dog named Max.")

        index.add(fid1, "User lives in New York City.", hid)
        index.add(fid2, "User loves eating pepperoni pizza.", hid)
        index.add(fid3, "User owns a golden retriever dog named Max.", hid)

        assert index.size() == 3
        assert index.size(hid) == 3

        # Query for location/city
        results = index.search("Where does the user live?", haystack_id=hid, top_k=2)
        assert len(results) == 2
        top_fid, top_score = results[0]
        assert top_fid == fid1
        assert top_score > 0.4

        # Query for dog/pet
        results = index.search("What pet does the user have?", haystack_id=hid, top_k=1)
        assert len(results) == 1
        assert results[0][0] == fid3

    def test_haystack_isolation(self, shared_model, id_gen):
        index = EmbeddingIndex(model=shared_model)
        hid_a = "haystack_A"
        hid_b = "haystack_B"

        fid_a = id_gen.fact_id(hid_a, "sess_1", 0, 0, "User enjoys playing tennis.")
        fid_b = id_gen.fact_id(hid_b, "sess_1", 0, 0, "User enjoys playing basketball.")

        index.add(fid_a, "User enjoys playing tennis.", hid_a)
        index.add(fid_b, "User enjoys playing basketball.", hid_b)

        assert index.size(hid_a) == 1
        assert index.size(hid_b) == 1

        results_a = index.search("sports", haystack_id=hid_a)
        assert len(results_a) == 1
        assert results_a[0][0] == fid_a

        results_b = index.search("sports", haystack_id=hid_b)
        assert len(results_b) == 1
        assert results_b[0][0] == fid_b

        # Querying an unknown haystack returns empty
        results_c = index.search("sports", haystack_id="unknown_haystack")
        assert len(results_c) == 0

    def test_remove_superseded_fact(self, shared_model, id_gen):
        index = EmbeddingIndex(model=shared_model)
        hid = "haystack_001"

        fid_old = id_gen.fact_id(hid, "sess_1", 0, 0, "User works as a software engineer at Company A.")
        fid_new = id_gen.fact_id(hid, "sess_2", 0, 0, "User works as a product manager at Company B.")

        index.add(fid_old, "User works as a software engineer at Company A.", hid)
        index.add(fid_new, "User works as a product manager at Company B.", hid)

        assert index.size(hid) == 2

        # Evict old superseded fact
        index.remove(fid_old)
        assert index.size(hid) == 1

        results = index.search("Where does the user work?", haystack_id=hid)
        assert len(results) == 1
        assert results[0][0] == fid_new

    def test_add_vector_hydration(self, shared_model, id_gen):
        index = EmbeddingIndex(model=shared_model)
        hid = "haystack_001"
        fid = id_gen.fact_id(hid, "sess_1", 0, 0, "User likes green tea.")

        # Compute raw vector externally
        vec = shared_model.encode("User likes green tea.")
        index.add_vector(fid, vec, hid)

        assert index.size(hid) == 1
        results = index.search("tea preferences", haystack_id=hid)
        assert len(results) == 1
        assert results[0][0] == fid

    def test_clear_and_validation(self, shared_model):
        index = EmbeddingIndex(model=shared_model)
        index.add("fid_1", "Test fact", "hid_1")
        assert index.size() == 1

        index.clear()
        assert index.size() == 0
        assert len(index.search("Test", haystack_id="hid_1")) == 0

        with pytest.raises(ValueError):
            index.add("", "text", "hid")

        with pytest.raises(ValueError):
            index.search("text", haystack_id="")


class TestEntityNameIndex:
    """Test suite for EntityNameIndex."""

    def test_add_and_find_candidates(self, shared_model, id_gen):
        index = EntityNameIndex(model=shared_model)
        hid = "haystack_001"

        eid1 = id_gen.entity_id(hid, "max", "pet")
        eid2 = id_gen.entity_id(hid, "maxwell", "person")
        eid3 = id_gen.entity_id(hid, "san francisco", "place")

        index.add(eid1, "Max", "pet", hid)
        index.add(eid2, "Maxwell", "person", hid)
        index.add(eid3, "San Francisco", "place", hid)

        assert index.size(hid) == 3

        # Search candidates for mentions of "Max" with entity_type="pet"
        candidates = index.find_candidates("Max", entity_type="pet", haystack_id=hid, threshold=0.7)
        assert len(candidates) >= 1

        # Matching entity type should be ranked first
        top_cand = candidates[0]
        assert top_cand["entity_id"] == eid1
        assert top_cand["entity_type"] == "pet"
        assert top_cand["type_match"] is True

    def test_threshold_filtering(self, shared_model, id_gen):
        index = EntityNameIndex(model=shared_model)
        hid = "haystack_001"

        eid1 = id_gen.entity_id(hid, "golden retriever", "pet")
        index.add(eid1, "Golden Retriever", "pet", hid)

        # High threshold for completely unrelated query returns empty
        candidates = index.find_candidates("Quantum Mechanics", entity_type="topic", haystack_id=hid, threshold=0.75)
        assert len(candidates) == 0

    def test_remove_entity(self, shared_model, id_gen):
        index = EntityNameIndex(model=shared_model)
        hid = "haystack_001"

        eid = id_gen.entity_id(hid, "buddy", "pet")
        index.add(eid, "Buddy", "pet", hid)
        assert index.size(hid) == 1

        index.remove(eid)
        assert index.size(hid) == 0
        candidates = index.find_candidates("Buddy", entity_type="pet", haystack_id=hid, threshold=0.5)
        assert len(candidates) == 0
