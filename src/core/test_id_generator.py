import uuid
import pytest
from src.core.id_generator import (
    IdGenerator,
    CollisionRegistry,
    HashCollisionError,
    InvalidSemanticInputError,
)


@pytest.fixture
def id_gen():
    return IdGenerator()


@pytest.fixture
def registry():
    return CollisionRegistry()


def test_deterministic_generation(id_gen):
    s1 = id_gen.session_id("haystack_1", "session_A")
    s2 = id_gen.session_id("haystack_1", "session_A")
    assert s1 == s2
    # Verify valid UUID format
    assert uuid.UUID(s1)


def test_turn_id_generation(id_gen):
    t1 = id_gen.turn_id("haystack_1", "session_A", 0, "user")
    t2 = id_gen.turn_id("haystack_1", "session_A", 0, "user")
    t3 = id_gen.turn_id("haystack_1", "session_A", 1, "user")
    assert t1 == t2
    assert t1 != t3


def test_fact_id_text_hash(id_gen):
    f1 = id_gen.fact_id("haystack_1", "session_A", 0, 0, "User likes apples")
    f2 = id_gen.fact_id("haystack_1", "session_A", 0, 0, "User likes apples")
    f3 = id_gen.fact_id("haystack_1", "session_A", 0, 0, "User likes bananas")
    assert f1 == f2
    assert f1 != f3


def test_entity_id_normalization(id_gen):
    e1 = id_gen.entity_id("haystack_1", "Max", "PET")
    e2 = id_gen.entity_id("haystack_1", " max ", "pet")
    assert e1 == e2


def test_alias_id_normalization(id_gen):
    a1 = id_gen.alias_id("haystack_1", "My Dog", "Max")
    a2 = id_gen.alias_id("haystack_1", "my dog", "max")
    assert a1 == a2


def test_speaker_id(id_gen):
    sp1 = id_gen.speaker_id("User")
    sp2 = id_gen.speaker_id("user")
    assert sp1 == sp2


def test_invalid_string_inputs(id_gen):
    with pytest.raises(InvalidSemanticInputError):
        id_gen.session_id("", "session_A")

    with pytest.raises(InvalidSemanticInputError):
        id_gen.session_id("haystack_1", "   ")

    with pytest.raises(InvalidSemanticInputError):
        id_gen.entity_id("haystack_1", None, "pet")  # type: ignore


def test_invalid_integer_inputs(id_gen):
    with pytest.raises(InvalidSemanticInputError):
        id_gen.turn_id("haystack_1", "session_A", -1, "user")

    with pytest.raises(InvalidSemanticInputError):
        id_gen.fact_id("haystack_1", "session_A", 0, -5, "some text")

    with pytest.raises(InvalidSemanticInputError):
        id_gen.turn_id("haystack_1", "session_A", "0", "user")  # type: ignore


def test_collision_registry_success(registry):
    registry.register("uuid-1234", "path/one")
    # Idempotent re-registration of exact same pair
    registry.register("uuid-1234", "path/one")
    assert registry._registry["uuid-1234"] == "path/one"


def test_collision_registry_collision(registry):
    registry.register("uuid-1234", "path/one")
    with pytest.raises(HashCollisionError) as exc_info:
        registry.register("uuid-1234", "path/two")
    assert "Hash collision detected" in str(exc_info.value)


def test_collision_registry_clear(registry):
    registry.register("uuid-1234", "path/one")
    registry.clear()
    assert len(registry._registry) == 0
    # Should allow registering uuid-1234 with different path now after clear
    registry.register("uuid-1234", "path/two")
