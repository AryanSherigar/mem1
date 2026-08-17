import hashlib
import uuid
from typing import Dict, Any


class IdGeneratorError(Exception):
    """Base exception for all ID generation and collision errors."""
    pass


class HashCollisionError(IdGeneratorError):
    """Raised when two distinct semantic paths map to the exact same UUID."""
    pass


class InvalidSemanticInputError(IdGeneratorError, ValueError):
    """Raised when an input parameter is None, empty, invalid type, or out of bounds."""
    pass


class IdGenerator:
    """Content-addressable, deterministic UUID generator for HydraDB nodes."""

    @staticmethod
    def _validate_str(value: str, field_name: str) -> str:
        if not isinstance(value, str):
            raise InvalidSemanticInputError(f"Field '{field_name}' must be a string, got {type(value).__name__}")
        trimmed = value.strip()
        if not trimmed:
            raise InvalidSemanticInputError(f"Field '{field_name}' cannot be empty or blank")
        return trimmed

    @staticmethod
    def _validate_non_negative_int(value: int, field_name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise InvalidSemanticInputError(f"Field '{field_name}' must be an integer, got {type(value).__name__}")
        if value < 0:
            raise InvalidSemanticInputError(f"Field '{field_name}' must be non-negative, got {value}")
        return value

    @staticmethod
    def _hash_to_uuid(semantic_path: str) -> str:
        if not semantic_path or not isinstance(semantic_path, str):
            raise InvalidSemanticInputError("semantic_path must be a non-empty string")
        digest = hashlib.sha256(semantic_path.encode("utf-8")).digest()
        return str(uuid.UUID(bytes=digest[:16]))

    def session_id(self, haystack_id: str, session_id: str) -> str:
        clean_hid = self._validate_str(haystack_id, "haystack_id")
        clean_sid = self._validate_str(session_id, "session_id")
        path = f"session:{clean_hid}:{clean_sid}"
        return self._hash_to_uuid(path)

    def turn_id(self, haystack_id: str, session_id: str, turn_index: int, role: str) -> str:
        clean_hid = self._validate_str(haystack_id, "haystack_id")
        clean_sid = self._validate_str(session_id, "session_id")
        clean_turn_idx = self._validate_non_negative_int(turn_index, "turn_index")
        clean_role = self._validate_str(role, "role").lower()
        path = f"turn:{clean_hid}:{clean_sid}:{clean_turn_idx}:{clean_role}"
        return self._hash_to_uuid(path)

    def fact_id(self, haystack_id: str, session_id: str, turn_index: int, fact_index: int, fact_text: str) -> str:
        clean_hid = self._validate_str(haystack_id, "haystack_id")
        clean_sid = self._validate_str(session_id, "session_id")
        clean_turn_idx = self._validate_non_negative_int(turn_index, "turn_index")
        clean_fact_idx = self._validate_non_negative_int(fact_index, "fact_index")
        clean_text = self._validate_str(fact_text, "fact_text")
        
        text_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()[:8]
        path = f"fact:{clean_hid}:{clean_sid}:{clean_turn_idx}:{clean_fact_idx}:{text_hash}"
        return self._hash_to_uuid(path)

    def entity_id(self, haystack_id: str, canonical_name: str, entity_type: str) -> str:
        clean_hid = self._validate_str(haystack_id, "haystack_id")
        clean_name = self._validate_str(canonical_name, "canonical_name").lower()
        clean_type = self._validate_str(entity_type, "entity_type").lower()
        path = f"entity:{clean_hid}:{clean_name}:{clean_type}"
        return self._hash_to_uuid(path)

    def alias_id(self, haystack_id: str, canonical_alias: str, parent_entity_canonical: str) -> str:
        clean_hid = self._validate_str(haystack_id, "haystack_id")
        clean_alias = self._validate_str(canonical_alias, "canonical_alias").lower()
        clean_parent = self._validate_str(parent_entity_canonical, "parent_entity_canonical").lower()
        path = f"alias:{clean_hid}:{clean_alias}:{clean_parent}"
        return self._hash_to_uuid(path)

    def speaker_id(self, role: str) -> str:
        clean_role = self._validate_str(role, "role").lower()
        path = f"speaker:{clean_role}"
        return self._hash_to_uuid(path)


class CollisionRegistry:
    """Tracks all generated IDs to detect hash collisions at ingestion time."""

    def __init__(self) -> None:
        self._registry: Dict[str, str] = {}  # id -> semantic_path

    def register(self, node_id: str, semantic_path: str) -> None:
        if not isinstance(node_id, str) or not node_id.strip():
            raise InvalidSemanticInputError("node_id must be a non-empty string")
        if not isinstance(semantic_path, str) or not semantic_path.strip():
            raise InvalidSemanticInputError("semantic_path must be a non-empty string")

        clean_id = node_id.strip()
        clean_path = semantic_path.strip()

        if clean_id in self._registry:
            existing_path = self._registry[clean_id]
            if existing_path != clean_path:
                raise HashCollisionError(
                    f"Hash collision detected! ID {clean_id} maps to both "
                    f"'{existing_path}' and '{clean_path}'"
                )
        self._registry[clean_id] = clean_path

    def clear(self) -> None:
        self._registry.clear()
