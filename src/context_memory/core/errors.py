"""Typed errors exposed by the deterministic domain boundary."""


class ContractValidationError(ValueError):
    """Raised when data violates Generic Context Ingestion Contract v1."""

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        super().__init__(f"{field}: {message}")


class ImmutableRecordConflictError(ValueError):
    """Raised when one logical record/chunk is replayed with new content."""


class GraphPayloadConflictError(ValueError):
    """Raised when one logical graph record is replayed with changed payload."""
