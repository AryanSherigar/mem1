"""Validated, scalar-only records for the checked-in HydraDB OpenCypher surface."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping

from context_memory.core.errors import ContractValidationError

NODE_LABELS = frozenset({"Session", "Turn", "Fact", "Entity", "Alias"})
RELATIONSHIP_TYPES = frozenset({"HAS_TURN", "EXTRACTED_FROM", "ABOUT", "HAS_ALIAS", "SUPERSEDES", "RELATES_TO"})
SCALAR = (str, int, float, bool)


def _required(value: str, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ContractValidationError(field, "must be a non-empty string")


def _properties(value: Mapping[str, object], field: str) -> None:
    for key, item in value.items():
        if not isinstance(key, str) or not key.replace("_", "").isalnum():
            raise ContractValidationError(field, "property keys must be identifier-safe")
        if not isinstance(item, SCALAR):
            raise ContractValidationError(field, "property values must be scalar")


@dataclass(frozen=True)
class GraphNode:
    graph_id: int
    label: str
    logical_key: str
    properties: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.graph_id, int) or self.graph_id < 0:
            raise ContractValidationError("graph_node.graph_id", "must be non-negative")
        if self.label not in NODE_LABELS:
            raise ContractValidationError("graph_node.label", "must be supported")
        _required(self.logical_key, "graph_node.logical_key")
        _properties(self.properties, "graph_node.properties")


@dataclass(frozen=True)
class GraphRelationship:
    graph_id: int
    relationship_type: str
    logical_key: str
    source_id: int
    destination_id: int
    source_label: str
    destination_label: str
    properties: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.graph_id, int) or self.graph_id < 0:
            raise ContractValidationError("graph_relationship.graph_id", "must be non-negative")
        if self.relationship_type not in RELATIONSHIP_TYPES:
            raise ContractValidationError("graph_relationship.relationship_type", "must be supported")
        _required(self.logical_key, "graph_relationship.logical_key")
        if not isinstance(self.source_id, int) or self.source_id < 0:
            raise ContractValidationError("graph_relationship.source_id", "must be non-negative")
        if not isinstance(self.destination_id, int) or self.destination_id < 0:
            raise ContractValidationError("graph_relationship.destination_id", "must be non-negative")
        if self.source_label not in NODE_LABELS or self.destination_label not in NODE_LABELS:
            raise ContractValidationError("graph_relationship", "endpoint labels must be supported")
        _properties(self.properties, "graph_relationship.properties")


@dataclass(frozen=True)
class GraphWritePlan:
    context_id: str
    plan_key: str
    nodes: tuple[GraphNode, ...]
    relationships: tuple[GraphRelationship, ...]

    def __post_init__(self) -> None:
        _required(self.context_id, "graph_plan.context_id")
        _required(self.plan_key, "graph_plan.plan_key")
        records = self.records()
        ids = [record.graph_id for record in records]
        if len(ids) != len(set(ids)):
            raise ContractValidationError("graph_plan", "graph IDs must be globally unique")
        for record in records:
            if record.properties.get("context_id") != self.context_id:
                raise ContractValidationError("graph_plan", "records must use plan context_id")

    def records(self) -> tuple[GraphNode | GraphRelationship, ...]:
        return (*self.nodes, *self.relationships)

    def payload_hash(self, record: GraphNode | GraphRelationship) -> str:
        if isinstance(record, GraphNode):
            payload: dict[str, object] = {"label": record.label, "id": record.graph_id, "properties": dict(record.properties)}
        else:
            payload = {"type": record.relationship_type, "id": record.graph_id, "source": record.source_id, "destination": record.destination_id, "source_label": record.source_label, "destination_label": record.destination_label, "properties": dict(record.properties)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
