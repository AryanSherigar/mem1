"""Ingestion-side graph batch construction and local HydraDB write orchestration."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256

from context_memory.core.graph import GraphNode, GraphRelationship, GraphWritePlan
from context_memory.core.logging import get_logger, timed_operation
from context_memory.ingestion.ports import GraphManifestStore, GraphTransport

logger = get_logger(__name__)


class GraphWriter:
    def __init__(self, manifest_store: GraphManifestStore, transport: GraphTransport) -> None:
        self._manifest_store = manifest_store
        self._transport = transport

    def write(self, plan: GraphWritePlan) -> tuple[str, ...]:
        with timed_operation(logger, "graph_writer.write", {"plan_key": plan.plan_key, "nodes": len(plan.nodes), "relationships": len(plan.relationships)}) as ctx:
            self._manifest_store.register(plan)
            bookmarks: list[str] = []
            nodes: dict[tuple[str, tuple[str, ...]], list[GraphNode]] = defaultdict(list)
            relationships: dict[tuple[str, str, str, tuple[str, ...]], list[GraphRelationship]] = defaultdict(list)
            for node in plan.nodes:
                nodes[(node.label, tuple(sorted(node.properties)))].append(node)
            for relationship in plan.relationships:
                relationships[(relationship.relationship_type, relationship.source_label, relationship.destination_label, tuple(sorted(relationship.properties)))].append(relationship)
            for (label, property_names), group in nodes.items():
                rows = self._node_rows(group)
                bookmark = self._transport.write(self._node_query(label, property_names), rows, self._key(plan, f"node-{label}-{'-'.join(property_names)}", rows))
                if bookmark:
                    bookmarks.append(bookmark)
            for (relationship_type, source_label, destination_label, property_names), group in relationships.items():
                rows = self._relationship_rows(group)
                bookmark = self._transport.write(self._relationship_query(relationship_type, source_label, destination_label, property_names), rows, self._key(plan, f"relationship-{relationship_type}-{source_label}-{destination_label}-{'-'.join(property_names)}", rows))
                if bookmark:
                    bookmarks.append(bookmark)
            ctx["bookmarks_received"] = len(bookmarks)
            return tuple(bookmarks)

    @staticmethod
    def _node_query(label: str, property_names: tuple[str, ...]) -> str:
        assignments = ", ".join(f"n.{name} = row.{name}" for name in property_names)
        return f"UNWIND $rows AS row MERGE (n {{id: row.id}}) SET n:{label}, {assignments}"

    @staticmethod
    def _relationship_query(relationship_type: str, source_label: str, destination_label: str, property_names: tuple[str, ...]) -> str:
        assignments = ", ".join(f"r.{name} = row.{name}" for name in property_names)
        return f"UNWIND $rows AS row MATCH (s:{source_label} {{id: row.source_id}}), (d:{destination_label} {{id: row.destination_id}}) MERGE (s)-[r:{relationship_type} {{id: row.id}}]->(d) SET {assignments}"

    @staticmethod
    def _node_rows(nodes: list[GraphNode]) -> list[dict[str, object]]:
        return [{"id": node.graph_id, **dict(node.properties)} for node in nodes]

    @staticmethod
    def _relationship_rows(relationships: list[GraphRelationship]) -> list[dict[str, object]]:
        return [{"id": item.graph_id, "source_id": item.source_id, "destination_id": item.destination_id, **dict(item.properties)} for item in relationships]

    @staticmethod
    def _key(plan: GraphWritePlan, phase: str, rows: list[dict[str, object]] | None = None) -> str:
        import json
        payload_bytes = json.dumps(rows, sort_keys=True, default=str).encode() if rows is not None else b""
        return f"context-memory-{sha256(f'{plan.context_id}\x00{plan.plan_key}\x00{phase}\x00'.encode() + payload_bytes).hexdigest()}"
