"""Maps validated extraction output to a `GraphWritePlan` (Milestone 8).

Nobody had written this mapping before: `GraphWriter` only knows how to write
an already-built plan (ADR-021/024); `EntityRegistry`/`TemporalUpdateClassifier`
only make bounded decisions (ADR-020). This module is the missing piece
between them — one `Chunk` + its accepted `ExtractedMemoryCandidate`s become
`Session`/`Turn`/`Fact`/`Entity`/`Alias` nodes and `HAS_TURN`/`EXTRACTED_FROM`/
`ABOUT`/`HAS_ALIAS` edges (labels/types from `core/graph.py`, already approved).

Deliberately does not emit `SUPERSEDES`. `TemporalUpdateClassifier.classify`
(ADR-020) requires a `FactState.predicate_key` to gate same-subject/same-
predicate before it will even consider calling the model — and nothing in the
current deterministic extraction baseline (ADR-019) produces a predicate for a
candidate; `ExtractedMemoryCandidate`/`ExtractionDraft` carry free text only.
Inventing a predicate heuristic here would be extraction-quality work smuggled
into graph-plan construction. This is a real, open gap (see ADR-030), not an
oversight: a future extractor milestone needs to produce `predicate_key`
before any fact can supersede another in the graph.

An unresolved entity mention (`EntityResolution.entity is None`) is skipped,
never forced into an `ABOUT` edge — the same guarantee ADR-020 already makes
at the resolution layer, preserved here rather than re-litigated.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Mapping

from context_memory.core.graph import GraphNode, GraphRelationship, GraphWritePlan
from context_memory.core.models import Chunk, ExtractedMemoryCandidate
from context_memory.core.resolution import EntityProfile
from context_memory.ingestion.extraction import ExtractionResult
from context_memory.ingestion.ports import GraphIdAllocator

ResolveEntity = Callable[[str, str, str], "EntityProfile | None"]
"""(context_id, surface, entity_type) -> resolved profile, or None if unresolved."""


def _scalar_properties(context_id: str, **fields: object) -> dict[str, object]:
    properties: dict[str, object] = {"context_id": context_id}
    for key, value in fields.items():
        if value is None:
            continue
        properties[key] = value
    return properties


class GraphPlanBuilder:
    """Builds one `GraphWritePlan` per chunk from its accepted extraction output."""

    def __init__(self, allocator: GraphIdAllocator) -> None:
        self._allocator = allocator

    def build(
        self,
        chunk: Chunk,
        extraction: ExtractionResult,
        resolve: ResolveEntity,
    ) -> GraphWritePlan:
        context_id = chunk.context_id
        nodes: dict[int, GraphNode] = {}
        relationships: dict[int, GraphRelationship] = {}

        session_key = chunk.session_id or f"context:{context_id}"
        session_node = self._session_node(context_id, session_key)
        turn_node = self._turn_node(chunk)
        nodes[session_node.graph_id] = session_node
        nodes[turn_node.graph_id] = turn_node
        has_turn = self._has_turn_edge(context_id, session_key, session_node, turn_node, chunk)
        relationships[has_turn.graph_id] = has_turn

        for candidate in extraction.accepted:
            fact_node = self._fact_node(chunk, candidate)
            nodes[fact_node.graph_id] = fact_node
            extracted_from = self._extracted_from_edge(context_id, candidate, fact_node, turn_node, chunk)
            relationships[extracted_from.graph_id] = extracted_from

            for entity_candidate in candidate.entities:
                profile = resolve(context_id, entity_candidate.surface, entity_candidate.entity_type or "other")
                if profile is None:
                    continue  # unresolved mention: no forced link (ADR-020)
                entity_node = self._entity_node(profile)
                nodes[entity_node.graph_id] = entity_node
                about = self._about_edge(context_id, candidate, fact_node, entity_node, entity_candidate.surface)
                relationships[about.graph_id] = about
                for alias_node, has_alias in self._alias_records(profile):
                    nodes[alias_node.graph_id] = alias_node
                    relationships[has_alias.graph_id] = has_alias

        plan_key = f"plan:{chunk.chunk_id}"
        return GraphWritePlan(
            context_id=context_id,
            plan_key=plan_key,
            nodes=tuple(nodes.values()),
            relationships=tuple(relationships.values()),
        )

    def _session_node(self, context_id: str, session_key: str) -> GraphNode:
        logical_key = f"session:{session_key}"
        graph_id = self._allocator.allocate_graph_id("session", context_id, logical_key)
        return GraphNode(graph_id, "Session", logical_key, _scalar_properties(context_id, session_key=session_key))

    def _turn_node(self, chunk: Chunk) -> GraphNode:
        logical_key = f"turn:{chunk.chunk_id}"
        graph_id = self._allocator.allocate_graph_id("turn", chunk.context_id, logical_key)
        properties = _scalar_properties(
            chunk.context_id,
            chunk_id=chunk.chunk_id,
            content_hash=chunk.content_hash,
            source_record_id=chunk.source_record_id,
            occurred_at=chunk.occurred_at.isoformat(),
            actor_role=chunk.actor_role,
            actor_id=chunk.actor_id,
        )
        return GraphNode(graph_id, "Turn", logical_key, properties)

    def _has_turn_edge(
        self, context_id: str, session_key: str, session_node: GraphNode, turn_node: GraphNode, chunk: Chunk
    ) -> GraphRelationship:
        logical_key = f"has_turn:{session_key}:{chunk.chunk_id}"
        graph_id = self._allocator.allocate_graph_id("has_turn", context_id, logical_key)
        return GraphRelationship(
            graph_id, "HAS_TURN", logical_key, session_node.graph_id, turn_node.graph_id,
            "Session", "Turn", _scalar_properties(context_id),
        )

    def _fact_node(self, chunk: Chunk, candidate: ExtractedMemoryCandidate) -> GraphNode:
        logical_key = f"fact:{candidate.candidate_id}"
        graph_id = self._allocator.allocate_graph_id("fact", chunk.context_id, logical_key)
        properties = _scalar_properties(
            chunk.context_id,
            text=candidate.text,
            memory_type=candidate.memory_type.value,
            scope_type=candidate.scope_type.value,
            scope_id=candidate.scope_id,
            source_chunk_id=chunk.chunk_id,
            source_start=candidate.source_span.source_start,
            source_end=candidate.source_span.source_end,
            confidence=candidate.confidence,
            observed_at=candidate.temporal.observed_at.isoformat(),
            valid_from=candidate.temporal.valid_from.isoformat() if candidate.temporal.valid_from else None,
            valid_to=candidate.temporal.valid_to.isoformat() if candidate.temporal.valid_to else None,
        )
        return GraphNode(graph_id, "Fact", logical_key, properties)

    def _extracted_from_edge(
        self, context_id: str, candidate: ExtractedMemoryCandidate, fact_node: GraphNode, turn_node: GraphNode, chunk: Chunk
    ) -> GraphRelationship:
        logical_key = f"extracted_from:{candidate.candidate_id}:{chunk.chunk_id}"
        graph_id = self._allocator.allocate_graph_id("extracted_from", context_id, logical_key)
        return GraphRelationship(
            graph_id, "EXTRACTED_FROM", logical_key, fact_node.graph_id, turn_node.graph_id,
            "Fact", "Turn", _scalar_properties(context_id),
        )

    def _entity_node(self, profile: EntityProfile) -> GraphNode:
        logical_key = f"entity:{profile.canonical_name}"
        return GraphNode(
            profile.graph_id, "Entity", logical_key,
            _scalar_properties(profile.context_id, canonical_name=profile.canonical_name, entity_type=profile.entity_type),
        )

    def _about_edge(
        self, context_id: str, candidate: ExtractedMemoryCandidate, fact_node: GraphNode, entity_node: GraphNode, surface: str
    ) -> GraphRelationship:
        logical_key = f"about:{candidate.candidate_id}:{entity_node.graph_id}"
        graph_id = self._allocator.allocate_graph_id("about", context_id, logical_key)
        return GraphRelationship(
            graph_id, "ABOUT", logical_key, fact_node.graph_id, entity_node.graph_id,
            "Fact", "Entity", _scalar_properties(context_id, surface=surface),
        )

    def _alias_records(self, profile: EntityProfile) -> list[tuple[GraphNode, GraphRelationship]]:
        records: list[tuple[GraphNode, GraphRelationship]] = []
        for alias in profile.aliases:
            alias_logical_key = f"alias:{alias}:{profile.graph_id}"
            alias_graph_id = self._allocator.allocate_graph_id("alias", profile.context_id, alias_logical_key)
            alias_node = GraphNode(
                alias_graph_id, "Alias", alias_logical_key,
                _scalar_properties(profile.context_id, canonical_alias=alias, entity_graph_id=profile.graph_id),
            )
            edge_logical_key = f"has_alias:{profile.graph_id}:{alias}"
            edge_graph_id = self._allocator.allocate_graph_id("has_alias", profile.context_id, edge_logical_key)
            has_alias = GraphRelationship(
                edge_graph_id, "HAS_ALIAS", edge_logical_key, profile.graph_id, alias_graph_id,
                "Entity", "Alias", _scalar_properties(profile.context_id),
            )
            records.append((alias_node, has_alias))
        return records
