"""Maps validated extraction output to a `GraphWritePlan` (Milestone 8).

Nobody had written this mapping before: `GraphWriter` only knows how to write
an already-built plan (ADR-021/024); `EntityRegistry`/`TemporalUpdateClassifier`
only make bounded decisions (ADR-020). This module is the missing piece
between them — one `Chunk` + its accepted `ExtractedMemoryCandidate`s become
`Session`/`Turn`/`Fact`/`Entity`/`Alias` nodes and `HAS_TURN`/`EXTRACTED_FROM`/
`ABOUT`/`HAS_ALIAS` edges (labels/types from `core/graph.py`, already approved).

Emits `SUPERSEDES` when the caller supplies both `update_classifier` and
`find_existing_facts` (optional constructor-time/call-time args) — closing
the gap ADR-030 originally left open once `ExtractedMemoryCandidate` gained
`action`/`predicate_key` fields. Now wired to a real data source in
production (`ingestion/fact_lookup.py`'s `HydraFactLookup`, constructed by
both `api/routes.py` and `evaluation/benchmark_runner.py`).

The existing-facts check runs for `ADD` candidates too, not only `UPDATE` —
confirmed live: `LLMExtractor.extract()` sees only the current turn's text
(no prior-facts context), so it has no real basis to say UPDATE vs ADD except
guessing from phrasing, and it guessed ADD on "Max moved to Seattle." after
"Max lives in Boston." in the same context. Trusting that guess as the sole
gate would make SUPERSEDES fire only when the extractor happens to phrase
things as a correction. `find_existing_facts` + `update_classifier` already
exist specifically to make this judgment correctly (including returning
NO_UPDATE when nothing should actually supersede), so they are the real
arbiter; `action` no longer gates whether they get asked, only `DELETE` is
excluded (separate semantics, not implemented here). Cost is bounded: a
classify() call only happens when `find_existing_facts` actually returns a
same-subject/same-predicate collision — an unrelated ADD with a fresh
predicate touches nothing beyond the lookup itself.

Every node also carries its `logical_key` as an actual graph property (not
just the Python-side manifest/registry bookkeeping field) — `retrieval.py`'s
graph expansion and `hydration.py`'s entity hydration both `MATCH`/`RETURN`
on `logical_key`, so it has to actually be written, not just computed.

An unresolved entity mention (`EntityResolution.entity is None`) is skipped,
never forced into an `ABOUT` edge — the same guarantee ADR-020 already makes
at the resolution layer, preserved here rather than re-litigated.
"""

from __future__ import annotations

from datetime import datetime, timezone
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


from context_memory.core.resolution import FactState, TemporalRelation
from context_memory.ingestion.resolution import TemporalUpdateClassifier

FindExistingFacts = Callable[[str, int, str], list[FactState]]
"""(context_id, subject_entity_id, predicate_key) -> list of active FactStates for that subject and predicate."""

class GraphPlanBuilder:
    """Builds one `GraphWritePlan` per chunk from its accepted extraction output."""

    def __init__(self, allocator: GraphIdAllocator) -> None:
        self._allocator = allocator

    def build(
        self,
        chunk: Chunk,
        extraction: ExtractionResult,
        resolve: ResolveEntity,
        update_classifier: TemporalUpdateClassifier | None = None,
        find_existing_facts: FindExistingFacts | None = None,
    ) -> GraphWritePlan:
        context_id = chunk.context_id
        nodes: dict[int, GraphNode] = {}
        relationships: dict[int, GraphRelationship] = {}

        session_key = chunk.session_id or f"context:{context_id}"
        session_node = self._session_node(chunk)
        turn_node = self._turn_node(chunk)
        nodes[session_node.graph_id] = session_node
        nodes[turn_node.graph_id] = turn_node
        has_turn = self._has_turn_edge(context_id, session_key, session_node, turn_node, chunk)
        relationships[has_turn.graph_id] = has_turn
        
        # Speaker node
        speaker_node = self._speaker_entity_node(chunk)
        nodes[speaker_node.graph_id] = speaker_node
        
        created_at_dt = datetime.now(timezone.utc)

        for candidate in extraction.accepted:
            fact_node = self._fact_node(chunk, candidate, created_at_dt)
            nodes[fact_node.graph_id] = fact_node
            
            extracted_from = self._extracted_from_edge(context_id, candidate, fact_node, turn_node, chunk)
            relationships[extracted_from.graph_id] = extracted_from
            
            stated_by = self._stated_by_edge(context_id, candidate, fact_node, speaker_node)
            relationships[stated_by.graph_id] = stated_by

            subject_entity_node = None
            for entity_candidate in candidate.entities:
                profile = resolve(context_id, entity_candidate.surface, entity_candidate.entity_type or "other")
                if profile is None:
                    continue  # unresolved mention: no forced link (ADR-020)
                entity_node = self._entity_node(profile)
                if subject_entity_node is None:
                    subject_entity_node = entity_node
                nodes[entity_node.graph_id] = entity_node
                about = self._about_edge(context_id, candidate, fact_node, entity_node, entity_candidate.surface)
                relationships[about.graph_id] = about
                for alias_node, has_alias in self._alias_records(profile):
                    nodes[alias_node.graph_id] = alias_node
                    relationships[has_alias.graph_id] = has_alias

            if candidate.action != "DELETE" and candidate.predicate_key and subject_entity_node and update_classifier and find_existing_facts:
                new_fact_state = FactState(
                    fact_id=candidate.candidate_id,
                    subject_entity_id=subject_entity_node.graph_id,
                    predicate_key=candidate.predicate_key,
                    text=candidate.text,
                    observed_at=candidate.temporal.observed_at,
                    valid_from=candidate.temporal.valid_from,
                    valid_to=candidate.temporal.valid_to
                )
                existing_facts = find_existing_facts(context_id, subject_entity_node.graph_id, candidate.predicate_key)
                
                for prior_fact in existing_facts:
                    decision = update_classifier.classify(new_fact=new_fact_state, prior_fact=prior_fact)
                    if decision.relation in (TemporalRelation.CORRECTION, TemporalRelation.STATE_CHANGE):
                        # Create SUPERSEDES edge
                        edge_logical_key = f"supersedes:{candidate.candidate_id}:{prior_fact.fact_id}"
                        edge_graph_id = self._allocator.allocate_graph_id("supersedes", context_id, edge_logical_key)
                        
                        prior_fact_graph_id = self._allocator.allocate_graph_id("fact", context_id, f"fact:{prior_fact.fact_id}")
                        supersedes_edge = GraphRelationship(
                            edge_graph_id, "SUPERSEDES", edge_logical_key, fact_node.graph_id, prior_fact_graph_id,
                            "Fact", "Fact", _scalar_properties(context_id)
                        )
                        relationships[supersedes_edge.graph_id] = supersedes_edge
                        
                        # Update old fact
                        prior_logical_key = f"fact:{prior_fact.fact_id}"
                        nodes[prior_fact_graph_id] = GraphNode(
                            prior_fact_graph_id, "Fact", prior_logical_key,
                            _scalar_properties(
                                context_id,
                                logical_key=prior_logical_key,
                                is_current=False,
                                superseded_at=int(decision.prior_superseded_at.timestamp()) if decision.prior_superseded_at else 9999999999,
                                valid_to=int(decision.prior_valid_to.timestamp()) if decision.prior_valid_to else 9999999999
                            )
                        )

        plan_key = f"plan:{chunk.chunk_id}"
        return GraphWritePlan(
            context_id=context_id,
            plan_key=plan_key,
            nodes=tuple(nodes.values()),
            relationships=tuple(relationships.values()),
        )

    def _session_node(self, chunk: Chunk) -> GraphNode:
        """A session's node identity is stable across every turn in it (same
        graph_id, same logical_key) — its properties must be too, or the second
        turn's write conflicts with the first's in `PostgresGraphManifestStore`
        (same logical key, different payload, rejected by design). Earlier this
        derived `date`/`date_epoch`/`turn_count`/`source_index` from *the current
        chunk*, which differs turn to turn — confirmed empirically: any session
        with more than one turn failed ingestion on its second turn. Session-level
        aggregates like turn count belong to a graph traversal (`count()` over
        `HAS_TURN`) or a query-time computation, not a value baked into the node
        at an arbitrary turn's write time.
        """
        session_key = chunk.session_id or "unknown"
        logical_key = f"session:{session_key}"
        graph_id = self._allocator.allocate_graph_id("session", chunk.context_id, logical_key)
        properties = _scalar_properties(
            chunk.context_id,
            logical_key=logical_key,
            session_id=session_key,
        )
        return GraphNode(graph_id, "Session", logical_key, properties)

    def _turn_node(self, chunk: Chunk) -> GraphNode:
        logical_key = f"turn:{chunk.chunk_id}"
        graph_id = self._allocator.allocate_graph_id("turn", chunk.context_id, logical_key)
        metadata = chunk.metadata or {}
        
        properties = _scalar_properties(
            chunk.context_id,
            logical_key=logical_key,
            chunk_id=chunk.chunk_id,
            content_hash=chunk.content_hash,
            source_record_id=chunk.source_record_id,
            occurred_at=chunk.occurred_at.isoformat(),
            actor_role=chunk.actor_role,
            actor_id=chunk.actor_id,
            turn_index=metadata.get("turn_index", 0),
            session_id=chunk.session_id or "unknown"
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

    def _speaker_entity_node(self, chunk: Chunk) -> GraphNode:
        """The speaker gets its own `speaker:` key namespace, deliberately not
        `entity:{role}`.

        Sharing the `entity:` namespace made this node collide with an ordinary
        extracted entity of the same name: the extractor genuinely emits
        "user"/"assistant" as entity surfaces, which canonicalize to the same
        `entity:assistant` logical_key and therefore the same allocated
        graph_id, but carry `entity_type="other"` instead of `"speaker"`. Same
        key, same id, different payload -- `PostgresGraphManifestStore` rejects
        that by design, and it aborted a real LongMemEval ingestion run with
        `GraphPayloadConflictError: node entity:assistant has a different
        immutable graph payload`. Separating the namespaces removes the whole
        collision class rather than trying to reconcile two payloads that
        legitimately differ.
        """
        role = chunk.actor_role or "unknown"
        logical_key = f"speaker:{role}"
        graph_id = self._allocator.allocate_graph_id("entity", chunk.context_id, logical_key)
        return GraphNode(
            graph_id, "Entity", logical_key,
            _scalar_properties(chunk.context_id, logical_key=logical_key, canonical_name=role, entity_type="speaker"),
        )

    def _stated_by_edge(
        self, context_id: str, candidate: ExtractedMemoryCandidate, fact_node: GraphNode, speaker_node: GraphNode
    ) -> GraphRelationship:
        logical_key = f"stated_by:{candidate.candidate_id}:{speaker_node.graph_id}"
        graph_id = self._allocator.allocate_graph_id("stated_by", context_id, logical_key)
        return GraphRelationship(
            graph_id, "STATED_BY", logical_key, fact_node.graph_id, speaker_node.graph_id,
            "Fact", "Entity", _scalar_properties(context_id)
        )

    def _fact_node(self, chunk: Chunk, candidate: ExtractedMemoryCandidate, created_at_dt: datetime) -> GraphNode:
        logical_key = f"fact:{candidate.candidate_id}"
        graph_id = self._allocator.allocate_graph_id("fact", chunk.context_id, logical_key)
        
        obs_epoch = int(candidate.temporal.observed_at.timestamp())
        v_from_epoch = int(candidate.temporal.valid_from.timestamp()) if candidate.temporal.valid_from else 0
        v_to_epoch = int(candidate.temporal.valid_to.timestamp()) if candidate.temporal.valid_to else 9999999999
        created_epoch = int(created_at_dt.timestamp())
        
        properties = _scalar_properties(
            chunk.context_id,
            logical_key=logical_key,
            text=candidate.text,
            speaker=chunk.actor_role,
            session_id=chunk.session_id or "unknown",
            memory_type=candidate.memory_type.value,
            scope_type=candidate.scope_type.value,
            scope_id=candidate.scope_id,
            # Written so `find_existing_facts` implementations can query "active
            # facts for this subject+predicate" directly off the graph — without
            # it, that lookup (the data source SUPERSEDES needs, see this
            # module's docstring) has nothing to filter on.
            predicate_key=candidate.predicate_key,
            source_chunk_id=chunk.chunk_id,
            source_start=candidate.source_span.source_start,
            source_end=candidate.source_span.source_end,
            content_hash=chunk.content_hash,
            confidence=candidate.confidence,
            observed_at=obs_epoch,
            superseded_at=9999999999,
            valid_from=v_from_epoch,
            valid_to=v_to_epoch,
            created_at=created_epoch,
            is_current=True,
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
            _scalar_properties(profile.context_id, logical_key=logical_key, canonical_name=profile.canonical_name, entity_type=profile.entity_type),
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
                _scalar_properties(profile.context_id, logical_key=alias_logical_key, canonical_alias=alias, entity_graph_id=profile.graph_id),
            )
            edge_logical_key = f"has_alias:{profile.graph_id}:{alias}"
            edge_graph_id = self._allocator.allocate_graph_id("has_alias", profile.context_id, edge_logical_key)
            has_alias = GraphRelationship(
                edge_graph_id, "HAS_ALIAS", edge_logical_key, profile.graph_id, alias_graph_id,
                "Entity", "Alias", _scalar_properties(profile.context_id),
            )
            records.append((alias_node, has_alias))
        return records
