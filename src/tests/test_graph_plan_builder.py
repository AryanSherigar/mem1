from __future__ import annotations

import unittest
from datetime import datetime, timezone

from context_memory.core.enums import MemoryScope, MemoryType
from context_memory.core.models import EntityCandidate, ExtractedMemoryCandidate, SourceDescriptor, SourceSpan, TemporalBounds
from context_memory.core.resolution import EntityProfile
from context_memory.core.validation import chunk_from_record
from context_memory.core.models import ContextBatch, ContextRecord
from context_memory.ingestion.extraction import ExtractionResult
from context_memory.ingestion.fakes import InMemoryGraphIdAllocator
from context_memory.ingestion.graph_plan_builder import GraphPlanBuilder


def _candidate(entities=()) -> ExtractedMemoryCandidate:
    return ExtractedMemoryCandidate(
        candidate_id="fact-001", text="Max likes walks", memory_type=MemoryType.SEMANTIC,
        scope_type=MemoryScope.SESSION, scope_id="session-001",
        source_span=SourceSpan("record-001", 0, 3), confidence=0.9,
        temporal=TemporalBounds(datetime(2026, 1, 10, 9, tzinfo=timezone.utc)), entities=entities,
    )


class GraphPlanBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.allocator = InMemoryGraphIdAllocator()
        self.builder = GraphPlanBuilder(self.allocator)
        record = ContextRecord(
            record_id="record-001", session_id="session-001",
            occurred_at=datetime(2026, 1, 10, 9, tzinfo=timezone.utc), content="Max likes walks",
        )
        batch = ContextBatch("ingestion-001", "context-001", SourceDescriptor("fixture", "fixture-001"), (record,))
        self.chunk = chunk_from_record(batch, record)

    def test_no_candidates_still_writes_session_and_turn(self) -> None:
        extraction = ExtractionResult("attempt-1", (), ())
        plan = self.builder.build(self.chunk, extraction, resolve=lambda *a: None)
        labels = {node.label for node in plan.nodes}
        self.assertEqual(labels, {"Session", "Turn", "Entity"})
        self.assertEqual({r.relationship_type for r in plan.relationships}, {"HAS_TURN"})

    def test_unresolved_entity_mention_creates_no_about_edge(self) -> None:
        candidate = _candidate(entities=(EntityCandidate("Max", "pet"),))
        extraction = ExtractionResult("attempt-1", (candidate,), ())
        plan = self.builder.build(self.chunk, extraction, resolve=lambda *a: None)  # always unresolved
        about_edges = [r for r in plan.relationships if r.relationship_type == "ABOUT"]
        self.assertEqual(len(about_edges), 0)
        # Note: speaker Entity is still created, but no ABOUT edge
        speaker_entities = [n for n in plan.nodes if n.label == "Entity" and n.properties.get("entity_type") == "speaker"]
        self.assertEqual(len(speaker_entities), 1)

    def test_resolved_entity_creates_entity_node_and_about_edge(self) -> None:
        candidate = _candidate(entities=(EntityCandidate("Max", "pet"),))
        extraction = ExtractionResult("attempt-1", (candidate,), ())
        profile = EntityProfile(99, self.chunk.context_id, "Max (pet)", "pet")
        plan = self.builder.build(self.chunk, extraction, resolve=lambda cid, surface, etype: profile)
        entity_nodes = [n for n in plan.nodes if n.label == "Entity" and n.properties.get("entity_type") != "speaker"]
        self.assertEqual(len(entity_nodes), 1)
        self.assertEqual(entity_nodes[0].properties["canonical_name"], "Max (pet)")
        about_edges = [r for r in plan.relationships if r.relationship_type == "ABOUT"]
        self.assertEqual(len(about_edges), 1)
        stated_by = [r for r in plan.relationships if r.relationship_type == "STATED_BY"]
        self.assertEqual(len(stated_by), 1)

    def test_aliases_produce_alias_nodes_and_has_alias_edges(self) -> None:
        entity_id = self.allocator.allocate_graph_id("entity", "context-001", "entity:max")
        profile = EntityProfile(entity_id, "context-001", "max", "pet", aliases=("my dog", "the retriever"))
        candidate = _candidate(entities=(EntityCandidate("my dog", "pet"),))
        extraction = ExtractionResult("attempt-1", (candidate,), ())
        plan = self.builder.build(self.chunk, extraction, resolve=lambda cid, surface, etype: profile)
        alias_nodes = [n for n in plan.nodes if n.label == "Alias"]
        has_alias = [r for r in plan.relationships if r.relationship_type == "HAS_ALIAS"]
        self.assertEqual(len(alias_nodes), 2)
        self.assertEqual(len(has_alias), 2)

    def test_every_node_and_relationship_carries_context_id(self) -> None:
        entity_id = self.allocator.allocate_graph_id("entity", "context-001", "entity:max")
        profile = EntityProfile(entity_id, "context-001", "max", "pet", aliases=("my dog",))
        candidate = _candidate(entities=(EntityCandidate("my dog", "pet"),))
        extraction = ExtractionResult("attempt-1", (candidate,), ())
        plan = self.builder.build(self.chunk, extraction, resolve=lambda cid, surface, etype: profile)
        for record in plan.records():
            self.assertEqual(record.properties["context_id"], "context-001")

    def test_repeated_build_is_deterministically_replayable(self) -> None:
        candidate = _candidate()
        extraction = ExtractionResult("attempt-1", (candidate,), ())
        plan_a = self.builder.build(self.chunk, extraction, resolve=lambda *a: None)
        plan_b = self.builder.build(self.chunk, extraction, resolve=lambda *a: None)
        self.assertEqual({n.graph_id for n in plan_a.nodes}, {n.graph_id for n in plan_b.nodes})
        self.assertEqual(plan_a.nodes, plan_b.nodes)

    def test_supersedes_edge_created(self) -> None:
        from context_memory.core.resolution import FactState, TemporalRelation, TemporalUpdateDecision
        
        class FakeTemporalClassifier:
            def classify(self, new_fact, prior_fact):
                return TemporalUpdateDecision(TemporalRelation.STATE_CHANGE, "test", prior_superseded_at=new_fact.observed_at)
                
        entity_id = self.allocator.allocate_graph_id("entity", "context-001", "entity:max")
        profile = EntityProfile(entity_id, "context-001", "max", "pet")
        candidate = ExtractedMemoryCandidate(
            candidate_id="fact-new", text="Max likes running", memory_type=MemoryType.SEMANTIC,
            scope_type=MemoryScope.SESSION, scope_id="session-001",
            source_span=SourceSpan("record-001", 0, 3), confidence=0.9,
            temporal=TemporalBounds(datetime(2026, 1, 10, 9, tzinfo=timezone.utc)),
            entities=(EntityCandidate("Max", "pet"),),
            action="UPDATE", predicate_key="likes"
        )
        extraction = ExtractionResult("attempt-1", (candidate,), ())
        prior_fact = FactState("fact-old", entity_id, "likes", "Max likes walking", datetime(2025, 1, 1, tzinfo=timezone.utc))
        
        def find_existing(context, subject, predicate):
            if subject == entity_id and predicate == "likes":
                return [prior_fact]
            return []
            
        plan = self.builder.build(
            self.chunk, extraction, lambda cid, s, e: profile, 
            update_classifier=FakeTemporalClassifier(), find_existing_facts=find_existing
        )
        
        supersedes = [r for r in plan.relationships if r.relationship_type == "SUPERSEDES"]
        self.assertEqual(len(supersedes), 1)
        self.assertEqual(supersedes[0].destination_id, self.allocator.allocate_graph_id("fact", "context-001", "fact:fact-old"))
        
        old_nodes = [n for n in plan.nodes if n.graph_id == supersedes[0].destination_id]
        self.assertEqual(len(old_nodes), 1)
        self.assertEqual(old_nodes[0].properties["is_current"], False)


if __name__ == "__main__":
    unittest.main()
