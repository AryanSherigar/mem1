from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_memory.ingestion.fakes import (
    DeterministicEntityResolutionModel,
    DeterministicTemporalUpdateModel,
    InMemoryGraphIdAllocator,
)
from context_memory.ingestion.resolution import EntityRegistry, TemporalUpdateClassifier
from context_memory.core.resolution import (
    EntityProfile,
    FactState,
    ResolutionStatus,
    TemporalRelation,
    canonicalize_entity_surface,
)


class EntityResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = EntityRegistry(InMemoryGraphIdAllocator())
        self.max = EntityProfile(7, "context-a", "Max", "pet", ("my dog", "the golden retriever"))
        self.registry.register(self.max)

    def test_unicode_canonicalization_is_stable(self) -> None:
        self.assertEqual(canonicalize_entity_surface("  CAFÉ\u00a0Dog "), "café dog")

    def test_exact_canonical_and_alias_do_not_call_model(self) -> None:
        model = DeterministicEntityResolutionModel()
        canonical = self.registry.resolve(context_id="context-a", surface=" max ", model=model)
        alias = self.registry.resolve(context_id="context-a", surface="MY DOG", model=model)
        self.assertEqual(canonical.status, ResolutionStatus.EXACT_CANONICAL)
        self.assertEqual(alias.status, ResolutionStatus.EXACT_ALIAS)
        self.assertEqual(model.calls, [])

    def test_context_isolation_creates_new_entity(self) -> None:
        resolution = self.registry.resolve(context_id="context-b", surface="Max")
        self.assertEqual(resolution.status, ResolutionStatus.NEW_ENTITY)
        self.assertNotEqual(resolution.entity.graph_id, self.max.graph_id)

    def test_bounded_model_resolves_non_exact_reference(self) -> None:
        model = DeterministicEntityResolutionModel({"him": 7})
        result = self.registry.resolve(
            context_id="context-a", surface="him", candidate_ids=(7,), model=model
        )
        self.assertEqual((result.status, result.entity.graph_id), (ResolutionStatus.MODEL_RESOLVED, 7))
        self.assertEqual(model.calls, [("context-a", "him", (7,))])

    def test_model_cannot_select_outside_bounded_candidates(self) -> None:
        model = DeterministicEntityResolutionModel({"him": 999})
        result = self.registry.resolve(
            context_id="context-a", surface="him", candidate_ids=(7,), model=model
        )
        self.assertEqual(result.status, ResolutionStatus.UNRESOLVED)

    def test_ambiguous_alias_without_model_is_unresolved(self) -> None:
        self.registry.register(EntityProfile(8, "context-a", "Rex", "pet", ("my dog",)))
        result = self.registry.resolve(context_id="context-a", surface="my dog")
        self.assertEqual(result.status, ResolutionStatus.UNRESOLVED)


class TemporalUpdateTests(unittest.TestCase):
    def fact(self, fact_id: str, observed_at: datetime, *, subject: int = 7, predicate: str = "breed", valid_from: datetime | None = None) -> FactState:
        return FactState(fact_id, subject, predicate, "Max is a dog", observed_at, valid_from)

    def setUp(self) -> None:
        self.start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.prior = self.fact("prior", self.start, valid_from=self.start)

    def test_correction_closes_knowledge_not_world_validity(self) -> None:
        new = self.fact("new", self.start + timedelta(days=2))
        model = DeterministicTemporalUpdateModel({("new", "prior"): TemporalRelation.CORRECTION})
        decision = TemporalUpdateClassifier(model).classify(new_fact=new, prior_fact=self.prior)
        self.assertEqual(decision.relation, TemporalRelation.CORRECTION)
        self.assertEqual(decision.prior_superseded_at, new.observed_at)
        self.assertIsNone(decision.prior_valid_to)

    def test_state_change_closes_world_validity_at_supported_time(self) -> None:
        change_time = self.start + timedelta(days=4)
        new = self.fact("new", self.start + timedelta(days=5), valid_from=change_time)
        model = DeterministicTemporalUpdateModel({("new", "prior"): TemporalRelation.STATE_CHANGE})
        decision = TemporalUpdateClassifier(model).classify(new_fact=new, prior_fact=self.prior)
        self.assertEqual(decision.prior_valid_to, change_time)

    def test_different_subject_or_predicate_never_calls_model(self) -> None:
        new = self.fact("new", self.start + timedelta(days=2), subject=8)
        model = DeterministicTemporalUpdateModel({("new", "prior"): TemporalRelation.CORRECTION})
        decision = TemporalUpdateClassifier(model).classify(new_fact=new, prior_fact=self.prior)
        self.assertEqual(decision.relation, TemporalRelation.NO_UPDATE)
        self.assertEqual(model.calls, [])

    def test_out_of_order_fact_cannot_supersede_known_newer_fact(self) -> None:
        earlier = self.fact("earlier", self.start - timedelta(days=1))
        model = DeterministicTemporalUpdateModel({("earlier", "prior"): TemporalRelation.STATE_CHANGE})
        decision = TemporalUpdateClassifier(model).classify(new_fact=earlier, prior_fact=self.prior)
        self.assertEqual(decision.relation, TemporalRelation.NO_UPDATE)
        self.assertEqual(model.calls, [])
