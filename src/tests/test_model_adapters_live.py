"""Live Fireworks/deepseek-v4-flash smoke for the M5 model adapters (ADR-029).

Mirrors test_hydradb_live.py's gating pattern: skipped unless a real
credential is present, never run in CI, makes real (billed) provider calls.
"""

from __future__ import annotations

import os
import unittest
from datetime import datetime, timezone

from context_memory.core.config import Config
from context_memory.core.resolution import EntityProfile, FactState, TemporalRelation
from context_memory.ingestion.model_adapters import LLMEntityResolutionModel, LLMTemporalUpdateModel

HAS_KEY = bool(os.environ.get("FIREWORKS_API_KEY") or os.environ.get("ENTITY_RESOLUTION_API_KEY"))


@unittest.skipUnless(HAS_KEY, "requires FIREWORKS_API_KEY (or ENTITY_RESOLUTION_API_KEY) for a real provider call")
class LiveEntityResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = LLMEntityResolutionModel(Config().get_entity_resolution_client())

    def test_resolves_alias_within_bounded_candidates(self) -> None:
        candidates = (
            EntityProfile(1, "context:1", "max the golden retriever", "pet", ("my dog",)),
            EntityProfile(2, "context:1", "max the colleague", "person"),
        )
        selected = self.model.resolve_entity(context_id="context:1", surface="my dog", candidates=candidates)
        self.assertEqual(selected, 1)

    def test_abstains_when_no_candidate_matches(self) -> None:
        candidates = (EntityProfile(3, "context:1", "sarah", "person"),)
        selected = self.model.resolve_entity(context_id="context:1", surface="my dog", candidates=candidates)
        self.assertIsNone(selected)


@unittest.skipUnless(HAS_KEY, "requires FIREWORKS_API_KEY (or TEMPORAL_UPDATE_API_KEY) for a real provider call")
class LiveTemporalUpdateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = LLMTemporalUpdateModel(Config().get_temporal_update_client())

    def test_real_world_move_is_a_state_change(self) -> None:
        prior = FactState("fact:1", 1, "lives_in", "Max lives in Boston", datetime(2026, 1, 1, tzinfo=timezone.utc))
        new = FactState(
            "fact:2", 1, "lives_in", "Max just moved to Seattle last week",
            datetime(2026, 2, 1, tzinfo=timezone.utc),
        )
        self.assertEqual(
            self.model.classify_update(new_fact=new, prior_fact=prior), TemporalRelation.STATE_CHANGE
        )

    def test_explicit_correction_is_not_a_state_change(self) -> None:
        prior = FactState("fact:1", 1, "lives_in", "Max lives in Seattle", datetime(2026, 1, 1, tzinfo=timezone.utc))
        new = FactState(
            "fact:2", 1, "lives_in", "I got that wrong earlier -- Max has always lived in Boston, never Seattle",
            datetime(2026, 1, 2, tzinfo=timezone.utc),
        )
        result = self.model.classify_update(new_fact=new, prior_fact=prior)
        self.assertNotEqual(result, TemporalRelation.STATE_CHANGE)


if __name__ == "__main__":
    unittest.main()
