from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone

from context_memory.core.llm_client import LLMClient
from context_memory.core.resolution import EntityProfile, FactState, TemporalRelation
from context_memory.ingestion.model_adapters import LLMEntityResolutionModel, LLMTemporalUpdateModel


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content: str | None, exception: Exception | None = None) -> None:
        self._content = content
        self._exception = exception
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _FakeResponse:
        self.calls.append(kwargs)
        if self._exception is not None:
            raise self._exception
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeOpenAIClient:
    def __init__(self, content: str | None = None, exception: Exception | None = None) -> None:
        self.chat = _FakeChat(_FakeCompletions(content, exception))


def _client(content: str | None = None, exception: Exception | None = None) -> LLMClient:
    fake = _FakeOpenAIClient(content, exception)
    return LLMClient(base_url="https://example.invalid/v1", api_key="fake-key", model_name="fake-model", client=fake)


class EntityResolutionModelTests(unittest.TestCase):
    def candidates(self) -> tuple[EntityProfile, ...]:
        return (
            EntityProfile(1, "context:1", "max the dog", "pet", ("my dog",)),
            EntityProfile(2, "context:1", "max the colleague", "person"),
        )

    def test_bounded_selection_returns_model_choice(self) -> None:
        model = LLMEntityResolutionModel(_client(json.dumps({"selected_graph_id": 1})))
        selected = model.resolve_entity(context_id="context:1", surface="my dog", candidates=self.candidates())
        self.assertEqual(selected, 1)

    def test_model_abstains_with_null(self) -> None:
        model = LLMEntityResolutionModel(_client(json.dumps({"selected_graph_id": None})))
        selected = model.resolve_entity(context_id="context:1", surface="Sam", candidates=self.candidates())
        self.assertIsNone(selected)

    def test_empty_candidates_never_calls_model(self) -> None:
        client = _client(json.dumps({"selected_graph_id": 1}))
        model = LLMEntityResolutionModel(client)
        selected = model.resolve_entity(context_id="context:1", surface="Sam", candidates=())
        self.assertIsNone(selected)

    def test_malformed_response_resolves_to_none_not_an_exception(self) -> None:
        model = LLMEntityResolutionModel(_client("not json"))
        selected = model.resolve_entity(context_id="context:1", surface="my dog", candidates=self.candidates())
        self.assertIsNone(selected)

    def test_out_of_bounds_id_is_returned_unfiltered_caller_must_reject(self) -> None:
        # The adapter does not enforce the bound itself; EntityRegistry.resolve does.
        # This test documents that boundary explicitly.
        model = LLMEntityResolutionModel(_client(json.dumps({"selected_graph_id": 999})))
        selected = model.resolve_entity(context_id="context:1", surface="my dog", candidates=self.candidates())
        self.assertEqual(selected, 999)


class TemporalUpdateModelTests(unittest.TestCase):
    def facts(self) -> tuple[FactState, FactState]:
        prior = FactState("fact:1", 1, "lives_in", "Max lives in Boston", datetime(2026, 1, 1, tzinfo=timezone.utc))
        new = FactState("fact:2", 1, "lives_in", "Max lives in Seattle", datetime(2026, 2, 1, tzinfo=timezone.utc))
        return new, prior

    def test_correction_relation_parses(self) -> None:
        new, prior = self.facts()
        model = LLMTemporalUpdateModel(_client(json.dumps({"relation": "correction"})))
        self.assertEqual(model.classify_update(new_fact=new, prior_fact=prior), TemporalRelation.CORRECTION)

    def test_state_change_relation_parses(self) -> None:
        new, prior = self.facts()
        model = LLMTemporalUpdateModel(_client(json.dumps({"relation": "state_change"})))
        self.assertEqual(model.classify_update(new_fact=new, prior_fact=prior), TemporalRelation.STATE_CHANGE)

    def test_invalid_json_resolves_to_unresolved(self) -> None:
        new, prior = self.facts()
        model = LLMTemporalUpdateModel(_client("{not valid json"))
        self.assertEqual(model.classify_update(new_fact=new, prior_fact=prior), TemporalRelation.UNRESOLVED)

    def test_invalid_enum_value_resolves_to_unresolved(self) -> None:
        new, prior = self.facts()
        model = LLMTemporalUpdateModel(_client(json.dumps({"relation": "made_up_value"})))
        self.assertEqual(model.classify_update(new_fact=new, prior_fact=prior), TemporalRelation.UNRESOLVED)


if __name__ == "__main__":
    unittest.main()
