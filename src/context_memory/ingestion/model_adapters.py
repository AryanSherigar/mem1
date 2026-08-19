"""Real (non-fake) implementations of the bounded LLM ports (ADR-027).

Both adapters only ever return a value the caller independently re-validates:
`EntityRegistry.resolve` rejects any `selected_graph_id` outside the supplied
candidate set (never trusted here), and `TemporalUpdateClassifier.classify`
only calls `classify_update` after same-subject/predicate/chronology gates
already passed. An adapter cannot widen either boundary; it can only narrow
to `UNRESOLVED`/`None` on invalid or abstaining model output.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from context_memory.core.llm_client import LLMClient, LLMClientError
from context_memory.core.resolution import EntityProfile, FactState, TemporalRelation

ENTITY_RESOLUTION_SYSTEM_PROMPT = (
    "You resolve an entity surface form to exactly one of a fixed, already-bounded "
    "candidate list, or to none if no candidate is the same real-world entity. "
    "Never invent a candidate id that is not in the supplied list. Prefer 'null' over "
    "a low-confidence guess."
)

TEMPORAL_UPDATE_SYSTEM_PROMPT = (
    "You classify how a new fact relates to a prior fact about the same subject and "
    "predicate, observed later in time. 'correction' means the prior fact was wrong "
    "and is being fixed; the real-world state never changed. 'state_change' means the "
    "real world changed and the prior fact was true until now. 'no_update' means the "
    "new fact does not actually supersede the prior one. Use 'unresolved' if the text "
    "gives no clear basis to decide."
)


class _EntityResolutionResponse(BaseModel):
    selected_graph_id: int | None


class _TemporalUpdateResponse(BaseModel):
    relation: Literal["correction", "state_change", "no_update", "unresolved"]


def _format_candidates(candidates: Sequence[EntityProfile]) -> str:
    lines = []
    for profile in candidates:
        aliases = ", ".join(profile.aliases) if profile.aliases else "(none)"
        lines.append(
            f"- graph_id={profile.graph_id}: name={profile.canonical_name!r} "
            f"type={profile.entity_type!r} aliases=[{aliases}]"
        )
    return "\n".join(lines)


class LLMEntityResolutionModel:
    """`ingestion.ports.EntityResolutionModel` backed by a real LLM call."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def resolve_entity(
        self, *, context_id: str, surface: str, candidates: Sequence[EntityProfile]
    ) -> int | None:
        if not candidates:
            return None
        user_prompt = (
            f"Surface form to resolve: {surface!r}\n\n"
            f"Candidates (context {context_id!r}):\n{_format_candidates(candidates)}\n\n"
            "Return the graph_id of the matching candidate, or null if none match."
        )
        try:
            result = self._client.structured_completion(
                ENTITY_RESOLUTION_SYSTEM_PROMPT, user_prompt, _EntityResolutionResponse
            )
        except LLMClientError:
            return None
        return result.selected_graph_id


class LLMTemporalUpdateModel:
    """`ingestion.ports.TemporalUpdateModel` backed by a real LLM call."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def classify_update(self, *, new_fact: FactState, prior_fact: FactState) -> TemporalRelation:
        user_prompt = (
            f"Prior fact (observed {prior_fact.observed_at.isoformat()}): {prior_fact.text!r}\n"
            f"New fact (observed {new_fact.observed_at.isoformat()}): {new_fact.text!r}\n\n"
            "Classify the relationship: correction, state_change, no_update, or unresolved."
        )
        try:
            result = self._client.structured_completion(
                TEMPORAL_UPDATE_SYSTEM_PROMPT, user_prompt, _TemporalUpdateResponse
            )
        except LLMClientError:
            return TemporalRelation.UNRESOLVED
        return TemporalRelation(result.relation)
