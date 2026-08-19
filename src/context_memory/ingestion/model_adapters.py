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
from context_memory.core.logging import get_logger, timed_operation
from context_memory.core.resolution import EntityProfile, FactState, TemporalRelation

logger = get_logger(__name__)

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
        with timed_operation(logger, "entity_resolution.disambiguate", {"surface": surface, "candidates_count": len(candidates)}) as ctx:
            user_prompt = (
                f"Surface form to resolve: {surface!r}\n\n"
                f"Candidates (context {context_id!r}):\n{_format_candidates(candidates)}\n\n"
                "Return the graph_id of the matching candidate, or null if none match."
            )
            try:
                result = self._client.structured_completion(
                    ENTITY_RESOLUTION_SYSTEM_PROMPT, user_prompt, _EntityResolutionResponse
                )
                ctx["selected_id"] = result.selected_graph_id
                return result.selected_graph_id
            except LLMClientError as error:
                logger.warning("Entity resolution model error for surface %r: %s", surface, error)
                return None


class LLMTemporalUpdateModel:
    """`ingestion.ports.TemporalUpdateModel` backed by a real LLM call."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def classify_update(self, *, new_fact: FactState, prior_fact: FactState) -> TemporalRelation:
        with timed_operation(logger, "temporal_update.classify", {"predicate": new_fact.predicate_key, "prior_id": prior_fact.fact_id, "new_id": new_fact.fact_id}) as ctx:
            user_prompt = (
                f"Prior fact (observed {prior_fact.observed_at.isoformat()}): {prior_fact.text!r}\n"
                f"New fact (observed {new_fact.observed_at.isoformat()}): {new_fact.text!r}\n\n"
                "Classify the relationship: correction, state_change, no_update, or unresolved."
            )
            try:
                result = self._client.structured_completion(
                    TEMPORAL_UPDATE_SYSTEM_PROMPT, user_prompt, _TemporalUpdateResponse
                )
                ctx["classified_relation"] = result.relation
                return TemporalRelation(result.relation)
            except LLMClientError as error:
                logger.warning("Temporal update classification error: %s", error)
                return TemporalRelation.UNRESOLVED


FACT_EXTRACTION_SYSTEM_PROMPT = (
    "You are a long-term memory extraction assistant. Extract atomic, enduring facts about the user "
    "or assistant from the dialogue turn. Ignore trivial chitchat, transient pleasantries, and greetings. "
    "For each extracted fact, provide:\n"
    "- text: The concise, self-contained atomic factual assertion.\n"
    "- action: 'ADD' for new facts, 'UPDATE' if updating an existing attribute, 'DELETE' if invalidating a past fact.\n"
    "- predicate_key: A clean snake_case predicate category (e.g. 'pet_name', 'favorite_food', 'location', 'hobby').\n"
    "- entities: List of specific named entities mentioned in the fact (e.g. ['Max', 'Seattle']).\n"
    "- confidence: Confidence score between 0.0 and 1.0.\n"
    "- exact_quote: The exact substring in the input text that provides direct evidence for this fact."
)


class _ExtractedFactItem(BaseModel):
    text: str
    action: Literal["ADD", "UPDATE", "DELETE"] = "ADD"
    predicate_key: str | None = None
    entities: list[str] = []
    confidence: float = 0.95
    exact_quote: str | None = None


class _FactExtractionResponse(BaseModel):
    facts: list[_ExtractedFactItem] = []


class LLMExtractor:
    """`ingestion.ports.Extractor` backed by structured LLM fact distillation."""

    extractor_name = "llm-extractor"
    extractor_version = "v1"

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def extract(self, record: ContextRecord) -> Sequence[ExtractionDraft]:
        import uuid
        from context_memory.core.enums import MemoryScope, MemoryType
        from context_memory.core.models import EntityCandidate, ExtractionDraft

        content = record.content
        if not content or not content.strip():
            return ()

        with timed_operation(logger, "extractor.extract", {"record_id": record.record_id, "content_len": len(content)}) as ctx:
            user_prompt = f"Speaker: {record.actor_role}\nContent: {content}\n\nExtract atomic facts:"
            try:
                res = self._client.structured_completion(
                    FACT_EXTRACTION_SYSTEM_PROMPT, user_prompt, _FactExtractionResponse
                )
            except Exception as e:
                logger.error("LLMExtractor failed structured extraction for record %s: %s", record.record_id, e)
                return ()

            drafts: list[ExtractionDraft] = []
            for item in res.facts:
                if not item.text or not item.text.strip():
                    continue
                # Calculate source span offsets
                quote = item.exact_quote or item.text
                start = content.find(quote)
                if start == -1:
                    start = 0
                    end = len(content)
                else:
                    end = start + len(quote)

                if end <= start:
                    end = max(len(content), start + 1)

                entities = tuple(EntityCandidate(surface=e.strip(), entity_type=None) for e in item.entities if e.strip())
                draft = ExtractionDraft(
                    candidate_id=f"cand-{uuid.uuid4().hex[:12]}",
                    text=item.text.strip(),
                    source_start=start,
                    source_end=end,
                    confidence=max(0.0, min(1.0, float(item.confidence))),
                    memory_type=MemoryType.SEMANTIC,
                    scope_type=MemoryScope.USER,
                    scope_id="user",
                    entities=entities,
                    action=item.action,
                    predicate_key=item.predicate_key,
                )
                drafts.append(draft)
            ctx["extracted_drafts"] = len(drafts)
            return tuple(drafts)
