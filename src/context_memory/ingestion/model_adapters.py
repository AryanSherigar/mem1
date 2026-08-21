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

from context_memory.core.config import Config
from context_memory.core.llm_client import LLMClient, LLMClientError
from context_memory.core.logging import get_logger, timed_operation
from context_memory.core.resolution import EntityProfile, FactState, TemporalRelation

logger = get_logger(__name__)


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

    def __init__(self, client: LLMClient, config: Config | None = None) -> None:
        self._client = client
        self._config = config or Config()

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
                    self._config.entity_resolution_system_prompt, user_prompt, _EntityResolutionResponse,
                    temperature=self._config.llm_temperature, max_tokens=self._config.entity_resolution_max_tokens,
                    timeout=self._config.entity_resolution_timeout_seconds,
                    max_retries=self._config.llm_structured_retry_attempts,
                )
                ctx["selected_id"] = result.selected_graph_id
                return result.selected_graph_id
            except Exception as error:
                # Was `except LLMClientError` only, which does not catch
                # openai.APITimeoutError (a different exception hierarchy) --
                # confirmed live: a single slow call propagated straight out of
                # this bounded-decision port, past graph_plan_builder's loop,
                # and aborted the *entire chunk* via orchestrator.run_chunk's
                # generic except, losing every fact in that turn over one
                # resolution timing out. This port's contract is already
                # "return None on any doubt" (EntityRegistry.resolve treats
                # UNRESOLVED as no-forced-link, never invents a candidate); a
                # provider timeout is exactly that kind of doubt.
                logger.warning("Entity resolution model error for surface %r: %s", surface, error)
                return None


class LLMTemporalUpdateModel:
    """`ingestion.ports.TemporalUpdateModel` backed by a real LLM call."""

    def __init__(self, client: LLMClient, config: Config | None = None) -> None:
        self._client = client
        self._config = config or Config()

    def classify_update(self, *, new_fact: FactState, prior_fact: FactState) -> TemporalRelation:
        with timed_operation(logger, "temporal_update.classify", {"predicate": new_fact.predicate_key, "prior_id": prior_fact.fact_id, "new_id": new_fact.fact_id}) as ctx:
            user_prompt = (
                f"Prior fact (observed {prior_fact.observed_at.isoformat()}): {prior_fact.text!r}\n"
                f"New fact (observed {new_fact.observed_at.isoformat()}): {new_fact.text!r}\n\n"
                "Classify the relationship: correction, state_change, no_update, or unresolved."
            )
            try:
                result = self._client.structured_completion(
                    self._config.temporal_update_system_prompt, user_prompt, _TemporalUpdateResponse,
                    temperature=self._config.llm_temperature, max_tokens=self._config.temporal_update_max_tokens,
                    timeout=self._config.temporal_update_timeout_seconds,
                    max_retries=self._config.llm_structured_retry_attempts,
                )
                ctx["classified_relation"] = result.relation
                return TemporalRelation(result.relation)
            except Exception as error:
                # Was `except LLMClientError` only. Confirmed live during a
                # 30-instance demo run: openai.APITimeoutError from Bedrock
                # propagated straight out of this bounded-decision port and
                # aborted the whole chunk via orchestrator.run_chunk's generic
                # except -- 42 timeouts in one instance, each one losing every
                # fact in that turn (not just the one supersession check) and
                # costing up to timeout x (sdk_max_retries+1) of wall clock
                # before the exception even surfaced. UNRESOLVED is this
                # port's existing "no clear basis to decide" outcome
                # (TemporalUpdateClassifier gates SUPERSEDES on
                # CORRECTION/STATE_CHANGE only), so a provider timeout
                # degrading to it is the contract working as designed, not a
                # new behavior.
                logger.warning("Temporal update classification error: %s", error)
                return TemporalRelation.UNRESOLVED


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

    def __init__(self, client: LLMClient, config: Config | None = None) -> None:
        self._client = client
        self._config = config or Config()

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
                    self._config.fact_extraction_system_prompt, user_prompt, _FactExtractionResponse,
                    temperature=self._config.llm_temperature, max_tokens=self._config.extractor_max_tokens,
                    timeout=self._config.extractor_timeout_seconds,
                    max_retries=self._config.llm_structured_retry_attempts,
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
