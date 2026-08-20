"""HydraDB-backed lookup for `find_existing_facts` (see `graph_plan_builder.py`).

The one legitimate reason to read `Fact` nodes for something other than
retrieval: `GraphPlanBuilder` needs "the active facts for this subject entity
and predicate" before it can decide whether a new UPDATE-action fact
supersedes one of them (ADR-030's SUPERSEDES mechanism). Without a real
implementation of this callable, that decision never runs — confirmed: neither
`api/routes.py` nor `evaluation/benchmark_runner.py` constructed one, so
`TemporalUpdateClassifier` was fully wired but never actually invoked, and
every knowledge-update question degrades to "old and new fact rank equally."
"""

from __future__ import annotations

from datetime import datetime, timezone

from context_memory.client.hydradb_http import HydraHttpTransport
from context_memory.core.logging import get_logger, timed_operation
from context_memory.core.resolution import FactState

logger = get_logger(__name__)

# Plain (non-UNWIND) multi-property MATCH — confirmed live against the running
# HydraDB build to work, unlike UNWIND-batched reads (see retrieval.py's Phase
# 2 comments for the extensive grammar restrictions that forced a per-fact
# loop there). This is already a single, non-batched query, so none of that
# applies here.
_FIND_EXISTING_CYPHER = """
MATCH (f:Fact {context_id: $context_id, predicate_key: $predicate_key, is_current: true})-[:ABOUT]->(e {id: $subject_id})
RETURN f.logical_key AS logical_key, f.text AS text, f.observed_at AS observed_at,
       f.valid_from AS valid_from, f.valid_to AS valid_to
"""


class HydraFactLookup:
    """`graph_plan_builder.FindExistingFacts` backed by a live HydraDB query.

    Pass `.find_existing` (not the instance) as the `find_existing_facts`
    constructor arg to `IngestionOrchestrator` — that's the callable shape
    `GraphPlanBuilder.build()` expects: `(context_id, subject_entity_id,
    predicate_key) -> list[FactState]`.
    """

    def __init__(self, transport: HydraHttpTransport) -> None:
        self._transport = transport

    def find_existing(self, context_id: str, subject_entity_id: int, predicate_key: str) -> list[FactState]:
        with timed_operation(
            logger, "fact_lookup.find_existing",
            {"context_id": context_id, "subject_entity_id": subject_entity_id, "predicate_key": predicate_key},
        ) as ctx:
            try:
                rows = self._transport.read(
                    _FIND_EXISTING_CYPHER,
                    {"context_id": context_id, "predicate_key": predicate_key, "subject_id": subject_entity_id},
                    None,
                )
            except Exception as error:
                # Never blocks ingestion: a failed lookup means "no known prior
                # fact," which just skips SUPERSEDES for this one write (same
                # as if this callable were never supplied), not a dropped turn.
                logger.warning(
                    "find_existing_facts lookup failed for %s/%s/%s: %s",
                    context_id, subject_entity_id, predicate_key, error,
                )
                return []

            results: list[FactState] = []
            for row in rows:
                logical_key = row.get("logical_key")
                text = row.get("text")
                observed_at = row.get("observed_at")
                if not logical_key or not text or observed_at is None:
                    continue
                try:
                    valid_from = row.get("valid_from")
                    valid_to = row.get("valid_to")
                    results.append(
                        FactState(
                            fact_id=str(logical_key).removeprefix("fact:"),
                            subject_entity_id=subject_entity_id,
                            predicate_key=predicate_key,
                            text=str(text),
                            observed_at=datetime.fromtimestamp(int(observed_at), tz=timezone.utc),
                            valid_from=datetime.fromtimestamp(int(valid_from), tz=timezone.utc) if valid_from else None,
                            # 9999999999 is the sentinel "no upper bound" (see
                            # graph_plan_builder.py's _fact_node) — real None,
                            # not a real-world date near year 2286.
                            valid_to=(
                                datetime.fromtimestamp(int(valid_to), tz=timezone.utc)
                                if valid_to and int(valid_to) < 9999999999
                                else None
                            ),
                        )
                    )
                except Exception as error:
                    logger.warning("find_existing_facts: skipping malformed row for %s: %s", logical_key, error)

            ctx["found"] = len(results)
            return results
