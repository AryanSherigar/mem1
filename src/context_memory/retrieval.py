"""Phase 2: Hybrid Retrieval Engine implementing the 4-phase retrieval pipeline."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from collections.abc import Sequence
from typing import Any
from pydantic import BaseModel

from context_memory.client.hydradb_http import HydraHttpTransport
from context_memory.core.config import Config
from context_memory.core.llm_client import LLMClient
from context_memory.core.logging import get_logger, timed_operation
from context_memory.ingestion.embedding import SentenceTransformerEmbedder

logger = get_logger(__name__)


@dataclass
class ScoredFact:
    fact_id: str
    text: str
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    structural_score: float = 0.0
    entity_boost: float = 0.0
    composite_score: float = 0.0
    speaker: str | None = None
    observed_at: int | None = None
    # 1-indexed rank within each channel's own result list -- None means the
    # fact did not appear in that channel at all. Captured at seeding time
    # (the SQL already returns rows in rank order) so Phase 3 can fuse by
    # Reciprocal Rank Fusion instead of summing raw, differently-scaled scores.
    semantic_rank: int | None = None
    keyword_rank: int | None = None


class DateRange(BaseModel):
    valid_from: datetime | None = None
    valid_to: datetime | None = None


class QueryRewriterOutput(BaseModel):
    decomposed_queries: list[str]
    synonyms: list[str]


class TemporalQueryResolver:
    def __init__(self, llm_client: LLMClient, config: Config | None = None) -> None:
        self._llm = llm_client
        self._config = config or Config()

    def resolve(self, question: str, question_date: datetime) -> DateRange:
        with timed_operation(logger, "retrieval.phase0.temporal_resolver", {"question_len": len(question)}) as ctx:
            system_prompt = self._config.temporal_resolver_system_prompt_template.format(
                question_date=question_date.isoformat()
            )
            try:
                response = self._llm.structured_completion(
                    system_prompt, question, DateRange,
                    temperature=self._config.llm_temperature, max_tokens=self._config.temporal_resolver_max_tokens,
                    timeout=self._config.temporal_resolver_timeout_seconds,
                    max_retries=self._config.llm_structured_retry_attempts,
                )
                if isinstance(response, DateRange):
                    buffer = timedelta(days=self._config.retrieval_temporal_buffer_days)
                    if response.valid_from:
                        response.valid_from -= buffer
                    if response.valid_to:
                        response.valid_to += buffer
                    ctx["valid_from"] = str(response.valid_from)
                    ctx["valid_to"] = str(response.valid_to)
                    return response
            except Exception as e:
                logger.warning("Temporal query resolver error: %s", e)
            return DateRange()


class QueryRewriter:
    def __init__(self, llm_client: LLMClient, config: Config | None = None) -> None:
        self._llm = llm_client
        self._config = config or Config()

    def rewrite(self, question: str) -> QueryRewriterOutput:
        with timed_operation(logger, "retrieval.phase0.query_rewriter", {"question_len": len(question)}) as ctx:
            try:
                response = self._llm.structured_completion(
                    self._config.query_rewriter_system_prompt, question, QueryRewriterOutput,
                    temperature=self._config.llm_temperature, max_tokens=self._config.query_rewriter_max_tokens,
                    timeout=self._config.query_rewriter_timeout_seconds,
                    max_retries=self._config.llm_structured_retry_attempts,
                )
                if isinstance(response, QueryRewriterOutput):
                    ctx["decomposed_count"] = len(response.decomposed_queries)
                    ctx["synonyms_count"] = len(response.synonyms)
                    return response
            except Exception as e:
                logger.warning("Query rewriter error: %s", e)
            return QueryRewriterOutput(decomposed_queries=[question], synonyms=[])


class HybridRetrievalEngine:
    def __init__(
        self,
        llm_client: LLMClient,
        embedder: SentenceTransformerEmbedder,
        pg_connection: object,
        hydra_client: HydraHttpTransport,
        config: Config | None = None,
        temporal_resolver_client: LLMClient | None = None,
        query_rewriter_client: LLMClient | None = None,
    ) -> None:
        """`llm_client` is the reader/answer-synthesis role, and also the default
        for temporal resolution/query rewriting when no role-specific client is
        given — so a single fake/client injected here still drives every LLM call
        in this engine, same as before role-specific clients existed. Pass
        `temporal_resolver_client`/`query_rewriter_client` explicitly (see
        `routes.py`, which builds them from `Config`) to actually split roles
        across different models."""
        self._llm = llm_client
        self._embedder = embedder
        self._pg = pg_connection
        self._hydra = hydra_client
        self._config = config or Config()
        self._temporal_resolver_client = temporal_resolver_client or llm_client
        self._query_rewriter_client = query_rewriter_client or llm_client

    def retrieve_and_answer(self, context_id: str, question: str, question_date: datetime, top_k: int | None = None) -> str:
        top_k = top_k if top_k is not None else self._config.retrieval_top_k
        with timed_operation(logger, "retrieval.retrieve_and_answer", {"context_id": context_id, "question_len": len(question)}) as ctx:
            # Phase 0: Temporal Resolution & Query Rewriting. Two independent
            # LLM calls -- the rewriter doesn't use temporal_bounds and the
            # resolver doesn't use expanded_query -- run concurrently rather
            # than one after the other; each is a real network round trip, so
            # this halves Phase 0's wall time for free.
            resolver = TemporalQueryResolver(self._temporal_resolver_client, self._config)
            rewriter = QueryRewriter(self._query_rewriter_client, self._config)
            with ThreadPoolExecutor(max_workers=2) as pool:
                temporal_future = pool.submit(resolver.resolve, question, question_date)
                rewriter_future = pool.submit(rewriter.rewrite, question)
                temporal_bounds = temporal_future.result()
                expanded_query = rewriter_future.result()

            # Phase 1: Semantic + Keyword Seeding
            seed_facts = self._semantic_and_keyword_seeding(
                context_id, question, expanded_query, top_k
            )
            ctx["seed_facts_count"] = len(seed_facts)

            # Phase 2: Graph Expansion & Temporal Filtering
            graph_data = self._hydradb_graph_expansion(context_id, seed_facts, temporal_bounds, question_date)
            ctx["graph_expanded_facts"] = len(graph_data)

            # Fallback: Populate missing fact text from PostgreSQL if empty
            missing_text_fids = [fid for fid, fact in seed_facts.items() if not fact.text]
            if missing_text_fids:
                try:
                    with self._pg.cursor() as cursor:
                        cursor.execute(
                            "SELECT fact_id, raw_text FROM fact_search_index WHERE context_id = %s AND fact_id = ANY(%s)",
                            (context_id, missing_text_fids)
                        )
                        for r_fid, r_text in cursor.fetchall():
                            if str(r_fid) in seed_facts:
                                seed_facts[str(r_fid)].text = r_text
                except Exception as e:
                    logger.debug("PostgreSQL fallback fact_search_index query skipped: %s", e)

            # Phase 3: 4-Factor Composite Scoring & Synthesis
            answer = self._composite_scoring_and_synthesis(question, seed_facts, graph_data, top_k, expanded_query, context_id)
            ctx["answer_len"] = len(answer)
            return answer

    def _semantic_and_keyword_seeding(
        self, context_id: str, question: str, expanded_query: QueryRewriterOutput, top_k: int
    ) -> dict[str, ScoredFact]:
        with timed_operation(logger, "retrieval.phase1.seeding", {"context_id": context_id, "top_k": top_k}) as ctx:
            limit = max(top_k * self._config.retrieval_overfetch_multiplier, self._config.retrieval_overfetch_floor)
            facts = {}

            # 1. Semantic Search
            vector = self._embedder.embed(question)
            vector_literal = "[" + ",".join(repr(float(v)) for v in vector) + "]"

            with self._pg.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT subject_id, embedding <=> %s::vector AS distance
                    FROM memory_embeddings
                    WHERE context_id = %s AND subject_kind = 'fact' AND is_active = true
                    ORDER BY distance ASC
                    LIMIT %s
                    """,
                    (vector_literal, context_id, limit)
                )
                for position, row in enumerate(cursor.fetchall(), start=1):
                    fact_id = str(row[0])
                    distance = float(row[1]) if row[1] is not None else 0.0
                    semantic_score = 1.0 / (1.0 + distance)
                    if fact_id not in facts:
                        facts[fact_id] = ScoredFact(fact_id, "", semantic_score=semantic_score, semantic_rank=position)
                    else:
                        facts[fact_id].semantic_score = max(facts[fact_id].semantic_score, semantic_score)
                        facts[fact_id].semantic_rank = min(facts[fact_id].semantic_rank or position, position)

            # 2. Keyword Search (BM25). Use websearch_to_tsquery with OR combination and @@ matching
            terms = [t.strip() for t in (expanded_query.synonyms + expanded_query.decomposed_queries + [question]) if t.strip()]
            search_query_str = " OR ".join(f'"{t}"' if " " in t else t for t in terms) if terms else question
            if search_query_str:
                with self._pg.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT fact_id, raw_text, ts_rank_cd(text_tsvector, websearch_to_tsquery('english', %s)) AS rank
                        FROM fact_search_index
                        WHERE context_id = %s AND is_active = true AND text_tsvector @@ websearch_to_tsquery('english', %s)
                        ORDER BY rank DESC
                        LIMIT %s
                        """,
                        (search_query_str, context_id, search_query_str, limit)
                    )
                    position = 0
                    for row in cursor.fetchall():
                        fact_id = str(row[0])
                        raw_text = str(row[1])
                        rank = float(row[2]) if row[2] is not None else 0.0
                        if rank <= 0.0:
                            continue
                        position += 1  # dense rank over accepted rows only, not the raw fetch
                        keyword_score = rank / (1.0 + rank)
                        if fact_id not in facts:
                            facts[fact_id] = ScoredFact(fact_id, raw_text, keyword_score=keyword_score, keyword_rank=position)
                        else:
                            facts[fact_id].keyword_score = max(facts[fact_id].keyword_score, keyword_score)
                            facts[fact_id].keyword_rank = min(facts[fact_id].keyword_rank or position, position)
                            if not facts[fact_id].text:
                                facts[fact_id].text = raw_text

            ctx["total_seeded_facts"] = len(facts)
            return facts

    def _hydradb_graph_expansion(
        self, context_id: str, seed_facts: dict[str, ScoredFact], temporal_bounds: DateRange, query_epoch: datetime
    ) -> dict[str, dict[str, Any]]:
        if not seed_facts:
            return {}

        with timed_operation(logger, "retrieval.phase2.graph_expansion", {"context_id": context_id, "seed_count": len(seed_facts)}) as ctx:
            graph_data = {}
            fact_logical_keys = [f"fact:{fid}" for fid in seed_facts.keys()]

            # Resolve each fact's integer graph_id from Postgres's own
            # graph_id_registry (the same registry ingestion allocated from —
            # this is its intended read side, not a new mechanism): HydraDB's
            # writes only ever match by `id`, and its UNWIND-batched reads
            # turned out just as narrow after four increasingly specific
            # rejections (labeled node patterns rejected, multi-property
            # patterns rejected, bare-scalar UNWIND rejected, and finally
            # "UNWIND batch read first projection must be the source field" —
            # all confirmed live). Rather than keep chasing an UNWIND-read
            # grammar that looks purpose-built for writes only, this queries
            # per-fact instead — the same shape graph_schema_proposal.md's own
            # Phase 2 pseudocode already used (`for fid in seed_fact_ids:
            # hydra_driver.execute(...)`), not a new pattern.
            graph_id_by_fact_key: dict[str, int] = {}
            try:
                with self._pg.cursor() as cursor:
                    cursor.execute(
                        "SELECT logical_key, graph_id FROM graph_id_registry WHERE node_kind = 'fact' AND context_id = %s AND logical_key = ANY(%s)",
                        (context_id, fact_logical_keys),
                    )
                    for logical_key, graph_id in cursor.fetchall():
                        graph_id_by_fact_key[logical_key] = int(graph_id)
            except Exception as e:
                logger.warning("graph_id_registry lookup error: %s", e)

            fact_key_by_graph_id = {gid: key for key, gid in graph_id_by_fact_key.items()}

            # 1. Fetch node bitemporal properties plus any connected entity, one
            # fact at a time. A plain (non-UNWIND) MATCH ... OPTIONAL MATCH ...
            # RETURN is fine — only the UNWIND-prefixed combined form was ever
            # rejected.
            #
            # Fetched concurrently, not one HTTP round trip after another: this
            # is a genuine N+1 (one call per seeded fact -- 60-80+ typical, per
            # the overfetch floor/multiplier) and every call is independent and
            # read-only. HydraHttpTransport holds no mutable per-call state (see
            # its own docstring/implementation -- headers/URL are fixed at
            # construction, each .read() is self-contained), so this has none
            # of the shared-connection hazard that keeps ingestion's writes
            # serial (PostgresGraphManifestStore et al. share one psycopg
            # connection; this is a different transport, a different direction
            # -- reads, not writes -- and has no such constraint).
            raw_nodes = []
            entity_key_by_fact: dict[str, str] = {}

            def _fetch_node(fact_key: str, graph_id: int) -> tuple[str, Sequence[dict[str, object]] | None]:
                node_cypher = f"""
                MATCH (f {{id: {int(graph_id)}}})
                OPTIONAL MATCH (f)-[:ABOUT]->(e)
                RETURN
                    f.text AS text,
                    f.speaker AS speaker,
                    f.valid_from AS valid_from,
                    f.valid_to AS valid_to,
                    f.observed_at AS observed_at,
                    f.superseded_at AS superseded_at,
                    f.memory_scope AS memory_scope,
                    e.logical_key AS entity_key
                """
                try:
                    # graph_id is inlined above (int()-coerced, so not injectable);
                    # no parameters remain to bind.
                    return fact_key, self._hydra.read(node_cypher, {}, None)
                except Exception as e:
                    logger.warning("HydraDB node query error for %s: %s", fact_key, e)
                    return fact_key, None

            with ThreadPoolExecutor(max_workers=self._config.retrieval_graph_fetch_workers) as pool:
                futures = [
                    pool.submit(_fetch_node, fact_key, graph_id)
                    for fact_key, graph_id in graph_id_by_fact_key.items()
                ]
                for future in as_completed(futures):
                    fact_key, rows = future.result()
                    if rows is None:
                        continue
                    for row in rows:
                        raw_nodes.append({**row, "fact_key": fact_key})
                        if row.get("entity_key"):
                            entity_key_by_fact[fact_key] = row["entity_key"]

            # Apply Bitemporal filtering in Python
            valid_fact_keys = set()
            entities = set()
            entity_to_facts = {}

            query_int = int(query_epoch.timestamp())
            chat_ttl_limit = int((query_epoch - timedelta(hours=self._config.retrieval_chat_ttl_hours)).timestamp())

            for row in raw_nodes:
                fact_key = row["fact_key"]
                valid_from = row.get("valid_from")
                valid_to = row.get("valid_to")
                observed_at = row.get("observed_at")
                superseded_at = row.get("superseded_at")
                memory_scope = row.get("memory_scope")

                # Temporal logic filtering
                if valid_from is not None and valid_from > query_int:
                    continue
                if valid_to is not None and valid_to < query_int:
                    continue
                if observed_at is not None and observed_at > query_int:
                    continue
                if superseded_at is not None and superseded_at < query_int:
                    continue

                # Chat scope TTL enforcement
                if memory_scope == "chat" and observed_at is not None and observed_at < chat_ttl_limit:
                    continue

                valid_fact_keys.add(fact_key)

                # Populate text, speaker, and observed_at on seed_facts
                fid = fact_key.replace("fact:", "", 1)
                if fid in seed_facts:
                    if row.get("text"):
                        seed_facts[fid].text = str(row["text"])
                    if row.get("speaker"):
                        seed_facts[fid].speaker = str(row["speaker"])
                    if observed_at is not None:
                        try:
                            seed_facts[fid].observed_at = int(observed_at)
                        except (ValueError, TypeError):
                            pass

                entity_key = entity_key_by_fact.get(fact_key)
                if entity_key:
                    entities.add(entity_key)
                    entity_to_facts.setdefault(entity_key, []).append(fact_key)

            # Prune seed_facts that failed temporal validation
            pruned_count = 0
            for fid in list(seed_facts.keys()):
                if f"fact:{fid}" not in valid_fact_keys:
                    del seed_facts[fid]
                    pruned_count += 1

            ctx["temporal_pruned_facts"] = pruned_count
            ctx["retained_valid_facts"] = len(seed_facts)

            if not seed_facts:
                return {}

            path_count_by_fact = {}
            hop_count_by_fact = {}

            # 2. Execute algo.MSpaths.
            #
            # Values are inlined rather than parameterized because this HydraDB
            # build rejects a list parameter here outright ("composite parameter
            # $entities is only supported as an UNWIND input", confirmed live),
            # so `$entities` is not an option for this call.
            #
            # Escaping via json.dumps, not a hand-rolled quote-doubler. Doubling
            # only `'` leaves backslashes untouched, and an entity ending in one
            # produces `'ent\'` -- the backslash escapes the closing quote, the
            # literal never terminates, and the rest of the query is swallowed.
            # That input is reachable: entity names come from LLM extraction of
            # user content, and `canonicalize_entity_surface` only NFKC-
            # normalizes, collapses whitespace, and casefolds -- it strips
            # neither backslashes nor quotes. json.dumps emits a correctly
            # escaped double-quoted literal, which Cypher accepts.
            if entities:
                escaped_entities = ", ".join(json.dumps(e) for e in entities)
                path_cypher = f"""
                CALL algo.MSpaths({{
                    sourceLabel: 'Entity',
                    sourceProperty: 'logical_key',
                    sourceValues: [{escaped_entities}],
                    relTypes: ['ABOUT'],
                    maxLen: {int(self._config.retrieval_graph_max_hops)}
                }}) YIELD path
                RETURN path
                """
                try:
                    path_res = self._hydra.read(path_cypher, {}, None)
                    for row in path_res:
                        path = row.get("path", [])
                        if len(path) >= 3:
                            hops = len(path) // 2
                            start_node = path[0]
                            end_node = path[-1]

                            start_key = start_node.get("logical_key") if isinstance(start_node, dict) else None
                            end_key = end_node.get("logical_key") if isinstance(end_node, dict) else None

                            for entity_key in (start_key, end_key):
                                if entity_key in entity_to_facts:
                                    for fact_key in entity_to_facts[entity_key]:
                                        path_count_by_fact[fact_key] = path_count_by_fact.get(fact_key, 0) + 1
                                        hop_count_by_fact[fact_key] = min(hop_count_by_fact.get(fact_key, hops), hops)
                except Exception as e:
                    logger.debug("algo.MSpaths query skipped: %s", e)

            for f_id in seed_facts.keys():
                fact_key = f"fact:{f_id}"
                entity_key = entity_key_by_fact.get(fact_key)
                entity_fact_count = len(entity_to_facts.get(entity_key, ())) if entity_key else 0
                graph_data[f_id] = {
                    # Carried through so scoring can apply the entity boost only
                    # to entities the *query* actually mentions, per
                    # FINAL_ARCHITECTURE.md's `if entity in query_entities`.
                    "entity_key": entity_key,
                    "hop_count": hop_count_by_fact.get(fact_key, 1),
                    "path_count": path_count_by_fact.get(fact_key, 0),
                    "entity_fact_count": entity_fact_count,
                }

            return graph_data

    def _query_entity_terms(self, question: str, expanded_query: QueryRewriterOutput | None) -> set[str]:
        """Lowercased tokens from the question (and its rewritten forms) used to
        decide whether a fact's linked entity is one the *query* mentions."""
        parts = [question]
        if expanded_query is not None:
            parts.extend(expanded_query.synonyms)
            parts.extend(expanded_query.decomposed_queries)
        text = " ".join(parts).casefold()
        return {token for token in re.findall(r"[a-z0-9]+", text) if len(token) > 2}

    def _sibling_facts(self, context_id: str, fact_ids: list[str], question: str) -> dict[str, str]:
        """Returns `{fact_id: raw_text}` for facts extracted from the same
        source turn as any fact in `fact_ids` -- the "neighboring turn" tier
        of FINAL_ARCHITECTURE.md's ADR-005 progressive evidence expansion
        (fact+span -> neighboring turn -> full chunk), accepted at design time
        but never actually implemented until now.

        Why this matters: a single turn is routinely split into several
        atomic facts at extraction time (by design), and those facts are then
        scored and ranked *independently* in Phase 3. Traced live: "User sold
        20 potted herb plants" and "Each potted herb plant was sold for $7.5"
        come from the same turn, but only the second one scored high enough
        to reach the reader on its own.

        Selection is Reciprocal-adjacent but not RRF: it's Semantic
        Continuity-Aware Retrieval (SCAR; Zhong et al. 2026,
        arxiv.org/abs/2606.16661), chosen over a flat "pull every same-turn
        fact" or a bare `ORDER BY` after the first version of this method hit
        a real, measured failure -- a shared LIMIT across every anchor's
        siblings starved out a needed fact (4 candidates for its own turn,
        available, but never reached: unrelated turns' siblings filled the
        cap first because nothing was scored or ordered). A same-turn fact
        still is not automatically the right one to add -- the actually-
        missing gold fact can just as easily live in a *different* turn
        entirely, which no amount of neighbor expansion reaches; the goal
        here is only to stop within-turn continuity from being either
        all-or-nothing or arbitrarily ordered.

        SCAR scores each candidate neighbor n of anchor a:
            S(a,n) = cos(query, n) - lambda * (1 - cos(a, n))
        and keeps n only if it clears a threshold set *relative to its own
        anchor's* query relevance:
            S(a,n) > gamma * cos(query, a)
        so a weakly-relevant anchor sets a low bar and a strongly-relevant
        one demands genuinely comparable neighbors, rather than one fixed cut
        for every fact regardless of how well it matched in the first place.
        Paper defaults (lambda=0.1, gamma=0.80) are used unchanged -- no
        tuning data of our own exists yet to justify moving them.
        """
        if not fact_ids:
            return {}
        vector = self._embedder.embed(question)
        vector_literal = "[" + ",".join(repr(float(v)) for v in vector) + "]"
        lam = self._config.retrieval_sibling_continuity_penalty
        gamma = self._config.retrieval_sibling_relevance_ratio

        with self._pg.cursor() as cursor:
            cursor.execute(
                """
                SELECT sibling.subject_id, fsi.raw_text,
                       (anchor.embedding <=> %s::vector) AS anchor_query_distance,
                       (sibling.embedding <=> %s::vector) AS sibling_query_distance,
                       (anchor.embedding <=> sibling.embedding) AS boundary_distance
                FROM memory_embeddings anchor
                JOIN memory_embeddings sibling
                  ON sibling.source_chunk_id = anchor.source_chunk_id
                 AND sibling.context_id = anchor.context_id
                 AND sibling.subject_kind = 'fact'
                 AND sibling.is_active = true
                JOIN fact_search_index fsi
                  ON fsi.fact_id = sibling.subject_id AND fsi.is_active = true
                WHERE anchor.context_id = %s
                  AND anchor.subject_kind = 'fact'
                  AND anchor.subject_id = ANY(%s)
                  AND sibling.subject_id != ALL(%s)
                """,
                (vector_literal, vector_literal, context_id, fact_ids, fact_ids),
            )
            rows = cursor.fetchall()

        # pgvector's <=> is cosine DISTANCE (0 = identical), so similarity is
        # (1 - distance). A sibling reachable via more than one anchor in this
        # batch is scored against each and kept at its best-scoring pairing --
        # SCAR is defined per anchor-neighbor pair, and there is no reason to
        # penalize a fact for happening to sit next to two relevant facts
        # instead of one.
        best: dict[str, tuple[float, str]] = {}
        for fact_id, raw_text, anchor_qd, sibling_qd, boundary_d in rows:
            anchor_relevance = 1.0 - float(anchor_qd)
            sibling_relevance = 1.0 - float(sibling_qd)
            score = sibling_relevance - lam * float(boundary_d)
            if score <= gamma * anchor_relevance:
                continue
            fact_id = str(fact_id)
            if fact_id not in best or score > best[fact_id][0]:
                best[fact_id] = (score, raw_text)

        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[: self._config.retrieval_sibling_fact_limit]
        return {fact_id: text for fact_id, (_, text) in ranked}

    def _composite_scoring_and_synthesis(
        self, question: str, facts: dict[str, ScoredFact], graph_data: dict, top_k: int,
        expanded_query: QueryRewriterOutput | None = None, context_id: str | None = None,
    ) -> str:
        with timed_operation(logger, "retrieval.phase3.scoring_and_synthesis", {"facts_to_score": len(facts)}) as ctx:
            path_cap = self._config.retrieval_structural_path_cap
            boost_cap = self._config.retrieval_entity_boost_cap
            rrf_k = self._config.retrieval_rrf_k
            query_terms = self._query_entity_terms(question, expanded_query)

            for f_id, fact in facts.items():
                g = graph_data.get(f_id, {"hop_count": 1, "path_count": 0, "entity_fact_count": 0})
                hop_count = g.get("hop_count", 1) or 1
                path_count = g.get("path_count", 0)
                entity_fact_count = g.get("entity_fact_count", 0)

                fact.structural_score = (1.0 / hop_count) * min(path_count, path_cap) / path_cap

                # Boost only entities the QUERY mentions -- FINAL_ARCHITECTURE.md
                # §"Entity boost" gates on `if entity in query_entities`, and
                # dropping that gate is not a small deviation: it hands the same
                # flat boost to every entity-linked fact in the corpus.
                # Measured on LongMemEval 118b2229 ("How long is my daily
                # commute to work?"): the gold fact carried the single highest
                # semantic score of all 113 seeds (0.716) but has no entity
                # link, so it scored 0.231 and ranked #39, while 15 unrelated
                # bike-training facts (semantic 0.58-0.69) each took +0.50 and
                # filled the entire top-15 the reader ever sees. The boost is
                # worth ~3x the entire spread of semantic scores, so ungated it
                # does not tie-break, it overrides.
                entity_key = g.get("entity_key") or ""
                canonical = entity_key.split(":", 1)[-1].casefold()
                entity_matches_query = bool(canonical) and any(
                    token in query_terms for token in re.findall(r"[a-z0-9]+", canonical) if len(token) > 2
                )
                if entity_fact_count > 0 and entity_matches_query:
                    fact.entity_boost = min(boost_cap / max(entity_fact_count, 1), boost_cap)
                else:
                    fact.entity_boost = 0.0

            # Reciprocal Rank Fusion, not a raw-score sum. The previous formula
            # added semantic_score, keyword_score, structural_score, and
            # entity_boost directly -- four differently-scaled signals (a
            # cosine-derived value, a BM25-derived value, a hop-count-derived
            # value, a capped constant) where whichever one happened to read
            # "big" for a given fact dominated the total regardless of how
            # relevant that fact actually was. That is the same failure shape
            # already fixed once for entity_boost specifically (query-gating);
            # RRF fixes it structurally for all four signals at once by fusing
            # on each fact's RANK POSITION within each channel instead of the
            # raw score value, which is scale-invariant by construction --
            # standard k=60 (Cormack et al., "Reciprocal Rank Fusion
            # Outperforms Condorcet and Individual Rank Learning Methods",
            # 2009). structural_score/entity_boost are not literal separate
            # retrieval result lists, so their "rank" is derived by sorting the
            # candidate set by that score descending; a fact with zero score in
            # a channel gets no rank and no term from it, the same "absent
            # means silent, not last-place" rule semantic/keyword already use.
            structural_rank_by_id = {
                f.fact_id: position
                for position, f in enumerate(
                    sorted((f for f in facts.values() if f.structural_score > 0), key=lambda f: -f.structural_score),
                    start=1,
                )
            }
            entity_rank_by_id = {
                f.fact_id: position
                for position, f in enumerate(
                    sorted((f for f in facts.values() if f.entity_boost > 0), key=lambda f: -f.entity_boost),
                    start=1,
                )
            }

            def _rrf_term(rank: int | None) -> float:
                return 1.0 / (rrf_k + rank) if rank is not None else 0.0

            for fact in facts.values():
                fact.composite_score = (
                    _rrf_term(fact.semantic_rank)
                    + _rrf_term(fact.keyword_rank)
                    + _rrf_term(structural_rank_by_id.get(fact.fact_id))
                    + _rrf_term(entity_rank_by_id.get(fact.fact_id))
                )

            # Sort facts
            ranked = sorted(facts.values(), key=lambda f: f.composite_score, reverse=True)

            # Abstention check. Was semantic-only: a fact found purely by BM25
            # keyword match (e.g. an exact name/term the embedding missed) with
            # zero structural support could trigger abstention even with a
            # strong keyword_score, because that score was never looked at.
            # Confirmed reachable: the query rewriter is on the same LLM call
            # path that's failed live in this session (credentials, timeouts),
            # and its failure zeroes every keyword_score for the whole request
            # (see _semantic_and_keyword_seeding) -- but that's a rewriter
            # failure feeding in a real zero, not this check's own bug; abstain
            # only when semantic, keyword, AND structural are all weak.
            threshold = self._config.retrieval_abstention_semantic_threshold
            if not ranked or (
                ranked[0].semantic_score < threshold
                and ranked[0].keyword_score < threshold
                and ranked[0].structural_score == 0
            ):
                ctx["abstention_triggered"] = True
                logger.info(
                    "Retrieval: Abstention triggered (semantic/keyword scores below %s, zero structural score)",
                    threshold,
                )
                return self._config.retrieval_abstention_message

            top_facts = ranked[:top_k]
            excluded = ranked[top_k:]
            ctx["top_fact_score"] = top_facts[0].composite_score if top_facts else 0.0
            ctx["candidates_considered"] = len(ranked)
            ctx["candidates_excluded"] = len(excluded)
            # Every traced retrieval miss so far has had the same shape: the
            # needed fact was seeded and scored, just not high enough to make
            # this cutoff -- crowded out by a fact that matched the question on
            # surface wording (e.g. any dollar amount for a "how much" question)
            # without matching its actual topic. Without this line, finding that
            # required a one-off diagnostic script re-running the whole pipeline
            # by hand each time; now it's one log line per query.
            if top_facts and excluded:
                logger.info(
                    "Retrieval: reader window cutoff (top_k=%d) — last included "
                    "(score=%.3f): %r | first excluded (score=%.3f): %r",
                    top_k, top_facts[-1].composite_score, (top_facts[-1].text or "")[:80],
                    excluded[0].composite_score, (excluded[0].text or "")[:80],
                )

            # Format context
            context_blocks = []
            for fact in top_facts:
                date_str = ""
                if fact.observed_at:
                    try:
                        date_str = datetime.fromtimestamp(fact.observed_at, tz=timezone.utc).strftime("%Y-%m-%d")
                    except Exception:
                        date_str = "Recent"
                else:
                    date_str = "Recent"
                speaker_str = fact.speaker or "user"
                context_blocks.append(f"[{date_str} | {speaker_str}]: {fact.text}")

            # Neighboring-turn expansion (ADR-005): pull in every other fact
            # extracted from the same source turn as a fact that already
            # earned its place in top_facts. No date/speaker available cheaply
            # here (would need another join), so these are appended under a
            # clearly labeled heading rather than interleaved as if they had
            # been independently ranked -- the reader should treat them as
            # "also said in the same breath as the fact above", not as
            # equally-scored top hits.
            if context_id is not None and self._config.retrieval_sibling_fact_limit > 0:
                try:
                    siblings = self._sibling_facts(context_id, [f.fact_id for f in top_facts], question)
                except Exception as e:
                    logger.debug("Sibling-fact expansion skipped: %s", e)
                    siblings = {}
                if siblings:
                    ctx["sibling_facts_added"] = len(siblings)
                    context_blocks.append("[related facts from the same conversation turns]:")
                    context_blocks.extend(f"- {text}" for text in siblings.values())

            context_str = "\n".join(context_blocks)

            prompt = self._config.reader_system_prompt_template.format(context=context_str)
            with timed_operation(logger, "retrieval.phase3.reader_synthesis"):
                return self._llm.text_completion(
                    prompt, question, temperature=self._config.reader_temperature,
                    max_tokens=self._config.reader_max_tokens, timeout=self._config.reader_timeout_seconds,
                )
