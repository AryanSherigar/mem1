"""Phase 2: Hybrid Retrieval Engine implementing the 4-phase retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
            # Phase 0: Temporal Resolution & Query Rewriting
            resolver = TemporalQueryResolver(self._temporal_resolver_client, self._config)
            temporal_bounds = resolver.resolve(question, question_date)

            rewriter = QueryRewriter(self._query_rewriter_client, self._config)
            expanded_query = rewriter.rewrite(question)

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
            answer = self._composite_scoring_and_synthesis(question, seed_facts, graph_data, top_k)
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
                for row in cursor.fetchall():
                    fact_id = str(row[0])
                    distance = float(row[1]) if row[1] is not None else 0.0
                    semantic_score = 1.0 / (1.0 + distance)
                    if fact_id not in facts:
                        facts[fact_id] = ScoredFact(fact_id, "", semantic_score=semantic_score)
                    else:
                        facts[fact_id].semantic_score = max(facts[fact_id].semantic_score, semantic_score)

            # 2. Keyword Search (BM25)
            keywords = " | ".join(expanded_query.synonyms + expanded_query.decomposed_queries)
            if keywords:
                with self._pg.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT fact_id, raw_text, ts_rank_cd(text_tsvector, plainto_tsquery('english', %s)) AS rank
                        FROM fact_search_index
                        WHERE context_id = %s AND is_active = true
                        ORDER BY rank DESC
                        LIMIT %s
                        """,
                        (keywords, context_id, limit)
                    )
                    for row in cursor.fetchall():
                        fact_id = str(row[0])
                        raw_text = str(row[1])
                        rank = float(row[2]) if row[2] is not None else 0.0
                        keyword_score = rank / (1.0 + rank)
                        if fact_id not in facts:
                            facts[fact_id] = ScoredFact(fact_id, raw_text, keyword_score=keyword_score)
                        else:
                            facts[fact_id].keyword_score = max(facts[fact_id].keyword_score, keyword_score)
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
            node_cypher = """
            MATCH (f {id: $fid})
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
            raw_nodes = []
            entity_key_by_fact: dict[str, str] = {}
            for fact_key, graph_id in graph_id_by_fact_key.items():
                try:
                    rows = self._hydra.read(node_cypher, {"fid": graph_id}, None)
                except Exception as e:
                    logger.warning("HydraDB node query error for %s: %s", fact_key, e)
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

            # 2. Execute algo.MSpaths
            if entities:
                path_cypher = """
                CALL algo.MSpaths({
                    sourceLabel: 'Entity',
                    sourceProperty: 'logical_key',
                    sourceValues: $entities,
                    relTypes: ['ABOUT'],
                    maxLen: $max_hops
                }) YIELD path
                RETURN path
                """
                try:
                    path_res = self._hydra.read(
                        path_cypher, {"entities": list(entities), "max_hops": self._config.retrieval_graph_max_hops}, None
                    )
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

            # 3. Traverse SUPERSEDES. Per-fact, same reasoning as step 1 above.
            # Note: its result isn't consumed by scoring yet (`sup_res` was
            # already dead — computed, never read — before this fix; left as
            # the same no-op pending real use rather than silently invented).
            supersedes_cypher = """
            MATCH (f {id: $fid})-[:SUPERSEDES*1..5]->(old)
            RETURN count(old) AS superseded_count
            """
            for fact_key in valid_fact_keys:
                graph_id = graph_id_by_fact_key.get(fact_key)
                if graph_id is None:
                    continue
                try:
                    self._hydra.read(supersedes_cypher, {"fid": graph_id}, None)
                except Exception as e:
                    logger.debug("SUPERSEDES query skipped for %s: %s", fact_key, e)

            for f_id in seed_facts.keys():
                fact_key = f"fact:{f_id}"
                graph_data[f_id] = {
                    "hop_count": hop_count_by_fact.get(fact_key, 1),
                    "path_count": path_count_by_fact.get(fact_key, 0),
                    "entity_fact_count": 1  # baseline; real inverse-frequency weighting not yet wired, see AGENTS/decisions
                }

            return graph_data

    def _composite_scoring_and_synthesis(
        self, question: str, facts: dict[str, ScoredFact], graph_data: dict, top_k: int
    ) -> str:
        with timed_operation(logger, "retrieval.phase3.scoring_and_synthesis", {"facts_to_score": len(facts)}) as ctx:
            path_cap = self._config.retrieval_structural_path_cap
            boost_cap = self._config.retrieval_entity_boost_cap
            divisor = self._config.retrieval_composite_score_divisor

            for f_id, fact in facts.items():
                g = graph_data.get(f_id, {"hop_count": 1, "path_count": 0, "entity_fact_count": 0})
                hop_count = g.get("hop_count", 1) or 1
                path_count = g.get("path_count", 0)
                entity_fact_count = g.get("entity_fact_count", 0)

                fact.structural_score = (1.0 / hop_count) * min(path_count, path_cap) / path_cap
                fact.entity_boost = min(boost_cap / max(entity_fact_count, 1), boost_cap)

                fact.composite_score = (fact.semantic_score + fact.keyword_score + fact.structural_score + fact.entity_boost) / divisor

            # Sort facts
            ranked = sorted(facts.values(), key=lambda f: f.composite_score, reverse=True)

            # Abstention check
            threshold = self._config.retrieval_abstention_semantic_threshold
            if not ranked or (ranked[0].semantic_score < threshold and ranked[0].structural_score == 0):
                ctx["abstention_triggered"] = True
                logger.info("Retrieval: Abstention triggered (max semantic score < %s and zero structural score)", threshold)
                return self._config.retrieval_abstention_message

            top_facts = ranked[:top_k]
            ctx["top_fact_score"] = top_facts[0].composite_score if top_facts else 0.0

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

            context_str = "\n".join(context_blocks)

            prompt = self._config.reader_system_prompt_template.format(context=context_str)
            with timed_operation(logger, "retrieval.phase3.reader_synthesis"):
                return self._llm.text_completion(
                    prompt, question, temperature=self._config.reader_temperature,
                    max_tokens=self._config.reader_max_tokens, timeout=self._config.reader_timeout_seconds,
                )
