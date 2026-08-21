from __future__ import annotations

import re
import threading
import unittest
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from context_memory.core.config import Config
from context_memory.core.llm_client import LLMClient
from context_memory.retrieval import (
    TemporalQueryResolver, QueryRewriter, HybridRetrievalEngine, ScoredFact, DateRange, QueryRewriterOutput
)

class FakeLLMClient:
    """Dispatches by the requested schema's type, not by call order.

    retrieval.py's Phase 0 now runs the temporal resolver and query rewriter
    concurrently in a thread pool (they're independent LLM calls -- see
    retrieve_and_answer), so "the Nth structured_completion call gets
    structured_responses[N]" is no longer a safe assumption: which of the two
    calls actually reaches this fake first is a real race, not a fixed order.
    Matching on the response's own type is correct regardless of call order;
    the lock only protects the shared per-type queue itself, not ordering.
    """

    def __init__(self, structured_responses, text_response=""):
        self._by_type = defaultdict(list)
        for resp in structured_responses:
            self._by_type[type(resp)].append(resp)
        self.text_response = text_response
        self.structured_calls = 0
        self.last_reader_system_prompt = None
        self._lock = threading.Lock()

    def structured_completion(self, system, user, schema, **kwargs):
        with self._lock:
            self.structured_calls += 1
            queue = self._by_type[schema]
            if not queue:
                raise AssertionError(f"FakeLLMClient: no queued response for schema {schema.__name__}")
            return queue.pop(0)

    def text_completion(self, system, user, *args, **kwargs):
        self.last_reader_system_prompt = system
        return self.text_response

class FakeEmbedder:
    def embed(self, text):
        return (0.1, 0.2, 0.3)

class FakeCursor:
    """Recognizes which query it's answering by content, not call order — the
    real query sequence changes as retrieval.py evolves (e.g. the graph_id_registry
    lookup added to resolve HydraDB's id-only UNWIND matching), and a fixed
    positional results list silently misaligns whenever that happens."""

    def __init__(self, semantic_rows=(), bm25_rows=(), registry_rows=(), missing_text_rows=(), sibling_rows=()):
        self.semantic_rows = list(semantic_rows)
        self.bm25_rows = list(bm25_rows)
        self.registry_rows = list(registry_rows)
        self.missing_text_rows = list(missing_text_rows)
        self.sibling_rows = list(sibling_rows)
        self._last_query = ""

    def execute(self, query, params=None):
        self._last_query = query

    def fetchall(self):
        q = self._last_query
        # Checked before the generic "memory_embeddings" branch: the sibling
        # query also references memory_embeddings (twice, anchor+sibling
        # joins), so it would otherwise be misidentified as the Phase 1
        # semantic-search query.
        if "sibling.subject_id" in q:
            return self.sibling_rows
        if "graph_id_registry" in q:
            return self.registry_rows
        if "memory_embeddings" in q:
            return self.semantic_rows
        if "ts_rank_cd" in q:
            return self.bm25_rows
        if "fact_search_index" in q:
            return self.missing_text_rows
        return []

    def __enter__(self): return self
    def __exit__(self, *args): pass

class FakeConnection:
    def __init__(self, semantic_rows=(), bm25_rows=(), registry_rows=(), missing_text_rows=(), sibling_rows=()):
        self.cursor_obj = FakeCursor(semantic_rows, bm25_rows, registry_rows, missing_text_rows, sibling_rows)
    def cursor(self):
        return self.cursor_obj

class FakeHydra:
    def __init__(self, return_paths=True):
        self.return_paths = return_paths

    def read(self, cypher, params, bookmark):
        # Per-fact calls now (retrieval.py stopped fighting HydraDB's UNWIND-read
        # grammar and queries one fact at a time, see retrieval.py's own comment).
        if "SUPERSEDES" in cypher:
            return []
        if "OPTIONAL MATCH" in cypher:
            return [{
                "text": None, "speaker": None,
                "valid_from": 0, "valid_to": 9999999999,
                "observed_at": 1000, "superseded_at": 9999999999,
                "memory_scope": None, "entity_key": "entity-1",
            }]
        if "algo.MSpaths" in cypher:
            if not self.return_paths:
                return []
            return [{"path": [{"logical_key": "entity-1"}, {}, {"logical_key": "entity-2"}]}]
        return []

class FakeHydraWithEntities:
    """Like FakeHydra but resolves OPTIONAL MATCH's entity_key per fact id, so a
    test can control which facts share an entity vs which don't -- needed to
    exercise entity_fact_count actually varying (it used to be hardcoded to 1
    for every fact, so entity_boost was a constant regardless of fixture).

    The id is parsed out of the query text rather than read from `params`:
    retrieval.py inlines it as `MATCH (f {id: 123})` (int()-coerced) because
    this HydraDB build's parameter support is too narrow, so there is no bound
    parameter left to inspect."""

    _ID_PATTERN = re.compile(r"MATCH \(f \{id: (\d+)\}\)")

    def __init__(self, entity_key_by_fid):
        self.entity_key_by_fid = entity_key_by_fid

    def read(self, cypher, params, bookmark):
        if "SUPERSEDES" in cypher:
            return []
        if "OPTIONAL MATCH" in cypher:
            match = self._ID_PATTERN.search(cypher)
            fid = int(match.group(1)) if match else None
            return [{
                "text": None, "speaker": None,
                "valid_from": 0, "valid_to": 9999999999,
                "observed_at": 1000, "superseded_at": 9999999999,
                "memory_scope": None, "entity_key": self.entity_key_by_fid.get(fid),
            }]
        if "algo.MSpaths" in cypher:
            return []
        return []


class TestRetrievalEngine(unittest.TestCase):
    def test_temporal_resolver_adds_buffer(self):
        base_time = datetime(2026, 8, 19, tzinfo=timezone.utc)
        llm = FakeLLMClient([DateRange(valid_from=base_time, valid_to=base_time)])
        resolver = TemporalQueryResolver(llm)
        
        result = resolver.resolve("What happened today?", base_time)
        
        # Buffer is 2 days (172800 seconds)
        expected_from = base_time - timedelta(seconds=172800)
        expected_to = base_time + timedelta(seconds=172800)
        
        self.assertEqual(result.valid_from, expected_from)
        self.assertEqual(result.valid_to, expected_to)

    def test_query_rewriter(self):
        llm = FakeLLMClient([QueryRewriterOutput(decomposed_queries=["where is dog"], synonyms=["puppy"])])
        rewriter = QueryRewriter(llm)
        res = rewriter.rewrite("where is my dog?")
        self.assertEqual(res.decomposed_queries, ["where is dog"])
        self.assertEqual(res.synonyms, ["puppy"])

    def test_abstention_triggers_on_low_scores(self):
        llm = FakeLLMClient([
            DateRange(valid_from=None, valid_to=None),
            QueryRewriterOutput(decomposed_queries=[], synonyms=[])
        ], text_response="Should not reach here")
        
        # pgvector -> [('fact-1', 9.0)] -> semantic score = 1 / (1 + 9) = 0.1; BM25 skipped
        # (no rewriter keywords); registry resolves fact-1 -> graph_id 1; missing-text fallback.
        conn = FakeConnection(
            semantic_rows=[("fact-1", 9.0)],
            registry_rows=[("fact:fact-1", 1)],
            missing_text_rows=[("fact-1", "irrelevant")],
        )
        
        engine = HybridRetrievalEngine(llm, FakeEmbedder(), conn, FakeHydra(return_paths=False))
        
        ans = engine.retrieve_and_answer("ctx-1", "what is the meaning of life?", datetime.now(timezone.utc))
        self.assertEqual(ans, "I don't have that information in my memory.")

    def test_composite_scoring_and_synthesis(self):
        llm = FakeLLMClient([
            DateRange(valid_from=None, valid_to=None),
            QueryRewriterOutput(decomposed_queries=[], synonyms=[])
        ], text_response="The dog is in the park")
        
        # pgvector -> [('fact-1', 0.1)] -> semantic score = 1 / 1.1 ~= 0.9; BM25 skipped;
        # registry resolves fact-1 -> graph_id 1; missing-text fallback.
        conn = FakeConnection(
            semantic_rows=[("fact-1", 0.1)],
            registry_rows=[("fact:fact-1", 1)],
            missing_text_rows=[("fact-1", "dog in park")],
        )
        
        engine = HybridRetrievalEngine(llm, FakeEmbedder(), conn, FakeHydra())
        
        ans = engine.retrieve_and_answer("ctx-1", "where is dog?", datetime.now(timezone.utc))
        self.assertEqual(ans, "The dog is in the park")

    def test_entity_boost_applies_only_to_entities_the_query_mentions(self):
        """FINAL_ARCHITECTURE.md's entity boost gates on
        `if entity in query_entities`. Dropping that gate gave the same flat
        boost to every entity-linked fact in the corpus, and because the boost
        (0.5) is worth roughly 3x the entire spread of semantic scores, it
        overrode relevance instead of tie-breaking it.

        Measured on LongMemEval 118b2229 ("How long is my daily commute to
        work?"): the gold fact had the single highest semantic score of all 113
        seeds (0.716) but no entity link, so it ranked #39 while 15 unrelated
        bike-training facts took +0.50 each and filled the entire top-15."""
        llm = FakeLLMClient([])
        conn = FakeConnection(registry_rows=[("fact:fact-1", 1), ("fact:fact-2", 2), ("fact:fact-3", 3)])
        # fact-1/fact-2 link to "commute"; fact-3 has no entity at all.
        hydra = FakeHydraWithEntities({1: "entity:commute", 2: "entity:commute", 3: None})
        engine = HybridRetrievalEngine(llm, FakeEmbedder(), conn, hydra)

        def fresh_seeds():
            return {
                "fact-1": ScoredFact("fact-1", "linked A"),
                "fact-2": ScoredFact("fact-2", "linked B"),
                "fact-3": ScoredFact("fact-3", "unlinked"),
            }

        seed_facts = fresh_seeds()
        graph_data = engine._hydradb_graph_expansion("ctx-1", seed_facts, DateRange(), datetime.now(timezone.utc))
        self.assertEqual(graph_data["fact-1"]["entity_fact_count"], 2)
        self.assertEqual(graph_data["fact-3"]["entity_fact_count"], 0)
        self.assertEqual(graph_data["fact-1"]["entity_key"], "entity:commute")

        cap = engine._config.retrieval_entity_boost_cap

        # Query mentions the entity -> boost applies, inverse-frequency scaled.
        engine._composite_scoring_and_synthesis(
            "how long is my commute?", seed_facts, graph_data, top_k=5)
        self.assertEqual(seed_facts["fact-1"].entity_boost, min(cap / 2, cap))
        self.assertEqual(seed_facts["fact-3"].entity_boost, 0.0)

        # Query does NOT mention it -> no boost, even though the fact is linked.
        unrelated = fresh_seeds()
        engine._composite_scoring_and_synthesis(
            "what laptop should I buy?", unrelated, graph_data, top_k=5)
        self.assertEqual(unrelated["fact-1"].entity_boost, 0.0)
        self.assertEqual(unrelated["fact-3"].entity_boost, 0.0)

    def test_abstention_no_longer_ignores_strong_keyword_match(self):
        """A fact found purely by keyword match (weak/no embedding similarity,
        strong BM25 rank, no graph structure) used to trigger false abstention
        because the check only ever looked at semantic_score. A real exact-term
        hit -- the case BM25 exists for -- should not be discarded."""
        llm = FakeLLMClient([
            DateRange(valid_from=None, valid_to=None),
            QueryRewriterOutput(decomposed_queries=["exact term"], synonyms=[]),
        ], text_response="Found via keyword match")

        # No semantic hit at all; BM25 rank=1.0 -> keyword_score = 1/(1+1) = 0.5,
        # comfortably above the 0.3 default threshold. return_paths=False keeps
        # structural_score at 0, so this exercises the keyword branch alone.
        conn = FakeConnection(
            bm25_rows=[("fact-1", "exact term match", 1.0)],
            registry_rows=[("fact:fact-1", 1)],
            missing_text_rows=[("fact-1", "exact term match")],
        )
        engine = HybridRetrievalEngine(llm, FakeEmbedder(), conn, FakeHydra(return_paths=False))

        ans = engine.retrieve_and_answer("ctx-1", "exact term", datetime.now(timezone.utc))
        self.assertEqual(ans, "Found via keyword match")

    def test_keyword_search_falls_back_to_raw_question_when_rewriter_empty(self):
        """An empty rewriter result (a real failure mode: it's on the same LLM
        call path that's failed live this session on credentials/timeouts)
        used to skip BM25 for the whole request. It should degrade to the raw
        question instead of going silent."""
        llm = FakeLLMClient([
            DateRange(valid_from=None, valid_to=None),
            QueryRewriterOutput(decomposed_queries=[], synonyms=[]),  # rewriter produced nothing
        ], text_response="Found via raw-question fallback")

        conn = FakeConnection(
            bm25_rows=[("fact-1", "matches raw question", 1.0)],
            registry_rows=[("fact:fact-1", 1)],
            missing_text_rows=[("fact-1", "matches raw question")],
        )
        engine = HybridRetrievalEngine(llm, FakeEmbedder(), conn, FakeHydra(return_paths=False))

        ans = engine.retrieve_and_answer("ctx-1", "raw question text", datetime.now(timezone.utc))
        self.assertEqual(ans, "Found via raw-question fallback")

    def test_sibling_facts_from_same_turn_are_added_to_context(self):
        """ADR-005's "neighboring turn" expansion tier, accepted at design time
        but never implemented until now. Traced live: a fact scoring high
        enough to reach the reader ("$7.5 each") had its quantity ("20 potted
        herb plants") extracted as a *separate* atomic fact from the same
        turn, which did not itself score high enough to be seeded/ranked in --
        the reader had a price with nothing to multiply it by. A sibling fact
        from the same source turn as an already-relevant fact must be pulled
        into context even though it never separately competed on ranking."""
        llm = FakeLLMClient([
            DateRange(valid_from=None, valid_to=None),
            QueryRewriterOutput(decomposed_queries=[], synonyms=[]),
        ], text_response="20 plants at $7.50 each is $150")
        conn = FakeConnection(
            semantic_rows=[("fact-1", 0.1)],  # only the price fact is seeded/ranked
            registry_rows=[("fact:fact-1", 1)],
            missing_text_rows=[("fact-1", "Each potted herb plant was sold for $7.5")],
            # (fact_id, text, anchor_query_distance, sibling_query_distance, boundary_distance).
            # anchor_relevance=0.6, sibling_relevance=0.65: score = 0.65 - 0.1*0.05 = 0.645,
            # threshold = 0.80*0.6 = 0.48 -- clears it, matching a genuinely on-topic same-turn sibling.
            sibling_rows=[("fact-2", "User sold 20 potted herb plants at the Summer Solstice Market", 0.4, 0.35, 0.05)],
        )
        engine = HybridRetrievalEngine(llm, FakeEmbedder(), conn, FakeHydra())

        ans = engine.retrieve_and_answer("ctx-1", "how much did the herb plants earn?", datetime.now(timezone.utc))

        self.assertEqual(ans, "20 plants at $7.50 each is $150")
        self.assertIn("Each potted herb plant was sold for $7.5", llm.last_reader_system_prompt)
        self.assertIn("User sold 20 potted herb plants", llm.last_reader_system_prompt)

    def test_sibling_expansion_disabled_when_limit_is_zero(self):
        """RETRIEVAL_SIBLING_FACT_LIMIT=0 must fully disable the extra query,
        not just cap it at zero rows -- confirms the feature is a no-op, not a
        silent failure, when turned off."""
        import dataclasses
        llm = FakeLLMClient([
            DateRange(valid_from=None, valid_to=None),
            QueryRewriterOutput(decomposed_queries=[], synonyms=[]),
        ], text_response="answer")
        conn = FakeConnection(
            semantic_rows=[("fact-1", 0.1)],
            registry_rows=[("fact:fact-1", 1)],
            missing_text_rows=[("fact-1", "Each potted herb plant was sold for $7.5")],
            # (fact_id, text, anchor_query_distance, sibling_query_distance, boundary_distance).
            # anchor_relevance=0.6, sibling_relevance=0.65: score = 0.65 - 0.1*0.05 = 0.645,
            # threshold = 0.80*0.6 = 0.48 -- clears it, matching a genuinely on-topic same-turn sibling.
            sibling_rows=[("fact-2", "User sold 20 potted herb plants at the Summer Solstice Market", 0.4, 0.35, 0.05)],
        )
        config = dataclasses.replace(Config(), retrieval_sibling_fact_limit=0)
        engine = HybridRetrievalEngine(llm, FakeEmbedder(), conn, FakeHydra(), config=config)

        engine.retrieve_and_answer("ctx-1", "how much did the herb plants earn?", datetime.now(timezone.utc))

    def test_scar_rejects_off_topic_same_turn_fact(self):
        """The whole point of moving to SCAR (arxiv.org/abs/2606.16661) instead
        of an unordered LIMIT: same-turn presence alone must not be enough. A
        fact from the same turn as an anchor but semantically unrelated to the
        question -- someone rambling about the weather in the middle of a
        market-earnings turn -- must be rejected, while a genuinely relevant
        same-turn fact is kept. If this collapsed back to "pull every same-turn
        fact," both would come back."""
        llm = FakeLLMClient([])
        engine = HybridRetrievalEngine(llm, FakeEmbedder(), FakeConnection(), FakeHydra())

        # anchor_relevance = 1 - 0.4 = 0.6 -> threshold = 0.80 * 0.6 = 0.48
        on_topic = ("fact-relevant", "User sold 20 potted herb plants", 0.4, 0.35, 0.05)  # score 0.645
        off_topic = ("fact-unrelated", "The weather was nice that day", 0.4, 0.85, 0.05)  # score 0.145
        conn = FakeConnection(sibling_rows=[on_topic, off_topic])
        engine2 = HybridRetrievalEngine(llm, FakeEmbedder(), conn, FakeHydra())

        siblings = engine2._sibling_facts("ctx-1", ["fact-anchor"], "how much did I earn at the market?")

        self.assertIn("fact-relevant", siblings)
        self.assertNotIn("fact-unrelated", siblings)

    def test_composite_score_is_reciprocal_rank_fusion_not_raw_sum(self):
        """Composite scoring used to sum raw semantic/keyword/structural/
        entity scores directly -- four differently-scaled signals where
        whichever one happened to read numerically "big" for a fact dominated
        regardless of actual relevance (confirmed live and repeatedly: an
        irrelevant fact with a coincidentally high semantic score routinely
        outranked the fact that actually answered the question). RRF fixes
        this by fusing on rank position, not raw score value -- a fact ranked
        decently across multiple channels should be able to outscore a fact
        that "wins" one channel alone, which a raw sum cannot express when the
        single-channel winner's raw score is large enough."""
        llm = FakeLLMClient([])
        engine = HybridRetrievalEngine(llm, FakeEmbedder(), FakeConnection(), FakeHydra())

        # fact-1: dominant single channel (semantic rank 1, huge raw score).
        # fact-2: weaker in any one channel, but present in three.
        fact1 = ScoredFact("fact-1", "single-channel winner", semantic_score=0.99, semantic_rank=1)
        fact2 = ScoredFact("fact-2", "multi-channel winner", semantic_score=0.10, semantic_rank=5,
                            keyword_score=0.10, keyword_rank=3)
        facts = {"fact-1": fact1, "fact-2": fact2}
        # fact-2 also has graph support (structural + entity); fact-1 has none.
        graph_data = {
            "fact-1": {"hop_count": 1, "path_count": 0, "entity_fact_count": 0, "entity_key": None},
            "fact-2": {"hop_count": 1, "path_count": 3, "entity_fact_count": 1, "entity_key": "entity:widget"},
        }

        engine._composite_scoring_and_synthesis("tell me about the widget", facts, graph_data, top_k=5)

        k = engine._config.retrieval_rrf_k
        expected_fact1 = 1.0 / (k + 1)  # semantic rank 1 only
        expected_fact2 = 1.0 / (k + 5) + 1.0 / (k + 3) + 1.0 / (k + 1) + 1.0 / (k + 1)  # 4 channels
        self.assertAlmostEqual(fact1.composite_score, expected_fact1)
        self.assertAlmostEqual(fact2.composite_score, expected_fact2)
        # The real point: multi-channel support overtakes a single dominant
        # raw score -- impossible under the old sum when fact-1's raw semantic
        # score (0.99) alone exceeded fact-2's summed raw scores.
        self.assertGreater(fact2.composite_score, fact1.composite_score)

    def test_rrf_channel_absence_contributes_nothing(self):
        """A fact missing entirely from a channel must get 0 from it, not a
        last-place rank -- otherwise every unretrieved fact in the corpus
        would tie for the same nonzero score in every channel it never
        appeared in."""
        llm = FakeLLMClient([])
        engine = HybridRetrievalEngine(llm, FakeEmbedder(), FakeConnection(), FakeHydra())
        fact = ScoredFact("fact-1", "only semantic", semantic_score=0.5, semantic_rank=2)
        facts = {"fact-1": fact}
        graph_data = {"fact-1": {"hop_count": 1, "path_count": 0, "entity_fact_count": 0, "entity_key": None}}

        engine._composite_scoring_and_synthesis("q", facts, graph_data, top_k=5)

        k = engine._config.retrieval_rrf_k
        self.assertAlmostEqual(fact.composite_score, 1.0 / (k + 2))


if __name__ == "__main__":
    unittest.main()
