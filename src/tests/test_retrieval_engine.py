from __future__ import annotations

import re
import unittest
from datetime import datetime, timedelta, timezone

from context_memory.core.llm_client import LLMClient
from context_memory.retrieval import (
    TemporalQueryResolver, QueryRewriter, HybridRetrievalEngine, ScoredFact, DateRange, QueryRewriterOutput
)

class FakeLLMClient:
    def __init__(self, structured_responses, text_response=""):
        self.structured_responses = structured_responses
        self.text_response = text_response
        self.structured_calls = 0

    def structured_completion(self, system, user, schema, **kwargs):
        resp = self.structured_responses[self.structured_calls]
        self.structured_calls += 1
        return resp

    def text_completion(self, system, user, *args, **kwargs):
        return self.text_response

class FakeEmbedder:
    def embed(self, text):
        return (0.1, 0.2, 0.3)

class FakeCursor:
    """Recognizes which query it's answering by content, not call order — the
    real query sequence changes as retrieval.py evolves (e.g. the graph_id_registry
    lookup added to resolve HydraDB's id-only UNWIND matching), and a fixed
    positional results list silently misaligns whenever that happens."""

    def __init__(self, semantic_rows=(), bm25_rows=(), registry_rows=(), missing_text_rows=()):
        self.semantic_rows = list(semantic_rows)
        self.bm25_rows = list(bm25_rows)
        self.registry_rows = list(registry_rows)
        self.missing_text_rows = list(missing_text_rows)
        self._last_query = ""

    def execute(self, query, params=None):
        self._last_query = query

    def fetchall(self):
        q = self._last_query
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
    def __init__(self, semantic_rows=(), bm25_rows=(), registry_rows=(), missing_text_rows=()):
        self.cursor_obj = FakeCursor(semantic_rows, bm25_rows, registry_rows, missing_text_rows)
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


if __name__ == "__main__":
    unittest.main()
