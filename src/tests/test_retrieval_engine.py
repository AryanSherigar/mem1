from __future__ import annotations

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

if __name__ == "__main__":
    unittest.main()
