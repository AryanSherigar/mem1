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

    def structured_completion(self, system, user, schema):
        resp = self.structured_responses[self.structured_calls]
        self.structured_calls += 1
        return resp
        
    def text_completion(self, system, user):
        return self.text_response

class FakeEmbedder:
    def embed(self, text):
        return (0.1, 0.2, 0.3)

class FakeCursor:
    def __init__(self, results):
        self.results = results
        self.idx = 0
    def execute(self, query, params=None):
        pass
    def fetchall(self):
        res = self.results[self.idx]
        self.idx += 1
        return res
    def __enter__(self): return self
    def __exit__(self, *args): pass

class FakeConnection:
    def __init__(self, results):
        self.cursor_obj = FakeCursor(results)
    def cursor(self):
        return self.cursor_obj

class FakeHydra:
    def __init__(self, return_paths=True):
        self.return_paths = return_paths

    def read(self, cypher, params, bookmark):
        if "RETURN \n            f.logical_key" in cypher or "RETURN" in cypher and "valid_from" in cypher:
            # First cypher query fetching nodes
            return [{
                "fact_key": key,
                "valid_from": 0,
                "valid_to": 9999999999,
                "observed_at": 1000,
                "superseded_at": 9999999999,
                "entity_key": "entity-1"
            } for key in params.get("fact_keys", [])]
        elif "algo.MSpaths" in cypher:
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
        
        # Return a fact with very low score
        # 1st query: pgvector -> [('fact-1', 9.0)] -> semantic score = 1 / (1 + 9) = 0.1
        # 2nd query: BM25 -> empty
        # 3rd query: missing text -> [('fact-1', 'irrelevant')]
        conn = FakeConnection([[("fact-1", 9.0)], [], [("fact-1", "irrelevant")]])
        
        engine = HybridRetrievalEngine(llm, FakeEmbedder(), conn, FakeHydra(return_paths=False))
        
        ans = engine.retrieve_and_answer("ctx-1", "what is the meaning of life?", datetime.now(timezone.utc))
        self.assertEqual(ans, "I don't have that information in my memory.")

    def test_composite_scoring_and_synthesis(self):
        llm = FakeLLMClient([
            DateRange(valid_from=None, valid_to=None),
            QueryRewriterOutput(decomposed_queries=[], synonyms=[])
        ], text_response="The dog is in the park")
        
        # 1st query: pgvector -> [('fact-1', 0.1)] -> semantic score = 1 / 1.1 ≈ 0.9
        # 2nd query: BM25 -> []
        # 3rd query: missing text -> [('fact-1', 'dog in park')]
        conn = FakeConnection([[("fact-1", 0.1)], [], [("fact-1", "dog in park")]])
        
        engine = HybridRetrievalEngine(llm, FakeEmbedder(), conn, FakeHydra())
        
        ans = engine.retrieve_and_answer("ctx-1", "where is dog?", datetime.now(timezone.utc))
        self.assertEqual(ans, "The dog is in the park")

if __name__ == "__main__":
    unittest.main()
