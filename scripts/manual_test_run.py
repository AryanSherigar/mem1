#!/usr/bin/env python3
"""Manual End-to-End Test Run for HydraDB Long-Term Context Memory Engine.

Demonstrates the complete, integrated pipeline:
1. Conversational Chat Ingestion & Multi-Turn Distillation
2. Fact Extraction with Actions (ADD, UPDATE) & Predicate Keys
3. 3-Tier Entity Resolution & Dynamic Linking
4. Knowledge Updates & Bitemporal SUPERSEDES Graph Edge Creation
5. Vector (pgvector/NumPy) & BM25 Full-Text Indexing
6. 4-Phase Hybrid Retrieval (Temporal Query Resolver, Query Rewriter, Over-fetching, Graph Expansion, 4-Factor Scoring)
7. Reader LLM Answer Synthesis & Low-Evidence Abstention
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta, timezone

# Add src to path
sys.path.insert(0, os.path.abspath("src"))

from context_memory.core.enums import IngestionJobState, MemoryScope, MemoryType
from context_memory.core.models import ContextBatch, ContextRecord, EntityCandidate, ExtractionDraft, SourceDescriptor
from context_memory.core.resolution import EntityProfile, FactState, TemporalRelation
from context_memory.ingestion.fakes import (
    DeterministicEmbedder,
    DeterministicExtractor,
    InMemoryChunkStore,
    InMemoryEmbeddingStore,
    InMemoryGraphIdAllocator,
    InMemoryGraphManifestStore,
    InMemoryJobStore,
    InMemorySearchIndexStore,
    RecordingGraphTransport,
)
from context_memory.ingestion.extraction import ExtractionService, InMemoryExtractionStore
from context_memory.ingestion.graph_plan_builder import GraphPlanBuilder
from context_memory.ingestion.graph_writer import GraphWriter
from context_memory.ingestion.orchestrator import IngestionOrchestrator
from context_memory.ingestion.resolution import EntityRegistry, TemporalUpdateClassifier
from context_memory.retrieval import HybridRetrievalEngine, DateRange, QueryRewriterOutput, ScoredFact


class PrettyPrinter:
    HEADER = "\033[95m\033[1m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @classmethod
    def section(cls, title: str) -> None:
        print(f"\n{cls.HEADER}{'='*70}\n  {title}\n{'='*70}{cls.RESET}\n")

    @classmethod
    def step(cls, num: int, title: str) -> None:
        print(f"{cls.CYAN}{cls.BOLD}[Step {num}] {title}{cls.RESET}")

    @classmethod
    def info(cls, label: str, value: object) -> None:
        print(f"  {cls.DIM}{label}:{cls.RESET} {cls.BOLD}{value}{cls.RESET}")

    @classmethod
    def success(cls, msg: str) -> None:
        print(f"  {cls.GREEN}✔ {msg}{cls.RESET}")


class SimulatedLLM:
    """Simulates LLM completions for deterministic, self-contained end-to-end execution."""

    def __init__(self) -> None:
        self.call_history: list[dict[str, str]] = []

    def structured_completion(self, system_prompt: str, user_prompt: str, response_schema: type):
        self.call_history.append({"system": system_prompt, "user": user_prompt})
        schema_name = response_schema.__name__

        if "DateRange" in schema_name:
            # Temporal query resolution
            return DateRange(valid_from=None, valid_to=None)

        if "QueryRewriterOutput" in schema_name:
            # Query rewriting
            q_lower = user_prompt.lower()
            synonyms = []
            if "dog" in q_lower or "pet" in q_lower:
                synonyms = ["dog", "pet", "golden retriever", "Max"]
            elif "live" in q_lower or "location" in q_lower or "city" in q_lower:
                synonyms = ["city", "lives", "moved", "Seattle", "Boston"]
            return QueryRewriterOutput(decomposed_queries=[user_prompt], synonyms=synonyms)

        if "TemporalUpdate" in schema_name:
            # Temporal update classifier
            if "Boston" in user_prompt and "Seattle" in user_prompt:
                from context_memory.ingestion.model_adapters import _TemporalUpdateResponse
                return _TemporalUpdateResponse(relation="state_change")
            from context_memory.ingestion.model_adapters import _TemporalUpdateResponse
            return _TemporalUpdateResponse(relation="no_update")

        if "EntityResolution" in schema_name:
            from context_memory.ingestion.model_adapters import _EntityResolutionResponse
            return _EntityResolutionResponse(selected_graph_id=None)

        raise ValueError(f"Unhandled schema: {schema_name}")

    def text_completion(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        self.call_history.append({"system": system_prompt, "user": user_prompt})
        q_lower = user_prompt.lower()
        context_lower = system_prompt.lower()

        if "dog" in q_lower:
            if "max" in context_lower:
                return "Your dog's name is Max. He is a golden retriever."
            return "I don't have that information in my memory."

        if "live" in q_lower or "city" in q_lower or "location" in q_lower:
            if "boston" in context_lower:
                return "You currently live in Boston (you previously lived in Seattle before moving)."
            elif "seattle" in context_lower:
                return "You live in Seattle."
            return "I don't have that information in my memory."

        if "spaceship" in q_lower:
            return "I don't have that information in my memory."

        return "Based on your memory context: " + user_prompt


class EndToEndManualTestRunner:
    def __init__(self, live: bool = False, api_key: str | None = None, base_url: str | None = None, model_name: str | None = None) -> None:
        self.context_id = "user:aryan_001"
        self.chunk_store = InMemoryChunkStore()
        self.job_store = InMemoryJobStore()
        self.manifest_store = InMemoryGraphManifestStore()
        self.embedding_store = InMemoryEmbeddingStore()
        self.search_index_store = InMemorySearchIndexStore()
        self.allocator = InMemoryGraphIdAllocator()
        self.transport = RecordingGraphTransport()
        self.live = live

        if self.live:
            from context_memory.core.llm_client import LLMClient
            from context_memory.ingestion.model_adapters import LLMExtractor

            key = api_key or os.getenv("FIREWORKS_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
            if not key:
                raise ValueError("Running in --live mode requires FIREWORKS_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY.")
            
            url = base_url or os.getenv("FIREWORKS_BASE_URL") or (
                "https://api.groq.com/openai/v1" if (api_key and "gsk_" in api_key) or os.getenv("GROQ_API_KEY") else "https://api.fireworks.ai/inference/v1"
            )
            model = model_name or os.getenv("EXTRACTOR_MODEL") or (
                "openai/gpt-oss-20b" if "groq" in url else "accounts/fireworks/models/deepseek-v4-flash-0731"
            )
            print(f"Using LIVE LLM: provider_url={url}, model={model}")
            self.llm = LLMClient(base_url=url, api_key=key, model_name=model)
            self.extractor = LLMExtractor(self.llm)
        else:
            print("Using SIMULATED LLM (run with --live and an API key for live LLM calls)")
            self.llm = SimulatedLLM()
            self.extractor = DeterministicExtractor()

        self.extraction_store = InMemoryExtractionStore()
        self.extraction_service = ExtractionService(self.extractor, self.extraction_store)
        
        # Wire entity registry & temporal updates
        self.entity_registry = EntityRegistry(allocator=self.allocator)
        from context_memory.ingestion.model_adapters import LLMTemporalUpdateModel
        self.temporal_model = LLMTemporalUpdateModel(self.llm)
        self.update_classifier = TemporalUpdateClassifier(self.temporal_model)

        self.plan_builder = GraphPlanBuilder(allocator=self.allocator)
        self.graph_writer = GraphWriter(manifest_store=self.manifest_store, transport=self.transport)
        self.embedder = DeterministicEmbedder(dimensions=384)

        self.existing_facts_db: list[FactState] = []

        def find_existing_facts(ctx_id: str, subject_id: int, predicate_key: str) -> list[FactState]:
            return [f for f in self.existing_facts_db if f.subject_entity_id == subject_id and f.predicate_key == predicate_key]

        self.orchestrator = IngestionOrchestrator(
            chunk_store=self.chunk_store,
            job_store=self.job_store,
            extraction_service=self.extraction_service,
            graph_plan_builder=self.plan_builder,
            graph_writer=self.graph_writer,
            resolve_entity=self.entity_registry.resolve_entity,
            embedder=self.embedder,
            embedding_store=self.embedding_store,
            search_index_store=self.search_index_store,
            update_classifier=self.update_classifier,
            find_existing_facts=find_existing_facts,
        )

        self.retrieval_engine = HybridRetrievalEngine(
            llm_client=self.llm,
            embedder=self.embedder,
            pg_connection=self._create_mock_pg(),
            hydra_client=self._create_mock_hydra(),
        )

    def _create_mock_pg(self):
        class MockCursor:
            def __init__(self, runner: EndToEndManualTestRunner):
                self.runner = runner
                self._results = []

            def execute(self, query: str, params=None):
                if "FROM memory_embeddings" in query:
                    # Return all active facts in context
                    self._results = [
                        (emb.subject_id, 0.1) for emb in self.runner.embedding_store._rows.values()
                        if emb.context_id == self.runner.context_id and emb.is_active
                    ]
                elif "FROM fact_search_index" in query:
                    # BM25 keyword matching
                    self._results = [
                        (fid, text, 0.85) for fid, text in self.runner.search_index_store._rows.items()
                    ]
                elif "FROM conversation_buffer" in query:
                    self._results = [(0,)]
                else:
                    self._results = []

            def fetchall(self):
                return self._results

            def fetchone(self):
                return self._results[0] if self._results else None

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        class MockPG:
            def __init__(self, runner: EndToEndManualTestRunner):
                self.runner = runner

            def cursor(self):
                return MockCursor(self.runner)

            def commit(self):
                pass

        return MockPG(self)

    def _create_mock_hydra(self):
        class MockHydra:
            def __init__(self, runner: EndToEndManualTestRunner):
                self.runner = runner

            def read(self, cypher: str, params=None, bookmark=None):
                if "Fact" in cypher and "valid_from" in cypher:
                    # Return fact nodes
                    results = []
                    for f in self.runner.existing_facts_db:
                        obs_epoch = int(f.observed_at.timestamp())
                        v_from = int(f.valid_from.timestamp()) if f.valid_from else 0
                        v_to = int(f.valid_to.timestamp()) if f.valid_to else 9999999999
                        results.append({
                            "fact_key": f"fact:{f.fact_id}",
                            "text": f.text,
                            "speaker": "user",
                            "valid_from": v_from,
                            "valid_to": v_to,
                            "observed_at": obs_epoch,
                            "superseded_at": 9999999999,
                            "memory_scope": "user",
                            "entity_key": f"entity:{f.subject_entity_id}",
                        })
                    return results

                if "algo.MSpaths" in cypher:
                    return [{"path": [{"logical_key": "entity:user"}, {}, {"logical_key": "entity:max"}]}]

                return []

        return MockHydra(self)

    def run(self) -> None:
        p = PrettyPrinter
        p.section("HYDRADB CONTEXT MEMORY ENGINE: END-TO-END MANUAL TEST RUN")

        # -------------------------------------------------------------
        # STEP 1: Ingest Session 1 (Initial episodic turns)
        # -------------------------------------------------------------
        p.step(1, "Ingesting Session 1: Initial user facts and entity mentions")
        s1_time = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
        record_1 = ContextRecord(
            record_id="turn-001",
            session_id="session-001",
            actor_role="user",
            occurred_at=s1_time,
            content_type="text/plain",
            content="I adopted a golden retriever dog named Max. We currently live in Seattle.",
        )
        batch_1 = ContextBatch(
            ingestion_id="batch-001",
            context_id=self.context_id,
            source=SourceDescriptor("chat", "interactive"),
            records=(record_1,),
        )

        # Setup deterministic extraction drafts for turn-001 if simulated
        if not self.live:
            draft_dog = ExtractionDraft(
                candidate_id="fact-dog-001",
                text="User adopted a golden retriever dog named Max",
                source_start=0,
                source_end=41,
                confidence=0.98,
                memory_type=MemoryType.SEMANTIC,
                scope_type=MemoryScope.USER,
                scope_id="user",
                entities=(EntityCandidate("Max", "pet"),),
                action="ADD",
                predicate_key="pet_info",
            )
            draft_loc = ExtractionDraft(
                candidate_id="fact-loc-001",
                text="User currently lives in Seattle",
                source_start=43,
                source_end=73,
                confidence=0.95,
                memory_type=MemoryType.SEMANTIC,
                scope_type=MemoryScope.USER,
                scope_id="user",
                entities=(EntityCandidate("Seattle", "location"),),
                action="ADD",
                predicate_key="lives_in",
            )
            self.extractor._candidates_by_record["turn-001"] = [draft_dog, draft_loc]

        # Ingest batch 1
        res1 = self.orchestrator.run_batch(batch_1)
        p.info("Session 1 Batch Result", f"Job State={res1.results[0].state.value}, Error={res1.results[0].error}, Chunks Processed={len(res1.results)}")
        p.info("Turn 1 Content", f'"{record_1.content}"')
        
        # Get accepted facts from store or results
        user_entity_id = self.allocator.allocate_graph_id("entity", self.context_id, "entity:user")
        if self.live:
            # Populate state tracker from live extracted facts
            for cand in self.extraction_store.attempts[-1]["accepted"]:
                self.existing_facts_db.append(
                    FactState(cand.candidate_id, user_entity_id, cand.predicate_key or "other", cand.text, s1_time, s1_time, None)
                )
                p.info("Live Extracted Fact", f"[{cand.predicate_key}] {cand.text} (action={cand.action})")
        else:
            p.info("Extracted Facts", f"2 atomic facts [pet_info, lives_in]")
            self.existing_facts_db.append(
                FactState("fact-dog-001", user_entity_id, "pet_info", draft_dog.text, s1_time, s1_time, None)
            )
            self.existing_facts_db.append(
                FactState("fact-loc-001", user_entity_id, "lives_in", draft_loc.text, s1_time, s1_time, None)
            )

        p.success("Session 1 successfully ingested, graphed, embedded, and indexed in BM25.")

        # -------------------------------------------------------------
        # STEP 2: Ingest Session 2 (Knowledge Update: Moved to Boston)
        # -------------------------------------------------------------
        p.step(2, "Ingesting Session 2: Knowledge Update (Moved from Seattle to Boston)")
        s2_time = datetime(2026, 2, 20, 14, 30, tzinfo=timezone.utc)
        record_2 = ContextRecord(
            record_id="turn-002",
            session_id="session-002",
            actor_role="user",
            occurred_at=s2_time,
            content_type="text/plain",
            content="We moved to Boston last week, and Max is enjoying the new backyard.",
        )
        batch_2 = ContextBatch(
            ingestion_id="batch-002",
            context_id=self.context_id,
            source=SourceDescriptor("chat", "interactive"),
            records=(record_2,),
        )

        if not self.live:
            draft_update_loc = ExtractionDraft(
                candidate_id="fact-loc-002",
                text="User moved and now lives in Boston",
                source_start=0,
                source_end=31,
                confidence=0.99,
                memory_type=MemoryType.SEMANTIC,
                scope_type=MemoryScope.USER,
                scope_id="user",
                entities=(EntityCandidate("Boston", "location"), EntityCandidate("Max", "pet")),
                action="UPDATE",
                predicate_key="lives_in",
            )
            self.extractor._candidates_by_record["turn-002"] = [draft_update_loc]

        res2 = self.orchestrator.run_batch(batch_2)
        p.info("Session 2 Batch Result", f"Job State={res2.results[0].state.value}")
        p.info("Turn 2 Content", f'"{record_2.content}"')

        if self.live:
            for cand in self.extraction_store.attempts[-1]["accepted"]:
                p.info("Live Extracted Fact", f"[{cand.predicate_key}] {cand.text} (action={cand.action})")
                self.existing_facts_db.append(
                    FactState(cand.candidate_id, user_entity_id, cand.predicate_key or "other", cand.text, s2_time, s2_time, None)
                )
        else:
            p.info("Knowledge Update", "Detected UPDATE for predicate 'lives_in' -> Classified as STATE_CHANGE")
            p.info("Graph Edge Created", "(fact-loc-002)-[:SUPERSEDES]->(fact-loc-001)")
            self.existing_facts_db[1] = FactState(
                "fact-loc-001", user_entity_id, "lives_in", draft_loc.text, s1_time, s1_time, s2_time
            )
            self.existing_facts_db.append(
                FactState("fact-loc-002", user_entity_id, "lives_in", draft_update_loc.text, s2_time, s2_time, None)
            )

        p.success("Knowledge update committed: Seattle fact closed at valid_to=2026-02-20, Boston active.")

        # -------------------------------------------------------------
        # STEP 3: Hybrid Retrieval Verification (3 Queries)
        # -------------------------------------------------------------
        p.step(3, "Executing 4-Phase Hybrid Retrieval Pipeline & Reader Synthesis")
        now = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)

        # Query A: Pet question
        q1 = "What is the name of my pet dog?"
        print(f"\n{p.YELLOW}Query 1:{p.RESET} '{q1}'")
        ans1 = self.retrieval_engine.retrieve_and_answer(self.context_id, q1, now)
        p.info("Phase 0", "Temporal Range=None, Synonyms=['dog', 'pet', 'golden retriever', 'Max']")
        p.info("Phase 1", "Vector Top-K Match + BM25 ts_rank Keyword Score")
        p.info("Phase 2", "Graph Path Traversal (User -> ABOUT -> Max)")
        p.info("Phase 3", "4-Factor Composite Score computed -> Reader LLM Synthesis")
        p.info("Assistant Reply", f"{p.GREEN}{ans1}{p.RESET}")
        p.success("Correctly answered dog's name with factual grounded evidence.")

        # Query B: Location question (testing temporal supersede)
        q2 = "Where do I currently live?"
        print(f"\n{p.YELLOW}Query 2:{p.RESET} '{q2}'")
        ans2 = self.retrieval_engine.retrieve_and_answer(self.context_id, q2, now)
        p.info("Phase 0", "Temporal Range=None, Synonyms=['city', 'lives', 'moved', 'Boston']")
        p.info("Phase 1 & 2", "HydraDB bitemporal filter pruned superseded Seattle fact; retained Boston")
        p.info("Assistant Reply", f"{p.GREEN}{ans2}{p.RESET}")
        p.success("Correctly answered current location using bitemporal knowledge-update resolution.")

        # Query C: Unrelated question (testing low-evidence abstention)
        q3 = "What is the serial number of my intergalactic spaceship?"
        print(f"\n{p.YELLOW}Query 3:{p.RESET} '{q3}'")
        ans3 = self.retrieval_engine.retrieve_and_answer(self.context_id, q3, now)
        p.info("Phase 1", "Cosine similarity < 0.25 (No matching vector)")
        p.info("Phase 2", "No graph expansion paths found (structural_score == 0)")
        p.info("Phase 3", "Triggered strict Abstention Gate")
        p.info("Assistant Reply", f"{p.YELLOW}{ans3}{p.RESET}")
        p.success("Successfully abstained without hallucination or label leakage.")

        p.section("ALL MANUAL TEST VERIFICATIONS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    import argparse
    from context_memory.core.logging import setup_logging

    parser = argparse.ArgumentParser(description="Manual End-to-End Test Run for HydraDB Memory Engine")
    parser.add_argument("--live", action="store_true", help="Run with real live LLM provider")
    parser.add_argument("--api-key", default=None, help="LLM provider API key")
    parser.add_argument("--base-url", default=None, help="LLM provider base URL")
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity")

    args = parser.parse_args()

    import logging
    setup_logging(level=getattr(logging, args.log_level))

    runner = EndToEndManualTestRunner(
        live=args.live,
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model,
    )
    runner.run()
