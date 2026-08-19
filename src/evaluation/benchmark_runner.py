"""LongMemEval Benchmark Runner.

Ingests LongMemEval historical sessions into the context memory system
and executes retrieval queries against the ingested context, writing
predictions in the official JSONL format: {"question_id": "...", "hypothesis": "..."}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from context_memory.client.hydradb_http import HydraHttpTransport
from context_memory.core.llm_client import LLMClient
from context_memory.ingestion.embedding import SentenceTransformerEmbedder
from context_memory.ingestion.extraction import ExtractionService
from context_memory.ingestion.graph_plan_builder import GraphPlanBuilder
from context_memory.ingestion.graph_writer import GraphWriter
from context_memory.ingestion.orchestrator import IngestionOrchestrator
from context_memory.ingestion.resolution import EntityRegistry
from context_memory.ingestion.sources.longmemeval import adapt_longmemeval_instance, parse_longmemeval_timestamp
from context_memory.persistence.postgres import (
    PostgresChunkStore,
    PostgresEmbeddingStore,
    PostgresExtractionStore,
    PostgresGraphManifestStore,
    PostgresJobStore,
    PostgresSearchIndexStore,
)
from context_memory.retrieval import HybridRetrievalEngine


def create_pipeline(
    pg_connection: Any,
    hydra_transport: Any,
    llm_client: LLMClient,
    embedder: Any,
    extraction_store: Any | None = None,
    extractor: Any | None = None,
) -> tuple[IngestionOrchestrator, HybridRetrievalEngine]:
    """Wires standard production ingestion orchestrator and hybrid retrieval engine."""
    chunk_store = PostgresChunkStore(pg_connection)
    job_store = PostgresJobStore(pg_connection)
    manifest_store = PostgresGraphManifestStore(pg_connection)
    embedding_store = PostgresEmbeddingStore(pg_connection)
    search_index_store = PostgresSearchIndexStore(pg_connection)
    ext_store = extraction_store or PostgresExtractionStore(pg_connection)

    if extractor is None:
        from context_memory.ingestion.model_adapters import LLMExtractor
        extractor = LLMExtractor(llm_client)

    extraction_service = ExtractionService(extractor, ext_store)
    entity_registry = EntityRegistry(allocator=chunk_store, model=None)
    plan_builder = GraphPlanBuilder(allocator=chunk_store)
    graph_writer = GraphWriter(manifest_store=manifest_store, transport=hydra_transport)

    from context_memory.ingestion.model_adapters import LLMTemporalUpdateModel
    from context_memory.ingestion.resolution import TemporalUpdateClassifier
    temporal_model = LLMTemporalUpdateModel(llm_client)
    update_classifier = TemporalUpdateClassifier(temporal_model)

    orchestrator = IngestionOrchestrator(
        chunk_store=chunk_store,
        job_store=job_store,
        extraction_service=extraction_service,
        graph_plan_builder=plan_builder,
        graph_writer=graph_writer,
        resolve_entity=entity_registry.resolve_entity,
        embedder=embedder,
        embedding_store=embedding_store,
        search_index_store=search_index_store,
        update_classifier=update_classifier,
    )

    retrieval_engine = HybridRetrievalEngine(
        llm_client=llm_client,
        embedder=embedder,
        pg_connection=pg_connection,
        hydra_client=hydra_transport,
    )

    return orchestrator, retrieval_engine


def evaluate_instance(
    instance: Mapping[str, Any],
    orchestrator: IngestionOrchestrator,
    retrieval_engine: HybridRetrievalEngine,
) -> dict[str, str]:
    """Ingest one benchmark instance and retrieve its hypothesis."""
    question_id = str(instance["question_id"])
    question = str(instance.get("question", ""))
    raw_date = instance.get("question_date")

    if raw_date:
        question_date = parse_longmemeval_timestamp(raw_date, "question_date")
    else:
        question_date = datetime.now(timezone.utc)

    # 1. Ingest historical context batch synchronously
    batch = adapt_longmemeval_instance(instance, ingestion_id=f"run:{question_id}")
    orchestrator.run_batch(batch)

    # 2. Retrieve answer
    hypothesis = retrieval_engine.retrieve_and_answer(
        context_id=batch.context_id,
        question=question,
        question_date=question_date,
    )

    return {
        "question_id": question_id,
        "hypothesis": hypothesis,
    }


def evaluate_dataset(
    instances: Sequence[Mapping[str, Any]],
    orchestrator: IngestionOrchestrator,
    retrieval_engine: HybridRetrievalEngine,
    output_path: Path,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict[str, str]]:
    """Evaluates a list of benchmark instances, streaming hypotheses to JSONL."""
    selected_instances = instances[offset:]
    if limit is not None:
        selected_instances = selected_instances[:limit]

    total = len(selected_instances)
    print(f"Starting benchmark run: {total} instances (offset={offset}, limit={limit})")
    print(f"Writing outputs to: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if offset > 0 and output_path.exists() else "w"

    results: list[dict[str, str]] = []
    start_time = time.perf_counter()

    with output_path.open(mode, encoding="utf-8") as out_file:
        for idx, instance in enumerate(selected_instances, start=1):
            q_id = instance.get("question_id", f"unknown-{idx}")
            t0 = time.perf_counter()
            try:
                record = evaluate_instance(instance, orchestrator, retrieval_engine)
                results.append(record)
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_file.flush()
                dur = time.perf_counter() - t0
                print(f"[{idx}/{total}] question_id={q_id} ({dur:.2f}s) -> hypothesis={record['hypothesis'][:60]!r}...")
            except Exception as exc:
                dur = time.perf_counter() - t0
                print(f"[{idx}/{total}] question_id={q_id} ({dur:.2f}s) ERROR: {exc}", file=sys.stderr)
                fallback_record = {"question_id": str(q_id), "hypothesis": "I do not have enough information to answer."}
                results.append(fallback_record)
                out_file.write(json.dumps(fallback_record, ensure_ascii=False) + "\n")
                out_file.flush()

    total_dur = time.perf_counter() - start_time
    avg_speed = total / total_dur if total_dur > 0 else 0.0
    print(f"Finished benchmark run: {len(results)} evaluated in {total_dur:.2f}s ({avg_speed:.2f} questions/s)")
    return results


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to local LongMemEval JSON dataset")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the output hypotheses JSONL")
    parser.add_argument("--limit", type=int, default=None, help="Maximum benchmark instances to evaluate")
    parser.add_argument("--offset", type=int, default=0, help="Starting index in the dataset (for resuming)")
    parser.add_argument(
        "--database-url",
        default=os.environ.get("CONTEXT_MEMORY_DATABASE_URL", "postgresql://context_memory@127.0.0.1:54329/context_memory"),
        help="PostgreSQL connection string",
    )
    parser.add_argument(
        "--hydradb-url",
        default=os.environ.get("CONTEXT_MEMORY_HYDRADB_URL", "http://127.0.0.1:8080"),
        help="HydraDB HTTP endpoint URL",
    )
    parser.add_argument(
        "--hydradb-token",
        default=os.environ.get("CONTEXT_MEMORY_HYDRADB_TOKEN", ""),
        help="Optional HydraDB bearer token",
    )
    parser.add_argument(
        "--hydradb-database",
        default=os.environ.get("CONTEXT_MEMORY_HYDRADB_DATABASE", "default"),
        help="HydraDB database name",
    )
    parser.add_argument(
        "--extractor",
        choices=["llm", "deterministic"],
        default="llm",
        help="Extractor implementation to use",
    )
    return parser.parse_args(args)


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"Error: input file {args.input} does not exist.", file=sys.stderr)
        return 1

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        print("Error: input dataset must contain a JSON array of instances.", file=sys.stderr)
        return 1

    import psycopg

    llm_base_url = os.environ.get("FIREWORKS_BASE_URL", os.environ.get("OPENAI_BASE_URL", "https://api.fireworks.ai/inference/v1"))
    llm_api_key = os.environ.get("FIREWORKS_API_KEY", os.environ.get("OPENAI_API_KEY", "fake-key"))
    llm_model = os.environ.get("EXTRACTOR_MODEL", "accounts/fireworks/models/deepseek-v4-flash-0731")

    print(f"Connecting to PostgreSQL at {args.database_url}...")
    with psycopg.connect(args.database_url, autocommit=True) as pg_conn:
        hydra_client = HydraHttpTransport(
            base_url=args.hydradb_url,
            bearer_token=args.hydradb_token or None,
            database=args.hydradb_database,
        )
        llm_client = LLMClient(base_url=llm_base_url, api_key=llm_api_key, model_name=llm_model)
        embedder = SentenceTransformerEmbedder()

        if args.extractor == "deterministic":
            from context_memory.ingestion.fakes import DeterministicExtractor
            extractor_impl = DeterministicExtractor()
        else:
            from context_memory.ingestion.model_adapters import LLMExtractor
            extractor_impl = LLMExtractor(llm_client)

        orchestrator, retrieval_engine = create_pipeline(
            pg_connection=pg_conn,
            hydra_transport=hydra_client,
            llm_client=llm_client,
            embedder=embedder,
            extractor=extractor_impl,
        )

        evaluate_dataset(
            instances=payload,
            orchestrator=orchestrator,
            retrieval_engine=retrieval_engine,
            output_path=args.output,
            limit=args.limit,
            offset=args.offset,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
