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
from concurrent.futures import ThreadPoolExecutor, as_completed
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


class PrefetchingExtractor:
    """Wraps a real `Extractor` and runs `.extract()` for many records concurrently
    ahead of `orchestrator.run_batch()`'s serial loop, then serves results from a
    warm cache — so the loop's per-turn cost drops from a full network round trip
    to a dict lookup.

    Why this is safe when parallelizing the batch loop itself is not: every store
    in this pipeline (`chunk_store`, `job_store`, `manifest_store`, ...) shares one
    `psycopg` connection, which is not safe for concurrent access from multiple
    threads. Extraction has no such shared state — it's a pure LLM network call,
    and the OpenAI SDK client is thread-safe for concurrent requests. This class
    only moves *that* call earlier; `orchestrator.run_batch()` still runs exactly
    as sequentially as before, and never touches Postgres/HydraDB concurrently.
    """

    def __init__(self, inner: Any, max_workers: int = 8, progress_every: int = 25) -> None:
        self._inner = inner
        self._max_workers = max_workers
        self._progress_every = progress_every
        self._cache: dict[str, Any] = {}
        # ExtractionService gates on these two attributes (see extraction.py's
        # `extract()`) — proxy them through so the wrapper is transparent to it.
        self.extractor_name = getattr(inner, "extractor_name", None)
        self.extractor_version = getattr(inner, "extractor_version", None)

    def prefetch(self, records: Sequence[Any]) -> None:
        """Runs `.extract()` for every record concurrently and warms the cache.
        Call this once per instance's records before `orchestrator.run_batch()`.

        Emits periodic progress with a live throughput-based ETA. This phase can
        dominate an instance's wall clock — a provider token-per-minute ceiling
        (Groq's free tier is 8k TPM) throttles it far below what the per-call
        latency alone suggests, and the SDK absorbs the resulting 429s as silent
        backoff. Without progress output there is no way to tell a slow run from
        a wedged one, which is exactly the hole this closes.
        """
        if not records:
            return
        total = len(records)
        done = failed = 0
        started = time.perf_counter()
        print(f"    prefetching {total} extractions ({self._max_workers} workers)...", flush=True)

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(self._inner.extract, record): record.record_id for record in records}
            for future in as_completed(futures):
                record_id = futures[future]
                try:
                    self._cache[record_id] = future.result()
                except Exception as exc:  # keep going; run_chunk's own error handling covers the rest
                    failed += 1
                    print(f"  prefetch extraction failed for {record_id}: {exc}", file=sys.stderr, flush=True)
                    self._cache[record_id] = ()
                done += 1
                if done % self._progress_every == 0 or done == total:
                    elapsed = time.perf_counter() - started
                    rate = done / elapsed if elapsed > 0 else 0.0
                    eta = (total - done) / rate if rate > 0 else 0.0
                    print(
                        f"      prefetch {done}/{total} ({100 * done / total:.0f}%)"
                        f" {rate * 60:.1f} calls/min elapsed {elapsed / 60:.1f}m"
                        f" eta {eta / 60:.1f}m" + (f" failed={failed}" if failed else ""),
                        flush=True,
                    )

    def extract(self, record: Any) -> Any:
        if record.record_id in self._cache:
            return self._cache.pop(record.record_id)
        # Cache miss (shouldn't happen if prefetch() covered the batch) — fall
        # back to a direct synchronous call so correctness never depends on
        # prefetch coverage, only speed does.
        return self._inner.extract(record)


def _report_rate_limits(llm_client: LLMClient) -> None:
    """Prints the provider's advertised rate limits up front.

    A tokens-per-minute ceiling, not per-call latency, is what actually bounds
    throughput here (measured: Groq's free tier advertises 8k TPM, which caps
    extraction at roughly 8-9 calls/min no matter how many workers are used).
    The SDK swallows the resulting 429s as backoff, so without this the run just
    looks mysteriously slow. Best-effort only — never fails the run.
    """
    try:
        raw = llm_client.client.chat.completions.with_raw_response.create(
            model=llm_client.model,
            messages=[{"role": "user", "content": "ok"}],
            max_tokens=8,
            timeout=30,
            **({"reasoning_effort": llm_client.reasoning_effort} if llm_client.reasoning_effort else {}),
        )
        headers = raw.headers
        limits = {k: v for k, v in headers.items() if "ratelimit" in k.lower()}
        if not limits:
            return
        tpm = limits.get("x-ratelimit-limit-tokens")
        rpm = limits.get("x-ratelimit-limit-requests")
        print(f"Provider rate limits: {tpm} tokens/min, {rpm} requests/min")
        if tpm and tpm.isdigit():
            # ~860 tokens is a measured short-turn extraction; real turns cost more,
            # so this is an optimistic ceiling, deliberately labeled as such.
            print(
                f"  -> optimistic ceiling ~{int(tpm) // 860} extraction calls/min"
                f" (real turns are longer, so expect fewer)"
            )
    except Exception as exc:
        print(f"  (could not read provider rate limits: {type(exc).__name__})", file=sys.stderr)


def create_pipeline(
    pg_connection: Any,
    hydra_transport: Any,
    llm_client: LLMClient,
    embedder: Any,
    extraction_store: Any | None = None,
    extractor: Any | None = None,
    extraction_workers: int = 0,
    progress_every: int = 25,
) -> tuple[IngestionOrchestrator, HybridRetrievalEngine, Any]:
    """Wires standard production ingestion orchestrator and hybrid retrieval engine.

    Returns `(orchestrator, retrieval_engine, extractor)` — the extractor is
    returned so the caller can call `.prefetch(records)` on it before
    `orchestrator.run_batch()` when `extraction_workers > 0` (see
    `PrefetchingExtractor`); it's a no-op passthrough otherwise.
    """
    chunk_store = PostgresChunkStore(pg_connection)
    job_store = PostgresJobStore(pg_connection)
    manifest_store = PostgresGraphManifestStore(pg_connection)
    embedding_store = PostgresEmbeddingStore(pg_connection)
    search_index_store = PostgresSearchIndexStore(pg_connection)
    ext_store = extraction_store or PostgresExtractionStore(pg_connection)

    if extractor is None:
        from context_memory.ingestion.model_adapters import LLMExtractor
        extractor = LLMExtractor(llm_client)

    if extraction_workers > 0:
        extractor = PrefetchingExtractor(
            extractor, max_workers=extraction_workers, progress_every=progress_every
        )

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

    return orchestrator, retrieval_engine, extractor


def evaluate_instance(
    instance: Mapping[str, Any],
    orchestrator: IngestionOrchestrator,
    retrieval_engine: HybridRetrievalEngine,
    extractor: Any | None = None,
) -> dict[str, str]:
    """Ingest one benchmark instance and retrieve its hypothesis."""
    question_id = str(instance["question_id"])
    question = str(instance.get("question", ""))
    raw_date = instance.get("question_date")

    if raw_date:
        question_date = parse_longmemeval_timestamp(raw_date, "question_date")
    else:
        question_date = datetime.now(timezone.utc)

    # 1. Ingest historical context batch synchronously. When a PrefetchingExtractor
    # is in play, warm every turn's extraction concurrently first — run_batch's own
    # loop stays serial (it must: shared psycopg connection), but its per-turn LLM
    # round trip becomes a cache hit.
    batch = adapt_longmemeval_instance(instance, ingestion_id=f"run:{question_id}")
    if extractor is not None and hasattr(extractor, "prefetch"):
        t_prefetch = time.perf_counter()
        extractor.prefetch(batch.records)
        print(f"    prefetched {len(batch.records)} extractions in {time.perf_counter() - t_prefetch:.1f}s")

    t_ingest = time.perf_counter()
    orchestrator.run_batch(batch)
    ingest_s = time.perf_counter() - t_ingest

    # 2. Retrieve answer
    t_retrieve = time.perf_counter()
    hypothesis = retrieval_engine.retrieve_and_answer(
        context_id=batch.context_id,
        question=question,
        question_date=question_date,
    )
    retrieve_s = time.perf_counter() - t_retrieve

    turns = len(batch.records)
    print(
        f"    ingest {ingest_s:.1f}s over {turns} turns ({ingest_s / max(turns, 1):.2f}s/turn)"
        f" | retrieve {retrieve_s:.1f}s"
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
    extractor: Any | None = None,
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
                record = evaluate_instance(instance, orchestrator, retrieval_engine, extractor)
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
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Emit a prefetch progress line every N completed extractions.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help=(
            "Root log level. INFO surfaces the per-stage [START]/[DONE] latency lines "
            "from `core.logging.timed_operation` — without it (the default) every stage "
            "timing in the pipeline is silently suppressed and runs are unprofilable."
        ),
    )
    parser.add_argument(
        "--extraction-workers",
        type=int,
        default=8,
        help=(
            "Concurrent extraction LLM calls to prefetch per instance (0 disables, "
            "restoring fully serial behavior). Only extraction is parallelized — "
            "graph/Postgres writes stay serial because the pipeline shares one "
            "psycopg connection."
        ),
    )
    return parser.parse_args(args)


def main() -> int:
    args = parse_args()

    import logging as _logging
    from context_memory.core.logging import setup_logging
    setup_logging(level=getattr(_logging, args.log_level))

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
        from pathlib import Path
        from context_memory.persistence.migrations import apply_migrations
        migrations_dir = Path(__file__).resolve().parents[2] / "db" / "migrations"
        if migrations_dir.exists():
            apply_migrations(pg_conn, migrations_dir)

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

        orchestrator, retrieval_engine, active_extractor = create_pipeline(
            pg_connection=pg_conn,
            hydra_transport=hydra_client,
            llm_client=llm_client,
            embedder=embedder,
            extractor=extractor_impl,
            extraction_workers=args.extraction_workers,
            progress_every=args.progress_every,
        )
        if args.extraction_workers > 0:
            print(f"Extraction prefetch enabled: {args.extraction_workers} concurrent calls per instance")
        _report_rate_limits(llm_client)

        evaluate_dataset(
            instances=payload,
            orchestrator=orchestrator,
            retrieval_engine=retrieval_engine,
            output_path=args.output,
            limit=args.limit,
            offset=args.offset,
            extractor=active_extractor,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
