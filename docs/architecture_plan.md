# Architecture Plan

| Field | Value |
|---|---|
| Status | Living. Single entry point for the accepted design. |
| Supersedes | `ARCHITECTURE.md`, `AGENT.md`, and five other removed documents (ADR-028) |
| Decision log | [decisions.md](decisions.md) |
| Roadmap | [ingestion_pipeline_roadmap.md](ingestion_pipeline_roadmap.md) |
| Repository guide | [../AGENTS.md](../AGENTS.md) |

## 1. Shape of the system

Two parts, both built on the checked-in local HydraDB (`hydradb/src/bin/graph-node.rs`) — a self-hosted Bolt/HTTP graph engine, not the hosted `api.hydradb.com` product (see §6).

- **Ingestion** — turns generic, timestamped context records into immutable SQL evidence plus a provenance-preserving, bitemporal graph. Implemented through Milestone 7; this document's authority for what exists.
- **Retrieval** — combines PostgreSQL/pgvector semantic seeding, HydraDB graph traversal, temporal filtering, evidence ranking, and answer/abstention generation. Designed in [graph_schema_proposal.md](graph_schema_proposal.md) and [retrieval_architecture.md](retrieval_architecture.md); not yet implemented (gated, Milestone 9+).

Storage split (ADR-001, unchanged by anything in this document):

- **PostgreSQL** owns immutable raw chunks, content hashes, the graph-ID registry, versioned embeddings (pgvector), and ingestion job/outbox state.
- **HydraDB** (local `graph-node`) owns `Session`/`Turn`/`Fact`/`Entity`/`Alias` graph records, `SUPERSEDES`/`ABOUT`/`HAS_ALIAS`/`RELATES_TO` relationships, and traversal — it is the source of truth for graph topology and bitemporal fact state. PostgreSQL and HydraDB never share a transaction; cross-store writes use idempotency keys, explicit job states, and verification (ADR-007).

The system serves two consumers through the same generic contract: a benchmark runtime (LongMemEval, via a dedicated adapter, ADR-018) and — once retrieval and a chat surface are built — an interactive memory layer. Neither gets source-specific branches in core ingestion (ADR-011).

## 2. Ingestion: what is actually built today

Verified by reading the code, not the roadmap prose, 2026-08-19. `PYTHONPATH=src python3.12 -m unittest discover -s src/tests` (tests live under `src/tests/` after a repository restructure) — 115 tests, 0 failures, 1 skipped (the opt-in embedding-model-download smoke test) with Postgres, the local `graph-node` (ADR-032), and the LLM provider credential all present.

| Stage | Module | State |
|---|---|---|
| Contract validation | `core/models.py`, `core/validation.py` | `ContextBatch`/`ContextRecord`, RFC3339 time, evaluation-field rejection. Done. |
| LongMemEval adapter | `ingestion/sources/longmemeval.py` | Stable-sorted, minute-timestamp parsing, evaluation labels stripped. Done. |
| Chunk persistence | `ingestion/service.py`, `persistence/postgres.py::PostgresChunkStore` | One immutable chunk per record, content-hash conflict detection, `pending_graph` job row in the same transaction. Done. |
| Deterministic extraction | `ingestion/extraction.py` | Accepts only `deterministic-fixture v1`; audit trail persisted with rejections. Real extractor deliberately still gated (ADR-019) — this document does not change that. No subject/predicate/object decomposition — see the `SUPERSEDES` gap below. |
| Entity resolution | `core/resolution.py`, `ingestion/resolution.py::EntityRegistry` | Exact canonical/alias match → bounded LLM tier → new-entity allocation. Model tier: real adapter on Fireworks `deepseek-v4-flash`, live-verified (ADR-029). |
| Temporal update classification | `ingestion/resolution.py::TemporalUpdateClassifier` | Same-subject/predicate/chronology gates before any model call; `CORRECTION` vs `STATE_CHANGE` closes knowledge time (`observed_at`/`superseded_at`) independent of world validity (`valid_from`/`valid_to`). Real adapter, live-verified (ADR-029) — but see the `SUPERSEDES` gap below: nothing in the pipeline calls it yet. |
| Graph-plan construction | `ingestion/graph_plan_builder.py` (new) | Maps accepted candidates + resolved entities into `Session`/`Turn`/`Fact`/`Entity`/`Alias` nodes and `HAS_TURN`/`EXTRACTED_FROM`/`ABOUT`/`HAS_ALIAS` edges. This mapping did not exist before ADR-030 — `GraphWriter` only knew how to write an already-built plan. **Does not emit `SUPERSEDES`**: that needs a `predicate_key` on extracted candidates, which the deterministic baseline doesn't produce (ADR-030). A real, open gap: `TemporalUpdateClassifier` is built and live-verified but unreachable from the ingestion path until a future extraction milestone adds predicate structure. |
| Graph-ID allocation | `persistence/postgres.py::allocate_graph_id` | Stable non-negative integer IDs, `(node_kind, context_id, logical_key)` registry. Done — this is why HydraDB's UUID-ID documents were removed (ADR-028); this project never needed them. |
| Graph write | `ingestion/graph_writer.py`, `client/hydradb_http.py` | Batched `UNWIND`/`MERGE` against the local HTTP query endpoint; PostgreSQL manifest rejects changed replays before any write. Wired into the orchestrator (below). Live-verified (ADR-032) against a real `graph-node` built from `hydradb/Dockerfile` — no native toolchain needed. |
| Embeddings | `ingestion/embedding.py`, `persistence/postgres.py::PostgresEmbeddingStore` | Implemented (ADR-027), wired into the orchestrator. |
| Recovery/outbox | `ingestion/orchestrator.py::IngestionOrchestrator`, `persistence/postgres.py::PostgresJobStore`, `core/enums.py::is_legal_job_transition` | Implemented (ADR-031) as **whole-chunk idempotent replay**, not fine-grained per-stage resumption — retrying re-runs extraction through embeddings from the top, relying on every stage's existing idempotency. Real cost tradeoff (wasted calls on retry), not a correctness gap; see ADR-031. |

## 3. What this change adds: LLM client and embeddings (ADR-027)

`main` branch had unmerged work — `src/core/llm_client.py`, `src/core/config.py`, `src/core/id_generator.py`, `src/db/embedding_index.py` — built against a different (in-memory, UUID-keyed) design. Evaluated each on its own merits against the accepted ports, not wholesale:

- **`LLMClient`** (OpenAI-compatible, provider-neutral, structured-output support) — adopted as-is behind the existing `EntityResolutionModel` and `TemporalUpdateModel` ports (`ingestion/ports.py`). Those ports and their bounded-candidate guardrail (`EntityRegistry.resolve` rejects any model output outside the supplied candidate set) do not change; only a real implementation is added alongside the deterministic fakes.
- **`sentence-transformers/all-MiniLM-L6-v2`** — adopted behind the `Embedder` port. This is not new: `retrieval_architecture.md` §3 already specified this exact model. It needs no API key, so — unlike the LLM-backed ports — embedding generation is locally verifiable today.
- **`EntityNameIndex`** (in-process cosine similarity over entity names) — adopted only as a non-authoritative Tier-2 semantic-blocking cache feeding candidate IDs into `EntityRegistry.resolve`, rebuilt from PostgreSQL on demand. It is a candidate-recall aid; exact/alias matching and the bounded-LLM tier remain the resolution authority, matching the "do not introduce a memory-only index as authoritative" position this project already took when it removed the in-memory-only assistant-runtime documents.
- **`IdGenerator`** (SHA-256 → UUID) — **not** adopted. `logical_key` strings (e.g. `f"entity:{canonical}"`) plus the PostgreSQL integer registry already give deterministic, replay-safe identity, and it's what the local `graph-node` mutation surface actually requires. Porting a UUID generator alongside an integer-ID registry would be two identity systems doing one job.

No LLM provider credential exists in this environment (`.env` has only `HYDRA_DB_API_KEY`). The LLM-backed adapters are implemented and unit-tested against an injected fake client; live verification is held pending credentials, the same posture Milestone 6 held for its HydraDB token before it arrived.

## 4. Retrieval (designed, not yet implemented)

Full design lives in [retrieval_architecture.md](retrieval_architecture.md) and [graph_schema_proposal.md](graph_schema_proposal.md); not duplicated here. Summary of the accepted shape:

1. Embed the question; exact pgvector cosine search scoped to `context_id` and model version — no ANN index before a measured latency need (ADR-004).
2. Expand through HydraDB: connected entities, `algo.MSpaths` bridging, `SUPERSEDES` chains for update-style questions.
3. Apply knowledge-time (`observed_at`/`superseded_at`) and, separately, world-validity (`valid_from`/`valid_to`) filters for the question's time.
4. Rank with `0.6 semantic + 0.4 structural` (uncalibrated starting point).
5. Assemble context progressively — exact span → neighboring turn → full chunk — never full-chunk-by-default (ADR-005).
6. Answer or abstain on evidence signals only, never evaluation labels (ADR-008).

Building this is Milestone 9+ and needs its own approval packet per this repository's working rules; nothing in this change starts it.

## 5. Bitemporal model

Unchanged, and confirmed by the hosted-API investigation (§6) to be worth keeping entirely in this project's own stores rather than delegated anywhere:

- `observed_at` / `superseded_at` — when the system knew or stopped preferring an assertion.
- `valid_from` / `valid_to` — when the assertion was true in the described world.
- A correction closes knowledge time; it must not automatically rewrite world validity (ADR-006). `core/resolution.py::TemporalUpdateDecision` already keeps these separate.

## 6. Hosted HydraDB v2 — evaluated, held (ADR-026)

A hosted product also exists at `api.hydradb.com` — full chunking/embedding/BM25/LLM-graph-extraction/temporal-reasoning as a managed service, unrelated code to the local `graph-node` this project uses. A credentialed round trip confirmed it works, but its structured `temporal_facts` output only populates for its own LLM-extraction path (not for bounded/deterministic `graph_payload` submissions) and carries a single time axis, so it cannot host this project's bitemporal state. It is not part of the active runtime. A one-way, rebuildable projection from the local system of record into hosted v2 (search convenience only, never authoritative) was scoped as a plausible future pattern but requires a real retrieval-quality comparison against pgvector/Postgres full-text search, plus a data-handling decision, before it's worth building. Local-vs-hosted is a good candidate for a future benchmarked comparison once retrieval (§4) exists to compare against.

## 7. Running local HydraDB (ADR-032)

No Rust toolchain needed — `compose.yaml` builds it from the checked-in
Dockerfile. One-time setup, then start:

```bash
mkdir -p .hydradb
printf '%s\n' 'local-development-token-32-bytes' > .hydradb/auth-token
docker compose --profile graph up -d graph-node
```

Profile-gated on purpose: plain `docker compose up` (most work only needs
Postgres) never triggers this service's multi-stage Rust build. Graph state
lives in Docker-managed named volumes, not a host bind mount, and is mounted
at the two paths the image's non-root user actually has write access to
(`/tmp/graph`, `/var/cache/slatedb`) — both non-obvious, both the result of a
restart actually breaking during setup; see ADR-032's addendum if reproducing
this by hand elsewhere. `docker compose down -v` wipes it for a clean slate.

Then `CONTEXT_MEMORY_HYDRADB_URL=http://127.0.0.1:8443`,
`CONTEXT_MEMORY_HYDRADB_TOKEN=local-development-token-32-bytes` unlocks
`tests/test_hydradb_live.py` and `tests/test_orchestrator_live.py`.

## 8. Document map after cleanup (ADR-028)

Removed (superseded by this document and the ADRs above; recoverable from git history): `ARCHITECTURE.md`, `AGENT.md`, `docs/semantic_memory_distillation.md`, `docs/startup_hydration_id_generation.md`, `docs/llm_model_config.md`, `docs/entity_resolution_strategy.md`, `docs/temporal_query_resolver.md`.

Kept, still authoritative: `AGENTS.md` (repository guide/working rules), `docs/decisions.md`, `docs/ingestion_pipeline_roadmap.md`, `docs/ingestion_contract_v1.md`, `docs/ingestion_structure.md`, `docs/extraction_baseline.md`, `docs/graph_schema_proposal.md`, `docs/retrieval_architecture.md`, `docs/architecture_flow_comparison.md` (historical review artifact — the reasoning trail that produced ADR-026/027/028, not a live spec).
