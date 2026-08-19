# Repository Guide

## Project Goal

Build a generic context-memory system on top of the checked-in HydraDB graph
layer has two planned components:

- an ingestion engine that turns timestamped chat sessions into a provenance-
  preserving temporal graph;
- a retrieval engine that combines semantic fact seeding, HydraDB traversal,
  temporal filtering, evidence ranking, and answer/abstention generation.

PostgreSQL/pgvector is the approved MVP persistence layer for immutable chunks,
embeddings, model versions, and ingestion state. HydraDB remains the graph and
traversal layer.

LongMemEval is the first acceptance benchmark and a source adapter. It must flow
through the same generic ingestion contracts and orchestration as other context
sources; it must not define core domain models or storage APIs.

Milestones 0–8 are implemented; M6's local HTTP adapter is live-verified
against a real `graph-node` (built via `hydradb/Dockerfile`, no native
toolchain needed — ADR-032). M4 is a
deterministic extraction baseline only. M5 and M7 have real provider-neutral
adapters (`LLMClient`-backed entity/temporal models on Fireworks
`deepseek-v4-flash`, ADR-029; `sentence-transformers` embedder) behind their
existing ports — both unit-tested against injected fakes and live-verified
against the real provider/model (opt-in tests, gated on a runtime-environment
credential, never run by default). M8's orchestrator (ADR-031) does whole-chunk
idempotent replay, not per-stage resumption, and does **not** yet emit
`SUPERSEDES` graph edges — that needs a `predicate_key` on extracted
candidates that the deterministic baseline doesn't produce (ADR-030); a real,
open, documented gap, not an oversight. Retrieval remains gated future work.
See [docs/architecture_plan.md](docs/architecture_plan.md) for the current
single-entry-point description of both parts.

## Repository Map

- `docs/architecture_plan.md`: single entry point for the accepted ingestion
  and retrieval design; start here.
- `docs/implemented_agentic_memory_guide.md`: beginner-friendly explanation of
  the implemented ingestion pipeline, graph, bitemporal model, embeddings, and
  current limits.
- `hydradb/`: upstream Rust graph database and its authoritative implementation
  documentation.
- `hydradb/architecture.md`: implemented storage, consistency, indexing,
  routing, and query architecture.
- `hydradb/cypher-compat.md`: authoritative accepted OpenCypher subset.
- `hydradb/DEVELOPMENT.md`: build and verification recipes.
- `docs/graph_schema_proposal.md`: initial ingestion data model and temporal
  graph proposal.
- `docs/retrieval_architecture.md`: initial ingestion/retrieval pseudocode and
  hybrid retrieval proposal.
- `docs/ingestion_pipeline_roadmap.md`: gated, step-by-step ingestion build plan
  and milestone status.
- `docs/decisions.md`: append-only architectural decision, rationale, tradeoff,
  hold-on, and revisit history.
- `docs/ingestion_contract_v1.md`: accepted generic input, derived-memory, and
  recovery-state contract.
- `docs/ingestion_structure.md`: ingestion-only package routing and upstream
  authority boundary.
- `docs/extraction_baseline.md`: executable M4 fixture matrix, baseline limits,
  and the required gate for any model-backed extractor.
- `src/context_memory/core/`: dependency-free contracts, validation, graph-plan
  types, errors, and temporal/entity decision primitives.
- `src/context_memory/ingestion/`: active generic ingest and graph-write
  orchestration; `sources/` owns source-specific mapping; `model_adapters.py`
  and `embedding.py` hold the real (non-fake) LLM/embedding port
  implementations from ADR-027; `semantic_blocking.py` holds the
  non-authoritative entity-name cache; `graph_plan_builder.py` maps accepted
  candidates to a `GraphWritePlan` (ADR-030, no `SUPERSEDES` yet);
  `orchestrator.py` is the M8 job-state-machine runner (ADR-031).
- `src/context_memory/client/`: local HydraDB public HTTP client boundary.
- `src/tests/`: deterministic contract and domain validation tests.
- `db/migrations/`: forward-only checksum-verified PostgreSQL schema changes.
- `compose.yaml`: local PostgreSQL 16 + pgvector (always-on) and local
  `graph-node` (profile-gated: `--profile graph`, ADR-032). Docker daemon required.

## Architectural Boundary

Treat HydraDB as the graph storage and traversal engine. Treat PostgreSQL as the
canonical raw-evidence, embedding, and ingestion-state store. Keep fact
extraction, entity resolution, temporal interpretation, retrieval ranking,
answer generation, abstention, benchmark adaptation, and model-provider code in
the application layer unless an upstream-quality HydraDB change is explicitly
approved.

HydraDB owns durable graph records, query execution, snapshots, writer
coordination, and traversal indexes. The application owns LLM extraction,
entity resolution, temporal interpretation, graph-plan construction, and raw
evidence. Use the checked-in HTTP/OpenCypher surface; hosted APIs are not part
of this runtime.

## Memory Model

Memory type and memory scope are separate dimensions:

- `episodic`: timestamped source experience with immutable chunk provenance;
- `semantic`: consolidated fact, preference, entity property, or relationship;
- `procedural`: instruction, workflow, rule, or learned skill;
- `chat`: current working scope, possibly ephemeral;
- `session`: one conversation episode;
- `user`: durable cross-session scope and primary MVP retrieval boundary;
- `organization`: optional shared scope, deferred until authorization and
  promotion policy exist.

Every source turn is episodic. Semantic or procedural memories derived from it
must retain source chunk id, hash, span offsets, speaker, session, and time.
Promotion between scopes is explicit application policy. Never infer or promote
organization memory automatically.

Keep knowledge time distinct from world-validity time. `observed_at` and
`superseded_at` define when an assertion was known/preferred by the system;
`valid_from` and `valid_to` define when it was true in the described world. A
correction closes knowledge time but must not automatically rewrite world
validity.

## Storage Responsibilities

PostgreSQL owns:

- immutable raw chunks, chunk boundaries, metadata, and content hashes;
- versioned fact/chunk embeddings through pgvector;
- ingestion jobs/outbox, attempts, errors, and completion state;
- optional PostgreSQL full-text indexes when sparse retrieval is introduced.

HydraDB owns:

- `Session`, `Turn`, `Fact`, `Entity`, and `Alias` graph records;
- provenance references to SQL chunks and source spans;
- bounded graph traversal and graph-side retrieval filters.

PostgreSQL and HydraDB do not share a transaction. Cross-store writes require
stable idempotency keys, explicit job states, graph verification, retries, and
safe replay. Never describe two separate client requests as atomic because they
run in one Python process.

## Current Proposed Data Flow

Ingestion:

1. Parse source payload through source-specific adapter.
2. Emit versioned generic context batch/record envelope using `context_id` as
   isolation key.
3. Persist immutable chunks, hashes, and ingestion job in PostgreSQL.
4. Extract atomic facts, memory type, source spans, and entities using validated
   structured output.
5. Assign deterministic scope defaults and resolve entities within active
   context.
6. Write validated graph batches to local HydraDB through Bolt.
7. Preserve updates with explicit temporal links; retain old evidence.
8. Persist versioned embeddings in PostgreSQL/pgvector.
9. Verify both stores, then mark ingestion job complete.

Retrieval:

1. Embed question and run exact pgvector search for fact IDs within active
   context and model version.
2. Expand through HydraDB relationships and update chains.
3. Apply question-time temporal constraints.
4. Rank semantic and structural evidence.
5. Expand context progressively: exact source span, neighboring sentence/turn
   when needed, then full chunk only when narrower evidence remains ambiguous.
6. Answer or abstain using runtime evidence signals only.

These choices remain draft. Do not copy pseudocode directly into production.
First resolve graph-wide id allocation, SQL chunk boundaries, fact-only versus
fact-plus-chunk embeddings, partial-failure recovery, and benchmark-calibrated
retrieval/context/abstention metrics. Use context-qualified `Entity.selector_key`
values for native path selection; pairwise `algo.MSpaths` must receive identical
qualified `sourceValues` and `targetValues`.

## Generic Ingestion and Benchmark Boundary

Core ingestion accepts only approved generic context models. Source-specific
adapters own parsing and mapping. Core packages must not import LongMemEval
schemas, branch on benchmark field names, or persist evaluation labels.

LongMemEval receives a dedicated adapter and approved script/endpoint that maps
sessions, turns, roles, content, and timestamps into generic contract, then
calls normal ingestion service. `question_id`, `_abs`, `has_answer`, expected
answers, and answer-session labels stay inside benchmark/evaluation code.

The local benchmark uses timezone-free minute timestamps in
`%Y/%m/%d (%a) %H:%M`; source arrays are not reliably chronological and some
session IDs repeat. The adapter stable-sorts using an explicit benchmark-local
ordering basis, qualifies canonical session IDs with source index, preserves raw
time/session values in metadata, and records skipped empty-turn counts.

## Benchmark Contract

Use `longmemeval_s_cleaned.json` as the initial acceptance dataset. Preserve
`question_id` and emit one JSONL record containing `question_id` and
`hypothesis` per question. Ingest historical sessions separately and use
`question_date` as query time for replay. Never use `has_answer` during
retrieval; it is evaluation-only evidence metadata.

Measure end-to-end answer quality and retrieval quality separately. Include
category breakdowns, evidence Recall@K/ranking metrics, temporal/update
accuracy, and abstention precision/recall. Oracle retrieval is diagnostic, not
the headline result.

## Working Rules

- Before a major milestone or architecture choice, explain functionality,
  alternatives, tradeoffs, risks, tests, and documentation impact; wait for
  explicit approval.
- Before implementing every ingestion-roadmap milestone, prepare the approval
  packet required by `docs/ingestion_pipeline_roadmap.md`, then stop and wait for
  explicit user approval. Approval is limited to the named milestone and exact
  decisions presented. Never infer approval from silence or general agreement.
- Record material architecture decisions in `docs/decisions.md` before their
  implementation. Use `Proposed`, `Accepted`, `Rejected`, `Deferred`, or
  `Superseded`; only explicit user approval can produce `Accepted`.
- Treat accepted decision entries as append-only history. When direction
  changes, add a superseding decision instead of rewriting earlier rationale.
- Record alternatives, rationale, tradeoffs, hold-ons, consequences,
  verification evidence, and revisit triggers. Held/deferred work must not enter
  implementation incidentally.
- Keep core ingestion source-neutral. New formats require adapters into generic
  context contract, not source-specific branches in domain, persistence,
  extraction, graph, embedding, or recovery layers.
- Treat design documents as living documents. Update affected architecture,
  code-flow, decisions, and documentation registers with approved behavior.
- Update roadmap milestone status and review all affected living documents in
  the same change set. Do not declare implementation or a milestone complete
  while roadmap, decisions, schema, retrieval flow, or agent guidance is stale.
- Preserve source text and provenance. Never silently rewrite historical
  evidence.
- Start with exact pgvector search. Add HNSW/IVFFlat only after a measured
  latency need and an exact-search recall comparison.
- Never use `_abs`, `has_answer`, expected answers, or question labels to make a
  runtime abstention decision.
- Keep provider adapters separate from domain logic. Prefer deterministic fakes
  for unit and CI tests; do not make paid model calls in CI.
- Never present deterministic extraction fixtures as model-quality evidence.
  Persist their `deterministic_fixture` / `baseline_only` identity, preserve
  invalid outputs with reasons, and require an approved ADR plus quality plan
  before enabling a model-backed extractor.
- Use LLMs only behind provider-neutral ports. First reduce entity candidates
  deterministically within `context_id`; accept only a returned candidate ID in
  that bounded set. Run temporal LLM judgment only after same-subject,
  same-predicate, and chronology gates. Invalid/abstaining output must remain
  unresolved, never create a cross-context link or forced supersession.
- Graph writes use local HydraDB HTTP with runtime URL/token. Register immutable
  graph payload hashes in PostgreSQL before writes; use global registry IDs and
  legal `UNWIND` statements. HTTP `query_id` is not a durable mutation key;
  local manifests plus deterministic `MERGE` identities make replay safe. Do
  not claim whole-plan atomicity before M8.
- Do not commit secrets, model keys, bearer tokens, generated benchmark data,
  or local object-store contents.
- Preserve unrelated user changes. Check `git status` and both staged and
  unstaged diffs before editing a dirty file.

## Verification

Run HydraDB commands from `hydradb/`:

```bash
just fmt-check
just check
just test
```

For OpenCypher changes, use `just test-opencypher`. For public Bolt/HTTP work,
use `just test-client-protocols`. Run `just ci` only for milestone-level or
pre-merge verification; it is the full matrix and can take tens of minutes.

Native OpenCypher builds require `libcypher-parser` and SuiteSparse GraphBLAS.
Keep `RUST_MIN_STACK=33554432` for OpenCypher test/runtime query futures; the
checked-in `justfile` exports it.

Application-layer tests should cover deterministic id generation, chunk
immutability and hash verification, structured extraction validation, memory
type/scope enums, context isolation, adapter-to-contract equivalence, absence of
source-specific imports/branches in core ingestion, idempotent cross-store
replay, partial failure recovery, supersession and temporal boundaries,
embedding model versioning, progressive context expansion, low-evidence
abstention without label leakage, and source-neutral plus LongMemEval golden
fixtures before full benchmark runs.

Run current application tests with:

```bash
PYTHONPATH=src python3.12 -m unittest discover -s src/tests -v
```

PostgreSQL integration needs Docker Desktop running, then:

```bash
docker compose up -d postgres
CONTEXT_MEMORY_TEST_DATABASE_URL=postgresql://context_memory@127.0.0.1:54329/context_memory PYTHONPATH=src .venv/bin/python -m unittest discover -s src/tests -v
```

HydraDB live verification is locally token-gated. No native toolchain needed —
build `graph-node` from the checked-in Dockerfile and run it per
`docs/architecture_plan.md` §7 (ADR-032). Supply `CONTEXT_MEMORY_HYDRADB_URL`,
`CONTEXT_MEMORY_HYDRADB_TOKEN`, and optional `CONTEXT_MEMORY_HYDRADB_DATABASE`
only through runtime environment, then run:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.test_hydradb_live -v
```
