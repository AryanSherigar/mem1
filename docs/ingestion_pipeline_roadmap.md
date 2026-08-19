# Ingestion Pipeline Roadmap

| Field | Value |
|---|---|
| Status | Living |
| Scope | MVP ingestion engine only |
| Last verified | 2026-08-17; Milestones 0–6 deterministic tests and local Docker HTTP smoke verified |
| Next review | Milestone 7 approval packet |
| Decision log | [decisions.md](decisions.md) |
| Accepted contract | [ingestion_contract_v1.md](ingestion_contract_v1.md) |
| Related architecture | [graph_schema_proposal.md](graph_schema_proposal.md), [retrieval_architecture.md](retrieval_architecture.md) |

## 1. Purpose

Build a restart-safe, source-neutral ingestion engine that converts generic context records into:

- immutable chunks, hashes, versioned embeddings, and ingestion state in PostgreSQL/pgvector;
- provenance-preserving episodic, semantic, and procedural graph records in HydraDB;
- bitemporal knowledge/update history that can be reconstructed at a caller-supplied query time;
- verifiable links between every derived fact and its original evidence.

LongMemEval is an acceptance adapter, not the ingestion domain model. Its sessions and turns must pass through the same canonical contracts, validators, stores, extraction, graph writing, embedding, and recovery flow used by other context sources.

Milestone 0 is approved and documented below. Every later milestone still
requires its own approval packet and explicit approval before implementation.

## 2. Mandatory Approval Gate

Before implementing **each milestone**, present a milestone approval packet containing:

1. current repository state and prerequisite evidence;
2. functionality and end-to-end data flow;
3. exact decisions requested from the user;
4. alternatives, recommendation, tradeoffs, and hold-ons;
5. files, schemas, dependencies, and interfaces proposed for change;
6. failure modes, privacy/security impact, and rollback path;
7. deterministic tests, acceptance commands, and expected evidence;
8. documentation impact, including roadmap and decision-log updates.

Then stop. Wait for explicit user approval. Approval for one milestone does not approve later milestones or unresolved design options. Do not create implementation files, migrations, dependencies, infrastructure, or model calls while a milestone is awaiting approval.

After implementation, provide verification evidence and update affected living documents before requesting approval for the next milestone.

## 3. MVP Ingestion Flow

```text
source-specific payload
    -> source adapter
    -> canonical ContextBatch / ContextRecord envelope
    -> validate source-neutral contract
    -> deterministically identify immutable chunk
    -> persist chunk + pending ingestion job in PostgreSQL
    -> extract structured candidate memories
    -> validate memory type, scope, source spans, and time fields
    -> resolve entities and classify update relationships
    -> write idempotent graph batches to local HydraDB
    -> persist versioned embeddings in PostgreSQL/pgvector
    -> verify SQL <-> graph provenance and counts
    -> mark ingestion job completed
```

Any failure leaves a replayable non-terminal job. Source evidence remains immutable.

## 4. Step-by-Step Milestones

### Milestone 0 — Freeze MVP Contracts

**Approval:** Approved 2026-08-16. Contract artifacts are ready for user verification.

**Goal:** Convert current proposals into a concrete, reviewable contract before code.

Proposed work:

- choose a small deterministic golden fixture covering ordinary facts, preferences, procedures, corrections, real-world changes, relative time, aliases, and duplicate text;
- define a source-neutral `ContextBatch`/`ContextRecord` envelope and versioning policy;
- define generic service boundary independently from HTTP, CLI, or benchmark adapters;
- decide initial generic ingestion surface: application service only or public HTTP endpoint;
- decide LongMemEval entrypoint: dedicated CLI/script (recommended for MVP) or benchmark-only endpoint;
- confirm application language/runtime and package layout;
- confirm SQL deployment assumption for local development and CI;
- decide chunk boundary for MVP: turn, multi-turn window, or another deterministic unit;
- decide embedding target: facts only or facts plus chunks;
- define collision-safe graph IDs and chunk IDs;
- define structured extraction, memory-type, scope, and temporal schemas;
- define ingestion job states and legal transitions.

Approval questions:

- chunk unit and minimum SQL columns;
- application language/framework;
- ID strategy and context isolation;
- fact-only versus fact-plus-chunk embedding experiment;
- initial job state machine;
- canonical generic context envelope and versioning;
- generic ingestion surface and LongMemEval adapter entrypoint.

Verification before exit:

- fixture reviewed by user;
- contracts illustrated with at least one complete input-to-record example;
- all accepted decisions recorded in `docs/decisions.md`;
- no unresolved decision silently encoded as final.

Exit criterion: explicit Milestone 0 approval.

### Milestone 1 — Application Skeleton and Deterministic Domain Models

**Goal:** Establish implementation surface without database or model behavior.

Proposed work:

- create application package, configuration boundary, and test layout;
- define typed domain models for source descriptors, context batches/records, sessions, turns, chunks, extracted memories, entities, source spans, temporal fields, embeddings, and ingestion jobs;
- keep source adapters outside core domain and orchestration modules;
- enforce enums for memory type and scope;
- add deterministic fake extractor, fake embedder, and fake storage ports;
- validate source offsets and content hashes.

Tests:

- valid and invalid model construction;
- enum rejection and organization-scope rejection;
- stable serialization;
- Unicode-safe source spans;
- deterministic hash generation;
- no gold evaluation fields in runtime models.

Exit criterion: deterministic unit suite passes; user verifies model shapes and boundaries.

### Milestone 2 — PostgreSQL Schema and Chunk Persistence

**Goal:** Make raw evidence and ingestion state durable.

Proposed work:

- add reviewed migrations for `evidence_chunks`, `memory_embeddings`, and `ingestion_jobs`;
- persist deterministic chunk IDs, raw text, hashes, observation time, metadata, and source coordinates;
- reject mutation when a known chunk ID arrives with different content;
- allow identical text in different turns without collapsing provenance;
- add repository methods and transaction boundaries;
- keep pgvector exact search available but do not add ANN indexes.

Tests:

- migration up/down or forward-only migration policy, as approved;
- idempotent insert/replay;
- same ID/different payload conflict;
- duplicate text/different source preservation;
- transaction rollback;
- model-version uniqueness;
- context/tenant filters present in every query.

Exit criterion: PostgreSQL round-trip and restart test passes; schema verified by user.

### Milestone 3 — Generic Input Boundary, LongMemEval Adapter, and Chunking

**Goal:** Prove generic sources and LongMemEval both enter one canonical ingestion path.

Proposed work:

- implement source-neutral ingestion service accepting approved versioned context envelopes;
- validate generic source, context, record, actor, content, occurrence-time, external-ID, idempotency, and metadata fields;
- implement LongMemEval adapter that parses `longmemeval_s_cleaned.json` and emits only canonical context records;
- validate LongMemEval parallel arrays; parse its custom minute timestamps, stable-sort unordered sessions, qualify repeated session IDs, and audit skipped empty turns inside adapter;
- expose approved LongMemEval script/endpoint that calls generic ingestion service rather than bypassing it;
- normalize only transport-level details while preserving exact raw content;
- compute deterministic chunk IDs and content hashes;
- create pending ingestion jobs in the same SQL transaction as chunks;
- exclude `has_answer`, expected answers, and `_abs` interpretation from runtime records.

Tests:

- malformed record rejection with useful error location;
- deterministic replay;
- timestamp and ordering boundaries;
- empty/large/Unicode turns;
- evaluation-only metadata isolation;
- generic fixture and mapped LongMemEval fixture produce equivalent canonical record shapes;
- core ingestion packages contain no imports or conditionals for LongMemEval field names.

Exit criterion: golden fixture produces user-reviewed chunks and job rows.

### Milestone 4 — Structured Extraction and Memory Classification

**Approval:** Approved 2026-08-16. Completed as deterministic baseline only; see [Extraction Baseline v1](extraction_baseline.md).

**Goal:** Produce validated candidate memories behind a provider-neutral adapter.

Proposed work:

- define prompt-independent extraction interface;
- implement deterministic fake first;
- validate atomic fact text, entities, memory type, requested scope, evidence offsets, confidence, and temporal hints;
- default raw turns to episodic and extracted claims to semantic;
- classify explicit rules/workflows as procedural candidates;
- prohibit automatic organization promotion;
- persist rejected extraction output and reason without corrupting source jobs.

Tests:

- schema-conformant and malformed outputs;
- unsupported enum handling;
- span text exactly matches immutable chunk;
- session/chat scope defaults; no automatic user promotion;
- procedural false-positive examples;
- deterministic fake end-to-end path;
- no paid model calls in CI.

Hold-on: selecting or enabling a real extraction model requires separate approval after fake-path verification.

Exit criterion: user approves extraction contract and reviewed examples.

### Milestone 5 — Entity Resolution and Temporal Update Classification

**Approval:** Approved 2026-08-16. Completed as a provider-neutral, deterministic/LLM-assisted decision layer; no graph writes.

**Goal:** Resolve graph identities and distinguish corrections from real-world changes.

Proposed work:

- exact canonical and alias matching first, then bounded LLM judgment for ambiguous supplied candidates;
- context-qualified `selector_key` generation;
- bounded candidate retrieval for ambiguous entities;
- classify `SUPERSEDES` candidates with LLM judgment only after same-subject/predicate and chronology gates;
- maintain `observed_at`/`superseded_at` knowledge intervals;
- change `valid_from`/`valid_to` only for supported world-validity updates;
- preserve unresolved contradictions instead of forcing supersession.

Tests:

- alias/coreference fixtures;
- same name in different contexts;
- correction versus state-change examples;
- out-of-order session ingestion;
- unresolved/invalid model output and bounded-candidate abstention;
- historical reconstruction at several question dates.

Hold-on: connecting a real model provider requires its own operational approval; M5 contains ports and deterministic fakes only.

Exit criterion: temporal examples reviewed and approved before graph writes are enabled.

### Milestone 6 — Local HydraDB Graph Write Adapter

**Approval:** Approved 2026-08-17. Local HTTP adapter implemented; local runtime smoke passed 2026-08-19 against a Docker-built `graph-node` (ADR-032).

**Goal:** Persist validated graph topology using the checked-in HydraDB server.

Proposed work:

- map validated application graph plans to legal `UNWIND`/`MERGE` batches;
- allocate stable graph IDs in PostgreSQL and register immutable payload hashes;
- write through the local HTTP query endpoint; retain deterministic `MERGE`
  identities because HTTP has no caller-owned durable idempotency field;
- retain SQL chunk IDs, hashes, source spans, memory type/scope, and temporal properties;
- verify read-after-write through a causal bookmark.

Tests:

- deterministic graph-plan/query/manifest tests;
- PostgreSQL manifest replay/conflict test;
- local HTTP write/read smoke, explicitly skipped unless local URL/token are supplied.

Exit criterion: deterministic suite passes; one isolated graph plan round-trips through local `graph-node`.

### Milestone 7 — Embedding Persistence

**Approval:** Approved 2026-08-19 (ADR-027). Implemented; local verification only (no provider credential required for this milestone's embedder).

**Goal:** Persist reproducible, versioned semantic seeds.

Proposed work:

- add provider-neutral embedder interface and deterministic fake;
- compute embeddings only after validated fact text exists;
- store model name, model version, dimensions, embedded-content hash, subject kind, and source chunk ID;
- preserve old embeddings after supersession while updating latest-state flags;
- implement exact scoped pgvector query and model-version filter;
- benchmark fact-only against fact-plus-chunk embeddings if Milestone 0 approves both.

Tests:

- model/dimension mismatch;
- re-embedding without source mutation;
- exact-search ordering;
- inactive/historical retrieval;
- context and model-version isolation;
- deterministic fake vector results.

Exit criterion: user reviews retrieval seeds and benchmark evidence.

### Milestone 8 — Cross-Store Orchestration and Recovery

**Approval:** Approved 2026-08-19 (ADR-030, ADR-031). Implemented as whole-chunk idempotent replay (not per-stage resumption — see ADR-031's hold-ons); `SUPERSEDES` graph edges deferred pending a `predicate_key` on extracted candidates (ADR-030, a real open gap, not an oversight).

**Goal:** Make the full pipeline restart-safe without claiming distributed atomicity.

Proposed work:

- implement approved ingestion job finite-state machine;
- define retryable versus terminal failures;
- resume from last verified state;
- verify SQL chunk, HydraDB graph, and embedding records before completion;
- add bounded retries, attempt history, error classification, and operator-visible diagnostics;
- support safe replay after process termination at every state boundary.

Tests:

- failure after chunk commit;
- failure during graph batches;
- failure after graph success but before embedding persistence;
- failure after embedding success but before completion marker;
- repeated restart and replay;
- payload conflict and manual-repair state.

Exit criterion: failure-injection matrix passes; user approves recovery semantics.

### Milestone 9 — End-to-End Ingestion Verification

**Goal:** Prove ingestion contract on a representative development slice.

Proposed work:

- ingest an approved source-neutral fixture through generic boundary;
- ingest a bounded LongMemEval development subset through dedicated adapter and the same generic boundary;
- verify source-to-fact provenance, memory-type distribution, entity isolation, temporal chains, SQL/graph reconciliation, and embedding coverage;
- publish deterministic run manifest with configuration/model versions;
- measure throughput and error rates without making performance claims from one run;
- document unresolved quality limitations.

Acceptance evidence:

- zero orphan graph facts or embeddings;
- every derived memory resolves to immutable source evidence and matching hash;
- replay produces no duplicate logical records;
- historical snapshots exclude future knowledge;
- all affected living documents marked updated or reviewed-no-change.

Exit criterion: explicit ingestion-MVP acceptance. Retrieval implementation remains separately gated.

## 5. Deferred Beyond Ingestion MVP

- organization memory and cross-user authorization;
- automatic procedural skill execution;
- approximate pgvector indexes;
- learned rerankers;
- production event streaming or CDC;
- distributed transaction coordinator;
- automatic source retention/deletion policy;
- additional source adapters beyond the generic contract and LongMemEval acceptance adapter.

Deferral means not designed or implemented without a new decision and approval packet.

## 6. Roadmap Status

| Milestone | Status | Approval |
|---|---|---|
| 0. Freeze MVP contracts | Completed — contract fixtures verified | Approved 2026-08-16 |
| 1. Domain models | Completed — deterministic suite verified | Approved 2026-08-16 |
| 2. PostgreSQL persistence | Completed — live PostgreSQL/pgvector integration verified | Approved 2026-08-16 |
| 3. Generic boundary/LongMemEval adapter/chunking | Completed — generic + live PostgreSQL tests and full-data dry run verified | Approved 2026-08-16 |
| 4. Extraction/classification | Completed — deterministic baseline, audit persistence, and live PostgreSQL test verified | Approved 2026-08-16 |
| 5. Entity/temporal resolution | Completed — bounded LLM decision ports, deterministic gates, and fixtures verified; real-provider hold-on closed (ADR-029, Fireworks deepseek-v4-flash, live-verified); no graph writes yet | Approved 2026-08-16 |
| 6. HydraDB adapter | Implemented and live-verified 2026-08-19 — local HTTP adapter, immutable graph manifest, deterministic tests, real write/read round trip against a Docker-built `graph-node` (ADR-032) | Approved 2026-08-17 |
| 7. Embeddings | Implemented — `sentence-transformers` embedder + `PostgresEmbeddingStore` against the existing `memory_embeddings` pgvector table; deterministic unit suite verified | Approved 2026-08-19 |
| 8. Recovery orchestration | Implemented — job state machine, whole-chunk idempotent replay, deterministic + real-Postgres suite verified; `SUPERSEDES` graph edges deferred (needs `predicate_key`, ADR-030) | Approved 2026-08-19 |
| 9. End-to-end verification | Partially unblocked: a real local `graph-node` is now built and live-verified (ADR-032), and `test_orchestrator_live.py` proves the full M8 pipeline against it. Still blocked on predicate extraction for full temporal-update/`SUPERSEDES` coverage (ADR-030) and a representative multi-record fixture run | Not approved |

M5's real-provider hold-on (see Milestone 5 above) is now closed by ADR-027/ADR-029: `LLMEntityResolutionModel`/`LLMTemporalUpdateModel` run against Fireworks `deepseek-v4-flash`, unit-tested with an injected fake client and live-verified against the real provider (`src/tests/test_model_adapters_live.py`, opt-in, gated on a runtime-environment credential).

Update this table after every approval, implementation, verification, rejection, or scope change.
