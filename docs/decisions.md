# Architectural Decisions

| Field | Value |
|---|---|
| Status | Living, append-only decision history |
| Last verified | 2026-08-17 against local HTTP Docker write/read smoke and PostgreSQL manifest implementation |
| Next review | Milestone 7 approval packet |
| Roadmap | [ingestion_pipeline_roadmap.md](ingestion_pipeline_roadmap.md) |

## 1. Governance

Use one status per decision:

- `Proposed`: awaiting explicit user decision;
- `Accepted`: explicitly approved by the user;
- `Rejected`: considered and explicitly rejected;
- `Deferred`: intentionally held beyond current milestone/MVP;
- `Superseded`: replaced by a later decision that links back to it.

Never convert `Proposed` to `Accepted` based on inference, implementation convenience, or silence. Accepted entries are historical records: do not rewrite their original decision or rationale. Add a new decision and mark the old entry `Superseded` when direction changes.

Every material entry records why, tradeoffs, consequences, revisit triggers, and related evidence. Implementation must stop when it encounters an unapproved decision that changes architecture, dependencies, schemas, data ownership, runtime behavior, security, evaluation, or operations.

## 2. Decision Index

| ID | Decision | Status | Date |
|---|---|---|---|
| ADR-001 | PostgreSQL and HydraDB responsibility split | Accepted | 2026-08-16 |
| ADR-002 | Memory type and memory scope are orthogonal | Accepted | 2026-08-16 |
| ADR-003 | Preserve immutable chunks and source-linked derivations | Accepted | 2026-08-16 |
| ADR-004 | Exact pgvector search before approximate indexing | Accepted | 2026-08-16 |
| ADR-005 | Progressive evidence expansion | Accepted | 2026-08-16 |
| ADR-006 | Separate knowledge time from world-validity time | Accepted | 2026-08-16 |
| ADR-007 | Cross-store job/outbox and replay instead of claimed atomicity | Accepted | 2026-08-16 |
| ADR-008 | Runtime abstention cannot use evaluation labels | Accepted | 2026-08-16 |
| ADR-009 | Organization memory deferred | Deferred | 2026-08-16 |
| ADR-010 | Per-milestone approval and living-document gates | Accepted | 2026-08-16 |
| ADR-011 | Generic context ingestion core with source adapters | Accepted | 2026-08-16 |
| ADR-012 | Generic v1 contract and internal-service boundary | Accepted | 2026-08-16 |
| ADR-013 | Record chunks and fact-only embeddings | Accepted | 2026-08-16 |
| ADR-014 | SQL graph-ID registry and per-chunk recovery jobs | Accepted | 2026-08-16 |
| ADR-015 | Python/PostgreSQL local baseline and golden fixtures | Accepted | 2026-08-16 |
| ADR-016 | Dependency-free Python domain skeleton | Accepted | 2026-08-16 |
| ADR-017 | PostgreSQL migration and immutable persistence baseline | Accepted | 2026-08-16 |
| ADR-018 | Dataset-verified LongMemEval normalization | Accepted | 2026-08-16 |
| ADR-019 | Deterministic extraction baseline and quality gate | Accepted | 2026-08-16 |
| ADR-020 | Bounded LLM assistance for entity and temporal ambiguity | Accepted | 2026-08-16 |
| ADR-021 | Bolt graph writer and immutable graph manifest | Accepted | 2026-08-16 |

## 3. Accepted Decisions

### ADR-001 — PostgreSQL and HydraDB Responsibility Split

**Status:** Accepted
**Decision:** PostgreSQL/pgvector owns immutable chunks, embeddings, model versions, and ingestion state. HydraDB owns graph records, temporal/update relationships, and traversal.

**Why:** HydraDB supports scalar properties and graph traversal but not vector properties. SQL provides durable chunks, vector persistence, filtering, transactions within SQL, and recovery metadata.

**Tradeoffs:** Two stores improve fit but create cross-store consistency work, additional local infrastructure, reconciliation needs, and more failure states.

**Consequences:** Every cross-store record needs stable IDs, content hashes, idempotent writes, verification, and replay. Never describe the combined workflow as one transaction.

**Revisit when:** Operational burden outweighs graph value, HydraDB adds suitable vector/evidence storage, or a single-store benchmark proves equivalent temporal/graph behavior.

### ADR-002 — Memory Type and Scope Are Orthogonal

**Status:** Accepted
**Decision:** Memory type is `episodic`, `semantic`, or `procedural`. Scope is `chat`, `session`, `user`, or optional `organization`.

**Why:** Type describes meaning; scope describes ownership/lifetime. Combining them would make promotion, filtering, and authorization ambiguous.

**Tradeoffs:** Additional fields and validation. Some memories require judgment during classification.

**Consequences:** Raw turns are episodic. Derived semantic/procedural memories retain episode evidence. Scope promotion is explicit policy.

**Revisit when:** A benchmark or production case requires an additional type or non-hierarchical scope.

### ADR-003 — Immutable Chunks and Source-Linked Derivations

**Status:** Accepted
**Decision:** PostgreSQL keeps canonical raw chunks. Derived graph memories carry chunk ID, content hash, and supporting offsets.

**Why:** Preserves auditability and permits selective evidence retrieval without losing raw context.

**Tradeoffs:** Offset handling, hash verification, duplicate storage references, and chunk retention requirements.

**Consequences:** Re-extraction creates new derived/versioned records; it never rewrites original evidence. Identical text in different turns must preserve distinct provenance.

**Revisit when:** Non-text sources require byte/page/time coordinates rather than character offsets.

### ADR-004 — Exact pgvector Search First

**Status:** Accepted
**Decision:** Use exact cosine search with mandatory context and model-version filters. Do not add HNSW/IVFFlat initially.

**Why:** Current scale does not justify recall loss or ANN tuning. Exact results provide evaluation ground truth.

**Tradeoffs:** Exact search scales linearly and may become slower at larger volume.

**Consequences:** Any ANN proposal needs measured latency pressure and Recall@K comparison against exact search.

**Revisit when:** Exact-search latency breaches an approved target on representative data.

### ADR-005 — Progressive Evidence Expansion

**Status:** Accepted
**Decision:** Reader context expands from fact plus exact span, to neighboring sentence/turn, to full chunk only when narrower evidence is insufficient.

**Why:** Limits distractor tokens while retaining access to qualifiers and narrative context.

**Tradeoffs:** Requires source offsets, context-expansion logic, and sufficiency criteria. Selective context can omit implicit information; full chunks can introduce noise.

**Consequences:** Benchmark against an unconditional full-chunk baseline using answer quality, evidence recall, false answers, tokens, and latency.

**Revisit when:** Selective context degrades supported-answer accuracy or chunk boundaries become very small.

### ADR-006 — Bitemporal Knowledge and World Validity

**Status:** Accepted
**Decision:** `observed_at`/`superseded_at` represent when the system knew or preferred an assertion. `valid_from`/`valid_to` represent when the assertion was true in the described world.

**Why:** Historical replay and temporal questions fail when ingestion/knowledge time is conflated with real-world validity.

**Tradeoffs:** More properties, temporal classification, and edge cases around corrections versus state changes.

**Consequences:** Corrections close knowledge time without automatically rewriting world validity. All replay queries apply `question_date` to knowledge intervals.

**Revisit when:** Full event-sourcing or richer temporal intervals become necessary.

### ADR-007 — Cross-Store Job/Outbox and Replay

**Status:** Accepted
**Decision:** Coordinate PostgreSQL and HydraDB through durable ingestion jobs, idempotent operations, verification, and replay. Do not introduce a distributed transaction coordinator for MVP.

**Why:** Stores cannot share a transaction. Durable state and replay are simpler and testable at MVP scale.

**Tradeoffs:** Temporary inconsistency, recovery states, reconciliation code, and operator-visible errors.

**Consequences:** Exact job states and write order require Milestone 0 approval. Failure injection is mandatory before MVP acceptance.

**Revisit when:** Recovery latency or temporary inconsistency violates a concrete requirement.

### ADR-008 — Honest Runtime Abstention

**Status:** Accepted
**Decision:** Runtime retrieval/generation cannot inspect `_abs`, `has_answer`, expected answers, or question labels to decide abstention.

**Why:** Those fields are evaluation truth and would leak the answerability label.

**Tradeoffs:** Evidence thresholds require calibration and can produce false abstentions or false answers.

**Consequences:** Measure abstention precision/recall separately. Keep evaluation metadata outside application persistence.

**Revisit when:** Never, unless benchmark contract changes which fields are genuinely available to a production caller.

### ADR-010 — Approval and Living-Document Gates

**Status:** Accepted
**Decision:** Every major milestone and material architecture change requires an approval packet and explicit user approval before implementation. Roadmap, decision log, and affected design documents change with the implementation.

**Why:** Prevents silent architectural drift and keeps evidence, rationale, and implementation synchronized.

**Tradeoffs:** More checkpoints and slower starts; lower rework and clearer ownership.

**Consequences:** No milestone is complete with stale applicable documentation. Approval is milestone-scoped, not blanket authorization.

**Revisit when:** User explicitly changes governance requirements.

### ADR-011 — Generic Context Ingestion Core with Source Adapters

**Status:** Accepted
**Decision:** Core ingestion accepts a versioned, source-neutral context envelope. LongMemEval-specific fields and transformations live in a dedicated adapter that calls the same generic ingestion service used by every other source.

**Why:** Ingestion engine is a reusable context platform component, not a benchmark loader. Keeping benchmark structure outside core prevents `question_id`, haystack arrays, gold labels, and evaluation assumptions from shaping storage and orchestration APIs.

**Alternatives:** Build directly around LongMemEval and generalize later; create a separate benchmark ingestion path; expose only a benchmark-specific endpoint.

**Tradeoffs:** Generic envelope requires upfront contract/versioning decisions and adapter tests. It adds one mapping layer. In return, core stays reusable, source validation stays isolated, and LongMemEval exercises the production path rather than a shortcut.

**Consequences:** Canonical isolation key is `context_id`, not `haystack_id`. Core packages must not import LongMemEval schemas or branch on benchmark fields. LongMemEval adapter maps sessions, turns, and timestamps into canonical records while keeping gold/evaluation fields outside runtime persistence.

**Revisit when:** A source cannot be represented without lossy mapping; extend the versioned generic contract through a new approved ADR rather than adding source-specific core branches.

## 4. Deferred Decision

### ADR-009 — Organization Memory

**Status:** Deferred
**Decision:** Exclude organization-scoped memory from ingestion MVP.

**Why:** Generic MVP needs a reliable user/context boundary first. Organization sharing introduces tenancy, authorization, provenance, and promotion-policy risks without validating core ingestion behavior.

**Tradeoff:** MVP will not demonstrate shared enterprise memory.

**Revisit when:** User-memory MVP passes and a concrete organization use case defines owners, readers, writers, retention, and promotion approval.

## 5. Decision Tracker and Hold-Ons

| ID | Topic | Status | Options / tradeoff | Required before |
|---|---|---|---|---|
| OPEN-001 | Application language/framework | Accepted — ADR-015 | Python 3.12 aligns with model/benchmark tooling | Milestone 1 |
| OPEN-002 | Chunk unit | Accepted — ADR-013 | One source record per immutable MVP chunk; multi-turn windows deferred | Milestone 2 migration |
| OPEN-003 | Final SQL columns and migration policy | Accepted — ADR-017 | Forward-only checksum SQL migrations; immutable chunks, jobs, registry, flexible vectors | Milestone 2 |
| OPEN-004 | Embedding targets | Accepted — ADR-013 | Fact embeddings first; chunk embeddings remain benchmark experiment | Milestone 7 |
| OPEN-005 | Graph ID allocation/isolation | Accepted — ADR-014 | PostgreSQL-backed global numeric graph-ID registry | Milestone 6 |
| OPEN-006 | Ingestion job state machine/write order | Accepted — ADR-014 | Per-chunk state contract; storage implementation in Milestone 2/8 | Milestone 8 |
| OPEN-007 | Extraction model/provider | Held | Quality/cost/latency/privacy tradeoff; deterministic baseline only | Separate model-backed extraction approval |
| OPEN-008 | Dedicated procedural-memory graph schema | Held | Better procedure semantics vs premature ontology complexity | After procedural retrieval requirement |
| OPEN-009 | PostgreSQL full-text/BM25 fusion | Held | Better lexical recall vs ranking/tuning complexity | After dense baseline evaluation |
| OPEN-010 | Approximate vector index | Held | Lower latency vs recall loss and filter behavior | After exact latency breach |
| OPEN-011 | Retention/deletion policy | Held | Storage/privacy obligations require product context | Before non-benchmark data |
| OPEN-012 | Generic ingestion surface | Accepted — ADR-012 | Internal application service; public HTTP endpoint deferred | Milestone 0 |
| OPEN-013 | LongMemEval entrypoint | Accepted — ADR-012 | Dedicated CLI/script calls generic service; endpoint deferred | Milestone 3 |
| OPEN-014 | LLM provider and operational policy | Held | Provider, structured-output reliability, privacy/PII, cost, latency, and fallback must be selected together | Before a real model adapter |

Held items must not enter implementation through convenience or incidental dependency behavior. Promote one to `Proposed` with an approval packet when its revisit trigger occurs.

## 6. New Decision Template

```markdown
### ADR-NNN — Title

**Status:** Proposed | Accepted | Rejected | Deferred | Superseded
**Date:** YYYY-MM-DD
**Supersedes / superseded by:** ADR-NNN or none

**Context:** What changed or what problem requires a decision?

**Decision:** Exact proposed or approved behavior.

**Why:** Evidence and reasoning.

**Alternatives:** Options considered.

**Tradeoffs / hold-ons:** Costs, risks, and intentionally deferred work.

**Consequences:** Code, schema, tests, operations, security, and documentation impact.

**Verification:** Evidence required to validate the decision.

**Revisit trigger:** Concrete condition that reopens it.
```

## 7. Approved Milestone 0 Decision Records

### ADR-012 — Generic v1 Contract and Internal-Service Boundary

**Status:** Accepted
**Date:** 2026-08-16

**Decision:** Core ingestion accepts [Generic Context Ingestion Contract v1](ingestion_contract_v1.md) through an in-process `IngestionService`. It accepts `ContextBatch` and `ContextRecord` only, and validates derived-memory candidates with explicit type, scope, source-span, and bitemporal fields before later graph/embedding work. LongMemEval gets a dedicated CLI/script adapter in Milestone 3; no public HTTP ingestion endpoint is part of MVP.

**Why:** Preserves a generic core while avoiding premature network/API compatibility commitments. A benchmark script remains reproducible and invokes the same ingestion path as future sources.

**Alternatives:** Public versioned HTTP endpoint now; LongMemEval-specific endpoint; benchmark-specific core path.

**Tradeoffs / hold-ons:** Callers are local-process only during MVP development. Public authentication, rate limiting, and API compatibility are deferred with the HTTP surface.

**Consequences:** Milestone 1 defines internal typed models and a service port. Source schemas stay in adapters. LongMemEval labels stay outside canonical contracts.

**Verification:** Generic and LongMemEval adapter-output fixtures conform to one v1 envelope and exclude evaluation labels.

**Revisit trigger:** A real external producer requires remote ingestion or cannot invoke the application service.

### ADR-013 — Record Chunks and Fact-Only Embeddings

**Status:** Accepted
**Date:** 2026-08-16

**Decision:** One accepted `ContextRecord` becomes one immutable raw MVP chunk. Persist embeddings for validated fact text only; retain raw chunks but do not write chunk embeddings initially.

**Why:** Record-level chunks preserve exact source provenance and make replay/conflict checks simple. Fact-only vectors constrain initial storage and model work while retaining a later comparison against chunk embeddings.

**Alternatives:** Multi-turn windows; semantic chunking; fact plus chunk embeddings from first release.

**Tradeoffs / hold-ons:** Some facts need broader reader context, handled by approved progressive evidence expansion. Multi-turn chunking and chunk embeddings need measured comparison rather than intuition.

**Consequences:** Source record text is immutable SQL evidence; extraction spans refer to that one record chunk. Milestone 2 must preserve duplicate text from distinct records.

**Verification:** Fixture contains duplicate text in distinct records; later persistence tests prove distinct provenance and idempotent replay.

**Revisit trigger:** Record chunks materially reduce answer quality or require unacceptable extraction/read amplification.

### ADR-014 — SQL Graph-ID Registry and Per-Chunk Recovery Jobs

**Status:** Accepted
**Date:** 2026-08-16

**Decision:** PostgreSQL owns a durable global mapping from logical graph identity to HydraDB’s non-negative numeric IDs. Create one recovery job per immutable chunk using the v1 state contract: `pending_graph`, `pending_embeddings`, `verifying`, `completed`, `retryable_failed`, `terminal_failed`, and `manual_repair`.

**Why:** A persisted registry avoids accepting hash collisions and makes IDs stable across retries/processes. Per-chunk jobs keep recovery scope small and evidence-specific.

**Alternatives:** Fixed numeric ranges; truncated deterministic hashes; one session/batch job; distributed transaction coordinator.

**Tradeoffs / hold-ons:** Registry adds SQL schema and allocation code. A source batch can be partially complete, so operators need job-level visibility. No distributed atomicity is claimed.

**Consequences:** Milestone 2 creates registry/job persistence behind ports; Milestone 8 implements recovery transitions and failure injection.

**Verification:** Later tests prove stable allocation, same-key replay, distinct-key uniqueness, legal transitions, and recovery after every cross-store boundary.

**Revisit trigger:** Registry allocation becomes a measured bottleneck or database ownership changes.

### ADR-015 — Python/PostgreSQL Local Baseline and Golden Fixtures

**Status:** Accepted
**Date:** 2026-08-16

**Decision:** Use Python 3.12 for application code. Local development and CI use containerized PostgreSQL 16 with pgvector. Adopt the generic and LongMemEval adapter-output fixtures in [docs/fixtures](fixtures/) as initial deterministic contract fixtures.

**Why:** Python aligns with benchmark adapters, structured model interfaces, and test fixtures. Containerized SQL makes local/CI behavior reproducible. Fixtures demonstrate generic-core equivalence before real providers or infrastructure.

**Alternatives:** Another language/runtime; host-installed database; no adapter fixture until Milestone 3.

**Tradeoffs / hold-ons:** Container runtime becomes a development prerequisite. Exact PostgreSQL schema, migration mechanism, and production deployment remain Milestone 2 decisions.

**Consequences:** Milestone 1 may add only Python package/test configuration and deterministic fakes. It must not add database/model dependencies or run paid calls.

**Verification:** Parse fixtures, validate v1 rules, and compare generic/LongMemEval canonical shapes in deterministic unit tests.

**Revisit trigger:** Deployment constraints require a different supported runtime or CI cannot provide the container service.

### ADR-016 — Dependency-Free Python Domain Skeleton

**Status:** Accepted
**Date:** 2026-08-16

**Decision:** Milestone 1 uses a `src/context_memory` Python 3.12 package with frozen dataclasses, explicit validators, standard-library `unittest`, protocol ports, and deterministic fakes. It adds no runtime third-party dependency, database driver, graph client, provider SDK, or source adapter.

**Why:** The approved v1 contract needs executable validation before storage/model decisions. Standard-library primitives keep the boundary inspectable and deterministic while avoiding premature dependency or persistence coupling.

**Alternatives:** Pydantic-based models; dependency injection framework; immediate PostgreSQL/HydraDB clients; real provider adapters.

**Tradeoffs / hold-ons:** Explicit validation is more code than declarative schema tooling. Dependency selection and concrete adapters remain later approval decisions.

**Consequences:** Domain constructors and `from_mapping` reject invalid contract versions, timestamps, labels, spans, enums, and scope. Ports describe future extraction, embedding, and chunk storage without operational behavior.

**Verification:** `PYTHONPATH=src python3.12 -m unittest discover -s tests -v` passes deterministic fixture, validation, fake, and immutable-conflict tests.

**Revisit trigger:** Contract complexity or a concrete external integration demonstrates that validated standard-library models are insufficient.

### ADR-017 — PostgreSQL Migration and Immutable Persistence Baseline

**Status:** Accepted
**Date:** 2026-08-16

**Decision:** Use containerized PostgreSQL 16 with pgvector, Psycopg 3 binary distribution, and forward-only numbered SQL migrations verified against a checksum ledger. Persist `evidence_chunks`, `graph_id_registry`, `ingestion_jobs`, and a flexible-dimension `memory_embeddings` table. Source-record chunks and jobs are inserted in one SQL transaction. All repository reads/replay lookups include `context_id`.

**Why:** Raw SQL migrations expose the schema and recovery contract directly without a migration framework. Psycopg’s binary package avoids local client-library setup for MVP. Flexible vector dimensions preserve model choice while exact search remains the initial policy.

**Alternatives:** Alembic; host-installed PostgreSQL; fixed 384-dimension vector column; a single session job; hash-derived graph IDs.

**Tradeoffs / hold-ons:** Docker daemon availability is required for integration verification. Forward-only migration correction requires a new migration rather than editing an applied file. Vector writes and approximate indexes remain deferred.

**Consequences:** Applied migration checksums are immutable. Same `(context_id, source_record_id)` with changed content conflicts. Duplicate text from different records remains distinct. Graph numeric IDs come from a SQL registry, not fixed ranges or truncated hashes.

**Verification:** Unit tests cover discovery, idempotence, duplicate versions, and checksum drift. Docker-gated integration tests cover PostgreSQL round trip, mutation conflict, duplicate-text provenance, registry stability, and transaction behavior.

**Revisit trigger:** Schema needs unsupported source types, migration operational cost becomes excessive, or flexible vector dimensions prevent required index/query behavior.

### ADR-018 — Dataset-Verified LongMemEval Normalization

**Status:** Accepted
**Date:** 2026-08-16

**Decision:** LongMemEval adapter parses `%Y/%m/%d (%a) %H:%M` timestamps, applies UTC only as a benchmark-local ordering basis, and stable-sorts sessions. It qualifies canonical session/record identity with source array index, preserves raw source identifiers/timestamps in metadata, and skips empty-text turns while recording a batch audit count.

**Why:** Local inspection of all 500 provided instances found timezone-free minute timestamps, unordered session arrays, repeated session IDs in 13 instances, and 12 empty turns. Earlier RFC 3339, already-sorted, unique-session assumptions were false.

**Alternatives:** Reject non-RFC timestamps; preserve source array order; require unique session IDs; persist empty chunks; claim normalized values are UTC source time.

**Tradeoffs / hold-ons:** Benchmark-local normalization preserves relative ordering but cannot establish real-world timezone semantics. Empty turns remain visible only in aggregate batch metadata, not as chunks. Later query adaptation must normalize `question_date` identically.

**Consequences:** The generic core stays unchanged. Adapter-specific metadata preserves source fidelity. Full benchmark dry run validates 500 batches and 246,738 canonical records without copying the dataset into the repository.

**Verification:** Unit, CLI dry-run, and live PostgreSQL service/replay tests pass. Full local dataset dry run completed with zero adapter validation failures.

**Revisit trigger:** Benchmark publishes source timezone semantics, changes timestamp format, or a production source requires a different chronology policy.

### ADR-019 — Deterministic Extraction Baseline and Quality Gate

**Status:** Accepted
**Date:** 2026-08-16

**Decision:** Milestone 4 accepts only `deterministic-fixture` extractor `v1`. Its SQL audit records are permanently labelled `extractor_kind=deterministic_fixture` and `quality_status=baseline_only`; the M4 migration rejects other values. Raw chunks remain episodic evidence. Extracted candidates default to semantic and session scope (chat when no session); procedural classification requires an explicit fixture type. Invalid drafts are retained with rejection reasons.

**Why:** A typed candidate schema and passing fake path can otherwise be mistaken for model-quality evidence. Fixed baseline labels, an explicit runtime guard, and fixture acceptance cases make the limitation executable rather than documentary only.

**Alternatives:** Add a real provider now; permit arbitrary extractor identifiers; silently discard invalid output; infer procedural memories from wording.

**Tradeoffs / hold-ons:** This proves provenance, validation, replay, and SQL audit behavior but says nothing about real extraction recall, precision, calibration, privacy, cost, or latency. User and organization promotion remain out of scope; a caller may explicitly provide user scope only after a future policy approval.

**Consequences:** Any model-backed provider requires a new approved ADR, evaluation plan, and forward-only migration to deliberately expand the baseline constraint. No code may describe M4 as production extraction quality.

**Verification:** Deterministic tests cover Unicode spans, defaults, explicit procedural classification, rejection retention, replay, non-fixture blocking, and PostgreSQL audit persistence. [Extraction Baseline v1](extraction_baseline.md) contains the acceptance matrix and production gate.

**Revisit trigger:** User approves a concrete model/provider and quality-evaluation packet.

### ADR-020 — Bounded LLM Assistance for Entity and Temporal Ambiguity

**Status:** Accepted
**Date:** 2026-08-16

**Decision:** M5 uses deterministic normalization, context isolation, exact canonical/alias matches, same-subject/predicate gating, and chronology checks first. A provider-neutral LLM port may resolve an entity only from an application-supplied bounded candidate set, and may classify an update only after the deterministic gates pass. Invalid selection, abstention, unrelated facts, and out-of-order evidence remain unresolved or no-update; they never produce a forced link. Real provider integration remains held.

**Why:** LLM judgment can resolve conversational ambiguity such as a pronoun or competing alias, but unrestricted identity selection or temporal classification can create cross-context links and false supersession.

**Alternatives:** Exact matching only; LLM on every entity/fact pair; embeddings before entity resolution; automatic acceptance of model output.

**Tradeoffs / hold-ons:** Bounded candidates can miss a valid remote antecedent until a later retrieval strategy supplies it. M5 does not establish provider quality, structured-output reliability, PII handling, cost, latency, or fallback behavior. It creates no HydraDB records.

**Consequences:** New code exposes ports and deterministic fakes, but no network/model SDK. Model output is constrained to existing candidate IDs or a closed update enum. `valid_to` comes only from supported incoming temporal evidence, never an LLM-invented timestamp.

**Verification:** Tests prove Unicode normalization, context isolation, exact-match bypass, bounded model selection, invalid model selection rejection, correction/state-change separation, same-subject/predicate gating, and out-of-order protection.

**Revisit trigger:** Before connecting any provider, approve an operational/model ADR covering vendor, schema validation, PII policy, retry/fallback, cost/latency budgets, evaluation metrics, and rollback.

### ADR-021 — Bolt Graph Writer and Immutable Graph Manifest

**Status:** Accepted
**Date:** 2026-08-16

**Decision:** Use official Neo4j Python-driver Bolt transport with HydraDB bearer authentication and caller-scoped `hydradb.idempotency_key`. Before writes, PostgreSQL registers immutable node/relationship logical identity, global graph ID, scalar payload, and payload hash. One legal `UNWIND` statement writes each node label or directed relationship type. Relationship IDs use the existing global SQL registry with relationship-specific kind names.

**Why:** Bolt exposes HydraDB durable mutation idempotency; HTTP does not expose a caller key. A SQL manifest rejects changed payloads before `MERGE ... SET` could overwrite graph properties. Global registry IDs replace unsafe partitioned ranges.

**Tradeoffs / hold-ons:** Each Bolt statement commits independently. The M6 writer can expose an incomplete plan after a failure; M8 owns job states, retry, reconciliation, and temporal-state mutation. Live verification waits for user-provided runtime URI/token.

**Consequences:** Graph records are scalar-only and restricted to approved labels/types. Source text remains SQL evidence; graph facts carry only provenance IDs/hashes/spans. No secret is stored in files. `SUPERSEDES` edges can be written, but updating an existing fact’s temporal properties is deferred to M8.

**Verification:** Deterministic plan/query/manifest tests, PostgreSQL migration/integration tests, driver import check, then a credentialed live HydraDB round trip with causal bookmark read.

**Revisit trigger:** HydraDB adds HTTP mutation idempotency, Bolt transport behavior changes, or M8 proves a different recovery boundary is required.

### ADR-022 — Hosted HydraDB Memory Ingestion Supersedes OSS Bolt Writes

**Status:** Accepted
**Date:** 2026-08-17

**Decision:** Use the hosted HydraDB HTTPS memory API as the active M6 integration. Generic records map to `POST /memories/add_memory` with a stable SQL chunk ID as `source_id`, explicit `infer: true`, tenant scope, optional sub-tenant scope, and `upsert: true`. PostgreSQL remains canonical for immutable raw evidence and provenance. Hosted HydraDB owns its internal graph, inference, processing state, and eventual recall; the application does not submit Cypher, custom nodes, relationships, or graph IDs.

**Why:** The selected runtime is the hosted HydraDB dashboard/API, not the checked-in OSS server. Requiring a Neo4j/Bolt URL would target a different deployment surface.

**Alternatives:** Continue with the checked-in OSS Bolt writer; use hosted knowledge upload for every source; remove local SQL provenance; wait for a custom hosted graph-write API.

**Tradeoffs / hold-ons:** Hosted processing is asynchronous and `upsert` overwrites a matching `source_id`; local immutable-chunk checks must run first. The dashboard term “database” must be confirmed against the API’s `tenant_id` at credentialed test time. M6 does not yet durably persist hosted receipts or poll terminal processing state; M8 owns outbox/replay/reconciliation. M5-derived facts/entities remain local audit/quality signals unless a later hosted custom-graph API is explicitly approved.

**Consequences:** Runtime configuration is `HYDRA_DB_API_KEY`, `HYDRADB_TENANT_ID`, optional `HYDRADB_SUB_TENANT_ID`, and optional `HYDRA_DB_API_URL`; no Neo4j URL. The historical ADR-021 implementation path is superseded and its forward-only SQL migration is retained as applied-schema history.

**Verification:** Deterministic request mapping, response/error, and processing-status tests. Credentialed smoke test queues one isolated memory only when the user supplies runtime credentials.

**Revisit trigger:** Credentialed dashboard test shows a database-to-tenant mismatch, hosted API version changes, custom graph-write support is required, or M8 introduces durable receipt recovery.

### ADR-023 — Checked-in HydraDB Local Bolt Runtime Supersedes Hosted API

**Status:** Accepted
**Date:** 2026-08-17

**Decision:** Use the checked-in HydraDB `graph-node` as the M6 runtime. The
application validates/extracts context, allocates stable graph IDs in
PostgreSQL, registers immutable graph payloads, and writes legal OpenCypher
`UNWIND` batches through the official Neo4j Bolt driver. Local development uses
a user-generated token file and local object-store path; no hosted API key is
required.

**Why:** The repository contains the graph durability, mutation, query, and
indexing implementation. It preserves application control over provenance,
temporal links, and explicit idempotent replay.

**Tradeoffs / hold-ons:** HydraDB does not transform raw text into semantic
facts. The LLM/deterministic extraction layer remains application-owned. Each
Bolt mutation is independently committed; M8 still owns job recovery and
reconciliation. Current source build requires Rust 1.91+, `just`,
libcypher-parser, and GraphBLAS; Docker is the preferred local runtime route.

**Consequences:** ADR-022 is superseded. Runtime variables are local
`CONTEXT_MEMORY_HYDRADB_URI`, `CONTEXT_MEMORY_HYDRADB_TOKEN`, and optional
`CONTEXT_MEMORY_HYDRADB_DATABASE`. The local token is not committed.

**Verification:** Deterministic graph writer/manifest tests, PostgreSQL replay
test, then a local `graph-node` Bolt write/read smoke using a causal bookmark.

### ADR-024 — Local HTTP Transport Supersedes Bolt Driver

**Status:** Accepted
**Date:** 2026-08-17

**Decision:** Use the checked-in `graph-node` HTTP query API for M6 writes and
reads. Requests carry local Bearer authentication, graph namespace, graph ID,
cell ID, query, parameters, and causal bookmark. `query_id` is request identity
only.

**Why:** Disposable runtime smoke reached `graph-node`, but its
`SlateDBGraph/0.1.0` product string is rejected by the official Neo4j Python
driver before a query runs. HTTP invokes the same checked-in query service
without that client-product gate.

**Tradeoffs / hold-ons:** HTTP lacks Bolt’s caller-provided durable mutation
idempotency metadata. PostgreSQL manifests reject changed replays; stable graph
IDs and `MERGE` make same-payload replays effect-idempotent. M8 still owns
operation state and partial-plan reconciliation.

**Consequences:** ADR-023's Bolt-client transport choice is superseded; its
local-runtime boundary remains accepted. Runtime variables are
`CONTEXT_MEMORY_HYDRADB_URL`, `CONTEXT_MEMORY_HYDRADB_TOKEN`, and optional
`CONTEXT_MEMORY_HYDRADB_DATABASE`.

**Verification:** HTTP mapping/typed-response tests plus disposable Docker
write/read smoke on the local adapter.

### ADR-025 — Ingestion-Only Package Routing Mirrors Relevant HydraDB Layers

**Status:** Accepted
**Date:** 2026-08-17

**Decision:** Move active generic ingestion and graph-write orchestration under
`context_memory.ingestion`; route source adapters under `ingestion.sources`.
Expose local graph-node transport from `context_memory.client` and application
contracts from `context_memory.core`. Preserve old module paths as temporary
compatibility imports.

**Why:** HydraDB separates core contracts, public clients, query/mutation work,
and internal engine/shard/indexer implementation. The application mirrors only
owned layers, not HydraDB storage, lease, routing, or index logic.

**Tradeoffs / hold-ons:** Incremental routing only; retrieval/chat are not
refactored. Compatibility paths add short-term duplication and stay until
downstream callers migrate.

**Verification:** Import-route test plus full ingestion/unit suite. No
retrieval/chat module changes under this ADR.
