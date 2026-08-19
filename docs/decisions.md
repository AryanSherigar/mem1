# Architectural Decisions

| Field | Value |
|---|---|
| Status | Living, append-only decision history |
| Last verified | 2026-08-19 against local HTTP Docker write/read smoke, PostgreSQL manifest implementation, deterministic M7 embedding/model-adapter unit suite, and a hosted HydraDB v2 sandbox investigation (ADR-026) |
| Next review | Milestone 8 approval packet |
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
| ADR-022 | Hosted HydraDB memory ingestion supersedes OSS Bolt writes | Superseded by ADR-023 | 2026-08-17 |
| ADR-023 | Checked-in HydraDB local Bolt runtime supersedes hosted API | Superseded by ADR-024 (transport only) | 2026-08-17 |
| ADR-024 | Local HTTP transport supersedes Bolt driver | Accepted | 2026-08-17 |
| ADR-025 | Ingestion-only package routing mirrors relevant HydraDB layers | Accepted | 2026-08-17 |
| ADR-026 | Hosted HydraDB v2 knowledge/memory API evaluated and held; local graph-node stays sole source of truth | Deferred | 2026-08-19 |
| ADR-027 | Provider-neutral LLM client and local embedding model adopted for M5/M7 | Accepted | 2026-08-19 |
| ADR-028 | Obsolete pulled-runtime documents removed | Accepted | 2026-08-19 |
| ADR-029 | Fireworks + deepseek-v4-flash as the M5 model provider | Accepted | 2026-08-19 |
| ADR-030 | Candidate-to-graph-plan mapping; SUPERSEDES deferred pending predicate_key | Accepted | 2026-08-19 |
| ADR-031 | M8 job state machine: whole-chunk idempotent replay, not per-stage resumption | Accepted | 2026-08-19 |
| ADR-032 | Local HydraDB graph-node built via Docker and live-verified | Accepted | 2026-08-19 |

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

### ADR-026 — Hosted HydraDB v2 Knowledge/Memory API Evaluated and Held

**Status:** Deferred
**Date:** 2026-08-19

**Decision:** A credentialed round trip against the hosted HydraDB v2 API (`api.hydradb.com`, `POST /context/ingest`, `GET /context/status`, `POST /query`) was run in a sandbox database. The local `graph-node` (ADR-023/ADR-024) remains the sole source of truth for graph topology and bitemporal fact state. The hosted v2 API is not adopted as part of the active runtime; it is parked as a documented option for a later, explicitly-scoped local-vs-hosted comparison.

**Why:** Probing showed hosted v2 is a managed knowledge/memory RAG product (its own chunking, embeddings, BM25, LLM entity/relation extraction, temporal reasoning), not a Cypher-like graph surface. Two findings rule it out as a source-of-truth candidate today: (1) its `temporal_facts` structured output (`event_start`/`event_end`/`date_precision`/`status`) is only populated for sources that go through its own LLM extraction — Bring-Your-Own-Graph (`graph_payload`) submissions, which are what bounded/deterministic ingestion (ADR-019/ADR-020) would have to use, get zero `temporal_facts` entries; (2) even where populated, `temporal_facts` carries one time axis (world/event validity) with no equivalent of `observed_at`/`superseded_at` (ADR-006), so it cannot host bitemporal state regardless. A BYOG relation also has no structured property bag to carry those fields ourselves (`source`/`target`/`predicate`/`context`/`temporal_details` only, `temporal_details` is a single freeform string).

**Alternatives considered:** Hybrid split (local graph-node as bitemporal system of record, hosted v2 as a one-way, rebuildable search projection fed by an outbox) — judged architecturally sound (comparable to a Postgres-behind-search-index pattern) but not adopted now because its own justification (hosted search quality beating pgvector/Postgres full-text search) is unverified, it adds a second operated dependency alongside the already-self-hosted `graph-node` cluster, and it would route conversational content to a third party before any PII/retention policy exists for this project.

**Tradeoffs / hold-ons:** No hosted-v2 code, config, or credentials are part of the active runtime. The `hackhydra-smoke` sandbox database and its one test record remain live in the vendor account from this investigation; not referenced by any code path. Revisiting this requires a real pgvector/FTS-vs-hosted-v2 retrieval-quality comparison and a data-handling decision, not just renewed interest.

**Consequences:** ADR-001's storage split (PostgreSQL: raw evidence/embeddings/job state; HydraDB: graph/temporal topology/traversal) stands unchanged and unambiguously: "HydraDB" throughout this decision log and the architecture plan means the local `graph-node`, never the hosted v2 API, unless a future ADR says otherwise.

**Verification:** Manual credentialed round trip (database create → knowledge ingest with `graph_payload` → status poll → query with `graph_context`/`temporal_facts`) and a second probe isolating LLM-extraction-path vs BYOG-path `temporal_facts` population, both run against a live sandbox database, 2026-08-19.

**Revisit trigger:** A scoped, approved comparison shows hosted v2 retrieval quality meaningfully exceeds pgvector/Postgres full-text search, and a data-handling/PII policy for external egress of conversational content is approved.

### ADR-027 — Provider-Neutral LLM Client and Local Embedding Model Adopted for M5/M7

**Status:** Accepted
**Date:** 2026-08-19

**Decision:** Adopt the OpenAI-compatible `LLMClient` (structured/text completion behind `base_url`/`api_key`/`model_name`, provider-agnostic across Fireworks/OpenRouter/Groq/OpenAI) as the concrete implementation behind the existing `EntityResolutionModel` and `TemporalUpdateModel` ports (`ingestion/ports.py`), unblocking M5's held real-provider connection. Adopt `sentence-transformers/all-MiniLM-L6-v2` (384-dim, CPU, no API key required — already the model named in `retrieval_architecture.md` §3) as the concrete `Embedder` port implementation, unblocking M7. The in-process `EntityNameIndex`/embedding-cache pattern is adopted only as a non-authoritative Tier-2 semantic-blocking cache for entity resolution, rebuilt from PostgreSQL/pgvector on demand; it is never the embedding source of truth (pgvector is, per ADR-013).

**Why:** Both ports already existed as deterministic-fake-only Protocols with the model-selection guardrail enforced by the caller (`EntityRegistry.resolve` rejects any ID outside the bounded candidate set — ports.py/resolution.py behavior is unchanged by this ADR). This closes the M5 hold-on and the M7 blocker without redesigning either boundary. `sentence-transformers` needs no provider credential, so embedding generation and pgvector persistence can be verified locally today, unlike the LLM-backed ports.

**Alternatives:** Call an API-based embedding endpoint through the same `LLMClient` instead of a local model — rejected for now because it would make M7 verification credential-gated too, for no accuracy benefit at this scale, and `retrieval_architecture.md` already approved the local model. Porting the UUID/sha256 `IdGenerator` from the evaluated main-branch work — rejected; the existing `logical_key` + PostgreSQL integer-registry design (`allocate_graph_id`, ADR-014) already implements the same determinism principle and is what HydraDB's local mutation surface requires, so the UUID generator would be redundant, not additive.

**Tradeoffs / hold-ons:** No LLM provider credential is present in this environment (`.env` carries only `HYDRA_DB_API_KEY`), so `LLMEntityResolutionModel`/`LLMTemporalUpdateModel` are implemented and unit-tested against an injected fake client only; live verification against a real provider remains held until credentials are supplied, exactly as M6 held for its HydraDB token. The semantic-blocking cache is process-local and cold-starts empty; it is a candidate-recall aid, never a correctness dependency, since exact/alias matching and the bounded-LLM tier remain the resolution authority.

**Consequences:** New dependencies (`openai`, `pydantic`, `python-dotenv`, `sentence-transformers`, `numpy`) added to `pyproject.toml`. New modules: `context_memory/core/llm_client.py`, `context_memory/core/config.py`, `context_memory/ingestion/model_adapters.py`, `context_memory/ingestion/embedding.py`, `context_memory/ingestion/semantic_blocking.py`, `context_memory/persistence/postgres.py::PostgresEmbeddingStore`. Roadmap M7 moves from "blocked" to "implemented, embedding generation locally verified"; M5's real-provider hold-on moves to "adapters implemented, live verification pending provider credentials."

**Verification:** Deterministic unit tests for both LLM-backed adapters using an injected fake OpenAI-shaped client (no network call); deterministic unit tests for the embedder using an injected fake encoder; an opt-in local smoke test (gated behind an explicit environment flag, mirroring `test_hydradb_live.py`) that loads the real `sentence-transformers` model and embeds text, since no credential is required for that path. PostgreSQL round-trip tests for `PostgresEmbeddingStore` follow the existing `CONTEXT_MEMORY_TEST_DATABASE_URL`-gated pattern.

**Revisit trigger:** A real LLM provider credential is supplied (triggers live M5 verification); measured embedding latency/accuracy motivates a different model; M8 recovery work changes how `pending_embeddings` jobs are retried.

### ADR-028 — Obsolete Pulled-Runtime Documents Removed

**Status:** Accepted
**Date:** 2026-08-19

**Decision:** Remove `ARCHITECTURE.md`, `AGENT.md`, `docs/semantic_memory_distillation.md`, `docs/startup_hydration_id_generation.md`, `docs/llm_model_config.md`, `docs/entity_resolution_strategy.md`, and `docs/temporal_query_resolver.md` from the repository. `docs/architecture_plan.md` is the new single entry point describing the accepted ingestion and retrieval design.

**Why:** All seven describe the "Assistant-memory runtime" design identified in `architecture_flow_comparison.md` §1 (in-process NumPy vector authority, UUID graph IDs, hosted-API-first framing, unbounded LLM ADD/UPDATE/DELETE, full-wipe re-ingestion) that Milestones 0–7 do not follow and that conflicts with Accepted ADRs (ADR-001 storage split, ADR-006 bitemporal model, ADR-014 integer graph IDs, ADR-019/ADR-020 bounded extraction/resolution). None of the seven are listed in `AGENTS.md`'s repository map; `graph_schema_proposal.md` and `retrieval_architecture.md`, which share their hackathon-era origin, already carry an explicit "current runtime decision" correction banner and stay. Keeping contradicting documents in the tree next to the accepted design was the actual source of confusion this decision resolves.

**Alternatives:** Keep the documents with a superseded banner (rejected — this repository already tried that for two documents and the other five never got one, and seven banners duplicate what a single removal note plus git history already provides); fold their salvageable content into `architecture_plan.md` first (checked — nothing in the seven is not already covered by an Accepted ADR or a surviving living document).

**Tradeoffs / hold-ons:** Full text remains recoverable from git history at the commit before this ADR; nothing here is destroyed, only removed from the working tree. `architecture_flow_comparison.md` itself is kept as the historical review artifact that produced this decision, not deleted.

**Consequences:** `AGENTS.md`'s repository map and `docs/ingestion_pipeline_roadmap.md` are updated in the same change set to stop referencing removed files and to add `docs/architecture_plan.md`.

**Verification:** `git log --follow` on any removed path recovers full history; `docs/architecture_plan.md` published in the same commit.

**Revisit trigger:** None expected; reopen only if a removed document's content is needed for a future audit, retrievable from git history without restoring it to the working tree.

### ADR-029 — Fireworks + deepseek-v4-flash as the M5 Model Provider

**Status:** Accepted
**Date:** 2026-08-19

**Decision:** Use Fireworks (`https://api.fireworks.ai/inference/v1`) with `accounts/fireworks/models/deepseek-v4-flash-0731` as the default model behind both `LLMEntityResolutionModel` and `LLMTemporalUpdateModel` (ADR-027). Credential resolution: `ENTITY_RESOLUTION_API_KEY`/`TEMPORAL_UPDATE_API_KEY` if set, else the shared `FIREWORKS_API_KEY`, so one key in `.env` is sufficient. This closes the M5 hold-on ("connecting a real model provider requires its own operational approval") that ADR-020 left open.

**Why:** User-supplied credential and explicit model choice. `LLMClient` was already provider-neutral (ADR-027); this is a configuration/default change, not a new code path.

**Tradeoffs / hold-ons:** No PII handling, cost, rate-limit, or retry policy has been decided for this provider — `architecture_flow_comparison.md`'s "Model/provider policy" row is still open beyond the bare provider/model choice made here. `structured_completion`'s `response_format: json_object` + `temperature: 0.0` request shape was verified against this exact model/endpoint (see Verification); no `max_tokens`/`top_k`/penalty tuning has been applied.

**Consequences:** `core/config.py` defaults changed from placeholder `gpt-oss-20b` to `accounts/fireworks/models/deepseek-v4-flash-0731` for both roles. `.env` (now at `src/.env` after the repository restructure) carries `FIREWORKS_API_KEY`; no key is logged or committed.

**Verification:** Live call against the real endpoint: (1) a bare `structured_completion` round trip, (2) `LLMEntityResolutionModel.resolve_entity` against a bounded 2-candidate set and against a no-match set, (3) `LLMTemporalUpdateModel.classify_update` on a correction-shaped and a state-change-shaped fact pair. Run 2026-08-19; see the session record for exact outputs. A checked-in opt-in live test (`src/tests/test_model_adapters_live.py`, gated on `FIREWORKS_API_KEY`/role-specific keys being present in the runtime environment, mirroring `test_hydradb_live.py`) now repeats this. Building that test surfaced and fixed a real bug: `core/config.py` initially auto-called `load_dotenv()` at import time (copied from `main` branch's script-style config); since `unittest discover` imports every test module to enumerate cases, that silently repopulated `FIREWORKS_API_KEY` from `src/.env` even when the shell didn't have it set, defeating the live test's `skipUnless` gate and risking a real billed call from a plain, non-targeted test run. Fixed by removing the auto-load — this module now only reads `os.environ`, exactly like `test_hydradb_live.py` already does for the HydraDB token; credentials must come from an explicitly-sourced `.env` or an exported var. Verified both directions: unset → all live-gated tests skip in 0.17s with zero network activity; explicitly sourced → all four live tests pass for real.

**Revisit trigger:** Response latency/cost motivates a different model; a PII/retry/rate-limit policy is approved and requires client changes beyond model selection.

### ADR-030 — Candidate-to-Graph-Plan Mapping; SUPERSEDES Deferred Pending `predicate_key`

**Status:** Accepted
**Date:** 2026-08-19

**Decision:** `ingestion/graph_plan_builder.py::GraphPlanBuilder` maps one `Chunk` + its accepted `ExtractedMemoryCandidate`s into a `GraphWritePlan`: one `Session` node (keyed by `record.session_id`, falling back to a per-`context_id` default bucket when absent), one `Turn` node per chunk, one `Fact` node per accepted candidate, one `Entity` node per resolved mention, `Alias` nodes for every alias already on the resolved `EntityProfile`, and `HAS_TURN`/`EXTRACTED_FROM`/`ABOUT`/`HAS_ALIAS` edges connecting them — labels/types unchanged from `core/graph.py` (already approved). An entity mention that resolves to `None` (`EntityResolution.entity is None`) gets no `ABOUT` edge — never a forced link, preserving ADR-020's guarantee at this layer too.

Explicitly out of scope for this builder: `SUPERSEDES` edges and `RELATES_TO` edges. Nothing is invented to fill the gap; see rationale.

**Why:** `TemporalUpdateClassifier.classify` (ADR-020) requires a `FactState.predicate_key` to run its same-subject/same-predicate gate before it will even consider calling the model. Nothing in the deterministic extraction baseline (ADR-019) produces a predicate for a candidate — `ExtractedMemoryCandidate`/`ExtractionDraft` carry free text only, no subject/predicate/object decomposition. Wiring `SUPERSEDES` here would mean inventing a predicate-extraction heuristic inside graph-plan construction, which is extraction-quality work, not plan-mapping work, and would misrepresent the deterministic baseline's own documented limits (ADR-019: "no code may describe M4 as production extraction quality"). `RELATES_TO` has no source data driving it at all yet (no entity-entity relation extraction exists anywhere in the pipeline).

**Alternatives:** Derive a weak predicate heuristic (e.g., first verb phrase, or reuse `memory_type`) just to unblock `SUPERSEDES` — rejected, this would silently encode an extraction-quality decision as if it were settled, the opposite of this project's own governance rule against inferring unresolved decisions. Skip building the graph-plan mapping until predicate extraction exists — rejected, the Session/Turn/Fact/Entity/Alias/`ABOUT` structure is fully specified today and has real value (queryable provenance graph) independent of supersession.

**Tradeoffs / hold-ons:** Corrections and real-world updates are not yet representable in the graph — every accepted candidate becomes a new `Fact` node with no link back to anything it might supersede, even when the deterministic/bounded machinery to *decide* that (ADR-020) already exists and is live-verified (ADR-029). This is a real, load-bearing gap, not a minor one: bitemporal correctness (ADR-006) has no graph-side representation until it's closed.

**Consequences:** A future extraction milestone must add `predicate_key` (or equivalent subject/predicate structure) to `ExtractedMemoryCandidate` before `GraphPlanBuilder` can be extended with `SUPERSEDES`. Until then, `TemporalUpdateClassifier` remains built, tested (deterministic + live, ADR-029), and unreachable from the real ingestion path — a documented, not hidden, gap.

**Verification:** Deterministic unit tests (`src/tests/test_graph_plan_builder.py`) cover: zero-candidate chunks still produce Session/Turn; unresolved mentions produce no `Entity`/`ABOUT`; resolved mentions produce both; aliases produce `Alias`/`HAS_ALIAS` pairs; every node/relationship carries `context_id` (required by `GraphWritePlan.__post_init__`); repeated builds are deterministically replayable (identical graph IDs).

**Revisit trigger:** A future extraction milestone adds predicate/subject-object structure to extracted candidates, at which point `SUPERSEDES` wiring becomes possible and should be added as its own ADR, not folded silently into this one.

### ADR-031 — M8 Job State Machine: Whole-Chunk Idempotent Replay, Not Per-Stage Resumption

**Status:** Accepted
**Date:** 2026-08-19

**Decision:** `ingestion/orchestrator.py::IngestionOrchestrator` drives each chunk through extraction → graph-plan construction → graph write → embedding persistence → verification → `completed`, transitioning `ingestion_jobs.state` at each boundary per the table in `docs/ingestion_contract_v1.md` §5 (now implemented as `core/enums.py::is_legal_job_transition`, table-driven and independently unit-tested). On any exception, the job moves to `retryable_failed` (transient/unclassified errors) or `terminal_failed` (`ContractValidationError`/`ImmutableRecordConflictError`/`GraphPayloadConflictError` — payload/policy problems retrying can't fix), capturing the last successfully-reached state. Retrying means calling the same entrypoint again: it **re-runs the whole chunk from the top** — extraction, resolution, graph-plan construction, the graph write, and embedding persistence all over again — relying on each stage's already-existing idempotent-replay behavior (`ExtractionService`'s stable `attempt_id`, `PostgresGraphManifestStore`, `PostgresEmbeddingStore`, `GraphWriter`'s `MERGE` identities) rather than resuming from a finer-grained checkpoint.

The state-transition table was generalized from the contract's single documented "next state" per row to "any state at or after the current one in happy-path order is a legal target," specifically so a full top-to-bottom replay can re-issue a state the job already reached (e.g. resuming `retryable_failed` straight to `pending_embeddings` when `last_verified_state` was already `pending_graph`) without that looking like an illegal regression.

**Why:** True per-stage resumption (skip straight to "just redo the embeddings step") needs a way to reload what extraction/resolution already decided without recomputing it — `ExtractionStore`/`EmbeddingStore` are currently write/audit-only ports with no read-back method, and building that read path is its own scoped piece of work, not a byproduct of wiring the state machine. Whole-chunk replay is strictly weaker but correct today, given every stage it touches was already built to tolerate exactly this.

**Alternatives:** Build the read-back ports now and do true per-stage resumption — rejected for this pass on scope grounds; flagged as future work below. Treat any failure as `terminal_failed` (no automatic retry at all) — rejected, it would make every transient network blip (the exact `_FailNTimesTransport` scenario the tests exercise) require manual intervention, defeating the point of M8.

**Tradeoffs / hold-ons:** Cost, not correctness: a chunk that fails at the embeddings stage after a successful (idempotent) graph write still re-extracts and re-submits an unchanged graph plan on retry — wasted LLM/embedding-model calls and network round trips, never wasted correctness, since every re-submitted stage is a verified no-op on unchanged input. Verification before `completed` is same-process evidence (nothing raised, plus one real `chunk_store.get` re-read) — not an independent re-read of HydraDB or pgvector, since `EmbeddingStore` has no `get`/`exists` port yet; a crash between `verifying` and `completed` is caught by the next replay re-doing the same idempotent work, not by a genuinely independent audit.

**Consequences:** New: `core/enums.py::is_legal_job_transition`, `core/errors.py::IllegalJobTransitionError`, `persistence/postgres.py::PostgresJobStore` (+ `fakes.py::InMemoryJobStore`), `ingestion/orchestrator.py::IngestionOrchestrator`. `core/models.py::IngestionJob` extended with `context_id`/`attempt_count`/`last_verified_state`/`last_error` to carry what the state machine needs (previously `job_id`/`chunk_id`/`state` only). `ports.py` gained `JobStore` and `EmbeddingStore` Protocols.

**Verification:** `src/tests/test_job_transitions.py` (11 tests, pure state-table logic, every table row plus the generalized skip-ahead/idempotent-replay rule). `src/tests/test_orchestrator.py` (6 tests, fakes only): happy path reaches `completed` and writes the expected graph/embedding shape; zero-candidate chunk still completes; a transient failure produces `retryable_failed` and a second call succeeds; a `completed` job is never reprocessed; a `terminal_failed` job blocks auto-retry. `PostgresJobStore` verified against real PostgreSQL: lifecycle to `completed`, `seed` idempotency (does not reset progress), illegal-transition rejection, `attempt_count` increment on failure.

**Revisit trigger:** Retry cost (re-running extraction/graph-write/embeddings on every retry) is measured and found too expensive, motivating `ExtractionStore`/`EmbeddingStore` read-back ports and true per-stage resumption as a follow-up ADR.

### ADR-032 — Local HydraDB Graph-Node Built via Docker and Live-Verified

**Status:** Accepted
**Date:** 2026-08-19

**Decision:** Build `graph-node` from `hydradb/Dockerfile`'s `runtime` target (`docker build --target runtime -t hydradb-graph-node:local hydradb/`) instead of a native `cargo build` — this environment has neither the Rust toolchain nor `just`/`libcypher-parser`/GraphBLAS installed, all of which the Dockerfile's `build-base` stage installs for you. Run it single-node, local-object-store, plaintext, exactly per `hydradb/README.md`'s "Run a local server" recipe, translated to `docker run` with the same env vars and ports 7687 (Bolt)/8443 (HTTP)/9090 (admin) mapped to `127.0.0.1`. State lives in `.hydradb/` (gitignored).

**Why:** ADR-021's and ADR-024's own verification sections said live verification was waiting on a user-provided runtime URI/token; every prior session left `test_hydradb_live.py` skipped. This closes that, using the exact build path the maintainers themselves ship for people without the native toolchain.

**Consequences:** `tests/test_hydradb_live.py` (previously always skipped in this environment) now passes for real. New `tests/test_orchestrator_live.py`: the full M8 pipeline — `IngestionOrchestrator` with real `PostgresChunkStore`/`PostgresJobStore`/`PostgresEmbeddingStore`, a real `sentence-transformers` embedder, and a real `HydraHttpTransport` against this node — reaches `completed`, then is independently re-read straight from HydraDB (not same-process evidence, unlike the fake-backed orchestrator tests) to confirm the entity actually landed in the graph. No LLM provider needed for this test: entity resolution here only needed exact/new-entity matching.

**Tradeoffs / hold-ons:** This is one plaintext, single-node, local-object-store instance — nothing about clustering, TLS, MinIO/S3 backing, or multi-node placement has been exercised. The container isn't part of `compose.yaml` (only Postgres is); anyone reproducing this needs to build the image and run it by hand per this ADR, or `compose.yaml` should gain a service for it as follow-up work.

**Verification:** `curl` write+read round trip matching the README's own documented expected output exactly (`{"type":"vertex_id","value":2}`). `src/tests/test_hydradb_live.py` (1 test) and `src/tests/test_orchestrator_live.py` (1 test) both pass against the running container. Full suite with Postgres + this node + the Fireworks credential all present: 115 tests, 1 skipped (the separately-opt-in embedding-model-download smoke test), 0 failures.

**Revisit trigger:** `compose.yaml` gains a `graph-node` service so this is reproducible with one `docker compose up` instead of a hand-built image; or a real multi-node/TLS/MinIO configuration is needed for a later milestone.

**Addendum, 2026-08-19 — `compose.yaml` service added, two real bugs found closing the revisit trigger above:**

`graph-node` is now a `compose.yaml` service, gated behind the `graph` Compose profile (`docker compose --profile graph up -d graph-node`) so plain `docker compose up` — most work only needs Postgres — never triggers its multi-stage Rust build. Two genuine, empirically-found problems came up productionizing the manual setup into a restartable service, neither hypothetical:

1. **Host bind mounts break on restart.** The original service mounted `./.hydradb/store` directly (matching the manual setup this ADR started with). First start: fine. Stop, then start again against the same data: every write failed with `object store error: Operation \`put_opts\` with mode \`PutMode::Update\` not yet implemented by LocalFileSystem(file:///data/store)`. graph-node's object-store layer does a conditional PUT for restart write-fencing that Docker Desktop's macOS bind-mount passthrough doesn't support. Fix: Docker-managed named volumes (`hydradb_store`, `hydradb_cache`) instead of host bind mounts — these go through the real filesystem inside Docker Desktop's Linux VM and don't hit the gap.
2. **Named volumes at an arbitrary path fail on the non-root container user.** Mounting the new named volumes at an arbitrary `/data/store`/`/data/cache` failed differently: `Permission denied (os error 13)` creating the writer-lease directory. The runtime image runs as uid 10001 (`graph`), and only `chown -R graph:graph`s two specific paths at build time — `/tmp/graph` and `/var/cache/slatedb` (see `hydradb/Dockerfile`'s `runtime-base` stage). A fresh named volume mounted at a path Docker has never seen starts out root-owned; mounted at a path the image already prepared, Docker copies that path's existing ownership onto the new volume. Fix: mount `hydradb_store`/`hydradb_cache` at exactly `/tmp/graph`/`/var/cache/slatedb`, not an invented path.

Both confirmed by actually stopping and restarting the Compose-managed container and re-running the same write/read round trip plus the full test suite (115 tests, 0 failures) against it — not just a first successful start.
