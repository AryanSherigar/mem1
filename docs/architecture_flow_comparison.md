# Architecture and Flow Comparison

**Status:** Review artifact, historical. The six conflicts in §6 are resolved:
the seven "Assistant-memory runtime" documents this review flagged were
removed (ADR-028) and [architecture_plan.md](architecture_plan.md) is now the
live single entry point. This document is kept for the reasoning trail, not as
a current spec — no implementation decision or migration is made by this
document.

**Compared sources:** [ARCHITECTURE.md](../ARCHITECTURE.md),
[AGENT.md](../AGENT.md), and the newly added subsystem documents, against the
implemented ingestion path and accepted ADRs in [decisions.md](decisions.md).

## 1. Two Designs Present in the Repository

The pulled documentation is not one fully consistent target architecture.

| Design                                      | Primary documents                                                                                                                          | Intended purpose                                                                                      |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| **Assistant-memory runtime**          | `ARCHITECTURE.md`, `AGENT.md`, `semantic_memory_distillation.md`, `startup_hydration_id_generation.md`, `llm_model_config.md`    | Interactive chat memory and LongMemEval evaluation, optimised for a fast prototype.                   |
| **Generic, durable ingestion engine** | `ingestion_contract_v1.md`, `ingestion_pipeline_roadmap.md`, `graph_schema_proposal.md`, `retrieval_architecture.md`, ADR-001–025 | Source-neutral evidence ingestion with replay, provenance, temporal correctness, and later retrieval. |

Our implemented Milestones 0–6 follow the second design. The first design is
useful as a future runtime/retrieval proposal, but it must not overwrite the
accepted ingestion contracts without a new approval.

## 2. Current Ingestion Flow

```text
generic source payload
  -> source adapter (LongMemEval is one adapter)
  -> ContextBatch / ContextRecord validation
  -> immutable one-record chunk + content hash
  -> PostgreSQL chunk + pending_graph job, one transaction
  -> extraction candidate + exact source-span validation
  -> bounded entity/update decision
  -> PostgreSQL graph-ID allocation + immutable graph manifest
  -> local HydraDB HTTP UNWIND/MERGE write + causal read verification
  -> pending_embeddings / later pgvector write
  -> cross-store verification -> completed
```

Implemented now: contract validation, LongMemEval normalization, immutable SQL
chunk persistence, deterministic extraction baseline, bounded resolution,
integer graph IDs, graph manifest, and local HydraDB HTTP graph writes.

Not implemented: real LLM provider, real embeddings/pgvector retrieval,
durable worker/outbox recovery, retrieval/chat API, startup hydration, and
benchmark prediction generation.

## 3. Pulled Runtime Flow

```text
chat turn
  -> respond immediately
  -> background task
  -> in-process embedding candidate lookup
  -> extractor LLM returns ADD / UPDATE / DELETE
  -> direct graph mutation + in-memory vector add/remove

user question
  -> LLM time-range inference
  -> in-memory vector top-K
  -> HydraDB graph expansion / SUPERSEDES traversal
  -> reader LLM answer
```

This flow assumes a chat runtime, an in-process vector index, direct action
classification, and a graph-led restart model. It is retrieval- and
interaction-centric rather than evidence-ingestion-centric.

## 4. Material Differences

| Area                  | Current ingestion design                                                                                        | Pulled runtime design                                                 | Consequence                                                                                                                                 |
| --------------------- | --------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Scope                 | Generic context ingestion; LongMemEval adapter only                                                             | Interactive chat plus benchmark runtime                               | Current core accepts exports, documents, and future connectors without source-specific branches.                                            |
| Durable authority     | PostgreSQL owns immutable raw chunks, job state, graph-ID registry, and embeddings; HydraDB owns graph topology | HydraDB graph is called the source of truth                           | Keep split authority. Graph alone cannot preserve SQL job/replay guarantees or vector rows.                                                 |
| Embeddings            | Versioned fact embeddings in PostgreSQL/pgvector; M7 pending                                                    | 384-dimension in-process NumPy`EmbeddingIndex`, rebuilt at startup  | Do not introduce a memory-only index as authoritative. A cache may later sit above pgvector.                                                |
| Graph transport       | Checked-in local`graph-node` HTTP query endpoint                                                              | Bolt-compatible driver/client                                         | Current HTTP adapter is verified against local HydraDB. Bolt document examples are stale for this repository path.                          |
| Identity              | PostgreSQL registry allocates stable non-negative integer IDs and rejects changed payloads                      | SHA-256-derived UUID strings                                          | HydraDB requires integer node IDs. UUID strategy cannot be used as stated; deterministic logical keys remain useful inputs to the registry. |
| Write consistency     | SQL transaction creates evidence/job; manifest plus idempotent`MERGE`; later outbox/reconciliation            | Python helper attempts graph and vector update together               | No cross-store transaction exists. Current job state machine correctly represents partial failure.                                          |
| Raw evidence          | Immutable chunk, source hash, exact Unicode span, provenance retained                                           | Turn-to-fact distillation; raw evidence less central                  | Current design is stronger for audit, correction, and generic-source reprocessing.                                                          |
| Memory update         | LLM only assists after deterministic subject, predicate, chronology, and candidate boundaries                   | Extractor chooses ADD/UPDATE/DELETE and target fact                   | Retain bounded decisions; never let a model select arbitrary graph IDs or delete evidence.                                                  |
| Temporal model        | Bitemporal: knowledge time (`observed_at`/`superseded_at`) separate from world validity                     | Update/delete examples set`valid_to` directly                       | Current model avoids treating a correction as a real-world state transition.                                                                |
| Entity resolution     | Context-scoped exact/alias match, then bounded LLM choice                                                       | Exact match, semantic blocking, LLM, later merge                      | Same direction, but semantic blocking depends on M7 embeddings. Merge/repoint behaviour remains future work.                                |
| Async work            | Durable ingestion-job state; worker/outbox planned in M8                                                        | Fire-and-forget background executor                                   | Do not use fire-and-forget: a process crash would lose work.                                                                                |
| Restart               | SQL jobs reconcile graph/embedding state; hydration not required for correctness                                | Re-query graph and re-embed in-memory index                           | If a cache is added, hydrate it from pgvector/SQL plus graph verification, not graph-only.                                                  |
| Benchmark isolation   | `context_id` and adapter metadata keep wilbenchmark labels out of runtime input                               | Per-question ephemeral graph wipe/re-ingest                           | Per-question isolation is an evaluation-harness policy, not a reason to make generic ingestion ephemeral.                                   |
| Retrieval evidence    | Approved progressive span -> neighboring text -> full chunk                                                     | Some runtime documents imply facts/turns directly into reader context | The selective-evidence policy already answers this: full chunk only when narrower evidence is insufficient.                                 |
| Model/provider policy | Provider-neutral ports and deterministic fakes; external credentials deferred                                   | Named models/providers and environment templates                      | Provider/model choice, PII, cost, rate limit, and retry policy need a separate approved milestone.                                          |

## 5. What Already Aligns

- Graph labels `Session`, `Turn`, `Fact`, `Entity`, and `Alias` align.
- Provenance (`EXTRACTED_FROM`), semantic links (`ABOUT`), aliases, and
  `SUPERSEDES` align conceptually.
- Memory type and scope are already explicit, orthogonal fields.
- LongMemEval must remain an adapter, with answer labels excluded from runtime
  ingestion.
- Hybrid retrieval remains the intended later shape: SQL semantic seeding,
  HydraDB traversal, evidence-grounded reader context, and abstention.
- LLM assistance is appropriate for extraction and ambiguity, after contract
  and policy checks.

## 6. Documentation Conflicts Requiring Reconciliation

1. `ARCHITECTURE.md` calls HydraDB the only source of truth and vectors an
   in-memory index; `retrieval_architecture.md` and accepted ADRs choose SQL
   chunks and pgvector. The latter matches implemented ingestion.
2. `AGENT.md`, `llm_model_config.md`, and several code snippets require Bolt;
   ADR-024 and the implementation use local HTTP because the checked-in server
   rejected the standard Neo4j driver identity.
3. `AGENT.md` and `startup_hydration_id_generation.md` prescribe UUID graph
   IDs; HydraDB’s local mutation surface requires non-negative integer IDs.
4. `semantic_memory_distillation.md` treats delete as graph/vector removal;
   current design preserves evidence and needs an approved retraction model.
5. `AGENT.md` describes a future `src/core`, `src/db`, `src/memory`, and chat
   layout. Current ingestion-only structure is deliberately
   `context_memory/core`, `ingestion`, `persistence`, and `client`; retrieval
   and chat have not been moved or created.
6. Several new documents contain absolute `file:///home/aryan-sherigar/...`
   links. They do not resolve in this checkout and should be converted to
   repository-relative links during the documentation-reconciliation milestone.

## 7. Recommended Direction Before Further Implementation

Keep the current ingestion architecture as the approved base. Treat the pulled
documents as requirements for later retrieval/chat work, after resolving the
conflicts above.

Proposed sequencing:

1. **M7:** pgvector fact embeddings, versioning, scoped exact search, then
   assess an optional in-memory cache as non-authoritative.
2. **M8:** durable outbox/worker, graph update/retraction mutations,
   reconciliation, and restart recovery.
3. **Retrieval approval:** reconcile `ARCHITECTURE.md`/`AGENT.md` with ADRs;
   choose HTTP transport, integer-ID adapter, cache policy, and benchmark
   isolation harness.
4. **Chat approval:** add async interactive endpoints only after durable worker
   semantics and authorization/retention policy exist.

Until approval, no source-of-truth switch, Bolt migration, UUID migration,
in-memory-vector authority, graph wipe strategy, or named external model
provider should be implemented.
