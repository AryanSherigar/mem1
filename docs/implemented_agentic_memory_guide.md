# Implemented Agentic Memory: Beginner Guide

**Status:** Living explanation of the implementation at commit `997c080`.
Read [architecture_plan.md](architecture_plan.md) for the accepted design and
[decisions.md](decisions.md) for the decision history. This guide explains
what exists today; it does not turn planned retrieval/chat work into completed
behavior.

## 1. What We Are Building

Think of the system as a careful memory librarian for an AI assistant.

When a source sends conversation context, the system does not simply put the
raw text into a vector database. It:

1. saves the original evidence without changing it;
2. derives small, source-linked memory facts from that evidence;
3. connects facts and entities in a graph;
4. records embeddings for semantic search later; and
5. remembers exactly which work succeeded, so a crash can be replayed safely.

“Agentic” here means controlled, multi-step decision-making with explicit
state and guardrails. It does **not** mean an unconstrained agent that can
rewrite history, choose arbitrary graph nodes, or silently delete memories.

The system has two parts:

| Part | State |
|---|---|
| Ingestion engine | Implemented through M8: accepts data, persists evidence, builds graph plans, writes graph data, creates fact embeddings, and records job state. |
| Retrieval/chat engine | Designed only. It will later search embeddings, traverse the graph, assemble evidence, and answer or abstain. |

LongMemEval is one input adapter used to evaluate the system. It is not the
core data model; future chat exports, documents, and connectors should enter
the same generic ingestion contract.

## 2. Big Picture: Where Data Lives

Two stores have deliberately different responsibilities.

```text
                         generic context record
                                   |
                     application ingestion pipeline
                         /                    \
                        /                      \
             PostgreSQL + pgvector         local HydraDB graph-node
             ---------------------         ------------------------
             exact raw chunks              Session / Turn / Fact
             content hashes                Entity / Alias
             extraction audit              provenance and entity edges
             graph-ID registry             graph traversal
             job / retry state             bitemporal fact properties
             fact embeddings
```

### Why not store everything in only one database?

- PostgreSQL is excellent for immutable evidence, transactions, audit rows,
  job state, and `pgvector` similarity search.
- HydraDB is excellent for graph-shaped relationships and traversals such as
  “find facts about this entity across sessions.”
- They cannot share one database transaction. The application therefore uses
  stable identities, manifests, job states, and replay instead of pretending a
  graph write and SQL write are atomic together.

This split is the central architecture decision (ADR-001 and ADR-007).

## 3. Main Terms

| Term | Plain meaning |
|---|---|
| `context_id` | Isolation boundary. Usually one user or tenant memory space. Every query and identity decision stays inside it. |
| `ContextBatch` | Versioned envelope containing records from one source ingestion attempt. |
| `ContextRecord` | One timestamped source item, such as one chat turn. |
| Chunk | Immutable stored copy of one accepted record in the MVP. No multi-turn chunking yet. |
| Episodic memory | Original timestamped experience: the raw chunk/turn. |
| Semantic memory | A concise fact derived from evidence, for example “Max is a golden retriever.” |
| Procedural memory | An instruction or workflow, for example “Run the backup before deployment.” |
| Graph plan | Validated, deterministic list of graph nodes and edges to write. |
| Embedding | A vector of numbers representing text meaning. Similar text tends to have nearby vectors. |
| Idempotent replay | Running the same unchanged work again gives the same result, not duplicate or mutated data. |

## 4. End-to-End Ingestion Flow

```text
1. source payload
      |
2. source adapter -> ContextBatch / ContextRecord
      |
3. contract validation
      |
4. immutable SQL chunk + pending_graph job
      |
5. extraction drafts -> validated memory candidates
      |
6. entity resolution (exact first; bounded LLM only if needed)
      |
7. GraphPlanBuilder -> nodes + relationships + integer IDs
      |
8. GraphWriter -> HydraDB HTTP UNWIND/MERGE batches
      |
9. local embedding model -> versioned pgvector rows
      |
10. verify local evidence and mark job completed
```

### Example input

```text
record_id: turn-17
session_id: session-3
occurred_at: 2026-08-19T10:00:00+00:00
role: user
content: "My dog Max is a golden retriever."
```

The raw text becomes one immutable chunk. An extraction result may produce the
semantic candidate:

```text
text: "Max is a golden retriever"
memory_type: semantic
source span: characters 7..33 in turn-17
entity: Max (pet)
```

The graph plan then contains:

```text
(Session)-[:HAS_TURN]->(Turn)
(Fact)-[:EXTRACTED_FROM]->(Turn)
(Fact)-[:ABOUT {surface: "Max"}]->(Entity)
```

The exact source chunk remains in PostgreSQL. The graph stores the fact,
provenance pointers, source span, time fields, and entity connection. The fact
text is embedded and its vector is stored in PostgreSQL/pgvector.

## 5. Step-by-Step: What Happens and Which Code Does It

| Step | What happens | Main files |
|---|---|---|
| 1. Adapt | Source-specific shape becomes generic input. LongMemEval timestamps are parsed, sessions are stable-sorted, empty turns are recorded, and evaluation labels are excluded. | `ingestion/sources/longmemeval.py` |
| 2. Validate | Required fields, timestamps, metadata, duplicate record IDs, content type, and forbidden evaluation fields are checked before persistence. | `core/models.py`, `core/validation.py` |
| 3. Persist evidence | One record becomes one deterministic chunk. SQL records its content hash and seeds a `pending_graph` job in the same SQL transaction. Changed content with the same logical chunk identity is rejected. | `ingestion/service.py`, `persistence/postgres.py::PostgresChunkStore` |
| 4. Extract candidates | Draft memories are converted to typed candidates. Their text spans must point into the exact raw source text; invalid drafts are stored as rejected audit data. | `ingestion/extraction.py` |
| 5. Resolve entities | Same-context canonical and alias matches are used first. Only an already-bounded candidate list can reach an LLM. If uncertain, no graph link is forced. | `core/resolution.py`, `ingestion/resolution.py`, `ingestion/model_adapters.py`, `ingestion/semantic_blocking.py` |
| 6. Plan graph | Valid candidates become `Session`, `Turn`, `Fact`, `Entity`, and `Alias` nodes plus provenance/entity edges. PostgreSQL allocates stable integer graph IDs. | `ingestion/graph_plan_builder.py`, `core/graph.py`, `persistence/postgres.py` |
| 7. Write graph | Immutable graph payload is registered first. Then legal grouped OpenCypher `UNWIND`/`MERGE` writes go to local HydraDB over HTTP. | `ingestion/graph_writer.py`, `client/hydradb_http.py`, `persistence/postgres.py::PostgresGraphManifestStore` |
| 8. Embed facts | Accepted fact text is encoded by a local sentence-transformers model. The normalized vector and model metadata become an immutable pgvector row. | `ingestion/embedding.py`, `persistence/postgres.py::PostgresEmbeddingStore` |
| 9. Track recovery | Orchestrator moves the job through legal states, marks failures, and safely reruns unchanged chunks from the beginning. | `ingestion/orchestrator.py`, `core/enums.py`, `persistence/postgres.py::PostgresJobStore` |

## 6. Source-Neutral Input and Immutable Evidence

`ContextBatch` and `ContextRecord` in `core/models.py` are deliberately
generic. Core code does not know LongMemEval field names. This prevents the
benchmark format from becoming a hidden dependency in storage, graph, or model
logic.

`chunk_from_record()` in `core/validation.py` makes the MVP boundary simple:
one accepted record equals one raw chunk. The chunk retains:

- the original text;
- content hash;
- source record/session/actor identifiers;
- source timestamp and metadata; and
- its `context_id`.

If a caller retries the same payload, the same content hash is accepted as a
no-op. If it claims to replay a record but supplies different text, SQL raises
an immutable-record conflict instead of overwriting evidence. This makes later
debugging possible: every fact can still point back to the exact text that
supported it.

## 7. Extraction: Current Implementation vs Future LLM Extraction

### What exists today

`ExtractionService` accepts only `deterministic-fixture v1` output. This is a
testable baseline, not a claim of production extraction quality.

For each draft, it creates an `ExtractedMemoryCandidate` only when all of the
following are valid:

- non-empty fact text;
- valid memory type and allowed scope;
- source offsets selecting a non-empty part of the original record;
- confidence between 0 and 1;
- source timestamp copied into `observed_at`;
- valid world-time bounds, when present; and
- entity surfaces, when supplied.

Raw chunks are episodic memory. Derived candidates are semantic or procedural;
the code intentionally rejects `episodic` as an extracted candidate because
the immutable raw chunk already is the episode.

Both accepted and rejected output are recorded by
`PostgresExtractionStore`. Rejections do not corrupt the raw chunk or stop an
otherwise valid chunk from being auditable.

### Where an LLM is implemented

`core/llm_client.py::LLMClient` is a provider-neutral OpenAI-compatible
wrapper. It sends a system prompt plus user prompt, requests JSON-object
output at temperature `0.0`, then validates the response with Pydantic.

The two implemented LLM adapter call sites are:

| LLM task | Module | Input boundary | Safe output |
|---|---|---|---|
| Entity ambiguity | `LLMEntityResolutionModel.resolve_entity` | One surface form plus a fixed list of same-context candidate entities | One supplied integer graph ID, or `null` |
| Correction vs state change | `LLMTemporalUpdateModel.classify_update` | Two facts that already passed subject, predicate, and chronology checks | `correction`, `state_change`, `no_update`, or `unresolved` |

The LLM cannot invent a new entity ID. `EntityRegistry` rejects a returned ID
outside the supplied shortlist. A malformed, failing, or abstaining model call
becomes unresolved rather than a forced link.

Important current limit: the orchestrator uses deterministic extraction only.
It does **not** yet call a real extractor LLM to create fact candidates. The
LLM adapters above are usable behind their ports, but the temporal update path
is not connected to graph supersession yet; see section 9.

## 8. Entity Resolution: Avoiding “Max” Matching the Wrong Max

Entity resolution happens only within `context_id`.

```text
"my dog" appears in new fact
       |
normalize Unicode, spaces, and case
       |
exact canonical match? ---- yes -> use entity
       |
exact alias match? -------- yes -> use entity
       |
bounded candidate list? --- yes -> ask LLM to choose one or abstain
       |
no candidates -----------------> allocate new entity ID
```

`canonicalize_entity_surface()` performs Unicode NFKC normalization, trims and
collapses whitespace, then case-folds. For example, `" Max "` and `"max"`
become the same comparison value.

`SemanticBlockingCache` can supply semantically close *candidate IDs* using a
process-local vector cache. It is not a source of truth: it only improves the
shortlist. Exact/alias rules and the bounded LLM decision remain authoritative.

## 9. How the Graph Is Made

`GraphPlanBuilder` turns a chunk and its accepted candidates into a
`GraphWritePlan`. This is a clean separation:

- extraction decides which candidates are valid;
- resolution decides whether an entity link is safe;
- graph planning describes graph records;
- graph writing only serializes that plan into legal HydraDB mutations.

### Nodes created now

| Node | Meaning | Typical identity key |
|---|---|---|
| `Session` | Source conversation/session, or a context fallback bucket | `session:<session key>` |
| `Turn` | One immutable raw chunk | `turn:<chunk id>` |
| `Fact` | One accepted semantic/procedural candidate | `fact:<candidate id>` |
| `Entity` | Canonical resolved person/place/pet/etc. | `entity:<canonical name>` |
| `Alias` | Alternative entity name | `alias:<alias>:<entity ID>` |

### Edges created now

| Edge | Meaning |
|---|---|
| `HAS_TURN` | Session contains a turn. |
| `EXTRACTED_FROM` | Fact is backed by this turn/chunk. |
| `ABOUT` | Fact describes this resolved entity. |
| `HAS_ALIAS` | Entity has this alternative name. |

Every graph record includes `context_id`. This keeps graph traversal and
identity isolated between users/tenants.

### Why PostgreSQL allocates IDs

HydraDB graph node/edge IDs must be non-negative integers. The SQL
`graph_id_registry` maps `(kind, context_id, logical_key)` to one stable
integer. It combines two useful properties:

- stable logical names make replays deterministic; and
- a database registry detects a collision/conflict instead of silently
  accepting a hash collision.

### How write safety works

Before HydraDB receives data, `PostgresGraphManifestStore` stores a hash of
each logical graph payload. Same payload + same logical key is a safe replay.
Changed payload + same logical key is rejected before graph mutation.

`GraphWriter` groups compatible records and writes statements shaped like:

```cypher
UNWIND $rows AS row
MERGE (n {id: row.id})
SET n:Fact, ...
```

and, for an edge:

```cypher
UNWIND $rows AS row
MATCH (s:Fact {id: row.source_id}), (d:Turn {id: row.destination_id})
MERGE (s)-[r:EXTRACTED_FROM {id: row.id}]->(d)
SET ...
```

`HydraHttpTransport` sends these to the checked-in local `graph-node` HTTP
query endpoint. HTTP `query_id` helps identify a request but is not treated as
durable server-side idempotency; the SQL manifest plus `MERGE` identities are
the replay guarantee.

## 10. Bitemporal Memory: Two Different Meanings of Time

Normal timestamps are insufficient for memories because “when the system
heard something” and “when it was true in the world” can differ.

| Time axis | Fields | Question answered |
|---|---|---|
| Knowledge time | `observed_at`, `superseded_at` | When did the system know or stop preferring this assertion? |
| World-validity time | `valid_from`, `valid_to` | When was this assertion true in the described world? |

Example:

```text
April 1: user says "I work at Acme."
May 1: user says "Actually, I have worked at Beta since January."
```

If the May statement corrects a previous misunderstanding, the system learned
the correction on May 1. It must close the old assertion’s **knowledge time**
at May 1, but it must not automatically claim Acme employment ended on May 1.

If instead the user says “I left Acme and joined Beta on May 1,” that is a
world state change. The old assertion can close its knowledge time at May 1 and
its world-validity time at the supported effective date.

`TemporalUpdateClassifier` enforces guards before any model call:

1. new fact must be observed later;
2. both facts must have the same resolved subject entity; and
3. both facts must have the same `predicate_key`.

Only then can an LLM classify correction/state-change/no-update/uncertain.

### Current bitemporal implementation limit

The data types and classifier implement the above decision rule. However,
current deterministic extraction does not emit `predicate_key`. Therefore
`GraphPlanBuilder` intentionally does not write `SUPERSEDES` edges or mutate
older fact time fields yet. That is an explicit open gap, documented in ADR-030
— not an accidental omission. A future approved extraction milestone must add
subject/predicate structure before graph-side bitemporal update chains can be
enabled safely.

## 11. Embeddings: How They Are Created and Why Retrieval Needs Them

An embedding is a compact list of numbers that represents the meaning of text.
Two sentences with related meaning often have nearby vectors, even when their
exact words differ.

For each accepted fact, `SentenceTransformerEmbedder`:

1. lazily loads local `sentence-transformers/all-MiniLM-L6-v2` on CPU;
2. encodes the fact text;
3. normalizes the vector; and
4. returns 384 floating-point values.

The orchestrator writes this as a `memory_embeddings` row with:

- `context_id` and fact identifier;
- source chunk identifier;
- model name and version;
- dimension count;
- hash of the embedded fact text; and
- active/inactive status.

The embedding store refuses to silently replace a vector produced for the same
fact/model/version from different text. A future model change creates a new
model version instead of destroying earlier vectors.

### How embeddings will help retrieval later

Retrieval is not implemented yet, but its intended sequence is:

```text
user question
   -> encode question with compatible embedding model
   -> pgvector cosine search, scoped by context_id + model version
   -> get semantically relevant fact IDs
   -> use facts as seeds for HydraDB graph traversal
   -> inspect provenance spans/chunks and time filters
   -> rank evidence, answer, or abstain
```

Example: question “What breed is Max?” can seed the fact “Max is a golden
retriever” even if the question and fact use different word order. The graph
then adds structured evidence: which entity “Max” means, which turn supports
the fact, and eventually which newer facts supersede it.

Embeddings find *meaningfully similar starting points*. The graph supplies
relationships and history. Source chunks supply proof. None alone is enough.

## 12. Recovery and Job States

Because SQL and HydraDB cannot commit as one atomic unit, every chunk has an
ingestion job.

```text
pending_graph
    -> pending_embeddings
    -> verifying
    -> completed

at any non-terminal step:
    -> retryable_failed | terminal_failed | manual_repair
```

- `retryable_failed`: transient/unclassified problem, such as a network
  failure. Safe automatic/manual replay is allowed.
- `terminal_failed`: invalid or conflicting payload. Retrying unchanged input
  cannot fix it.
- `manual_repair`: requires a human decision.
- `completed`: terminal; it is not processed again.

Current M8 strategy is **whole-chunk replay**. On retry, it re-runs extraction,
resolution, graph plan construction, graph write, and embedding persistence.
This can waste work, especially model calls, but is correct because each stage
rejects changed payloads and accepts an unchanged replay. Fine-grained resume
needs read-back ports for stored extraction/embedding results and is future
work.

Current verification before `completed` re-reads the SQL chunk and confirms
its content hash. It is not yet an independent re-read of HydraDB and pgvector;
a future reconciliation/audit step can strengthen that evidence.

## 13. Code Map

```text
src/context_memory/
  core/
    models.py              typed input, chunk, candidate, embedding, job models
    validation.py          hashes, chunk construction, contract checks
    graph.py               scalar-only graph plan/node/edge types
    resolution.py          entity normalization and bitemporal decision types
    enums.py               memory and job-state vocabularies
    llm_client.py          provider-neutral structured LLM wrapper
    config.py              explicit environment-driven provider configuration

  ingestion/
    service.py             basic batch -> immutable chunk boundary
    extraction.py          deterministic extraction baseline and audit shape
    resolution.py          bounded entity/update decision logic
    model_adapters.py      real LLM implementations of bounded ports
    semantic_blocking.py   non-authoritative entity-name candidate cache
    graph_plan_builder.py  candidate/entity decisions -> graph records
    graph_writer.py        graph plan -> legal grouped HydraDB mutations
    embedding.py           local MiniLM fact embedding implementation
    orchestrator.py        full chunk workflow, jobs, retry classification
    ports.py               interfaces separating policy from infrastructure
    fakes.py               deterministic test doubles; no paid CI calls
    sources/longmemeval.py LongMemEval-specific mapping only

  persistence/
    postgres.py            chunks, extraction audit, IDs, manifests, vectors, jobs
    migrations.py          forward-only migration discovery/application

  client/
    hydradb_http.py        HTTP client for checked-in local graph-node

src/tests/
  test_*.py                contract, graph, embedding, adapter, and recovery tests
```

## 14. Important Non-Claims and Next Work

The following are intentionally not claimed as complete:

- real LLM **fact extraction** is still gated; current extraction is a
  deterministic fixture baseline;
- graph `SUPERSEDES` update chains are deferred until candidates include a
  reliable `predicate_key`;
- local HydraDB HTTP graph writes have deterministic/manifest coverage, but
  require a live graph-node runtime and token for end-to-end live verification;
- retrieval, answer generation, temporal query interpretation, and chat API
  are Milestone 9+ work, requiring their own approval;
- whole-chunk retries are correct but not cost-optimal;
- organization-scope promotion requires separate authorization policy.

This separation is deliberate: it keeps the current ingestion foundation
auditable and safe while future retrieval quality work is added incrementally.
