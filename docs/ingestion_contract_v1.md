# Generic Context Ingestion Contract v1

| Field | Value |
|---|---|
| Status | Accepted; deterministic domain reference implemented in Milestone 1 |
| Contract version | `v1` |
| Approved | 2026-08-16 |
| Decision records | [ADR-012 through ADR-018](decisions.md) |
| Roadmap | [ingestion_pipeline_roadmap.md](ingestion_pipeline_roadmap.md) |

## 1. Boundary

The core accepts a versioned generic envelope. A source adapter parses its
payload and emits this envelope. Core never imports source schemas, interprets
benchmark labels, or branches on source-specific field names.

```text
source payload -> source adapter -> IngestionService.ingest(batch)
             -> generic validation/chunking/persistence/extraction/graph/recovery
```

`IngestionService` is an in-process application interface in MVP. No HTTP
ingestion endpoint is authorized. LongMemEval uses a dedicated CLI/script in
Milestone 3; that script invokes the same service.

Milestone 3 provides the shared `IngestionService`, LongMemEval adapter, and
local CLI. It does not provide graph writes, extraction, embeddings, or a model
provider.

## 2. Canonical Envelope

### ContextBatch

| Field | Type | Required | Rule |
|---|---|---:|---|
| `contract_version` | string | yes | Exactly `v1` |
| `ingestion_id` | string | yes | Caller retry/idempotency identity for this batch |
| `context_id` | string | yes | Active isolation boundary; non-empty and stable for replay |
| `source_type` | string | yes | Adapter/source name, for example `longmemeval` or `chat_export` |
| `source_external_id` | string | yes | Stable source-side batch identity |
| `records` | array of `ContextRecord` | yes | Non-empty; ordered as supplied by adapter |
| `metadata` | object | no | JSON-compatible source metadata; no evaluation labels |

### ContextRecord

| Field | Type | Required | Rule |
|---|---|---:|---|
| `record_id` | string | yes | Stable within `context_id`; source id or deterministic adapter id |
| `session_id` | string | no | Source conversation/session identifier |
| `actor_id` | string | no | Source participant identity when available |
| `actor_role` | string | no | Source role such as `user`, `assistant`, `system`, or `unknown` |
| `occurred_at` | RFC 3339 timestamp | yes | Time observed in source; offsets required unless source policy approves UTC normalization |
| `content_type` | string | yes | MVP accepts `text/plain` only |
| `content` | string | yes | Exact source text; cannot be silently normalized or rewritten |
| `metadata` | object | no | JSON-compatible record metadata, excluding evaluation-only fields |

### Invariants

1. `(context_id, record_id)` identifies one immutable logical source record.
2. A replay with identical canonical content is idempotent. Same logical record
   with different canonical content is a conflict, never an overwrite.
3. `records` retain adapter order. Adapters requiring chronological order must
   establish it before invoking core and preserve source ordering metadata.
4. Empty `content`, unsupported `content_type`, unknown contract version,
   malformed timestamp, duplicate `record_id`, and non-object metadata fail
   validation before persistence.
5. `question_id`, `question_type`, `has_answer`, `_abs`, expected answers, and
   answer-session labels remain in evaluation code. They are not v1 fields.

## 3. Derived Memory Candidate Contract

This is not source input. It is the validated output shape required from a
future extractor before graph or embedding work. Raw records remain episodic
evidence even when extraction produces no candidate.

| Field | Type | Required | Rule |
|---|---|---:|---|
| `candidate_id` | string | yes | Stable only within one extraction attempt |
| `text` | string | yes | Atomic claim/instruction; non-empty |
| `memory_type` | enum | yes | `episodic`, `semantic`, or `procedural` |
| `scope_type` | enum | yes | `chat`, `session`, or `user`; `organization` rejected in MVP |
| `scope_id` | string | yes | Matches the selected scope owner |
| `source_record_id` | string | yes | Record providing direct evidence; later resolved to chunk ID |
| `source_start` / `source_end` | integer | yes | Zero-based, end-exclusive Unicode code-point offsets into exact source `content` |
| `confidence` | number | yes | Inclusive range `0.0..1.0`; does not bypass validation/policy |
| `observed_at` | timestamp | yes | Copied from source record; extractor cannot invent it |
| `valid_from` / `valid_to` | timestamp or null | no | World-validity claim; `valid_from <= valid_to` when both supplied |
| `entities` | array | no | Candidate entity surfaces; resolution occurs later |

Validation rules:

1. `source_start < source_end <= len(source.content)` and the selected span
   must be non-empty.
2. `episodic` candidates preserve source experience. Extracted factual claims
   default to `semantic`; explicit repeatable instructions/workflows may be
   `procedural`.
3. `scope_type = organization` is rejected until an approved authorization and
   promotion policy exists.
4. `observed_at` is knowledge-availability time. `valid_from`/`valid_to`, when
   supported by evidence, describe world-validity time. A correction does not
   automatically rewrite world-validity fields.
5. Invalid candidates are retained only as diagnostic/rejected extraction output
   when persistence is implemented; they never create graph facts.

### 3.1 Entity and Temporal Interpretation (M5)

Entity resolution and update classification are a later interpretation step, separate from immutable evidence and extraction output. It receives the candidate, entity surfaces, and explicit fixture/provider hints such as `predicate_key`; it does not rewrite the candidate or source chunk.

- Canonical entity matching is Unicode `NFKC`, trimmed/collapsed whitespace, and case-folded within one `context_id`.
- Exact canonical/alias matches do not need an LLM.
- An LLM may choose only from a bounded, application-supplied candidate set in the same context. An invalid selection or abstention is `unresolved`, never a new cross-context identity link.
- An LLM may classify `correction`, `state_change`, `no_update`, or `unresolved` only after equal subject entity, equal explicit `predicate_key`, and newer `observed_at` are established.
- `correction` closes knowledge time only. `state_change` may close world validity at an already-supported incoming `valid_from` (or observation time); the LLM does not invent timestamps.

## 4. Approved Storage and Identity Shape

MVP chunk unit: exactly one accepted `ContextRecord` becomes one immutable raw
chunk. No multi-turn merging or text splitting in MVP. Chunk identifier is a
stable textual logical key derived from `context_id`, `record_id`, and canonical
content identity; its final encoded form is implementation detail, but the raw
content hash must be persisted and checked on replay.

HydraDB node IDs require non-negative integers. PostgreSQL therefore owns a
durable global graph-ID registry. Registry key is `(node_kind, context_id,
logical_key)`; allocation returns one stable non-negative integer for every
logical graph node. This avoids hash-collision acceptance and permits safe
replay across processes.

Embedding target: validated fact text only. Raw chunks remain durable in SQL;
chunk embeddings are a held benchmark experiment, not an MVP write requirement.

Local development and CI target PostgreSQL 16 with pgvector through a
containerized service. Milestone 2 implements `evidence_chunks`,
`graph_id_registry`, `ingestion_jobs`, and future-ready `memory_embeddings`
through forward-only checksum-verified SQL migrations. Every persistence read
and replay lookup carries `context_id`; chunk IDs alone are not an authorization
boundary.

## 5. Cross-Store Job State Contract

One ingestion job exists per immutable chunk. PostgreSQL inserts the chunk and
its initial job state in one SQL transaction. Local HydraDB graph writes and
embedding writes occur afterward and are verified before completion. PostgreSQL
allocates graph IDs; every graph record carries a SQL chunk/provenance reference.

| State | Meaning | Allowed next state |
|---|---|---|
| `pending_graph` | SQL chunk/job durable; graph write not yet verified | `pending_embeddings`, `retryable_failed`, `terminal_failed`, `manual_repair` |
| `pending_embeddings` | Graph write verified; fact embeddings not yet verified | `verifying`, `retryable_failed`, `terminal_failed`, `manual_repair` |
| `verifying` | SQL, graph, and embedding provenance/count checks running | `completed`, `retryable_failed`, `terminal_failed`, `manual_repair` |
| `completed` | All required records verified | none |
| `retryable_failed` | Safe automatic/manual replay permitted | prior non-terminal state, `terminal_failed`, `manual_repair` |
| `terminal_failed` | Invalid or policy-forbidden payload; no automatic retry | `manual_repair` only |
| `manual_repair` | Human intervention required | approved non-terminal state or `terminal_failed` |

Attempt history, error class, and last verified stage are required when the job
model is implemented. No state implies a distributed transaction.

## 6. Golden Fixtures

The following fixture pair is normative for v1 shape validation:

- [generic_context_batch_v1.json](fixtures/generic_context_batch_v1.json):
  direct generic input with episodic fact, preference, procedure, correction,
  relative-time phrase, alias, and duplicate text from different records.
- [longmemeval_adapter_expected_context_batch_v1.json](fixtures/longmemeval_adapter_expected_context_batch_v1.json):
  expected canonical output for a minimal LongMemEval-style input. It proves
  benchmark labels are excluded and adapter output has the same v1 shape.

The raw LongMemEval-style input is documented inside the expected fixture under
`adapter_fixture_note`; a runnable minimal raw fixture also lives at
[longmemeval_adapter_input_v1.json](fixtures/longmemeval_adapter_input_v1.json).
Neither is a runtime benchmark dataset.

### LongMemEval Source-Normalization Policy

Inspection of the local `longmemeval_s_cleaned.json` found 500 instances,
23,867 session timestamps, and 500 question timestamps. All 24,367 values use
`%Y/%m/%d (%a) %H:%M`: minute precision, no timezone. Session arrays are not
chronological; 3,167 adjacent date inversions occur. Thirteen instances repeat
a session ID (at most twice), and 12 turns have empty content.

The LongMemEval adapter therefore:

1. parses the custom timestamp and attaches UTC only as a benchmark-local
   ordering basis, never as a claim about source timezone;
2. preserves raw timestamp, `source_timezone: "unknown"`, minute precision,
   and source array index in record metadata;
3. stable-sorts sessions by parsed timestamp while preserving turn order;
4. qualifies canonical session/record IDs with source array index, while
   preserving the raw session ID in metadata;
5. skips empty-text turns and stores `source_empty_turn_count` in batch
   metadata. Generic v1 still rejects empty records from other callers.

## 7. Deferred Items

- public HTTP ingestion endpoint;
- non-text content types;
- multi-turn/window chunking;
- chunk embeddings;
- retention, deletion, encryption, authorization, and organization scope for
  non-benchmark data;
- real extraction/embedding model selection.

Any change to required v1 fields requires a new contract version and approved
ADR. Additive optional metadata is allowed only when it cannot affect core
meaning, idempotency, scope, or evaluation integrity.
