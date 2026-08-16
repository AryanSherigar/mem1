# Agent Memory Engineering Instructions

This document provides strict, programmatic constraints and reference material for agents working on the HydraDB LongMemEval codebase. You MUST adhere to these constraints to ensure correct benchmark evaluation and database compatibility.

## 1. Codebase Structure

We are implementing the system using a modular, atomic file architecture suited for production scalability:

```text
src/
├── core/
│   ├── llm_client.py         # Universal LLMClient class with structured output support
│   ├── id_generator.py       # Content-addressable UUID generator
│   └── config.py             # Environment configuration and model allocation
├── db/
│   ├── graph_client.py       # HydraDB Bolt Cypher writer and connections
│   └── embedding_index.py    # In-memory EmbeddingIndex and EntityNameIndex (numpy)
├── memory/
│   ├── engine.py             # Core Product: add_turn_async(), search_memories(), generate_reply(), hydrate()
│   ├── retrieval.py          # Temporal pruning, semantic seeding, algo.MSpaths
│   └── distillation.py       # Semantic memory distillation and SUPERSEDES updates
├── entities/
│   ├── resolver.py           # Canonical matching and MERGED_INTO logic
│   └── semantic_blocking.py  # Tier 2 semantic blocking and Tier 3 LLM disambiguation
├── chat/
│   └── interactive_chat.py   # CLI chat interface testing persistence
├── api/
│   ├── server.py             # FastAPI application and uvicorn runner
│   └── routes.py             # HTTP REST endpoints (POST /memory, GET /search)
└── evaluation/
    └── benchmark_runner.py   # Evaluator over longmemeval_s_cleaned.json -> predictions.jsonl
```

## 2. HydraDB Cypher Constraints (Bolt 5.x)

You MUST NOT use the following Cypher features, as HydraDB does not support them:
- `IN [list]`, `ENDS WITH`, `CONTAINS`, `IS NULL` in `WHERE` clauses.
- `RETURN *`
- Aggregation functions `min()` and `max()`.
- `DISTINCT` inside aggregate arguments.
- `WITH` aliasing or filtering.
- Unbounded variable-length paths (e.g., `*` or `*1..`). MUST specify max (e.g., `*1..5`).
- `ON CREATE` / `ON MATCH` inside `MERGE`.
- Storing Float Arrays (Embeddings MUST live in `src/db/embedding_index.py`'s `EmbeddingIndex`).

## 3. ID Generation

IDs are generated as UUID strings derived deterministically from a SHA-256 hash of the unique semantic path. This guarantees perfect reproducibility without ever hitting an artificial ceiling or relying on sequential state.

Implementation snippet for ID generation:
```python
import hashlib
import uuid

def content_hash_uuid(semantic_path: str) -> str:
    """Generate a deterministic UUID format string from a semantic path."""
    digest = hashlib.sha256(semantic_path.encode("utf-8")).digest()
    return str(uuid.UUID(bytes=digest[:16]))
```

## 4. Entity Types (Strict Enum)

The LLM extraction prompt MUST enforce this exact list of string literals for `entity_type`:
`"person"`, `"pet"`, `"place"`, `"organization"`, `"event"`, `"creative_work"`, `"product"`, `"activity"`, `"preference"`, `"topic"`, `"other"`.

## 5. Pydantic Schemas

Use these precise schemas when invoking the unified `LLMClient.structured_completion()`:

**Extraction:**
```python
class ExtractedMemory(BaseModel):
    fact: str
    entity_name: str
    entity_type: str
    action: Literal["ADD", "UPDATE", "DELETE"]
    target_fact_id: int | None = None

class MemoryExtractionPayload(BaseModel):
    memories: list[ExtractedMemory]
```

**Temporal Inference:**
```python
class TimeRangeInference(BaseModel):
    start: str | None = Field(default=None, description="YYYY/MM/DD or None")
    end: str | None = Field(default=None, description="YYYY/MM/DD or None")
```

**Entity Resolution:**
```python
class EntityResolutionDecision(BaseModel):
    mention_index: int
    decision: Literal["MATCH", "NEW"]
    matched_entity_id: int | None = None
    confidence: Literal["high", "medium", "low"]
    reasoning: str

class EntityResolutionBatch(BaseModel):
    decisions: list[EntityResolutionDecision]
```

## 6. Retrieval Algorithm Constants

When implementing `search_memories()` in `src/memory/engine.py`, strictly adhere to these constants:
- **Temporal Buffer**: `±2 days` (172,800 seconds) padding applied to inferred `[start, end]`.
- **Semantic Seeding**: `top_k = 10` (return top 10 facts from `EmbeddingIndex`).
- **Semantic/Structural Weights**: `SEMANTIC = 0.6`, `STRUCTURAL = 0.4`.
- **Structural Score Formula**: `(1.0 / hop_count) * min(path_count, 3) / 3.0`
- **Context Size**: Pass `top_n = 15` facts to the Reader LLM.
- **Abstention Cutoff**: If `semantic_score < 0.25` AND `len(expanded_facts) == 0` -> Immediately return `"I don't have that information in my memory."`

## 7. Model Allocation (`.env`)

```bash
EXTRACTOR_MODEL="gpt-oss-20b"
READER_MODEL="gpt-oss-120b"
JUDGE_MODEL="meta-llama/llama-3.3-70b-instruct"
```
Ensure `LLMClient` uses standard OpenAI JSON object mode: `response_format={"type": "json_object"}`.
