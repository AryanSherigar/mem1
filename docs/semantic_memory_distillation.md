# Semantic Memory Distillation — LLM Extraction & Ingestion Pipeline

Follow-up to [graph_schema_proposal.md](file:///home/aryan-sherigar/projects/hydradb-hackathon/docs/graph_schema_proposal.md) and [retrieval_architecture.md](file:///home/aryan-sherigar/projects/hydradb-hackathon/docs/retrieval_architecture.md).

---

## 1. Overview & Architectural Motivation

At its core, semantic memory distillation adopts an **LLM-as-a-function** pattern (inspired by Mem0) to transform unstructured conversational turns into structured memory units. Instead of relying on brittle heuristic NLP or static regular expressions, a fast extraction model extracts atomic facts, classifies action types (`ADD`, `UPDATE`, `DELETE`), resolves affected entities, and writes them into our dual-storage architecture:
1. **HydraDB Knowledge Graph**: Stores discrete entities, facts, speakers, temporal validity (`valid_from`, `valid_to`, `is_current`), and provenance relationships (`EXTRACTED_FROM`, `ABOUT`, `STATED_BY`, `SUPERSEDES`).
2. **In-Process Embedding Matrix**: Stores normalized 384-dimensional vector embeddings for sub-millisecond semantic seeding.

Unlike a pure flat vector database (which struggles with multi-hop relationships, temporal reasoning, and contradiction invalidation), this distillation pipeline populates a **hybrid temporal graph** that natively supports LongMemEval's complex retrieval tasks.

---

## 2. End-to-End Distillation Pipeline

```mermaid
flowchart TD
    UserTurn[User / Assistant Message Turn] --> MainAgent[Immediate Main Agent Response]
    UserTurn -->|Async Background Task| CandidateLookup[1. Semantic Candidate Lookup<br/>EmbeddingIndex.top_k]
    CandidateLookup --> ExtractorLLM[2. Extractor LLM<br/>Structured JSON / Pydantic]
    
    ExtractorLLM --> ActionRouter{Action Classifier}
    
    ActionRouter -->|ADD| CreateFact[3a. HydraDB CREATE Fact + ABOUT + STATED_BY<br/>+ EmbeddingIndex.add]
    ActionRouter -->|UPDATE| SupersedeFact[3b. HydraDB CREATE Fact -[:SUPERSEDES]-> OldFact<br/>SET old.is_current=false, old.valid_to=epoch<br/>+ EmbeddingIndex.replace]
    ActionRouter -->|DELETE| InvalidateFact[3c. HydraDB SET fact.is_current=false<br/>+ EmbeddingIndex.remove]
```

The pipeline operates in five distinct phases:

### Phase 1: Context & Candidate Memory Retrieval
Before invoking the extraction LLM, the turn text is embedded using `all-MiniLM-L6-v2`. We query the in-process `EmbeddingIndex` for the top-$k$ ($k \approx 5$) most semantically relevant existing facts scoped to the current `haystack_id`. These candidates are injected into the extractor prompt so the LLM can identify updates or invalidations of prior knowledge.

### Phase 2: The Extractor Prompt & Structured Output
The extractor is invoked with a strict system prompt and enforced Pydantic schema. It strips out conversational filler and temporary banter, isolating concrete facts, preferences, user constraints, or entity state updates.

### Phase 3: Action Classification (`ADD` / `UPDATE` / `DELETE`)
Every extracted memory is assigned an operational intent:
* **`ADD`**: A novel fact or entity relationship has been established.
* **`UPDATE`**: An existing fact is modified or corrected (e.g., *"Actually, I switched to Go for scripting"*). Points to the `superseded_fact_id`.
* **`DELETE`**: An existing fact is explicitly negated or forgotten. Points to the `target_fact_id`.

### Phase 4: Dual-Store Ingestion (HydraDB + Embedding Index)
Mutations are applied atomically to both HydraDB (via Bolt 5.x Cypher) and the in-process `EmbeddingIndex`.

### Phase 5: Asynchronous Execution
Memory extraction runs off the critical path in background workers / async tasks, ensuring user interactions experience zero memory ingestion latency.

---

## 3. Pydantic Schemas & Data Contracts

```python
from typing import Literal
from pydantic import BaseModel, Field


class ExtractedMemory(BaseModel):
    """Atomic memory unit extracted from conversational turn."""
    fact: str = Field(
        description="Self-contained, atomic statement of fact, preference, or event."
    )
    entity_name: str = Field(
        description="Canonical name of the primary subject entity (e.g. 'Max', 'Python', 'Berlin')."
    )
    entity_type: str = Field(
        description="Type category: 'person', 'pet', 'preference', 'topic', 'location', etc."
    )
    action: Literal["ADD", "UPDATE", "DELETE"] = Field(
        description="Operational action for memory management."
    )
    target_fact_id: int | None = Field(
        default=None,
        description="HydraDB ID of existing fact being updated or deleted (from provided candidates)."
    )


class MemoryExtractionPayload(BaseModel):
    """Full extraction response from extractor LLM."""
    memories: list[ExtractedMemory]
```

---

## 4. HydraDB Cypher Mutation Mappings

All graph writes adhere strictly to HydraDB's supported Cypher subset ([cypher-compat.md](file:///home/aryan-sherigar/projects/hydradb-hackathon/hydradb/cypher-compat.md)): integer node IDs, scalar properties, single statement per request.

### 4.1 `ADD` Action

1. **HydraDB Graph Write**:
```cypher
CREATE (f:Fact {
  id: $fact_id,
  haystack_id: $haystack_id,
  text: $fact_text,
  speaker: $speaker,
  session_id: $session_id,
  valid_from: $timestamp,
  valid_to: 9999999999,
  created_at: $created_at,
  is_current: true
})-[:EXTRACTED_FROM]->(t:Turn {id: $turn_id, haystack_id: $haystack_id})
```

2. **Entity Linkage**:
```cypher
MATCH (f:Fact {id: $fact_id, haystack_id: $haystack_id}), (e:Entity {id: $entity_id, haystack_id: $haystack_id})
CREATE (f)-[:ABOUT]->(e)
```

3. **Embedding Update**:
```python
embedding_index.add(fact_id=fact_id, text=fact_text, haystack_id=haystack_id)
```

---

### 4.2 `UPDATE` Action (Knowledge Update $\rightarrow$ `SUPERSEDES`)

When a user updates an existing fact (e.g., replacing an old preference or state):

1. **HydraDB Graph Write (Create new fact with `SUPERSEDES` edge)**:
```cypher
CREATE (f_new:Fact {
  id: $new_fact_id,
  haystack_id: $haystack_id,
  text: $fact_text,
  speaker: $speaker,
  session_id: $session_id,
  valid_from: $timestamp,
  valid_to: 9999999999,
  created_at: $created_at,
  is_current: true
})-[:SUPERSEDES]->(f_old:Fact {id: $target_fact_id, haystack_id: $haystack_id})
```

2. **HydraDB Invalidate Old Fact**:
```cypher
MATCH (f_old:Fact {id: $target_fact_id, haystack_id: $haystack_id})
SET f_old.is_current = false
SET f_old.valid_to = $timestamp
```

3. **Embedding Synchronization**:
```python
embedding_index.remove(target_fact_id)
embedding_index.add(fact_id=new_fact_id, text=fact_text, haystack_id=haystack_id)
```

> [!NOTE]
> Setting `valid_to` and creating the `SUPERSEDES` edge allows temporal reasoning queries to reconstruct past states while ensuring semantic similarity searches exclusively retrieve `is_current: true` facts.

---

### 4.3 `DELETE` Action

1. **HydraDB Invalidation**:
```cypher
MATCH (f:Fact {id: $target_fact_id, haystack_id: $haystack_id})
SET f.is_current = false
SET f.valid_to = $timestamp
```

2. **Embedding Removal**:
```python
embedding_index.remove(target_fact_id)
```

---

## 5. Dual-Store Synchronization & Consistency

Because HydraDB does not store vector arrays as native properties, the in-process `EmbeddingIndex` (defined in [retrieval_architecture.md](file:///home/aryan-sherigar/projects/hydradb-hackathon/docs/retrieval_architecture.md)) acts as the semantic acceleration layer, while HydraDB holds the ground-truth relational and temporal graph.

| Event | In-Process `EmbeddingIndex` | HydraDB Graph Store |
|---|---|---|
| **Add Fact** | `EmbeddingIndex.add(id, text, hid)` | `CREATE (f:Fact {is_current: true})` |
| **Update Fact** | `EmbeddingIndex.remove(old_id)`<br/>`EmbeddingIndex.add(new_id, text, hid)` | `CREATE (f_new)-[:SUPERSEDES]->(f_old)`<br/>`SET f_old.is_current = false` |
| **Delete Fact** | `EmbeddingIndex.remove(id)` | `SET f.is_current = false` |

Both updates are packaged into a single transactional Python helper function `ingest_memory_action(...)` to prevent drift.

---

## 6. Asynchronous Background Execution Blueprint

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from neo4j import GraphDatabase

executor = ThreadPoolExecutor(max_workers=4)


class AsyncMemoryDistillationWorker:
    def __init__(self, hydra_driver, embedding_index, extractor_llm, id_gen):
        self.driver = hydra_driver
        self.embedding_index = embedding_index
        self.llm = extractor_llm
        self.id_gen = id_gen

    async def enqueue_turn_distillation(self, turn_data: dict):
        """Non-blocking fire-and-forget ingestion task."""
        loop = asyncio.get_running_loop()
        loop.run_in_executor(executor, self._process_turn_sync, turn_data)

    def _process_turn_sync(self, turn: dict):
        # 1. Retrieve candidate facts from in-memory index
        candidates = self.embedding_index.search(
            query_text=turn["content"],
            haystack_id=turn["haystack_id"],
            top_k=5
        )
        
        # 2. Extract facts via structured LLM prompt
        extraction: MemoryExtractionPayload = self.llm.extract(
            content=turn["content"],
            speaker=turn["role"],
            candidates=candidates
        )
        
        # 3. Apply graph & vector mutations
        for mem in extraction.memories:
            self._apply_memory_action(mem, turn)

    def _apply_memory_action(self, mem: ExtractedMemory, turn: dict):
        hid = turn["haystack_id"]
        ts = turn["timestamp"]
        
        if mem.action == "ADD":
            fact_id = self.id_gen.next_id()
            entity_id = self._get_or_create_entity(mem.entity_name, mem.entity_type, hid)
            
            # HydraDB Cypher write
            with self.driver.session() as session:
                session.run(
                    "CREATE (f:Fact {id: $fid, haystack_id: $hid, text: $txt, "
                    "speaker: $spk, session_id: $sid, valid_from: $ts, valid_to: 9999999999, "
                    "created_at: $ts, is_current: true})",
                    fid=fact_id, hid=hid, txt=mem.fact, spk=turn["role"], sid=turn["session_id"], ts=ts
                )
                session.run(
                    "MATCH (f:Fact {id: $fid, haystack_id: $hid}), (e:Entity {id: $eid, haystack_id: $hid}) "
                    "CREATE (f)-[:ABOUT]->(e)",
                    fid=fact_id, eid=entity_id, hid=hid
                )
            self.embedding_index.add(fact_id, mem.fact, hid)

        elif mem.action == "UPDATE" and mem.target_fact_id:
            new_fid = self.id_gen.next_id()
            with self.driver.session() as session:
                session.run(
                    "CREATE (f_new:Fact {id: $new_fid, haystack_id: $hid, text: $txt, "
                    "speaker: $spk, session_id: $sid, valid_from: $ts, valid_to: 9999999999, "
                    "created_at: $ts, is_current: true})"
                    "-[:SUPERSEDES]->(f_old:Fact {id: $old_fid, haystack_id: $hid})",
                    new_fid=new_fid, old_fid=mem.target_fact_id, hid=hid, txt=mem.fact,
                    spk=turn["role"], sid=turn["session_id"], ts=ts
                )
                session.run(
                    "MATCH (f_old:Fact {id: $old_fid, haystack_id: $hid}) "
                    "SET f_old.is_current = false, f_old.valid_to = $ts",
                    old_fid=mem.target_fact_id, hid=hid, ts=ts
                )
            self.embedding_index.remove(mem.target_fact_id)
            self.embedding_index.add(new_fid, mem.fact, hid)

        elif mem.action == "DELETE" and mem.target_fact_id:
            with self.driver.session() as session:
                session.run(
                    "MATCH (f:Fact {id: $fid, haystack_id: $hid}) "
                    "SET f.is_current = false, f.valid_to = $ts",
                    fid=mem.target_fact_id, hid=hid, ts=ts
                )
            self.embedding_index.remove(mem.target_fact_id)
```

---

## 7. Implementation Checklist & Verification

- [ ] **Structured LLM Parser**: Verify JSON adherence and schema validation via Pydantic using GPT-4o-mini / Claude 3.5 Haiku.
- [ ] **Prompt Candidate Pruning**: Ensure no more than 5 candidate facts are provided to avoid prompt bloat and latency spikes.
- [ ] **HydraDB Driver Pooling**: Ensure the Bolt driver connection pool is reused across background worker threads.
- [ ] **Dual Index Invalidation**: Verify `EmbeddingIndex.remove` is triggered whenever `SUPERSEDES` or `DELETE` executes.
- [ ] **Abstention & Empty Extractions**: Support empty `memories: []` outputs gracefully for turns containing solely conversational filler.
