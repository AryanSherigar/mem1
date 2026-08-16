# HydraDB Long-Term Memory Engine

This project implements a production-ready long-term memory engine for AI assistants, built on **HydraDB** (a Bolt-compatible graph database). It is designed to evaluate cleanly against the **LongMemEval** benchmark while simultaneously serving as the backend for continuous, interactive chat applications.

For a comprehensive breakdown of specific subsystems, please refer to the deep-dive documents in the `docs/` directory.

---

## 1. Core Architecture

The system operates across three interconnected layers:

1. **HydraDB Graph Storage**: The source of truth. Stores Sessions, Turns, Facts, and Entities. It handles multi-hop reasoning (`algo.MSpaths`) and knowledge update invalidations (`SUPERSEDES`).
2. **In-Memory Semantic Index**: Since HydraDB stores scalar properties, all embedding vectors (384-dimensional `all-MiniLM-L6-v2`) live in a fast, in-process numpy matrix (`EmbeddingIndex`). It is kept in perfect sync with the graph.
3. **LLM Extraction & Synthesis**: Fast OSS models (`gpt-oss-20b`) extract structured facts from conversational turns. Large frontier models (`gpt-oss-120b`) synthesize answers from graph traversals.

## 2. Graph Schema Overview

The graph schema is designed to directly answer all LongMemEval question types with minimal traversal hops.

* **Nodes**:
  * `Session`: A discrete chat session.
  * `Turn`: A single user or assistant message.
  * `Fact`: An atomic factual statement extracted from a turn.
  * `Entity`: A resolved canonical subject (e.g., "max").
  * `Alias`: An alternative surface form of an entity (e.g., "my dog").
* **Key Relationships**:
  * `(Fact)-[:ABOUT]->(Entity)`: The core semantic bridge linking scattered facts.
  * `(Fact)-[:SUPERSEDES]->(Fact)`: Knowledge updates (e.g., changing jobs). Marks old facts as inactive.
  * `(Entity)-[:HAS_ALIAS]->(Alias)`: Entity resolution targets.
  * `(Entity)-[:MERGED_INTO]->(Entity)`: Reversible audit trails for continuous entity deduplication.

> 📚 **Deep Dive**: [docs/graph_schema_proposal.md](docs/graph_schema_proposal.md)

## 3. The Retrieval Pipeline

When a user asks a question, the system executes a 3-phase hybrid retrieval:

1. **Temporal Pruning**: An LLM extracts relative date ranges (e.g., "last month" -> `[start_epoch, end_epoch]`). Facts outside this window are pruned before semantic search.
2. **Semantic Seeding**: The `EmbeddingIndex` finds the top 10 most relevant facts using numpy cosine similarity.
3. **Graph Traversal**: 
   * Seed facts are mapped to their `Entity` nodes.
   * `algo.MSpaths` (a native HydraDB multi-source path algorithm) bridges these entities across distant sessions to find hidden connections.
   * `SUPERSEDES` paths are traversed to find the most recent valid state.
4. **Reader Synthesis**: The retrieved, timestamped facts are passed to the Reader LLM to generate the final response.

> 📚 **Deep Dive**: [docs/retrieval_architecture.md](docs/retrieval_architecture.md) & [docs/temporal_query_resolver.md](docs/temporal_query_resolver.md)

## 4. Entity Resolution & Memory Distillation

Because this engine supports **interactive, long-lived chat sessions**, it implements robust, continuous memory distillation:

* **Distillation**: A 5-phase LLM-as-a-function pipeline evaluates whether new information should `ADD`, `UPDATE` (supersede), or `DELETE` existing memories.
* **Entity Resolution (3-Tier)**: 
  1. Exact string match on aliases.
  2. Semantic blocking (`cosine_sim > 0.75`).
  3. LLM-batched disambiguation.
* **Startup Hydration**: On server restart, the engine queries HydraDB to automatically rebuild its in-memory vector index, ensuring no memories are lost across server reboots.

> 📚 **Deep Dive**: [docs/entity_resolution_strategy.md](docs/entity_resolution_strategy.md) & [docs/semantic_memory_distillation.md](docs/semantic_memory_distillation.md)

## 5. System Modes

The engine seamlessly powers two modes of operation:
* **Interactive Mode**: Async ingestion, background fact extraction, and `hydrate()` on startup.
* **Benchmark Mode**: Fast, ephemeral per-question ingestion that loops through the 500 questions in `longmemeval_s_cleaned.json` to generate `predictions.jsonl` with zero cross-test contamination.
