# Hybrid Retrieval Architecture — PostgreSQL/pgvector Seeding + HydraDB Traversal

> **Current runtime decision (2026-08-17):** direct OpenCypher traversal against
> the checked-in local HydraDB server is the active design. The application
> builds provenance-preserving graph plans; HydraDB stores and traverses them.

Follow-up to [graph_schema_proposal.md](graph_schema_proposal.md).

Canonical ingestion input is defined in [ingestion_contract_v1.md](ingestion_contract_v1.md).

Implementation remains gated by [ingestion_pipeline_roadmap.md](ingestion_pipeline_roadmap.md). Approved rationale and open hold-ons are tracked in [decisions.md](decisions.md).

---

## 0. Why PostgreSQL Alongside HydraDB

The original in-process Python dictionary/NumPy design is no longer the MVP storage decision. It loses embeddings on restart, has no durable chunk source of truth, and cannot provide reliable replay or model-version tracking.

The approved MVP boundary is:

- **PostgreSQL** stores immutable raw chunks, chunk metadata and hashes, embeddings, model versions, and ingestion/checkpoint state.
- **HydraDB** stores facts, entities, provenance references, memory type/scope,
  temporal/update links, and traversal topology.

PostgreSQL with [pgvector](https://github.com/pgvector/pgvector) keeps evidence and vectors in one durable SQL system. It supports exact and approximate nearest-neighbor search, metadata filters, and normal SQL transactions. PostgreSQL also leaves a path to sparse retrieval through [built-in full-text search](https://www.postgresql.org/docs/current/textsearch-tables.html).

This does **not** make PostgreSQL and HydraDB one atomic transaction. Cross-store ingestion requires idempotency, explicit job states, verification, and replay after partial failure.

Core storage/retrieval uses generic `context_id` and source metadata. LongMemEval enters through a dedicated adapter that emits the same canonical records as any other source; benchmark fields never appear in core SQL or graph contracts. Its timezone-free minute timestamps use one explicit benchmark-local ordering basis; `question_date` must use that same normalization during later evaluation/retrieval adaptation.

---

## 1. Storage Responsibility Decision

HydraDB cannot store a float-array property:

From [cypher-compat.md](../hydradb/cypher-compat.md):

> Property values are integers, floats, booleans and strings.

Scalar types only. No lists, no arrays, no binary blobs. An embedding vector (384 floats) cannot be stored as a node property.

**Consequence**: embeddings live in PostgreSQL/pgvector, not HydraDB and not a process-local dictionary. `Fact.id`, `context_id`, and `source_chunk_id` form the cross-store join contract.

Canonical ownership:

| Data | Store | Notes |
|---|---|---|
| Raw chunk text | PostgreSQL | Immutable evidence source |
| Chunk metadata/hash | PostgreSQL | Session, turn range, role, token count, source integrity |
| Embedding vector/model version | PostgreSQL/pgvector | Re-embeddable without rewriting chunks or graph history |
| Atomic fact text | HydraDB | Graph retrieval unit with source span reference |
| Entities and relationships | HydraDB | Semantic and temporal topology |
| Ingestion job/outbox state | PostgreSQL | Drives retries and cross-store verification |

Exact chunk boundaries and whether MVP stores fact embeddings only or both fact and chunk embeddings remain benchmark-driven decisions.

---

## 2. Exact pgvector Search Before Approximate Indexing

**MVP decision: PostgreSQL/pgvector exact cosine search with a mandatory `context_id` filter.**

pgvector performs exact nearest-neighbor search by default. HNSW and IVFFlat trade recall and operational complexity for speed; neither is justified before measurement at LongMemEval scale.

```sql
SELECT subject_id, 1 - (embedding <=> :query_embedding) AS cosine_similarity
FROM memory_embeddings
WHERE context_id = :context_id
  AND subject_kind = 'fact'
  AND is_active = true
  AND model_name = :model_name
  AND model_version = :model_version
ORDER BY embedding <=> :query_embedding
LIMIT :top_k;
```

Add HNSW only after exact-search latency violates an approved target. Any approximate index must be evaluated against exact-search Recall@K and must preserve tenant/context filtering behavior.

---

## 3. Embedding Model

**Choice: `sentence-transformers/all-MiniLM-L6-v2`**

| Property | Value |
|---|---|
| Dimensions | 384 |
| Parameters | 22M |
| Speed (CPU) | ~5ms per sentence |
| Model size | ~80 MB |
| License | Apache 2.0 |

**Reasoning**:
- **Not the extraction or reader model** — this is a tiny encoder, not a generative LLM. The extraction model (GPT-4o-mini or similar) and the reader model are separate, larger models.
- **384 dimensions fit the MVP** — compact vectors reduce SQL storage and exact-search work. Larger or newer models require the same golden-data benchmark before replacement.
- **Well-tested for semantic similarity** — MiniLM-L6-v2 consistently performs well on STS benchmarks and is the most-used sentence-transformer model. For matching question text to fact text (short sentence pairs), it's in the sweet spot.
- **CPU-only is fine** — at ~5ms per embedding, ingesting 1,200 facts takes ~6 seconds. Query embedding is instant. No GPU needed.

The model is loaded once at process start and shared across ingestion and retrieval.

---

## 4. Three-Phase Retrieval — End to End

### 4.0 Draft SQL Persistence Contract

The schema below establishes responsibilities, not final chunk boundaries. Migration details remain subject to fixture-driven review.

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE evidence_chunks (
    chunk_id TEXT PRIMARY KEY,
    context_id TEXT NOT NULL,
    contract_version TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_external_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    turn_start INTEGER NOT NULL,
    turn_end INTEGER NOT NULL,
    role TEXT,
    raw_text TEXT NOT NULL,
    token_count INTEGER,
    content_hash TEXT NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE memory_embeddings (
    embedding_id BIGSERIAL PRIMARY KEY,
    context_id TEXT NOT NULL,
    subject_kind TEXT NOT NULL CHECK (subject_kind IN ('fact', 'chunk')),
    subject_id TEXT NOT NULL,
    source_chunk_id TEXT NOT NULL REFERENCES evidence_chunks(chunk_id),
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    embedding vector(384) NOT NULL,
    embedded_content_hash TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (context_id, subject_kind, subject_id, model_name, model_version)
);

CREATE TABLE ingestion_jobs (
    job_id TEXT PRIMARY KEY,
    context_id TEXT NOT NULL,
    contract_version TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_chunk_id TEXT NOT NULL REFERENCES evidence_chunks(chunk_id),
    state TEXT NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX memory_embeddings_scope_idx
    ON memory_embeddings (context_id, subject_kind, is_active);

CREATE INDEX evidence_chunks_source_idx
    ON evidence_chunks (context_id, session_id, turn_start);
```

Superseded embeddings remain durable for historical/temporal retrieval. `is_active` controls default latest-state seeding; it does not delete history. Re-embedding inserts a new model-version row rather than mutating source chunks.

### 4.1 Generic Ingestion Pseudocode

Source adapters first map payloads into canonical context sessions/records. LongMemEval adapter is one caller of this function, not a branch inside it.

```python
def ingest_context_session(session: ContextSession, hydra_driver, sql_store, embedder):
    """
    Ingest one session into PostgreSQL + HydraDB using idempotent job state.
    Called once per canonical source session. No benchmark-specific fields.
    """
    context_id = session.context_id
    session_id = session.session_id
    date_str = session.occurred_at
    date_epoch = date_str_to_epoch(date_str)
    now_epoch = int(time.time())

    # 1. Create Session node
    hydra_driver.execute(
        "CREATE (s:Session {id: $id, context_id: $hid, session_id: $sid, date: $date, "
        "date_epoch: $dep, turn_count: $tc, source_index: $hi})",
        id=alloc_session_id(), hid=context_id, sid=session_id, date=date_str,
        dep=date_epoch, tc=len(session.records), hi=session.source_index
    )

    for turn_idx, turn in enumerate(session.records):
        turn_id = alloc_turn_id()
        chunk = build_immutable_chunk(
            context_id=context_id,
            session_id=session_id,
            turn_index=turn_idx,
            role=turn.actor_role,
            raw_text=turn.content,
            observed_at=date_str,
        )
        sql_store.upsert_chunk_and_job(chunk, state='chunk_stored')

        # 2. Create Turn node + HAS_TURN edge
        hydra_driver.execute(
            "CREATE (s:Session {id: $sid, context_id: $hid})-[:HAS_TURN {turn_index: $ti}]->"
            "(t:Turn {id: $tid, context_id: $hid, turn_index: $ti, role: $role, "
            "source_chunk_id: $cid, content_hash: $hash, session_id: $sessid})",
            sid=session_node_id, hid=context_id, tid=turn_id, ti=turn_idx,
            role=turn.actor_role, cid=chunk.id, hash=chunk.content_hash,
            sessid=session_id
        )

        # 3. LLM: Extract atomic facts from turn text with strict entity_type enum
        # Prompt enforces JSON schema with enum:
        # ['person', 'pet', 'place', 'organization', 'event', 'creative_work',
        #  'product', 'activity', 'preference', 'topic', 'other']
        facts = llm_extract_facts(turn.content, turn.actor_role, session_id)
        # Returns: [{"text": "...", "entities": [{"name": "...", "type": "<VALID_ENUM>"}],
        #            "valid_from_hint": epoch|None}, ...]

        for fact_data in facts:
            fact_id = alloc_fact_id()
            valid_from = fact_data.get('valid_from_hint') or date_epoch
            valid_to = 9999999999  # world-validity interval; not knowledge recency
            memory_type = validate_memory_type(fact_data.get('memory_type', 'semantic'))
            scope_type, scope_id = assign_scope(fact_data, session, context_id)

            # 4. Create Fact node + EXTRACTED_FROM edge
            hydra_driver.execute(
                "CREATE (f:Fact {id: $fid, context_id: $hid, text: $text, speaker: $spk, "
            "session_id: $sid, memory_type: $mt, scope_type: $st, scope_id: $scope, "
            "source_chunk_id: $cid, source_start: $start, source_end: $end, content_hash: $hash, "
            "observed_at: $observed, superseded_at: 9999999999, valid_from: $vf, valid_to: $vt, "
                "created_at: $ca, is_current: true})"
                "-[:EXTRACTED_FROM]->(t:Turn {id: $tid, context_id: $hid})",
                fid=fact_id, hid=context_id, text=fact_data['text'], spk=turn.actor_role,
                sid=session_id, mt=memory_type, st=scope_type, scope=scope_id,
                cid=chunk.id, start=fact_data['source_start'], end=fact_data['source_end'],
                hash=chunk.content_hash, observed=date_epoch, vf=valid_from, vt=valid_to,
                ca=now_epoch, tid=turn_id
            )

            # 5. Persist versioned fact embedding in PostgreSQL/pgvector
            sql_store.upsert_embedding(
                context_id=context_id,
                subject_kind='fact',
                subject_id=str(fact_id),
                source_chunk_id=chunk.id,
                vector=embedder.encode(fact_data['text']),
                embedded_content_hash=hash_text(fact_data['text']),
                model_name=embedder.model_name,
                model_version=embedder.model_version,
            )

            # 6. Resolve entities: for each entity in the fact (scoped to context_id)
            for ent in fact_data['entities']:
                entity_id = resolve_or_create_entity(
                    hydra_driver, ent['name'], ent['type'], context_id
                )
                # 6a. Create ABOUT edge
                hydra_driver.execute(
                    "CREATE (f:Fact {id: $fid, context_id: $hid})-[:ABOUT]->(e:Entity {id: $eid, context_id: $hid})",
                    fid=fact_id, eid=entity_id, hid=context_id
                )

            # 7. Check for knowledge updates (hybrid: deterministic + LLM)
            superseded = detect_supersession(hydra_driver, fact_id, fact_data, context_id)
            if superseded:
                old_fact_id = superseded['old_fact_id']
                # 7a. Create SUPERSEDES edge
                hydra_driver.execute(
                    "CREATE (f_new:Fact {id: $new, context_id: $hid})-[:SUPERSEDES "
                    "{superseded_at: $at}]->(f_old:Fact {id: $old, context_id: $hid})",
                    new=fact_id, old=old_fact_id, at=now_epoch, hid=context_id
                )
                # 7b. Close knowledge-time interval. Close world validity only
                # when classification says this was a real state change.
                close_superseded_fact(
                    hydra_driver,
                    old_fact_id=old_fact_id,
                    superseded_at=date_epoch,
                    close_validity=superseded.get('closes_validity', False),
                    valid_to=valid_from,
                    context_id=context_id,
                )
                # 7c. Preserve historical vector; exclude it from latest-state seeding
                sql_store.set_embedding_active('fact', str(old_fact_id), False)

            # 8. Create STATED_BY edge to User/Assistant entity (scoped to context_id)
            speaker_entity_id = 9999999 if turn.actor_role == 'user' else 9999998
            hydra_driver.execute(
                "CREATE (f:Fact {id: $fid, context_id: $hid})-[:STATED_BY]->"
                "(e:Entity {id: $eid, context_id: $hid})",
                fid=fact_id, eid=speaker_entity_id, hid=context_id
            )

        verify_graph_chunk_links(hydra_driver, sql_store, chunk.id, context_id)
        sql_store.mark_ingestion_job(chunk.id, state='completed')

VALID_ENTITY_TYPES = {
    'person', 'pet', 'place', 'organization', 'event',
    'creative_work', 'product', 'activity', 'preference', 'topic', 'other'
}

def resolve_or_create_entity(hydra_driver, name: str, entity_type: str, context_id: str) -> int:
    """
    Illustrative M6 graph-write sketch only. M5 now implements the prior
    decision layer: deterministic context-scoped matching, followed by bounded
    provider-neutral LLM selection for ambiguous supplied candidates. This
    pseudocode must consume that decision rather than perform an unrestricted
    lookup or provider call itself.
    """
    canonical_name = name.strip().lower()
    selector_key = f"{context_id}::{canonical_name}"
    # Enforce enum whitelist; map unknown/invented types to 'other'
    sanitized_type = entity_type if entity_type in VALID_ENTITY_TYPES else 'other'

    # Check for existing canonical entity in this context
    existing = hydra_driver.execute(
        "MATCH (e:Entity {canonical_name: $cname, context_id: $hid}) RETURN e.id AS id",
        cname=canonical_name, hid=context_id
    )
    if existing:
        return existing[0]['id']

    # Check if this name is an alias of an existing entity in this context
    alias_match = hydra_driver.execute(
        "MATCH (e:Entity {context_id: $hid})-[:HAS_ALIAS]->(a:Alias {canonical_alias: $cname, context_id: $hid}) RETURN e.id AS id",
        cname=canonical_name, hid=context_id
    )
    if alias_match:
        return alias_match[0]['id']

    # New entity: create Entity node
    new_id = alloc_entity_id()
    hydra_driver.execute(
        "CREATE (e:Entity {id: $id, context_id: $hid, name: $name, canonical_name: $cname, selector_key: $skey, entity_type: $etype})",
        id=new_id, hid=context_id, name=name, cname=canonical_name,
        skey=selector_key, etype=sanitized_type
    )
    return new_id
```

### 4.2 Retrieval Pseudocode — Three Phases

```python
def retrieve(query: ContextQuery, hydra_driver, sql_store, embedder) -> str:
    """
    Generic three-phase retrieval. LongMemEval maps its question record into
    ContextQuery and maps the returned answer to the benchmark hypothesis.
    """
    q_text = query.text
    strategy_hint = query.strategy_hint  # optional; never required
    q_date = query.query_time
    q_context_id = query.context_id

    # ================================================================
    # PHASE 1: Durable semantic seeding (PostgreSQL/pgvector, scoped by context_id)
    # ================================================================
    top_k = 10  # tunable

    query_vector = embedder.encode(q_text)
    seed_results = sql_store.search_embeddings(
        query_vector=query_vector,
        context_id=q_context_id,
        subject_kind='fact',
        model_name=embedder.model_name,
        model_version=embedder.model_version,
        # Historical replay needs inactive embeddings so graph knowledge-time
        # filtering can reconstruct what was current at question_date.
        active_only=(q_date is None),
        top_k=top_k,
    )
    # seed_results: [(fact_id, cosine_sim), ...] strictly from q_context_id

    seed_fact_ids = [fid for fid, _ in seed_results]
    seed_scores = {fid: score for fid, score in seed_results}

    # ================================================================
    # PHASE 2: Graph expansion (HydraDB, scoped by context_id)
    # ================================================================

    # 2a. Collect entities connected to seed facts within the context
    seed_entities = set()
    for fid in seed_fact_ids:
        result = hydra_driver.execute(
            "MATCH (f:Fact {id: $fid, context_id: $hid})-[:ABOUT]->(e:Entity {context_id: $hid}) "
            "RETURN e.id AS eid, e.canonical_name AS name",
            fid=fid, hid=q_context_id
        )
        for row in result:
            seed_entities.add((row['eid'], row['name']))

    entity_selector_keys = [f"{q_context_id}::{name}" for _, name in seed_entities]

    # 2b. Graph expansion via algo.MSpaths — find facts connected
    #     through shared entities across sessions within this context
    expanded_facts = {}  # fact_id → {text, hop_count, path_count}

    if len(entity_selector_keys) >= 2:
        # Multi-entity bridging: find paths between seed entities
        # through ABOUT edges (Fact↔Entity connections)
        paths = hydra_driver.execute(
            "CALL algo.MSpaths({"
            "  sourceLabel: 'Entity',"
            "  sourceProperty: 'selector_key',"
            "  sourceValues: $selector_keys,"
            "  targetValues: $selector_keys,"
            "  pairwise: true,"
            "  relTypes: ['ABOUT'],"
            "  relDirection: 'both',"
            "  maxLen: 4,"
            "  pathCount: 5,"
            "  fairRelationshipVariants: true,"
            "  resultLimit: 100"
            "}) YIELD path RETURN path",
            selector_keys=entity_selector_keys
        )
        # Parse paths to extract intermediate Fact nodes belonging to this context
        for path in paths:
            for node, hops in extract_facts_from_path(path):
                # Extra safety check: verify context_id matches
                if node.get('context_id') == q_context_id:
                    if node['id'] not in expanded_facts:
                        expanded_facts[node['id']] = {
                            'text': node.get('text', ''),
                            'hop_count': hops,
                            'path_count': 1
                        }
                    else:
                        expanded_facts[node['id']]['path_count'] += 1

    # 2c. Single-entity expansion: for each seed entity, get all connected
    #     current facts in the same context
    for eid, _ in seed_entities:
        result = hydra_driver.execute(
            "MATCH (f:Fact {context_id: $hid})-[:ABOUT]->(e:Entity {id: $eid, context_id: $hid}) "
            "WHERE f.is_current = true "
            "RETURN f.id AS fid, f.text AS text",
            eid=eid, hid=q_context_id
        )
        for row in result:
            if row['fid'] not in expanded_facts and row['fid'] not in seed_scores:
                expanded_facts[row['fid']] = {
                    'text': row['text'],
                    'hop_count': 1,
                    'path_count': 1
                }

    # 2d. Knowledge-update expansion: follow SUPERSEDES chains within the context
    if strategy_hint == 'knowledge-update' or detects_update_query(q_text):
        for fid in seed_fact_ids:
            result = hydra_driver.execute(
                "MATCH (f:Fact {id: $fid, context_id: $hid})-[:SUPERSEDES*1..5]->(f_old:Fact {context_id: $hid}) "
                "RETURN f_old.id AS fid, f_old.text AS text",
                fid=fid, hid=q_context_id
            )
            for row in result:
                expanded_facts[row['fid']] = {
                    'text': row['text'], 'hop_count': 1, 'path_count': 1
                }
            # Also check if seed was superseded BY something newer in this context
            result = hydra_driver.execute(
                "MATCH (f_new:Fact {context_id: $hid})-[:SUPERSEDES*1..5]->(f:Fact {id: $fid, context_id: $hid}) "
                "RETURN f_new.id AS fid, f_new.text AS text",
                fid=fid, hid=q_context_id
            )
            for row in result:
                expanded_facts[row['fid']] = {
                    'text': row['text'], 'hop_count': 1, 'path_count': 1
                }

    # 2e. Historical replay cutoff for every question
    if q_date:
        query_epoch = date_str_to_epoch(q_date)
        observable_ids = set()
        all_candidate_ids = list(seed_scores.keys()) + list(expanded_facts.keys())
        for fid in all_candidate_ids:
            result = hydra_driver.execute(
                "MATCH (f:Fact {id: $fid, context_id: $hid}) "
                "WHERE f.observed_at <= $qe AND f.superseded_at > $qe "
                "RETURN f.id AS fid",
                fid=fid, hid=q_context_id, qe=query_epoch
            )
            for row in result:
                observable_ids.add(row['fid'])
        seed_scores = {k: v for k, v in seed_scores.items() if k in observable_ids}
        expanded_facts = {k: v for k, v in expanded_facts.items() if k in observable_ids}

    # Temporal-reasoning questions apply a second, distinct validity-time filter.
    if strategy_hint == 'temporal-reasoning' or detects_temporal_query(q_text):
        target_epoch = resolve_target_epoch(q_text, q_date, hydra_driver, q_context_id)
        seed_scores, expanded_facts = filter_by_validity_window(
            seed_scores, expanded_facts, target_epoch, hydra_driver, q_context_id
        )

    # ================================================================
    # PHASE 3: Hybrid scoring + context assembly
    # ================================================================
    SEMANTIC_WEIGHT = 0.6
    STRUCTURAL_WEIGHT = 0.4

    all_facts = {}

    # Score seed facts
    for fid, sem_score in seed_scores.items():
        structural_score = 1.0  # direct hit
        combined = SEMANTIC_WEIGHT * sem_score + STRUCTURAL_WEIGHT * structural_score
        all_facts[fid] = {'score': combined, 'semantic': sem_score}

    # Score expanded facts
    for fid, info in expanded_facts.items():
        sem_score = seed_scores.get(fid, 0.0)
        # Structural score: inversely proportional to hop distance,
        # boosted by number of connecting paths
        structural_score = (1.0 / info['hop_count']) * min(info['path_count'], 3) / 3
        combined = SEMANTIC_WEIGHT * sem_score + STRUCTURAL_WEIGHT * structural_score
        if fid not in all_facts or all_facts[fid]['score'] < combined:
            all_facts[fid] = {'score': combined, 'semantic': sem_score}

    # Sort by combined score, take top-N for context
    ranked = sorted(all_facts.items(), key=lambda x: x[1]['score'], reverse=True)
    top_n = 15  # context window budget

    # ---- Abstention check ----
    ABSTENTION_THRESHOLD = 0.25
    if not ranked or (ranked[0][1]['semantic'] < ABSTENTION_THRESHOLD
                      and len(expanded_facts) == 0):
        # Low confidence: best semantic score is weak AND no graph connections found
        # → abstain rather than hallucinate
        return "I don't have that information in my memory."

    # ---- Assemble context for reader LLM ----
    context_evidence = []
    for fid, scores in ranked[:top_n]:
        # Fetch graph fact + SQL source pointer (strictly scoped by context_id)
        result = hydra_driver.execute(
            "MATCH (f:Fact {id: $fid, context_id: $hid})-[:EXTRACTED_FROM]->(t:Turn {context_id: $hid})"
            "<-[:HAS_TURN]-(s:Session {context_id: $hid}) "
            "RETURN f.text AS fact, f.speaker AS speaker, "
            "f.source_chunk_id AS chunk_id, f.source_start AS source_start, "
            "f.source_end AS source_end, f.content_hash AS content_hash, "
            "s.date AS date, s.session_id AS session_id",
            fid=fid, hid=q_context_id
        )
        for row in result:
            evidence = sql_store.load_evidence_progressively(
                chunk_id=row['chunk_id'],
                expected_hash=row['content_hash'],
                source_start=row['source_start'],
                source_end=row['source_end'],
                question=q_text,
                levels=('source_span', 'neighboring_turn', 'full_chunk'),
            )
            context_evidence.append({
                'fact': row['fact'],
                'evidence': evidence,
                'speaker': row['speaker'],
                'date': row['date'],
                'session_id': row['session_id'],
            })

    # ---- Generate answer with reader LLM ----
    answer = reader_llm_generate(q_text, context_evidence, strategy_hint)
    return answer
```

### Phase 2 — Correction from Schema Update

> [!WARNING]
> The task description mentions traversing `SAME_AS` edges in Phase 2. **`SAME_AS` was removed** from the schema in the previous revision — co-reference is now handled entirely by `HAS_ALIAS` (Entity → Alias). The relationship types traversed in Phase 2 are:
> - `ABOUT` (Fact ↔ Entity) — the primary join between facts and entities
> - `SUPERSEDES` (Fact → Fact) — knowledge-update chains
> - `RELATES_TO` (Entity ↔ Entity) — semantic entity-entity links
> - `HAS_ALIAS` (Entity → Alias) — used at query time for entity name resolution, not in `algo.MSpaths` traversal

### `algo.MSpaths` — Confirmed Compatibility

The `algo.MSpaths` call in Phase 2b is valid against the [confirmed Cypher constraints](../hydradb/cypher-compat.md):

| Constraint | Status |
|---|---|
| Config map with named keys | ✅ Uses `sourceLabel`, `sourceProperty`, `sourceValues`, and `targetValues` |
| `YIELD path` + `RETURN path` — only yielded columns | ✅ |
| `relTypes` as a list | ✅ `['ABOUT']` — single type to avoid ambiguity |
| `maxLen` required (bounded) | ✅ `maxLen: 4` |
| No `WITH` filtering, no `IN`, no `CONTAINS` | ✅ Not used |
| `sourceValues`/`targetValues` accept list parameters | ✅ Same list of context-qualified keys via `$selector_keys` |
| One statement per request | ✅ |

> [!NOTE]
> We use `relTypes: ['ABOUT']` (single type) in the `MSpaths` call rather than mixing `ABOUT` + `RELATES_TO` + `SUPERSEDES`. Reason: `MSpaths` finds paths between Entity nodes via shared Facts — the `ABOUT` relationship is the bridge. `SUPERSEDES` and `RELATES_TO` expansion are handled by separate, targeted MATCH queries (2c, 2d) where the semantics are clearer and we can apply appropriate `WHERE` filters.

---

## 5. Abstention Integration

Abstention is a runtime evidence decision. It must never inspect `question_id` suffixes, `has_answer`, expected answers, or any other evaluation-only field.

Initial MVP gate:

```text
IF no ranked evidence:
    -> abstain
ELSE IF semantic support is below threshold
        AND graph expansion adds no corroborating evidence:
    -> abstain
ELSE IF temporal resolution fails or contradiction remains unresolved:
    -> abstain with the corresponding reason
ELSE:
    -> generate from source-linked context
```

The numeric threshold is uncalibrated until a held-out development split exists. Tune it on retrieval signals and answer behavior, then report abstention precision/recall separately on the untouched evaluation set.

---

## 6. Graph and SQL Schema Diff

`graph_schema_proposal.md` now carries the cross-store provenance contract:

- `Turn.source_chunk_id` and `Turn.content_hash` point to immutable SQL evidence.
- `Fact.source_chunk_id`, `source_start`, and `source_end` identify the supporting span.
- `Fact.observed_at` separates knowledge-availability time from real-world validity time.
- `Fact.superseded_at` closes the knowledge-time interval without necessarily changing world validity.
- `Fact.memory_type`, `scope_type`, and `scope_id` support memory taxonomy and hierarchy.
- `Entity.selector_key` qualifies native path selectors with `context_id`.

Join and lifecycle contract:

- **Write**: persist immutable chunk and ingestion job in PostgreSQL, write/verify HydraDB nodes and edges, persist versioned embeddings, then mark job complete.
- **Read**: pgvector returns fact identifiers; HydraDB expands/ranks graph evidence; PostgreSQL returns verified source spans and optional wider context.
- **Supersede**: preserve old graph facts and embeddings for history; set latest-state flags without deleting evidence.
- **Recover**: replay incomplete jobs idempotently and verify both stores before completion.

---

## 7. Context Construction and Summary of Decisions

### 7.1 Selective Evidence vs Full Raw Chunk

**MVP decision: progressive evidence expansion, not unconditional full-chunk injection.**

Context construction proceeds through bounded levels:

1. **Fact + exact source span** — smallest evidence directly supporting the fact.
2. **Neighboring sentence or turn** — add when the span contains unresolved pronouns, negation, comparison, or temporal language.
3. **Full raw chunk** — use only when narrower evidence remains ambiguous or the reader explicitly needs broader narrative context.

Every level includes speaker, session, timestamp, chunk identifier, and content hash. The reader receives the smallest sufficient context under a token budget. Raw chunks remain retrievable for audit even when only a span enters the prompt.

This is a benchmarked policy, not a permanent heuristic. Compare selective evidence against full-chunk baselines using answer accuracy, evidence recall, false-answer rate, context tokens, and latency.

### 7.2 Approved MVP Decisions

| Decision | MVP choice | Reason / status |
|---|---|---|
| Raw chunk storage | PostgreSQL | Durable canonical evidence; exact columns still fixture-driven |
| Embedding storage | PostgreSQL + pgvector | Persistence, model versioning, SQL filters, exact/ANN path |
| Similarity engine | Exact pgvector cosine search | Preserve recall at current scale; benchmark before HNSW/IVFFlat |
| Embedding target | Fact embeddings first; benchmark chunk embeddings | Final fact-only vs fact-plus-chunk choice remains open |
| Embedding model | `all-MiniLM-L6-v2` (384 dimensions) | Initial reproducible baseline; adapter-bound and replaceable |
| Phase 2 traversal | Scoped `algo.MSpaths` plus targeted `MATCH` queries | Graph-native expansion with context-qualified selector keys |
| Hybrid scoring | Initial `0.6 semantic + 0.4 structural` | Uncalibrated starting point; must be tuned on development data |
| Reader context | Progressive span -> neighbor -> full chunk | Reduce distractors while retaining recoverable source context |
| Superseded evidence | Preserve; mark inactive for latest-state seeding | Historical and temporal questions require old states |
| Abstention | Evidence/temporal/contradiction gate | No benchmark-label leakage |
| Cross-store writes | Idempotent job/outbox + verification | PostgreSQL and HydraDB cannot share one transaction |
| Organization memory | Deferred | Requires tenant authorization and promotion policy |
| `SAME_AS` | Removed | Co-reference handled by canonical entities and `HAS_ALIAS` |
