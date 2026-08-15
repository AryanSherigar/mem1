# Hybrid Retrieval Architecture — Semantic Seeding + Graph Traversal

Follow-up to [graph_schema_proposal.md](file:///home/aryan-sherigar/.gemini/antigravity-cli/brain/203a0cc1-b4d3-40f7-be5d-e8e73e65c45d/graph_schema_proposal.md).

---

## 0. Why No Separate Vector Database — Confirmed Understanding

At `longmemeval_s` scale (~1,200 facts, ~40 sessions, ~500 entities), a separate vector DB adds operational complexity (second service, consistency sync on `SUPERSEDES` invalidations) that dwarfs any performance benefit. In-process embedding search over a few hundred vectors is sub-millisecond with numpy. The sync problem is real: every `SUPERSEDES` edge must atomically update `is_current` in HydraDB **and** remove/update the embedding in the vector index. With in-process storage, both happen in the same Python process, in the same ingestion transaction. No distributed consistency problem to solve.

---

## 1. Embedding Property Type Decision

**Answer: No. HydraDB cannot store a float-array property.**

From [cypher-compat.md](file:///home/aryan-sherigar/projects/hydradb-hackathon/hydradb/cypher-compat.md) line 242:

> Property values are integers, floats, booleans and strings.

Scalar types only. No lists, no arrays, no binary blobs. An embedding vector (384 floats) cannot be stored as a node property.

**Consequence**: Embeddings live in an **in-process Python structure** — a dictionary mapping `Fact.id → numpy.ndarray`. The `Fact` node in HydraDB carries no embedding property. The join key is `Fact.id` (an integer), which exists in both stores.

---

## 2. numpy vs FAISS

**Recommendation: Pure numpy. FAISS is unnecessary.**

| Factor | numpy | FAISS (`faiss-cpu`) |
|---|---|---|
| Scale | ~1,200 facts × 384 dims = **1.8 MB** matrix | Same data, indexed |
| Similarity computation | `np.dot` on normalized vectors: **<1ms** for 1,200 rows | Also <1ms, but adds index build overhead |
| Dependencies | Already required (transitive via sentence-transformers) | Extra C++ compiled dependency, potential install issues |
| Complexity | 15 lines of code | Index creation, serialization, re-building on ingestion |
| Medium tier ceiling | ~15,000 facts × 384 dims = **23 MB**, still **<10ms** | Unnecessary until >100K vectors |

At 1,200 facts, numpy cosine similarity is a single matrix multiplication — there is nothing for FAISS to optimize. Even at the medium tier (~15K facts, which we're not targeting), numpy stays under 10ms. FAISS's IVF/HNSW indexing provides speedup only when brute-force exceeds tens of milliseconds, which requires >100K vectors.

**No new dependency. No index management. No serialization.**

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
- **384-dim is ideal for numpy brute-force** — smaller vectors mean faster dot products and less memory. 768-dim models (e.g., `bge-base`) offer marginal quality improvement but double storage and compute for no benefit at our recall-oriented task.
- **Well-tested for semantic similarity** — MiniLM-L6-v2 consistently performs well on STS benchmarks and is the most-used sentence-transformer model. For matching question text to fact text (short sentence pairs), it's in the sweet spot.
- **CPU-only is fine** — at ~5ms per embedding, ingesting 1,200 facts takes ~6 seconds. Query embedding is instant. No GPU needed.

The model is loaded once at process start and shared across ingestion and retrieval.

---

## 4. Three-Phase Retrieval — End to End

### 4.0 In-Memory Embedding Index Structure

```python
class EmbeddingIndex:
    """In-process embedding store with haystack_id partitioning. No external dependencies beyond numpy."""
    
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self._dirty = True                     # rebuild matrix on next query
        self._embeddings: dict[int, np.ndarray] = {}  # fact_id → embedding (384,)
        self._haystacks: dict[int, str] = {}          # fact_id → haystack_id
        self.fact_ids: list[int] = []          # parallel to rows in self.matrix
        self.fact_haystacks: list[str] = []    # parallel to rows in self.matrix
        self.matrix: np.ndarray | None = None  # shape (n_facts, 384), L2-normalized
    
    def add(self, fact_id: int, text: str, haystack_id: str):
        """Called once per Fact at ingestion time."""
        vec = self.model.encode(text, normalize_embeddings=True)
        self._embeddings[fact_id] = vec
        self._haystacks[fact_id] = haystack_id
        self._dirty = True
    
    def remove(self, fact_id: int):
        """Called when a Fact is superseded (is_current → false)."""
        self._embeddings.pop(fact_id, None)
        self._haystacks.pop(fact_id, None)
        self._dirty = True
    
    def _rebuild(self):
        """Stack all embeddings into a single matrix for batch similarity."""
        self.fact_ids = list(self._embeddings.keys())
        self.fact_haystacks = [self._haystacks[fid] for fid in self.fact_ids]
        if self.fact_ids:
            self.matrix = np.stack([self._embeddings[fid] for fid in self.fact_ids])
        else:
            self.matrix = np.empty((0, 384))
        self._dirty = False
    
    def search(self, query_text: str, haystack_id: str, top_k: int = 10,
               include_non_current: bool = False) -> list[tuple[int, float]]:
        """
        Returns list of (fact_id, cosine_similarity) sorted descending,
        strictly filtered to the requested haystack_id.
        """
        if self._dirty:
            self._rebuild()
        if self.matrix is None or len(self.fact_ids) == 0:
            return []
        
        # Filter indices belonging to this haystack
        mask = [h == haystack_id for h in self.fact_haystacks]
        if not any(mask):
            return []
        
        scoped_ids = [self.fact_ids[i] for i, m in enumerate(mask) if m]
        scoped_matrix = self.matrix[mask]
        
        query_vec = self.model.encode(query_text, normalize_embeddings=True)
        # Cosine similarity = dot product on L2-normalized vectors
        scores = scoped_matrix @ query_vec  # shape (n_scoped_facts,)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(scoped_ids[i], float(scores[i])) for i in top_indices]
```

> [!NOTE]
> The `remove()` method is called when `SUPERSEDES` marks a fact as `is_current = false`. For `knowledge-update` questions specifically, we need historical facts too — the caller can maintain a **second** `EmbeddingIndex` instance that retains all embeddings (never calls `remove`), or re-add superseded facts temporarily. The primary index only holds `is_current = true` facts.

### 4.1 Ingestion Pseudocode

```python
def ingest_session(session: dict, hydra_driver, embedding_index: EmbeddingIndex, haystack_id: str):
    """
    Ingest one session from longmemeval_s_cleaned.json into HydraDB + embedding index.
    Called once per session, in haystack order, with explicit haystack_id.
    """
    session_id = session['session_id']
    date_str = session['date']              # e.g. '2024-03-15'
    date_epoch = date_str_to_epoch(date_str)
    now_epoch = int(time.time())

    # 1. Create Session node
    hydra_driver.execute(
        "CREATE (s:Session {id: $id, haystack_id: $hid, session_id: $sid, date: $date, "
        "date_epoch: $dep, turn_count: $tc, haystack_index: $hi})",
        id=alloc_session_id(), hid=haystack_id, sid=session_id, date=date_str,
        dep=date_epoch, tc=len(session['turns']), hi=session['index']
    )

    for turn_idx, turn in enumerate(session['turns']):
        turn_id = alloc_turn_id()

        # 2. Create Turn node + HAS_TURN edge
        hydra_driver.execute(
            "CREATE (s:Session {id: $sid, haystack_id: $hid})-[:HAS_TURN {turn_index: $ti}]->"
            "(t:Turn {id: $tid, haystack_id: $hid, turn_index: $ti, role: $role, "
            "content: $content, has_answer: $ha, session_id: $sessid})",
            sid=session_node_id, hid=haystack_id, tid=turn_id, ti=turn_idx,
            role=turn['role'], content=turn['content'],
            ha=turn.get('has_answer', False), sessid=session_id
        )

        # 3. LLM: Extract atomic facts from turn text with strict entity_type enum
        # Prompt enforces JSON schema with enum:
        # ['person', 'pet', 'place', 'organization', 'event', 'creative_work', 
        #  'product', 'activity', 'preference', 'topic', 'other']
        facts = llm_extract_facts(turn['content'], turn['role'], session_id)
        # Returns: [{"text": "...", "entities": [{"name": "...", "type": "<VALID_ENUM>"}], 
        #            "valid_from_hint": epoch|None}, ...]

        for fact_data in facts:
            fact_id = alloc_fact_id()
            valid_from = fact_data.get('valid_from_hint') or date_epoch
            valid_to = 9999999999  # open-ended until superseded

            # 4. Create Fact node + EXTRACTED_FROM edge
            hydra_driver.execute(
                "CREATE (f:Fact {id: $fid, haystack_id: $hid, text: $text, speaker: $spk, "
                "session_id: $sid, valid_from: $vf, valid_to: $vt, "
                "created_at: $ca, is_current: true})"
                "-[:EXTRACTED_FROM]->(t:Turn {id: $tid, haystack_id: $hid})",
                fid=fact_id, hid=haystack_id, text=fact_data['text'], spk=turn['role'],
                sid=session_id, vf=valid_from, vt=valid_to,
                ca=now_epoch, tid=turn_id
            )

            # 5. Compute embedding and add to in-process index (scoped to haystack_id)
            embedding_index.add(fact_id, fact_data['text'], haystack_id=haystack_id)

            # 6. Resolve entities: for each entity in the fact (scoped to haystack_id)
            for ent in fact_data['entities']:
                entity_id = resolve_or_create_entity(
                    hydra_driver, ent['name'], ent['type'], haystack_id
                )
                # 6a. Create ABOUT edge
                hydra_driver.execute(
                    "CREATE (f:Fact {id: $fid, haystack_id: $hid})-[:ABOUT]->(e:Entity {id: $eid, haystack_id: $hid})",
                    fid=fact_id, eid=entity_id, hid=haystack_id
                )

            # 7. Check for knowledge updates (hybrid: deterministic + LLM)
            superseded = detect_supersession(hydra_driver, fact_id, fact_data, haystack_id)
            if superseded:
                old_fact_id = superseded['old_fact_id']
                # 7a. Create SUPERSEDES edge
                hydra_driver.execute(
                    "CREATE (f_new:Fact {id: $new, haystack_id: $hid})-[:SUPERSEDES "
                    "{superseded_at: $at}]->(f_old:Fact {id: $old, haystack_id: $hid})",
                    new=fact_id, old=old_fact_id, at=now_epoch, hid=haystack_id
                )
                # 7b. Mark old fact as no longer current
                hydra_driver.execute(
                    "MATCH (f:Fact {id: $old, haystack_id: $hid}) SET f.is_current = false, "
                    "f.valid_to = $vt",
                    old=old_fact_id, vt=valid_from, hid=haystack_id
                )
                # 7c. Remove old fact from primary embedding index
                embedding_index.remove(old_fact_id)

            # 8. Create STATED_BY edge to User/Assistant entity (scoped to haystack_id)
            speaker_entity_id = 9999999 if turn['role'] == 'user' else 9999998
            hydra_driver.execute(
                "CREATE (f:Fact {id: $fid, haystack_id: $hid})-[:STATED_BY]->"
                "(e:Entity {id: $eid, haystack_id: $hid})",
                fid=fact_id, eid=speaker_entity_id, hid=haystack_id
            )

VALID_ENTITY_TYPES = {
    'person', 'pet', 'place', 'organization', 'event', 
    'creative_work', 'product', 'activity', 'preference', 'topic', 'other'
}

def resolve_or_create_entity(hydra_driver, name: str, entity_type: str, haystack_id: str) -> int:
    """
    Deterministic/canonical entity resolution with strict enum enforcement,
    scoped strictly to the current haystack_id.
    """
    canonical_name = name.strip().lower()
    # Enforce enum whitelist; map unknown/invented types to 'other'
    sanitized_type = entity_type if entity_type in VALID_ENTITY_TYPES else 'other'

    # Check for existing canonical entity in this haystack
    existing = hydra_driver.execute(
        "MATCH (e:Entity {canonical_name: $cname, haystack_id: $hid}) RETURN e.id AS id",
        cname=canonical_name, hid=haystack_id
    )
    if existing:
        return existing[0]['id']

    # Check if this name is an alias of an existing entity in this haystack
    alias_match = hydra_driver.execute(
        "MATCH (e:Entity {haystack_id: $hid})-[:HAS_ALIAS]->(a:Alias {canonical_alias: $cname, haystack_id: $hid}) RETURN e.id AS id",
        cname=canonical_name, hid=haystack_id
    )
    if alias_match:
        return alias_match[0]['id']

    # New entity: create Entity node
    new_id = alloc_entity_id()
    hydra_driver.execute(
        "CREATE (e:Entity {id: $id, haystack_id: $hid, name: $name, canonical_name: $cname, entity_type: $etype})",
        id=new_id, hid=haystack_id, name=name, cname=canonical_name, etype=sanitized_type
    )
    return new_id
```

### 4.2 Retrieval Pseudocode — Three Phases

```python
def retrieve(question: dict, hydra_driver, embedding_index: EmbeddingIndex) -> str:
    """
    Three-phase retrieval for a single LongMemEval question.
    Returns the generated answer string (the 'hypothesis').
    """
    q_text = question['question']
    q_type = question['question_type']
    q_date = question.get('question_date')
    q_haystack_id = question.get('haystack_id', question['question_id'])
    is_abstention = question['question_id'].endswith('_abs')

    # ================================================================
    # PHASE 1: Semantic seeding (in-process, numpy, scoped by haystack_id)
    # ================================================================
    top_k = 10  # tunable

    seed_results = embedding_index.search(q_text, haystack_id=q_haystack_id, top_k=top_k)
    # seed_results: [(fact_id, cosine_sim), ...] strictly from q_haystack_id

    seed_fact_ids = [fid for fid, _ in seed_results]
    seed_scores = {fid: score for fid, score in seed_results}

    # ================================================================
    # PHASE 2: Graph expansion (HydraDB, scoped by haystack_id)
    # ================================================================

    # 2a. Collect entities connected to seed facts within the haystack
    seed_entities = set()
    for fid in seed_fact_ids:
        result = hydra_driver.execute(
            "MATCH (f:Fact {id: $fid, haystack_id: $hid})-[:ABOUT]->(e:Entity {haystack_id: $hid}) "
            "RETURN e.id AS eid, e.canonical_name AS name",
            fid=fid, hid=q_haystack_id
        )
        for row in result:
            seed_entities.add((row['eid'], row['name']))

    entity_names = [name for _, name in seed_entities]

    # 2b. Graph expansion via algo.MSpaths — find facts connected
    #     through shared entities across sessions within this haystack
    expanded_facts = {}  # fact_id → {text, hop_count, path_count}

    if len(entity_names) >= 2:
        # Multi-entity bridging: find paths between seed entities
        # through ABOUT edges (Fact↔Entity connections)
        paths = hydra_driver.execute(
            "CALL algo.MSpaths({"
            "  sourceLabel: 'Entity',"
            "  sourceProperty: 'canonical_name',"
            "  sourceValues: $names,"
            "  pairwise: true,"
            "  relTypes: ['ABOUT'],"
            "  relDirection: 'both',"
            "  maxLen: 4,"
            "  pathCount: 5,"
            "  fairRelationshipVariants: true,"
            "  resultLimit: 100"
            "}) YIELD path RETURN path",
            names=entity_names
        )
        # Parse paths to extract intermediate Fact nodes belonging to this haystack
        for path in paths:
            for node, hops in extract_facts_from_path(path):
                # Extra safety check: verify haystack_id matches
                if node.get('haystack_id') == q_haystack_id:
                    if node['id'] not in expanded_facts:
                        expanded_facts[node['id']] = {
                            'text': node.get('text', ''),
                            'hop_count': hops,
                            'path_count': 1
                        }
                    else:
                        expanded_facts[node['id']]['path_count'] += 1

    # 2c. Single-entity expansion: for each seed entity, get all connected
    #     current facts in the same haystack
    for eid, _ in seed_entities:
        result = hydra_driver.execute(
            "MATCH (f:Fact {haystack_id: $hid})-[:ABOUT]->(e:Entity {id: $eid, haystack_id: $hid}) "
            "WHERE f.is_current = true "
            "RETURN f.id AS fid, f.text AS text",
            eid=eid, hid=q_haystack_id
        )
        for row in result:
            if row['fid'] not in expanded_facts and row['fid'] not in seed_scores:
                expanded_facts[row['fid']] = {
                    'text': row['text'],
                    'hop_count': 1,
                    'path_count': 1
                }

    # 2d. Knowledge-update expansion: follow SUPERSEDES chains within the haystack
    if q_type == 'knowledge-update':
        for fid in seed_fact_ids:
            result = hydra_driver.execute(
                "MATCH (f:Fact {id: $fid, haystack_id: $hid})-[:SUPERSEDES*1..5]->(f_old:Fact {haystack_id: $hid}) "
                "RETURN f_old.id AS fid, f_old.text AS text",
                fid=fid, hid=q_haystack_id
            )
            for row in result:
                expanded_facts[row['fid']] = {
                    'text': row['text'], 'hop_count': 1, 'path_count': 1
                }
            # Also check if seed was superseded BY something newer in this haystack
            result = hydra_driver.execute(
                "MATCH (f_new:Fact {haystack_id: $hid})-[:SUPERSEDES*1..5]->(f:Fact {id: $fid, haystack_id: $hid}) "
                "RETURN f_new.id AS fid, f_new.text AS text",
                fid=fid, hid=q_haystack_id
            )
            for row in result:
                expanded_facts[row['fid']] = {
                    'text': row['text'], 'hop_count': 1, 'path_count': 1
                }

    # 2e. Temporal filtering (for temporal-reasoning questions within haystack)
    if q_type == 'temporal-reasoning' and q_date:
        query_epoch = date_str_to_epoch(q_date)
        # Re-filter: keep only facts valid at query time
        temporal_filter_ids = set()
        all_candidate_ids = list(seed_scores.keys()) + list(expanded_facts.keys())
        for fid in all_candidate_ids:
            result = hydra_driver.execute(
                "MATCH (f:Fact {id: $fid, haystack_id: $hid}) "
                "WHERE f.valid_from <= $qe AND f.valid_to >= $qe "
                "RETURN f.id AS fid",
                fid=fid, hid=q_haystack_id, qe=query_epoch
            )
            for row in result:
                temporal_filter_ids.add(row['fid'])
        # Remove facts outside the validity window
        seed_scores = {k: v for k, v in seed_scores.items() if k in temporal_filter_ids}
        expanded_facts = {k: v for k, v in expanded_facts.items() if k in temporal_filter_ids}

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
    if is_abstention or (ranked and ranked[0][1]['semantic'] < ABSTENTION_THRESHOLD
                         and len(expanded_facts) == 0):
        # Low confidence: best semantic score is weak AND no graph connections found
        # → abstain rather than hallucinate
        return "I don't have that information in my memory."

    # ---- Assemble context for reader LLM ----
    context_facts = []
    for fid, scores in ranked[:top_n]:
        # Fetch full fact text + provenance (strictly scoped by haystack_id)
        result = hydra_driver.execute(
            "MATCH (f:Fact {id: $fid, haystack_id: $hid})-[:EXTRACTED_FROM]->(t:Turn {haystack_id: $hid})"
            "<-[:HAS_TURN]-(s:Session {haystack_id: $hid}) "
            "RETURN f.text AS fact, f.speaker AS speaker, "
            "s.date AS date, s.session_id AS session_id",
            fid=fid, hid=q_haystack_id
        )
        for row in result:
            context_facts.append(
                f"[{row['date']}, {row['speaker']}]: {row['fact']}"
            )

    # ---- Generate answer with reader LLM ----
    answer = reader_llm_generate(q_text, context_facts, q_type)
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

The `algo.MSpaths` call in Phase 2b is valid against the [confirmed Cypher constraints](file:///home/aryan-sherigar/projects/hydradb-hackathon/hydradb/cypher-compat.md):

| Constraint | Status |
|---|---|
| Config map with named keys | ✅ Uses `sourceLabel`, `sourceProperty`, `sourceValues`, etc. |
| `YIELD path` + `RETURN path` — only yielded columns | ✅ |
| `relTypes` as a list | ✅ `['ABOUT']` — single type to avoid ambiguity |
| `maxLen` required (bounded) | ✅ `maxLen: 4` |
| No `WITH` filtering, no `IN`, no `CONTAINS` | ✅ Not used |
| `sourceValues` accepts a list parameter | ✅ List of strings via `$names` parameter |
| One statement per request | ✅ |

> [!NOTE]
> We use `relTypes: ['ABOUT']` (single type) in the `MSpaths` call rather than mixing `ABOUT` + `RELATES_TO` + `SUPERSEDES`. Reason: `MSpaths` finds paths between Entity nodes via shared Facts — the `ABOUT` relationship is the bridge. `SUPERSEDES` and `RELATES_TO` expansion are handled by separate, targeted MATCH queries (2c, 2d) where the semantics are clearer and we can apply appropriate `WHERE` filters.

---

## 5. Abstention Integration

Abstention is **not a separate system** — it's a threshold check in Phase 3 scoring:

```
IF  best_semantic_score < 0.25  AND  graph_expansion_found_nothing:
    → Abstain ("I don't have that information in my memory.")
ELSE:
    → Proceed to reader LLM with assembled context
```

The two signals are complementary:
- **Semantic score alone** can be low for poorly-worded questions about real facts → don't abstain yet
- **Graph expansion alone** can find spurious connections to distractor-session entities → don't trust structure alone
- **Both weak** → high confidence the information genuinely isn't in memory → abstain

The `0.25` threshold is a starting point to tune on the ~50 abstention questions in the dataset (questions where `question_id` ends with `_abs`).

---

## 6. Schema Diff

**No changes needed to `graph_schema_proposal.md`.**

Embeddings cannot be stored as HydraDB properties (float-array not supported), so no new column is added to the Fact property table. The embedding index is a purely in-process Python structure (`EmbeddingIndex` class above), keyed by `Fact.id`.

The join contract is:
- **Write path**: `Fact.id` (int) is written to both HydraDB (as node identity) and `EmbeddingIndex._embeddings` (as dict key) during ingestion
- **Read path**: Phase 1 returns `Fact.id` values, which Phase 2 uses in HydraDB `MATCH (f:Fact {id: $fid})` queries
- **Delete path**: When `SUPERSEDES` marks a fact `is_current = false`, `EmbeddingIndex.remove(fact_id)` is called in the same ingestion function

**The only new artifact is the `EmbeddingIndex` class**, which exists as Python code alongside the ingestion/retrieval scripts, not as a schema change.

---

## 7. Summary of Decisions

| Decision | Choice | Key Reason |
|---|---|---|
| Embedding storage | In-process Python dict + numpy array | HydraDB has no array property type |
| Similarity engine | numpy brute-force | <1ms at 1,200 facts; FAISS adds complexity for zero benefit |
| Embedding model | `all-MiniLM-L6-v2` (384-dim, 22M params) | Smallest/fastest sentence-transformer; CPU-only; not the extraction or reader model |
| Phase 2 traversal | `algo.MSpaths` for multi-entity bridging + targeted MATCH for SUPERSEDES/single-entity | Separates concerns; avoids mixing rel types in one MSpaths call |
| Hybrid scoring | `0.6 × semantic + 0.4 × structural` | Simple weighted combination; no learned ranker |
| Abstention | Threshold on (best_semantic_score, graph_expansion_count) | Not a separate system; integrated into Phase 3 scoring |
| `SAME_AS` in Phase 2 | **Removed** — was already dropped from schema | Co-reference handled by `HAS_ALIAS` |
