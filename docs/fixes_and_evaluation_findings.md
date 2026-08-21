# Fixes and Evaluation Findings

This document is a running record of two things, kept together because they
came from the same investigation: **(1)** every fix applied to ingestion and
retrieval while diagnosing real LongMemEval failures, and **(2)** the full
diagnosis of what went wrong in the 30-instance evaluation run — traced
against actual ingested data and actual retrieved context, not inferred from
the score alone. See `docs/decisions.md` for the formal ADR-numbered
rationale behind the larger architectural calls; this document is the
working detail underneath a subset of those ADRs (chiefly ADR-034/035) plus
everything done after them.

---

## 1. Fixes applied

### 1.1 Ingestion correctness

- **Speaker/entity namespace collision (`graph_plan_builder.py`).** The
  speaker node lived at `entity:{role}`. The extractor genuinely emits
  "user"/"assistant" as ordinary entity surfaces too, which canonicalize to
  the same logical key and the same allocated graph ID but a different
  `entity_type` ("speaker" vs "other"). Same key, same ID, different
  payload — `PostgresGraphManifestStore` rejects that by design. Confirmed
  live: this aborted a real LongMemEval ingestion run with
  `GraphPayloadConflictError: node entity:assistant has a different
  immutable graph payload`. Fixed by moving the speaker node to its own
  `speaker:` namespace, which removes the collision class entirely rather
  than reconciling two payloads that legitimately differ.

- **Entity coverage in extraction (`core/config.py`,
  `FACT_EXTRACTION_SYSTEM_PROMPT`).** "Named entities" alone was read
  strictly as proper nouns, so any fact about a common-noun topic (commute,
  rent, audiobooks) got an empty `entities` list. Measured: 691 of 2,685
  facts (25.7%) on one real instance had no entity at all — including every
  "commute" fact for a question whose gold answer was specifically about the
  user's commute. Entities are what graph expansion traverses and what the
  entity boost keys on, so an unlinked fact is invisible to both. Prompt now
  asks for salient topic nouns in addition to named entities; verified live
  this recovers entity links without disturbing named-entity extraction
  quality.

- **Swallowed timeouts in entity-resolution/temporal-update ports
  (`model_adapters.py`).** Both ports only caught `LLMClientError`, not
  `openai.APITimeoutError` (a different exception hierarchy). Confirmed live
  during the 30-instance run: 42 timeouts in one instance alone, each one
  propagating out of a bounded-decision port and aborting the *entire chunk*
  via the orchestrator's generic exception handler — losing every fact in
  that turn over one classification timing out. Both ports already had a
  defined "no clear basis to decide" fallback (`None` / `UNRESOLVED`);
  broadened to `except Exception` so a timeout degrades to that existing
  contract instead of aborting the chunk.

- **Malformed-JSON recovery generalized (`llm_client.py`).** An earlier fix
  special-cased literal prefixes (`{{`, `{\n{`) and stripped one character.
  Bedrock's `openai.gpt-oss-20b` deterministically (6/6 calls, not
  intermittent) emits a `{"` prefix instead — two characters, not a matched
  pattern, so the special-casing could not recover it. Replaced with a scan
  for the first balanced JSON object anywhere in the text (still strips ```
  fences first), which uniformly handles the `{"` case, leading prose, and
  trailing commentary instead of enumerating malformation shapes one at a
  time.

- **HydraDB Cypher injection in entity-value escaping (`retrieval.py`).** A
  hand-rolled `'` doubler left backslashes untouched — an entity ending in
  one produces `'ent\'`, whose trailing backslash escapes the closing quote
  and swallows the rest of the query. Reachable input: entity names come
  from LLM extraction of user content, and `canonicalize_entity_surface`
  only NFKC-normalizes and casefolds; it strips neither backslashes nor
  quotes. Switched to `json.dumps` for correct escaping. A dead, shadowed
  copy of the per-fact node query (left over from an earlier edit) was
  removed in the same pass.

### 1.2 Retrieval correctness

- **`SUPERSEDES` was wired but dead (ADR-034).** Every component existed and
  was unit-tested, but no production caller ever supplied the
  `find_existing_facts` callback the edge-creation logic is gated on, so
  `LLMTemporalUpdateModel` was never invoked and no `SUPERSEDES` edge was
  ever written. Fixed by adding `ingestion/fact_lookup.py::HydraFactLookup`
  as the real data source, constructed in both `api/routes.py` and
  `evaluation/benchmark_runner.py`. Also stopped gating the check on
  `action == UPDATE` — confirmed live, `LLMExtractor` sees only the current
  turn (no prior-facts context) and labelled "Max moved to Seattle." as
  `ADD` immediately after "Max lives in Boston." in the same context, so
  gating on the extractor's own guess meant supersession only fired when the
  model happened to phrase things as an explicit correction. Verified live
  end-to-end: `SUPERSEDES` edge created, prior fact's `is_current` flipped
  to `False`.

- **Entity boost was ungated (ADR-035).** A prior fix made `entity_boost`
  binary (cap if any entity linked, else 0) but dropped
  `FINAL_ARCHITECTURE.md`'s `if entity in query_entities` condition — every
  entity-linked fact got the same flat boost regardless of whether the query
  was actually about that entity. Measured on "How long is my daily commute
  to work?": the gold fact had the single highest semantic score of all 113
  seeds (0.716) but no entity link, so it ranked #39 while 15 unrelated
  bike-training facts each took the full boost and filled the entire top-15.
  Now applies inverse-frequency scaled, only when the fact's linked entity
  actually appears in the question (or its rewritten forms).

- **Abstention ignored `keyword_score`; BM25 went silent on rewriter
  failure (ADR-035).** A strong BM25-only match with weak embedding
  similarity could still trigger abstention. Separately, an empty rewriter
  result (same LLM path that has failed live on credentials/timeouts) used
  to skip keyword search for the whole request rather than degrading to the
  raw question. Both fixed.

- **Composite scoring replaced with Reciprocal Rank Fusion.** The formula
  summed four raw, differently-scaled scores (`semantic_score +
  keyword_score + structural_score + entity_boost`) directly. Whichever
  signal happened to read numerically "big" for a fact dominated the total
  regardless of actual relevance — the same failure shape already fixed once
  for `entity_boost` specifically, but present structurally across all four
  signals. Replaced with RRF: each fact's rank *position* within each
  channel (not its raw score) determines its contribution,
  `1/(k + rank)` per channel, k=60 (Cormack et al. 2009's standard
  constant). A fact absent from a channel contributes 0 from it rather than
  tying for last place. Two tests lock in the exact property this exists
  for: multi-channel presence can now outscore a single dominant raw score,
  which the old additive formula could not express.

- **Reader window widened, 15 → 20 (`retrieval_top_k`).** Cheap, low-risk
  mitigation for facts that scored well but fell just outside the cutoff —
  confirmed live this alone pulled a needed fact (a per-unit price) into
  context that had previously been excluded.

- **Sibling-fact expansion — the structural fix for split facts
  (`retrieval.py::_sibling_facts`), now scored with SCAR, not a flat
  `ORDER BY`.** Implements `FINAL_ARCHITECTURE.md`'s own ADR-005
  ("progressive evidence expansion: fact → neighboring turn → full chunk"),
  accepted at design time but never actually built. Went through two real
  iterations, both driven by live evidence rather than designed up front:

  **v1 (unordered, flat pull).** Traced live: extraction splits one turn
  into several atomic facts, and Phase 3 scores them independently — "User
  sold 20 potted herb plants" and "Each potted herb plant was sold for
  $7.5" come from the same turn, but a question needing both
  (20 × $7.5 = $150, one component of a larger $495 total) got the price
  without the quantity, because only the price scored high enough to be
  individually ranked. v1 pulled in every other fact from the same source
  turn as any top-K fact via `memory_embeddings.source_chunk_id`, capped at
  a shared `LIMIT 20` with no ordering.

  **Why v1 wasn't enough, found by re-testing rather than assumed working:**
  requesting numerical evidence for this exact fix surfaced that it hadn't
  actually fixed the traced case. Diagnosed precisely: the needed sibling
  (the $7.5/unit fact) had only 4 real candidate siblings for its own turn —
  comfortably under the cap on its own — but the query pools siblings across
  *all* facts in the top-20 combined, and unrelated turns' siblings filled
  the shared, unordered 20-row cap first. The needed fact was available and
  never reached.

  **v2 — SCAR (Semantic Continuity-Aware Retrieval; Zhong et al. 2026,
  [arxiv.org/abs/2606.16661](https://arxiv.org/abs/2606.16661)).** Chosen
  over hand-rolling a heuristic: this is a solved problem in the RAG
  literature (parent-document retrieval, sentence-window retrieval, and
  auto-merging retrieval all address the same "small chunk for precision,
  larger context for the LLM" tension), and SCAR is the entry that scores
  *which* neighbors earn inclusion instead of pulling a fixed radius or a
  flat top-N — the same gap v1 had. Its formula, using embeddings already
  stored in `memory_embeddings` (no extra embedding calls beyond the one
  question vector):

  ```
  S(anchor, neighbor) = cos(query, neighbor) − λ · (1 − cos(anchor, neighbor))
  keep neighbor only if:  S(anchor, neighbor) > γ · cos(query, anchor)
  ```

  Paper defaults used unchanged (λ=0.1, γ=0.80) — no tuning data of our own
  exists yet to justify moving them. **Why this is the right shape for the
  underlying tension, not just a bigger hammer:** a same-turn fact is a
  *plausible* signal that a neighbor is relevant, not proof — the actually-
  missing gold fact can just as easily live in a completely different turn,
  which no amount of neighbor expansion reaches (that remains a Phase 1
  recall problem, out of scope for this fix). SCAR's threshold is relative
  to *each anchor's own* query relevance rather than one fixed cutoff for
  every fact: a weakly-relevant anchor sets a low bar (loosely-related
  context still gets a chance), a strongly-relevant anchor demands genuinely
  comparable neighbors (a strong hit doesn't get to drag in everything
  physically nearby). This is exactly the requested property — biased
  toward the anchor's own turn, but not an absolute or all-or-nothing bias.

  **Benefit, measured, not asserted:** the exact money-aggregation case that
  exposed v1's gap now answers **$495** (exact gold match; was $345 before
  any of this, still $345 after v1). Live run shows 10 siblings admitted out
  of the candidate pool (down from v1's uncapped-priority 20), and a
  dedicated test proves the actual selectivity property: an on-topic
  same-turn candidate (score 0.645, clears a 0.48 threshold) is kept; an
  off-topic same-turn candidate (score 0.145, same anchor, same threshold)
  is rejected — the property a flat pull-everything or a flat `ORDER BY`
  cannot express, since both would keep or drop them identically regardless
  of relevance. New config: `RETRIEVAL_SIBLING_FACT_LIMIT` (default 20, 0
  disables), `RETRIEVAL_SIBLING_CONTINUITY_PENALTY` (λ, default 0.1),
  `RETRIEVAL_SIBLING_RELEVANCE_RATIO` (γ, default 0.80).

- **Cutoff-boundary diagnostic logging.** Every retrieval now logs the
  last-included vs. first-excluded fact and their scores at the top-K
  cutoff. Every traced miss so far had the same shape (a topically-irrelevant
  fact with a coincidentally high score crowding out the relevant one); this
  makes that visible in normal logs going forward instead of requiring a
  one-off diagnostic script re-run by hand each time.

### 1.3 Reader prompt

`READER_SYSTEM_PROMPT_TEMPLATE` (`core/config.py`) gained two targeted
instructions, not a general rewrite:

- If the question needs a total, count, or duration spanning multiple facts,
  identify every matching fact first, then compute from all of them — do not
  answer from a partial subset or return a single fact's value directly when
  the question asks for a combination of several.
- If the question is a recommendation ask and the context states a specific
  relevant prior preference, the answer must build on that specific
  preference, not just stay on the same general topic.

Both target the exact two zero-score categories from §2 below. Not yet
verified against a live re-run (explicitly deferred at the user's request —
implemented but not tested).

### 1.4 Latency (all read-only / no correctness risk, verified via 149
unit tests + one live end-to-end call reproducing the correct answer)

- **Phase 0 parallelized.** Temporal resolution and query rewriting are two
  independent LLM calls (the rewriter doesn't use `temporal_bounds`, the
  resolver doesn't use `expanded_query`) that were running sequentially. Now
  concurrent via a 2-worker thread pool. Measured with real LLM calls,
  isolated from the rest of the pipeline: **3.36s sequential → 0.85s
  concurrent (75% cut, 2.51s saved)**.
- **Phase 2's N+1 parallelized.** HydraDB rejects UNWIND-batched reads (see
  `retrieval.py`'s own extensive comments on this, discovered through four
  rounds of live 400 errors earlier in this project), forcing one HTTP
  round trip per seeded fact — 60-80+ typical given the overfetch floor.
  Verified `HydraHttpTransport` holds no shared mutable per-call state
  (headers/URL fixed at construction, each `.read()` self-contained), so
  this is safe to parallelize, unlike ingestion's writes, which share one
  psycopg connection and genuinely cannot. Now concurrent, 8 workers by
  default (`RETRIEVAL_GRAPH_FETCH_WORKERS`). Measured against 70 real fact
  IDs from an already-ingested instance: **0.57s sequential (8ms/call) →
  0.06s concurrent (89% cut, 0.50s saved)**.
- **Dead SUPERSEDES probe already removed.** An earlier session had flagged
  a per-fact SUPERSEDES-traversal loop whose result was computed but never
  consumed — pure wasted latency. Checked before doing anything further: it
  is already gone from the codebase (removed in an earlier commit), so no
  action was needed here.
- **Caught mid-fix:** parallelizing Phase 0 broke an unstated assumption in
  the test fakes — `FakeLLMClient` indexed queued responses by call order,
  which stops being safe once two calls genuinely race. It passed once by
  GIL timing luck, not correctness. Fixed the fake to dispatch by the
  requested response's type instead of call order; confirmed deterministic
  across 5 repeated runs.

**Honest scope of these numbers, checked before claiming a benchmark-level
win:** both fixes are real, measured, and correct — but they only touch
`retrieve_and_answer`, and retrieval is a small fraction of total per-instance
wall time. From the 30-instance run's own per-instance ingest/retrieve split:

```
ingest total (30 instances):   24,603.2s  (99.7% of measured phase time)
retrieve total (30 instances):     85.9s  ( 0.3%)
```

Ingestion — serial per-turn LLM extraction calls, ~500 turns/instance — is
what actually dominates a LongMemEval run, and nothing in this latency pass
touched it. A 75-89% cut on 0.3% of total time is real but does not
meaningfully change how long a full run takes; see §2.7 for what an actual
re-run would cost.

---

## 2. Evaluation findings: the 30-instance run

### 2.1 Setup

Stratified sample, 5 instances per category, fixed random seed (reproducible)
drawn from all six LongMemEval-S question types. Scored with the official
grader (`xiaowu0162/LongMemEval`'s `src/evaluation/evaluate_qa.py`), judge
model `deepseek.v3.2` (chosen after `gpt-5.4`/`gpt-5.5` returned real 401s on
this account, Claude 400'd on this endpoint's request shape, and
`gpt-oss-120b` hit the same reasoning-exhaustion bug as our extractor —
confirmed live, not assumed).

### 2.2 Data legitimacy — checked before looking anywhere else

For every one of the 10 failing instances: **turn count in the source data
== ingestion-job count == completed-job count, exactly**, and each has
2,700-2,950 real extracted facts indexed (not empty or stub output). Not
"completed for the sake of being completed" — ingestion is not where these
two categories fail.

### 2.3 Full results

| category | accuracy | n |
|---|---:|---:|
| single-session-user | 80.0% | 5 |
| single-session-assistant | 60.0% | 5 |
| knowledge-update | 60.0% | 5 |
| temporal-reasoning | 40.0% | 5 |
| single-session-preference | 0.0% | 5 |
| multi-session | 0.0% | 5 |
| **overall** | **40.0%** | **30** |

An earlier partial read (19/30, before `single-session-user` and
`temporal-reasoning` had finished ingesting) showed 31.6% — the missing
categories were pulling the average down by their absence, not by scoring
badly.

### 2.4 `multi-session` (0/5) — root cause

Every one of the 5 instances requires *deriving* an answer from two or more
separate facts (summing amounts, counting distinct events, or subtracting
two ages). The reader has no structured step for this and fails in one of
two shapes:

- **(a) Computes from a partial subset that survived ranking.** Total
  market-earnings question: three facts contribute to the $495 gold total
  ($225 + $150 + $120). All three were ingested. Two ($225, $120) reached
  the reader directly. The third needs `20 potted herb plants × $7.5/plant
  = $150`; the per-unit-price fact ranked just outside the top-15 the reader
  saw, crowded out by unrelated dollar-amount facts (real-estate prices,
  gross income) that scored higher on surface similarity to "money." The
  reader answered $345 from the two facts it had — internally consistent,
  externally wrong.
- **(b) Returns a single raw fact directly instead of attempting the
  derivation.** "How old was I when Alex was born?" (gold: 11, from
  32 − 21). Both ages ("Alex is 21 years old", "The user is 32 years old")
  were ingested facts. The reader answered "21" — Alex's own stated age,
  echoed back verbatim, with no subtraction attempted. (A first hypothesis
  here — that there were two different people both named Alex and the
  system picked the wrong one — was checked against the actual source
  transcript and found to be false: there is only one Alex in the haystack,
  in the exact designated answer session. Corrected before reporting.)

One instance (`ef9cf60a`, "How much did I spend on gifts for my sister?",
gold $300 = $200 necklace + $100 spa gift card, both facts ingested and both
individually ranking in the top 2 by raw semantic score) did **not**
reproduce identically on rerun against the same static data — the original
run answered "I don't have enough information," a fresh rerun answered
"$100." Both wrong, differently wrong, on identical inputs — a genuine
reader-side non-determinism component stacking on top of the structural gap,
not purely a retrieval defect.

### 2.5 `single-session-preference` (0/5) — root cause

Graded against a rubric, not exact-match, so a partially-relevant response
can still fail. Two distinct shapes traced:

- **The specific preference fact never reaches the reader.** "What should I
  serve for dinner this weekend with my homegrown ingredients?" — gold wants
  the answer to reference the user's homegrown cherry tomatoes and herbs.
  The fact "The user has harvested cherry tomatoes from their garden." was
  ingested but never made the top-60 seeded candidate pool at all (a Phase 1
  embedding-recall miss, not a ranking-cutoff miss) — no amount of
  downstream re-ranking could have recovered it.
- **The fact is available and the reader answers generically anyway.** A
  coffee-creamer recommendation question's gold rubric wanted variations on
  an *existing* stated recipe (almond milk, vanilla, honey) plus cost/sugar
  reduction goals; the reader produced an unrelated new recipe (rose petal,
  lavender, collagen peptides, MCT oil) that never engaged with the specific
  prior preference at all.
- A third case (rearranging bedroom furniture) technically mentioned the
  right keywords (the dresser, mid-century modern style) but the response
  was mostly about Wi-Fi signal strength, with furniture placement framed
  around it — off-topic content diluting an otherwise on-topic answer well
  past what the rubric wanted.

### 2.6 What's fixed vs. still open, honestly

**Fixed and verified this pass, with live before/after numbers:** the
entity-boost query-gating bug (commute fact: rank #39/113, excluded →
#4/138, included), missing common-noun entity coverage (25.7% of facts with
no entity → 0% on a fresh live re-test), the speaker/entity namespace
collision (36 conflicts on old contexts → 0 on 30 fresh ones), RRF composite
scoring (proven via test, not just claimed), sibling-fact expansion —
upgraded to SCAR after v1 was re-tested and found not to have actually fixed
its own motivating case (see §1.2) — the exact case it was built for now
answers $495 exactly, both parallelization fixes (75% and 89% cuts,
measured).

**Implemented but not yet live-verified:** the reader prompt's aggregation
and preference-fidelity instructions (explicitly deferred per instruction —
implemented, not re-run against live data yet).

**Still open, not yet attempted:**

- Phase 1 embedding recall can still miss a topically-narrow fact entirely
  (the tomato/garden case) — RRF, sibling expansion, and SCAR's continuity
  scoring all operate on facts that were at least seeded; a fact that never
  enters the candidate pool at all, or was never extracted onto a turn SCAR
  can reach, is unaffected by any of them.
- The reader-side non-determinism observed on `ef9cf60a` (same inputs,
  different wrong answers across runs) has not been investigated further —
  worth knowing if reproducibility matters for future evaluation runs.
- SCAR's λ/γ are the paper's published defaults, unvalidated against this
  system's own data — no tuning pass has been run.

### 2.7 Cost of running this again

**A full re-run needs fresh ingestion, not just re-running retrieval against
the already-ingested contexts.** The entity-coverage prompt fix changes what
extraction produces; the 30 contexts already ingested reflect the *old*
prompt. Re-scoring against old data would not exercise most of what changed
this pass.

**Ingestion, not retrieval, sets the clock — and nothing in this pass
touched ingestion.** From the measured 30-instance split in §1.4: ingestion
was 99.7% of wall time, retrieval 0.3%. The 75-89% retrieval speedups
translate to roughly **0.3% × 0.8 ≈ 0.25% off total wall time** — real, but
not the number that matters for planning a re-run.

**Estimate, from the actual measured run, not a projection:**

```
measured: 30 instances = 27,693s = 7.7h   (923s/instance average)
```

| scope | estimate |
|---|---|
| same 30-instance stratified sample, re-run fresh | **~7.5-7.7h**, materially unchanged from the original run |
| full 500-instance LongMemEval-S | **~128h ≈ 5.3 days**, linear extrapolation from the measured 30 |

Both are the *ingestion* cost — extraction is a real per-turn LLM call
(~500 turns/instance average), run mostly-concurrently within an instance
via the existing prefetch mechanism but still bounded by provider
throughput; nothing added this pass changes that arithmetic. If ingestion
speed itself needs to come down for a re-run to be practical, that is a
separate, not-yet-investigated piece of work — this pass's scope was
retrieval quality and retrieval latency, not extraction throughput.
