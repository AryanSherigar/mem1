"""Single source of truth for every tunable value in the system: model
allocation per role, LLM call limits, retrieval scoring/thresholds, ingestion
concurrency, and every system prompt. Nothing below is read from `os.environ`
anywhere else in the codebase — if a value needs to be dialed, it's a field
here, not a literal buried in `retrieval.py`/`model_adapters.py`/`engine.py`.

Deliberately does NOT auto-load `.env` on import. `main` branch's version of
this module did (`load_dotenv()` at module scope) — that is an import-time
side effect: merely importing this module (e.g. `unittest discover` importing
every test file to enumerate cases, including a live-gated one it never runs)
silently repopulates `FIREWORKS_API_KEY` into the process environment from
disk, defeating any `skipUnless(os.environ.get(...))` gate and risking a real
billed call from a plain test run. Same posture `test_hydradb_live.py` already
takes for `CONTEXT_MEMORY_HYDRADB_TOKEN`: credentials come from the runtime
environment the caller explicitly set up (`source src/.env`, an exported var,
or a deploy-time secret), never from an implicit file read triggered by import.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from context_memory.core.llm_client import LLMClient

# Default provider. Groq, measured against Fireworks on this exact workload:
# ~0.95s warm per extraction call vs 0.8s, but Fireworks' `deepseek-v4-flash`
# cold-started at 226s for an 11-token prompt, and — decisively — returned
# EMPTY content whenever its completion budget was exhausted by hidden
# reasoning (`finish_reason=length`, `content=''`), which is a systematic
# defect of that model, not an occasional blip. Groq's gpt-oss-120b returned
# valid JSON on the first attempt every time. Any OpenAI-compatible endpoint
# still works — set LLM_BASE_URL/LLM_API_KEY/LLM_MODEL (or the per-role
# overrides below) to point anywhere.
_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
# qwen3.6-27b is the one model on this endpoint that accepts
# `reasoning_effort="none"` — genuinely no hidden reasoning. Measured against
# gpt-oss-120b on the same extraction prompt: 0.17s vs 0.50-0.76s, and 36
# completion tokens vs 160-330 (the rest being reasoning the caller never sees
# but pays for in both latency and budget). It also fixes a correctness bug,
# not just speed: the same call WITHOUT reasoning disabled returns HTTP 400
# "Failed to validate JSON" because reasoning text corrupts the JSON body.
# gpt-oss models only accept low/medium/high, so reasoning cannot be turned off
# there at all — which is what produced the empty-content-at-budget-exhaustion
# failures seen on both Fireworks' deepseek and Groq's gpt-oss.
_DEFAULT_MODEL = "qwen/qwen3.6-27b"


def _role_base_url(env_name: str) -> str:
    """Per-role override, else generic LLM_BASE_URL, else the legacy
    FIREWORKS_BASE_URL (kept so existing .env files keep working), else default."""
    return os.getenv(
        env_name,
        os.getenv("LLM_BASE_URL", os.getenv("FIREWORKS_BASE_URL", _DEFAULT_BASE_URL)),
    )


def _role_api_key(env_name: str) -> str:
    return os.getenv(
        env_name,
        os.getenv("LLM_API_KEY", os.getenv("GROQ_API_KEY", os.getenv("FIREWORKS_API_KEY", ""))),
    )


def _role_model(env_name: str) -> str:
    return os.getenv(env_name, os.getenv("LLM_MODEL", _DEFAULT_MODEL))


# ---------------------------------------------------------------------------
# Prompts. Every system prompt used anywhere in the pipeline lives here, not
# as a module-level constant next to the code that calls it.
# ---------------------------------------------------------------------------

ENTITY_RESOLUTION_SYSTEM_PROMPT = (
    "You resolve an entity surface form (a name, nickname, or pronoun-adjacent mention) to "
    "exactly one of a fixed, already-bounded candidate list, or to none if no candidate is "
    "the same real-world entity. Treat aliases and partial names (e.g. a first name matching "
    "a candidate's alias list) as potential matches, but never invent a candidate id that is "
    "not in the supplied list. When two candidates are both plausible, or no candidate is a "
    "confident match, prefer 'null' over a low-confidence guess — a missed link is cheaper "
    "than a wrong merge."
)

TEMPORAL_UPDATE_SYSTEM_PROMPT = (
    "You classify how a new fact relates to a prior fact about the same subject and "
    "predicate, observed later in time. 'correction' means the prior fact was wrong "
    "and is being fixed (e.g. 'my dog's name is Max, not Mac'); the real-world state "
    "never changed. 'state_change' means the real world changed and the prior fact was "
    "true until now (e.g. 'I moved to Seattle' after a prior 'I live in Austin'). "
    "'no_update' means the new fact does not actually supersede the prior one — they can "
    "both stay true (e.g. two different pets, not a renaming of the same one). Use "
    "'unresolved' only if the text truly gives no clear basis to decide; do not default "
    "to it just because the distinction is subtle."
)

# Deliberately terse. This is sent on EVERY extraction call, and measured on a
# real LongMemEval instance the fixed overhead (this prompt + the JSON schema)
# was 2237 chars against a mean turn content of only 882 — 72% of each request
# was boilerplate, which at a tokens-per-minute rate limit directly caps
# throughput. Field-by-field descriptions were removed because the JSON schema
# already declares every field, its type, and its enum values; only guidance
# the schema *cannot* express is kept here.
FACT_EXTRACTION_SYSTEM_PROMPT = (
    "Extract atomic, enduring facts about the user or assistant from the dialogue turn. "
    "Skip chitchat, pleasantries, and anything true only within this conversation. "
    "Each fact must stand alone: resolve pronouns, and split compound statements into separate facts. "
    "predicate_key is a snake_case category (e.g. pet_name, location, hobby). "
    # "named entities" alone was read strictly as proper nouns, so any fact
    # about a common-noun topic got an empty entity list -- 691 of 2685 facts
    # (25.7%) on a real LongMemEval instance, including every "commute" fact
    # for a question whose gold answer was about the user's commute. Entities
    # are what graph expansion traverses and what the query-gated entity boost
    # keys on, so an unlinked fact is invisible to both.
    "entities lists the key subjects of the fact: named entities (people, places, "
    "products) AND the salient topic nouns (e.g. commute, audiobooks, rent). "
    "exact_quote is the verbatim substring evidencing the fact. "
    "confidence: >0.9 plain assertions, <0.6 hedged or implied. "
    "action: ADD, or UPDATE/DELETE if it changes or invalidates an earlier fact. "
    "No durable facts means an empty facts list — do not invent one. "
    "Return only the JSON object."
)

# {question_date} is substituted at call time (current reference time for the question).
TEMPORAL_RESOLVER_SYSTEM_PROMPT_TEMPLATE = (
    "You are a temporal query resolver. Current reference time is {question_date} (UTC). "
    "Analyze the question and extract the world-validity time range [valid_from, valid_to] "
    "only if it references a specific time, explicitly (a date) or relatively (e.g. 'yesterday', "
    "'last month', 'this summer') — resolve relative expressions against the current reference "
    "time above. If the question has no temporal anchor at all, return null for both rather than "
    "guessing a range."
)

QUERY_REWRITER_SYSTEM_PROMPT = (
    "You are a query expansion assistant for a fact-retrieval system. Decompose complex or "
    "multi-part questions into atomic sub-questions, one per distinct piece of information "
    "needed — leave simple, already-atomic questions as a single item rather than splitting "
    "for its own sake. Extract a short list of key entity/predicate synonyms (e.g. 'dog' -> "
    "'pet', 'puppy') to maximize keyword recall; skip synonyms for terms that are already "
    "unambiguous. Prefer fewer, higher-value items over an exhaustive list."
)

# {context} is substituted at call time (the assembled, ranked evidence block).
# Two additions here (aggregation, preference-fidelity) target a specific,
# diagnosed failure pattern, not a general rewrite. Traced against real
# retrieved context on the 30-instance LongMemEval sample:
#
# - multi-session questions ("total I earned", "how many pieces of furniture",
#   "how old was I when X was born") need combining 2+ facts by sum, count, or
#   subtraction. Without being told to do this explicitly, the reader either
#   answered from whichever partial subset of facts it had (internally
#   consistent, silently wrong) or echoed a single raw fact back instead of
#   computing what was asked.
# - single-session-preference questions are graded against a rubric that
#   wants the answer anchored to a specific prior-stated preference. Without
#   being told to check for one, the reader sometimes gave a generic,
#   same-topic-but-unrelated answer even when the specific preference fact
#   was present in context.
READER_SYSTEM_PROMPT_TEMPLATE = (
    "You are a memory-grounded assistant. Answer the user's question using only the facts "
    "in the memory context below — do not use outside knowledge or assumptions beyond what "
    "is stated. Each fact is timestamped; if facts conflict, trust the most recent one. If "
    "the question asks for a total, count, or duration that spans multiple facts, first "
    "identify every matching fact in the context, then compute the answer from all of them "
    "— do not answer from a partial subset, and do not return a single fact's value directly "
    "when the question asks for a combination of several. If the question is asking for a "
    "recommendation and the context states a specific relevant preference (a prior choice, "
    "style, constraint, or thing they already liked), your answer must build on that specific "
    "preference, not just stay on the same general topic. If the context does not contain "
    "enough information to answer, say so plainly instead of guessing. Answer directly and "
    "concisely, in a natural conversational tone — do not restate the context verbatim or "
    "mention that you were given a context.\n\n"
    "Memory context:\n{context}"
)

# Base prompt for the alternate context-aware extractor (`LLMExtractionService`,
# `ingestion/extraction.py`). Recent-turns/candidate-facts context is appended
# by the caller at call time — that's assembled content, not a tunable, so it
# stays inline there.
LLM_EXTRACTION_SERVICE_SYSTEM_PROMPT = (
    "You are an expert extraction AI. Extract atomic facts from the current turn, each one "
    "understandable without the rest of the conversation.\n"
    "Identify the appropriate action (ADD, UPDATE, DELETE) — use UPDATE/DELETE only when the "
    "turn clearly changes or invalidates one of the existing candidate facts listed below.\n"
    "If updating an existing state or property, specify a predicate_key (e.g. 'pet_breed').\n"
    "Respond with the JSON object only — no reasoning or commentary before it.\n"
)


@dataclass(frozen=True)
class Config:
    # -- Ingestion/resolution roles (ADR-027/029) --------------------------
    entity_resolution_base_url: str = field(default_factory=lambda: _role_base_url("ENTITY_RESOLUTION_BASE_URL"))
    entity_resolution_api_key: str = field(default_factory=lambda: _role_api_key("ENTITY_RESOLUTION_API_KEY"))
    entity_resolution_model: str = field(default_factory=lambda: _role_model("ENTITY_RESOLUTION_MODEL"))

    temporal_update_base_url: str = field(default_factory=lambda: _role_base_url("TEMPORAL_UPDATE_BASE_URL"))
    temporal_update_api_key: str = field(default_factory=lambda: _role_api_key("TEMPORAL_UPDATE_API_KEY"))
    temporal_update_model: str = field(default_factory=lambda: _role_model("TEMPORAL_UPDATE_MODEL"))

    extractor_base_url: str = field(default_factory=lambda: _role_base_url("EXTRACTOR_BASE_URL"))
    extractor_api_key: str = field(default_factory=lambda: _role_api_key("EXTRACTOR_API_KEY"))
    extractor_model: str = field(default_factory=lambda: _role_model("EXTRACTOR_MODEL"))

    # -- Retrieval roles ------------------------------------------------
    reader_base_url: str = field(default_factory=lambda: _role_base_url("READER_BASE_URL"))
    reader_api_key: str = field(default_factory=lambda: _role_api_key("READER_API_KEY"))
    reader_model: str = field(default_factory=lambda: _role_model("READER_MODEL"))

    temporal_resolver_base_url: str = field(default_factory=lambda: _role_base_url("TEMPORAL_RESOLVER_BASE_URL"))
    temporal_resolver_api_key: str = field(default_factory=lambda: _role_api_key("TEMPORAL_RESOLVER_API_KEY"))
    temporal_resolver_model: str = field(default_factory=lambda: _role_model("TEMPORAL_RESOLVER_MODEL"))

    query_rewriter_base_url: str = field(default_factory=lambda: _role_base_url("QUERY_REWRITER_BASE_URL"))
    query_rewriter_api_key: str = field(default_factory=lambda: _role_api_key("QUERY_REWRITER_API_KEY"))
    query_rewriter_model: str = field(default_factory=lambda: _role_model("QUERY_REWRITER_MODEL"))

    # -- LLM call limits, sized per role ------------------------------------
    # A 241-second call that returned an empty completion (hit its own output
    # ceiling before producing valid JSON) is what an unbounded call actually
    # does — confirmed live, not a theoretical risk. Every role gets a cap.
    # One shared cap isn't right either: extraction reasons over a whole turn
    # before emitting JSON and was the one role that hit 2048 and came back
    # empty (confirmed live), while classification/selection roles emit a few
    # words of JSON and gain nothing from a large ceiling but pay for one in
    # provider-side worst-case latency. Every role below is sized to its own
    # output shape rather than sharing one number.
    llm_temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.0")))
    # Confirmed live (LongMemEval pilot): a reasoning model can spend its whole
    # completion budget on hidden reasoning and return empty content — the cap
    # bounds worst-case latency but doesn't prevent this. One retry (with a
    # blunter prompt and, only on a genuine budget-exhaustion `finish_reason`,
    # a doubled budget) recovers most of these instead of silently losing a
    # turn's facts. See `LLMClient.structured_completion`.
    llm_structured_retry_attempts: int = field(default_factory=lambda: int(os.getenv("LLM_STRUCTURED_RETRY_ATTEMPTS", "1")))
    # The OpenAI SDK's own retry count. Its default of 2 multiplies every timeout
    # (45s cap became 135s of real wall clock — measured). Kept at 1 rather than 0
    # because Groq rate-limits on tokens-per-minute and a 429 deserves one retry.
    llm_sdk_max_retries: int = field(default_factory=lambda: int(os.getenv("LLM_SDK_MAX_RETRIES", "1")))
    # Separate from both retry counts above: a 429 raises out of `.create()`
    # rather than returning unparseable content, so neither of them ever saw it
    # and rate-limited calls failed outright, silently dropping a turn's facts
    # (confirmed live against Groq's 8k TPM free tier). Higher than the others
    # because a tokens-per-minute limiter legitimately needs several waits in a
    # row under concurrent prefetch; each wait honors the provider's own stated
    # retry-after, so this is mostly-idle time, not repeated load.
    llm_rate_limit_max_retries: int = field(default_factory=lambda: int(os.getenv("LLM_RATE_LIMIT_MAX_RETRIES", "5")))
    # How the response shape is described to the model: "typedef" (compact
    # TS/BAML-style class declaration) or "json_schema" (pydantic's own output).
    # Measured on the extraction schema: 839 chars as raw JSON Schema, 479 with
    # noise keys stripped, 225 as a typedef. That blob rides on every single
    # call, so under a tokens-per-minute limit it is throughput, not cosmetics.
    # Groq reports no prompt caching (verified: identical prompts billed at full
    # price twice, no `cached_tokens` field), so nothing amortizes this for us.
    llm_schema_format: str = field(default_factory=lambda: os.getenv("LLM_SCHEMA_FORMAT", "typedef"))
    # Sent only when non-empty, because valid values are model-specific
    # (qwen3.6: none|default — gpt-oss: low|medium|high) and an unsupported
    # value is a hard 400. Empty string omits the parameter entirely, which is
    # the right behavior for any provider/model that doesn't know it.
    llm_reasoning_effort: str = field(default_factory=lambda: os.getenv("LLM_REASONING_EFFORT", "none"))

    entity_resolution_max_tokens: int = field(default_factory=lambda: int(os.getenv("ENTITY_RESOLUTION_MAX_TOKENS", "512")))
    entity_resolution_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("ENTITY_RESOLUTION_TIMEOUT_SECONDS", "20")))

    temporal_update_max_tokens: int = field(default_factory=lambda: int(os.getenv("TEMPORAL_UPDATE_MAX_TOKENS", "512")))
    temporal_update_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("TEMPORAL_UPDATE_TIMEOUT_SECONDS", "20")))

    # Extraction reasons over the whole turn before emitting JSON — this is the
    # role that actually hit the old 2048 shared cap and came back empty/truncated
    # (confirmed live). Given more headroom and more time to use it.
    extractor_max_tokens: int = field(default_factory=lambda: int(os.getenv("EXTRACTOR_MAX_TOKENS", "4096")))
    extractor_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("EXTRACTOR_TIMEOUT_SECONDS", "45")))

    # Reader writes a full prose answer, so gets more room than a structured-json
    # role, and a bit of warmth (0.0 elsewhere is deliberate for determinism on
    # classification/extraction; a synthesized answer reads better slightly above
    # frozen-greedy without materially risking groundedness given the prompt's
    # context-only instruction).
    reader_temperature: float = field(default_factory=lambda: float(os.getenv("READER_TEMPERATURE", "0.2")))
    reader_max_tokens: int = field(default_factory=lambda: int(os.getenv("READER_MAX_TOKENS", "1536")))
    reader_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("READER_TIMEOUT_SECONDS", "25")))

    temporal_resolver_max_tokens: int = field(default_factory=lambda: int(os.getenv("TEMPORAL_RESOLVER_MAX_TOKENS", "512")))
    temporal_resolver_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("TEMPORAL_RESOLVER_TIMEOUT_SECONDS", "15")))

    query_rewriter_max_tokens: int = field(default_factory=lambda: int(os.getenv("QUERY_REWRITER_MAX_TOKENS", "768")))
    query_rewriter_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("QUERY_REWRITER_TIMEOUT_SECONDS", "15")))

    # -- Embedding (Milestone 7) ------------------------------------------
    embedding_model_name: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    )
    embedding_model_version: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL_VERSION", "1"))

    # -- Retrieval tuning (FINAL_ARCHITECTURE.md §12) ----------------------
    # 15 -> 20. Every traced multi-session/preference retrieval miss on the
    # 30-instance LongMemEval sample had the same shape: the needed fact was
    # seeded and scored, just outside this cutoff, crowded out by a fact that
    # matched the question's wording (e.g. any dollar figure for a "how much"
    # question) without matching its actual topic. This doesn't fix the
    # underlying ranking-relevance gap (see retrieval.phase3's cutoff-boundary
    # log line, added alongside this change, for that diagnosis going
    # forward) -- it's a cheap, low-risk mitigation: a wider reader window
    # gives near-miss facts more room to still get seen while the real ranking
    # fix (surface-similarity vs topical-relevance) is designed properly.
    retrieval_top_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "20")))
    # Neighboring-turn expansion (ADR-005's accepted-but-never-implemented
    # "fact + span -> neighboring turn -> full chunk" tier). Traced live: a
    # question needing "20 potted herb plants" x "$7.5 each" got the price but
    # not the quantity, because the extraction prompt splits one turn into
    # several atomic facts and Phase 3 scores them independently -- pulling in
    # every other fact from the same source turn as an already-relevant one
    # re-unites what extraction split apart. 0 disables it entirely.
    retrieval_sibling_fact_limit: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_SIBLING_FACT_LIMIT", "20")))
    # SCAR (Semantic Continuity-Aware Retrieval; Zhong et al. 2026,
    # arxiv.org/abs/2606.16661) scores which siblings actually earn a spot,
    # rather than an unordered LIMIT -- confirmed live this mattered: a needed
    # sibling (4 candidates for its own turn) lost out to unrelated turns'
    # siblings under a shared, unordered cap. lambda penalizes a candidate for
    # being semantically distant from the fact that pulled it in; gamma sets
    # the bar a candidate must clear *relative to its own anchor's* query
    # relevance (a weak anchor -> a low bar; a strong anchor -> a high one).
    # Paper defaults, unchanged -- no tuning data of our own exists yet.
    retrieval_sibling_continuity_penalty: float = field(default_factory=lambda: float(os.getenv("RETRIEVAL_SIBLING_CONTINUITY_PENALTY", "0.1")))
    retrieval_sibling_relevance_ratio: float = field(default_factory=lambda: float(os.getenv("RETRIEVAL_SIBLING_RELEVANCE_RATIO", "0.80")))
    retrieval_overfetch_multiplier: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_OVERFETCH_MULTIPLIER", "4")))
    retrieval_overfetch_floor: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_OVERFETCH_FLOOR", "60")))
    retrieval_chat_ttl_hours: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_CHAT_TTL_HOURS", "24")))
    retrieval_temporal_buffer_days: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_TEMPORAL_BUFFER_DAYS", "2")))
    # Session-[:HAS_TURN]->Turn-[:EXTRACTED_FROM]->Fact-[:ABOUT]->Entity is the
    # deepest real path in the schema; MSpaths hops fact-to-fact via a shared
    # Entity. 4 hops routinely walks past directly-relevant facts into
    # loosely-associated ones, and each hop is a real per-fact HydraDB round
    # trip (no batched UNWIND reads — see retrieval.py), so it's a latency cost
    # for mostly-noise recall. 3 still reaches one shared-entity hop past the
    # seed set.
    retrieval_graph_max_hops: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_GRAPH_MAX_HOPS", "3")))
    # Phase 2 fetches one fact's graph node per HydraDB HTTP call (a genuine
    # N+1 -- HydraDB rejects UNWIND-batched reads, see retrieval.py's own
    # comments), so a retrieval with the overfetch floor's ~60-80 seeded facts
    # was ~60-80 sequential round trips. Concurrent, since each call is
    # independent/read-only and the local HydraDB HTTP transport holds no
    # shared per-call state. 8 matches the same default already used for
    # concurrent extraction prefetch in benchmark_runner.py.
    retrieval_graph_fetch_workers: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_GRAPH_FETCH_WORKERS", "8")))
    retrieval_structural_path_cap: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_STRUCTURAL_PATH_CAP", "3")))
    retrieval_entity_boost_cap: float = field(default_factory=lambda: float(os.getenv("RETRIEVAL_ENTITY_BOOST_CAP", "0.5")))
    # Composite scoring fuses semantic/keyword/structural/entity by Reciprocal
    # Rank Fusion (rank position within each channel, not raw score value) --
    # replaced a raw weighted sum of the four differently-scaled scores, which
    # let whichever one happened to read numerically "big" for a fact dominate
    # regardless of actual relevance (confirmed live and repeatedly on
    # LongMemEval traces: a topically-irrelevant fact with a coincidentally
    # high semantic/entity score routinely outranked the fact that actually
    # answered the question). 60 is RRF's standard constant (Cormack et al.
    # 2009); lower values weight top-ranked-in-any-single-channel facts more
    # heavily, higher values flatten the fusion toward facts that rank
    # decently across several channels rather than winning any one of them.
    retrieval_rrf_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_RRF_K", "60")))
    retrieval_abstention_semantic_threshold: float = field(
        default_factory=lambda: float(os.getenv("RETRIEVAL_ABSTENTION_SEMANTIC_THRESHOLD", "0.3"))
    )
    retrieval_abstention_message: str = field(
        default_factory=lambda: os.getenv("RETRIEVAL_ABSTENTION_MESSAGE", "I don't have that information in my memory.")
    )

    # -- Ingestion concurrency ---------------------------------------------
    ingestion_executor_max_workers: int = field(default_factory=lambda: int(os.getenv("INGESTION_EXECUTOR_MAX_WORKERS", "1")))

    # -- Storage / transport connection settings ---------------------------
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "CONTEXT_MEMORY_DATABASE_URL",
            os.getenv("DATABASE_URL", "postgresql://context_memory@127.0.0.1:54329/context_memory"),
        )
    )
    hydradb_url: str = field(
        default_factory=lambda: os.getenv(
            "CONTEXT_MEMORY_HYDRADB_URL", os.getenv("HYDRA_DB_HOST", "http://127.0.0.1:8080")
        )
    )
    hydradb_token: str = field(
        default_factory=lambda: os.getenv(
            "CONTEXT_MEMORY_HYDRADB_TOKEN",
            os.getenv("HYDRA_DB_API_KEY", "context-memory-local-smoke-token-32b"),
        )
    )
    hydradb_database: str = field(
        default_factory=lambda: os.getenv("CONTEXT_MEMORY_HYDRADB_DATABASE", os.getenv("HYDRA_DB_GRAPH_ID", "default"))
    )
    hydradb_request_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("HYDRADB_REQUEST_TIMEOUT_SECONDS", "15"))
    )

    # -- Prompts (module-level strings above; fields here so a caller can
    #    override per-instance without editing source) ----------------------
    entity_resolution_system_prompt: str = ENTITY_RESOLUTION_SYSTEM_PROMPT
    temporal_update_system_prompt: str = TEMPORAL_UPDATE_SYSTEM_PROMPT
    fact_extraction_system_prompt: str = FACT_EXTRACTION_SYSTEM_PROMPT
    temporal_resolver_system_prompt_template: str = TEMPORAL_RESOLVER_SYSTEM_PROMPT_TEMPLATE
    query_rewriter_system_prompt: str = QUERY_REWRITER_SYSTEM_PROMPT
    reader_system_prompt_template: str = READER_SYSTEM_PROMPT_TEMPLATE
    llm_extraction_service_system_prompt: str = LLM_EXTRACTION_SERVICE_SYSTEM_PROMPT

    def _client(self, base_url: str, api_key: str, model: str) -> LLMClient:
        return LLMClient(
            base_url=base_url, api_key=api_key, model_name=model,
            sdk_max_retries=self.llm_sdk_max_retries,
            reasoning_effort=self.llm_reasoning_effort,
            rate_limit_max_retries=self.llm_rate_limit_max_retries,
            schema_format=self.llm_schema_format,
        )

    def get_entity_resolution_client(self) -> LLMClient:
        return self._client(self.entity_resolution_base_url, self.entity_resolution_api_key, self.entity_resolution_model)

    def get_temporal_update_client(self) -> LLMClient:
        return self._client(self.temporal_update_base_url, self.temporal_update_api_key, self.temporal_update_model)

    def get_extractor_client(self) -> LLMClient:
        return self._client(self.extractor_base_url, self.extractor_api_key, self.extractor_model)

    def get_reader_client(self) -> LLMClient:
        return self._client(self.reader_base_url, self.reader_api_key, self.reader_model)

    def get_temporal_resolver_client(self) -> LLMClient:
        return self._client(self.temporal_resolver_base_url, self.temporal_resolver_api_key, self.temporal_resolver_model)

    def get_query_rewriter_client(self) -> LLMClient:
        return self._client(self.query_rewriter_base_url, self.query_rewriter_api_key, self.query_rewriter_model)
