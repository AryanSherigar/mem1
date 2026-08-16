# LLM Model Selection & Concurrency Configuration

Follow-up to [graph_schema_proposal.md](file:///home/aryan-sherigar/projects/hydradb-hackathon/docs/graph_schema_proposal.md), [retrieval_architecture.md](file:///home/aryan-sherigar/projects/hydradb-hackathon/docs/retrieval_architecture.md), [semantic_memory_distillation.md](file:///home/aryan-sherigar/projects/hydradb-hackathon/docs/semantic_memory_distillation.md), and [temporal_query_resolver.md](file:///home/aryan-sherigar/projects/hydradb-hackathon/docs/temporal_query_resolver.md).

---

## 1. Overview & Constraints

The system requires LLM capabilities across three distinct roles:
1. **Extractor LLM (Ingestion)**: High call volume (~400 turns $\times$ structured JSON extraction + entity resolution).
2. **Reader LLM (Retrieval QA)**: 500 questions $\times$ reasoning over dated facts and synthesis.
3. **Judge LLM (Evaluation)**: Scoring predictions against gold answers via LongMemEval's judge prompt.

### Free Tier & Credit Optimization Strategy

| Role | Recommended Model & Provider | Why Selected | Cost / Free Limit |
|---|---|---|---|
| **Extractor** | **`gpt-oss-20b`** (Fireworks / OpenRouter / vLLM) | Ultra-fast structured JSON extraction; 20B parameters capture implicit and nuanced facts without 70B latency | **Free / Starter Credits** (Fraction of a cent per session) |
| **Reader** | **`gpt-oss-120b`** (Fireworks / OpenRouter / vLLM) | 120B frontier reasoning for complex multi-session synthesis, knowledge updates, and calendar date math | **Starter Credits / Minimal** (Only 1 call per question) |
| **Judge** | **OpenRouter** (`meta-llama/llama-3.3-70b-instruct` or `openai/gpt-4o-mini`) | High fidelity to LongMemEval's judge rubrics; evaluates hypothesis equivalence | **Free** via OpenRouter `:free` tier or minimal starter credit usage |

---

## 2. Universal OpenAI-Compatible Interface

Both **Fireworks AI** and **OpenRouter** (as well as Groq and Google AI Studio) implement the standard OpenAI REST protocol. This allows our entire Python pipeline to use a **single unified LLM client class** without vendor lock-in.

```python
import os
import json
from openai import OpenAI
from pydantic import BaseModel


class LLMClient:
    """Unified client supporting Fireworks, OpenRouter, Groq, and OpenAI."""
    
    def __init__(self, base_url: str, api_key: str, model_name: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model_name

    def structured_completion(self, system_prompt: str, user_prompt: str, response_schema: type[BaseModel]) -> BaseModel:
        """Call LLM with enforced JSON schema."""
        # Using JSON object mode compatible across all open-source API providers
        schema_json = json.dumps(response_schema.model_json_schema())
        augmented_system = f"{system_prompt}\n\nYou MUST return a valid JSON object strictly matching this schema:\n{schema_json}"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": augmented_system},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        raw_text = response.choices[0].message.content
        return response_schema.model_validate_json(raw_text)

    def text_completion(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        """Standard text generation for Reader and Judge."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content or ""
```

---

## 3. Provider Configuration Reference

### 3.1 Fireworks AI Configuration
- **Base URL**: `https://api.fireworks.ai/inference/v1`
- **Recommended Extraction Model**: `accounts/fireworks/models/llama-v3p1-8b-instruct`
- **Recommended Reader Model**: `accounts/fireworks/models/llama-v3p3-70b-instruct` or `accounts/fireworks/models/qwen2p5-72b-instruct`
- **Rate Limits**: 600 RPM (starter tier) — extremely high throughput.

### 3.2 OpenRouter Configuration
- **Base URL**: `https://openrouter.ai/api/v1`
- **Recommended Judge Model**: `meta-llama/llama-3.3-70b-instruct` or `openai/gpt-4o-mini`
- **Free Tier Models**: `meta-llama/llama-3.3-70b-instruct:free`, `google/gemini-2.0-flash-exp:free`
- **Headers required**: `HTTP-Referer` and `X-Title` (OpenRouter metadata).

### 3.3 Environment Variables Template (`.env`)
```bash
# HydraDB Connection
HYDRADB_BOLT_URI="bolt://127.0.0.1:7687"
HYDRADB_AUTH_TOKEN="your_hydradb_token"

# Ingestion / Extraction LLM (Fireworks / OpenRouter / vLLM)
EXTRACTOR_BASE_URL="https://api.fireworks.ai/inference/v1"
EXTRACTOR_API_KEY="your_api_key"
EXTRACTOR_MODEL="gpt-oss-20b"

# Reader / QA LLM (Fireworks / OpenRouter / vLLM)
READER_BASE_URL="https://api.fireworks.ai/inference/v1"
READER_API_KEY="your_api_key"
READER_MODEL="gpt-oss-120b"

# Judge LLM (OpenRouter)
JUDGE_BASE_URL="https://openrouter.ai/api/v1"
JUDGE_API_KEY="sk-or-..."
JUDGE_MODEL="meta-llama/llama-3.3-70b-instruct"
```

---

## 4. Concurrency & Rate Limiting

To balance speed with rate limit stability during benchmark runs:

```python
import asyncio
from typing import Callable, Coroutine, Any

class ConcurrencyLimiter:
    """Controls parallel LLM calls to prevent 429 rate limit errors."""
    
    def __init__(self, max_concurrent: int = 5, backoff_seconds: float = 2.0):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.backoff = backoff_seconds

    async def run(self, async_fn: Callable[..., Coroutine[Any, Any, Any]], *args, **kwargs):
        async with self.semaphore:
            retries = 3
            for attempt in range(retries):
                try:
                    return await async_fn(*args, **kwargs)
                except Exception as e:
                    if "429" in str(e) or "rate" in str(e).lower():
                        await asyncio.sleep(self.backoff * (2 ** attempt))
                    else:
                        raise e
            raise RuntimeError(f"Failed after {retries} retries due to rate limits.")
```

### Recommended Concurrency Settings
* **Fireworks AI (Extraction & Reading)**: `max_concurrent = 8` (fast processing without hitting rate bounds).
* **OpenRouter Free Tier (Judge / QA)**: `max_concurrent = 2` (free tier typically enforces 20 RPM).
* **OpenRouter Paid / Starter Credits**: `max_concurrent = 5`.

---

## 5. Summary of Completed Architectural Decisions

| # | System Area | Final Decision | Reference Document |
|---|---|---|---|
| **1** | Graph Schema & Cypher Subset | `Session`, `Turn`, `Fact`, `Entity`, `Alias` with `SUPERSEDES`, `ABOUT`, `RELATES_TO`, `MERGED_INTO` | [graph_schema_proposal.md](file:///home/aryan-sherigar/projects/hydradb-hackathon/docs/graph_schema_proposal.md) |
| **2** | Hybrid Retrieval | 3-phase: In-memory `all-MiniLM-L6-v2` semantic seeding + `algo.MSpaths` graph expansion + $0.6/0.4$ scoring | [retrieval_architecture.md](file:///home/aryan-sherigar/projects/hydradb-hackathon/docs/retrieval_architecture.md) |
| **3** | Memory Distillation | 5-phase LLM-as-a-function pipeline with `ADD`/`UPDATE`/`DELETE` mapping to `SUPERSEDES` | [semantic_memory_distillation.md](file:///home/aryan-sherigar/projects/hydradb-hackathon/docs/semantic_memory_distillation.md) |
| **4** | Entity Resolution | 3-tier: Exact $\rightarrow$ Semantic blocking ($>0.75$) $\rightarrow$ Batched LLM confirmation with `MERGED_INTO` audit | [entity_resolution_strategy.md](file:///home/aryan-sherigar/projects/hydradb-hackathon/docs/entity_resolution_strategy.md) |
| **5** | Startup & ID Generation | Content-addressable SHA-256 hash IDs, HydraDB graph hydration, wipe-and-reingest | [startup_hydration_id_generation.md](file:///home/aryan-sherigar/projects/hydradb-hackathon/docs/startup_hydration_id_generation.md) |
| **6** | Benchmark Execution | Per-Question ephemeral graph ingestion (~35 sessions per question) | [graph_schema_proposal.md](file:///home/aryan-sherigar/projects/hydradb-hackathon/docs/graph_schema_proposal.md) |
| **7** | Temporal Query Resolver | LongMemEval Time-Aware Query Expansion with LLM date inference, $\pm 2$ days padding | [temporal_query_resolver.md](file:///home/aryan-sherigar/projects/hydradb-hackathon/docs/temporal_query_resolver.md) |
| **8** | LLM Selection & Config | Fireworks (`llama-3.1-8b`, `llama-3.3-70b`) + OpenRouter (`llama-3.3-70b`) via unified OpenAI client | [llm_model_config.md](file:///home/aryan-sherigar/projects/hydradb-hackathon/docs/llm_model_config.md) |
