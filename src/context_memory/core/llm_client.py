"""Provider-neutral OpenAI-compatible LLM client (ADR-027).

Behind `ingestion.ports.EntityResolutionModel` / `TemporalUpdateModel` only.
Never called from core; never called directly from CI (see fakes.py).
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from context_memory.core.logging import get_logger, timed_operation

logger = get_logger(__name__)


class LLMClientError(RuntimeError):
    """Raised when a provider response cannot be parsed or validated."""


class LLMClient:
    """Thin structured/text completion wrapper over any OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        *,
        client: Any | None = None,
        sdk_max_retries: int = 1,
        reasoning_effort: str = "",
    ) -> None:
        if not base_url or not model_name:
            raise ValueError("base_url and model_name must be non-empty")
        self.base_url = base_url if base_url.endswith("/") else f"{base_url}/"
        self.api_key = api_key
        self.model = model_name
        self.sdk_max_retries = sdk_max_retries
        # Only forwarded when truthy — valid values differ per model family and
        # an unsupported one is a hard 400, so "" must mean "omit entirely".
        self.reasoning_effort = reasoning_effort
        self._client = client

    def _apply_reasoning_effort(self, create_kwargs: dict[str, Any]) -> None:
        if self.reasoning_effort:
            create_kwargs["reasoning_effort"] = self.reasoning_effort

    @property
    def client(self) -> Any:
        """Lazily construct the underlying OpenAI SDK client (never at import time).

        `sdk_max_retries` is passed explicitly because the SDK's own default is 2,
        which silently *multiplies* whatever per-call `timeout` we set: a 45s
        timeout became 45s x 3 attempts = 135s of real wall clock before anything
        surfaced. That was measured, not assumed — one extraction call logged
        163947 ms against a 45s configured timeout. Retries are still worth having
        for 429/5xx (Groq rate-limits on tokens-per-minute), so this is lowered
        rather than disabled.
        """
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.base_url, api_key=self.api_key, max_retries=self.sdk_max_retries
            )
        return self._client

    def structured_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: type[BaseModel],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float | None = None,
        max_retries: int = 1,
    ) -> BaseModel:
        """Call the model with a JSON-object response constrained to `response_schema`.

        `max_tokens`/`timeout` default to `None` (provider default, unbounded) only
        for direct callers that don't go through `Config` — every real call site in
        this codebase passes both, specifically because an uncapped call has hung
        for 4+ minutes and returned nothing (empty completion, hit the model's own
        output ceiling before producing valid JSON) — confirmed against the live
        provider, not a hypothetical. See `core/config.py`'s per-role
        `<role>_max_tokens`/`<role>_timeout_seconds` fields (sized differently per
        role — extraction reasons before emitting JSON and needs more headroom
        than a one-word classification call).

        A capped budget bounds worst-case latency but doesn't stop the model from
        spending the *entire* cap on hidden reasoning and returning empty content —
        confirmed live on this exact model/provider (`completion_tokens` landed
        precisely at the configured cap, `raw_response=''`). `max_retries` (default
        1) recovers from that: on an empty/unparseable response, retries with a
        blunter "stop reasoning, emit JSON now" nudge, and — only when the failure's
        `finish_reason` was `length` (budget genuinely exhausted, not malformed
        content) — a doubled token budget for that one retry.
        """
        schema_name = response_schema.__name__
        with timed_operation(logger, f"llm.structured_completion[{schema_name}]", {"model": self.model, "prompt_chars": len(user_prompt)}) as ctx:
            schema_json = json.dumps(response_schema.model_json_schema())
            augmented_system = (
                f"{system_prompt}\n\nYou MUST return a valid JSON object strictly matching this schema:\n{schema_json}"
            )
            current_user_prompt = user_prompt
            current_max_tokens = max_tokens
            last_error: Exception | None = None

            for attempt in range(max_retries + 1):
                create_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": augmented_system},
                        {"role": "user", "content": current_user_prompt},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": temperature,
                }
                if current_max_tokens is not None:
                    create_kwargs["max_tokens"] = current_max_tokens
                if timeout is not None:
                    create_kwargs["timeout"] = timeout
                self._apply_reasoning_effort(create_kwargs)
                response = self.client.chat.completions.create(**create_kwargs)
                if hasattr(response, "usage") and response.usage:
                    ctx["prompt_tokens"] = response.usage.prompt_tokens
                    ctx["completion_tokens"] = response.usage.completion_tokens
                    ctx["total_tokens"] = response.usage.total_tokens
                finish_reason = getattr(response.choices[0], "finish_reason", None)

                raw_text = response.choices[0].message.content
                try:
                    if not raw_text:
                        raise ValueError("empty completion content")
                    return response_schema.model_validate_json(raw_text)
                except Exception as error:  # pydantic ValidationError, malformed JSON, or empty content
                    last_error = error
                    if attempt < max_retries:
                        ran_out_of_budget = finish_reason == "length"
                        logger.warning(
                            "structured_completion[%s] attempt %d/%d returned unparseable output "
                            "(finish_reason=%s) — retrying%s",
                            schema_name, attempt + 1, max_retries + 1, finish_reason,
                            " with a doubled token budget" if ran_out_of_budget and current_max_tokens else "",
                        )
                        current_user_prompt = (
                            f"{user_prompt}\n\n(Your previous response was empty or not valid JSON. Stop "
                            "reasoning and respond with ONLY the JSON object now — no other text.)"
                        )
                        if ran_out_of_budget and current_max_tokens is not None:
                            current_max_tokens = min(current_max_tokens * 2, 16384)
                        continue
                    logger.error(
                        "Failed to parse LLM structured completion to %s after %d attempt(s): raw_response=%r",
                        schema_name, max_retries + 1, raw_text,
                    )
                    raise LLMClientError(f"model response did not match {response_schema.__name__}") from last_error

    def text_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        *,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> str:
        """Plain text generation, used for reader/answer-generation."""
        with timed_operation(logger, "llm.text_completion", {"model": self.model, "prompt_chars": len(user_prompt)}) as ctx:
            create_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
            }
            if max_tokens is not None:
                create_kwargs["max_tokens"] = max_tokens
            if timeout is not None:
                create_kwargs["timeout"] = timeout
            self._apply_reasoning_effort(create_kwargs)
            response = self.client.chat.completions.create(**create_kwargs)
            if hasattr(response, "usage") and response.usage:
                ctx["prompt_tokens"] = response.usage.prompt_tokens
                ctx["completion_tokens"] = response.usage.completion_tokens
                ctx["total_tokens"] = response.usage.total_tokens

            content = response.choices[0].message.content or ""
            ctx["response_chars"] = len(content)
            return content
