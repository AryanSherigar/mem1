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

    def __init__(self, base_url: str, api_key: str, model_name: str, *, client: Any | None = None) -> None:
        if not base_url or not model_name:
            raise ValueError("base_url and model_name must be non-empty")
        self.base_url = base_url if base_url.endswith("/") else f"{base_url}/"
        self.api_key = api_key
        self.model = model_name
        self._client = client

    @property
    def client(self) -> Any:
        """Lazily construct the underlying OpenAI SDK client (never at import time)."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def structured_completion(self, system_prompt: str, user_prompt: str, response_schema: type[BaseModel]) -> BaseModel:
        """Call the model with a JSON-object response constrained to `response_schema`."""
        schema_name = response_schema.__name__
        with timed_operation(logger, f"llm.structured_completion[{schema_name}]", {"model": self.model, "prompt_chars": len(user_prompt)}) as ctx:
            schema_json = json.dumps(response_schema.model_json_schema())
            augmented_system = (
                f"{system_prompt}\n\nYou MUST return a valid JSON object strictly matching this schema:\n{schema_json}"
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": augmented_system},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            if hasattr(response, "usage") and response.usage:
                ctx["prompt_tokens"] = response.usage.prompt_tokens
                ctx["completion_tokens"] = response.usage.completion_tokens
                ctx["total_tokens"] = response.usage.total_tokens

            raw_text = response.choices[0].message.content
            try:
                parsed = response_schema.model_validate_json(raw_text)
                return parsed
            except Exception as error:  # pydantic ValidationError or malformed JSON
                logger.error("Failed to parse LLM structured completion to %s: raw_response=%r", schema_name, raw_text)
                raise LLMClientError(f"model response did not match {response_schema.__name__}") from error

    def text_completion(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        """Plain text generation, for future reader/answer-generation use."""
        with timed_operation(logger, "llm.text_completion", {"model": self.model, "prompt_chars": len(user_prompt)}) as ctx:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            if hasattr(response, "usage") and response.usage:
                ctx["prompt_tokens"] = response.usage.prompt_tokens
                ctx["completion_tokens"] = response.usage.completion_tokens
                ctx["total_tokens"] = response.usage.total_tokens

            content = response.choices[0].message.content or ""
            ctx["response_chars"] = len(content)
            return content
