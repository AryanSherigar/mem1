"""Provider-neutral OpenAI-compatible LLM client (ADR-027).

Behind `ingestion.ports.EntityResolutionModel` / `TemporalUpdateModel` only.
Never called from core; never called directly from CI (see fakes.py).
"""

from __future__ import annotations

import json
import random
import re
import time
import types
from collections.abc import Sequence
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel

from context_memory.core.logging import get_logger, timed_operation

logger = get_logger(__name__)

# Groq's 429 body states exactly how long to wait ("Please try again in 4.995s").
# Honoring it beats a blind exponential backoff: too short re-triggers the limit,
# too long wastes budget the limiter has already released.
_RETRY_AFTER_PATTERN = re.compile(r"try again in ([0-9.]+)s")


def _rate_limit_delay(error: Exception, attempt: int) -> float:
    """Seconds to wait before retrying a rate-limited call.

    Prefers the provider's own stated delay; falls back to capped exponential
    backoff with jitter. Jitter matters here specifically because extraction
    prefetch runs several workers concurrently — without it, every worker that
    hit the same limit wakes at the same instant and immediately re-trips it.
    """
    match = _RETRY_AFTER_PATTERN.search(str(error))
    if match:
        try:
            return min(float(match.group(1)) + random.uniform(0.1, 0.5), 60.0)
        except ValueError:
            pass
    return min(2.0 ** attempt + random.uniform(0.1, 0.5), 60.0)


# Keys pydantic emits that constrain nothing the model needs: "title" is just a
# prettified field name it already has, and "default" describes what the caller
# does with an omitted field, not what the model may return. Stripping both, plus
# whitespace-free separators, cut the extraction schema 839 -> 479 chars (43%).
# That blob is sent on every single call, so at a tokens-per-minute rate limit it
# is a direct throughput cost, not a cosmetic one.
_JSON_DECODER = json.JSONDecoder()


def _recover_json_object(raw_text: str) -> str:
    """Returns the first parseable JSON object in `raw_text`.

    Providers wrap or corrupt the JSON body in ways `response_format:
    json_object` does not prevent. Observed live, deterministically (6/6 calls),
    from Bedrock's `openai.gpt-oss-20b`: a spurious `{"` prefix, i.e.
    `{"{"facts":[...]}` where `{"facts":[...]}` was meant. Markdown ``` fences
    and leading prose are the other common shapes.

    Scanning for the first balanced object handles all of them uniformly. The
    prior approach special-cased literal prefixes (`{{`, `{\\n{`) and stripped a
    single character, which by construction could not fix the `{"` case above --
    two characters, and not a matched pattern. Enumerating malformations is a
    losing game; finding the JSON is not.

    Returns the input unchanged when nothing parses, so the caller's existing
    error path still reports the real provider output.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        json.loads(text)
        return text
    except ValueError:
        pass

    # `raw_decode` stops at the end of the first valid value, so trailing junk
    # is tolerated; scanning every '{' also skips any leading junk.
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = _JSON_DECODER.raw_decode(text, index)
        except ValueError:
            continue
        if isinstance(value, dict):
            # Re-serialize rather than slicing the original: raw_decode returns
            # the parsed value, and the source span may still contain the
            # corruption we are recovering from.
            return json.dumps(value)
    return raw_text


_SCHEMA_NOISE_KEYS = frozenset({"title", "default"})

_schema_cache: dict[type[BaseModel], str] = {}


def _strip_schema_noise(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _strip_schema_noise(v) for k, v in node.items() if k not in _SCHEMA_NOISE_KEYS}
    if isinstance(node, list):
        return [_strip_schema_noise(v) for v in node]
    return node


def _compact_schema_json(response_schema: type[BaseModel]) -> str:
    """Serialized once per schema class, then cached — the result is identical
    for every call with the same schema, and this runs on a per-turn hot path."""
    cached = _schema_cache.get(response_schema)
    if cached is None:
        cached = json.dumps(
            _strip_schema_noise(response_schema.model_json_schema()), separators=(",", ":")
        )
        _schema_cache[response_schema] = cached
    return cached


_PRIMITIVE_TYPE_NAMES = {str: "string", int: "int", float: "float", bool: "bool"}

_typedef_cache: dict[type[BaseModel], str] = {}


def _model_name(model: type[BaseModel]) -> str:
    # Internal response models are underscore-prefixed by convention
    # (`_FactExtractionResponse`); that prefix is a Python visibility hint and
    # means nothing to the model, so it is not spent as prompt tokens.
    return model.__name__.lstrip("_")


def _render_type(annotation: Any) -> str:
    origin = get_origin(annotation)

    if origin is Literal:
        return " | ".join(json.dumps(value) for value in get_args(annotation))

    if origin is Union or origin is types.UnionType:
        args = get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        suffix = "?" if type(None) in args else ""
        if not non_none:
            return "null"
        return _render_type(non_none[0]) + suffix

    if origin in (list, tuple, set, frozenset, Sequence):
        args = get_args(annotation)
        return (_render_type(args[0]) if args else "any") + "[]"

    if origin is dict:
        return "map"

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _model_name(annotation)

    return _PRIMITIVE_TYPE_NAMES.get(annotation, getattr(annotation, "__name__", "any"))


def _render_typedef(model: type[BaseModel], emitted: list[type[BaseModel]] | None = None) -> str:
    """Renders a pydantic model as a compact TypeScript/BAML-style type
    declaration instead of JSON Schema.

    Measured on the extraction schema: 839 chars as pydantic JSON Schema, 479
    with the noise keys stripped, 225 as a typedef — 73% off the original, on a
    blob sent with every single call. JSON Schema spends most of its length on
    structural scaffolding (`"properties"`, `"type":"string"`, `"$defs"`,
    `"$ref"`) that a typedef expresses positionally. `response_format` is still
    set to `json_object`, so JSON output remains enforced by the provider; this
    only changes how the *shape* is described.
    """
    emitted = [] if emitted is None else emitted
    if model in emitted:
        return ""
    emitted.append(model)

    nested_blocks: list[str] = []
    lines: list[str] = []
    for field_name, field in model.model_fields.items():
        annotation = field.annotation
        inner = annotation
        origin = get_origin(annotation)
        if origin in (list, tuple, set, frozenset, Sequence):
            args = get_args(annotation)
            inner = args[0] if args else None
        elif origin is Union or origin is types.UnionType:
            non_none = [a for a in get_args(annotation) if a is not type(None)]
            inner = non_none[0] if non_none else None
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            block = _render_typedef(inner, emitted)
            if block:
                nested_blocks.append(block)
        lines.append(f"  {field_name} {_render_type(annotation)}")

    block = "\n".join([f"class {_model_name(model)} {{", *lines, "}"])
    return "\n".join([*nested_blocks, block])


def _compact_schema_typedef(response_schema: type[BaseModel]) -> str:
    cached = _typedef_cache.get(response_schema)
    if cached is None:
        cached = _render_typedef(response_schema)
        _typedef_cache[response_schema] = cached
    return cached


def _is_rate_limit_error(error: Exception) -> bool:
    if type(error).__name__ == "RateLimitError":
        return True
    return getattr(error, "status_code", None) == 429


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
        rate_limit_max_retries: int = 5,
        schema_format: str = "typedef",
    ) -> None:
        if not base_url or not model_name:
            raise ValueError("base_url and model_name must be non-empty")
        self.base_url = base_url if base_url.endswith("/") else f"{base_url}/"
        self.api_key = api_key
        self.model = model_name
        self.sdk_max_retries = sdk_max_retries
        self.rate_limit_max_retries = rate_limit_max_retries
        # "typedef" (compact TS/BAML-style) or "json_schema" (pydantic's own).
        # Provider-side JSON enforcement comes from `response_format`, not from
        # this — it only describes the shape, so switching formats cannot make
        # output non-JSON, only differently guided.
        self.schema_format = schema_format
        # Only forwarded when truthy — valid values differ per model family and
        # an unsupported one is a hard 400, so "" must mean "omit entirely".
        self.reasoning_effort = reasoning_effort
        self._client = client

    def _apply_reasoning_effort(self, create_kwargs: dict[str, Any]) -> None:
        if self.reasoning_effort:
            create_kwargs["reasoning_effort"] = self.reasoning_effort

    def _create_with_rate_limit_retry(self, create_kwargs: dict[str, Any]) -> Any:
        """Issues the API call, retrying only on rate limiting.

        Distinct from `structured_completion`'s own retry loop, which handles a
        different failure: a 200 response whose *content* is empty/unparseable.
        A 429 never reaches that loop — it raises out of `.create()` — so it was
        previously not retried at all here, and the SDK's own retry budget
        (`sdk_max_retries`, deliberately lowered to 1 because it silently
        multiplies timeouts) is far too small for a token-per-minute limiter
        that can need multiple waits in a row. Confirmed live, not theoretical:
        extraction prefetch against Groq's 8k TPM free tier failed outright with
        `RateLimitError` and silently dropped those turns' facts.
        """
        last_error: Exception | None = None
        for attempt in range(self.rate_limit_max_retries + 1):
            try:
                return self.client.chat.completions.create(**create_kwargs)
            except Exception as error:
                if not _is_rate_limit_error(error) or attempt == self.rate_limit_max_retries:
                    raise
                last_error = error
                delay = _rate_limit_delay(error, attempt)
                logger.warning(
                    "Rate limited (attempt %d/%d); waiting %.1fs before retry",
                    attempt + 1, self.rate_limit_max_retries + 1, delay,
                )
                time.sleep(delay)
        raise last_error if last_error else RuntimeError("unreachable")

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
            if self.schema_format == "typedef":
                augmented_system = (
                    f"{system_prompt}\n\nReturn JSON matching:\n{_compact_schema_typedef(response_schema)}"
                )
            else:
                augmented_system = (
                    f"{system_prompt}\n\nReturn JSON matching this schema:\n{_compact_schema_json(response_schema)}"
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
                response = self._create_with_rate_limit_retry(create_kwargs)
                if hasattr(response, "usage") and response.usage:
                    ctx["prompt_tokens"] = response.usage.prompt_tokens
                    ctx["completion_tokens"] = response.usage.completion_tokens
                    ctx["total_tokens"] = response.usage.total_tokens
                finish_reason = getattr(response.choices[0], "finish_reason", None)

                raw_text = response.choices[0].message.content
                try:
                    if not raw_text:
                        raise ValueError("empty completion content")
                    return response_schema.model_validate_json(_recover_json_object(raw_text))
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
            response = self._create_with_rate_limit_retry(create_kwargs)
            if hasattr(response, "usage") and response.usage:
                ctx["prompt_tokens"] = response.usage.prompt_tokens
                ctx["completion_tokens"] = response.usage.completion_tokens
                ctx["total_tokens"] = response.usage.total_tokens

            content = response.choices[0].message.content or ""
            ctx["response_chars"] = len(content)
            return content
