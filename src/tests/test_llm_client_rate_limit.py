from __future__ import annotations

import json
import unittest
from unittest import mock

from pydantic import BaseModel

from context_memory.core.llm_client import (
    LLMClient,
    _is_rate_limit_error,
    _rate_limit_delay,
)


class _Schema(BaseModel):
    ok: bool


class _FakeRateLimitError(Exception):
    """Mimics the provider SDK's RateLimitError by class name, which is what
    `_is_rate_limit_error` matches on -- the real one can't be constructed
    without a live httpx response object."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


_FakeRateLimitError.__name__ = "RateLimitError"

_GROQ_429_BODY = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
    "`qwen/qwen3.6-27b` ... on tokens per minute (TPM): Limit 8000, Used 3826, "
    "Requested 4840. Please try again in 4.995s.'}}"
)


class _Message:
    def __init__(self, content): self.content = content


class _Choice:
    def __init__(self, content): self.message = _Message(content); self.finish_reason = "stop"


class _Response:
    def __init__(self, content): self.choices = [_Choice(content)]; self.usage = None


class _Completions:
    """Raises rate-limit errors for the first `fail_times` calls, then succeeds."""

    def __init__(self, fail_times: int, error: Exception | None = None) -> None:
        self.fail_times = fail_times
        self.error = error or _FakeRateLimitError(_GROQ_429_BODY)
        self.calls = 0
        self.last_kwargs = None

    def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if self.calls <= self.fail_times:
            raise self.error
        return _Response(json.dumps({"ok": True}))


class _Chat:
    def __init__(self, completions): self.completions = completions


class _FakeClient:
    def __init__(self, completions): self.chat = _Chat(completions)


def _client(completions, **kwargs) -> LLMClient:
    return LLMClient(
        base_url="https://example.invalid/v1", api_key="k", model_name="m",
        client=_FakeClient(completions), **kwargs,
    )


class RateLimitRetryTests(unittest.TestCase):
    def test_parses_provider_stated_retry_delay(self) -> None:
        # Groq states the wait in the 429 body; honoring it beats blind backoff.
        delay = _rate_limit_delay(_FakeRateLimitError(_GROQ_429_BODY), attempt=0)
        self.assertGreaterEqual(delay, 4.995)
        self.assertLess(delay, 5.6)  # stated delay + jitter only

    def test_falls_back_to_exponential_backoff_without_stated_delay(self) -> None:
        delay = _rate_limit_delay(_FakeRateLimitError("429 too many requests"), attempt=3)
        self.assertGreaterEqual(delay, 8.0)  # 2**3
        self.assertLess(delay, 9.0)

    def test_delay_is_capped(self) -> None:
        self.assertLessEqual(_rate_limit_delay(_FakeRateLimitError("nope"), attempt=20), 60.0)

    def test_identifies_rate_limit_by_status_code(self) -> None:
        err = Exception("boom")
        err.status_code = 429
        self.assertTrue(_is_rate_limit_error(err))
        self.assertFalse(_is_rate_limit_error(Exception("boom")))

    def test_retries_then_succeeds(self) -> None:
        """A 429 raises out of .create() and so never reached the empty-content
        retry loop -- rate-limited calls previously failed outright, silently
        dropping a turn's facts."""
        completions = _Completions(fail_times=2)
        client = _client(completions, rate_limit_max_retries=5)
        with mock.patch("context_memory.core.llm_client.time.sleep") as sleep:
            result = client.structured_completion("sys", "usr", _Schema)
        self.assertTrue(result.ok)
        self.assertEqual(completions.calls, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_gives_up_after_max_retries(self) -> None:
        completions = _Completions(fail_times=99)
        client = _client(completions, rate_limit_max_retries=2)
        with mock.patch("context_memory.core.llm_client.time.sleep"):
            with self.assertRaises(Exception) as caught:
                client.structured_completion("sys", "usr", _Schema)
        self.assertEqual(type(caught.exception).__name__, "RateLimitError")
        self.assertEqual(completions.calls, 3)  # initial + 2 retries

    def test_non_rate_limit_error_is_not_retried(self) -> None:
        """Only rate limiting should be retried here; anything else must surface
        immediately rather than being slowly retried against a real provider."""
        completions = _Completions(fail_times=99, error=ValueError("bad request"))
        client = _client(completions, rate_limit_max_retries=5)
        with mock.patch("context_memory.core.llm_client.time.sleep") as sleep:
            with self.assertRaises(ValueError):
                client.structured_completion("sys", "usr", _Schema)
        self.assertEqual(completions.calls, 1)
        sleep.assert_not_called()

    def test_text_completion_also_retries(self) -> None:
        completions = _Completions(fail_times=1)
        client = _client(completions, rate_limit_max_retries=3)
        with mock.patch("context_memory.core.llm_client.time.sleep"):
            out = client.text_completion("sys", "usr")
        self.assertEqual(completions.calls, 2)
        self.assertIn("ok", out)



class SchemaCompactionTests(unittest.TestCase):
    """The schema blob rides on every call and Groq reports no prompt caching
    (verified: identical prompts billed at full price twice, no `cached_tokens`
    field), so nothing amortizes it -- under a tokens-per-minute limit its size
    is throughput, not cosmetics."""

    def test_typedef_is_substantially_smaller_than_json_schema(self) -> None:
        from context_memory.core.llm_client import _compact_schema_json, _compact_schema_typedef
        from context_memory.ingestion.model_adapters import _FactExtractionResponse

        raw = json.dumps(_FactExtractionResponse.model_json_schema())
        compact = _compact_schema_json(_FactExtractionResponse)
        typedef = _compact_schema_typedef(_FactExtractionResponse)
        self.assertLess(len(compact), len(raw))
        self.assertLess(len(typedef), len(compact))
        self.assertLess(len(typedef), len(raw) * 0.4)  # measured ~73% reduction

    def test_typedef_renders_enums_optionals_lists_and_nesting(self) -> None:
        from context_memory.core.llm_client import _compact_schema_typedef
        from context_memory.ingestion.model_adapters import _FactExtractionResponse

        out = _compact_schema_typedef(_FactExtractionResponse)
        self.assertIn('action "ADD" | "UPDATE" | "DELETE"', out)   # Literal -> union
        self.assertIn("predicate_key string?", out)                 # Optional -> ?
        self.assertIn("entities string[]", out)                     # list -> []
        self.assertIn("facts ExtractedFactItem[]", out)             # nested model ref
        self.assertIn("class ExtractedFactItem {", out)             # nested block emitted
        # The leading underscore is a Python visibility convention with no
        # meaning to the model; it must not be spent as prompt tokens.
        self.assertNotIn("_ExtractedFactItem", out)

    def test_json_schema_strips_only_noise_keys(self) -> None:
        from context_memory.core.llm_client import _compact_schema_json
        from context_memory.ingestion.model_adapters import _FactExtractionResponse

        compact = json.loads(_compact_schema_json(_FactExtractionResponse))
        item = compact["$defs"]["_ExtractedFactItem"]
        self.assertNotIn("title", item)
        # Constraints the model actually needs must survive.
        self.assertEqual(item["properties"]["action"]["enum"], ["ADD", "UPDATE", "DELETE"])
        self.assertEqual(item["required"], ["text"])

    def test_both_formats_are_cached_per_schema_class(self) -> None:
        from context_memory.core.llm_client import _compact_schema_json, _compact_schema_typedef
        from context_memory.ingestion.model_adapters import _FactExtractionResponse

        self.assertIs(_compact_schema_typedef(_FactExtractionResponse),
                      _compact_schema_typedef(_FactExtractionResponse))
        self.assertIs(_compact_schema_json(_FactExtractionResponse),
                      _compact_schema_json(_FactExtractionResponse))

    def test_schema_format_selects_representation(self) -> None:
        completions = _Completions(fail_times=0)
        client = _client(completions, schema_format="typedef")
        client.structured_completion("sys", "usr", _Schema)
        sent = completions.last_kwargs["messages"][0]["content"]
        self.assertIn("class Schema {", sent)
        self.assertNotIn('"properties"', sent)

        completions2 = _Completions(fail_times=0)
        client2 = _client(completions2, schema_format="json_schema")
        client2.structured_completion("sys", "usr", _Schema)
        sent2 = completions2.last_kwargs["messages"][0]["content"]
        self.assertIn('"properties"', sent2)


if __name__ == "__main__":
    unittest.main()
