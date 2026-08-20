"""LLM client abstraction.

Agents depend on this narrow interface (`generate` / `generate_structured`) rather
than on the provider SDK directly. That keeps the provider swappable in one file
and lets routing and agent logic be tested offline without burning API quota.
"""

import json
import logging
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from aegis.config import settings
from aegis.harness.cost import estimate_tokens, record_usage

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the LLM call fails or returns unusable output."""


def _record_gemini_usage(model: str, resp, prompt: str, system: str) -> None:
    """Book a Gemini call against the request's cost meter.

    The SDK reports token counts on the response; when it does not, the call is still
    recorded from a character estimate and flagged as such, so a missing field shows up
    as an approximate number rather than as free.
    """
    meta = getattr(resp, "usage_metadata", None)
    tokens_in = getattr(meta, "prompt_token_count", None)
    tokens_out = getattr(meta, "candidates_token_count", None)
    if tokens_in is None or tokens_out is None:
        record_usage(
            model,
            estimate_tokens(system + prompt),
            estimate_tokens(getattr(resp, "text", "") or ""),
            estimated=True,
        )
        return
    record_usage(model, int(tokens_in), int(tokens_out or 0))


class LLMClient(Protocol):
    """Minimal surface every agent needs from a language model."""

    async def generate(self, prompt: str, *, system: str = "", temperature: float = 0.3) -> str:
        """Return a free-text completion."""
        ...

    async def generate_structured(
        self, prompt: str, *, schema: type[T], system: str = "", temperature: float = 0.0
    ) -> T:
        """Return a completion parsed into `schema`."""
        ...


class GeminiClient:
    """Google Gemini implementation, using the async surface of google-genai."""

    def __init__(self, api_key: str = "", model: str = ""):
        from google import genai  # imported lazily so tests don't need the SDK configured

        self._api_key = api_key or settings.google_api_key
        if not self._api_key:
            raise LLMError("GOOGLE_API_KEY is not set — cannot create the Gemini client.")
        self._model = model or settings.gemini_model
        self._client = genai.Client(api_key=self._api_key)

    async def generate(self, prompt: str, *, system: str = "", temperature: float = 0.3) -> str:
        from google.genai import types

        try:
            resp = await self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system or None,
                    temperature=temperature,
                ),
            )
        except Exception as exc:  # SDK raises provider-specific errors
            raise LLMError(f"Gemini generate_content failed: {exc}") from exc

        _record_gemini_usage(self._model, resp, prompt, system)
        text = (resp.text or "").strip()
        if not text:
            raise LLMError("Gemini returned an empty response.")
        return text

    async def generate_structured(
        self, prompt: str, *, schema: type[T], system: str = "", temperature: float = 0.0
    ) -> T:
        from google.genai import types

        try:
            resp = await self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system or None,
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
        except Exception as exc:
            raise LLMError(f"Gemini structured call failed: {exc}") from exc

        _record_gemini_usage(self._model, resp, prompt, system)
        # `.parsed` is populated when the SDK could validate the response itself;
        # fall back to parsing `.text` so a schema mismatch surfaces as LLMError.
        parsed = getattr(resp, "parsed", None)
        if isinstance(parsed, schema):
            return parsed
        try:
            return schema.model_validate(json.loads(resp.text or ""))
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            raise LLMError(
                f"Gemini returned output that does not match {schema.__name__}: {exc}"
            ) from exc


def _customer_message(prompt: str) -> str:
    """Pull the customer's message back out of a composed prompt."""
    for line in prompt.splitlines():
        if line.startswith("Customer message:"):
            return line.split(":", 1)[1].strip()
    return prompt.strip()


class FakeLLMClient:
    """Deterministic offline client for tests and for running the API without an API key.

    It is not a model: routing falls back to keyword matching and replies are canned.
    Selected with `LLM_PROVIDER=fake` so the request path can be exercised
    end-to-end in CI.
    """

    RECOMMENDATION_HINTS = (
        "recommend", "suggest", "which plan", "which policy", "best plan", "best policy",
        "looking for", "should i buy", "shopping for", "need insurance", "compare plans",
        "budget",
    )

    async def generate(self, prompt: str, *, system: str = "", temperature: float = 0.3) -> str:
        reply = f"[fake-llm] no model configured; echoing prompt tail: {prompt.strip()[-180:]}"
        # Priced at zero, but still counted: the budget and the loop limits have to
        # behave the same offline as they do against a real provider.
        record_usage("fake", estimate_tokens(system + prompt), estimate_tokens(reply),
                     estimated=True)
        return reply

    async def generate_structured(
        self, prompt: str, *, schema: type[T], system: str = "", temperature: float = 0.0
    ) -> T:
        record_usage("fake", estimate_tokens(system + prompt), 16, estimated=True)
        fields = set(schema.model_fields)
        if {"action", "product_id"} <= fields:
            # Execution loop: one search, then answer. Deterministic, and it exercises
            # both halves of the loop (act, then revise into a stop) without a model.
            first_step = "Observations so far: none" in prompt
            return schema.model_validate(
                {
                    "action": "search_products" if first_step else "answer",
                    "thought": "fake client: search once, then answer",
                    "query": _customer_message(prompt),
                }
            )
        if {"query", "product_type"} <= fields:
            # Catalog planning: search on the raw message, invent no filters.
            return schema.model_validate(
                {
                    "query": prompt.split(":", 1)[-1].strip(),
                    "product_type": "",
                    "max_monthly_premium": None,
                }
            )
        if {"query"} == fields:
            # Query reformulation: the offline client has no vocabulary to map, so it
            # returns the question unchanged and the retry is a measured no-op.
            return schema.model_validate({"query": prompt.split(":", 1)[-1].strip()})
        if {"is_injection"} <= fields:
            # The fake client cannot judge intent; leave the decision to the patterns.
            return schema.model_validate({"is_injection": False, "reason": "fake client"})
        if {"intent", "confidence"} <= fields:
            lowered = prompt.lower()
            hit = any(hint in lowered for hint in self.RECOMMENDATION_HINTS)
            return schema.model_validate(
                {
                    "intent": "recommendation" if hit else "faq",
                    "confidence": 0.6,
                    "reasoning": "keyword match" if hit else "default to faq",
                }
            )
        raise LLMError(f"FakeLLMClient has no canned response for {schema.__name__}")


def build_llm_client() -> LLMClient:
    """Construct the client selected by configuration."""
    provider = settings.llm_provider.lower()
    if provider == "fake":
        logger.warning("Using FakeLLMClient — responses are canned, not model-generated.")
        return FakeLLMClient()
    if provider == "gemini":
        return GeminiClient()
    raise LLMError(f"Unknown LLM provider: {settings.llm_provider!r} (expected 'gemini' or 'fake')")
