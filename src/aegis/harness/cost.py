"""Token, latency, and cost accounting for a single request.

Every LLM call records its usage into the meter bound to the current request, so agents
do not have to thread a counter through their signatures. Two things come out of that:

1. The trace can say what each step cost, not just what it did.
2. A request can be stopped when it exceeds its budget, instead of looping until the
   provider bill notices.

Usage is attributed with `measure()`, which snapshots the meter around a block and
reports the delta plus wall-clock latency.
"""

import contextvars
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Iterator

from aegis.config import settings

logger = logging.getLogger(__name__)

# USD per 1M tokens, as (input, output). A snapshot of published list prices — treat it
# as configuration to refresh, not as a fact the code can verify.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.0-flash-lite": (0.075, 0.30),
    "text-embedding-004": (0.0, 0.0),
    # The offline client is free; it is priced so reports still have a row for it.
    "fake": (0.0, 0.0),
}

CHARS_PER_TOKEN = 4  # only used when a provider reports no usage metadata


class BudgetExceeded(RuntimeError):
    """Raised when a request has spent its allowance and must stop."""


@dataclass(frozen=True)
class Usage:
    """What a call, a step, or a whole request consumed."""

    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    estimated: bool = False

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            calls=self.calls + other.calls,
            tokens_in=self.tokens_in + other.tokens_in,
            tokens_out=self.tokens_out + other.tokens_out,
            cost_usd=round(self.cost_usd + other.cost_usd, 8),
            estimated=self.estimated or other.estimated,
        )

    def __sub__(self, other: "Usage") -> "Usage":
        return Usage(
            calls=self.calls - other.calls,
            tokens_in=self.tokens_in - other.tokens_in,
            tokens_out=self.tokens_out - other.tokens_out,
            cost_usd=round(self.cost_usd - other.cost_usd, 8),
            estimated=self.estimated or other.estimated,
        )


@dataclass(frozen=True)
class Budget:
    """Per-request ceilings. Zero disables that particular limit."""

    max_llm_calls: int = 0
    max_tokens: int = 0
    max_cost_usd: float = 0.0

    @classmethod
    def from_settings(cls) -> "Budget":
        return cls(
            max_llm_calls=settings.max_llm_calls,
            max_tokens=settings.max_total_tokens,
            max_cost_usd=settings.max_cost_usd,
        )

    @classmethod
    def unlimited(cls) -> "Budget":
        return cls()


def price(model: str, tokens_in: int, tokens_out: int) -> float:
    """Cost in USD for one call, or 0.0 for a model with no published price here."""
    rate_in, rate_out = MODEL_PRICING.get(model, (0.0, 0.0))
    return round((tokens_in * rate_in + tokens_out * rate_out) / 1_000_000, 8)


def estimate_tokens(text: str) -> int:
    """Rough token count for providers that report no usage metadata."""
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class CostMeter:
    """Running total for one request, with the budget it has to stay inside."""

    budget: Budget = field(default_factory=Budget.unlimited)
    total: Usage = field(default_factory=Usage)
    by_model: dict[str, Usage] = field(default_factory=dict)

    def record(
        self,
        model: str,
        tokens_in: int,
        tokens_out: int,
        *,
        estimated: bool = False,
    ) -> Usage:
        """Add one model call to the running total."""
        usage = Usage(
            calls=1,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=price(model, tokens_in, tokens_out),
            estimated=estimated,
        )
        self.total = self.total + usage
        self.by_model[model] = self.by_model.get(model, Usage()) + usage
        return usage

    def snapshot(self) -> Usage:
        return replace(self.total)

    def remaining_calls(self) -> int | None:
        if not self.budget.max_llm_calls:
            return None
        return max(0, self.budget.max_llm_calls - self.total.calls)

    def over_budget(self) -> str:
        """Return a human-readable reason if the budget is spent, else ''."""
        b = self.budget
        if b.max_llm_calls and self.total.calls >= b.max_llm_calls:
            return f"llm_calls {self.total.calls}/{b.max_llm_calls}"
        if b.max_tokens and self.total.tokens >= b.max_tokens:
            return f"tokens {self.total.tokens}/{b.max_tokens}"
        if b.max_cost_usd and self.total.cost_usd >= b.max_cost_usd:
            return f"cost ${self.total.cost_usd:.4f}/${b.max_cost_usd:.4f}"
        return ""

    def ensure_within_budget(self) -> None:
        """Raise before starting work that the request can no longer afford."""
        reason = self.over_budget()
        if reason:
            raise BudgetExceeded(reason)


_NULL_METER = CostMeter()
_current: contextvars.ContextVar[CostMeter | None] = contextvars.ContextVar(
    "aegis_cost_meter", default=None
)


def current_meter() -> CostMeter:
    """The meter for the request in flight, or a throwaway one outside a request."""
    return _current.get() or _NULL_METER


@contextmanager
def use_meter(meter: CostMeter) -> Iterator[CostMeter]:
    """Bind a meter for the duration of a request."""
    token = _current.set(meter)
    try:
        yield meter
    finally:
        _current.reset(token)


def record_usage(model: str, tokens_in: int, tokens_out: int, *, estimated: bool = False) -> None:
    """Called by LLM clients after every call."""
    current_meter().record(model, tokens_in, tokens_out, estimated=estimated)


@dataclass
class Measurement:
    """Latency and usage attributable to one traced step."""

    duration_ms: float = 0.0
    usage: Usage = field(default_factory=Usage)

    def trace_fields(self) -> dict:
        """Kwargs for a `TraceStep`, so a step can report what it cost."""
        return {
            "duration_ms": self.duration_ms,
            "tokens_in": self.usage.tokens_in,
            "tokens_out": self.usage.tokens_out,
            "cost_usd": self.usage.cost_usd,
        }


@contextmanager
def measure() -> Iterator[Measurement]:
    """Time a block and attribute any model usage inside it to that block."""
    meter = current_meter()
    before = meter.snapshot()
    m = Measurement()
    started = time.perf_counter()
    try:
        yield m
    finally:
        m.duration_ms = round((time.perf_counter() - started) * 1000, 1)
        m.usage = meter.snapshot() - before
