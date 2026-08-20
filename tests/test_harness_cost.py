"""Cost accounting: what a request spent, and what happens when it runs out."""

import pytest

from aegis.agents.faq import FAQAgent
from aegis.harness.context import AgentContext
from aegis.harness.cost import (
    Budget,
    BudgetExceeded,
    CostMeter,
    Usage,
    current_meter,
    measure,
    price,
    use_meter,
)
from aegis.llm import FakeLLMClient


def test_price_uses_the_published_rate_for_the_model():
    # 1M in + 1M out at 0.10 / 0.40 per million.
    assert price("gemini-2.0-flash", 1_000_000, 1_000_000) == pytest.approx(0.50)
    assert price("a-model-nobody-priced", 1_000_000, 1_000_000) == 0.0


def test_usage_adds_and_subtracts():
    a = Usage(calls=1, tokens_in=10, tokens_out=5, cost_usd=0.01)
    b = Usage(calls=2, tokens_in=20, tokens_out=5, cost_usd=0.02)

    assert (a + b).calls == 3
    assert (b - a).tokens_in == 10
    assert (a + b).tokens == 40


async def test_llm_calls_are_recorded_against_the_bound_meter():
    meter = CostMeter()
    with use_meter(meter):
        await FakeLLMClient().generate("a question")

    assert meter.total.calls == 1
    assert meter.total.tokens_in > 0
    assert meter.total.estimated is True  # the offline client has no usage metadata


async def test_measure_attributes_only_what_happened_inside_it():
    meter = CostMeter()
    llm = FakeLLMClient()
    with use_meter(meter):
        await llm.generate("outside the block")
        with measure() as m:
            await llm.generate("inside the block")

    assert m.usage.calls == 1  # not the earlier call
    assert meter.total.calls == 2
    assert m.trace_fields()["tokens_in"] > 0
    assert m.duration_ms >= 0


def test_meter_is_scoped_to_its_block():
    outer = current_meter()
    with use_meter(CostMeter()) as inner:
        assert current_meter() is inner
    assert current_meter() is outer


def test_budget_stops_the_request_when_calls_run_out():
    meter = CostMeter(budget=Budget(max_llm_calls=2))
    meter.record("fake", 10, 10)
    meter.ensure_within_budget()  # one call left

    meter.record("fake", 10, 10)
    with pytest.raises(BudgetExceeded, match="llm_calls 2/2"):
        meter.ensure_within_budget()


def test_budget_stops_on_tokens_and_on_cost():
    tokens = CostMeter(budget=Budget(max_tokens=50))
    tokens.record("fake", 40, 20)
    assert "tokens" in tokens.over_budget()

    cost = CostMeter(budget=Budget(max_cost_usd=0.0001))
    cost.record("gemini-2.0-flash", 10_000, 10_000)
    assert "cost" in cost.over_budget()


def test_an_unlimited_budget_never_binds():
    meter = CostMeter(budget=Budget.unlimited())
    for _ in range(50):
        meter.record("gemini-2.0-flash", 1_000, 1_000)

    assert meter.over_budget() == ""
    assert meter.remaining_calls() is None


async def test_trace_steps_carry_their_own_cost():
    from tests.test_agents import _retriever

    agent = FAQAgent(FakeLLMClient(), retriever=await _retriever())
    meter = CostMeter()
    with use_meter(meter):
        result = await agent.run("what does travel insurance cover?", ctx=AgentContext(meter=meter))

    answer = [s for s in result.trace if s.action == "llm_answer"][0]
    assert answer.tokens_in > 0
    assert answer.duration_ms is not None
    retrieval = [s for s in result.trace if s.action == "retrieve"][0]
    assert retrieval.tokens_in == 0  # retrieval costs latency, not tokens
    assert retrieval.duration_ms is not None
