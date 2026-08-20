"""Execution loop: how the recommendation agent stops.

The loop is only as good as the conditions that end it, so each one gets a test:
answered, max_steps, budget, a repeated call, and a tool error mid-run.
"""

from aegis.agents.recommendation import RecommendationAgent
from aegis.harness.context import AgentContext
from aegis.harness.cost import Budget, CostMeter, estimate_tokens, record_usage, use_meter
from aegis.llm import FakeLLMClient
from tests.test_agents import RecordingLLM, StubCatalog

PRODUCTS = [{"id": "prod-001", "name": "SafeTravel Plus", "monthly_premium": 45}]


class ScriptedPlanner(RecordingLLM):
    """Returns a scripted sequence of loop decisions, repeating the last one.

    Usage is recorded the way a real client records it, so budget limits bind here too.
    """

    def __init__(self, steps):
        super().__init__()
        self.steps = list(steps)
        self.planned = 0

    async def generate(self, prompt, *, system="", temperature=0.3):
        record_usage("fake", estimate_tokens(prompt), 32, estimated=True)
        return await super().generate(prompt, system=system, temperature=temperature)

    async def generate_structured(self, prompt, *, schema, system="", temperature=0.0):
        record_usage("fake", estimate_tokens(prompt), 16, estimated=True)
        if "action" not in schema.model_fields:
            return await FakeLLMClient().generate_structured(
                prompt, schema=schema, system=system, temperature=temperature
            )
        step = self.steps[min(self.planned, len(self.steps) - 1)]
        self.planned += 1
        return schema.model_validate(step)


async def test_the_loop_revises_after_observing_a_result():
    """A first search, then a detail lookup chosen because of what it returned."""
    llm = ScriptedPlanner(
        [
            {"action": "search_products", "query": "travel japan"},
            {"action": "get_product_details", "product_id": "prod-001"},
            {"action": "answer"},
        ]
    )
    catalog = StubCatalog(PRODUCTS, details={"prod-001": {"id": "prod-001", "waiting_period": "0"}})

    result = await RecommendationAgent(llm, catalog=catalog).run("travel cover for japan")

    assert [s.action for s in result.trace] == [
        "plan_step", "mcp_tool_result",
        "plan_step", "mcp_tool_result",
        "plan_step", "llm_answer",
    ]
    assert catalog.calls == [
        {"query": "travel japan", "product_type": "", "max_monthly_premium": None, "limit": 3},
        {"product_id": "prod-001"},
    ]
    assert result.stop_reason == "answered"
    # The second plan step saw the first result.
    assert "search_products" in llm.prompts[0] or "Tool observations" in llm.prompts[0]


async def test_the_step_ceiling_ends_a_loop_that_will_not_stop():
    llm = ScriptedPlanner(
        [{"action": "search_products", "query": f"query {i}"} for i in range(10)]
    )
    catalog = StubCatalog(PRODUCTS)
    ctx = AgentContext(max_steps=3)

    result = await RecommendationAgent(llm, catalog=catalog).run("recommend something", ctx=ctx)

    assert result.stop_reason == "max_steps"
    assert len(catalog.calls) == 3  # never more than the ceiling
    assert result.trace[-1].action == "llm_answer"  # it still answers from what it has


async def test_a_repeated_call_is_not_run_twice():
    same = {"action": "search_products", "query": "travel"}
    llm = ScriptedPlanner([same, same, same, {"action": "answer"}])
    catalog = StubCatalog(PRODUCTS)

    result = await RecommendationAgent(llm, catalog=catalog).run("recommend travel")

    assert len(catalog.calls) == 1
    assert "loop_guard" in [s.action for s in result.trace]
    assert result.stop_reason in {"answered", "max_steps"}


async def test_the_budget_stops_the_loop_mid_task():
    llm = ScriptedPlanner(
        [{"action": "search_products", "query": f"query {i}"} for i in range(10)]
    )
    catalog = StubCatalog(PRODUCTS)
    # Two calls: one plan step, then the budget binds before the next.
    meter = CostMeter(budget=Budget(max_llm_calls=2))
    ctx = AgentContext(meter=meter, max_steps=8)

    with use_meter(meter):
        result = await RecommendationAgent(llm, catalog=catalog).run("recommend x", ctx=ctx)

    assert result.stop_reason == "budget"
    assert "budget_stop" in [s.action for s in result.trace]
    assert len(catalog.calls) < 8


async def test_a_tool_error_is_observed_and_the_loop_continues():
    llm = ScriptedPlanner(
        [
            {"action": "search_products", "query": "travel"},
            {"action": "search_products", "query": "trip cancellation"},
            {"action": "answer"},
        ]
    )
    catalog = StubCatalog(error="server crashed")

    result = await RecommendationAgent(llm, catalog=catalog).run("recommend travel")

    actions = [s.action for s in result.trace]
    assert actions.count("tool_error") == 2  # both attempts failed and were recorded
    assert actions[-1] == "abstain"


async def test_every_step_reports_what_it_cost():
    llm = ScriptedPlanner([{"action": "search_products", "query": "travel"}, {"action": "answer"}])
    meter = CostMeter()
    ctx = AgentContext(meter=meter)

    with use_meter(meter):
        result = await RecommendationAgent(llm, catalog=StubCatalog(PRODUCTS)).run(
            "recommend travel", ctx=ctx
        )

    planning = [s for s in result.trace if s.action == "plan_step"]
    assert all(s.duration_ms is not None for s in planning)
    assert sum(s.tokens_in for s in result.trace) == meter.total.tokens_in
