"""Supervisor routing behaviour, exercised against a scripted LLM."""

import pytest

from aegis.agents.base import AgentResult, TraceStep
from aegis.agents.supervisor import Supervisor
from aegis.harness.context import AgentContext
from aegis.llm import LLMError
from aegis.memory.session import Turn


class ScriptedLLM:
    """Returns a fixed routing decision, or raises to simulate a classifier outage."""

    def __init__(self, intent="faq", confidence=0.9, fail=False):
        self.intent, self.confidence, self.fail = intent, confidence, fail

    async def generate(self, prompt, *, system="", temperature=0.3):
        return "unused"

    async def generate_structured(self, prompt, *, schema, system="", temperature=0.0):
        if self.fail:
            raise LLMError("simulated classifier outage")
        return schema.model_validate(
            {"intent": self.intent, "confidence": self.confidence, "reasoning": "scripted"}
        )


class StubAgent:
    def __init__(self, name):
        self.name = name
        self.calls = []

    async def run(self, message, *, ctx=None):
        self.calls.append((message, ctx.session_id if ctx else "default"))
        return AgentResult(
            reply=f"{self.name} handled it",
            trace=[TraceStep(agent=self.name, action="llm_answer")],
        )


def build(llm):
    agents = {"faq": StubAgent("faq"), "recommendation": StubAgent("recommendation")}
    return Supervisor(llm, agents), agents


@pytest.mark.parametrize("intent", ["faq", "recommendation"])
async def test_routes_to_the_classified_agent(intent):
    supervisor, agents = build(ScriptedLLM(intent=intent))

    result = await supervisor.handle("some question", ctx=AgentContext(session_id="s1"))

    assert result.agent_used == intent
    assert result.reply == f"{intent} handled it"
    assert agents[intent].calls == [("some question", "s1")]
    assert result.trace[0].action == "route"
    assert result.trace[-1].agent == intent


async def test_low_confidence_is_recorded_in_the_trace():
    supervisor, _ = build(ScriptedLLM(intent="faq", confidence=0.2))

    result = await supervisor.handle("hmm")

    assert any(step.action == "low_confidence" for step in result.trace)


async def test_classifier_failure_degrades_to_faq_instead_of_erroring():
    supervisor, _ = build(ScriptedLLM(fail=True))

    result = await supervisor.handle("what does travel insurance cover?")

    assert result.agent_used == "faq"
    assert "routing failed" in result.trace[0].detail


async def test_history_is_supplied_to_the_router():
    """A follow-up has to be routed in the light of the turn before it."""
    seen = []

    class PromptCapturingLLM(ScriptedLLM):
        async def generate_structured(self, prompt, *, schema, system="", temperature=0.0):
            seen.append(prompt)
            return await super().generate_structured(
                prompt, schema=schema, system=system, temperature=temperature
            )

    supervisor, _ = build(PromptCapturingLLM())
    ctx = AgentContext(history=[Turn(role="user", content="which travel plan should I get?")])

    await supervisor.handle("what about for a family?", ctx=ctx)

    assert "Earlier turns:" in seen[0]
    assert "which travel plan should I get?" in seen[0]


def test_supervisor_requires_both_agents():
    with pytest.raises(ValueError, match="recommendation"):
        Supervisor(ScriptedLLM(), {"faq": StubAgent("faq")})
