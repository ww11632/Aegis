"""Supervisor routing behaviour, exercised against a scripted LLM."""

import pytest

from aegis.agents.base import AgentResult, TraceStep
from aegis.agents.supervisor import Supervisor
from aegis.llm import LLMError


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

    async def run(self, message, *, session_id="default"):
        self.calls.append((message, session_id))
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

    reply, agent_used, trace = await supervisor.handle("some question", session_id="s1")

    assert agent_used == intent
    assert reply == f"{intent} handled it"
    assert agents[intent].calls == [("some question", "s1")]
    assert trace[0].action == "route"
    assert trace[-1].agent == intent


async def test_low_confidence_is_recorded_in_the_trace():
    supervisor, _ = build(ScriptedLLM(intent="faq", confidence=0.2))

    _, _, trace = await supervisor.handle("hmm")

    assert any(step.action == "low_confidence" for step in trace)


async def test_classifier_failure_degrades_to_faq_instead_of_erroring():
    supervisor, _ = build(ScriptedLLM(fail=True))

    _, agent_used, trace = await supervisor.handle("what does travel insurance cover?")

    assert agent_used == "faq"
    assert "routing failed" in trace[0].detail


def test_supervisor_requires_both_agents():
    with pytest.raises(ValueError, match="recommendation"):
        Supervisor(ScriptedLLM(), {"faq": StubAgent("faq")})
