"""Supervisor agent — classifies user intent and routes to the appropriate sub-agent.

Uses structured output so routing is a parsed enum rather than free text that has to
be string-matched.
"""

import logging
from typing import Literal

from pydantic import BaseModel, Field

from aegis.agents.base import Agent, AgentResult, TraceStep
from aegis.harness.context import AgentContext
from aegis.harness.cost import measure, use_meter
from aegis.harness.permissions import PendingApproval
from aegis.llm import LLMClient, LLMError

logger = logging.getLogger(__name__)

Intent = Literal["faq", "recommendation"]

DEFAULT_INTENT: Intent = "faq"
LOW_CONFIDENCE = 0.5


class RoutingDecision(BaseModel):
    """Structured routing output from the classifier."""

    intent: Intent = Field(description="Which agent should handle this message")
    confidence: float = Field(ge=0.0, le=1.0, description="Classifier confidence, 0.0-1.0")
    reasoning: str = Field(default="", description="Short justification for the choice")


ROUTER_SYSTEM_PROMPT = """You route insurance customer-support messages to one of two agents.

- "faq": the customer wants information — how something works, what is covered, how to
  file a claim, definitions, timelines, policy administration.
- "recommendation": the customer wants help choosing a product — which plan or policy
  suits them, comparisons for their own situation, "what should I get" questions.

Examples:
message: "What does travel insurance cover?" -> faq (0.95)
message: "How long does a claim take to process?" -> faq (0.95)
message: "What is the difference between term and whole life?" -> faq (0.9)
message: "I'm 30 with two kids, which life policy should I get?" -> recommendation (0.95)
message: "Recommend a plan for a 2-week trip to Japan under $60/month." -> recommendation (0.95)
message: "I just bought a house, what do I need?" -> recommendation (0.8)

If the message asks for information *and* a suggestion, prefer "recommendation".
Return your confidence honestly: use a value below 0.5 when the message is too vague.

When earlier turns are supplied, route the latest message in their light: a short
follow-up usually continues the subject of the turn before it."""


class SupervisorResult(BaseModel):
    """What one full route-then-answer pass produced."""

    model_config = {"arbitrary_types_allowed": True}

    reply: str
    agent_used: str
    trace: list[TraceStep] = Field(default_factory=list)
    pending_approval: PendingApproval | None = None
    stop_reason: str = "answered"


class Supervisor:
    """Classifies intent, delegates to a sub-agent, and assembles the execution trace."""

    name = "supervisor"

    def __init__(self, llm: LLMClient, agents: dict[str, Agent]):
        missing = {"faq", "recommendation"} - set(agents)
        if missing:
            raise ValueError(f"Supervisor is missing required agents: {sorted(missing)}")
        self._llm = llm
        self._agents = agents

    async def route(self, message: str, ctx: AgentContext | None = None) -> RoutingDecision:
        """Classify a message into a routing decision, degrading to the default intent."""
        prompt = f"message: {message}"
        if ctx is not None and ctx.history:
            prompt = f"Earlier turns:\n{ctx.history_prompt()}\n\n{prompt}"
        try:
            decision = await self._llm.generate_structured(
                prompt,
                schema=RoutingDecision,
                system=ROUTER_SYSTEM_PROMPT,
            )
        except LLMError:
            # A classifier failure should not take down the request: answering as FAQ is
            # the safer default, and the trace records that routing was degraded.
            logger.exception("Routing failed; defaulting to %s", DEFAULT_INTENT)
            return RoutingDecision(
                intent=DEFAULT_INTENT, confidence=0.0, reasoning="routing failed"
            )
        logger.info(
            "routed intent=%s confidence=%.2f reasoning=%s",
            decision.intent, decision.confidence, decision.reasoning,
        )
        return decision

    async def handle(
        self, message: str, *, ctx: AgentContext | None = None
    ) -> SupervisorResult:
        """Run the full route-then-answer flow."""
        ctx = ctx or AgentContext()
        # Bind the request's meter for everything downstream, so what the agents check
        # their budget against is the same object the LLM clients record into.
        with use_meter(ctx.meter):
            return await self._handle(message, ctx)

    async def _handle(self, message: str, ctx: AgentContext) -> SupervisorResult:
        with measure() as m:
            decision = await self.route(message, ctx)
        trace = [
            TraceStep(
                agent=self.name,
                action="route",
                detail=(
                    f"intent={decision.intent} "
                    f"confidence={decision.confidence:.2f} — {decision.reasoning}"
                ),
                **m.trace_fields(),
            )
        ]
        if decision.confidence < LOW_CONFIDENCE:
            trace.append(
                TraceStep(
                    agent=self.name,
                    action="low_confidence",
                    detail=f"confidence below {LOW_CONFIDENCE}; proceeding with {decision.intent}",
                )
            )

        agent = self._agents[decision.intent]
        result: AgentResult = await agent.run(message, ctx=ctx)
        trace.extend(result.trace)
        return SupervisorResult(
            reply=result.reply,
            agent_used=agent.name,
            trace=trace,
            pending_approval=result.pending_approval,
            stop_reason=result.stop_reason,
        )
