"""Recommendation Agent — recommends products from the MCP catalog.

This agent runs an execution loop rather than a single planned call: it decides a next
step, runs it, reads the result, and decides again, until it has enough to answer or the
harness stops it.

    plan ──▶ act ──▶ observe ──▶ revise ──┐
      ▲                                   │
      └───────────────────────────────────┘

Four things end the loop, and every one of them is recorded in the trace:

- **answered** — the model says it has what it needs.
- **max_steps** — the step ceiling; the agent answers from whatever it gathered.
- **budget** — calls, tokens, or cost ran out mid-task (see `harness.cost`).
- **awaiting_approval** — the next step is an external side effect, so it is previewed
  and held instead of executed (see `harness.permissions`).

Tool output is untrusted input: every result is neutralised and fenced before it goes
back into the prompt, and a result that carried injected instructions taints the request
so it can no longer reach anything but a read.
"""

import json
import logging
from typing import Literal

from pydantic import BaseModel, Field

from aegis.agents.base import AgentResult, TraceStep
from aegis.guardrails.untrusted import UNTRUSTED_SYSTEM_RULE, sanitize_and_fence
from aegis.harness.context import AgentContext
from aegis.harness.cost import BudgetExceeded, measure
from aegis.harness.permissions import describe_call
from aegis.llm import LLMClient, LLMError
from aegis.tools.mcp_client import MCPToolError, ProductCatalogClient

logger = logging.getLogger(__name__)

Action = Literal["search_products", "get_product_details", "request_advisor_callback", "answer"]

PLANNER_SYSTEM_PROMPT = f"""You plan the next step for an insurance recommendation agent.

Tools:
- search_products(query, product_type, max_monthly_premium): find candidate products.
  product_type is one of "travel", "life-term", "life-whole", "auto", "home", "health",
  "pet", "renters" — only when the customer clearly wants that type, otherwise "".
  max_monthly_premium only when the customer stated a budget.
- get_product_details(product_id): full record for one product, when a candidate looks
  right but you need its exclusions, waiting periods, or full feature list.
- request_advisor_callback(product_id, reason, contact_hint): ask a human advisor to call
  the customer. This leaves the system, so it is held for approval before it runs. Use it
  only when the customer asked to speak to someone, or when their situation genuinely
  cannot be settled from the catalog.
- answer: stop searching and write the recommendation.

Rules:
- Choose "answer" as soon as the observations can support a recommendation. One good
  search is usually enough; do not keep searching for its own sake.
- Never repeat a call you have already made with the same arguments.
- Do not invent constraints the customer did not express.

{UNTRUSTED_SYSTEM_RULE}"""

ANSWER_SYSTEM_PROMPT = f"""You are the recommendation agent for Aegis, an insurance assistant.

Recommend from the catalog products provided, and nothing else. Rules:
- Recommend at most two products, naming each and its monthly premium and coverage limit
  exactly as given.
- Say in one sentence why each fits what the customer described.
- If none of the products fit, say so instead of stretching to recommend one.
- Be concise: at most 6 sentences. Do not invent products, prices, or features.

Risk boundaries (these are hard rules):
- Never state or imply guaranteed eligibility, underwriting approval, claim acceptance,
  investment returns, or a coverage outcome. No "you will be approved", "your claim will
  be paid", "guaranteed returns", "risk-free", "100% covered".
- Present recommendations as informational guidance, not as an offer of cover.
- Say that eligibility and pricing depend on underwriting, and mention material
  exclusions, waiting periods, or uncertainty where they are relevant.
- For anything that turns on the customer's own circumstances, point them to a human
  advisor rather than deciding it yourself.

{UNTRUSTED_SYSTEM_RULE}"""

NO_PRODUCTS_REPLY = (
    "I couldn't find a product in our catalog that matches what you described. "
    "Our advisors can help you look at options directly at 1-800-AEGIS."
)

APPROVAL_REPLY = (
    "Before I do that I need your confirmation, because it creates a request in our "
    "advisor queue:\n\n    {preview}\n\nConfirm and I'll send it; otherwise I can keep "
    "answering questions here."
)


class NextStep(BaseModel):
    """The model's choice of what to do next."""

    action: Action = Field(description="Which tool to call, or 'answer' to stop and reply")
    thought: str = Field(default="", description="One sentence on why this step")
    query: str = Field(default="", description="search_products: keywords for the need")
    product_type: str = Field(default="", description="search_products: exact type filter")
    max_monthly_premium: float | None = Field(
        default=None, description="search_products: monthly budget ceiling in USD"
    )
    product_id: str = Field(default="", description="get_product_details / callback: catalog id")
    reason: str = Field(default="", description="callback: what the customer wants to discuss")
    contact_hint: str = Field(default="", description="callback: preferred time, never a number")

    def arguments(self) -> dict:
        """The arguments this step passes to its tool."""
        if self.action == "search_products":
            args = {"query": self.query}
            if self.product_type:
                args["product_type"] = self.product_type
            if self.max_monthly_premium is not None:
                args["max_monthly_premium"] = self.max_monthly_premium
            return args
        if self.action == "get_product_details":
            return {"product_id": self.product_id}
        if self.action == "request_advisor_callback":
            return {
                "product_id": self.product_id,
                "reason": self.reason,
                "contact_hint": self.contact_hint,
            }
        return {}


class RecommendationAgent:
    """Recommends products retrieved through the MCP product catalog server."""

    name = "recommendation"

    def __init__(self, llm: LLMClient, catalog: ProductCatalogClient | None = None, limit: int = 3):
        self._llm = llm
        self._catalog = catalog
        self._limit = limit

    async def run(self, message: str, *, ctx: AgentContext | None = None) -> AgentResult:
        ctx = ctx or AgentContext()
        if self._catalog is None:
            return await self._answer_without_catalog(message, ctx)

        trace: list[TraceStep] = []
        observations: list[str] = []
        products: list[dict] = []
        seen: set[str] = set()
        stop_reason = "max_steps"
        pending = None

        for step in range(1, ctx.step_limit + 1):
            try:
                ctx.meter.ensure_within_budget()
            except BudgetExceeded as exc:
                trace.append(
                    TraceStep(
                        agent=self.name,
                        action="budget_stop",
                        detail=(
                            f"step {step}: budget exhausted ({exc}) — "
                            "answering with what I have"
                        ),
                    )
                )
                stop_reason = "budget"
                break

            decision = await self._plan(message, observations, ctx, trace, step)
            if decision is None:  # planning failed outright
                stop_reason = "planning_failed"
                break

            if decision.action == "answer":
                stop_reason = "answered"
                break

            call_signature = describe_call(decision.action, decision.arguments())
            if call_signature in seen:
                # Revising into the same call means the loop is not progressing. Say so
                # in the observations and let the next step choose differently.
                trace.append(
                    TraceStep(
                        agent=self.name,
                        action="loop_guard",
                        detail=f"repeat of {call_signature} — not run again",
                    )
                )
                observations.append(
                    f"[step {step}] {call_signature} was already run; its result is above. "
                    "Choose a different call or answer."
                )
                continue
            seen.add(call_signature)

            verdict = ctx.check_tool(decision.action)
            if verdict.denied:
                trace.append(
                    TraceStep(
                        agent=self.name,
                        action="tool_denied",
                        detail=f"{call_signature} — {verdict.reason}",
                    )
                )
                observations.append(
                    f"[step {step}] {call_signature} was refused: {verdict.reason}. "
                    "Answer without it."
                )
                continue

            if verdict.requires_approval:
                pending = self._request_approval(decision, call_signature, ctx, trace)
                if pending is None:
                    # Nowhere to record the request, so it cannot be approved later —
                    # which makes it a refusal, not a pause.
                    observations.append(
                        f"[step {step}] {call_signature} could not be held for approval. "
                        "Answer without it."
                    )
                    continue
                stop_reason = "awaiting_approval"
                break

            outcome = await self._execute(decision, call_signature, ctx, trace, step)
            observations.append(outcome.observation)
            products.extend(outcome.products)

        if stop_reason == "awaiting_approval":
            return AgentResult(
                reply=APPROVAL_REPLY.format(preview=pending.preview),
                trace=trace,
                pending_approval=pending,
                stop_reason=stop_reason,
            )

        reply = await self._write_answer(message, observations, products, ctx, trace, stop_reason)
        return AgentResult(reply=reply, trace=trace, stop_reason=stop_reason)

    # --- loop steps ------------------------------------------------------------------

    async def _plan(
        self,
        message: str,
        observations: list[str],
        ctx: AgentContext,
        trace: list[TraceStep],
        step: int,
    ) -> NextStep | None:
        """Ask the model what to do next, given everything observed so far."""
        prompt = self._planner_prompt(message, observations, ctx)
        try:
            with measure() as m:
                decision = await self._llm.generate_structured(
                    prompt, schema=NextStep, system=PLANNER_SYSTEM_PROMPT
                )
        except LLMError:
            logger.exception("Planning failed at step %d", step)
            trace.append(
                TraceStep(
                    agent=self.name, action="plan_error", detail=f"step {step}: planner failed"
                )
            )
            return None

        detail = describe_call(decision.action, decision.arguments())
        if decision.thought:
            detail = f"{detail} — {decision.thought}"
        trace.append(
            TraceStep(
                agent=self.name,
                action="plan_step",
                detail=f"step {step}/{ctx.step_limit}: {detail}",
                **m.trace_fields(),
            )
        )
        return decision

    def _planner_prompt(self, message: str, observations: list[str], ctx: AgentContext) -> str:
        parts = []
        if ctx.history:
            parts.append(f"Conversation so far:\n{ctx.history_prompt()}")
        parts.append(f"Customer message: {message}")
        if observations:
            parts.append("Observations so far:\n" + "\n\n".join(observations))
        else:
            parts.append("Observations so far: none — this is the first step.")
        return "\n\n".join(parts)

    def _request_approval(
        self,
        decision: NextStep,
        call_signature: str,
        ctx: AgentContext,
        trace: list[TraceStep],
    ):
        """Hold an external side effect and describe it for a human to approve."""
        if ctx.approvals is None:
            trace.append(
                TraceStep(
                    agent=self.name,
                    action="tool_denied",
                    detail=f"{call_signature} — approval required but no approval store is wired",
                )
            )
            return None
        pending = ctx.approvals.create(
            decision.action,
            decision.arguments(),
            session_id=ctx.session_id,
            agent=self.name,
        )
        trace.append(
            TraceStep(
                agent=self.name,
                action="approval_required",
                detail=f"{call_signature} held as approval {pending.id}",
            )
        )
        return pending

    async def _execute(
        self,
        decision: NextStep,
        call_signature: str,
        ctx: AgentContext,
        trace: list[TraceStep],
        step: int,
    ) -> "_Outcome":
        """Run one tool call and turn its result into an observation."""
        try:
            with measure() as m:
                if decision.action == "search_products":
                    payload = await self._catalog.search_products(
                        query=decision.query,
                        product_type=decision.product_type,
                        max_monthly_premium=decision.max_monthly_premium,
                        limit=self._limit,
                    )
                elif decision.action == "get_product_details":
                    payload = await self._catalog.get_product_details(decision.product_id)
                else:
                    # Reached only when the deployment does not require approval for
                    # external side effects; the permission check has already run.
                    payload = await self._catalog.request_advisor_callback(
                        product_id=decision.product_id,
                        reason=decision.reason,
                        contact_hint=decision.contact_hint,
                    )
        except MCPToolError as exc:
            logger.exception("MCP call failed: %s", call_signature)
            trace.append(
                TraceStep(agent=self.name, action="tool_error", detail=f"{call_signature}: {exc}")
            )
            return _Outcome(
                observation=f"[step {step}] {call_signature} failed: {exc}", products=[]
            )

        found = payload if isinstance(payload, list) else ([payload] if payload else [])
        ids = ", ".join(p.get("id", "?") for p in found if isinstance(p, dict)) or "none"

        # Tool output crosses the trust boundary here.
        screened = sanitize_and_fence(json.dumps(found, indent=2), source="product_catalog")
        if screened.tainted:
            ctx.taint(screened.summary)
            trace.append(
                TraceStep(agent=self.name, action="untrusted_neutralised", detail=screened.summary)
            )

        trace.append(
            TraceStep(
                agent=self.name,
                action="mcp_tool_result",
                detail=f"{len(found)} result(s): {ids}",
                **m.trace_fields(),
            )
        )
        return _Outcome(
            observation=f"[step {step}] {call_signature} returned:\n{screened.text}",
            products=[p for p in found if isinstance(p, dict) and "id" in p],
        )

    async def _write_answer(
        self,
        message: str,
        observations: list[str],
        products: list[dict],
        ctx: AgentContext,
        trace: list[TraceStep],
        stop_reason: str,
    ) -> str:
        """Write the recommendation from what the loop gathered."""
        if not products:
            trace.append(
                TraceStep(
                    agent=self.name,
                    action="abstain",
                    detail=f"no catalog products gathered (stop_reason={stop_reason})",
                )
            )
            return NO_PRODUCTS_REPLY

        parts = []
        if ctx.history:
            parts.append(f"Conversation so far:\n{ctx.history_prompt()}")
        parts.append("Tool observations:\n" + "\n\n".join(observations))
        parts.append(f"Customer message: {message}")
        if stop_reason in {"max_steps", "budget"}:
            parts.append(
                "You have run out of steps. Answer from the observations above and say "
                "plainly if they are not enough."
            )
        prompt = "\n\n".join(parts)

        with measure() as m:
            reply = await self._llm.generate(prompt, system=ANSWER_SYSTEM_PROMPT, temperature=0.4)
        logger.info(
            "recommendation_agent answered session=%s products=%d stop_reason=%s",
            ctx.session_id, len(products), stop_reason,
        )
        trace.append(
            TraceStep(
                agent=self.name,
                action="llm_answer",
                detail=f"grounded in {len(products)} catalog product(s), stop_reason={stop_reason}",
                **m.trace_fields(),
            )
        )
        return reply

    async def _answer_without_catalog(self, message: str, ctx: AgentContext) -> AgentResult:
        """Degraded mode: the catalog is down, so say what kind of cover to look for."""
        with measure() as m:
            reply = await self._llm.generate(
                f"Customer message: {message}\n\n"
                "The product catalog is unavailable. Explain in two sentences what kind of "
                "cover they should look for, and say you cannot quote specific products.",
                system=ANSWER_SYSTEM_PROMPT,
                temperature=0.4,
            )
        return AgentResult(
            reply=reply,
            trace=[
                TraceStep(
                    agent=self.name,
                    action="llm_answer",
                    detail="catalog unavailable — answered without product data",
                    **m.trace_fields(),
                )
            ],
            stop_reason="degraded",
        )


class _Outcome:
    """One tool call's contribution to the loop."""

    def __init__(self, observation: str, products: list[dict]):
        self.observation = observation
        self.products = products
