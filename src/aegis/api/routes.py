"""API route definitions.

`/chat` is where the harness is assembled for a request: memory is loaded, a budget and a
permission set are attached, the meter is bound, and everything the run consumed is
settled into the response before it goes back.

    input guard -> context (memory, budget, permissions) -> supervisor -> agent loop
                -> output guard -> memory write -> cost settlement

`/approvals/{id}` is the other half of the loop the agent cannot close by itself: an
external side effect it previewed, that a human then allows or refuses.
"""

import logging
import time

from fastapi import APIRouter, HTTPException, Request

from aegis.agents.base import TraceStep
from aegis.agents.supervisor import Supervisor
from aegis.api.schemas import (
    AgentStep,
    ApprovalDecision,
    ApprovalResult,
    ChatRequest,
    ChatResponse,
    CostSummary,
    GuardrailFlag,
    HealthResponse,
    PendingApprovalOut,
    SessionView,
)
from aegis.config import settings
from aegis.guardrails.input_guard import BLOCKED_REPLY, check_input_async
from aegis.guardrails.output_guard import check_output
from aegis.harness.context import AgentContext
from aegis.harness.cost import Budget, BudgetExceeded, CostMeter, measure, use_meter
from aegis.harness.permissions import ApprovalStore, PendingApproval
from aegis.memory.session import SessionStore, Turn
from aegis.tools.mcp_client import MCPToolError

logger = logging.getLogger(__name__)

router = APIRouter()

BUDGET_REPLY = (
    "This question took more work than I'm allowed to spend on one request. Please try a "
    "narrower question, or contact our support team at 1-800-AEGIS."
)


def get_supervisor(request: Request) -> Supervisor:
    """Pull the agent graph built during startup off application state."""
    return request.app.state.supervisor


def get_sessions(request: Request) -> SessionStore:
    return request.app.state.sessions


def get_approvals(request: Request) -> ApprovalStore:
    return request.app.state.approvals


def _to_flags(flags) -> list[GuardrailFlag]:
    return [GuardrailFlag(type=f.type, detail=f.detail) for f in flags]


def _to_steps(steps: list[TraceStep]) -> list[AgentStep]:
    return [AgentStep(**step.model_dump()) for step in steps]


def _cost_summary(meter: CostMeter, started: float) -> CostSummary:
    total = meter.total
    return CostSummary(
        llm_calls=total.calls,
        tokens_in=total.tokens_in,
        tokens_out=total.tokens_out,
        cost_usd=round(total.cost_usd, 6),
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
        estimated=total.estimated,
        budget_exhausted=bool(meter.over_budget()),
    )


def _to_pending(approval: PendingApproval | None) -> PendingApprovalOut | None:
    if approval is None:
        return None
    return PendingApprovalOut(
        id=approval.id,
        tool=approval.tool,
        arguments=approval.arguments,
        preview=approval.preview,
        expires_at=approval.expires_at.isoformat(timespec="seconds"),
    )


@router.get("/health", response_model=HealthResponse)
async def health():
    """Readiness probe."""
    return HealthResponse(status="ok", version=settings.app_version)


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    """Main chat endpoint."""
    started = time.perf_counter()
    meter = CostMeter(budget=Budget.from_settings())
    sessions = get_sessions(request)

    with use_meter(meter):
        with measure() as screening:
            guarded_in = await check_input_async(
                req.message, llm=getattr(request.app.state, "llm", None)
            )
        # The classifier stage is a model call like any other, so it is traced like one.
        # Patterns alone cost nothing and get no step.
        screen_steps = (
            [
                TraceStep(
                    agent="input_guard",
                    action="screen",
                    detail="patterns cleared it; classifier checked the message",
                    **screening.trace_fields(),
                )
            ]
            if screening.usage.calls
            else []
        )
        if guarded_in.blocked:
            return ChatResponse(
                reply=BLOCKED_REPLY,
                agent_used="none",
                guardrail_flags=_to_flags(guarded_in.flags),
                trace=_to_steps(
                    screen_steps
                    + [
                        TraceStep(
                            agent="input_guard",
                            action="block",
                            detail=guarded_in.flags[0].detail,
                        )
                    ]
                ),
                cost=_cost_summary(meter, started),
                stop_reason="blocked",
            )

        ctx = AgentContext(
            session_id=req.session_id,
            history=await sessions.history(req.session_id),
            state=await sessions.load_state(req.session_id),
            meter=meter,
            approvals=get_approvals(request),
        )

        supervisor = get_supervisor(request)
        try:
            result = await supervisor.handle(guarded_in.text, ctx=ctx)
        except BudgetExceeded as exc:
            # The budget stopped the request before any agent could answer.
            logger.warning("Request abandoned: budget exceeded (%s)", exc)
            return ChatResponse(
                reply=BUDGET_REPLY,
                agent_used="none",
                guardrail_flags=_to_flags(guarded_in.flags),
                trace=_to_steps(
                    screen_steps
                    + [TraceStep(agent="harness", action="budget_stop", detail=str(exc))]
                ),
                cost=_cost_summary(meter, started),
                stop_reason="budget",
            )

        with measure() as screening_out:
            guarded_out = check_output(result.reply, agent=result.agent_used)
        steps: list[TraceStep] = screen_steps + list(result.trace)
        if guarded_out.blocked or guarded_out.flags:
            if guarded_out.blocked:
                action = "block"
            elif "pii" in guarded_out.flag_types:
                action = "mask"
            else:
                action = "disclose"
            steps.append(
                TraceStep(
                    agent="output_guard",
                    action=action,
                    detail=", ".join(f.detail for f in guarded_out.flags),
                    **screening_out.trace_fields(),
                )
            )

        if ctx.tainted:
            steps.append(
                TraceStep(
                    agent="harness",
                    action="tainted",
                    detail=(
                        "read-only for the rest of this request: " + "; ".join(ctx.taint_reasons)
                    ),
                )
            )

        # Memory records the guarded text on both sides: masked in, policy-checked out.
        await sessions.append(req.session_id, Turn(role="user", content=guarded_in.text))
        await sessions.append(
            req.session_id,
            Turn(role="assistant", content=guarded_out.text, agent=result.agent_used),
        )
        ctx.state.update(
            {
                "last_agent": result.agent_used,
                "last_stop_reason": result.stop_reason,
                "pending_approval_id": result.pending_approval.id
                if result.pending_approval
                else None,
            }
        )
        await sessions.save_state(req.session_id, ctx.state)

        return ChatResponse(
            reply=guarded_out.text,
            agent_used=result.agent_used,
            guardrail_flags=_to_flags(guarded_in.flags + guarded_out.flags),
            trace=_to_steps(steps),
            cost=_cost_summary(meter, started),
            stop_reason=result.stop_reason,
            pending_approval=_to_pending(result.pending_approval),
        )


@router.post("/approvals/{approval_id}", response_model=ApprovalResult)
async def resolve_approval(approval_id: str, decision: ApprovalDecision, request: Request):
    """Allow or refuse a held external tool call.

    The agent previewed the call and stopped; this is the step it cannot take itself.
    Resolving consumes the approval either way, so a call can never run twice.
    """
    approvals = get_approvals(request)
    pending = approvals.get(approval_id)
    if pending is None:
        raise HTTPException(status_code=404, detail="No such approval, or it has expired")

    approvals.resolve(approval_id)
    sessions = get_sessions(request)

    if decision.decision == "deny":
        reply = "Understood — I haven't sent that request."
        await sessions.append(
            pending.session_id, Turn(role="assistant", content=reply, agent=pending.agent)
        )
        logger.info("approval denied id=%s call=%s", approval_id, pending.preview)
        return ApprovalResult(
            id=approval_id,
            tool=pending.tool,
            status="denied",
            reply=reply,
            trace=[
                AgentStep(
                    agent="approval", action="denied", detail=f"{pending.preview} was not run"
                )
            ],
        )

    catalog = getattr(request.app.state.runtime, "catalog", None)
    if catalog is None:
        raise HTTPException(status_code=503, detail="The tool server is unavailable")

    started = time.perf_counter()
    try:
        result = await catalog.call(pending.tool, pending.arguments)
    except MCPToolError as exc:
        logger.exception("Approved call failed: %s", pending.preview)
        return ApprovalResult(
            id=approval_id,
            tool=pending.tool,
            status="failed",
            reply="I couldn't complete that request. Our support team can help at 1-800-AEGIS.",
            trace=[AgentStep(agent="approval", action="tool_error", detail=str(exc))],
        )

    elapsed = round((time.perf_counter() - started) * 1000, 1)
    ticket = result.get("ticket_id", "") if isinstance(result, dict) else ""
    reply = (
        f"Done — an advisor will be in touch. Your reference is {ticket}."
        if ticket
        else "Done — that request has been sent."
    )
    await sessions.append(
        pending.session_id, Turn(role="assistant", content=reply, agent=pending.agent)
    )
    logger.info("approval executed id=%s call=%s", approval_id, pending.preview)
    return ApprovalResult(
        id=approval_id,
        tool=pending.tool,
        status="executed",
        reply=reply,
        result=result if isinstance(result, dict) else {"result": result},
        trace=[
            AgentStep(
                agent="approval",
                action="executed",
                detail=f"{pending.preview} approved and run",
                duration_ms=elapsed,
            )
        ],
    )


@router.get("/sessions/{session_id}", response_model=SessionView)
async def read_session(session_id: str, request: Request):
    """Inspect what the assistant remembers about a session."""
    sessions = get_sessions(request)
    turns = await sessions.history(session_id, limit=50)
    return SessionView(
        session_id=session_id,
        turns=[t.model_dump(mode="json") for t in turns],
        state=await sessions.load_state(session_id),
    )


@router.delete("/sessions/{session_id}", status_code=204)
async def clear_session(session_id: str, request: Request):
    """Forget a session — the deletion path memory needs to be allowed to hold anything."""
    await get_sessions(request).clear(session_id)
