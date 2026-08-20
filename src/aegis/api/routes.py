"""API route definitions."""

import logging

from fastapi import APIRouter, Request

from aegis.agents.base import TraceStep
from aegis.agents.supervisor import Supervisor
from aegis.api.schemas import AgentStep, ChatRequest, ChatResponse, GuardrailFlag, HealthResponse
from aegis.config import settings
from aegis.guardrails.input_guard import BLOCKED_REPLY, check_input_async
from aegis.guardrails.output_guard import check_output

logger = logging.getLogger(__name__)

router = APIRouter()


def get_supervisor(request: Request) -> Supervisor:
    """Pull the agent graph built during startup off application state."""
    return request.app.state.supervisor


def _to_flags(flags) -> list[GuardrailFlag]:
    return [GuardrailFlag(type=f.type, detail=f.detail) for f in flags]


@router.get("/health", response_model=HealthResponse)
async def health():
    """Readiness probe."""
    return HealthResponse(status="ok", version=settings.app_version)


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    """Main chat endpoint.

    Flow: input guardrail -> supervisor -> agent (RAG or MCP tools) -> output guardrail.
    """
    guarded_in = await check_input_async(req.message, llm=getattr(request.app.state, "llm", None))
    if guarded_in.blocked:
        return ChatResponse(
            reply=BLOCKED_REPLY,
            agent_used="none",
            guardrail_flags=_to_flags(guarded_in.flags),
            trace=[
                AgentStep(
                    agent="input_guard", action="block", detail=guarded_in.flags[0].detail
                )
            ],
        )

    supervisor = get_supervisor(request)
    reply, agent_used, trace = await supervisor.handle(guarded_in.text, session_id=req.session_id)

    guarded_out = check_output(reply, agent=agent_used)
    steps: list[TraceStep] = list(trace)
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
            )
        )

    return ChatResponse(
        reply=guarded_out.text,
        agent_used=agent_used,
        guardrail_flags=_to_flags(guarded_in.flags + guarded_out.flags),
        trace=[AgentStep(**step.model_dump()) for step in steps],
    )
