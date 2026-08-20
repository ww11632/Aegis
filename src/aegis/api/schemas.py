"""API request / response schemas."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat request."""

    message: str = Field(..., min_length=1, max_length=4096, description="User message")
    session_id: str = Field(
        default="default", description="Session identifier for conversation context"
    )


class GuardrailFlag(BaseModel):
    """A single guardrail detection."""

    type: str = Field(..., description="e.g. 'pii', 'injection', 'unsafe_content'")
    detail: str = Field(default="", description="Human-readable detail")


class AgentStep(BaseModel):
    """One step in the agent execution trace, with what it cost."""

    agent: str
    action: str
    detail: str = ""
    duration_ms: float | None = Field(default=None, description="Wall-clock time for this step")
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


class CostSummary(BaseModel):
    """What the whole request consumed."""

    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    estimated: bool = Field(
        default=False,
        description="True when a provider reported no usage and tokens were estimated",
    )
    budget_exhausted: bool = False


class PendingApprovalOut(BaseModel):
    """A tool call held back for a human decision."""

    id: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    preview: str = Field(..., description="The call as it will run, for the approver to read")
    expires_at: str = ""


class ChatResponse(BaseModel):
    """Chat endpoint response."""

    reply: str
    agent_used: str = Field(..., description="Which agent handled this request")
    guardrail_flags: list[GuardrailFlag] = Field(default_factory=list)
    trace: list[AgentStep] = Field(
        default_factory=list, description="Execution trace for transparency"
    )
    cost: CostSummary = Field(default_factory=CostSummary)
    stop_reason: str = Field(
        default="answered",
        description="Why the run ended: answered, max_steps, budget, awaiting_approval, ...",
    )
    pending_approval: PendingApprovalOut | None = Field(
        default=None, description="Set when the agent needs approval before acting"
    )


class ApprovalDecision(BaseModel):
    """A human's verdict on a held tool call."""

    decision: Literal["approve", "deny"] = Field(..., description="Run the call, or discard it")


class ApprovalResult(BaseModel):
    """Outcome of resolving an approval."""

    id: str
    tool: str
    status: Literal["executed", "denied", "expired", "failed"]
    reply: str = ""
    result: dict[str, Any] | None = None
    trace: list[AgentStep] = Field(default_factory=list)


class SessionView(BaseModel):
    """Stored memory for one session."""

    session_id: str
    turns: list[dict[str, Any]] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = ""
