"""API request / response schemas."""

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
    """One step in the agent execution trace."""

    agent: str
    action: str
    detail: str = ""


class ChatResponse(BaseModel):
    """Chat endpoint response."""

    reply: str
    agent_used: str = Field(..., description="Which agent handled this request")
    guardrail_flags: list[GuardrailFlag] = Field(default_factory=list)
    trace: list[AgentStep] = Field(
        default_factory=list, description="Execution trace for transparency"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = ""
