"""Shared agent types.

Kept free of FastAPI/pydantic-API imports so agent code stays independent of the
transport layer; `api.schemas` mirrors these for the wire format.
"""

from typing import Protocol

from pydantic import BaseModel, Field

from aegis.harness.context import AgentContext
from aegis.harness.permissions import PendingApproval


class TraceStep(BaseModel):
    """One observable step in handling a request.

    Carries what happened *and* what it cost: a trace that cannot answer "how much did
    this take?" only covers half of what an operator needs after the fact.
    """

    agent: str
    action: str
    detail: str = ""
    duration_ms: float | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


class AgentResult(BaseModel):
    """What a sub-agent hands back to the supervisor."""

    model_config = {"arbitrary_types_allowed": True}

    reply: str
    trace: list[TraceStep] = Field(default_factory=list)
    # Set when the agent stopped to ask for approval instead of finishing the task.
    pending_approval: PendingApproval | None = None
    # Why the execution loop ended: "answered", "max_steps", "budget", "awaiting_approval".
    stop_reason: str = "answered"


class Agent(Protocol):
    """Contract every sub-agent implements."""

    name: str

    async def run(self, message: str, *, ctx: AgentContext | None = None) -> AgentResult:
        """Handle a user message and return a reply plus its trace."""
        ...
