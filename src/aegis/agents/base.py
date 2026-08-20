"""Shared agent types.

Kept free of FastAPI/pydantic-API imports so agent code stays independent of the
transport layer; `api.schemas` mirrors these for the wire format.
"""

from typing import Protocol

from pydantic import BaseModel, Field


class TraceStep(BaseModel):
    """One observable step in handling a request."""

    agent: str
    action: str
    detail: str = ""


class AgentResult(BaseModel):
    """What a sub-agent hands back to the supervisor."""

    reply: str
    trace: list[TraceStep] = Field(default_factory=list)


class Agent(Protocol):
    """Contract every sub-agent implements."""

    name: str

    async def run(self, message: str, *, session_id: str = "default") -> AgentResult:
        """Handle a user message and return a reply plus its trace."""
        ...
