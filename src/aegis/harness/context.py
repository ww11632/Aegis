"""Request context — the object the harness hands to an agent.

Everything an agent needs that is not the message itself, and everything the harness
needs to keep control of it: who is asking, what was said before, what the agent may
call, what it has already spent, and whether untrusted content has tainted the request.

Agents receive this rather than a growing list of keyword arguments, so adding a harness
concern does not change every agent signature.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from aegis.config import settings
from aegis.harness.cost import Budget, CostMeter
from aegis.harness.permissions import ApprovalStore, Decision, PermissionSet, check
from aegis.memory.session import Turn, format_history

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """One request's slice of the harness."""

    session_id: str = "default"
    history: list[Turn] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    permissions: PermissionSet = field(default_factory=PermissionSet.default)
    meter: CostMeter = field(default_factory=lambda: CostMeter(budget=Budget.from_settings()))
    approvals: ApprovalStore | None = None
    max_steps: int = 0
    # Set when untrusted content had to be neutralised during this request.
    tainted: bool = False
    taint_reasons: list[str] = field(default_factory=list)

    @property
    def step_limit(self) -> int:
        return self.max_steps or settings.max_agent_steps

    def history_prompt(self) -> str:
        """Prior turns rendered for a prompt, or '' when this is the first one."""
        return format_history(self.history)

    def taint(self, reason: str) -> None:
        """Mark the request as carrying untrusted instructions.

        Irreversible for the rest of the request: `permissions.check` then refuses
        anything above a read.
        """
        self.tainted = True
        self.taint_reasons.append(reason)
        logger.warning("Request tainted by untrusted content: %s", reason)

    def check_tool(self, tool: str) -> Decision:
        """Permission decision for a tool call in this context."""
        return check(tool, self.permissions, tainted=self.tainted)
