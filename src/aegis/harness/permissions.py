"""Tool permissions — what an agent may do without asking, and what needs a human.

Permission is not a yes/no flag. Tools are classified by the kind of damage a wrong call
can do, and each level gets a different control:

    read              allow, but record every call in the trace
    local_write       allow inside the system's own storage, recorded
    execute           allow under a timeout and the request budget
    external          preview the call, get approval, then execute

Two rules sit on top of the levels:

- Unknown tools are denied. A tool nobody classified is a tool nobody reasoned about.
- A request whose context was tainted by untrusted content (see `guardrails.untrusted`)
  is dropped to read-only, approval or not. Content fetched from outside must not be
  able to reach a side effect, which is the step that turns a prompt injection into an
  incident.
"""

import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from aegis.config import settings

logger = logging.getLogger(__name__)


class Level(str, Enum):
    """Impact class of a tool call, lowest to highest."""

    READ = "read"
    LOCAL_WRITE = "local_write"
    EXECUTE = "execute"
    EXTERNAL = "external"


# Every tool the agents can reach, and what it can damage. Adding a tool means adding it
# here — `check()` denies anything missing.
TOOL_LEVELS: dict[str, Level] = {
    "search_products": Level.READ,
    "get_product_details": Level.READ,
    "request_advisor_callback": Level.EXTERNAL,
}


@dataclass(frozen=True)
class PermissionSet:
    """What the current request is allowed to do."""

    allowed: frozenset[Level] = frozenset(
        {Level.READ, Level.LOCAL_WRITE, Level.EXECUTE, Level.EXTERNAL}
    )
    needs_approval: frozenset[Level] = frozenset({Level.EXTERNAL})

    @classmethod
    def default(cls) -> "PermissionSet":
        needs = (
            frozenset({Level.EXTERNAL}) if settings.require_approval_for_external else frozenset()
        )
        return cls(needs_approval=needs)

    @classmethod
    def read_only(cls) -> "PermissionSet":
        return cls(allowed=frozenset({Level.READ}), needs_approval=frozenset())


@dataclass(frozen=True)
class Decision:
    """Outcome of a permission check."""

    tool: str
    level: Level | None
    allowed: bool
    requires_approval: bool = False
    reason: str = ""

    @property
    def denied(self) -> bool:
        return not self.allowed


def check(tool: str, permissions: PermissionSet, *, tainted: bool = False) -> Decision:
    """Decide whether `tool` may run now, needs approval, or is refused outright."""
    level = TOOL_LEVELS.get(tool)
    if level is None:
        return Decision(tool, None, allowed=False, reason="tool is not in the permission registry")

    if tainted and level is not Level.READ:
        return Decision(
            tool,
            level,
            allowed=False,
            reason=(
                "context contains untrusted external content; "
                f"{level.value} tools are disabled for this request"
            ),
        )

    if level not in permissions.allowed:
        return Decision(tool, level, allowed=False, reason=f"{level.value} is not permitted here")

    if level in permissions.needs_approval:
        return Decision(
            tool,
            level,
            allowed=True,
            requires_approval=True,
            reason=f"{level.value} side effect requires approval",
        )

    return Decision(tool, level, allowed=True, reason=f"{level.value} permitted")


@dataclass
class PendingApproval:
    """A tool call held back, described well enough for a human to judge it."""

    id: str
    tool: str
    arguments: dict[str, Any]
    preview: str
    session_id: str = "default"
    agent: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
        + timedelta(seconds=settings.approval_ttl_seconds)
    )

    @property
    def expired(self) -> bool:
        return datetime.now(timezone.utc) >= self.expires_at


def describe_call(tool: str, arguments: dict[str, Any]) -> str:
    """Render a tool call the way it should be shown to the person approving it."""
    args = ", ".join(f"{k}={v!r}" for k, v in sorted(arguments.items()))
    return f"{tool}({args})"


class ApprovalStore:
    """In-memory store of pending approvals.

    Process-local and therefore single-instance only: a production deployment needs this
    in shared storage so any worker can resolve an approval raised by another.
    """

    def __init__(self):
        self._pending: dict[str, PendingApproval] = {}

    def create(
        self,
        tool: str,
        arguments: dict[str, Any],
        *,
        session_id: str = "default",
        agent: str = "",
    ) -> PendingApproval:
        self._purge()
        approval = PendingApproval(
            id=secrets.token_urlsafe(12),
            tool=tool,
            arguments=dict(arguments),
            preview=describe_call(tool, arguments),
            session_id=session_id,
            agent=agent,
        )
        self._pending[approval.id] = approval
        logger.info("approval pending id=%s call=%s", approval.id, approval.preview)
        return approval

    def get(self, approval_id: str) -> PendingApproval | None:
        self._purge()
        return self._pending.get(approval_id)

    def resolve(self, approval_id: str) -> PendingApproval | None:
        """Remove and return an approval — it can only be acted on once."""
        self._purge()
        return self._pending.pop(approval_id, None)

    def _purge(self) -> None:
        for key in [k for k, v in self._pending.items() if v.expired]:
            logger.info("approval expired id=%s", key)
            del self._pending[key]

    def __len__(self) -> int:
        self._purge()
        return len(self._pending)
