"""Session memory — conversation history and task state for one session.

Two of the three memory kinds the harness needs:

- **Session memory**: the turns of this conversation, so "what about for a family?" can
  resolve against what was asked before it.
- **Task state**: small facts the runtime carries between turns — which agent answered
  last, what it looked up, whether an approval is outstanding.

Long-term memory (durable user preferences, learned conventions) is deliberately not
here: it needs a retention policy and a deletion path before it should hold anything
about a customer.

What gets stored is the *guarded* text: the user turn is written after the input guard
has masked PII, and the assistant turn after the output guard has run. Raw identifiers
are never persisted, and a reply that the output policy replaced is remembered as the
replacement, not the original.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Literal, Protocol

import asyncpg
from pydantic import BaseModel, Field

from aegis.config import settings

logger = logging.getLogger(__name__)

TURNS_TABLE = "session_turns"
STATE_TABLE = "session_state"

Role = Literal["user", "assistant"]


class Turn(BaseModel):
    """One message in a conversation."""

    role: Role
    content: str
    agent: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def format_history(turns: list[Turn], max_chars: int = 1200) -> str:
    """Render history for a prompt, oldest first, trimmed from the front when long."""
    if not turns:
        return ""
    lines = [f"{t.role}: {t.content.strip()}" for t in turns]
    rendered = "\n".join(lines)
    while len(rendered) > max_chars and len(lines) > 1:
        lines.pop(0)
        rendered = "\n".join(lines)
    return rendered


class SessionStore(Protocol):
    """Persistence the harness needs for a session."""

    async def append(self, session_id: str, turn: Turn) -> None: ...

    async def history(self, session_id: str, limit: int = 0) -> list[Turn]: ...

    async def load_state(self, session_id: str) -> dict[str, Any]: ...

    async def save_state(self, session_id: str, state: dict[str, Any]) -> None: ...

    async def clear(self, session_id: str) -> None: ...


class InMemorySessionStore:
    """Process-local store — for tests and for running without a database."""

    def __init__(self):
        self._turns: dict[str, list[Turn]] = {}
        self._state: dict[str, dict[str, Any]] = {}

    async def append(self, session_id: str, turn: Turn) -> None:
        self._turns.setdefault(session_id, []).append(turn)

    async def history(self, session_id: str, limit: int = 0) -> list[Turn]:
        turns = self._turns.get(session_id, [])
        n = limit or settings.session_memory_turns
        return turns[-n:] if n else list(turns)

    async def load_state(self, session_id: str) -> dict[str, Any]:
        return dict(self._state.get(session_id, {}))

    async def save_state(self, session_id: str, state: dict[str, Any]) -> None:
        self._state[session_id] = dict(state)

    async def clear(self, session_id: str) -> None:
        self._turns.pop(session_id, None)
        self._state.pop(session_id, None)


class PgSessionStore:
    """PostgreSQL-backed session memory, sharing the database the vector store uses."""

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    @classmethod
    async def connect(cls, dsn: str = "", **pool_kwargs) -> "PgSessionStore":
        pool = await asyncpg.create_pool(dsn or settings.database_url, **pool_kwargs)
        return cls(pool)

    async def close(self) -> None:
        await self._pool.close()

    async def setup(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TURNS_TABLE} (
                    id BIGSERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    agent TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            # History is always read as "the last N turns of one session".
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS {TURNS_TABLE}_session_idx "
                f"ON {TURNS_TABLE} (session_id, id DESC)"
            )
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {STATE_TABLE} (
                    session_id TEXT PRIMARY KEY,
                    state JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )

    async def append(self, session_id: str, turn: Turn) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"INSERT INTO {TURNS_TABLE} (session_id, role, content, agent) "
                f"VALUES ($1, $2, $3, $4)",
                session_id, turn.role, turn.content, turn.agent,
            )

    async def history(self, session_id: str, limit: int = 0) -> list[Turn]:
        n = limit or settings.session_memory_turns
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT role, content, agent, created_at FROM {TURNS_TABLE} "
                f"WHERE session_id = $1 ORDER BY id DESC LIMIT $2",
                session_id, n,
            )
        return [Turn(**dict(row)) for row in reversed(rows)]

    async def load_state(self, session_id: str) -> dict[str, Any]:
        async with self._pool.acquire() as conn:
            raw = await conn.fetchval(
                f"SELECT state FROM {STATE_TABLE} WHERE session_id = $1", session_id
            )
        return json.loads(raw) if raw else {}

    async def save_state(self, session_id: str, state: dict[str, Any]) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {STATE_TABLE} (session_id, state, updated_at)
                VALUES ($1, $2::jsonb, now())
                ON CONFLICT (session_id) DO UPDATE SET
                    state = EXCLUDED.state, updated_at = now()
                """,
                session_id, json.dumps(state),
            )

    async def clear(self, session_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(f"DELETE FROM {TURNS_TABLE} WHERE session_id = $1", session_id)
            await conn.execute(f"DELETE FROM {STATE_TABLE} WHERE session_id = $1", session_id)
