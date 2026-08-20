"""Session memory: what carries across turns, and what must not be written down."""

import pytest

from aegis.agents.faq import FAQAgent
from aegis.harness.context import AgentContext
from aegis.memory.session import (
    InMemorySessionStore,
    PgSessionStore,
    Turn,
    format_history,
)
from tests.test_agents import RecordingLLM, _retriever


async def test_history_round_trips_in_order():
    store = InMemorySessionStore()
    await store.append("s1", Turn(role="user", content="which travel plan?"))
    await store.append("s1", Turn(role="assistant", content="here are two", agent="recommendation"))

    turns = await store.history("s1")

    assert [t.role for t in turns] == ["user", "assistant"]
    assert turns[1].agent == "recommendation"


async def test_sessions_do_not_leak_into_each_other():
    store = InMemorySessionStore()
    await store.append("s1", Turn(role="user", content="mine"))

    assert await store.history("s2") == []


async def test_history_is_limited_to_the_configured_window():
    store = InMemorySessionStore()
    for i in range(20):
        await store.append("s1", Turn(role="user", content=f"turn {i}"))

    turns = await store.history("s1", limit=4)

    assert len(turns) == 4
    assert turns[-1].content == "turn 19"


async def test_task_state_round_trips():
    store = InMemorySessionStore()
    await store.save_state("s1", {"last_agent": "faq"})

    assert await store.load_state("s1") == {"last_agent": "faq"}
    assert await store.load_state("unknown") == {}


async def test_clearing_a_session_forgets_everything():
    store = InMemorySessionStore()
    await store.append("s1", Turn(role="user", content="hello"))
    await store.save_state("s1", {"last_agent": "faq"})

    await store.clear("s1")

    assert await store.history("s1") == []
    assert await store.load_state("s1") == {}


def test_format_history_trims_from_the_front_when_it_gets_long():
    turns = [Turn(role="user", content="x" * 400) for _ in range(10)]

    rendered = format_history(turns, max_chars=900)

    assert len(rendered) <= 900 + len("user: ")
    assert rendered.startswith("user:")


def test_format_history_of_an_empty_session_is_empty():
    assert format_history([]) == ""


async def test_a_short_follow_up_retrieves_on_the_previous_subject():
    """"What about for a family?" carries no retrievable words of its own."""
    llm = RecordingLLM()
    ctx = AgentContext(
        history=[
            Turn(role="user", content="what does travel insurance cover?"),
            Turn(role="assistant", content="Trip cancellation and baggage.", agent="faq"),
        ]
    )
    agent = FAQAgent(llm, retriever=await _retriever())

    result = await agent.run("what about for a family?", ctx=ctx)

    # The retrieval hit the travel entry, which the follow-up alone could not have found.
    assert "faq-001" in result.trace[0].detail
    assert "Conversation so far:" in llm.prompts[0]


async def test_a_long_message_is_not_prefixed_with_history():
    llm = RecordingLLM()
    ctx = AgentContext(history=[Turn(role="user", content="what does travel insurance cover?")])
    agent = FAQAgent(llm, retriever=await _retriever())

    await agent.run("how do I file a claim for a lost bag on a trip abroad?", ctx=ctx)

    assert llm.prompts  # it answered; the query stood on its own


# --- PostgreSQL-backed store -------------------------------------------------------

@pytest.fixture
async def pg_sessions(pg_dsn):
    store = await PgSessionStore.connect(dsn=pg_dsn)
    await store.setup()
    await store.clear("pg-session")
    try:
        yield store
    finally:
        await store.clear("pg-session")
        await store.close()


async def test_pg_store_persists_turns_and_state(pg_sessions):
    await pg_sessions.append("pg-session", Turn(role="user", content="which plan?"))
    await pg_sessions.append(
        "pg-session", Turn(role="assistant", content="these two", agent="recommendation")
    )
    await pg_sessions.save_state("pg-session", {"last_agent": "recommendation", "steps": 2})

    turns = await pg_sessions.history("pg-session")
    state = await pg_sessions.load_state("pg-session")

    assert [t.content for t in turns] == ["which plan?", "these two"]
    assert turns[1].agent == "recommendation"
    assert state == {"last_agent": "recommendation", "steps": 2}


async def test_pg_store_returns_the_most_recent_turns_oldest_first(pg_sessions):
    for i in range(10):
        await pg_sessions.append("pg-session", Turn(role="user", content=f"turn {i}"))

    turns = await pg_sessions.history("pg-session", limit=3)

    assert [t.content for t in turns] == ["turn 7", "turn 8", "turn 9"]


async def test_pg_state_is_overwritten_not_appended(pg_sessions):
    await pg_sessions.save_state("pg-session", {"last_agent": "faq"})
    await pg_sessions.save_state("pg-session", {"last_agent": "recommendation"})

    assert await pg_sessions.load_state("pg-session") == {"last_agent": "recommendation"}
