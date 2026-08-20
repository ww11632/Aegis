"""Shared test fixtures.

Tests run entirely offline: the LLM is the deterministic fake client, embeddings are
lexical, and PostgreSQL comes from `pgserver` (embedded binaries, includes pgvector) so
no Docker daemon is needed. Tests that need a database skip when pgserver is missing.
"""

from pathlib import Path

import pytest

from aegis.config import settings

PGDATA = Path(__file__).parent / ".pgdata"


@pytest.fixture(autouse=True)
def offline_llm():
    """Force the deterministic client so tests never hit the Gemini API."""
    original = settings.llm_provider
    settings.llm_provider = "fake"
    yield
    settings.llm_provider = original


@pytest.fixture(scope="session")
def pg_dsn() -> str:
    """A live PostgreSQL instance with pgvector, shared by the whole test session."""
    pgserver = pytest.importorskip("pgserver", reason="pip install pgserver to run DB tests")
    server = pgserver.get_server(PGDATA, cleanup_mode="stop")
    return server.get_uri()


@pytest.fixture
async def pg_store(pg_dsn):
    """A PgVectorStore against a clean table."""
    from aegis.rag.vector_store import TABLE, PgVectorStore

    store = await PgVectorStore.connect(dsn=pg_dsn)
    await store.setup()
    async with store._pool.acquire() as conn:  # noqa: SLF001 — test-only reset
        await conn.execute(f"TRUNCATE {TABLE}")
    try:
        yield store
    finally:
        await store.close()
