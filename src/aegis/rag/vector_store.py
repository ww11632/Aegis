"""Vector storage and similarity search.

`PgVectorStore` is the real store (PostgreSQL + pgvector, cosine distance).
`InMemoryVectorStore` mirrors its behaviour for unit tests that do not need a database.

Vectors are sent as text literals with an explicit `::vector` cast rather than through a
binary codec, which keeps the store free of a numpy dependency.
"""

import json
import logging
from typing import Protocol

import asyncpg
from pydantic import BaseModel

from aegis.config import settings

logger = logging.getLogger(__name__)

TABLE = "faq_chunks"


class Chunk(BaseModel):
    """A knowledge-base entry to be indexed."""

    faq_id: str
    question: str
    answer: str
    category: str = ""
    tags: list[str] = []

    @property
    def content(self) -> str:
        """The text that actually gets embedded."""
        return f"Q: {self.question}\nA: {self.answer}"


class RetrievedChunk(BaseModel):
    """A chunk returned by similarity search."""

    faq_id: str
    question: str
    answer: str
    category: str = ""
    score: float = 0.0  # cosine similarity in [-1, 1]; higher is closer


def _to_literal(embedding: list[float]) -> str:
    return "[" + ",".join(repr(float(v)) for v in embedding) + "]"


class VectorStore(Protocol):
    """Storage operations the retriever depends on."""

    async def setup(self) -> None: ...

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int: ...

    async def search(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]: ...

    async def count(self) -> int: ...


class PgVectorStore:
    """PostgreSQL + pgvector implementation."""

    def __init__(self, pool: asyncpg.Pool, dim: int = 0):
        self._pool = pool
        self._dim = dim or settings.embedding_dim

    @classmethod
    async def connect(cls, dsn: str = "", dim: int = 0, **pool_kwargs) -> "PgVectorStore":
        """Open a connection pool for the configured database."""
        pool = await asyncpg.create_pool(dsn or settings.database_url, **pool_kwargs)
        return cls(pool, dim=dim)

    async def close(self) -> None:
        await self._pool.close()

    async def setup(self) -> None:
        """Create the extension, table, and index if they are not already present."""
        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TABLE} (
                    id BIGSERIAL PRIMARY KEY,
                    faq_id TEXT UNIQUE NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT '',
                    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
                    content TEXT NOT NULL,
                    embedding VECTOR({self._dim}) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            # HNSW over cosine distance: the query path below must use the same operator
            # class (`<=>`) for the index to be usable.
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS {TABLE}_embedding_idx "
                f"ON {TABLE} USING hnsw (embedding vector_cosine_ops)"
            )

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        if not chunks:
            return 0
        rows = [
            (
                c.faq_id, c.question, c.answer, c.category,
                json.dumps(c.tags), c.content, _to_literal(e),
            )
            for c, e in zip(chunks, embeddings)
        ]
        async with self._pool.acquire() as conn:
            await conn.executemany(
                f"""
                INSERT INTO {TABLE} (faq_id, question, answer, category, tags, content, embedding)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7::vector)
                ON CONFLICT (faq_id) DO UPDATE SET
                    question = EXCLUDED.question,
                    answer = EXCLUDED.answer,
                    category = EXCLUDED.category,
                    tags = EXCLUDED.tags,
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding
                """,
                rows,
            )
        logger.info("Upserted %d chunks into %s", len(rows), TABLE)
        return len(rows)

    async def search(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT faq_id, question, answer, category,
                       1 - (embedding <=> $1::vector) AS score
                FROM {TABLE}
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                _to_literal(embedding),
                top_k,
            )
        return [RetrievedChunk(**dict(row)) for row in rows]

    async def count(self) -> int:
        async with self._pool.acquire() as conn:
            return await conn.fetchval(f"SELECT count(*) FROM {TABLE}")


class InMemoryVectorStore:
    """Cosine search over a Python list — for unit tests, never for serving."""

    def __init__(self):
        self._rows: dict[str, tuple[Chunk, list[float]]] = {}

    async def setup(self) -> None:
        return None

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        for chunk, embedding in zip(chunks, embeddings):
            self._rows[chunk.faq_id] = (chunk, embedding)
        return len(chunks)

    async def search(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        scored = [
            (self._cosine(embedding, vec), chunk) for chunk, vec in self._rows.values()
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            RetrievedChunk(
                faq_id=c.faq_id, question=c.question, answer=c.answer, category=c.category, score=s
            )
            for s, c in scored[:top_k]
        ]

    async def count(self) -> int:
        return len(self._rows)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return 0.0 if na == 0 or nb == 0 else dot / (na * nb)
