"""Ingest the FAQ knowledge base into pgvector.

Usage:
    python -m aegis.rag.ingest
    LLM_PROVIDER=fake python -m aegis.rag.ingest   # offline lexical embeddings
"""

import asyncio
import json
import logging
from pathlib import Path

from aegis.config import settings
from aegis.rag.embeddings import Embedder, build_embedder
from aegis.rag.vector_store import Chunk, PgVectorStore, VectorStore

logger = logging.getLogger(__name__)


def load_faq_chunks(path: Path | None = None) -> list[Chunk]:
    """Read the FAQ knowledge base from disk.

    One FAQ entry is one chunk: the entries are already short and self-contained, so
    splitting them further would only break question/answer pairs apart.
    """
    source = path or settings.data_dir / "faq_knowledge.json"
    records = json.loads(source.read_text())
    return [
        Chunk(
            faq_id=r["id"],
            question=r["question"],
            answer=r["answer"],
            category=r.get("category", ""),
            tags=r.get("tags", []),
        )
        for r in records
    ]


async def ingest(store: VectorStore, embedder: Embedder, path: Path | None = None) -> int:
    """Embed and upsert every FAQ entry. Safe to re-run — upserts are keyed on faq_id."""
    chunks = load_faq_chunks(path)
    await store.setup()
    embeddings = await embedder.embed_documents([c.content for c in chunks])
    return await store.upsert(chunks, embeddings)


async def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(name)s | %(message)s")
    embedder = build_embedder()
    store = await PgVectorStore.connect(dim=embedder.dim)
    try:
        written = await ingest(store, embedder)
        total = await store.count()
        print(f"Ingested {written} chunks with the {embedder.name} embedder ({total} rows total).")
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(_main())
