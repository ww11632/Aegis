"""RAG retrieval: embed the query, search the vector store, format context for the LLM."""

import logging

from aegis.config import settings
from aegis.rag.embeddings import Embedder
from aegis.rag.vector_store import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Query-time half of the RAG pipeline."""

    def __init__(self, embedder: Embedder, store: VectorStore, min_score: float | None = None):
        self._embedder = embedder
        self._store = store
        self._min_score = settings.rag_min_score if min_score is None else min_score

    async def retrieve(self, query: str, top_k: int = 0) -> list[RetrievedChunk]:
        """Return the top-k most similar chunks, dropping anything below the score floor."""
        k = top_k or settings.rag_top_k
        embedding = await self._embedder.embed_query(query)
        chunks = await self._store.search(embedding, k)
        kept = [c for c in chunks if c.score >= self._min_score]
        logger.info(
            "retrieved %d/%d chunks for query=%r ids=%s",
            len(kept), len(chunks), query[:60], [c.faq_id for c in kept],
        )
        return kept


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render chunks as numbered sources so the model can cite them as [1], [2], ..."""
    return "\n\n".join(
        f"[{i}] (id={c.faq_id}, category={c.category or 'n/a'})\n"
        f"Q: {c.question}\nA: {c.answer}"
        for i, c in enumerate(chunks, start=1)
    )
