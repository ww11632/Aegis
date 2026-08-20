"""Embedding generation.

`GeminiEmbedder` is the real implementation. `LexicalEmbedder` is a deterministic
offline stand-in (hashed bag-of-words) so retrieval can be exercised in tests and CI
without an API key — it is a weak lexical baseline, not a semantic model, and any
metric produced with it must be labelled as such.
"""

import hashlib
import logging
import math
import re
from typing import Protocol

from aegis.config import settings
from aegis.llm import LLMError

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9']+")


class Embedder(Protocol):
    """Turns text into vectors. Queries and documents are embedded separately because
    embedding models are trained with asymmetric task types."""

    dim: int
    name: str

    async def embed_query(self, text: str) -> list[float]: ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class GeminiEmbedder:
    """Google text-embedding-004 via google-genai."""

    name = "gemini"

    def __init__(self, api_key: str = "", model: str = "", dim: int = 0):
        from google import genai

        self._api_key = api_key or settings.google_api_key
        if not self._api_key:
            raise LLMError("GOOGLE_API_KEY is not set — cannot create the embedding client.")
        self._model = model or settings.embedding_model
        self.dim = dim or settings.embedding_dim
        self._client = genai.Client(api_key=self._api_key)

    async def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        try:
            resp = await self._client.aio.models.embed_content(
                model=self._model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type, output_dimensionality=self.dim
                ),
            )
        except Exception as exc:
            raise LLMError(f"Embedding call failed: {exc}") from exc

        vectors = [list(e.values or []) for e in (resp.embeddings or [])]
        if len(vectors) != len(texts) or any(len(v) != self.dim for v in vectors):
            raise LLMError(
                f"Embedding API returned {len(vectors)} vectors for {len(texts)} inputs "
                f"(expected dimension {self.dim})"
            )
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text], "RETRIEVAL_QUERY"))[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # text-embedding-004 accepts batches; keep them modest to stay under request limits.
        out: list[list[float]] = []
        batch = 32
        for i in range(0, len(texts), batch):
            out.extend(await self._embed(texts[i : i + batch], "RETRIEVAL_DOCUMENT"))
        return out


class LexicalEmbedder:
    """Deterministic hashed bag-of-words vectors — offline only.

    Retrieval works by lexical overlap, so it is a floor on what real embeddings should
    achieve, not a substitute for them.
    """

    name = "lexical"

    def __init__(self, dim: int = 0):
        self.dim = dim or settings.embedding_dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            # pgvector cannot compute cosine distance against a zero vector.
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


def build_embedder() -> Embedder:
    """Construct the embedder matching the configured provider."""
    if settings.llm_provider.lower() == "fake":
        logger.warning("Using LexicalEmbedder — lexical overlap only, not semantic search.")
        return LexicalEmbedder()
    return GeminiEmbedder()
