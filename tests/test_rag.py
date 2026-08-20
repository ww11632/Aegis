"""RAG pipeline: embedding, storage, similarity search, retrieval."""


from aegis.rag.embeddings import LexicalEmbedder
from aegis.rag.ingest import ingest, load_faq_chunks
from aegis.rag.retriever import Retriever, format_context
from aegis.rag.vector_store import Chunk, InMemoryVectorStore


async def _seeded_store(store):
    embedder = LexicalEmbedder()
    chunks = [
        Chunk(faq_id="faq-001", question="What does travel insurance cover?",
              answer="Trip cancellation, medical emergencies abroad, and lost baggage.",
              category="travel"),
        Chunk(faq_id="faq-002", question="How do I file a claim?",
              answer="Log in or call the claims hotline, then upload supporting documents.",
              category="claims"),
        Chunk(faq_id="faq-003", question="What is term versus whole life insurance?",
              answer="Term covers a fixed period; whole life covers you for life and builds value.",
              category="life-insurance"),
    ]
    await store.setup()
    await store.upsert(chunks, await embedder.embed_documents([c.content for c in chunks]))
    return embedder, store


async def test_similarity_search_ranks_the_relevant_chunk_first():
    embedder, store = await _seeded_store(InMemoryVectorStore())

    hits = await store.search(await embedder.embed_query("how do I file a claim"), top_k=3)

    assert hits[0].faq_id == "faq-002"
    assert hits[0].score > hits[-1].score


async def test_retriever_returns_top_k():
    embedder, store = await _seeded_store(InMemoryVectorStore())
    retriever = Retriever(embedder, store)

    chunks = await retriever.retrieve("what does travel insurance cover", top_k=2)

    assert chunks[0].faq_id == "faq-001"
    assert len(chunks) == 2


async def test_retriever_score_floor_drops_weak_matches():
    embedder, store = await _seeded_store(InMemoryVectorStore())
    strict = Retriever(embedder, store, min_score=0.99)

    assert await strict.retrieve("something entirely unrelated to insurance") == []


async def test_format_context_numbers_sources_for_citation():
    embedder, store = await _seeded_store(InMemoryVectorStore())
    chunks = await Retriever(embedder, store).retrieve("file a claim", top_k=2)

    context = format_context(chunks)

    assert context.startswith("[1]")
    assert "[2]" in context
    assert "faq-002" in context


def test_knowledge_base_loads_from_disk():
    chunks = load_faq_chunks()

    assert len(chunks) >= 5
    assert all(c.faq_id and c.question and c.answer for c in chunks)
    assert chunks[0].content.startswith("Q: ")


# --- Integration: real PostgreSQL + pgvector ---------------------------------------

async def test_pgvector_round_trip(pg_store):
    embedder, store = await _seeded_store(pg_store)

    assert await store.count() == 3
    hits = await store.search(await embedder.embed_query("term versus whole life"), top_k=2)
    assert hits[0].faq_id == "faq-003"
    assert 0.0 <= hits[0].score <= 1.0


async def test_pgvector_upsert_is_idempotent(pg_store):
    embedder, store = await _seeded_store(pg_store)
    await _seeded_store(pg_store)

    assert await store.count() == 3


async def test_ingest_indexes_the_whole_knowledge_base(pg_store):
    embedder = LexicalEmbedder()

    written = await ingest(pg_store, embedder)

    assert written == len(load_faq_chunks())
    assert await pg_store.count() == written
    hits = await Retriever(embedder, pg_store).retrieve("how do I file a claim", top_k=3)
    assert hits and hits[0].score > 0
