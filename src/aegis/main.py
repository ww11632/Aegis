"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aegis.agents.faq import FAQAgent
from aegis.agents.recommendation import RecommendationAgent
from aegis.agents.supervisor import Supervisor
from aegis.api.routes import router
from aegis.config import settings
from aegis.llm import LLMClient, build_llm_client
from aegis.rag.embeddings import build_embedder
from aegis.rag.retriever import Retriever
from aegis.rag.vector_store import PgVectorStore
from aegis.tools.mcp_client import ProductCatalogClient

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Runtime:
    """Resources owned by the application for its lifetime."""

    supervisor: Supervisor
    llm: LLMClient
    store: PgVectorStore | None = None
    catalog: ProductCatalogClient | None = None

    async def aclose(self) -> None:
        if self.catalog is not None:
            await self.catalog.close()
        if self.store is not None:
            await self.store.close()


async def build_runtime() -> Runtime:
    """Wire up the agent graph and its backing resources.

    A missing database or catalog server degrades the affected agent rather than
    failing startup: the trace records that the answer was ungrounded, which is more
    useful in development than a service that refuses to boot.
    """
    llm = build_llm_client()

    retriever = None
    store = None
    try:
        embedder = build_embedder()
        store = await PgVectorStore.connect(dim=embedder.dim)
        await store.setup()
        indexed = await store.count()
        retriever = Retriever(embedder, store)
        logger.info("RAG ready — %d chunks indexed, embedder=%s", indexed, embedder.name)
        if indexed == 0:
            logger.warning("Knowledge base is empty — run: python -m aegis.rag.ingest")
    except Exception:
        logger.exception("RAG unavailable — FAQ answers will not be grounded")
        if store is not None:
            await store.close()
            store = None

    catalog: ProductCatalogClient | None = ProductCatalogClient()
    try:
        await catalog.start()
        logger.info("MCP catalog ready — %s", catalog.server_info)
    except Exception:
        logger.exception("MCP catalog unavailable — recommendations will not use the catalog")
        catalog = None

    supervisor = Supervisor(
        llm,
        {
            "faq": FAQAgent(llm, retriever=retriever),
            "recommendation": RecommendationAgent(llm, catalog=catalog),
        },
    )
    return Runtime(supervisor=supervisor, llm=llm, store=store, catalog=catalog)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle hooks."""
    logger.info(
        "Aegis starting up — provider=%s model=%s", settings.llm_provider, settings.gemini_model
    )
    runtime = await build_runtime()
    app.state.runtime = runtime
    app.state.supervisor = runtime.supervisor
    app.state.llm = runtime.llm
    yield
    logger.info("Aegis shutting down")
    await runtime.aclose()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Multi-agent GenAI system with RAG, MCP tools, and semantic guardrails.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
