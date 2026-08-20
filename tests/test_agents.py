"""Agent behaviour: grounding, abstention, and tool use."""

import sys

import pytest

from aegis.agents.faq import NO_CONTEXT_REPLY, FAQAgent
from aegis.agents.recommendation import NO_PRODUCTS_REPLY, RecommendationAgent
from aegis.llm import FakeLLMClient, LLMError
from aegis.rag.embeddings import LexicalEmbedder
from aegis.rag.retriever import Retriever
from aegis.rag.vector_store import Chunk, InMemoryVectorStore
from aegis.tools.mcp_client import MCPToolError, ProductCatalogClient


class RecordingLLM(FakeLLMClient):
    """Fake client that remembers what it was asked."""

    def __init__(self):
        self.prompts: list[str] = []
        self.systems: list[str] = []

    async def generate(self, prompt, *, system="", temperature=0.3):
        self.prompts.append(prompt)
        self.systems.append(system)
        return "generated answer"


async def _retriever(min_score: float = 0.0) -> Retriever:
    embedder = LexicalEmbedder()
    store = InMemoryVectorStore()
    chunks = [
        Chunk(faq_id="faq-001", question="What does travel insurance cover?",
              answer="Trip cancellation, medical emergencies abroad, and lost baggage."),
        Chunk(faq_id="faq-002", question="How do I file a claim?",
              answer="Call the claims hotline and upload supporting documents."),
    ]
    await store.upsert(chunks, await embedder.embed_documents([c.content for c in chunks]))
    return Retriever(embedder, store, min_score=min_score)


# --- FAQ agent ---------------------------------------------------------------------

async def test_faq_agent_grounds_the_answer_in_retrieved_sources():
    llm = RecordingLLM()
    agent = FAQAgent(llm, retriever=await _retriever())

    result = await agent.run("what does travel insurance cover?")

    assert result.reply == "generated answer"
    assert [s.action for s in result.trace] == ["retrieve", "llm_answer"]
    assert "faq-001" in result.trace[0].detail
    assert "[1]" in llm.prompts[0] and "Trip cancellation" in llm.prompts[0]
    assert "ONLY the numbered sources" in llm.systems[0]


async def test_faq_agent_abstains_when_nothing_is_retrieved():
    llm = RecordingLLM()
    agent = FAQAgent(llm, retriever=await _retriever(min_score=0.99))

    result = await agent.run("what is the capital of France?")

    assert result.reply == NO_CONTEXT_REPLY
    # retrieve -> observe nothing -> reformulate -> still nothing -> abstain
    assert [s.action for s in result.trace] == ["retrieve", "rephrase", "abstain"]
    assert llm.prompts == []  # no answer is written without grounding


async def test_faq_agent_degrades_when_the_retriever_is_missing():
    llm = RecordingLLM()

    result = await FAQAgent(llm, retriever=None).run("what does travel insurance cover?")

    assert result.reply == "generated answer"
    assert "without grounding" in result.trace[0].detail


# --- Recommendation agent ----------------------------------------------------------

class StubCatalog:
    def __init__(self, products=None, error: str = "", details=None):
        self.products = products if products is not None else []
        self.error = error
        self.details = details or {}
        self.calls: list[dict] = []

    async def search_products(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise MCPToolError(self.error)
        return self.products

    async def get_product_details(self, product_id):
        self.calls.append({"product_id": product_id})
        if self.error:
            raise MCPToolError(self.error)
        return self.details.get(product_id, {})


async def test_recommendation_agent_grounds_the_answer_in_catalog_products():
    llm = RecordingLLM()
    catalog = StubCatalog([{"id": "prod-001", "name": "SafeTravel Plus", "monthly_premium": 45}])

    result = await RecommendationAgent(llm, catalog=catalog).run("recommend travel cover")

    # plan -> act -> observe -> plan again -> answer
    assert [s.action for s in result.trace] == [
        "plan_step", "mcp_tool_result", "plan_step", "llm_answer",
    ]
    assert "prod-001" in result.trace[1].detail
    assert "SafeTravel Plus" in llm.prompts[0]
    assert catalog.calls[0]["query"]
    assert result.stop_reason == "answered"


async def test_recommendation_agent_reports_no_match_when_the_catalog_is_empty():
    result = await RecommendationAgent(RecordingLLM(), catalog=StubCatalog([])).run("recommend x")

    assert result.reply == NO_PRODUCTS_REPLY
    assert [s.action for s in result.trace] == [
        "plan_step", "mcp_tool_result", "plan_step", "abstain",
    ]


async def test_recommendation_agent_survives_a_tool_failure():
    catalog = StubCatalog(error="server crashed")

    result = await RecommendationAgent(RecordingLLM(), catalog=catalog).run("recommend x")

    assert result.reply == NO_PRODUCTS_REPLY
    # The failure is observed and fed back to the planner, not raised to the caller.
    assert "tool_error" in [step.action for step in result.trace]
    errors = [step.detail for step in result.trace if step.action == "tool_error"]
    assert "server crashed" in errors[0]


async def test_recommendation_agent_stops_when_planning_fails():
    class PlannerFailsLLM(RecordingLLM):
        async def generate_structured(self, prompt, *, schema, system="", temperature=0.0):
            raise LLMError("planner unavailable")

    catalog = StubCatalog([{"id": "prod-001", "name": "SafeTravel Plus"}])

    result = await RecommendationAgent(PlannerFailsLLM(), catalog=catalog).run("recommend travel")

    assert catalog.calls == []  # a planner that cannot plan does not get to call tools
    assert result.stop_reason == "planning_failed"
    assert result.reply == NO_PRODUCTS_REPLY


async def test_recommendation_agent_against_the_real_mcp_server():
    llm = RecordingLLM()
    async with ProductCatalogClient([sys.executable, "-m", "aegis.tools.product_catalog"]) as c:
        result = await RecommendationAgent(llm, catalog=c).run(
            "I need travel insurance for a family trip to Japan"
        )

    assert result.trace[1].action == "mcp_tool_result"
    assert "prod-" in result.trace[1].detail
    assert "monthly_premium" in llm.prompts[0]


@pytest.mark.parametrize("catalog", [None])
async def test_recommendation_agent_degrades_without_a_catalog(catalog):
    result = await RecommendationAgent(RecordingLLM(), catalog=catalog).run("recommend something")

    assert "catalog unavailable" in result.trace[0].detail
