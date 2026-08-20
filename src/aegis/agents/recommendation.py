"""Recommendation Agent — recommends products from the MCP catalog.

Two LLM turns around one tool call: the model plans the catalog query (structured
output), the MCP tool returns real products, and the model writes the recommendation
from those products only.
"""

import json
import logging

from pydantic import BaseModel, Field

from aegis.agents.base import AgentResult, TraceStep
from aegis.llm import LLMClient, LLMError
from aegis.tools.mcp_client import MCPToolError, ProductCatalogClient

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You turn an insurance customer's message into a product catalog query.

Fill in:
- query: the customer's need in a few keywords (e.g. "family travel japan trip cancellation")
- product_type: one of "travel", "life-term", "life-whole", "auto", "home", "health",
  "pet", "renters" — only if the customer clearly wants that type, otherwise "".
- max_monthly_premium: a monthly budget in USD only if the customer stated one, else null.

Do not invent constraints the customer did not express."""

ANSWER_SYSTEM_PROMPT = """You are the recommendation agent for Aegis, an insurance assistant.

Recommend from the catalog products provided, and nothing else. Rules:
- Recommend at most two products, naming each and its monthly premium and coverage limit
  exactly as given.
- Say in one sentence why each fits what the customer described.
- If none of the products fit, say so instead of stretching to recommend one.
- Be concise: at most 6 sentences. Do not invent products, prices, or features.

Risk boundaries (these are hard rules):
- Never state or imply guaranteed eligibility, underwriting approval, claim acceptance,
  investment returns, or a coverage outcome. No "you will be approved", "your claim will
  be paid", "guaranteed returns", "risk-free", "100% covered".
- Present recommendations as informational guidance, not as an offer of cover.
- Say that eligibility and pricing depend on underwriting, and mention material
  exclusions, waiting periods, or uncertainty where they are relevant.
- For anything that turns on the customer's own circumstances, point them to a human
  advisor rather than deciding it yourself."""

NO_PRODUCTS_REPLY = (
    "I couldn't find a product in our catalog that matches what you described. "
    "Our advisors can help you look at options directly at 1-800-AEGIS."
)


class CatalogQuery(BaseModel):
    """Tool arguments the planner turn produces."""

    query: str = Field(default="", description="Keywords describing the customer's need")
    product_type: str = Field(default="", description="Exact product type filter, or empty")
    max_monthly_premium: float | None = Field(
        default=None, description="Monthly budget ceiling in USD, or null"
    )


class RecommendationAgent:
    """Recommends products retrieved through the MCP product catalog server."""

    name = "recommendation"

    def __init__(self, llm: LLMClient, catalog: ProductCatalogClient | None = None, limit: int = 3):
        self._llm = llm
        self._catalog = catalog
        self._limit = limit

    async def run(self, message: str, *, session_id: str = "default") -> AgentResult:
        if self._catalog is None:
            reply = await self._llm.generate(
                f"Customer message: {message}\n\n"
                "The product catalog is unavailable. Explain in two sentences what kind of "
                "cover they should look for, and say you cannot quote specific products.",
                system=ANSWER_SYSTEM_PROMPT,
                temperature=0.4,
            )
            return AgentResult(
                reply=reply,
                trace=[
                    TraceStep(
                        agent=self.name,
                        action="llm_answer",
                        detail="catalog unavailable — answered without product data",
                    )
                ],
            )

        plan = await self._plan(message)
        trace = [
            TraceStep(
                agent=self.name,
                action="plan_tool_call",
                detail=(
                    f"search_products(query={plan.query!r}, product_type={plan.product_type!r}, "
                    f"max_monthly_premium={plan.max_monthly_premium})"
                ),
            )
        ]

        try:
            products = await self._catalog.search_products(
                query=plan.query or message,
                product_type=plan.product_type,
                max_monthly_premium=plan.max_monthly_premium,
                limit=self._limit,
            )
        except MCPToolError as exc:
            logger.exception("MCP catalog lookup failed")
            trace.append(TraceStep(agent=self.name, action="tool_error", detail=str(exc)))
            return AgentResult(reply=NO_PRODUCTS_REPLY, trace=trace)

        trace.append(
            TraceStep(
                agent=self.name,
                action="mcp_tool_result",
                detail=(
                    f"{len(products)} product(s): "
                    + ", ".join(p.get("id", "?") for p in products)
                ),
            )
        )
        if not products:
            return AgentResult(reply=NO_PRODUCTS_REPLY, trace=trace)

        prompt = (
            f"Catalog products (JSON):\n{json.dumps(products, indent=2)}\n\n"
            f"Customer message: {message}"
        )
        reply = await self._llm.generate(prompt, system=ANSWER_SYSTEM_PROMPT, temperature=0.4)
        logger.info(
            "recommendation_agent answered session=%s products=%d", session_id, len(products)
        )
        trace.append(
            TraceStep(
                agent=self.name,
                action="llm_answer",
                detail=f"grounded in {len(products)} catalog product(s)",
            )
        )
        return AgentResult(reply=reply, trace=trace)

    async def _plan(self, message: str) -> CatalogQuery:
        """Ask the model for tool arguments, falling back to a raw keyword search."""
        try:
            return await self._llm.generate_structured(
                f"Customer message: {message}",
                schema=CatalogQuery,
                system=PLANNER_SYSTEM_PROMPT,
            )
        except LLMError:
            logger.exception("Tool-call planning failed; searching with the raw message")
            return CatalogQuery(query=message)
