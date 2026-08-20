"""MCP server exposing the insurance product catalog.

Run standalone over stdio:
    python -m aegis.tools.product_catalog

The Recommendation Agent talks to this process as an MCP client
(see `aegis.tools.mcp_client`), so the tool boundary is a real protocol boundary:
the agent only knows tool names, schemas, and JSON results.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from functools import lru_cache

from mcp.server.mcpserver import MCPServer

from aegis.config import settings

logger = logging.getLogger(__name__)

server = MCPServer(
    name="aegis-product-catalog",
    version="0.1.0",
    instructions="Look up Aegis insurance products by need, type, and budget.",
)


# Callback requests raised in this process. Production would write to the CRM instead.
_CALLBACK_QUEUE: list[dict] = []


@lru_cache(maxsize=1)
def _catalog() -> list[dict]:
    """Load the catalog once per process."""
    return json.loads((settings.data_dir / "product_catalog.json").read_text())


def _matches(product: dict, terms: list[str]) -> int:
    """Score a product against query terms by counting term hits in its searchable text."""
    haystack = " ".join(
        [
            product.get("name", ""),
            product.get("type", ""),
            product.get("description", ""),
            " ".join(product.get("features", [])),
            " ".join(product.get("suitable_for", [])),
        ]
    ).lower()
    return sum(1 for term in terms if term in haystack)


@server.tool(
    description=(
        "Search the insurance product catalog. Returns products ranked by how well they "
        "match the query terms, then by customer rating."
    )
)
def search_products(
    query: str = "",
    product_type: str = "",
    max_monthly_premium: float | None = None,
    limit: int = 3,
) -> list[dict]:
    """Search products.

    Args:
        query: Free-text description of the customer's need, e.g. "family travel to Japan".
        product_type: Optional exact filter on product type, e.g. "travel", "life-term", "auto".
        max_monthly_premium: Optional budget ceiling in USD per month.
        limit: Maximum number of products to return (1-10).
    """
    terms = [t for t in query.lower().split() if len(t) > 2]
    limit = max(1, min(limit, 10))

    results = []
    for product in _catalog():
        if product_type and product.get("type") != product_type:
            continue
        premium = product.get("monthly_premium", 0)
        if max_monthly_premium is not None and premium > max_monthly_premium:
            continue
        results.append((_matches(product, terms), product.get("rating", 0.0), product))

    results.sort(key=lambda row: (row[0], row[1]), reverse=True)
    logger.info("search_products query=%r type=%r -> %d hits", query, product_type, len(results))
    return [product for _, _, product in results[:limit]]


@server.tool(description="Fetch the full record for a single product by its catalog id.")
def get_product_details(product_id: str) -> dict:
    """Get one product.

    Args:
        product_id: Catalog id, e.g. "prod-001".
    """
    for product in _catalog():
        if product.get("id") == product_id:
            return product
    return {"error": f"No product with id {product_id!r}"}


@server.tool(
    description=(
        "Request a callback from a human advisor about a product. This creates a real "
        "record in the advisor queue and is an external side effect: the caller must "
        "obtain approval before invoking it."
    )
)
def request_advisor_callback(
    product_id: str = "", reason: str = "", contact_hint: str = ""
) -> dict:
    """Queue a callback request for a human advisor.

    Args:
        product_id: Catalog id the customer is asking about, e.g. "prod-001".
        reason: One sentence on what the customer wants to discuss.
        contact_hint: How the customer prefers to be reached, e.g. "weekday mornings".
            Never a phone number or an email address — the advisor system already holds
            the customer's contact details.
    """
    ticket = {
        "ticket_id": f"cb-{uuid.uuid4().hex[:8]}",
        "product_id": product_id,
        "reason": reason[:280],
        "contact_hint": contact_hint[:120],
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    # An MVP stand-in for the CRM write this would be in production. It is kept as a real
    # side effect — the queue changes — so the approval path has something to gate.
    _CALLBACK_QUEUE.append(ticket)
    logger.info("request_advisor_callback queued %s", ticket["ticket_id"])
    return ticket


def main() -> None:
    """Entry point for `python -m aegis.tools.product_catalog`."""
    # stdout is the MCP transport, so logs must go to stderr.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | mcp-server | %(message)s")
    server.run("stdio")


if __name__ == "__main__":
    main()
