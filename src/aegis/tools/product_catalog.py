"""MCP server exposing the insurance product catalog.

Run standalone over stdio:
    python -m aegis.tools.product_catalog

The Recommendation Agent talks to this process as an MCP client
(see `aegis.tools.mcp_client`), so the tool boundary is a real protocol boundary:
the agent only knows tool names, schemas, and JSON results.
"""

import json
import logging
from functools import lru_cache

from mcp.server.mcpserver import MCPServer

from aegis.config import settings

logger = logging.getLogger(__name__)

server = MCPServer(
    name="aegis-product-catalog",
    version="0.1.0",
    instructions="Look up Aegis insurance products by need, type, and budget.",
)


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


def main() -> None:
    """Entry point for `python -m aegis.tools.product_catalog`."""
    # stdout is the MCP transport, so logs must go to stderr.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | mcp-server | %(message)s")
    server.run("stdio")


if __name__ == "__main__":
    main()
