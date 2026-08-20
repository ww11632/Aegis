"""MCP integration: the client talks to the catalog server over stdio, as a subprocess."""

import sys

import pytest

from aegis.tools.mcp_client import MCPToolError, ProductCatalogClient

SERVER = [sys.executable, "-m", "aegis.tools.product_catalog"]


@pytest.fixture
async def catalog():
    async with ProductCatalogClient(SERVER) as client:
        yield client


async def test_handshake_reports_the_server(catalog):
    assert catalog.server_info.startswith("aegis-product-catalog")


async def test_tools_are_advertised_over_the_protocol(catalog):
    assert sorted(await catalog.list_tool_names()) == ["get_product_details", "search_products"]


async def test_search_returns_catalog_products(catalog):
    products = await catalog.search_products(query="travel insurance for a family trip", limit=2)

    assert 1 <= len(products) <= 2
    assert products[0]["type"] == "travel"
    assert {"id", "name", "monthly_premium", "coverage_limit"} <= set(products[0])


async def test_search_honours_the_type_filter(catalog):
    products = await catalog.search_products(query="cover my car", product_type="auto", limit=5)

    assert products
    assert all(p["type"] == "auto" for p in products)


async def test_search_honours_the_budget_filter(catalog):
    products = await catalog.search_products(query="insurance", max_monthly_premium=40, limit=10)

    assert products
    assert all(p["monthly_premium"] <= 40 for p in products)


async def test_get_product_details(catalog):
    product = await catalog.get_product_details("prod-002")

    assert product["id"] == "prod-002"
    assert product["name"]


async def test_unknown_product_id_returns_an_error_payload(catalog):
    assert "error" in await catalog.get_product_details("prod-does-not-exist")


async def test_unknown_tool_raises(catalog):
    with pytest.raises(MCPToolError):
        await catalog.call("no_such_tool", {})


async def test_calling_a_stopped_client_raises():
    client = ProductCatalogClient(SERVER)
    with pytest.raises(MCPToolError):
        await client.search_products(query="anything")
