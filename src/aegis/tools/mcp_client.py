"""MCP client for the product catalog server.

The server runs as a child process and speaks MCP over stdio. The session is owned by a
dedicated background task: anyio task groups must be entered and exited from the same
task, and FastAPI's startup and shutdown hooks give no such guarantee.
"""

import asyncio
import json
import logging
import sys
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

SERVER_COMMAND = [sys.executable, "-m", "aegis.tools.product_catalog"]
CALL_TIMEOUT_SECONDS = 20.0


class MCPToolError(RuntimeError):
    """Raised when a tool call fails or returns unusable content."""


def _unwrap(result: Any) -> Any:
    """Pull the payload out of an MCP CallToolResult."""
    if getattr(result, "is_error", False):
        detail = result.content[0].text if result.content else "unknown error"
        raise MCPToolError(f"tool returned an error: {detail}")
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict) and "result" in structured:
        return structured["result"]
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text is None:
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    raise MCPToolError("tool returned no readable content")


class ProductCatalogClient:
    """Long-lived MCP client session against the product catalog server."""

    def __init__(self, command: list[str] | None = None):
        self._command = command or SERVER_COMMAND
        self._session: ClientSession | None = None
        self._task: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._error: BaseException | None = None
        self.server_info: str = ""

    async def start(self) -> None:
        """Spawn the server process and complete the MCP handshake."""
        self._task = asyncio.create_task(self._serve(), name="mcp-product-catalog")
        ready = asyncio.create_task(self._ready.wait())
        done, _ = await asyncio.wait({ready, self._task}, return_when=asyncio.FIRST_COMPLETED)
        if ready not in done:  # the session task died before signalling readiness
            ready.cancel()
            raise MCPToolError(f"MCP server failed to start: {self._error!r}")

    async def _serve(self) -> None:
        params = StdioServerParameters(command=self._command[0], args=self._command[1:])
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    init = await session.initialize()
                    self.server_info = f"{init.server_info.name} {init.server_info.version}"
                    self._session = session
                    logger.info("MCP session established with %s", self.server_info)
                    self._ready.set()
                    await self._stop.wait()
        except BaseException as exc:  # noqa: BLE001 — surfaced through start()/call()
            self._error = exc
            logger.exception("MCP session ended unexpectedly")
        finally:
            self._session = None
            self._ready.set()

    async def close(self) -> None:
        """Shut the session down and reap the server process."""
        self._stop.set()
        if self._task is not None:
            await asyncio.wait({self._task}, timeout=5)
            self._task = None

    async def list_tool_names(self) -> list[str]:
        session = self._require_session()
        result = await session.list_tools()
        return [tool.name for tool in result.tools]

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        """Invoke a tool by name and return its decoded payload."""
        session = self._require_session()
        try:
            result = await asyncio.wait_for(
                session.call_tool(name, arguments), timeout=CALL_TIMEOUT_SECONDS
            )
        except TimeoutError as exc:
            raise MCPToolError(f"tool {name!r} timed out after {CALL_TIMEOUT_SECONDS}s") from exc
        return _unwrap(result)

    async def search_products(
        self,
        query: str = "",
        product_type: str = "",
        max_monthly_premium: float | None = None,
        limit: int = 3,
    ) -> list[dict]:
        args: dict[str, Any] = {"query": query, "limit": limit}
        if product_type:
            args["product_type"] = product_type
        if max_monthly_premium is not None:
            args["max_monthly_premium"] = max_monthly_premium
        payload = await self.call("search_products", args)
        return payload if isinstance(payload, list) else []

    async def get_product_details(self, product_id: str) -> dict:
        payload = await self.call("get_product_details", {"product_id": product_id})
        return payload if isinstance(payload, dict) else {}

    async def request_advisor_callback(
        self, product_id: str = "", reason: str = "", contact_hint: str = ""
    ) -> dict:
        """External side effect — only call this after the permission layer approved it."""
        payload = await self.call(
            "request_advisor_callback",
            {"product_id": product_id, "reason": reason, "contact_hint": contact_hint},
        )
        return payload if isinstance(payload, dict) else {}

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise MCPToolError(f"MCP session is not running ({self._error!r})")
        return self._session

    async def __aenter__(self) -> "ProductCatalogClient":
        await self.start()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.close()
