"""Regression tests for the third code review round: the simplification (no network)."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from clawifi import AsyncClawifi, Clawifi
from langchain_core.tools import ToolException

import langchain_clawifi._keyless as keyless_module
from langchain_clawifi import ClawifiClientConfigError, ClawifiFetch, ClawifiSearch
from langchain_clawifi._keyless import AsyncKeylessClient, ClawifiKeylessError, KeylessClient
from tests.unit_tests.conftest import Router
from tests.unit_tests.fake_mcp import FakeMcpServer

# 1. No mirroring: one given client means the other mode fails clearly -------------


def test_only_sync_client_then_async_call_fails_clearly(router: Router) -> None:
    tool = ClawifiSearch(client=Clawifi(api_key="k"))
    with pytest.raises(ToolException, match="Pass async_client too"):
        asyncio.run(tool.ainvoke({"query": "mcp"}))
    assert router.rest == [] and router.mcp.requests == []


def test_only_async_client_then_sync_call_fails_clearly(router: Router) -> None:
    tool = ClawifiFetch(async_client=AsyncClawifi(api_key="k"))
    with pytest.raises(ToolException, match="Pass client too"):
        tool.invoke({"url": "https://example.com"})
    assert router.rest == [] and router.mcp.requests == []


def test_error_type_is_a_clawifi_error() -> None:
    tool = ClawifiSearch(client=Clawifi(api_key="k"))

    async def open_async() -> None:
        async with tool._async_rest():
            pass

    with pytest.raises(ClawifiClientConfigError, match="or set an API key"):
        asyncio.run(open_async())


def test_api_key_builds_both_modes(router: Router) -> None:
    tool = ClawifiFetch(api_key="k")
    tool.invoke({"url": "https://example.com"})
    asyncio.run(tool.ainvoke({"url": "https://example.com"}))
    assert [r.headers["authorization"] for r in router.rest] == ["Bearer k", "Bearer k"]


def test_both_clients_given_both_modes_work(router: Router) -> None:
    tool = ClawifiFetch(client=Clawifi(api_key="s"), async_client=AsyncClawifi(api_key="a"))
    tool.invoke({"url": "https://example.com"})
    asyncio.run(tool.ainvoke({"url": "https://example.com"}))
    assert [r.headers["authorization"] for r in router.rest] == ["Bearer s", "Bearer a"]


def test_mirroring_helpers_are_gone() -> None:
    assert not hasattr(keyless_module, "http_settings_copyable")
    assert not hasattr(keyless_module, "require_copyable")


# 3. Sync keyless: close() drops the HTTP client; the next call builds a new one


def test_tool_close_then_invoke_works() -> None:
    server = FakeMcpServer()
    tool = ClawifiSearch(keyless_client=KeylessClient(transport=httpx.MockTransport(server)))
    tool.invoke({"query": "a"})
    tool.close()
    assert tool.invoke({"query": "b"})["results"]
    tool.close()
    assert server.open_sessions() == 0


# 4. A transport passed in is never closed --------------------------------------


class TrackingTransport(httpx.MockTransport):
    def __init__(self, handler) -> None:  # noqa: ANN001
        super().__init__(handler)
        self.closed = False

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.closed = True


def test_sync_client_never_closes_a_given_transport() -> None:
    transport = TrackingTransport(FakeMcpServer())
    client = KeylessClient(transport=transport)
    client.call_tool("search_free", {"query": "a"})
    client.close()
    assert transport.closed is False


def test_async_client_never_closes_a_given_transport() -> None:
    transport = TrackingTransport(FakeMcpServer())
    client = AsyncKeylessClient(transport=transport)
    asyncio.run(client.call_tool("search_free", {"query": "a"}))
    asyncio.run(client.aclose())
    assert transport.closed is False


# 5. httpx and RuntimeError failures become ClawifiKeylessError / ToolException -----


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("Cannot send a request, as the client has been closed."),
        httpx.ReadTimeout("timed out"),
        httpx.RemoteProtocolError("bad framing"),
    ],
)
def test_transport_failures_are_wrapped(error: Exception) -> None:
    def broken(request: httpx.Request) -> httpx.Response:
        raise error

    with pytest.raises(ClawifiKeylessError, match="unreachable"):
        KeylessClient(transport=httpx.MockTransport(broken)).call_tool(
            "search_free", {"query": "a"}
        )
    with pytest.raises(ClawifiKeylessError, match="unreachable"):
        asyncio.run(
            AsyncKeylessClient(transport=httpx.MockTransport(broken)).call_tool(
                "search_free", {"query": "a"}
            )
        )
    tool = ClawifiSearch(keyless_client=KeylessClient(transport=httpx.MockTransport(broken)))
    with pytest.raises(ToolException, match="unreachable"):
        tool.invoke({"query": "a"})
