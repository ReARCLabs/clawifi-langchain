"""Regression tests for the second code review round (2026-11-26, no network)."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from clawifi import AsyncClawifi, Clawifi
from langchain_core.tools import ToolException

from langchain_clawifi import ClawifiClientConfigError, ClawifiFetch, ClawifiSearch
from langchain_clawifi._keyless import (
    AsyncKeylessClient,
    KeylessClient,
    http_settings_copyable,
    retry_after_seconds,
)
from tests.unit_tests.conftest import Router
from tests.unit_tests.fake_mcp import FakeMcpServer

# Finding 2: one MCP session is reused and ended with DELETE ---------------------


def test_async_calls_in_one_loop_share_one_session() -> None:
    server = FakeMcpServer()
    client = AsyncKeylessClient(transport=httpx.MockTransport(server))

    async def run() -> None:
        for _ in range(3):
            await client.call_tool("search_free", {"query": "mcp"})
        assert server.open_sessions() == 1
        await client.aclose()

    asyncio.run(run())
    assert server.methods().count("initialize") == 1
    assert server.methods().count("tools/call") == 3
    assert server.deleted_sessions == ["sess-1-1"]
    assert server.open_sessions() == 0


def test_new_event_loop_ends_the_old_session(router: Router) -> None:
    client = AsyncKeylessClient()  # default transport, patched to the fakes
    asyncio.run(client.call_tool("search_free", {"query": "a"}))
    asyncio.run(client.call_tool("search_free", {"query": "b"}))
    assert router.mcp.deleted_sessions == ["sess-1-1"]  # ended from the second loop
    assert router.mcp.open_sessions() == 1

    async def finish() -> None:
        await client.aclose()

    asyncio.run(finish())  # aclose from yet another loop still ends the session
    assert router.mcp.open_sessions() == 0


def test_tool_aclose_and_close_end_sessions() -> None:
    server = FakeMcpServer()
    transport = httpx.MockTransport(server)
    tool = ClawifiSearch(
        keyless_client=KeylessClient(transport=transport),
        async_keyless_client=AsyncKeylessClient(transport=transport),
    )

    async def run() -> None:
        await tool.ainvoke({"query": "mcp"})
        await tool.ainvoke({"query": "mcp"})
        await tool.aclose()

    asyncio.run(run())
    tool.invoke({"query": "mcp"})
    tool.close()
    assert server.methods().count("initialize") == 2  # one async session, one sync session
    assert server.open_sessions() == 0


def test_sync_close_without_session_sends_nothing() -> None:
    server = FakeMcpServer(session_id=None)
    client = KeylessClient(transport=httpx.MockTransport(server))
    client.call_tool("fetch_url", {"url": "https://example.com"})
    client.close()
    assert server.deleted_sessions == []


# Finding 3: a custom transport or proxy is never silently dropped ---------------


def test_default_client_is_mirrored(router: Router) -> None:
    out = asyncio.run(ClawifiSearch(client=Clawifi(api_key="k")).ainvoke({"query": "mcp"}))
    assert out["results"]
    assert router.rest[0].headers["authorization"] == "Bearer k"


@pytest.mark.parametrize(
    "client",
    [
        lambda: Clawifi(api_key="k", transport=httpx.MockTransport(lambda r: httpx.Response(200))),
        lambda: Clawifi(
            api_key="k", transport=httpx.HTTPTransport(proxy="http://proxy.invalid:3128")
        ),
    ],
)
def test_custom_sync_client_is_not_mirrored_to_async(client) -> None:
    tool = ClawifiFetch(client=client())
    with pytest.raises(ToolException, match="Pass async_client= as well"):
        asyncio.run(tool.ainvoke({"url": "https://example.com"}))


def test_custom_async_client_is_not_mirrored_to_sync() -> None:
    async_client = AsyncClawifi(
        api_key="k", transport=httpx.MockTransport(lambda r: httpx.Response(200))
    )
    with pytest.raises(ToolException, match="Pass client= as well"):
        ClawifiSearch(async_client=async_client).invoke({"query": "mcp"})


def test_proxy_mount_is_detected() -> None:
    assert http_settings_copyable(Clawifi(api_key="k"))
    proxied = Clawifi(api_key="k")
    proxied._client = httpx.Client(proxy="http://proxy.invalid:3128")
    assert not http_settings_copyable(proxied)

    async def open_async_mirror() -> None:
        async with ClawifiSearch(client=proxied)._async_rest():
            pass

    with pytest.raises(ClawifiClientConfigError, match="Pass async_client= as well"):
        asyncio.run(open_async_mirror())


def test_environment_proxy_counts_as_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
    assert http_settings_copyable(Clawifi(api_key="k"))


# Finding 7: undecodable SSE events (pings) are skipped ----------------------------


def test_sse_ping_events_are_skipped() -> None:
    server = FakeMcpServer(sse=True)
    server.sse_ping = True
    out = ClawifiSearch(keyless_client=KeylessClient(transport=httpx.MockTransport(server))).invoke(
        {"query": "mcp", "max_results": 2}
    )
    assert len(out["results"]) == 2


def test_sse_with_only_unreadable_events_still_fails_clearly() -> None:
    server = FakeMcpServer()
    server.fail_next = httpx.Response(
        200, text="data: ping\n\ndata: {oops\n\n", headers={"content-type": "text/event-stream"}
    )
    tool = ClawifiSearch(keyless_client=KeylessClient(transport=httpx.MockTransport(server)))
    with pytest.raises(ToolException, match="2 could not be decoded"):
        tool.invoke({"query": "mcp"})


# Finding 9: only finite, non-negative Retry-After values -----------------------


@pytest.mark.parametrize("value", ["nan", "NaN", "inf", "-inf", "Infinity", "-1", "-0.5"])
def test_retry_after_rejects_non_finite_and_negative(value: str) -> None:
    assert retry_after_seconds(value) is None


def test_retry_after_accepts_zero_and_finite() -> None:
    assert retry_after_seconds("0") == 0
    assert retry_after_seconds("1.5") == 1.5


def test_nan_retry_after_is_not_reported() -> None:
    server = FakeMcpServer()
    server.fail_next = httpx.Response(
        429, json={"error": {"message": "busy"}}, headers={"Retry-After": "nan"}
    )
    client = KeylessClient(transport=httpx.MockTransport(server))
    with pytest.raises(Exception) as info:
        client.call_tool("search_free", {"query": "mcp"})
    assert info.value.retry_after is None  # type: ignore[attr-defined]
    assert "Retry after" not in str(info.value)


# Finding 10 (checked in this package too): a later max_results change counts ----


def test_max_results_set_after_construction_is_a_user_request() -> None:
    server = FakeMcpServer()
    tool = ClawifiSearch(keyless_client=KeylessClient(transport=httpx.MockTransport(server)))
    assert "Asked for" not in tool.invoke({"query": "mcp"})["keyless_note"]
    tool.max_results = 9
    assert "Asked for 9 results; capped at 3." in tool.invoke({"query": "mcp"})["keyless_note"]
