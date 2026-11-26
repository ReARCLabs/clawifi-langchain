"""Regression tests for the second code review round (2026-11-26, no network)."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from langchain_core.tools import ToolException

from langchain_clawifi import ClawifiSearch
from langchain_clawifi._keyless import (
    AsyncKeylessClient,
    KeylessClient,
    retry_after_seconds,
)
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
