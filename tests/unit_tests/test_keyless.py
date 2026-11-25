from __future__ import annotations

import httpx
import pytest
from langchain_core.tools import ToolException

from langchain_clawifi import ClawifiFetch, ClawifiSearch
from langchain_clawifi._keyless import (
    AsyncKeylessClient,
    ClawifiKeylessError,
    KeylessClient,
    mcp_url,
)
from tests.unit_tests.fake_mcp import FakeMcpServer


def keyless_pair(server: FakeMcpServer) -> dict:
    transport = httpx.MockTransport(server)
    return {
        "keyless_client": KeylessClient(transport=transport),
        "async_keyless_client": AsyncKeylessClient(transport=transport),
    }


def test_search_keyless_handshake_and_cap() -> None:
    server = FakeMcpServer()
    tool = ClawifiSearch(**keyless_pair(server))

    out = tool.invoke({"query": "mcp"})

    assert server.methods() == ["initialize", "notifications/initialized", "tools/call"]
    init = server.requests[0][1]["params"]
    assert init["protocolVersion"] == "2025-06-18" and init["clientInfo"]["name"]
    assert server.calls() == [{"name": "search_free", "arguments": {"query": "mcp"}}]
    assert [r["rank"] for r in out["results"]] == [1, 2, 3]
    assert set(out["results"][0]) == {"rank", "title", "url", "snippet"}
    assert "at most 3 results" in out["keyless_note"]
    assert "Asked for 5 results; capped at 3." in out["keyless_note"]  # tool default is 5


def test_search_keyless_reports_dropped_filters_and_reuses_session() -> None:
    server = FakeMcpServer()
    tool = ClawifiSearch(**keyless_pair(server))
    out = tool.invoke(
        {
            "query": "mcp",
            "max_results": 2,
            "time_range": "week",
            "include_domains": ["a.com"],
            "exclude_domains": ["b.com"],
        }
    )
    assert len(out["results"]) == 2
    assert "Not applied: time_range, include_domains, exclude_domains." in out["keyless_note"]
    assert "capped" not in out["keyless_note"]
    assert server.calls()[0]["arguments"] == {"query": "mcp"}  # filters are never sent
    tool.invoke({"query": "again"})
    assert server.methods().count("initialize") == 1


def test_fetch_keyless_same_fields_as_keyed() -> None:
    server = FakeMcpServer()
    tool = ClawifiFetch(**keyless_pair(server))
    out = tool.invoke({"url": "https://example.com", "query": "what is it"})
    assert server.calls() == [
        {"name": "fetch_url", "arguments": {"url": "https://example.com", "query": "what is it"}}
    ]
    assert out == {
        "url": "https://example.com/",
        "title": "Example Domain",
        "content_status": "ok",
        "content": "Body",
    }


def test_sse_replies_are_decoded() -> None:
    server = FakeMcpServer(sse=True)
    out = ClawifiSearch(**keyless_pair(server)).invoke({"query": "mcp", "max_results": 1})
    assert out["results"] == [
        {"rank": 1, "title": "T1", "url": "https://example.com/1", "snippet": "s1"}
    ]


def test_structured_content_is_used() -> None:
    server = FakeMcpServer(structured=True)
    out = ClawifiFetch(**keyless_pair(server)).invoke({"url": "https://example.com"})
    assert out["title"] == "Example Domain"


def test_server_without_session_header() -> None:
    server = FakeMcpServer(session_id=None)
    out = ClawifiFetch(**keyless_pair(server)).invoke({"url": "https://example.com"})
    assert out["content"] == "Body"
    assert all("mcp-session-id" not in r.headers for r, _ in server.requests)


def test_expired_session_is_reopened_once() -> None:
    server = FakeMcpServer()
    server.expire_session_once = True
    out = ClawifiFetch(**keyless_pair(server)).invoke({"url": "https://example.com"})
    assert out["content"] == "Body"
    assert server.methods().count("initialize") == 2


def test_rate_limit_429_carries_retry_after() -> None:
    server = FakeMcpServer()
    server.fail_next = httpx.Response(
        429,
        json={"error": {"code": "rate_limited", "message": "Too many keyless calls"}},
        headers={"Retry-After": "7"},
    )
    client = KeylessClient(transport=httpx.MockTransport(server))
    with pytest.raises(ClawifiKeylessError) as info:
        client.call_tool("search_free", {"query": "mcp"})
    assert info.value.status == 429
    assert info.value.retry_after == 7
    assert "Too many keyless calls" in str(info.value) and "Retry after 7 s" in str(info.value)


def test_rate_limit_becomes_tool_exception() -> None:
    server = FakeMcpServer()
    server.fail_next = httpx.Response(429, text="slow down", headers={"Retry-After": "3"})
    with pytest.raises(ToolException, match="HTTP 429"):
        ClawifiSearch(**keyless_pair(server)).invoke({"query": "mcp"})


def test_daily_limit_401() -> None:
    server = FakeMcpServer()
    server.fail_next = httpx.Response(
        401, json={"error": "keyless_limit", "message": "keyless limit reached"}
    )
    with pytest.raises(ToolException, match="keyless limit reached"):
        ClawifiFetch(**keyless_pair(server)).invoke({"url": "https://example.com"})


def test_tool_error_result() -> None:
    server = FakeMcpServer()
    server.tool_error = "Sign in to use this tool"
    with pytest.raises(ToolException, match="Sign in to use this tool"):
        ClawifiFetch(**keyless_pair(server)).invoke({"url": "https://example.com"})


def test_jsonrpc_error() -> None:
    server = FakeMcpServer(sse=True)
    server.rpc_error = {"code": -32602, "message": "Unknown tool"}
    with pytest.raises(ToolException, match="Unknown tool"):
        ClawifiSearch(**keyless_pair(server)).invoke({"query": "mcp"})


def test_unreachable_server() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = KeylessClient(transport=httpx.MockTransport(down))
    with pytest.raises(ToolException, match="unreachable"):
        ClawifiSearch(keyless_client=client).invoke({"query": "mcp"})


async def test_async_search_and_fetch_keyless() -> None:
    server = FakeMcpServer(sse=True)
    pair = keyless_pair(server)
    out = await ClawifiSearch(**pair).ainvoke({"query": "mcp"})
    assert len(out["results"]) == 3 and out["keyless_note"]
    page = await ClawifiFetch(**pair).ainvoke({"url": "https://example.com"})
    assert page["title"] == "Example Domain"


async def test_async_rate_limit() -> None:
    server = FakeMcpServer()
    server.fail_next = httpx.Response(
        429, json={"error": {"message": "busy"}}, headers={"Retry-After": "2"}
    )
    with pytest.raises(ToolException, match="busy"):
        await ClawifiSearch(**keyless_pair(server)).ainvoke({"query": "mcp"})


def test_key_present_uses_rest_not_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeMcpServer()
    tool = ClawifiSearch(api_key="k", **keyless_pair(server))
    assert tool._keyless(is_async=False) is False
    monkeypatch.setenv("CLAWIFI_API_KEY", "env-key")
    assert ClawifiFetch()._keyless(is_async=True) is False


def test_mcp_url_follows_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    assert mcp_url() == "https://api.clawifi.dev/mcp"
    assert mcp_url("https://staging.example/") == "https://staging.example/mcp"
    monkeypatch.setenv("CLAWIFI_API_BASE", "https://env.example")
    assert mcp_url() == "https://env.example/mcp"
