"""Regression tests for the fourth review round: per-call keyless sessions (no network)."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from clawifi import AsyncClawifi, Clawifi
from langchain_core.tools import ToolException

from langchain_clawifi import ClawifiFetch, ClawifiSearch
from langchain_clawifi._keyless import AsyncKeylessClient, ClawifiKeylessError, KeylessClient
from tests.unit_tests.conftest import Router
from tests.unit_tests.fake_mcp import FakeMcpServer


def async_tool(server: FakeMcpServer) -> ClawifiSearch:
    return ClawifiSearch(
        async_keyless_client=AsyncKeylessClient(transport=httpx.MockTransport(server))
    )


# 1. Every call opens and ends its own session -----------------------------------


def test_two_concurrent_ainvokes_open_two_sessions_and_end_both() -> None:
    server = FakeMcpServer()
    tool = async_tool(server)

    async def both() -> list:
        return await asyncio.gather(tool.ainvoke({"query": "a"}), tool.ainvoke({"query": "b"}))

    first, second = asyncio.run(both())
    assert first["results"] and second["results"]
    assert server.sessions_opened == 2
    assert sorted(server.deleted_sessions) == ["sess-1-1", "sess-1-2"]
    assert server.open_sessions() == 0
    calls = [r for r, body in server.requests if body.get("method") == "tools/call"]
    assert sorted(r.headers["mcp-session-id"] for r in calls) == ["sess-1-1", "sess-1-2"]


def test_sequential_sync_calls_each_end_their_session() -> None:
    server = FakeMcpServer()
    tool = ClawifiSearch(keyless_client=KeylessClient(transport=httpx.MockTransport(server)))
    tool.invoke({"query": "a"})
    tool.invoke({"query": "b"})
    assert server.methods().count("initialize") == 2
    assert server.deleted_sessions == ["sess-1-1", "sess-1-2"]
    assert server.open_sessions() == 0


@pytest.mark.parametrize(
    "setup",
    [
        lambda s: setattr(s, "tool_error", "Sign in to use this tool"),
        lambda s: setattr(s, "fail_next", httpx.Response(429, json={"error": {"message": "busy"}})),
        lambda s: setattr(s, "rpc_error", {"code": -32602, "message": "bad args"}),
    ],
)
def test_session_is_deleted_when_the_call_fails(setup) -> None:  # noqa: ANN001
    for make_call in (
        lambda s: KeylessClient(transport=httpx.MockTransport(s)).call_tool(
            "search_free", {"query": "a"}
        ),
        lambda s: asyncio.run(
            AsyncKeylessClient(transport=httpx.MockTransport(s)).call_tool(
                "search_free", {"query": "a"}
            )
        ),
    ):
        server = FakeMcpServer()
        setup(server)
        with pytest.raises(ClawifiKeylessError):
            make_call(server)
        assert server.deleted_sessions == ["sess-1-1"]
        assert server.open_sessions() == 0


def _assert_lost_session_handled(server: FakeMcpServer, out: dict) -> None:
    assert out["success"] is True
    assert server.methods().count("initialize") == 2
    assert server.deleted_sessions == ["sess-1-2"]  # the lost session is already gone
    assert server.open_sessions() == 0


def test_lost_session_404_retries_once_sync() -> None:
    server = FakeMcpServer()
    server.expire_session_once = True
    out = KeylessClient(transport=httpx.MockTransport(server)).call_tool(
        "search_free", {"query": "a"}
    )
    _assert_lost_session_handled(server, out)


def test_lost_session_404_retries_once_async() -> None:
    server = FakeMcpServer()
    server.expire_session_once = True
    client = AsyncKeylessClient(transport=httpx.MockTransport(server))
    out = asyncio.run(client.call_tool("search_free", {"query": "a"}))
    _assert_lost_session_handled(server, out)


def test_delete_failures_are_ignored() -> None:
    server = FakeMcpServer()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            raise RuntimeError("client has been closed")
        return server(request)

    assert KeylessClient(transport=httpx.MockTransport(handler)).call_tool(
        "search_free", {"query": "a"}
    )
    assert asyncio.run(
        AsyncKeylessClient(transport=httpx.MockTransport(handler)).call_tool(
            "search_free", {"query": "a"}
        )
    )


def test_no_shared_session_state_on_the_clients() -> None:
    sync_client = KeylessClient()
    async_client = AsyncKeylessClient()
    for client in (sync_client, async_client):
        assert not hasattr(client, "_session")
        assert not hasattr(client, "session_id")


# 2. close()/aclose() --------------------------------------------------------------


def test_close_only_drops_the_sync_http_client() -> None:
    server = FakeMcpServer()
    client = KeylessClient(transport=httpx.MockTransport(server))
    client.call_tool("search_free", {"query": "a"})
    client.close()
    assert client._http is None
    assert server.deleted_sessions == ["sess-1-1"]  # ended by the call itself, not by close()
    client.call_tool("search_free", {"query": "b"})  # a new HTTP client is built
    assert server.open_sessions() == 0


def test_async_aclose_is_a_no_op() -> None:
    server = FakeMcpServer()
    client = AsyncKeylessClient(transport=httpx.MockTransport(server))
    asyncio.run(client.call_tool("search_free", {"query": "a"}))
    assert server.deleted_sessions == ["sess-1-1"]  # ended by the call
    asyncio.run(client.aclose())
    assert server.deleted_sessions == ["sess-1-1"]  # aclose sends nothing more


# 3. Error wrapping includes InvalidURL and StreamError ----------------------------


def test_invalid_url_is_wrapped() -> None:
    tool = ClawifiFetch(api_base="http://[bad")
    with pytest.raises(ToolException, match="unreachable"):
        tool.invoke({"url": "https://example.com"})
    with pytest.raises(ClawifiKeylessError):
        asyncio.run(AsyncKeylessClient(api_base="http://[bad").call_tool("fetch_url", {"url": "x"}))


@pytest.mark.parametrize(
    "error",
    [
        httpx.StreamConsumed(),
        httpx.InvalidURL("bad"),
        RuntimeError("closed"),
        httpx.ConnectError("x"),
    ],
)
def test_all_transport_errors_are_wrapped(error: Exception) -> None:
    def broken(request: httpx.Request) -> httpx.Response:
        raise error

    with pytest.raises(ClawifiKeylessError):
        KeylessClient(transport=httpx.MockTransport(broken)).call_tool(
            "search_free", {"query": "a"}
        )
    with pytest.raises(ClawifiKeylessError):
        asyncio.run(
            AsyncKeylessClient(transport=httpx.MockTransport(broken)).call_tool(
                "search_free", {"query": "a"}
            )
        )


# 5. A key plus one client: the other mode is built with the key ------------------


def test_key_and_only_sync_client_async_call_uses_the_key(router: Router) -> None:
    tool = ClawifiSearch(api_key="k", client=Clawifi(api_key="sync"))
    out = asyncio.run(tool.ainvoke({"query": "mcp"}))
    assert out["results"]
    assert router.rest[0].headers["authorization"] == "Bearer k"


def test_env_key_and_only_async_client_sync_call_uses_the_key(
    router: Router, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAWIFI_API_KEY", "env-key")
    tool = ClawifiFetch(async_client=AsyncClawifi(api_key="async"))
    tool.invoke({"url": "https://example.com"})
    assert router.rest[0].headers["authorization"] == "Bearer env-key"


def test_no_key_and_one_client_still_fails_clearly() -> None:
    tool = ClawifiSearch(client=Clawifi(api_key="sync"))
    with pytest.raises(ToolException, match="Pass async_client too"):
        asyncio.run(tool.ainvoke({"query": "mcp"}))
