"""Regression tests for the fifth review round: the single-POST model (no network)."""

from __future__ import annotations

import asyncio
import threading
import time

import httpx
import pytest
from clawifi import AsyncClawifi, Clawifi
from langchain_core.tools import ToolException

import langchain_clawifi._keyless as keyless_module
from langchain_clawifi import ClawifiFetch, ClawifiSearch
from langchain_clawifi._keyless import (
    DELETE_TIMEOUT,
    AsyncKeylessClient,
    ClawifiKeylessError,
    KeylessClient,
    clean_key,
    resolve_key,
)
from tests.unit_tests.conftest import Router
from tests.unit_tests.fake_mcp import FakeMcpServer

ARGS = ("search_free", {"query": "mcp"})


def sync_call(server, **kwargs):  # noqa: ANN001, ANN201
    return KeylessClient(transport=httpx.MockTransport(server), **kwargs).call_tool(*ARGS)


def async_call(server):  # noqa: ANN001, ANN201
    return asyncio.run(AsyncKeylessClient(transport=httpx.MockTransport(server)).call_tool(*ARGS))


CALLERS = [pytest.param(sync_call, id="sync"), pytest.param(async_call, id="async")]


# 1. One POST per call; initialize only when the server asks ----------------------


@pytest.mark.parametrize("call", CALLERS)
@pytest.mark.parametrize("sse", [False, True])
def test_single_post_success(call, sse: bool) -> None:  # noqa: ANN001
    server = FakeMcpServer(sse=sse)
    assert call(server)["success"] is True
    assert server.methods() == ["tools/call"]
    assert server.delete_requests == []
    request = server.requests[0][0]
    assert request.headers["mcp-protocol-version"] == "2025-06-18"
    assert "mcp-session-id" not in request.headers


@pytest.mark.parametrize("call", CALLERS)
@pytest.mark.parametrize("style", ["rpc-code", "rpc-message", "http400"])
def test_not_initialized_falls_back_to_initialize(call, style: str) -> None:  # noqa: ANN001
    server = FakeMcpServer(require_init=True, init_error=style)
    assert call(server)["success"] is True
    assert server.methods() == [
        "tools/call",
        "initialize",
        "notifications/initialized",
        "tools/call",
    ]
    assert server.deleted_sessions == ["sess-1-1"]  # the session it was given is ended
    assert server.open_sessions() == 0


@pytest.mark.parametrize("call", CALLERS)
def test_delete_uses_a_short_timeout(call) -> None:  # noqa: ANN001
    server = FakeMcpServer(require_init=True)
    call(server)
    timeout = server.delete_requests[0].extensions["timeout"]
    assert DELETE_TIMEOUT <= 2 and all(v == DELETE_TIMEOUT for v in timeout.values())


@pytest.mark.parametrize("call", CALLERS)
def test_delete_only_when_the_server_gave_a_session_id(call) -> None:  # noqa: ANN001
    server = FakeMcpServer(require_init=True, session_id=None)
    assert call(server)["success"] is True
    assert "initialize" in server.methods()
    assert server.delete_requests == []


@pytest.mark.parametrize("call", CALLERS)
def test_server_still_not_initialized_is_a_clear_error(call) -> None:  # noqa: ANN001
    server = FakeMcpServer()
    refuse = httpx.Response(400, text="Bad Request: Missing session ID")

    def handler(request: httpx.Request) -> httpx.Response:
        response = server(request)
        if request.method == "POST" and b'"tools/call"' in request.content:
            return refuse
        return response

    with pytest.raises(ClawifiKeylessError, match="still asks for initialization"):
        call(handler)


@pytest.mark.parametrize("call", CALLERS)
def test_failed_delete_is_ignored(call) -> None:  # noqa: ANN001
    server = FakeMcpServer(require_init=True)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            raise httpx.ReadTimeout("slow")
        return server(request)

    assert call(handler)["success"] is True


def test_cancellation_skips_the_delete() -> None:
    server = FakeMcpServer(require_init=True)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and server.methods().count("initialize"):
            if b'"tools/call"' in request.content:
                raise asyncio.CancelledError
        return server(request)

    with pytest.raises(asyncio.CancelledError):
        async_call(handler)
    assert server.delete_requests == []


def test_plain_400_is_not_mistaken_for_initialization() -> None:
    server = FakeMcpServer()
    server.fail_next = httpx.Response(400, json={"error": {"message": "query is required"}})
    with pytest.raises(ClawifiKeylessError) as info:
        sync_call(server)
    assert info.value.status == 400
    assert server.methods() == ["tools/call"]


# 2. Only httpx errors and the closed-client RuntimeError are wrapped ---------------


@pytest.mark.parametrize("call", CALLERS)
@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("refused"),
        httpx.StreamConsumed(),
        httpx.InvalidURL("bad"),
        RuntimeError("Cannot send a request, as the client has been closed."),
    ],
)
def test_transport_errors_are_wrapped(call, error: Exception) -> None:  # noqa: ANN001
    def broken(request: httpx.Request) -> httpx.Response:
        raise error

    with pytest.raises(ClawifiKeylessError, match="unreachable"):
        call(broken)


@pytest.mark.parametrize("call", CALLERS)
def test_other_runtime_errors_propagate(call) -> None:  # noqa: ANN001
    def broken(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("a bug somewhere else")

    with pytest.raises(RuntimeError, match="a bug somewhere else") as info:
        call(broken)
    assert not isinstance(info.value, ClawifiKeylessError)


def test_closed_client_is_rebuilt() -> None:
    server = FakeMcpServer()
    client = KeylessClient(transport=httpx.MockTransport(server))
    client.call_tool(*ARGS)
    client._client().close()
    assert client.call_tool(*ARGS)["success"] is True


# 3. A close error never hides the call's own error -----------------------------


def test_429_details_survive_a_failing_close(monkeypatch: pytest.MonkeyPatch) -> None:
    async def failing_aclose(self: httpx.AsyncClient) -> None:
        raise RuntimeError("Cannot close: the client has been closed")

    monkeypatch.setattr(httpx.AsyncClient, "aclose", failing_aclose)
    server = FakeMcpServer()
    server.fail_next = httpx.Response(
        429, json={"error": {"message": "busy"}}, headers={"Retry-After": "7"}
    )
    with pytest.raises(ClawifiKeylessError) as info:
        async_call(server)
    assert info.value.status == 429 and info.value.retry_after == 7


def test_close_error_after_success_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    async def failing_aclose(self: httpx.AsyncClient) -> None:
        raise httpx.ReadError("close failed")

    monkeypatch.setattr(httpx.AsyncClient, "aclose", failing_aclose)
    with pytest.raises(ClawifiKeylessError, match="close failed"):
        async_call(FakeMcpServer())


def test_429_details_reach_the_tool() -> None:
    server = FakeMcpServer()
    server.fail_next = httpx.Response(
        429, json={"error": {"message": "busy"}}, headers={"Retry-After": "5"}
    )
    tool = ClawifiSearch(
        async_keyless_client=AsyncKeylessClient(transport=httpx.MockTransport(server))
    )
    with pytest.raises(ToolException, match="HTTP 429.*Retry after 5 s"):
        asyncio.run(tool.ainvoke({"query": "mcp"}))


# 4. Concurrent first sync calls build one HTTP client ----------------------------


def test_concurrent_first_sync_calls_build_one_client(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[object] = []
    real_client = httpx.Client

    class SlowClient(real_client):  # type: ignore[misc, valid-type]
        def __init__(self, *args: object, **kwargs: object) -> None:
            time.sleep(0.05)  # widen the race window
            built.append(self)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(keyless_module.httpx, "Client", SlowClient)
    server = FakeMcpServer()
    client = KeylessClient(transport=httpx.MockTransport(server))
    threads = [threading.Thread(target=client.call_tool, args=ARGS) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(built) == 1
    assert server.methods().count("tools/call") == 8


# 5. The client built with the key keeps the given client's api_base -------------


def test_async_client_built_with_key_copies_api_base(router: Router) -> None:
    given = Clawifi(api_key="sync", api_base="https://alt.example")
    tool = ClawifiSearch(api_key="k", client=given)
    asyncio.run(tool.ainvoke({"query": "mcp"}))
    request = router.rest[0]
    assert request.url.host == "alt.example"
    assert request.headers["authorization"] == "Bearer k"


def test_sync_client_built_with_key_copies_api_base(router: Router) -> None:
    given = AsyncClawifi(api_key="async", api_base="https://alt.example")
    ClawifiFetch(api_key="k", async_client=given).invoke({"url": "https://example.com"})
    assert router.rest[0].url.host == "alt.example"


def test_tool_api_base_wins_over_the_given_client(router: Router) -> None:
    given = Clawifi(api_key="sync", api_base="https://alt.example")
    tool = ClawifiFetch(api_key="k", client=given, api_base="https://own.example")
    asyncio.run(tool.ainvoke({"url": "https://example.com"}))
    assert router.rest[0].url.host == "own.example"


# 6. A whitespace key counts as no key, for the argument and the env ----------------


def test_key_helpers() -> None:
    assert clean_key(None) is None
    assert clean_key("   ") is None
    assert clean_key(" k ") == "k"


def test_whitespace_argument_key_means_keyless() -> None:
    assert ClawifiSearch(api_key="   ")._keyless() is True


def test_whitespace_env_key_means_keyless(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAWIFI_API_KEY", " \t ")
    assert resolve_key(None) is None
    assert ClawifiSearch()._keyless() is True


def test_whitespace_key_with_one_client_still_fails_clearly() -> None:
    tool = ClawifiSearch(api_key="  ", client=Clawifi(api_key="sync"))
    with pytest.raises(ToolException, match="Pass async_client too"):
        asyncio.run(tool.ainvoke({"query": "mcp"}))


def test_padded_key_is_used_stripped(router: Router) -> None:
    ClawifiFetch(api_key="  k  ").invoke({"url": "https://example.com"})
    assert router.rest[0].headers["authorization"] == "Bearer k"
