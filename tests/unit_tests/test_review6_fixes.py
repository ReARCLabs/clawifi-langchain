"""Regression tests for the sixth review round (no network)."""

from __future__ import annotations

import asyncio
import copy
import json
import pickle

import httpx
import pytest
from langchain_core.tools import ToolException

from langchain_clawifi import ClawifiFetch, ClawifiSearch
from langchain_clawifi._keyless import AsyncKeylessClient, ClawifiKeylessError, KeylessClient
from tests.unit_tests.conftest import Router
from tests.unit_tests.fake_mcp import FakeMcpServer

COPIES = [
    pytest.param(copy.deepcopy, id="deepcopy"),
    pytest.param(lambda tool: tool.model_copy(deep=True), id="model_copy"),
    pytest.param(lambda tool: pickle.loads(pickle.dumps(tool)), id="pickle"),
]


# 1. Tools can be copied and pickled, fresh and after a call ------------------------


@pytest.mark.parametrize("make_copy", COPIES)
def test_fresh_tools_copy(make_copy) -> None:  # noqa: ANN001
    for tool in (ClawifiSearch(), ClawifiFetch(api_key="k", max_chars=10)):
        clone = make_copy(tool)
        assert type(clone) is type(tool) and clone.name == tool.name
    assert make_copy(ClawifiFetch(api_key="k", max_chars=10)).max_chars == 10


@pytest.mark.parametrize("make_copy", COPIES)
def test_tools_copy_after_keyless_calls_and_the_copy_works(make_copy, router: Router) -> None:  # noqa: ANN001
    tool = ClawifiSearch()
    tool.invoke({"query": "a"})  # builds the sync keyless client and its HTTP client
    asyncio.run(tool.ainvoke({"query": "b"}))
    clone = make_copy(tool)
    assert clone.invoke({"query": "c"})["results"]
    assert asyncio.run(clone.ainvoke({"query": "d"}))["results"]
    assert tool.invoke({"query": "e"})["results"]  # the original still works too


@pytest.mark.parametrize("make_copy", COPIES)
def test_tools_copy_after_a_rest_call_and_the_copy_works(make_copy, router: Router) -> None:  # noqa: ANN001
    tool = ClawifiFetch(api_key="k")
    tool.invoke({"url": "https://example.com"})  # builds the tool's own sync REST client
    clone = make_copy(tool)
    assert clone._sync is None  # left out of the copy, rebuilt on demand
    assert clone.invoke({"url": "https://example.com"})["title"] == "Example Domain"
    assert [r.headers["authorization"] for r in router.rest] == ["Bearer k", "Bearer k"]


@pytest.mark.parametrize("make_copy", [copy.deepcopy, lambda c: pickle.loads(pickle.dumps(c))])
def test_keyless_clients_copy_after_use(make_copy, router: Router) -> None:  # noqa: ANN001
    client = KeylessClient()
    client.call_tool("search_free", {"query": "a"})
    clone = make_copy(client)
    assert clone._http is None
    assert clone.call_tool("search_free", {"query": "b"})["success"] is True
    async_clone = make_copy(AsyncKeylessClient())
    assert asyncio.run(async_clone.call_tool("search_free", {"query": "c"}))["success"] is True


# 2. Only explicit "not initialized" signals trigger the initialize fallback --------


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(400, text="invalid session parameter"),
        httpx.Response(400, json={"error": {"message": "session_token must be a string"}}),
        httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "error": {"code": -32602, "message": "invalid session parameter"},
            },
        ),
    ],
)
def test_unrelated_session_messages_do_not_initialize(response: httpx.Response) -> None:
    server = FakeMcpServer()
    server.fail_next = response
    tool = ClawifiSearch(keyless_client=KeylessClient(transport=httpx.MockTransport(server)))
    with pytest.raises(ToolException, match="session"):
        tool.invoke({"query": "a"})
    assert server.methods() == ["tools/call"]  # no initialize


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(400, text="Bad Request: Missing session ID"),
        httpx.Response(400, text="Bad Request: Mcp-Session-Id header is required"),
        httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "error": {"code": -32600, "message": "Server not initialized"},
            },
        ),
        httpx.Response(
            200, json={"jsonrpc": "2.0", "id": 1, "error": {"code": -32002, "message": "nope"}}
        ),
    ],
)
def test_explicit_signals_initialize(response: httpx.Response) -> None:
    server = FakeMcpServer()
    server.fail_next = response
    out = KeylessClient(transport=httpx.MockTransport(server)).call_tool(
        "search_free", {"query": "a"}
    )
    assert out["success"] is True
    assert server.methods() == [
        "tools/call",
        "initialize",
        "notifications/initialized",
        "tools/call",
    ]


# 3. A JSON-RPC error with a null id is reported, not lost ---------------------------


NULL_ID_ERROR = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}


def null_id_response(sse: bool) -> httpx.Response:
    if sse:
        return httpx.Response(
            200,
            text=f"event: message\ndata: {json.dumps(NULL_ID_ERROR)}\n\n",
            headers={"content-type": "text/event-stream"},
        )
    return httpx.Response(200, json=NULL_ID_ERROR)


@pytest.mark.parametrize("sse", [False, True])
def test_null_id_error_is_raised_sync(sse: bool) -> None:
    server = FakeMcpServer()
    server.fail_next = null_id_response(sse)
    with pytest.raises(ClawifiKeylessError, match="Parse error") as info:
        KeylessClient(transport=httpx.MockTransport(server)).call_tool(
            "search_free", {"query": "a"}
        )
    assert info.value.code == -32700


@pytest.mark.parametrize("sse", [False, True])
def test_null_id_error_is_raised_async(sse: bool) -> None:
    server = FakeMcpServer()
    server.fail_next = null_id_response(sse)
    client = AsyncKeylessClient(transport=httpx.MockTransport(server))
    with pytest.raises(ClawifiKeylessError, match="Parse error") as info:
        asyncio.run(client.call_tool("search_free", {"query": "a"}))
    assert info.value.code == -32700


def test_matching_id_wins_over_a_null_id_error() -> None:
    server = FakeMcpServer()
    body = [
        {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}},
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"type": "text", "text": '{"success": true}'}]},
        },
    ]
    server.fail_next = httpx.Response(200, json=body)
    assert KeylessClient(transport=httpx.MockTransport(server)).call_tool(
        "search_free", {"query": "a"}
    ) == {"success": True}
