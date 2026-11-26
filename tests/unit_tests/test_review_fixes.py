"""Regression tests for the 2026-11-26 code review findings (no network)."""

from __future__ import annotations

import asyncio
import inspect
import json
import tomllib
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from clawifi import AsyncClawifi, Clawifi
from langchain_core.tools import ToolException

import langchain_clawifi._keyless as keyless_module
from langchain_clawifi import ClawifiFetch, ClawifiSearch
from langchain_clawifi._keyless import KeylessClient, retry_after_seconds
from tests.unit_tests.conftest import Router
from tests.unit_tests.fake_mcp import FakeMcpServer

# -- Finding 1: async clients must not stay bound to the first event loop ----------


def test_keyless_async_works_across_two_event_loops(router: Router) -> None:
    tool = ClawifiSearch()  # no key: keyless MCP path, client built by the tool
    first = asyncio.run(tool.ainvoke({"query": "mcp"}))
    second = asyncio.run(tool.ainvoke({"query": "mcp"}))
    assert first["results"] and second["results"]
    assert router.mcp.methods().count("tools/call") == 2


def test_rest_async_works_across_two_event_loops(router: Router) -> None:
    tool = ClawifiFetch(api_key="k")
    for _ in range(2):
        out = asyncio.run(tool.ainvoke({"url": "https://example.com"}))
        assert out["title"] == "Example Domain"
    assert [r.url.path for r in router.rest] == ["/fetch", "/fetch"]
    assert all(r.headers["authorization"] == "Bearer k" for r in router.rest)


# -- Finding 2: unreadable replies become ClawifiKeylessError / ToolException -------


def _tool_with(server: FakeMcpServer) -> ClawifiSearch:
    return ClawifiSearch(keyless_client=KeylessClient(transport=httpx.MockTransport(server)))


@pytest.mark.parametrize(
    "response, message",
    [
        (
            httpx.Response(
                200, text="data: {not json\n\n", headers={"content-type": "text/event-stream"}
            ),
            "could not be read",
        ),
        (
            httpx.Response(200, text="<html>oops</html>", headers={"content-type": "text/html"}),
            "could not be read",
        ),
        (
            httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "error": "Unknown tool"}),
            "Unknown tool",
        ),
        (
            httpx.Response(
                200, json={"jsonrpc": "2.0", "id": 1, "result": ["not", "an", "object"]}
            ),
            "not an object",
        ),
    ],
)
def test_unreadable_replies_become_tool_exception(response: httpx.Response, message: str) -> None:
    server = FakeMcpServer()
    server.fail_next = response
    with pytest.raises(ToolException, match=message):
        _tool_with(server).invoke({"query": "mcp"})


def test_unreadable_reply_is_a_keyless_error() -> None:
    server = FakeMcpServer()
    server.fail_next = httpx.Response(
        200, text="nope", headers={"content-type": "application/json"}
    )
    client = KeylessClient(transport=httpx.MockTransport(server))
    with pytest.raises(keyless_module.ClawifiKeylessError) as info:
        client.call_tool("search_free", {"query": "mcp"})
    assert info.value.status == 200


# -- Finding 3: any pre-built REST client means REST, on both sync and async paths --


def test_keyless_only_without_key_and_without_clients() -> None:
    assert ClawifiSearch()._keyless() is True
    assert ClawifiSearch(client=Clawifi())._keyless() is False
    assert ClawifiSearch(async_client=AsyncClawifi())._keyless() is False
    assert ClawifiSearch(api_key="k")._keyless() is False


# -- Finding 6: the note only claims a cap when the user asked for more than 3 ----


def test_note_only_mentions_cap_for_user_requests() -> None:
    def note(tool: ClawifiSearch, **args: Any) -> str:
        return tool.invoke({"query": "mcp", **args})["keyless_note"]

    server = FakeMcpServer()
    default_tool = _tool_with(server)
    assert "Asked for" not in note(default_tool)
    assert "Asked for" not in note(default_tool, max_results=3)
    assert "Asked for 10 results; capped at 3." in note(default_tool, max_results=10)
    configured = ClawifiSearch(
        max_results=8, keyless_client=KeylessClient(transport=httpx.MockTransport(server))
    )
    assert "Asked for 8 results; capped at 3." in note(configured)


# -- Finding 9: httpx is a declared dependency; no private clawifi helpers --------


def test_httpx_is_a_declared_dependency() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    deps = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    assert any(dep.replace(" ", "").startswith("httpx>=") for dep in deps)


def test_no_private_clawifi_module_is_imported() -> None:
    source = inspect.getsource(keyless_module)
    assert "clawifi.errors" not in source


def test_retry_after_seconds() -> None:
    now = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    assert retry_after_seconds("7") == 7
    assert retry_after_seconds(" 2.5 ") == 2.5
    assert retry_after_seconds("-3") is None
    assert (
        retry_after_seconds(format_datetime(now + timedelta(seconds=30), usegmt=True), now=now)
        == 30
    )
    assert (
        retry_after_seconds(format_datetime(now - timedelta(seconds=30), usegmt=True), now=now) == 0
    )
    assert retry_after_seconds("soon") is None
    assert retry_after_seconds(None) is None
    assert retry_after_seconds("") is None


def test_retry_after_http_date_reaches_the_error() -> None:
    server = FakeMcpServer()
    later = datetime.now(UTC) + timedelta(seconds=120)
    server.fail_next = httpx.Response(
        429,
        json={"error": {"message": "busy"}},
        headers={"Retry-After": format_datetime(later, usegmt=True)},
    )
    client = KeylessClient(transport=httpx.MockTransport(server))
    with pytest.raises(keyless_module.ClawifiKeylessError) as info:
        client.call_tool("search_free", {"query": "mcp"})
    assert info.value.retry_after is not None and 100 < info.value.retry_after <= 120
    assert json.dumps(info.value.status) == "429"
