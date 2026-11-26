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


# 1. No session state is kept on the clients ------------------------------------


def test_no_shared_session_state_on_the_clients() -> None:
    sync_client = KeylessClient()
    async_client = AsyncKeylessClient()
    for client in (sync_client, async_client):
        assert not hasattr(client, "_session")
        assert not hasattr(client, "session_id")


# 3. Error wrapping includes InvalidURL and StreamError ----------------------------


def test_invalid_url_is_wrapped() -> None:
    tool = ClawifiFetch(api_base="http://exa\tmple")
    with pytest.raises(ToolException, match="unreachable"):
        tool.invoke({"url": "https://example.com"})
    with pytest.raises(ClawifiKeylessError):
        asyncio.run(
            AsyncKeylessClient(api_base="http://exa\tmple").call_tool("fetch_url", {"url": "x"})
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
