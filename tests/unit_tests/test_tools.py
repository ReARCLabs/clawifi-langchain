from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from clawifi import AsyncClawifi, Clawifi
from langchain_core.tools import ToolException

from langchain_clawifi import ClawifiFetch, ClawifiSearch

SEARCH_RESPONSE = {
    "success": True,
    "query": "mcp",
    "provider": "free_auto",
    "providers_tried": ["engine_a"],
    "paid_providers_used": [],
    "result_count": 2,
    "results": [
        {
            "url": "https://example.com/a",
            "title": "A",
            "snippet": "first",
            "rank": 1,
            "provider": "engine_a",
            "published_at": None,
            "data": {},
        },
        {
            "url": "https://example.com/b",
            "title": "B",
            "snippet": "second",
            "rank": 2,
            "provider": "engine_a",
            "data": {},
        },
    ],
    "errors": [],
    "cache_hit": False,
}

FETCH_RESPONSE = {
    "success": True,
    "source": "generic_web",
    "mode_used": "http_fetch",
    "entity_type": "WebDocument",
    "content_status": "ok",
    "cache_hit": False,
    "errors": [],
    "data": {
        "url": "https://example.com",
        "title": "Example Domain",
        "content_markdown": "# Example Domain\n\nThis domain is for use in examples.",
    },
}


class Recorder:
    def __init__(self, routes: dict[str, tuple[int, dict[str, Any]]]) -> None:
        self.routes = routes
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status, payload = self.routes[request.url.path]
        return httpx.Response(status, json=payload)

    @property
    def last_body(self) -> dict[str, Any]:
        return json.loads(self.requests[-1].content)


def sync_client(rec: Recorder, api_key: str | None = "test-key") -> Clawifi:
    return Clawifi(api_key=api_key, transport=httpx.MockTransport(rec), max_retries=0)


def async_client(rec: Recorder, api_key: str | None = "test-key") -> AsyncClawifi:
    return AsyncClawifi(api_key=api_key, transport=httpx.MockTransport(rec), max_retries=0)


def test_search_sends_query_and_formats_results() -> None:
    rec = Recorder({"/search/free": (200, SEARCH_RESPONSE)})
    tool = ClawifiSearch(client=sync_client(rec), max_results=3)

    out = tool.invoke({"query": "mcp", "time_range": "week"})

    assert rec.requests[0].method == "POST"
    assert rec.last_body == {"query": "mcp", "max_results": 3, "time_range": "week"}
    assert rec.requests[0].headers["Authorization"] == "Bearer test-key"
    assert set(out) == {"results"}  # no keyless_note with a key
    assert out["results"][0] == {
        "rank": 1,
        "title": "A",
        "url": "https://example.com/a",
        "snippet": "first",
    }
    assert len(out["results"]) == 2


def test_search_keyless_sends_no_authorization_header() -> None:
    rec = Recorder({"/search/free": (200, SEARCH_RESPONSE)})
    tool = ClawifiSearch(client=sync_client(rec, api_key=None))
    tool.invoke({"query": "mcp"})
    assert "Authorization" not in rec.requests[0].headers


def test_search_all_engines_failed_raises_tool_exception() -> None:
    failed = {**SEARCH_RESPONSE, "success": False, "results": [], "errors": [{"code": "x"}]}
    rec = Recorder({"/search/free": (200, failed)})
    tool = ClawifiSearch(client=sync_client(rec))
    with pytest.raises(ToolException):
        tool.invoke({"query": "mcp"})


def test_search_api_error_becomes_tool_exception() -> None:
    error = {"error": {"code": "insufficient_credits", "message": "Not enough credits"}}
    rec = Recorder({"/search/free": (402, error)})
    tool = ClawifiSearch(client=sync_client(rec))
    with pytest.raises(ToolException):
        tool.invoke({"query": "mcp"})


def test_search_error_can_be_returned_to_the_agent() -> None:
    error = {"error": {"code": "unauthorized", "message": "bad key"}}
    rec = Recorder({"/search/free": (401, error)})
    tool = ClawifiSearch(client=sync_client(rec), handle_tool_error=True)
    out = tool.invoke({"query": "mcp"})
    assert isinstance(out, str) and out


async def test_search_async() -> None:
    rec = Recorder({"/search/free": (200, SEARCH_RESPONSE)})
    tool = ClawifiSearch(async_client=async_client(rec))
    out = await tool.ainvoke({"query": "mcp", "include_domains": ["example.com"]})
    assert rec.last_body["include_domains"] == ["example.com"]
    assert set(out) == {"results"}
    assert out["results"][1]["url"] == "https://example.com/b"


def test_fetch_returns_markdown_and_passes_query() -> None:
    rec = Recorder({"/fetch": (200, FETCH_RESPONSE)})
    tool = ClawifiFetch(client=sync_client(rec))

    out = tool.invoke({"url": "https://example.com", "query": "what is it for"})

    assert rec.last_body == {"url": "https://example.com", "query": "what is it for"}
    assert out["title"] == "Example Domain"
    assert out["content"].startswith("# Example Domain")
    assert out["content_status"] == "ok"
    assert "truncated" not in out


def test_fetch_truncates_long_content() -> None:
    long = {**FETCH_RESPONSE, "data": {**FETCH_RESPONSE["data"], "content_markdown": "x" * 50}}
    rec = Recorder({"/fetch": (200, long)})
    tool = ClawifiFetch(client=sync_client(rec), max_chars=10, render="auto")
    out = tool.invoke({"url": "https://example.com"})
    assert rec.last_body == {"url": "https://example.com", "render": "auto"}
    assert out["content"] == "x" * 10
    assert out["truncated"] is True


def test_fetch_unreadable_page_raises() -> None:
    blocked = {**FETCH_RESPONSE, "success": False, "content_status": "blocked"}
    rec = Recorder({"/fetch": (200, blocked)})
    with pytest.raises(ToolException, match="blocked"):
        ClawifiFetch(client=sync_client(rec)).invoke({"url": "https://example.com"})


async def test_fetch_async() -> None:
    rec = Recorder({"/fetch": (200, FETCH_RESPONSE)})
    tool = ClawifiFetch(async_client=async_client(rec))
    out = await tool.ainvoke({"url": "https://example.com"})
    assert out["url"] == "https://example.com"


def test_api_key_from_env_is_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAWIFI_API_KEY", "env-key")
    tool = ClawifiSearch()
    assert tool.clawifi_api_key is not None
    assert tool.clawifi_api_key.get_secret_value() == "env-key"
    assert "env-key" not in repr(tool)
    assert tool._get_client().config.api_key == "env-key"


def test_no_key_builds_keyless_client() -> None:
    tool = ClawifiFetch()
    assert tool.clawifi_api_key is None
    assert tool._get_client().config.api_key is None
