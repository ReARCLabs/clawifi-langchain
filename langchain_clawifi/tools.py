"""LangChain tools for Clawifi web search and page reading."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from clawifi import AsyncClawifi, Clawifi, ClawifiError
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from langchain_core.utils import secret_from_env
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr

from langchain_clawifi._keyless import (
    KEYLESS_FETCH_TOOL,
    KEYLESS_MAX_RESULTS,
    KEYLESS_SEARCH_TOOL,
    AsyncKeylessClient,
    KeylessClient,
    fetch_arguments,
    keyless_note,
    require_copyable,
    search_arguments,
)

DEFAULT_MAX_CHARS = 20_000

_RESULT_FIELDS = ("rank", "title", "url", "snippet", "published_at")


class _ClawifiToolBase(BaseTool):
    """Shared client handling for the Clawifi tools.

    With an API key (``api_key`` or ``CLAWIFI_API_KEY``) the tools call the Clawifi
    REST API through the official SDK. Without a key they call the keyless trial
    tools of Clawifi's hosted MCP server, which has lower limits.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    clawifi_api_key: SecretStr | None = Field(
        default_factory=secret_from_env("CLAWIFI_API_KEY", default=None),
        alias="api_key",
    )
    """Clawifi API key. Optional: without it the keyless trial is used."""

    api_base: str | None = None
    """Override the API base URL. Defaults to the SDK default or ``CLAWIFI_API_BASE``."""

    client: Clawifi | None = Field(default=None, exclude=True)
    """Optional pre-built sync client (always used when set)."""

    async_client: AsyncClawifi | None = Field(default=None, exclude=True)
    """Optional pre-built async client (always used when set)."""

    keyless_client: KeylessClient | None = Field(default=None, exclude=True)
    """Optional pre-built keyless MCP client, used only when there is no key."""

    async_keyless_client: AsyncKeylessClient | None = Field(default=None, exclude=True)
    """Optional pre-built async keyless MCP client, used only when there is no key."""

    _sync: Clawifi | None = PrivateAttr(default=None)

    def _key(self) -> str | None:
        return self.clawifi_api_key.get_secret_value() if self.clawifi_api_key else None

    def _keyless(self) -> bool:
        """Use the keyless trial only with no key and no pre-built REST client at all."""
        return self.client is None and self.async_client is None and self._key() is None

    def _get_client(self) -> Clawifi:
        if self.client is not None:
            return self.client
        if self._sync is None:
            if self.async_client is not None:
                # Only an async client was given: mirror it if nothing would be lost.
                require_copyable(self.async_client, "async_client", "client")
                self._sync = Clawifi(config=self.async_client.config)
            else:
                self._sync = Clawifi(api_key=self._key(), api_base=self.api_base)
        return self._sync

    @asynccontextmanager
    async def _async_rest(self) -> AsyncIterator[AsyncClawifi]:
        """Yield an async REST client bound to the running event loop.

        A client passed as ``async_client`` is used as is. Otherwise a new one is
        opened for this call and closed after it, so nothing stays bound to an
        event loop that ``asyncio.run`` has already closed.
        """
        if self.async_client is not None:
            yield self.async_client
            return
        if self.client is not None:
            # Only a sync client was given: mirror it if nothing would be lost.
            require_copyable(self.client, "client", "async_client")
            client = AsyncClawifi(config=self.client.config)
        else:
            client = AsyncClawifi(api_key=self._key(), api_base=self.api_base)
        try:
            yield client
        finally:
            await client.aclose()

    def close(self) -> None:
        """End the keyless MCP session (if any) and close the clients this tool built."""
        if self.keyless_client is not None:
            self.keyless_client.close()
        if self._sync is not None:
            self._sync.close()
            self._sync = None

    async def aclose(self) -> None:
        """End the async keyless MCP session (if any). Call it from the loop that used it."""
        if self.async_keyless_client is not None:
            await self.async_keyless_client.aclose()

    def _get_keyless_client(self) -> KeylessClient:
        if self.keyless_client is None:
            self.keyless_client = KeylessClient(api_base=self.api_base)
        return self.keyless_client

    def _get_async_keyless_client(self) -> AsyncKeylessClient:
        if self.async_keyless_client is None:
            self.async_keyless_client = AsyncKeylessClient(api_base=self.api_base)
        return self.async_keyless_client


class ClawifiSearchInput(BaseModel):
    """Input for :class:`ClawifiSearch`."""

    query: str = Field(description="The search query.")
    max_results: int | None = Field(
        default=None, ge=1, le=50, description="How many results to return."
    )
    time_range: Literal["day", "week", "month", "year"] | None = Field(
        default=None, description="Only return results from this recent period."
    )
    include_domains: list[str] | None = Field(
        default=None, description="Only return results from these domains."
    )
    exclude_domains: list[str] | None = Field(
        default=None, description="Leave out results from these domains."
    )


class ClawifiSearch(_ClawifiToolBase):
    """Search the web with Clawifi free search.

    Always returns ``{"results": [...]}``. With a key this calls
    ``POST /search/free``. Without a key it calls the hosted MCP trial tool
    ``search_free`` and adds ``keyless_note``: at most 3 results, and the domain
    and time filters are not applied.

    Setup:
        .. code-block:: bash

            pip install -U langchain-clawifi
            export CLAWIFI_API_KEY="..."  # optional

    Instantiate:
        .. code-block:: python

            from langchain_clawifi import ClawifiSearch

            tool = ClawifiSearch(max_results=5)

    Invoke:
        .. code-block:: python

            tool.invoke({"query": "what is the Model Context Protocol"})
    """

    name: str = "clawifi_search"
    description: str = (
        "Search the web. Returns results with title, url and snippet. "
        "Use clawifi_fetch to read a result's page."
    )
    args_schema: type[BaseModel] = ClawifiSearchInput

    max_results: int = 5
    """Default number of results when the call does not set one."""

    country: str | None = None
    language: str | None = None

    def _body(self, max_results: int | None, **filters: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"max_results": max_results or self.max_results}
        if self.country:
            body["country"] = self.country
        if self.language:
            body["language"] = self.language
        body.update({k: v for k, v in filters.items() if v is not None})
        return body

    @staticmethod
    def _items(response: dict[str, Any]) -> list[dict[str, Any]]:
        if not response.get("success", True) and not response.get("results"):
            errors = response.get("errors") or []
            raise ToolException(f"Clawifi search failed: {errors}")
        return [
            {k: item.get(k) for k in _RESULT_FIELDS if item.get(k) is not None}
            for item in response.get("results") or []
            if isinstance(item, dict)
        ]

    def _requested(self, max_results: int | None) -> int | None:
        """The result count the user asked for: per call, or set on the tool; else None."""
        if max_results is not None:
            return max_results
        if "max_results" in self.model_fields_set:
            return self.max_results
        return None

    def _keyless_result(
        self, response: dict[str, Any], body: dict[str, Any], requested: int | None
    ) -> dict[str, Any]:
        limit = min(requested or KEYLESS_MAX_RESULTS, KEYLESS_MAX_RESULTS)
        dropped = [
            name
            for name in ("time_range", "include_domains", "exclude_domains", "country", "language")
            if body.get(name)
        ]
        return {
            "results": self._items(response)[:limit],
            "keyless_note": keyless_note(dropped, requested),
        }

    def _run(
        self,
        query: str,
        max_results: int | None = None,
        time_range: str | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict[str, Any]:
        body = self._body(
            max_results,
            time_range=time_range,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )
        try:
            if self._keyless():
                response = self._get_keyless_client().call_tool(
                    KEYLESS_SEARCH_TOOL, search_arguments(query)
                )
                return self._keyless_result(response, body, self._requested(max_results))
            response = self._get_client().search_free(query, **body)
        except ClawifiError as exc:
            raise ToolException(str(exc)) from exc
        return {"results": self._items(response)}

    async def _arun(
        self,
        query: str,
        max_results: int | None = None,
        time_range: str | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> dict[str, Any]:
        body = self._body(
            max_results,
            time_range=time_range,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )
        try:
            if self._keyless():
                response = await self._get_async_keyless_client().call_tool(
                    KEYLESS_SEARCH_TOOL, search_arguments(query)
                )
                return self._keyless_result(response, body, self._requested(max_results))
            async with self._async_rest() as client:
                response = await client.search_free(query, **body)
        except ClawifiError as exc:
            raise ToolException(str(exc)) from exc
        return {"results": self._items(response)}


class ClawifiFetchInput(BaseModel):
    """Input for :class:`ClawifiFetch`."""

    url: str = Field(description="The URL of the page or PDF to read.")
    query: str | None = Field(
        default=None,
        description=(
            "Optional question. When set, only the passages of the page that "
            "answer it are returned."
        ),
    )


class ClawifiFetch(_ClawifiToolBase):
    """Read a web page or PDF as Markdown with Clawifi.

    With a key this calls ``POST /fetch``. Without a key it calls the hosted MCP
    trial tool ``fetch_url``; the result has the same fields.

    Instantiate:
        .. code-block:: python

            from langchain_clawifi import ClawifiFetch

            tool = ClawifiFetch()

    Invoke:
        .. code-block:: python

            tool.invoke({"url": "https://example.com", "query": "what is this page for?"})
    """

    name: str = "clawifi_fetch"
    description: str = (
        "Read a web page or PDF and return its main content as Markdown. "
        "Pass a query to get only the passages that answer it."
    )
    args_schema: type[BaseModel] = ClawifiFetchInput

    max_chars: int = DEFAULT_MAX_CHARS
    """Cut the returned Markdown to this many characters."""

    render: Literal["never", "auto", "always"] | None = None
    """Browser rendering mode; ``None`` keeps the API default. Not sent without a key."""

    def _body(self, query: str | None) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if query:
            body["query"] = query
        if self.render:
            body["render"] = self.render
        return body

    def _format(self, response: dict[str, Any]) -> dict[str, Any]:
        data = response.get("data") or {}
        if not response.get("success", True):
            errors = response.get("errors") or []
            status = response.get("content_status")
            raise ToolException(f"Clawifi could not read the page ({status}): {errors}")
        content = data.get("content_markdown") or ""
        truncated = len(content) > self.max_chars
        result = {
            "url": data.get("url") or data.get("source_url"),
            "title": data.get("title"),
            "content_status": response.get("content_status"),
            "content": content[: self.max_chars],
        }
        if truncated:
            result["truncated"] = True
        return result

    def _run(
        self,
        url: str,
        query: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict[str, Any]:
        try:
            if self._keyless():
                response = self._get_keyless_client().call_tool(
                    KEYLESS_FETCH_TOOL, fetch_arguments(url, query)
                )
            else:
                response = self._get_client().fetch(url, **self._body(query))
        except ClawifiError as exc:
            raise ToolException(str(exc)) from exc
        return self._format(response)

    async def _arun(
        self,
        url: str,
        query: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> dict[str, Any]:
        try:
            if self._keyless():
                response = await self._get_async_keyless_client().call_tool(
                    KEYLESS_FETCH_TOOL, fetch_arguments(url, query)
                )
            else:
                async with self._async_rest() as client:
                    response = await client.fetch(url, **self._body(query))
        except ClawifiError as exc:
            raise ToolException(str(exc)) from exc
        return self._format(response)
