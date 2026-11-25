"""LangChain tools for Clawifi web search and page reading."""

from __future__ import annotations

from typing import Any, Literal

from clawifi import AsyncClawifi, Clawifi, ClawifiError
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from langchain_core.utils import secret_from_env
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr

DEFAULT_MAX_CHARS = 20_000

_RESULT_FIELDS = ("rank", "title", "url", "snippet", "published_at")


class _ClawifiToolBase(BaseTool):
    """Shared client handling for the Clawifi tools.

    The API key is read from ``CLAWIFI_API_KEY`` when it is not passed. Without
    a key, the calls go out keyless and use Clawifi's free trial limits.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    clawifi_api_key: SecretStr | None = Field(
        default_factory=secret_from_env("CLAWIFI_API_KEY", default=None),
        alias="api_key",
    )
    """Clawifi API key. Optional: keyless calls use the free trial limits."""

    api_base: str | None = None
    """Override the API base URL. Defaults to the SDK default or ``CLAWIFI_API_BASE``."""

    client: Clawifi | None = Field(default=None, exclude=True)
    """Optional pre-built sync client."""

    async_client: AsyncClawifi | None = Field(default=None, exclude=True)
    """Optional pre-built async client."""

    _sync: Clawifi | None = PrivateAttr(default=None)
    _async: AsyncClawifi | None = PrivateAttr(default=None)

    def _key(self) -> str | None:
        return self.clawifi_api_key.get_secret_value() if self.clawifi_api_key else None

    def _get_client(self) -> Clawifi:
        if self.client is not None:
            return self.client
        if self._sync is None:
            self._sync = Clawifi(api_key=self._key(), api_base=self.api_base)
        return self._sync

    def _get_async_client(self) -> AsyncClawifi:
        if self.async_client is not None:
            return self.async_client
        if self._async is None:
            self._async = AsyncClawifi(api_key=self._key(), api_base=self.api_base)
        return self._async


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
    """Search the web with Clawifi free search (``POST /search/free``).

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
        "Search the web. Returns a list of results with title, url and snippet. "
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
    def _format(response: dict[str, Any]) -> list[dict[str, Any]]:
        if not response.get("success", True) and not response.get("results"):
            errors = response.get("errors") or []
            raise ToolException(f"Clawifi search failed: {errors}")
        return [
            {k: item.get(k) for k in _RESULT_FIELDS if item.get(k) is not None}
            for item in response.get("results") or []
        ]

    def _run(
        self,
        query: str,
        max_results: int | None = None,
        time_range: str | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> list[dict[str, Any]]:
        body = self._body(
            max_results,
            time_range=time_range,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )
        try:
            response = self._get_client().search_free(query, **body)
        except ClawifiError as exc:
            raise ToolException(str(exc)) from exc
        return self._format(response)

    async def _arun(
        self,
        query: str,
        max_results: int | None = None,
        time_range: str | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> list[dict[str, Any]]:
        body = self._body(
            max_results,
            time_range=time_range,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )
        try:
            response = await self._get_async_client().search_free(query, **body)
        except ClawifiError as exc:
            raise ToolException(str(exc)) from exc
        return self._format(response)


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
    """Read a web page or PDF as Markdown with Clawifi (``POST /fetch``).

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
    """Browser rendering mode; ``None`` keeps the API default."""

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
            response = await self._get_async_client().fetch(url, **self._body(query))
        except ClawifiError as exc:
            raise ToolException(str(exc)) from exc
        return self._format(response)
