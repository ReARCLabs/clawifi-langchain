# langchain-clawifi

LangChain tools for [Clawifi](https://clawifi.dev): web search and page reading for agents.

- `ClawifiSearch` searches the web (`POST /search/free`) and returns title, URL and snippet for each result.
- `ClawifiFetch` reads a page or PDF (`POST /fetch`) and returns its main content as Markdown. Pass a `query` to get only the passages that answer it.

Both tools wrap the official [`clawifi`](https://pypi.org/project/clawifi/) Python SDK and support sync and async calls.

Source: <https://github.com/ReARCLabs/clawifi-langchain>

## Install

```bash
pip install -U langchain-clawifi
```

## Credentials

Set `CLAWIFI_API_KEY`, or pass `api_key=` to a tool. Keys come from [clawifi.dev](https://clawifi.dev/docs/authentication).

```bash
export CLAWIFI_API_KEY="..."
```

Without a key the tools still work, through the keyless trial below.

## Keyless trial

The Clawifi REST API needs a key. Without one, the tools call the trial tools of Clawifi's hosted MCP server instead ([Remote MCP](https://clawifi.dev/docs/remote-mcp)): `search_free` and `fetch_url` at `https://api.clawifi.dev/mcp`. The trial's limits, from the Clawifi docs:

- 12 calls a minute and 50 a day per address. On a shared host (serverless, CI) many apps can share one outbound address, so set `CLAWIFI_API_KEY` in production.
- Search returns at most 3 results, and domain and time filters are not applied.
- Fetched pages are cut at 20,000 characters; with `query`, up to 3 passages come back.
- Usually the day's limit comes back as HTTP 401 ("keyless limit reached") and the minute's limit as HTTP 429 with `Retry-After`.

A free account has 1,000 credits a month ([sign up](https://clawifi.dev/signup)). With a key, the tools use the REST API and your plan's limits.

Without a key, `ClawifiSearch` returns a dict instead of a list. The result items have the same fields, and `keyless_note` says which limits applied:

```python
{
    "results": [{"rank": 1, "title": "...", "url": "https://...", "snippet": "..."}],
    "keyless_note": "Keyless trial through Clawifi's hosted MCP server: at most 3 results, ...",
}
```

`ClawifiFetch` returns the same fields with or without a key. The `render` option is not sent without a key.

## Use the tools directly

```python
from langchain_clawifi import ClawifiFetch, ClawifiSearch

search = ClawifiSearch(max_results=5)
results = search.invoke({"query": "what is the Model Context Protocol"})
# [{"rank": 1, "title": "...", "url": "https://...", "snippet": "..."}, ...]

fetch = ClawifiFetch(max_chars=8000)
page = fetch.invoke({"url": results[0]["url"], "query": "who maintains it?"})
# {"url": "...", "title": "...", "content_status": "ok", "content": "..."}
```

## Use them in an agent

```python
from langchain.agents import create_agent
from langchain_clawifi import ClawifiFetch, ClawifiSearch

agent = create_agent(
    model="anthropic:claude-sonnet-5-5",
    tools=[ClawifiSearch(), ClawifiFetch()],
)
agent.invoke({"messages": [{"role": "user", "content": "Summarize the latest MCP spec changes."}]})
```

## Options

| Tool | Option | Default | Meaning |
| --- | --- | --- | --- |
| both | `api_key` | `CLAWIFI_API_KEY` | API key; optional |
| both | `api_base` | SDK default | Override the API URL |
| both | `client` / `async_client` | built on first call | Pass your own `clawifi.Clawifi` / `AsyncClawifi` |
| `ClawifiSearch` | `max_results` | `5` | Results per call when the call does not set one |
| `ClawifiSearch` | `country`, `language` | none | Passed to the search |
| `ClawifiFetch` | `max_chars` | `20000` | Cut the returned Markdown to this length |
| `ClawifiFetch` | `render` | API default | `"never"`, `"auto"` or `"always"` |

Call arguments the model can set: `query`, `max_results`, `time_range` (`day`/`week`/`month`/`year`), `include_domains`, `exclude_domains` for search; `url`, `query` for fetch.

API errors, including keyless trial limits, are raised as `ToolException`. Set `handle_tool_error=True` on a tool to return the error text to the model instead.

## Costs

Each search or fetch call uses Clawifi credits; see [Pricing and credits](https://clawifi.dev/docs/pricing-and-credits).

## Development

```bash
pip install -e . pytest pytest-asyncio "langchain-tests>=1.1.9,<2"
pytest
```

The unit tests mock every HTTP call and never reach the network.

## License

MIT
