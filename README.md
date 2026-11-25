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

Without a key the tools still work as a keyless trial with lower limits (see [Rate limits](https://clawifi.dev/docs/rate-limits)).

```bash
export CLAWIFI_API_KEY="..."
```

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

API errors are raised as `ToolException`. Set `handle_tool_error=True` on a tool to return the error text to the model instead.

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
