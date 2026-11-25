from __future__ import annotations

from langchain_tests.unit_tests import ToolsUnitTests

from langchain_clawifi import ClawifiFetch, ClawifiSearch


class TestClawifiSearchStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[ClawifiSearch]:
        return ClawifiSearch

    @property
    def tool_constructor_params(self) -> dict:
        return {"max_results": 3}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"query": "what is the Model Context Protocol"}

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return ({"CLAWIFI_API_KEY": "env-key"}, {}, {"clawifi_api_key": "env-key"})


class TestClawifiFetchStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[ClawifiFetch]:
        return ClawifiFetch

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"url": "https://example.com"}

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return ({"CLAWIFI_API_KEY": "env-key"}, {}, {"clawifi_api_key": "env-key"})
