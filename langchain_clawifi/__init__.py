"""LangChain integration for Clawifi: web search and page reading for agents."""

from importlib import metadata

from langchain_clawifi._keyless import ClawifiKeylessError
from langchain_clawifi.tools import (
    ClawifiFetch,
    ClawifiFetchInput,
    ClawifiSearch,
    ClawifiSearchInput,
)

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = ""

__all__ = [
    "ClawifiFetch",
    "ClawifiKeylessError",
    "ClawifiFetchInput",
    "ClawifiSearch",
    "ClawifiSearchInput",
    "__version__",
]
