"""Tracing hooks.

This file intentionally avoids binding to one provider. Students can plug in LangSmith,
Langfuse, OpenTelemetry, or simple JSON traces.
"""

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Literal, TypeVar

from multi_agent_research_lab.core.config import Settings

T = TypeVar("T")
RunType = Literal["tool", "chain", "llm", "retriever", "embedding", "prompt", "parser"]


def configure_langsmith(settings: Settings) -> None:
    """Export LangSmith settings so the SDK can pick them up at runtime."""

    if not settings.langsmith_tracing or not settings.langsmith_api_key:
        return
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal span context used by the skeleton.

    TODO(student): Replace or augment with LangSmith/Langfuse provider spans.
    """

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started


def run_with_langsmith_trace(
    name: str,
    run_type: RunType,
    function: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Run a function inside a LangSmith trace when the SDK is installed and enabled."""

    if os.environ.get("LANGSMITH_TRACING", "").lower() != "true":
        return function(*args, **kwargs)
    try:
        from langsmith import traceable
    except ImportError:
        return function(*args, **kwargs)

    traced = traceable(run_type=run_type, name=name)(function)
    return traced(*args, **kwargs)
