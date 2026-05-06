"""Search client abstraction for ResearcherAgent."""

import json
from typing import Any, cast
from urllib.error import URLError
from urllib.request import Request, urlopen

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Provider-agnostic search client skeleton."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query.

        Uses Tavily when configured. Otherwise returns local seed documents so the
        lab can run without paid search.
        """

        if self.settings.tavily_api_key:
            return self._tavily_search(query, max_results)
        return self._mock_search(query, max_results)

    def _mock_search(self, query: str, max_results: int) -> list[SourceDocument]:
        seeds = [
            SourceDocument(
                title="Anthropic: Building effective agents",
                url="https://www.anthropic.com/engineering/building-effective-agents",
                snippet=(
                    "Practical agent systems often work best when workflows are explicit, "
                    "tools are scoped, and agent autonomy is added only where it helps."
                ),
                metadata={"provider": "local", "query": query},
            ),
            SourceDocument(
                title="OpenAI Agents SDK orchestration and handoffs",
                url="https://developers.openai.com/",
                snippet=(
                    "Agent orchestration benefits from clear handoff contracts, structured "
                    "outputs, guardrails, and observability."
                ),
                metadata={"provider": "local", "query": query},
            ),
            SourceDocument(
                title="LangGraph workflow concepts",
                url="https://langchain-ai.github.io/langgraph/concepts/",
                snippet=(
                    "Graph-based agent workflows model nodes, edges, conditional routing, "
                    "state updates, and stop conditions."
                ),
                metadata={"provider": "local", "query": query},
            ),
        ]
        return seeds[:max_results]

    def _tavily_search(self, query: str, max_results: int) -> list[SourceDocument]:
        try:
            body = json.dumps(
                {
                    "api_key": self.settings.tavily_api_key,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": False,
                }
            ).encode("utf-8")
            request = Request(
                "https://api.tavily.com/search",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                payload = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, URLError, json.JSONDecodeError):
            return self._mock_search(query, max_results)
        return [
            SourceDocument(
                title=str(item.get("title") or "Untitled source"),
                url=str(item["url"]) if item.get("url") else None,
                snippet=str(item.get("content") or item.get("snippet") or ""),
                metadata={"provider": "tavily"},
            )
            for item in _result_items(payload, max_results)
        ]


def _result_items(payload: dict[str, Any], max_results: int) -> list[dict[str, Any]]:
    results = payload.get("results", [])
    if not isinstance(results, list):
        return []
    return [item for item in results[:max_results] if isinstance(item, dict)]
