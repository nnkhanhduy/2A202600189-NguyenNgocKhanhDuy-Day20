"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import configure_langsmith
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_langsmith(settings)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline."""

    _init()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    response = LLMClient().complete(
        system_prompt=(
            "You are a single-agent research baseline. Answer directly, note uncertainty, "
            "and keep the response concise."
        ),
        user_prompt=f"Query: {query}\nAudience: {request.audience}",
    )
    state.final_answer = response.content
    state.add_trace_event("baseline", {"cost_usd": response.cost_usd})
    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow skeleton."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    result = workflow.run(state)
    console.print(result.model_dump_json(indent=2))


@app.command("benchmark")
def benchmark_command(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run baseline and multi-agent benchmark, then write a markdown report."""

    _init()
    settings = get_settings()
    artifact_store = LocalArtifactStore()
    llm_provider = "OpenAI API" if settings.openai_api_key else "offline fallback"
    search_provider = "Tavily API" if settings.tavily_api_key else "local mock search"
    trace_provider = (
        "LangSmith plus local JSON trace"
        if settings.langsmith_tracing and settings.langsmith_api_key
        else "local JSON trace"
    )

    def baseline_runner(item: str) -> ResearchState:
        request = ResearchQuery(query=item)
        state = ResearchState(request=request)
        response = LLMClient().complete(
            system_prompt="You are a single-agent research baseline.",
            user_prompt=f"Query: {item}\nAudience: {request.audience}",
        )
        state.final_answer = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={
                    "mode": "single-agent",
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event("baseline", {"cost_usd": response.cost_usd})
        return state

    def multi_agent_runner(item: str) -> ResearchState:
        return MultiAgentWorkflow().run(ResearchState(request=ResearchQuery(query=item)))

    baseline_state, baseline_metrics = run_benchmark("single-agent", query, baseline_runner)
    multi_agent_state, multi_agent_metrics = run_benchmark("multi-agent", query, multi_agent_runner)
    baseline_trace = artifact_store.write_text(
        "baseline_trace.json",
        baseline_state.model_dump_json(indent=2),
    )
    multi_agent_trace = artifact_store.write_text(
        "multi_agent_trace.json",
        multi_agent_state.model_dump_json(indent=2),
    )
    report = render_markdown_report(
        [baseline_metrics, multi_agent_metrics],
        query=query,
        llm_provider=llm_provider,
        search_provider=search_provider,
        trace_provider=trace_provider,
        trace_files=[str(baseline_trace), str(multi_agent_trace)],
    )
    path = artifact_store.write_text("benchmark_report.md", report)
    console.print(Panel.fit(str(path), title="Benchmark Report Written"))


if __name__ == "__main__":
    app()
