"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(
    metrics: list[BenchmarkMetrics],
    query: str | None = None,
    llm_provider: str | None = None,
    search_provider: str | None = None,
    trace_provider: str | None = None,
    trace_files: list[str] | None = None,
) -> str:
    """Render benchmark metrics to markdown."""

    lines = [
        "# Benchmark Report",
        "",
        "This report compares single-agent and multi-agent runs using automated lab metrics. "
        "Quality is a heuristic score and should be complemented by peer review.",
        "",
        "## Run Configuration",
        "",
        f"- Query: {_escape(query or 'Not recorded')}",
        f"- LLM provider: {_escape(llm_provider or 'Not recorded')}",
        f"- Search provider: {_escape(search_provider or 'Not recorded')}",
        f"- Trace provider: {_escape(trace_provider or 'local JSON trace')}",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Notes |",
        "|---|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        lines.append(
            f"| {_escape(item.run_name)} | {item.latency_seconds:.2f} | {cost} | "
            f"{quality} | {_escape(item.notes)} |"
        )
    lines.extend(
        [
            "",
            "## Human Review",
            "",
            "- Single-agent is simpler and faster, but it does not separate evidence collection, "
            "analysis, and writing.",
            "- Multi-agent has more moving parts, but the trace is easier to inspect because each "
            "stage writes to shared state.",
            "- Prefer multi-agent for research tasks that need source collection, critique, and "
            "a polished final answer.",
            "- Prefer single-agent for short questions where orchestration overhead is not useful.",
            "",
            "## Trace Evidence",
            "",
            "- LangSmith/Langfuse screenshot: TODO(add screenshot path or shared trace URL).",
            *[_trace_line(path) for path in trace_files or []],
            "",
            "## Failure Modes And Fixes",
            "",
            "| Failure mode | Impact | Fix |",
            "|---|---|---|",
            "| Missing or invalid OpenAI key | Falls back to offline response or API call fails | "
            "Set OPENAI_API_KEY and rerun benchmark |",
            "| Missing Tavily key | Uses local mock sources instead of live web results | "
            "Set TAVILY_API_KEY or document mock-search limitation |",
            "| Weak citations | Final answer may cite too few sources | "
            "Add critic checks and require source IDs in writer prompt |",
            "| Max iterations reached | Workflow stops before final answer | "
            "Tune MAX_ITERATIONS and inspect route_history |",
        ]
    )
    return "\n".join(lines) + "\n"


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _trace_line(path: str) -> str:
    return f"- Local trace artifact: `{_escape(path)}`"
