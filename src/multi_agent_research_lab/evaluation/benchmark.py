"""Benchmark skeleton for single-agent vs multi-agent."""

from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency and summarize basic quality/cost signals."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    costs: list[float] = []
    for result in state.agent_results:
        cost = result.metadata.get("cost_usd")
        if isinstance(cost, int | float):
            costs.append(float(cost))
    estimated_cost = sum(costs) if costs else None
    quality_score = _heuristic_quality_score(state)
    notes = (
        f"sources={len(state.sources)}; "
        f"errors={len(state.errors)}; "
        f"citation_coverage={_citation_coverage(state):.0%}"
    )
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=estimated_cost,
        quality_score=quality_score,
        notes=notes,
    )
    return state, metrics


def _citation_coverage(state: ResearchState) -> float:
    if not state.sources or not state.final_answer:
        return 0.0
    cited = sum(
        1
        for index in range(1, len(state.sources) + 1)
        if f"[{index}]" in state.final_answer
    )
    return cited / len(state.sources)


def _heuristic_quality_score(state: ResearchState) -> float:
    score = 0.0
    if state.final_answer and len(state.final_answer.split()) >= 40:
        score += 3.0
    if state.research_notes:
        score += 2.0
    if state.analysis_notes:
        score += 2.0
    if state.sources:
        score += 1.5
    score += min(1.5, _citation_coverage(state) * 1.5)
    if state.errors:
        score -= min(3.0, len(state.errors))
    return max(0.0, min(10.0, score))
