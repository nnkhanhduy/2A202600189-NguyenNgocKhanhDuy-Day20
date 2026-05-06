# Benchmark Report

This report compares single-agent and multi-agent runs using automated lab metrics. Quality is a heuristic score and should be complemented by peer review.

## Run Configuration

- Query: Explain multi-agent systems
- LLM provider: OpenAI API
- Search provider: Tavily API
- Trace provider: LangSmith plus local JSON trace

| Run | Latency (s) | Cost (USD) | Quality | Notes |
|---|---:|---:|---:|---|
| single-agent | 12.60 | 0.0004 | 3.0 | sources=0; errors=0; citation_coverage=0% |
| multi-agent | 25.23 | 0.0014 | 10.0 | sources=5; errors=0; citation_coverage=100% |

## Human Review

- Single-agent is simpler and faster, but it does not separate evidence collection, analysis, and writing.
- Multi-agent has more moving parts, but the trace is easier to inspect because each stage writes to shared state.
- Prefer multi-agent for research tasks that need source collection, critique, and a polished final answer.
- Prefer single-agent for short questions where orchestration overhead is not useful.

## Trace Evidence

- LangSmith screenshot: `reports/screenshot.png`.
- Local trace artifact: `reports\baseline_trace.json`
- Local trace artifact: `reports\multi_agent_trace.json`

## Failure Modes And Fixes

| Failure mode | Impact | Fix |
|---|---|---|
| Missing or invalid OpenAI key | Falls back to offline response or API call fails | Set OPENAI_API_KEY and rerun benchmark |
| Missing Tavily key | Uses local mock sources instead of live web results | Set TAVILY_API_KEY or document mock-search limitation |
| Weak citations | Final answer may cite too few sources | Add critic checks and require source IDs in writer prompt |
| Max iterations reached | Workflow stops before final answer | Tune MAX_ITERATIONS and inspect route_history |
