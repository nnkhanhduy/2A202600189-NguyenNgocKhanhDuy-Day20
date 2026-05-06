"""LangGraph workflow skeleton."""

from typing import Protocol, cast

from multi_agent_research_lab.agents import (
    AnalystAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import run_with_langsmith_trace


class RunnableAgent(Protocol):
    def run(self, state: ResearchState) -> ResearchState:
        """Run one workflow node."""
        ...


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def build(self) -> object:
        """Create the graph object.

        The lab can be upgraded to LangGraph later; this returns the node mapping used
        by `run`, keeping orchestration separate from agent internals.
        """

        return {
            "supervisor": SupervisorAgent(),
            "researcher": ResearcherAgent(),
            "analyst": AnalystAgent(),
            "writer": WriterAgent(),
        }

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state.

        Runs supervisor-controlled routing until the supervisor emits `done`.
        """

        return run_with_langsmith_trace(
            "multi_agent_workflow",
            "chain",
            self._run,
            state,
        )

    def _run(self, state: ResearchState) -> ResearchState:
        """Run the supervisor-controlled loop."""

        graph = self.build()
        agents = _typed_graph(graph)

        while True:
            state = _run_agent("supervisor", agents["supervisor"], state)
            route = state.route_history[-1]
            if route == "done":
                return state
            agent = agents.get(route)
            if agent is None:
                state.errors.append(f"Unknown route from supervisor: {route}")
                state.record_route("done")
                return state
            state = _run_agent(route, agent, state)


def _typed_graph(graph: object) -> dict[str, RunnableAgent]:
    if not isinstance(graph, dict):
        raise TypeError("Workflow graph must be a dictionary of runnable agents")
    return cast(dict[str, RunnableAgent], graph)


def _run_agent(name: str, agent: RunnableAgent, state: ResearchState) -> ResearchState:
    return run_with_langsmith_trace(f"agent:{name}", "chain", agent.run, state)
