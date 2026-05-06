"""Supervisor / router skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route.

        Routes through the required lab stages and stops on completion or guardrail limits.
        """

        route = self._next_route(state)
        state.record_route(route)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.SUPERVISOR,
                content=f"Next route: {route}",
                metadata={"iteration": state.iteration, "errors": len(state.errors)},
            )
        )
        state.add_trace_event(self.name, {"next_route": route, "iteration": state.iteration})
        return state

    def _next_route(self, state: ResearchState) -> str:
        if state.iteration >= self.settings.max_iterations:
            state.errors.append("Supervisor stopped workflow: max_iterations reached")
            return "done"
        if state.final_answer:
            return "done"
        if not state.research_notes:
            return "researcher"
        if not state.analysis_notes:
            return "analyst"
        return "writer"
