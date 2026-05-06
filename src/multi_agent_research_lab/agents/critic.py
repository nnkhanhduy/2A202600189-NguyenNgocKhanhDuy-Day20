"""Optional critic agent skeleton for bonus work."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings.

        Adds lightweight citation coverage feedback without another LLM call.
        """

        if not state.final_answer:
            state.errors.append("CriticAgent skipped: missing final_answer")
            return state

        source_refs = sum(
            1
            for index in range(1, len(state.sources) + 1)
            if f"[{index}]" in state.final_answer
        )
        coverage = source_refs / len(state.sources) if state.sources else 0.0
        content = (
            f"Citation coverage: {coverage:.0%} "
            f"({source_refs}/{len(state.sources)} sources referenced)."
        )
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=content,
                metadata={"citation_coverage": coverage},
            )
        )
        state.add_trace_event(self.name, {"citation_coverage": coverage})
        return state
