from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from agents.clarity import clarity_agent
from agents.research import research_agent
from agents.synthesis import synthesis_agent
from agents.validator import validator_agent
from state import ResearchState


def _route_after_research(state: ResearchState) -> str:
    score = int(state.get("confidence_score") or 0)
    if score >= 6:
        print("[Router] High confidence — proceeding to synthesis.")
        return "synthesis_agent"
    print("[Router] Lower model confidence — sending to validator for a second opinion.")
    return "validator_agent"


def _route_after_validator(state: ResearchState) -> str:
    verdict = (state.get("validation_result") or "").strip().lower()
    if verdict == "sufficient":
        print("[Router] Validation sufficient — synthesizing.")
        return "synthesis_agent"
    attempts = int(state.get("research_attempts") or 0)
    reason = (state.get("validation_reason") or "").strip()
    excerpt = (reason[:200] + "…") if len(reason) > 200 else reason
    if verdict == "insufficient" and attempts < 3:
        if excerpt:
            print(f"[Router] Research insufficient — retrying. Note: {excerpt}")
        else:
            print("[Router] Insufficient data — additional research pass.")
        return "research_agent"
    print("[Router] Insufficient after max attempts — synthesizing with available data.")
    return "synthesis_agent"


def build_graph():
    workflow = StateGraph(ResearchState)
    workflow.add_node("clarity_agent", clarity_agent)
    workflow.add_node("research_agent", research_agent)
    workflow.add_node("validator_agent", validator_agent)
    workflow.add_node("synthesis_agent", synthesis_agent)

    workflow.add_edge(START, "clarity_agent")
    workflow.add_edge("clarity_agent", "research_agent")

    workflow.add_conditional_edges(
        "research_agent",
        _route_after_research,
        {
            "validator_agent": "validator_agent",
            "synthesis_agent": "synthesis_agent",
        },
    )

    workflow.add_conditional_edges(
        "validator_agent",
        _route_after_validator,
        {
            "research_agent": "research_agent",
            "synthesis_agent": "synthesis_agent",
        },
    )

    workflow.add_edge("synthesis_agent", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
