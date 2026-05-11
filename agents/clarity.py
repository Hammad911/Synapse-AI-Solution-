from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from agents.context import transcript_block
from agents.llm import get_llm
from json_utils import parse_llm_json
from state import ResearchState


def clarity_agent(state: ResearchState) -> dict:
    print("[Clarity Agent] Evaluating query precision and company anchor...")
    llm = get_llm(temperature=0)
    context = transcript_block(state, tail=24)
    query = state.get("query") or ""

    system = SystemMessage(
        content=(
            "You gate a business research assistant. Decide if the user has given enough to run "
            "a targeted company research pass.\n\n"
            "MARK AS clear ONLY when at least one of these holds:\n"
            "- The latest query names a specific company, brand, or clearly identifiable "
            "listed entity (ticker + exchange counts), AND the user asks something answerable "
            "with public info (products, CEO, financials, news, competitors of THAT company, etc.).\n"
            "- OR earlier messages already name the company AND the latest line is a reasonable "
            "follow-up (e.g. 'What about their CEO?', 'compared to who?', 'more on revenue') "
            "without switching to a different unnamed subject.\n\n"
            "MARK AS needs_clarification when:\n"
            "- The user uses vague references ('that company', 'them', 'their stock') with NO "
            "company established in the conversation.\n"
            "- The ask is so broad there is no company target ('compare tech giants' with no names).\n"
            "- Critical disambiguation is missing (e.g. 'Apple' could mean fruit vs Apple Inc. and "
            "the user gave no context — ask which they mean).\n\n"
            "Respond ONLY with raw JSON, no markdown: "
            '{"clarity_status":"clear"|"needs_clarification",'
            '"clarification_question":"" OR one short question}.\n'
            "Use an empty string for clarification_question when status is clear."
        )
    )
    human = HumanMessage(
        content=(
            f"Conversation (most recent last):\n{context}\n\n"
            f"Latest user message to classify:\n{query}"
        )
    )
    raw = llm.invoke([system, human]).content
    if isinstance(raw, list):
        raw = str(raw)
    parsed = parse_llm_json(raw)
    status = str(parsed.get("clarity_status", "needs_clarification")).strip().lower()
    if status not in ("clear", "needs_clarification"):
        status = "needs_clarification"
    question = (parsed.get("clarification_question") or "").strip()

    if status == "needs_clarification":
        clarification = question or (
            "Which company (or stock) should I research, and what aspect matters most "
            "(e.g. financials, leadership, recent news)?"
        )
        interrupt(clarification)

    return {
        "clarity_status": "clear",
        "clarification_question": "",
    }
