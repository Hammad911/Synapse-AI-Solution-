from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agents.context import transcript_block
from agents.llm import get_llm
from json_utils import parse_llm_json
from state import ResearchState


def validator_agent(state: ResearchState) -> dict:
    print("[Validator Agent] Assessing research quality vs the user's question...")
    llm = get_llm(temperature=0)
    query = state.get("query") or ""
    findings = state.get("research_findings") or ""
    history = transcript_block(state, tail=6)

    system = SystemMessage(
        content=(
            "You QA web research before it is shown to a user.\n"
            "Decide if the findings are sufficient to answer the user's business question "
            "with useful specificity (not guesses): enough concrete facts, relevant scope, "
            "and no total mismatch.\n"
            "sufficient: a competent analyst could answer main points from the text.\n"
            "insufficient: missing key dimensions, contradictory/empty results, wrong company, "
            "or question needs numbers/news that are absent.\n\n"
            "Return ONLY raw JSON: "
            '{"validation_result":"sufficient"|"insufficient",'
            '"reason":"<one or two sentences; if insufficient, say what to look for next>"}\n'
            "No markdown."
        )
    )
    human = HumanMessage(
        content=(
            f"Conversation tail:\n{history}\n\n"
            f"User question:\n{query}\n\n"
            f"Research bundle:\n{findings[:18000]}"
        )
    )
    raw = llm.invoke([system, human]).content
    if isinstance(raw, list):
        raw = str(raw)
    parsed = parse_llm_json(raw)
    result = str(parsed.get("validation_result", "insufficient")).strip().lower()
    if result not in ("sufficient", "insufficient"):
        result = "insufficient"
    reason = str(parsed.get("reason") or "").strip()
    return {"validation_result": result, "validation_reason": reason}
