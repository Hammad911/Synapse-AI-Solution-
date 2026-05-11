from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.context import transcript_block
from agents.llm import get_llm
from state import ResearchState


def synthesis_agent(state: ResearchState) -> dict:
    print("[Synthesis Agent] Structuring the final answer...")
    llm = get_llm(temperature=0.25)
    query = state.get("query") or ""
    findings = state.get("research_findings") or ""
    history = transcript_block(state, max_messages=40)

    system = SystemMessage(
        content=(
            "You are a senior business research analyst.\n"
            "Using the research findings and the conversation, answer the user's latest question.\n\n"
            "FORMAT (use these headings; keep tight, skimmable prose and bullets):\n"
            "## Direct answer\n"
            "2–4 sentences.\n"
            "## Key facts\n"
            "Bullets with dates, figures, or names when the sources provide them.\n"
            "## Context & developments\n"
            "Relevant background, recent news, or strategic moves if supported by findings.\n"
            "## Gaps & caveats\n"
            "What is uncertain, not in the sources, or requires primary documents — be explicit.\n"
            "## Sources\n"
            "Brief note that the answer is based on retrieved web snippets (no fake URLs).\n\n"
            "Rules: do not invent facts; if findings are thin, say so. Tie follow-ups to prior "
            "user messages when they reference 'they/the company' using the conversation."
        )
    )
    human = HumanMessage(
        content=(
            f"Full conversation:\n{history}\n\n"
            f"Latest question:\n{query}\n\n"
            f"Research findings:\n{findings[:22000]}"
        )
    )
    response = llm.invoke([system, human])
    text = response.content
    if isinstance(text, list):
        text = str(text)
    text = (text or "").strip()
    new_messages = list(state.get("messages") or [])
    new_messages.append(AIMessage(content=text))
    return {"final_response": text, "messages": new_messages}
