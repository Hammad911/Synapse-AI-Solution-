from __future__ import annotations

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from state import ResearchState


def format_conversation_for_llm(
    messages: list[BaseMessage] | None,
    *,
    max_messages: int | None = None,
    max_chars_per_msg: int = 2000,
) -> str:
    """Turn message history into a readable transcript for prompts."""
    if not messages:
        return "(no prior messages)"

    slice_msgs = messages if max_messages is None else messages[-max_messages:]
    lines: list[str] = []
    for m in slice_msgs:
        if isinstance(m, HumanMessage):
            role = "User"
        elif isinstance(m, AIMessage):
            role = "Assistant"
        else:
            role = type(m).__name__
        content = getattr(m, "content", "")
        if isinstance(content, list):
            content = " ".join(str(part) for part in content)
        content = (content or "").strip()
        if len(content) > max_chars_per_msg:
            content = content[: max_chars_per_msg - 3] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def transcript_block(
    state: ResearchState,
    *,
    tail: int | None = None,
    max_messages: int | None = None,
) -> str:
    msgs = list(state.get("messages") or [])
    if tail is not None:
        msgs = msgs[-tail:]
    return format_conversation_for_llm(msgs, max_messages=max_messages)
