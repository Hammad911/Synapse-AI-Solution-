from __future__ import annotations

import os
import textwrap
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from graph import build_graph


def main() -> None:
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
    print(
        textwrap.dedent(
            """\
            Multi-agent business research assistant (LangGraph)
            ------------------------------------------------------------
            Ask about a specific company and what you want to learn.
            Follow-ups work (e.g. competitors, CEO) once a company is in context.

            Keys: set GROQ_API_KEY and TAVILY_API_KEY in .env
            Web UI:  python server.py   →  http://127.0.0.1:8000
            Type Ctrl+C to exit.
            """
        ).strip()
    )

    if not os.environ.get("GROQ_API_KEY"):
        print("\n[Error] GROQ_API_KEY missing. Add it to .env (see https://console.groq.com).\n")
        return
    if not os.environ.get("TAVILY_API_KEY"):
        print("\n[Error] TAVILY_API_KEY missing. Add it to .env (see https://app.tavily.com).\n")
        return

    graph = build_graph()
    messages: list = []

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not user_text:
            continue

        thread_id = uuid.uuid4().hex
        config = {"configurable": {"thread_id": thread_id}}

        messages.append(HumanMessage(content=user_text))
        initial = {
            "messages": messages,
            "query": user_text,
            "clarity_status": "",
            "clarification_question": "",
            "research_findings": "",
            "confidence_score": 0,
            "validation_result": "",
            "validation_reason": "",
            "research_attempts": 0,
            "final_response": "",
        }
        print("\n[Orchestrator] Starting graph run...")
        try:
            result = graph.invoke(initial, config=config)
        except Exception as e:
            print(f"\n[Error] Graph run failed: {e}\n")
            messages.pop()
            continue

        while result.get("__interrupt__"):
            question = result["__interrupt__"][0].value
            print(f"\n[Clarification needed] {question}")
            try:
                answer = input("You (clarify): ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye.")
                return
            if not answer:
                print("[Orchestrator] Empty answer; canceling this turn.")
                messages.pop()
                break
            messages.append(HumanMessage(content=answer))
            print("\n[Orchestrator] Resuming after clarification...")
            try:
                result = graph.invoke(
                    Command(
                        resume=answer,
                        update={
                            "messages": messages,
                            "query": answer,
                            "validation_reason": "",
                        },
                    ),
                    config=config,
                )
            except Exception as e:
                print(f"\n[Error] Resume failed: {e}\n")
                messages.pop()
                break

        if result.get("__interrupt__"):
            continue

        messages = result.get("messages") or messages
        final = (result.get("final_response") or "").strip()
        if final:
            print(f"\nAssistant:\n{final}")
        else:
            print("\nAssistant: (no final response produced)")


if __name__ == "__main__":
    main()
