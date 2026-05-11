from __future__ import annotations

import json
import re


def _sanitize_json_string_values(payload: str) -> str:
    """
    LLMs often emit raw newlines / control chars inside JSON string values, which json.loads
    rejects. Walk the payload and escape those characters when inside double-quoted strings.
    """
    out: list[str] = []
    i = 0
    in_string = False
    escape = False
    n = len(payload)
    while i < n:
        c = payload[i]
        if escape:
            out.append(c)
            escape = False
            i += 1
            continue
        if c == "\\":
            out.append(c)
            escape = True
            i += 1
            continue
        if c == '"':
            in_string = not in_string
            out.append(c)
            i += 1
            continue
        if in_string:
            o = ord(c)
            if c == "\n":
                out.append("\\n")
            elif c == "\r":
                out.append("\\r")
            elif c == "\t":
                out.append("\\t")
            elif c in ("\u2028", "\u2029"):
                out.append(" ")
            elif o < 32:
                out.append(" ")
            else:
                out.append(c)
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _strip_json_fences(raw: str) -> str:
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*\n(.*)\n```\s*$", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def parse_llm_json(raw: str) -> dict:
    """Parse JSON from an LLM response, stripping fences and fixing common malformed strings."""
    text = _strip_json_fences(raw)

    def _loads(blob: str) -> dict:
        return json.loads(blob)

    for candidate in (text, _sanitize_json_string_values(text)):
        try:
            return _loads(candidate)
        except json.JSONDecodeError:
            continue

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        chunk = text[start : end + 1]
        for candidate in (chunk, _sanitize_json_string_values(chunk)):
            try:
                return _loads(candidate)
            except json.JSONDecodeError:
                continue

    raise json.JSONDecodeError("Could not parse LLM JSON after sanitization", raw, 0)
