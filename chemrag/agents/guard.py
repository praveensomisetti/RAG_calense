"""Input guard: length cap and prompt-injection heuristics (deterministic)."""

from __future__ import annotations

import re

from chemrag.agents.context import AgentContext, Timer, event
from chemrag.schemas import WarningItem
from chemrag.state import TurnState

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"disregard (all |the )?(previous|prior|above) ",
    r"(reveal|print|show|repeat|output) (me )?(your|the) (system )?(prompt|instructions)",
    r"\bsystem prompt\b",
    r"\byou are now\b",
    r"\bact as\b",
    r"\bjailbreak\b",
    r"\bdeveloper mode\b",
    r";\s*--",
    r"\b(drop|delete|truncate|alter|insert|update)\s+(table|from|into|database)\b",
    r"<\s*/?\s*(system|assistant|user_question)\s*>",
]
_INJ = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)
_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")


def guard_node(state: TurnState, ctx: AgentContext) -> dict:
    t = Timer()
    q = state.question.strip()
    warnings: list[WarningItem] = []
    limit = ctx.settings.max_question_chars
    if len(q) > limit:
        q = q[:limit]
        warnings.append(WarningItem(code="input_truncated", message=f"Question truncated to {limit} characters."))
    removed: list[str] = []
    if _INJ.search(q):
        kept = []
        for sent in _SENTENCES.split(q):
            if _INJ.search(sent):
                removed.append(sent.strip())
            elif sent.strip():
                kept.append(sent.strip())
        q = " ".join(kept)
        warnings.append(WarningItem(
            code="possible_injection",
            message="Ignored instruction-like text in the question: " + " | ".join(r[:80] for r in removed),
        ))
    update: dict = {"clean_question": q, "warnings": warnings}
    if not q:
        update["refusal_reason"] = "The question contained no answerable content after removing instructions."
        update["response_type"] = "refusal"
    update["trace"] = [event("guard", "input checked" + (f"; removed {len(removed)} instruction-like sentence(s)"
                                                         if removed else ""), t, removed=removed)]
    return update
