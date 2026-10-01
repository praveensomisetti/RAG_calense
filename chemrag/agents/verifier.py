"""Verifier / guardrail (deterministic).

Checks the draft against the fact sheet: every number must come from the facts or the query rows, every
narrative bullet must cite existing facts, no medical advice, no leaked prompt text. On failure the
LLM narrative is dropped and the templated details (built only from facts) are used instead.
"""

from __future__ import annotations

import re
from typing import Any

from chemrag.agents.context import AgentContext, Timer, event
from chemrag.schemas import Draft, Verification, WarningItem
from chemrag.state import TurnState

NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(%?)")
FACT_REF = re.compile(r"\[(F\d+)\]")
ADVICE = re.compile(r"\b(is (perfectly |completely )?safe to use|you should (not )?(use|avoid|stop)|safe for (you|your)|"
                    r"recommend (using|avoiding)|will (not )?cause cancer|is harmless|no health risk)\b", re.IGNORECASE)
LEAK = re.compile(r"(untrusted user input|<user_question>|Strict rules:|Output only the JSON)", re.IGNORECASE)


def _numbers_in(obj: Any, out: set[str]) -> None:
    if obj is None or isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(str(int(obj)) if float(obj).is_integer() else str(obj))
        return
    if isinstance(obj, str):
        for m in NUM.finditer(obj):
            out.add(m.group(1).replace(",", ""))
        for part in re.findall(r"\d+", obj):
            out.add(str(int(part)))
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _numbers_in(v, out)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers_in(v, out)


def allowed_numbers(state: TurnState) -> set[str]:
    allowed: set[str] = set()
    for f in state.facts:
        _numbers_in(f.statement, allowed)
        _numbers_in(f.value, allowed)
    for r in state.results:
        _numbers_in(r.rows, allowed)
        _numbers_in(r.totals, allowed)
        _numbers_in(r.notes, allowed)
    _numbers_in(state.question, allowed)
    _numbers_in(state.assumptions, allowed)
    _numbers_in([w.message for w in state.warnings], allowed)
    for i in range(1, 11):  # small ordinals/list counters ("top 3", "2 companies")
        allowed.add(str(i))
    return allowed


def check_text(text: str, allowed: set[str], fact_ids: set[str], require_refs: bool) -> list[str]:
    problems = []
    for ref in FACT_REF.findall(text):
        if ref not in fact_ids:
            problems.append(f"cites unknown fact {ref}")
    stripped = FACT_REF.sub("", text)
    for m in NUM.finditer(stripped):
        whole = m.group(1).replace(",", "")
        if m.group(3) == "%":
            problems.append(f"computed percentage '{m.group(0)}' not in facts")
            continue
        if whole not in allowed and str(int(whole)) not in allowed:
            problems.append(f"number {m.group(0)} not supported by query results")
    if require_refs:
        for line in text.splitlines():
            if line.strip().startswith("-") and NUM.search(FACT_REF.sub("", line)) and not FACT_REF.search(line):
                problems.append(f"bullet with a number but no fact reference: {line.strip()[:60]}")
    if ADVICE.search(text):
        problems.append("contains health/safety advice")
    if LEAK.search(text):
        problems.append("echoes system prompt text")
    return problems


def verifier_node(state: TurnState, ctx: AgentContext) -> dict:
    t = Timer()
    draft = state.draft
    allowed = allowed_numbers(state)
    fact_ids = {f.id for f in state.facts}
    short_problems = check_text(draft.answer_short, allowed, fact_ids, require_refs=False)
    narrative = draft.answer_details
    template_details = None
    for ev in reversed(state.trace):
        if ev.agent == "synthesizer":
            template_details = ev.output.get("template_details")
            break
    detail_problems = check_text(narrative, allowed, fact_ids, require_refs=draft.details_mode == "llm")
    needs_evidence = [r for r in state.results if r.status == "ok" and r.tool != "dataset_coverage"]
    checks = {"short_numbers_grounded": not short_problems, "details_grounded": not detail_problems,
              "evidence_present": bool(state.evidence) or state.response_type != "answer" or not needs_evidence}
    warnings: list[WarningItem] = []
    update: dict = {}
    if detail_problems and draft.details_mode == "llm":
        warnings.append(WarningItem(code="verifier_fallback",
                                    message="The generated narrative failed grounding checks and was replaced by "
                                            "templated details: " + "; ".join(detail_problems[:3])))
        update["draft"] = Draft(answer_short=draft.answer_short, answer_details=template_details or "",
                                details_mode="template")
    if short_problems:  # should be impossible (templated) - surface loudly if it happens
        warnings.append(WarningItem(code="verifier_error", severity="error",
                                    message="Short answer failed grounding: " + "; ".join(short_problems)))
    if not checks["evidence_present"]:
        warnings.append(WarningItem(code="no_evidence", message="Answer has no supporting evidence rows."))
    passed = all(checks.values())
    update.update({
        "verification": Verification(passed=passed, checks=checks, problems=short_problems + detail_problems),
        "warnings": warnings,
        "trace": [event("verifier", "passed" if passed else "issues: " + "; ".join((short_problems + detail_problems)[:3]),
                        t, checks=checks)],
    })
    return update
