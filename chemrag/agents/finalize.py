"""Assemble the output contract (Response) and compute the documented confidence score."""

from __future__ import annotations

from chemrag.agents.context import AgentContext, Timer, event
from chemrag.agents.synthesizer import SAFETY_NOTE
from chemrag.schemas import Confidence, Response, WarningItem
from chemrag.state import TurnState

PENALTY = {
    "ambiguous_entity": 0.7, "entity_not_found": 0.6, "verifier_fallback": 0.85, "verifier_error": 0.5,
    "empty_result": 0.9, "possible_injection": 0.85, "data_conflict": 0.97, "date_range_clipped": 0.95,
    "brand_multiple_companies": 0.95, "no_evidence": 0.7, "input_truncated": 0.9, "question_truncated": 0.85,
}


def confidence(state: TurnState) -> Confidence:
    if state.response_type == "clarification":
        return Confidence(score=0.0, level="low", rationale="needs clarification before answering")
    parts = []
    intent_conf = min((s.intent_confidence for s in state.plan.subtasks), default=1.0) if state.plan else 1.0
    res_scores = [c.score for r in state.resolutions if r.status == "resolved" for c in r.chosen]
    res_conf = min(res_scores, default=1.0)
    score = min(intent_conf, res_conf)
    parts.append(f"intent {intent_conf:.2f}, entity match {res_conf:.2f}")
    seen = set()
    for w in state.warnings:
        if w.code in PENALTY and w.code not in seen:
            seen.add(w.code)
            score *= PENALTY[w.code]
            parts.append(f"{w.code} x{PENALTY[w.code]}")
    level = "high" if score >= 0.85 else "medium" if score >= 0.6 else "low"
    return Confidence(score=round(score, 3), level=level, rationale="; ".join(parts))


def dedupe_warnings(ws: list[WarningItem]) -> list[WarningItem]:
    seen, out = set(), []
    for w in ws:
        key = (w.code, w.message)
        if key not in seen:
            seen.add(key)
            out.append(w)
    return out


def finalize_node(state: TurnState, ctx: AgentContext) -> dict:
    t = Timer()
    rtype = state.response_type
    draft = state.draft
    if rtype == "clarification" and state.clarification:
        short = state.clarification.question
        details = "\n".join(f"{i}. {o.name} ({o.type}; {o.detail})" for i, o in enumerate(state.clarification.options, 1))
    elif draft is None:  # refusal before synthesis (guard / out of scope)
        short = state.refusal_reason or "I can only answer questions about the cosmetics chemical dataset."
        details = SAFETY_NOTE
    else:
        short, details = draft.answer_short, draft.answer_details
    trace = state.trace + [event("finalize", f"response_type={rtype}", t)]
    for i, ev in enumerate(trace, 1):
        ev.step = i
    llm_name = getattr(ctx.llm, "name", "none")
    resp = Response(
        request_id=state.request_id, response_type=rtype, question=state.question, answer_short=short,
        answer_details=details, evidence=state.evidence, facts=state.facts, query_plan=trace,
        clarification=state.clarification, assumptions=list(dict.fromkeys(state.assumptions)),
        warnings=dedupe_warnings(state.warnings), confidence=confidence(state),
        meta={"db_sha256": ctx.engine.meta.get("csv_sha256"), "llm": llm_name if ctx.llm.available else "none",
              "embedding_model": ctx.settings.embed_model,
              "vectors": "on" if ctx.resolver._vectors_on() else "off (lexical only)", "version": "0.1.0",
              "elapsed_ms": round(sum(e.elapsed_ms for e in trace), 1)},
    )
    return {"response": resp, "trace": [trace[-1]]}
