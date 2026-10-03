"""Planner / orchestrator: scope check, intent classification, decomposition.

LLM (OpenAI by default, structured output) when available; deterministic keyword rules otherwise or on failure.
The planner never decides *which tools run* - routers and the query agent do that deterministically.
"""

from __future__ import annotations

import re

from chemrag.agents.context import AgentContext, Timer, event
from chemrag.etl.normalize import norm_key
from chemrag.llm.base import LLMPlan, LLMUnavailable, prompt, wrap_user
from chemrag.schemas import EntityType, Intent, Plan, SubTask, WarningItem
from chemrag.state import TurnState

MEDICAL = re.compile(
    r"\b(is|are|was|were)\b[^?.]*\b(safe|unsafe|dangerous|harmful|toxic|bad for|good for|healthy|carcinogenic)\b"
    r"|\bshould (i|we|my)\b|\bcan i (use|apply|wear)\b|\b(cause|causes|give|get)s? (me |you )?cancer\b"
    r"|\b(pregnan\w*|breastfeed\w*)\b|\bfor my (baby|kid|kids|child|children|skin|face)\b|\bside effects?\b"
    r"|\b(health|medical) (risk|advice|effect)s?\b|\brisk to (me|my|you)\b",
    re.IGNORECASE,
)
DOMAIN = re.compile(
    r"\b(product|chemical|cas\b|brand|compan|cosmetic|categor|discontinu|report|ingredient|contain|trend|"
    r"data|dataset|record|row|makeup|nail|hair|skin|lip|shampoo|sunscreen|lotion|removed|reformulat|cdph|cscp|"
    r"manufactur|substance|shade|variant|prop ?65|toxic|safe|items?|goods|lipstick|polish|perfume|fragrance)\w*",
    re.IGNORECASE,
)
INTENT_RULES: list[tuple[Intent, re.Pattern]] = [
    (Intent.COVERAGE, re.compile(r"\b(date range|time range|what years|which years|time period|coverage|"
                                 r"how far back|span of|cover(s|ed)?\b.*\b(years|dates|period))", re.IGNORECASE)),
    (Intent.DATA_QUALITY, re.compile(r"\b(data quality|missing|invalid|duplicat\w*|null|bad data|wrong|errors?|"
                                     r"inconsisten\w*|malformed|dirty|look(s)? wrong|data (problems?|issues?)|"
                                     r"problems? (in|with) the data|anomal\w*)\b", re.IGNORECASE)),
    (Intent.TREND, re.compile(r"\b(trends?|over time|by year|per year|each year|over the years|annual\w*|"
                              r"yearly|timeline|time series|year over year|year by year|year-by-year|"
                              r"changed? (over|across|through) (time|the years))\b", re.IGNORECASE)),
    (Intent.COMPARE, re.compile(r"\b(compare|comparison|compared|versus|vs\.?|difference between)\b", re.IGNORECASE)),
    (Intent.SUMMARIZE, re.compile(r"\b(summari[sz]e|summary|overview|profile|tell me about|describe)\b", re.IGNORECASE)),
    (Intent.LOOKUP, re.compile(r"\b(how many|count|number of|total)\b", re.IGNORECASE)),
    (Intent.LIST, re.compile(r"\b(which|list|show|find|what|give me|products? (with|containing|that))\b", re.IGNORECASE)),
]
SPLIT = re.compile(
    r"\?\s+(?=\S)|;\s*|\.\s+(?=[A-Z])|,?\s+and\s+(?=(?:how many|which|what|show|list|are there|did|compare|"
    r"summari[sz]e|also)\b)|,?\s+also\s+",
    re.IGNORECASE,
)


def classify(text: str) -> tuple[Intent, float]:
    for intent, pat in INTENT_RULES:
        if pat.search(text):
            return intent, 0.9
    return Intent.LIST, 0.75


def has_domain_signal(text: str, ctx: AgentContext) -> bool:
    if DOMAIN.search(text) or re.search(r"\d{2,7}-\d{2}-\d", text):
        return True
    toks = norm_key(text).split()
    gaz = ctx.resolver.gazetteer()
    for n in range(1, 6):
        for i in range(len(toks) - n + 1):
            g = " ".join(toks[i:i + n])
            if len(g) >= 4 and g in gaz:
                return True
    # misspelled chemical names ("titanium dioxyde") still count as being about the dataset
    from rapidfuzz import fuzz, process

    chems = ctx.resolver._choices.get(EntityType.CHEMICAL, [])
    for n in (1, 2, 3):
        for i in range(len(toks) - n + 1):
            g = " ".join(toks[i:i + n])
            if len(g) >= 6 and process.extractOne(g, chems, scorer=fuzz.ratio, score_cutoff=88):
                return True
    return False


def split_parts(text: str, ctx: AgentContext) -> tuple[list[str], str | None]:
    """Deterministic sub-question boundaries. Parts keep the user's exact words (entity names intact)."""
    parts = [p.strip(" ,") for p in SPLIT.split(text) if p and len(p.strip(" ,?.")) > 3] or [text]
    note = None
    if len(parts) > ctx.settings.max_subtasks:
        note = f"question had {len(parts)} parts; only the first {ctx.settings.max_subtasks} are answered"
    return parts[: ctx.settings.max_subtasks], note


def _subtasks(parts: list[str], intents: list[tuple[Intent, float]]) -> list[SubTask]:
    return [SubTask(id=f"t{i + 1}", text=p, intent=it, intent_confidence=conf, depends_on=[f"t{i}"] if i else [])
            for i, (p, (it, conf)) in enumerate(zip(parts, intents))]


def rules_plan(text: str, ctx: AgentContext) -> Plan:
    if MEDICAL.search(text):
        scope = "medical_advice"
    elif not has_domain_signal(text, ctx):
        scope = "unrelated"
    else:
        scope = "in_scope"
    if scope == "unrelated":
        return Plan(subtasks=[SubTask(id="t1", text=text, intent=Intent.OUT_OF_SCOPE)], mode="rules", scope=scope)
    parts, note = split_parts(text, ctx)
    intents = [(Intent.LOOKUP, 0.8) if scope == "medical_advice" else classify(p) for p in parts]
    return Plan(subtasks=_subtasks(parts, intents), mode="rules", scope=scope, note=note)


def llm_plan(text: str, ctx: AgentContext) -> Plan:
    """LLM decides scope (and intent where the keyword rules are unsure); the deterministic splitter decides the
    sub-question boundaries and keeps the user's exact wording.

    Measured with gpt-4o-mini: letting the LLM split/rewrite broke exact names ("Nail Polish and Enamel" ->
    "Nail Polish"), split "X or CAS Y" into two questions and split comparisons apart. Explicit keyword signals
    therefore win; the LLM fills in where the rules only had a default guess.
    """
    out: LLMPlan = ctx.llm.structured(prompt("planner"), wrap_user(text), LLMPlan)
    has_signal = has_domain_signal(text, ctx)
    if out.scope == "unrelated" and not has_signal:
        return Plan(subtasks=[SubTask(id="t1", text=text, intent=Intent.OUT_OF_SCOPE)], mode="llm", scope="unrelated")
    scope = "medical_advice" if MEDICAL.search(text) else ("in_scope" if out.scope == "unrelated" else out.scope)
    parts, note = split_parts(text, ctx)
    llm_intents = [Intent(s.intent) for s in out.subtasks if s.text.strip()]
    same_shape = len(llm_intents) == len(parts)
    intents: list[tuple[Intent, float]] = []
    for i, p in enumerate(parts):
        rule_intent, rule_conf = classify(p)
        if scope == "medical_advice":
            intents.append((Intent.LOOKUP, 0.8))
        elif rule_conf >= 0.9 or not same_shape or llm_intents[i] == Intent.OUT_OF_SCOPE:
            intents.append((rule_intent, rule_conf))  # explicit keyword signal (or LLM split differently)
        else:
            intents.append((llm_intents[i], 0.85))  # rules only had a default guess; trust the LLM
    return Plan(subtasks=_subtasks(parts, intents), mode="llm+rules", scope=scope, note=note)


def planner_node(state: TurnState, ctx: AgentContext) -> dict:
    t = Timer()
    text = state.clean_question
    warnings: list[WarningItem] = []
    if ctx.llm.available:
        try:
            plan = llm_plan(text, ctx)
        except LLMUnavailable as e:
            warnings.append(WarningItem(code="llm_unavailable", severity="info",
                                        message=f"Planner fell back to rules: {e}"))
            plan = rules_plan(text, ctx)
    else:
        plan = rules_plan(text, ctx)
    if plan.note:
        warnings.append(WarningItem(code="question_truncated", message=plan.note))
    update: dict = {"plan": plan, "current": 0, "warnings": warnings}
    if plan.scope == "unrelated":
        update["response_type"] = "refusal"
        update["refusal_reason"] = ("This assistant only answers questions about the California Safe Cosmetics "
                                    "Program chemical disclosure dataset.")
    summary = f"scope={plan.scope}; " + "; ".join(f"{s.id}:{s.intent.value}" for s in plan.subtasks)
    update["trace"] = [event("planner", summary, t, mode=plan.mode,
                             subtasks=[s.model_dump(mode="json") for s in plan.subtasks], scope=plan.scope)]
    return update
