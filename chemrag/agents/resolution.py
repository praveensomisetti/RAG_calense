"""Resolver node (entities), record-retrieval node (products) and the clarification node."""

from __future__ import annotations

from langgraph.types import Command, interrupt

from chemrag.agents.context import AgentContext, Timer, event
from chemrag.schemas import (
    Clarification,
    ClarificationOption,
    EntityType,
    Mention,
    ProductResolution,
    Resolution,
    WarningItem,
)
from chemrag.state import TurnState

PROPER_HINTS = [EntityType.COMPANY, EntityType.BRAND]


def is_blocking(r: Resolution) -> bool:
    """A failed low-confidence cue phrase is noise; anything the user clearly named must resolve."""
    return not (r.mention.source == "cue" and r.mention.confidence < 0.7)


def resolver_node(state: TurnState, ctx: AgentContext) -> dict:
    t = Timer()
    sub = state.current_subtask
    ex = state.extraction_for(sub.id)
    resolutions: list[Resolution] = []
    warnings: list[WarningItem] = []
    assumptions: list[str] = []
    for m in ex.mentions if ex else []:
        if m.type == EntityType.PRODUCT:
            continue
        hints = PROPER_HINTS if (m.type == EntityType.UNKNOWN and m.source == "cue") else None
        r = ctx.resolver.resolve(m, sub.id, hints)
        if r.status == "ambiguous" and state.options.assume_best and r.candidates:
            top = r.candidates[0]
            alts = ", ".join(c.display_name for c in r.candidates[1:5])
            r = r.model_copy(update={"status": "resolved", "chosen": [top],
                                     "note": f"'{m.text}' was ambiguous; proceeding with {top.display_name}"})
            warnings.append(WarningItem(code="ambiguous_entity",
                                        message=f"'{m.text}' could also mean: {alts}. Used {top.display_name}."))
        if r.status == "resolved" and r.note:
            assumptions.append(r.note)
        if r.status == "resolved" and r.mention.type == EntityType.CAS and len(r.chosen) > 1:
            warnings.append(WarningItem(code="data_conflict", severity="info", message=r.note or ""))
        if r.status == "not_found" and not is_blocking(r):
            warnings.append(WarningItem(code="ignored_phrase", severity="info",
                                        message=f"Ignored '{m.text}': it does not match any company or brand."))
        resolutions.append(r)
    if ctx.resolver.vector_warning and not any(w.code == "vector_unavailable" for w in state.warnings):
        warnings.append(WarningItem(code="vector_unavailable", severity="info", message=ctx.resolver.vector_warning))
    summary = "; ".join(
        f"'{r.mention.text}' -> {r.status}"
        + (f" {', '.join(f'{c.entity_type.value}:{c.display_name} [{c.method} {c.score:.2f}]' for c in r.chosen[:3])}"
           if r.chosen else "")
        for r in resolutions) or "nothing to resolve"
    return {"resolutions": resolutions, "warnings": warnings, "assumptions": assumptions,
            "trace": [event("resolver", f"{sub.id}: {summary}", t,
                            resolutions=[r.model_dump(mode="json", exclude={"subtask_id"}) for r in resolutions])]}


def retrieval_node(state: TurnState, ctx: AgentContext) -> dict:
    """Semantic/lexical record retrieval for product-name mentions -> CDPHIds (evidence still comes from SQL)."""
    t = Timer()
    sub = state.current_subtask
    ex = state.extraction_for(sub.id)
    brand_keys, company_keys = [], []
    for r in state.latest_resolutions(sub.id):
        for c in r.chosen:
            if c.entity_type == EntityType.BRAND:
                brand_keys.append(c.canonical_id)
            elif c.entity_type == EntityType.COMPANY:
                company_keys.append(c.canonical_id)
    out: list[ProductResolution] = []
    warnings: list[WarningItem] = []
    assumptions: list[str] = []
    for m in ex.mentions:
        if m.type != EntityType.PRODUCT:
            continue
        pr = ctx.resolver.resolve_product(m, sub.id, brand_keys or None, company_keys or None)
        if m.confidence < 0.7 and pr.status != "resolved":
            # a guessed product phrase that does not clearly match is dropped (the brand still applies)
            pr = pr.model_copy(update={"status": "not_found", "note": f"no specific product matched '{m.text}'"})
        if pr.status == "ambiguous" and state.options.assume_best:
            top = pr.candidates[0]
            pr = pr.model_copy(update={"status": "resolved", "cdph_ids": top["cdph_ids"],
                                       "note": f"'{m.text}' matched several products; using all "
                                               f"{len(top['cdph_ids'])} named '{top['product_name']}'"})
            warnings.append(WarningItem(code="ambiguous_entity", message=pr.note))
        if pr.status == "resolved" and pr.note:
            assumptions.append(pr.note)
        out.append(pr)
    summary = "; ".join(f"'{p.mention.text}' -> {p.status} ({len(p.cdph_ids)} CDPHIds)" for p in out)
    return {"product_resolutions": out, "warnings": warnings, "assumptions": assumptions,
            "trace": [event("retrieval", f"{sub.id}: {summary}", t,
                            mode="lexical+vector" if ctx.resolver._vectors_on() else "lexical",
                            products=[p.model_dump(mode="json", exclude={"subtask_id"}) for p in out])]}


# ------------------------------------------------------------------ routing helpers
def pending_step(state: TurnState, extra_res: list[Resolution] = (), extra_prod: list[ProductResolution] = ()) -> str:
    sub = state.current_subtask
    latest = {r.mention.text.casefold(): r for r in state.latest_resolutions(sub.id)}
    latest.update({r.mention.text.casefold(): r for r in extra_res})
    if any(r.status == "ambiguous" for r in latest.values()):
        return "clarify"
    prods = {p.mention.text.casefold(): p for p in state.latest_product_resolutions(sub.id)}
    prods.update({p.mention.text.casefold(): p for p in extra_prod})
    if any(p.status == "ambiguous" for p in prods.values()):
        return "clarify"
    ex = state.extraction_for(sub.id)
    wanted = {m.text.casefold() for m in (ex.mentions if ex else []) if m.type == EntityType.PRODUCT}
    if wanted - set(prods):
        return "retrieval"
    return "query"


def clarify_node(state: TurnState, ctx: AgentContext) -> Command:
    t = Timer()
    sub = state.current_subtask
    amb = next((r for r in state.latest_resolutions(sub.id) if r.status == "ambiguous"), None)
    pamb = None if amb else next((p for p in state.latest_product_resolutions(sub.id) if p.status == "ambiguous"), None)
    if amb:
        options = [ClarificationOption(type=c.entity_type.value, name=c.display_name, canonical_id=c.canonical_id,
                                       detail=f"{c.n_products:,} products" + (f", match {c.score:.2f}" if c.score < 1 else ""))
                   for c in amb.candidates[:6]]
        kind = amb.candidates[0].entity_type.value if amb.candidates else "entity"
        clar = Clarification(question=f"Which {kind.replace('_', ' ')} did you mean by '{amb.mention.text}'?",
                             mention=amb.mention.text, options=options)
    else:
        options = [ClarificationOption(type="product", name=c["product_name"],
                                       detail=f"{len(c['cdph_ids'])} product(s); "
                                              f"{', '.join((c['companies'] or [])[:2])}")
                   for c in pamb.candidates[:8]]
        clar = Clarification(question=f"Which product did you mean by '{pamb.mention.text}'? "
                                      "You can also add the brand or company to narrow it down.",
                             mention=pamb.mention.text, options=options)
        if pamb.note:
            clar.question = f"{pamb.note}. " + clar.question

    if state.options.interactive:
        choice = interrupt(clar.model_dump(mode="json"))
        if isinstance(choice, int) and 0 <= choice < len(options):
            chosen = clar.options[choice]
            mention = (amb or pamb).mention
            ev = event("clarify", f"user chose '{chosen.name}' for '{mention.text}'", t)
            if amb:
                cand = amb.candidates[choice]
                new = Resolution(subtask_id=sub.id, mention=mention, status="resolved", chosen=[cand],
                                 candidates=amb.candidates, note=f"user clarified '{mention.text}' = {cand.display_name}")
                nxt = pending_step(state, extra_res=[new])
                return Command(goto=nxt, update={"resolutions": [new], "trace": [ev],
                                                 "assumptions": [new.note]})
            cand = pamb.candidates[choice]
            new_p = ProductResolution(subtask_id=sub.id, mention=mention, status="resolved", cdph_ids=cand["cdph_ids"],
                                      candidates=pamb.candidates, note=f"user clarified product = {cand['product_name']}")
            nxt = pending_step(state, extra_prod=[new_p])
            return Command(goto=nxt, update={"product_resolutions": [new_p], "trace": [ev],
                                             "assumptions": [new_p.note]})
    return Command(goto="finalize", update={
        "clarification": clar, "response_type": "clarification",
        "trace": [event("clarify", f"needs clarification: {clar.question}", t,
                        options=[o.model_dump() for o in clar.options])],
    })


__all__ = ["Mention", "clarify_node", "pending_step", "resolver_node", "retrieval_node"]
