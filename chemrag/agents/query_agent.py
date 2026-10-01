"""Structured query agent (deterministic): resolved entities + constraints -> Filters -> tool calls.

Also owns the data-grounded safety checks that need the database:
out-of-range dates, unresolved entities, empty-result diagnostics, dominant-chemical and trade-secret
warnings, cross-company brands, synonym/family notes and CAS repairs.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from datetime import date

from chemrag.agents.context import AgentContext, Timer, event
from chemrag.agents.resolution import is_blocking
from chemrag.query.filters import Filters
from chemrag.schemas import (
    DATE_FIELD_LABELS,
    DateConstraint,
    EntityType,
    Extraction,
    Intent,
    SubTask,
    ToolResult,
    TraceEvent,
    WarningItem,
)
from chemrag.state import TurnState

TYPE_FIELD = {
    EntityType.CHEMICAL: ("chem_group_ids", "chemical"),
    EntityType.COMPANY: ("company_keys", "company"),
    EntityType.BRAND: ("brand_keys", "brand"),
    EntityType.PRIMARY_CATEGORY: ("primary_category_keys", "primary_category"),
    EntityType.SUBCATEGORY: ("subcategory_keys", "subcategory"),
}
RELAX_ORDER = [
    ("dates", "date range"), ("subcategory_keys", "subcategory"), ("primary_category_keys", "category"),
    ("discontinued", "discontinued status"), ("chem_removed", "removal status"), ("brand_keys", "brand"),
    ("company_keys", "company"), ("cdph_ids", "product"), ("chem_group_ids", "chemical"), ("cas_numbers", "CAS number"),
]


MISSING_REF = re.compile(r"\b(contain|contains|containing|with|have|has|for|by|of)\s+(it|them|this|that|those|these|"
                         r"this chemical|that chemical|the chemical|the brand|the company|the product)\b\s*[?.!]*$"
                         r"|^\s*(which|what) products\??\s*$", re.IGNORECASE)


def _family_question(state: TurnState, sub: SubTask) -> bool:
    """'Which retinoids ...' names a chemical family, so the user wants the chemicals listed."""
    return sub.intent == Intent.LIST and any(
        c.method == "family" for r in state.latest_resolutions(sub.id) if r.status == "resolved" for c in r.chosen)


class Plan:
    """Per-subtask accumulation of filters, blockers and notes."""

    def __init__(self) -> None:
        self.data: dict = {k: [] for k in ["chem_group_ids", "cas_numbers", "company_keys", "brand_keys", "cdph_ids",
                                           "primary_category_keys", "subcategory_keys", "exclude_chem_group_ids"]}
        self.labels: dict[str, list[str]] = {}
        self.blockers: list[str] = []
        self.per_mention: list[tuple[EntityType, str, dict, dict]] = []  # for compare
        self.brand_companies: dict[str, list[int]] = {}
        self.include_trade_secret = False

    def add(self, field: str, value, label_key: str, label: str) -> None:
        if value not in self.data[field]:
            self.data[field].append(value)
        if label not in self.labels.setdefault(label_key, []):
            self.labels[label_key].append(label)


def build_plan(state: TurnState, sub: SubTask) -> Plan:
    p = Plan()
    for r in state.latest_resolutions(sub.id):
        if r.status == "not_found":
            if is_blocking(r):
                p.blockers.append(r.note or f"'{r.mention.text}' not found")
            continue
        if r.status != "resolved":
            continue
        part: dict = {}
        labels: dict = {}
        for c in r.chosen:
            if r.mention.type == EntityType.CAS:
                names = " / ".join(x.display_name for x in r.chosen)
                label = f"{names} (CAS {c.matched_alias})"
                p.add("cas_numbers", c.matched_alias, "cas", label)
                part.setdefault("cas_numbers", []).append(c.matched_alias)
                labels["cas"] = [label]
                break
            field, lk = TYPE_FIELD[c.entity_type]
            if r.mention.negated and c.entity_type == EntityType.CHEMICAL:
                p.add("exclude_chem_group_ids", c.canonical_id, "exclude", c.display_name)
                continue
            p.add(field, c.canonical_id, lk, c.display_name)
            part.setdefault(field, []).append(c.canonical_id)
            labels.setdefault(lk, []).append(c.display_name)
            if c.display_name == "Trade Secret":
                p.include_trade_secret = True
            if c.entity_type == EntityType.BRAND and len(c.extra.get("company_keys", [])) > 1:
                p.brand_companies[c.display_name] = c.extra["company_keys"]
        if part:
            etype = EntityType.CAS if r.mention.type == EntityType.CAS else r.chosen[0].entity_type
            p.per_mention.append((etype, r.mention.text, part, labels))
    for pr in state.latest_product_resolutions(sub.id):
        if pr.status == "resolved":
            for cid in pr.cdph_ids:
                p.add("cdph_ids", cid, "product", f"'{pr.mention.text}'")
        elif pr.status == "not_found" and pr.mention.confidence >= 0.7:
            p.blockers.append(pr.note or f"product '{pr.mention.text}' not found")
    return p


def make_filters(p: Plan, ex: Extraction, dates: list[DateConstraint]) -> Filters:
    return Filters(**p.data, dates=dates, discontinued=ex.discontinued, chem_removed=ex.chem_removed,
                   include_trade_secret=p.include_trade_secret, labels=p.labels)


def check_dates(dates: list[DateConstraint], ctx: AgentContext, sub_id: str
                ) -> tuple[list[DateConstraint], list[str], list[WarningItem]]:
    """Clip partially-overlapping ranges; report fully out-of-range ones (answer without querying)."""
    if not dates:
        return dates, [], []
    ctx.engine.run_sql("coverage_check", sub_id, {"fields": [d.field for d in dates]},
                       "SELECT value FROM dataset_meta WHERE key = 'coverage'", [])
    kept, out_of_range, warnings = [], [], []
    for d in dates:
        lo, hi = ctx.engine.coverage(d.field)
        label = DATE_FIELD_LABELS[d.field]
        if (d.end and d.end < lo) or (d.start and d.start > hi):
            msg = (f"{label} in this dataset ranges from {lo.isoformat()} to {hi.isoformat()}; "
                   f"there is no data for '{d.source_text}'.")
            if d.field == "chem_removed_date":
                bad = ctx.engine.run_sql("dq_lookup", sub_id, {"issue": "removed_date_future"},
                                         "SELECT count(*) AS n, min(row_id) AS sample_row_id FROM dq_issues "
                                         "WHERE issue_code = 'removed_date_future'", [])[1][0]
                if bad["n"]:
                    msg += (f" Note: {bad['n']} rows carry an invalid ChemicalDateRemoved in 2103/2104 (e.g. row "
                            f"{bad['sample_row_id']}); these are treated as 'removed, date unknown'.")
            out_of_range.append(msg)
            continue
        start, end = d.start, d.end
        if start and start < lo or end and end > hi:
            start = max(start, lo) if start else start
            end = min(end, hi) if end else end
            warnings.append(WarningItem(code="date_range_clipped", severity="info",
                                        message=f"'{d.source_text}' partly falls outside the {label} coverage "
                                                f"({lo.isoformat()} to {hi.isoformat()}); clipped to "
                                                f"{start or lo} to {end or hi}."))
        kept.append(d.model_copy(update={"start": start, "end": end}))
    return kept, out_of_range, warnings


def diagnose_empty(f: Filters, ctx: AgentContext, sub_id: str) -> list[str]:
    notes = []
    for field, label in RELAX_ORDER:
        val = getattr(f, field)
        if val in (None, [], False) and field not in ("discontinued", "chem_removed"):
            continue
        if field in ("discontinued", "chem_removed") and val is None:
            continue
        relaxed = f.without(field)
        if relaxed.is_empty():
            continue
        _, totals = ctx.engine.product_totals(relaxed, sub_id, tool="diagnostic_relaxation")
        notes.append(f"without the {label} constraint: {totals['n_products']:,} products")
    return notes


def run_subtask(state: TurnState, sub: SubTask, ctx: AgentContext
                ) -> tuple[list[ToolResult], list[WarningItem], list[str]]:
    eng = ctx.engine
    s = ctx.settings
    warnings: list[WarningItem] = []
    assumptions: list[str] = []
    if sub.intent == Intent.COVERAGE:
        return [eng.dataset_coverage(sub.id)], warnings, assumptions
    if sub.intent == Intent.DATA_QUALITY:
        return [eng.dq_summary(sub.id)], warnings, assumptions
    if sub.intent == Intent.OUT_OF_SCOPE:
        return [], warnings, assumptions

    ex = state.extraction_for(sub.id) or Extraction(subtask_id=sub.id)
    p = build_plan(state, sub)
    if p.blockers:
        return [ToolResult(call_id="-", subtask_id=sub.id, tool="none", description="entity not found",
                           status="not_found", notes=p.blockers)], [
            WarningItem(code="entity_not_found", message=b) for b in p.blockers], assumptions

    dates, out_of_range, w = check_dates(ex.dates, ctx, sub.id)
    warnings += w
    if out_of_range:
        warnings += [WarningItem(code="out_of_range_date", message=m) for m in out_of_range]
        return [ToolResult(call_id="-", subtask_id=sub.id, tool="coverage_check", description="date out of range",
                           status="no_data_in_range", notes=out_of_range)], warnings, assumptions

    f = make_filters(p, ex, dates)
    if state.plan and state.plan.scope == "medical_advice" and f.is_empty():
        return [], warnings, assumptions
    if f.is_empty() and sub.intent in (Intent.LIST, Intent.LOOKUP) and not ex.group_by and MISSING_REF.search(sub.text):
        return [ToolResult(call_id="-", subtask_id=sub.id, tool="none", description="missing entity",
                           status="skipped", notes=["missing_entity"])], warnings, assumptions
    limit = min(ex.limit or (10 if ex.group_by else state.options.limit), s.max_list_limit)
    results: list[ToolResult] = []

    if sub.intent == Intent.TREND:
        field = ex.trend_field or "initial_reported"
        if not ex.trend_field or field == "initial_reported":
            assumptions.append("Trend counts distinct products by the year they were first reported "
                               "(InitialDateReported).")
        results.append(eng.trend_by_year(f, sub.id, date_field=field))
    elif sub.intent == Intent.COMPARE:
        results += run_compare(p, ex, dates, f, sub, ctx, warnings)
    elif sub.intent == Intent.SUMMARIZE:
        results.append(eng.count_products(f, sub.id))
        results.append(eng.chemicals_for(f, sub.id, limit=8))
        results.append(eng.count_products(f, sub.id, group_by="subcategory", limit=5))
        results.append(eng.trend_by_year(f, sub.id))
    elif (ex.asks_for_chemicals or _family_question(state, sub)) and not ex.group_by:
        results.append(eng.chemicals_for(f, sub.id, limit=max(limit, 50)))
    elif ex.group_by:
        gb = ex.group_by[0]
        results.append(eng.count_products(f, sub.id, group_by=gb, limit=limit))
    elif sub.intent == Intent.LOOKUP:
        results.append(eng.count_products(f, sub.id))
    else:
        results.append(eng.find_products(f, sub.id, limit=limit, offset=state.options.offset))

    main = results[0] if results else None
    if main is not None and main.status == "empty":
        notes = diagnose_empty(f, ctx, sub.id)
        main.notes += notes
        warnings.append(WarningItem(code="empty_result",
                                    message=f"No products match all constraints ({f.describe()})."
                                            + (" Diagnostics: " + "; ".join(notes) if notes else "")))
    warnings += post_checks(results, f, p, ctx)
    if not any(a.startswith("'Contains' includes") for a in state.assumptions):
        assumptions += contains_assumption(results, f)
    return results, warnings, assumptions


def run_compare(p: Plan, ex: Extraction, dates, base: Filters, sub: SubTask, ctx: AgentContext,
                warnings: list[WarningItem]) -> list[ToolResult]:
    by_type: OrderedDict[EntityType, list] = OrderedDict()
    for etype, text, part, labels in p.per_mention:
        by_type.setdefault(etype, []).append((text, part, labels))
    dim = next((t for t, items in by_type.items() if len(items) >= 2), None)
    if dim is None:
        warnings.append(WarningItem(code="compare_needs_two", message="A comparison needs at least two companies, "
                                    "brands, categories or chemicals; showing a single count instead."))
        return [ctx.engine.count_products(base, sub.id)]
    results = []
    for _text, part, labels in by_type[dim]:
        data = {**p.data}
        for field, _ in TYPE_FIELD.values():
            if field in part or (dim == EntityType.CAS and field == "cas_numbers"):
                data[field] = []
        if dim == EntityType.CAS:
            data["cas_numbers"] = []
        data.update(part)
        common_labels = {k: v for k, v in p.labels.items() if k not in labels}
        f = Filters(**data, dates=dates, discontinued=ex.discontinued, chem_removed=ex.chem_removed,
                    include_trade_secret=p.include_trade_secret, labels={**common_labels, **labels})
        r = ctx.engine.chemicals_for(f, sub.id, limit=10) if ex.asks_for_chemicals else ctx.engine.count_products(f, sub.id)
        r.notes.append(f"compare_item={'/'.join(sum(labels.values(), []))}")
        results.append(r)
    return results


def post_checks(results: list[ToolResult], f: Filters, p: Plan, ctx: AgentContext) -> list[WarningItem]:
    th = ctx.settings.thresholds
    out: list[WarningItem] = []
    for r in results:
        n = r.totals.get("n_products", 0)
        if r.tool == "find_products" and r.truncated:
            out.append(WarningItem(code="truncated", severity="info",
                                   message=f"Listing {len(r.rows)} of {n:,} matching products; use --limit/--page "
                                           "to see more. Counts are always over the full result."))
        if r.tool in ("chemicals_for",) and r.rows and n:
            top = r.rows[0]
            share = top["n_products"] / n
            second = r.rows[1]["n_products"] if len(r.rows) > 1 else 0
            if share > th.dominance_share and n >= 20 and top["n_products"] > 2 * second:
                out.append(WarningItem(code="dominant_chemical", severity="info",
                                       message=f"{top['chemical']} appears in {top['n_products']:,} of {n:,} matching "
                                               f"products ({share:.0%}); it dominates this dataset (81% of all rows).",
                                       row_ids=[top["sample_row_id"]]))
            if r.truncated:
                out.append(WarningItem(code="truncated", severity="info",
                                       message=f"Showing {len(r.rows)} of {r.totals.get('n_chemicals')} chemicals."))
        if r.tool == "count_products" and r.truncated:
            out.append(WarningItem(code="truncated", severity="info",
                                   message=f"Showing top {len(r.rows)} of {r.totals.get('n_groups')} groups."))
        ts = r.totals.get("n_products_trade_secret")
        if ts:
            out.append(WarningItem(code="trade_secret_present", severity="info",
                                   message=f"{ts:,} matching products also report an undisclosed 'Trade Secret' "
                                           "chemical, which is excluded from chemical lists.",
                                   row_ids=[r.totals["trade_secret_sample_row_id"]]))
        if r.totals.get("n_rows_cas_repaired"):
            out.append(WarningItem(code="data_conflict", severity="info",
                                   message=f"{r.totals['n_rows_cas_repaired']:,} matched rows had a malformed or "
                                           "missing CAS number that was repaired or filled from the chemical's CasId."))
    for brand, comps in p.brand_companies.items():
        if not f.company_keys:
            out.append(WarningItem(code="brand_multiple_companies", severity="info",
                                   message=f"Brand '{brand}' is reported by {len(comps)} different companies; "
                                           "results include all of them."))
    if f.chem_group_ids:
        rel = ctx.engine.family_relatives(f.chem_group_ids)
        if rel:
            fam = rel[0]["family_label"]
            names = ", ".join(f"{x['group_name']} ({x['n_products']:,} products overall)" for x in rel[:4])
            out.append(WarningItem(code="related_chemicals", severity="info",
                                   message=f"Related {fam} entries are reported separately and not included: {names}."))
        members = ctx.engine.group_members(f.chem_group_ids)
        for m in members:
            if len(m["member_names"]) > 1:
                out.append(WarningItem(code="synonyms_merged", severity="info",
                                       message=f"'{m['group_name']}' includes {len(m['member_names'])} reported names: "
                                               + "; ".join(m["member_names"])))
    return dedupe(out)


def contains_assumption(results: list[ToolResult], f: Filters) -> list[str]:
    if not (f.chem_group_ids or f.cas_numbers) or f.chem_removed is not None:
        return []
    for r in results:
        removed = r.totals.get("n_products_chem_removed")
        if removed:
            return [f"'Contains' includes chemicals that were later removed/reformulated: {removed:,} of "
                    f"{r.totals['n_products']:,} matching products had the chemical removed."]
    return []


def dedupe(items: list[WarningItem]) -> list[WarningItem]:
    seen, out = set(), []
    for w in items:
        if (w.code, w.message) not in seen:
            seen.add((w.code, w.message))
            out.append(w)
    return out


def query_node(state: TurnState, ctx: AgentContext) -> dict:
    t = Timer()
    sub = state.current_subtask
    results, warnings, assumptions = run_subtask(state, sub, ctx)
    calls = ctx.engine.drain_calls()
    extra: dict = {}
    if any(r.status == "skipped" and "missing_entity" in r.notes for r in results):
        from chemrag.schemas import Clarification

        extra = {"response_type": "clarification", "clarification": Clarification(
            question="Which chemical, CAS number, brand, company or category do you mean? For example: "
                     "'Which products contain CAS 75-07-0?'", mention=sub.text)}
    trace: list[TraceEvent] = [event(
        "query", f"{sub.id}: " + (", ".join(f"{r.tool}[{r.status}]" for r in results) or "no query needed"), t,
        filters=[r.description for r in results])]
    for c in calls:
        trace.append(TraceEvent(agent="query", mode="sql", summary=f"{c.id} {c.tool} -> {c.row_count} rows",
                                output={"id": c.id, "tool": c.tool, "params": c.params, "sql": c.sql,
                                        "bound_params": c.bound_params, "row_count": c.row_count,
                                        "result_sha1": c.result_sha1},
                                elapsed_ms=c.elapsed_ms))
    return {"results": results, "tool_calls": calls, "warnings": warnings, "assumptions": assumptions,
            "trace": trace, "current": state.current + 1, **extra}


__all__ = ["date", "query_node", "run_subtask"]
