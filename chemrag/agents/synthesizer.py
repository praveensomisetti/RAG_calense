"""Answer synthesizer.

* Facts (F1..Fn) are built deterministically from tool results - every number in an answer traces to one.
* `answer_short` is always templated from facts (never LLM-written, so it cannot contain an invented number).
* `answer_details` = optional Gemini narrative bullets citing [F#] (checked by the verifier) + a deterministic
  listing/breakdown section.
"""

from __future__ import annotations

from typing import Any

from chemrag.agents.context import AgentContext, Timer, event
from chemrag.llm.base import LLMNarrative, LLMUnavailable, prompt
from chemrag.schemas import (
    DATE_FIELD_LABELS,
    Draft,
    EvidenceItem,
    Fact,
    Intent,
    SubTask,
    ToolResult,
    WarningItem,
)
from chemrag.state import TurnState

SAFETY_NOTE = ("This dataset records what companies reported to the California Safe Cosmetics Program; it does not "
               "assess safety, exposure or health risk.")


def n(x: Any) -> str:
    return f"{x:,}" if isinstance(x, int) else str(x)


def plural(k: int, word: str) -> str:
    return f"{n(k)} {word}" + ("" if k == 1 else ("es" if word.endswith(("s", "x")) else "s"))


class FactSheet:
    def __init__(self) -> None:
        self.facts: list[Fact] = []

    def add(self, statement: str, value: Any = None, call_id: str | None = None) -> str:
        fid = f"F{len(self.facts) + 1}"
        self.facts.append(Fact(id=fid, statement=statement, value=value, call_id=call_id))
        return fid


def _subject(r: ToolResult) -> str:
    return r.description


def facts_for(r: ToolResult, fs: FactSheet, sub: SubTask) -> tuple[str, list[str], list[str]]:
    """Return (short sentence, detail lines, fact ids) for one tool result."""
    t = r.totals
    ids: list[str] = []
    lines: list[str] = []
    if r.status == "not_found":
        msg = "; ".join(r.notes)
        ids.append(fs.add(f"Entity not found: {msg}", None, r.call_id))
        return f"No answer: {msg.rstrip('.?')}.", [], ids
    if r.status == "no_data_in_range":
        msg = " ".join(r.notes)
        ids.append(fs.add(msg, None, r.call_id))
        return f"No data in range. {msg}", [], ids

    if r.tool in ("find_products", "count_products") and "n_products" in t and not (r.tool == "count_products"
                                                                                        and r.rows):
        np_ = t["n_products"]
        ids.append(fs.add(f"Products {_subject(r)}: {n(np_)}", np_, r.call_id))
        ids.append(fs.add(f"Matching rows: {n(t['n_rows'])}; report records (ChemicalId): {n(t['n_report_records'])}",
                          t["n_rows"], r.call_id))
        ids.append(fs.add(f"Companies: {n(t['n_companies'])}; brands: {n(t['n_brands'])}", t["n_companies"], r.call_id))
        ids.append(fs.add(f"Discontinued products among them: {n(t['n_discontinued_products'])}",
                          t["n_discontinued_products"], r.call_id))
        if r.description.startswith("containing"):
            ids.append(fs.add(f"Products where the chemical was later removed/reformulated: "
                              f"{n(t['n_products_chem_removed'])}", t["n_products_chem_removed"], r.call_id))
        if not np_:
            lines += ["No products match all constraints."] + [f"- {x}" for x in r.notes if x.startswith("without")]
            return f"No products found {_subject(r)}.", lines, ids
        short = f"{plural(np_, 'product')} {_subject(r)}"
        if np_ and t["n_discontinued_products"] and "discontinued" not in r.description:
            short += f" ({n(t['n_discontinued_products'])} of them discontinued)"
        if " from " not in r.description:
            short += f", reported by {n(t['n_companies'])} compan{'y' if t['n_companies'] == 1 else 'ies'}"
        short += "."
        if r.tool == "find_products" and r.rows:
            shown = len(r.rows)
            lines.append(f"Products ({shown} of {n(np_)} shown):")
            for row in r.rows:
                bits = [f"**{row['product_name']}**", f"{row.get('brand_name') or '(no brand)'} / {row['company_name']}",
                        "chemicals: " + ", ".join(row["chemicals"])]
                if row.get("discontinued_date"):
                    bits.append(f"discontinued {row['discontinued_date']}")
                if row.get("chemical_removed"):
                    bits.append("chemical later removed")
                lines.append(f"- {' — '.join(bits)} (CDPHId {row['cdph_id']}, row {row['first_row_id']})")
        return short, lines, ids

    if r.tool == "count_products" and r.rows:
        gb = next((x.split("=", 1)[1] for x in r.notes if x.startswith("group_by=")), "group")
        ids.append(fs.add(f"Products {_subject(r)}: {n(t['n_products'])}; distinct {gb} values: "
                          f"{n(t.get('n_groups'))}", t["n_products"], r.call_id))
        lines.append(f"Products by {gb.replace('_', ' ')} (top {len(r.rows)} of {n(t.get('n_groups'))}):")
        for i, row in enumerate(r.rows, 1):
            fid = fs.add(f"{row['label']}: {n(row['n_products'])} products", row["n_products"], r.call_id)
            ids.append(fid)
            lines.append(f"{i}. {row['label']} — {plural(row['n_products'], 'product')} (sample row {row['sample_row_id']})")
        top = ", ".join(f"{row['label']} ({n(row['n_products'])})" for row in r.rows[:3])
        label = {"company": "companies", "brand": "brands", "subcategory": "subcategories",
                 "primary_category": "categories", "chemical": "chemicals"}.get(gb, gb)
        return f"Top {label} by number of products {_subject(r)}: {top}.", lines, ids

    if r.tool == "chemicals_for":
        np_ = t.get("n_products", 0)
        ids.append(fs.add(f"Products {_subject(r)}: {n(np_)}; distinct chemicals reported: "
                          f"{n(t.get('n_chemicals', 0))}", np_, r.call_id))
        if t.get("n_products_trade_secret"):
            ids.append(fs.add(f"Products that also report an undisclosed 'Trade Secret' chemical: "
                              f"{n(t['n_products_trade_secret'])}", t["n_products_trade_secret"], r.call_id))
        lines.append(f"Chemicals reported ({len(r.rows)} of {n(t.get('n_chemicals', 0))}), by number of products:")
        for row in r.rows:
            ids.append(fs.add(f"{row['chemical']}: {n(row['n_products'])} products "
                              f"({n(row['n_products_removed'])} with the chemical removed)", row["n_products"],
                              r.call_id))
            extra = []
            if len(row["reported_names"]) > 1 or row["reported_names"][0] != row["chemical"]:
                extra.append("reported as " + "; ".join(row["reported_names"]))
            if row["cas_numbers"]:
                extra.append("CAS " + ", ".join(row["cas_numbers"]))
            if row["n_products_removed"]:
                extra.append(f"removed in {n(row['n_products_removed'])}")
            lines.append(f"- **{row['chemical']}** — {plural(row['n_products'], 'product')}"
                         + (f" ({'; '.join(extra)})" if extra else "") + f" (sample row {row['sample_row_id']})")
        if not r.rows:
            return f"No chemicals are reported for {_subject(r)}.", lines, ids
        top = ", ".join(f"{row['chemical']} ({n(row['n_products'])})" for row in r.rows[:3])
        return (f"{plural(t.get('n_chemicals', 0), 'chemical')} reported across {plural(np_, 'product')} "
                f"{_subject(r)}; most common: {top}."), lines, ids

    if r.tool == "trend_by_year":
        field = DATE_FIELD_LABELS.get(t.get("date_field", "initial_reported"), t.get("date_field"))
        nz = [row for row in r.rows if row["n_products"]]
        ids.append(fs.add(f"Products {_subject(r)}: {n(t['n_products'])}", t["n_products"], r.call_id))
        lines.append(f"Distinct products by year of {field}:")
        peak = max(r.rows, key=lambda x: x["n_products"]) if r.rows else None
        width = max((row["n_products"] for row in r.rows), default=0) or 1
        for row in r.rows:
            ids.append(fs.add(f"{row['year']}: {n(row['n_products'])} products (by {field})", row["n_products"],
                              r.call_id))
            bar = "█" * max(1 if row["n_products"] else 0, round(24 * row["n_products"] / width))
            lines.append(f"- {row['year']}: {n(row['n_products']):>7} {bar}")
        if not nz:
            return f"No dated records for {_subject(r)} by {field}.", lines, ids
        first, last = nz[0], nz[-1]
        return (f"Products {_subject(r)} by year of {field}: peak in {peak['year']} ({n(peak['n_products'])}); "
                f"{first['year']}: {n(first['n_products'])}, {last['year']}: {n(last['n_products'])} "
                f"(total {n(t['n_products'])} products)."), lines, ids

    if r.tool == "dataset_coverage":
        cov = t["coverage"]
        for field, c in cov.items():
            ids.append(fs.add(f"{field}: {c['min']} to {c['max']} ({n(c['non_null'])} non-null rows)", c, r.call_id))
            lines.append(f"- {field}: {c['min']} → {c['max']} ({n(c['non_null'])} rows with a value)")
        ids.append(fs.add(f"Rows: {n(t['n_rows'])}; products: {n(t['n_products'])}; companies: {n(t['n_companies'])}; "
                          f"chemical names: {n(t['n_chemical_names'])}", t["n_rows"], r.call_id))
        ir = cov["initial_reported"]
        return (f"The dataset covers products first reported between {ir['min']} and {ir['max']} "
                f"({n(t['n_rows'])} rows, {n(t['n_products'])} products, {n(t['n_companies'])} companies)."), lines, ids

    if r.tool == "dq_summary":
        for row in r.rows:
            ids.append(fs.add(f"{row['issue_code']} in {row['column_name']}: {n(row['n_rows'])} rows "
                              f"({row['action_taken']})", row["n_rows"], r.call_id))
            lines.append(f"- {row['issue_code']} ({row['column_name']}): {n(row['n_rows'])} rows — "
                         f"{row['action_taken']}; e.g. rows {', '.join(map(str, row['sample_row_ids']))}")
        for k, v in t.items():
            ids.append(fs.add(f"{k.replace('_', ' ')}: {n(v)}", v, r.call_id))
            lines.append(f"- {k.replace('_', ' ')}: {n(v)}")
        top = ", ".join(f"{row['issue_code']} ({n(row['n_rows'])})" for row in r.rows[:3])
        return f"Main data-quality issues: {top}; plus {n(t['exact_duplicate_rows'])} exact duplicate rows.", lines, ids
    return "", lines, ids


def compare_section(results: list[ToolResult], fs: FactSheet) -> tuple[str, list[str], list[str]]:
    ids, lines, parts = [], [], []
    for r in results:
        label = next((x.split("=", 1)[1] for x in r.notes if x.startswith("compare_item=")), r.description)
        v = r.totals.get("n_products", 0)
        ids.append(fs.add(f"{label}: {n(v)} products ({r.description})", v, r.call_id))
        parts.append(f"{label}: {n(v)} products")
        lines.append(f"- **{label}** — {n(v)} products ({r.description})")
        if r.tool == "chemicals_for":
            for row in r.rows[:5]:
                lines.append(f"  - {row['chemical']}: {n(row['n_products'])}")
    return "Comparison — " + "; ".join(parts) + ".", lines, ids


def collect_evidence(state: TurnState, ctx: AgentContext, support: dict[str, list[str]]) -> list[EvidenceItem]:
    per_result = [(r.call_id, list(r.evidence_row_ids)) for r in state.results if r.evidence_row_ids]
    ordered: list[tuple[int, str]] = []
    seen: set[int] = set()
    while any(rows for _, rows in per_result) and len(ordered) < ctx.settings.evidence_limit:
        for cid, rows in per_result:
            if rows:
                rid = rows.pop(0)
                if rid not in seen:
                    seen.add(rid)
                    ordered.append((rid, cid))
    rows = {r["row_id"]: r for r in ctx.engine.fetch_evidence([rid for rid, _ in ordered])}
    out = []
    for i, (rid, cid) in enumerate(ordered, 1):
        r = rows[rid]
        fields = {"ProductName": r["product_name"], "CompanyName": r["company_name"], "BrandName": r["brand_name"],
                  "PrimaryCategory": r["primary_category"], "SubCategory": r["subcategory"],
                  "ChemicalName": r["chemical_name"], "CasNumber": r["cas_number"],
                  "InitialDateReported": r["initial_reported"]}
        if r["cas_raw"] != r["cas_number"]:
            fields["CasNumber_raw"] = r["cas_raw"]
        if r["csf"]:
            fields["CSF"] = r["csf"]
        if r["discontinued_date"]:
            fields["DiscontinuedDate"] = r["discontinued_date"]
        if r["chem_removed_date_raw"]:
            fields["ChemicalDateRemoved"] = r["chem_removed_date"] or f"{r['chem_removed_date_raw']} (invalid)"
        out.append(EvidenceItem(ref=f"E{i}", row_id=rid, cdph_id=r["cdph_id"], csf_id=r["csf_id"],
                                chemical_id=r["chemical_id"], cas_id=r["cas_id"], fields=fields,
                                supports=support.get(cid, [])))
    return out


def llm_bullets(state: TurnState, facts: list[Fact], ctx: AgentContext) -> list[str]:
    payload = {
        "question": state.clean_question,
        "facts": [{"id": f.id, "statement": f.statement} for f in facts[:60]],
        "assumptions": state.assumptions,
        "warnings": [w.message for w in state.warnings if w.severity != "info"][:6],
    }
    import json

    out: LLMNarrative = ctx.llm.structured(prompt("synthesizer"), json.dumps(payload, default=str), LLMNarrative)
    return [b.strip() for b in out.bullets if b.strip()][:6]


def synthesizer_node(state: TurnState, ctx: AgentContext) -> dict:
    t = Timer()
    fs = FactSheet()
    shorts: list[str] = []
    sections: list[str] = []
    support: dict[str, list[str]] = {}
    warnings: list[WarningItem] = []
    plan = state.plan
    for sub in plan.subtasks if plan else []:
        res = [r for r in state.results if r.subtask_id == sub.id]
        if not res:
            continue
        if sub.intent == Intent.COMPARE and len(res) > 1:
            short, lines, ids = compare_section(res, fs)
            for r in res:
                support[r.call_id] = ids
        else:
            short, lines, ids = "", [], []
            for i, r in enumerate(res):
                s, ls, fids = facts_for(r, fs, sub)
                support.setdefault(r.call_id, []).extend(fids[:3])
                if s and (i == 0 or (sub.intent == Intent.SUMMARIZE and i < 3)):
                    short = f"{short} {s}".strip() if sub.intent == Intent.SUMMARIZE and i else (short or s)
                lines += ([""] if lines and ls else []) + ls
                ids += fids
        if short:
            shorts.append(short)
        if lines:
            header = f"**{sub.text}**" if len(plan.subtasks) > 1 else ""
            sections.append("\n".join(([header, ""] if header else []) + lines))

    response_type = state.response_type
    statuses = [r.status for r in state.results]
    if response_type == "answer" and statuses and all(s in ("not_found", "no_data_in_range") for s in statuses):
        response_type = "no_data"
    if plan and plan.scope == "medical_advice":
        response_type = "refusal"
        factual = (" Factually: " + shorts[0]) if shorts else ""
        shorts = [f"I can't provide health or safety advice. {SAFETY_NOTE}{factual}"]
        sections.insert(0, "For health questions, consult a healthcare provider or CDPH resources "
                           "(https://www.cdph.ca.gov/Programs/CCDPHP/DEODC/OHB/CSCP).")
    if state.response_type == "refusal" and not (plan and plan.scope == "medical_advice"):
        shorts = [state.refusal_reason or "I can only answer questions about the cosmetics chemical dataset."]
        sections = ["Try e.g. 'Which products contain CAS 75-07-0?' or 'What chemicals are reported for Sally Hansen?'"]

    template_details = "\n\n".join(s for s in sections if s.strip())
    details, mode = template_details, "template"
    facts = fs.facts
    bullets: list[str] = []
    if ctx.llm.available and response_type in ("answer", "no_data") and facts:
        try:
            bullets = llm_bullets(state, facts, ctx)
            if bullets:
                details = "\n".join(f"- {b}" for b in bullets) + ("\n\n" + template_details if template_details else "")
                mode = "llm"
        except LLMUnavailable as e:
            warnings.append(WarningItem(code="llm_unavailable", severity="info",
                                        message=f"Details written from templates: {e}"))
    evidence = collect_evidence(state, ctx, support) if response_type in ("answer", "refusal", "no_data") else []
    short = " ".join(shorts) or "No answer could be produced."
    draft = Draft(answer_short=short, answer_details=details, details_mode=mode)
    return {"facts": facts, "evidence": evidence, "draft": draft, "response_type": response_type,
            "warnings": warnings,
            "trace": [event("synthesizer", f"{len(facts)} facts, {len(evidence)} evidence rows, details={mode}", t,
                            mode=mode, template_details=template_details if mode == "llm" else None,
                            llm_bullets=bullets)]}
