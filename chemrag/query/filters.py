"""Typed filters and their compilation into a whitelisted, parameterised SQL WHERE clause.

Values are always bound as `?` parameters. Column names come only from the fixed fragments below,
never from user text or LLM output.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel

from chemrag.schemas import DateConstraint

# Product-level date fields live on dim_product; the removal date is per product-chemical row.
PRODUCT_DATE_FIELDS = {"initial_reported", "most_recent_reported", "discontinued_date"}
ROW_DATE_FIELDS = {"chem_removed_date"}


class Filters(BaseModel):
    chem_group_ids: list[int] = []
    cas_numbers: list[str] = []
    company_keys: list[int] = []
    brand_keys: list[int] = []
    cdph_ids: list[int] = []
    primary_category_keys: list[int] = []
    subcategory_keys: list[int] = []
    dates: list[DateConstraint] = []
    discontinued: bool | None = None
    chem_removed: bool | None = None
    exclude_chem_group_ids: list[int] = []
    include_trade_secret: bool = False
    # Display labels only (for descriptions); never used in SQL.
    labels: dict[str, list[str]] = {}

    def is_empty(self) -> bool:
        return not any([self.chem_group_ids, self.cas_numbers, self.company_keys, self.brand_keys, self.cdph_ids,
                        self.primary_category_keys, self.subcategory_keys, self.dates,
                        self.discontinued is not None, self.chem_removed is not None])

    def describe(self) -> str:
        """Natural-language description, e.g. 'containing Acetaldehyde (CAS 75-07-0) from Revlon, discontinued in 2015'."""
        lab = self.labels
        parts: list[str] = []
        if lab.get("chemical") or lab.get("cas"):
            parts.append("containing " + " or ".join(lab.get("chemical", []) + lab.get("cas", [])))
        if lab.get("product"):
            parts.append("named " + " or ".join(lab["product"]))
        if lab.get("brand"):
            parts.append("of brand " + " or ".join(lab["brand"]))
        if lab.get("company"):
            parts.append("from " + " or ".join(lab["company"]))
        cats = lab.get("subcategory", []) + lab.get("primary_category", [])
        if cats:
            parts.append("in " + " / ".join(cats))
        date_fields = {d.field for d in self.dates}
        if self.discontinued is True and "discontinued_date" not in date_fields:
            parts.append("that are discontinued")
        if self.discontinued is False:
            parts.append("that are not discontinued")
        if self.chem_removed is True and "chem_removed_date" not in date_fields:
            parts.append("where the chemical was later removed")
        if self.chem_removed is False:
            parts.append("where the chemical was not removed")
        verbs = {"initial_reported": "first reported", "most_recent_reported": "most recently reported",
                 "discontinued_date": "discontinued", "chem_removed_date": "with the chemical removed"}
        for d in self.dates:
            if d.start and d.end and d.start.year == d.end.year and d.start.month == 1 and d.end.month == 12:
                rng = f"in {d.start.year}"
            elif d.start and d.end:
                rng = f"between {d.start} and {d.end}"
            elif d.start:
                rng = f"on or after {d.start}"
            else:
                rng = f"on or before {d.end}"
            parts.append(f"{verbs[d.field]} {rng}")
        if lab.get("exclude"):
            parts.append("excluding " + " / ".join(lab["exclude"]))
        out = ""
        for part in parts:
            out += (" " if part.startswith(("that ", "where ")) else ", ") + part if out else part
        return out or "in the whole dataset"

    def without(self, *fields: str) -> Filters:
        """Copy with some constraints dropped (used for empty-result diagnostics)."""
        data = self.model_dump()
        defaults = Filters().model_dump()
        for f in fields:
            data[f] = defaults[f]
        return Filters(**data)


def _in(column: str, values: list[Any], params: list[Any], negate: bool = False) -> str:
    params.extend(values)
    marks = ", ".join("?" for _ in values)
    return f"{column} {'NOT IN' if negate else 'IN'} ({marks})"


def _date_range(column: str, d: DateConstraint, params: list[Any]) -> str:
    preds = [f"{column} IS NOT NULL"]
    if d.start:
        preds.append(f"{column} >= ?")
        params.append(d.start)
    if d.end:
        preds.append(f"{column} <= ?")
        params.append(d.end)
    return " AND ".join(preds)


def compile_filters(f: Filters, alias: str = "f") -> tuple[str, list[Any]]:
    """Return `(where_sql, params)` selecting non-duplicate fact rows that satisfy all constraints.

    Row-level constraints (chemical, CAS, removal) select *which rows* are evidence. Product-level
    constraints (category, product dates, discontinued) are applied as `cdph_id IN (...)` so a
    product matches even when its category and chemical facts live on different rows.
    """
    p: list[Any] = []
    w: list[str] = [f"NOT {alias}.is_exact_dup"]
    asks_trade_secret = f.include_trade_secret
    if not asks_trade_secret:
        w.append(f"NOT {alias}.is_trade_secret")
    if f.chem_group_ids:
        w.append(_in(f"{alias}.chem_group_id", f.chem_group_ids, p))
    if f.cas_numbers:
        w.append(_in(f"{alias}.cas_number", f.cas_numbers, p))
    if f.exclude_chem_group_ids:
        w.append(_in(f"{alias}.chem_group_id", f.exclude_chem_group_ids, p, negate=True))
    if f.company_keys:
        w.append(_in(f"{alias}.company_key", f.company_keys, p))
    if f.brand_keys:
        w.append(_in(f"{alias}.brand_key", f.brand_keys, p))
    if f.cdph_ids:
        w.append(_in(f"{alias}.cdph_id", f.cdph_ids, p))
    if f.chem_removed is True:
        # Raw value, so rows whose removal date was invalid (e.g. year 2103) still count as removed.
        w.append(f"{alias}.chem_removed_date_raw IS NOT NULL")
    elif f.chem_removed is False:
        w.append(f"{alias}.chem_removed_date_raw IS NULL")
    if f.primary_category_keys:
        sub: list[Any] = []
        cond = _in("primary_category_key", f.primary_category_keys, sub)
        w.append(f"{alias}.cdph_id IN (SELECT cdph_id FROM fact_report WHERE NOT is_exact_dup AND {cond})")
        p.extend(sub)
    if f.subcategory_keys:
        sub = []
        cond = _in("subcategory_key", f.subcategory_keys, sub)
        w.append(f"{alias}.cdph_id IN (SELECT cdph_id FROM fact_report WHERE NOT is_exact_dup AND {cond})")
        p.extend(sub)
    if f.discontinued is True:
        w.append(f"{alias}.cdph_id IN (SELECT cdph_id FROM dim_product WHERE discontinued_date IS NOT NULL)")
    elif f.discontinued is False:
        w.append(f"{alias}.cdph_id IN (SELECT cdph_id FROM dim_product WHERE discontinued_date IS NULL)")
    for d in f.dates:
        if d.field in PRODUCT_DATE_FIELDS:
            sub = []
            cond = _date_range(d.field, d, sub)
            w.append(f"{alias}.cdph_id IN (SELECT cdph_id FROM dim_product WHERE {cond})")
            p.extend(sub)
        elif d.field in ROW_DATE_FIELDS:
            w.append(_date_range(f"{alias}.{d.field}", d, p))
        else:  # pragma: no cover - guarded by the DateField Literal
            raise ValueError(f"unsupported date field {d.field}")
    return " AND ".join(w), p


def year_range(start_year: int | None, end_year: int | None) -> tuple[date | None, date | None]:
    return (date(start_year, 1, 1) if start_year else None, date(end_year, 12, 31) if end_year else None)
