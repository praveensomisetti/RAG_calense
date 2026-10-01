"""Deterministic, parameterised query tools over the read-only DuckDB database.

Every tool:
  * builds SQL from a fixed template + `compile_filters` (no LLM-written SQL, ever),
  * logs a ToolCall (exact SQL, bound params, row count, result hash) for the query plan / replay,
  * returns a ToolResult carrying authoritative totals and row_ids for citation.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any

import duckdb

from chemrag.query.filters import PRODUCT_DATE_FIELDS, Filters, compile_filters
from chemrag.schemas import ToolCall, ToolResult

GROUP_DIMENSIONS: dict[str, tuple[str, str, str]] = {
    # name: (key expression, label expression, join clause)
    "company": ("m.company_key", "c.company_name", "LEFT JOIN dim_company c ON c.company_key = m.company_key"),
    "brand": ("m.brand_key", "coalesce(b.brand_name, '(no brand)')", "LEFT JOIN dim_brand b ON b.brand_key = m.brand_key"),
    "chemical": ("m.chem_group_id", "g.group_name",
                 "LEFT JOIN dim_chemical_group g ON g.chem_group_id = m.chem_group_id"),
    "subcategory": ("m.subcategory_key", "s.subcategory",
                    "LEFT JOIN dim_subcategory s ON s.subcategory_key = m.subcategory_key"),
    "primary_category": ("m.primary_category_key", "pc.primary_category",
                         "LEFT JOIN dim_primary_category pc ON pc.primary_category_key = m.primary_category_key"),
}

EVIDENCE_COLUMNS = (
    "row_id, cdph_id, csf_id, chemical_id, cas_id, product_name, csf, company_name, brand_name, primary_category, "
    "subcategory, chemical_name, cas_number, cas_raw, cas_status, initial_reported, most_recent_reported, "
    "discontinued_date, chem_removed_date, chem_removed_date_raw"
)


def _jsonable(v: Any) -> Any:
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


class QueryError(RuntimeError):
    pass


class QueryEngine:
    def __init__(self, db_path: Path, memory_limit: str = "1GB", threads: int = 2, timeout_s: float = 10.0,
                 evidence_limit: int = 25):
        if not Path(db_path).exists():
            raise QueryError(f"Database not found at {db_path}. Run `chemrag build` first.")
        self.con = duckdb.connect(str(db_path), read_only=True)
        self.con.execute(f"SET memory_limit='{memory_limit}'")
        self.con.execute(f"SET threads={int(threads)}")
        self.con.execute("SET enable_external_access=false")
        self.timeout_s = timeout_s
        self.evidence_limit = evidence_limit
        self.calls: list[ToolCall] = []
        self._counter = 0
        self.meta = {k: json.loads(v) for k, v in self.con.execute("SELECT key, value FROM dataset_meta").fetchall()}

    # ------------------------------------------------------------------ plumbing
    def _query(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        timer = threading.Timer(self.timeout_s, self.con.interrupt)
        timer.start()
        try:
            cur = self.con.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, (_jsonable(v) for v in row))) for row in cur.fetchall()]
        except duckdb.InterruptException as e:  # pragma: no cover
            raise QueryError(f"query exceeded {self.timeout_s}s") from e
        finally:
            timer.cancel()

    def run_sql(self, tool: str, subtask_id: str, params: dict[str, Any], sql: str,
                bound: list[Any]) -> tuple[str, list[dict[str, Any]]]:
        t0 = time.perf_counter()
        rows = self._query(sql, bound)
        self._counter += 1
        call_id = f"q{self._counter}"
        digest = hashlib.sha1(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()
        self.calls.append(ToolCall(
            id=call_id, subtask_id=subtask_id, tool=tool, params=params, sql=" ".join(sql.split()),
            bound_params=[_jsonable(b) for b in bound], row_count=len(rows), result_sha1=digest,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
        ))
        return call_id, rows

    def drain_calls(self) -> list[ToolCall]:
        calls, self.calls = self.calls, []
        return calls

    def replay(self, call: ToolCall) -> tuple[bool, int]:
        rows = self._query(call.sql, call.bound_params)
        digest = hashlib.sha1(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()
        return digest == call.result_sha1, len(rows)

    def coverage(self, field: str) -> tuple[date, date]:
        c = self.meta["coverage"][field]
        return date.fromisoformat(c["min"]), date.fromisoformat(c["max"])

    # ------------------------------------------------------------------ tools
    def product_totals(self, f: Filters, subtask_id: str, tool: str = "count_products") -> tuple[str, dict]:
        where, bound = compile_filters(f)
        sql = f"""
            WITH m AS (SELECT f.* FROM fact_report f WHERE {where})
            SELECT count(DISTINCT m.cdph_id) AS n_products, count(*) AS n_rows,
                   count(DISTINCT m.chemical_id) AS n_report_records,
                   count(DISTINCT m.company_key) AS n_companies, count(DISTINCT m.brand_key) AS n_brands,
                   count(DISTINCT m.chem_group_id) AS n_chemicals,
                   count(DISTINCT CASE WHEN p.discontinued_date IS NOT NULL THEN m.cdph_id END) AS n_discontinued_products,
                   count(DISTINCT CASE WHEN m.chem_removed_date_raw IS NOT NULL THEN m.cdph_id END) AS n_products_chem_removed,
                   count(CASE WHEN m.cas_status IN ('repaired', 'from_casid') THEN 1 END) AS n_rows_cas_repaired
            FROM m JOIN dim_product p USING (cdph_id)"""
        call_id, rows = self.run_sql(tool, subtask_id, {"filters": f.describe()}, sql, bound)
        return call_id, rows[0]

    def evidence_sample(self, f: Filters, subtask_id: str, limit: int | None = None) -> list[int]:
        """Stratified evidence: one row per (product, chemical), round-robin across chemicals so the
        dominant chemical (titanium dioxide) cannot crowd out everything else."""
        where, bound = compile_filters(f)
        sql = f"""
            WITH m AS (SELECT f.* FROM fact_report f WHERE {where}),
                 one AS (SELECT row_id, cdph_id, chem_group_id FROM m
                         QUALIFY row_number() OVER (PARTITION BY cdph_id, chem_group_id ORDER BY row_id) = 1)
            SELECT row_id FROM one
            ORDER BY row_number() OVER (PARTITION BY chem_group_id ORDER BY cdph_id), chem_group_id
            LIMIT ?"""
        _, rows = self.run_sql("evidence_sample", subtask_id, {"filters": f.describe()}, sql,
                               bound + [limit or self.evidence_limit])
        return [r["row_id"] for r in rows]

    def find_products(self, f: Filters, subtask_id: str, limit: int = 20, offset: int = 0) -> ToolResult:
        totals_id, totals = self.product_totals(f, subtask_id, tool="find_products.totals")
        where, bound = compile_filters(f)
        sql = f"""
            WITH m AS (SELECT f.* FROM fact_report f WHERE {where})
            SELECT m.cdph_id, any_value(p.product_name) AS product_name, any_value(p.brand_name) AS brand_name,
                   any_value(p.company_name) AS company_name, any_value(p.discontinued_date) AS discontinued_date,
                   list_sort(list_distinct(list(m.chemical_name))) AS chemicals,
                   count(*) AS n_rows, bool_or(m.chem_removed_date_raw IS NOT NULL) AS chemical_removed,
                   min(m.row_id) AS first_row_id
            FROM m JOIN dim_product p USING (cdph_id)
            GROUP BY m.cdph_id
            ORDER BY product_name, m.cdph_id
            LIMIT ? OFFSET ?"""
        call_id, rows = self.run_sql("find_products", subtask_id,
                                     {"filters": f.describe(), "limit": limit, "offset": offset},
                                     sql, bound + [limit, offset])
        evidence = [r["first_row_id"] for r in rows][: self.evidence_limit]
        n = totals["n_products"]
        return ToolResult(
            call_id=call_id, subtask_id=subtask_id, tool="find_products", description=f.describe(), rows=rows,
            totals=totals, evidence_row_ids=evidence, truncated=n > offset + len(rows),
            status="ok" if n else "empty", notes=[f"totals from {totals_id}"],
        )

    def count_products(self, f: Filters, subtask_id: str, group_by: str | None = None,
                       limit: int = 20) -> ToolResult:
        totals_id, totals = self.product_totals(f, subtask_id)
        if not group_by:
            ev = self.evidence_sample(f, subtask_id) if totals["n_products"] else []
            return ToolResult(call_id=totals_id, subtask_id=subtask_id, tool="count_products",
                              description=f.describe(), totals=totals, evidence_row_ids=ev,
                              status="ok" if totals["n_products"] else "empty")
        key, label, join = GROUP_DIMENSIONS[group_by]
        where, bound = compile_filters(f, exclude_trade_secret=group_by == "chemical")
        sql = f"""
            WITH m AS (SELECT f.* FROM fact_report f WHERE {where})
            SELECT {key} AS key, any_value({label}) AS label, count(DISTINCT m.cdph_id) AS n_products,
                   count(*) AS n_rows, min(m.row_id) AS sample_row_id, count(*) OVER () AS n_groups
            FROM m {join}
            GROUP BY {key}
            ORDER BY n_products DESC, label
            LIMIT ?"""
        call_id, rows = self.run_sql("count_products", subtask_id,
                                     {"filters": f.describe(), "group_by": group_by, "limit": limit},
                                     sql, bound + [limit])
        n_groups = rows[0]["n_groups"] if rows else 0
        for r in rows:
            r.pop("n_groups", None)
        totals["n_groups"] = n_groups
        return ToolResult(
            call_id=call_id, subtask_id=subtask_id, tool="count_products", description=f.describe(), rows=rows,
            totals=totals, evidence_row_ids=[r["sample_row_id"] for r in rows][: self.evidence_limit],
            truncated=n_groups > len(rows), status="ok" if rows else "empty",
            notes=[f"group_by={group_by}", f"totals from {totals_id}"],
        )

    def chemicals_for(self, f: Filters, subtask_id: str, limit: int = 50) -> ToolResult:
        totals_id, totals = self.product_totals(f, subtask_id, tool="chemicals_for.totals")
        where, bound = compile_filters(f, exclude_trade_secret=True)
        sql = f"""
            WITH m AS (SELECT f.* FROM fact_report f WHERE {where})
            SELECT m.chem_group_id, any_value(g.group_name) AS chemical,
                   list_sort(list_distinct(list(m.chemical_name))) AS reported_names,
                   list_sort(list_distinct(list(m.cas_number) FILTER (WHERE m.cas_number IS NOT NULL))) AS cas_numbers,
                   count(DISTINCT m.cdph_id) AS n_products,
                   count(DISTINCT CASE WHEN m.chem_removed_date_raw IS NOT NULL THEN m.cdph_id END) AS n_products_removed,
                   count(*) AS n_rows, min(m.row_id) AS sample_row_id, count(*) OVER () AS n_groups
            FROM m JOIN dim_chemical_group g ON g.chem_group_id = m.chem_group_id
            GROUP BY m.chem_group_id
            ORDER BY n_products DESC, chemical
            LIMIT ?"""
        call_id, rows = self.run_sql("chemicals_for", subtask_id, {"filters": f.describe(), "limit": limit},
                                     sql, bound + [limit])
        n_groups = rows[0]["n_groups"] if rows else 0
        for r in rows:
            r.pop("n_groups", None)
        totals["n_chemicals"] = n_groups
        notes = [f"totals from {totals_id}"]
        if not f.include_trade_secret and not f.chem_group_ids:
            ts_where, ts_bound = compile_filters(f)
            ts_sql = f"""SELECT count(DISTINCT f.cdph_id) AS n_products, min(f.row_id) AS sample_row_id
                         FROM fact_report f WHERE {ts_where} AND f.is_trade_secret"""
            _, ts_rows = self.run_sql("trade_secret_count", subtask_id, {"filters": f.describe()}, ts_sql, ts_bound)
            totals["n_products_trade_secret"] = ts_rows[0]["n_products"]
            totals["trade_secret_sample_row_id"] = ts_rows[0]["sample_row_id"]
        return ToolResult(
            call_id=call_id, subtask_id=subtask_id, tool="chemicals_for", description=f.describe(), rows=rows,
            totals=totals, evidence_row_ids=[r["sample_row_id"] for r in rows][: self.evidence_limit],
            truncated=n_groups > len(rows), status="ok" if rows else "empty", notes=notes,
        )

    def trend_by_year(self, f: Filters, subtask_id: str, date_field: str = "initial_reported") -> ToolResult:
        totals_id, totals = self.product_totals(f, subtask_id, tool="trend_by_year.totals")
        where, bound = compile_filters(f)
        col = f"p.{date_field}" if date_field in PRODUCT_DATE_FIELDS else "m.chem_removed_date"
        sql = f"""
            WITH m AS (SELECT f.* FROM fact_report f WHERE {where})
            SELECT year({col}) AS year, count(DISTINCT m.cdph_id) AS n_products, count(*) AS n_rows,
                   min(m.row_id) AS sample_row_id
            FROM m JOIN dim_product p USING (cdph_id)
            WHERE {col} IS NOT NULL
            GROUP BY 1 ORDER BY 1"""
        call_id, rows = self.run_sql("trend_by_year", subtask_id,
                                     {"filters": f.describe(), "date_field": date_field}, sql, bound)
        lo, hi = self.coverage(date_field)
        by_year = {r["year"]: r for r in rows}
        filled = [by_year.get(y, {"year": y, "n_products": 0, "n_rows": 0, "sample_row_id": None})
                  for y in range(lo.year, hi.year + 1)]
        totals["date_field"] = date_field
        totals["n_products_with_date"] = sum(r["n_products"] for r in rows)
        return ToolResult(
            call_id=call_id, subtask_id=subtask_id, tool="trend_by_year", description=f.describe(), rows=filled,
            totals=totals, evidence_row_ids=[r["sample_row_id"] for r in rows if r["sample_row_id"]][: self.evidence_limit],
            status="ok" if rows else "empty", notes=[f"totals from {totals_id}", f"years {lo.year}-{hi.year}"],
        )

    def dataset_coverage(self, subtask_id: str) -> ToolResult:
        sql = "SELECT key, value FROM dataset_meta WHERE key IN ('coverage', 'n_rows', 'n_products', 'n_companies', 'n_brands', 'n_chemical_names', 'n_chemical_groups', 'n_exact_dup_rows') ORDER BY key"
        call_id, rows = self.run_sql("dataset_coverage", subtask_id, {}, sql, [])
        data = {r["key"]: json.loads(r["value"]) for r in rows}
        return ToolResult(call_id=call_id, subtask_id=subtask_id, tool="dataset_coverage",
                          description="dataset coverage", rows=[data], totals=data)

    def dq_summary(self, subtask_id: str) -> ToolResult:
        sql = """SELECT issue_code, column_name, count(*) AS n_rows, any_value(action_taken) AS action_taken,
                        list(row_id ORDER BY row_id)[1:3] AS sample_row_ids
                 FROM dq_issues GROUP BY issue_code, column_name ORDER BY n_rows DESC"""
        call_id, rows = self.run_sql("dq_summary", subtask_id, {}, sql, [])
        extra_sql = """SELECT count(*) FILTER (WHERE is_exact_dup) AS exact_duplicate_rows,
                              count(*) FILTER (WHERE cas_raw IS NULL) AS rows_cas_null_in_source,
                              count(*) FILTER (WHERE cas_number IS NULL AND NOT is_trade_secret) AS rows_without_valid_cas,
                              count(*) FILTER (WHERE is_trade_secret) AS trade_secret_rows
                       FROM fact_report"""
        _, extra = self.run_sql("dq_summary.totals", subtask_id, {}, extra_sql, [])
        ev = [rid for r in rows for rid in r["sample_row_ids"][:1]][: self.evidence_limit]
        return ToolResult(call_id=call_id, subtask_id=subtask_id, tool="dq_summary", description="data quality",
                          rows=rows, totals=extra[0], evidence_row_ids=ev)

    def fetch_evidence(self, row_ids: list[int]) -> list[dict[str, Any]]:
        if not row_ids:
            return []
        marks = ", ".join("?" for _ in row_ids)
        rows = self._query(f"SELECT {EVIDENCE_COLUMNS} FROM fact_report WHERE row_id IN ({marks})", list(row_ids))
        order = {rid: i for i, rid in enumerate(row_ids)}
        return sorted(rows, key=lambda r: order[r["row_id"]])

    def family_relatives(self, chem_group_ids: list[int]) -> list[dict[str, Any]]:
        """Other groups in the same family (e.g. Retinyl palmitate when asking about Retinol)."""
        if not chem_group_ids:
            return []
        marks = ", ".join("?" for _ in chem_group_ids)
        sql = f"""SELECT g2.chem_group_id, g2.group_name, g2.family_label, g2.n_products
                  FROM dim_chemical_group g1 JOIN dim_chemical_group g2 ON g1.family = g2.family
                  WHERE g1.chem_group_id IN ({marks}) AND g2.chem_group_id NOT IN ({marks})
                  ORDER BY g2.n_products DESC"""
        return self._query(sql, list(chem_group_ids) * 2)

    def group_members(self, chem_group_ids: list[int]) -> list[dict[str, Any]]:
        marks = ", ".join("?" for _ in chem_group_ids)
        return self._query(f"SELECT chem_group_id, group_name, member_names, cas_numbers, n_products "
                           f"FROM dim_chemical_group WHERE chem_group_id IN ({marks})", list(chem_group_ids))

    def table(self, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        """Internal read helper for resolver indexes (fixed SQL only, not exposed to agents' inputs)."""
        return self._query(sql, params or [])

    def close(self) -> None:
        self.con.close()
