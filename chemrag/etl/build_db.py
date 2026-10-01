"""ETL: raw CSV -> cleaned, typed DuckDB with canonical dimension tables.

Tables written (see PLAN.md §2):
  raw_rows, fact_report, dim_product, dim_chemical, dim_chemical_group, dim_company, dim_brand,
  dim_primary_category, dim_subcategory, entity_alias, dq_issues, dataset_meta
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import yaml

from chemrag.etl.cas import normalize_cas
from chemrag.etl.normalize import clean_text, company_match_key, norm_key

SOURCE_COLUMNS = [
    "CDPHId", "ProductName", "CSFId", "CSF", "CompanyId", "CompanyName", "BrandName", "PrimaryCategoryId",
    "PrimaryCategory", "SubCategoryId", "SubCategory", "CasId", "CasNumber", "ChemicalId", "ChemicalName",
    "InitialDateReported", "MostRecentDateReported", "DiscontinuedDate", "ChemicalCreatedAt",
    "ChemicalUpdatedAt", "ChemicalDateRemoved", "ChemicalCount",
]
DATE_COLUMNS = {
    "InitialDateReported": "initial_reported",
    "MostRecentDateReported": "most_recent_reported",
    "DiscontinuedDate": "discontinued_date",
    "ChemicalCreatedAt": "chem_created_at",
    "ChemicalUpdatedAt": "chem_updated_at",
    "ChemicalDateRemoved": "chem_removed_date",
}
MIN_VALID_DATE = pd.Timestamp("2000-01-01")
TRADE_SECRET = "Trade Secret"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _most_common(values: pd.Series) -> str:
    return Counter(v for v in values if v is not None).most_common(1)[0][0]


def _load_groups(path: Path, chemical_names: set[str]) -> tuple[dict[str, dict], dict, dict]:
    cfg = yaml.safe_load(path.read_text())
    families: dict[str, str] = cfg.get("families", {})
    groups: dict[str, dict] = {}  # member name -> group spec
    for g in cfg.get("groups", []):
        missing = [m for m in g["members"] if m not in chemical_names]
        if missing:
            raise ValueError(f"chemical_groups.yaml: unknown chemical names {missing} in group {g['name']!r}")
        fam = g.get("family")
        if fam and fam not in families:
            raise ValueError(f"chemical_groups.yaml: unknown family {fam!r}")
        for m in g["members"]:
            if m in groups:
                raise ValueError(f"chemical_groups.yaml: {m!r} is in two groups")
            groups[m] = {"name": g["name"], "family": fam, "synonyms": g.get("synonyms", [])}
    for name, syns in cfg.get("singleton_synonyms", {}).items():
        if name not in chemical_names:
            raise ValueError(f"chemical_groups.yaml: unknown singleton {name!r}")
        if name in groups:
            raise ValueError(f"chemical_groups.yaml: {name!r} is both grouped and a singleton")
        groups[name] = {"name": name, "family": None, "synonyms": syns}
    for name in chemical_names - set(groups):
        groups[name] = {"name": name, "family": None, "synonyms": []}
    return groups, families, cfg.get("family_synonyms", {})


def _category_short_forms(name: str) -> list[str]:
    """'Makeup Products (non-permanent)' -> ['Makeup Products', 'Makeup'] (matching aliases only)."""
    base = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()
    out = [base] if base != name else []
    no_products = re.sub(r"\s+Products$", "", base).strip()
    if no_products != base and len(no_products) >= 4:
        out.append(no_products)
    return out


def build_database(csv_path: Path, db_path: Path, groups_path: Path, log=print) -> dict:
    t0 = time.time()
    sha = file_sha256(csv_path)
    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[""])
    if list(raw.columns) != SOURCE_COLUMNS:
        raise ValueError(f"Unexpected CSV columns: {list(raw.columns)}")
    raw.insert(0, "row_id", range(1, len(raw) + 1))
    log(f"read {len(raw):,} rows ({time.time() - t0:.1f}s)")

    dq: list[tuple[int, str, str, str | None, str]] = []  # row_id, column, issue, raw, action
    df = pd.DataFrame({"row_id": raw["row_id"]})
    for col in ["CDPHId", "CSFId", "CompanyId", "PrimaryCategoryId", "SubCategoryId", "CasId", "ChemicalId",
                "ChemicalCount"]:
        df[col] = pd.to_numeric(raw[col], errors="coerce").astype("Int64")
    text_cols = ["ProductName", "CSF", "CompanyName", "BrandName", "PrimaryCategory", "SubCategory", "ChemicalName"]
    for col in text_cols:
        df[col] = raw[col].map(clean_text)
    placeholder = raw["BrandName"].notna() & df["BrandName"].isna()
    for rid, val in zip(raw.loc[placeholder, "row_id"], raw.loc[placeholder, "BrandName"]):
        dq.append((rid, "BrandName", "placeholder_null", val, "set NULL"))
    ws = raw["SubCategory"].notna() & (raw["SubCategory"] != raw["SubCategory"].str.strip())
    log(f"text cleaned: {int(placeholder.sum())} brand placeholders -> NULL, {int(ws.sum())} subcategory values trimmed")

    # ---- dates ---------------------------------------------------------------------------------
    parsed = {c: pd.to_datetime(raw[c], format="%m/%d/%Y", errors="coerce") for c in DATE_COLUMNS}
    max_valid = parsed["ChemicalUpdatedAt"].max() + timedelta(days=365)
    for src, dst in DATE_COLUMNS.items():
        p = parsed[src]
        unparsable = raw[src].notna() & p.isna()
        for rid, val in zip(raw.loc[unparsable, "row_id"], raw.loc[unparsable, src]):
            dq.append((rid, src, "unparsable_date", val, "set NULL"))
        bad = p.notna() & ((p > max_valid) | (p < MIN_VALID_DATE))
        issue = "removed_date_future" if src == "ChemicalDateRemoved" else "date_out_of_range"
        for rid, val in zip(raw.loc[bad, "row_id"], raw.loc[bad, src]):
            dq.append((rid, src, issue, val, "set NULL (raw kept)"))
        df[dst] = p.where(~bad).dt.date
    df["chem_removed_date_raw"] = raw["ChemicalDateRemoved"]
    disc_before = parsed["DiscontinuedDate"].notna() & (parsed["DiscontinuedDate"] < parsed["InitialDateReported"])
    rem_before = df["chem_removed_date"].notna() & (parsed["ChemicalDateRemoved"] < parsed["ChemicalCreatedAt"])
    for rid in raw.loc[disc_before, "row_id"]:
        dq.append((rid, "DiscontinuedDate", "discontinued_before_initial", None, "kept, flagged"))
    for rid in raw.loc[rem_before, "row_id"]:
        dq.append((rid, "ChemicalDateRemoved", "removed_before_created", None, "kept, flagged"))

    # ---- CAS -----------------------------------------------------------------------------------
    norm = raw["CasNumber"].map(normalize_cas)
    df["cas_raw"] = raw["CasNumber"]
    df["cas_number"] = norm.map(lambda t: t[0])
    df["cas_status"] = norm.map(lambda t: t[1])
    valid_by_casid = (
        df[df["cas_status"].isin(["valid", "repaired"])].groupby("CasId")["cas_number"].agg(lambda s: sorted(set(s)))
    )
    unique_by_casid = {k: v[0] for k, v in valid_by_casid.items() if len(v) == 1}
    fill = df["cas_status"].isin(["invalid", "missing"]) & df["CasId"].map(lambda c: c in unique_by_casid)
    df.loc[fill, "cas_number"] = df.loc[fill, "CasId"].map(unique_by_casid)
    df.loc[fill, "cas_status"] = "from_casid"
    for rid, val, st in zip(df["row_id"], df["cas_raw"], df["cas_status"]):
        if st in ("repaired", "from_casid", "invalid"):
            dq.append((rid, "CasNumber", f"cas_{st}", val, {"repaired": "normalised", "from_casid":
                       "filled from CasId", "invalid": "set NULL"}[st]))
    log("CAS status: " + json.dumps(df["cas_status"].value_counts().to_dict()))

    df["is_trade_secret"] = df["ChemicalName"].eq(TRADE_SECRET)
    for rid in df.loc[df["CompanyName"].str.casefold().isin(["test", "testing"]), "row_id"]:
        dq.append((rid, "CompanyName", "suspect_test_record", "Test", "kept, flagged"))
    df["is_exact_dup"] = raw[SOURCE_COLUMNS].duplicated(keep="first")

    # ---- chemicals -----------------------------------------------------------------------------
    names = set(df["ChemicalName"].dropna())
    groups, families, family_syns = _load_groups(groups_path, names)
    group_names = sorted({g["name"] for g in groups.values()}, key=str.casefold)
    group_id = {n: i + 1 for i, n in enumerate(group_names)}
    df["chem_group_id"] = df["ChemicalName"].map(lambda n: group_id[groups[n]["name"]])

    nd = df[~df["is_exact_dup"]]
    grp_rows = []
    for gname, gid in group_id.items():
        sub = nd[nd["chem_group_id"] == gid]
        fam = next(g["family"] for g in groups.values() if g["name"] == gname)
        grp_rows.append({
            "chem_group_id": gid, "group_name": gname, "family": fam, "family_label": families.get(fam),
            "member_names": sorted(set(sub["ChemicalName"])),
            "cas_numbers": sorted(set(sub["cas_number"].dropna())),
            "n_products": int(sub["CDPHId"].nunique()), "n_rows": len(sub),
            "is_trade_secret": gname == TRADE_SECRET,
        })
    dim_group = pd.DataFrame(grp_rows)
    dim_chem = (
        nd.groupby("CasId").agg(
            chemical_name=("ChemicalName", "first"), chem_group_id=("chem_group_id", "first"),
            cas_numbers=("cas_number", lambda s: sorted(set(s.dropna()))), n_products=("CDPHId", "nunique"),
        ).reset_index().rename(columns={"CasId": "cas_id"})
    )

    # ---- companies / brands / categories --------------------------------------------------------
    df["company_norm"] = df["CompanyName"].map(norm_key)
    comp_keys = {k: i + 1 for i, k in enumerate(sorted(set(df["company_norm"])))}
    df["company_key"] = df["company_norm"].map(comp_keys)
    dim_company = nd.assign(company_key=df["company_key"]).groupby("company_key").agg(
        company_name=("CompanyName", _most_common), company_ids=("CompanyId", lambda s: sorted(set(int(x) for x in s))),
        n_products=("CDPHId", "nunique"), n_brands=("BrandName", "nunique"),
    ).reset_index()
    dim_company["match_key"] = dim_company["company_name"].map(company_match_key)

    df["brand_norm"] = df["BrandName"].map(lambda b: norm_key(b) or None)
    brand_keys = {k: i + 1 for i, k in enumerate(sorted(set(df["brand_norm"].dropna())))}
    df["brand_key"] = df["brand_norm"].map(brand_keys).astype("Int64")
    dim_brand = df[~df["is_exact_dup"] & df["brand_key"].notna()].groupby("brand_key").agg(
        brand_name=("BrandName", _most_common), company_keys=("company_key", lambda s: sorted(set(int(x) for x in s))),
        n_products=("CDPHId", "nunique"),
    ).reset_index()
    dim_brand["brand_key"] = dim_brand["brand_key"].astype(int)

    for col, key in [("PrimaryCategory", "primary_category_key"), ("SubCategory", "subcategory_key")]:
        keys = {k: i + 1 for i, k in enumerate(sorted(set(df[col].map(norm_key))))}
        df[key] = df[col].map(norm_key).map(keys)
    dim_primary = df[~df["is_exact_dup"]].groupby("primary_category_key").agg(
        primary_category=("PrimaryCategory", _most_common),
        source_ids=("PrimaryCategoryId", lambda s: sorted(set(int(x) for x in s))),
        n_products=("CDPHId", "nunique"),
    ).reset_index()
    dim_sub = df[~df["is_exact_dup"]].groupby("subcategory_key").agg(
        subcategory=("SubCategory", _most_common),
        source_ids=("SubCategoryId", lambda s: sorted(set(int(x) for x in s))),
        primary_categories=("PrimaryCategory", lambda s: sorted(set(s))),
        n_products=("CDPHId", "nunique"),
    ).reset_index()

    # ---- products ------------------------------------------------------------------------------
    nd = df[~df["is_exact_dup"]]
    dim_product = nd.groupby("CDPHId").agg(
        product_name=("ProductName", "first"), company_key=("company_key", "first"),
        company_name=("CompanyName", "first"), brand_key=("brand_key", "first"), brand_name=("BrandName", "first"),
        initial_reported=("initial_reported", "min"), most_recent_reported=("most_recent_reported", "max"),
        discontinued_date=("discontinued_date", "max"),
        subcategory_keys=("subcategory_key", lambda s: sorted(set(int(x) for x in s))),
        n_csf=("CSFId", "nunique"),
        has_removed_chemical=("chem_removed_date", lambda s: bool(s.notna().any())),
    ).reset_index().rename(columns={"CDPHId": "cdph_id"})
    n_chem = nd[~nd["is_trade_secret"]].groupby("CDPHId")["chem_group_id"].nunique()
    dim_product["n_chemicals"] = dim_product["cdph_id"].map(n_chem).fillna(0).astype(int)
    dim_product["company_name"] = dim_product["company_key"].map(dim_company.set_index("company_key")["company_name"])
    dim_product["brand_name"] = dim_product["brand_key"].map(dim_brand.set_index("brand_key")["brand_name"])
    dim_product["product_norm"] = dim_product["product_name"].map(norm_key)

    # ---- unified alias table for the resolver / vector index -----------------------------------
    alias_rows: list[dict] = []

    def add(etype: str, cid, display: str, alias: str, kind: str, n: int, extra: dict | None = None):
        key = norm_key(alias) if kind != "cas" else alias
        if key:
            alias_rows.append({"entity_type": etype, "canonical_id": None if cid is None else int(cid),
                               "display_name": display, "alias": alias, "alias_norm": key, "kind": kind,
                               "n_products": int(n), "extra": json.dumps(extra or {})})

    for r in dim_group.itertuples():
        add("chemical", r.chem_group_id, r.group_name, r.group_name, "name", r.n_products)
        for m in r.member_names:
            add("chemical", r.chem_group_id, r.group_name, m, "member", r.n_products)
        for c in r.cas_numbers:
            add("chemical", r.chem_group_id, r.group_name, c, "cas", r.n_products)
        syns = next((g["synonyms"] for g in groups.values() if g["name"] == r.group_name), [])
        for s in syns:
            add("chemical", r.chem_group_id, r.group_name, s, "synonym", r.n_products)
    fam_products = {f: int(nd[nd["chem_group_id"].isin(dim_group.loc[dim_group["family"] == f, "chem_group_id"])]
                           ["CDPHId"].nunique()) for f in families}
    for fam, label in families.items():
        for s in [label] + family_syns.get(fam, []):
            add("chemical_family", None, label, s, "family", fam_products[fam], {"family": fam})
    for r in dim_company.itertuples():
        add("company", r.company_key, r.company_name, r.company_name, "name", r.n_products)
        if r.match_key and r.match_key != norm_key(r.company_name):
            add("company", r.company_key, r.company_name, r.match_key, "match_key", r.n_products)
    for r in dim_brand.itertuples():
        add("brand", r.brand_key, r.brand_name, r.brand_name, "name", r.n_products,
            {"company_keys": r.company_keys})
    for r in dim_primary.itertuples():
        add("primary_category", r.primary_category_key, r.primary_category, r.primary_category, "name", r.n_products)
        for short in _category_short_forms(r.primary_category):
            add("primary_category", r.primary_category_key, r.primary_category, short, "short_name", r.n_products)
    for r in dim_sub.itertuples():
        add("subcategory", r.subcategory_key, r.subcategory, r.subcategory, "name", r.n_products,
            {"primary_categories": r.primary_categories})
        for short in _category_short_forms(r.subcategory):
            add("subcategory", r.subcategory_key, r.subcategory, short, "short_name", r.n_products,
                {"primary_categories": r.primary_categories})
    entity_alias = pd.DataFrame(alias_rows).drop_duplicates(["entity_type", "canonical_id", "alias_norm", "kind"])

    # ---- fact table ----------------------------------------------------------------------------
    fact = pd.DataFrame({
        "row_id": df["row_id"], "cdph_id": df["CDPHId"], "csf_id": df["CSFId"], "chemical_id": df["ChemicalId"],
        "cas_id": df["CasId"], "company_id": df["CompanyId"], "primary_category_id": df["PrimaryCategoryId"],
        "subcategory_id": df["SubCategoryId"], "product_name": df["ProductName"], "csf": df["CSF"],
        "company_name": df["CompanyName"], "brand_name": df["BrandName"], "primary_category": df["PrimaryCategory"],
        "subcategory": df["SubCategory"], "chemical_name": df["ChemicalName"], "cas_raw": df["cas_raw"],
        "cas_number": df["cas_number"], "cas_status": df["cas_status"], "chem_group_id": df["chem_group_id"],
        "company_key": df["company_key"], "brand_key": df["brand_key"],
        "primary_category_key": df["primary_category_key"], "subcategory_key": df["subcategory_key"],
        **{v: df[v] for v in DATE_COLUMNS.values()},
        "chem_removed_date_raw": df["chem_removed_date_raw"], "chemical_count": df["ChemicalCount"],
        "is_exact_dup": df["is_exact_dup"], "is_trade_secret": df["is_trade_secret"],
    })
    dq_df = pd.DataFrame(dq, columns=["row_id", "column_name", "issue_code", "raw_value", "action_taken"])

    coverage = {}
    for col in DATE_COLUMNS.values():
        s = pd.to_datetime(fact[col])
        coverage[col] = {"min": str(s.min().date()), "max": str(s.max().date()), "non_null": int(s.notna().sum())}
    meta = {
        "csv_sha256": sha, "csv_path": str(csv_path.name), "built_at": datetime.now().isoformat(timespec="seconds"),
        "n_rows": len(fact), "n_exact_dup_rows": int(fact["is_exact_dup"].sum()),
        "n_products": int(dim_product.shape[0]), "n_companies": int(dim_company.shape[0]),
        "n_brands": int(dim_brand.shape[0]), "n_chemical_groups": int(dim_group.shape[0]),
        "n_chemical_names": len(names), "coverage": coverage,
        "dq_counts": dq_df["issue_code"].value_counts().to_dict(),
    }

    # ---- write ---------------------------------------------------------------------------------
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".tmp.duckdb")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    con.execute("SET memory_limit='1GB'; SET threads=2")
    tables = {
        "raw_rows": raw, "fact_report": fact, "dim_product": dim_product, "dim_chemical": dim_chem,
        "dim_chemical_group": dim_group, "dim_company": dim_company, "dim_brand": dim_brand,
        "dim_primary_category": dim_primary, "dim_subcategory": dim_sub, "entity_alias": entity_alias,
        "dq_issues": dq_df,
    }
    for name, frame in tables.items():
        con.register("_frame", frame)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _frame")
        con.unregister("_frame")
    con.execute("CREATE TABLE dataset_meta (key VARCHAR PRIMARY KEY, value JSON)")
    con.executemany("INSERT INTO dataset_meta VALUES (?, ?)", [(k, json.dumps(v)) for k, v in meta.items()])
    con.close()
    tmp.replace(db_path)
    log(f"wrote {db_path} in {time.time() - t0:.1f}s: " + ", ".join(f"{k}={len(v):,}" for k, v in tables.items()))
    return meta
