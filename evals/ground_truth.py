"""Hand-written pandas ground truth over the RAW CSV.

Deliberately independent of the chemrag package (no shared ETL, normaliser or SQL), so bugs in cleaning,
resolution or query templates show up as eval failures instead of being reproduced by the checker.

Each entry returns a dict of expected values plus `row_ok(row) -> bool`, which says whether a cited raw row
supports the answer (used for citation precision).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

CSV = Path(__file__).resolve().parent.parent / "data" / "raw" / "interviewtestdataset.csv"

TIO2 = {"Titanium dioxide", "Titanium dioxide (airborne, unbound particles of respirable size)"}
RETINYL_PALMITATE = {"Retinyl palmitate", "Vitamin A palmitate", "Retinol palmitate"}
COCAMIDE_DEA = {"Cocamide diethanolamine", "Cocamide DEA", "Cocamide diethanolamine (DEA)",
                "Diethanolamides of the fatty acids of coconut oil", "Cocamide"}
FORMALDEHYDE = {"Formaldehyde", "Formaldehyde (gas)", "Formaldehyde solution"}
CARBON_BLACK = {"Carbon black", "Carbon black (airborne, unbound particles of respirable size)"}
SILICA = {"Silica, crystalline (airborne particles of respirable size)"}
RETINOIDS = {"Retinol", "Vitamin A", "Retinyl palmitate", "Vitamin A palmitate", "Retinol palmitate",
             "Retinyl acetate", "Acetic acid, retinyl ester", "All-trans retinoic acid",
             "Retinol/retinyl esters, when in daily dosages in excess of 10,000 IU, or 3,000 retinol equivalents."}


@lru_cache(maxsize=1)
def raw() -> pd.DataFrame:
    df = pd.read_csv(CSV, dtype=str, keep_default_na=False, na_values=[""])
    df.insert(0, "row_id", range(1, len(df) + 1))
    for c in ["ProductName", "CompanyName", "BrandName", "PrimaryCategory", "SubCategory", "ChemicalName", "CasNumber"]:
        df[c] = df[c].str.strip()
    df.loc[df["BrandName"].str.casefold().isin(["na", "n/a", "none", "null", "-"]), "BrandName"] = None
    for c in ["InitialDateReported", "MostRecentDateReported", "DiscontinuedDate", "ChemicalDateRemoved"]:
        df[c + "_dt"] = pd.to_datetime(df[c], format="%m/%d/%Y", errors="coerce")
    df["dup"] = df.drop(columns=["row_id"]).duplicated()
    return df


def nd() -> pd.DataFrame:
    d = raw()
    return d[~d["dup"]]


def products_in_category(col: str, value: str) -> set[str]:
    d = nd()
    return set(d.loc[d[col] == value, "CDPHId"])


def by_row(df: pd.DataFrame):
    ids = set(df["row_id"])
    return lambda row: row["row_id"] in ids


def chem_set(df: pd.DataFrame) -> set[str]:
    return set(df.loc[df["ChemicalName"] != "Trade Secret", "ChemicalName"])


def gt_products(names: set[str] | None = None, cas: str | None = None) -> pd.DataFrame:
    d = nd()
    if names is not None:
        d = d[d["ChemicalName"].isin(names)]
    if cas is not None:
        d = d[d["CasNumber"] == cas]
    return d


def _first_reported_year(d: pd.DataFrame) -> pd.Series:
    return nd().groupby("CDPHId")["InitialDateReported_dt"].min().dt.year


# ----------------------------------------------------------------------------------------------- cases
def case_cas_acetaldehyde():
    d = gt_products(cas="75-07-0")
    return {"n_products": d["CDPHId"].nunique(), "n_discontinued": d.loc[d["DiscontinuedDate"].notna(), "CDPHId"].nunique(),
            "row_ok": by_row(d)}


def case_name_acetaldehyde():
    d = gt_products({"Acetaldehyde"})
    return {"n_products": d["CDPHId"].nunique(), "row_ok": by_row(d)}


def case_group(names: set[str]):
    def f():
        d = gt_products(names)
        return {"n_products": d["CDPHId"].nunique(), "row_ok": by_row(d)}
    return f


def case_sally_hansen_nail_polish():
    d = nd()
    prods = products_in_category("SubCategory", "Nail Polish and Enamel")
    d = d[(d["BrandName"].str.strip().str.casefold() == "sally hansen") & d["CDPHId"].isin(prods)]
    return {"n_products": d["CDPHId"].nunique(), "chemicals": chem_set(d), "row_ok": by_row(d)}


def case_company_chemicals(company: str, exclude: set[str] = frozenset()):
    def f():
        d = nd()
        d = d[(d["CompanyName"] == company)]
        all_products = d["CDPHId"].nunique()
        d2 = d[~d["ChemicalName"].isin(exclude)]
        out = {"chemicals": chem_set(d2), "row_ok": by_row(d2)}
        if not exclude:
            out["n_products"] = all_products
        return out
    return f


def case_discontinued_2015_rp():
    d = gt_products(RETINYL_PALMITATE)
    d = d[d["DiscontinuedDate_dt"].dt.year == 2015]
    return {"n_products": d["CDPHId"].nunique(), "row_ok": by_row(d)}


def case_revlon_removed():
    d = nd()
    d = d[(d["BrandName"].str.casefold() == "revlon") & d["ChemicalDateRemoved"].notna()]
    return {"n_products": d["CDPHId"].nunique(), "chemicals": chem_set(d), "row_ok": by_row(d)}


def case_trend_nail():
    d = nd()
    prods = products_in_category("PrimaryCategory", "Nail Products")
    first = d[d["CDPHId"].isin(prods)].groupby("CDPHId")["InitialDateReported_dt"].min().dt.year
    return {"trend": first.value_counts().sort_index().to_dict(),
            "row_ok": lambda row: row["CDPHId"] in prods}


def case_trend_aii_recent():
    d = nd()
    d = d[d["CompanyName"] == "American International Industries"]
    last = d.groupby("CDPHId")["MostRecentDateReported_dt"].max().dt.year
    return {"trend": last.value_counts().sort_index().to_dict(), "row_ok": by_row(d)}


def case_compare_carbon_black():
    d = gt_products(CARBON_BLACK)
    out = []
    for cat in ["Makeup Products (non-permanent)", "Nail Products"]:
        out.append(d[d["CDPHId"].isin(products_in_category("PrimaryCategory", cat))]["CDPHId"].nunique())
    return {"compare": out, "row_ok": by_row(d)}


def case_top_companies_silica():
    d = gt_products(SILICA)
    top = d.groupby("CompanyName")["CDPHId"].nunique().sort_values(ascending=False).head(5)
    return {"top": [(k, int(v)) for k, v in top.items()], "row_ok": by_row(d)}


def case_2019():
    d = nd()
    first = d.groupby("CDPHId")["InitialDateReported_dt"].min().dt.year
    prods = set(first[first == 2019].index)
    sub = d[d["CDPHId"].isin(prods)].groupby("SubCategory")["CDPHId"].nunique().sort_values(ascending=False)
    return {"n_products": len(prods), "top_t2": [(k, int(v)) for k, v in sub.head(3).items()],
            "row_ok": lambda row: row["CDPHId"] in prods}


def case_glovers():
    d = nd()
    d = d[d["ProductName"] == "Glover's Medicated Shampoo"]
    return {"chemicals": chem_set(d), "row_ok": by_row(d)}


def case_bpa_baby():
    d = gt_products({"Bisphenol A (BPA)"})
    d = d[d["CDPHId"].isin(products_in_category("PrimaryCategory", "Baby Products"))]
    return {"n_products": d["CDPHId"].nunique(), "row_ok": lambda row: False}


def case_missing_cas():
    return {"dq_null_cas": int(raw()["CasNumber"].isna().sum()), "row_ok": lambda row: True}


def case_coverage():
    d = raw()
    return {"coverage": (d["InitialDateReported_dt"].min().date().isoformat(),
                         d["InitialDateReported_dt"].max().date().isoformat()), "row_ok": lambda row: True}


def case_companies():
    top = nd().groupby("CompanyName")["CDPHId"].nunique().sort_values(ascending=False).head(3)
    return {"top": [(k, int(v)) for k, v in top.items()], "row_ok": lambda row: True}


def case_last3_discontinued():
    d = nd()
    hi = raw()["InitialDateReported_dt"].max()
    x = d[(d["DiscontinuedDate_dt"].dt.year >= hi.year - 2) & (d["DiscontinuedDate_dt"] <= hi)]
    return {"n_products": x["CDPHId"].nunique(), "row_ok": by_row(x)}


def case_trade_secret():
    d = gt_products({"Trade Secret"})
    return {"n_products": d["CDPHId"].nunique(), "row_ok": by_row(d)}


def case_acetaldehyde_multi():
    d = gt_products({"Acetaldehyde"})
    return {"n_products": d["CDPHId"].nunique(),
            "n_products_t2": d.loc[d["DiscontinuedDate"].notna(), "CDPHId"].nunique(), "row_ok": by_row(d)}


def case_retinoids_skin():
    d = gt_products(RETINOIDS)
    d = d[d["CDPHId"].isin(products_in_category("PrimaryCategory", "Skin Care Products"))]
    return {"n_products": d["CDPHId"].nunique(), "chemicals": chem_set(d), "row_ok": by_row(d)}


def case_none():
    return {"row_ok": lambda row: True}


GROUND_TRUTH = {
    "cas_acetaldehyde": case_cas_acetaldehyde,
    "name_acetaldehyde": case_name_acetaldehyde,
    "cas_bare_digits": case_cas_acetaldehyde,
    "cas_bad_check_digit": case_none,
    "misspelled_tio2": case_group(TIO2),
    "synonym_vitamin_a_palmitate": case_group(RETINYL_PALMITATE),
    "synonym_cocamide_dea": case_group(COCAMIDE_DEA),
    "formaldehyde_variants": case_group(FORMALDEHYDE),
    "brand_subcategory_chemicals": case_sally_hansen_nail_polish,
    "ambiguous_brand_pure": case_none,
    "company_chemicals_loreal": case_company_chemicals("L'Oreal USA"),
    "discontinued_2024_out_of_range": case_none,
    "discontinued_2015_retinyl_palmitate": case_discontinued_2015_rp,
    "removed_by_revlon": case_revlon_removed,
    "removed_2103_invalid": case_none,
    "trend_nail_products": case_trend_nail,
    "trend_company_most_recent": case_trend_aii_recent,
    "compare_carbon_black": case_compare_carbon_black,
    "top_companies_silica": case_top_companies_silica,
    "multi_part_2019": case_2019,
    "product_glovers": case_glovers,
    "generic_product_lipstick": case_none,
    "empty_bpa_baby": case_bpa_baby,
    "data_quality_cas": case_missing_cas,
    "coverage": case_coverage,
    "medical_refusal": case_none,
    "out_of_scope_weather": case_none,
    "prompt_injection": case_companies,
    "missing_entity": case_none,
    "relative_last_3_years": case_last3_discontinued,
    "trade_secret": case_trade_secret,
    "unknown_chemical": case_none,
    "exclusion_loreal": case_company_chemicals("L'Oreal USA", exclude=TIO2),
    "multi_part_inherit": case_acetaldehyde_multi,
    "family_retinoids_skin": case_retinoids_skin,
}
