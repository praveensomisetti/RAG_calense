"""Run the golden set and score it against the independent pandas ground truth.

Metrics: response-type accuracy, intent accuracy, entity-resolution accuracy, answer correctness,
citation precision (cited rows really satisfy the question), warning recall, refusal/clarification accuracy.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml

from chemrag.orchestrator import Orchestrator
from chemrag.state import RunOptions
from evals.ground_truth import GROUND_TRUTH, raw

HERE = Path(__file__).resolve().parent

# Budget-friendly subset (~30 LLM calls): one question per behaviour the brief asks for.
QUICK_IDS = [
    "cas_acetaldehyde",                  # CAS lookup with citations
    "multi_part_inherit",                # typo + two-part question
    "chemical_or_cas",                   # brief example 1: chemical X or CAS
    "brand_subcategory_chemicals",       # brief example 2: chemicals for brand in subcategory
    "discontinued_2024_out_of_range",    # brief example 3: discontinued in 2024 -> no data in range
    "trend_nail_products",               # brief example 4: trends over time
    "synonym_vitamin_a_palmitate",       # synonym resolution
    "ambiguous_brand_pure",              # ambiguity -> clarification
    "compare_carbon_black",              # compare intent
    "medical_refusal",                   # safety: medical advice refused
]


def _results(values: dict, sub_id: str) -> list:
    return [r for r in values.get("results", []) if r.subtask_id == sub_id]


def check_values(expected: dict, values: dict) -> dict[str, bool]:
    out: dict[str, bool] = {}
    t1, t2 = _results(values, "t1"), _results(values, "t2")
    main = t1[0] if t1 else None
    for key, exp in expected.items():
        if key == "row_ok":
            continue
        got: Any = None
        if key == "n_products":
            got = main.totals.get("n_products") if main else None
        elif key == "n_discontinued":
            got = main.totals.get("n_discontinued_products") if main else None
        elif key == "n_products_t2":
            got = t2[0].totals.get("n_products") if t2 else None
        elif key == "chemicals":
            res = next((r for r in t1 if r.tool == "chemicals_for"), None)
            got = {n for row in res.rows for n in row["reported_names"]} if res else None
        elif key == "trend":
            res = next((r for r in t1 if r.tool == "trend_by_year"), None)
            got = {row["year"]: row["n_products"] for row in res.rows if row["n_products"]} if res else None
            exp = {int(k): int(v) for k, v in exp.items()}
        elif key == "compare":
            got = [r.totals.get("n_products") for r in t1]
        elif key in ("top", "top_t2"):
            res = (t2 if key == "top_t2" else t1)
            rows = res[0].rows[: len(exp)] if res else []
            got = [(row["label"], row["n_products"]) for row in rows]
            out[key] = [(a.casefold(), b) for a, b in got] == [(a.casefold(), b) for a, b in exp]
            continue
        elif key == "dq_null_cas":
            got = main.totals.get("rows_cas_null_in_source") if main else None
        elif key == "coverage":
            cov = main.totals["coverage"]["initial_reported"] if main else {}
            got = (cov.get("min"), cov.get("max"))
        out[key] = got == exp
    return out


def main(no_llm: bool = False, only: list[str] | None = None, out_dir: Path | None = None,
         set_name: str = "golden", quick: bool = False) -> int:
    cases = yaml.safe_load((HERE / f"{set_name}.yaml").read_text())
    if quick:
        cases = [c for c in cases if c["id"] in QUICK_IDS]
        set_name = f"{set_name}_quick"
    if only:
        cases = [c for c in cases if c["id"] in only]
    orch = Orchestrator(no_llm=no_llm)
    data = raw().set_index("row_id")
    rows, t0 = [], time.time()
    for c in cases:
        gt = GROUND_TRUTH[c.get("gt", c["id"])]()
        t = time.time()
        resp, values = orch.ask_full(c["q"], RunOptions(limit=20), save=False)
        elapsed = time.time() - t
        plan = values.get("plan")
        intents = [s.intent.value for s in plan.subtasks] if plan else []
        resolved = {ch.display_name for r in values.get("resolutions", []) if r.status == "resolved" for ch in r.chosen}
        exp_entities = c.get("entities", [])
        ent_hits = sum(1 for e in exp_entities if e in resolved)
        vals = check_values(gt, values)
        cited = [e.row_id for e in resp.evidence]
        ok_cited = sum(1 for rid in cited if gt["row_ok"]({**data.loc[rid].to_dict(), "row_id": rid}))
        warn_codes = {w.code for w in resp.warnings}
        must = c.get("must_warn", [])
        row = {
            "id": c["id"], "question": c["q"], "type_expected": c["type"], "type_got": resp.response_type,
            "type_ok": resp.response_type == c["type"],
            "intents_expected": c.get("intents"), "intents_got": intents,
            "intent_ok": None if "intents" not in c else intents == c["intents"],
            "entities_expected": exp_entities, "entities_hit": ent_hits,
            "values": vals, "values_ok": all(vals.values()) if vals else None,
            "cited": len(cited), "cited_ok": ok_cited,
            "must_warn": must, "warn_missing": [w for w in must if w not in warn_codes],
            "answer_short": resp.answer_short, "confidence": resp.confidence.level, "seconds": round(elapsed, 2),
        }
        row["pass"] = (row["type_ok"] and row["intent_ok"] is not False and ent_hits == len(exp_entities)
                       and row["values_ok"] is not False and not row["warn_missing"] and ok_cited == len(cited))
        rows.append(row)
        print(f"{'PASS' if row['pass'] else 'FAIL'} {c['id']:<38} type={resp.response_type:<13} "
              f"values={vals} cited={ok_cited}/{len(cited)} {elapsed:.1f}s")

    def rate(xs):
        xs = [x for x in xs if x is not None]
        return (sum(bool(x) for x in xs) / len(xs)) if xs else None

    metrics = {
        "cases": len(rows), "passed": sum(r["pass"] for r in rows),
        "response_type_accuracy": rate([r["type_ok"] for r in rows]),
        "intent_accuracy": rate([r["intent_ok"] for r in rows]),
        "entity_resolution_accuracy": (sum(r["entities_hit"] for r in rows) /
                                       max(1, sum(len(r["entities_expected"]) for r in rows))),
        "answer_correctness": rate([r["values_ok"] for r in rows]),
        "citation_precision": sum(r["cited_ok"] for r in rows) / max(1, sum(r["cited"] for r in rows)),
        "warning_recall": 1 - sum(len(r["warn_missing"]) for r in rows) / max(1, sum(len(r["must_warn"]) for r in rows)),
        "refusal_clarification_accuracy": rate([r["type_ok"] for r in rows
                                                if r["type_expected"] in ("refusal", "clarification", "no_data")]),
        "mode": "no-llm" if no_llm or not orch.llm.available else orch.llm.name,
        "vectors": "on" if orch.resolver._vectors_on() else "off",
        "seconds_total": round(time.time() - t0, 1),
    }
    out_dir = out_dir or HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{set_name}_" + metrics["mode"].replace(":", "_")
    (out_dir / f"eval_{tag}.json").write_text(json.dumps({"metrics": metrics, "cases": rows}, indent=2, default=str))
    md = [f"# Eval report: {set_name} set ({metrics['mode']}, vectors {metrics['vectors']})", "",
          "| metric | value |", "|---|---|"]
    for k, v in metrics.items():
        md.append(f"| {k} | {v:.3f} |" if isinstance(v, float) else f"| {k} | {v} |")
    md += ["", "| id | pass | type | values | cited ok | answer |", "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['id']} | {'✅' if r['pass'] else '❌'} | {r['type_got']} | "
                  f"{'/'.join(k for k, v in r['values'].items() if v) or '-'} | {r['cited_ok']}/{r['cited']} | "
                  f"{r['answer_short'][:90].replace('|', '/')} |")
    (out_dir / f"eval_{tag}.md").write_text("\n".join(md) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["passed"] == metrics["cases"] else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(main(no_llm="--no-llm" in sys.argv, set_name="holdout" if "--holdout" in sys.argv else "golden",
                          quick="--quick" in sys.argv))
