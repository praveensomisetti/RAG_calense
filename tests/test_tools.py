from chemrag.query.filters import Filters


def test_acetaldehyde_counts(engine):
    r = engine.find_products(Filters(cas_numbers=["75-07-0"]), "t1", limit=5)
    assert r.totals["n_products"] == 30 and r.totals["n_rows"] == 56 and r.totals["n_discontinued_products"] == 13
    assert r.truncated and len(r.rows) == 5 and r.evidence_row_ids


def test_every_call_is_logged_with_sql(engine):
    engine.drain_calls()
    engine.count_products(Filters(cas_numbers=["75-07-0"]), "t1")
    calls = engine.drain_calls()
    assert calls and all("?" in c.sql and c.bound_params for c in calls if c.tool == "count_products")
    for c in calls:
        same, _ = engine.replay(c)
        assert same


def test_trend_fills_years_and_sums(engine):
    r = engine.trend_by_year(Filters(), "t1")
    years = [row["year"] for row in r.rows]
    assert years == list(range(2009, 2021))
    assert sum(row["n_products"] for row in r.rows) == 36972


def test_chemicals_for_reports_trade_secret_separately(engine):
    r = engine.chemicals_for(Filters(), "t1", limit=3)
    assert r.rows[0]["chemical"] == "Titanium dioxide"
    assert all(row["chemical"] != "Trade Secret" for row in r.rows)
    assert r.totals["n_products_trade_secret"] == 405


def test_evidence_is_stratified(engine):
    rows = engine.fetch_evidence(engine.evidence_sample(Filters(), "t1", limit=10))
    assert len({r["chemical_name"] for r in rows}) > 3  # not all titanium dioxide


def test_group_by_company(engine):
    r = engine.count_products(Filters(), "t1", group_by="company", limit=3)
    assert r.rows[0]["label"] == "American International Industries" and r.rows[0]["n_products"] == 1964


def test_coverage(engine):
    lo, hi = engine.coverage("initial_reported")
    assert (lo.isoformat(), hi.isoformat()) == ("2009-06-17", "2020-06-23")
