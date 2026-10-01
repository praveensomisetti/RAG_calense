"""Invariants of the built database (numbers verified independently in profiling)."""


def q(engine, sql):
    return engine.table(sql)[0]


def test_row_counts(engine):
    r = q(engine, "SELECT count(*) n, min(row_id) lo, max(row_id) hi, count(DISTINCT row_id) d FROM fact_report")
    assert r == {"n": 114635, "lo": 1, "hi": 114635, "d": 114635}


def test_exact_duplicates_flagged(engine):
    assert q(engine, "SELECT count(*) n FROM fact_report WHERE is_exact_dup")["n"] == 254


def test_products_and_companies(engine):
    assert q(engine, "SELECT count(*) n FROM dim_product")["n"] == 36972
    assert q(engine, "SELECT count(DISTINCT company_name) n FROM fact_report")["n"] == 606


def test_no_future_dates_in_clean_columns(engine):
    r = q(engine, "SELECT max(chem_removed_date) m, count(*) FILTER (WHERE regexp_matches(chem_removed_date_raw, '/21[0-9]{2}$')) bad "
                  "FROM fact_report")
    assert str(r["m"]) < "2021" and r["bad"] == 115


def test_brand_placeholders_are_null(engine):
    r = q(engine, "SELECT count(*) n FROM fact_report WHERE lower(brand_name) IN ('na','n/a','none')")
    assert r["n"] == 0


def test_subcategory_trimmed(engine):
    assert q(engine, "SELECT count(*) n FROM fact_report WHERE subcategory <> trim(subcategory)")["n"] == 0


def test_cas_numbers_valid_or_null(engine):
    from chemrag.etl.cas import check_digit_ok

    cas = [r["cas_number"] for r in engine.table("SELECT DISTINCT cas_number FROM fact_report WHERE cas_number IS NOT NULL")]
    assert cas and all(check_digit_ok(c) for c in cas)


def test_every_chemical_name_has_a_group(engine):
    assert q(engine, "SELECT count(*) n FROM fact_report WHERE chem_group_id IS NULL")["n"] == 0


def test_raw_rows_preserved(engine):
    r = q(engine, "SELECT CasNumber FROM raw_rows WHERE row_id = (SELECT min(row_id) FROM fact_report "
                  "WHERE cas_raw = 'asdf')")
    assert r["CasNumber"] == "asdf"
