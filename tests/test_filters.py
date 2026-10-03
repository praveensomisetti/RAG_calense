from datetime import date

from chemrag.query.filters import Filters, compile_filters
from chemrag.schemas import DateConstraint


def test_values_are_bound_not_interpolated():
    evil = "x'); DROP TABLE fact_report; --"
    where, params = compile_filters(Filters(cas_numbers=[evil], company_keys=[1, 2]))
    assert evil not in where and evil in params and where.count("?") == len(params)


def test_product_level_constraints_use_subqueries():
    f = Filters(chem_group_ids=[5], subcategory_keys=[7], discontinued=True,
                dates=[DateConstraint(field="initial_reported", start=date(2019, 1, 1), end=date(2019, 12, 31))])
    where, params = compile_filters(f)
    assert "f.chem_group_id IN (?)" in where
    assert "cdph_id IN (SELECT cdph_id FROM fact_report WHERE NOT is_exact_dup AND subcategory_key IN (?))" in where
    assert "dim_product WHERE discontinued_date IS NOT NULL" in where
    assert params == [5, 7, date(2019, 1, 1), date(2019, 12, 31)]


def test_trade_secret_excluded_only_for_chemical_listings():
    assert "is_trade_secret" not in compile_filters(Filters())[0]
    assert "NOT f.is_trade_secret" in compile_filters(Filters(), exclude_trade_secret=True)[0]


def test_removed_uses_raw_column():
    assert "chem_removed_date_raw IS NOT NULL" in compile_filters(Filters(chem_removed=True))[0]


def test_describe_and_without():
    f = Filters(chem_group_ids=[1], labels={"chemical": ["Talc"]}, discontinued=True)
    assert f.describe() == "containing Talc that are discontinued"
    assert f.without("discontinued").discontinued is None


def test_chemical_and_cas_are_alternatives():
    where, params = compile_filters(Filters(chem_group_ids=[1], cas_numbers=["75-07-0"]))
    assert "(f.chem_group_id IN (?) OR f.cas_number IN (?))" in where and params == [1, "75-07-0"]
