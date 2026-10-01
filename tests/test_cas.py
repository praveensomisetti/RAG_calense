import pytest

from chemrag.etl.cas import check_digit_ok, near_miss_suggestions, normalize_cas


@pytest.mark.parametrize("raw,expected", [
    ("13463-67-7", ("13463-67-7", "valid")),
    ("13463-67-7 ", ("13463-67-7", "repaired")),
    ("13463677", ("13463-67-7", "repaired")),
    ("79812", ("79-81-2", "repaired")),
    ("CAS #79-81-2", ("79-81-2", "repaired")),
    ("93 15 2", ("93-15-2", "repaired")),
    ("RN:  93�15�2", ("93-15-2", "repaired")),
    ("101--54-2", ("101-54-2", "repaired")),
    ("75070", ("75-07-0", "repaired")),
    ("0", (None, "sentinel_zero")),
    ("", (None, "missing")),
    (None, (None, "missing")),
    (float("nan"), (None, "missing")),
    ("asdf", (None, "invalid")),
    ("13463-67-", (None, "invalid")),          # truncated: not guessed
    ("50-78-25", (None, "invalid")),           # must not be reshaped into another CAS
    ("201-228-5", (None, "invalid")),          # EC number, not CAS
    ("79-82-1", (None, "invalid")),            # check digit fails
    ("11103-57-438", (None, "invalid")),
])
def test_normalize_cas(raw, expected):
    assert normalize_cas(raw) == expected


def test_check_digit():
    assert check_digit_ok("75-07-0") and check_digit_ok("50-00-0") and not check_digit_ok("75-07-1")


def test_near_miss():
    assert near_miss_suggestions("75-07-1", ["75-07-0", "50-00-0"]) == ["75-07-0"]
