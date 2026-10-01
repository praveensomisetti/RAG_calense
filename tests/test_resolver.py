import pytest

from chemrag.schemas import EntityType as E
from chemrag.schemas import Mention


def res(resolver, text, etype=E.UNKNOWN):
    return resolver.resolve(Mention(type=etype, text=text, source="regex"), "t1")


@pytest.mark.parametrize("text,etype,expected", [
    ("acetaldehide", E.UNKNOWN, "Acetaldehyde"),
    ("titanium dioxde", E.CHEMICAL, "Titanium dioxide"),
    ("formaldehide", E.CHEMICAL, "Formaldehyde"),
    ("vitamin A palmitate", E.CHEMICAL, "Retinyl palmitate"),
    ("BPA", E.CHEMICAL, "Bisphenol A (BPA)"),
    ("methylene chloride", E.CHEMICAL, "Dichloromethane (Methylene chloride)"),
    ("Sally Hansen", E.UNKNOWN, "Sally Hansen"),
    ("loreal usa", E.COMPANY, "L'Oreal USA"),
    ("nail products", E.UNKNOWN, "Nail Products"),
    ("Makeup Products", E.UNKNOWN, "Makeup Products (non-permanent)"),
])
def test_resolves(resolver, text, etype, expected):
    r = res(resolver, text, etype)
    assert r.status == "resolved" and r.chosen[0].display_name == expected


@pytest.mark.parametrize("text", ["75-07-0", "75070", "75 07 0", "CAS 75-07-0"])
def test_cas_variants(resolver, text):
    r = res(resolver, text, E.CAS)
    assert r.status == "resolved" and r.chosen[0].display_name == "Acetaldehyde"


def test_bad_check_digit_suggests(resolver):
    r = res(resolver, "75-07-1", E.CAS)
    assert r.status == "not_found" and "75-07-0" in r.note


def test_valid_cas_not_in_dataset(resolver):
    r = res(resolver, "1071-83-6", E.CAS)
    assert r.status == "not_found" and "does not appear" in r.note


def test_cas_with_multiple_names(resolver):
    r = res(resolver, "79-81-2", E.CAS)
    assert r.status == "resolved" and len(r.chosen) == 2


def test_ambiguous_brand(resolver):
    r = res(resolver, "Pure", E.BRAND)
    assert r.status == "ambiguous" and len(r.candidates) >= 3


def test_family_expansion(resolver):
    r = res(resolver, "retinoids", E.CHEMICAL)
    assert r.status == "resolved" and len(r.chosen) == 5 and all(c.method == "family" for c in r.chosen)


def test_exact_group_beats_family(resolver):
    r = res(resolver, "talc", E.CHEMICAL)
    assert [c.display_name for c in r.chosen] == ["Talc"]


def test_not_found(resolver):
    assert res(resolver, "glyphosate", E.CHEMICAL).status == "not_found"


def test_product_lookup(resolver):
    p = resolver.resolve_product(Mention(type=E.PRODUCT, text="Glovers Medicated Shampoo", source="regex"), "t1")
    assert p.status == "resolved" and p.cdph_ids == [3]
    g = resolver.resolve_product(Mention(type=E.PRODUCT, text="Lipstick", source="regex"), "t1")
    assert g.status == "ambiguous"
