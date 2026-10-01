from chemrag.agents.verifier import allowed_numbers, check_text
from chemrag.schemas import Fact
from chemrag.state import TurnState


def state_with(facts):
    return TurnState(request_id="x", question="Which products contain CAS 75-07-0?", facts=facts)


def test_grounded_text_passes():
    s = state_with([Fact(id="F1", statement="Products containing Acetaldehyde: 30", value=30)])
    allowed = allowed_numbers(s)
    assert check_text("- 30 products contain it [F1]", allowed, {"F1"}, require_refs=True) == []


def test_invented_number_fails():
    s = state_with([Fact(id="F1", statement="Products: 30", value=30)])
    problems = check_text("- 31 products contain it [F1]", allowed_numbers(s), {"F1"}, require_refs=True)
    assert any("31" in p for p in problems)


def test_missing_reference_and_unknown_fact():
    s = state_with([Fact(id="F1", statement="Products: 30", value=30)])
    problems = check_text("- 30 products\n- see [F9]", allowed_numbers(s), {"F1"}, require_refs=True)
    assert any("no fact reference" in p for p in problems) and any("F9" in p for p in problems)


def test_percentages_and_advice_rejected():
    s = state_with([Fact(id="F1", statement="Products: 30", value=30)])
    problems = check_text("- 50% of them [F1]. It is safe to use.", allowed_numbers(s), {"F1"}, require_refs=True)
    assert any("percentage" in p for p in problems) and any("advice" in p for p in problems)
