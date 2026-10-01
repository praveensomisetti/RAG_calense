from chemrag.graph import route_after_guard, route_after_query, route_after_resolver
from chemrag.schemas import (
    Candidate,
    EntityType,
    Extraction,
    Intent,
    Mention,
    Plan,
    ProductResolution,
    Resolution,
    SubTask,
)
from chemrag.state import TurnState


def base(**kw):
    plan = Plan(subtasks=[SubTask(id="t1", text="q", intent=Intent.LIST),
                          SubTask(id="t2", text="q2", intent=Intent.LOOKUP)], mode="rules")
    return TurnState(request_id="r", question="q", plan=plan, **kw)


def m(text, etype=EntityType.CHEMICAL):
    return Mention(type=etype, text=text, source="regex")


def cand():
    return Candidate(entity_type=EntityType.CHEMICAL, canonical_id=1, display_name="X", matched_alias="x",
                     score=1.0, method="exact")


def test_guard_refusal():
    assert route_after_guard(base(response_type="refusal")) == "refuse"
    assert route_after_guard(base()) == "ok"


def test_resolver_routes():
    ok = Resolution(subtask_id="t1", mention=m("a"), status="resolved", chosen=[cand()])
    amb = Resolution(subtask_id="t1", mention=m("b"), status="ambiguous", candidates=[cand(), cand()])
    ex = Extraction(subtask_id="t1", mentions=[m("a")])
    assert route_after_resolver(base(resolutions=[ok], extractions=[ex])) == "sql"
    assert route_after_resolver(base(resolutions=[ok, amb], extractions=[ex])) == "clarify"
    ex_p = Extraction(subtask_id="t1", mentions=[m("a"), m("Glover's", EntityType.PRODUCT)])
    assert route_after_resolver(base(resolutions=[ok], extractions=[ex_p])) == "hybrid"
    done = ProductResolution(subtask_id="t1", mention=m("Glover's", EntityType.PRODUCT), status="resolved", cdph_ids=[3])
    assert route_after_resolver(base(resolutions=[ok], extractions=[ex_p], product_resolutions=[done])) == "sql"


def test_clarification_answer_supersedes():
    amb = Resolution(subtask_id="t1", mention=m("b"), status="ambiguous", candidates=[cand()])
    fixed = Resolution(subtask_id="t1", mention=m("b"), status="resolved", chosen=[cand()])
    ex = Extraction(subtask_id="t1", mentions=[m("b")])
    assert route_after_resolver(base(resolutions=[amb, fixed], extractions=[ex])) == "sql"


def test_subtask_loop():
    assert route_after_query(base(current=1)) == "next_subtask"
    assert route_after_query(base(current=2)) == "done"
