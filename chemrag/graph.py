"""LangGraph wiring: agents are nodes, routing is deterministic conditional edges.

    START → guard → planner → extractor → resolver ─┬─(sql)──────→ query ─┬─(next sub-task)→ extractor
                 ↘ refuse     ↘ refuse              ├─(hybrid)→ retrieval ↗└─(done)→ synthesizer → verifier → finalize → END
                                                    └─(clarify)→ clarify ──(answer / interrupt-resume)──↗
"""

from __future__ import annotations

from functools import partial

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from chemrag.agents.context import AgentContext
from chemrag.agents.extractor import extractor_node
from chemrag.agents.finalize import finalize_node
from chemrag.agents.guard import guard_node
from chemrag.agents.planner import planner_node
from chemrag.agents.query_agent import query_node
from chemrag.agents.resolution import clarify_node, pending_step, resolver_node, retrieval_node
from chemrag.agents.synthesizer import synthesizer_node
from chemrag.agents.verifier import verifier_node
from chemrag.state import TurnState


# ---------------------------------------------------------------- routers (pure functions of state)
def route_after_guard(state: TurnState) -> str:
    return "refuse" if state.response_type == "refusal" else "ok"


def route_after_planner(state: TurnState) -> str:
    return "refuse" if state.response_type == "refusal" else "extract"


def route_after_resolver(state: TurnState) -> str:
    return {"clarify": "clarify", "retrieval": "hybrid", "query": "sql"}[pending_step(state)]


def route_after_retrieval(state: TurnState) -> str:
    return "clarify" if pending_step(state) == "clarify" else "sql"


def route_after_query(state: TurnState) -> str:
    return "next_subtask" if state.plan and state.current < len(state.plan.subtasks) else "done"


def build_graph(ctx: AgentContext, checkpointer=None):
    g = StateGraph(TurnState)
    g.add_node("guard", partial(guard_node, ctx=ctx))
    g.add_node("planner", partial(planner_node, ctx=ctx))
    g.add_node("extractor", partial(extractor_node, ctx=ctx))
    g.add_node("resolver", partial(resolver_node, ctx=ctx))
    g.add_node("retrieval", partial(retrieval_node, ctx=ctx))
    g.add_node("clarify", partial(clarify_node, ctx=ctx), destinations=("query", "retrieval", "clarify", "finalize"))
    g.add_node("query", partial(query_node, ctx=ctx))
    g.add_node("synthesizer", partial(synthesizer_node, ctx=ctx))
    g.add_node("verifier", partial(verifier_node, ctx=ctx))
    g.add_node("finalize", partial(finalize_node, ctx=ctx))

    g.add_edge(START, "guard")
    g.add_conditional_edges("guard", route_after_guard, {"ok": "planner", "refuse": "finalize"})
    g.add_conditional_edges("planner", route_after_planner, {"extract": "extractor", "refuse": "synthesizer"})
    g.add_edge("extractor", "resolver")
    g.add_conditional_edges("resolver", route_after_resolver,
                            {"sql": "query", "hybrid": "retrieval", "clarify": "clarify"})
    g.add_conditional_edges("retrieval", route_after_retrieval, {"sql": "query", "clarify": "clarify"})
    g.add_conditional_edges("query", route_after_query, {"next_subtask": "extractor", "done": "synthesizer"})
    g.add_edge("synthesizer", "verifier")
    g.add_edge("verifier", "finalize")
    g.add_edge("finalize", END)
    if checkpointer is None:
        # State is made of our own Pydantic models; allow exactly these modules in the checkpoint codec.
        serde = JsonPlusSerializer(allowed_msgpack_modules=[("chemrag.state", "RunOptions"), *SCHEMA_TYPES])
        checkpointer = MemorySaver(serde=serde)
    return g.compile(checkpointer=checkpointer)


SCHEMA_TYPES = [("chemrag.schemas", n) for n in (
    "Intent", "SubTask", "Plan", "EntityType", "Mention", "DateConstraint", "Extraction", "Candidate", "Resolution",
    "ProductResolution", "ToolCall", "ToolResult", "Fact", "EvidenceItem", "WarningItem", "TraceEvent", "Confidence",
    "ClarificationOption", "Clarification", "Draft", "Verification", "Response")]
