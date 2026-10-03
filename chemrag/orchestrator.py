"""Public entry point: build dependencies once, run questions through the LangGraph app, persist traces."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from typing import Any

from langgraph.types import Command

from chemrag.agents.context import AgentContext
from chemrag.graph import build_graph
from chemrag.llm.null_client import make_llm
from chemrag.query.tools import QueryEngine
from chemrag.retrieval.resolver import EntityResolver
from chemrag.schemas import Response
from chemrag.settings import Settings, get_settings
from chemrag.state import RunOptions, TurnState


class Orchestrator:
    def __init__(self, settings: Settings | None = None, no_llm: bool = False, use_vectors: bool = True,
                 llm: Any = None):
        self.settings = settings or get_settings()
        s = self.settings
        self.engine = QueryEngine(s.db_path, s.duckdb_memory_limit, s.duckdb_threads, evidence_limit=s.evidence_limit)
        self.resolver = EntityResolver(self.engine, s, use_vectors=use_vectors)
        self.llm = llm if llm is not None else make_llm(s, disabled=no_llm)
        self.ctx = AgentContext(settings=s, engine=self.engine, resolver=self.resolver, llm=self.llm)
        self.app = build_graph(self.ctx)

    def ask(self, question: str, options: RunOptions | None = None,
            on_clarify: Callable[[dict], int | None] | None = None, save: bool = True) -> Response:
        return self.ask_full(question, options, on_clarify, save)[0]

    def ask_full(self, question: str, options: RunOptions | None = None,
                 on_clarify: Callable[[dict], int | None] | None = None,
                 save: bool = True) -> tuple[Response, dict]:
        """Like `ask`, but also returns the final graph state (used by evals and tests)."""
        rid = uuid.uuid4().hex[:12]
        config = {"configurable": {"thread_id": rid}, "recursion_limit": 60}
        opts = options or RunOptions()
        usage_before = dict(getattr(self.llm, "usage", {}) or {})
        if on_clarify is not None:
            opts = opts.model_copy(update={"interactive": True})
        out = self.app.invoke(TurnState(request_id=rid, question=question, options=opts), config)
        while isinstance(out, dict) and out.get("__interrupt__"):
            payload = out["__interrupt__"][0].value
            choice = on_clarify(payload) if on_clarify else None
            out = self.app.invoke(Command(resume=choice), config)
        values = self.app.get_state(config).values
        resp = values["response"]
        resp = resp if isinstance(resp, Response) else Response.model_validate(resp)
        usage_after = getattr(self.llm, "usage", None)
        if usage_after:  # LLM tokens spent on this question (cost visibility)
            resp.meta["llm_tokens"] = {k: v - usage_before.get(k, 0) for k, v in usage_after.items()}
        if save:
            self._save(resp, values)
        return resp, values

    def _save(self, resp: Response, values: dict) -> None:
        d = self.settings.runs_dir
        d.mkdir(parents=True, exist_ok=True)
        calls = [c.model_dump(mode="json") if hasattr(c, "model_dump") else c for c in values.get("tool_calls", [])]
        (d / f"{resp.request_id}.json").write_text(json.dumps(
            {"response": resp.model_dump(mode="json"), "tool_calls": calls}, indent=2, default=str))

    def replay(self, request_id: str) -> list[dict]:
        from chemrag.schemas import ToolCall

        data = json.loads((self.settings.runs_dir / f"{request_id}.json").read_text())
        out = []
        for c in data["tool_calls"]:
            call = ToolCall.model_validate(c)
            same, n = self.engine.replay(call)
            out.append({"id": call.id, "tool": call.tool, "rows": n, "identical": same})
        return out
