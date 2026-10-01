"""LangGraph state: one typed object flows through every node.

List fields use an `operator.add` reducer, so nodes return only what they add and the trace stays
append-only (a later clarification appends a newer Resolution rather than mutating the old one).
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal

from pydantic import BaseModel

from chemrag.schemas import (
    Clarification,
    Draft,
    EvidenceItem,
    Extraction,
    Fact,
    Plan,
    ProductResolution,
    Resolution,
    Response,
    ToolCall,
    ToolResult,
    TraceEvent,
    Verification,
    WarningItem,
)


class RunOptions(BaseModel):
    interactive: bool = False  # CLI may pause for clarification via LangGraph interrupt()
    assume_best: bool = False  # proceed with top candidate on ambiguity (with a warning)
    limit: int = 20
    offset: int = 0


class TurnState(BaseModel):
    request_id: str
    question: str
    options: RunOptions = RunOptions()
    clean_question: str = ""
    refusal_reason: str | None = None
    plan: Plan | None = None
    current: int = 0
    extractions: Annotated[list[Extraction], operator.add] = []
    resolutions: Annotated[list[Resolution], operator.add] = []
    product_resolutions: Annotated[list[ProductResolution], operator.add] = []
    tool_calls: Annotated[list[ToolCall], operator.add] = []
    results: Annotated[list[ToolResult], operator.add] = []
    assumptions: Annotated[list[str], operator.add] = []
    warnings: Annotated[list[WarningItem], operator.add] = []
    trace: Annotated[list[TraceEvent], operator.add] = []
    clarification: Clarification | None = None
    response_type: Literal["answer", "clarification", "refusal", "no_data"] = "answer"
    facts: list[Fact] = []
    evidence: list[EvidenceItem] = []
    draft: Draft | None = None
    verification: Verification | None = None
    response: Response | None = None

    # ---------------------------------------------------------------- helpers used by nodes / routers
    def extraction_for(self, subtask_id: str) -> Extraction | None:
        found = [e for e in self.extractions if e.subtask_id == subtask_id]
        return found[-1] if found else None

    def latest_resolutions(self, subtask_id: str) -> list[Resolution]:
        """Most recent resolution per mention (clarification answers supersede earlier ones)."""
        latest: dict[str, Resolution] = {}
        for r in self.resolutions:
            if r.subtask_id == subtask_id:
                latest[r.mention.text.casefold()] = r
        return list(latest.values())

    def latest_product_resolutions(self, subtask_id: str) -> list[ProductResolution]:
        latest: dict[str, ProductResolution] = {}
        for r in self.product_resolutions:
            if r.subtask_id == subtask_id:
                latest[r.mention.text.casefold()] = r
        return list(latest.values())

    @property
    def current_subtask(self):
        if self.plan and self.current < len(self.plan.subtasks):
            return self.plan.subtasks[self.current]
        return None
