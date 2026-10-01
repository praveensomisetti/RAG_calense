"""Pydantic contracts for every message passed between agents, and for the final response."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- planner
class Intent(str, Enum):
    LOOKUP = "lookup"  # how many / count
    LIST = "list"  # which / show / list
    COMPARE = "compare"
    SUMMARIZE = "summarize"
    TREND = "trend"
    DATA_QUALITY = "data_quality"
    COVERAGE = "coverage"
    OUT_OF_SCOPE = "out_of_scope"


class SubTask(BaseModel):
    id: str
    text: str
    intent: Intent
    intent_confidence: float = 1.0
    depends_on: list[str] = []


class Plan(BaseModel):
    subtasks: list[SubTask]
    mode: Literal["llm", "rules"]
    scope: Literal["in_scope", "medical_advice", "unrelated"] = "in_scope"
    note: str | None = None


# ---------------------------------------------------------------- extraction
class EntityType(str, Enum):
    CHEMICAL = "chemical"
    CHEMICAL_FAMILY = "chemical_family"
    CAS = "cas"
    COMPANY = "company"
    BRAND = "brand"
    PRODUCT = "product"
    PRIMARY_CATEGORY = "primary_category"
    SUBCATEGORY = "subcategory"
    UNKNOWN = "unknown"


class Mention(BaseModel):
    type: EntityType
    text: str
    source: Literal["regex", "gazetteer", "cue", "llm", "inherited", "clarification"]
    confidence: float = 1.0
    negated: bool = False  # "excluding X"


DateField = Literal["initial_reported", "most_recent_reported", "discontinued_date", "chem_removed_date"]

DATE_FIELD_LABELS: dict[str, str] = {
    "initial_reported": "InitialDateReported",
    "most_recent_reported": "MostRecentDateReported",
    "discontinued_date": "DiscontinuedDate",
    "chem_removed_date": "ChemicalDateRemoved",
}


class DateConstraint(BaseModel):
    field: DateField
    start: date | None = None  # inclusive
    end: date | None = None  # inclusive
    source_text: str = ""
    defaulted_field: bool = False  # field picked by the default rule -> recorded as an assumption


GroupBy = Literal["year", "company", "brand", "chemical", "subcategory", "primary_category"]
Metric = Literal["products", "report_records", "chemicals", "companies", "brands"]


class Extraction(BaseModel):
    subtask_id: str
    mentions: list[Mention] = []
    dates: list[DateConstraint] = []
    discontinued: bool | None = None
    chem_removed: bool | None = None
    group_by: list[GroupBy] = []
    asks_for_chemicals: bool = False  # "what chemicals ..." -> chemicals_for
    trend_field: DateField | None = None
    limit: int | None = None
    mode: Literal["llm", "rules", "llm+rules"] = "rules"


# ---------------------------------------------------------------- resolution
class Candidate(BaseModel):
    entity_type: EntityType
    canonical_id: int | None
    display_name: str
    matched_alias: str
    score: float
    lexical: float = 0.0
    semantic: float | None = None
    method: Literal["exact", "cas", "alias", "fuzzy", "vector", "family"]
    n_products: int = 0
    extra: dict[str, Any] = {}


class Resolution(BaseModel):
    subtask_id: str
    mention: Mention
    status: Literal["resolved", "ambiguous", "not_found"]
    chosen: list[Candidate] = []
    candidates: list[Candidate] = []
    note: str | None = None


class ProductResolution(BaseModel):
    subtask_id: str
    mention: Mention
    status: Literal["resolved", "ambiguous", "not_found"]
    cdph_ids: list[int] = []
    candidates: list[dict[str, Any]] = []  # {product_name, cdph_ids, brand, company, score}
    note: str | None = None


# ---------------------------------------------------------------- query
class ToolCall(BaseModel):
    id: str
    subtask_id: str
    tool: str
    params: dict[str, Any]
    sql: str
    bound_params: list[Any]
    row_count: int
    result_sha1: str
    elapsed_ms: float


class ToolResult(BaseModel):
    call_id: str
    subtask_id: str
    tool: str
    description: str  # human readable filter description, e.g. "chemical = Acetaldehyde"
    rows: list[dict[str, Any]] = []
    totals: dict[str, Any] = {}
    evidence_row_ids: list[int] = []
    truncated: bool = False
    status: Literal["ok", "empty", "no_data_in_range", "not_found", "skipped"] = "ok"
    notes: list[str] = []


# ---------------------------------------------------------------- answer
class Fact(BaseModel):
    id: str
    statement: str
    value: Any = None
    call_id: str | None = None


class EvidenceItem(BaseModel):
    ref: str
    row_id: int
    cdph_id: int
    csf_id: int | None
    chemical_id: int
    cas_id: int
    fields: dict[str, Any]
    supports: list[str] = []


class WarningItem(BaseModel):
    code: str
    message: str
    severity: Literal["info", "warn", "error"] = "warn"
    row_ids: list[int] = []


class TraceEvent(BaseModel):
    step: int = 0
    agent: str
    mode: str = "deterministic"
    summary: str
    output: dict[str, Any] = {}
    elapsed_ms: float = 0.0


class Confidence(BaseModel):
    score: float
    level: Literal["high", "medium", "low"]
    rationale: str


class ClarificationOption(BaseModel):
    type: str
    name: str
    detail: str = ""
    canonical_id: int | None = None


class Clarification(BaseModel):
    question: str
    mention: str = ""
    options: list[ClarificationOption] = []


class Draft(BaseModel):
    answer_short: str
    answer_details: str
    details_mode: Literal["llm", "template"] = "template"


class Verification(BaseModel):
    passed: bool
    checks: dict[str, bool] = {}
    problems: list[str] = []


class Response(BaseModel):
    request_id: str
    response_type: Literal["answer", "clarification", "refusal", "no_data"]
    question: str
    answer_short: str
    answer_details: str
    evidence: list[EvidenceItem] = []
    facts: list[Fact] = []
    query_plan: list[TraceEvent] = []
    clarification: Clarification | None = None
    assumptions: list[str] = []
    warnings: list[WarningItem] = []
    confidence: Confidence
    meta: dict[str, Any] = Field(default_factory=dict)
