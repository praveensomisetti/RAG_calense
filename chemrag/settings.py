"""Runtime configuration, read once from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def _path(env: str, default: str) -> Path:
    p = Path(os.getenv(env, default))
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class Thresholds:
    resolve_high: float = 0.90  # >= : resolved without comment
    resolve_low: float = 0.75  # >= : resolved with an assumption; below -> not_found
    ambiguity_margin: float = 0.05  # top1 - top2 < margin (different entities) -> ambiguous
    lexical_weight: float = 0.65  # fused = w*lexical + (1-w)*semantic
    dominance_share: float = 0.50  # one chemical > 50% of matched products -> warning


@dataclass(frozen=True)
class Settings:
    csv_path: Path
    db_path: Path
    vector_path: Path
    groups_path: Path
    runs_dir: Path
    llm_provider: str
    llm_model: str
    llm_timeout_s: float
    gemini_api_key: str | None
    embed_model: str
    embed_batch: int
    torch_threads: int
    duckdb_memory_limit: str
    duckdb_threads: int
    list_limit: int = 20
    max_list_limit: int = 200
    evidence_limit: int = 25
    max_question_chars: int = 1000
    max_subtasks: int = 4
    thresholds: Thresholds = field(default_factory=Thresholds)

    @property
    def manifest_path(self) -> Path:
        return self.vector_path.parent / "index_manifest.json"

    @property
    def llm_enabled(self) -> bool:
        return self.llm_provider == "gemini" and bool(self.gemini_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    provider = os.getenv("CHEMRAG_LLM_PROVIDER", "gemini").strip().lower()
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None
    return Settings(
        csv_path=_path("CHEMRAG_CSV_PATH", "data/raw/interviewtestdataset.csv"),
        db_path=_path("CHEMRAG_DB_PATH", "data/processed/cscp.duckdb"),
        vector_path=_path("CHEMRAG_VECTOR_PATH", "data/processed/chroma"),
        groups_path=_path("CHEMRAG_GROUPS_PATH", "config/chemical_groups.yaml"),
        runs_dir=_path("CHEMRAG_RUNS_DIR", "runs"),
        llm_provider=provider,
        llm_model=os.getenv("CHEMRAG_LLM_MODEL", "gemini-3.8-flash"),
        llm_timeout_s=float(os.getenv("CHEMRAG_LLM_TIMEOUT_S", "20")),
        gemini_api_key=key,
        embed_model=os.getenv("CHEMRAG_EMBED_MODEL", "thenlper/gte-large"),
        embed_batch=int(os.getenv("CHEMRAG_EMBED_BATCH", "32")),
        torch_threads=int(os.getenv("CHEMRAG_TORCH_THREADS", str(min(4, os.cpu_count() or 1)))),
        duckdb_memory_limit=os.getenv("CHEMRAG_DUCKDB_MEMORY_LIMIT", "1GB"),
        duckdb_threads=int(os.getenv("CHEMRAG_DUCKDB_THREADS", "2")),
    )
