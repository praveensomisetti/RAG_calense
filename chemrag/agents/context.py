from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from chemrag.query.tools import QueryEngine
from chemrag.retrieval.resolver import EntityResolver
from chemrag.schemas import TraceEvent
from chemrag.settings import Settings


@dataclass
class AgentContext:
    """Long-lived dependencies shared by graph nodes (kept out of the serialisable state)."""

    settings: Settings
    engine: QueryEngine
    resolver: EntityResolver
    llm: Any  # LLMClient


class Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def ms(self) -> float:
        return round((time.perf_counter() - self.t0) * 1000, 2)


def event(agent: str, summary: str, timer: Timer, mode: str = "deterministic", **output: Any) -> TraceEvent:
    return TraceEvent(agent=agent, mode=mode, summary=summary, output=output, elapsed_ms=timer.ms())
