from __future__ import annotations

from chemrag.llm.base import LLMUnavailable
from chemrag.settings import Settings


class NullClient:
    """Used when no API key is configured or --no-llm is passed: every agent falls back to rules."""

    name = "none"
    available = False

    def __init__(self, reason: str = "LLM disabled"):
        self.reason = reason

    def structured(self, system, user, schema):
        raise LLMUnavailable(self.reason)


def make_llm(settings: Settings, disabled: bool = False):
    if disabled:
        return NullClient("--no-llm")
    if settings.llm_provider == "none":
        return NullClient("CHEMRAG_LLM_PROVIDER=none")
    if not settings.gemini_api_key:
        return NullClient("GEMINI_API_KEY not set")
    from chemrag.llm.gemini_client import GeminiClient

    try:
        return GeminiClient(settings.gemini_api_key, settings.llm_model, settings.llm_timeout_s)
    except Exception as e:  # pragma: no cover
        return NullClient(f"Gemini client init failed: {e}")
