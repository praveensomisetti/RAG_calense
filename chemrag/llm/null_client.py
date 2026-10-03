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
    """Pick the LLM behind the one-method `LLMClient` interface. OpenAI is the default provider."""
    if disabled:
        return NullClient("--no-llm")
    provider = settings.llm_provider
    if provider == "none":
        return NullClient("CHEMRAG_LLM_PROVIDER=none")
    if provider == "openai":
        if not settings.openai_api_key:
            return NullClient("OPENAI_API_KEY not set")
        from chemrag.llm.openai_client import OpenAIClient

        try:
            return OpenAIClient(settings.openai_api_key, settings.llm_model, settings.llm_timeout_s,
                                reasoning_effort=settings.llm_reasoning_effort, base_url=settings.openai_base_url)
        except LLMUnavailable as e:
            return NullClient(str(e))
    if provider == "gemini":  # optional alternative: pip install -e ".[gemini]"
        if not settings.gemini_api_key:
            return NullClient("GEMINI_API_KEY not set")
        try:
            from chemrag.llm.gemini_client import GeminiClient

            return GeminiClient(settings.gemini_api_key, settings.llm_model, settings.llm_timeout_s)
        except Exception as e:  # noqa: BLE001 - missing optional package or bad config -> rules mode
            return NullClient(f"Gemini client unavailable: {e}")
    return NullClient(f"unknown CHEMRAG_LLM_PROVIDER={provider!r} (use openai, gemini or none)")
