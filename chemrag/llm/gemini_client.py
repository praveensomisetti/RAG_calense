"""Gemini via the official `google-genai` SDK, JSON-schema constrained output, validated with Pydantic."""

from __future__ import annotations

import json
import logging
import time
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from chemrag.llm.base import LLMUnavailable

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class GeminiClient:
    def __init__(self, api_key: str, model: str, timeout_s: float = 20.0, max_retries: int = 2):
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:  # pragma: no cover
            raise LLMUnavailable("google-genai is not installed") from e
        self._types = types
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
        self.model = model
        self.name = f"gemini:{model}"
        self.available = True
        self.max_retries = max_retries
        self.calls = 0

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        types = self._types
        cfg = types.GenerateContentConfig(
            system_instruction=system, temperature=0.0, response_mime_type="application/json",
            response_schema=schema,
        )
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                self.calls += 1
                resp = self.client.models.generate_content(model=self.model, contents=user, config=cfg)
                parsed = getattr(resp, "parsed", None)
                if isinstance(parsed, schema):
                    return parsed
                return schema.model_validate(json.loads(resp.text or ""))
            except (ValidationError, json.JSONDecodeError) as e:
                last = e  # one more try, then give up -> caller falls back to rules
                log.warning("Gemini output failed validation (%s)", e)
            except Exception as e:  # 429 / 5xx / timeout / auth
                last = e
                msg = str(e)
                if any(code in msg for code in ("401", "403", "API key", "PERMISSION_DENIED", "NOT_FOUND", "404")):
                    break  # not retryable
                time.sleep(min(8.0, 1.5 * 2 ** attempt))
        raise LLMUnavailable(f"Gemini call failed: {type(last).__name__}: {last}")

    def list_models(self) -> list[str]:
        return sorted(m.name for m in self.client.models.list())
