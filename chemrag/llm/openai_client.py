"""OpenAI via the official `openai` SDK: Structured Outputs (`chat.completions.parse` with a Pydantic schema).

* Reasoning-family models (gpt-5*, o-series) do not accept `temperature`; they get `reasoning_effort`
  (default "low": these are short classification/extraction tasks, so deep reasoning only adds cost/latency).
* Other models get `temperature=0` for repeatable output.
* If the API rejects either parameter for a given model, it is dropped and the call retried once.
* 429/5xx/timeouts are retried by the SDK itself (`max_retries`); anything that still fails raises
  LLMUnavailable, and the calling agent falls back to deterministic rules.
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from chemrag.llm.base import LLMUnavailable

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

REASONING_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")


class OpenAIClient:
    def __init__(self, api_key: str, model: str, timeout_s: float = 20.0, max_retries: int = 2,
                 reasoning_effort: str | None = "low", base_url: str | None = None, client: Any = None):
        try:
            import openai
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise LLMUnavailable("the `openai` package is not installed") from e
        self._openai = openai
        self.client = client or OpenAI(api_key=api_key, timeout=timeout_s, max_retries=max_retries,
                                       base_url=base_url or None)
        self.model = model
        self.name = f"openai:{model}"
        self.available = True
        self.reasoning_effort = reasoning_effort or None
        self.is_reasoning = model.startswith(REASONING_PREFIXES)
        self.calls = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    def _kwargs(self, system: str, user: str, schema: type[BaseModel]) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": schema,
        }
        if self.is_reasoning:
            if self.reasoning_effort:
                kw["reasoning_effort"] = self.reasoning_effort
        else:
            kw["temperature"] = 0
        return kw

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        oa = self._openai
        kw = self._kwargs(system, user, schema)
        for _ in range(3):
            try:
                self.calls += 1
                resp = self.client.chat.completions.parse(**kw)
                if getattr(resp, "usage", None):
                    self.usage["input_tokens"] += resp.usage.prompt_tokens or 0
                    self.usage["output_tokens"] += resp.usage.completion_tokens or 0
                msg = resp.choices[0].message
                if getattr(msg, "refusal", None):
                    raise LLMUnavailable(f"model refused: {msg.refusal}")
                if msg.parsed is None:
                    raise LLMUnavailable("no structured output returned")
                return msg.parsed if isinstance(msg.parsed, schema) else schema.model_validate(msg.parsed)
            except oa.BadRequestError as e:
                text = str(e).lower()
                if "temperature" in kw and "temperature" in text:
                    kw.pop("temperature")
                    continue
                if "reasoning_effort" in kw and "reasoning" in text:
                    kw.pop("reasoning_effort")
                    continue
                raise LLMUnavailable(f"OpenAI rejected the request: {e}") from e
            except (oa.LengthFinishReasonError, oa.ContentFilterFinishReasonError, ValidationError) as e:
                raise LLMUnavailable(f"OpenAI output unusable: {type(e).__name__}") from e
            except (oa.AuthenticationError, oa.PermissionDeniedError, oa.NotFoundError, oa.APIConnectionError) as e:
                # Hard failures (bad key, unknown model, no network) will not fix themselves within this run:
                # open the circuit so the remaining agents go straight to rules instead of waiting on retries.
                self.available = False
                raise LLMUnavailable(f"OpenAI unavailable ({type(e).__name__}): {e}") from e
            except oa.OpenAIError as e:  # rate limit after SDK retries, timeout, 5xx
                raise LLMUnavailable(f"OpenAI call failed: {type(e).__name__}: {e}") from e
        raise LLMUnavailable("OpenAI call failed after parameter fallbacks")

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self.client.models.list())
