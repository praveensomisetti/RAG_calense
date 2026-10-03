"""OpenAIClient without network: a fake SDK client checks request shaping, parsing, fallbacks and errors."""

from types import SimpleNamespace

import httpx
import openai
import pytest
from openai.lib._pydantic import to_strict_json_schema

from chemrag.llm.base import LLMExtraction, LLMNarrative, LLMPlan, LLMSubTask, LLMUnavailable
from chemrag.llm.openai_client import OpenAIClient


def test_llm_schemas_are_strict_structured_output_compatible():
    for schema in (LLMPlan, LLMExtraction, LLMNarrative):
        js = to_strict_json_schema(schema)
        assert js["additionalProperties"] is False and set(js["required"]) == set(js["properties"])


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def parse(self, **kw):
        self.calls.append(kw)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        msg = SimpleNamespace(parsed=out, refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))


def client_with(model, outcomes):
    comp = FakeCompletions(outcomes)
    fake = SimpleNamespace(chat=SimpleNamespace(completions=comp))
    return OpenAIClient("sk-test", model, client=fake), comp


def bad_request(msg):
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.BadRequestError(msg, response=httpx.Response(400, request=req), body=None)


def test_reasoning_model_uses_reasoning_effort_not_temperature():
    c, comp = client_with("gpt-5-mini", [LLMSubTask(text="q", intent="lookup")])
    out = c.structured("sys", "user", LLMSubTask)
    assert out.intent == "lookup"
    kw = comp.calls[0]
    assert kw["reasoning_effort"] == "low" and "temperature" not in kw and kw["response_format"] is LLMSubTask
    assert c.usage == {"input_tokens": 100, "output_tokens": 20}


def test_non_reasoning_model_uses_temperature_zero():
    c, comp = client_with("gpt-4.1-mini", [LLMSubTask(text="q", intent="list")])
    c.structured("sys", "user", LLMSubTask)
    assert comp.calls[0]["temperature"] == 0 and "reasoning_effort" not in comp.calls[0]


def test_unsupported_parameter_is_dropped_and_retried():
    c, comp = client_with("gpt-5-mini", [bad_request("Unsupported value: 'reasoning_effort'"),
                                         LLMSubTask(text="q", intent="trend")])
    assert c.structured("s", "u", LLMSubTask).intent == "trend"
    assert "reasoning_effort" not in comp.calls[1]


def test_failures_become_llm_unavailable():
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    err = openai.RateLimitError("quota", response=httpx.Response(429, request=req), body=None)
    c, _ = client_with("gpt-5-mini", [err])
    with pytest.raises(LLMUnavailable):
        c.structured("s", "u", LLMSubTask)


def test_hard_failure_opens_circuit():
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    c, _ = client_with("gpt-5-mini", [openai.APIConnectionError(request=req)])
    with pytest.raises(LLMUnavailable):
        c.structured("s", "u", LLMSubTask)
    assert c.available is False
