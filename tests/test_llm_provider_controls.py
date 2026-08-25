from __future__ import annotations

import json
import sys
from pathlib import Path

FLOWEVO_SRC = Path(__file__).resolve().parents[1] / "src" / "flowevo"
if str(FLOWEVO_SRC) not in sys.path:
    sys.path.insert(0, str(FLOWEVO_SRC))

from runtime.config import (  # noqa: E402
    GenerationSettings,
    RuntimeLLMConfig,
    SkillContextBudgets,
)
from runtime.llm_client import LLMClient  # noqa: E402


def _config() -> RuntimeLLMConfig:
    settings = GenerationSettings(temperature=0.0, max_output_tokens=256)
    return RuntimeLLMConfig(
        provider="openrouter", api_key="test-key",
        base_url="https://api.deepseek.com", model="deepseek-v4-flash",
        app_name="test", skill_top_k=1,
        skill_context_budgets=SkillContextBudgets(1, 1, 1, 1),
        draft=settings, repair=settings, config_path="", local_override_path="",
    )


class _Response:
    ok = True
    status_code = 200
    headers = {}
    text = "{}"

    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


class _StreamResponse(_Response):
    def __init__(self, chunks):
        super().__init__({})
        self._chunks = chunks

    def iter_lines(self, decode_unicode=True):
        for chunk in self._chunks:
            yield "data: " + json.dumps(chunk)
        yield "data: [DONE]"


def test_alfworld_can_disable_deepseek_thinking(monkeypatch):
    payloads = []

    def post(_url, *, headers, json, timeout):
        payloads.append(dict(json))
        return _Response({
            "choices": [{"message": {"content": "Act: look"},
                         "finish_reason": "stop"}],
            "usage": {},
        })

    monkeypatch.setattr("runtime.llm_client.requests.post", post)
    client = LLMClient(_config())
    result = client.generate(
        instructions="robot", input_text="choose",
        settings=GenerationSettings(temperature=0.0, max_output_tokens=256,
                                    thinking="disabled"),
    )
    assert result.text == "Act: look"
    assert payloads[0]["thinking"] == {"type": "disabled"}


def test_length_retry_increases_large_budget_instead_of_reducing(monkeypatch):
    payloads = []

    def post(_url, *, headers, json, timeout):
        payloads.append(dict(json))
        if len(payloads) == 1:
            return _Response({
                "choices": [{"message": {"content": "", "reasoning_content": "thinking"},
                             "finish_reason": "length"}],
            })
        return _Response({
            "choices": [{"message": {"content": "done"},
                         "finish_reason": "stop"}], "usage": {},
        })

    monkeypatch.setattr("runtime.llm_client.requests.post", post)
    client = LLMClient(_config())
    monkeypatch.setattr(client, "_retry_backoff", lambda *args, **kwargs: None)
    result = client.generate(
        instructions="solve", input_text="problem",
        settings=GenerationSettings(temperature=0.0, max_output_tokens=2048),
    )
    assert result.text == "done"
    assert [payload["max_tokens"] for payload in payloads] == [2048, 4096]


def test_extractor_reasoning_length_falls_back_to_non_thinking_json(monkeypatch):
    payloads = []
    timeouts = []

    def post(_url, *, headers, json, timeout):
        payloads.append(dict(json))
        timeouts.append(timeout)
        if len(payloads) == 1:
            return _Response({
                "choices": [{
                    "message": {"content": "", "reasoning_content": "long reasoning"},
                    "finish_reason": "length",
                }],
            })
        return _Response({
            "choices": [{"message": {"content": '{"phases": []}'},
                         "finish_reason": "stop"}], "usage": {},
        })

    monkeypatch.setattr("runtime.llm_client.requests.post", post)
    client = LLMClient(_config())
    monkeypatch.setattr(client, "_retry_backoff", lambda *args, **kwargs: None)
    result = client.generate(
        instructions="return json", input_text="events",
        settings=GenerationSettings(
            temperature=0.1, max_output_tokens=8192, thinking="enabled",
            reasoning_effort="low",
            request_timeout_seconds=120,
            fallback_disable_thinking_on_length=True,
            fallback_response_format="json_object"),
    )

    assert result.text == '{"phases": []}'
    assert payloads[0]["thinking"] == {"type": "enabled"}
    assert payloads[0]["reasoning_effort"] == "low"
    assert "response_format" not in payloads[0]
    assert payloads[1]["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in payloads[1]
    assert payloads[1]["response_format"] == {"type": "json_object"}
    assert [payload["max_tokens"] for payload in payloads] == [8192, 8192]
    assert timeouts == [(10.0, 120.0), (10.0, 120.0)]


def test_streaming_preserves_complete_reasoning_and_final_content(monkeypatch):
    payloads = []

    def post(_url, *, headers, json, timeout, stream):
        payloads.append(dict(json))
        return _StreamResponse([
            {"choices": [{"delta": {"reasoning_content": "reason part 1 "},
                           "finish_reason": None}]},
            {"choices": [{"delta": {"reasoning_content": "part 2"},
                           "finish_reason": None}]},
            {"choices": [{"delta": {"content": '{"ok":true}'},
                           "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 10, "completion_tokens": 8,
                       "total_tokens": 18,
                       "completion_tokens_details": {"reasoning_tokens": 5}}},
        ])

    monkeypatch.setattr("runtime.llm_client.requests.post", post)
    client = LLMClient(_config())
    result = client.generate(
        instructions="extract", input_text="events",
        settings=GenerationSettings(
            temperature=0.1, max_output_tokens=384000,
            thinking="enabled", reasoning_effort="low",
            request_timeout_seconds=120, stream_response=True),
    )

    assert result.text == '{"ok":true}'
    assert result.reasoning_text == "reason part 1 part 2"
    assert result.reasoning_tokens == 5
    assert result.finish_reason == "stop"
    assert payloads[0]["stream"] is True
    assert payloads[0]["max_tokens"] == 384000
