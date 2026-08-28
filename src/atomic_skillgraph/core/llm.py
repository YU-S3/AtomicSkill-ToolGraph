"""LLM 协议：v2.0 层不直接依赖 FlowEvo 的 LLMClient。

真实 API 与 Mock 都实现该协议，便于无 API smoke 与真实实验切换。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


LLM_USAGE_FIELDS = (
    "prompt_tokens", "completion_tokens", "total_tokens",
    "reasoning_tokens", "call_count", "latency_ms",
)


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    provider: str = ""
    model: str = ""
    reasoning_text: str = ""
    reasoning_tokens: int = 0
    finish_reason: str = ""


@dataclass
class LLMUsage:
    """累计 token/调用统计。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    call_count: int = 0
    latency_ms: float = 0.0

    def add(self, response: LLMResponse) -> None:
        self.prompt_tokens += response.prompt_tokens
        self.completion_tokens += response.completion_tokens
        self.total_tokens += response.total_tokens
        self.reasoning_tokens += response.reasoning_tokens
        self.call_count += 1
        self.latency_ms += response.latency_ms

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "call_count": self.call_count,
            "latency_ms": round(self.latency_ms, 1),
        }


def snapshot_llm_usage(llm) -> dict[str, int | float]:
    """Take a provider-independent cumulative usage snapshot."""
    usage = getattr(llm, "usage", None)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "reasoning_tokens": int(getattr(usage, "reasoning_tokens", 0) or 0),
        "call_count": int(getattr(usage, "call_count", 0) or 0),
        "latency_ms": float(getattr(usage, "latency_ms", 0.0) or 0.0),
    }


def diff_llm_usage(before: dict[str, int | float],
                   after: dict[str, int | float]) -> dict[str, int | float]:
    """Return a non-negative per-stage delta between two snapshots."""
    delta: dict[str, int | float] = {}
    for field_name in LLM_USAGE_FIELDS:
        value = max(0.0, float(after.get(field_name, 0) or 0)
                    - float(before.get(field_name, 0) or 0))
        delta[field_name] = (value if field_name == "latency_ms"
                             else int(value))
    return delta


def record_llm_usage_bucket(metrics: dict, bucket: str,
                            delta: dict[str, int | float]) -> None:
    """Accumulate one agent/stage delta in an auditable trace metric."""
    usage_by_agent = metrics.setdefault("llm_usage_by_agent", {})
    current = dict(usage_by_agent.get(bucket) or {})
    for field_name in LLM_USAGE_FIELDS:
        value = float(current.get(field_name, 0) or 0) + float(
            delta.get(field_name, 0) or 0)
        current[field_name] = (value if field_name == "latency_ms"
                               else int(value))
    current["latency_ms"] = round(float(current["latency_ms"]), 3)
    usage_by_agent[bucket] = current


@runtime_checkable
class LLM(Protocol):
    """统一的生成协议：instructions=system 指令，input_text=任务输入。

    temperature 为可选覆盖（如交互环境用贪心解码 temperature=0.0，
    与 FlowEvo 的 ALFWorld 生成设置一致）。
    """

    def generate(self, instructions: str, input_text: str,
                 *, temperature: float | None = None,
                 thinking: str | None = None,
                 structured_output: bool = False) -> LLMResponse:
        """生成一次补全。"""
        ...

    @property
    def usage(self) -> LLMUsage:
        """累计用量（用于成本指标）。"""
        ...
