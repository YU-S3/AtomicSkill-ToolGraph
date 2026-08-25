"""Benchmark Adapter 协议（设计文档 v2.0 §52）。

Adapter 必须暴露：
    load_tasks() / parse_task_type() / run_environment_action() / get_observation()
    / verify_task() / extract_state_summary() / validate_atomic_effect()
    / compile_tool_artifact() / replay_tool()
不要求所有 Benchmark 实现完全相同（§52）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..core.llm import LLM, LLMUsage


# ---------------------------------------------------------------------------
# 规范任务模型（v2.0 内部统一任务表示）
# ---------------------------------------------------------------------------

@dataclass
class Task:
    task_id: str
    benchmark: str
    task_type: str = ""
    goal: str = ""
    context: dict[str, Any] = field(default_factory=dict)   # benchmark 特定载荷
    state: dict[str, Any] = field(default_factory=dict)     # 初始状态快照
    target_effects: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "benchmark": self.benchmark,
            "task_type": self.task_type,
            "goal": self.goal,
            "context": self.context,
            "state": self.state,
            "target_effects": self.target_effects,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        return cls(
            task_id=str(data.get("task_id", "")),
            benchmark=str(data.get("benchmark", "")),
            task_type=str(data.get("task_type", "")),
            goal=str(data.get("goal", "")),
            context=dict(data.get("context") or {}),
            state=dict(data.get("state") or {}),
            target_effects=list(data.get("target_effects") or []),
            metadata=dict(data.get("metadata") or {}),
        )


# ---------------------------------------------------------------------------
# 验证反馈
# ---------------------------------------------------------------------------

@dataclass
class VerifyResult:
    passed: bool = False
    tests: list[str] = field(default_factory=list)
    executed_code: str = ""
    stdout: str = ""
    stderr: str = ""
    timeout: bool = False
    feedback: dict[str, Any] = field(default_factory=dict)
    failure_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "tests": self.tests,
            "executed_code": self.executed_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timeout": self.timeout,
            "feedback": self.feedback,
            "failure_type": self.failure_type,
        }


# ---------------------------------------------------------------------------
# 代码生成结果（code/math 类）
# ---------------------------------------------------------------------------

@dataclass
class CodeAttempt:
    index: int = 0
    stage: str = "draft"
    code: str = ""
    verify: VerifyResult = field(default_factory=VerifyResult)
    usage: LLMUsage = field(default_factory=LLMUsage)

    @property
    def passed(self) -> bool:
        return self.verify.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "stage": self.stage,
            "code": self.code,
            "verify": self.verify.to_dict(),
            "usage": self.usage.to_dict(),
        }


@dataclass
class CodeRunResult:
    success: bool = False
    attempts: list[CodeAttempt] = field(default_factory=list)
    candidate_code: str = ""
    feedback: VerifyResult = field(default_factory=VerifyResult)
    failure_type: str = ""
    first_attempt_success: bool = False
    retry_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "attempts": [a.to_dict() for a in self.attempts],
            "candidate_code": self.candidate_code,
            "feedback": self.feedback.to_dict(),
            "failure_type": self.failure_type,
            "first_attempt_success": self.first_attempt_success,
            "retry_count": self.retry_count,
        }


# ---------------------------------------------------------------------------
# 交互环境单步结果与 episode 结果
# ---------------------------------------------------------------------------

@dataclass
class EnvStepResult:
    observation: str = ""
    score: float = 0.0
    done: bool = False
    won: bool = False
    admissible_commands: list[str] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    accepted: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation,
            "score": self.score,
            "done": self.done,
            "won": self.won,
            "admissible_commands": self.admissible_commands,
            "state": self.state,
            "accepted": self.accepted,
        }


@dataclass
class EnvRunResult:
    success: bool = False
    actions: list[dict[str, Any]] = field(default_factory=list)      # ActionRecord.to_dict()
    states: list[dict[str, Any]] = field(default_factory=list)       # state snapshots
    steps: int = 0
    failure_type: str = ""
    usage: LLMUsage = field(default_factory=LLMUsage)
    direct_used: bool = False
    seeded_used: bool = False
    dynamic_used: bool = False
    final_observation: str = ""
    # 原地降级（in-place fallback）续跑载荷：失败点的当前观察与可行动作
    current_observation: str = ""
    current_admissible: list[str] = field(default_factory=list)
    atomic_complete: bool = False
    # API/网络/超时异常属于基础设施证据，不与任务动作失败混在一起。
    infrastructure_errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "actions": self.actions,
            "states": self.states,
            "steps": self.steps,
            "failure_type": self.failure_type,
            "usage": self.usage.to_dict(),
            "direct_used": self.direct_used,
            "seeded_used": self.seeded_used,
            "dynamic_used": self.dynamic_used,
            "final_observation": self.final_observation,
            "current_observation": self.current_observation,
            "current_admissible": self.current_admissible,
            "atomic_complete": self.atomic_complete,
            "infrastructure_errors": self.infrastructure_errors,
        }


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """§52 要求的适配器协议。"""

    name: str

    def load_tasks(self, limit: int, task_type: str | None = None) -> list[Task]: ...

    def parse_task_type(self, task: Task) -> str: ...

    def verify_task(self, task: Task, candidate: str) -> VerifyResult: ...

    def replay_tool(self, tool, bindings: dict[str, Any],
                    before: dict[str, Any]) -> dict[str, Any]: ...

    def execute_tool(self, task: Task, tool, parameters: dict[str, Any],
                     state: dict[str, Any]) -> dict[str, Any]: ...

    def generate_code(self, task: Task, llm: LLM, *,
                      seed_context: str = "",
                      max_repairs: int = 2) -> CodeRunResult: ...

    def run_env_episode(self, task: Task, llm: LLM, *,
                        seed_context: str = "",
                        direct_steps: list[dict[str, Any]] | None = None,
                        max_steps: int = 50,
                        resume: dict[str, Any] | None = None,
                        stop_effects: list[dict[str, Any]] | None = None,
                        effect_inputs: dict[str, Any] | None = None,
                        node_ref: str = "",
                        phase_goal: str = "") -> EnvRunResult: ...


# ---------------------------------------------------------------------------
# 通用 goal 解析工具（弱参数提取；ALFWorld 真实环境可复用 vendored param_extractor）
# ---------------------------------------------------------------------------

def parse_goal_params(goal: str, input_names: list[str]) -> dict[str, Any]:
    """从 goal 文本中弱提取对象/位置参数（用于原子节点参数绑定兜底）。"""
    params: dict[str, Any] = {}
    goal_lower = goal.lower()
    for name in input_names:
        lower = name.lower()
        if lower in ("object_location", "source_location", "source"):
            match = re.search(r"\bfrom\s+(?:the\s+)?([a-z]+(?:\s+\d+)?)", goal_lower)
            if match:
                params[name] = match.group(1).strip()
        elif "location" in lower or lower in ("recep", "target", "target_location", "sink"):
            # 单词边界避免把 ``heat an object`` 末尾的 ``at`` 误当成
            # 位置介词，从而错误绑定 target_location="an"。
            match = re.search(r"\b(?:in|on|into|to|at)\b\s+(?:the\s+)?([a-z0-9 ]+?)(?:\.|$|\s)", goal_lower)
            if match and match.group(1).strip():
                params[name] = match.group(1).strip()
            else:
                recep_match = re.search(r"(countertop|shelf|drawer|cabinet|fridge|microwave|sink|garbagecan|toilet|bathtub|stoveburner|coffeemachine|dresser|desk|sidetable|ottoman|bed|safe|box|basin|cart|laundryhamper|pan|pot|bowl|plate|mug|cup|bottle|knife|fork|spoon|butterknife|plate|lettuce|tomato|potato|apple|bread|egg|laptop|cellphone|book|newspaper|pen|pencil|creditcard|key|remotecontrol|watch|tissue|soap|candle|cloth|spraybottle|teddybear|statue|vase)", goal_lower)
                if recep_match:
                    params[name] = recep_match.group(1).strip()
        elif "object" in lower or lower in ("obj", "target_object", "item"):
            # 取第一个名词（排除容器/位置词）
            obj_match = re.search(r"(?:take|get|pick|put|move|clean|heat|cool|examine|use|look at|place|find|grab)\s+(?:up\s+)?(?:(?:the|a|an|some)\s+)?([a-z0-9 ]+?)(?:\s+(?:and|with|from|in|on|to|into|at|then)|\.|$|\s)", goal_lower)
            if obj_match and obj_match.group(1).strip():
                params[name] = obj_match.group(1).strip()
    return {k: v for k, v in params.items() if v}
