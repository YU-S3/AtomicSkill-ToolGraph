"""Trace 记录规范与 Task 执行实例（设计文档 v2.0 §20、§24）。

- TraceRecord：一次任务执行的可原子化记录（成功/失败都要支持事后学习）
- TaskExecutionInstance：Runtime 运行记录，不进入长期 SkillGraph
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .status import ErrorKind, ExecutionMode


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@dataclass
class ActionRecord:
    """单步环境动作记录。"""

    step: int = 0
    name: str = ""                    # 动作名（如 take / heat / go to）
    params: dict[str, Any] = field(default_factory=dict)
    observation: str = ""             # 执行后的观察
    accepted: bool = True             # 环境是否接受
    mode: ExecutionMode = ExecutionMode.DYNAMIC
    node_ref: str = ""                # 所属原子节点
    tool_ref: str = ""                # 使用的 Tool 引用（可为空）
    origin: str = "agent"             # agent | tool | framework_discovery

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "name": self.name,
            "params": self.params,
            "observation": self.observation,
            "accepted": self.accepted,
            "mode": self.mode.value,
            "node_ref": self.node_ref,
            "tool_ref": self.tool_ref,
            "origin": self.origin,
        }


@dataclass
class AttemptRecord:
    """一次候选生成/验证尝试（代码类任务）。"""

    index: int = 0
    stage: str = "draft"              # draft | repair | direct_tool
    candidate: str = ""
    passed: bool = False
    feedback: dict[str, Any] = field(default_factory=dict)
    failure_type: str = ""
    repair_source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "stage": self.stage,
            "candidate": self.candidate,
            "passed": self.passed,
            "feedback": self.feedback,
            "failure_type": self.failure_type,
            "repair_source": self.repair_source,
        }


@dataclass
class NodeValidationResult:
    """节点级验证结果（三级验证体系中的 atomic/composite 层）。"""

    node_ref: str = ""
    level: str = "atomic"             # atomic | composite | tool
    passed: bool = False
    checks: dict[str, bool] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_ref": self.node_ref,
            "level": self.level,
            "passed": self.passed,
            "checks": self.checks,
            "messages": self.messages,
            "before": self.before,
            "after": self.after,
        }


@dataclass
class TraceRecord:
    """v2.0 规范 Trace（§24）。"""

    trace_id: str = field(default_factory=lambda: new_id("trace"))
    task_id: str = ""
    task_type: str = ""
    task_goal: str = ""
    benchmark: str = ""
    start_mode: str = "cold"          # cold | warm
    planning_mode: str = "pure_dynamic"
    retrieved_skill_refs: list[str] = field(default_factory=list)
    selected_composite: str = ""
    planned_atomic_nodes: list[dict[str, Any]] = field(default_factory=list)
    realized_atomic_nodes: list[dict[str, Any]] = field(default_factory=list)
    implementation_refs: list[str] = field(default_factory=list)
    tool_refs: list[str] = field(default_factory=list)
    tool_parameters: list[dict[str, Any]] = field(default_factory=list)
    actions: list[ActionRecord] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    attempts: list[AttemptRecord] = field(default_factory=list)
    state_snapshots: list[dict[str, Any]] = field(default_factory=list)
    node_validators: list[NodeValidationResult] = field(default_factory=list)
    validation_layers: dict[str, Any] = field(default_factory=dict)
    candidate_code: str = ""
    benchmark_result: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    failure_type: str = ""
    token_cost: float = 0.0
    latency_ms: float = 0.0
    retries: int = 0
    provenance: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    # -- 便捷访问 ------------------------------------------------------------
    def final_state(self) -> dict[str, Any]:
        if self.state_snapshots:
            return dict(self.state_snapshots[-1].get("state") or {})
        return {}

    def initial_state(self) -> dict[str, Any]:
        if self.state_snapshots:
            return dict(self.state_snapshots[0].get("state") or {})
        return {}

    def direct_use_count(self) -> int:
        return int(self.metrics.get("direct_reuse_count", 0))

    def seeded_use_count(self) -> int:
        return int(self.metrics.get("seeded_generation_count", 0))

    def dynamic_use_count(self) -> int:
        return int(self.metrics.get("dynamic_generation_count", 0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "task_goal": self.task_goal,
            "benchmark": self.benchmark,
            "start_mode": self.start_mode,
            "planning_mode": self.planning_mode,
            "retrieved_skill_refs": self.retrieved_skill_refs,
            "selected_composite": self.selected_composite,
            "planned_atomic_nodes": self.planned_atomic_nodes,
            "realized_atomic_nodes": self.realized_atomic_nodes,
            "implementation_refs": self.implementation_refs,
            "tool_refs": self.tool_refs,
            "tool_parameters": self.tool_parameters,
            "actions": [a.to_dict() for a in self.actions],
            "observations": self.observations,
            "attempts": [a.to_dict() for a in self.attempts],
            "state_snapshots": self.state_snapshots,
            "node_validators": [v.to_dict() for v in self.node_validators],
            "validation_layers": self.validation_layers,
            "candidate_code": self.candidate_code,
            "benchmark_result": self.benchmark_result,
            "success": self.success,
            "failure_type": self.failure_type,
            "token_cost": self.token_cost,
            "latency_ms": self.latency_ms,
            "retries": self.retries,
            "provenance": self.provenance,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceRecord":
        return cls(
            trace_id=str(data.get("trace_id") or new_id("trace")),
            task_id=str(data.get("task_id", "")),
            task_type=str(data.get("task_type", "")),
            task_goal=str(data.get("task_goal", "")),
            benchmark=str(data.get("benchmark", "")),
            start_mode=str(data.get("start_mode", "cold")),
            planning_mode=str(data.get("planning_mode", "pure_dynamic")),
            retrieved_skill_refs=list(data.get("retrieved_skill_refs") or []),
            selected_composite=str(data.get("selected_composite", "")),
            planned_atomic_nodes=list(data.get("planned_atomic_nodes") or []),
            realized_atomic_nodes=list(data.get("realized_atomic_nodes") or []),
            implementation_refs=list(data.get("implementation_refs") or []),
            tool_refs=list(data.get("tool_refs") or []),
            tool_parameters=list(data.get("tool_parameters") or []),
            actions=[_action_from_dict(a) for a in (data.get("actions") or [])],
            observations=list(data.get("observations") or []),
            attempts=[AttemptRecord(**a) for a in (data.get("attempts") or [])],
            state_snapshots=list(data.get("state_snapshots") or []),
            node_validators=[NodeValidationResult(**v) for v in (data.get("node_validators") or [])],
            validation_layers=dict(data.get("validation_layers") or {}),
            candidate_code=str(data.get("candidate_code", "")),
            benchmark_result=dict(data.get("benchmark_result") or {}),
            success=bool(data.get("success", False)),
            failure_type=str(data.get("failure_type", "")),
            token_cost=float(data.get("token_cost", 0.0)),
            latency_ms=float(data.get("latency_ms", 0.0)),
            retries=int(data.get("retries", 0)),
            provenance=dict(data.get("provenance") or {}),
            metrics=dict(data.get("metrics") or {}),
        )


def _action_from_dict(data: dict[str, Any]) -> ActionRecord:
    mode_raw = data.get("mode", "dynamic")
    try:
        mode = ExecutionMode(mode_raw)
    except ValueError:
        mode = ExecutionMode.DYNAMIC
    return ActionRecord(
        step=int(data.get("step", 0)),
        name=str(data.get("name", "")),
        params=dict(data.get("params") or {}),
        observation=str(data.get("observation", "")),
        accepted=bool(data.get("accepted", True)),
        mode=mode,
        node_ref=str(data.get("node_ref", "")),
        tool_ref=str(data.get("tool_ref", "")),
        origin=str(data.get("origin", "agent")),
    )


@dataclass
class TaskExecutionInstance:
    """单次任务执行记录（§20），不进入长期 SkillGraph。"""

    execution_id: str = field(default_factory=lambda: new_id("exec"))
    task_id: str = ""
    task_type: str = ""
    benchmark: str = ""
    start_mode: str = "cold"
    runtime_graph: dict[str, Any] = field(default_factory=dict)
    node_results: list[dict[str, Any]] = field(default_factory=list)
    benchmark_result: dict[str, Any] = field(default_factory=dict)
    trace_ref: str = ""
    success: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "benchmark": self.benchmark,
            "start_mode": self.start_mode,
            "runtime_graph": self.runtime_graph,
            "node_results": self.node_results,
            "benchmark_result": self.benchmark_result,
            "trace_ref": self.trace_ref,
            "success": self.success,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskExecutionInstance":
        return cls(
            execution_id=str(data.get("execution_id") or new_id("exec")),
            task_id=str(data.get("task_id", "")),
            task_type=str(data.get("task_type", "")),
            benchmark=str(data.get("benchmark", "")),
            start_mode=str(data.get("start_mode", "cold")),
            runtime_graph=dict(data.get("runtime_graph") or {}),
            node_results=list(data.get("node_results") or []),
            benchmark_result=dict(data.get("benchmark_result") or {}),
            trace_ref=str(data.get("trace_ref", "")),
            success=bool(data.get("success", False)),
            metrics=dict(data.get("metrics") or {}),
            timestamp=str(data.get("timestamp", "")),
        )


def error_kind_of(failure_type: str) -> ErrorKind:
    """把字符串 failure_type 映射到 §34 的错误分类。"""
    for kind in ErrorKind:
        if kind.value == failure_type:
            return kind
    return ErrorKind.UNKNOWN
