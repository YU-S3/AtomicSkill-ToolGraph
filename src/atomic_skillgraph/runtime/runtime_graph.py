"""Runtime Graph 与 Task Execution Instance（设计文档 v2.0 §20、§43）。

只保存为执行记录，不进入长期 SkillGraph。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.binding_ir import BindingSpec
from ..core.edge_ir import GraphEdge
from ..core.refs import SkillRef
from ..core.status import EdgeType, ExecutionMode
from ..core.trace_ir import (NodeExecutionStatus, NodeValidationResult,
                             TaskExecutionInstance, TraceRecord)


@dataclass
class PlannedNode:
    """规划出的原子节点（执行前）。"""

    ref: SkillRef
    step_id: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    source: str = "retrieval"       # composite | retrieval | planner_llm
    target_effects: list[dict[str, Any]] = field(default_factory=list)
    dynamic: bool = False
    # Budget is part of the executable Runtime Graph semantics.  ``dynamic``
    # describes the execution route; it must not be used to guess whether this
    # occurrence is a whole-task fallback or a bounded local gap.
    budget_scope: str = "atomic"     # task | atomic | gap
    occurrence_id: str = ""
    origin_step_id: str = ""
    binding_specs: dict[str, BindingSpec] = field(default_factory=dict)
    branch_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"ref": str(self.ref), "step_id": self.step_id,
                "occurrence_id": self.occurrence_id,
                "origin_step_id": self.origin_step_id,
                "binding_specs": {key: value.to_dict()
                                  for key, value in self.binding_specs.items()},
                "branch_id": self.branch_id,
                "params": self.params, "source": self.source,
                "target_effects": self.target_effects, "dynamic": self.dynamic,
                "budget_scope": self.budget_scope}


@dataclass
class RuntimePlan:
    """一次任务的最小充分 Runtime Graph 规划。"""

    start_mode: str = "cold"              # cold | warm
    composite_ref: str = ""
    nodes: list[PlannedNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    audit: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_mode": self.start_mode,
            "composite_ref": self.composite_ref,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "retrieved": self.retrieved,
            "notes": self.notes,
            "audit": self.audit,
        }


@dataclass
class RuntimeNodeState:
    """运行时原子节点状态。"""

    ref: str = ""
    step_id: str = ""
    mode: ExecutionMode = ExecutionMode.DYNAMIC
    params: dict[str, Any] = field(default_factory=dict)
    impl_ref: str = ""
    tool_refs: list[str] = field(default_factory=list)
    validation: NodeValidationResult | None = None
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)
    passed: bool = False
    fallback_reason: str = ""
    target_effects: list[dict[str, Any]] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    occurrence_id: str = ""
    origin_step_id: str = ""
    outputs: dict[str, Any] = field(default_factory=dict)
    attempt_started: bool = False
    executed_action_count: int = 0
    execution_status: NodeExecutionStatus = NodeExecutionStatus.NOT_STARTED
    satisfied_without_execution: bool = False
    binding_provenance: dict[str, Any] = field(default_factory=dict)
    budget_scope: str = "atomic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "step_id": self.step_id,
            "mode": self.mode.value,
            "params": self.params,
            "impl_ref": self.impl_ref,
            "tool_refs": self.tool_refs,
            "validation": self.validation.to_dict() if self.validation else None,
            "before": self.before,
            "after": self.after,
            "passed": self.passed,
            "fallback_reason": self.fallback_reason,
            "target_effects": self.target_effects,
            "attempts": self.attempts,
            "occurrence_id": self.occurrence_id,
            "origin_step_id": self.origin_step_id,
            "outputs": self.outputs,
            "attempt_started": self.attempt_started,
            "executed_action_count": self.executed_action_count,
            "execution_status": self.execution_status.value,
            "satisfied_without_execution": self.satisfied_without_execution,
            "binding_provenance": self.binding_provenance,
            "budget_scope": self.budget_scope,
        }


class RuntimeGraph:
    """任务级运行图（§4.7）：当前任务的最小充分 Skill 子图 + 执行状态。"""

    def __init__(self, task_id: str, plan: RuntimePlan) -> None:
        self.task_id = task_id
        self.plan = plan
        self.nodes: list[RuntimeNodeState] = [
            RuntimeNodeState(ref=str(node.ref), step_id=node.step_id or f"step_{index:03d}",
                             occurrence_id=node.occurrence_id or
                             node.step_id or f"step_{index:03d}",
                             origin_step_id=node.origin_step_id,
                             params=dict(node.params),
                             target_effects=list(node.target_effects),
                             budget_scope=node.budget_scope)
            for index, node in enumerate(plan.nodes)
        ]
        self.edges: list[GraphEdge] = list(plan.edges) or self._sequential_edges()
        self.metrics: dict[str, Any] = {
            "direct_reuse_count": 0,
            "direct_attempt_count": 0,
            "direct_started_count": 0,
            "direct_success_count": 0,
            "seeded_generation_count": 0,
            "dynamic_generation_count": 0,
            "tool_calls": 0,
            "llm_tokens": 0,
        }

    def append_dynamic_gap(self, missing_effects: list[dict[str, Any]],
                           params: dict[str, Any], *,
                           occurrence_id: str = "task_gap_000") -> int:
        """Append one explicit, auditable task-level Dynamic occurrence."""
        index = len(self.plan.nodes)
        step_id = f"step_{index:03d}"
        planned = PlannedNode(
            ref=SkillRef("runtime.dynamic.task_gap", "0.0.0"),
            step_id=step_id, occurrence_id=occurrence_id,
            params=dict(params), source="task_gap",
            target_effects=[dict(item) for item in missing_effects],
            dynamic=True,
            budget_scope="gap",
        )
        previous = self.plan.nodes[-1] if self.plan.nodes else None
        self.plan.nodes.append(planned)
        state = RuntimeNodeState(
            ref=str(planned.ref), step_id=step_id,
            occurrence_id=occurrence_id, params=dict(params),
            target_effects=list(planned.target_effects),
            budget_scope=planned.budget_scope,
        )
        self.nodes.append(state)
        if previous is not None:
            edge = GraphEdge(
                source=str(previous.ref), target=str(planned.ref),
                type=EdgeType.NEXT, scope="runtime",
                source_step=previous.step_id, target_step=step_id,
                metadata={"task_id": self.task_id,
                          "reason": "explicit_task_gap"},
            )
            self.plan.edges.append(edge)
            self.edges.append(edge)
        return index

    def mark_fallback(self, index: int, mode: ExecutionMode, reason: str) -> None:
        node = self.nodes[index]
        node.mode = mode
        node.fallback_reason = reason

    def record_usage(self, mode: ExecutionMode) -> None:
        if mode == ExecutionMode.DIRECT:
            self.metrics["direct_attempt_count"] += 1
            self.metrics["direct_started_count"] += 1
        elif mode == ExecutionMode.SEEDED:
            self.metrics["seeded_generation_count"] += 1
        else:
            self.metrics["dynamic_generation_count"] += 1

    def add_tokens(self, usage) -> None:
        self.metrics["llm_tokens"] += int(getattr(usage, "total_tokens", 0))

    def record_direct_success(self) -> None:
        self.metrics["direct_success_count"] += 1
        # Backward-compatible metric now means realized successful reuse.
        self.metrics["direct_reuse_count"] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "composite_refs": [self.plan.composite_ref] if self.plan.composite_ref else [],
            "atomic_refs": [node.ref for node in self.nodes],
            "implementation_refs": [n.impl_ref for n in self.nodes if n.impl_ref],
            "tool_refs": sorted({t for n in self.nodes for t in n.tool_refs}),
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    def _sequential_edges(self) -> list[GraphEdge]:
        edges: list[GraphEdge] = []
        for left, right in zip(self.nodes, self.nodes[1:]):
            edges.append(GraphEdge(
                source=left.ref, target=right.ref, type=EdgeType.NEXT,
                scope="runtime", source_step=left.step_id,
                target_step=right.step_id,
                metadata={"task_id": self.task_id},
            ))
        return edges

    def build_execution_instance(self, task, trace_ref: str,
                                 benchmark_result: dict[str, Any],
                                 success: bool) -> TaskExecutionInstance:
        return TaskExecutionInstance(
            task_id=task.task_id,
            task_type=task.task_type,
            benchmark=task.benchmark,
            start_mode=self.plan.start_mode,
            runtime_graph=self.to_dict(),
            node_results=[n.to_dict() for n in self.nodes],
            benchmark_result=benchmark_result,
            trace_ref=trace_ref,
            success=success,
            metrics=dict(self.metrics),
        )

    def apply_to_trace(self, trace: TraceRecord) -> None:
        """把运行图信息写回 TraceRecord（§24 的 planned/realized 字段）。"""
        trace.planning_mode = "atomic_runtime" if self.plan.nodes else "pure_dynamic"
        trace.planned_atomic_nodes = [
            node.to_dict()
            for node in self.plan.nodes
        ]
        trace.realized_atomic_nodes = [n.to_dict() for n in self.nodes]
        trace.implementation_refs = [n.impl_ref for n in self.nodes if n.impl_ref]
        # Only a Direct attempt that actually started is Tool usage evidence.
        # Merely resolving/selecting a Tool must not create a call or failure.
        trace.tool_refs = sorted({
            str(tool_ref)
            for node in self.nodes
            for attempt in node.attempts
            if bool(attempt.get("started"))
            and str(attempt.get("mode") or "") == ExecutionMode.DIRECT.value
            for tool_ref in (attempt.get("tool_refs") or [])
            if tool_ref
        })
        trace.selected_composite = self.plan.composite_ref
        trace.retrieved_skill_refs = [h.get("ref", "") for h in self.plan.retrieved]
        trace.metrics.update(self.metrics)
