"""Runtime Graph 与 Task Execution Instance（设计文档 v2.0 §20、§43）。

只保存为执行记录，不进入长期 SkillGraph。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..core.binding_ir import BindingSpec, binding_slot_name
from ..core.edge_ir import GraphEdge
from ..core.predicates import (
    distinct_claims_conflict, normalize_value, ordered_predicate_args,
    predicate_fact_signature)
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
    # A role may belong to one or more task-level cardinality groups.  Values
    # realized by a validated occurrence in one branch are excluded from the
    # same group in later branches.  Keeping this in Runtime IR makes instance
    # distinctness an executable/auditable constraint instead of a prompt hint.
    distinct_bindings: dict[str, list[str]] = field(default_factory=dict)
    # Branch identity is scoped by distinctness group.  A node/role can
    # participate in multiple cardinality contracts whose occurrence
    # boundaries are not necessarily the same.
    distinct_branch_ids: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"ref": str(self.ref), "step_id": self.step_id,
                "occurrence_id": self.occurrence_id,
                "origin_step_id": self.origin_step_id,
                "binding_specs": {key: value.to_dict()
                                  for key, value in self.binding_specs.items()},
                "branch_id": self.branch_id,
                "distinct_bindings": {
                    key: list(value)
                    for key, value in self.distinct_bindings.items()
                },
                "distinct_branch_ids": dict(self.distinct_branch_ids),
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
    # A cardinality Gap may validate several witnesses in one occurrence.
    # Keep those collection-valued results separate from scalar ``outputs``:
    # ordinary DATA_FLOW edges continue to consume one materialized value,
    # while distinctness/exclusion logic can explicitly consume this set.
    distinct_witness_outputs: dict[str, list[Any]] = field(
        default_factory=dict)
    attempt_started: bool = False
    executed_action_count: int = 0
    execution_status: NodeExecutionStatus = NodeExecutionStatus.NOT_STARTED
    satisfied_without_execution: bool = False
    binding_provenance: dict[str, Any] = field(default_factory=dict)
    budget_scope: str = "atomic"
    distinct_bindings: dict[str, list[str]] = field(default_factory=dict)
    distinct_branch_ids: dict[str, str] = field(default_factory=dict)

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
            "distinct_witness_outputs": {
                role: list(values)
                for role, values in self.distinct_witness_outputs.items()
            },
            "attempt_started": self.attempt_started,
            "executed_action_count": self.executed_action_count,
            "execution_status": self.execution_status.value,
            "satisfied_without_execution": self.satisfied_without_execution,
            "binding_provenance": self.binding_provenance,
            "budget_scope": self.budget_scope,
            "distinct_bindings": {
                key: list(value)
                for key, value in self.distinct_bindings.items()
            },
            "distinct_branch_ids": dict(self.distinct_branch_ids),
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
                             budget_scope=node.budget_scope,
                             distinct_bindings={
                                 key: list(value)
                                 for key, value in node.distinct_bindings.items()
                             },
                             distinct_branch_ids=dict(
                                 node.distinct_branch_ids))
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
                           task_target_effects: list[dict[str, Any]] | None = None,
                           occurrence_id: str = "task_gap_000") -> int:
        """Append one explicit, auditable task-level Dynamic occurrence."""
        index = len(self.plan.nodes)
        step_id = f"step_{index:03d}"
        distinct_bindings, distinct_branch_ids = _gap_distinct_constraints(
            missing_effects, task_target_effects or missing_effects,
            occurrence_id=occurrence_id)
        planned = PlannedNode(
            ref=SkillRef("runtime.dynamic.task_gap", "0.0.0"),
            step_id=step_id, occurrence_id=occurrence_id,
            params=dict(params), source="task_gap",
            target_effects=[dict(item) for item in missing_effects],
            dynamic=True,
            budget_scope="gap",
            branch_id=occurrence_id,
            distinct_bindings=distinct_bindings,
            distinct_branch_ids=distinct_branch_ids,
        )
        previous = self.plan.nodes[-1] if self.plan.nodes else None
        self.plan.nodes.append(planned)
        state = RuntimeNodeState(
            ref=str(planned.ref), step_id=step_id,
            occurrence_id=occurrence_id, params=dict(params),
            target_effects=list(planned.target_effects),
            budget_scope=planned.budget_scope,
            distinct_bindings={
                role: list(groups)
                for role, groups in distinct_bindings.items()
            },
            distinct_branch_ids=dict(distinct_branch_ids),
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

    def distinct_exclusions(self, index: int) -> dict[str, set[Any]]:
        """Return validated values claimed by earlier cardinality branches.

        A value flows freely inside its own branch (Acquire -> Place), but it
        cannot ground the same distinctness group in another branch.  Only
        passed runtime nodes are evidence, so failed hypotheses never reserve
        an instance.
        """
        if index < 0 or index >= len(self.plan.nodes):
            return {}
        current = self.plan.nodes[index]
        if not current.distinct_bindings:
            return {}
        exclusions: dict[str, set[Any]] = {}
        for current_role, raw_groups in current.distinct_bindings.items():
            groups = {str(item) for item in raw_groups if str(item)}
            if not groups:
                continue
            for prior_index in range(index):
                prior_plan = self.plan.nodes[prior_index]
                prior_state = self.nodes[prior_index]
                if not prior_state.passed:
                    continue
                for prior_role, prior_groups in (
                        prior_plan.distinct_bindings or {}).items():
                    shared_groups = groups & {
                        str(item) for item in prior_groups if str(item)}
                    if not shared_groups:
                        continue
                    crosses_branch = any(
                        _distinct_branch_id(current, group)
                        and _distinct_branch_id(prior_plan, group)
                        and _distinct_branch_id(current, group)
                        != _distinct_branch_id(prior_plan, group)
                        for group in shared_groups)
                    if not crosses_branch:
                        continue
                    values = _validated_distinct_witnesses(
                        prior_plan, prior_state, str(prior_role))
                    concrete_values = {
                        value for value in values
                        if value not in (None, "") and not (
                            isinstance(value, str) and value.startswith("$"))
                    }
                    if concrete_values:
                        exclusions.setdefault(str(current_role), set()).update(
                            concrete_values)
        return exclusions

    def distinct_occurrence_ordinal(self, index: int, group: str) -> int:
        """Return the stable occurrence ordinal for one distinctness group."""
        if index < 0 or index >= len(self.plan.nodes):
            return -1
        current = self.plan.nodes[index]
        identity = _distinct_branch_id(current, str(group))
        if not identity:
            return -1
        identities: list[str] = []
        for node in self.plan.nodes:
            participates = any(
                str(group) in {str(item) for item in groups}
                for groups in (node.distinct_bindings or {}).values())
            if not participates:
                continue
            candidate = _distinct_branch_id(node, str(group))
            if candidate and candidate not in identities:
                identities.append(candidate)
        return identities.index(identity) if identity in identities else -1

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


def distinct_values_conflict(left: Any, right: Any) -> bool:
    """Whether two claims may denote the same instance.

    Class-to-instance matching is deliberately symmetric and fail-closed. Two
    different grounded instances remain distinct, while either direction of
    ``mug`` <-> ``mug_1`` conflicts.
    """
    return distinct_claims_conflict(left, right)


def _distinct_branch_id(node: PlannedNode, group: str) -> str:
    return str((node.distinct_branch_ids or {}).get(group)
               or node.branch_id or "")


def _validated_distinct_witnesses(
        planned: PlannedNode, state: RuntimeNodeState, role: str) -> set[Any]:
    """Return every concrete validated witness, falling back only if absent."""
    claimed = state.outputs.get(role)
    if claimed in (None, ""):
        claimed = state.params.get(role)
    # Collection outputs are explicit validated occurrence results.  They are
    # merged with fact-derived witnesses for exclusion governance, but never
    # exposed through the scalar output channel used by DATA_FLOW.
    witnesses: set[str] = {
        normalize_value(value)
        for value in state.distinct_witness_outputs.get(role, [])
        if value not in (None, "")
    }
    facts = list((state.after or {}).get("facts") or [])
    bindings = {**dict(planned.params or {}), **dict(state.params or {})}
    for effect in planned.target_effects or []:
        if not isinstance(effect, dict):
            continue
        items = ordered_predicate_args(
            str(effect.get("predicate") or ""),
            dict(effect.get("args") or {}))
        role_positions = [
            index for index, (_arg, value) in enumerate(items)
            if binding_slot_name(value) == role
        ]
        if not role_positions:
            continue
        expected_predicate = _fact_predicate(
            str(effect.get("predicate") or ""))
        for fact in facts:
            parsed = _parse_fact(str(fact))
            if parsed is None:
                continue
            predicate, actual_values = parsed
            if predicate != expected_predicate or len(actual_values) != len(items):
                continue
            compatible = True
            for position, (_arg, expected) in enumerate(items):
                actual = actual_values[position]
                slot = binding_slot_name(expected)
                bound = bindings.get(slot) if slot else expected
                if position in role_positions:
                    bound = claimed if claimed not in (None, "") else bound
                if (bound not in (None, "")
                        and not str(bound).startswith("$")
                        and not distinct_values_conflict(bound, actual)):
                    compatible = False
                    break
            if compatible:
                witnesses.update(
                    normalize_value(actual_values[position])
                    for position in role_positions)
    if witnesses:
        return set(witnesses)
    return {claimed} if claimed not in (None, "") else set()


def _fact_predicate(value: str) -> str:
    normalized, _order = predicate_fact_signature(value)
    return {
        "object_at_location": "object_at",
        "object_in_receptacle": "object_at",
        "object_in_container": "object_at",
    }.get(normalized, normalized)


def _parse_fact(fact: str) -> tuple[str, list[str]] | None:
    match = re.fullmatch(r"\s*([a-zA-Z0-9_.]+)\((.*)\)\s*", fact)
    if match is None:
        return None
    values = [normalize_value(item.strip())
              for item in match.group(2).split(",")]
    return _fact_predicate(match.group(1)), values


def _gap_distinct_constraints(
        missing_effects: list[dict[str, Any]],
        task_target_effects: list[dict[str, Any]], *,
        occurrence_id: str) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Project cardinality contracts onto an explicit Task Gap occurrence.

    Missing targets are a filtered subset of the original task targets. Their
    group identity must retain the original target index; re-numbering the
    subset would collide whenever an earlier target is already satisfied.
    """
    distinct_bindings: dict[str, list[str]] = {}
    branch_ids: dict[str, str] = {}
    used_target_indices: set[int] = set()
    targets = list(task_target_effects or [])
    for missing in missing_effects or []:
        if (not isinstance(missing, dict)
                or int(missing.get("cardinality", 1) or 1) <= 1):
            continue
        distinct_arg = str(missing.get("distinct_by") or "")
        if not distinct_arg:
            continue
        target_index = _matching_task_target_index(
            missing, targets, used_target_indices)
        if target_index is None:
            continue
        used_target_indices.add(target_index)
        predicate = _gap_group_predicate(
            str(missing.get("predicate") or ""))
        group = f"target_{target_index:03d}:{predicate}:{distinct_arg}"
        args = dict(missing.get("args") or {})
        role = binding_slot_name(args.get(distinct_arg)) or distinct_arg
        distinct_bindings.setdefault(role, []).append(group)
        branch_ids[group] = str(occurrence_id or "task_gap_000")
    return distinct_bindings, branch_ids


def _matching_task_target_index(
        missing: dict[str, Any], targets: list[dict[str, Any]],
        used: set[int]) -> int | None:
    for index, target in enumerate(targets):
        if index not in used and isinstance(target, dict) and target == missing:
            return index
    wanted = (
        _gap_group_predicate(str(missing.get("predicate") or "")),
        dict(missing.get("args") or {}),
        int(missing.get("cardinality", 1) or 1),
        str(missing.get("distinct_by") or ""),
    )
    for index, target in enumerate(targets):
        if index in used or not isinstance(target, dict):
            continue
        candidate = (
            _gap_group_predicate(str(target.get("predicate") or "")),
            dict(target.get("args") or {}),
            int(target.get("cardinality", 1) or 1),
            str(target.get("distinct_by") or ""),
        )
        if candidate == wanted:
            return index
    return None


def _gap_group_predicate(value: str) -> str:
    return {
        "object.in_receptacle": "object.at_location",
        "object.in.receptacle": "object.at_location",
        "object_in_receptacle": "object.at_location",
        "object.in_container": "object.at_location",
        "object.in.container": "object.at_location",
        "object_in_container": "object.at_location",
    }.get(str(value or ""), str(value or ""))
