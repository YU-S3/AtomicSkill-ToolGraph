"""Deterministic executor for Runtime Graph edge semantics.

The LLM produces or repairs node behaviour; graph traversal, retry limits,
conditions and data transfer are framework responsibilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.edge_ir import GraphEdge
from ..core.status import EdgeType


@dataclass
class GraphExecutionState:
    completed: set[str] = field(default_factory=set)
    failed: set[str] = field(default_factory=set)
    attempts: dict[str, int] = field(default_factory=dict)
    loop_counts: dict[str, int] = field(default_factory=dict)
    values: dict[str, dict[str, Any]] = field(default_factory=dict)


class RuntimeGraphExecutor:
    """Small scheduling kernel shared by environment and synthetic tests."""

    def __init__(self, step_ids: list[str], edges: list[GraphEdge]) -> None:
        self.step_ids = list(step_ids)
        self.edges = list(edges)

    def initial_steps(self, context: dict[str, Any] | None = None) -> list[str]:
        context = context or {}
        incoming = {edge.target_step for edge in self.edges
                    if edge.type in {EdgeType.NEXT, EdgeType.BRANCH,
                                     EdgeType.PARALLEL}}
        return [step for step in self.step_ids if step not in incoming]

    def next_steps(self, current: str, success: bool,
                   state: GraphExecutionState,
                   context: dict[str, Any] | None = None) -> list[str]:
        """Resolve outgoing control edges with bounded retry/loop semantics."""
        context = context or {}
        if success:
            state.completed.add(current)
        else:
            state.failed.add(current)
        outgoing = [edge for edge in self.edges if edge.source_step == current]

        if not success:
            for edge in outgoing:
                if edge.type == EdgeType.RETRY:
                    count = state.attempts.get(current, 1)
                    if count < int(edge.policy.get("max_attempts", 1)):
                        state.attempts[current] = count + 1
                        return [edge.target_step or current]
            fallbacks = [edge.target_step for edge in outgoing
                         if edge.type == EdgeType.FALLBACK
                         and _condition_matches(edge.policy.get("on"), context)]
            return _unique(fallbacks)

        selected: list[str] = []
        for edge in outgoing:
            if edge.type == EdgeType.DATA_FLOW:
                self.apply_data_flow(edge, state)
                continue
            if edge.type in {EdgeType.NEXT, EdgeType.PARALLEL}:
                selected.append(edge.target_step)
            elif edge.type == EdgeType.BRANCH and evaluate_condition(edge.condition, context):
                selected.append(edge.target_step)
            elif edge.type == EdgeType.LOOP and evaluate_condition(edge.condition, context):
                count = state.loop_counts.get(edge.edge_id, 0)
                if count < int(edge.policy.get("max_iterations", 1)):
                    state.loop_counts[edge.edge_id] = count + 1
                    selected.append(edge.target_step)
        return _unique(selected)

    @staticmethod
    def apply_data_flow(edge: GraphEdge, state: GraphExecutionState) -> None:
        source_key = str(edge.mapping.get("source_output") or "")
        target_key = str(edge.mapping.get("target_input") or "")
        if not source_key or not target_key:
            return
        source_values = state.values.get(edge.source_step, {})
        if source_key in source_values:
            state.values.setdefault(edge.target_step, {})[target_key] = source_values[source_key]


def evaluate_condition(condition: dict[str, Any], context: dict[str, Any]) -> bool:
    """Evaluate a safe declarative condition; never executes arbitrary code."""
    if not condition:
        return False
    if "all" in condition:
        return all(evaluate_condition(dict(item), context)
                   for item in condition.get("all") or [])
    if "any" in condition:
        return any(evaluate_condition(dict(item), context)
                   for item in condition.get("any") or [])
    if "not" in condition:
        return not evaluate_condition(dict(condition.get("not") or {}), context)
    field = str(condition.get("field") or "")
    actual = _lookup(context, field)
    if "equals" in condition:
        return actual == condition["equals"]
    if "not_equals" in condition:
        return actual != condition["not_equals"]
    if "in" in condition:
        return actual in (condition.get("in") or [])
    if "exists" in condition:
        return (actual is not None) == bool(condition["exists"])
    if "truthy" in condition:
        return bool(actual) == bool(condition["truthy"])
    return False


def _lookup(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split(".") if path else []:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _condition_matches(expected: Any, context: dict[str, Any]) -> bool:
    if isinstance(expected, list):
        return context.get("failure_type") in expected or "any" in expected
    return expected in ("any", context.get("failure_type"))


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))
