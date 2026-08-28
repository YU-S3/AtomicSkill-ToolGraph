"""Structured parameter bindings shared by planning, runtime and validation.

Legacy ``$task.*`` values are explicit task bindings.  Legacy ``$flow.*``
values are deliberately unresolved: guessing their producer was the source of
cross-occurrence data-flow corruption in multi-object workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class BindingKind(str, Enum):
    LITERAL = "literal"
    TASK = "task"
    DATA_FLOW = "data_flow"
    STATE = "state"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class BindingSpec:
    kind: BindingKind
    value: Any = None
    task_role: str = ""
    source_step: str = ""
    source_output: str = ""
    state_predicate: str = ""
    state_entity_role: str = ""
    symbol: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "value": self.value,
            "task_role": self.task_role,
            "source_step": self.source_step,
            "source_output": self.source_output,
            "state_predicate": self.state_predicate,
            "state_entity_role": self.state_entity_role,
            "symbol": self.symbol,
        }

    @classmethod
    def from_value(cls, value: Any) -> "BindingSpec":
        if isinstance(value, BindingSpec):
            return value
        if isinstance(value, dict) and value.get("kind") in {
                item.value for item in BindingKind}:
            return cls(
                kind=BindingKind(str(value["kind"])), value=value.get("value"),
                task_role=str(value.get("task_role") or ""),
                source_step=str(value.get("source_step") or ""),
                source_output=str(value.get("source_output") or ""),
                state_predicate=str(value.get("state_predicate") or ""),
                state_entity_role=str(value.get("state_entity_role") or ""),
                symbol=str(value.get("symbol") or ""),
            )
        if isinstance(value, str) and value.startswith("$task."):
            return cls(BindingKind.TASK, task_role=value[len("$task."):],
                       symbol=value)
        if isinstance(value, str) and value.startswith("$"):
            return cls(BindingKind.UNRESOLVED, symbol=value)
        return cls(BindingKind.LITERAL, value=value)


def is_symbolic_binding(value: Any) -> bool:
    if isinstance(value, BindingSpec):
        return value.kind == BindingKind.UNRESOLVED
    if isinstance(value, str):
        return value.startswith("$")
    if isinstance(value, dict):
        if value.get("kind") == BindingKind.UNRESOLVED.value:
            return True
        return any(is_symbolic_binding(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(is_symbolic_binding(item) for item in value)
    return False


def is_concrete_binding(value: Any) -> bool:
    if value in (None, ""):
        return False
    if isinstance(value, BindingSpec):
        return value.kind == BindingKind.LITERAL and is_concrete_binding(value.value)
    if isinstance(value, str):
        return not value.startswith("$")
    if isinstance(value, dict):
        if value.get("kind") in {item.value for item in BindingKind}:
            spec = BindingSpec.from_value(value)
            return (spec.kind == BindingKind.LITERAL
                    and is_concrete_binding(spec.value))
        return bool(value) and all(is_concrete_binding(item)
                                   for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return bool(value) and all(is_concrete_binding(item) for item in value)
    return True


def resolve_binding(spec: BindingSpec, task_params: dict[str, Any],
                    outputs_by_step: dict[str, dict[str, Any]] | None = None,
                    state_values: dict[str, Any] | None = None) -> Any:
    if spec.kind == BindingKind.LITERAL:
        return spec.value if is_concrete_binding(spec.value) else None
    if spec.kind == BindingKind.TASK:
        value = task_params.get(spec.task_role)
        return value if is_concrete_binding(value) else None
    if spec.kind == BindingKind.DATA_FLOW:
        value = ((outputs_by_step or {}).get(spec.source_step) or {}).get(
            spec.source_output)
        return value if is_concrete_binding(value) else None
    if spec.kind == BindingKind.STATE:
        value = (state_values or {}).get(spec.state_entity_role)
        return value if is_concrete_binding(value) else None
    return None
