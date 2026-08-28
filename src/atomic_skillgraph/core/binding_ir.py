"""Structured parameter bindings shared by planning, runtime and validation.

Legacy ``$task.*`` values are explicit task bindings.  Legacy ``$flow.*``
values are deliberately unresolved: guessing their producer was the source of
cross-occurrence data-flow corruption in multi-object workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BindingKind(str, Enum):
    LITERAL = "literal"
    TASK = "task"
    DATA_FLOW = "data_flow"
    STATE = "state"
    UNRESOLVED = "unresolved"


class BindingResolutionState(str, Enum):
    """Planning-time state of one input slot.

    ``PENDING_DATA_FLOW`` is source-closed even though the concrete value only
    exists after an earlier occurrence executes.  ``RUNTIME_RESOLVABLE`` is
    likewise source-closed for Seeded/Dynamic execution, but deliberately does
    not make a Direct invocation eligible.
    """

    RESOLVED = "resolved"
    PENDING_DATA_FLOW = "pending_data_flow"
    RUNTIME_RESOLVABLE = "runtime_resolvable"
    UNRESOLVABLE = "unresolvable"


@dataclass(frozen=True)
class ResolutionPolicy:
    """Allowed evidence sources and their deterministic precedence."""

    precedence: tuple[str, ...] = ("task", "data_flow", "state", "runtime")
    allow_literal: bool = True
    require_runtime_anchor: bool = True

    def allows(self, source: str) -> bool:
        normalized = str(source or "").strip().lower()
        return ((normalized == "literal" and self.allow_literal)
                or normalized in self.precedence)


@dataclass(frozen=True)
class BindingProvenance:
    """Auditable provenance for one resolved or deferred binding."""

    source: str = ""
    role: str = ""
    source_step: str = ""
    source_output: str = ""
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "role": self.role,
            "source_step": self.source_step,
            "source_output": self.source_output,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class SlotRequirement:
    """Execution requirements for an Atomic input slot.

    The flags may be declared directly on an Atomic ``inputs`` item.  Missing
    metadata remains conservative: a slot used by a core Effect is semantic,
    while an auxiliary slot is optional unless explicitly marked otherwise.
    """

    name: str
    semantic_required: bool = False
    direct_required: bool = False
    runtime_resolvable: bool = False
    allowed_sources: frozenset[str] = field(default_factory=lambda: frozenset({
        "literal", "task", "data_flow", "state", "runtime",
    }))
    anchor_roles: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "semantic_required": self.semantic_required,
            "direct_required": self.direct_required,
            "runtime_resolvable": self.runtime_resolvable,
            "allowed_sources": sorted(self.allowed_sources),
            "anchor_roles": list(self.anchor_roles),
        }

    @classmethod
    def from_input(cls, declaration: dict[str, Any], *,
                   semantic_required: bool = False) -> "SlotRequirement":
        raw_sources = declaration.get("allowed_sources")
        sources = (frozenset(str(item).strip().lower() for item in raw_sources
                             if str(item).strip())
                   if isinstance(raw_sources, (list, tuple, set, frozenset))
                   else cls.__dataclass_fields__["allowed_sources"].default_factory())
        raw_anchors = declaration.get("anchor_roles") or []
        if isinstance(raw_anchors, str):
            raw_anchors = [raw_anchors]
        return cls(
            name=str(declaration.get("name") or ""),
            semantic_required=bool(
                declaration.get("semantic_required", semantic_required)),
            direct_required=bool(declaration.get("direct_required", False)),
            runtime_resolvable=bool(declaration.get("runtime_resolvable", False)),
            allowed_sources=sources,
            anchor_roles=tuple(str(item) for item in raw_anchors if str(item)),
        )


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
    resolution_state: BindingResolutionState | None = None
    provenance: BindingProvenance | None = None

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
            "resolution_state": (self.resolution_state.value
                                 if self.resolution_state is not None else ""),
            "provenance": (self.provenance.to_dict()
                           if self.provenance is not None else {}),
        }

    @classmethod
    def from_value(cls, value: Any) -> "BindingSpec":
        if isinstance(value, BindingSpec):
            return value
        if isinstance(value, dict) and value.get("kind") in {
                item.value for item in BindingKind}:
            raw_state = str(value.get("resolution_state") or "")
            raw_provenance = value.get("provenance") or {}
            return cls(
                kind=BindingKind(str(value["kind"])), value=value.get("value"),
                task_role=str(value.get("task_role") or ""),
                source_step=str(value.get("source_step") or ""),
                source_output=str(value.get("source_output") or ""),
                state_predicate=str(value.get("state_predicate") or ""),
                state_entity_role=str(value.get("state_entity_role") or ""),
                symbol=str(value.get("symbol") or ""),
                resolution_state=(BindingResolutionState(raw_state)
                                  if raw_state in {
                                      item.value for item in BindingResolutionState
                                  } else None),
                provenance=(BindingProvenance(
                    source=str(raw_provenance.get("source") or ""),
                    role=str(raw_provenance.get("role") or ""),
                    source_step=str(raw_provenance.get("source_step") or ""),
                    source_output=str(raw_provenance.get("source_output") or ""),
                    evidence=tuple(str(item) for item in
                                   (raw_provenance.get("evidence") or [])),
                ) if isinstance(raw_provenance, dict) and raw_provenance else None),
            )
        if isinstance(value, str) and value.startswith("$task."):
            return cls(BindingKind.TASK, task_role=value[len("$task."):],
                       symbol=value)
        if isinstance(value, str) and value.startswith("$flow."):
            # The producer occurrence is intentionally not guessed here.  A
            # DATA_FLOW edge must later supply ``source_step`` exactly.
            return cls(BindingKind.UNRESOLVED,
                       source_output=value[len("$flow."):], symbol=value)
        if isinstance(value, str) and value.startswith("$inputs."):
            return cls(BindingKind.UNRESOLVED, symbol=value)
        if isinstance(value, str) and value.startswith("$"):
            return cls(BindingKind.UNRESOLVED, symbol=value)
        return cls(BindingKind.LITERAL, value=value)


def binding_slot_name(value: Any) -> str:
    """Return the semantic slot named by any supported placeholder syntax."""

    spec = BindingSpec.from_value(value)
    if spec.kind == BindingKind.TASK:
        return spec.task_role
    if spec.kind == BindingKind.DATA_FLOW:
        return spec.source_output
    symbol = spec.symbol if isinstance(spec.symbol, str) else ""
    if not symbol.startswith("$"):
        return ""
    path = symbol[1:].split(".")
    if len(path) == 1:
        return path[0]
    if len(path) == 2 and path[0] in {"inputs", "task", "flow"}:
        return path[1]
    return ""


def source_name_for_kind(kind: BindingKind) -> str:
    return {
        BindingKind.LITERAL: "literal",
        BindingKind.TASK: "task",
        BindingKind.DATA_FLOW: "data_flow",
        BindingKind.STATE: "state",
        BindingKind.UNRESOLVED: "runtime",
    }[kind]


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
