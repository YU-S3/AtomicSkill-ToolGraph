"""Fail-closed binding and route validation for a compiled RuntimePlan."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..core.binding_ir import BindingKind, BindingSpec, is_concrete_binding


@dataclass
class NodeBindingReport:
    step_id: str
    semantic_required_slots: list[str] = field(default_factory=list)
    semantic_bound_slots: list[str] = field(default_factory=list)
    unresolved_semantic_slots: list[str] = field(default_factory=list)
    implementation_missing_slots: list[str] = field(default_factory=list)
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    executable_routes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return vars(self)


@dataclass
class PlanValidationReport:
    passed: bool
    node_reports: list[NodeBindingReport] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed,
                "node_reports": [item.to_dict() for item in self.node_reports],
                "errors": self.errors}


def semantic_required_slots(effects: list[dict[str, Any]]) -> set[str]:
    slots: set[str] = set()
    for effect in effects or []:
        for value in (effect.get("args") or {}).values():
            if isinstance(value, str):
                match = re.fullmatch(r"\$(?:inputs|task|flow)\.([A-Za-z_][A-Za-z0-9_]*)", value)
                if match:
                    slots.add(match.group(1))
    return slots


def validate_plan_bindings(plan, registry, task) -> PlanValidationReport:
    reports: list[NodeBindingReport] = []
    errors: list[str] = []
    step_ids = {node.origin_step_id or node.step_id for node in plan.nodes}
    for node in plan.nodes:
        atomic = registry.get(node.ref)
        effects = list(node.target_effects or getattr(atomic, "effects", []) or [])
        required = semantic_required_slots(effects)
        report = NodeBindingReport(step_id=node.origin_step_id or node.step_id,
                                   semantic_required_slots=sorted(required))
        for slot in sorted(required):
            value = node.params.get(slot)
            spec = node.binding_specs.get(slot, BindingSpec.from_value(value))
            report.sources[slot] = spec.to_dict()
            bound = is_concrete_binding(value)
            if spec.kind == BindingKind.DATA_FLOW:
                source_node = next((item for item in plan.nodes
                                    if (item.origin_step_id or item.step_id)
                                    == spec.source_step), None)
                source_atomic = (registry.get(source_node.ref)
                                 if source_node is not None else None)
                bound = bool(
                    spec.source_step in step_ids and spec.source_output
                    and source_atomic is not None
                    and any(str(item.get("name") or "") == spec.source_output
                            for item in getattr(source_atomic, "outputs", []) or []))
            if bound:
                report.semantic_bound_slots.append(slot)
            else:
                report.unresolved_semantic_slots.append(slot)
                errors.append(
                    f"unresolved_semantic_binding:step={report.step_id}:slot={slot}")
        if not report.unresolved_semantic_slots:
            report.executable_routes.extend(["seeded", "dynamic"])
        reports.append(report)
    return PlanValidationReport(not errors, reports, errors)


def validate_composite_binding_closure(composite, registry) -> PlanValidationReport:
    """Validate persisted occurrence bindings without executing the graph."""
    incoming: dict[tuple[str, str], Any] = {}
    for edge in composite.edge_objects():
        if str(edge.type.value) != "data_flow":
            continue
        target_input = str((edge.mapping or {}).get("target_input") or "")
        source_output = str((edge.mapping or {}).get("source_output") or "")
        if target_input and source_output:
            incoming[(edge.target_step, target_input)] = edge
    reports: list[NodeBindingReport] = []
    errors: list[str] = []
    steps = {str(step["step_id"]): step for step in composite.step_instances()}
    for step_id, step in steps.items():
        try:
            from ..core.refs import SkillRef
            atomic = registry.get(SkillRef.parse(str(step["node_ref"])))
        except ValueError:
            atomic = None
        if atomic is None:
            errors.append(f"exact_child_version_missing:step={step_id}")
            continue
        required = semantic_required_slots(list(getattr(atomic, "effects", []) or []))
        report = NodeBindingReport(step_id=step_id,
                                   semantic_required_slots=sorted(required))
        params = dict(step.get("params") or {})
        for slot in sorted(required):
            value = params.get(slot)
            spec = BindingSpec.from_value(value)
            edge = incoming.get((step_id, slot))
            if edge is not None:
                spec = BindingSpec(
                    BindingKind.DATA_FLOW, source_step=edge.source_step,
                    source_output=str(edge.mapping.get("source_output") or ""))
            report.sources[slot] = spec.to_dict()
            source_step = steps.get(spec.source_step) if spec.source_step else None
            source_atomic = None
            if source_step is not None:
                try:
                    from ..core.refs import SkillRef
                    source_atomic = registry.get(SkillRef.parse(
                        str(source_step["node_ref"])))
                except ValueError:
                    source_atomic = None
            output_exists = bool(source_atomic is not None and any(
                str(item.get("name") or "") == spec.source_output
                for item in (getattr(source_atomic, "outputs", []) or [])))
            bound = (spec.kind == BindingKind.LITERAL
                     and is_concrete_binding(spec.value))
            bound = bound or (spec.kind == BindingKind.TASK and bool(spec.task_role))
            bound = bound or (spec.kind == BindingKind.STATE
                              and bool(spec.state_predicate))
            bound = bound or (spec.kind == BindingKind.DATA_FLOW
                              and output_exists)
            if bound:
                report.semantic_bound_slots.append(slot)
            else:
                report.unresolved_semantic_slots.append(slot)
                errors.append(
                    f"unresolved_semantic_binding:step={step_id}:slot={slot}")
        reports.append(report)
    return PlanValidationReport(not errors, reports, errors)
