"""Fail-closed source-closure and route validation for Runtime plans."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..core.binding_ir import (
    BindingKind,
    BindingProvenance,
    BindingResolutionState,
    BindingSpec,
    SlotRequirement,
    is_concrete_binding,
    source_name_for_kind,
)
from ..core.status import EdgeType
from .output_materializer import validate_output_materializer


@dataclass
class NodeBindingReport:
    step_id: str
    semantic_required_slots: list[str] = field(default_factory=list)
    semantic_bound_slots: list[str] = field(default_factory=list)
    unresolved_semantic_slots: list[str] = field(default_factory=list)
    pending_data_flow_slots: list[str] = field(default_factory=list)
    runtime_resolvable_slots: list[str] = field(default_factory=list)
    implementation_missing_slots: list[str] = field(default_factory=list)
    requirements: dict[str, dict[str, Any]] = field(default_factory=dict)
    resolution_states: dict[str, str] = field(default_factory=dict)
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    binding_provenance: dict[str, dict[str, Any]] = field(default_factory=dict)
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
        if not isinstance(effect, dict):
            continue
        for value in (effect.get("args") or {}).values():
            if isinstance(value, str):
                match = re.fullmatch(
                    r"\$(?:(?:inputs|task|flow)\.)?([A-Za-z_][A-Za-z0-9_]*)",
                    value)
                if match:
                    slots.add(match.group(1))
    return slots


def slot_requirements_for(atomic: Any,
                          effects: list[dict[str, Any]]) -> dict[str, SlotRequirement]:
    semantic = semantic_required_slots(effects)
    result: dict[str, SlotRequirement] = {}
    for declaration in list(getattr(atomic, "inputs", []) or []):
        if not isinstance(declaration, dict) or not declaration.get("name"):
            continue
        requirement = SlotRequirement.from_input(
            declaration, semantic_required=str(declaration["name"]) in semantic)
        result[requirement.name] = requirement
    for name in semantic:
        result.setdefault(name, SlotRequirement(name=name, semantic_required=True))
    return result


def validate_plan_bindings(plan: Any, registry: Any, task: Any
                           ) -> PlanValidationReport:
    """Validate parameter *source closure*, not premature concreteness."""

    reports: list[NodeBindingReport] = []
    errors: list[str] = []
    node_by_step: dict[str, tuple[int, Any]] = {}
    for index, node in enumerate(plan.nodes):
        node_by_step[_runtime_step(node, index)] = (index, node)
        origin = str(getattr(node, "origin_step_id", "") or "")
        if origin:
            node_by_step.setdefault(origin, (index, node))
    incoming, edge_errors = _incoming_flow_specs(plan, registry, node_by_step)
    errors.extend(edge_errors)
    task_params = dict(getattr(task, "context", {}).get("params") or {})

    for index, node in enumerate(plan.nodes):
        step_id = _runtime_step(node, index)
        atomic = registry.get(node.ref)
        effects = list(getattr(node, "target_effects", [])
                       or getattr(atomic, "effects", []) or [])
        requirements = slot_requirements_for(atomic, effects)
        report = NodeBindingReport(
            step_id=step_id,
            semantic_required_slots=sorted(
                name for name, requirement in requirements.items()
                if requirement.semantic_required),
            requirements={name: requirement.to_dict()
                          for name, requirement in requirements.items()},
        )
        specs: dict[str, BindingSpec] = {}
        params = dict(getattr(node, "params", {}) or {})
        declared_specs = dict(getattr(node, "binding_specs", {}) or {})
        for name in requirements:
            specs[name] = incoming.get((step_id, name),
                                       declared_specs.get(
                                           name, BindingSpec.from_value(
                                               params.get(name))))

        states = _resolve_requirement_states(
            requirements, specs, params, task_params, persistent=False)
        for name, requirement in requirements.items():
            spec = specs[name]
            state = states[name]
            report.sources[name] = spec.to_dict()
            report.binding_provenance[name] = _provenance(
                name, spec, state).to_dict()
            report.resolution_states[name] = state.value
            if state == BindingResolutionState.RESOLVED:
                if requirement.semantic_required:
                    report.semantic_bound_slots.append(name)
            elif state == BindingResolutionState.PENDING_DATA_FLOW:
                report.pending_data_flow_slots.append(name)
                if requirement.semantic_required:
                    report.semantic_bound_slots.append(name)
            elif state == BindingResolutionState.RUNTIME_RESOLVABLE:
                report.runtime_resolvable_slots.append(name)
            else:
                if requirement.semantic_required:
                    report.unresolved_semantic_slots.append(name)
                    errors.append(
                        f"unresolved_semantic_binding:step={step_id}:slot={name}")
            if (requirement.direct_required
                    and state not in {BindingResolutionState.RESOLVED,
                                      BindingResolutionState.PENDING_DATA_FLOW}):
                report.implementation_missing_slots.append(name)

        required = [name for name, requirement in requirements.items()
                    if requirement.semantic_required or requirement.direct_required]
        direct_ready = all(states[name] in {
            BindingResolutionState.RESOLVED,
            BindingResolutionState.PENDING_DATA_FLOW,
        } for name in required)
        semantic_closed = all(
            states[name] != BindingResolutionState.UNRESOLVABLE
            for name, requirement in requirements.items()
            if requirement.semantic_required)
        if direct_ready:
            report.executable_routes.append("direct")
        if semantic_closed:
            report.executable_routes.extend(["seeded", "dynamic"])
        reports.append(report)
    return PlanValidationReport(not errors, reports, errors)


def validate_plan_source_closure(plan: Any, registry: Any, task: Any
                                 ) -> PlanValidationReport:
    """Preferred explicit name; retained alongside the old public API."""

    return validate_plan_bindings(plan, registry, task)


def validate_composite_binding_closure(composite: Any, registry: Any
                                       ) -> PlanValidationReport:
    """Validate persisted occurrence sources without requiring task values."""

    incoming: dict[tuple[str, str], BindingSpec] = {}
    errors: list[str] = []
    steps = {str(step["step_id"]): step for step in composite.step_instances()}
    order = {step_id: index for index, step_id in enumerate(steps)}
    for edge in composite.edge_objects():
        if edge.type != EdgeType.DATA_FLOW:
            continue
        target_input = str((edge.mapping or {}).get("target_input") or "")
        source_output = str((edge.mapping or {}).get("source_output") or "")
        source_step = steps.get(edge.source_step)
        target_step = steps.get(edge.target_step)
        source_atomic = _atomic_of_step(source_step, registry)
        if source_step is None or target_step is None:
            errors.append(f"data_flow_unknown_step:{edge.edge_id}")
            continue
        if order[edge.source_step] >= order[edge.target_step]:
            errors.append(f"data_flow_source_not_prior:{edge.edge_id}")
            continue
        materializer = validate_output_materializer(source_atomic, source_output)
        if not materializer.passed:
            errors.append(
                f"data_flow_output_unmaterializable:{edge.source_step}:"
                f"{source_output}:{'|'.join(materializer.errors)}")
            continue
        if target_input and source_output:
            incoming[(edge.target_step, target_input)] = BindingSpec(
                BindingKind.DATA_FLOW, source_step=edge.source_step,
                source_output=source_output,
                symbol=f"$flow.{source_output}")

    reports: list[NodeBindingReport] = []
    for step_id, step in steps.items():
        atomic = _atomic_of_step(step, registry)
        if atomic is None:
            errors.append(f"exact_child_version_missing:step={step_id}")
            continue
        effects = list(getattr(atomic, "effects", []) or [])
        requirements = slot_requirements_for(atomic, effects)
        report = NodeBindingReport(
            step_id=step_id,
            semantic_required_slots=sorted(
                name for name, requirement in requirements.items()
                if requirement.semantic_required),
            requirements={name: requirement.to_dict()
                          for name, requirement in requirements.items()},
        )
        params = dict(step.get("params") or {})
        specs = {name: incoming.get(
            (step_id, name), BindingSpec.from_value(params.get(name)))
                 for name in requirements}
        states = _resolve_requirement_states(
            requirements, specs, params, {}, persistent=True)
        for name, requirement in requirements.items():
            spec, state = specs[name], states[name]
            report.sources[name] = spec.to_dict()
            report.binding_provenance[name] = _provenance(
                name, spec, state).to_dict()
            report.resolution_states[name] = state.value
            if state == BindingResolutionState.RESOLVED:
                if requirement.semantic_required:
                    report.semantic_bound_slots.append(name)
            elif state == BindingResolutionState.PENDING_DATA_FLOW:
                report.pending_data_flow_slots.append(name)
                if requirement.semantic_required:
                    report.semantic_bound_slots.append(name)
            elif state == BindingResolutionState.RUNTIME_RESOLVABLE:
                report.runtime_resolvable_slots.append(name)
            elif requirement.semantic_required:
                report.unresolved_semantic_slots.append(name)
                errors.append(
                    f"unresolved_semantic_binding:step={step_id}:slot={name}")
            if requirement.direct_required and state not in {
                    BindingResolutionState.RESOLVED,
                    BindingResolutionState.PENDING_DATA_FLOW}:
                report.implementation_missing_slots.append(name)
        semantic_closed = not report.unresolved_semantic_slots
        direct_ready = not report.implementation_missing_slots and all(
            states[name] in {BindingResolutionState.RESOLVED,
                             BindingResolutionState.PENDING_DATA_FLOW}
            for name, requirement in requirements.items()
            if requirement.semantic_required)
        if direct_ready:
            report.executable_routes.append("direct")
        if semantic_closed:
            report.executable_routes.extend(["seeded", "dynamic"])
        reports.append(report)
    return PlanValidationReport(not errors, reports, errors)


def _resolve_requirement_states(
        requirements: dict[str, SlotRequirement], specs: dict[str, BindingSpec],
        params: dict[str, Any], task_params: dict[str, Any], *, persistent: bool
        ) -> dict[str, BindingResolutionState]:
    states: dict[str, BindingResolutionState] = {}
    for _ in range(max(1, len(requirements) + 1)):
        changed = False
        for name, requirement in requirements.items():
            state = _resolution_state(
                requirement, specs[name], params.get(name), task_params,
                states, persistent=persistent)
            if states.get(name) != state:
                states[name] = state
                changed = True
        if not changed:
            break
    return states


def _runtime_step(node: Any, index: int) -> str:
    return str(getattr(node, "step_id", "") or
               getattr(node, "origin_step_id", "") or f"step_{index:03d}")


def _incoming_flow_specs(plan: Any, registry: Any,
                         nodes: dict[str, tuple[int, Any]]) -> tuple[
                             dict[tuple[str, str], BindingSpec], list[str]]:
    incoming: dict[tuple[str, str], BindingSpec] = {}
    errors: list[str] = []
    for edge in list(getattr(plan, "edges", []) or []):
        if edge.type != EdgeType.DATA_FLOW:
            continue
        source_entry, target_entry = nodes.get(edge.source_step), nodes.get(edge.target_step)
        source_output = str((edge.mapping or {}).get("source_output") or "")
        target_input = str((edge.mapping or {}).get("target_input") or "")
        if source_entry is None or target_entry is None:
            errors.append(f"data_flow_unknown_step:{edge.edge_id}")
            continue
        if source_entry[0] >= target_entry[0]:
            errors.append(f"data_flow_source_not_prior:{edge.edge_id}")
            continue
        source_atomic = registry.get(source_entry[1].ref)
        validation = validate_output_materializer(source_atomic, source_output)
        if not validation.passed:
            errors.append(
                f"data_flow_output_unmaterializable:{edge.source_step}:"
                f"{source_output}:{'|'.join(validation.errors)}")
            continue
        incoming[(edge.target_step, target_input)] = BindingSpec(
            BindingKind.DATA_FLOW, source_step=edge.source_step,
            source_output=source_output, symbol=f"$flow.{source_output}")
    return incoming, errors


def _resolution_state(requirement: SlotRequirement, spec: BindingSpec,
                      raw_value: Any, task_params: dict[str, Any],
                      states: dict[str, BindingResolutionState], *,
                      persistent: bool) -> BindingResolutionState:
    source = source_name_for_kind(spec.kind)
    if source not in requirement.allowed_sources:
        return BindingResolutionState.UNRESOLVABLE
    if is_concrete_binding(raw_value):
        return BindingResolutionState.RESOLVED
    if spec.kind == BindingKind.LITERAL:
        return (BindingResolutionState.RESOLVED
                if is_concrete_binding(spec.value)
                else BindingResolutionState.UNRESOLVABLE)
    if spec.kind == BindingKind.TASK:
        if persistent and spec.task_role:
            return BindingResolutionState.RESOLVED
        return (BindingResolutionState.RESOLVED
                if is_concrete_binding(task_params.get(spec.task_role))
                else BindingResolutionState.UNRESOLVABLE)
    if spec.kind == BindingKind.DATA_FLOW:
        return (BindingResolutionState.PENDING_DATA_FLOW
                if spec.source_step and spec.source_output
                else BindingResolutionState.UNRESOLVABLE)
    anchors_closed = bool(requirement.anchor_roles) and all(
        states.get(role) in {BindingResolutionState.RESOLVED,
                             BindingResolutionState.PENDING_DATA_FLOW}
        for role in requirement.anchor_roles)
    if spec.kind == BindingKind.STATE:
        selector_present = bool(spec.state_predicate or spec.state_entity_role)
        if selector_present and requirement.runtime_resolvable and anchors_closed:
            return BindingResolutionState.RUNTIME_RESOLVABLE
        return BindingResolutionState.UNRESOLVABLE
    if (spec.kind == BindingKind.UNRESOLVED and requirement.runtime_resolvable
            and anchors_closed):
        return BindingResolutionState.RUNTIME_RESOLVABLE
    return BindingResolutionState.UNRESOLVABLE


def _provenance(name: str, spec: BindingSpec,
                state: BindingResolutionState) -> BindingProvenance:
    if spec.provenance is not None:
        return spec.provenance
    return BindingProvenance(
        source=source_name_for_kind(spec.kind), role=name,
        source_step=spec.source_step, source_output=spec.source_output,
        evidence=(state.value,))


def _atomic_of_step(step: dict[str, Any] | None, registry: Any) -> Any:
    if step is None:
        return None
    try:
        from ..core.refs import SkillRef
        return registry.get(SkillRef.parse(str(step.get("node_ref") or "")))
    except (TypeError, ValueError):
        return None
