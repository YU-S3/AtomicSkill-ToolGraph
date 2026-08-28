"""Deterministic Composite lifecycle; ordinary tasks never execute Drafts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..core.refs import SkillRef
from ..core.status import EdgeType, SkillNodeKind, SkillStatus
from ..graph.graph import composite_step_order
from ..runtime.plan_validator import validate_composite_binding_closure
from ..validation.composite_validator import CompositeFailureCode


@dataclass(frozen=True)
class CompositeLifecycleDecision:
    status: SkillStatus
    reason: str


def evaluate_composite(composite, registry, *, min_support: int
                       ) -> CompositeLifecycleDecision:
    closure = validate_composite_binding_closure(composite, registry)
    composite.metadata["binding_closure"] = closure.to_dict()
    _steps, graph = composite_step_instances_for_validation(composite, registry)
    composite.metadata["graph_validation"] = {
        "passed": graph.passed, "errors": list(graph.errors)}
    if not closure.passed:
        return CompositeLifecycleDecision(SkillStatus.SHADOW,
                                          "binding_closure_failed")
    if not graph.passed:
        return CompositeLifecycleDecision(SkillStatus.SHADOW,
                                          "graph_validation_failed")
    child_statuses = _child_statuses(composite, registry)
    if any(status in {SkillStatus.SHADOW, SkillStatus.SUPPRESSED,
                      SkillStatus.RETIRED} for status in child_statuses):
        return CompositeLifecycleDecision(SkillStatus.SHADOW,
                                          "child_structurally_unusable")
    if any(status == SkillStatus.DRAFT for status in child_statuses):
        return CompositeLifecycleDecision(SkillStatus.DRAFT,
                                          "awaiting_child_activation")
    support = int((composite.metadata.get("statistics") or {}).get(
        "support_count", 0))
    if support >= max(2, int(min_support)):
        return CompositeLifecycleDecision(SkillStatus.ACTIVE,
                                          "independent_support_satisfied")
    return CompositeLifecycleDecision(SkillStatus.DRAFT,
                                      "awaiting_independent_support")


def composite_step_instances_for_validation(composite, registry):
    # A Draft child is a valid exact version and blocks execution/promotion, but
    # it does not make the Composite structure invalid.  Normal runtime callers
    # keep composite_step_order's strict Active-only default.
    return composite_step_order(
        composite, registry, allow_draft_children=True)


def reevaluate_waiting_composites(registry, *, min_support: int
                                  ) -> list[dict[str, str]]:
    """Re-evaluate blocked Drafts after Atomic evidence changes.

    This closes the common two-trace sequence: trace one creates a Draft Atomic
    and a Draft Composite; trace two activates the Atomic before the matching
    Composite occurrence is aligned and promoted.
    """
    events: list[dict[str, str]] = []
    for composite in registry.list_all_versions(SkillNodeKind.COMPOSITE):
        if composite.status != SkillStatus.DRAFT:
            continue
        before = composite.status
        decision = evaluate_composite(
            composite, registry, min_support=min_support)
        composite.status = decision.status
        composite.metadata.setdefault("candidate", {})[
            "lifecycle_reason"] = decision.reason
        registry.update_runtime_state(composite)
        if before != SkillStatus.ACTIVE and decision.status == SkillStatus.ACTIVE:
            # A revised Composite may become promotable only after a Draft
            # child receives independent support.  Promotion itself establishes
            # replacement lineage even if no later trace happens to align the
            # Composite and call CompositeRevisionBuilder.apply again.
            for ancestor in list(
                    composite.metadata.get("derived_from_refs") or []):
                ancestor_ref = str(ancestor or "")
                if not ancestor_ref or ancestor_ref == str(composite.ref):
                    continue
                registry.add_edge(
                    str(composite.ref), ancestor_ref, EdgeType.SUPERSEDES,
                    evidence=[f"lifecycle_promotion:{composite.ref}"],
                    metadata={
                        "reason": "validated_revised_composite_promoted",
                        "origin": "composite_lifecycle_reevaluation",
                    })
        if decision.status != before:
            events.append({
                "composite_ref": str(composite.ref),
                "from": before.value,
                "to": decision.status.value,
                "reason": decision.reason,
            })
    return events


def apply_self_sufficient_evidence(
        composite, registry, *, passed: bool,
        failure_codes: Iterable[str] = (),
        task_gap_proved_missing_effect: bool = False,
        minimum_rate: float = 0.5) -> CompositeLifecycleDecision:
    """Record one self-sufficient outcome and apply deterministic suppression.

    Callers should invoke this exactly once for the selected Composite at the
    pre-gap boundary.  Task/benchmark success is deliberately not an input.
    """
    codes = {str(getattr(code, "value", code)) for code in failure_codes}
    stats = dict(composite.metadata.get("statistics") or {})
    stats["self_sufficient_validation_count"] = int(
        stats.get("self_sufficient_validation_count", 0)) + 1
    if passed:
        stats["self_sufficient_success_count"] = int(
            stats.get("self_sufficient_success_count", 0)) + 1
        stats["consecutive_self_sufficient_failures"] = 0
    else:
        stats["self_sufficient_failure_count"] = int(
            stats.get("self_sufficient_failure_count", 0)) + 1
        stats["consecutive_self_sufficient_failures"] = int(
            stats.get("consecutive_self_sufficient_failures", 0)) + 1
    if CompositeFailureCode.TASK_GAP_REQUIRED.value in codes:
        stats["task_gap_required_count"] = int(
            stats.get("task_gap_required_count", 0)) + 1
    composite.metadata["statistics"] = stats
    composite.metadata["last_composite_failure_codes"] = sorted(codes)

    reason = "self_sufficient_success" if passed else "self_sufficient_failure"
    immediate_structural = bool(codes & {
        CompositeFailureCode.DATA_FLOW_FAILED.value,
        CompositeFailureCode.CHILD_VERSION_MISSING.value,
    })
    if immediate_structural:
        reason = "structured_runtime_validation_failed"
        composite.status = SkillStatus.SUPPRESSED
    elif task_gap_proved_missing_effect:
        reason = "task_gap_proved_missing_effect"
        composite.status = SkillStatus.SUPPRESSED
    elif int(stats.get("consecutive_self_sufficient_failures", 0)) >= 2:
        reason = "consecutive_self_sufficient_failures"
        composite.status = SkillStatus.SUPPRESSED
    else:
        count = int(stats.get("self_sufficient_validation_count", 0))
        rate = int(stats.get("self_sufficient_success_count", 0)) / max(count, 1)
        if count >= 3 and rate < float(minimum_rate):
            reason = "low_self_sufficient_success_rate"
            composite.status = SkillStatus.SUPPRESSED

    if composite.status == SkillStatus.SUPPRESSED:
        composite.metadata["suppression_reason"] = reason
    registry.update_runtime_state(composite)
    return CompositeLifecycleDecision(composite.status, reason)


def _child_statuses(composite, registry) -> list[SkillStatus]:
    statuses: list[SkillStatus] = []
    for step in composite.step_instances():
        try:
            child = registry.get(SkillRef.parse(str(step.get("node_ref") or "")))
        except ValueError:
            child = None
        if child is not None:
            statuses.append(child.status)
    return statuses
