"""Deterministic Composite lifecycle; ordinary tasks never execute Drafts."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.status import SkillStatus
from ..graph.graph import composite_step_order
from ..runtime.plan_validator import validate_composite_binding_closure


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
    support = int((composite.metadata.get("statistics") or {}).get(
        "support_count", 0))
    if support >= max(2, int(min_support)):
        return CompositeLifecycleDecision(SkillStatus.ACTIVE,
                                          "independent_support_satisfied")
    return CompositeLifecycleDecision(SkillStatus.DRAFT,
                                      "awaiting_independent_support")


def composite_step_instances_for_validation(composite, registry):
    # Draft children themselves must still be exact Active Atomic versions;
    # composite_step_order checks this and current runtime control support.
    return composite_step_order(composite, registry)
