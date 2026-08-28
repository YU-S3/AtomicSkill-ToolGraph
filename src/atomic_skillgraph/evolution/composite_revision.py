"""Persist Composite revision lineage without mutating historical structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.refs import SkillRef
from ..core.status import EdgeType, SkillStatus
from .trace_graph_reconstructor import TraceGraphRevision


_INSERT_REVISIONS = {
    "new_capability_insert",
    "existing_capability_insert",
    "repeated_occurrence_insert",
}


@dataclass
class CompositeRevisionResult:
    revision_kind: str = "no_revision"
    composite_ref: str = ""
    derived_from: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)
    suppressed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision_kind": self.revision_kind,
            "composite_ref": self.composite_ref,
            "derived_from": self.derived_from,
            "supersedes": self.supersedes,
            "suppressed": self.suppressed,
        }


class CompositeRevisionBuilder:
    """Attach lineage and govern an exact selected parent Composite."""

    def __init__(self, registry) -> None:
        self.registry = registry

    def apply(self, composite, revision: TraceGraphRevision | None, *,
              trace_id: str) -> CompositeRevisionResult:
        revision = revision or TraceGraphRevision()
        outcome = CompositeRevisionResult(
            revision_kind=revision.revision_kind,
            composite_ref=str(composite.ref),
        )
        metadata = dict(composite.metadata or {})
        history = list(metadata.get("graph_revision_history") or [])
        record = revision.to_dict()
        record["source_trace_id"] = trace_id
        if not any(str(item.get("source_trace_id") or "") == trace_id
                   for item in history):
            history.append(record)
        metadata["graph_revision_history"] = history[-100:]

        derived = list(metadata.get("derived_from_refs") or [])
        parent_ref = str(revision.selected_composite_ref or "")
        if (revision.revision_kind in _INSERT_REVISIONS and parent_ref
                and parent_ref != str(composite.ref)):
            if parent_ref not in derived:
                derived.append(parent_ref)
            self.registry.add_edge(
                str(composite.ref), parent_ref, EdgeType.DERIVED_FROM,
                evidence=[trace_id], metadata={
                    "revision_kind": revision.revision_kind,
                    "task_gap_proof": bool(
                        revision.task_gap_proved_missing_effect),
                })
            outcome.derived_from.append(parent_ref)
        metadata["derived_from_refs"] = derived
        composite.metadata = metadata
        self.registry.update_runtime_state(composite)

        # Promotion establishes replacement; a Draft records derivation only.
        if composite.status == SkillStatus.ACTIVE:
            for ancestor in derived:
                if ancestor == str(composite.ref):
                    continue
                self.registry.add_edge(
                    str(composite.ref), ancestor, EdgeType.SUPERSEDES,
                    evidence=[trace_id], metadata={
                        "reason": "validated_revised_composite_promoted"})
                outcome.supersedes.append(ancestor)

        if (revision.task_gap_proved_missing_effect
                and revision.revision_kind in _INSERT_REVISIONS):
            suppressed = self.suppress_proven_incomplete_parent(
                revision, trace_id=trace_id,
                replacement_ref=str(composite.ref))
            if suppressed:
                outcome.suppressed.append(suppressed)
        return outcome

    def suppress_proven_incomplete_parent(
            self, revision: TraceGraphRevision, *, trace_id: str,
            replacement_ref: str = "") -> str:
        """Suppress only the exact parent proven non-self-sufficient by code."""
        if not revision.task_gap_proved_missing_effect:
            return ""
        parent_text = str(revision.selected_composite_ref or "")
        if not parent_text or parent_text == replacement_ref:
            return ""
        try:
            parent_ref = SkillRef.parse(parent_text)
        except ValueError:
            return ""
        parent = self.registry.get(parent_ref)
        if parent is None or parent.status in {
                SkillStatus.SUPPRESSED, SkillStatus.RETIRED}:
            return ""
        # Evidence counters are updated once by system._update_skill_evidence
        # from ``selected_composite_self``.  Revision application owns only
        # structural lineage/status; incrementing here would double count the
        # same successful trace before the system evidence phase runs.
        parent.metadata["suppression_reason"] = (
            "task_gap_proved_missing_effect")
        parent.metadata["suppression_trace_id"] = trace_id
        if replacement_ref:
            parent.metadata["suppressed_by_revision_ref"] = replacement_ref
        parent.status = SkillStatus.SUPPRESSED
        self.registry.update_runtime_state(parent)
        return str(parent.ref)
