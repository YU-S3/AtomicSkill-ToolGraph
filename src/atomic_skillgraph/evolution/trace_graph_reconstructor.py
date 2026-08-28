"""Reconstruct an auditable graph revision from one successful runtime trace."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..core.binding_ir import binding_slot_name, is_concrete_binding
from ..core.refs import SkillRef
from ..runtime.contract_matcher import match_effect_contract


@dataclass
class TraceGraphRevision:
    selected_composite_ref: str = ""
    realized_occurrences: list[dict[str, Any]] = field(default_factory=list)
    inserted_occurrences: list[dict[str, Any]] = field(default_factory=list)
    reused_atomic_refs: list[str] = field(default_factory=list)
    new_atomic_refs: list[str] = field(default_factory=list)
    replaced_occurrences: list[dict[str, Any]] = field(default_factory=list)
    gap_effects: list[dict[str, Any]] = field(default_factory=list)
    revision_kind: str = "no_revision"
    task_gap_proved_missing_effect: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_composite_ref": self.selected_composite_ref,
            "realized_occurrences": self.realized_occurrences,
            "inserted_occurrences": self.inserted_occurrences,
            "reused_atomic_refs": self.reused_atomic_refs,
            "new_atomic_refs": self.new_atomic_refs,
            "replaced_occurrences": self.replaced_occurrences,
            "gap_effects": self.gap_effects,
            "revision_kind": self.revision_kind,
            "task_gap_proved_missing_effect": self.task_gap_proved_missing_effect,
            "notes": self.notes,
        }


class TraceGraphReconstructor:
    """Classify what a successful task-gap changed in the executed graph."""

    def __init__(self, registry) -> None:
        self.registry = registry

    def reconstruct(self, *, trace, atomic_result,
                    selected_composite=None) -> TraceGraphRevision:
        selected = selected_composite or self._selected_composite(trace)
        selected_ref = str(getattr(selected, "ref", "") or
                           getattr(trace, "selected_composite", "") or "")
        spans = [_span_dict(item) for item in
                 (getattr(trace, "runtime_spans", None) or [])]
        candidates = list(getattr(atomic_result, "candidates", None) or [])
        segments = list(getattr(atomic_result, "segments", None) or [])
        decisions = list(getattr(atomic_result, "decisions", None) or [])
        extracted_occurrences: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            segment = dict(segments[index] if index < len(segments)
                           else getattr(candidate, "segment", None) or {})
            source_kind, occurrence_id = _segment_source(segment, spans)
            decision = str(decisions[index] if index < len(decisions) else
                           ("reuse" if getattr(candidate, "alignment", None)
                            and candidate.alignment.matched else "add"))
            extracted_occurrences.append({
                "phase_id": str(segment.get("phase_id") or f"phase_{index:03d}"),
                "skill_ref": str(candidate.skill.ref),
                "source_kind": source_kind,
                "runtime_occurrence_id": occurrence_id,
                "event_start": segment.get("event_start"),
                "event_end": segment.get("event_end"),
                "params": dict(segment.get("params") or {}),
                "before": dict(segment.get("before") or {}),
                "after": dict(segment.get("after") or {}),
                "preconditions": [dict(item) for item in
                                  (segment.get("preconditions") or [])
                                  if isinstance(item, dict)],
                "effect": [dict(item) for item in
                           (segment.get("effect") or [])
                           if isinstance(item, dict)],
                "negative_effect": [dict(item) for item in
                                    (segment.get("negative_effect") or [])
                                    if isinstance(item, dict)],
                "extraction_method": str(
                    segment.get("extraction_method") or ""),
                "atomic_decision": decision,
            })

        # A selected parent is immutable graph evidence.  Extractor phases are
        # allowed to omit a zero-action ALREADY_SATISFIED occurrence, but a
        # graph revision must not consequently forget that parent step.  Start
        # from every exact, code-validated parent occurrence and insert only
        # the extracted task-gap phases into that complete sequence.
        parent_complete = True
        if selected is not None and selected_ref:
            parent_occurrences = self._validated_parent_occurrences(
                trace, selected, spans)
            parent_complete = (
                len(parent_occurrences) == len(selected.step_instances())
                and all(bool(item.get("passed"))
                        for item in parent_occurrences))
            gap_occurrences = [item for item in extracted_occurrences
                               if item.get("source_kind") == "task_gap"]
            occurrences = parent_occurrences + gap_occurrences
        else:
            occurrences = extracted_occurrences

        gap_effects = _gap_effects(trace)
        inserted = [item for item in occurrences
                    if item.get("source_kind") == "task_gap"]
        reused = [str(item["skill_ref"]) for item in occurrences
                  if str(item.get("atomic_decision") or "").startswith("reuse")]
        new = [str(item["skill_ref"]) for item in occurrences
               if not str(item.get("atomic_decision") or "").startswith("reuse")]
        revision = TraceGraphRevision(
            selected_composite_ref=selected_ref,
            realized_occurrences=occurrences,
            inserted_occurrences=inserted,
            reused_atomic_refs=list(dict.fromkeys(reused)),
            new_atomic_refs=list(dict.fromkeys(new)),
            gap_effects=gap_effects,
        )
        gap_effect_coverage = _gap_occurrences_cover_missing_effects(
            inserted, gap_effects,
            dict((getattr(trace, "provenance", None) or {}).get(
                "realized_params") or {}))
        revision.task_gap_proved_missing_effect = _strong_gap_proof(
            trace, selected, spans, gap_effects, inserted,
            action_effect_coverage=gap_effect_coverage)
        if selected is not None and inserted and not parent_complete:
            revision.revision_kind = "no_revision"
            revision.notes.append(
                "selected_parent_occurrence_reconstruction_incomplete")
            return revision
        if selected is not None and inserted and gap_effects \
                and not gap_effect_coverage:
            revision.revision_kind = "observation_only_gap"
            revision.notes.append(
                "task_gap_actions_did_not_produce_the_missing_effects")
            return revision

        analysis = getattr(trace, "task_gap_analysis", None)
        analysis_dict = (analysis.to_dict() if hasattr(analysis, "to_dict")
                         else dict(analysis or {}))
        has_finalization_span = any(
            span.get("kind") == "benchmark_finalization" for span in spans)
        # ``benchmark_only_finalization`` in TaskGapAnalysis means the formal
        # target was already satisfied at the pre-gap boundary; it does *not*
        # mean that the whole successful trace contains no learnable planned
        # capabilities.  Classify this special case only when an actual
        # finalization span exists and the Extractor produced no learnable
        # occurrence at all.  Otherwise cold successful workflows would never
        # be allowed to form their first Composite.
        if (analysis_dict.get("benchmark_only_finalization")
                and has_finalization_span and not occurrences
                and not inserted and not gap_effects):
            revision.revision_kind = "benchmark_finalization_only"
            return revision
        if selected is None or not selected_ref:
            revision.revision_kind = "no_revision"
            return revision
        if not inserted:
            if gap_effects:
                revision.notes.append(
                    "task_gap_was_code_verified_but_no_validated_atomic_phase_was_extracted")
            revision.revision_kind = "no_revision"
            return revision

        selected_steps = list(selected.step_instances())
        selected_contracts = {
            self._contract_key(str(step.get("node_ref") or ""))
            for step in selected_steps
        }
        inserted_contracts = {
            self._contract_key(str(item.get("skill_ref") or ""))
            for item in inserted
        }
        repeated = bool(inserted_contracts & selected_contracts)
        if repeated and self._matching_planned_occurrence_failed(
                trace, inserted_contracts, spans):
            revision.revision_kind = "implementation_repair"
            revision.replaced_occurrences = self._failed_matching_occurrences(
                trace, inserted_contracts, spans)
        elif repeated:
            revision.revision_kind = "repeated_occurrence_insert"
        elif any(not str(item.get("atomic_decision") or "").startswith("reuse")
                 for item in inserted):
            revision.revision_kind = "new_capability_insert"
        else:
            revision.revision_kind = "existing_capability_insert"
        return revision

    def _validated_parent_occurrences(
            self, trace, selected, spans: list[dict[str, Any]],
            ) -> list[dict[str, Any]]:
        """Rebuild the exact selected-parent occurrence sequence from runtime.

        The occurrence may legitimately contain zero actions.  Its validation,
        exact child version, bindings and order still come from the Runtime
        Graph, so it remains part of a revised Composite.
        """
        realized = [dict(item) for item in
                    (getattr(trace, "realized_atomic_nodes", None) or [])]
        gap_ids = {str(span.get("occurrence_id") or "") for span in spans
                   if span.get("kind") == "task_gap"}
        planned = [item for item in realized
                   if str(item.get("occurrence_id") or "") not in gap_ids
                   and not str(item.get("ref") or "").startswith(
                       "skill://runtime.dynamic.task_gap")]
        spans_by_occurrence = {
            str(span.get("occurrence_id") or ""): span for span in spans
            if span.get("kind") == "planned_node"
        }
        used: set[int] = set()
        result: list[dict[str, Any]] = []
        for index, step in enumerate(selected.step_instances()):
            origin_step = str(step.get("step_id") or f"step_{index:03d}")
            expected_ref = _normalized_skill_ref(
                str(step.get("node_ref") or ""))
            match_index = next((
                candidate_index for candidate_index, node in enumerate(planned)
                if candidate_index not in used
                and str(node.get("origin_step_id") or "") == origin_step
                and _normalized_skill_ref(str(node.get("ref") or ""))
                == expected_ref), None)
            # Compatibility for old traces that predate origin_step_id: retain
            # exact order and version, never match merely by logical id.
            if match_index is None:
                match_index = next((
                    candidate_index for candidate_index, node in enumerate(planned)
                    if candidate_index not in used
                    and _normalized_skill_ref(str(node.get("ref") or ""))
                    == expected_ref), None)
            if match_index is None:
                continue
            used.add(match_index)
            node = planned[match_index]
            occurrence_id = str(node.get("occurrence_id") or
                                f"parent_{index:03d}")
            span = spans_by_occurrence.get(occurrence_id, {})
            try:
                atomic = self.registry.get(SkillRef.parse(expected_ref))
            except ValueError:
                atomic = None
            result.append({
                "phase_id": f"parent_{origin_step}",
                "skill_ref": expected_ref,
                "source_kind": "planned_node",
                "runtime_occurrence_id": occurrence_id,
                "origin_step_id": origin_step,
                "event_start": int(span.get("action_start", 0) or 0),
                "event_end": int(span.get("action_end", 0) or 0),
                "params": dict(node.get("params") or {}),
                "before": dict(node.get("before") or {}),
                "after": dict(node.get("after") or node.get("before") or {}),
                "preconditions": [dict(item) for item in
                                  (getattr(atomic, "preconditions", []) or [])
                                  if isinstance(item, dict)],
                "effect": [dict(item) for item in
                           (getattr(atomic, "effects", []) or [])
                           if isinstance(item, dict)],
                "negative_effect": [dict(item) for item in
                                    ((getattr(atomic, "metadata", {}) or {}).get(
                                        "observed_negative_effects") or [])
                                    if isinstance(item, dict)],
                "extraction_method": "validated_parent_occurrence",
                "atomic_decision": "reuse_exact_parent",
                "execution_status": str(node.get("execution_status") or ""),
                "passed": bool(node.get("passed")),
            })
        return result

    def _selected_composite(self, trace):
        text = str(getattr(trace, "selected_composite", "") or "")
        if not text:
            return None
        try:
            return self.registry.get(SkillRef.parse(text))
        except ValueError:
            return None

    def _contract_key(self, ref_text: str) -> str:
        try:
            ref = SkillRef.parse(ref_text)
            atomic = self.registry.get(ref)
        except ValueError:
            atomic = None
        if atomic is None:
            return _logical_ref(ref_text)
        effects = [item for item in (getattr(atomic, "effects", None) or [])
                   if isinstance(item, dict)]
        return json.dumps(effects, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"))

    def _matching_planned_occurrence_failed(
            self, trace, contracts: set[str], spans: list[dict[str, Any]]) -> bool:
        return bool(self._failed_matching_occurrences(trace, contracts, spans))

    def _failed_matching_occurrences(
            self, trace, contracts: set[str], spans: list[dict[str, Any]],
            ) -> list[dict[str, Any]]:
        gap_ids = {str(span.get("occurrence_id") or "") for span in spans
                   if span.get("kind") == "task_gap"}
        result: list[dict[str, Any]] = []
        for node in getattr(trace, "realized_atomic_nodes", None) or []:
            if str(node.get("occurrence_id") or "") in gap_ids:
                continue
            if (not bool(node.get("passed"))
                    and self._contract_key(str(node.get("ref") or
                                               node.get("node_ref") or ""))
                    in contracts):
                result.append({
                    "occurrence_id": str(node.get("occurrence_id") or ""),
                    "step_id": str(node.get("step_id") or ""),
                    "atomic_ref": str(node.get("ref") or node.get("node_ref") or ""),
                    "implementation_ref": str(node.get("impl_ref") or ""),
                    "tool_refs": list(node.get("tool_refs") or []),
                })
        return result


def _strong_gap_proof(trace, selected, spans: list[dict[str, Any]],
                      gap_effects: list[dict[str, Any]],
                      inserted: list[dict[str, Any]], *,
                      action_effect_coverage: bool) -> bool:
    if selected is None or not bool(getattr(trace, "success", False)) or not gap_effects:
        return False
    if not inserted or not action_effect_coverage:
        return False
    gap_spans = [span for span in spans if span.get("kind") == "task_gap"]
    if not gap_spans or not any(int(span.get("action_end", 0)) >
                                int(span.get("action_start", 0))
                                for span in gap_spans):
        return False
    gap_ids = {str(span.get("occurrence_id") or "") for span in gap_spans}
    planned = [node for node in (getattr(trace, "realized_atomic_nodes", None) or [])
               if str(node.get("occurrence_id") or "") not in gap_ids
               and not str(node.get("ref") or "").startswith("skill://runtime.dynamic.task_gap")]
    used: set[int] = set()
    for step in selected.step_instances():
        origin_step = str(step.get("step_id") or "")
        expected_ref = _normalized_skill_ref(str(step.get("node_ref") or ""))
        match_index = next((
            index for index, node in enumerate(planned)
            if index not in used
            and str(node.get("origin_step_id") or "") == origin_step
            and _normalized_skill_ref(str(node.get("ref") or ""))
            == expected_ref), None)
        if match_index is None:
            match_index = next((
                index for index, node in enumerate(planned)
                if index not in used
                and _normalized_skill_ref(str(node.get("ref") or ""))
                == expected_ref), None)
        if match_index is None or not bool(planned[match_index].get("passed")):
            return False
        used.add(match_index)
    gap_passed = any(
        getattr(item, "level", "") == "task_gap" and bool(getattr(item, "passed", False))
        for item in (getattr(trace, "node_validators", None) or []))
    gap_passed = gap_passed or any(
        bool(node.get("passed")) and
        str(node.get("occurrence_id") or "") in gap_ids
        for node in (getattr(trace, "realized_atomic_nodes", None) or []))
    return gap_passed


def _gap_occurrences_cover_missing_effects(
        inserted: list[dict[str, Any]], missing_effects: list[dict[str, Any]],
        realized_params: dict[str, Any] | None = None) -> bool:
    """Require action-caused, Extractor-validated positive delta evidence.

    A fact newly visible after opening/observing a container is deliberately
    absent from an origin-aware phase's ``effect`` list.  Such a reveal cannot
    prove that the selected Composite lacked a state-changing capability.
    """
    if not inserted or not missing_effects:
        return False
    base_bindings = dict(realized_params or {})
    for raw_target in missing_effects:
        if not isinstance(raw_target, dict):
            return False
        required = max(1, int(raw_target.get("cardinality", 1) or 1))
        target = _ground_effect_contract(
            {**raw_target, "cardinality": 1, "distinct_by": ""},
            base_bindings)
        if target is None:
            return False
        distinct_role = str(raw_target.get("distinct_by") or "")
        covered = 0
        distinct_witnesses: set[str] = set()
        for occurrence in inserted:
            occurrence_bindings = dict(occurrence.get("params") or {})
            for effect in occurrence.get("effect") or []:
                grounded_effect = (_ground_effect_contract(
                    effect, occurrence_bindings)
                    if isinstance(effect, dict) else None)
                if (grounded_effect is None
                        or not match_effect_contract(
                            grounded_effect, target).passed):
                    continue
                if distinct_role:
                    value = (dict(grounded_effect.get("args") or {}).get(
                        distinct_role) or occurrence_bindings.get(distinct_role))
                    if not is_concrete_binding(value):
                        continue
                    witness = str(value)
                    if witness in distinct_witnesses:
                        continue
                    distinct_witnesses.add(witness)
                    covered += 1
                else:
                    covered += max(
                        1, int(grounded_effect.get("cardinality", 1) or 1))
                if covered >= required:
                    break
            if covered >= required:
                break
        if covered < required:
            return False
    return True


def _ground_effect_contract(effect: dict[str, Any],
                            bindings: dict[str, Any]) \
        -> dict[str, Any] | None:
    """Ground one predicate in its own binding scope, never a merged scope."""
    grounded = dict(effect)
    args: dict[str, Any] = {}
    for name, raw in dict(effect.get("args") or {}).items():
        role = binding_slot_name(raw)
        value = bindings.get(role) if role else raw
        if not is_concrete_binding(value):
            return None
        args[str(name)] = value
    grounded["args"] = args
    return grounded


def _segment_source(segment: dict[str, Any], spans: list[dict[str, Any]]) \
        -> tuple[str, str]:
    explicit_kind = str(segment.get("source_kind") or "")
    explicit_occurrence = str(segment.get("runtime_occurrence_id") or "")
    if explicit_kind:
        return explicit_kind, explicit_occurrence
    start = segment.get("event_start")
    end = segment.get("event_end")
    if isinstance(start, int):
        end_value = int(end if isinstance(end, int) else start + 1)
        for span in spans:
            span_start = int(span.get("action_start", 0))
            span_end = int(span.get("action_end", span_start))
            if start >= span_start and end_value <= span_end:
                return str(span.get("kind") or "planned_node"), str(
                    span.get("occurrence_id") or "")
    return "planned_node", explicit_occurrence


def _gap_effects(trace) -> list[dict[str, Any]]:
    analysis = getattr(trace, "task_gap_analysis", None)
    payload = analysis.to_dict() if hasattr(analysis, "to_dict") else dict(analysis or {})
    return [dict(item) for item in (payload.get("missing_effects") or [])
            if isinstance(item, dict)]


def _span_dict(value) -> dict[str, Any]:
    return value.to_dict() if hasattr(value, "to_dict") else dict(value or {})


def _logical_ref(value: str) -> str:
    text = str(value or "")
    if text.startswith("skill://"):
        text = text[len("skill://"):]
    return text.rsplit("@", 1)[0]


def _normalized_skill_ref(value: str) -> str:
    try:
        return str(SkillRef.parse(str(value or "")))
    except ValueError:
        return str(value or "")
