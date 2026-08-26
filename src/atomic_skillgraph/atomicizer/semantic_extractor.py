"""LLM-assisted semantic trace extraction with deterministic evidence gates.

The agent may *propose* macro phases and implicit dependencies.  It never writes
the graph directly: event ranges, effects, parameters and causal relevance are
reconstructed and checked from the persisted trace by code.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..core.predicates import (
    _fact_to_predicate,
    StateSnapshot,
    bind_args,
    compute_effects,
    evaluate_predicate,
    normalize_value,
)
from ..core.trace_ir import TraceRecord
from .effect_extractor import (
    _FACT_FAMILY_NAMES,
    _family_of,
    extract_effect,
    parameterize_predicates,
)


EXTRACTOR_PROMPT = """You are the Trace Extractor Agent in a reusable capability-learning system.
You have not been given a benchmark taxonomy, a task type, a catalogue of operations, or a predefined
workflow. Your input is one successful trace normalized into structured events. Every event contains
the executed action, grounded arguments, the state immediately before and after it, positive and
negative state deltas, acceptance, and execution provenance.

The adapter separates positive_effects (world transitions attributable to the accepted action) from
observed_effects (pre-existing facts newly revealed by perception). observed_effects may justify a
later external Precondition or setup search, but they are not capability Effects and must never define
an Atomic occurrence or be copied into effect_predicates.

Some environments expose a successful terminal relation without repeating it in the final observation.
In that case an event may contain terminal_verified_effects. Each such item is a concrete target
relation independently certified by code from benchmark success plus all required terminal state facts.
It may anchor a terminal Atomic occurrence and be copied into effect_predicates. It is not an ordinary
single-action delta: infer the smallest causal terminal occurrence from its evidence and never claim
that the final action alone implements the relation.

Discover the smallest sufficient set of reusable Atomic capability occurrences from this evidence.
An Atomic occurrence is an independently meaningful and independently verifiable state transition
with a coherent intent, explicit external inputs/outputs, and a minimal causal action subsequence.
Infer its boundary; do not map the trace onto an assumed list of operations. Repeated checks, failed
attempts, loops, exploration branches, and recovery detours are not capabilities unless an accepted
event from them is causally necessary for the verified transition.

Use state evidence rather than action wording as authority. Setup actions and temporary helper state
belong inside an occurrence only when they are necessary to replay its core transition. A durable
transition should be a separate occurrence when it is independently useful, independently consumed,
or required by the task goal. Never merge distinct effect-producing events merely to minimize the
number of phases: one phase has one primary effect-producing boundary, although one indivisible event
may have multiple simultaneous deltas. This rule must be applied from the observed data; no
domain-specific boundary rule is supplied.

Invent a concise snake_case intent that describes the reusable transition. The name must remain valid
when every concrete entity is replaced by another entity with the same semantic role. Do not include
instance identifiers, concrete object/location/device classes, task wording, or several sequential
intents in one name. If a learned contract in known_atomic_contracts has equivalent validated Effects,
compatible I/O, and the same atomic boundary, reuse its canonical_name; otherwise propose a new name.
Put every episode-specific value in parameter_roles. Include all external values needed to replay the
minimal occurrence, including a required execution position or resource when the state evidence shows
that the core action depends on it. In effect_predicates, copy only predicate names that are actually
present in the structured state deltas, without arguments. Never invent an action, entity, parameter,
state, Effect, Tool, Skill, or dependency.

In precondition_predicates, list only predicates whose truth is semantically necessary for the core
transition to be executable, not every fact that happened to be true beforehand. Incidental results
of earlier operations must be excluded. Code accepts only listed predicates that are actually
witnessed and grounded in the core event's before state.

Return ONLY JSON:
{
  "phases": [
    {
      "phase_id": "phase_000",
      "intent": "snake_case_capability_name",
      "event_start": 0,
      "event_end": 3,
      "parameter_roles": {"semantic_role": "observed value"},
      "effect_predicates": ["predicate.name.copied.from.event"],
      "precondition_predicates": ["necessary.predicate.from.before.state"],
      "rationale": "why this range is one capability"
    }
  ],
  "discarded_event_indices": [4, 5],
  "discard_reasons": {"4": "exploration|duplicate|loop|failed_attempt|recovery"},
  "workflow_summary": "short semantic summary"
}

Requirements: ranges must be valid, non-overlapping and ordered; effect_predicates must be non-empty;
precondition_predicates must be present (it may be an empty list);
prefer the fewest independently verifiable occurrences sufficient for the goal; do not infer success or
atomicity merely from action wording.
"""


GRAPH_PROMPT = """You are the Composite Graph Proposal Agent. You receive a successful task goal and
code-validated Atomic occurrences. Each occurrence includes its validated skill reference, event range,
parameters, preconditions, effects, and available tool references.

Propose the minimal reusable capability/tool-reference composition that achieves the task. Use only the
provided phase IDs, skill refs and tool refs. Preserve parameter roles. Prefer causal dependencies over
raw temporal adjacency. Do not include exploration, loops, retries, recovery, or implementation-internal
operations. Add an implicit dependency only when it is semantically necessary but not already explicit
from Effect-to-Precondition or parameter data flow.

Return ONLY JSON:
{
  "ordered_phase_ids": ["phase_000", "phase_001"],
  "summary": "what reusable method this Composite represents",
  "implicit_dependencies": [
    {"source_phase_id": "phase_000", "target_phase_id": "phase_001",
     "relation": "requires_skill", "reason": "short reason"}
  ],
  "tool_plan": [
    {"phase_id": "phase_000", "skill_ref": "skill://...@...", "tool_refs": ["tool://...@..."]}
  ]
}
"""


@dataclass
class SemanticExtraction:
    events: list[dict[str, Any]] = field(default_factory=list)
    phases: list[dict[str, Any]] = field(default_factory=list)
    discarded_event_indices: list[int] = field(default_factory=list)
    workflow_summary: str = ""
    method: str = "rule_fallback"
    errors: list[str] = field(default_factory=list)
    proposal: dict[str, Any] = field(default_factory=dict)
    validated_phases: list[dict[str, Any]] = field(default_factory=list)
    slice_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [_compact_event(event) for event in self.events],
            "phases": [_phase_audit_view(phase) for phase in self.phases],
            "proposal": self.proposal,
            "validated_phases": [_phase_audit_view(phase)
                                 for phase in self.validated_phases],
            "slice_diagnostics": self.slice_diagnostics,
            "discarded_event_indices": self.discarded_event_indices,
            "workflow_summary": self.workflow_summary,
            "method": self.method,
            "errors": self.errors,
        }


class SemanticExtractorAgent:
    """Independent LLM session for semantic segmentation and graph proposals."""

    def __init__(self, llm=None, *, thinking: str = "enabled") -> None:
        self.llm = llm
        self.thinking = thinking if thinking in {"enabled", "disabled"} else "enabled"

    def extract(self, trace: TraceRecord, *,
                known_atomic_contracts: list[dict[str, Any]] | None = None
                ) -> SemanticExtraction:
        events = build_structured_events(trace)
        result = SemanticExtraction(events=events)
        if not events or self.llm is None:
            result.errors.append("no_events_or_llm")
            return result
        payload = {
            "task": {
                "goal": trace.task_goal,
            },
            "events": [_compact_event(event) for event in events],
            # Later traces see the already validated catalog.  This is evidence
            # sharing between independent Extractor sessions, not shared chat
            # state: the LLM proposes reuse and code still verifies the Effect.
            "known_atomic_contracts": list(known_atomic_contracts or []),
        }
        try:
            response = self.llm.generate(
                instructions=EXTRACTOR_PROMPT,
                input_text=json.dumps(payload, ensure_ascii=False),
                temperature=0.1, thinking=self.thinking,
                structured_output=True)
            proposal = _split_overmerged_phase_proposals(
                events, _parse_json_object(response.text))
            result.proposal = proposal
            phases, errors = validate_phase_proposal(trace, events, proposal)
            result.validated_phases = phases
            result.errors.extend(errors)
            if phases:
                sliced, diagnostics = causal_slice(
                    phases, list(trace.provenance.get("target_effects") or []),
                    trace.initial_state(),
                    task_params=dict(trace.provenance.get("semantic_params")
                                     or trace.provenance.get("params") or {}))
                result.slice_diagnostics = diagnostics
                if sliced:
                    retained_prefix: list[dict[str, Any]] = []
                    for phase in sliced:
                        phase["replay_prefix_actions"] = [
                            dict(action) for action in retained_prefix]
                        retained_prefix.extend(
                            dict(action) for action in (phase.get("actions") or []))
                    result.phases = sliced
                    result.discarded_event_indices = sorted({
                        int(item) for item in proposal.get("discarded_event_indices", [])
                        if isinstance(item, int) and 0 <= item < len(events)
                    })
                    result.workflow_summary = str(proposal.get("workflow_summary") or "")[:500]
                    result.method = "llm_proposal_code_validated"
                else:
                    result.errors.append("causal_slice_missing_goal_or_capability_dependency")
        except Exception as exc:  # extraction failure must not turn task success into infra failure
            result.errors.append(f"extractor_error:{type(exc).__name__}:{exc}")
        return result

    def propose_graph(self, trace: TraceRecord,
                      occurrences: list[dict[str, Any]]) -> dict[str, Any]:
        if self.llm is None or len(occurrences) < 2:
            return {}
        payload = {
            "task": {"goal": trace.task_goal},
            "validated_occurrences": occurrences,
        }
        try:
            response = self.llm.generate(
                instructions=GRAPH_PROMPT,
                input_text=json.dumps(payload, ensure_ascii=False),
                temperature=0.1, thinking=self.thinking,
                structured_output=True)
            proposal = _parse_json_object(response.text)
            return validate_graph_proposal(proposal, occurrences)
        except Exception:
            return {}


def build_structured_events(trace: TraceRecord) -> list[dict[str, Any]]:
    """Convert persisted actions/snapshots into auditable transition events."""
    actions = [a.to_dict() if hasattr(a, "to_dict") else dict(a) for a in trace.actions]
    snapshots = [dict(item.get("state") or {}) for item in trace.state_snapshots]
    terminal_by_event: dict[int, list[dict[str, Any]]] = {}
    diagnostics = dict(trace.metrics.get("runtime_diagnostics") or {})
    for certificate in diagnostics.get("terminal_verified_effects") or []:
        if not isinstance(certificate, dict):
            continue
        try:
            event_index = int(certificate.get("action_index"))
        except (TypeError, ValueError):
            continue
        effect = certificate.get("effect")
        if (0 <= event_index < len(actions) and isinstance(effect, dict)
                and effect.get("predicate")):
            terminal_by_event.setdefault(event_index, []).append({
                **dict(effect),
                "certificate": {
                    key: value for key, value in certificate.items()
                    if key != "effect"
                },
            })
    events: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        before = snapshots[index] if index < len(snapshots) else {}
        after = snapshots[index + 1] if index + 1 < len(snapshots) else before
        positive, negative = compute_effects(StateSnapshot(before), StateSnapshot(after))
        observed_predicates = [
            predicate for fact in ((after.get("meta") or {}).get(
                "last_observed_facts") or [])
            if (predicate := _fact_to_predicate(str(fact))) is not None
        ]
        observed_keys = {repr(item) for item in observed_predicates}
        causal_positive = [item for item in positive
                           if repr(item) not in observed_keys]
        events.append({
            "event_index": index,
            "step": int(action.get("step", index)),
            "action": str(action.get("name") or ""),
            "name": str(action.get("name") or ""),
            "params": dict(action.get("params") or {}),
            "accepted": bool(action.get("accepted", True)),
            "mode": str(action.get("mode") or "dynamic"),
            "node_ref": str(action.get("node_ref") or ""),
            "tool_ref": str(action.get("tool_ref") or ""),
            "before": before,
            "after": after,
            "positive_effects": causal_positive,
            "terminal_verified_effects": terminal_by_event.get(index, []),
            "observed_effects": observed_predicates,
            "negative_effects": negative,
            "state_changed": bool(causal_positive or negative),
            "observation_changed": bool(observed_predicates),
        })
    return events


def _extract_origin_aware_effect(events: list[dict[str, Any]], start: int,
                                 end: int,
                                 params: dict[str, Any]) -> Any:
    """Materialize a phase contract without losing transition provenance.

    Structured events are the source of truth for action-caused deltas.  Their
    ``observed_effects`` are intentionally excluded here: newly visible facts
    may be used as knowledge by the runtime, but they are not capabilities
    produced by the action that revealed them.
    """
    positive: list[dict[str, Any]] = []
    negative: list[dict[str, Any]] = []
    positive_keys: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    negative_keys: set[tuple[str, tuple[tuple[str, str], ...]]] = set()

    for event in events[start:end + 1]:
        if not bool(event.get("accepted", True)):
            continue
        for raw in event.get("negative_effects") or []:
            if not isinstance(raw, dict):
                continue
            inner = raw.get("not") if isinstance(raw.get("not"), dict) else raw
            key = _predicate_key(inner)
            positive = [item for item in positive
                        if _predicate_key(item) != key]
            positive_keys.discard(key)
            if key not in negative_keys:
                negative.append(dict(raw))
                negative_keys.add(key)
        for raw in _event_capability_effects(event):
            if not isinstance(raw, dict):
                continue
            key = _predicate_key(raw)
            retained_negative: list[dict[str, Any]] = []
            for item in negative:
                inner = (item.get("not")
                         if isinstance(item.get("not"), dict) else item)
                if _predicate_key(inner) != key:
                    retained_negative.append(item)
            negative = retained_negative
            negative_keys.discard(key)
            if key not in positive_keys:
                positive.append(dict(raw))
                positive_keys.add(key)

    return extract_effect(
        events[start]["before"], events[end]["after"], params,
        positive_effects=positive, negative_effects=negative)


def _split_overmerged_phase_proposals(
        events: list[dict[str, Any]], proposal: dict[str, Any]) -> dict[str, Any]:
    """Split a proposed macro at independently observed Effect producers."""
    normalized = dict(proposal or {})
    expanded: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(proposal.get("phases") or []):
        if not isinstance(raw, dict):
            expanded.append(raw)
            continue
        try:
            start, end = int(raw["event_start"]), int(raw["event_end"])
        except (KeyError, TypeError, ValueError):
            expanded.append(dict(raw))
            continue
        if start < 0 or end < start or end >= len(events):
            expanded.append(dict(raw))
            continue
        producers: list[tuple[int, list[str]]] = []
        for index in range(start, end + 1):
            event = events[index]
            if not bool(event.get("accepted", True)):
                continue
            names = sorted({str(item.get("predicate") or "")
                            for item in _event_capability_effects(event)
                            if isinstance(item, dict) and item.get("predicate")})
            if names:
                producers.append((index, names))
        if len(producers) <= 1:
            expanded.append(dict(raw))
            continue
        base_id = str(raw.get("phase_id") or f"phase_{ordinal:03d}")
        for offset, (index, names) in enumerate(producers):
            item = dict(raw)
            item["phase_id"] = f"{base_id}_effect_{offset:02d}"
            item["event_start"] = index
            item["event_end"] = index
            item["effect_predicates"] = names
            item["rationale"] = (
                f"Code split independent effect producer event {index} from "
                f"the proposed macro; original rationale: "
                f"{str(raw.get('rationale') or '')}")[:500]
            expanded.append(item)
    normalized["phases"] = expanded
    normalized["code_boundary_normalization"] = {
        "method": "split_independent_effect_producers",
        "raw_phase_count": len(proposal.get("phases") or []),
        "normalized_phase_count": len(expanded),
    }
    return normalized


def validate_phase_proposal(trace: TraceRecord, events: list[dict[str, Any]],
                            proposal: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Materialize phases exclusively from trace evidence; reject invented fields."""
    errors: list[str] = []
    phases: list[dict[str, Any]] = []
    occupied: set[int] = set()
    previous_core_end = -1
    known_values = _known_values(trace, events)
    for ordinal, raw in enumerate(proposal.get("phases") or []):
        if not isinstance(raw, dict):
            errors.append(f"phase_{ordinal}:not_object")
            continue
        try:
            start, end = int(raw["event_start"]), int(raw["event_end"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"phase_{ordinal}:invalid_range")
            continue
        raw_start, raw_end = start, end
        indices = set(range(start, end + 1))
        if start < 0 or end < start or end >= len(events) or occupied & indices:
            errors.append(f"phase_{ordinal}:range_out_of_bounds_or_overlap")
            continue
        declared = {_declared_predicate_name(item)
                    for item in (raw.get("effect_predicates") or [])}
        declared.discard("")
        semantic_core_names = set(declared)
        if not semantic_core_names:
            errors.append(f"phase_{ordinal}:missing_effect_predicates")
            continue
        evidence_end = (_last_core_effect_event(events, start, end,
                                                semantic_core_names)
                        if semantic_core_names else None)
        roles = _validated_roles(dict(raw.get("parameter_roles") or {}), known_values)
        parameter_end = evidence_end if evidence_end is not None else end
        observed_params = _event_params(events[start:parameter_end + 1])
        roles = _refine_roles_with_phase_evidence(roles, observed_params)
        # LLM supplies semantic role names, but values must be trace-grounded.
        # Core-event roles take precedence over a broader task-level synonym so
        # two semantically different locations cannot collapse into one slot.
        params = dict(roles)
        # Preserve the executed action's semantic role even when the LLM used
        # the same entity value under a different task-level role.  For
        # For example, a final destination must not suppress an observed source
        # location merely because both values share an entity family.
        for key, value in observed_params.items():
            if value not in (None, ""):
                params[_canonical_role_name(str(key))] = value
        params.update(_infer_execution_location_roles(
            events, start, evidence_end if evidence_end is not None else end,
            params, semantic_core_names))
        effect = _extract_origin_aware_effect(events, start, end, params)
        declared_aligned = True
        if declared:
            matched = [item for item in effect.positive
                       if str(item.get("predicate")) in declared]
            if matched:
                effect.positive = matched
            else:
                # LLM 的 effect_predicates 只是语义提案，不能覆盖真实
                # before/after 差分；命名不一致也不能删除已观察到的 Effect。
                declared_aligned = False
        # A non-canonical declaration cannot be trusted, but the intent can
        # still select one *observed* core predicate.  It never creates Effect.
        if not declared_aligned:
            preferred = _intent_predicates(str(raw.get("intent") or ""))
            matched = [item for item in effect.positive
                       if str(item.get("predicate")) in preferred]
            if matched:
                effect.positive = matched
        if not effect.positive:
            errors.append(f"phase_{ordinal}:no_observed_stable_effect")
            continue
        _refresh_effect_identity(effect)
        # The verified Effect and task contract own the public role names.
        # LLM names such as ``target_object``/``associated_object`` remain
        # useful proposal evidence, but must not fragment the executable I/O
        # contract when the task already exposes ``object`` and
        # ``associated_entity`` for the same grounded participants.
        params = _canonicalize_verified_roles(
            params, effect.positive, trace.provenance)
        params.update(_terminal_certificate_params(
            events[evidence_end if evidence_end is not None else end], params))
        # Remove trailing navigation/transport after the core transition.  The
        # retained range remains an actual contiguous occurrence in the trace.
        core_names = {str(item.get("predicate")) for item in effect.positive}
        trimmed_end = _last_core_effect_event(
            events, start, end, core_names, core_effects=effect.positive)
        if trimmed_end is not None and trimmed_end < end:
            end = trimmed_end
            effect = _extract_origin_aware_effect(events, start, end, params)
            effect.positive = [item for item in effect.positive
                               if str(item.get("predicate")) in core_names]
            # ``extract_effect`` assigns its family before the semantic filter.
            # A widened phase can also observe helper Effects, so recomputing
            # after trimming must refresh the identity before role pruning.
            _refresh_effect_identity(effect)
        params = _prune_phase_params(
            _canonical_phase_params(params, effect.primary_family),
            effect.positive, events[start:end + 1],
            core_event=events[trimmed_end if trimmed_end is not None else end])
        causal_window_start = max(0, previous_core_end + 1)
        causal_event_indices, slice_diagnostics = _minimal_causal_event_indices(
            events, causal_window_start, end, core_names, params,
            core_effects=effect.positive)
        if causal_event_indices:
            start, end = min(causal_event_indices), max(causal_event_indices)
        else:
            causal_event_indices = list(range(start, end + 1))
        core_index = int(slice_diagnostics.get("core_event_index", end))
        # Recompute I/O, Preconditions and Validator from the canonical role
        # contract, then retain only the already evidence-validated core Effect.
        canonical_effect = _extract_origin_aware_effect(
            events, core_index, core_index, params)
        canonical_effect.positive = [item for item in canonical_effect.positive
                                     if str(item.get("predicate")) in core_names]
        terminal_preconditions = _terminal_certificate_preconditions(
            events[core_index], params)
        if terminal_preconditions:
            # A terminal certificate is a stronger code-level contract than
            # an LLM's optional precondition list. Only evidence already true
            # immediately before the terminal boundary is admitted, so a
            # simultaneous final movement is not mislabeled as a precondition.
            canonical_effect.preconditions = terminal_preconditions
            canonical_effect.validator["pre_checks"] = sorted({
                str(item.get("predicate") or "")
                for item in terminal_preconditions if item.get("predicate")})
        elif "precondition_predicates" in raw:
            declared_preconditions = {
                _declared_predicate_name(item)
                for item in (raw.get("precondition_predicates") or [])
            }
            declared_preconditions.discard("")
            canonical_effect.preconditions = [
                item for item in canonical_effect.preconditions
                if str(item.get("predicate") or "") in declared_preconditions
            ]
            canonical_effect.validator["pre_checks"] = sorted({
                str(item.get("predicate") or "")
                for item in canonical_effect.preconditions
                if item.get("predicate")
            })
        canonical_effect.validator["post_checks"] = sorted(core_names)
        if canonical_effect.positive:
            _refresh_effect_identity(canonical_effect)
            effect = canonical_effect
        occupied |= indices
        previous_core_end = end
        proposed_intent = _safe_name(str(raw.get("intent") or "atomic"))
        # Logical identity is derived mechanically from the verified core
        # Effect.  The independently proposed phrase is retained as an alias;
        # no benchmark operation catalogue is used to name the node.
        intent = _safe_name(str(effect.suggested_name or proposed_intent))
        phases.append({
            "phase_id": str(raw.get("phase_id") or f"phase_{ordinal:03d}"),
            "name": intent,
            "proposed_intent": proposed_intent,
            "kind": "env",
            "actions": [dict(events[index]) for index in causal_event_indices],
            "before": events[start]["before"],
            "after": events[end]["after"],
            "contract_before": events[core_index]["before"],
            "params": params,
            "effect": effect.positive,
            "negative_effect": effect.negative,
            "preconditions": effect.preconditions,
            "summary": str(raw.get("rationale") or effect.summary)[:500],
            "event_start": start,
            "event_end": end,
            "proposal_event_start": raw_start,
            "proposal_event_end": raw_end,
            "entry_event_index": causal_window_start,
            "causal_event_indices": causal_event_indices,
            "removed_internal_event_indices": sorted(
                set(range(causal_window_start, raw_end + 1))
                - set(causal_event_indices)),
            "effect_producer_indices": [core_index],
            "terminal_effect_origin": any(
                _predicate_key(item) in {
                    _predicate_key(effect)
                    for effect in (events[core_index].get(
                        "terminal_verified_effects") or [])
                }
                for item in effect.positive),
            "event_slice_validated": True,
            "replay_safe": bool(slice_diagnostics.get("replay_safe")),
            "event_slice_diagnostics": slice_diagnostics,
            "extraction_method": "llm_proposal_code_validated",
            "validation": {"observed_effect": True, "roles_grounded": True,
                           "declared_effect_aligned": declared_aligned,
                           "state_evidence_precedence": True,
                           "identity_from_verified_effect": True,
                           "terminal_certificate_preconditions": bool(
                               terminal_preconditions),
                           "trailing_internal_events_removed": trimmed_end is not None
                           and trimmed_end < raw_end,
                           "internal_causal_slice_applied": len(causal_event_indices)
                           < len(indices)},
        })
    phases.sort(key=lambda item: (item["event_start"], item["event_end"]))
    return phases, errors


def causal_slice(phases: list[dict[str, Any]], target_effects: list[dict[str, Any]],
                 initial_state: dict[str, Any], *,
                 task_params: dict[str, Any] | None = None
                 ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Backward causal slice for a partially observable environment.

    A validated occurrence is selected when it produces a goal predicate or a
    precondition required by a later selected occurrence.  A precondition that
    is witnessed in the occurrence's real ``before`` snapshot but has no earlier
    capability producer is an exogenous environment condition, not a reason to
    reject the workflow.  This keeps exploration out of the capability graph.
    """
    task_params = dict(task_params or {})
    goals: list[dict[str, Any]] = []
    for item in target_effects:
        if not isinstance(item, dict) or not item.get("predicate"):
            continue
        # Requirements are a multiset, not a set. A cardinality-two placement
        # goal must retain two independently produced occurrences during the
        # backward slice.
        count = max(1, int(item.get("cardinality", 1) or 1))
        goals.extend(dict(item) for _ in range(count))
    diagnostics: dict[str, Any] = {
        "goal_effects": goals,
        "selected_phase_ids": [],
        "removed_phase_ids": [],
        "removed_phases": [],
        "dependencies": [],
        "external_preconditions": [],
        "unresolved_requirements": [],
    }
    if not goals:
        diagnostics["selected_phase_ids"] = [str(p.get("phase_id")) for p in phases]
        return phases, diagnostics

    # requirement = predicate + the parameter environment in which placeholders
    # must be resolved + the downstream phase that consumes it.
    needed: list[tuple[dict[str, Any], dict[str, Any], str]] = [
        (goal, task_params, "task_goal") for goal in goals]
    selected_indices: set[int] = set()
    initial_snapshot = StateSnapshot(initial_state)

    for index in range(len(phases) - 1, -1, -1):
        phase = phases[index]
        params = dict(phase.get("params") or {})
        produced = list(phase.get("effect") or [])
        matched_indices: list[int] = []
        used_effects: set[int] = set()
        for requirement_index, requirement in enumerate(needed):
            for effect_index, effect in enumerate(produced):
                if effect_index in used_effects:
                    continue
                if _predicates_compatible(
                        effect, params, requirement[0], requirement[1]):
                    matched_indices.append(requirement_index)
                    used_effects.add(effect_index)
                    break
        if not matched_indices:
            continue
        selected_indices.add(index)
        matched_set = set(matched_indices)
        needed = [requirement for requirement_index, requirement in enumerate(needed)
                  if requirement_index not in matched_set]

        for precondition in (phase.get("preconditions") or []):
            producer_index = _latest_earlier_producer(
                phases, index, precondition, params)
            if producer_index is not None:
                producer = phases[producer_index]
                requirement = (dict(precondition), params,
                               str(phase.get("phase_id") or index))
                if not any(_same_requirement(requirement, existing)
                           for existing in needed):
                    needed.append(requirement)
                diagnostics["dependencies"].append({
                    "source_phase_id": str(producer.get("phase_id") or producer_index),
                    "target_phase_id": str(phase.get("phase_id") or index),
                    "predicate": str(precondition.get("predicate") or ""),
                    "relation": "effect_satisfies_precondition",
                })
                continue

            bound = _bind_predicate(precondition, params)
            # A phase can be widened beyond the LLM-proposed span so the event
            # slicer can recover a required navigation/open action.  That wider
            # snapshot is useful for replay, but it is not the semantic entry
            # state of the atomic capability.  Preconditions that already hold
            # immediately before the core action are environment/observation
            # inputs, not unresolved requirements that must be produced by an
            # earlier Atomic node.
            before_snapshot = StateSnapshot(
                phase.get("contract_before") or phase.get("before") or {})
            if evaluate_predicate(before_snapshot, bound):
                source = ("initial_state" if evaluate_predicate(initial_snapshot, bound)
                          else "observed_before_phase")
                external = {
                    "phase_id": str(phase.get("phase_id") or index),
                    "predicate": bound,
                    "source": source,
                }
                diagnostics["external_preconditions"].append(external)
                phase.setdefault("validation", {}).setdefault(
                    "external_preconditions", []).append(external)
            else:
                requirement = (dict(precondition), params,
                               str(phase.get("phase_id") or index))
                if not any(_same_requirement(requirement, existing)
                           for existing in needed):
                    needed.append(requirement)

    selected = [phase for index, phase in enumerate(phases)
                if index in selected_indices]
    diagnostics["selected_phase_ids"] = [str(p.get("phase_id")) for p in selected]
    diagnostics["removed_phase_ids"] = [
        str(phase.get("phase_id")) for index, phase in enumerate(phases)
        if index not in selected_indices]
    diagnostics["removed_phases"] = [
        {"phase_id": str(phase.get("phase_id")),
         "reason": "not_causally_required_for_task_goal"}
        for index, phase in enumerate(phases) if index not in selected_indices]
    diagnostics["unresolved_requirements"] = [
        {"predicate": _bind_predicate(predicate, params), "consumer": consumer}
        for predicate, params, consumer in needed]
    return (selected if selected and not needed else []), diagnostics


def _phase_audit_view(phase: dict[str, Any]) -> dict[str, Any]:
    """Persist the evidence-bearing phase fields without duplicating full events."""
    return {
        "phase_id": phase.get("phase_id"), "name": phase.get("name"),
        "proposed_intent": phase.get("proposed_intent"),
        "event_start": phase.get("event_start"), "event_end": phase.get("event_end"),
        "causal_event_indices": list(phase.get("causal_event_indices") or []),
        "effect_producer_indices": list(phase.get("effect_producer_indices") or []),
        "retained_actions": [
            str(action.get("action") or action.get("name") or "")
            for action in (phase.get("actions") or [])
            if str(action.get("action") or action.get("name") or "")
        ],
        "removed_internal_event_indices": list(
            phase.get("removed_internal_event_indices") or []),
        "event_slice_validated": bool(phase.get("event_slice_validated")),
        "replay_safe": bool(phase.get("replay_safe")),
        "event_slice_diagnostics": dict(
            phase.get("event_slice_diagnostics") or {}),
        "params": dict(phase.get("params") or {}),
        "preconditions": list(phase.get("preconditions") or []),
        "effect": list(phase.get("effect") or []),
        "negative_effect": list(phase.get("negative_effect") or []),
        "summary": phase.get("summary"),
        "validation": dict(phase.get("validation") or {}),
    }


def _bind_predicate(predicate: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    bound = dict(predicate)
    args = bind_args(dict(predicate.get("args") or {}), params, params)
    for key, value in list(args.items()):
        if isinstance(value, str) and value.startswith("$task."):
            args[key] = params.get(value.split(".", 1)[1], value)
    bound["args"] = args
    return bound


def _predicates_compatible(left: dict[str, Any], left_params: dict[str, Any],
                           right: dict[str, Any], right_params: dict[str, Any]) -> bool:
    if str(left.get("predicate") or "") != str(right.get("predicate") or ""):
        return False
    left_args = (_bind_predicate(left, left_params).get("args") or {})
    right_args = (_bind_predicate(right, right_params).get("args") or {})
    common = set(left_args) & set(right_args)
    if not common:
        return True
    return all(_values_compatible(left_args[key], right_args[key]) for key in common)


def _values_compatible(left: Any, right: Any) -> bool:
    left_value, right_value = normalize_value(left), normalize_value(right)
    if not left_value or not right_value:
        return True
    if left_value.startswith("$") or right_value.startswith("$"):
        return left_value == right_value
    if left_value == right_value:
        return True
    # A class-valued slot may match a concrete ALFWorld instance, but two
    # distinct concrete instances must never be merged.
    left_instance = bool(re.search(r"_\d+$", left_value))
    right_instance = bool(re.search(r"_\d+$", right_value))
    if left_instance and right_instance:
        return False
    strip_instance = lambda value: re.sub(r"_\d+$", "", value)
    return strip_instance(left_value) == strip_instance(right_value)


def _latest_earlier_producer(phases: list[dict[str, Any]], before_index: int,
                             precondition: dict[str, Any],
                             consumer_params: dict[str, Any]) -> int | None:
    for index in range(before_index - 1, -1, -1):
        producer = phases[index]
        producer_params = dict(producer.get("params") or {})
        if any(_predicates_compatible(effect, producer_params,
                                      precondition, consumer_params)
               for effect in (producer.get("effect") or [])):
            return index
    return None


def _same_requirement(
        left: tuple[dict[str, Any], dict[str, Any], str],
        right: tuple[dict[str, Any], dict[str, Any], str]) -> bool:
    return _predicates_compatible(left[0], left[1], right[0], right[1])


def validate_graph_proposal(proposal: dict[str, Any],
                            occurrences: list[dict[str, Any]]) -> dict[str, Any]:
    """Allow only supplied occurrences/refs and forward, acyclic dependencies."""
    by_id = {str(item["phase_id"]): item for item in occurrences}
    ordered = [str(item) for item in (proposal.get("ordered_phase_ids") or [])]
    if set(ordered) != set(by_id) or len(ordered) != len(by_id):
        ordered = [str(item["phase_id"]) for item in occurrences]
    positions = {phase_id: index for index, phase_id in enumerate(ordered)}
    dependencies: list[dict[str, Any]] = []
    for raw in proposal.get("implicit_dependencies") or []:
        if not isinstance(raw, dict):
            continue
        source = str(raw.get("source_phase_id") or "")
        target = str(raw.get("target_phase_id") or "")
        if source in by_id and target in by_id and positions[source] < positions[target]:
            dependencies.append({
                "source_phase_id": source, "target_phase_id": target,
                "relation": "requires_skill", "reason": str(raw.get("reason") or "")[:300],
            })
    allowed_skills = {str(item.get("skill_ref")) for item in occurrences}
    allowed_tools = {str(ref) for item in occurrences for ref in (item.get("tool_refs") or [])}
    tool_plan = []
    for raw in proposal.get("tool_plan") or []:
        if not isinstance(raw, dict) or str(raw.get("phase_id")) not in by_id:
            continue
        skill_ref = str(raw.get("skill_ref") or "")
        refs = [str(ref) for ref in (raw.get("tool_refs") or []) if str(ref) in allowed_tools]
        if skill_ref in allowed_skills:
            tool_plan.append({"phase_id": str(raw["phase_id"]),
                              "skill_ref": skill_ref, "tool_refs": refs})
    return {"ordered_phase_ids": ordered,
            "summary": str(proposal.get("summary") or "")[:500],
            "implicit_dependencies": dependencies, "tool_plan": tool_plan,
            "validated": True}


def _parse_json_object(text: str) -> dict[str, Any]:
    candidate = str(text or "").strip()
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.I | re.S)
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end < start:
        raise ValueError("extractor returned no JSON object")
    payload = json.loads(candidate[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("extractor JSON must be an object")
    return payload


def _compact_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_index": event["event_index"], "action": event["action"],
        "params": event["params"], "accepted": event["accepted"],
        "mode": event["mode"], "positive_effects": event["positive_effects"],
        "terminal_verified_effects": event.get("terminal_verified_effects") or [],
        "observed_effects": event.get("observed_effects") or [],
        "negative_effects": event["negative_effects"],
        "before_facts": list(event["before"].get("facts") or []),
        "after_facts": list(event["after"].get("facts") or []),
        "before_inventory": list(event["before"].get("inventory") or []),
        "after_inventory": list(event["after"].get("inventory") or []),
    }


def _event_params(events: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for event in events:
        for key, value in (event.get("params") or {}).items():
            if value not in (None, ""):
                # Later actions are closer to the phase's core Effect and take
                # precedence over an earlier failed attempt on another entity.
                result[str(key)] = value
    return result


def _known_values(trace: TraceRecord, events: list[dict[str, Any]]) -> set[str]:
    values = {normalize_value(value) for value in (trace.provenance.get("params") or {}).values()}
    for event in events:
        values |= {normalize_value(value) for value in (event.get("params") or {}).values()}
        for state_key in ("before", "after"):
            values |= {normalize_value(value) for value in (event[state_key].get("inventory") or [])}
            for fact in event[state_key].get("facts") or []:
                predicate = _fact_to_predicate(str(fact))
                if isinstance(predicate, dict):
                    values |= {normalize_value(value)
                               for value in (predicate.get("args") or {}).values()}
    return {value for value in values if value}


def _infer_execution_location_roles(
        events: list[dict[str, Any]], start: int, end: int,
        params: dict[str, Any], core_names: set[str]) -> dict[str, Any]:
    """Infer an external co-location input from the real core-event state.

    This is predicate-driven rather than verb- or benchmark-driven.  If the
    entity changed by the core event is known to be at a location and the
    agent is at that same location immediately before the event, replaying the
    occurrence requires that location.  The value becomes ``<role>_location``
    unless an existing input already binds it.
    """
    core_index = _last_core_effect_event(events, start, end, core_names)
    if core_index is None:
        return {}
    event = events[core_index]
    before_predicates = _state_predicates(event.get("before") or {})
    agent_locations = {
        normalize_value((predicate.get("args") or {}).get("arg0")
                        or (predicate.get("args") or {}).get("location"))
        for predicate in before_predicates
        if str(predicate.get("predicate") or "") in {"agent_at", "agent.at"}
    }
    agent_locations.discard("")
    if not agent_locations:
        return {}

    effect_entities = {
        normalize_value(value)
        for effect in _event_capability_effects(event)
        if str(effect.get("predicate") or "") in core_names
        for value in (effect.get("args") or {}).values()
    }
    inferred: dict[str, Any] = {}
    normalized_params = {str(role): normalize_value(value)
                         for role, value in params.items() if value not in (None, "")}
    occupied_locations = set(normalized_params.values()) & agent_locations
    if occupied_locations:
        return inferred

    for predicate in before_predicates:
        if str(predicate.get("predicate") or "") != "object.at_location":
            continue
        args = predicate.get("args") or {}
        entity = normalize_value(args.get("object"))
        location = normalize_value(args.get("location"))
        if not entity or entity not in effect_entities or location not in agent_locations:
            continue
        matching_roles = [role for role, value in normalized_params.items()
                          if value == entity or (
                              value and not re.search(r"_\d+$", value)
                              and re.sub(r"_\d+$", "", entity) == value)]
        if not matching_roles:
            continue
        role = sorted(matching_roles)[0]
        location_role = f"{role}_location"
        if location_role not in params:
            inferred[location_role] = location
    return inferred


def _validated_roles(roles: dict[str, Any], known_values: set[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in roles.items():
        normalized = normalize_value(value)
        family = re.sub(r"_\d+$", "", normalized)
        concrete = bool(re.search(r"_\d+$", normalized))
        grounded = (normalized in known_values if concrete else any(
            family == re.sub(r"_\d+$", "", known) for known in known_values))
        if normalized and grounded:
            result[_safe_name(str(key))] = value
    return result


def _refine_roles_with_phase_evidence(roles: dict[str, Any],
                                      observed_params: dict[str, Any]) -> dict[str, Any]:
    """Let phase-local executed parameters refine an LLM semantic role.

    Whole-trace grounding only proves that an entity exists somewhere.  It does
    not prove that mug_2 was the object manipulated in this phase.  Same-role
    action parameters are authoritative; a generic semantic value may be
    specialized to a unique concrete phase value across role names.
    """
    observed_by_role: dict[str, Any] = {}
    for key, value in observed_params.items():
        observed_by_role.setdefault(_canonical_role_name(str(key)), value)
    refined: dict[str, Any] = {}
    for raw_role, proposed in roles.items():
        role = _canonical_role_name(str(raw_role))
        same_role = observed_by_role.get(role)
        if same_role not in (None, ""):
            refined[role] = same_role
            continue
        # Never specialize one semantic role using a same-family value from a
        # different role (for example a final destination from an acquisition
        # source merely because both are cabinets).
        refined[role] = proposed
    return refined


def _canonical_role_name(role: str) -> str:
    aliases = {
        "source_location": "object_location",
        "source_receptacle": "object_location",
        "destination": "target_location",
        "destination_location": "target_location",
        "target_receptacle": "target_location",
    }
    return aliases.get(str(role), str(role))


def _declared_predicate_name(value: Any) -> str:
    """Normalize LLM declarations such as ``agent_holds(x)`` to IR names."""
    raw = str(value or "").strip().split("(", 1)[0]
    return _predicate_name_from_fact(raw)


def _intent_predicates(intent: str) -> set[str]:
    """No operation-name lexicon is used to infer state semantics.

    Kept as an internal compatibility hook for callers, but Extractor proposals
    must explicitly cite predicates observed in the event deltas.
    """
    return set()


def _last_core_effect_event(events: list[dict[str, Any]], start: int, end: int,
                            core_names: set[str], *,
                            core_effects: list[dict[str, Any]] | None = None
                            ) -> int | None:
    matches = [index for index in range(start, end + 1)
               if any(str(item.get("predicate")) in core_names
                      and (not core_effects or any(
                          _predicate_key(item) == _predicate_key(anchor)
                          for anchor in core_effects))
                       for item in _event_capability_effects(events[index]))]
    return max(matches) if matches else None


def _minimal_causal_event_indices(events: list[dict[str, Any]], start: int, end: int,
                                  core_names: set[str],
                                  params: dict[str, Any], *,
                                  core_effects: list[dict[str, Any]] | None = None
                                  ) -> tuple[list[int], dict[str, Any]]:
    """Select an event-level causal subtrace inside one LLM phase.

    The LLM proposes a semantic boundary; code starts from the event that truly
    produced the core Effect and recursively retains only earlier producers of
    operational preconditions.  Exploration, repeated examine/open/close loops
    and failed recovery detours therefore remain in the raw trace but not in the
    Atomic occurrence or mined Tool body.
    """
    core_index = _last_core_effect_event(
        events, start, end, core_names, core_effects=core_effects)
    if core_index is None:
        return [], {"replay_safe": False, "reason": "core_effect_producer_missing",
                    "unresolved_requirements": []}
    initial_keys = {_predicate_key(predicate) for predicate in
                    _state_predicates(events[start].get("before") or {})}
    selected: set[int] = {core_index}
    pending = [core_index]
    unresolved: list[dict[str, Any]] = []
    while pending:
        consumer_index = pending.pop()
        requirements = _event_operational_requirements(
            events[consumer_index], params)
        for requirement in requirements:
            key = _predicate_key(requirement)
            if key in initial_keys:
                continue
            producer_index = _latest_event_producer(
                events, start, consumer_index, requirement)
            if producer_index is not None and producer_index not in selected:
                selected.add(producer_index)
                pending.append(producer_index)
            elif producer_index is None:
                unresolved.append({"consumer_event_index": consumer_index,
                                   "predicate": requirement})
    retained = sorted(selected)
    # Raw reaching definitions are conservative: a later navigation can look
    # like the producer of agent_at only because omitted exploration moved the
    # agent away in the original trace.  Re-evaluate the sparse subsequence
    # from the true window entry and greedily remove every non-core event whose
    # absence still preserves all operational requirements and the core Effect.
    for candidate_index in reversed([index for index in retained
                                     if index != core_index]):
        trial = [index for index in retained if index != candidate_index]
        if _forward_validate_event_slice(
                events, start, trial, core_effects or [
                    effect for effect in
                    _event_capability_effects(events[core_index])
                    if str(effect.get("predicate") or "") in core_names],
                params):
            retained = trial
    selected = set(retained)
    forward_valid = _forward_validate_event_slice(
        events, start, retained, core_effects or [
            effect for effect in
            _event_capability_effects(events[core_index])
            if str(effect.get("predicate") or "") in core_names],
        params)
    replay_safe = (bool(retained) and core_index in selected
                   and all(events[index].get("accepted", True) for index in retained)
                   and not unresolved and forward_valid)
    return retained, {
        "core_event_index": core_index,
        "effect_producer_indices": [core_index],
        "unresolved_requirements": unresolved,
        "retained_event_indices": retained,
        "replay_safe": replay_safe,
        "counterfactual_forward_validated": forward_valid,
        "one_minimal_by_reaching_definition": True,
    }


def slice_event_occurrence(events: list[dict[str, Any]], *,
                           window_start: int, window_end: int,
                           core_effects: list[dict[str, Any]],
                           params: dict[str, Any]) -> dict[str, Any]:
    """Public evidence gate shared by extraction, Tool mining and repair.

    ``window_end`` is inclusive.  The result contains original event objects;
    callers may parameterize their actions only after this code-validated
    sparse slice succeeds.
    """
    if (not events or window_start < 0 or window_end < window_start
            or window_end >= len(events)):
        return {
            "retained_event_indices": [], "events": [], "replay_safe": False,
            "reason": "invalid_event_window", "effect_producer_indices": [],
        }
    bound_effects = [_bind_predicate(dict(effect), params)
                     for effect in core_effects
                     if isinstance(effect, dict) and effect.get("predicate")]
    core_names = {str(effect.get("predicate") or "") for effect in bound_effects}
    core_names.discard("")
    if not core_names:
        return {
            "retained_event_indices": [], "events": [], "replay_safe": False,
            "reason": "core_effect_missing", "effect_producer_indices": [],
        }
    retained, diagnostics = _minimal_causal_event_indices(
        events, window_start, window_end, core_names, params,
        core_effects=bound_effects)
    return {
        **diagnostics,
        "events": [dict(events[index]) for index in retained],
        "retained_event_indices": retained,
    }


def _forward_validate_event_slice(events: list[dict[str, Any]], entry_index: int,
                                  retained: list[int],
                                  core_effects: list[dict[str, Any]],
                                  params: dict[str, Any]) -> bool:
    """Validate a sparse occurrence using only persisted state transitions.

    This is deliberately not an environment simulator.  It applies the real
    add/delete predicates recorded for retained accepted events and checks each
    event's grounded operational reads.  Therefore it can prove that removing
    a loop is safe without inventing an action or an unobserved transition.
    """
    if not retained or not (0 <= entry_index < len(events)):
        return False
    state_keys = {_predicate_key(predicate) for predicate in
                  _state_predicates(events[entry_index].get("before") or {})}
    for index in retained:
        event = events[index]
        if not bool(event.get("accepted", True)):
            return False
        requirements = _event_operational_requirements(event, params)
        if any(_predicate_key(requirement) not in state_keys
               for requirement in requirements):
            return False
        for effect in event.get("negative_effects") or []:
            state_keys.discard(_predicate_key(effect))
        for effect in _event_capability_effects(event):
            state_keys.add(_predicate_key(effect))
    return all(_predicate_key(effect) in state_keys for effect in core_effects)


def _state_predicates(state: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for fact in state.get("facts") or []:
        predicate = _fact_to_predicate(str(fact))
        if isinstance(predicate, dict):
            result.append(predicate)
    return result


def _event_operational_requirements(event: dict[str, Any],
                                    params: dict[str, Any]) -> list[dict[str, Any]]:
    """Infer event reads from grounded state relations, without a verb table."""
    positives = [item for item in _event_capability_effects(event)
                 if isinstance(item, dict)]
    event_params = {**dict(params or {}), **dict(event.get("params") or {})}

    # A parameter-free transition whose only active change is agent location
    # is a producer/setup event. First observations may add object facts, but
    # those observations are not reads required to perform the movement.
    active = [str(item.get("predicate") or "") for item in positives
              if str(item.get("predicate") or "") != "object.exists"]
    if (not event.get("params")
            and any(name in {"agent_at", "agent.at"} for name in active)
            and all(name in {"agent_at", "agent.at", "object.at_location"}
                    for name in active)):
        return []

    before = _state_predicates(event.get("before") or {})
    participants = {normalize_value(value) for value in event_params.values()
                    if value not in (None, "")}
    participants.discard("")
    related_locations: set[str] = set()
    for predicate in before:
        if str(predicate.get("predicate") or "") != "object.at_location":
            continue
        args = predicate.get("args") or {}
        if _matches_any(args.get("object"), participants):
            related_locations.add(normalize_value(args.get("location")))
    related_locations.discard("")

    requirements: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    for predicate in before:
        args = predicate.get("args") or {}
        values = {normalize_value(value) for value in args.values()}
        if "object" in args:
            # An object-bearing relation is selected through its object, never
            # merely because an unrelated object shares a location participant.
            mentions_participant = _matches_any(args.get("object"), participants)
        else:
            mentions_participant = bool(values & participants)
        name = str(predicate.get("predicate") or "")
        if name in {"agent_at", "agent.at", "container.open"}:
            mentions_participant = mentions_participant or bool(
                values & related_locations)
        if not mentions_participant:
            continue
        key = _predicate_key(predicate)
        if key not in seen:
            seen.add(key)
            requirements.append(predicate)
    return requirements


def _matches_any(value: Any, candidates: set[str]) -> bool:
    normalized = normalize_value(value)
    if normalized in candidates:
        return True
    family = re.sub(r"_\d+$", "", normalized)
    return any(candidate and not re.search(r"_\d+$", candidate)
               and family == re.sub(r"_\d+$", "", candidate)
               for candidate in candidates)


def _predicate_mentions_phase_entity(predicate: dict[str, Any],
                                     params: dict[str, Any]) -> bool:
    args = predicate.get("args") or {}
    object_value = normalize_value(params.get("object", ""))
    locations = {normalize_value(value) for key, value in params.items()
                 if key != "object" and value not in (None, "")}
    locations.discard("")

    def matches(value: Any, target: str) -> bool:
        normalized = normalize_value(value)
        if not normalized or not target:
            return False
        if re.search(r"_\d+$", target):
            return normalized == target
        return re.sub(r"_\d+$", "", normalized) == re.sub(r"_\d+$", "", target)

    # Object-bearing predicates are about that exact semantic participant.  A
    # matching location must not turn ``object_at(mug_2, cabinet_1)`` into an
    # operational dependency of the occurrence acting on ``mug_1``.  Only
    # predicates without an object argument (for example ``container.open`` or
    # ``agent_at``) may be selected through a location role alone.
    if "object" in args:
        return matches(args.get("object"), object_value)
    for key in ("location", "container", "arg0"):
        if key in args and any(matches(args.get(key), value) for value in locations):
            return True
    return False


def _latest_event_producer(events: list[dict[str, Any]], start: int, end: int,
                           requirement: dict[str, Any]) -> int | None:
    key = _predicate_key(requirement)
    for index in range(end - 1, start - 1, -1):
        if not events[index].get("accepted", True):
            continue
        if any(_predicate_key(effect) == key
               for effect in _event_capability_effects(events[index])):
            return index
    return None


def _predicate_key(predicate: dict[str, Any]) -> tuple[str, tuple[tuple[str, str], ...]]:
    return (str(predicate.get("predicate") or ""),
            tuple(sorted((str(key), normalize_value(value))
                         for key, value in (predicate.get("args") or {}).items())))


def _event_capability_effects(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Action deltas plus separately certified terminal relations.

    Certificate audit metadata is stripped before predicate comparison and
    parameterization.  ``observed_effects`` remain excluded.
    """
    effects: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    for raw in [*(event.get("positive_effects") or []),
                *(event.get("terminal_verified_effects") or [])]:
        if not isinstance(raw, dict) or not raw.get("predicate"):
            continue
        predicate = {"predicate": str(raw.get("predicate")),
                     "args": dict(raw.get("args") or {})}
        key = _predicate_key(predicate)
        if key not in seen:
            seen.add(key)
            effects.append(predicate)
    return effects


def _canonicalize_verified_roles(
        params: dict[str, Any], effects: list[dict[str, Any]],
        provenance: dict[str, Any]) -> dict[str, Any]:
    """Align proposal aliases to task semantic roles using grounded equality.

    This is benchmark-agnostic: role names and values come exclusively from
    the persisted task contract and verified Effect arguments. No operation,
    object, device, or dataset vocabulary is consulted.
    """
    canonical = dict(params or {})
    task_roles = {
        str(role): value
        for source in (dict(provenance.get("semantic_params") or {}),
                       dict(provenance.get("params") or {}))
        for role, value in source.items() if value not in (None, "")
    }
    effect_values = [
        (str(argument), value)
        for effect in effects
        for argument, value in (effect.get("args") or {}).items()
        if value not in (None, "")
    ]

    def matches(grounded: Any, declared: Any) -> bool:
        left, right = normalize_value(grounded), normalize_value(declared)
        if not left or not right:
            return False
        if left == right:
            return True
        # A task-level family may bind a concrete occurrence, never vice versa.
        return (not re.search(r"_\d+$", right)
                and re.sub(r"_\d+$", "", left)
                == re.sub(r"_\d+$", "", right))

    preferred_by_value: dict[str, str] = {}
    for argument, value in effect_values:
        candidates = [role for role, declared in task_roles.items()
                      if matches(value, declared)]
        lexical = [role for role in candidates
                   if argument in role or role in argument]
        aliases = [role for role, current in canonical.items()
                   if matches(value, current)]
        if argument in candidates:
            preferred = argument
        elif argument in canonical and matches(value, canonical[argument]):
            preferred = argument
        elif lexical:
            preferred = sorted(lexical, key=lambda role: (len(role), role))[0]
        elif len(aliases) == 1:
            preferred = aliases[0]
        else:
            # Predicate argument names provide a stable benchmark-independent
            # fallback. Composite occurrence mapping may later bind this role
            # to an equivalent task role through grounded co-reference.
            preferred = argument
        surface_value = next(
            (canonical[role] for role in aliases if role == preferred),
            next((canonical[role] for role in aliases), value))
        canonical[preferred] = surface_value
        preferred_by_value[normalize_value(value)] = preferred

    task_role_names = set(task_roles)
    for role, value in list(canonical.items()):
        preferred = preferred_by_value.get(normalize_value(value))
        if (preferred and role != preferred and role not in task_role_names):
            canonical.pop(role, None)
    return canonical


def _terminal_certificate_params(event: dict[str, Any],
                                 params: dict[str, Any]) -> dict[str, Any]:
    """Recover missing replay roles from pre-terminal certificate evidence."""
    before = StateSnapshot(event.get("before") or {})
    enriched: dict[str, Any] = {}
    known_values = {normalize_value(value) for value in params.values()}
    for terminal in event.get("terminal_verified_effects") or []:
        certificate = dict(terminal.get("certificate") or {})
        if bool(certificate.get("standalone_action_effect", True)):
            continue
        for fact in certificate.get("evidence_facts") or []:
            predicate = _fact_to_predicate(str(fact))
            if predicate is None or not evaluate_predicate(before, predicate):
                continue
            for role, value in (predicate.get("args") or {}).items():
                normalized = normalize_value(value)
                if normalized and normalized not in known_values:
                    enriched.setdefault(str(role), value)
                    known_values.add(normalized)
    return enriched


def _terminal_certificate_preconditions(
        event: dict[str, Any], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Compile a terminal certificate's witnessed reads into Preconditions."""
    before = StateSnapshot(event.get("before") or {})
    grounded: list[dict[str, Any]] = []
    for terminal in event.get("terminal_verified_effects") or []:
        certificate = dict(terminal.get("certificate") or {})
        if bool(certificate.get("standalone_action_effect", True)):
            continue
        for fact in certificate.get("evidence_facts") or []:
            predicate = _fact_to_predicate(str(fact))
            # Facts established only by the terminal event (for example the
            # destination agent location) are post-state evidence, not reads.
            if (predicate is not None
                    and evaluate_predicate(before, predicate)):
                grounded.append(predicate)
    parameterized = parameterize_predicates(grounded, params)
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    for predicate in parameterized:
        # Every reusable certificate precondition must be fully bindable.
        values = list((predicate.get("args") or {}).values())
        if any(isinstance(value, str) and not value.startswith("$inputs.")
               for value in values):
            continue
        key = _predicate_key(predicate)
        if key not in seen:
            seen.add(key)
            unique.append(predicate)
    return unique


def _canonical_phase_params(params: dict[str, Any], family: str) -> dict[str, Any]:
    """Preserve trace-grounded semantic roles without a capability whitelist."""
    return {_safe_name(str(key)): value for key, value in params.items()
            if value not in (None, "")}


def _prune_phase_params(params: dict[str, Any], effects: list[dict[str, Any]],
                        events: list[dict[str, Any]], *,
                        core_event: dict[str, Any] | None = None) -> dict[str, Any]:
    """Keep only roles evidenced by this proposed occurrence.

    This prevents whole-task bindings from leaking into an Atomic interface
    without relying on a family-specific role whitelist.
    """
    evidence_values = {
        normalize_value(value)
        for effect in effects for value in (effect.get("args") or {}).values()
    }
    action_text = "\n".join(str(event.get("action") or event.get("name") or "")
                            for event in events).lower().replace("_", " ")
    for event in events:
        evidence_values |= {normalize_value(value)
                            for value in (event.get("params") or {}).values()}
    core_roles = {
        _canonical_role_name(str(role))
        for role in dict((core_event or {}).get("params") or {})
    }
    kept: dict[str, Any] = {}
    for role, value in params.items():
        normalized = normalize_value(value)
        surface = normalized.replace("_", " ")
        if (normalized in evidence_values
                or (surface and re.search(
                    rf"(?<![a-z0-9]){re.escape(surface)}(?![a-z0-9])", action_text))):
            kept[role] = value
    by_value: dict[str, list[str]] = {}
    for role, value in kept.items():
        by_value.setdefault(normalize_value(value), []).append(role)
    for roles in by_value.values():
        # When multiple semantic labels bind the same concrete participant,
        # prefer the role named by the verified Effect argument.  This removes
        # aliases such as ``required_position=container`` without a
        # capability- or benchmark-specific role whitelist.
        effect_argument_roles = {
            _canonical_role_name(str(argument_role))
            for effect in effects
            for argument_role, argument_value in (effect.get("args") or {}).items()
            if normalize_value(argument_value)
            == normalize_value(kept.get(roles[0]))
        }
        direct_matches = [role for role in roles if role in effect_argument_roles]
        if direct_matches:
            preferred = sorted(direct_matches)[0]
            for role in roles:
                if role != preferred:
                    kept.pop(role, None)
            continue
        core_matches = [role for role in roles if role in core_roles]
        if not core_matches:
            continue
        for role in roles:
            if role not in core_matches:
                kept.pop(role, None)
    return kept


def _refresh_effect_identity(effect: Any) -> None:
    """Refresh name/summary after semantic filtering removed helper Effects."""
    families = sorted({_family_of(item) for item in effect.positive if _family_of(item)})
    effect.primary_family = families[0] if families else ""
    if effect.primary_family:
        predicate = str((effect.positive[0] or {}).get("predicate") or "")
        name, summary = _FACT_FAMILY_NAMES.get(
            effect.primary_family,
            (effect.primary_family, f"Verified transition: {predicate}"))
        effect.suggested_name = name
        effect.summary = summary


def _predicate_name_from_fact(fact: Any) -> str:
    raw = str(fact).split("(", 1)[0]
    aliases = {
        "agent_holds": "agent.holds", "object_at": "object.at_location",
        "object_exists": "object.exists", "object_heated": "object.heated",
        "object_cleaned": "object.cleaned", "object_cooled": "object.cooled",
        "container_open": "container.open", "object_lit": "object.lit",
        "object_toggled": "object.toggled",
    }
    return aliases.get(raw, raw if "." in raw else raw.replace("_", "."))


def _safe_name(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_") or "atomic"
