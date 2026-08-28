"""Failure-driven, Git-like repair branches with strict replay gates.

The main registries are copied into a branch-local overlay.  A rescue-derived
Tool and its Implementation are changed only in that overlay, replayed on the
source task without LLM fallback, graph-validated, and merged by switching the
main recommended pointers only after every gate passes.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from ..core.refs import SkillRef, ToolRef, bump_version
from ..core.predicates import _fact_to_predicate, bind_args
from ..core.skill_ir import (
    AbstractAtomicSkill,
    CompositeSkill,
    ImplementationAtom,
    ToolBinding,
)
from ..core.status import EdgeType, ExecutionMode, SkillNodeKind, SkillStatus, ToolLifecycle
from ..core.tool_ir import ToolAsset
from ..core.trace_ir import TraceRecord
from ..graph.registry import SkillGraphRegistry
from ..graph.validator import validate_graph
from ..persistence_guard import validate_long_term_asset
from ..tools.admission_adapter import AdmissionEngine
from ..tools.registry import ToolRegistry
from ..atomicizer.effect_extractor import (
    is_fully_parameterized_predicate,
    parameterize_predicates,
)
from ..atomicizer.semantic_extractor import (
    build_structured_events,
    slice_event_occurrence,
)


class FailureBranchManager:
    """Turn failed attempts into isolated repair branches.

    A failed Direct attempt followed by a successful Seeded/Dynamic attempt has
    positive causal evidence: the cumulative action suffix reached the same
    Atomic Effect.  This is the only branch type automatically merged today.
    Other failed attempts still receive independent branch manifests, but stay
    ``awaiting_success_evidence`` rather than mutating main speculatively.
    """

    def __init__(self, data_dir: str | Path, registry: SkillGraphRegistry,
                 tool_registry: ToolRegistry, adapter, config, llm=None) -> None:
        self.data_dir = Path(data_dir)
        self.root = self.data_dir / "evolution" / "branches"
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry = registry
        self.tool_registry = tool_registry
        self.adapter = adapter
        self.config = config
        self.llm = llm

    def process(self, trace: TraceRecord, task) -> list[dict[str, Any]]:
        if self.config.freeze_skills:
            return []
        # A plan that could not bind/compile or an attempt that never started
        # has no executable repair evidence.  Creating a branch from it would
        # turn planner/budget failures into fake Tool/Skill mutations.
        if trace.failure_stage in {"planning", "budget"}:
            return []
        events: list[dict[str, Any]] = []
        for node_index, node in enumerate(trace.realized_atomic_nodes):
            attempts = list(node.get("attempts") or [])
            for attempt_index, attempt in enumerate(attempts):
                if bool(attempt.get("passed")):
                    continue
                incident = self._incident(trace, node, node_index,
                                          attempt, attempt_index, attempts)
                if not incident:
                    continue
                if incident["mode"] == ExecutionMode.DIRECT.value:
                    event = self._process_direct_incident(trace, task, node, incident)
                elif (incident["mode"] == ExecutionMode.SEEDED.value
                      and incident.get("rescue")):
                    event = self._process_atomic_guideline_incident(
                        trace, task, node, incident)
                else:
                    event = self._record_unverified_branch(incident)
                events.append(event)
        events.extend(self._process_code_attempts(trace, task))
        composite_validation = dict(trace.validation_layers.get("composite") or {})
        if trace.selected_composite and not composite_validation.get("passed", True):
            incident = {
                "branch_id": _safe_id(f"{trace.trace_id}-composite-revision"),
                "trace_id": trace.trace_id, "task_id": trace.task_id,
                "task_type": trace.task_type, "benchmark": trace.benchmark,
                "node_ref": trace.selected_composite, "mode": "composite",
                "failure_type": "composite_validation_error",
                "validation": composite_validation,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            events.append(self._record_unverified_branch(
                incident, "graph_patch_requires_successful_order_evidence"))
        return events

    def _process_atomic_guideline_incident(self, trace: TraceRecord, task,
                                           node: dict[str, Any],
                                           incident: dict[str, Any]) -> dict[str, Any]:
        if self.llm is None or not hasattr(self.adapter, "run_env_episode"):
            return self._record_unverified_branch(incident,
                                                  "strict_seeded_replay_unavailable")
        try:
            source_ref = SkillRef.parse(str(incident.get("node_ref") or ""))
        except ValueError:
            return self._record_unverified_branch(incident, "invalid_skill_ref")
        original = self.registry.get(source_ref)
        if not isinstance(original, AbstractAtomicSkill):
            return self._record_unverified_branch(incident, "atomic_missing")
        base_main_digest = _bank_digest(self.data_dir)
        branch_dir, branch_registry, branch_tools = self._branch_snapshot(incident)
        # The failed occurrence is versioned evidence.  A newer recommended
        # contract must not silently become the repair parent; latest is used
        # only to allocate a collision-free child version.
        parent = original
        latest = branch_registry.get_latest(original.ref.logical_id) or original
        payload = original.to_dict()
        payload["version"] = bump_version(latest.ref.version, "patch")
        payload["status"] = SkillStatus.ACTIVE.value
        rescue = dict(incident.get("rescue") or {})
        start = int(rescue.get("action_start") or 0)
        end = int(rescue.get("action_end") or start)
        preparation = _prepare_repair_evidence(
            trace, start, end, list(parent.effects),
            dict(node.get("params") or {}))
        if not bool(preparation.get("replay_safe")):
            return self._record_unverified_branch(
                _repair_preparation_incident(incident, preparation),
                str(preparation.get("reason") or "unsafe_repair_event_slice"))
        event_slice = dict(preparation["event_slice"])
        repair_bindings = dict(preparation["repair_bindings"])
        rescue_actions = list(preparation["steps"])
        if len(rescue_actions) > 12:
            return self._record_unverified_branch(
                incident, "rescue_not_atomic_after_denoising")
        if validate_long_term_asset(
                {"repair_action_template": rescue_actions},
                asset_kind="atomic_repair_template"):
            return self._record_unverified_branch(
                incident, "repair_template_contains_private_or_instance_literal")
        guideline = dict(payload.get("guideline") or {})
        rules = list(guideline.get("rules") or [])
        guideline_steps = [_guideline_placeholders(step) for step in rescue_actions]
        repair_rule = ("仅当常规执行失败且核心 Effect 尚未满足时，"
                       "使用经原任务严格重放验证的参数化补救模板："
                       + " -> ".join(guideline_steps))
        duplicate_rule = repair_rule in parent.guideline_rules()
        if not duplicate_rule:
            # A later failure trace still names the exact version that failed.
            # Reuse only a previously verified descendant of that *same*
            # version; never substitute an unrelated recommended/latest node.
            # This preserves exact causal lineage while allowing repeated
            # evidence to accumulate without creating version churn.
            linked_repairs = [
                item for item in self.registry.list_all_versions(
                    SkillNodeKind.ABSTRACT_ATOMIC)
                if (isinstance(item, AbstractAtomicSkill)
                    and item.ref.logical_id == original.ref.logical_id
                    and str((item.metadata or {}).get("repair_parent_ref") or "")
                    == str(original.ref)
                    and repair_rule in item.guideline_rules())
            ]
            if linked_repairs:
                parent = max(
                    linked_repairs,
                    key=lambda item: int(
                        (item.metadata or {}).get("repair_replay_count") or 0),
                )
                duplicate_rule = True
        if not duplicate_rule and repair_rule not in rules:
            rules.append(repair_rule)
        guideline["rules"] = rules
        payload["guideline"] = guideline
        repair_evidence_ref = self.registry.evidence_store.put(
            "atomic_repair",
            {"incident": incident, "event_slice": event_slice,
             "repair_bindings": repair_bindings,
             "rescue_actions": rescue_actions,
             "repair_guard": _repair_guard(
                 dict(rescue.get("before") or {}), repair_bindings)},
            trace_id=trace.trace_id,
            event_start=start,
            event_end=end,
        )
        metadata = dict(payload.get("metadata") or {})
        metadata["repair_branch_ref"] = _branch_ref(incident["branch_id"])
        metadata["repair_parent_ref"] = str(parent.ref)
        metadata["repair_source_trace_id"] = trace.trace_id
        metadata["repair_generalized"] = True
        metadata["repair_action_template"] = rescue_actions
        metadata["repair_replay_trace_ids"] = [trace.trace_id]
        metadata["repair_replay_count"] = 1
        metadata["repair_parameter_roles"] = sorted(
            str(key) for key in (node.get("params") or {}))
        metadata["repair_guard"] = _repair_guard(
            dict(rescue.get("before") or {}), repair_bindings)
        metadata["repair_evidence_ref"] = repair_evidence_ref
        payload["metadata"] = metadata
        candidate = parent if duplicate_rule else AbstractAtomicSkill.from_dict(payload)
        if not duplicate_rule:
            branch_registry.register(candidate)
            branch_registry.add_edge(
                str(candidate.ref), str(parent.ref), EdgeType.SUPERSEDES,
                evidence=[trace.trace_id],
                metadata={"branch_ref": _branch_ref(incident["branch_id"]),
                          "strict_seeded_replay": True})
        branch_report = validate_graph(branch_registry, branch_tools)
        seed_context = "\n".join(
            [f"[Candidate Atomic Skill] {candidate.summary}"]
            + [f"  - {rule}" for rule in candidate.guideline_rules()]
            + ["[Current Task Bindings]"]
            + [f"  - $inputs.{key}={value}"
               for key, value in repair_bindings.items()])
        try:
            replay = self.adapter.run_env_episode(
                task, self.llm, seed_context=seed_context,
                max_steps=self.config.max_steps,
                stop_effects=list(parent.effects),
                effect_inputs=repair_bindings,
                node_ref=str(candidate.ref),
                phase_goal=parent.summary)
            replay_actions = list(getattr(replay, "actions", None) or [])
            effect_reached = bool(getattr(replay, "atomic_complete", False))
            seeded_only = (bool(getattr(replay, "seeded_used", False))
                           and not bool(getattr(replay, "dynamic_used", False)))
            template_executed = _actions_follow_template(
                replay_actions, rescue_actions, repair_bindings,
                candidate_ref=str(candidate.ref))
            replay_passed = effect_reached and seeded_only and template_executed
            replay_payload = replay.to_dict() if hasattr(replay, "to_dict") else {
                "success": bool(getattr(replay, "success", False)),
                "failure_type": str(getattr(replay, "failure_type", "")),
            }
            replay_payload["atomic_effect_reached"] = effect_reached
            replay_payload["seeded_only"] = seeded_only
            replay_payload["repair_template_executed"] = template_executed
            replay_payload["verified_repair"] = replay_passed
        except Exception as exc:  # noqa: BLE001
            replay_passed = False
            replay_payload = {"success": False,
                              "error": f"{type(exc).__name__}: {exc}"}
        manifest = {
            **incident, "kind": "atomic_guideline_repair",
            "base_main_digest": base_main_digest,
            "source_skill_ref": str(original.ref),
            "candidate_skill_ref": str(candidate.ref),
            "candidate_patch": ({"reuse_guideline_rule": repair_rule}
                                if duplicate_rule else
                                {"add_guideline_rule": repair_rule}),
            "branch_graph_validation": branch_report.to_dict(),
            "strict_seeded_replay": replay_payload,
            "merge_audit": {"target_candidate_conditioned": True,
                            "direct_route_used": False,
                            "dynamic_fallback_allowed": False,
                            "source_task_id": task.task_id,
                            "source_trace_id": trace.trace_id},
            "replay_bindings": repair_bindings,
            "repair_event_slice": _repair_slice_audit(event_slice),
        }
        if not branch_report.passed or not replay_passed:
            manifest.update({"status": "rejected", "reason":
                             "branch_graph_invalid" if not branch_report.passed
                             else "strict_seeded_replay_failed"})
            return self._write_manifest(branch_dir, manifest)
        if _bank_digest(self.data_dir) != base_main_digest:
            manifest.update({
                "status": "rejected",
                "reason": "main_bank_changed_during_repair",
                "main_digest_after_replay": _bank_digest(self.data_dir),
            })
            return self._write_manifest(branch_dir, manifest)

        # A semantically identical, already verified rule is runtime evidence,
        # not a new capability definition.  Revalidate it on the new source
        # task and accumulate independent trace support without version churn.
        if duplicate_rule:
            parent_metadata = dict(parent.metadata or {})
            evidence_ids = list(parent_metadata.get("repair_replay_trace_ids") or [])
            if trace.trace_id not in evidence_ids:
                evidence_ids.append(trace.trace_id)
            parent_metadata.update({
                "repair_replay_trace_ids": evidence_ids,
                "repair_replay_count": len(evidence_ids),
                "last_repair_branch_ref": _branch_ref(incident["branch_id"]),
                "repair_evidence_ref": repair_evidence_ref,
            })
            parent.metadata = parent_metadata
            before_update = _bank_digest(self.data_dir)
            self.registry.update_runtime_state(parent)
            main_report = validate_graph(self.registry, self.tool_registry)
            manifest.update({
                "status": "evidence_updated" if main_report.passed else "rejected",
                "reason": ("existing_parameterized_repair_revalidated"
                           if main_report.passed else "main_graph_invalid"),
                "main_digest_before_merge": before_update,
                "main_digest_after_merge": _bank_digest(self.data_dir),
                "main_graph_validation": main_report.to_dict(),
            })
            return self._write_manifest(branch_dir, manifest)
        before_merge = _bank_digest(self.data_dir)
        self.registry.register(candidate)
        self.registry.add_edge(str(candidate.ref), str(parent.ref),
                               EdgeType.SUPERSEDES, evidence=[trace.trace_id],
                               metadata={"branch_ref": _branch_ref(incident["branch_id"]),
                                         "strict_seeded_replay": True})
        main_report = validate_graph(self.registry, self.tool_registry)
        if not main_report.passed:
            self.registry.rollback(parent.ref.logical_id, parent.ref.version)
        manifest.update({
            "status": "merged" if main_report.passed else "rejected",
            "reason": ("strict_seeded_replay_and_graph_validation_passed"
                       if main_report.passed else "main_graph_invalid"),
            "main_digest_before_merge": before_merge,
            "main_digest_after_merge": _bank_digest(self.data_dir),
            "main_graph_validation": main_report.to_dict(),
        })
        return self._write_manifest(branch_dir, manifest)

    def _process_code_attempts(self, trace: TraceRecord, task) -> list[dict[str, Any]]:
        attempts = [attempt.to_dict() if hasattr(attempt, "to_dict") else dict(attempt)
                    for attempt in trace.attempts]
        events: list[dict[str, Any]] = []
        for index, attempt in enumerate(attempts):
            if attempt.get("stage") != "direct_tool" or attempt.get("passed"):
                continue
            rescue = next((later for later in attempts[index + 1:]
                           if later.get("passed")), None)
            branch_id = _safe_id(f"{trace.trace_id}-code-{index}-direct")
            incident = {
                "branch_id": branch_id, "trace_id": trace.trace_id,
                "task_id": trace.task_id, "task_type": trace.task_type,
                "benchmark": trace.benchmark, "node_ref":
                    (trace.realized_atomic_nodes[0].get("ref", "")
                     if trace.realized_atomic_nodes else ""),
                "attempt_index": index, "mode": "direct",
                "failure_type": str(attempt.get("failure_type") or "tool_execution_error"),
                "tool_refs": list(trace.tool_refs), "rescue": rescue,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            if not rescue or not trace.candidate_code:
                events.append(self._record_unverified_branch(incident))
                continue
            events.append(self._process_code_direct_incident(trace, task, incident))
        return events

    def _process_code_direct_incident(self, trace: TraceRecord, task,
                                      incident: dict[str, Any]) -> dict[str, Any]:
        if not self.config.features.enable_tool_evolution:
            return self._record_unverified_branch(incident, "tool_evolution_disabled")
        if not incident.get("tool_refs"):
            return self._record_unverified_branch(incident, "tool_ref_missing")
        try:
            source_ref = ToolRef.parse(str(incident["tool_refs"][0]))
        except ValueError:
            return self._record_unverified_branch(incident, "invalid_tool_ref")
        original = self.tool_registry.get(source_ref)
        if original is None:
            return self._record_unverified_branch(incident, "tool_missing")

        branch_dir, branch_registry, branch_tools = self._branch_snapshot(incident)
        latest = branch_tools.get_latest(original.tool_id) or original
        payload = original.to_dict()
        payload["version"] = bump_version(latest.ref.version, "patch")
        payload["status"] = ToolLifecycle.DRAFT.value
        payload["summary"] = f"Failure-repair candidate for {original.summary}"
        payload["artifact"] = {**dict(original.artifact), "code": trace.candidate_code}
        replay_tests = list(trace.benchmark_result.get("tests") or [])
        if replay_tests:
            payload["tests"] = [{"kind": "replay", "tests": replay_tests,
                                 "source_trace_id": trace.trace_id}]
        payload["statistics"] = {"support_count": 1, "call_count": 0,
                                 "success_count": 0, "failure_count": 0,
                                 "utility": 0.5}
        payload["lineage"] = {**dict(original.lineage),
                              "supersedes": str(original.ref),
                              "repair_source_trace_id": trace.trace_id}
        payload["provenance"] = {**dict(original.provenance),
                                 "source_trace_ids": [trace.trace_id],
                                 "extraction_method": "failed_direct_plus_code_repair"}
        candidate = ToolAsset.from_dict(payload)
        manifest = {**incident, "kind": "code_tool_repair",
                    "base_main_digest": _bank_digest(self.data_dir),
                    "source_tool_ref": str(original.ref),
                    "candidate_tool_ref": str(candidate.ref),
                    "candidate_diff": _tool_diff(original, candidate),
                    "status": "candidate"}
        if candidate.structural_hash() == original.structural_hash():
            manifest.update({"status": "rejected", "reason": "no_semantic_change"})
            return self._write_manifest(branch_dir, manifest)
        verify = self.adapter.verify_task(task, candidate.artifact_body())
        manifest["source_task_strict_direct_replay"] = (
            verify.to_dict() if hasattr(verify, "to_dict") else dict(verify))
        if not bool(getattr(verify, "passed", False)):
            manifest.update({"status": "rejected", "reason": "source_task_replay_failed"})
            return self._write_manifest(branch_dir, manifest)
        admission = AdmissionEngine(
            timeout_seconds=self.config.thresholds.admission_timeout_seconds).admit(candidate)
        manifest["strict_direct_replay"] = admission.to_dict()
        if not admission.passed:
            candidate.status = ToolLifecycle.SHADOW
            branch_tools.register(candidate)
            manifest.update({"status": "rejected", "reason": "admission_failed"})
            return self._write_manifest(branch_dir, manifest)
        candidate.statistics = {"support_count": 1, "call_count": 0,
                                "success_count": 0, "failure_count": 0,
                                "utility": 0.5}
        candidate.record_usage(True, usage_mode="direct")
        branch_tools.register(candidate)
        branch_tools.recommend(candidate.ref)
        evolved_impls = self._bind_candidate_implementations(
            branch_registry, source_ref, candidate.ref, incident["branch_id"],
            exact_impl_ref=str(incident.get("implementation_ref") or ""))
        branch_report = validate_graph(branch_registry, branch_tools)
        manifest["branch_graph_validation"] = branch_report.to_dict()
        manifest["candidate_implementation_refs"] = [str(item.ref)
                                                       for item in evolved_impls]
        if not branch_report.passed or not evolved_impls:
            manifest.update({"status": "rejected", "reason": "branch_graph_invalid"})
            return self._write_manifest(branch_dir, manifest)
        before_merge = _bank_digest(self.data_dir)
        self.tool_registry.register(candidate)
        self.tool_registry.recommend(candidate.ref)
        for impl in evolved_impls:
            self.registry.register(impl)
            parent = str(impl.execution_policy.get("repair_parent_impl") or "")
            if parent:
                self.registry.add_edge(str(impl.ref), parent, EdgeType.SUPERSEDES,
                                       evidence=[trace.trace_id],
                                       metadata={"branch_ref": _branch_ref(incident["branch_id"]),
                                                 "strict_direct_replay": True})
        main_report = validate_graph(self.registry, self.tool_registry)
        if not main_report.passed:
            self.tool_registry.rollback(original.tool_id, original.ref.version)
            for impl in evolved_impls:
                parent = str(impl.execution_policy.get("repair_parent_impl") or "")
                if parent:
                    try:
                        parent_ref = SkillRef.parse(parent)
                        self.registry.rollback(parent_ref.logical_id, parent_ref.version)
                    except ValueError:
                        pass
        manifest.update({
            "status": "merged" if main_report.passed else "rejected",
            "reason": ("strict_direct_replay_and_graph_validation_passed"
                       if main_report.passed else "main_graph_invalid"),
            "main_digest_before_merge": before_merge,
            "main_digest_after_merge": _bank_digest(self.data_dir),
            "main_graph_validation": main_report.to_dict(),
            "merge_audit": {"target_attempt_forced_mode": "direct",
                            "seeded_fallback_allowed": False,
                            "dynamic_fallback_allowed": False,
                            "source_task_id": task.task_id,
                            "source_trace_id": trace.trace_id},
        })
        return self._write_manifest(branch_dir, manifest)

    def _incident(self, trace: TraceRecord, node: dict[str, Any], node_index: int,
                  attempt: dict[str, Any], attempt_index: int,
                  attempts: list[dict[str, Any]]) -> dict[str, Any]:
        rescue = next((candidate for candidate in attempts[attempt_index + 1:]
                       if bool(candidate.get("passed"))), None)
        ref = str(node.get("ref") or node.get("node_ref") or "")
        branch_id = _safe_id(
            f"{trace.trace_id}-{node_index}-{attempt_index}-{attempt.get('mode', 'unknown')}")
        return {
            "branch_id": branch_id,
            "trace_id": trace.trace_id,
            "task_id": trace.task_id,
            "task_type": trace.task_type,
            "benchmark": trace.benchmark,
            "node_ref": ref,
            "occurrence_id": str(node.get("occurrence_id") or ""),
            "implementation_ref": str(node.get("impl_ref") or ""),
            "node_index": node_index,
            "attempt_index": attempt_index,
            "mode": str(attempt.get("mode") or "unknown"),
            "failure_type": str(attempt.get("failure_type") or "unknown"),
            "tool_refs": list(attempt.get("tool_refs") or []),
            "before": dict(attempt.get("before") or node.get("before") or {}),
            "after": dict(attempt.get("after") or {}),
            "action_start": int(attempt.get("action_start") or 0),
            "action_end": int(attempt.get("action_end") or 0),
            "rescue": dict(rescue) if rescue else None,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def _process_direct_incident(self, trace: TraceRecord, task,
                                 node: dict[str, Any],
                                 incident: dict[str, Any]) -> dict[str, Any]:
        if not self.config.features.enable_tool_evolution:
            return self._record_unverified_branch(incident,
                                                  "tool_evolution_disabled")
        rescue = incident.get("rescue") or {}
        tool_refs = incident.get("tool_refs") or []
        if not rescue or not tool_refs:
            return self._record_unverified_branch(incident)
        try:
            source_ref = ToolRef.parse(str(tool_refs[0]))
        except ValueError:
            return self._record_unverified_branch(incident, "invalid_tool_ref")
        original = self.tool_registry.get(source_ref)
        if original is None:
            return self._record_unverified_branch(incident, "tool_missing")

        try:
            atomic_ref = SkillRef.parse(str(node.get("ref") or node.get("node_ref") or ""))
        except ValueError:
            atomic_ref = None
        atomic = self.registry.get(atomic_ref) if atomic_ref else None
        core_effects = list(getattr(atomic, "effects", None) or node.get("effects") or [])
        if not core_effects:
            return self._record_unverified_branch(incident, "repair_core_effect_missing")
        rescue_start = int(rescue.get("action_start") or incident.get("action_end") or 0)
        rescue_end = int(rescue.get("action_end") or rescue_start)
        preparation = _prepare_repair_evidence(
            trace, rescue_start, rescue_end, core_effects,
            dict(node.get("params") or {}))
        if not bool(preparation.get("replay_safe")):
            return self._record_unverified_branch(
                _repair_preparation_incident(incident, preparation),
                str(preparation.get("reason") or "unsafe_repair_event_slice"))
        event_slice = dict(preparation["event_slice"])
        repair_bindings = dict(preparation["repair_bindings"])
        generalized_steps = list(preparation["steps"])
        bound_core_effects = list(preparation["bound_core_effects"])

        branch_dir, branch_registry, branch_tools = self._branch_snapshot(incident)
        candidate = self._repair_tool(
            original, trace, node, incident, rescue, branch_tools,
            event_slice=event_slice, core_effects=bound_core_effects,
            repair_bindings=repair_bindings, steps=generalized_steps)
        manifest = dict(incident)
        manifest.update({
            "kind": "tool_repair",
            "base_main_digest": _bank_digest(self.data_dir),
            "source_tool_ref": str(original.ref),
            "candidate_tool_ref": str(candidate.ref),
            "candidate_diff": _tool_diff(original, candidate),
            "repair_event_slice": _repair_slice_audit(event_slice),
            "replay_bindings": repair_bindings,
            "status": "candidate",
        })
        if candidate.structural_hash() == original.structural_hash():
            manifest.update({"status": "rejected", "reason": "no_semantic_change"})
            return self._write_manifest(branch_dir, manifest)

        admission = AdmissionEngine(
            replay_fn=getattr(self.adapter, "replay_tool", None),
            timeout_seconds=self.config.thresholds.admission_timeout_seconds,
        ).admit(candidate)
        manifest["strict_direct_replay"] = admission.to_dict()
        if not admission.passed:
            candidate.status = ToolLifecycle.SHADOW
            branch_tools.register(candidate)
            manifest.update({"status": "rejected", "reason": "strict_direct_replay_failed"})
            return self._write_manifest(branch_dir, manifest)

        # Strict replay is executable success evidence, not merely admission metadata.
        candidate.statistics = {"support_count": 1, "call_count": 0,
                                "success_count": 0, "failure_count": 0,
                                "utility": 0.5}
        candidate.record_usage(True, usage_mode="direct")
        branch_tools.register(candidate)
        branch_tools.recommend(candidate.ref)
        evolved_impls = self._bind_candidate_implementations(
            branch_registry, source_ref, candidate.ref, incident["branch_id"],
            exact_impl_ref=str(incident.get("implementation_ref") or ""))
        graph_report = validate_graph(branch_registry, branch_tools)
        manifest["branch_graph_validation"] = graph_report.to_dict()
        manifest["candidate_implementation_refs"] = [str(item.ref)
                                                       for item in evolved_impls]
        if not graph_report.passed or not evolved_impls:
            manifest.update({"status": "rejected", "reason":
                             "branch_graph_invalid" if not graph_report.passed
                             else "no_bound_implementation"})
            return self._write_manifest(branch_dir, manifest)

        before_merge = _bank_digest(self.data_dir)
        self.tool_registry.register(candidate)
        self.tool_registry.recommend(candidate.ref)
        for impl in evolved_impls:
            self.registry.register(impl)
            parent = str(impl.execution_policy.get("repair_parent_impl") or "")
            if parent:
                self.registry.add_edge(str(impl.ref), parent, EdgeType.SUPERSEDES,
                                       evidence=[trace.trace_id],
                                       metadata={"branch_ref": _branch_ref(incident["branch_id"]),
                                                 "strict_direct_replay": True})
        main_report = validate_graph(self.registry, self.tool_registry)
        if not main_report.passed:
            # The old recommended pointers are recoverable; reject loudly rather
            # than silently presenting an invalid merge as successful.
            self.tool_registry.rollback(original.tool_id, original.ref.version)
            for impl in evolved_impls:
                parent = str(impl.execution_policy.get("repair_parent_impl") or "")
                if parent:
                    try:
                        parent_ref = SkillRef.parse(parent)
                        self.registry.rollback(parent_ref.logical_id, parent_ref.version)
                    except ValueError:
                        pass
            manifest.update({"status": "rejected", "reason": "main_graph_invalid",
                             "main_graph_validation": main_report.to_dict()})
            return self._write_manifest(branch_dir, manifest)

        manifest.update({
            "status": "merged",
            "reason": "strict_direct_replay_and_graph_validation_passed",
            "merged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "main_digest_before_merge": before_merge,
            "main_digest_after_merge": _bank_digest(self.data_dir),
            "main_graph_validation": main_report.to_dict(),
            "merge_audit": {
                "target_attempt_forced_mode": "direct",
                "seeded_fallback_allowed": False,
                "dynamic_fallback_allowed": False,
                "source_task_id": task.task_id,
                "source_trace_id": trace.trace_id,
            },
        })
        return self._write_manifest(branch_dir, manifest)

    def _repair_tool(self, original: ToolAsset, trace: TraceRecord,
                     node: dict[str, Any], incident: dict[str, Any],
                     rescue: dict[str, Any], branch_tools: ToolRegistry, *,
                     event_slice: dict[str, Any],
                     core_effects: list[dict[str, Any]],
                     repair_bindings: dict[str, Any],
                     steps: list[str]) -> ToolAsset:
        actions = [action.to_dict() if hasattr(action, "to_dict") else dict(action)
                   for action in trace.actions]
        prefix_end = int(incident.get("action_start") or 0)
        prefix = [str(action.get("name") or "") for action in actions[:prefix_end]
                  if str(action.get("name") or "").strip()]
        latest = branch_tools.get_latest(original.tool_id) or original
        data = original.to_dict()
        data["version"] = bump_version(latest.ref.version, "patch")
        data["status"] = ToolLifecycle.DRAFT.value
        data["summary"] = f"Failure-repair candidate for {original.summary}"
        data["artifact"] = {"template": "\n".join(steps), "steps": steps}
        data["tests"] = [{
            "kind": "replay",
            "bindings": repair_bindings,
            "steps": steps,
            "before": dict(incident.get("before") or {}),
            "after": dict(rescue.get("after") or node.get("after") or {}),
            "expected_effects": parameterize_predicates(
                core_effects, repair_bindings),
            "prefix": prefix,
            "source_trace_id": trace.trace_id,
            "source": {"task_id": trace.task_id,
                       "env_index": (trace.provenance or {}).get("env_index")},
        }]
        data["statistics"] = {"support_count": 1, "call_count": 0,
                              "success_count": 0, "failure_count": 0,
                              "utility": 0.5}
        data["lineage"] = {**dict(original.lineage),
                           "supersedes": str(original.ref),
                           "repair_source_trace_id": trace.trace_id,
                           "repair_source_attempt": incident["attempt_index"]}
        data["provenance"] = {**dict(original.provenance),
                              "source_trace_ids": [trace.trace_id],
                              "extraction_method":
                                  "failed_direct_plus_causal_rescue_slice",
                              "repair_event_slice": {
                                  key: value for key, value in event_slice.items()
                                  if key != "events"}}
        return ToolAsset.from_dict(data)

    @staticmethod
    def _bind_candidate_implementations(registry: SkillGraphRegistry,
                                        old_tool: ToolRef, new_tool: ToolRef,
                                        branch_id: str, *,
                                        exact_impl_ref: str = ""
                                        ) -> list[ImplementationAtom]:
        evolved: list[ImplementationAtom] = []
        implementations: list[ImplementationAtom] = []
        if exact_impl_ref:
            try:
                exact = registry.get(SkillRef.parse(exact_impl_ref))
            except ValueError:
                exact = None
            if isinstance(exact, ImplementationAtom):
                implementations = [exact]
        else:
            # Legacy code-task incidents did not persist an Implementation ref.
            # Environment runtime incidents always take the exact branch above.
            implementations = registry.list_by_kind(
                SkillNodeKind.IMPLEMENTATION_ATOMIC)
        for impl in implementations:
            if not any(binding.tool_ref == old_tool for binding in impl.tool_bindings):
                continue
            latest = registry.get_latest(impl.ref.logical_id) or impl
            bindings = [
                ToolBinding(tool_ref=new_tool if binding.tool_ref == old_tool else binding.tool_ref,
                            role=binding.role,
                            parameter_mapping=dict(binding.parameter_mapping))
                for binding in impl.tool_bindings
            ]
            candidate = ImplementationAtom(
                ref=SkillRef(impl.ref.logical_id,
                             bump_version(latest.ref.version, "patch")),
                abstract_ref=impl.abstract_ref,
                tool_bindings=bindings,
                execution_policy={**dict(impl.execution_policy),
                                  "repair_branch_ref": _branch_ref(branch_id),
                                  "repair_parent_impl": str(impl.ref),
                                  "strict_direct_replay": True},
                compatibility=dict(impl.compatibility),
                quality={"use_count": 0, "success_count": 1,
                         "failure_count": 0, "utility": 0.6},
                status=SkillStatus.ACTIVE,
            )
            registry.register(candidate)
            registry.add_edge(str(candidate.ref), str(impl.ref), EdgeType.SUPERSEDES,
                              metadata={"branch_ref": _branch_ref(branch_id)},
                              evidence=[_branch_ref(branch_id)])
            evolved.append(candidate)
        return evolved

    def _branch_snapshot(self, incident: dict[str, Any]) -> tuple[Path, SkillGraphRegistry, ToolRegistry]:
        branch_dir = self.root / incident["branch_id"]
        bank = branch_dir / "bank"
        skill_dst, tool_dst = bank / "skill_graph", bank / "tools"
        branch_dir.mkdir(parents=True, exist_ok=True)
        if not skill_dst.exists():
            shutil.copytree(self.registry.root, skill_dst)
        if not tool_dst.exists():
            shutil.copytree(self.tool_registry.root, tool_dst)
        # Branch Tool assets contain only evidence refs.  The branch is a
        # mutable local validation workspace (not a portable frozen export),
        # so copy its private Tool evidence as well; otherwise even diagnostic
        # shadow versions would inherit dangling refs and fail closed before
        # branch validation can run.
        evidence_src = self.tool_registry.evidence_store.root
        evidence_dst = bank / "evidence" / "tool_tests"
        if evidence_src.exists() and not evidence_dst.exists():
            shutil.copytree(evidence_src, evidence_dst)
        return branch_dir, SkillGraphRegistry(skill_dst), ToolRegistry(tool_dst)

    def _record_unverified_branch(self, incident: dict[str, Any],
                                  reason: str = "no_successful_rescue") -> dict[str, Any]:
        branch_dir, branch_registry, branch_tools = self._branch_snapshot(incident)
        candidates = self._create_shadow_candidates(
            incident, branch_registry, branch_tools, reason)
        branch_report = validate_graph(branch_registry, branch_tools)
        manifest = {**incident, "kind": f"{incident['mode']}_repair",
                    "status": "awaiting_success_evidence", "reason": reason,
                    "base_main_digest": _bank_digest(self.data_dir),
                    "shadow_candidates": candidates,
                    "branch_graph_validation": branch_report.to_dict(),
                    "merge_allowed": False,
                    "merge_gate": "successful_source_task_replay_required"}
        return self._write_manifest(branch_dir, manifest)

    @staticmethod
    def _create_shadow_candidates(incident: dict[str, Any],
                                  registry: SkillGraphRegistry,
                                  tools: ToolRegistry,
                                  reason: str) -> list[dict[str, Any]]:
        """Materialize diagnostic candidates only inside an unverified branch.

        With no successful rescue there is no sound executable replacement to
        learn from.  We still create an immutable patch candidate containing
        the observed failure and a concrete review rule.  It remains shadow /
        draft and can never be merged by this path.
        """
        candidates: list[dict[str, Any]] = []
        failure_type = _failure_category(incident.get("failure_type"))
        branch_id = str(incident["branch_id"])
        branch_ref = _branch_ref(branch_id)
        incident_evidence = registry.evidence_store.put(
            "repair_incident", incident,
            trace_id=str(incident.get("trace_id") or ""))
        node_ref_text = str(incident.get("node_ref") or "")
        try:
            node_ref = SkillRef.parse(node_ref_text)
        except ValueError:
            node_ref = None
        source = registry.get(node_ref) if node_ref else None
        if isinstance(source, AbstractAtomicSkill):
            latest = registry.get_latest(source.ref.logical_id) or source
            payload = source.to_dict()
            payload["version"] = bump_version(latest.ref.version, "patch")
            payload["status"] = SkillStatus.SHADOW.value
            failure_modes = list(payload.get("failure_modes") or [])
            observation = {
                "type": failure_type,
                "source_trace_id": incident.get("trace_id"),
                "branch_ref": branch_ref,
                "evidence_ref": incident_evidence,
                "verified_repair": False,
            }
            if observation not in failure_modes:
                failure_modes.append(observation)
            payload["failure_modes"] = failure_modes
            guideline = dict(payload.get("guideline") or {})
            rules = list(guideline.get("rules") or [])
            rule = (f"未验证候选：执行前检查 {failure_type}；只有在原失败任务上"
                    "严格重放成功后才允许合并。")
            if rule not in rules:
                rules.append(rule)
            guideline["rules"] = rules
            payload["guideline"] = guideline
            metadata = dict(payload.get("metadata") or {})
            metadata.update({"repair_branch_ref": branch_ref,
                             "repair_evidence_ref": incident_evidence,
                             "repair_parent_ref": str(source.ref),
                             "verified_repair": False,
                             "awaiting_reason": reason})
            payload["metadata"] = metadata
            candidate = AbstractAtomicSkill.from_dict(payload)
            registry.register(candidate)
            candidates.append({"kind": candidate.kind.value,
                               "ref": str(candidate.ref),
                               "parent_ref": str(source.ref),
                               "change": "failure_observation_and_guard_rule"})
        elif isinstance(source, CompositeSkill):
            latest = registry.get_latest(source.ref.logical_id) or source
            payload = source.to_dict()
            payload["version"] = bump_version(latest.ref.version, "patch")
            payload["status"] = SkillStatus.SHADOW.value
            guideline = dict(payload.get("guideline") or {})
            rules = list(guideline.get("rules") or [])
            rule = (f"未验证候选：组合顺序出现 {failure_type}；需要成功顺序轨迹"
                    "并在原任务严格重放后才能替换主版本。")
            if rule not in rules:
                rules.append(rule)
            guideline["rules"] = rules
            payload["guideline"] = guideline
            metadata = dict(payload.get("metadata") or {})
            metadata.update({"repair_branch_ref": branch_ref,
                             "repair_evidence_ref": incident_evidence,
                             "repair_parent_ref": str(source.ref),
                             "verified_repair": False,
                             "awaiting_reason": reason})
            payload["metadata"] = metadata
            candidate = CompositeSkill.from_dict(payload)
            registry.register(candidate)
            candidates.append({"kind": candidate.kind.value,
                               "ref": str(candidate.ref),
                               "parent_ref": str(source.ref),
                               "change": "ordering_failure_guard_rule"})

        for text_ref in list(incident.get("tool_refs") or [])[:1]:
            try:
                tool_ref = ToolRef.parse(str(text_ref))
            except ValueError:
                continue
            source_tool = tools.get(tool_ref)
            if source_tool is None:
                continue
            latest_tool = tools.get_latest(source_tool.tool_id) or source_tool
            payload = source_tool.to_dict()
            payload["version"] = bump_version(latest_tool.ref.version, "patch")
            payload["status"] = ToolLifecycle.SHADOW.value
            payload["summary"] = f"Unverified failure-diagnosis candidate: {source_tool.summary}"
            tests = list(payload.get("tests") or [])
            tests.append({"kind": "failure_observation",
                          "source_trace_id": incident.get("trace_id"),
                          "source_task_id": incident.get("task_id"),
                          "failure_type": failure_type,
                          "passed": False})
            payload["tests"] = tests
            provenance = dict(payload.get("provenance") or {})
            provenance.update({"repair_branch_ref": branch_ref,
                               "repair_parent_ref": str(source_tool.ref),
                               "verified_repair": False,
                               "awaiting_reason": reason})
            payload["provenance"] = provenance
            payload["statistics"] = {"support_count": 0, "call_count": 0,
                                     "success_count": 0, "failure_count": 0,
                                     "utility": 0.0}
            candidate_tool = ToolAsset.from_dict(payload)
            tools.register(candidate_tool)
            candidates.append({"kind": "tool",
                               "ref": str(candidate_tool.ref),
                               "parent_ref": str(source_tool.ref),
                               "change": "failure_regression_case_only"})
        return candidates

    @staticmethod
    def _write_manifest(branch_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
        path = branch_dir / "manifest.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return manifest


def _parameterize_rescue_actions(actions: list[str], params: dict[str, Any]) -> list[str]:
    result: list[str] = []
    values = _safe_replacement_values(actions, params)
    for raw in actions:
        text = str(raw).strip()
        if not text:
            continue
        for value, name in values:
            pattern = rf"(?<![a-z0-9]){re.escape(value)}(?![a-z0-9])"
            text = re.sub(pattern, "{" + name + "}", text,
                          flags=re.IGNORECASE)
        # If a concrete instance remains after replacing every known binding,
        # the action is episode-specific and cannot enter a reusable repair.
        if _contains_concrete_instance(text):
            continue
        # The shared event-level causal slicer has already removed exploration,
        # recovery loops and irrelevant state toggles. Do not rewrite the
        # verified path by recognizing domain verbs here.
        if not result or result[-1] != text:
            result.append(text)
    return result


def _prepare_repair_evidence(trace: TraceRecord, start: int, end: int,
                             core_effects: list[dict[str, Any]],
                             params: dict[str, Any]) -> dict[str, Any]:
    """Resolve one concrete rescue occurrence before causal slicing.

    Generic task bindings such as ``apple`` cannot safely anchor a core Effect
    while a rescue window contains both ``apple_1`` and ``apple_2``.  Resolve
    the complete accepted rescue window first (including structured action
    params), fail closed on multiple occurrences, and only then invoke the
    shared extractor slice with concretely bound Effects.  Atomic guideline and
    Tool repair both consume this exact result so neither path can silently
    reinterpret the retained suffix.
    """
    window_events = _repair_window_events(trace, start, end)
    if window_events is None:
        event_slice = _validated_repair_event_slice(
            trace, start, end, core_effects, params)
        return {
            "replay_safe": False,
            "reason": "unsafe_repair_event_slice",
            "event_slice": event_slice,
        }

    repair_bindings, binding_candidates, ambiguous_roles = (
        _resolve_repair_event_bindings(window_events, params))
    if ambiguous_roles:
        return {
            "replay_safe": False,
            "reason": "ambiguous_binding",
            "repair_bindings": repair_bindings,
            "binding_candidates": binding_candidates,
            "ambiguous_parameter_roles": ambiguous_roles,
            "event_slice": {
                "events": [],
                "retained_event_indices": [],
                "effect_producer_indices": [],
                "replay_safe": False,
                "event_slice_validated": False,
                "reason": "ambiguous_binding",
                "window_start": start,
                "window_end": end - 1,
                "window_event_indices": [
                    int(event["event_index"]) for event in window_events],
            },
        }

    bound_core_effects = [
        _bind_repair_effect(effect, repair_bindings)
        for effect in core_effects if isinstance(effect, dict)]
    event_slice = _validated_repair_event_slice(
        trace, start, end, bound_core_effects, repair_bindings)
    if not bool(event_slice.get("replay_safe")):
        return {
            "replay_safe": False,
            "reason": "unsafe_repair_event_slice",
            "repair_bindings": repair_bindings,
            "binding_candidates": binding_candidates,
            "bound_core_effects": bound_core_effects,
            "event_slice": event_slice,
        }

    retained_events = [dict(event) for event in (event_slice.get("events") or [])]
    retained_indices = {
        int(index) for index in (event_slice.get("retained_event_indices") or [])}
    retained_indices.update(
        int(event.get("event_index", -1)) for event in retained_events)
    producer_indices = {
        int(index) for index in (event_slice.get("effect_producer_indices") or [])}
    if not producer_indices or not producer_indices.issubset(retained_indices):
        return {
            "replay_safe": False,
            "reason": "effect_producer_not_retained",
            "repair_bindings": repair_bindings,
            "binding_candidates": binding_candidates,
            "bound_core_effects": bound_core_effects,
            "event_slice": event_slice,
        }

    raw_actions = [
        str(event.get("action") or event.get("name") or "")
        for event in retained_events
        if str(event.get("action") or event.get("name") or "").strip()]
    steps = _parameterize_rescue_actions(raw_actions, repair_bindings)
    if not steps:
        return {
            "replay_safe": False,
            "reason": "no_generalizable_rescue_actions",
            "repair_bindings": repair_bindings,
            "binding_candidates": binding_candidates,
            "bound_core_effects": bound_core_effects,
            "event_slice": event_slice,
        }

    producer_events = sorted(
        (event for event in retained_events
         if int(event.get("event_index", -1)) in producer_indices),
        key=lambda event: int(event.get("event_index", -1)))
    producer_templates: list[str] = []
    for event in producer_events:
        action = str(event.get("action") or event.get("name") or "")
        generalized = _parameterize_rescue_actions([action], repair_bindings)
        if not generalized:
            producer_templates = []
            break
        producer_templates.append(generalized[-1])
    producer_retained = (
        len(producer_events) == len(producer_indices)
        and len(producer_templates) == len(producer_indices)
        and all(any(_normalize_action(producer) == _normalize_action(step)
                    for step in steps)
                for producer in producer_templates)
        and bool(producer_templates)
        and _normalize_action(producer_templates[-1]) == _normalize_action(steps[-1])
    )
    if not producer_retained:
        return {
            "replay_safe": False,
            "reason": "effect_producer_not_retained",
            "repair_bindings": repair_bindings,
            "binding_candidates": binding_candidates,
            "bound_core_effects": bound_core_effects,
            "event_slice": event_slice,
        }
    return {
        "replay_safe": True,
        "reason": "repair_evidence_validated",
        "repair_bindings": repair_bindings,
        "binding_candidates": binding_candidates,
        "bound_core_effects": bound_core_effects,
        "event_slice": event_slice,
        "steps": steps,
        "effect_producer_indices": sorted(producer_indices),
    }


def _repair_window_events(trace: TraceRecord, start: int,
                          end: int) -> list[dict[str, Any]] | None:
    actions = [action.to_dict() if hasattr(action, "to_dict") else dict(action)
               for action in trace.actions]
    if start < 0 or end <= start or end > len(actions):
        return None
    return [
        {**dict(action), "action": str(action.get("name") or ""),
         "event_index": index}
        for index, action in enumerate(actions[start:end], start=start)
        if bool(action.get("accepted", True))
        and str(action.get("name") or "").strip()
    ]


def _resolve_repair_event_bindings(
        events: list[dict[str, Any]],
        params: dict[str, Any],
        ) -> tuple[dict[str, Any], dict[str, list[str]], list[str]]:
    """Return concrete bindings and every grounded candidate per generic role."""
    resolved = dict(params)
    candidates_by_role: dict[str, list[str]] = {}
    ambiguous: list[str] = []
    for raw_name, raw_value in params.items():
        name = str(raw_name)
        value = _normalize_occurrence_value(raw_value)
        if not value:
            continue
        if _is_concrete_occurrence(value):
            resolved[name] = value
            candidates_by_role[name] = [value]
            continue

        family = _occurrence_family(value)
        candidates: set[str] = set()
        allow_same_role_any_family = _is_semantic_generic_value(value, name)
        for event in events:
            for event_role, event_value in dict(event.get("params") or {}).items():
                occurrence = _normalize_occurrence_value(event_value)
                if not _is_concrete_occurrence(occurrence):
                    continue
                same_family = _occurrence_family(occurrence) == family
                same_role_generic = (
                    str(event_role) == name and allow_same_role_any_family)
                if same_family or same_role_generic:
                    candidates.add(occurrence)
            action = _normalize_occurrence_value(
                event.get("action") or event.get("name") or "")
            family_pattern = re.escape(family).replace(r"\ ", r"[_ ]+")
            for match in re.finditer(
                    rf"(?<![a-z0-9]){family_pattern}[_ ]+(\d+)(?![a-z0-9])",
                    action, flags=re.IGNORECASE):
                candidates.add(f"{family} {match.group(1)}")

        ordered = sorted(candidates)
        candidates_by_role[name] = ordered
        if len(ordered) > 1:
            ambiguous.append(name)
        elif len(ordered) == 1:
            resolved[name] = ordered[0]
    return resolved, candidates_by_role, sorted(ambiguous)


def _normalize_occurrence_value(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower().replace("_", " "))


def _is_concrete_occurrence(value: str) -> bool:
    return bool(re.search(r"(?:^| )\d+$", str(value)))


def _occurrence_family(value: str) -> str:
    return re.sub(r"(?: |_)\d+$", "", _normalize_occurrence_value(value))


def _is_semantic_generic_value(value: str, role: str) -> bool:
    normalized_role = _normalize_occurrence_value(role)
    return value == normalized_role or value in {
        "object", "item", "thing", "entity", "container", "receptacle",
        "location", "place", "target", "source", "destination",
    }


def _repair_preparation_incident(incident: dict[str, Any],
                                 preparation: dict[str, Any]) -> dict[str, Any]:
    payload = dict(incident)
    event_slice = preparation.get("event_slice")
    if isinstance(event_slice, dict):
        payload["repair_event_slice"] = _repair_slice_audit(event_slice)
    for key in ("ambiguous_parameter_roles", "binding_candidates",
                "repair_bindings"):
        if key in preparation:
            payload[key] = preparation[key]
    return payload


def _validated_repair_event_slice(trace: TraceRecord, start: int, end: int,
                                  core_effects: list[dict[str, Any]],
                                  params: dict[str, Any]) -> dict[str, Any]:
    """Use the normal extractor's causal evidence gate for repair branches.

    ``action_end`` in attempts is exclusive.  Some code-only/unit adapters do
    not persist per-action states; those cases remain eligible only through the
    existing strict source-task replay and are explicitly marked as such.
    Environment traces with snapshots fail closed when their event slice is not
    replay-safe.
    """
    raw_actions = [action.to_dict() if hasattr(action, "to_dict") else dict(action)
                   for action in trace.actions]
    if start < 0 or end <= start or end > len(raw_actions):
        return {
            "events": [], "retained_event_indices": [], "replay_safe": False,
            "event_slice_validated": False, "reason": "invalid_repair_window",
            "effect_producer_indices": [],
        }
    if len(trace.state_snapshots) < len(raw_actions) + 1:
        fallback_events = [
            {**dict(action), "action": str(action.get("name") or ""),
             "event_index": index}
            for index, action in enumerate(raw_actions[start:end], start=start)
            if bool(action.get("accepted", True))
            and str(action.get("name") or "").strip()
        ]
        return {
            "events": fallback_events,
            "retained_event_indices": [int(item["event_index"])
                                       for item in fallback_events],
            "effect_producer_indices": ([int(fallback_events[-1]["event_index"])]
                                        if fallback_events else []),
            "replay_safe": bool(fallback_events),
            "event_slice_validated": False,
            "reason": "per_event_state_unavailable_strict_replay_required",
        }
    result = slice_event_occurrence(
        build_structured_events(trace), window_start=start, window_end=end - 1,
        core_effects=core_effects, params=params)
    result["event_slice_validated"] = True
    return result


def _repair_slice_audit(event_slice: dict[str, Any]) -> dict[str, Any]:
    audit = {key: value for key, value in event_slice.items() if key != "events"}
    audit["retained_actions"] = [
        str(event.get("action") or event.get("name") or "")
        for event in (event_slice.get("events") or [])]
    return audit


def _bind_repair_effect(effect: dict[str, Any],
                        bindings: dict[str, Any]) -> dict[str, Any]:
    payload = dict(effect)
    args = bind_args(dict(effect.get("args") or {}), bindings, bindings)
    for key, value in list(args.items()):
        if isinstance(value, str) and value.startswith("$task."):
            args[key] = bindings.get(value.split(".", 1)[1], value)
    payload["args"] = args
    return payload


def _safe_replacement_values(actions: list[str],
                             params: dict[str, Any]) -> list[tuple[str, str]]:
    """Resolve abstract role values only when one concrete occurrence exists."""
    joined = "\n".join(str(action) for action in actions)
    values: list[tuple[str, str]] = []
    for name, raw_value in params.items():
        value = str(raw_value).strip()
        if not value:
            continue
        normalized = value.replace("_", " ")
        if re.search(r"(?:[_ ]\d+)$", value):
            values.extend([(value, str(name)), (normalized, str(name))])
            continue
        family = re.sub(r"(?:[_ ]\d+)$", "", normalized)
        matches = {
            match.group(0).strip() for match in re.finditer(
                rf"(?<![a-z0-9]){re.escape(family)}(?:[_ ]\d+)?(?![a-z0-9])",
                joined, flags=re.IGNORECASE)}
        concrete = {item for item in matches if re.search(r"(?:[_ ]\d+)$", item)}
        if len({item.lower().replace("_", " ") for item in concrete}) == 1:
            resolved = next(iter(concrete))
            values.extend([(resolved, str(name)),
                           (resolved.replace("_", " "), str(name))])
        elif not concrete and matches:
            values.append((family, str(name)))
    unique = {(value.lower(), name): (value, name) for value, name in values}
    return sorted(unique.values(), key=lambda item: len(item[0]), reverse=True)


def _resolve_repair_bindings(actions: list[str],
                             params: dict[str, Any]) -> dict[str, Any]:
    """Resolve generic task roles to the unique concrete rescue occurrence."""
    resolved = dict(params)
    for value, name in _safe_replacement_values(actions, params):
        if re.search(r"(?:[_ ]\d+)$", value):
            current = str(resolved.get(name, ""))
            if not re.search(r"(?:[_ ]\d+)$", current):
                resolved[name] = value.replace("_", " ")
    return resolved


def _ambiguous_repair_roles(actions: list[str],
                            params: dict[str, Any]) -> list[str]:
    joined = "\n".join(str(action) for action in actions)
    ambiguous: list[str] = []
    for name, raw_value in params.items():
        value = str(raw_value).strip().replace("_", " ")
        if not value or re.search(r" \d+$", value):
            continue
        matches = {match.group(0).lower().replace("_", " ")
                   for match in re.finditer(
                       rf"(?<![a-z0-9]){re.escape(value)}(?:[_ ]\d+)(?![a-z0-9])",
                       joined, flags=re.IGNORECASE)}
        if len(matches) > 1:
            ambiguous.append(str(name))
    return sorted(ambiguous)


def _repair_guard(before: dict[str, Any],
                  bindings: dict[str, Any]) -> dict[str, Any]:
    predicates: list[dict[str, Any]] = []
    for fact in before.get("facts") or []:
        predicate = _fact_to_predicate(str(fact))
        if not isinstance(predicate, dict):
            continue
        parameterized = parameterize_predicates([predicate], bindings)
        if parameterized and is_fully_parameterized_predicate(parameterized[0]):
            predicates.extend(parameterized)
    return {
        "trigger": "primary_execution_failed_and_atomic_effect_unmet",
        "preconditions": predicates,
    }


def _contains_concrete_instance(text: str) -> bool:
    return bool(re.search(r"(?<![{$a-z0-9])(?:[a-z][a-z0-9]*[ _])+\d+(?![a-z0-9}])",
                          str(text), flags=re.IGNORECASE))


def _guideline_placeholders(step: str) -> str:
    return re.sub(r"\{([a-z_][a-z0-9_]*)\}", r"$inputs.\1", str(step))


def _actions_follow_template(actions: list[Any], template: list[str],
                             params: dict[str, Any], *,
                             candidate_ref: str = "") -> bool:
    """Require the replay to execute the proposed repair template in order."""
    rendered = [_render_template(step, params) for step in template]
    normalized_actions: list[str] = []
    for raw in actions:
        action = raw if isinstance(raw, dict) else {
            "name": str(getattr(raw, "name", raw)),
            "accepted": bool(getattr(raw, "accepted", True)),
            "mode": str(getattr(raw, "mode", "")),
            "node_ref": str(getattr(raw, "node_ref", "")),
        }
        if not bool(action.get("accepted", True)):
            continue
        if str(action.get("mode") or "") != ExecutionMode.SEEDED.value:
            continue
        if candidate_ref and str(action.get("node_ref") or "") != candidate_ref:
            continue
        name = str(action.get("name") or "").strip()
        if name:
            normalized_actions.append(_normalize_action(name))
    cursor = 0
    for expected in rendered:
        expected_norm = _normalize_action(expected)
        while cursor < len(normalized_actions) and normalized_actions[cursor] != expected_norm:
            cursor += 1
        if cursor >= len(normalized_actions):
            return False
        cursor += 1
    return bool(rendered)


def _render_template(step: str, params: dict[str, Any]) -> str:
    text = str(step)
    for name, value in params.items():
        text = text.replace("{" + str(name) + "}", str(value))
    return text


def _normalize_action(step: str) -> str:
    return re.sub(r"\s+", " ", str(step).strip().lower().replace("_", " "))


def _tool_diff(original: ToolAsset, candidate: ToolAsset) -> dict[str, Any]:
    old = list(original.artifact.get("steps") or original.artifact_body().splitlines())
    new = list(candidate.artifact.get("steps") or candidate.artifact_body().splitlines())
    return {
        "relation": "equivalent" if old == new else
                    "extended" if all(step in new for step in old) else "replaced",
        "old_steps": old,
        "new_steps": new,
        "unified_diff": list(difflib.unified_diff(old, new,
                                                   fromfile=str(original.ref),
                                                   tofile=str(candidate.ref),
                                                   lineterm="")),
    }


def _bank_digest(data_dir: Path) -> str:
    digest = hashlib.sha256()
    for dirname in ("skill_graph", "tools"):
        root = data_dir / dirname
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            digest.update(str(path.relative_to(data_dir)).replace("\\", "/").encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _safe_id(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text)[:180]


def _branch_ref(branch_id: Any) -> str:
    """Portable opaque identity for a trace-derived repair branch."""
    digest = hashlib.sha256(str(branch_id or "").encode("utf-8")).hexdigest()
    return f"branch:{digest}"


def _failure_category(value: Any) -> str:
    """Map raw error text to a portable framework-level category."""
    text = str(value or "").lower()
    categories = (
        "precondition", "data_flow", "control_flow", "binding", "budget",
        "planning", "timeout", "llm", "tool", "effect", "validation",
        "environment",
    )
    return next((category for category in categories
                 if category.replace("_", " ") in text
                 or category in text), "unknown_failure")
