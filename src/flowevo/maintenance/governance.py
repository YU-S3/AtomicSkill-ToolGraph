"""Governance kernel with runtime audit for code skills and primitives."""

from __future__ import annotations

from pathlib import Path
import re

from agent.generator import BaseGenerator, GenerationError
from core.schemas import BenchmarkTaskInstance, CodeTaskInstance, PrimitiveCard, SkillCard, VerifierFeedback
from env.sandbox import Sandbox
from env.benchmark_adapter import load_benchmark_tasks
from eval.verifier import verify_task
from memory.negative_memory_store import NegativeMemoryStore
from memory.primitive_store import PrimitiveStore
from memory.skill_registry import SkillRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class GovernanceKernel:
    """Apply minimal runtime governance and periodic audit for skills."""

    _SEED_MODE_ORDER = {
        "guard_only_seed": 0,
        "skeleton_only_seed": 1,
        "code_excerpt_seed": 2,
    }
    _SHADOW_ACTION_NEGATIVE_SOURCES = (
        "compile_rejection",
        "trace_audit",
        "runtime_failure",
        "audit_failure",
        "prune_reason",
    )

    def __init__(
        self,
        registry: SkillRegistry,
        primitive_store: PrimitiveStore | None = None,
        negative_memory_store: NegativeMemoryStore | None = None,
        sandbox: Sandbox | None = None,
        repair_generator: BaseGenerator | None = None,
    ) -> None:
        self.registry = registry
        self.primitive_store = primitive_store
        self.negative_memory_store = negative_memory_store
        self.sandbox = sandbox or Sandbox()
        self.repair_generator = repair_generator

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.exists():
            return path
        if not path.is_absolute():
            repo_relative = PROJECT_ROOT / path
            if repo_relative.exists():
                return repo_relative
        return path

    def _load_skill_code(self, card: SkillCard) -> str:
        return self._resolve_path(card.code_path).read_text(encoding="utf-8")

    def _infer_humaneval_task_id(self, raw_value: object) -> str:
        match = re.search(r"HumanEval[/_](\d+)", str(raw_value or ""))
        if not match:
            return ""
        return f"HumanEval/{match.group(1)}"

    def _load_task_for_skill(self, card: SkillCard) -> BenchmarkTaskInstance | None:
        benchmark = str(card.preconditions.get("benchmark") or card.family or "").strip().lower()
        if benchmark != "humaneval":
            return None
        target_task_id = None
        for item in [card.skill_id, *card.lineage]:
            target_task_id = self._infer_humaneval_task_id(item)
            if target_task_id:
                break
        tasks = load_benchmark_tasks(benchmark="humaneval", profile="full", dataset_split="")
        if target_task_id is None:
            entry_point = card.signature.get("entry_point")
            for task in tasks:
                inferred = task.entry_point
                if not inferred and task.test_list:
                    text = "\n".join(task.test_list)
                    if entry_point and entry_point in text:
                        return task
                if inferred == entry_point:
                    return task
            return None
        for task in tasks:
            if task.task_id == target_task_id:
                return task
        return None

    def _build_audit_task(self, card: SkillCard) -> BenchmarkTaskInstance | None:
        task = self._load_task_for_skill(card)
        if task is None:
            return None
        return task.model_copy(deep=True)

    def _build_standard_replay_task(self, card: SkillCard) -> BenchmarkTaskInstance | None:
        task = self._load_task_for_skill(card)
        if task is None:
            return None
        return task.model_copy(deep=True)

    def _verify_card(self, card: SkillCard) -> VerifierFeedback | None:
        task = self._build_audit_task(card)
        if task is None:
            return None
        code = self._load_skill_code(card)
        return verify_task(task=task, candidate=code, sandbox=self.sandbox)

    def _record_negative_memory(
        self,
        *,
        artifact_id: str,
        artifact_kind: str,
        benchmark: str,
        lifecycle_phase: str = "",
        task_pattern: str,
        failure_type: str,
        source: str,
        summary: str,
        severity: float,
    ) -> None:
        if self.negative_memory_store is None:
            return
        self.negative_memory_store.add(
            artifact_id=artifact_id,
            artifact_kind=artifact_kind,
            benchmark=benchmark,
            lifecycle_phase=lifecycle_phase,
            task_pattern=task_pattern,
            planning_mode="audit",
            failure_type=failure_type,
            source=source,
            summary=summary,
            severity=severity,
        )

    def _demoted_seed_mode(self, current_mode: str) -> str:
        mode = str(current_mode or "").strip()
        if mode == "code_excerpt_seed":
            return "skeleton_only_seed"
        if mode == "skeleton_only_seed":
            return "guard_only_seed"
        return mode or "guard_only_seed"

    def _shadow_negative_memory_counts(self, skill_id: str, *, lifecycle_phase: str) -> dict[str, int]:
        if self.negative_memory_store is None or lifecycle_phase != "phase_b":
            return {}
        counts: dict[str, int] = {}
        for source in self._SHADOW_ACTION_NEGATIVE_SOURCES:
            count = self.negative_memory_store.count_for_artifact(
                skill_id,
                source=source,
                lifecycle_phase=lifecycle_phase,
            )
            if count > 0:
                counts[source] = count
        return counts

    def _apply_shadow_transfer_controls(
        self,
        *,
        card: SkillCard,
        lifecycle_phase: str = "",
    ) -> dict[str, object] | None:
        counts = self._shadow_negative_memory_counts(card.skill_id, lifecycle_phase=lifecycle_phase)
        if not counts:
            return None

        prune_support = int(counts.get("audit_failure", 0)) + int(counts.get("prune_reason", 0))
        suppress_support = (
            int(counts.get("runtime_failure", 0))
            + int(counts.get("trace_audit", 0))
            + int(counts.get("compile_rejection", 0))
        )
        if prune_support <= 0 and suppress_support <= 0:
            return None

        next_status = "pruned" if prune_support > 0 else "suppressed"
        updated = self.registry.set_status(card.skill_id, next_status) or card
        benchmark = str(card.preconditions.get("benchmark") or card.family or "").strip().lower()
        task_pattern = str(card.preconditions.get("task_pattern", "") or "")
        if next_status == "pruned":
            self._record_negative_memory(
                artifact_id=card.skill_id,
                artifact_kind="skill",
                benchmark=benchmark,
                lifecycle_phase=lifecycle_phase,
                task_pattern=task_pattern,
                failure_type="shadow_governance_prune",
                source="prune_reason",
                summary="phase-b governance pruned shadow skill after repeated negative memory support",
                severity=0.95,
            )
            reason = "shadow_negative_memory_pruned"
        else:
            self._record_negative_memory(
                artifact_id=card.skill_id,
                artifact_kind="skill",
                benchmark=benchmark,
                lifecycle_phase=lifecycle_phase,
                task_pattern=task_pattern,
                failure_type="shadow_governance_suppress",
                source="trace_audit",
                summary="phase-b governance suppressed shadow skill after negative memory support",
                severity=0.65,
            )
            reason = "shadow_negative_memory_suppressed"
        return {
            "artifact_id": card.skill_id,
            "artifact_kind": "skill",
            "audited": False,
            "passed": False,
            "status": updated.status,
            "repair_attempted": False,
            "repair_succeeded": False,
            "pruned": bool(updated.status == "pruned"),
            "toxic_risk": True,
            "reason": reason,
        }

    def _apply_transfer_controls(self, *, lifecycle_phase: str = "") -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        for card in self.registry.list_all():
            benchmark = str(card.preconditions.get("benchmark") or card.family or "").strip().lower()
            if benchmark != "humaneval":
                continue
            if str(card.status) == "shadow":
                shadow_result = self._apply_shadow_transfer_controls(card=card, lifecycle_phase=lifecycle_phase)
                if shadow_result is not None:
                    results.append(shadow_result)
                continue
            if str(card.status) != "active":
                continue
            positive = int(card.positive_transfer_count or 0)
            negative = int(card.negative_transfer_count or 0)
            utility = float(card.utility or 0.0)
            task_pattern = str(card.preconditions.get("task_pattern", "") or "")
            if negative >= 3 and positive <= 1 and utility < 0.35:
                updated = self.registry.set_status(card.skill_id, "pruned") or card
                self._record_negative_memory(
                    artifact_id=card.skill_id,
                    artifact_kind="skill",
                    benchmark=benchmark,
                    lifecycle_phase=lifecycle_phase,
                    task_pattern=task_pattern,
                    failure_type="governance_prune",
                    source="prune_reason",
                    summary="negative transfer kept growing with low utility; pruned by governance",
                    severity=0.9,
                )
                results.append(
                    {
                        "artifact_id": card.skill_id,
                        "artifact_kind": "skill",
                        "audited": False,
                        "passed": False,
                        "status": updated.status,
                        "repair_attempted": False,
                        "repair_succeeded": False,
                        "pruned": True,
                        "toxic_risk": True,
                        "reason": "governance_prune_low_utility_negative_transfer",
                    }
                )
                continue
            if negative >= 2 and positive == 0:
                updated = self.registry.set_status(card.skill_id, "suppressed") or card
                self._record_negative_memory(
                    artifact_id=card.skill_id,
                    artifact_kind="skill",
                    benchmark=benchmark,
                    lifecycle_phase=lifecycle_phase,
                    task_pattern=task_pattern,
                    failure_type="governance_suppress",
                    source="trace_audit",
                    summary="suppressed after repeated negative transfer with zero positive transfer",
                    severity=0.6,
                )
                results.append(
                    {
                        "artifact_id": card.skill_id,
                        "artifact_kind": "skill",
                        "audited": False,
                        "passed": False,
                        "status": updated.status,
                        "repair_attempted": False,
                        "repair_succeeded": False,
                        "pruned": False,
                        "toxic_risk": True,
                        "reason": "governance_suppress_zero_positive_transfer",
                    }
                )
                continue
            if positive > 0 and negative > positive:
                current_mode = str((card.compile_metadata or {}).get("governance_seed_mode_override") or card.last_used_routing_mode or "").strip()
                if not current_mode:
                    current_mode = "code_excerpt_seed"
                repaired_mode = self._demoted_seed_mode(current_mode)
                if repaired_mode != current_mode:
                    updated_card = card.model_copy(deep=True)
                    updated_card.compile_metadata = dict(updated_card.compile_metadata or {})
                    updated_card.compile_metadata["governance_seed_mode_override"] = repaired_mode
                    self.registry.upsert(updated_card)
                    results.append(
                        {
                            "artifact_id": card.skill_id,
                            "artifact_kind": "skill",
                            "audited": False,
                            "passed": True,
                            "status": updated_card.status,
                            "repair_attempted": True,
                            "repair_succeeded": True,
                            "pruned": False,
                            "toxic_risk": False,
                            "reason": f"governance_demote_seed_mode:{current_mode}->{repaired_mode}",
                        }
                    )
        return results

    def _attempt_skill_repair(
        self,
        *,
        card: SkillCard,
        task: BenchmarkTaskInstance,
        feedback: VerifierFeedback,
        standard_task: BenchmarkTaskInstance | None = None,
    ) -> dict[str, object]:
        if self.repair_generator is None:
            return {
                "attempted": False,
                "succeeded": False,
                "failure_reason": "missing_repair_generator",
                "feedback": None,
                "candidate_code": "",
                "standard_replay_attempted": False,
                "standard_replay_passed": False,
            }
        previous_code = self._load_skill_code(card)
        try:
            repair_output = self.repair_generator.repair(
                task=task,
                previous_code=previous_code,
                feedback=feedback,
                attempt_index=1,
                retrieved_skills=[],
                planning_mode="pure_dynamic",
                policy_directives=[],
                workflow_memories=[],
                template_priors=[],
                primitive_helpers=[],
            )
        except GenerationError as exc:
            return {
                "attempted": True,
                "succeeded": False,
                "failure_reason": str(exc),
                "feedback": None,
                "candidate_code": "",
                "standard_replay_attempted": False,
                "standard_replay_passed": False,
            }

        candidate_code = repair_output.code
        if not candidate_code.strip():
            return {
                "attempted": True,
                "succeeded": False,
                "failure_reason": "empty_repair_candidate",
                "feedback": None,
                "candidate_code": "",
                "standard_replay_attempted": False,
                "standard_replay_passed": False,
            }

        repaired_feedback = verify_task(task=task, candidate=candidate_code, sandbox=self.sandbox)
        if repaired_feedback.passed:
            standard_replay_attempted = standard_task is not None
            standard_replay_passed = True
            if standard_task is not None:
                standard_feedback = verify_task(task=standard_task, candidate=candidate_code, sandbox=self.sandbox)
                standard_replay_passed = bool(standard_feedback.passed)
                if not standard_replay_passed:
                    return {
                        "attempted": True,
                        "succeeded": False,
                        "failure_reason": (standard_feedback.stderr or standard_feedback.summary or "standard_regression_replay_failed")[:500],
                        "feedback": standard_feedback,
                        "candidate_code": candidate_code,
                        "standard_replay_attempted": standard_replay_attempted,
                        "standard_replay_passed": False,
                    }
            path = self.registry.save_code(card.skill_id, candidate_code)
            updated_card = self.registry.get(card.skill_id) or card.model_copy(deep=True)
            updated_card.code_path = str(path)
            self.registry.upsert(updated_card)
            return {
                "attempted": True,
                "succeeded": True,
                "failure_reason": "",
                "feedback": repaired_feedback,
                "candidate_code": candidate_code,
                "standard_replay_attempted": standard_replay_attempted,
                "standard_replay_passed": standard_replay_passed,
            }

        failure_reason = repaired_feedback.stderr or repaired_feedback.summary or "repair_regression_failed"
        return {
            "attempted": True,
            "succeeded": False,
            "failure_reason": failure_reason[:500],
            "feedback": repaired_feedback,
            "candidate_code": candidate_code,
            "standard_replay_attempted": False,
            "standard_replay_passed": False,
        }

    def _verify_primitive(self, card: PrimitiveCard) -> dict[str, object]:
        path = self._resolve_path(card.code_path)
        if not path.exists():
            return {"passed": False, "reason": "missing_primitive_code"}
        code = path.read_text(encoding="utf-8")
        program = "\n\n".join(
            [
                code.rstrip(),
                f"assert callable({card.helper_name})",
            ]
        )
        result = self.sandbox.run(program)
        return {
            "passed": bool(result["passed"]),
            "reason": (result["stderr"] or result["stdout"] or "").strip()[:500],
        }

    def audit_skill(self, skill_id: str, *, lifecycle_phase: str = "") -> dict:
        card = self.registry.get(skill_id)
        if card is None:
            return {
                "artifact_id": skill_id,
                "artifact_kind": "skill",
                "audited": False,
                "reason": "missing_skill",
                "repair_attempted": False,
                "repair_succeeded": False,
                "pruned": False,
                "toxic_risk": False,
            }
        feedback = self._verify_card(card)
        if feedback is None:
            updated = self.registry.update_audit_result(skill_id=skill_id, passed=False)
            self.registry.append_failure(
                skill_id,
                task_id="audit",
                benchmark=card.family,
                planning_mode="audit",
                verifier_message="audit task could not be reconstructed",
                failure_type="audit_setup_failure",
            )
            self._record_negative_memory(
                artifact_id=skill_id,
                artifact_kind="skill",
                benchmark=card.family,
                lifecycle_phase=lifecycle_phase,
                task_pattern=card.preconditions.get("task_pattern", ""),
                failure_type="audit_setup_failure",
                source="audit_failure",
                summary="audit task could not be reconstructed",
                severity=0.8,
            )
            return {
                "artifact_id": skill_id,
                "artifact_kind": "skill",
                "audited": True,
                "passed": False,
                "status": updated.status if updated else card.status,
                "reason": "task_reconstruction_failed",
                "repair_attempted": False,
                "repair_succeeded": False,
                "repair_failure_reason": "task_reconstruction_failed",
                "pruned": bool((updated.status if updated else card.status) == "pruned"),
                "toxic_risk": True,
            }

        passed = feedback.passed
        repair_attempted = False
        repair_succeeded = False
        repair_failure_reason = ""
        standard_replay_attempted = False
        standard_replay_passed = False
        if passed:
            updated = self.registry.update_audit_result(skill_id=skill_id, passed=True)
            latest = self.registry.get(skill_id)
        else:
            task = self._build_audit_task(card)
            standard_task = self._build_standard_replay_task(card)
            repair_result = {
                "attempted": False,
                "succeeded": False,
                "failure_reason": "",
                "feedback": None,
                "standard_replay_attempted": False,
                "standard_replay_passed": False,
            }
            if task is not None:
                repair_result = self._attempt_skill_repair(card=card, task=task, feedback=feedback, standard_task=standard_task)
            repair_attempted = bool(repair_result["attempted"])
            repair_succeeded = bool(repair_result["succeeded"])
            repair_failure_reason = str(repair_result["failure_reason"] or "")
            standard_replay_attempted = bool(repair_result.get("standard_replay_attempted"))
            standard_replay_passed = bool(repair_result.get("standard_replay_passed"))
            self.registry.append_failure(
                skill_id,
                task_id=feedback.task_id,
                benchmark=feedback.benchmark,
                planning_mode="audit",
                verifier_message=feedback.stderr or feedback.summary,
                failure_type="audit_failure_repaired" if repair_succeeded else "audit_failure",
            )
            if repair_succeeded:
                updated = self.registry.update_audit_result(skill_id=skill_id, passed=True)
                latest = self.registry.get(skill_id)
                if latest is not None:
                    latest = self.registry.set_status(skill_id, "active")
                if self.negative_memory_store is not None:
                    self.negative_memory_store.clear_for_artifact(skill_id, source="audit_failure")
                    self.negative_memory_store.clear_for_artifact(skill_id, source="prune_reason")
            else:
                updated = self.registry.update_audit_result(skill_id=skill_id, passed=False)
                latest = self.registry.get(skill_id)
                if latest is not None:
                    forced_status = "pruned" if latest.audit_fail_count >= self.registry.prune_audit_fail_threshold else "suppressed"
                    latest = self.registry.set_status(skill_id, forced_status)
                self._record_negative_memory(
                    artifact_id=skill_id,
                    artifact_kind="skill",
                    benchmark=card.family,
                    lifecycle_phase=lifecycle_phase,
                    task_pattern=card.preconditions.get("task_pattern", ""),
                    failure_type="audit_failure",
                    source="audit_failure",
                    summary=feedback.stderr or feedback.summary or "audit failure",
                    severity=0.8,
                )
                if latest is not None and latest.status == "pruned":
                    self._record_negative_memory(
                        artifact_id=skill_id,
                        artifact_kind="skill",
                        benchmark=card.family,
                        lifecycle_phase=lifecycle_phase,
                        task_pattern=card.preconditions.get("task_pattern", ""),
                        failure_type="audit_prune",
                        source="prune_reason",
                        summary=repair_failure_reason or feedback.stderr or feedback.summary or "artifact pruned after repeated audit failures",
                        severity=0.95,
                    )
        final_status = latest.status if latest else (updated.status if updated else card.status)
        return {
            "artifact_id": skill_id,
            "artifact_kind": "skill",
            "audited": True,
            "passed": bool(passed or repair_succeeded),
            "status": final_status,
            "returncode": feedback.returncode,
            "failure_type": "" if (passed or repair_succeeded) else "audit_failure",
            "audit_test_case_count": len(getattr(self._build_audit_task(card), "test_list", []) or []),
            "failure_count": 0 if (passed or repair_succeeded) else 1,
            "repair_attempted": repair_attempted,
            "repair_succeeded": repair_succeeded,
            "repair_failure_reason": repair_failure_reason,
            "standard_replay_attempted": standard_replay_attempted,
            "standard_replay_passed": standard_replay_passed,
            "pruned": bool(final_status == "pruned"),
            "toxic_risk": bool(not (passed or repair_succeeded)),
        }

    def audit_primitive(self, primitive_id: str, *, lifecycle_phase: str = "") -> dict[str, object]:
        if self.primitive_store is None:
            return {
                "artifact_id": primitive_id,
                "artifact_kind": "primitive",
                "audited": False,
                "reason": "missing_primitive_store",
                "repair_attempted": False,
                "repair_succeeded": False,
                "pruned": False,
                "toxic_risk": False,
            }
        card = self.primitive_store.get(primitive_id)
        if card is None:
            return {
                "artifact_id": primitive_id,
                "artifact_kind": "primitive",
                "audited": False,
                "reason": "missing_primitive",
                "repair_attempted": False,
                "repair_succeeded": False,
                "pruned": False,
                "toxic_risk": False,
            }
        result = self._verify_primitive(card)
        updates = self.primitive_store.record_feedback([primitive_id], success=bool(result["passed"]))
        latest = updates[0] if updates else (self.primitive_store.get(primitive_id) or card)
        if not result["passed"] and self.negative_memory_store is not None:
            self.negative_memory_store.add(
                artifact_id=primitive_id,
                artifact_kind="primitive",
                benchmark=card.benchmark,
                lifecycle_phase=lifecycle_phase,
                task_pattern=card.task_pattern,
                planning_mode="audit",
                failure_type="audit_failure",
                source="audit_failure",
                summary=str(result["reason"] or "primitive audit failure"),
                severity=0.7,
            )
        return {
            "artifact_id": primitive_id,
            "artifact_kind": "primitive",
            "audited": True,
            "passed": bool(result["passed"]),
            "status": latest.status,
            "failure_type": "" if result["passed"] else "audit_failure",
            "audit_test_case_count": 0,
            "failure_count": 0 if result["passed"] else 1,
            "repair_attempted": False,
            "repair_succeeded": False,
            "repair_failure_reason": "",
            "pruned": False,
            "toxic_risk": bool(not result["passed"]),
        }

    def periodic_audit(self, top_k: int = 3, *, lifecycle_phase: str = "") -> list[dict]:
        results = self._apply_transfer_controls(lifecycle_phase=lifecycle_phase)
        results.extend(self.audit_skill(card.skill_id, lifecycle_phase=lifecycle_phase) for card in self.registry.top_for_audit(top_k=top_k))
        if self.primitive_store is not None:
            results.extend(
                self.audit_primitive(card.primitive_id, lifecycle_phase=lifecycle_phase)
                for card in self.primitive_store.top_for_audit(top_k=top_k)
            )
        return results
