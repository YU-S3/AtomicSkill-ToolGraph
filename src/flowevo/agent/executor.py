"""Executors for MBPP-first workflows."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from agent.generator import BaseGenerator, GenerationError, HeuristicGenerator, OracleGenerator
from agent.skill_compatibility import task_compatibility_context
from core.schemas import (
    ActionRecord,
    AttemptRecord,
    BenchmarkTaskInstance,
    ExecutionTrace,
    GenerationEvent,
    VerifierFeedback,
    WorkflowPlan,
)
from core.utils import infer_task_entry_point
from env.sandbox import Sandbox
from eval.verifier import verify_task

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CodeTaskExecutor:
    """Execute oracle or dynamic workflows on code tasks."""

    def __init__(
        self,
        sandbox: Sandbox,
        max_repairs: int = 2,
        dynamic_generator: Optional[BaseGenerator] = None,
        oracle_generator: Optional[BaseGenerator] = None,
        exact_direct_postcall_wrapper_enabled: bool = False,
        direct_postcall_max_local_patch: int = 0,
    ) -> None:
        self.sandbox = sandbox
        self.max_repairs = max_repairs
        self.dynamic_generator = dynamic_generator or HeuristicGenerator()
        self.oracle_generator = oracle_generator or OracleGenerator()
        self.exact_direct_postcall_wrapper_enabled = bool(exact_direct_postcall_wrapper_enabled)
        self.direct_postcall_max_local_patch = int(direct_postcall_max_local_patch or 0)

    def _record_action(self, actions: list[ActionRecord], tool_name: str, tool_input_summary: str, tool_output_summary: str) -> None:
        actions.append(
            ActionRecord(
                tool_name=tool_name,
                tool_input_summary=tool_input_summary,
                tool_output_summary=tool_output_summary,
            )
        )

    def _invoke_generator(self, generator: BaseGenerator, method_name: str, **kwargs: Any) -> GenerationOutput:
        method = getattr(generator, method_name)
        signature = inspect.signature(method)
        accepted = {
            key: value
            for key, value in kwargs.items()
            if key in signature.parameters
        }
        return method(**accepted)

    def _classify_failure(self, feedback: VerifierFeedback) -> str:
        if feedback.summary == "generation_failed":
            return "generation_error"
        if feedback.timeout:
            return "timeout"
        stderr = feedback.stderr or ""
        if "AssertionError" in stderr:
            return "test_failure"
        if feedback.returncode != 0:
            return "runtime_error"
        if not feedback.passed:
            return "test_failure"
        return ""

    def _append_attempt(
        self,
        attempts: list[AttemptRecord],
        attempt_index: int,
        parent_attempt: Optional[int],
        stage: str,
        candidate_code: str,
        feedback: VerifierFeedback | None,
        failure_type: str,
        repair_source: str,
        notes: str,
    ) -> None:
        attempts.append(
            AttemptRecord(
                attempt_index=attempt_index,
                parent_attempt=parent_attempt,
                stage=stage,
                candidate_code=candidate_code,
                verifier_feedback=feedback,
                failure_type=failure_type,
                repair_source=repair_source,
                notes=notes,
            )
        )

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.exists():
            return path
        if not path.is_absolute():
            repo_relative = PROJECT_ROOT / path
            if repo_relative.exists():
                return repo_relative
        return path

    def _load_skill_code(self, code_path: str) -> str:
        return self._resolve_path(code_path).read_text(encoding="utf-8")

    def _import_skill_module(self, code_path: str, entry_point: str | None) -> None:
        resolved_path = self._resolve_path(code_path)
        module_name = f"retrieved_skill_{uuid.uuid4().hex[:8]}"
        spec = importlib.util.spec_from_file_location(module_name, resolved_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to build import spec for skill: {code_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if entry_point and not hasattr(module, entry_point):
            raise AttributeError(f"Skill module missing expected entry point: {entry_point}")

    def _adapt_skill_candidate_for_task(self, candidate: str, *, skill_entry_point: str, task_entry_point: str | None) -> str:
        if not task_entry_point or not skill_entry_point or task_entry_point == skill_entry_point:
            return candidate
        return (
            candidate.rstrip()
            + "\n\n"
            + f"def {task_entry_point}(*args, **kwargs):\n"
            + f"    return {skill_entry_point}(*args, **kwargs)\n"
        )

    def _make_skill_runtime_failure(self, task: BenchmarkTaskInstance, candidate_code: str, error: Exception) -> VerifierFeedback:
        return VerifierFeedback(
            benchmark=task.benchmark,
            task_id=task.task_id,
            dataset_split="",
            passed=False,
            stdout="",
            stderr=f"Skill runtime load failed: {type(error).__name__}: {error}",
            returncode=1,
            timeout=False,
            executed_code=candidate_code,
            failed_tests=["Retrieved skill failed before verifier execution."],
            summary="skill_load_failed",
        )

    def _make_generation_failure(self, task: BenchmarkTaskInstance, stage: str, error: str) -> VerifierFeedback:
        return VerifierFeedback(
            benchmark=task.benchmark,
            task_id=task.task_id,
            dataset_split="",
            passed=False,
            stdout="",
            stderr=f"Generation failed during {stage}: {error}",
            returncode=1,
            timeout=False,
            executed_code="",
            failed_tests=["LLM-backed generation failed before verifier execution."],
            summary="generation_failed",
        )

    def _selected_skill_row(self, plan: WorkflowPlan, retrieval_scores: list[dict[str, Any]]) -> dict[str, Any]:
        selected_skill_id = str(plan.selected_skill_id or "").strip()
        if not selected_skill_id:
            return {}
        return next(
            (item for item in retrieval_scores if str(item.get("skill_id", "")) == selected_skill_id),
            {},
        )

    def _compatibility_like_failure(
        self,
        *,
        task: BenchmarkTaskInstance,
        plan: WorkflowPlan,
        feedback: VerifierFeedback,
        retrieval_scores: list[dict[str, Any]],
    ) -> bool:
        if not bool(getattr(plan, "soft_aware_enabled", False)):
            return False
        if feedback.passed:
            return False
        selected_row = self._selected_skill_row(plan, retrieval_scores)
        selected_entry_point = str(selected_row.get("entry_point", "") or (selected_row.get("signature") or {}).get("entry_point") or "")
        task_entry_point = str(getattr(task, "entry_point", "") or "")
        stderr_text = str(feedback.stderr or "")
        lowered = stderr_text.lower()
        context = task_compatibility_context(task, runtime_error_text=stderr_text)
        if selected_entry_point and task_entry_point and selected_entry_point != task_entry_point:
            return True
        if "file_path" in set(selected_row.get("environment_affordance") or []) and not context["has_file_path_evidence"]:
            return True
        if any(token in lowered for token in ("syntaxerror", "indentationerror", "typeerror", "nameerror", "attributeerror", "importerror")):
            return True
        if str(selected_row.get("role", "")) == "guard" and not context["has_guard_evidence"]:
            return True
        return False

    def _normalize_direct_skill_candidate(
        self,
        candidate: str,
        *,
        task: BenchmarkTaskInstance,
        skill_entry_point: str,
    ) -> str:
        normalized = self._adapt_skill_candidate_for_task(
            candidate.rstrip() + "\n",
            skill_entry_point=skill_entry_point,
            task_entry_point=infer_task_entry_point(task),
        )
        return normalized.rstrip() + "\n"

    def _build_direct_trace(
        self,
        *,
        task: BenchmarkTaskInstance,
        mode: str,
        plan: WorkflowPlan,
        retrieval_scores: list[dict[str, Any]],
        utility_before: float | None,
        actions: list[ActionRecord],
        attempts: list[AttemptRecord],
        generation_events: list[GenerationEvent],
        code_history: list[str],
        verifier_feedback: list[VerifierFeedback],
        execution_outputs: list[dict[str, Any]],
        feedback: VerifierFeedback,
        failure_type: str,
        used_skills: list[str],
        skill_hit: bool,
        fallback_triggered: bool,
        start_time: float,
        artifact_import_latency_ms: float,
        artifact_verification_latency_ms: float,
        injected_workflow_memory_ids: list[str],
    ) -> ExecutionTrace:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        totals = self._llm_totals(generation_events)
        return ExecutionTrace(
            trace_id=f"trace_{uuid.uuid4().hex[:8]}",
            task_id=task.task_id,
            family=task.benchmark,
            query=self._task_query(task),
            benchmark=task.benchmark,
            retrieved_skills=plan.retrieved_skills,
            retrieval_scores=retrieval_scores,
            planning_mode=plan.planning_mode,
            task_pattern=plan.task_pattern,
            selected_skill_id=plan.selected_skill_id,
            seed_routing_mode=plan.seed_routing_mode,
            transfer_utility_score=plan.transfer_utility_score,
            estimated_seed_token_cost=plan.estimated_seed_token_cost,
            historical_positive_transfer=plan.historical_positive_transfer,
            negative_transfer_risk=plan.negative_transfer_risk,
            skill_seeded=plan.skill_seeded,
            seed_source=plan.seed_source,
            selection_reason=plan.selection_reason,
            utility_before=utility_before,
            utility_after=None,
            plan=plan,
            actions=actions,
            attempts=attempts,
            generation_events=generation_events,
            code_history=code_history,
            verifier_feedback=verifier_feedback,
            final_result=self._final_result_payload(
                mode=mode,
                feedback=feedback,
                failure_type=failure_type,
                skill_hit=skill_hit,
                fallback_triggered=fallback_triggered,
                task=task,
                attempts=attempts,
            ),
            execution_outputs=execution_outputs,
            success=bool(feedback.passed),
            token_cost=float(totals["total_tokens"]),
            latency_ms=latency_ms,
            artifact_execution_latency_ms=artifact_import_latency_ms + artifact_verification_latency_ms,
            artifact_import_latency_ms=artifact_import_latency_ms,
            artifact_verification_latency_ms=artifact_verification_latency_ms,
            llm_prompt_tokens_total=int(totals["prompt_tokens"]),
            llm_completion_tokens_total=int(totals["completion_tokens"]),
            llm_total_tokens_total=int(totals["total_tokens"]),
            llm_latency_ms_total=float(totals["latency_ms"]),
            llm_call_count=int(totals["call_count"]),
            policy_version=plan.policy_version,
            matched_workflow_memory_ids=plan.matched_workflow_memory_ids,
            injected_workflow_memory_ids=injected_workflow_memory_ids,
            matched_template_ids=plan.matched_template_ids,
            matched_primitive_ids=plan.matched_primitive_ids,
            draft_injected_template_ids=[],
            repair_injected_template_ids=[],
            injected_primitive_ids=[],
            failure_type=failure_type,
            used_skills=used_skills,
            skill_hit=skill_hit,
            fallback_triggered=fallback_triggered,
            **self._operational_metrics(actions, attempts, generation_events, plan.selected_skill_id, injected_workflow_memory_ids),
            **self._structured_trace_fields(
                plan=plan,
                generation_events=generation_events,
                candidate_code=code_history[-1] if code_history else "",
                task_entry_point=infer_task_entry_point(task) or "",
            ),
        )

    def _run_direct_postcall_wrapper(
        self,
        *,
        task: BenchmarkTaskInstance,
        plan: WorkflowPlan,
        skill_step_payload: dict[str, Any],
        feedback: VerifierFeedback,
        retrieval_scores: list[dict[str, Any]],
        actions: list[ActionRecord],
        attempts: list[AttemptRecord],
        code_history: list[str],
        verifier_feedback: list[VerifierFeedback],
        execution_outputs: list[dict[str, Any]],
        generation_events: list[GenerationEvent],
        generator: BaseGenerator,
    ) -> VerifierFeedback:
        if not code_history:
            return feedback
        skill_entry_point = str(skill_step_payload.get("entry_point", "") or infer_task_entry_point(task) or "")
        normalized_candidate = self._normalize_direct_skill_candidate(
            code_history[-1],
            task=task,
            skill_entry_point=skill_entry_point,
        )
        if normalized_candidate != code_history[-1]:
            code_history.append(normalized_candidate)
            self._record_action(
                actions,
                tool_name="postcall_normalize",
                tool_input_summary=f"task_id={task.task_id}, skill_id={plan.selected_skill_id}",
                tool_output_summary="normalized direct skill output and rechecked interface",
            )
            feedback = verify_task(task=task, candidate=normalized_candidate, sandbox=self.sandbox)
            verifier_feedback.append(feedback)
            execution_outputs.append(feedback.model_dump(mode="json"))
            self._append_attempt(
                attempts=attempts,
                attempt_index=len(attempts),
                parent_attempt=0,
                stage="direct_postcall_normalize",
                candidate_code=normalized_candidate,
                feedback=feedback,
                failure_type=self._classify_failure(feedback),
                repair_source="postcall_wrapper",
                notes="post-call normalization and interface check",
            )
            self._record_action(
                actions,
                tool_name="run_tests",
                tool_input_summary=f"attempt_index={len(attempts)-1}",
                tool_output_summary=f"passed={feedback.passed}, returncode={feedback.returncode}, timeout={feedback.timeout}",
            )
            if feedback.passed:
                return feedback
        patch_budget = max(int(plan.direct_postcall_max_local_patch or self.direct_postcall_max_local_patch or 0), 0)
        if patch_budget <= 0:
            return feedback
        selected_row = self._selected_skill_row(plan, retrieval_scores)
        selected_row = {**selected_row, "seed_routing_mode": "guard_only_seed"}
        for patch_index in range(patch_budget):
            try:
                repair_output = self._invoke_generator(
                    generator,
                    "repair",
                    task=task,
                    previous_code=code_history[-1],
                    feedback=feedback,
                    attempt_index=patch_index + 1,
                    retrieved_skills=[selected_row] if selected_row else [],
                    planning_mode="skill_seeded_dynamic",
                    seed_context_skill_modes={str(plan.selected_skill_id or ""): "guard_only_seed"} if plan.selected_skill_id else {},
                    seed_routing_mode="guard_only_seed",
                    transfer_utility_score=plan.transfer_utility_score,
                    estimated_seed_token_cost=plan.estimated_seed_token_cost,
                    historical_positive_transfer=plan.historical_positive_transfer,
                    negative_transfer_risk=plan.negative_transfer_risk,
                    policy_directives=[
                        *list(plan.policy_directives or []),
                        "postcall_wrapper: normalize interface/output only",
                        "postcall_wrapper: preserve the retrieved skill body and make the smallest viable patch",
                    ],
                    workflow_memories=[],
                    template_priors=[],
                    primitive_helpers=[],
                )
                candidate = repair_output.code
                if repair_output.event is not None:
                    generation_events.append(repair_output.event)
                    execution_outputs.append(repair_output.event.model_dump(mode="json"))
            except GenerationError as exc:
                generation_events.append(exc.event)
                execution_outputs.append(exc.event.model_dump(mode="json"))
                feedback = self._make_generation_failure(task=task, stage="direct_postcall_patch", error=str(exc))
                verifier_feedback.append(feedback)
                self._append_attempt(
                    attempts=attempts,
                    attempt_index=len(attempts),
                    parent_attempt=0,
                    stage="direct_postcall_patch",
                    candidate_code="",
                    feedback=feedback,
                    failure_type=self._classify_failure(feedback),
                    repair_source="postcall_wrapper",
                    notes="post-call local patch generation failed",
                )
                return feedback
            candidate, _payload = self._extract_skill_plan_header(candidate)
            code_history.append(candidate)
            self._record_action(
                actions,
                tool_name="postcall_local_patch",
                tool_input_summary=f"task_id={task.task_id}, skill_id={plan.selected_skill_id}",
                tool_output_summary="generated one minimal local patch after direct skill failure",
            )
            feedback = verify_task(task=task, candidate=candidate, sandbox=self.sandbox)
            verifier_feedback.append(feedback)
            execution_outputs.append(feedback.model_dump(mode="json"))
            self._append_attempt(
                attempts=attempts,
                attempt_index=len(attempts),
                parent_attempt=0,
                stage="direct_postcall_patch",
                candidate_code=candidate,
                feedback=feedback,
                failure_type=self._classify_failure(feedback),
                repair_source="postcall_wrapper",
                notes="post-call minimal local patch",
            )
            self._record_action(
                actions,
                tool_name="run_tests",
                tool_input_summary=f"attempt_index={len(attempts)-1}",
                tool_output_summary=f"passed={feedback.passed}, returncode={feedback.returncode}, timeout={feedback.timeout}",
            )
            break
        return feedback

    def _llm_totals(self, generation_events: list[GenerationEvent]) -> dict[str, float]:
        return {
            "prompt_tokens": sum(event.prompt_tokens for event in generation_events),
            "completion_tokens": sum(event.completion_tokens for event in generation_events),
            "total_tokens": sum(event.total_tokens for event in generation_events),
            "latency_ms": sum(event.latency_ms for event in generation_events),
            "call_count": len(generation_events),
        }

    def _extract_skill_plan_header(self, code: str) -> tuple[str, dict[str, Any]]:
        lines = code.splitlines()
        if not lines or lines[0].strip() != "# SKILL_PLAN":
            return code, {}
        payload_lines: list[str] = []
        end_index = -1
        for index, line in enumerate(lines[1:], start=1):
            if line.strip() == "# END_SKILL_PLAN":
                end_index = index
                break
            payload_lines.append(line[2:] if line.startswith("# ") else line.lstrip("#"))
        if end_index < 0:
            return code, {}
        try:
            payload = json.loads("\n".join(payload_lines).strip() or "{}")
        except json.JSONDecodeError:
            return code, {}
        stripped_code = "\n".join(lines[end_index + 1 :]).lstrip()
        if stripped_code and not stripped_code.endswith("\n"):
            stripped_code += "\n"
        return stripped_code, payload

    def _extract_realized_skill_nodes(
        self,
        candidate_code: str,
        *,
        task_entry_point: str,
        planned_nodes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not planned_nodes:
            return []
        try:
            tree = ast.parse(candidate_code)
        except SyntaxError:
            return []
        function_node = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and (not task_entry_point or node.name == task_entry_point)
            ),
            None,
        )
        if function_node is None:
            return []
        realized_nodes: list[dict[str, Any]] = []
        target_names = {
            str(item.get("name", "")).strip()
            for item in planned_nodes
            if str(item.get("kind", "")) == "helper_call" and str(item.get("name", "")).strip()
        }
        if target_names:
            realized_names: list[str] = []
            for node in ast.walk(function_node):
                if not isinstance(node, ast.Call):
                    continue
                call_name = ""
                if isinstance(node.func, ast.Name):
                    call_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    call_name = node.func.attr
                if call_name in target_names and call_name not in realized_names:
                    realized_names.append(call_name)
            realized_nodes.extend({"kind": "helper_call", "name": name} for name in realized_names)
        if any(str(item.get("kind", "")) == "guard_check" for item in planned_nodes) and self._guard_check_realized(function_node):
            realized_nodes.append({"kind": "guard_check", "name": task_entry_point or "guard_check"})
        return realized_nodes

    def _guard_check_realized(self, function_node: ast.FunctionDef) -> bool:
        param_names = {
            arg.arg
            for arg in (*function_node.args.posonlyargs, *function_node.args.args, *function_node.args.kwonlyargs)
        }

        def references_param(node: ast.AST) -> bool:
            return any(isinstance(item, ast.Name) and item.id in param_names for item in ast.walk(node))

        def guard_expr(node: ast.AST) -> bool:
            if isinstance(node, ast.BoolOp):
                return bool(node.values) and all(guard_expr(value) for value in node.values)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                return guard_expr(node.operand)
            if isinstance(node, ast.Compare):
                return references_param(node)
            if isinstance(node, ast.Call):
                call_name = ""
                if isinstance(node.func, ast.Name):
                    call_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    call_name = node.func.attr
                return call_name in {"isinstance", "len"} and references_param(node)
            return False

        leading_statements = function_node.body[:3]
        for stmt in leading_statements:
            if isinstance(stmt, (ast.Assert, ast.Raise)):
                return True
            if isinstance(stmt, ast.Return) and guard_expr(stmt.value):
                return True
            if isinstance(stmt, ast.If) and guard_expr(stmt.test):
                if any(
                    isinstance(node, (ast.Assert, ast.Raise, ast.Return, ast.Assign, ast.AnnAssign))
                    for node in ast.walk(stmt)
                ):
                    return True
        return False

    def _inline_scaffold_realized(
        self,
        candidate_code: str,
        *,
        task_entry_point: str,
    ) -> bool:
        if not task_entry_point:
            return False
        try:
            tree = ast.parse(candidate_code)
        except SyntaxError:
            return False
        return any(
            isinstance(node, ast.FunctionDef) and node.name == task_entry_point
            for node in tree.body
        )

    def _structured_trace_fields(
        self,
        *,
        plan: WorkflowPlan,
        generation_events: list[GenerationEvent],
        candidate_code: str,
        task_entry_point: str,
    ) -> dict[str, Any]:
        latest_event = generation_events[-1] if generation_events else None
        planned_nodes = list((latest_event.planned_skill_nodes if latest_event is not None else []) or [])
        planned_count = int(latest_event.planned_skill_node_count if latest_event is not None else len(planned_nodes))
        realized_nodes = self._extract_realized_skill_nodes(
            candidate_code,
            task_entry_point=task_entry_point,
            planned_nodes=planned_nodes,
        )
        latest_role = str(latest_event.source_structural_role if latest_event is not None else "").strip()
        if (
            not realized_nodes
            and planned_count
            and plan.planning_mode == "skill_seeded_dynamic"
            and plan.planned_usage_subtype == "plan_skeleton"
            and latest_role == "local_scaffold"
            and self._inline_scaffold_realized(candidate_code, task_entry_point=task_entry_point)
        ):
            realized_nodes = [
                {
                    "kind": "inline_scaffold",
                    "name": task_entry_point or str(plan.selected_skill_id or "inline_scaffold"),
                }
            ]
        realized_count = len(realized_nodes)
        gap_count = max(planned_count - realized_count, 0)
        evidence_source = ""
        realized_usage_subtype = ""
        gap_reason = ""
        if planned_count and realized_count:
            evidence_source = "structured"
            realized_usage_subtype = plan.planned_usage_subtype
            gap_reason = "partial_structured_realization" if gap_count else ""
        elif plan.planning_mode == "skill_seeded_dynamic" and latest_event is not None and (
            latest_event.scaffold_hint_injected
            or latest_event.guard_checker_emitted
            or bool(latest_event.conditioned_skill_ids)
        ):
            evidence_source = "heuristic"
            realized_usage_subtype = plan.planned_usage_subtype
            gap_reason = "planned_nodes_not_realized" if planned_count else ""
        return {
            "planned_usage_subtype": plan.planned_usage_subtype,
            "realized_usage_subtype": realized_usage_subtype,
            "realization_evidence_source": evidence_source,
            "subtype_realization_gap_reason": gap_reason,
            "planned_skill_nodes": planned_nodes,
            "realized_skill_nodes": realized_nodes,
            "planned_skill_node_count": planned_count,
            "realized_skill_node_count": realized_count,
            "structured_realization_rate": round(realized_count / planned_count, 4) if planned_count else 0.0,
            "planned_realized_gap_count": gap_count,
            "planned_realized_gap_rate": round(gap_count / planned_count, 4) if planned_count else 0.0,
            "skill_node_insertion_count": realized_count,
        }

    def _template_sort_key(self, template: dict[str, Any]) -> tuple[int, int, str]:
        return (
            -int(template.get("priority", 0)),
            -int(template.get("success_count", 0)),
            str(template.get("template_id", "")),
        )

    def _operational_metrics(
        self,
        actions: list[ActionRecord],
        attempts: list[AttemptRecord],
        generation_events: list[GenerationEvent],
        selected_skill_id: str,
        injected_workflow_memory_ids: list[str],
    ) -> dict[str, Any]:
        operational_tools = {"call_skill", "draft_code", "run_tests", "inspect_failure", "repair_code", "finalize"}
        operational_event_count = sum(1 for action in actions if action.tool_name in operational_tools)
        skill_mediated_event_count = sum(1 for action in actions if action.tool_name == "call_skill")
        skill_mediated_event_count += sum(
            1 for attempt in attempts if attempt.stage in {"draft_from_skill", "repair_with_skill"}
        )
        first_attempt_success = bool(attempts and attempts[0].verifier_feedback and attempts[0].verifier_feedback.passed)
        # Count every attempt after the first one as a retry so direct-skill misses
        # followed by fallback dynamic generation are measured fairly.
        retry_count = max(len(attempts) - 1, 0)
        draft_prompt_tokens = 0
        repair_prompt_tokens = 0
        task_context_tokens = 0
        tests_context_tokens = 0
        feedback_context_tokens = 0
        previous_code_context_tokens = 0
        seed_summary_tokens = 0
        seed_context_tokens = 0
        scaffold_tokens = 0
        seed_context_skill_count = 0
        seed_context_mode = ""
        seed_context_char_count = 0
        seed_context_contains_code = False
        seed_context_contains_tests = False
        repair_round_count = 0
        full_rewrite_repair_count = 0
        local_patch_repair_count = 0
        compact_seeded_prompt_used = False
        for event in generation_events:
            if event.generation_stage == "draft":
                draft_prompt_tokens = int(event.prompt_tokens)
            elif event.generation_stage.startswith("repair_"):
                repair_prompt_tokens += int(event.prompt_tokens)
                repair_round_count += 1
                if event.repair_strategy == "local_patch":
                    local_patch_repair_count += 1
                else:
                    full_rewrite_repair_count += 1
            task_context_tokens += int(event.task_context_tokens or 0)
            tests_context_tokens += int(event.tests_context_tokens or 0)
            feedback_context_tokens += int(event.feedback_context_tokens or 0)
            previous_code_context_tokens += int(event.previous_code_context_tokens or 0)
            seed_summary_tokens += int(event.seed_summary_tokens or 0)
            seed_context_tokens += int(event.seed_context_tokens or 0)
            scaffold_tokens += int(event.scaffold_tokens or 0)
            compact_seeded_prompt_used = compact_seeded_prompt_used or bool(event.compact_seeded_prompt_used)
            seed_context_skill_count = max(seed_context_skill_count, int(event.seed_context_skill_count or 0))
            seed_context_char_count += int(event.seed_context_char_count or 0)
            seed_context_contains_code = seed_context_contains_code or bool(event.seed_context_contains_code)
            seed_context_contains_tests = seed_context_contains_tests or bool(event.seed_context_contains_tests)
            if not seed_context_mode and str(event.seed_context_mode or "").strip():
                seed_context_mode = str(event.seed_context_mode or "").strip()
        artifact_mediated = bool(selected_skill_id or injected_workflow_memory_ids)
        num_repairs = sum(1 for attempt in attempts if attempt.stage in {"repair", "repair_with_skill"})
        repair_failure_types: list[str] = []
        for current, nxt in zip(attempts, attempts[1:]):
            if nxt.stage not in {"repair", "repair_with_skill"}:
                continue
            failure_type = str(current.failure_type or "").strip()
            if failure_type and failure_type not in repair_failure_types:
                repair_failure_types.append(failure_type)
        return {
            "first_attempt_success": first_attempt_success,
            "retry_count": retry_count,
            "operational_event_count": operational_event_count,
            "skill_mediated_event_count": skill_mediated_event_count,
            "skill_mediated_event_rate": round(skill_mediated_event_count / operational_event_count, 4) if operational_event_count else 0.0,
            "draft_prompt_tokens": draft_prompt_tokens,
            "task_context_tokens": task_context_tokens,
            "tests_context_tokens": tests_context_tokens,
            "feedback_context_tokens": feedback_context_tokens,
            "previous_code_context_tokens": previous_code_context_tokens,
            "seed_summary_tokens": seed_summary_tokens,
            "seed_context_tokens": seed_context_tokens,
            "scaffold_tokens": scaffold_tokens,
            "seed_context_skill_count": seed_context_skill_count,
            "seed_context_mode": seed_context_mode,
            "seed_context_char_count": seed_context_char_count,
            "seed_context_contains_code": seed_context_contains_code,
            "seed_context_contains_tests": seed_context_contains_tests,
            "repair_prompt_tokens": repair_prompt_tokens,
            "num_repairs": num_repairs,
            "repair_round_count": repair_round_count,
            "full_rewrite_repair_count": full_rewrite_repair_count,
            "local_patch_repair_count": local_patch_repair_count,
            "repair_failure_types": repair_failure_types,
            "repair_heavy": bool(repair_prompt_tokens > draft_prompt_tokens),
            "compact_seeded_prompt_used": compact_seeded_prompt_used,
            "artifact_mediated": artifact_mediated,
        }

    def _task_query(self, task: BenchmarkTaskInstance) -> str:
        return task.prompt or task.text

    def _final_result_payload(
        self,
        *,
        mode: str,
        feedback: VerifierFeedback,
        failure_type: str,
        skill_hit: bool,
        fallback_triggered: bool,
        task: BenchmarkTaskInstance,
        attempts: list[AttemptRecord],
    ) -> dict[str, Any]:
        return {
            "mode": mode,
            "num_attempts": len(attempts),
            "passed": feedback.passed,
            "returncode": feedback.returncode,
            "timeout": feedback.timeout,
            "final_failure_type": failure_type,
            "skill_hit": skill_hit,
            "fallback_triggered": fallback_triggered,
            "dataset_split": "",
            "metrics": dict(feedback.metrics),
        }

    def _contextualized_retrieval_scores(self, plan: WorkflowPlan, retrieval_scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if plan.planning_mode != "skill_seeded_dynamic" or not retrieval_scores:
            return retrieval_scores
        seed_context_skill_ids = [str(skill_id).strip() for skill_id in (plan.seed_context_skill_ids or []) if str(skill_id).strip()]
        if not seed_context_skill_ids:
            selected_skill_id = str(plan.selected_skill_id or "")
            seed_context_skill_ids = [selected_skill_id] if selected_skill_id else []
        if not seed_context_skill_ids:
            return retrieval_scores
        selected_rows = [
            dict(item, _preserve_prompt_order=True)
            for skill_id in seed_context_skill_ids
            for item in retrieval_scores
            if str(item.get("skill_id", "")) == skill_id
        ]
        if not selected_rows:
            return retrieval_scores
        return selected_rows

    def _select_draft_templates(
        self,
        *,
        plan: WorkflowPlan,
        retrieval_scores: list[dict[str, Any]],
        matched_templates: list[dict[str, Any]],
        template_injection_enabled: bool,
    ) -> list[dict[str, Any]]:
        if not template_injection_enabled:
            return []
        candidates = [
            template
            for template in matched_templates
            if template.get("template_kind") in {"seeded_context", "draft_scaffold", ""}
        ]
        candidates.sort(key=self._template_sort_key)
        return candidates[:1]

    def _select_repair_templates(
        self,
        *,
        plan: WorkflowPlan,
        matched_templates: list[dict[str, Any]],
        feedback: VerifierFeedback,
        failure_type: str,
        template_injection_enabled: bool,
    ) -> list[dict[str, Any]]:
        if not template_injection_enabled:
            return []
        candidates = []
        for template in matched_templates:
            if template.get("template_kind") not in {"repair_motif", "seeded_context"}:
                continue
            trigger = template.get("trigger_signature") or {}
            trigger_failure = str(trigger.get("failure_type", ""))
            if trigger_failure and trigger_failure not in {failure_type, feedback.summary, "unknown"}:
                continue
            candidates.append(template)
        candidates.sort(key=self._template_sort_key)
        return candidates[:1]

    def _run_skill_step(
        self,
        task: BenchmarkTaskInstance,
        skill_step_payload: dict[str, Any],
        actions: list[ActionRecord],
        attempts: list[AttemptRecord],
        code_history: list[str],
        verifier_feedback: list[VerifierFeedback],
        execution_outputs: list[dict[str, Any]],
    ) -> tuple[VerifierFeedback, str, float, float]:
        skill_id = str(skill_step_payload.get("skill_id", ""))
        code_path = str(skill_step_payload.get("code_path", ""))
        task_entry_point = infer_task_entry_point(task)
        skill_entry_point = str(skill_step_payload.get("entry_point", "") or task_entry_point or "")
        self._record_action(
            actions,
            tool_name="call_skill",
            tool_input_summary=f"skill_id={skill_id}, code_path={code_path}",
            tool_output_summary="loading retrieved skill",
        )
        try:
            import_started = time.perf_counter()
            candidate = self._load_skill_code(code_path)
            self._import_skill_module(code_path, skill_entry_point or task_entry_point)
            artifact_import_latency_ms = (time.perf_counter() - import_started) * 1000.0
            candidate = self._adapt_skill_candidate_for_task(
                candidate,
                skill_entry_point=skill_entry_point,
                task_entry_point=task_entry_point,
            )
            verify_started = time.perf_counter()
            feedback = verify_task(task=task, candidate=candidate, sandbox=self.sandbox)
            artifact_verification_latency_ms = (time.perf_counter() - verify_started) * 1000.0
        except Exception as exc:  # noqa: BLE001
            resolved_path = self._resolve_path(code_path)
            candidate = resolved_path.read_text(encoding="utf-8") if code_path and resolved_path.exists() else ""
            feedback = self._make_skill_runtime_failure(task=task, candidate_code=candidate, error=exc)
            artifact_import_latency_ms = 0.0
            artifact_verification_latency_ms = 0.0

        code_history.append(candidate)
        verifier_feedback.append(feedback)
        execution_outputs.append(feedback.model_dump(mode="json"))
        self._append_attempt(
            attempts=attempts,
            attempt_index=0,
            parent_attempt=None,
            stage="skill",
            candidate_code=candidate,
            feedback=feedback,
            failure_type=self._classify_failure(feedback),
            repair_source="retrieved_skill",
            notes=f"skill attempt via {skill_id}",
        )
        self._record_action(
            actions,
            tool_name="run_tests",
            tool_input_summary="attempt_index=0, source=skill",
            tool_output_summary=f"passed={feedback.passed}, returncode={feedback.returncode}, timeout={feedback.timeout}",
        )
        return feedback, skill_id, artifact_import_latency_ms, artifact_verification_latency_ms

    def run(
        self,
        task: BenchmarkTaskInstance,
        plan: WorkflowPlan,
        mode: str,
        retrieval_scores: list[dict[str, Any]] | None = None,
        matched_workflow_memories: list[dict[str, Any]] | None = None,
        matched_templates: list[dict[str, Any]] | None = None,
        matched_primitives: list[dict[str, Any]] | None = None,
        template_injection_enabled: bool = True,
    ) -> ExecutionTrace:
        start_time = time.perf_counter()
        actions: list[ActionRecord] = []
        attempts: list[AttemptRecord] = []
        generation_events: list[GenerationEvent] = []
        code_history: list[str] = []
        verifier_feedback: list[VerifierFeedback] = []
        execution_outputs: list[dict[str, Any]] = []
        failure_type = ""
        retrieval_scores = retrieval_scores or []
        matched_workflow_memories = matched_workflow_memories or []
        matched_templates = matched_templates or []
        matched_primitives = matched_primitives or []
        contextualized_retrieval_scores = self._contextualized_retrieval_scores(plan, retrieval_scores)
        injected_primitive_ids = [str(item.get("primitive_id", "")) for item in matched_primitives if item.get("primitive_id")]
        injected_workflow_memory_ids = [str(item.get("memory_id", "")) for item in matched_workflow_memories if item.get("memory_id")]
        used_skills: list[str] = []
        skill_hit = False
        fallback_triggered = False
        selected_skill_id = plan.selected_skill_id
        utility_before = None
        artifact_import_latency_ms = 0.0
        artifact_verification_latency_ms = 0.0
        for item in retrieval_scores:
            if item.get("skill_id") == selected_skill_id:
                utility_before = float(item.get("utility", 0.0))
                break

        generator = self.oracle_generator if mode == "oracle" else self.dynamic_generator
        repair_source = "oracle" if mode == "oracle" else "llm_runtime"
        feedback: VerifierFeedback | None = None
        attempt_offset = 0
        plan_steps = list(plan.steps)

        if plan_steps and plan_steps[0].step_type == "call_skill":
            feedback, used_skill, artifact_import_latency_ms, artifact_verification_latency_ms = self._run_skill_step(
                task=task,
                skill_step_payload=plan_steps[0].payload,
                actions=actions,
                attempts=attempts,
                code_history=code_history,
                verifier_feedback=verifier_feedback,
                execution_outputs=execution_outputs,
            )
            used_skills.append(used_skill)
            skill_hit = feedback.passed
            failure_type = self._classify_failure(feedback)
            if (
                not feedback.passed
                and bool(plan.exact_direct_postcall_wrapper_enabled or self.exact_direct_postcall_wrapper_enabled)
            ):
                feedback = self._run_direct_postcall_wrapper(
                    task=task,
                    plan=plan,
                    skill_step_payload=plan_steps[0].payload,
                    feedback=feedback,
                    retrieval_scores=retrieval_scores,
                    actions=actions,
                    attempts=attempts,
                    code_history=code_history,
                    verifier_feedback=verifier_feedback,
                    execution_outputs=execution_outputs,
                    generation_events=generation_events,
                    generator=generator,
                )
                skill_hit = feedback.passed
                failure_type = self._classify_failure(feedback)
            if feedback.passed:
                self._record_action(actions, "finalize", "mode=dynamic, source=skill, attempts=1", "success=True")
                return self._build_direct_trace(
                    task=task,
                    mode=mode,
                    plan=plan,
                    retrieval_scores=retrieval_scores,
                    utility_before=utility_before,
                    actions=actions,
                    attempts=attempts,
                    generation_events=generation_events,
                    code_history=code_history,
                    verifier_feedback=verifier_feedback,
                    execution_outputs=execution_outputs,
                    feedback=feedback,
                    failure_type=failure_type,
                    used_skills=used_skills,
                    skill_hit=True,
                    fallback_triggered=False,
                    start_time=start_time,
                    artifact_import_latency_ms=artifact_import_latency_ms,
                    artifact_verification_latency_ms=artifact_verification_latency_ms,
                    injected_workflow_memory_ids=injected_workflow_memory_ids,
                )
            if bool(plan.exact_direct_postcall_wrapper_enabled or self.exact_direct_postcall_wrapper_enabled):
                self._record_action(actions, "finalize", "mode=dynamic, source=skill, attempts={}".format(len(attempts)), "success=False")
                return self._build_direct_trace(
                    task=task,
                    mode=mode,
                    plan=plan,
                    retrieval_scores=retrieval_scores,
                    utility_before=utility_before,
                    actions=actions,
                    attempts=attempts,
                    generation_events=generation_events,
                    code_history=code_history,
                    verifier_feedback=verifier_feedback,
                    execution_outputs=execution_outputs,
                    feedback=feedback,
                    failure_type=failure_type,
                    used_skills=used_skills,
                    skill_hit=False,
                    fallback_triggered=False,
                    start_time=start_time,
                    artifact_import_latency_ms=artifact_import_latency_ms,
                    artifact_verification_latency_ms=artifact_verification_latency_ms,
                    injected_workflow_memory_ids=injected_workflow_memory_ids,
                )
            attempt_offset = 1
            fallback_triggered = True
            self._record_action(
                actions,
                tool_name="fallback_dynamic",
                tool_input_summary=f"skill_id={used_skill}, passed={feedback.passed}",
                tool_output_summary="switching to baseline dynamic workflow",
            )

        draft_stage = "draft_from_skill" if plan.planning_mode == "skill_seeded_dynamic" else "draft"
        repair_stage = "repair_with_skill" if plan.planning_mode == "skill_seeded_dynamic" else "repair"
        repair_source = "skill_seeded" if plan.planning_mode == "skill_seeded_dynamic" else repair_source
        draft_injected_templates = self._select_draft_templates(
            plan=plan,
            retrieval_scores=retrieval_scores,
            matched_templates=matched_templates,
            template_injection_enabled=template_injection_enabled,
        )
        repair_injected_template_ids: list[str] = []

        try:
            draft_output = self._invoke_generator(
                generator,
                "draft",
                task=task,
                retrieved_skills=contextualized_retrieval_scores,
                planning_mode=plan.planning_mode,
                seed_context_skill_modes=plan.seed_context_skill_modes,
                seed_routing_mode=plan.seed_routing_mode,
                transfer_utility_score=plan.transfer_utility_score,
                estimated_seed_token_cost=plan.estimated_seed_token_cost,
                historical_positive_transfer=plan.historical_positive_transfer,
                negative_transfer_risk=plan.negative_transfer_risk,
                policy_directives=plan.policy_directives,
                workflow_memories=matched_workflow_memories,
                template_priors=draft_injected_templates,
                primitive_helpers=matched_primitives,
            )
            candidate = draft_output.code
            if draft_output.event is not None:
                generation_events.append(draft_output.event)
                execution_outputs.append(draft_output.event.model_dump(mode="json"))
        except GenerationError as exc:
            candidate = ""
            generation_events.append(exc.event)
            execution_outputs.append(exc.event.model_dump(mode="json"))
            feedback = self._make_generation_failure(task=task, stage=draft_stage, error=str(exc))
            failure_type = self._classify_failure(feedback)
            verifier_feedback.append(feedback)
            self._append_attempt(
                attempts=attempts,
                attempt_index=attempt_offset,
                parent_attempt=None if attempt_offset == 0 else 0,
                stage=draft_stage,
                candidate_code="",
                feedback=feedback,
                failure_type=failure_type,
                repair_source=repair_source,
                notes="initial draft generation failed",
            )
            self._record_action(
                actions,
                tool_name="draft_code",
                tool_input_summary=f"mode={mode}, benchmark={task.benchmark}, planning_mode={plan.planning_mode}, task_id={task.task_id}",
                tool_output_summary=f"generation failed via {repair_source}",
            )
            self._record_action(actions, "finalize", f"mode={mode}, attempts={len(attempts)}", "success=False")
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            totals = self._llm_totals(generation_events)
            final_result = self._final_result_payload(
                mode=mode,
                feedback=feedback,
                failure_type=failure_type,
                skill_hit=skill_hit,
                fallback_triggered=fallback_triggered,
                task=task,
                attempts=attempts,
            )
            return ExecutionTrace(
                trace_id=f"trace_{uuid.uuid4().hex[:8]}",
                task_id=task.task_id,
                family=task.benchmark,
                query=self._task_query(task),
                benchmark=task.benchmark,
                retrieved_skills=plan.retrieved_skills,
                retrieval_scores=retrieval_scores,
                planning_mode=plan.planning_mode,
                task_pattern=plan.task_pattern,
                selected_skill_id=plan.selected_skill_id,
                seed_routing_mode=plan.seed_routing_mode,
                transfer_utility_score=plan.transfer_utility_score,
                estimated_seed_token_cost=plan.estimated_seed_token_cost,
                historical_positive_transfer=plan.historical_positive_transfer,
                negative_transfer_risk=plan.negative_transfer_risk,
                skill_seeded=plan.skill_seeded,
                seed_source=plan.seed_source,
                selection_reason=plan.selection_reason,
                utility_before=utility_before,
                utility_after=None,
                plan=plan,
                actions=actions,
                attempts=attempts,
                generation_events=generation_events,
                code_history=code_history,
                verifier_feedback=verifier_feedback,
                final_result=final_result,
                execution_outputs=execution_outputs,
                success=False,
                token_cost=float(totals["total_tokens"]),
                latency_ms=latency_ms,
                artifact_execution_latency_ms=artifact_import_latency_ms + artifact_verification_latency_ms,
                artifact_import_latency_ms=artifact_import_latency_ms,
                artifact_verification_latency_ms=artifact_verification_latency_ms,
                llm_prompt_tokens_total=int(totals["prompt_tokens"]),
                llm_completion_tokens_total=int(totals["completion_tokens"]),
                llm_total_tokens_total=int(totals["total_tokens"]),
                llm_latency_ms_total=float(totals["latency_ms"]),
                llm_call_count=int(totals["call_count"]),
                policy_version=plan.policy_version,
                matched_workflow_memory_ids=plan.matched_workflow_memory_ids,
                injected_workflow_memory_ids=injected_workflow_memory_ids,
                matched_template_ids=plan.matched_template_ids,
                matched_primitive_ids=plan.matched_primitive_ids,
                draft_injected_template_ids=[str(item.get("template_id", "")) for item in draft_injected_templates if item.get("template_id")],
                repair_injected_template_ids=[],
                injected_primitive_ids=injected_primitive_ids,
                failure_type=failure_type,
                used_skills=used_skills,
                skill_hit=skill_hit,
                fallback_triggered=fallback_triggered,
                **self._operational_metrics(actions, attempts, generation_events, plan.selected_skill_id, injected_workflow_memory_ids),
                **self._structured_trace_fields(
                    plan=plan,
                    generation_events=generation_events,
                    candidate_code="",
                    task_entry_point=infer_task_entry_point(task) or "",
                ),
            )

        candidate, _skill_plan_payload = self._extract_skill_plan_header(candidate)
        code_history.append(candidate)
        self._record_action(
            actions,
            tool_name="draft_code",
            tool_input_summary=f"mode={mode}, benchmark={task.benchmark}, planning_mode={plan.planning_mode}, task_id={task.task_id}",
            tool_output_summary=f"draft created via {repair_source}",
        )
        feedback = verify_task(task=task, candidate=candidate, sandbox=self.sandbox)
        failure_type = self._classify_failure(feedback)
        verifier_feedback.append(feedback)
        execution_outputs.append(feedback.model_dump(mode="json"))
        self._append_attempt(
            attempts=attempts,
            attempt_index=attempt_offset,
            parent_attempt=None if attempt_offset == 0 else 0,
            stage=draft_stage,
            candidate_code=candidate,
            feedback=feedback,
            failure_type=failure_type,
            repair_source=repair_source,
            notes="initial draft attempt" if attempt_offset == 0 else "fallback dynamic draft attempt",
        )
        self._record_action(
            actions,
            tool_name="run_tests",
            tool_input_summary=f"attempt_index={attempt_offset}",
            tool_output_summary=f"passed={feedback.passed}, returncode={feedback.returncode}, timeout={feedback.timeout}",
        )

        if (
            mode == "dynamic"
            and not feedback.passed
            and not fallback_triggered
            and self._compatibility_like_failure(
                task=task,
                plan=plan,
                feedback=feedback,
                retrieval_scores=retrieval_scores,
            )
            and str(getattr(plan, "fallback_planning_mode", "") or "").strip() == "pure_dynamic"
        ):
            fallback_triggered = True
            self._record_action(
                actions,
                tool_name="fallback_dynamic",
                tool_input_summary=f"task_id={task.task_id}, selected_skill_id={plan.selected_skill_id}",
                tool_output_summary="soft-aware compatibility fallback to pure_dynamic",
            )
            try:
                fallback_output = self._invoke_generator(
                    generator,
                    "draft",
                    task=task,
                    retrieved_skills=[],
                    planning_mode="pure_dynamic",
                    seed_context_skill_modes={},
                    seed_routing_mode="",
                    transfer_utility_score=0.0,
                    estimated_seed_token_cost=0.0,
                    historical_positive_transfer=0.0,
                    negative_transfer_risk=0.0,
                    policy_directives=plan.policy_directives,
                    workflow_memories=matched_workflow_memories,
                    template_priors=draft_injected_templates,
                    primitive_helpers=matched_primitives,
                )
                candidate = fallback_output.code
                if fallback_output.event is not None:
                    generation_events.append(fallback_output.event)
                    execution_outputs.append(fallback_output.event.model_dump(mode="json"))
                candidate, _fallback_skill_plan_payload = self._extract_skill_plan_header(candidate)
                code_history.append(candidate)
                self._record_action(
                    actions,
                    tool_name="draft_code",
                    tool_input_summary=f"mode={mode}, benchmark={task.benchmark}, planning_mode=pure_dynamic, task_id={task.task_id}",
                    tool_output_summary="soft-aware fallback draft created via llm_runtime",
                )
                feedback = verify_task(task=task, candidate=candidate, sandbox=self.sandbox)
                failure_type = self._classify_failure(feedback)
                verifier_feedback.append(feedback)
                execution_outputs.append(feedback.model_dump(mode="json"))
                self._append_attempt(
                    attempts=attempts,
                    attempt_index=attempt_offset + 1,
                    parent_attempt=attempt_offset,
                    stage="draft",
                    candidate_code=candidate,
                    feedback=feedback,
                    failure_type=failure_type,
                    repair_source="soft_aware_fallback_dynamic",
                    notes="soft-aware compatibility fallback draft",
                )
                self._record_action(
                    actions,
                    tool_name="run_tests",
                    tool_input_summary=f"attempt_index={attempt_offset + 1}",
                    tool_output_summary=f"passed={feedback.passed}, returncode={feedback.returncode}, timeout={feedback.timeout}",
                )
                attempt_offset += 1
            except GenerationError as exc:
                generation_events.append(exc.event)
                execution_outputs.append(exc.event.model_dump(mode="json"))
                feedback = self._make_generation_failure(task=task, stage="fallback_dynamic", error=str(exc))
                failure_type = self._classify_failure(feedback)
                verifier_feedback.append(feedback)
                self._append_attempt(
                    attempts=attempts,
                    attempt_index=attempt_offset + 1,
                    parent_attempt=attempt_offset,
                    stage="draft",
                    candidate_code="",
                    feedback=feedback,
                    failure_type=failure_type,
                    repair_source="soft_aware_fallback_dynamic",
                    notes="soft-aware fallback draft generation failed",
                )
                self._record_action(
                    actions,
                    tool_name="draft_code",
                    tool_input_summary=f"mode={mode}, benchmark={task.benchmark}, planning_mode=pure_dynamic, task_id={task.task_id}",
                    tool_output_summary="soft-aware fallback generation failed",
                )
                attempt_offset += 1

        parent_attempt = attempt_offset
        repair_count = 0
        while mode == "dynamic" and not feedback.passed and repair_count < self.max_repairs:
            repair_count += 1
            failure_type = self._classify_failure(feedback)
            self._record_action(
                actions,
                tool_name="inspect_failure",
                tool_input_summary=f"attempt_index={parent_attempt}, stderr={feedback.stderr[:160]}",
                tool_output_summary=f"failure_type={failure_type or 'unknown'}",
            )
            try:
                repair_injected_templates = self._select_repair_templates(
                    plan=plan,
                    matched_templates=matched_templates,
                    feedback=feedback,
                    failure_type=failure_type,
                    template_injection_enabled=template_injection_enabled,
                )
                for item in repair_injected_templates:
                    template_id = str(item.get("template_id", ""))
                    if template_id and template_id not in repair_injected_template_ids:
                        repair_injected_template_ids.append(template_id)
                repair_output = self._invoke_generator(
                    generator,
                    "repair",
                    task=task,
                    previous_code=candidate,
                    feedback=feedback,
                    attempt_index=repair_count,
                    retrieved_skills=contextualized_retrieval_scores,
                    planning_mode=plan.planning_mode,
                    seed_context_skill_modes=plan.seed_context_skill_modes,
                    seed_routing_mode=plan.seed_routing_mode,
                    transfer_utility_score=plan.transfer_utility_score,
                    estimated_seed_token_cost=plan.estimated_seed_token_cost,
                    historical_positive_transfer=plan.historical_positive_transfer,
                    negative_transfer_risk=plan.negative_transfer_risk,
                    policy_directives=plan.policy_directives,
                    workflow_memories=matched_workflow_memories,
                    template_priors=repair_injected_templates,
                    primitive_helpers=matched_primitives,
                )
                candidate = repair_output.code
                if repair_output.event is not None:
                    generation_events.append(repair_output.event)
                    execution_outputs.append(repair_output.event.model_dump(mode="json"))
            except GenerationError as exc:
                generation_events.append(exc.event)
                execution_outputs.append(exc.event.model_dump(mode="json"))
                feedback = self._make_generation_failure(task=task, stage=f"repair_{repair_count}", error=str(exc))
                failure_type = self._classify_failure(feedback)
                verifier_feedback.append(feedback)
                self._append_attempt(
                    attempts=attempts,
                    attempt_index=attempt_offset + repair_count,
                    parent_attempt=parent_attempt,
                    stage=repair_stage,
                    candidate_code="",
                    feedback=feedback,
                    failure_type=failure_type,
                    repair_source=repair_source,
                    notes=f"repair generation failed on attempt {repair_count}",
                )
                self._record_action(
                    actions,
                    tool_name="repair_code",
                    tool_input_summary=f"repair_attempt={repair_count}, parent_attempt={parent_attempt}, planning_mode={plan.planning_mode}",
                    tool_output_summary=f"{repair_source} generation failed",
                )
                break
            candidate, _repair_skill_plan_payload = self._extract_skill_plan_header(candidate)
            code_history.append(candidate)
            self._record_action(
                actions,
                tool_name="repair_code",
                tool_input_summary=f"repair_attempt={repair_count}, parent_attempt={parent_attempt}, planning_mode={plan.planning_mode}",
                tool_output_summary=f"{repair_source} repair generated new candidate",
            )

            feedback = verify_task(task=task, candidate=candidate, sandbox=self.sandbox)
            failure_type = self._classify_failure(feedback)
            verifier_feedback.append(feedback)
            execution_outputs.append(feedback.model_dump(mode="json"))
            self._append_attempt(
                attempts=attempts,
                attempt_index=attempt_offset + repair_count,
                parent_attempt=parent_attempt,
                stage=repair_stage,
                candidate_code=candidate,
                feedback=feedback,
                failure_type=failure_type,
                repair_source=repair_source,
                notes=f"repair attempt {repair_count}",
            )
            self._record_action(
                actions,
                tool_name="run_tests",
                tool_input_summary=f"attempt_index={attempt_offset + repair_count}",
                tool_output_summary=f"passed={feedback.passed}, returncode={feedback.returncode}, timeout={feedback.timeout}",
            )
            parent_attempt = attempt_offset + repair_count

        self._record_action(actions, "finalize", f"mode={mode}, attempts={len(attempts)}", f"success={feedback.passed}")
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        totals = self._llm_totals(generation_events)
        final_result = self._final_result_payload(
            mode=mode,
            feedback=feedback,
            failure_type=failure_type,
            skill_hit=skill_hit,
            fallback_triggered=fallback_triggered,
            task=task,
            attempts=attempts,
        )

        return ExecutionTrace(
            trace_id=f"trace_{uuid.uuid4().hex[:8]}",
            task_id=task.task_id,
            family=task.benchmark,
            query=self._task_query(task),
            benchmark=task.benchmark,
            retrieved_skills=plan.retrieved_skills,
            retrieval_scores=retrieval_scores,
            planning_mode=plan.planning_mode,
            task_pattern=plan.task_pattern,
            selected_skill_id=plan.selected_skill_id,
            seed_routing_mode=plan.seed_routing_mode,
            transfer_utility_score=plan.transfer_utility_score,
            estimated_seed_token_cost=plan.estimated_seed_token_cost,
            historical_positive_transfer=plan.historical_positive_transfer,
            negative_transfer_risk=plan.negative_transfer_risk,
            skill_seeded=plan.skill_seeded,
            seed_source=plan.seed_source,
            selection_reason=plan.selection_reason,
            utility_before=utility_before,
            utility_after=None,
            plan=plan,
            actions=actions,
            attempts=attempts,
            generation_events=generation_events,
            code_history=code_history,
            verifier_feedback=verifier_feedback,
            final_result=final_result,
            execution_outputs=execution_outputs,
            success=feedback.passed,
            token_cost=float(totals["total_tokens"]),
            latency_ms=latency_ms,
            artifact_execution_latency_ms=artifact_import_latency_ms + artifact_verification_latency_ms,
            artifact_import_latency_ms=artifact_import_latency_ms,
            artifact_verification_latency_ms=artifact_verification_latency_ms,
            llm_prompt_tokens_total=int(totals["prompt_tokens"]),
            llm_completion_tokens_total=int(totals["completion_tokens"]),
            llm_total_tokens_total=int(totals["total_tokens"]),
            llm_latency_ms_total=float(totals["latency_ms"]),
            llm_call_count=int(totals["call_count"]),
            policy_version=plan.policy_version,
            matched_workflow_memory_ids=plan.matched_workflow_memory_ids,
            injected_workflow_memory_ids=injected_workflow_memory_ids,
            matched_template_ids=plan.matched_template_ids,
            matched_primitive_ids=plan.matched_primitive_ids,
            draft_injected_template_ids=[str(item.get("template_id", "")) for item in draft_injected_templates if item.get("template_id")],
            repair_injected_template_ids=repair_injected_template_ids,
            injected_primitive_ids=injected_primitive_ids,
            failure_type=failure_type,
            used_skills=used_skills,
            skill_hit=skill_hit,
            fallback_triggered=fallback_triggered,
            **self._operational_metrics(actions, attempts, generation_events, plan.selected_skill_id, injected_workflow_memory_ids),
            **self._structured_trace_fields(
                plan=plan,
                generation_events=generation_events,
                candidate_code=code_history[-1] if code_history else "",
                task_entry_point=infer_task_entry_point(task) or "",
            ),
        )
