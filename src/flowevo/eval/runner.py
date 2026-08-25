"""Benchmark runner for MBPP-first workflow baselines."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent.executor import CodeTaskExecutor
from agent.generator import BaseGenerator, LLMGenerator
from agent.planner import CodeTaskPlanner
from agent.retriever import Retriever
from agent.skill_compatibility import task_compatibility_context
from compiler.admission import CompileAdmissionResult, admit_compiled_trace, existing_dedup_keys
from compiler.primitive_compiler import PrimitiveCompiler
from compiler.template_compiler import TemplateCompiler
from compiler.workflow_memory_compiler import WorkflowMemoryCompiler
from core.utils import infer_task_arg_count, infer_task_entry_point, infer_task_pattern
from env.sandbox import Sandbox
from governance.audit import audit_trace
from governance.utility import score_episode
from maintenance.governance import GovernanceKernel
from memory.negative_memory_store import NegativeMemoryStore
from memory.policy_store import PolicyStore
from memory.primitive_store import PrimitiveStore
from memory.skill_registry import SkillRegistry
from memory.template_store import TemplateStore
from memory.trace_store import TraceStore
from memory.workflow_memory_store import WorkflowMemoryStore
from eval.verifier import verify_task
from runtime.config import load_runtime_config
from compiler.strategy_compiler import StrategyCompiler
from core.generator_state import build_generator_state
from core.meta_update import meta_update
from memory.strategy_registry import StrategyRegistry

HIGH_CONFIDENCE_NEGATIVE_SOURCES = frozenset({"runtime_failure", "audit_failure", "prune_reason"})
STRING_SEEDED_QUERY_PATTERN = re.compile(
    r"\b(string|strings|character|characters|letter|letters|word|words|uppercase|lowercase|split)\b"
)


@dataclass
class RunnerConfig:
    mode: str = "dynamic"
    benchmark: str = "humaneval"
    dataset_split: str = ""
    profile: str = "full"
    limit: int = 3
    timeout_seconds: float = 5.0
    retrieval_top_k: int = 3
    retrieval_threshold: float = 5.0
    direct_skill_threshold: float | None = None
    seed_threshold: float | None = None
    seed_guard_entry_threshold: float | None = None
    seeded_shadow_promotion_margin: float = 0.0
    planner_seed_pool_top_k: int | None = None
    compatibility_hard_gate: bool = False
    soft_aware_first: bool = False
    planner_negative_memory_enabled: bool = False
    planner_state_reset: str = "never"
    selector_exact_compat_bonus: float = 0.0
    selector_active_utility_bonus: float = 0.0
    selector_shadow_penalty: float = 0.0
    selector_low_confidence_abstain_threshold: float | None = None
    bifurcated_skill_router_enabled: bool = False
    bifurcated_exact_or_seeded_enabled: bool = False
    selective_integrated_enabled: bool = False
    seeded_disable_local_insertion: bool = False
    exact_direct_postcall_wrapper_enabled: bool = False
    direct_postcall_max_local_patch: int = 0
    audit_top_k: int = 0
    disable_retrieval: bool = False
    direct_skill_only: bool = False
    allow_unsupported_family_direct_admission: bool = False
    allow_non_exact_direct_reuse: bool = False
    admit_seeded_compiles: bool = False
    disable_direct_skill_execution: bool = False
    disable_skill_seeded: bool = False
    disable_governance: bool = False
    disable_policy: bool = False
    frozen_eval: bool = False
    disable_policy_update: bool = False
    freeze_policy_updates: bool = False
    normalize_retrieval_ties_for_planner: bool = False
    preserve_prompt_skill_order: bool = False
    enable_string_seeded_bias: bool = False
    enable_count_seeded_bias: bool = False
    proof_compact_seeded_context: bool = False
    proof_compact_seeded_repair: bool = False
    disable_template_injection: bool = False
    disable_template_growth: bool = False
    disable_primitive_helper_context: bool = False
    disable_primitive_growth: bool = False
    disable_executable_artifact_growth: bool = False
    novelty_budget_pattern_arity_limit: int | None = None
    novelty_budget_task_pattern_limit: int | None = None
    disable_textual_workflow_memory_growth: bool = True
    disable_workflow_memory_growth: bool = False
    disable_textual_workflow_memory_retrieval: bool = True
    disable_negative_memory_write: bool = False
    disable_skill_reuse: bool = False
    trace_root: str = "data/traces"
    registry_path: str = "data/skills/registry.json"
    config_path: str = "configs/default.yaml"
    local_config_path: str | None = None
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    skill_top_k: int | None = None
    draft_total_tokens: int | None = None
    repair_total_tokens: int | None = None
    draft_per_skill_tokens: int | None = None
    repair_per_skill_tokens: int | None = None
    policy_directive_min_confidence: float | None = None
    policy_directive_max_count: int | None = None
    policy_path: str = "data/policy/state.json"
    template_path: str = "data/tasks/templates.json"
    primitive_path: str = "data/primitives/registry.json"
    negative_memory_path: str = "data/governance/negative_memory.json"
    workflow_memory_path: str = "data/tasks/workflow_memory.json"
    strategy_path: str = "data/strategies/registry.json"
    workflow_memory_mode: str = "disabled"
    governance_mode: str = "disabled"
    setting_name: str = ""
    phase_a_bank_source: str = ""
    phase_b_usage_setting: str = ""
    task_order_seed: int | None = None
    phase_a_seed: int | None = None
    claim_target: str = ""
    mbpp_score_protocol: str = "table1_compatible"


class BenchmarkRunner:
    """Run benchmark tasks under oracle or dynamic workflow mode."""

    def __init__(
        self,
        config: RunnerConfig,
        python_executable: str = sys.executable,
        dynamic_generator: BaseGenerator | None = None,
    ) -> None:
        self.config = config
        self.mode = config.mode
        self.claim_target = config.claim_target
        self.mbpp_score_protocol = str(config.mbpp_score_protocol or "table1_compatible")
        self.frozen_eval = config.frozen_eval
        self.retrieval_top_k = config.retrieval_top_k
        self.retrieval_threshold = config.retrieval_threshold
        self.direct_skill_threshold = config.direct_skill_threshold
        self.seed_threshold = config.seed_threshold
        self.seed_guard_entry_threshold = config.seed_guard_entry_threshold
        self.seeded_shadow_promotion_margin = config.seeded_shadow_promotion_margin
        self.planner_seed_pool_top_k = config.planner_seed_pool_top_k
        self.compatibility_hard_gate = config.compatibility_hard_gate
        self.soft_aware_first = config.soft_aware_first
        self.planner_negative_memory_enabled = config.planner_negative_memory_enabled
        self.planner_state_reset = str(config.planner_state_reset or "never")
        self.selector_exact_compat_bonus = float(config.selector_exact_compat_bonus or 0.0)
        self.selector_active_utility_bonus = float(config.selector_active_utility_bonus or 0.0)
        self.selector_shadow_penalty = float(config.selector_shadow_penalty or 0.0)
        self.selector_low_confidence_abstain_threshold = config.selector_low_confidence_abstain_threshold
        self.bifurcated_skill_router_enabled = bool(config.bifurcated_skill_router_enabled)
        self.bifurcated_exact_or_seeded_enabled = bool(config.bifurcated_exact_or_seeded_enabled)
        self.selective_integrated_enabled = bool(config.selective_integrated_enabled)
        self.seeded_disable_local_insertion = bool(config.seeded_disable_local_insertion)
        self.exact_direct_postcall_wrapper_enabled = bool(config.exact_direct_postcall_wrapper_enabled)
        self.direct_postcall_max_local_patch = int(config.direct_postcall_max_local_patch or 0)
        self.audit_top_k = config.audit_top_k
        self.disable_retrieval = config.disable_retrieval
        self.direct_skill_only = config.direct_skill_only
        self.allow_unsupported_family_direct_admission = config.allow_unsupported_family_direct_admission
        self.allow_non_exact_direct_reuse = config.allow_non_exact_direct_reuse
        self.admit_seeded_compiles = config.admit_seeded_compiles
        self.disable_direct_skill_execution = config.disable_direct_skill_execution
        self.disable_skill_seeded = config.disable_skill_seeded
        self.disable_governance = config.disable_governance
        self.disable_policy = config.disable_policy
        self.disable_policy_update = config.disable_policy_update
        self.freeze_policy_updates = config.freeze_policy_updates
        self.normalize_retrieval_ties_for_planner = config.normalize_retrieval_ties_for_planner
        self.preserve_prompt_skill_order = config.preserve_prompt_skill_order
        self.enable_string_seeded_bias = config.enable_string_seeded_bias
        self.enable_count_seeded_bias = config.enable_count_seeded_bias
        self.proof_compact_seeded_context = config.proof_compact_seeded_context
        self.proof_compact_seeded_repair = config.proof_compact_seeded_repair
        self.disable_template_injection = config.disable_template_injection
        self.disable_template_growth = config.disable_template_growth
        self.disable_primitive_helper_context = config.disable_primitive_helper_context
        self.disable_primitive_growth = config.disable_primitive_growth
        self.disable_executable_artifact_growth = config.disable_executable_artifact_growth
        self.novelty_budget_pattern_arity_limit = config.novelty_budget_pattern_arity_limit
        self.novelty_budget_task_pattern_limit = config.novelty_budget_task_pattern_limit
        self.disable_textual_workflow_memory_growth = config.disable_textual_workflow_memory_growth
        self.disable_workflow_memory_growth = config.disable_workflow_memory_growth
        self.disable_textual_workflow_memory_retrieval = config.disable_textual_workflow_memory_retrieval
        self.disable_negative_memory_write = config.disable_negative_memory_write
        self.disable_skill_reuse = config.disable_skill_reuse
        self.workflow_memory_mode = config.workflow_memory_mode
        self.governance_mode = "disabled" if config.disable_governance else (config.governance_mode or "full")
        if self.frozen_eval:
            self.disable_executable_artifact_growth = True
            self.disable_policy_update = True
            self.freeze_policy_updates = True
            self.disable_template_growth = True
            self.disable_primitive_growth = True
            self.disable_workflow_memory_growth = True
            self.disable_negative_memory_write = True
            self.disable_governance = True
            self.governance_mode = "disabled"
            self.config.disable_executable_artifact_growth = True
            self.config.disable_policy_update = True
            self.config.freeze_policy_updates = True
            self.config.disable_template_growth = True
            self.config.disable_primitive_growth = True
            self.config.disable_workflow_memory_growth = True
            self.config.disable_negative_memory_write = True
            self.config.disable_governance = True
            self.config.governance_mode = "disabled"
        self.policy_directive_min_confidence = config.policy_directive_min_confidence
        self.policy_directive_max_count = config.policy_directive_max_count
        self.sandbox = Sandbox(python_executable=python_executable, timeout_seconds=config.timeout_seconds)
        self.planner = CodeTaskPlanner()
        self.skill_registry = SkillRegistry(file_path=config.registry_path)
        self.policy_store = PolicyStore(file_path=config.policy_path)
        self.template_store = TemplateStore(file_path=config.template_path)
        self.primitive_store = PrimitiveStore(file_path=config.primitive_path)
        self.negative_memory_store = NegativeMemoryStore(file_path=config.negative_memory_path)
        self.workflow_memory_store = WorkflowMemoryStore(file_path=config.workflow_memory_path)
        self.strategy_registry = StrategyRegistry(file_path=config.strategy_path)
        self.template_compiler = TemplateCompiler()
        self.primitive_compiler = PrimitiveCompiler()
        self.workflow_memory_compiler = WorkflowMemoryCompiler()
        self.compile_dedup_keys = existing_dedup_keys(self.skill_registry)
        self.retriever = Retriever(
            registry=self.skill_registry,
            negative_memory_store=self.negative_memory_store,
            top_k=config.retrieval_top_k,
        )
        runtime_dynamic_generator = dynamic_generator
        if config.mode == "dynamic" and runtime_dynamic_generator is None:
            runtime_config = load_runtime_config(
                config_path=config.config_path,
                local_override_path=config.local_config_path,
                provider=config.provider,
                model=config.model,
                base_url=config.base_url,
                skill_top_k=config.skill_top_k,
                draft_total_tokens=config.draft_total_tokens,
                repair_total_tokens=config.repair_total_tokens,
                draft_per_skill_tokens=config.draft_per_skill_tokens,
                repair_per_skill_tokens=config.repair_per_skill_tokens,
            )
            runtime_dynamic_generator = LLMGenerator(runtime_config=runtime_config)
        if runtime_dynamic_generator is not None:
            setattr(runtime_dynamic_generator, "preserve_prompt_skill_order", config.preserve_prompt_skill_order)
        self.executor = CodeTaskExecutor(
            sandbox=self.sandbox,
            max_repairs=2,
            dynamic_generator=runtime_dynamic_generator,
            exact_direct_postcall_wrapper_enabled=self.exact_direct_postcall_wrapper_enabled,
            direct_postcall_max_local_patch=self.direct_postcall_max_local_patch,
        )
        # v7: LLM-enhanced strategy compilation (fallback to heuristic if unavailable)
        _strategy_llm = getattr(runtime_dynamic_generator, "client", None) if runtime_dynamic_generator is not None else None
        _strategy_settings = getattr(runtime_dynamic_generator, "runtime_config", None)
        _strategy_settings = getattr(_strategy_settings, "draft", None) if _strategy_settings is not None else None
        self.strategy_compiler = StrategyCompiler(
            llm_client=_strategy_llm,
            llm_settings=_strategy_settings,
        )
        self.trace_store = TraceStore(root=config.trace_root)
        self.governance = GovernanceKernel(
            registry=self.skill_registry,
            primitive_store=self.primitive_store,
            negative_memory_store=self.negative_memory_store,
            sandbox=self.sandbox,
            repair_generator=self.executor.dynamic_generator,
        )
        self._planner_negative_memory: dict[tuple[str, str], int] = {}
        self.disable_contrastive_eval = False
        self._generator_state = build_generator_state(
            skill_registry=self.skill_registry,
            template_store=self.template_store,
            policy_store=self.policy_store,
            benchmark=config.benchmark,
        )

    def _mbpp_hidden_scoring_enabled(self, task) -> bool:
        return str(getattr(task, "benchmark", "") or "").strip().lower() == "mbpp"

    def _rescore_mbpp_trace(self, *, task, trace) -> None:
        if not self._mbpp_hidden_scoring_enabled(task):
            return
        final_candidate = trace.code_history[-1] if trace.code_history else ""
        if not final_candidate:
            return
        score_protocol = str(self.mbpp_score_protocol or "table1_compatible")
        final_hidden_feedback = verify_task(
            task=task,
            candidate=final_candidate,
            sandbox=self.sandbox,
            scope="hidden_eval",
            score_protocol=score_protocol,
        )
        trace.verifier_feedback.append(final_hidden_feedback)
        trace.execution_outputs.append(final_hidden_feedback.model_dump(mode="json"))
        trace.success = bool(final_hidden_feedback.passed)
        trace.failure_type = self.executor._classify_failure(final_hidden_feedback)
        if trace.attempts:
            first_candidate = str(trace.attempts[0].candidate_code or "").strip()
            if first_candidate:
                first_feedback = verify_task(
                    task=task,
                    candidate=first_candidate,
                    sandbox=self.sandbox,
                    scope="hidden_eval",
                    score_protocol=score_protocol,
                )
                trace.first_attempt_success = bool(first_feedback.passed)
                trace.final_result["mbpp_first_attempt_hidden_feedback"] = first_feedback.model_dump(mode="json")
        trace.final_result["mbpp_score_protocol"] = score_protocol
        trace.final_result["mbpp_visible_test_count"] = len(getattr(task, "visible_public_tests", []) or getattr(task, "test_list", []) or [])
        trace.final_result["mbpp_hidden_feedback"] = final_hidden_feedback.model_dump(mode="json")

    def _episode_provenance(self, trace) -> str:
        if not trace.success:
            return ""
        if trace.planning_mode == "pure_dynamic":
            return "pure_dynamic_success"
        if trace.planning_mode == "skill_seeded_dynamic":
            return "seeded_dynamic_success"
        if trace.planning_mode == "direct_skill":
            return "direct_skill_reuse"
        return ""

    def _apply_online_utility_distillation(self, *, benchmark: str) -> dict[str, Any]:
        if "utility_distilled" not in str(self.config.setting_name or ""):
            return {"enabled": False, "suppressed_skill_ids": [], "kept_skill_ids": []}
        kept_ids = self.skill_registry.utility_distilled_skill_ids(
            benchmark=benchmark,
            retain_per_cluster=1,
            second_margin=0.25,
        )
        suppressed_skill_ids: list[str] = []
        for card in self.skill_registry.list_all():
            card_benchmark = str(card.preconditions.get("benchmark", card.family) or "").strip().lower()
            if card_benchmark != benchmark or str(card.status) != "active":
                continue
            if str(card.skill_id) in kept_ids:
                continue
            self.skill_registry.set_status(card.skill_id, "suppressed")
            suppressed_skill_ids.append(str(card.skill_id))
        return {
            "enabled": True,
            "kept_skill_ids": sorted(kept_ids),
            "suppressed_skill_ids": suppressed_skill_ids,
        }

    def _novelty_budget_exhausted(self, task, trace) -> bool:
        task_pattern = trace.task_pattern or infer_task_pattern(task)
        task_arity = infer_task_arg_count(task)
        benchmark = trace.benchmark or trace.family
        active_cards = list(self.skill_registry.list_active())

        pattern_limit = self.novelty_budget_task_pattern_limit
        if pattern_limit is not None and pattern_limit > 0:
            pattern_matches = 0
            for card in active_cards:
                if (card.preconditions.get("benchmark", card.family) or "") != benchmark:
                    continue
                if str(card.preconditions.get("task_pattern", "")) != task_pattern:
                    continue
                pattern_matches += 1
                if pattern_matches >= pattern_limit:
                    return True

        arity_limit = self.novelty_budget_pattern_arity_limit
        if arity_limit is None or arity_limit <= 0:
            return False
        active_matches = 0
        for card in active_cards:
            if (card.preconditions.get("benchmark", card.family) or "") != benchmark:
                continue
            if str(card.preconditions.get("task_pattern", "")) != task_pattern:
                continue
            args = card.signature.get("args", []) if isinstance(card.signature, dict) else []
            skill_arity = len(args) if isinstance(args, list) else None
            if skill_arity != task_arity:
                continue
            active_matches += 1
            if active_matches >= arity_limit:
                return True
        return False

    def _has_redundant_pattern_solution(self, task, trace) -> bool:
        if self.novelty_budget_task_pattern_limit is None:
            return False
        task_pattern = trace.task_pattern or infer_task_pattern(task)
        benchmark = trace.benchmark or trace.family
        active_pattern_matches = [
            card
            for card in self.skill_registry.list_active()
            if (card.preconditions.get("benchmark", card.family) or "") == benchmark
            and str(card.preconditions.get("task_pattern", "")) == task_pattern
        ]
        if not active_pattern_matches:
            return False
        seen_candidates = list(trace.hard_filter_survivors or []) or list(trace.recall_candidates or [])
        return any(str(item.get("task_pattern", "")) == task_pattern for item in seen_candidates)

    def _growth_budget_rejection_reason(self, task, trace) -> str:
        if self._has_redundant_pattern_solution(task, trace):
            return "redundant_pattern_solution"
        if self._novelty_budget_exhausted(task, trace):
            return "novelty_budget_exhausted"
        return ""

    def _online_compile(self, task, trace) -> CompileAdmissionResult:
        if self.frozen_eval or self.disable_executable_artifact_growth:
            return CompileAdmissionResult(attempted=False, admitted=False)
        if trace.episode_provenance not in {"pure_dynamic_success", "seeded_dynamic_success"}:
            return CompileAdmissionResult(attempted=False, admitted=False)
        rejection_reason = self._growth_budget_rejection_reason(task, trace)
        if rejection_reason:
            return CompileAdmissionResult(
                attempted=True,
                admitted=False,
                rejection_stage="growth_budget",
                rejection_reason=rejection_reason,
                status="skipped",
                benchmark=trace.benchmark or trace.family,
            )
        try:
            return admit_compiled_trace(
                trace,
                provenance=trace.episode_provenance,
                registry=self.skill_registry,
                timeout_seconds=self.config.timeout_seconds,
                dedup_keys=self.compile_dedup_keys,
                force_shadow=trace.episode_provenance == "seeded_dynamic_success" and not self.admit_seeded_compiles,
                allow_unsupported_family_direct_admission=self.allow_unsupported_family_direct_admission,
            )
        except ValueError as exc:
            if "has no passing attempt to compile" not in str(exc):
                raise
            return CompileAdmissionResult(
                attempted=True,
                admitted=False,
                rejection_stage="compile_source",
                rejection_reason="no_passing_attempt",
                status="skipped",
                benchmark=trace.benchmark or trace.family,
            )

    def _record_runtime_governance(self, task, trace) -> tuple[float | None, str]:
        utility_after = None
        status_after = ""
        skill_id = trace.selected_skill_id
        if not skill_id or self.governance_mode == "disabled":
            return utility_after, status_after

        if trace.planning_mode == "direct_skill":
            usage_success = trace.skill_hit
            usage_mode = "direct"
        elif trace.planning_mode == "skill_seeded_dynamic":
            usage_success = trace.success
            usage_mode = "seeded"
        else:
            return utility_after, status_after

        seeded_usage = trace.planning_mode == "skill_seeded_dynamic" and trace.seeded_use_count > 0
        updated = self.skill_registry.record_usage(
            skill_id=skill_id,
            success=usage_success,
            usage_mode=usage_mode,
            routing_mode=trace.seed_routing_mode if seeded_usage else "",
            first_attempt_success=trace.first_attempt_success if seeded_usage else None,
            retry_count=trace.retry_count if seeded_usage else None,
            token_cost_to_pass=trace.llm_total_tokens_total if (seeded_usage and trace.success) else None,
            transfer_positive=bool(seeded_usage and trace.success),
            transfer_negative=bool(seeded_usage and not trace.success),
        )
        if updated is not None:
            utility_after = updated.utility
            status_after = updated.status
            trace.utility_after = utility_after

        if not usage_success and trace.verifier_feedback:
            final_feedback = trace.verifier_feedback[-1]
            self.skill_registry.append_failure(
                skill_id,
                task_id=task.task_id,
                benchmark=task.benchmark,
                planning_mode=trace.planning_mode,
                verifier_message=final_feedback.stderr or final_feedback.summary,
                failure_type=trace.failure_type or "skill_usage_failure",
            )
            refreshed = self.skill_registry.get(skill_id)
            if refreshed is not None:
                status_after = refreshed.status
                utility_after = refreshed.utility
                trace.utility_after = refreshed.utility
        return utility_after, status_after

    def _retrieve(self, task) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        include_shadow_for_seeded = bool(
            (
                self.disable_direct_skill_execution
                or self.bifurcated_skill_router_enabled
                or self.bifurcated_exact_or_seeded_enabled
            )
            and not self.disable_skill_seeded
            and not self.direct_skill_only
        )
        retrieval_top_k = self.retrieval_top_k
        if include_shadow_for_seeded and self.enable_string_seeded_bias:
            query_text = " ".join(part for part in (getattr(task, "text", ""), getattr(task, "prompt", "")) if part).lower()
            if STRING_SEEDED_QUERY_PATTERN.search(query_text):
                # String-oriented seeded queries need a slightly wider shadow
                # candidate window so aware-only helpers like
                # `min_flip_to_make_string_alternate` are not dropped on
                # skill-id tie breaks before the planner can inspect them.
                retrieval_top_k = max(retrieval_top_k, 6)
        if include_shadow_for_seeded and self.planner_seed_pool_top_k:
            retrieval_top_k = max(retrieval_top_k, int(self.planner_seed_pool_top_k))
        if self.mode != "dynamic" or self.disable_retrieval or self.disable_skill_reuse:
            return [], {
                "active_skill_bank_at_episode_start": self.retriever.active_bank_snapshot(
                    include_shadow=include_shadow_for_seeded
                ),
                "recall_candidates": [],
                "hard_filter_survivors": [],
                "hard_filter_drop_reasons": {},
                "rerank_score_decomposition": {},
            }
        retrieval_scores, diagnostics = self.retriever.retrieve_with_diagnostics(
            task=task,
            top_k=retrieval_top_k,
            include_shadow=include_shadow_for_seeded,
        )
        if self.normalize_retrieval_ties_for_planner:
            retrieval_scores = sorted(
                retrieval_scores,
                key=lambda item: (
                    float(item.get("score", 0.0)),
                    int(bool(item.get("exact_entry_match", False))),
                    float(item.get("utility", 0.0)),
                    int(item.get("call_count", 0)),
                    str(item.get("skill_id", "")),
                ),
                reverse=True,
            )
        if any(
            (
                self.selector_exact_compat_bonus,
                self.selector_active_utility_bonus,
                self.selector_shadow_penalty,
                self.selector_low_confidence_abstain_threshold is not None,
            )
        ):
            retrieval_scores = self._apply_selector_tuning(task, retrieval_scores)
        return retrieval_scores, diagnostics

    def _apply_selector_tuning(self, task, retrieval_scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tuned: list[dict[str, Any]] = []
        for item in retrieval_scores:
            candidate = dict(item)
            adjusted = float(candidate.get("score", 0.0) or 0.0)
            if bool(candidate.get("exact_entry_match", False)):
                adjusted += self.selector_exact_compat_bonus
            elif bool(candidate.get("signature_compatible", False)):
                adjusted += 0.35 * self.selector_exact_compat_bonus
            if str(candidate.get("status", "")) == "active":
                adjusted += self.selector_active_utility_bonus * float(candidate.get("utility", 0.0) or 0.0)
            elif str(candidate.get("status", "")) == "shadow":
                adjusted -= self.selector_shadow_penalty
            candidate["score"] = round(adjusted, 4)
            candidate["selector_tuned_score"] = round(adjusted, 4)
            tuned.append(candidate)
        tuned.sort(
            key=lambda item: (
                float(item.get("score", 0.0)),
                int(bool(item.get("exact_entry_match", False))),
                float(item.get("utility", 0.0)),
                str(item.get("skill_id", "")),
            ),
            reverse=True,
        )
        if tuned and self.selector_low_confidence_abstain_threshold is not None:
            if float(tuned[0].get("score", 0.0) or 0.0) < float(self.selector_low_confidence_abstain_threshold):
                tuned[0]["selector_low_confidence_abstain"] = True
                if (
                    self.direct_skill_only
                    or self.selective_integrated_enabled
                    or self.bifurcated_skill_router_enabled
                    or self.bifurcated_exact_or_seeded_enabled
                ):
                    tuned[0]["direct_eligible"] = False
        return tuned

    def _planner_task_family(self, *, benchmark: str, task_pattern: str) -> str:
        return str(task_pattern or benchmark or "generic")

    def _planner_negative_memory_snapshot(self, *, task_family: str) -> dict[str, int]:
        if not self.planner_negative_memory_enabled:
            return {}
        return {
            skill_key: int(count)
            for (family, skill_key), count in self._planner_negative_memory.items()
            if family == task_family and int(count or 0) > 0
        }

    def _reset_planner_state(self) -> None:
        self._planner_negative_memory.clear()
        self._generator_state = build_generator_state(
            skill_registry=self.skill_registry,
            template_store=self.template_store,
            policy_store=self.policy_store,
            benchmark=self.config.benchmark,
        )

    def _record_planner_negative_memory(self, task, trace) -> None:
        if not self.planner_negative_memory_enabled:
            return
        if str(trace.planning_mode or "") != "skill_seeded_dynamic":
            return
        selected_skill_id = str(trace.selected_skill_id or "").strip()
        if not selected_skill_id:
            return
        selected_row = next(
            (item for item in (trace.retrieval_scores or []) if str(item.get("skill_id", "")) == selected_skill_id),
            {},
        )
        task_family = self._planner_task_family(benchmark=task.benchmark, task_pattern=str(trace.task_pattern or ""))
        final_feedback = trace.verifier_feedback[-1] if trace.verifier_feedback else None
        final_stderr = str(getattr(final_feedback, "stderr", "") or "")
        context = task_compatibility_context(task, runtime_error_text=final_stderr)
        selected_skill_family = str(selected_row.get("skill_family", "") or "")
        selected_entry_point = str(selected_row.get("entry_point", "") or (selected_row.get("signature") or {}).get("entry_point") or "")
        compatibility_reasons = [str(item) for item in (trace.planner_compatibility_veto_reasons or []) if str(item).strip()]
        mismatch = False
        if compatibility_reasons:
            mismatch = True
        elif selected_entry_point and getattr(task, "entry_point", "") and selected_entry_point != str(task.entry_point or "") and not trace.success:
            mismatch = True
        elif "file_path" in set(selected_row.get("environment_affordance") or []) and not context["has_file_path_evidence"]:
            mismatch = True
        elif str(trace.failure_type or "") in {"runtime_error", "generation_error"} and not trace.success:
            mismatch = True
        if not mismatch:
            return
        self._planner_negative_memory[(task_family, selected_skill_id)] = self._planner_negative_memory.get((task_family, selected_skill_id), 0) + 1
        if selected_skill_family:
            key = (task_family, selected_skill_family)
            self._planner_negative_memory[key] = self._planner_negative_memory.get(key, 0) + 1

    def _negative_memory_toxicity_scores(
        self,
        candidate: dict[str, Any] | None,
        *,
        usage_subtype: str = "",
    ) -> tuple[float, float]:
        if candidate is None:
            return 0.0, 0.0
        artifact_id = str(candidate.get("skill_id", "")).strip()
        if not artifact_id:
            return 0.0, 0.0
        high_confidence_score = self.negative_memory_store.toxicity_score_for_artifact(
            artifact_id,
            usage_subtype=usage_subtype or None,
            sources=HIGH_CONFIDENCE_NEGATIVE_SOURCES,
        )
        all_source_score = self.negative_memory_store.toxicity_score_for_artifact(
            artifact_id,
            usage_subtype=usage_subtype or None,
        )
        return high_confidence_score, all_source_score

    def _runtime_governance_block_reason(self, task, candidate: dict[str, Any] | None, usage_subtype: str = "") -> str:
        if self.governance_mode != "full" or candidate is None:
            return ""
        reasons: list[str] = []
        high_confidence_score, _ = self._negative_memory_toxicity_scores(candidate, usage_subtype=usage_subtype)
        if high_confidence_score >= 1.0:
            reasons.append("recent_toxic_score")
        return ",".join(reasons)

    def _policy_context_for_trace(self, *, benchmark: str, task_pattern: str, selected_skill_id: str = "", usage_subtype: str = "") -> dict[str, Any]:
        task_family = self._planner_task_family(benchmark=benchmark, task_pattern=task_pattern)
        if self.disable_policy:
            return {
                "version": 0,
                "direct_skill_delta": 0.0,
                "seeded_delta": 0.0,
                "prompt_directives": [],
                "task_family": task_family,
                "planner_negative_memory": self._planner_negative_memory_snapshot(task_family=task_family),
            }
        context = self.policy_store.context(
            benchmark=benchmark,
            task_pattern=task_pattern,
            selected_skill_id=selected_skill_id,
            usage_subtype=usage_subtype,
        )
        if (
            self.governance_mode == "full"
            and selected_skill_id
            and self.negative_memory_store.toxicity_score_for_artifact(selected_skill_id, usage_subtype=usage_subtype or None) >= 1.0
        ):
            context["direct_skill_delta"] = round(float(context.get("direct_skill_delta", 0.0)) - 0.3, 4)
            context["seeded_delta"] = round(float(context.get("seeded_delta", 0.0)) - 0.3, 4)
            directives = list(context.get("prompt_directives", []))
            if "avoid negatively remembered skill reuse" not in directives:
                directives.append("avoid negatively remembered skill reuse")
            context["prompt_directives"] = directives
        context["task_family"] = task_family
        context["planner_negative_memory"] = self._planner_negative_memory_snapshot(task_family=task_family)
        return self._filter_policy_context(context)

    def _selected_skill_entry_point(self, trace) -> str:
        if not trace.selected_skill_id:
            return ""
        for item in trace.retrieval_scores or []:
            if str(item.get("skill_id", "")) != str(trace.selected_skill_id):
                continue
            signature = item.get("signature", {}) or {}
            return str(item.get("entry_point") or signature.get("entry_point") or "")
        return ""

    def _scaffold_function_inserted(self, task, trace) -> bool:
        if str(trace.planned_usage_subtype or "") != "plan_skeleton":
            return False
        selected_entry_point = self._selected_skill_entry_point(trace)
        task_entry_point = infer_task_entry_point(task)
        if not selected_entry_point or not task_entry_point or selected_entry_point == task_entry_point:
            return False
        pattern = re.compile(rf"\b{re.escape(selected_entry_point)}\s*\(")
        return any(bool(pattern.search(code or "")) for code in trace.code_history)

    def _derive_realized_usage_subtype(self, trace) -> tuple[str, str, int, str]:
        if not trace.selected_skill_id:
            return "", "no_selected_skill", 0, "none"
        if (
            str(trace.planned_usage_subtype or "")
            and (
                float(trace.structured_realization_rate or 0.0) > 0.0
                or int(trace.realized_skill_node_count or 0) > 0
                or str(trace.realization_evidence_source or "") == "structured"
            )
        ):
            realized_subtype = str(trace.realized_usage_subtype or trace.planned_usage_subtype or "")
            insertion_count = max(
                int(trace.skill_node_insertion_count or 0),
                int(trace.realized_skill_node_count or 0),
                1,
            )
            return realized_subtype, "", insertion_count, "structured"
        if trace.planning_mode == "direct_skill":
            if trace.macro_call_executed:
                return "macro_call", "", 1, "structured"
            return "", "direct_skill_not_hit", 0, "none"
        if trace.planning_mode != "skill_seeded_dynamic":
            return "", "planning_mode_not_seeded", 0, "none"

        planned = str(trace.planned_usage_subtype or "")
        if not planned:
            return "", "no_planned_subtype", 0, "none"

        action_text = " ".join(
            f"{action.tool_input_summary} {action.tool_output_summary}" for action in trace.actions
        ).lower()
        code_text = "\n".join(trace.code_history).lower()
        if planned == "plan_skeleton":
            if trace.scaffold_hint_injected or trace.scaffold_function_inserted:
                return "plan_skeleton", "", 1, "structured"
            if any(marker in action_text + " " + code_text for marker in ["subgoal", "step", "helper", "scaffold"]) or trace.skill_mediated_event_count > 0:
                return "plan_skeleton", "", 1, "heuristic"
            return "", "missing_plan_scaffold_evidence", 0, "none"
        if planned == "guard_checker":
            if trace.guard_checker_emitted:
                return "guard_checker", "", 1, "structured"
            if any(marker in action_text + " " + code_text for marker in ["assert ", "raise ", "guard", "check", "verify"]):
                return "guard_checker", "", 1, "heuristic"
            return "", "missing_guard_evidence", 0, "none"
        insertion_count = 1 if trace.skill_mediated_event_count > 0 else 0
        if insertion_count:
            return planned, "", insertion_count, "heuristic"
        return planned, "", 0, "none"

    def _causal_stage_outcome(self, trace) -> str:
        if trace.runtime_skill_block_reason:
            return "runtime_blocked"
        if trace.selected_skill_id and trace.planned_usage_subtype and trace.realized_usage_subtype:
            return "correct_realized_reuse"
        if trace.selected_skill_id and trace.planned_usage_subtype and not trace.realized_usage_subtype:
            return "selected_but_unrealized"
        if trace.planner_visible_candidates and not trace.selected_skill_id:
            return "planner_rejected"
        if trace.recall_candidates and not trace.hard_filter_survivors:
            return "hard_filtered"
        if not trace.recall_candidates:
            return "not_recalled"
        return "unknown"

    def _retrieve_workflow_memories(self, task) -> list[dict[str, Any]]:
        if self.mode != "dynamic":
            return []
        if self.workflow_memory_mode == "disabled" or self.disable_textual_workflow_memory_retrieval:
            return []
        task_pattern = infer_task_pattern(task)
        matched = self.workflow_memory_store.match(task.benchmark, task_pattern, limit=2)
        return [card.model_dump(mode="json") for card in matched]

    def _match_templates(self, task) -> list[dict[str, Any]]:
        if self.mode != "dynamic":
            return []
        task_pattern = infer_task_pattern(task)
        matched = self.template_store.match(task.benchmark, task_pattern, limit=4)
        return [template.model_dump(mode="json") for template in matched]

    def _match_primitives(self, task) -> list[dict[str, Any]]:
        if self.mode != "dynamic" or self.disable_primitive_helper_context:
            return []
        task_pattern = infer_task_pattern(task)
        matched = self.primitive_store.match(task.benchmark, task_pattern, limit=2)
        return [card.model_dump(mode="json") for card in matched]

    def _filter_policy_context(self, policy_context: dict[str, Any]) -> dict[str, Any]:
        if self.disable_policy:
            filtered = dict(policy_context)
            filtered.setdefault("version", 0)
            filtered.setdefault("direct_skill_delta", 0.0)
            filtered.setdefault("seeded_delta", 0.0)
            filtered.setdefault("prompt_directives", [])
            filtered.setdefault("task_family", "")
            filtered.setdefault("planner_negative_memory", {})
            return filtered
        filtered = dict(policy_context)
        directives = [str(item) for item in filtered.get("prompt_directives", []) if str(item).strip()]
        confidence = max(
            abs(float(filtered.get("direct_skill_delta", 0.0))),
            abs(float(filtered.get("seeded_delta", 0.0))),
        )
        min_confidence = self.policy_directive_min_confidence
        if min_confidence is not None and confidence < float(min_confidence):
            directives = []
        max_count = self.policy_directive_max_count
        if max_count is not None:
            directives = directives[: max(0, int(max_count))]
        filtered["prompt_directives"] = directives
        return filtered

    def _enrich_policy_with_generator_state(
        self,
        policy_context: dict[str, Any],
        task_pattern: str,
        retrieved_skill_ids: list[str],
    ) -> dict[str, Any]:
        """Merge GeneratorState's three layers into policy_context for Planner/Generator."""
        gs = self._generator_state
        enriched = dict(policy_context)

        # meta-update layer: stack routing biases (additive, not overwrite)
        enriched["direct_skill_delta"] = round(
            float(enriched.get("direct_skill_delta", 0.0)) + gs.routing_biases.direct_skill_delta, 4
        )
        enriched["seeded_delta"] = round(
            float(enriched.get("seeded_delta", 0.0)) + gs.routing_biases.seeded_delta, 4
        )

        # meta-update layer: merge prompt directives (deduplicated append)
        existing_directives = list(enriched.get("prompt_directives", []))
        for directive in gs.prompt_directives:
            if directive and directive not in existing_directives:
                existing_directives.append(directive)
        enriched["prompt_directives"] = existing_directives

        # action layer: annotate retrieved skills with generator_state priors
        skill_priors: dict[str, dict[str, Any]] = {}
        for skill_id in retrieved_skill_ids:
            entry = gs.skill_action_space.get(skill_id)
            if entry is not None:
                skill_priors[skill_id] = {
                    "gs_role": entry.role,
                    "gs_insertion_mode": entry.insertion_mode,
                    "gs_preferred_usage": entry.preferred_usage,
                    "gs_utility": entry.utility,
                }
        enriched["generator_state_skill_priors"] = skill_priors

        # structure layer: inject matching motifs as extra directives
        for motif in gs.active_motifs:
            if motif.task_pattern == task_pattern and motif.success_rate >= 0.6:
                arrow = " \u2192 "
                skill_chain = arrow.join(motif.skill_sequence)
                motif_directive = (
                    f"prefer workflow motif [{skill_chain}] "
                    f"for task pattern {motif.task_pattern} "
                    f"(support={motif.support_count}, success_rate={motif.success_rate:.2f})"
                )
                if motif_directive not in enriched["prompt_directives"]:
                    enriched["prompt_directives"].append(motif_directive)

        # version tracking
        enriched["generator_state_version"] = gs.version

        return enriched

    def _update_templates(self, trace) -> tuple[list[str], list[str], str]:
        # Core settings that disable template injection should not accumulate
        # template growth and then report it as if it were causal evidence.
        if self.frozen_eval or self.disable_template_growth:
            return [], [], "template_growth_frozen"
        if self.disable_template_injection:
            return [], [], "template_growth_disabled_without_injection"
        updated_templates = []
        summaries: list[str] = []
        injected_template_ids = list(dict.fromkeys(trace.draft_injected_template_ids + trace.repair_injected_template_ids))
        if injected_template_ids:
            injected_updates = self.template_store.record_match_feedback(injected_template_ids, success=trace.success)
            if injected_updates:
                updated_templates.extend(injected_updates)
                summaries.append(
                    f"injected_feedback={'success' if trace.success else 'failure'}:{','.join(template.template_id for template in injected_updates)}"
                )
        if trace.success:
            mined_candidates = self.template_compiler.compile_success_templates(trace)
            observed = self.template_store.observe_candidates(mined_candidates)
            if observed:
                updated_templates.extend(observed)
                summaries.append(f"observed:{','.join(template.template_id for template in observed)}")
        unique_templates = {template.template_id: template for template in updated_templates}
        ids = list(unique_templates)
        kinds = [unique_templates[template_id].template_kind for template_id in ids]
        return ids, kinds, "; ".join(summaries) if summaries else "no_template_change"

    def _update_primitives(self, trace) -> tuple[list[str], bool, str]:
        if self.frozen_eval or self.disable_primitive_growth:
            return [], False, "primitive_growth_frozen"
        if self.disable_executable_artifact_growth or self.disable_primitive_helper_context:
            return [], False, "primitive_growth_disabled"
        updated_cards = []
        summaries: list[str] = []
        if trace.injected_primitive_ids:
            feedback_updates = self.primitive_store.record_feedback(trace.injected_primitive_ids, success=trace.success)
            if feedback_updates:
                updated_cards.extend(feedback_updates)
                summaries.append(
                    f"injected_feedback={'success' if trace.success else 'failure'}:{','.join(card.primitive_id for card in feedback_updates)}"
                )
        admitted = False
        if trace.success:
            candidates = self.primitive_compiler.compile_success_primitives(trace)
            for candidate in candidates:
                updated_cards.append(self.primitive_store.observe_candidate(candidate.card, candidate.code))
            if candidates:
                admitted = True
                summaries.append(f"observed:{','.join(candidate.card.primitive_id for candidate in candidates)}")
        unique_cards = {card.primitive_id: card for card in updated_cards}
        return list(unique_cards), admitted, "; ".join(summaries) if summaries else "no_primitive_change"

    def _update_workflow_memory(self, trace) -> list[str]:
        if self.frozen_eval or self.disable_workflow_memory_growth:
            return []
        if self.workflow_memory_mode == "disabled" or self.disable_textual_workflow_memory_growth:
            return []
        candidate = self.workflow_memory_compiler.compile_success_memory(trace)
        if candidate is None:
            return []
        updated = self.workflow_memory_store.upsert(candidate)
        return [updated.memory_id]

    def _record_negative_memory(self, task, trace, compile_result: CompileAdmissionResult) -> list[str]:
        if self.frozen_eval or self.disable_negative_memory_write:
            return []
        entry_ids: list[str] = []
        lifecycle_phase = str(task.metadata.get("lifecycle_phase", "")) if hasattr(task, "metadata") else ""
        if trace.selected_skill_id and (not trace.success or trace.failure_type):
            entry = self.negative_memory_store.add(
                artifact_id=trace.selected_skill_id,
                artifact_kind="skill",
                benchmark=task.benchmark,
                lifecycle_phase=lifecycle_phase,
                task_pattern=trace.task_pattern,
                planning_mode=trace.planning_mode,
                usage_subtype=trace.realized_usage_subtype or trace.planned_usage_subtype,
                failure_type=trace.failure_type,
                source="runtime_failure",
                summary=(trace.verifier_feedback[-1].stderr if trace.verifier_feedback else "") or trace.failure_type or "runtime failure",
                trace_id=trace.trace_id,
                severity=0.7,
            )
            entry_ids.append(entry.entry_id)
        if compile_result.attempted and not compile_result.admitted and compile_result.skill_id:
            entry = self.negative_memory_store.add(
                artifact_id=compile_result.skill_id,
                artifact_kind="skill",
                benchmark=task.benchmark,
                lifecycle_phase=lifecycle_phase,
                task_pattern=trace.task_pattern,
                planning_mode=trace.planning_mode,
                usage_subtype=trace.realized_usage_subtype or trace.planned_usage_subtype,
                failure_type=trace.compile_rejection_stage,
                source="compile_rejection",
                summary=trace.compile_rejection_reason or "compile rejection",
                trace_id=trace.trace_id,
                severity=0.5,
            )
            entry_ids.append(entry.entry_id)
        for warning in trace.audit_warnings:
            if trace.selected_skill_id:
                entry = self.negative_memory_store.add(
                    artifact_id=trace.selected_skill_id,
                    artifact_kind="skill",
                    benchmark=task.benchmark,
                    lifecycle_phase=lifecycle_phase,
                    task_pattern=trace.task_pattern,
                    planning_mode=trace.planning_mode,
                    usage_subtype=trace.realized_usage_subtype or trace.planned_usage_subtype,
                    failure_type=trace.failure_type,
                    source="trace_audit",
                    summary=warning,
                    trace_id=trace.trace_id,
                    severity=0.4,
                )
                entry_ids.append(entry.entry_id)
        return entry_ids

    def run_episode(self, task) -> dict:
        retrieval_scores, retrieval_diagnostics = self._retrieve(task)
        matched_workflow_memories = self._retrieve_workflow_memories(task)
        matched_templates = self._match_templates(task)
        matched_primitives = self._match_primitives(task)
        task_pattern = infer_task_pattern(task)
        top1 = retrieval_scores[0] if retrieval_scores else None
        top1_skill_id = top1.get("skill_id", "") if top1 else ""
        top1_score = float(top1.get("score", 0.0)) if top1 else 0.0
        selected_skill_before = self.skill_registry.get(top1_skill_id) if top1_skill_id else None
        selected_skill_status_before = selected_skill_before.status if selected_skill_before is not None else ""
        selected_skill_utility_before = float(selected_skill_before.utility) if selected_skill_before is not None else None
        policy_context = self._policy_context_for_trace(
            benchmark=task.benchmark,
            task_pattern=task_pattern,
            selected_skill_id=top1_skill_id,
        )
        # v7 third equation: enrich policy_context with GeneratorState
        policy_context = self._enrich_policy_with_generator_state(
            policy_context,
            task_pattern=task_pattern,
            retrieved_skill_ids=[str(item.get("skill_id", "")) for item in retrieval_scores if item.get("skill_id")],
        )
        # v7 mainline 2: Strategy Injection — inject strategy hints into directives
        _injected_strategy_ids: list[str] = []
        _strategy_holdout = False
        if self.mode == "dynamic" and not self.disable_skill_reuse:
            matched_strategies = self.strategy_registry.match_by_pattern(
                task_pattern=task_pattern, top_k=2,
            )
            # Exploration holdout: 20% chance of skipping injection for
            # ContrastiveEval.  Only when meta_update is active (not frozen).
            if matched_strategies and not self.frozen_eval:
                import random as _rng
                holdout_rng = _rng.Random(hash((task.task_id, self._generator_state.version)))
                _strategy_holdout = holdout_rng.random() < 0.2
            if _strategy_holdout:
                matched_strategies = []
            strategy_texts: list[str] = []
            for strat in matched_strategies:
                if strat.status != "active":
                    continue
                if strat.total_support < 5:
                    # New strategy: insufficient data — 50% exploratory injection
                    import random as _explore_rng
                    if _explore_rng.Random(hash((task.task_id, strat.strategy_id))).random() < 0.5:
                        strategy_texts.append(strat.to_injection_text())
                        _injected_strategy_ids.append(strat.strategy_id)
                elif strat.contrastive_delta > 0:
                    # Verified effective: always inject
                    strategy_texts.append(strat.to_injection_text())
                    _injected_strategy_ids.append(strat.strategy_id)
                # else: delta <= 0 with enough support → skip (verified harmful/neutral)
            if strategy_texts:
                existing_directives = list(policy_context.get("prompt_directives", []))
                existing_directives.append("\n\n".join(strategy_texts))
                policy_context["prompt_directives"] = existing_directives
                policy_context["injected_strategy_ids"] = _injected_strategy_ids
        plan = self.planner.build_plan(
            task=task,
            mode=self.mode,
            retrieved_skills=retrieval_scores,
            matched_templates=matched_templates,
            task_pattern=task_pattern,
            retrieval_threshold=self.retrieval_threshold,
            direct_skill_threshold=self.direct_skill_threshold,
            seed_threshold=self.seed_threshold,
            seed_guard_entry_threshold=self.seed_guard_entry_threshold,
            direct_skill_only=self.direct_skill_only,
            allow_non_exact_direct_reuse=self.allow_non_exact_direct_reuse,
            disable_skill_seeded=self.disable_skill_seeded,
            disable_direct_skill_execution=self.disable_direct_skill_execution,
            seeded_shadow_promotion_margin=self.seeded_shadow_promotion_margin,
            planner_seed_pool_top_k=self.planner_seed_pool_top_k,
            enable_string_seeded_bias=self.enable_string_seeded_bias,
            enable_count_seeded_bias=self.enable_count_seeded_bias,
            compatibility_hard_gate=self.compatibility_hard_gate,
            soft_aware_first=self.soft_aware_first,
            bifurcated_skill_router_enabled=self.bifurcated_skill_router_enabled,
            bifurcated_exact_or_seeded_enabled=self.bifurcated_exact_or_seeded_enabled,
            selective_integrated_enabled=self.selective_integrated_enabled,
            seeded_disable_local_insertion=self.seeded_disable_local_insertion,
            exact_direct_postcall_wrapper_enabled=self.exact_direct_postcall_wrapper_enabled,
            direct_postcall_max_local_patch=self.direct_postcall_max_local_patch,
            policy_context=policy_context,
        )
        original_planned_usage_subtype = plan.planned_usage_subtype
        if plan.selected_skill_id and original_planned_usage_subtype:
            subtype_context = self._policy_context_for_trace(
                benchmark=task.benchmark,
                task_pattern=task_pattern,
                selected_skill_id=plan.selected_skill_id,
                usage_subtype=original_planned_usage_subtype,
            )
            base_directives = [str(item) for item in plan.policy_directives if str(item).strip()]
            for directive in subtype_context.get("prompt_directives", []):
                directive = str(directive).strip()
                if directive and directive not in base_directives:
                    base_directives.append(directive)
            plan.policy_directives = base_directives
        if (
            self.proof_compact_seeded_context
            and (
                self.disable_direct_skill_execution
                or self.bifurcated_skill_router_enabled
                or self.bifurcated_exact_or_seeded_enabled
            )
            and plan.planning_mode == "skill_seeded_dynamic"
            and "proof_compact_seeded_context" not in plan.policy_directives
        ):
            plan.policy_directives = [*plan.policy_directives, "proof_compact_seeded_context"]
        if (
            self.proof_compact_seeded_repair
            and (
                self.disable_direct_skill_execution
                or self.bifurcated_skill_router_enabled
                or self.bifurcated_exact_or_seeded_enabled
            )
            and plan.planning_mode == "skill_seeded_dynamic"
            and "proof_compact_seeded_repair" not in plan.policy_directives
        ):
            plan.policy_directives = [*plan.policy_directives, "proof_compact_seeded_repair"]
        blocked_candidate = next(
            (item for item in retrieval_scores if str(item.get("skill_id", "")) == str(plan.selected_skill_id or "")),
            None,
        )
        high_confidence_toxic_score, all_source_toxic_score = self._negative_memory_toxicity_scores(
            blocked_candidate,
            usage_subtype=original_planned_usage_subtype,
        )
        runtime_skill_block_reason = self._runtime_governance_block_reason(
            task,
            blocked_candidate,
            usage_subtype=original_planned_usage_subtype,
        )
        if runtime_skill_block_reason:
            plan.selection_reason = f"{plan.selection_reason}; runtime_block={runtime_skill_block_reason}"
            plan.planning_mode = "pure_dynamic"
            plan.skill_seeded = False
            plan.seed_source = ""
            plan.selected_skill_id = ""
            plan.planned_usage_subtype = ""
        plan.matched_primitive_ids = [str(item.get("primitive_id", "")) for item in matched_primitives if item.get("primitive_id")]
        plan.matched_workflow_memory_ids = [str(item.get("memory_id", "")) for item in matched_workflow_memories if item.get("memory_id")]
        trace = self.executor.run(
            task=task,
            plan=plan,
            mode=self.mode,
            retrieval_scores=retrieval_scores,
            matched_workflow_memories=matched_workflow_memories,
            matched_templates=matched_templates,
            matched_primitives=matched_primitives,
            template_injection_enabled=not self.disable_template_injection,
        )
        trace.setting_name = self.config.setting_name
        trace.phase_a_bank_source = self.config.phase_a_bank_source
        trace.phase_b_usage_setting = self.config.phase_b_usage_setting
        trace.task_order_seed = self.config.task_order_seed
        trace.phase_a_seed = self.config.phase_a_seed
        trace.seed_routing_mode = plan.seed_routing_mode
        trace.transfer_utility_score = plan.transfer_utility_score
        trace.estimated_seed_token_cost = plan.estimated_seed_token_cost
        trace.historical_positive_transfer = plan.historical_positive_transfer
        trace.negative_transfer_risk = plan.negative_transfer_risk
        trace.planner_compatibility_score = plan.planner_compatibility_score
        trace.planner_compatibility_veto_reasons = list(plan.planner_compatibility_veto_reasons)
        trace.planner_soft_hint_reasons = list(plan.planner_soft_hint_reasons)
        trace.planned_usage_subtype = original_planned_usage_subtype
        trace.runtime_skill_block_reason = runtime_skill_block_reason
        trace.active_skill_bank_at_episode_start = list(retrieval_diagnostics.get("active_skill_bank_at_episode_start", []))
        trace.recall_candidates = list(retrieval_diagnostics.get("recall_candidates", []))
        trace.hard_filter_survivors = list(retrieval_diagnostics.get("hard_filter_survivors", []))
        trace.hard_filter_drop_reasons = dict(retrieval_diagnostics.get("hard_filter_drop_reasons", {}))
        trace.rerank_score_decomposition = dict(retrieval_diagnostics.get("rerank_score_decomposition", {}))
        trace.planner_visible_candidates = list(plan.planner_visible_candidates)
        trace.planner_rejection_reasons = dict(plan.planner_rejection_reasons)
        scaffold_events = [event for event in trace.generation_events if event.generation_stage.startswith("draft") or event.generation_stage.startswith("repair")]
        trace.scaffold_hint_injected = any(bool(event.scaffold_hint_injected) for event in scaffold_events)
        trace.guard_checker_emitted = any(bool(event.guard_checker_emitted) for event in scaffold_events)
        trace.guard_checker_context_injected = any(bool(event.guard_checker_context_injected) for event in scaffold_events)
        trace.guard_checker_feedback_used = any(bool(event.guard_checker_feedback_used) for event in scaffold_events)
        trace.scaffold_suppression_reason = next(
            (str(event.scaffold_suppression_reason) for event in scaffold_events if str(event.scaffold_suppression_reason)),
            "",
        )
        trace.repair_seeded_demoted = any(bool(event.repair_seeded_demoted) for event in scaffold_events)
        trace.repair_demotion_reason = next(
            (str(event.repair_demotion_reason) for event in scaffold_events if str(event.repair_demotion_reason)),
            "",
        )
        trace.macro_call_executed = bool(trace.planning_mode == "direct_skill" and trace.skill_hit)
        trace.scaffold_function_inserted = self._scaffold_function_inserted(task, trace)
        realized_usage_subtype, subtype_gap_reason, insertion_count, evidence_source = self._derive_realized_usage_subtype(trace)
        trace.realized_usage_subtype = realized_usage_subtype
        trace.realization_evidence_source = evidence_source
        if trace.scaffold_suppression_reason and trace.planned_usage_subtype == "plan_skeleton" and not realized_usage_subtype:
            trace.subtype_realization_gap_reason = trace.scaffold_suppression_reason
        else:
            trace.subtype_realization_gap_reason = subtype_gap_reason
        trace.skill_node_insertion_count = insertion_count
        self._rescore_mbpp_trace(task=task, trace=trace)
        if trace.selected_skill_id:
            if trace.planning_mode == "direct_skill":
                trace.direct_use_count = 1
                if trace.success:
                    trace.direct_success_count = 1
                else:
                    trace.direct_failure_count = 1
            elif trace.planning_mode == "skill_seeded_dynamic":
                trace.seeded_use_count = 1
                if trace.success:
                    trace.seeded_success_count = 1
                else:
                    trace.seeded_failure_count = 1
        trace.task_pattern = task_pattern
        trace.episode_provenance = self._episode_provenance(trace)
        trace.policy_version = plan.policy_version
        trace.matched_workflow_memory_ids = plan.matched_workflow_memory_ids
        trace.matched_template_ids = plan.matched_template_ids
        trace.matched_primitive_ids = plan.matched_primitive_ids
        utility_after, status_after = self._record_runtime_governance(task, trace)
        compile_result = self._online_compile(task, trace)
        trace.compiled_skill_admitted = compile_result.admitted
        trace.compiled_skill_admission_bank = compile_result.admission_bank
        if compile_result.attempted:
            trace.replay_validation_count = 1
            if compile_result.skill_id:
                trace.compiled_skill_ids = [compile_result.skill_id]
            if compile_result.provenance:
                trace.compiled_skill_provenance = [compile_result.provenance]
            trace.compiled_skill_status = compile_result.status
            trace.compile_gate_checks = dict(compile_result.checks)
            trace.compile_rejection_stage = compile_result.rejection_stage
            trace.compile_rejection_reason = compile_result.rejection_reason
            trace.compiled_skill_seed_replay_pass_rate = float(compile_result.metrics.get("seed_replay_pass_rate", 0.0) or 0.0)
            trace.compiled_skill_seed_replay_attempt_count = int(compile_result.metrics.get("seed_replay_attempt_count", 0) or 0)
            trace.compiled_skill_seed_replay_pass_count = int(compile_result.metrics.get("seed_replay_pass_count", 0) or 0)
            if not bool(compile_result.checks.get("fresh_world_source_replay", True)):
                trace.replay_failure_count = 1
        # v7 mainline 2: compile StrategyCard from admitted skill
        if compile_result.admitted and compile_result.skill_id and not self.frozen_eval:
            compiled_skill = self.skill_registry.get(compile_result.skill_id)
            if compiled_skill is not None:
                skill_code = ""
                if compile_result.code_path:
                    from pathlib import Path as _Path
                    code_file = _Path(compile_result.code_path)
                    if code_file.exists():
                        skill_code = code_file.read_text(encoding="utf-8")
                strategy = self.strategy_compiler.compile_strategy(
                    trace=trace,
                    skill_card=compiled_skill,
                    skill_code=skill_code,
                )
                if strategy is not None:
                    self.strategy_registry.add(strategy)
        trace.compiled_primitive_ids, trace.compiled_primitive_admitted, trace.primitive_update_summary = self._update_primitives(trace)
        if self.disable_policy:
            updated_policy_version, policy_update_summary = trace.policy_version, "policy_disabled"
        elif self.frozen_eval or self.disable_policy_update or self.freeze_policy_updates:
            updated_policy_version, policy_update_summary = trace.policy_version, "policy_frozen"
        else:
            updated_policy_version, policy_update_summary = self.policy_store.update_after_episode(
                trace=trace,
                selected_skill_status_after=status_after,
            )
        trace.updated_template_ids, trace.updated_template_kinds, trace.template_update_summary = self._update_templates(trace)
        trace.updated_workflow_memory_ids = self._update_workflow_memory(trace)
        trace.utility_score = score_episode(trace)
        if self.governance_mode == "full":
            trace.audit_warnings = audit_trace(trace)
            trace.negative_memory_ids = self._record_negative_memory(task, trace, compile_result)
        else:
            trace.audit_warnings = []
            trace.negative_memory_ids = []
        trace.final_result["utility_distillation"] = self._apply_online_utility_distillation(benchmark=task.benchmark)
        trace.causal_stage_outcome = self._causal_stage_outcome(trace)
        trace.policy_update_summary = f"next_policy_version={updated_policy_version}; {policy_update_summary}"
        self._record_planner_negative_memory(task, trace)
        # v7 mainline 2: ContrastiveEval — record strategy usage outcome
        _strategy_guided = bool(_injected_strategy_ids)
        if not self.frozen_eval and not self.disable_contrastive_eval:
            if _injected_strategy_ids:
                for sid in _injected_strategy_ids:
                    self.strategy_registry.record_guided_usage(sid, success=trace.success)
            else:
                self.strategy_registry.record_unguided_usage(task_pattern, success=trace.success)
        # v7 third equation: update GeneratorState after episode
        if not self.frozen_eval:
            self._generator_state = meta_update(
                state=self._generator_state,
                trace=trace,
                skill_registry=self.skill_registry,
            )
        saved_path = self.trace_store.save(trace)
        final_feedback = trace.verifier_feedback[-1] if trace.verifier_feedback else None
        return {
            "task_id": task.task_id,
            "benchmark": task.benchmark,
            "mode": self.mode,
            "planning_mode": trace.planning_mode,
            "task_pattern": trace.task_pattern,
            "episode_provenance": trace.episode_provenance,
            "passed": trace.success,
            "trace_file": os.fspath(saved_path),
            "attempts": len(trace.code_history),
            "returncode": final_feedback.returncode if final_feedback else None,
            "timeout": final_feedback.timeout if final_feedback else None,
            "failure_type": trace.failure_type,
            "stderr": final_feedback.stderr if final_feedback else "",
            "retrieved_skills": trace.retrieved_skills,
            "retrieval_scores": trace.retrieval_scores,
            "used_skills": trace.used_skills,
            "selected_skill_id": trace.selected_skill_id,
            "skill_seeded": trace.skill_seeded,
            "seed_source": trace.seed_source,
            "selection_reason": trace.selection_reason,
            "active_skill_bank_at_episode_start": trace.active_skill_bank_at_episode_start,
            "recall_candidates": trace.recall_candidates,
            "hard_filter_survivors": trace.hard_filter_survivors,
            "hard_filter_drop_reasons": trace.hard_filter_drop_reasons,
            "rerank_score_decomposition": trace.rerank_score_decomposition,
            "planner_visible_candidates": trace.planner_visible_candidates,
            "planner_rejection_reasons": trace.planner_rejection_reasons,
            "causal_stage_outcome": trace.causal_stage_outcome,
            "planned_usage_subtype": trace.planned_usage_subtype,
            "realized_usage_subtype": trace.realized_usage_subtype,
            "realization_evidence_source": trace.realization_evidence_source,
            "subtype_realization_gap_reason": trace.subtype_realization_gap_reason,
            "macro_call_executed": trace.macro_call_executed,
            "scaffold_hint_injected": trace.scaffold_hint_injected,
            "scaffold_function_inserted": trace.scaffold_function_inserted,
            "guard_checker_emitted": trace.guard_checker_emitted,
            "guard_checker_context_injected": trace.guard_checker_context_injected,
            "guard_checker_feedback_used": trace.guard_checker_feedback_used,
            "scaffold_suppression_reason": trace.scaffold_suppression_reason,
            "repair_seeded_demoted": trace.repair_seeded_demoted,
            "repair_demotion_reason": trace.repair_demotion_reason,
            "runtime_skill_block_reason": trace.runtime_skill_block_reason,
            "trace_audit_only_block_candidate": bool(
                blocked_candidate
                and all_source_toxic_score >= 1.0
                and high_confidence_toxic_score < 1.0
            ),
            "skill_node_insertion_count": trace.skill_node_insertion_count,
            "skill_hit": trace.skill_hit,
            "fallback_triggered": trace.fallback_triggered,
            "utility_before": trace.utility_before,
            "utility_after": utility_after,
            "status_after": status_after,
            "latency_ms": trace.latency_ms,
            "disable_retrieval": self.disable_retrieval,
            "direct_skill_only": self.direct_skill_only,
            "disable_skill_seeded": self.disable_skill_seeded,
            "disable_governance": self.disable_governance,
            "disable_policy": self.disable_policy,
            "top1_skill_id": top1_skill_id,
            "top1_score": top1_score,
            "selected_skill_status_before": selected_skill_status_before,
            "selected_skill_status_after": status_after or selected_skill_status_before,
            "selected_skill_utility_before": selected_skill_utility_before,
            "selected_skill_utility_after": utility_after if utility_after is not None else selected_skill_utility_before,
            "llm_prompt_tokens_total": trace.llm_prompt_tokens_total,
            "llm_completion_tokens_total": trace.llm_completion_tokens_total,
            "llm_total_tokens_total": trace.llm_total_tokens_total,
            "llm_latency_ms_total": trace.llm_latency_ms_total,
            "llm_call_count": trace.llm_call_count,
            "compiled_skill_ids": trace.compiled_skill_ids,
            "compiled_skill_provenance": trace.compiled_skill_provenance,
            "compiled_skill_admitted": trace.compiled_skill_admitted,
            "compiled_skill_admission_bank": trace.compiled_skill_admission_bank,
            "compiled_skill_status": trace.compiled_skill_status,
            "compile_gate_checks": trace.compile_gate_checks,
            "compile_rejection_stage": trace.compile_rejection_stage,
            "compile_rejection_reason": trace.compile_rejection_reason,
            "compiled_skill_seed_replay_pass_rate": trace.compiled_skill_seed_replay_pass_rate,
            "compiled_skill_seed_replay_attempt_count": trace.compiled_skill_seed_replay_attempt_count,
            "compiled_skill_seed_replay_pass_count": trace.compiled_skill_seed_replay_pass_count,
            "policy_version": trace.policy_version,
            "policy_update_summary": trace.policy_update_summary,
            "matched_workflow_memory_ids": trace.matched_workflow_memory_ids,
            "injected_workflow_memory_ids": trace.injected_workflow_memory_ids,
            "updated_workflow_memory_ids": trace.updated_workflow_memory_ids,
            "matched_template_ids": trace.matched_template_ids,
            "draft_injected_template_ids": trace.draft_injected_template_ids,
            "repair_injected_template_ids": trace.repair_injected_template_ids,
            "updated_template_ids": trace.updated_template_ids,
            "updated_template_kinds": trace.updated_template_kinds,
            "template_update_summary": trace.template_update_summary,
            "matched_primitive_ids": trace.matched_primitive_ids,
            "injected_primitive_ids": trace.injected_primitive_ids,
            "compiled_primitive_ids": trace.compiled_primitive_ids,
            "compiled_primitive_admitted": trace.compiled_primitive_admitted,
            "primitive_update_summary": trace.primitive_update_summary,
            "utility_score": trace.utility_score,
            "audit_warnings": trace.audit_warnings,
            "negative_memory_ids": trace.negative_memory_ids,
            "first_attempt_success": trace.first_attempt_success,
            "retry_count": trace.retry_count,
            "operational_event_count": trace.operational_event_count,
            "skill_mediated_event_count": trace.skill_mediated_event_count,
            "skill_mediated_event_rate": trace.skill_mediated_event_rate,
            "draft_prompt_tokens": trace.draft_prompt_tokens,
            "seed_context_tokens": trace.seed_context_tokens,
            "scaffold_tokens": trace.scaffold_tokens,
            "seed_context_skill_count": trace.seed_context_skill_count,
            "seed_context_mode": trace.seed_context_mode,
            "seed_routing_mode": trace.seed_routing_mode,
            "transfer_utility_score": trace.transfer_utility_score,
            "estimated_seed_token_cost": trace.estimated_seed_token_cost,
            "historical_positive_transfer": trace.historical_positive_transfer,
            "negative_transfer_risk": trace.negative_transfer_risk,
            "planner_compatibility_score": trace.planner_compatibility_score,
            "planner_compatibility_veto_reasons": trace.planner_compatibility_veto_reasons,
            "planner_soft_hint_reasons": trace.planner_soft_hint_reasons,
            "seed_context_char_count": trace.seed_context_char_count,
            "seed_context_contains_code": trace.seed_context_contains_code,
            "seed_context_contains_tests": trace.seed_context_contains_tests,
            "repair_prompt_tokens": trace.repair_prompt_tokens,
            "repair_round_count": trace.repair_round_count,
            "local_patch_repair_count": trace.local_patch_repair_count,
            "full_rewrite_repair_count": trace.full_rewrite_repair_count,
            "repair_failure_types": trace.repair_failure_types,
            "artifact_mediated": trace.artifact_mediated,
            "direct_use_count": trace.direct_use_count,
            "direct_success_count": trace.direct_success_count,
            "direct_failure_count": trace.direct_failure_count,
            "seeded_use_count": trace.seeded_use_count,
            "seeded_success_count": trace.seeded_success_count,
            "seeded_failure_count": trace.seeded_failure_count,
            "replay_validation_count": trace.replay_validation_count,
            "replay_failure_count": trace.replay_failure_count,
            "workflow_memory_mode": self.workflow_memory_mode,
            "governance_mode": self.governance_mode,
            "generator_state_version": self._generator_state.version,
            "injected_strategy_ids": _injected_strategy_ids,
            "strategy_guided": _strategy_guided,
            "strategy_holdout": _strategy_holdout,
        }

    def run_tasks(self, tasks: list[Any]) -> dict[str, Any]:
        episodes: list[dict[str, Any]] = []
        for index, task in enumerate(tasks):
            if self.planner_state_reset == "per_task":
                self._reset_planner_state()
            elif self.planner_state_reset == "every_5_tasks" and index > 0 and index % 5 == 0:
                self._reset_planner_state()
            episodes.append(self.run_episode(task))
        audit_results = []
        if self.governance_mode == "full" and self.audit_top_k > 0:
            lifecycle_phase = str(tasks[0].metadata.get("lifecycle_phase", "")) if tasks else ""
            audit_results = self.governance.periodic_audit(top_k=self.audit_top_k, lifecycle_phase=lifecycle_phase)
        passed = sum(1 for episode in episodes if episode["passed"])
        return {
            "benchmark": self.config.benchmark,
            "profile": self.config.profile,
            "mode": self.mode,
            "num_tasks": len(tasks),
            "num_passed": passed,
            "num_failed": len(tasks) - passed,
            "episodes": episodes,
            "audit_results": audit_results,
            "config": asdict(self.config),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FlowEvo benchmark runner")
    parser.add_argument("--benchmark", choices=["humaneval"], required=True)
    parser.add_argument("--dataset-split", default="")
    parser.add_argument("--profile", choices=["full"], default="full")
    parser.add_argument("--mode", choices=["oracle", "dynamic"], default=None)
    parser.add_argument("--oracle", action="store_true", help="Backward-compatible alias for --mode oracle.")
    parser.add_argument("--limit", type=int, default=3, help="Number of tasks to run.")
    parser.add_argument("--timeout", type=float, default=5.0, help="Sandbox timeout in seconds.")
    parser.add_argument("--retrieval-top-k", type=int, default=3)
    parser.add_argument("--retrieval-threshold", type=float, default=5.0)
    parser.add_argument("--direct-skill-threshold", type=float, default=None)
    parser.add_argument("--seed-threshold", type=float, default=None)
    parser.add_argument("--audit-top-k", type=int, default=0)
    parser.add_argument("--disable-retrieval", action="store_true")
    parser.add_argument("--direct-skill-only", action="store_true")
    parser.add_argument("--allow-non-exact-direct-reuse", action="store_true")
    parser.add_argument("--disable-direct-skill-execution", action="store_true")
    parser.add_argument("--disable-skill-seeded", action="store_true")
    parser.add_argument("--disable-governance", action="store_true")
    parser.add_argument("--disable-policy", action="store_true")
    parser.add_argument("--frozen-eval", action="store_true")
    parser.add_argument("--disable-policy-update", action="store_true")
    parser.add_argument("--disable-template-injection", action="store_true")
    parser.add_argument("--disable-template-growth", action="store_true")
    parser.add_argument("--disable-primitive-helper-context", action="store_true")
    parser.add_argument("--disable-primitive-growth", action="store_true")
    parser.add_argument("--disable-executable-artifact-growth", action="store_true")
    parser.add_argument("--disable-textual-workflow-memory-growth", action="store_true")
    parser.add_argument("--disable-workflow-memory-growth", action="store_true")
    parser.add_argument("--disable-textual-workflow-memory-retrieval", action="store_true")
    parser.add_argument("--disable-negative-memory-write", action="store_true")
    parser.add_argument("--disable-skill-reuse", action="store_true")
    parser.add_argument("--workflow-memory-path", default="data/tasks/workflow_memory.json")
    parser.add_argument("--workflow-memory-mode", choices=["disabled", "rules_text_summary"], default="disabled")
    parser.add_argument("--governance-mode", choices=["disabled", "passive", "full"], default="disabled")
    parser.add_argument("--trace-root", default="data/traces")
    parser.add_argument("--registry-path", default="data/skills/registry.json")
    parser.add_argument("--config-path", default="configs/default.yaml")
    parser.add_argument("--local-config-path", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--skill-top-k", type=int, default=None)
    parser.add_argument("--draft-total-tokens", type=int, default=None)
    parser.add_argument("--repair-total-tokens", type=int, default=None)
    parser.add_argument("--draft-per-skill-tokens", type=int, default=None)
    parser.add_argument("--repair-per-skill-tokens", type=int, default=None)
    parser.add_argument("--policy-directive-min-confidence", type=float, default=None)
    parser.add_argument("--policy-directive-max-count", type=int, default=None)
    parser.add_argument("--policy-path", default="data/policy/state.json")
    parser.add_argument("--template-path", default="data/tasks/templates.json")
    parser.add_argument("--primitive-path", default="data/primitives/registry.json")
    parser.add_argument("--negative-memory-path", default="data/governance/negative_memory.json")
    return parser.parse_args()


def resolve_mode(args: argparse.Namespace) -> str:
    if args.mode is not None:
        return args.mode
    if args.oracle:
        return "oracle"
    return "dynamic"


def build_runner_config(args: argparse.Namespace) -> RunnerConfig:
    return RunnerConfig(
        mode=resolve_mode(args),
        benchmark=args.benchmark,
        dataset_split=args.dataset_split,
        profile=args.profile,
        limit=args.limit,
        timeout_seconds=args.timeout,
        retrieval_top_k=args.retrieval_top_k,
        retrieval_threshold=args.retrieval_threshold,
        direct_skill_threshold=args.direct_skill_threshold,
        seed_threshold=args.seed_threshold,
        audit_top_k=args.audit_top_k,
        disable_retrieval=args.disable_retrieval,
        direct_skill_only=args.direct_skill_only,
        allow_non_exact_direct_reuse=args.allow_non_exact_direct_reuse,
        disable_direct_skill_execution=args.disable_direct_skill_execution,
        disable_skill_seeded=args.disable_skill_seeded,
        disable_governance=args.disable_governance,
        disable_policy=args.disable_policy,
        frozen_eval=args.frozen_eval,
        disable_policy_update=args.disable_policy_update,
        normalize_retrieval_ties_for_planner=getattr(args, "normalize_retrieval_ties_for_planner", False),
        preserve_prompt_skill_order=getattr(args, "preserve_prompt_skill_order", False),
        disable_template_injection=args.disable_template_injection,
        disable_template_growth=args.disable_template_growth,
        disable_primitive_helper_context=args.disable_primitive_helper_context,
        disable_primitive_growth=args.disable_primitive_growth,
        disable_executable_artifact_growth=args.disable_executable_artifact_growth,
        disable_textual_workflow_memory_growth=args.disable_textual_workflow_memory_growth,
        disable_workflow_memory_growth=args.disable_workflow_memory_growth,
        disable_textual_workflow_memory_retrieval=args.disable_textual_workflow_memory_retrieval,
        disable_negative_memory_write=args.disable_negative_memory_write,
        disable_skill_reuse=args.disable_skill_reuse,
        trace_root=args.trace_root,
        registry_path=args.registry_path,
        config_path=args.config_path,
        local_config_path=args.local_config_path,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        skill_top_k=args.skill_top_k,
        draft_total_tokens=args.draft_total_tokens,
        repair_total_tokens=args.repair_total_tokens,
        draft_per_skill_tokens=args.draft_per_skill_tokens,
        repair_per_skill_tokens=args.repair_per_skill_tokens,
        policy_path=args.policy_path,
        template_path=args.template_path,
        primitive_path=args.primitive_path,
        negative_memory_path=args.negative_memory_path,
        workflow_memory_path=args.workflow_memory_path,
        workflow_memory_mode=args.workflow_memory_mode,
        governance_mode=args.governance_mode,
        policy_directive_min_confidence=args.policy_directive_min_confidence,
        policy_directive_max_count=args.policy_directive_max_count,
    )


def main() -> None:
    args = parse_args()
    config = build_runner_config(args)
    from env.benchmark_adapter import load_benchmark_tasks

    tasks = load_benchmark_tasks(
        benchmark=args.benchmark,
        profile=args.profile,
        limit=args.limit,
        dataset_split=args.dataset_split or "",
    )
    runner = BenchmarkRunner(config=config)
    result = runner.run_tasks(tasks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
