"""Workflow planners for FlowEvo.

This module now provides a minimal benchmark-oriented planner for the dynamic
workflow baseline while keeping the older generic planner available.
"""

from __future__ import annotations

import ast
import re
import uuid
from typing import Any

from agent.skill_compatibility import compatibility_assessment
from core.schemas import CodeTaskInstance, TaskInstance, WorkflowPlan, WorkflowStep


class Planner:
    """Create workflow plans for the earlier generic MVP flow."""

    def build_plan(self, task: TaskInstance, retrieved_skills: list[str] = None) -> WorkflowPlan:
        """Build a simple fixed plan for a generic task."""
        skills = retrieved_skills or []
        return WorkflowPlan(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            task_id=task.task_id,
            retrieved_skills=skills,
            steps=[
                WorkflowStep(step_id="s1", step_type="inspect_schema", payload={"tables": task.table_paths}),
                WorkflowStep(step_id="s2", step_type="write_code", payload={"target": "solution.py"}),
                WorkflowStep(step_id="s3", step_type="run_code", payload={"entry": "solution.py"}),
                WorkflowStep(step_id="s4", step_type="finalize", payload={"format": "text"}),
            ],
        )


class CodeTaskPlanner:
    """Build plans for oracle and dynamic code-task execution."""

    _NO_SEED_THRESHOLD = 2.6
    _SKELETON_THRESHOLD = 3.5
    _CODE_EXCERPT_THRESHOLD = 4.6
    _TOP2_MARGIN = 0.3
    _ROUTING_MODE_RANK = {
        "no_seed": 0,
        "guard_only_seed": 1,
        "skeleton_only_seed": 2,
        "code_excerpt_seed": 3,
    }
    _COMPATIBILITY_DIRECT_THRESHOLD = 0.55
    _COMPATIBILITY_SOFT_ONLY_THRESHOLD = 0.8
    _BIFURCATED_DIRECT_MARGIN = 0.5

    _COLLECTION_OR_TEXT = {"str", "list", "tuple", "dict", "set"}
    _SCALAR_ONLY = {"number", "bool"}
    _COUNT_GEOMETRY_KEYWORDS = (
        "area",
        "circle",
        "cylinder",
        "pentagon",
        "polygon",
        "radius",
        "rectangle",
        "rectangles",
        "triangle",
        "volume",
    )
    _STRING_QUERY_KEYWORDS = (
        "character",
        "characters",
        "letter",
        "letters",
        "lowercase",
        "lower case",
        "string",
        "strings",
        "uppercase",
        "upper case",
        "word",
        "words",
    )
    _STRING_HELPER_KEYWORDS = {
        "char",
        "chars",
        "character",
        "characters",
        "lower",
        "lowercase",
        "remove",
        "split",
        "string",
        "strings",
        "upper",
        "uppercase",
        "word",
        "words",
    }
    _NON_STRING_GENERIC_KEYWORDS = {
        "area",
        "bit",
        "cylinder",
        "divisor",
        "flip",
        "pentagon",
        "power",
        "sum",
        "volume",
    }
    _ARITHMETIC_SEARCH_HELPER_KEYWORDS = {
        "divisor",
        "divisors",
        "factor",
        "factors",
        "gcd",
        "lcm",
        "multiple",
        "multiples",
        "prime",
        "primes",
    }

    def _annotation_kind(self, annotation: object) -> str:
        text = str(annotation or "").strip().lower()
        if not text:
            return ""
        if "str" in text:
            return "str"
        if any(token in text for token in ("int", "float", "number")):
            return "number"
        if "bool" in text:
            return "bool"
        if "list" in text:
            return "list"
        if "tuple" in text:
            return "tuple"
        if "dict" in text:
            return "dict"
        if "set" in text:
            return "set"
        return ""

    def _literal_kind(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return "str"
            if isinstance(node.value, bool):
                return "bool"
            if isinstance(node.value, (int, float)):
                return "number"
            return ""
        if isinstance(node, ast.List):
            return "list"
        if isinstance(node, ast.Tuple):
            return "tuple"
        if isinstance(node, ast.Dict):
            return "dict"
        if isinstance(node, ast.Set):
            return "set"
        return ""

    def _assert_literal_profile(self, raw_test: str, *, entry_point: str | None = None) -> tuple[set[str], set[str]]:
        arg_kinds: set[str] = set()
        return_kinds: set[str] = set()
        try:
            module = ast.parse(raw_test)
        except SyntaxError:
            return arg_kinds, return_kinds
        for stmt in module.body:
            if not isinstance(stmt, ast.Assert):
                continue
            test_expr = stmt.test
            if not isinstance(test_expr, ast.Compare) or not isinstance(test_expr.left, ast.Call):
                continue
            call = test_expr.left
            func = call.func
            if isinstance(func, ast.Name) and entry_point and func.id != entry_point:
                continue
            for arg in call.args:
                kind = self._literal_kind(arg)
                if kind:
                    arg_kinds.add(kind)
            for comparator in test_expr.comparators:
                kind = self._literal_kind(comparator)
                if kind:
                    return_kinds.add(kind)
        return arg_kinds, return_kinds

    def _task_literal_profile(self, task: CodeTaskInstance) -> tuple[set[str], set[str]]:
        arg_kinds: set[str] = set()
        return_kinds: set[str] = set()
        for raw_test in task.test_list:
            test_args, test_returns = self._assert_literal_profile(raw_test, entry_point=task.entry_point)
            arg_kinds.update(test_args)
            return_kinds.update(test_returns)
        return arg_kinds, return_kinds

    def _candidate_literal_profile(self, candidate: dict[str, Any]) -> tuple[set[str], set[str]]:
        arg_kinds: set[str] = set()
        return_kinds: set[str] = set()
        signature = candidate.get("signature") or {}
        if isinstance(signature, dict):
            for item in signature.get("args", []):
                if isinstance(item, dict):
                    kind = self._annotation_kind(item.get("annotation"))
                    if kind:
                        arg_kinds.add(kind)
            return_kind = self._annotation_kind(signature.get("returns"))
            if return_kind:
                return_kinds.add(return_kind)
        entry_point = ""
        if isinstance(signature, dict):
            entry_point = str(signature.get("entry_point") or "")
        if not entry_point:
            entry_point = str(candidate.get("entry_point") or "")
        for test in candidate.get("tests", []):
            if not isinstance(test, dict):
                continue
            test_args, test_returns = self._assert_literal_profile(str(test.get("code") or ""), entry_point=entry_point or None)
            arg_kinds.update(test_args)
            return_kinds.update(test_returns)
        return arg_kinds, return_kinds

    def _task_query_text(self, task: CodeTaskInstance) -> str:
        return " ".join(part for part in (task.text, task.prompt) if part).strip().lower()

    def _text_tokens(self, text: str) -> set[str]:
        return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}", text.lower()))

    def _candidate_semantic_tokens(self, candidate: dict[str, Any]) -> set[str]:
        tokens: set[str] = set()
        entry_point = str((candidate.get("signature") or {}).get("entry_point") or candidate.get("entry_point") or "")
        tokens.update(self._text_tokens(entry_point))
        for test in candidate.get("tests", []):
            if isinstance(test, dict):
                tokens.update(self._text_tokens(str(test.get("code") or "")))
        return tokens

    def _obvious_seed_type_mismatch(self, task: CodeTaskInstance, candidate: dict[str, Any]) -> bool:
        task_arg_kinds, task_return_kinds = self._task_literal_profile(task)
        if not task_arg_kinds and not task_return_kinds:
            return False
        candidate_arg_kinds, candidate_return_kinds = self._candidate_literal_profile(candidate)
        if not candidate_arg_kinds and not candidate_return_kinds:
            return False
        if task_arg_kinds & self._COLLECTION_OR_TEXT and candidate_arg_kinds and candidate_arg_kinds <= self._SCALAR_ONLY:
            return True
        if task_return_kinds & self._COLLECTION_OR_TEXT and candidate_return_kinds and candidate_return_kinds <= self._SCALAR_ONLY:
            return True
        if task_arg_kinds and candidate_arg_kinds and task_arg_kinds.isdisjoint(candidate_arg_kinds):
            return True
        if task_return_kinds and candidate_return_kinds and task_return_kinds.isdisjoint(candidate_return_kinds):
            return True
        return False

    def _seeded_rank_adjustment(
        self,
        task: CodeTaskInstance,
        candidate: dict[str, Any],
        *,
        disable_direct_skill_execution: bool = False,
        enable_string_seeded_bias: bool = False,
        enable_count_seeded_bias: bool = False,
    ) -> float:
        task_text = self._task_query_text(task)
        candidate_tokens = self._candidate_semantic_tokens(candidate)
        preconditions = candidate.get("preconditions") or {}
        candidate_pattern = str(preconditions.get("task_pattern") or candidate.get("task_pattern") or "")
        adjustment = 0.0
        if enable_string_seeded_bias and any(keyword in task_text for keyword in self._STRING_QUERY_KEYWORDS):
            if candidate_tokens & self._STRING_HELPER_KEYWORDS:
                adjustment += 0.12
            if candidate_tokens & self._NON_STRING_GENERIC_KEYWORDS and not candidate_tokens & self._STRING_HELPER_KEYWORDS:
                adjustment -= 0.08
        if disable_direct_skill_execution and str(candidate.get("status", "")) == "shadow":
            seed_replay_pass_rate = float(candidate.get("seed_replay_pass_rate", 0.0) or 0.0)
            seed_replay_attempt_count = int(candidate.get("seed_replay_attempt_count", 0) or 0)
            if seed_replay_attempt_count > 0 and seed_replay_pass_rate > 0.0:
                adjustment += min(0.35, 0.2 * seed_replay_pass_rate + 0.03 * min(seed_replay_attempt_count, 5))
        if not enable_count_seeded_bias:
            return adjustment
        task_arg_kinds, task_return_kinds = self._task_literal_profile(task)
        if str(candidate.get("status", "")) != "shadow":
            return adjustment
        if str(candidate.get("provenance", "")) != "seeded_dynamic_success":
            return adjustment
        if candidate_pattern not in {"function_synthesis", "generic_function_synthesis", "mbpp_general"}:
            return adjustment
        if "count" not in task_text:
            return adjustment
        if any(keyword in task_text for keyword in self._COUNT_GEOMETRY_KEYWORDS):
            return adjustment
        if task_arg_kinds and not task_arg_kinds <= {"number"}:
            return adjustment
        if task_return_kinds and not task_return_kinds <= {"number"}:
            return adjustment
        return adjustment - 0.05

    def _effective_seed_score(
        self,
        candidate: dict[str, Any],
        *,
        disable_direct_skill_execution: bool,
        seeded_shadow_promotion_margin: float,
    ) -> float:
        score = float(candidate.get("score", 0.0))
        status = str(candidate.get("status", ""))
        if disable_direct_skill_execution and status == "shadow":
            return score + max(0.0, seeded_shadow_promotion_margin)
        return score

    def _planned_usage_subtype(self, planning_mode: str, selected_skill: dict[str, Any] | None) -> str:
        if planning_mode == "direct_skill":
            return "macro_call"
        if planning_mode != "skill_seeded_dynamic" or selected_skill is None:
            return ""
        routing_mode = str(selected_skill.get("seed_routing_mode", "") or "")
        if routing_mode == "guard_only_seed":
            return "guard_checker"
        if routing_mode in {"skeleton_only_seed", "code_excerpt_seed"}:
            return "plan_skeleton"
        preferred = [str(item) for item in (selected_skill.get("preferred_usage_subtypes") or []) if str(item)]
        if "plan_skeleton" in preferred:
            return "plan_skeleton"
        if "guard_checker" in preferred:
            return "guard_checker"
        if bool(selected_skill.get("exact_entry_match", False)):
            return "plan_skeleton"
        return "guard_checker"

    def _seed_rank_tuple(
        self,
        task: CodeTaskInstance,
        candidate: dict[str, Any],
        *,
        disable_direct_skill_execution: bool,
        seeded_shadow_promotion_margin: float,
        enable_string_seeded_bias: bool,
        enable_count_seeded_bias: bool,
    ) -> tuple[float, float, float, float]:
        effective_score = self._effective_seed_score(
            candidate,
            disable_direct_skill_execution=disable_direct_skill_execution,
            seeded_shadow_promotion_margin=seeded_shadow_promotion_margin,
        )
        seed_score = effective_score + self._seeded_rank_adjustment(
            task,
            candidate,
            disable_direct_skill_execution=disable_direct_skill_execution,
            enable_string_seeded_bias=enable_string_seeded_bias,
            enable_count_seeded_bias=enable_count_seeded_bias,
        )
        if bool(candidate.get("direct_eligible", False)) and task.benchmark != "gsm8k" and not disable_direct_skill_execution:
            seed_score -= 0.25
        shadow_seed_confidence = 0.0
        if str(candidate.get("status", "")) == "shadow":
            shadow_seed_confidence = float(candidate.get("seed_replay_pass_rate", 0.0) or 0.0)
        return (
            seed_score,
            shadow_seed_confidence,
            float(candidate.get("compat", 0.0)),
            float(candidate.get("utility", 0.0)),
        )

    def _seed_context_skill_modes(
        self,
        task: CodeTaskInstance,
        *,
        planning_mode: str,
        retrieved: list[dict[str, Any]],
        selected_skill: dict[str, Any] | None,
        selected_routing_mode: str,
    ) -> dict[str, str]:
        if planning_mode != "skill_seeded_dynamic" or selected_skill is None or selected_routing_mode == "no_seed":
            return {}
        selected_id = str(selected_skill.get("skill_id", "")).strip()
        if not selected_id:
            return {}
        ordered: list[dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in [selected_skill, *retrieved]:
            skill_id = str(candidate.get("skill_id", "")).strip()
            if not skill_id or skill_id in seen:
                continue
            seen.add(skill_id)
            ordered.append(candidate)
        seed_modes: dict[str, str] = {selected_id: selected_routing_mode}
        top_utility = float(selected_skill.get("transfer_utility_score", 0.0))
        for candidate in ordered[1:]:
            if len(seed_modes) >= 2:
                break
            candidate_id = str(candidate.get("skill_id", "")).strip()
            if not candidate_id:
                continue
            if self._obvious_seed_type_mismatch(task, candidate):
                continue
            if float(top_utility - float(candidate.get("transfer_utility_score", 0.0))) > self._TOP2_MARGIN:
                continue
            seed_modes[candidate_id] = "guard_only_seed"
        return seed_modes

    def _merge_rejection_reasons(
        self,
        primary: dict[str, list[str]],
        secondary: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        merged: dict[str, list[str]] = {str(key): list(value) for key, value in primary.items()}
        for key, reasons in secondary.items():
            bucket = merged.setdefault(str(key), [])
            for reason in reasons:
                text = str(reason).strip()
                if text and text not in bucket:
                    bucket.append(text)
        return merged

    def _bifurcated_direct_view(
        self,
        retrieved: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
        direct_view = [dict(candidate) for candidate in retrieved]
        direct_candidates = [
            candidate
            for candidate in direct_view
            if bool(candidate.get("direct_eligible", False))
            and (bool(candidate.get("exact_entry_match", False)) or bool(candidate.get("signature_compatible", False)))
        ]
        rejection_reasons: dict[str, list[str]] = {}
        if not direct_candidates:
            return direct_view, rejection_reasons
        top = direct_candidates[0]
        top_score = float(top.get("score", 0.0) or 0.0)
        second_score = float(direct_candidates[1].get("score", 0.0) or 0.0) if len(direct_candidates) > 1 else None
        high_confidence = bool(top.get("exact_entry_match", False))
        if not high_confidence and not bool(top.get("selector_low_confidence_abstain", False)):
            high_confidence = second_score is None or (top_score - second_score) >= self._BIFURCATED_DIRECT_MARGIN
        top_skill_id = str(top.get("skill_id", "")).strip()
        if high_confidence and top_skill_id:
            for candidate in direct_view:
                skill_id = str(candidate.get("skill_id", "")).strip()
                if (
                    skill_id
                    and skill_id != top_skill_id
                    and bool(candidate.get("direct_eligible", False))
                    and (bool(candidate.get("exact_entry_match", False)) or bool(candidate.get("signature_compatible", False)))
                ):
                    candidate["direct_eligible"] = False
                    rejection_reasons.setdefault(skill_id, []).append("bifurcated_direct_lower_ranked_candidate")
            return direct_view, rejection_reasons
        for candidate in direct_view:
            skill_id = str(candidate.get("skill_id", "")).strip()
            if not skill_id:
                continue
            if bool(candidate.get("direct_eligible", False)) and (
                bool(candidate.get("exact_entry_match", False)) or bool(candidate.get("signature_compatible", False))
            ):
                candidate["direct_eligible"] = False
                rejection_reasons.setdefault(skill_id, []).append("bifurcated_direct_ambiguous_or_low_confidence")
        return direct_view, rejection_reasons

    def _exact_or_seeded_direct_view(
        self,
        retrieved: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
        direct_view = [dict(candidate) for candidate in retrieved]
        rejection_reasons: dict[str, list[str]] = {}
        exact_candidates = [
            candidate
            for candidate in direct_view
            if bool(candidate.get("direct_eligible", False)) and bool(candidate.get("exact_entry_match", False))
        ]
        if not exact_candidates:
            for candidate in direct_view:
                skill_id = str(candidate.get("skill_id", "")).strip()
                if not skill_id:
                    continue
                if bool(candidate.get("direct_eligible", False)) and bool(candidate.get("signature_compatible", False)):
                    candidate["direct_eligible"] = False
                    rejection_reasons.setdefault(skill_id, []).append("exact_or_seeded_non_exact_direct_demoted")
            return direct_view, rejection_reasons

        exact_candidates.sort(
            key=lambda item: (
                float(item.get("score", 0.0) or 0.0),
                float(item.get("utility", 0.0) or 0.0),
                str(item.get("skill_id", "")),
            ),
            reverse=True,
        )
        top = exact_candidates[0]
        top_score = float(top.get("score", 0.0) or 0.0)
        second_score = float(exact_candidates[1].get("score", 0.0) or 0.0) if len(exact_candidates) > 1 else None
        top_skill_id = str(top.get("skill_id", "")).strip()
        high_confidence = not bool(top.get("selector_low_confidence_abstain", False))
        if high_confidence and second_score is not None:
            high_confidence = (top_score - second_score) >= self._BIFURCATED_DIRECT_MARGIN
        if high_confidence and top_skill_id:
            for candidate in direct_view:
                skill_id = str(candidate.get("skill_id", "")).strip()
                if not skill_id or skill_id == top_skill_id:
                    continue
                if bool(candidate.get("direct_eligible", False)):
                    candidate["direct_eligible"] = False
                    rejection_reasons.setdefault(skill_id, []).append("exact_or_seeded_direct_lower_ranked_candidate")
            return direct_view, rejection_reasons

        for candidate in direct_view:
            skill_id = str(candidate.get("skill_id", "")).strip()
            if not skill_id:
                continue
            if bool(candidate.get("direct_eligible", False)):
                candidate["direct_eligible"] = False
                rejection_reasons.setdefault(skill_id, []).append("exact_or_seeded_direct_ambiguous_or_low_confidence")
        return direct_view, rejection_reasons

    def _decide_bifurcated_mode(
        self,
        task: CodeTaskInstance,
        retrieved: list[dict[str, Any]],
        benchmark: str,
        direct_skill_threshold: float,
        seed_threshold: float,
        allow_non_exact_direct_reuse: bool,
        disable_skill_seeded: bool,
        seeded_shadow_promotion_margin: float,
        planner_seed_pool_top_k: int | None,
        enable_string_seeded_bias: bool,
        enable_count_seeded_bias: bool,
        seed_guard_entry_threshold: float | None,
        compatibility_hard_gate: bool,
        policy_context: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any] | None, str, dict[str, list[str]], dict[str, Any]]:
        direct_view, direct_only_rejections = self._bifurcated_direct_view(retrieved)
        direct_mode, selected_skill, selection_reason, direct_rejections, utility_breakdown = self._decide_planning_mode(
            task=task,
            retrieved=direct_view,
            benchmark=benchmark,
            direct_skill_threshold=direct_skill_threshold,
            seed_threshold=seed_threshold,
            direct_skill_only=True,
            allow_non_exact_direct_reuse=allow_non_exact_direct_reuse,
            disable_skill_seeded=disable_skill_seeded,
            disable_direct_skill_execution=False,
            seeded_shadow_promotion_margin=seeded_shadow_promotion_margin,
            planner_seed_pool_top_k=planner_seed_pool_top_k,
            enable_string_seeded_bias=enable_string_seeded_bias,
            enable_count_seeded_bias=enable_count_seeded_bias,
            seed_guard_entry_threshold=seed_guard_entry_threshold,
            compatibility_hard_gate=compatibility_hard_gate,
            soft_aware_first=False,
            bifurcated_skill_router_enabled=False,
            bifurcated_exact_or_seeded_enabled=False,
            selective_integrated_enabled=False,
            policy_context=policy_context,
        )
        merged_direct_rejections = self._merge_rejection_reasons(direct_rejections, direct_only_rejections)
        if direct_mode == "direct_skill":
            return (
                direct_mode,
                selected_skill,
                f"{selection_reason}; bifurcated direct lane selected the strongest reusable callable",
                merged_direct_rejections,
                utility_breakdown,
            )

        if benchmark == "humaneval":
            humaneval_rejections = dict(merged_direct_rejections)
            for candidate in retrieved:
                skill_id = str(candidate.get("skill_id", "")).strip()
                if not skill_id:
                    continue
                humaneval_rejections.setdefault(skill_id, []).append("bifurcated_humaneval_abstain_to_dynamic")
            return (
                "pure_dynamic",
                None,
                "bifurcated router abstained to dynamic after direct lane miss on humaneval",
                humaneval_rejections,
                {},
            )

        seed_mode, seed_skill, seed_reason, seed_rejections, seed_utility = self._decide_planning_mode(
            task=task,
            retrieved=[dict(candidate) for candidate in retrieved],
            benchmark=benchmark,
            direct_skill_threshold=direct_skill_threshold,
            seed_threshold=seed_threshold,
            direct_skill_only=False,
            allow_non_exact_direct_reuse=False,
            disable_skill_seeded=disable_skill_seeded,
            disable_direct_skill_execution=True,
            seeded_shadow_promotion_margin=seeded_shadow_promotion_margin,
            planner_seed_pool_top_k=planner_seed_pool_top_k,
            enable_string_seeded_bias=enable_string_seeded_bias,
            enable_count_seeded_bias=enable_count_seeded_bias,
            seed_guard_entry_threshold=seed_guard_entry_threshold,
            compatibility_hard_gate=compatibility_hard_gate,
            soft_aware_first=False,
            bifurcated_skill_router_enabled=False,
            bifurcated_exact_or_seeded_enabled=False,
            selective_integrated_enabled=False,
            policy_context=policy_context,
        )
        merged_seed_rejections = self._merge_rejection_reasons(seed_rejections, merged_direct_rejections)
        if seed_mode == "skill_seeded_dynamic":
            return (
                seed_mode,
                seed_skill,
                f"{seed_reason}; bifurcated seeded lane used a prior-only proof route",
                merged_seed_rejections,
                seed_utility,
            )
        return (
            "pure_dynamic",
            None,
            "bifurcated router abstained from both direct execution and seeded prior injection",
            merged_seed_rejections,
            {},
        )

    def _decide_exact_or_seeded_mode(
        self,
        task: CodeTaskInstance,
        retrieved: list[dict[str, Any]],
        benchmark: str,
        direct_skill_threshold: float,
        seed_threshold: float,
        disable_skill_seeded: bool,
        seeded_shadow_promotion_margin: float,
        planner_seed_pool_top_k: int | None,
        enable_string_seeded_bias: bool,
        enable_count_seeded_bias: bool,
        seed_guard_entry_threshold: float | None,
        compatibility_hard_gate: bool,
        policy_context: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any] | None, str, dict[str, list[str]], dict[str, Any]]:
        direct_view, direct_only_rejections = self._exact_or_seeded_direct_view(retrieved)
        direct_mode, selected_skill, selection_reason, direct_rejections, utility_breakdown = self._decide_planning_mode(
            task=task,
            retrieved=direct_view,
            benchmark=benchmark,
            direct_skill_threshold=direct_skill_threshold,
            seed_threshold=seed_threshold,
            direct_skill_only=True,
            allow_non_exact_direct_reuse=False,
            disable_skill_seeded=disable_skill_seeded,
            disable_direct_skill_execution=False,
            seeded_shadow_promotion_margin=seeded_shadow_promotion_margin,
            planner_seed_pool_top_k=planner_seed_pool_top_k,
            enable_string_seeded_bias=enable_string_seeded_bias,
            enable_count_seeded_bias=enable_count_seeded_bias,
            seed_guard_entry_threshold=seed_guard_entry_threshold,
            compatibility_hard_gate=compatibility_hard_gate,
            soft_aware_first=False,
            bifurcated_skill_router_enabled=False,
            bifurcated_exact_or_seeded_enabled=False,
            selective_integrated_enabled=False,
            policy_context=policy_context,
        )
        merged_direct_rejections = self._merge_rejection_reasons(direct_rejections, direct_only_rejections)
        if direct_mode == "direct_skill":
            return (
                direct_mode,
                selected_skill,
                f"{selection_reason}; exact-or-seeded kept the exact-match direct lane",
                merged_direct_rejections,
                utility_breakdown,
            )

        if benchmark == "humaneval":
            humaneval_rejections = dict(merged_direct_rejections)
            for candidate in retrieved:
                skill_id = str(candidate.get("skill_id", "")).strip()
                if not skill_id:
                    continue
                humaneval_rejections.setdefault(skill_id, []).append("exact_or_seeded_humaneval_abstain_to_dynamic")
            return (
                "pure_dynamic",
                None,
                "exact-or-seeded abstained to dynamic after direct lane miss on humaneval",
                humaneval_rejections,
                {},
            )

        seed_mode, seed_skill, seed_reason, seed_rejections, seed_utility = self._decide_planning_mode(
            task=task,
            retrieved=[dict(candidate) for candidate in retrieved],
            benchmark=benchmark,
            direct_skill_threshold=direct_skill_threshold,
            seed_threshold=seed_threshold,
            direct_skill_only=False,
            allow_non_exact_direct_reuse=False,
            disable_skill_seeded=disable_skill_seeded,
            disable_direct_skill_execution=True,
            seeded_shadow_promotion_margin=seeded_shadow_promotion_margin,
            planner_seed_pool_top_k=planner_seed_pool_top_k,
            enable_string_seeded_bias=enable_string_seeded_bias,
            enable_count_seeded_bias=enable_count_seeded_bias,
            seed_guard_entry_threshold=seed_guard_entry_threshold,
            compatibility_hard_gate=compatibility_hard_gate,
            soft_aware_first=False,
            bifurcated_skill_router_enabled=False,
            bifurcated_exact_or_seeded_enabled=False,
            selective_integrated_enabled=False,
            policy_context=policy_context,
        )
        merged_seed_rejections = self._merge_rejection_reasons(seed_rejections, merged_direct_rejections)
        if seed_mode == "skill_seeded_dynamic":
            return (
                seed_mode,
                seed_skill,
                f"{seed_reason}; exact-or-seeded demoted non-exact direct reuse and selected the proof prior lane",
                merged_seed_rejections,
                seed_utility,
            )
        return (
            "pure_dynamic",
            None,
            "exact-or-seeded abstained from both exact direct execution and seeded prior injection",
            merged_seed_rejections,
            {},
        )

    def _signature_utility_component(self, candidate: dict[str, Any]) -> float:
        if bool(candidate.get("exact_entry_match", False)):
            return 1.25
        if bool(candidate.get("signature_compatible", False)):
            return 0.85
        return -0.75

    def _transfer_utility_breakdown(self, task: CodeTaskInstance, candidate: dict[str, Any]) -> dict[str, Any]:
        guard_entry_threshold = self._NO_SEED_THRESHOLD
        if candidate.get("_seed_guard_entry_threshold_override") is not None:
            guard_entry_threshold = float(candidate["_seed_guard_entry_threshold_override"])
        retrieval_score = float(candidate.get("score", 0.0))
        historical_positive_transfer = float(candidate.get("historical_positive_transfer", 0.0) or 0.0)
        estimated_seed_token_cost = float(candidate.get("estimated_seed_token_cost", 0.0) or 0.0)
        negative_transfer_risk = float(candidate.get("negative_transfer_risk", 0.0) or 0.0)
        type_mismatch = self._obvious_seed_type_mismatch(task, candidate)
        type_mismatch_penalty = 1.25 if type_mismatch else 0.0
        utility = (
            0.18 * retrieval_score
            + self._signature_utility_component(candidate)
            + 0.90 * historical_positive_transfer
            - 1.15 * estimated_seed_token_cost
            - 1.00 * negative_transfer_risk
            - type_mismatch_penalty
        )
        routing_mode = "no_seed"
        if utility >= self._CODE_EXCERPT_THRESHOLD:
            routing_mode = "code_excerpt_seed"
        elif utility >= self._SKELETON_THRESHOLD:
            routing_mode = "skeleton_only_seed"
        elif utility >= guard_entry_threshold:
            routing_mode = "guard_only_seed"
        governance_override = str(candidate.get("governance_seed_mode_override", "") or "").strip()
        if governance_override in self._ROUTING_MODE_RANK:
            if self._ROUTING_MODE_RANK[governance_override] < self._ROUTING_MODE_RANK[routing_mode]:
                routing_mode = governance_override
        return {
            "transfer_utility_score": round(utility, 4),
            "historical_positive_transfer": round(historical_positive_transfer, 4),
            "estimated_seed_token_cost": round(estimated_seed_token_cost, 4),
            "negative_transfer_risk": round(negative_transfer_risk, 4),
            "type_mismatch_penalty": round(type_mismatch_penalty, 4),
            "seed_routing_mode": routing_mode,
            "governance_seed_mode_override": governance_override,
            "type_mismatch": type_mismatch,
        }

    def planner_compatibility_score(
        self,
        task: CodeTaskInstance,
        candidate: dict[str, Any],
        *,
        policy_context: dict[str, Any] | None = None,
        runtime_error_text: str = "",
    ) -> dict[str, Any]:
        task_family = str((policy_context or {}).get("task_family") or "")
        negative_memory = dict((policy_context or {}).get("planner_negative_memory", {}) or {})
        skill_id = str(candidate.get("skill_id", "")).strip()
        skill_family = str(candidate.get("skill_family", "") or "")
        memory_hits = int(negative_memory.get(skill_id, 0) or 0)
        if skill_family:
            memory_hits += int(negative_memory.get(skill_family, 0) or 0)
        assessment = compatibility_assessment(
            task,
            candidate,
            runtime_error_text=runtime_error_text,
            negative_memory_hits=memory_hits,
        )
        context = dict(assessment.get("context") or {})
        if task_family and not context.get("task_family"):
            context["task_family"] = task_family
        assessment["context"] = context
        return assessment

    def _direct_rejection_reasons(
        self,
        candidate: dict[str, Any],
        benchmark: str,
        direct_skill_threshold: float,
        compatibility: dict[str, Any] | None = None,
        compatibility_hard_gate: bool = False,
    ) -> list[str]:
        reasons: list[str] = []
        score = float(candidate.get("score", 0.0))
        compat = float(candidate.get("compat", 0.0))
        if score < direct_skill_threshold:
            reasons.append("below_direct_threshold")
        if compat < 1.5:
            reasons.append("direct_compat_below_threshold")
        if not (bool(candidate.get("exact_entry_match", False)) or bool(candidate.get("signature_compatible", False))):
            reasons.append("missing_direct_signature_compatibility")
        if not bool(candidate.get("direct_eligible", False)):
            reasons.append("direct_not_eligible")
        if bool(candidate.get("selector_low_confidence_abstain", False)):
            reasons.append("selector_low_confidence_abstain")
        if compatibility_hard_gate and compatibility is not None:
            reasons.extend(f"planner_veto:{reason}" for reason in compatibility.get("hard_veto", []))
            if float(compatibility.get("score", 0.0) or 0.0) < self._COMPATIBILITY_DIRECT_THRESHOLD:
                reasons.append("planner_compatibility_below_direct_threshold")
        return reasons

    def _seed_rejection_reasons(
        self,
        task: CodeTaskInstance,
        candidate: dict[str, Any],
        benchmark: str,
        seed_threshold: float,
        direct_skill_only: bool,
        disable_skill_seeded: bool,
        disable_direct_skill_execution: bool,
        seeded_shadow_promotion_margin: float,
        enable_string_seeded_bias: bool,
        enable_count_seeded_bias: bool,
        seed_guard_entry_threshold: float | None,
        compatibility: dict[str, Any] | None,
        compatibility_hard_gate: bool,
        soft_aware_first: bool,
    ) -> list[str]:
        if direct_skill_only or disable_skill_seeded:
            return ["skill_seeded_disabled"]
        reasons: list[str] = []
        score = self._effective_seed_score(
            candidate,
            disable_direct_skill_execution=disable_direct_skill_execution,
            seeded_shadow_promotion_margin=seeded_shadow_promotion_margin,
        ) + self._seeded_rank_adjustment(
            task,
            candidate,
            disable_direct_skill_execution=disable_direct_skill_execution,
            enable_string_seeded_bias=enable_string_seeded_bias,
            enable_count_seeded_bias=enable_count_seeded_bias,
        )
        compat = float(candidate.get("compat", 0.0))
        if score < seed_threshold:
            reasons.append("below_seed_threshold")
        if compat < 0.5:
            reasons.append("seed_compat_below_threshold")
        if bool(candidate.get("selector_low_confidence_abstain", False)):
            reasons.append("selector_low_confidence_abstain")
        if compatibility_hard_gate and compatibility is not None:
            reasons.extend(f"planner_veto:{reason}" for reason in compatibility.get("hard_veto", []))
            if float(compatibility.get("score", 0.0) or 0.0) < self._COMPATIBILITY_DIRECT_THRESHOLD:
                reasons.append("planner_compatibility_below_seed_threshold")
        if not bool(candidate.get("hard_filter_passed", True)):
            reasons.append("hard_filter_failed")
        seeded_candidate = dict(candidate)
        seeded_candidate["_seed_guard_entry_threshold_override"] = seed_guard_entry_threshold
        utility_breakdown = self._transfer_utility_breakdown(task, seeded_candidate)
        if compatibility is not None:
            routing_mode = str(utility_breakdown.get("seed_routing_mode", "no_seed"))
            if routing_mode in {"skeleton_only_seed", "code_excerpt_seed"} and compatibility.get("soft_only"):
                reasons.append("planner_compatibility_soft_hint_only")
            if soft_aware_first and routing_mode in {"skeleton_only_seed", "code_excerpt_seed"}:
                reasons.append("soft_aware_prefers_hint_only")
        if bool(utility_breakdown.get("type_mismatch", False)):
            reasons.append("seed_type_profile_mismatch")
        if str(utility_breakdown.get("seed_routing_mode", "no_seed")) == "no_seed":
            reasons.append("transfer_utility_below_threshold")
        return reasons

    def _decide_planning_mode(
        self,
        task: CodeTaskInstance,
        retrieved: list[dict[str, Any]],
        benchmark: str,
        direct_skill_threshold: float,
        seed_threshold: float,
        direct_skill_only: bool,
        allow_non_exact_direct_reuse: bool,
        disable_skill_seeded: bool,
        disable_direct_skill_execution: bool,
        seeded_shadow_promotion_margin: float,
        planner_seed_pool_top_k: int | None,
        enable_string_seeded_bias: bool,
        enable_count_seeded_bias: bool,
        seed_guard_entry_threshold: float | None,
        compatibility_hard_gate: bool,
        soft_aware_first: bool,
        bifurcated_skill_router_enabled: bool,
        bifurcated_exact_or_seeded_enabled: bool,
        selective_integrated_enabled: bool,
        policy_context: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any] | None, str, dict[str, list[str]], dict[str, Any]]:
        if not retrieved:
            return "pure_dynamic", None, "no_retrieved_skills", {}, {}
        if bifurcated_exact_or_seeded_enabled:
            return self._decide_exact_or_seeded_mode(
                task=task,
                retrieved=retrieved,
                benchmark=benchmark,
                direct_skill_threshold=direct_skill_threshold,
                seed_threshold=seed_threshold,
                disable_skill_seeded=disable_skill_seeded,
                seeded_shadow_promotion_margin=seeded_shadow_promotion_margin,
                planner_seed_pool_top_k=planner_seed_pool_top_k,
                enable_string_seeded_bias=enable_string_seeded_bias,
                enable_count_seeded_bias=enable_count_seeded_bias,
                seed_guard_entry_threshold=seed_guard_entry_threshold,
                compatibility_hard_gate=compatibility_hard_gate,
                policy_context=policy_context,
            )
        if bifurcated_skill_router_enabled:
            return self._decide_bifurcated_mode(
                task=task,
                retrieved=retrieved,
                benchmark=benchmark,
                direct_skill_threshold=direct_skill_threshold,
                seed_threshold=seed_threshold,
                allow_non_exact_direct_reuse=allow_non_exact_direct_reuse,
                disable_skill_seeded=disable_skill_seeded,
                seeded_shadow_promotion_margin=seeded_shadow_promotion_margin,
                planner_seed_pool_top_k=planner_seed_pool_top_k,
                enable_string_seeded_bias=enable_string_seeded_bias,
                enable_count_seeded_bias=enable_count_seeded_bias,
                seed_guard_entry_threshold=seed_guard_entry_threshold,
                compatibility_hard_gate=compatibility_hard_gate,
                policy_context=policy_context,
            )

        rejection_reasons: dict[str, list[str]] = {}

        for candidate in retrieved:
            score = float(candidate.get("score", 0.0))
            exact_match = bool(candidate.get("exact_entry_match", False))
            signature_compatible = bool(candidate.get("signature_compatible", False))
            compat = float(candidate.get("compat", 0.0))
            compatibility = self.planner_compatibility_score(task, candidate, policy_context=policy_context)
            candidate["planner_compatibility_score"] = float(compatibility.get("score", 0.0) or 0.0)
            candidate["planner_compatibility_veto_reasons"] = list(compatibility.get("hard_veto", []))
            candidate["planner_soft_hint_reasons"] = list(compatibility.get("soft_only", []))
            allow_non_exact_direct = direct_skill_only or allow_non_exact_direct_reuse
            direct_allowed = (
                task.benchmark != "gsm8k"
                and not disable_direct_skill_execution
                and bool(candidate.get("direct_eligible", False))
            )
            if (
                direct_allowed
                and (exact_match or (allow_non_exact_direct and signature_compatible))
                and score >= direct_skill_threshold
                and compat >= 1.5
                and (
                    not compatibility_hard_gate
                    or (
                        not compatibility.get("hard_veto")
                        and float(compatibility.get("score", 0.0) or 0.0) >= self._COMPATIBILITY_DIRECT_THRESHOLD
                    )
                )
            ):
                if exact_match:
                    return (
                        "direct_skill",
                        candidate,
                        "top1 exact entry match with strong retrieval score and compat",
                        rejection_reasons,
                        {
                            "planner_compatibility_score": float(compatibility.get("score", 0.0) or 0.0),
                            "planner_compatibility_veto_reasons": list(compatibility.get("hard_veto", [])),
                            "planner_soft_hint_reasons": list(compatibility.get("soft_only", [])),
                        },
                    )
                return (
                    "direct_skill",
                    candidate,
                    "signature-compatible direct skill with strong retrieval score and compat",
                    rejection_reasons,
                    {
                        "planner_compatibility_score": float(compatibility.get("score", 0.0) or 0.0),
                        "planner_compatibility_veto_reasons": list(compatibility.get("hard_veto", [])),
                        "planner_soft_hint_reasons": list(compatibility.get("soft_only", [])),
                    },
                )
            if task.benchmark == "gsm8k" and (exact_match or signature_compatible):
                rejection_reasons.setdefault(str(candidate.get("skill_id", "")), []).append("benchmark_disables_direct_skill")
            if disable_direct_skill_execution and (exact_match or signature_compatible):
                rejection_reasons.setdefault(str(candidate.get("skill_id", "")), []).append("direct_skill_execution_disabled")
            if signature_compatible and not exact_match and not direct_skill_only and not disable_skill_seeded:
                rejection_reasons.setdefault(str(candidate.get("skill_id", "")), []).append("prefer_seeded_for_non_exact_signature_match")
            reasons = self._direct_rejection_reasons(
                candidate,
                benchmark,
                direct_skill_threshold,
                compatibility=compatibility,
                compatibility_hard_gate=compatibility_hard_gate,
            )
            if reasons:
                rejection_reasons.setdefault(str(candidate.get("skill_id", "")), [])
                rejection_reasons[str(candidate.get("skill_id", ""))].extend(reasons)

        if direct_skill_only or disable_skill_seeded:
            for candidate in retrieved:
                skill_id = str(candidate.get("skill_id", ""))
                rejection_reasons.setdefault(skill_id, []).append("skill_seeded_disabled")
            return "pure_dynamic", None, "direct-skill gate not met and skill-seeded planning disabled", rejection_reasons, {}

        seeded_pool = list(retrieved)
        if planner_seed_pool_top_k is not None and planner_seed_pool_top_k > 0:
            seeded_pool = seeded_pool[:planner_seed_pool_top_k]
        seeded_candidates: list[tuple[tuple[float, float, float, float, float, int], dict[str, Any], bool, dict[str, Any]]] = []
        for index, candidate in enumerate(seeded_pool):
            compatibility = self.planner_compatibility_score(task, candidate, policy_context=policy_context)
            effective_score = self._effective_seed_score(
                candidate,
                disable_direct_skill_execution=disable_direct_skill_execution,
                seeded_shadow_promotion_margin=seeded_shadow_promotion_margin,
            )
            seed_score, shadow_seed_confidence, compat, utility = self._seed_rank_tuple(
                task,
                candidate,
                disable_direct_skill_execution=disable_direct_skill_execution,
                seeded_shadow_promotion_margin=seeded_shadow_promotion_margin,
                enable_string_seeded_bias=enable_string_seeded_bias,
                enable_count_seeded_bias=enable_count_seeded_bias,
            )
            compat = float(candidate.get("compat", 0.0))
            candidate["_seed_guard_entry_threshold_override"] = seed_guard_entry_threshold
            utility_breakdown = self._transfer_utility_breakdown(task, candidate)
            if selective_integrated_enabled:
                if bool(candidate.get("selector_low_confidence_abstain", False)):
                    utility_breakdown["seed_routing_mode"] = "no_seed"
                    utility_breakdown["selective_seed_blocked"] = "low_confidence_abstain"
                elif bool(candidate.get("direct_eligible", False)) and (
                    bool(candidate.get("exact_entry_match", False))
                    or bool(candidate.get("signature_compatible", False))
                ):
                    utility_breakdown["seed_routing_mode"] = "no_seed"
                    utility_breakdown["selective_seed_blocked"] = "reserved_for_direct_lane"
            if (
                str(utility_breakdown.get("seed_routing_mode", "no_seed")) == "no_seed"
                and seed_score >= seed_threshold
                and float(candidate.get("compat", 0.0) or 0.0) >= 1.0
                and not compatibility.get("hard_veto")
                and not str(utility_breakdown.get("selective_seed_blocked", "")).strip()
            ):
                utility_breakdown["seed_routing_mode"] = "guard_only_seed"
            if compatibility_hard_gate and compatibility.get("hard_veto"):
                utility_breakdown["seed_routing_mode"] = "no_seed"
            elif compatibility.get("soft_only") and str(utility_breakdown.get("seed_routing_mode", "")) in {
                "skeleton_only_seed",
                "code_excerpt_seed",
            }:
                utility_breakdown["seed_routing_mode"] = "guard_only_seed"
            if soft_aware_first and str(utility_breakdown.get("seed_routing_mode", "")) in {"skeleton_only_seed", "code_excerpt_seed"}:
                utility_breakdown["seed_routing_mode"] = "guard_only_seed"
            utility_breakdown["planner_compatibility_score"] = float(compatibility.get("score", 0.0) or 0.0)
            utility_breakdown["planner_compatibility_veto_reasons"] = list(compatibility.get("hard_veto", []))
            utility_breakdown["planner_soft_hint_reasons"] = list(compatibility.get("soft_only", []))
            candidate["transfer_utility_score"] = float(utility_breakdown.get("transfer_utility_score", 0.0))
            candidate["historical_positive_transfer"] = float(utility_breakdown.get("historical_positive_transfer", 0.0))
            candidate["estimated_seed_token_cost"] = float(utility_breakdown.get("estimated_seed_token_cost", 0.0))
            candidate["negative_transfer_risk"] = float(utility_breakdown.get("negative_transfer_risk", 0.0))
            candidate["seed_routing_mode"] = str(utility_breakdown.get("seed_routing_mode", ""))
            candidate["type_mismatch_penalty"] = float(utility_breakdown.get("type_mismatch_penalty", 0.0))
            candidate["planner_compatibility_score"] = float(compatibility.get("score", 0.0) or 0.0)
            candidate["planner_compatibility_veto_reasons"] = list(compatibility.get("hard_veto", []))
            candidate["planner_soft_hint_reasons"] = list(compatibility.get("soft_only", []))
            seed_type_mismatch = bool(utility_breakdown.get("type_mismatch", False))
            if (
                bool(candidate.get("hard_filter_passed", True))
                and seed_score >= seed_threshold
                and compat >= 0.5
                and not seed_type_mismatch
                and str(utility_breakdown.get("seed_routing_mode", "no_seed")) != "no_seed"
            ):
                seeded_candidates.append(
                    (
                        (
                            seed_score,
                            float(utility_breakdown.get("transfer_utility_score", 0.0)),
                            shadow_seed_confidence,
                            compat,
                            utility,
                            -index,
                        ),
                        candidate,
                        str(candidate.get("status", "")) == "shadow"
                        and (
                            effective_score > float(candidate.get("score", 0.0))
                            or shadow_seed_confidence > 0.0
                        ),
                        utility_breakdown,
                    )
                )
                continue
            reasons = self._seed_rejection_reasons(
                task,
                candidate,
                benchmark,
                seed_threshold,
                direct_skill_only,
                disable_skill_seeded,
                disable_direct_skill_execution,
                seeded_shadow_promotion_margin,
                enable_string_seeded_bias,
                enable_count_seeded_bias,
                seed_guard_entry_threshold,
                compatibility,
                compatibility_hard_gate,
                soft_aware_first,
            )
            if reasons:
                rejection_reasons.setdefault(str(candidate.get("skill_id", "")), [])
                rejection_reasons[str(candidate.get("skill_id", ""))].extend(reasons)
        if seeded_candidates:
            _, candidate, used_shadow_promotion, utility_breakdown = max(seeded_candidates, key=lambda item: item[0])
            reason = (
                "retrieved prior is useful enough to seed dynamic generation via "
                f"{utility_breakdown.get('seed_routing_mode', 'guard_only_seed')}"
            )
            if used_shadow_promotion:
                reason = f"{reason}; selected promoted shadow helper for seeded-only routing"
            if utility_breakdown.get("planner_soft_hint_reasons"):
                reason = f"{reason}; compatibility keeps this seed in hint-only mode"
            if soft_aware_first:
                reason = f"{reason}; soft-aware-first preserved a no-skill fallback"
            if selective_integrated_enabled:
                reason = f"{reason}; selective route kept this as a planning prior only"
            return "skill_seeded_dynamic", candidate, reason, rejection_reasons, utility_breakdown
        return "pure_dynamic", None, "retrieval too weak for direct or selective seeded planning", rejection_reasons, {}

    def build_plan(
        self,
        task: CodeTaskInstance,
        mode: str,
            retrieved_skills: list[dict[str, Any]] | None = None,
            matched_templates: list[dict[str, Any]] | None = None,
            task_pattern: str = "",
            retrieval_threshold: float = 5.0,
            direct_skill_threshold: float | None = None,
            seed_threshold: float | None = None,
            seed_guard_entry_threshold: float | None = None,
            direct_skill_only: bool = False,
            allow_non_exact_direct_reuse: bool = False,
            disable_skill_seeded: bool = False,
            disable_direct_skill_execution: bool = False,
            seeded_shadow_promotion_margin: float = 0.0,
            planner_seed_pool_top_k: int | None = None,
            enable_string_seeded_bias: bool = False,
            enable_count_seeded_bias: bool = False,
            compatibility_hard_gate: bool = False,
            soft_aware_first: bool = False,
            bifurcated_skill_router_enabled: bool = False,
            bifurcated_exact_or_seeded_enabled: bool = False,
            selective_integrated_enabled: bool = False,
            seeded_disable_local_insertion: bool = False,
            exact_direct_postcall_wrapper_enabled: bool = False,
            direct_postcall_max_local_patch: int = 0,
            policy_context: dict[str, Any] | None = None,
    ) -> WorkflowPlan:
        """Build a benchmark workflow plan."""
        retrieved = retrieved_skills or []
        policy = policy_context or {}
        steps: list[WorkflowStep]
        planning_mode = "pure_dynamic"
        selected_skill = None
        seed_context_skill_ids: list[str] = []
        seed_context_skill_modes: dict[str, str] = {}
        planned_usage_subtype = ""
        seed_routing_mode = ""
        transfer_utility_score = 0.0
        estimated_seed_token_cost = 0.0
        historical_positive_transfer = 0.0
        negative_transfer_risk = 0.0
        planner_compatibility_score = 0.0
        planner_compatibility_veto_reasons: list[str] = []
        planner_soft_hint_reasons: list[str] = []
        seed_source = ""
        selection_reason = "oracle mode"
        planner_rejection_reasons: dict[str, list[str]] = {}
        policy_version = int(policy.get("version", 0))
        policy_directives = list(policy.get("prompt_directives", []))
        matched_template_ids = [str(item.get("template_id", "")) for item in (matched_templates or []) if item.get("template_id")]

        if mode == "oracle":
            steps = [
                WorkflowStep(step_id="s1", step_type="draft_code", payload={"mode": "oracle"}),
                WorkflowStep(step_id="s2", step_type="run_tests", payload={"verifier": task.benchmark}),
                WorkflowStep(step_id="s3", step_type="finalize", payload={"mode": "oracle"}),
            ]
        elif mode == "dynamic":
            direct_threshold = direct_skill_threshold if direct_skill_threshold is not None else retrieval_threshold
            direct_threshold = max(0.0, direct_threshold - float(policy.get("direct_skill_delta", 0.0)))
            seed_threshold_value = seed_threshold if seed_threshold is not None else max(0.5, retrieval_threshold * 0.6)
            seed_threshold_value = max(0.0, seed_threshold_value - float(policy.get("seeded_delta", 0.0)))
            planning_mode, selected_skill, selection_reason, planner_rejection_reasons, utility_breakdown = self._decide_planning_mode(
                task=task,
                retrieved=retrieved,
                benchmark=task.benchmark,
                direct_skill_threshold=direct_threshold,
                seed_threshold=seed_threshold_value,
                direct_skill_only=direct_skill_only,
                allow_non_exact_direct_reuse=allow_non_exact_direct_reuse,
                disable_skill_seeded=disable_skill_seeded,
                disable_direct_skill_execution=disable_direct_skill_execution,
                seeded_shadow_promotion_margin=seeded_shadow_promotion_margin,
                planner_seed_pool_top_k=planner_seed_pool_top_k,
                enable_string_seeded_bias=enable_string_seeded_bias,
                enable_count_seeded_bias=enable_count_seeded_bias,
                seed_guard_entry_threshold=seed_guard_entry_threshold,
                compatibility_hard_gate=compatibility_hard_gate,
                soft_aware_first=soft_aware_first,
                bifurcated_skill_router_enabled=bifurcated_skill_router_enabled,
                bifurcated_exact_or_seeded_enabled=bifurcated_exact_or_seeded_enabled,
                selective_integrated_enabled=selective_integrated_enabled,
                policy_context=policy,
            )
            if matched_template_ids:
                selection_reason = f"{selection_reason}; matched templates={', '.join(matched_template_ids)}"
            if selected_skill is not None:
                seed_source = selected_skill.get("code_path", "")
                seed_routing_mode = str(utility_breakdown.get("seed_routing_mode", "") or "")
                transfer_utility_score = float(utility_breakdown.get("transfer_utility_score", 0.0) or 0.0)
                estimated_seed_token_cost = float(utility_breakdown.get("estimated_seed_token_cost", 0.0) or 0.0)
                historical_positive_transfer = float(utility_breakdown.get("historical_positive_transfer", 0.0) or 0.0)
                negative_transfer_risk = float(utility_breakdown.get("negative_transfer_risk", 0.0) or 0.0)
                planner_compatibility_score = float(utility_breakdown.get("planner_compatibility_score", selected_skill.get("planner_compatibility_score", 0.0)) or 0.0)
                planner_compatibility_veto_reasons = list(
                    utility_breakdown.get("planner_compatibility_veto_reasons", selected_skill.get("planner_compatibility_veto_reasons", []))
                    or []
                )
                planner_soft_hint_reasons = list(
                    utility_breakdown.get("planner_soft_hint_reasons", selected_skill.get("planner_soft_hint_reasons", []))
                    or []
                )
            planned_usage_subtype = self._planned_usage_subtype(planning_mode, selected_skill)
            seed_context_skill_modes = self._seed_context_skill_modes(
                task,
                planning_mode=planning_mode,
                retrieved=retrieved,
                selected_skill=selected_skill,
                selected_routing_mode=seed_routing_mode or "guard_only_seed",
            )
            seed_context_skill_ids = list(seed_context_skill_modes)

            steps = []
            if planning_mode == "direct_skill" and selected_skill is not None:
                steps.append(
                    WorkflowStep(
                        step_id="s0",
                        step_type="call_skill",
                        payload={
                            "skill_id": selected_skill["skill_id"],
                            "score": selected_skill["score"],
                            "code_path": selected_skill["code_path"],
                            "entry_point": selected_skill.get("entry_point", ""),
                        },
                    )
                )
                steps.extend(
                    [
                        WorkflowStep(step_id="s1", step_type="run_tests", payload={"verifier": task.benchmark}),
                        WorkflowStep(step_id="s2", step_type="finalize", payload={"mode": "dynamic"}),
                    ]
                )
            else:
                draft_step_type = "draft_from_skill" if planning_mode == "skill_seeded_dynamic" else "draft_code"
                repair_step_type = "repair_with_skill" if planning_mode == "skill_seeded_dynamic" else "repair_code"
                draft_payload = {"mode": "dynamic", "planning_mode": planning_mode}
                repair_payload = {"max_repairs": 2, "planning_mode": planning_mode}
                if selected_skill is not None:
                    draft_payload["skill_id"] = selected_skill["skill_id"]
                    draft_payload["seed_source"] = selected_skill.get("code_path", "")
                    repair_payload["skill_id"] = selected_skill["skill_id"]
                    repair_payload["seed_source"] = selected_skill.get("code_path", "")

                steps.extend(
                    [
                        WorkflowStep(step_id="s1", step_type=draft_step_type, payload=draft_payload),
                        WorkflowStep(step_id="s2", step_type="run_tests", payload={"verifier": task.benchmark}),
                        WorkflowStep(step_id="s3", step_type="inspect_failure", payload={"enabled": True}),
                        WorkflowStep(step_id="s4", step_type=repair_step_type, payload=repair_payload),
                        WorkflowStep(step_id="s5", step_type="run_tests", payload={"verifier": task.benchmark}),
                        WorkflowStep(step_id="s6", step_type="finalize", payload={"mode": "dynamic"}),
                    ]
                )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        return WorkflowPlan(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            task_id=task.task_id,
            retrieved_skills=[item["skill_id"] for item in retrieved],
            planner_visible_candidates=[str(item.get("skill_id", "")) for item in retrieved if item.get("skill_id")],
            planner_rejection_reasons=planner_rejection_reasons if mode == "dynamic" else {},
            planning_mode=planning_mode,
            task_pattern=task_pattern,
            selected_skill_id=selected_skill["skill_id"] if selected_skill is not None else "",
            seed_context_skill_ids=seed_context_skill_ids,
            seed_context_skill_modes=seed_context_skill_modes,
            planned_usage_subtype=planned_usage_subtype,
            seed_routing_mode=seed_routing_mode,
            transfer_utility_score=transfer_utility_score,
            estimated_seed_token_cost=estimated_seed_token_cost,
            historical_positive_transfer=historical_positive_transfer,
            negative_transfer_risk=negative_transfer_risk,
            planner_compatibility_score=planner_compatibility_score,
            planner_compatibility_veto_reasons=planner_compatibility_veto_reasons,
            planner_soft_hint_reasons=planner_soft_hint_reasons,
            skill_seeded=planning_mode == "skill_seeded_dynamic",
            soft_aware_enabled=bool(soft_aware_first and planning_mode == "skill_seeded_dynamic"),
            fallback_planning_mode="pure_dynamic" if soft_aware_first and planning_mode == "skill_seeded_dynamic" else "",
            bifurcated_skill_router_enabled=bifurcated_skill_router_enabled,
            bifurcated_exact_or_seeded_enabled=bifurcated_exact_or_seeded_enabled,
            selective_integrated_enabled=selective_integrated_enabled,
            seeded_disable_local_insertion=bool(seeded_disable_local_insertion and planning_mode == "skill_seeded_dynamic"),
            exact_direct_postcall_wrapper_enabled=bool(exact_direct_postcall_wrapper_enabled and planning_mode == "direct_skill"),
            direct_postcall_max_local_patch=int(direct_postcall_max_local_patch if planning_mode == "direct_skill" else 0),
            seed_source=seed_source,
            selection_reason=selection_reason,
            policy_version=policy_version,
            policy_directives=[
                *policy_directives,
                *(
                    [
                        "exact_or_seeded_prior_only",
                        "if no exact direct match exists, prefer the proof prior lane over non-exact direct reuse",
                        "do not insert local hard skill nodes; keep the seed as a planning prior",
                    ]
                    if bifurcated_exact_or_seeded_enabled and planning_mode == "skill_seeded_dynamic"
                    else []
                ),
                *(
                    [
                        "bifurcated_seed_prior_only",
                        "use the retrieved seed only as a planning prior; do not insert local hard skill nodes",
                        "prefer first-attempt seed realization; if the prior is weak, abstain back to a fresh dynamic plan",
                    ]
                    if bifurcated_skill_router_enabled and planning_mode == "skill_seeded_dynamic"
                    else []
                ),
                *(
                    [
                        "selective_seed_prior_only",
                        "do not insert local skill nodes; use the retrieved seed only to bias the initial plan",
                        "prefer realizing the seed in the initial draft rather than deferring it to repair",
                    ]
                    if selective_integrated_enabled and planning_mode == "skill_seeded_dynamic"
                    else []
                ),
                *(
                    [
                        "postcall_wrapper: normalize interface/output only",
                        "postcall_wrapper: if patching is needed, keep it local and minimal",
                        "postcall_wrapper: do not enter an open-ended repair loop",
                    ]
                    if exact_direct_postcall_wrapper_enabled and planning_mode == "direct_skill"
                    else []
                ),
            ],
            matched_template_ids=matched_template_ids,
            steps=steps,
        )
