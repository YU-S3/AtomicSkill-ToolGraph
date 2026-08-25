"""Candidate generation and repair abstractions for executable benchmark experiments."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.schemas import BenchmarkTaskInstance, CodeTaskInstance, GenerationEvent, VerifierFeedback
from runtime.config import RuntimeLLMConfig
from runtime.llm_client import LLMClient, LLMClientError, LLMResponse


@dataclass(frozen=True)
class GenerationOutput:
    code: str
    event: GenerationEvent | None = None


@dataclass(frozen=True)
class PromptAssembly:
    text: str
    seed_context_text: str = ""
    scaffold_text: str = ""
    repair_strategy: str = ""
    task_context_text: str = ""
    tests_context_text: str = ""
    feedback_context_text: str = ""
    previous_code_context_text: str = ""
    seed_summary_text: str = ""
    compact_seeded_prompt_used: bool = False


class GenerationError(RuntimeError):
    """Raised when generation fails after an LLM call or prompt assembly."""

    def __init__(self, message: str, event: GenerationEvent) -> None:
        super().__init__(message)
        self.event = event


class BaseGenerator:
    """Abstract generator interface for code tasks."""

    def draft(
        self,
        task: BenchmarkTaskInstance,
        retrieved_skills: list[dict[str, Any]] | None = None,
        planning_mode: str = "pure_dynamic",
        seed_context_skill_modes: dict[str, str] | None = None,
        seed_routing_mode: str = "",
        transfer_utility_score: float = 0.0,
        estimated_seed_token_cost: float = 0.0,
        historical_positive_transfer: float = 0.0,
        negative_transfer_risk: float = 0.0,
        policy_directives: list[str] | None = None,
        workflow_memories: list[dict[str, Any]] | None = None,
        template_priors: list[dict[str, Any]] | None = None,
        primitive_helpers: list[dict[str, Any]] | None = None,
    ) -> GenerationOutput:
        raise NotImplementedError

    def repair(
        self,
        task: BenchmarkTaskInstance,
        previous_code: str,
        feedback: VerifierFeedback,
        attempt_index: int,
        retrieved_skills: list[dict[str, Any]] | None = None,
        planning_mode: str = "pure_dynamic",
        seed_context_skill_modes: dict[str, str] | None = None,
        seed_routing_mode: str = "",
        transfer_utility_score: float = 0.0,
        estimated_seed_token_cost: float = 0.0,
        historical_positive_transfer: float = 0.0,
        negative_transfer_risk: float = 0.0,
        policy_directives: list[str] | None = None,
        workflow_memories: list[dict[str, Any]] | None = None,
        template_priors: list[dict[str, Any]] | None = None,
        primitive_helpers: list[dict[str, Any]] | None = None,
    ) -> GenerationOutput:
        raise NotImplementedError


class OracleGenerator(BaseGenerator):
    """Debug-only generator that returns benchmark reference solutions."""

    def draft(
        self,
        task: BenchmarkTaskInstance,
        retrieved_skills: list[dict[str, Any]] | None = None,
        planning_mode: str = "pure_dynamic",
        seed_context_skill_modes: dict[str, str] | None = None,
        seed_routing_mode: str = "",
        transfer_utility_score: float = 0.0,
        estimated_seed_token_cost: float = 0.0,
        historical_positive_transfer: float = 0.0,
        negative_transfer_risk: float = 0.0,
        policy_directives: list[str] | None = None,
        workflow_memories: list[dict[str, Any]] | None = None,
        template_priors: list[dict[str, Any]] | None = None,
        primitive_helpers: list[dict[str, Any]] | None = None,
    ) -> GenerationOutput:
        return GenerationOutput(code=(task.canonical_solution or "").rstrip() + "\n")

    def repair(
        self,
        task: BenchmarkTaskInstance,
        previous_code: str,
        feedback: VerifierFeedback,
        attempt_index: int,
        retrieved_skills: list[dict[str, Any]] | None = None,
        planning_mode: str = "pure_dynamic",
        seed_context_skill_modes: dict[str, str] | None = None,
        seed_routing_mode: str = "",
        transfer_utility_score: float = 0.0,
        estimated_seed_token_cost: float = 0.0,
        historical_positive_transfer: float = 0.0,
        negative_transfer_risk: float = 0.0,
        policy_directives: list[str] | None = None,
        workflow_memories: list[dict[str, Any]] | None = None,
        template_priors: list[dict[str, Any]] | None = None,
        primitive_helpers: list[dict[str, Any]] | None = None,
    ) -> GenerationOutput:
        return self.draft(task=task)


class HeuristicGenerator(BaseGenerator):
    """Cheap fallback generator used in tests and non-LLM paths."""

    def _signature(self, task: CodeTaskInstance) -> str:
        entry_point = task.entry_point or self._infer_entry_point(task)
        args = self._infer_args(task)
        return f"def {entry_point}({', '.join(args)}):"

    def _infer_entry_point(self, task: CodeTaskInstance) -> str:
        if task.entry_point:
            return task.entry_point
        for test_case in task.test_list:
            match = re.search(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", test_case)
            if match:
                return match.group(1)
        return "solve"

    def _infer_args(self, task: CodeTaskInstance) -> list[str]:
        entry_point = self._infer_entry_point(task)
        params = ["x"]
        for test_case in task.test_list:
            match = re.search(rf"{re.escape(entry_point)}\((.*?)\)", test_case)
            if not match:
                continue
            arg_count = 1 if not match.group(1).strip() else match.group(1).count(",") + 1
            params = [f"arg{i}" for i in range(arg_count)]
            break
        return params

    def _guess_default_return(self, task: CodeTaskInstance, feedback: VerifierFeedback | None = None, previous_code: str = "") -> str:
        text = "\n".join([task.prompt, task.text, task.test, "\n".join(task.test_list), previous_code, getattr(feedback, "stderr", "")]).lower()
        if any(token in text for token in ["list", "array", "append", "elements"]):
            return "[]"
        if any(token in text for token in ["string", "substring", "palindrome", "text"]):
            return "''"
        if any(token in text for token in ["true", "false", "check", "predicate", "is_", "whether"]):
            return "False"
        return "0"

    def _seeded_draft(self, task: CodeTaskInstance, retrieved_skills: list[dict[str, Any]] | None) -> str:
        signature = self._signature(task)
        if retrieved_skills:
            top = retrieved_skills[0]
            entry_point = str(top.get("entry_point") or top.get("signature", {}).get("entry_point") or "")
            if entry_point and entry_point != self._infer_entry_point(task):
                args = ", ".join(self._infer_args(task))
                return f"{signature}\n    return {entry_point}(*[{args}])\n"
        return f"{signature}\n    return {self._guess_default_return(task)}\n"

    def draft(
        self,
        task: BenchmarkTaskInstance,
        retrieved_skills: list[dict[str, Any]] | None = None,
        planning_mode: str = "pure_dynamic",
        seed_context_skill_modes: dict[str, str] | None = None,
        seed_routing_mode: str = "",
        transfer_utility_score: float = 0.0,
        estimated_seed_token_cost: float = 0.0,
        historical_positive_transfer: float = 0.0,
        negative_transfer_risk: float = 0.0,
        policy_directives: list[str] | None = None,
        workflow_memories: list[dict[str, Any]] | None = None,
        template_priors: list[dict[str, Any]] | None = None,
        primitive_helpers: list[dict[str, Any]] | None = None,
    ) -> GenerationOutput:
        if planning_mode == "skill_seeded_dynamic":
            return GenerationOutput(code=self._seeded_draft(task, retrieved_skills))
        return GenerationOutput(code=f"{self._signature(task)}\n    return {self._guess_default_return(task)}\n")

    def repair(
        self,
        task: BenchmarkTaskInstance,
        previous_code: str,
        feedback: VerifierFeedback,
        attempt_index: int,
        retrieved_skills: list[dict[str, Any]] | None = None,
        planning_mode: str = "pure_dynamic",
        seed_context_skill_modes: dict[str, str] | None = None,
        seed_routing_mode: str = "",
        transfer_utility_score: float = 0.0,
        estimated_seed_token_cost: float = 0.0,
        historical_positive_transfer: float = 0.0,
        negative_transfer_risk: float = 0.0,
        policy_directives: list[str] | None = None,
        workflow_memories: list[dict[str, Any]] | None = None,
        template_priors: list[dict[str, Any]] | None = None,
        primitive_helpers: list[dict[str, Any]] | None = None,
    ) -> GenerationOutput:
        signature = self._signature(task)
        if planning_mode == "skill_seeded_dynamic":
            return GenerationOutput(code=self._seeded_draft(task, retrieved_skills))
        return GenerationOutput(code=f"{signature}\n    return {self._guess_default_return(task, feedback, previous_code)}\n")


class LLMGenerator(HeuristicGenerator):
    """LLM-backed generator for MBPP and HumanEval."""

    _COMBINATORICS_HINT_KEYWORDS = (
        "sequence",
        "sequences",
        "prefix sum",
        "prefix sums",
        "rectangle",
        "rectangles",
        "circle",
        "radius",
        "ways",
    )
    _COUNTING_HELPER_PREFIXES = ("skill_humaneval_HumanEval_", "skill_mbpp_235")
    _GEOMETRIC_COUNTING_KEYWORDS = ("rectangle", "rectangles", "circle", "radius")
    _SKILL_PLAN_START = "# SKILL_PLAN"
    _SKILL_PLAN_END = "# END_SKILL_PLAN"

    def __init__(self, runtime_config: RuntimeLLMConfig, client: LLMClient | None = None) -> None:
        self.runtime_config = runtime_config
        self.client = client or LLMClient(runtime_config)

    def _event(
        self,
        *,
        stage: str,
        response: LLMResponse,
        conditioned_skill_ids: list[str],
        planned_skill_nodes: list[dict[str, Any]] | None = None,
        planned_skill_node_count: int = 0,
        source_structural_role: str = "",
        scaffold_hint_injected: bool = False,
        guard_checker_emitted: bool = False,
        guard_checker_context_injected: bool = False,
        guard_checker_feedback_used: bool = False,
        scaffold_suppression_reason: str = "",
        repair_seeded_demoted: bool = False,
        repair_demotion_reason: str = "",
        seed_context_skill_count: int = 0,
        seed_context_mode: str = "",
        seed_routing_mode: str = "",
        seed_context_char_count: int = 0,
        seed_context_contains_code: bool = False,
        seed_context_contains_tests: bool = False,
        transfer_utility_score: float = 0.0,
        estimated_seed_token_cost: float = 0.0,
        historical_positive_transfer: float = 0.0,
        negative_transfer_risk: float = 0.0,
        compact_seeded_prompt_used: bool = False,
        task_context_tokens: int = 0,
        tests_context_tokens: int = 0,
        feedback_context_tokens: int = 0,
        previous_code_context_tokens: int = 0,
        seed_summary_tokens: int = 0,
        seed_context_tokens: int = 0,
        scaffold_tokens: int = 0,
        repair_strategy: str = "",
    ) -> GenerationEvent:
        return GenerationEvent(
            generation_stage=stage,
            provider=response.provider,
            model=response.model,
            conditioned_skill_ids=conditioned_skill_ids,
            planned_skill_nodes=list(planned_skill_nodes or []),
            planned_skill_node_count=planned_skill_node_count,
            source_structural_role=source_structural_role,
            scaffold_hint_injected=scaffold_hint_injected,
            guard_checker_emitted=guard_checker_emitted,
            guard_checker_context_injected=guard_checker_context_injected,
            guard_checker_feedback_used=guard_checker_feedback_used,
            scaffold_suppression_reason=scaffold_suppression_reason,
            repair_seeded_demoted=repair_seeded_demoted,
            repair_demotion_reason=repair_demotion_reason,
            seed_context_skill_count=seed_context_skill_count,
            seed_context_mode=seed_context_mode,
            seed_routing_mode=seed_routing_mode,
            seed_context_char_count=seed_context_char_count,
            seed_context_contains_code=seed_context_contains_code,
            seed_context_contains_tests=seed_context_contains_tests,
            transfer_utility_score=transfer_utility_score,
            estimated_seed_token_cost=estimated_seed_token_cost,
            historical_positive_transfer=historical_positive_transfer,
            negative_transfer_risk=negative_transfer_risk,
            compact_seeded_prompt_used=compact_seeded_prompt_used,
            task_context_tokens=task_context_tokens,
            tests_context_tokens=tests_context_tokens,
            feedback_context_tokens=feedback_context_tokens,
            previous_code_context_tokens=previous_code_context_tokens,
            seed_summary_tokens=seed_summary_tokens,
            seed_context_tokens=seed_context_tokens,
            scaffold_tokens=scaffold_tokens,
            repair_strategy=repair_strategy,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            latency_ms=response.latency_ms,
        )

    def _prompt_component_units(self, text: str) -> int:
        return len(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\s]", text))

    def _allocate_prompt_component_tokens(self, total_tokens: int, components: dict[str, str]) -> dict[str, int]:
        keys = list(components)
        if total_tokens <= 0:
            return {key: 0 for key in keys}
        units = {key: self._prompt_component_units(value) for key, value in components.items() if str(value).strip()}
        if not units:
            return {key: 0 for key in keys}
        total_units = sum(units.values())
        if total_units <= 0:
            return {key: 0 for key in keys}
        allocations = {key: 0 for key in keys}
        assigned = 0
        remainders: list[tuple[float, str]] = []
        for key, unit_count in units.items():
            raw = total_tokens * (unit_count / total_units)
            base = int(raw)
            allocations[key] = base
            assigned += base
            remainders.append((raw - base, key))
        for _fraction, key in sorted(remainders, reverse=True):
            if assigned >= total_tokens:
                break
            allocations[key] += 1
            assigned += 1
        return allocations

    def _skill_sort_key(self, skill: dict[str, Any]) -> tuple[float, int, float, int, str]:
        return (
            float(skill.get("score", 0.0)),
            int(bool(skill.get("exact_entry_match", False))),
            float(skill.get("utility", 0.0)),
            int(skill.get("call_count", 0)),
            str(skill.get("skill_id", "")),
        )

    def _select_skills(self, retrieved_skills: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        if not retrieved_skills:
            return []
        if any(bool(item.get("_preserve_prompt_order", False)) for item in retrieved_skills):
            return list(retrieved_skills[: self.runtime_config.skill_top_k])
        skills = sorted(retrieved_skills, key=self._skill_sort_key, reverse=True)
        return skills[: self.runtime_config.skill_top_k]

    def _instructions(self, *, stage: str, planning_mode: str, policy_directives: list[str]) -> str:
        if planning_mode == "skill_seeded_dynamic" and stage == "draft":
            parts = [
                "Write raw Python only.",
                "Preserve the requested entry point.",
                "Use the retrieved prior only as a shallow scaffold.",
                "Prefer the shortest correct implementation.",
            ]
        else:
            parts = [
                "Write Python that passes deterministic tests.",
                "Return raw Python only.",
                "Preserve the requested entry point.",
            ]
            if planning_mode == "skill_seeded_dynamic":
                parts.append("A retrieved skill prior is available. Use it only if it shortens the search.")
        if policy_directives:
            parts.append("Persisted runtime policy directives:\n- " + "\n- ".join(policy_directives))
        if stage == "repair":
            parts.append("Fix the candidate using verifier feedback. Prefer minimal edits.")
        return "\n\n".join(parts)

    def _signature_block(self, skill: dict[str, Any]) -> str:
        signature = skill.get("signature") or {}
        entry_point = str(skill.get("entry_point") or signature.get("entry_point") or "").strip()
        args: list[str] = []
        for item in signature.get("args", []) or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip() or "arg"
            annotation = str(item.get("annotation", "")).strip()
            args.append(f"{name}: {annotation}" if annotation else name)
        returns = str(signature.get("returns", "")).strip()
        suffix = f" -> {returns}" if returns else ""
        return f"def {entry_point}({', '.join(args)}){suffix}:"

    def _skill_source_text(self, skill: dict[str, Any]) -> str:
        code_path = str(skill.get("code_path", "") or "").strip()
        if not code_path:
            return ""
        path = Path(code_path)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def _normalized_code_excerpt(self, skill: dict[str, Any], *, max_lines: int = 30) -> str:
        source = self._skill_source_text(skill)
        if not source:
            return ""
        lines = [line.rstrip() for line in source.splitlines()]
        if not lines:
            return ""
        return "\n".join(lines[:max_lines]).strip()

    def _helper_functions(self, skill: dict[str, Any]) -> list[str]:
        source = self._skill_source_text(skill)
        if not source:
            return []
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        entry_point = str(skill.get("entry_point") or (skill.get("signature") or {}).get("entry_point") or "").strip()
        return [
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name != entry_point
        ][:6]

    def _replay_assertions(self, skill: dict[str, Any], *, limit: int = 2) -> list[str]:
        assertions: list[str] = []
        for test in skill.get("tests", []) or []:
            if not isinstance(test, dict):
                continue
            for line in str(test.get("code", "") or "").splitlines():
                stripped = line.strip()
                if stripped.startswith("assert "):
                    assertions.append(stripped)
                if len(assertions) >= limit:
                    return assertions
        return assertions

    def _clip_chars(self, text: str, *, max_chars: int) -> str:
        compact = " ".join(str(text or "").split())
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 3].rstrip() + "..."

    def _seed_transfer_instruction(self, subtype: str) -> str:
        if subtype == "guard_checker":
            return (
                "borrow precondition design, boundary checks, and local sanity patterns; "
                "do not copy task-specific constants, names, or the exact body."
            )
        return (
            "borrow decomposition, helper structure, and edge-case handling; "
            "do not copy task-specific constants, names, or the exact body."
        )

    def _guard_only_seed_block(self, skill: dict[str, Any]) -> str:
        replay_assertions = self._replay_assertions(skill, limit=1)
        hint_parts = []
        if replay_assertions:
            hint_parts.append(replay_assertions[0].replace("assert ", "check "))
        failure_log = list(skill.get("failure_log") or [])
        if failure_log:
            verifier_message = str(failure_log[-1].get("verifier_message", "") or "")
            if verifier_message:
                hint_parts.append(verifier_message[:60].strip())
        if not hint_parts:
            hint_parts.append(f"preserve edge checks around {skill.get('entry_point', '') or 'the helper'}")
        return self._clip_chars("guard_hint: " + "; ".join(part for part in hint_parts if part), max_chars=120)

    def _skeleton_only_seed_block(self, skill: dict[str, Any]) -> str:
        signature = self._signature_block(skill)
        helper_functions = self._helper_functions(skill)
        helper_text = f"helpers={','.join(helper_functions[:3])}" if helper_functions else "helpers=none"
        replay_assertions = self._replay_assertions(skill, limit=1)
        assertion_text = replay_assertions[0] if replay_assertions else ""
        parts = [
            f"signature: {signature}",
            helper_text,
            f"edge: {assertion_text}" if assertion_text else "",
        ]
        return self._clip_chars(" | ".join(part for part in parts if part), max_chars=220)

    def _code_excerpt_seed_block(self, skill: dict[str, Any], *, subtype: str) -> str:
        signature = self._signature_block(skill)
        helper_functions = self._helper_functions(skill)
        excerpt = self._normalized_code_excerpt(skill, max_lines=12)
        replay_assertions = self._replay_assertions(skill, limit=2)
        parts = [f"signature: {signature}"]
        if helper_functions:
            parts.append("helpers: " + ", ".join(helper_functions[:4]))
        if excerpt:
            parts.append(f"normalized_code_excerpt:\n```python\n{excerpt}\n```")
        if replay_assertions:
            parts.append("replay_assertions:\n- " + "\n- ".join(replay_assertions))
        parts.append(f"transfer_instruction: {self._seed_transfer_instruction(subtype or 'plan_skeleton')}")
        text = "\n\n".join(part for part in parts if part)
        return text[:500].rstrip() + ("..." if len(text) > 500 else "")

    def _seed_context_summary(
        self,
        skills: list[dict[str, Any]],
        *,
        subtype: str,
        seed_context_text: str,
        seed_routing_mode: str = "",
        transfer_utility_score: float = 0.0,
        estimated_seed_token_cost: float = 0.0,
        historical_positive_transfer: float = 0.0,
        negative_transfer_risk: float = 0.0,
    ) -> dict[str, Any]:
        return {
            "seed_context_skill_count": len(skills) if seed_context_text else 0,
            "seed_context_mode": subtype if seed_context_text else "",
            "seed_routing_mode": seed_routing_mode if seed_context_text else "",
            "seed_context_char_count": len(seed_context_text),
            "seed_context_contains_code": "normalized_code_excerpt" in seed_context_text or "```python" in seed_context_text,
            "seed_context_contains_tests": "replay_assertions" in seed_context_text or "assert " in seed_context_text,
            "transfer_utility_score": round(float(transfer_utility_score), 4),
            "estimated_seed_token_cost": round(float(estimated_seed_token_cost), 4),
            "historical_positive_transfer": round(float(historical_positive_transfer), 4),
            "negative_transfer_risk": round(float(negative_transfer_risk), 4),
        }

    def _visible_tests(self, task: CodeTaskInstance) -> str:
        if task.benchmark == "gsm8k":
            return "Hidden numeric verifier checks that solve() returns the final answer."
        return str(task.test_list)

    def _one_line_task(self, task: CodeTaskInstance, *, max_chars: int = 180) -> str:
        text = " ".join(str(task.prompt or task.text or "").split())
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    def _shape_of_value(self, value: Any) -> str:
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int) and not isinstance(value, bool):
            return "int"
        if isinstance(value, float):
            return "float"
        if isinstance(value, str):
            return "str"
        if isinstance(value, list):
            inner = self._shape_of_value(value[0]) if value else "any"
            return f"list[{inner}]"
        if isinstance(value, tuple):
            if not value:
                return "tuple[]"
            if len(value) <= 3:
                return "tuple[" + ",".join(self._shape_of_value(item) for item in value) + "]"
            return f"tuple[{self._shape_of_value(value[0])},...]"
        if isinstance(value, dict):
            return "dict"
        if value is None:
            return "None"
        return "any"

    def _parse_assert_shapes(self, test_line: str, entry_point: str) -> tuple[int | None, list[str], str] | None:
        try:
            stmt = ast.parse(test_line.strip()).body[0]
        except SyntaxError:
            return None
        if not isinstance(stmt, ast.Assert):
            return None
        call: ast.Call | None = None
        expected_node: ast.AST | None = None
        test_expr = stmt.test
        if isinstance(test_expr, ast.Compare) and len(test_expr.ops) == 1 and len(test_expr.comparators) == 1:
            left = test_expr.left
            if (
                isinstance(left, ast.Call)
                and isinstance(left.func, ast.Name)
                and left.func.id == entry_point
            ):
                call = left
                expected_node = test_expr.comparators[0]
        elif isinstance(test_expr, ast.Call) and isinstance(test_expr.func, ast.Name) and test_expr.func.id == entry_point:
            call = test_expr
        if call is None:
            return None
        arg_shapes: list[str] = []
        for arg in call.args:
            try:
                arg_shapes.append(self._shape_of_value(ast.literal_eval(arg)))
            except Exception:
                arg_shapes.append("any")
        return_shape = "bool"
        if expected_node is not None:
            try:
                return_shape = self._shape_of_value(ast.literal_eval(expected_node))
            except Exception:
                return_shape = "any"
        return len(call.args), arg_shapes, return_shape

    def _shape_summary_from_tests(self, task: CodeTaskInstance, max_examples: int = 2) -> str:
        if task.benchmark == "gsm8k":
            return "shape_summary: arg_count=1; input_shape=text; return_shape=str"
        entry_point = self._infer_entry_point(task)
        parsed = None
        examples: list[str] = []
        for test_line in getattr(task, "test_list", []) or []:
            stripped = str(test_line).strip()
            if stripped and len(examples) < max_examples:
                examples.append(stripped if len(stripped) <= 180 else stripped[:177].rstrip() + "...")
            if parsed is None:
                parsed = self._parse_assert_shapes(stripped, entry_point)
        if parsed is None:
            summary = "shape_summary: arg_count=unknown; input_shape=unknown; return_shape=unknown"
        else:
            arg_count, arg_shapes, return_shape = parsed
            shape_label = ",".join(arg_shapes) if arg_shapes else "none"
            summary = (
                f"shape_summary: arg_count={arg_count}; input_shape={shape_label}; return_shape={return_shape}"
            )
        if not examples:
            return summary
        lines = [summary, "examples:"] + [f"- {example}" for example in examples]
        return "\n".join(lines)

    def _compact_feedback_summary(
        self,
        feedback: VerifierFeedback,
        *,
        max_failed_tests: int = 1,
        max_stderr_chars: int = 120,
    ) -> str:
        failure_type = str(getattr(feedback, "failure_type", "") or "").strip()
        if not failure_type:
            stderr_text = str(getattr(feedback, "stderr", "") or "").strip().lower()
            if "assert" in stderr_text:
                failure_type = "test_failure"
            elif stderr_text:
                failure_type = "runtime_error"
            else:
                failure_type = "unknown"
        failed_tests = [
            str(item).strip()
            for item in list(getattr(feedback, "failed_tests", []) or [])[:max_failed_tests]
            if str(item).strip()
        ]
        stderr_digest = " ".join(str(getattr(feedback, "stderr", "") or "").split())
        if len(stderr_digest) > max_stderr_chars:
            stderr_digest = stderr_digest[: max_stderr_chars - 3].rstrip() + "..."
        lines = [f"failure_summary: type={failure_type}"]
        if failed_tests:
            lines.extend(f"- first_failed_test={item}" for item in failed_tests)
        if stderr_digest:
            lines.append(f"- stderr_digest={stderr_digest}")
        return "\n".join(lines)

    def _compact_previous_candidate(
        self,
        previous_code: str,
        *,
        max_lines: int | None = None,
        max_chars: int | None = None,
    ) -> str:
        compact_lines: list[str] = []
        for raw_line in previous_code.rstrip().splitlines():
            stripped = raw_line.rstrip()
            if not stripped.strip():
                continue
            if stripped.lstrip().startswith("#"):
                continue
            compact_lines.append(stripped)
        if max_lines is not None and max_lines > 0:
            compact_lines = compact_lines[:max_lines]
        compact = "\n".join(compact_lines).strip() or previous_code.rstrip()
        if max_chars is not None and max_chars > 0 and len(compact) > max_chars:
            compact = compact[: max_chars - 3].rstrip() + "..."
        return compact

    def _compact_seed_summary(
        self,
        retrieved_skills: list[dict[str, Any]],
        *,
        subtype: str,
        include_task_pattern: bool = True,
        include_role: bool = True,
    ) -> str:
        if not retrieved_skills:
            return ""
        skill = retrieved_skills[0]
        entry = str(skill.get("entry_point") or skill.get("signature", {}).get("entry_point") or "").strip()
        task_pattern = str(skill.get("task_pattern", "")).strip()
        role = "guard" if subtype == "guard_checker" else "scaffold"
        lines = ["seed_summary:"]
        for part in (
            f"- subtype={subtype}" if subtype else "",
            f"- entry={entry}" if entry else "",
            f"- task_pattern={task_pattern}" if include_task_pattern and task_pattern else "",
            f"- role={role}" if include_role else "",
        ):
            if part:
                lines.append(part)
        return "\n".join(lines)

    def _compact_seeded_plan_prompt(
        self,
        task: CodeTaskInstance,
        *,
        retrieved_skills: list[dict[str, Any]],
        seed_context_text: str = "",
        counting_hint: str = "",
        scaffold_hint_injected: bool = False,
    ) -> str:
        task_context = "\n".join(
            [
                f"entry_point: {self._infer_entry_point(task)}",
                f"task: {self._one_line_task(task)}",
            ]
        )
        tests_context = self._shape_summary_from_tests(task)
        seed_summary = self._compact_seed_summary(retrieved_skills, subtype="plan_skeleton")
        scaffold_anchor = self._plan_skeleton_anchor(task, retrieved_skills=retrieved_skills) if scaffold_hint_injected else ""
        instruction = "instruction: use only the scaffold shape; solve the task directly; keep code short."
        parts = [task_context, tests_context, counting_hint, seed_summary, seed_context_text, scaffold_anchor, instruction]
        return "\n\n".join(part for part in parts if part)

    def _compact_seeded_guard_prompt(
        self,
        task: CodeTaskInstance,
        *,
        retrieved_skills: list[dict[str, Any]],
        seed_context_text: str = "",
    ) -> str:
        task_context = "\n".join(
            [
                f"entry_point: {self._infer_entry_point(task)}",
                f"task: {self._one_line_task(task)}",
            ]
        )
        tests_context = self._shape_summary_from_tests(task)
        seed_summary = self._compact_seed_summary(retrieved_skills, subtype="guard_checker")
        instruction = "instruction: add only the minimal boolean or precondition guard needed for correctness."
        return "\n\n".join(part for part in (task_context, tests_context, seed_summary, seed_context_text, instruction) if part)

    def _compact_seeded_guard_repair_prompt(
        self,
        task: CodeTaskInstance,
        *,
        previous_code: str,
        feedback: VerifierFeedback,
        retrieved_skills: list[dict[str, Any]],
        seed_context_text: str = "",
    ) -> str:
        task_context = "\n".join(
            [
                f"entry_point: {self._infer_entry_point(task)}",
                f"task: {self._one_line_task(task)}",
            ]
        )
        tests_context = self._shape_summary_from_tests(task)
        feedback_context = self._compact_feedback_summary(feedback)
        seed_summary = self._compact_seed_summary(retrieved_skills, subtype="guard_checker")
        previous_candidate = f"previous_candidate:\n{self._compact_previous_candidate(previous_code)}"
        instruction = "repair_instruction: fix the failing logic with minimal semantic change; return the full corrected function."
        return "\n\n".join(
            part
            for part in (
                task_context,
                tests_context,
                previous_candidate,
                feedback_context,
                seed_summary,
                seed_context_text,
                instruction,
            )
            if part
        )

    def _compact_seeded_plan_repair_prompt(
        self,
        task: CodeTaskInstance,
        *,
        previous_code: str,
        feedback: VerifierFeedback,
        retrieved_skills: list[dict[str, Any]],
        seed_context_text: str = "",
        repair_sanity_hint: str = "",
    ) -> str:
        task_context = "\n".join(
            [
                f"entry_point: {self._infer_entry_point(task)}",
                f"task: {self._one_line_task(task)}",
            ]
        )
        tests_context = self._shape_summary_from_tests(task)
        feedback_context = self._compact_feedback_summary(feedback)
        seed_summary = self._compact_seed_summary(retrieved_skills, subtype="plan_skeleton")
        previous_candidate = f"previous_candidate:\n{self._compact_previous_candidate(previous_code)}"
        instruction = "repair_instruction: fix the failing logic with minimal semantic change; return the full corrected function."
        parts = [
            task_context,
            tests_context,
            previous_candidate,
            feedback_context,
            seed_summary,
            seed_context_text,
            repair_sanity_hint,
            instruction,
        ]
        return "\n\n".join(part for part in parts if part)

    def _serialize_skill_context(
        self,
        skills: list[dict[str, Any]],
        subtype: str = "",
        skill_modes: dict[str, str] | None = None,
    ) -> str:
        if not skills:
            return ""
        rendered_parts: list[str] = []
        observed_modes: list[str] = []
        for index, skill in enumerate(skills, start=1):
            skill_id = str(skill.get("skill_id", "")).strip()
            mode = str((skill_modes or {}).get(skill_id) or skill.get("seed_routing_mode") or "skeleton_only_seed")
            observed_modes.append(mode)
            if mode == "guard_only_seed":
                rendered_parts.append(f"prior_{index} guard: {self._guard_only_seed_block(skill)}")
            elif mode == "code_excerpt_seed":
                rendered_parts.append(self._code_excerpt_seed_block(skill, subtype=subtype))
            else:
                rendered_parts.append(f"prior_{index} scaffold: {self._skeleton_only_seed_block(skill)}")
        if subtype:
            rendered_parts.append(f"transfer_instruction: {self._seed_transfer_instruction(subtype)}")
        text = "\n\n".join(part for part in rendered_parts if part)
        primary_mode = observed_modes[0] if observed_modes else ""
        if primary_mode == "guard_only_seed":
            budget = 180 if len(observed_modes) <= 1 else 220
            return self._clip_chars(text, max_chars=budget)
        if primary_mode == "skeleton_only_seed":
            budget = 260 if len(observed_modes) <= 1 else 320
            return self._clip_chars(text, max_chars=budget)
        if primary_mode == "code_excerpt_seed" and len(text) > 520:
            return text[:517].rstrip() + "..."
        return text

    def _plan_skeleton_anchor(
        self,
        task: CodeTaskInstance,
        *,
        retrieved_skills: list[dict[str, Any]],
    ) -> str:
        entry_point = self._infer_entry_point(task)
        skill = retrieved_skills[0] if retrieved_skills else {}
        hint_entry = str(skill.get("entry_point") or skill.get("signature", {}).get("entry_point") or "").strip()
        lines = [
            "starter_scaffold:",
            f"def {entry_point}(...):",
            f"    # seed_anchor: {hint_entry or 'scaffold'}",
            "    core_result = ...  # TODO(core_logic)",
            "    return core_result",
        ]
        return "\n".join(lines)

    def _serialize_template_priors(self, template_priors: list[dict[str, Any]] | None) -> str:
        if not template_priors:
            return ""
        lines = ["template_priors:"]
        for item in template_priors:
            lines.append(
                f"- template_id={item.get('template_id', '')}; kind={item.get('template_kind', '')}; "
                f"directives={item.get('prompt_directives', [])}; preferred_skill_ids={item.get('preferred_skill_ids', [])}"
            )
        return "\n".join(lines)

    def _serialize_workflow_memories(self, workflow_memories: list[dict[str, Any]] | None) -> str:
        if not workflow_memories:
            return ""
        lines = ["workflow_memories:"]
        for item in workflow_memories:
            lines.append(
                f"- memory_id={item.get('memory_id', '')}; workflow_summary={item.get('workflow_summary', '')}; "
                f"repair_summary={item.get('repair_summary', '')}; applicability={item.get('applicability_notes', '')}"
            )
        return "\n".join(lines)

    def _serialize_primitives(self, primitive_helpers: list[dict[str, Any]] | None) -> str:
        if not primitive_helpers:
            return ""
        lines = ["primitive_helpers:"]
        for item in primitive_helpers:
            code_excerpt = ""
            code_path = str(item.get("code_path", "") or "")
            if code_path:
                path = Path(code_path)
                if path.exists():
                    code_excerpt = path.read_text(encoding="utf-8")[:220].strip()
            lines.append(
                f"- primitive_id={item.get('primitive_id', '')}; helper_name={item.get('helper_name', '')}; "
                f"target_entry_point={item.get('target_entry_point', '')}; code_excerpt={code_excerpt}"
            )
        return "\n".join(lines)

    def _subtype_for_prompt(self, planning_mode: str, retrieved_skills: list[dict[str, Any]]) -> str:
        if planning_mode != "skill_seeded_dynamic" or not retrieved_skills:
            return ""
        routing_mode = str(retrieved_skills[0].get("seed_routing_mode", "") or "")
        if routing_mode == "guard_only_seed":
            return "guard_checker"
        if routing_mode in {"skeleton_only_seed", "code_excerpt_seed"}:
            return "plan_skeleton"
        preferred = [str(item) for item in (retrieved_skills[0].get("preferred_usage_subtypes") or []) if str(item)]
        if "plan_skeleton" in preferred:
            return "plan_skeleton"
        if "guard_checker" in preferred:
            return "guard_checker"
        return "plan_skeleton" if bool(retrieved_skills[0].get("exact_entry_match", False)) else "guard_checker"

    def _plan_skeleton_scaffold_status(
        self,
        planning_mode: str,
        retrieved_skills: list[dict[str, Any]],
    ) -> tuple[bool, str]:
        if planning_mode != "skill_seeded_dynamic" or not retrieved_skills:
            return False, ""
        if self._subtype_for_prompt(planning_mode, retrieved_skills) != "plan_skeleton":
            return False, ""
        top = retrieved_skills[0]
        exact_entry_match = bool(top.get("exact_entry_match", False))
        signature_compatible = bool(top.get("signature_compatible", False))
        score = float(top.get("score", 0.0))
        if (exact_entry_match or signature_compatible) and score >= 8.0:
            return True, ""
        return False, "low_confidence_scaffold_suppressed"

    def _seeded_counting_hint(
        self,
        task: CodeTaskInstance,
        *,
        planning_mode: str,
        retrieved_skills: list[dict[str, Any]],
        repair: bool = False,
    ) -> str:
        if planning_mode != "skill_seeded_dynamic" or not retrieved_skills:
            return ""
        subtype = self._subtype_for_prompt(planning_mode, retrieved_skills)
        if subtype != "plan_skeleton":
            return ""
        scaffold_hint_injected, _ = self._plan_skeleton_scaffold_status(planning_mode, retrieved_skills)
        if scaffold_hint_injected:
            return ""
        top_skill_id = str(retrieved_skills[0].get("skill_id", ""))
        if not any(top_skill_id.startswith(prefix) for prefix in self._COUNTING_HELPER_PREFIXES):
            return ""
        query_text = " ".join(
            part for part in (getattr(task, "prompt", ""), getattr(task, "text", "")) if part
        ).lower()
        if not any(keyword in query_text for keyword in self._COMBINATORICS_HINT_KEYWORDS):
            return ""
        if repair:
            return (
                "repair_counting_hint: this is a combinatorial counting problem; ignore unrelated bitwise or helper-specific "
                "semantics, fix the recurrence/formula/counting logic directly, and preserve the target entry point."
            )
        return (
            "counting_hint: this is a combinatorial counting task; treat the retrieved helper as loose structure only, "
            "ignore unrelated bitwise semantics, and derive the count/formula directly from the statement and tests."
        )

    def _repair_counting_sanity_hint(
        self,
        task: CodeTaskInstance,
        *,
        planning_mode: str,
        retrieved_skills: list[dict[str, Any]],
    ) -> str:
        if planning_mode != "skill_seeded_dynamic" or not retrieved_skills:
            return ""
        subtype = self._subtype_for_prompt(planning_mode, retrieved_skills)
        if subtype != "plan_skeleton":
            return ""
        query_text = " ".join(
            part for part in (getattr(task, "prompt", ""), getattr(task, "text", "")) if part
        ).lower()
        if not all(keyword in query_text for keyword in self._GEOMETRIC_COUNTING_KEYWORDS):
            top_skill_id = str(retrieved_skills[0].get("skill_id", ""))
            if not any(top_skill_id.startswith(prefix) for prefix in self._COUNTING_HELPER_PREFIXES):
                return ""
            return ""
        return (
            "repair_sanity_hint: satisfy the small checked cases before generalizing, and do not introduce symmetry "
            "multipliers or geometric counting shortcuts unless they match the provided tests exactly."
        )

    def _draft_input(
        self,
        task: CodeTaskInstance,
        *,
        retrieved_skills: list[dict[str, Any]],
        seed_context_skill_modes: dict[str, str],
        planning_mode: str,
        policy_directives: list[str],
        workflow_memories: list[dict[str, Any]],
        template_priors: list[dict[str, Any]],
        primitive_helpers: list[dict[str, Any]],
    ) -> PromptAssembly:
        subtype = self._subtype_for_prompt(planning_mode, retrieved_skills)
        scaffold_hint_injected, _ = self._plan_skeleton_scaffold_status(planning_mode, retrieved_skills)
        counting_hint = self._seeded_counting_hint(
            task,
            planning_mode=planning_mode,
            retrieved_skills=retrieved_skills,
        )
        template_block = self._serialize_template_priors(template_priors)
        workflow_block = self._serialize_workflow_memories(workflow_memories)
        primitive_block = self._serialize_primitives(primitive_helpers)
        proof_compact_seeded_context = "proof_compact_seeded_context" in {
            str(item).strip() for item in (policy_directives or []) if str(item).strip()
        }
        test_compatible_compact = not any(isinstance(skill.get("compile_metadata"), dict) for skill in retrieved_skills)
        compact_seeded_path = (
            planning_mode == "skill_seeded_dynamic"
            and (proof_compact_seeded_context or test_compatible_compact)
            and not template_block
            and not workflow_block
            and not primitive_block
        )
        if compact_seeded_path and subtype == "plan_skeleton":
            seed_context_text = self._serialize_skill_context(
                retrieved_skills,
                subtype="plan_skeleton",
                skill_modes=seed_context_skill_modes,
            )
            task_context_text = "\n".join(
                [
                    f"entry_point: {self._infer_entry_point(task)}",
                    f"task: {self._one_line_task(task)}",
                ]
            )
            tests_context_text = self._shape_summary_from_tests(task)
            seed_summary_text = self._compact_seed_summary(retrieved_skills, subtype="plan_skeleton")
            scaffold_text = self._plan_skeleton_anchor(task, retrieved_skills=retrieved_skills) if scaffold_hint_injected else counting_hint
            if counting_hint and scaffold_hint_injected:
                scaffold_text = "\n".join(part for part in (counting_hint, scaffold_text) if part)
            return PromptAssembly(
                text=self._compact_seeded_plan_prompt(
                    task,
                    retrieved_skills=retrieved_skills,
                    seed_context_text=seed_context_text,
                    counting_hint=counting_hint,
                    scaffold_hint_injected=scaffold_hint_injected,
                ),
                task_context_text=task_context_text,
                tests_context_text=tests_context_text,
                seed_summary_text=seed_summary_text,
                seed_context_text=seed_context_text,
                scaffold_text=scaffold_text,
                compact_seeded_prompt_used=True,
            )
        if compact_seeded_path and subtype == "guard_checker":
            seed_context_text = self._serialize_skill_context(
                retrieved_skills,
                subtype="guard_checker",
                skill_modes=seed_context_skill_modes,
            )
            return PromptAssembly(
                text=self._compact_seeded_guard_prompt(
                    task,
                    retrieved_skills=retrieved_skills,
                    seed_context_text=seed_context_text,
                ),
                task_context_text="\n".join(
                    [
                        f"entry_point: {self._infer_entry_point(task)}",
                        f"task: {self._one_line_task(task)}",
                    ]
                ),
                tests_context_text=self._shape_summary_from_tests(task),
                seed_summary_text=self._compact_seed_summary(retrieved_skills, subtype="guard_checker"),
                seed_context_text=seed_context_text,
                scaffold_text="guard_checker_hint:",
                compact_seeded_prompt_used=True,
            )
        seed_context_text = ""
        scaffold_text = counting_hint
        task_context_text = "\n".join(
            [
                f"benchmark: {task.benchmark}",
                f"task_id: {task.task_id}",
                f"entry_point: {self._infer_entry_point(task)}",
                f"task: {task.prompt or task.text}",
            ]
        )
        tests_context_text = f"tests: {self._visible_tests(task)}"
        parts = [
            task_context_text,
            tests_context_text,
        ]
        if counting_hint:
            parts.append(counting_hint)
        if template_block:
            parts.append(template_block)
        if workflow_block:
            parts.append(workflow_block)
        if primitive_block:
            parts.append(primitive_block)
        if planning_mode == "skill_seeded_dynamic":
            seed_context_text = self._serialize_skill_context(
                retrieved_skills,
                subtype=subtype or "plan_skeleton",
                skill_modes=seed_context_skill_modes,
            )
            if seed_context_text:
                parts.append(seed_context_text)
            if subtype == "plan_skeleton" and scaffold_hint_injected:
                instruction = "plan_skeleton_instruction: Use only the scaffold shape; keep the target entry point."
                parts.append(instruction)
                scaffold_text = "\n".join(part for part in [scaffold_text, instruction] if part)
            elif subtype == "guard_checker":
                guard_slots = "\n".join(
                    [
                        "forbidden_move_slot: keep only the unsafe moves to avoid",
                        "precondition_check_slot: keep only lightweight precondition checks",
                        "verification_hint_slot: keep only concise local verification hints",
                    ]
                )
                parts.append(guard_slots)
                scaffold_text = "\n".join(part for part in [scaffold_text, guard_slots] if part)
        return PromptAssembly(
            text="\n\n".join(part for part in parts if part),
            seed_context_text=seed_context_text,
            scaffold_text=scaffold_text,
            task_context_text=task_context_text,
            tests_context_text=tests_context_text,
        )

    def _repair_input(
        self,
        task: CodeTaskInstance,
        *,
        previous_code: str,
        feedback: VerifierFeedback,
        retrieved_skills: list[dict[str, Any]],
        seed_context_skill_modes: dict[str, str],
        planning_mode: str,
        policy_directives: list[str],
        workflow_memories: list[dict[str, Any]],
        template_priors: list[dict[str, Any]],
        primitive_helpers: list[dict[str, Any]],
        repair_seeded_demoted: bool = False,
    ) -> PromptAssembly:
        subtype = self._subtype_for_prompt(planning_mode, retrieved_skills)
        scaffold_hint_injected, _ = self._plan_skeleton_scaffold_status(planning_mode, retrieved_skills)
        counting_hint = self._seeded_counting_hint(
            task,
            planning_mode=planning_mode,
            retrieved_skills=retrieved_skills,
            repair=True,
        )
        repair_sanity_hint = self._repair_counting_sanity_hint(
            task,
            planning_mode=planning_mode,
            retrieved_skills=retrieved_skills,
        )
        template_block = self._serialize_template_priors(template_priors)
        workflow_block = self._serialize_workflow_memories(workflow_memories)
        primitive_block = self._serialize_primitives(primitive_helpers)
        proof_compact_seeded_context = "proof_compact_seeded_context" in {
            str(item).strip() for item in (policy_directives or []) if str(item).strip()
        }
        proof_compact_seeded_repair = "proof_compact_seeded_repair" in {
            str(item).strip() for item in (policy_directives or []) if str(item).strip()
        }
        test_compatible_compact = not any(isinstance(skill.get("compile_metadata"), dict) for skill in retrieved_skills)
        compact_seeded_path = (
            planning_mode == "skill_seeded_dynamic"
            and (proof_compact_seeded_context or test_compatible_compact)
            and not template_block
            and not workflow_block
            and not primitive_block
        )
        if compact_seeded_path and subtype == "guard_checker":
            seed_context_text = self._serialize_skill_context(
                retrieved_skills,
                subtype="guard_checker",
                skill_modes=seed_context_skill_modes,
            )
            task_context_text = "\n".join(
                [
                    f"entry_point: {self._infer_entry_point(task)}",
                    f"task: {self._one_line_task(task)}",
                ]
            )
            tests_context_text = self._shape_summary_from_tests(task, max_examples=1 if proof_compact_seeded_repair else 2)
            feedback_context_text = self._compact_feedback_summary(
                feedback,
                max_failed_tests=1,
                max_stderr_chars=80 if proof_compact_seeded_repair else 120,
            )
            previous_code_context_text = self._compact_previous_candidate(
                previous_code,
                max_lines=18 if proof_compact_seeded_repair else None,
                max_chars=520 if proof_compact_seeded_repair else None,
            )
            seed_summary_text = self._compact_seed_summary(
                retrieved_skills,
                subtype="guard_checker",
                include_task_pattern=not proof_compact_seeded_repair,
            )
            prompt_parts = [
                task_context_text,
                tests_context_text,
                f"previous_candidate:\n{previous_code_context_text}",
                feedback_context_text,
                seed_summary_text,
                seed_context_text,
                "repair_instruction: fix the failing logic with minimal semantic change; return the full corrected function.",
            ]
            return PromptAssembly(
                text="\n\n".join(part for part in prompt_parts if part),
                task_context_text=task_context_text,
                tests_context_text=tests_context_text,
                feedback_context_text=feedback_context_text,
                previous_code_context_text=previous_code_context_text,
                seed_summary_text=seed_summary_text,
                seed_context_text=seed_context_text,
                scaffold_text="guard_checker_hint:",
                repair_strategy="full_rewrite",
                compact_seeded_prompt_used=True,
            )
        if compact_seeded_path and subtype == "plan_skeleton":
            seed_context_text = self._serialize_skill_context(
                retrieved_skills,
                subtype="plan_skeleton",
                skill_modes=seed_context_skill_modes,
            )
            task_context_text = "\n".join(
                [
                    f"entry_point: {self._infer_entry_point(task)}",
                    f"task: {self._one_line_task(task)}",
                ]
            )
            tests_context_text = self._shape_summary_from_tests(task, max_examples=1 if proof_compact_seeded_repair else 2)
            feedback_context_text = self._compact_feedback_summary(
                feedback,
                max_failed_tests=1,
                max_stderr_chars=80 if proof_compact_seeded_repair else 120,
            )
            previous_code_context_text = self._compact_previous_candidate(
                previous_code,
                max_lines=18 if proof_compact_seeded_repair else None,
                max_chars=520 if proof_compact_seeded_repair else None,
            )
            seed_summary_text = self._compact_seed_summary(
                retrieved_skills,
                subtype="plan_skeleton",
                include_task_pattern=not proof_compact_seeded_repair,
            )
            prompt_parts = [
                task_context_text,
                tests_context_text,
                f"previous_candidate:\n{previous_code_context_text}",
                feedback_context_text,
                seed_summary_text,
                seed_context_text,
                repair_sanity_hint,
                "repair_instruction: fix the failing logic with minimal semantic change; return the full corrected function.",
            ]
            return PromptAssembly(
                text="\n\n".join(part for part in prompt_parts if part),
                task_context_text=task_context_text,
                tests_context_text=tests_context_text,
                feedback_context_text=feedback_context_text,
                previous_code_context_text=previous_code_context_text,
                seed_summary_text=seed_summary_text,
                seed_context_text=seed_context_text,
                scaffold_text="\n".join(
                    part
                    for part in (
                        counting_hint,
                        repair_sanity_hint,
                    )
                    if part
                ),
                repair_strategy="full_rewrite",
                compact_seeded_prompt_used=True,
            )
        seed_context_text = ""
        scaffold_text = "\n".join(part for part in (counting_hint, repair_sanity_hint) if part)
        task_context_text = "\n".join(
            [
                f"benchmark: {task.benchmark}",
                f"task_id: {task.task_id}",
                f"entry_point: {self._infer_entry_point(task)}",
                f"task: {task.prompt or task.text}",
            ]
        )
        tests_context_text = f"tests: {self._visible_tests(task)}"
        previous_code_context_text = previous_code.rstrip()
        feedback_context_text = (
            f"summary={feedback.summary}\nstderr={feedback.stderr}\nfailed_tests={feedback.failed_tests}"
        )
        parts = [
            task_context_text,
            tests_context_text,
            f"previous_candidate:\n{previous_code_context_text}",
            f"verifier_feedback:\n{feedback_context_text}",
        ]
        if counting_hint:
            parts.append(counting_hint)
        if repair_sanity_hint:
            parts.append(repair_sanity_hint)
        if template_block:
            parts.append(template_block)
        if workflow_block:
            parts.append(workflow_block)
        if primitive_block:
            parts.append(primitive_block)
        if planning_mode == "skill_seeded_dynamic":
            seed_context_text = self._serialize_skill_context(
                retrieved_skills,
                subtype=subtype or "plan_skeleton",
                skill_modes=seed_context_skill_modes,
            )
            if seed_context_text:
                parts.append(seed_context_text)
            if subtype == "plan_skeleton" and scaffold_hint_injected and not repair_seeded_demoted:
                repair_scaffold_hint = "repair_scaffold_hint: keep the earlier scaffold if it still matches the target entry point."
                parts.append(repair_scaffold_hint)
                scaffold_text = "\n".join(part for part in (scaffold_text, repair_scaffold_hint) if part)
            if subtype == "guard_checker":
                precondition_slot = "precondition_check_slot: preserve only the necessary checker constraints while fixing the code."
                parts.append(precondition_slot)
                scaffold_text = "\n".join(part for part in (scaffold_text, precondition_slot) if part)
        return PromptAssembly(
            text="\n\n".join(part for part in parts if part),
            seed_context_text=seed_context_text,
            scaffold_text=scaffold_text,
            task_context_text=task_context_text,
            tests_context_text=tests_context_text,
            feedback_context_text=feedback_context_text,
            previous_code_context_text=previous_code_context_text,
            repair_strategy=("local_patch" if subtype == "guard_checker" else "full_rewrite") if planning_mode == "skill_seeded_dynamic" else "",
        )

    def _sanitize_response(self, text: str) -> str:
        stripped = text.strip()
        stripped = re.sub(r"^```(?:python)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
        return stripped.strip() + "\n"

    def _strip_existing_skill_plan_header(self, code: str) -> str:
        lines = code.splitlines()
        if not lines or lines[0].strip() != self._SKILL_PLAN_START:
            return code
        for index, line in enumerate(lines[1:], start=1):
            if line.strip() == self._SKILL_PLAN_END:
                return "\n".join(lines[index + 1 :]).lstrip() + ("\n" if lines[index + 1 :] else "")
        return code

    def _skill_plan_payload(
        self,
        task: CodeTaskInstance,
        *,
        planning_mode: str,
        selected_skills: list[dict[str, Any]],
        subtype: str,
        policy_directives: list[str] | None = None,
    ) -> dict[str, Any] | None:
        if planning_mode != "skill_seeded_dynamic" or not selected_skills:
            return None
        directives = {
            str(item).strip()
            for item in (policy_directives or [])
            if str(item).strip()
        }
        if (
            "selective_seed_prior_only" in directives
            or "bifurcated_seed_prior_only" in directives
            or "exact_or_seeded_prior_only" in directives
            or "do not insert local skill nodes; use the retrieved seed only to bias the initial plan" in directives
            or "use the retrieved seed only as a planning prior; do not insert local hard skill nodes" in directives
            or "if no exact direct match exists, prefer the proof prior lane over non-exact direct reuse" in directives
        ):
            return None
        top = selected_skills[0]
        helper_name = str(top.get("entry_point") or top.get("signature", {}).get("entry_point") or "").strip()
        role = "local_scaffold" if subtype == "plan_skeleton" else "guard_checker"
        nodes: list[dict[str, Any]] = []
        if subtype == "guard_checker":
            nodes.append(
                {
                    "kind": "guard_check",
                    "name": task.entry_point or self._infer_entry_point(task),
                    "source_skill_entry_point": helper_name,
                }
            )
        elif helper_name:
            nodes.append({"kind": "helper_call", "name": helper_name})
        return {
            "role": role,
            "skill_id": str(top.get("skill_id", "")),
            "planned_usage_subtype": subtype,
            "entry_point": self._infer_entry_point(task),
            "nodes": nodes,
        }

    def _attach_skill_plan_header(self, code: str, payload: dict[str, Any] | None) -> str:
        if not payload:
            return code
        clean_code = self._strip_existing_skill_plan_header(code).rstrip()
        header = "\n".join(
            [
                self._SKILL_PLAN_START,
                "# " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                self._SKILL_PLAN_END,
                "",
            ]
        )
        return header + clean_code + "\n"

    def _code_emits_guard_checker(self, code: str, entry_point: str) -> bool:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        function_node = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and (not entry_point or node.name == entry_point)
            ),
            None,
        )
        if function_node is None:
            return False
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

        for stmt in function_node.body[:3]:
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

    def _call(
        self,
        *,
        stage: str,
        instructions: str,
        input_text: str,
        settings,
        conditioned_skill_ids: list[str],
        scaffold_hint_injected: bool = False,
        guard_checker_requested: bool = False,
        task_entry_point: str = "",
        guard_checker_context_injected: bool = False,
        guard_checker_feedback_used: bool = False,
        scaffold_suppression_reason: str = "",
        repair_seeded_demoted: bool = False,
        repair_demotion_reason: str = "",
        skill_plan_payload: dict[str, Any] | None = None,
        task_context_text: str = "",
        tests_context_text: str = "",
        feedback_context_text: str = "",
        previous_code_context_text: str = "",
        seed_summary_text: str = "",
        seed_context_text: str = "",
        scaffold_text: str = "",
        repair_strategy: str = "",
        compact_seeded_prompt_used: bool = False,
        seed_context_skill_count: int = 0,
        seed_context_mode: str = "",
        seed_routing_mode: str = "",
        seed_context_char_count: int = 0,
        seed_context_contains_code: bool = False,
        seed_context_contains_tests: bool = False,
        transfer_utility_score: float = 0.0,
        estimated_seed_token_cost: float = 0.0,
        historical_positive_transfer: float = 0.0,
        negative_transfer_risk: float = 0.0,
    ) -> GenerationOutput:
        try:
            response = self.client.generate(instructions=instructions, input_text=input_text, settings=settings)
        except LLMClientError as exc:
            event = GenerationEvent(
                generation_stage=stage,
                provider=self.runtime_config.provider,
                model=self.runtime_config.model,
                conditioned_skill_ids=conditioned_skill_ids,
                planned_skill_nodes=list((skill_plan_payload or {}).get("nodes", [])),
                planned_skill_node_count=len(list((skill_plan_payload or {}).get("nodes", []))),
                source_structural_role=str((skill_plan_payload or {}).get("role", "")),
                scaffold_hint_injected=scaffold_hint_injected,
                guard_checker_emitted=False,
                guard_checker_context_injected=guard_checker_context_injected,
                guard_checker_feedback_used=guard_checker_feedback_used,
                scaffold_suppression_reason=scaffold_suppression_reason,
                repair_seeded_demoted=repair_seeded_demoted,
                repair_demotion_reason=repair_demotion_reason,
                seed_context_skill_count=seed_context_skill_count,
                seed_context_mode=seed_context_mode,
                seed_routing_mode=seed_routing_mode,
                seed_context_char_count=seed_context_char_count,
                seed_context_contains_code=seed_context_contains_code,
                seed_context_contains_tests=seed_context_contains_tests,
                transfer_utility_score=transfer_utility_score,
                estimated_seed_token_cost=estimated_seed_token_cost,
                historical_positive_transfer=historical_positive_transfer,
                negative_transfer_risk=negative_transfer_risk,
                compact_seeded_prompt_used=compact_seeded_prompt_used,
                task_context_tokens=0,
                tests_context_tokens=0,
                feedback_context_tokens=0,
                previous_code_context_tokens=0,
                seed_summary_tokens=0,
                seed_context_tokens=0,
                scaffold_tokens=0,
                repair_strategy=repair_strategy,
                error_summary=str(exc),
            )
            raise GenerationError(str(exc), event) from exc

        code = self._sanitize_response(response.text)
        code = self._attach_skill_plan_header(code, skill_plan_payload)
        guard_checker_emitted = bool(guard_checker_requested and self._code_emits_guard_checker(code, task_entry_point))
        component_tokens = self._allocate_prompt_component_tokens(
            response.prompt_tokens,
            {
                "task_context_tokens": task_context_text,
                "tests_context_tokens": tests_context_text,
                "feedback_context_tokens": feedback_context_text,
                "previous_code_context_tokens": previous_code_context_text,
                "seed_summary_tokens": seed_summary_text,
                "seed_context_tokens": seed_context_text,
                "scaffold_tokens": scaffold_text,
            },
        )
        return GenerationOutput(
            code=code,
            event=self._event(
                stage=stage,
                response=response,
                conditioned_skill_ids=conditioned_skill_ids,
                planned_skill_nodes=list((skill_plan_payload or {}).get("nodes", [])),
                planned_skill_node_count=len(list((skill_plan_payload or {}).get("nodes", []))),
                source_structural_role=str((skill_plan_payload or {}).get("role", "")),
                scaffold_hint_injected=scaffold_hint_injected,
                guard_checker_emitted=guard_checker_emitted,
                guard_checker_context_injected=guard_checker_context_injected,
                guard_checker_feedback_used=guard_checker_feedback_used,
                scaffold_suppression_reason=scaffold_suppression_reason,
                repair_seeded_demoted=repair_seeded_demoted,
                repair_demotion_reason=repair_demotion_reason,
                seed_context_skill_count=seed_context_skill_count,
                seed_context_mode=seed_context_mode,
                seed_routing_mode=seed_routing_mode,
                seed_context_char_count=seed_context_char_count,
                seed_context_contains_code=seed_context_contains_code,
                seed_context_contains_tests=seed_context_contains_tests,
                transfer_utility_score=transfer_utility_score,
                estimated_seed_token_cost=estimated_seed_token_cost,
                historical_positive_transfer=historical_positive_transfer,
                negative_transfer_risk=negative_transfer_risk,
                compact_seeded_prompt_used=compact_seeded_prompt_used,
                task_context_tokens=int(component_tokens.get("task_context_tokens", 0)),
                tests_context_tokens=int(component_tokens.get("tests_context_tokens", 0)),
                feedback_context_tokens=int(component_tokens.get("feedback_context_tokens", 0)),
                previous_code_context_tokens=int(component_tokens.get("previous_code_context_tokens", 0)),
                seed_summary_tokens=int(component_tokens.get("seed_summary_tokens", 0)),
                seed_context_tokens=int(component_tokens.get("seed_context_tokens", 0)),
                scaffold_tokens=int(component_tokens.get("scaffold_tokens", 0)),
                repair_strategy=repair_strategy,
            ),
        )

    def draft(
        self,
        task: BenchmarkTaskInstance,
        retrieved_skills: list[dict[str, Any]] | None = None,
        planning_mode: str = "pure_dynamic",
        seed_context_skill_modes: dict[str, str] | None = None,
        seed_routing_mode: str = "",
        transfer_utility_score: float = 0.0,
        estimated_seed_token_cost: float = 0.0,
        historical_positive_transfer: float = 0.0,
        negative_transfer_risk: float = 0.0,
        policy_directives: list[str] | None = None,
        workflow_memories: list[dict[str, Any]] | None = None,
        template_priors: list[dict[str, Any]] | None = None,
        primitive_helpers: list[dict[str, Any]] | None = None,
    ) -> GenerationOutput:
        selected_skills = [] if planning_mode == "pure_dynamic" else self._select_skills(retrieved_skills)
        scaffold_hint_injected, scaffold_suppression_reason = self._plan_skeleton_scaffold_status(planning_mode, selected_skills)
        subtype = self._subtype_for_prompt(planning_mode, selected_skills)
        guard_checker_context_injected = bool(planning_mode == "skill_seeded_dynamic" and subtype == "guard_checker")
        task_entry_point = self._infer_entry_point(task)
        instructions = self._instructions(stage="draft", planning_mode=planning_mode, policy_directives=policy_directives or [])
        prompt = self._draft_input(
            task,
            retrieved_skills=selected_skills,
            seed_context_skill_modes=seed_context_skill_modes or {},
            planning_mode=planning_mode,
            policy_directives=policy_directives or [],
            workflow_memories=workflow_memories or [],
            template_priors=template_priors or [],
            primitive_helpers=primitive_helpers or [],
        )
        seed_context_summary = self._seed_context_summary(
            selected_skills,
            subtype=subtype or "plan_skeleton",
            seed_context_text=prompt.seed_context_text,
            seed_routing_mode=seed_routing_mode,
            transfer_utility_score=transfer_utility_score,
            estimated_seed_token_cost=estimated_seed_token_cost,
            historical_positive_transfer=historical_positive_transfer,
            negative_transfer_risk=negative_transfer_risk,
        )
        return self._call(
            stage="draft",
            instructions=instructions,
            input_text=prompt.text,
            settings=self.runtime_config.draft,
            conditioned_skill_ids=[str(item.get("skill_id", "")) for item in selected_skills if item.get("skill_id")],
            scaffold_hint_injected=scaffold_hint_injected,
            guard_checker_requested=guard_checker_context_injected,
            task_entry_point=task_entry_point,
            guard_checker_context_injected=guard_checker_context_injected,
            guard_checker_feedback_used=False,
            scaffold_suppression_reason=scaffold_suppression_reason,
            task_context_text=prompt.task_context_text,
            tests_context_text=prompt.tests_context_text,
            seed_summary_text=prompt.seed_summary_text,
            seed_context_text=prompt.seed_context_text,
            scaffold_text=prompt.scaffold_text,
            compact_seeded_prompt_used=prompt.compact_seeded_prompt_used,
            seed_context_skill_count=seed_context_summary["seed_context_skill_count"],
            seed_context_mode=seed_context_summary["seed_context_mode"],
            seed_routing_mode=seed_context_summary["seed_routing_mode"],
            seed_context_char_count=seed_context_summary["seed_context_char_count"],
            seed_context_contains_code=seed_context_summary["seed_context_contains_code"],
            seed_context_contains_tests=seed_context_summary["seed_context_contains_tests"],
            transfer_utility_score=seed_context_summary["transfer_utility_score"],
            estimated_seed_token_cost=seed_context_summary["estimated_seed_token_cost"],
            historical_positive_transfer=seed_context_summary["historical_positive_transfer"],
            negative_transfer_risk=seed_context_summary["negative_transfer_risk"],
            skill_plan_payload=self._skill_plan_payload(
                task,
                planning_mode=planning_mode,
                selected_skills=selected_skills,
                subtype=subtype,
                policy_directives=policy_directives or [],
            ),
        )

    def repair(
        self,
        task: BenchmarkTaskInstance,
        previous_code: str,
        feedback: VerifierFeedback,
        attempt_index: int,
        retrieved_skills: list[dict[str, Any]] | None = None,
        planning_mode: str = "pure_dynamic",
        seed_context_skill_modes: dict[str, str] | None = None,
        seed_routing_mode: str = "",
        transfer_utility_score: float = 0.0,
        estimated_seed_token_cost: float = 0.0,
        historical_positive_transfer: float = 0.0,
        negative_transfer_risk: float = 0.0,
        policy_directives: list[str] | None = None,
        workflow_memories: list[dict[str, Any]] | None = None,
        template_priors: list[dict[str, Any]] | None = None,
        primitive_helpers: list[dict[str, Any]] | None = None,
    ) -> GenerationOutput:
        selected_skills = [] if planning_mode == "pure_dynamic" else self._select_skills(retrieved_skills)
        scaffold_hint_injected, scaffold_suppression_reason = self._plan_skeleton_scaffold_status(planning_mode, selected_skills)
        subtype = self._subtype_for_prompt(planning_mode, selected_skills)
        guard_checker_context_injected = bool(planning_mode == "skill_seeded_dynamic" and subtype == "guard_checker")
        guard_checker_feedback_used = bool(guard_checker_context_injected)
        task_entry_point = self._infer_entry_point(task)
        repair_seeded_demoted = bool(
            planning_mode == "skill_seeded_dynamic"
            and subtype == "plan_skeleton"
            and scaffold_hint_injected
        )
        repair_demotion_reason = "seeded_first_attempt_failed" if repair_seeded_demoted else ""
        instructions = self._instructions(stage="repair", planning_mode=planning_mode, policy_directives=policy_directives or [])
        prompt = self._repair_input(
            task,
            previous_code=previous_code,
            feedback=feedback,
            retrieved_skills=selected_skills,
            seed_context_skill_modes=seed_context_skill_modes or {},
            planning_mode=planning_mode,
            policy_directives=policy_directives or [],
            workflow_memories=workflow_memories or [],
            template_priors=template_priors or [],
            primitive_helpers=primitive_helpers or [],
            repair_seeded_demoted=repair_seeded_demoted,
        )
        seed_context_summary = self._seed_context_summary(
            selected_skills,
            subtype=subtype or "plan_skeleton",
            seed_context_text=prompt.seed_context_text,
            seed_routing_mode=seed_routing_mode,
            transfer_utility_score=transfer_utility_score,
            estimated_seed_token_cost=estimated_seed_token_cost,
            historical_positive_transfer=historical_positive_transfer,
            negative_transfer_risk=negative_transfer_risk,
        )
        return self._call(
            stage=f"repair_{attempt_index}",
            instructions=instructions,
            input_text=prompt.text,
            settings=self.runtime_config.repair,
            conditioned_skill_ids=[str(item.get("skill_id", "")) for item in selected_skills if item.get("skill_id")],
            scaffold_hint_injected=scaffold_hint_injected and not repair_seeded_demoted,
            guard_checker_requested=guard_checker_context_injected,
            task_entry_point=task_entry_point,
            guard_checker_context_injected=guard_checker_context_injected,
            guard_checker_feedback_used=guard_checker_feedback_used,
            scaffold_suppression_reason=scaffold_suppression_reason,
            repair_seeded_demoted=repair_seeded_demoted,
            repair_demotion_reason=repair_demotion_reason,
            task_context_text=prompt.task_context_text,
            tests_context_text=prompt.tests_context_text,
            feedback_context_text=prompt.feedback_context_text,
            previous_code_context_text=prompt.previous_code_context_text,
            seed_summary_text=prompt.seed_summary_text,
            seed_context_text=prompt.seed_context_text,
            scaffold_text=prompt.scaffold_text,
            repair_strategy=prompt.repair_strategy,
            compact_seeded_prompt_used=prompt.compact_seeded_prompt_used,
            seed_context_skill_count=seed_context_summary["seed_context_skill_count"],
            seed_context_mode=seed_context_summary["seed_context_mode"],
            seed_routing_mode=seed_context_summary["seed_routing_mode"],
            seed_context_char_count=seed_context_summary["seed_context_char_count"],
            seed_context_contains_code=seed_context_summary["seed_context_contains_code"],
            seed_context_contains_tests=seed_context_summary["seed_context_contains_tests"],
            transfer_utility_score=seed_context_summary["transfer_utility_score"],
            estimated_seed_token_cost=seed_context_summary["estimated_seed_token_cost"],
            historical_positive_transfer=seed_context_summary["historical_positive_transfer"],
            negative_transfer_risk=seed_context_summary["negative_transfer_risk"],
            skill_plan_payload=self._skill_plan_payload(
                task,
                planning_mode=planning_mode,
                selected_skills=selected_skills,
                subtype=subtype,
                policy_directives=policy_directives or [],
            ),
        )


OpenRouterGenerator = LLMGenerator
