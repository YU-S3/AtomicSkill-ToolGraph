"""Minimal workflow-trace compiler for executable benchmark experiments."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from core.schemas import ExecutionTrace, SkillCard


class Compiler:
    """Compile a successful trace into an executable skill artifact."""

    def __init__(self, skills_root: str = "data/skills") -> None:
        self.skills_root = Path(skills_root)
        self.skills_root.mkdir(parents=True, exist_ok=True)

    def _select_passing_attempt(self, trace: ExecutionTrace):
        for attempt in reversed(trace.attempts):
            feedback = attempt.verifier_feedback
            if feedback is not None and feedback.passed:
                return attempt

        paired = list(zip(trace.code_history, trace.verifier_feedback))
        for index, (candidate_code, feedback) in reversed(list(enumerate(paired))):
            if feedback.passed:
                return type(
                    "PassingAttempt",
                    (),
                    {
                        "attempt_index": index,
                        "candidate_code": candidate_code,
                        "verifier_feedback": feedback,
                    },
                )()
        raise ValueError(f"Trace {trace.trace_id} has no passing attempt to compile.")

    def _safe_task_id(self, task_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]+", "_", task_id)

    def _derive_name(self, trace: ExecutionTrace, entry_point: str) -> str:
        benchmark = trace.benchmark or trace.family
        safe_task_id = self._safe_task_id(trace.task_id)
        return f"{benchmark}_{safe_task_id}_{entry_point}"

    def _extract_entry_point(self, code: str) -> str:
        match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code)
        if match:
            return match.group(1)
        return "solve"

    def _build_skill_source(self, trace: ExecutionTrace, candidate_code: str) -> str:
        if trace.benchmark == "humaneval":
            return trace.query.rstrip() + "\n" + candidate_code.rstrip() + "\n"
        return candidate_code.rstrip() + "\n"

    def _expr_to_string(self, node: ast.AST | None) -> str:
        if node is None:
            return "Any"
        if hasattr(ast, "unparse"):
            return ast.unparse(node)
        return "Any"

    def _extract_signature(self, skill_source: str, entry_point: str) -> dict[str, Any]:
        tree = ast.parse(skill_source)
        function_node = next(
            (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == entry_point),
            None,
        )
        args: list[dict[str, str]] = []
        returns = "Any"
        if function_node is not None:
            for arg in function_node.args.args:
                args.append({"name": arg.arg, "annotation": self._expr_to_string(arg.annotation)})
            returns = self._expr_to_string(function_node.returns)
        return {"entry_point": entry_point, "args": args, "returns": returns}

    def _extract_literal_replay_cases(self, executed_code: str, entry_point: str) -> list[tuple[list[Any], Any]]:
        if not executed_code or not entry_point:
            return []
        try:
            tree = ast.parse(executed_code)
        except SyntaxError:
            return []
        cases: list[tuple[list[Any], Any]] = []
        for node in tree.body:
            if not isinstance(node, ast.Assert):
                continue
            expr = node.test
            if not isinstance(expr, ast.Compare) or len(expr.ops) != 1 or not isinstance(expr.ops[0], ast.Eq):
                continue
            if not isinstance(expr.left, ast.Call):
                continue
            call = expr.left
            if not isinstance(call.func, ast.Name) or call.func.id != entry_point:
                continue
            try:
                args = [ast.literal_eval(arg) for arg in call.args]
                expected = ast.literal_eval(expr.comparators[0])
            except Exception:
                continue
            cases.append((args, expected))
        return cases

    def _infer_task_pattern(
        self,
        trace: ExecutionTrace,
        *,
        entry_point: str,
        replay_cases: list[tuple[list[Any], Any]],
    ) -> str:
        text = (trace.query or "").lower()
        if trace.benchmark == "humaneval":
            return "function_synthesis"
        numeric_formula_hint = any(
            token in text
            for token in (
                "angle",
                "area",
                "volume",
                "radius",
                "perimeter",
                "distance",
                "power",
                "temperature",
                "double",
                "triple",
                "formula",
            )
        ) or entry_point.startswith(("area", "volume", "find_", "even_", "double", "third_", "convert_"))
        if trace.benchmark == "gsm8k":
            if any(token in text for token in ("percent", "%", "discount", "tax", "interest", "tip")):
                return "gsm8k_percentage"
            if any(token in text for token in ("combination", "combinations", "permutation", "arrangement", "ways", "choose")):
                return "gsm8k_counting_combinatorics"
            if any(token in text for token in ("ratio", "rate", "per ", "each minute", "each hour", "mph")):
                return "gsm8k_rate_ratio"
            if any(token in text for token in ("dollar", "dollars", "cent", "cents", "price", "cost", "money", "hour", "minute", "day", "week")):
                return "gsm8k_time_money"
            if any(token in text for token in ("remaining", "left", "twice", "half", "sum of", "difference", "equation")):
                return "gsm8k_algebraic_word_problem"
            return "gsm8k_arithmetic"
        if any(token in text for token in ("sort", "ascending", "descending", "ordered", "rearrange")):
            return "sort"
        if any(
            token in text
            for token in (
                "count",
                "number of",
                "digits",
                "digit",
                "frequency",
                "occurrence",
                "ways",
            )
        ):
            return "count"
        if replay_cases and all(isinstance(expected, bool) for _args, expected in replay_cases):
            arity = max((len(args) for args, _expected in replay_cases), default=0)
            if arity <= 1:
                return "scalar_guard_predicate"
            return "multiarg_relation_or_classifier"
        if replay_cases:
            scalar_numeric_cases = all(
                all(not isinstance(arg, (list, tuple, dict, set, str, bytes)) for arg in args)
                and isinstance(expected, (int, float))
                and not isinstance(expected, bool)
                for args, expected in replay_cases
            )
            if scalar_numeric_cases and numeric_formula_hint:
                arity = max((len(args) for args, _expected in replay_cases), default=0)
                if arity <= 1:
                    return "numeric_formula_single_arg"
                return "numeric_formula_multi_arg"
        if any(
            token in text
            for token in (
                "check",
                "find",
                "search",
                "whether",
                "valid",
                "sum of",
                "minimum",
                "maximum",
                "divisor",
                "angle",
                "area",
                "volume",
            )
        ):
            if replay_cases:
                first_args = replay_cases[0][0]
                if len(first_args) >= 2 and isinstance(first_args[0], (list, tuple, str)):
                    return "predicate_or_search"
            return "predicate_or_search"
        return "generic_function_synthesis"

    def _helper_kind(self, trace: ExecutionTrace) -> str:
        if trace.benchmark == "humaneval":
            return "humaneval_function_skill"
        if trace.benchmark == "gsm8k":
            return "gsm8k_function_skill"
        return "generic_function_skill"

    def _helper_names(self, skill_source: str, *, entry_point: str) -> list[str]:
        try:
            tree = ast.parse(skill_source)
        except SyntaxError:
            return []
        return [
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name != entry_point
        ]

    def _helper_shape_key(self, helper_names: list[str]) -> str:
        if not helper_names:
            return "no_helpers"
        normalized = sorted({re.sub(r"\d+$", "", name) for name in helper_names if name})
        if not normalized:
            return "no_helpers"
        return "|".join(normalized[:4])

    def _transfer_cluster_key(
        self,
        trace: ExecutionTrace,
        *,
        signature: dict[str, Any],
        task_pattern: str,
        helper_shape_key: str,
    ) -> str:
        arg_count = len(signature.get("args") or [])
        returns = str(signature.get("returns", "") or "Any")
        benchmark = str(trace.benchmark or trace.family or "")
        return f"{benchmark}::{task_pattern}::{arg_count}::{returns}::{helper_shape_key}"

    def _preferred_usage_subtypes(self, trace: ExecutionTrace, *, task_pattern: str) -> list[str]:
        trace_subtypes = [
            str(item)
            for item in (
                trace.realized_usage_subtype,
                trace.planned_usage_subtype,
            )
            if str(item).strip()
        ]
        if trace_subtypes:
            return list(dict.fromkeys(trace_subtypes))
        if trace.benchmark == "gsm8k":
            return ["plan_skeleton", "guard_checker"]
        if task_pattern in {"function_synthesis", "generic_function_synthesis", "humaneval_function_synthesis"}:
            return ["plan_skeleton", "guard_checker"]
        if task_pattern in {
            "count",
            "numeric_formula_single_arg",
            "numeric_formula_multi_arg",
            "sort",
        }:
            return ["plan_skeleton"]
        return ["guard_checker"]

    def _compile_metadata(
        self,
        trace: ExecutionTrace,
        *,
        preferred_usage_subtypes: list[str],
        signature: dict[str, Any],
        task_pattern: str,
        helper_names: list[str],
        replay_assertion_count: int,
        source_line_count: int,
    ) -> dict[str, Any]:
        helper_shape_key = self._helper_shape_key(helper_names)
        return {
            "source_realized_usage_subtype": str(trace.realized_usage_subtype or ""),
            "source_structural_role": str(trace.realized_usage_subtype or trace.planned_usage_subtype or ""),
            "source_planned_nodes": list(trace.planned_skill_nodes or []),
            "source_realized_nodes": list(trace.realized_skill_nodes or []),
            "compile_episode_provenance": str(trace.episode_provenance or ""),
            "phase_a_bank_source": str(trace.phase_a_bank_source or ""),
            "phase_a_seed": trace.phase_a_seed,
            "admission_mode": "shadow" if trace.episode_provenance == "seeded_dynamic_success" else "active_candidate",
            "preferred_usage_subtypes": list(preferred_usage_subtypes),
            "helper_names": list(helper_names),
            "helper_function_count": len(helper_names),
            "helper_shape_key": helper_shape_key,
            "replay_assertion_count": int(replay_assertion_count),
            "source_line_count": int(source_line_count),
            "transfer_cluster_key": self._transfer_cluster_key(
                trace,
                signature=signature,
                task_pattern=task_pattern,
                helper_shape_key=helper_shape_key,
            ),
        }

    def _direct_eligible(self, trace: ExecutionTrace, entry_point: str) -> bool:
        if trace.benchmark in {"gsm8k", "humaneval"}:
            return False
        return bool(entry_point)

    def _build_preconditions(
        self,
        trace: ExecutionTrace,
        entry_point: str,
        *,
        direct_eligible: bool,
        preferred_usage_subtypes: list[str],
        helper_kind: str,
        task_pattern: str,
    ) -> dict[str, Any]:
        return {
            "benchmark": trace.benchmark,
            "language": "python",
            "callable_type": "python_function",
            "entry_point": entry_point,
            "task_pattern": task_pattern,
            "helper_kind": helper_kind,
            "negative_preconditions": {
                "fallback_required_when": [
                    "benchmark_mismatch",
                    "missing_exact_entry_match",
                ]
            },
            "direct_eligible": bool(direct_eligible),
            "preferred_usage_subtypes": list(preferred_usage_subtypes),
        }

    def _build_tests(
        self,
        trace: ExecutionTrace,
        executed_code: str,
        skill_source: str,
        entry_point: str,
    ) -> list[dict]:
        if trace.benchmark == "humaneval":
            marker = "def check(candidate):"
            if marker in executed_code:
                replay = executed_code.split(marker, 1)[1]
                replay = marker + replay
                replay = replay.replace(f"check({entry_point})", "")
                replay = replay.rstrip() + f"\n\ncheck({entry_point})"
                return [{"name": "verifier_replay", "type": "python_assertions", "code": replay, "source": "humaneval_verifier"}]
            return []

        if skill_source not in executed_code:
            return []
        suffix = executed_code.split(skill_source, 1)[1].strip()
        if not suffix:
            return []
        source = "gsm8k_verifier" if trace.benchmark == "gsm8k" else "assert_verifier"
        return [{"name": "verifier_replay", "type": "python_assertions", "code": suffix, "source": source}]

    def compile(self, trace: ExecutionTrace, provenance: str) -> tuple[SkillCard, str]:
        if not trace.success:
            raise ValueError(f"Trace {trace.trace_id} is not successful and cannot be compiled.")

        attempt = self._select_passing_attempt(trace)
        candidate_code = attempt.candidate_code.rstrip() + "\n"
        feedback = attempt.verifier_feedback
        skill_source = self._build_skill_source(trace, candidate_code)
        entry_point = self._extract_entry_point(skill_source)
        direct_eligible = self._direct_eligible(trace, entry_point)
        helper_kind = self._helper_kind(trace)
        replay_cases = self._extract_literal_replay_cases(feedback.executed_code, entry_point)
        task_pattern = self._infer_task_pattern(trace, entry_point=entry_point, replay_cases=replay_cases)
        preferred_usage_subtypes = self._preferred_usage_subtypes(trace, task_pattern=task_pattern)
        safe_task_id = self._safe_task_id(trace.task_id)
        skill_id = f"skill_{trace.benchmark}_{safe_task_id}_{trace.trace_id}"
        name = self._derive_name(trace, entry_point)
        tests = self._build_tests(trace, feedback.executed_code, skill_source, entry_point)
        signature = self._extract_signature(skill_source, entry_point)
        helper_names = self._helper_names(skill_source, entry_point=entry_point)
        source_line_count = len([line for line in skill_source.splitlines() if line.strip()])
        code_path = self.skills_root / f"{skill_id}.py"

        card = SkillCard(
            skill_id=skill_id,
            name=name,
            description=f"Compiled executable skill for {trace.benchmark} task {trace.task_id} from successful trace {trace.trace_id}.",
            family=trace.benchmark or trace.family,
            signature=signature,
            preconditions=self._build_preconditions(
                trace,
                entry_point,
                direct_eligible=direct_eligible,
                preferred_usage_subtypes=preferred_usage_subtypes,
                helper_kind=helper_kind,
                task_pattern=task_pattern,
            ),
            direct_eligible=direct_eligible,
            preferred_usage_subtypes=preferred_usage_subtypes,
            direct_reuse_evidence_count=1 if direct_eligible and trace.benchmark != "gsm8k" else 0,
            code_path=str(code_path),
            tests=tests,
            utility=1.0,
            failure_log=[],
            lineage=[trace.trace_id],
            provenance=provenance,
            compile_metadata=self._compile_metadata(
                trace,
                preferred_usage_subtypes=preferred_usage_subtypes,
                signature=signature,
                task_pattern=task_pattern,
                helper_names=helper_names,
                replay_assertion_count=len(tests),
                source_line_count=source_line_count,
            ),
            status="draft",
        )
        return card, skill_source
