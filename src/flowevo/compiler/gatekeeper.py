"""Minimal gatekeeper for compiled code skills."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

from core.schemas import SkillCard
from env.sandbox import Sandbox


@dataclass
class GatekeeperResult:
    """Outcome of gatekeeper checks."""

    allowed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


class GateKeeper:
    """Run minimal validation before a compiled skill enters the registry."""

    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox

    def _unit_test_replay(self, code: str, card: SkillCard) -> tuple[bool, str]:
        test_blocks = [test.get("code", "") for test in card.tests if test.get("code")]
        if not test_blocks:
            return True, "no replay test available"
        program = "\n\n".join([code.rstrip(), *[block.rstrip() for block in test_blocks]])
        result = self.sandbox.run(program)
        if result["passed"]:
            return True, "unit-test replay passed"
        return False, f"unit-test replay failed: {result['stderr'] or result['stdout']}"

    def _static_safety_scan(self, code: str) -> tuple[bool, str]:
        banned_imports = {"os", "subprocess", "socket", "requests", "pathlib"}
        banned_calls = {"eval", "exec", "compile", "__import__", "open"}
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return False, f"syntax error: {exc}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in banned_imports:
                        return False, f"banned import: {root}"
            if isinstance(node, ast.ImportFrom):
                module = (node.module or "").split(".")[0]
                if module in banned_imports:
                    return False, f"banned import: {module}"
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in banned_calls:
                    return False, f"banned call: {node.func.id}"
        return True, "static safety scan passed"

    def _trivial_solution_check(self, code: str, card: SkillCard) -> tuple[bool, str]:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return False, f"syntax error: {exc}"

        function_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if not function_nodes:
            return False, "no function definition found"

        for function_node in function_nodes:
            body = function_node.body
            if len(body) == 1 and isinstance(body[0], ast.Return):
                value = body[0].value
                if isinstance(value, ast.Constant) and value.value in {None, True, False, "", 0}:
                    return False, "single constant-return solution detected"
                if isinstance(value, (ast.List, ast.Set, ast.Tuple)) and not value.elts:
                    return False, "single empty-container return detected"
                if isinstance(value, ast.Dict) and not value.keys:
                    return False, "single empty-container return detected"
        if card.provenance in {"dynamic_success", "pure_dynamic_success", "seeded_dynamic_success"} and "canonical_solution" in code:
            return False, "dynamic skill appears to embed oracle-only artifacts"
        return True, "trivial-solution check passed"

    def _negative_precondition_adequacy_check(self, card: SkillCard) -> tuple[bool, str]:
        negative = card.preconditions.get("negative_preconditions") or {}
        fallbacks = list(negative.get("fallback_required_when") or [])
        if not fallbacks:
            return False, "negative-precondition adequacy failed: missing fallback rules"
        if "missing_exact_entry_match" not in fallbacks and "benchmark_mismatch" not in fallbacks:
            return False, "negative-precondition adequacy failed: missing entry/benchmark fallback rule"
        return True, "negative-precondition adequacy passed"

    def _extract_literal_replay_cases(self, card: SkillCard) -> list[tuple[list[object], object]]:
        entry_point = str(card.signature.get("entry_point", "")).strip()
        if not entry_point:
            return []
        cases: list[tuple[list[object], object]] = []
        for test in card.tests:
            block = str(test.get("code", "")).strip()
            if not block:
                continue
            try:
                tree = ast.parse(block)
            except SyntaxError:
                continue
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

    def _supported_task_pattern(self, card: SkillCard) -> str:
        return str((card.preconditions or {}).get("task_pattern", "") or "")

    def _humaneval_seedable_family(self, card: SkillCard) -> bool:
        benchmark = str((card.preconditions or {}).get("benchmark", card.family) or "")
        return benchmark == "humaneval" or self._supported_task_pattern(card) in {
            "function_synthesis",
            "generic_function_synthesis",
            "humaneval_function_synthesis",
        }

    def _mutate_scalar(self, value: object) -> object | None:
        if isinstance(value, bool):
            return not value
        if isinstance(value, int):
            return value + 1
        if isinstance(value, float):
            return value + 1.0
        if isinstance(value, str):
            return value + "_x"
        return None

    def _shape_preserving_seed_variants(self, args: list[object]) -> list[list[object]]:
        if not args:
            return []
        variants: list[list[object]] = []
        base = list(args)
        first = base[0]
        if isinstance(first, list):
            if first:
                variants.append([list(first) + [first[0]], *args[1:]])
                variants.append([list(reversed(first)), *args[1:]])
            else:
                filler = args[1] if len(args) > 1 else 0
                variants.append([[filler], *args[1:]])
        elif isinstance(first, str):
            variants.append([first + (first[:1] or "x"), *args[1:]])
        else:
            mutated = self._mutate_scalar(first)
            if mutated is not None:
                changed = list(args)
                changed[0] = mutated
                variants.append(changed)
        if len(args) > 1:
            second = args[1]
            mutated_second = self._mutate_scalar(second)
            if mutated_second is not None:
                changed = list(args)
                changed[1] = mutated_second
                variants.append(changed)
        unique: list[list[object]] = []
        seen: set[str] = set()
        for variant in variants:
            marker = repr(variant)
            if marker in seen:
                continue
            seen.add(marker)
            unique.append(variant)
        return unique

    def _source_oracle_seed_replay_program(self, entry_point: str, args: list[object], task_pattern: str) -> str | None:
        variants = self._shape_preserving_seed_variants(args)
        if task_pattern not in {"count", "mbpp_general"}:
            return None
        if not variants:
            return None
        lines = [
            f"_base_args = {repr(list(args))}",
            f"_base = {entry_point}(*_base_args)",
            "assert _base is not None",
        ]
        for index, variant in enumerate(variants[:2]):
            lines.extend(
                [
                    f"_variant_{index} = {repr(list(variant))}",
                    f"_result_{index} = {entry_point}(*_variant_{index})",
                    f"assert type(_result_{index}) is type(_base)",
                ]
            )
        return "\n".join(lines)

    def _source_oracle_seed_replay(self, code: str, card: SkillCard) -> tuple[int, int, dict[str, int]]:
        task_pattern = self._supported_task_pattern(card)
        entry_point = str(card.signature.get("entry_point", "")).strip()
        cases = self._extract_literal_replay_cases(card)
        if not entry_point or not cases:
            return 0, 0, {"no_literal_cases": 1}
        attempted = 0
        passed = 0
        reason_histogram: dict[str, int] = {}
        for args, _expected in cases:
            program = self._source_oracle_seed_replay_program(entry_point, args, task_pattern)
            if not program:
                reason_histogram["unsupported_seed_replay_family"] = reason_histogram.get("unsupported_seed_replay_family", 0) + 1
                continue
            attempted += 1
            result = self.sandbox.run("\n\n".join([code.rstrip(), program.rstrip()]))
            if result["passed"]:
                passed += 1
                reason_histogram["passed"] = reason_histogram.get("passed", 0) + 1
            else:
                key = "seed_replay_failed"
                reason_histogram[key] = reason_histogram.get(key, 0) + 1
        return attempted, passed, reason_histogram

    def _sort_perturbation_program(self, entry_point: str, args: list[object]) -> str | None:
        if not args:
            return None
        seq = args[0]
        if not isinstance(seq, (list, tuple)):
            return None
        mutated = list(reversed(list(seq)))
        extra_args = ", ".join(repr(arg) for arg in args[1:])
        tail = f", {extra_args}" if extra_args else ""
        return "\n".join(
            [
                f"_perturbed = {repr(mutated)}",
                f"_result = {entry_point}(_perturbed{tail})",
                "assert _result == sorted(_perturbed)",
                "assert sorted(_result) == sorted(_perturbed)",
            ]
        )

    def _count_perturbation_program(self, entry_point: str, args: list[object]) -> str | None:
        if len(args) < 2:
            return None
        seq = args[0]
        target = args[1]
        if not isinstance(seq, list):
            return None
        distractor = "__distractor__" if target != "__distractor__" else "__alt_distractor__"
        if isinstance(target, int):
            distractor = target + 1 if target != target + 1 else target + 2
        elif isinstance(target, float):
            distractor = target + 1.0
        elif isinstance(target, bool):
            distractor = not target
        elif isinstance(target, str):
            distractor = target + "_x"
        extra_args = ", ".join(repr(arg) for arg in args[2:])
        extra_tail = f", {extra_args}" if extra_args else ""
        return "\n".join(
            [
                f"_base_seq = {repr(seq)}",
                f"_target = {repr(target)}",
                f"_base = {entry_point}(_base_seq, _target{extra_tail})",
                f"assert {entry_point}(_base_seq + [{repr(distractor)}], _target{extra_tail}) == _base",
                f"assert {entry_point}(_base_seq + [_target], _target{extra_tail}) == _base + 1",
            ]
        )

    def _predicate_or_search_perturbation_program(self, entry_point: str, args: list[object]) -> str | None:
        if len(args) < 2:
            return None
        seq = args[0]
        target = args[1]
        if isinstance(seq, list):
            distractor = "__distractor__" if target != "__distractor__" else "__alt_distractor__"
            if isinstance(target, int):
                distractor = target + 1 if target != target + 1 else target + 2
            elif isinstance(target, float):
                distractor = target + 1.0
            elif isinstance(target, bool):
                distractor = not target
            elif isinstance(target, str):
                distractor = target + "_x"
            extra_args = ", ".join(repr(arg) for arg in args[2:])
            extra_tail = f", {extra_args}" if extra_args else ""
            return "\n".join(
                [
                    f"_base_seq = {repr(seq)}",
                    f"_target = {repr(target)}",
                    f"_base = {entry_point}(_base_seq, _target{extra_tail})",
                    f"assert {entry_point}(_base_seq + [{repr(distractor)}], _target{extra_tail}) == _base",
                ]
            )
        if isinstance(seq, str) and isinstance(target, str):
            distractor = "_x" if target != "_x" else "_y"
            extra_args = ", ".join(repr(arg) for arg in args[2:])
            extra_tail = f", {extra_args}" if extra_args else ""
            return "\n".join(
                [
                    f"_base_seq = {repr(seq)}",
                    f"_target = {repr(target)}",
                    f"_base = {entry_point}(_base_seq, _target{extra_tail})",
                    f"assert {entry_point}(_base_seq + {repr(distractor)}, _target{extra_tail}) == _base",
                ]
            )
        return None

    def _scalar_guard_predicate_perturbation_program(self, entry_point: str, args: list[object]) -> str | None:
        if len(args) != 1:
            return None
        mutated = self._mutate_scalar(args[0])
        if mutated is None:
            return None
        return "\n".join(
            [
                f"_arg = {repr(args[0])}",
                f"_base = {entry_point}(_arg)",
                f"_repeat = {entry_point}(_arg)",
                "assert isinstance(_base, bool)",
                "assert _base == _repeat",
                f"_mutated = {repr(mutated)}",
                f"assert isinstance({entry_point}(_mutated), bool)",
            ]
        )

    def _multiarg_relation_or_classifier_perturbation_program(self, entry_point: str, args: list[object]) -> str | None:
        if len(args) < 2 or any(isinstance(arg, (list, tuple, dict, set)) for arg in args):
            return None
        mutated = list(args)
        replacement = self._mutate_scalar(mutated[0])
        if replacement is None:
            return None
        mutated[0] = replacement
        arg_list = ", ".join(repr(arg) for arg in args)
        mutated_list = ", ".join(repr(arg) for arg in mutated)
        return "\n".join(
            [
                f"_args = [{arg_list}]",
                f"_base = {entry_point}(*_args)",
                f"_repeat = {entry_point}(*_args)",
                "assert isinstance(_base, bool)",
                "assert _base == _repeat",
                f"_mutated = [{mutated_list}]",
                f"assert isinstance({entry_point}(*_mutated), bool)",
            ]
        )

    def _numeric_formula_single_arg_perturbation_program(self, entry_point: str, args: list[object]) -> str | None:
        if len(args) != 1 or isinstance(args[0], (list, tuple, dict, set, str, bytes, bool)):
            return None
        mutated = self._mutate_scalar(args[0])
        if mutated is None or isinstance(mutated, bool):
            return None
        return "\n".join(
            [
                f"_arg = {repr(args[0])}",
                f"_base = {entry_point}(_arg)",
                f"_repeat = {entry_point}(_arg)",
                "assert isinstance(_base, (int, float))",
                "assert _base == _repeat",
                f"_mutated = {repr(mutated)}",
                f"assert isinstance({entry_point}(_mutated), (int, float))",
            ]
        )

    def _numeric_formula_multi_arg_perturbation_program(self, entry_point: str, args: list[object]) -> str | None:
        if len(args) < 2 or any(isinstance(arg, (list, tuple, dict, set, str, bytes, bool)) for arg in args):
            return None
        mutated = list(args)
        replacement = self._mutate_scalar(mutated[0])
        if replacement is None or isinstance(replacement, bool):
            return None
        mutated[0] = replacement
        arg_list = ", ".join(repr(arg) for arg in args)
        mutated_list = ", ".join(repr(arg) for arg in mutated)
        return "\n".join(
            [
                f"_args = [{arg_list}]",
                f"_base = {entry_point}(*_args)",
                f"_repeat = {entry_point}(*_args)",
                "assert isinstance(_base, (int, float))",
                "assert _base == _repeat",
                f"_mutated = [{mutated_list}]",
                f"assert isinstance({entry_point}(*_mutated), (int, float))",
            ]
        )

    def _same_family_perturbation_replay(self, code: str, card: SkillCard) -> tuple[bool, bool, str]:
        task_pattern = self._supported_task_pattern(card)
        entry_point = str(card.signature.get("entry_point", "")).strip()
        cases = self._extract_literal_replay_cases(card)
        if not entry_point or not cases:
            return True, False, "unsupported_family_replay"
        builders = {
            "sort": self._sort_perturbation_program,
            "count": self._count_perturbation_program,
            "predicate_or_search": self._predicate_or_search_perturbation_program,
            "scalar_guard_predicate": self._scalar_guard_predicate_perturbation_program,
            "multiarg_relation_or_classifier": self._multiarg_relation_or_classifier_perturbation_program,
            "numeric_formula_single_arg": self._numeric_formula_single_arg_perturbation_program,
            "numeric_formula_multi_arg": self._numeric_formula_multi_arg_perturbation_program,
        }
        builder = builders.get(task_pattern)
        if builder is None:
            return True, False, "unsupported_family_replay"
        program = None
        for args, _expected in cases:
            program = builder(entry_point, args)
            if program:
                break
        if not program:
            return True, False, "unsupported_family_replay"
        result = self.sandbox.run("\n\n".join([code.rstrip(), program.rstrip()]))
        if result["passed"]:
            return True, True, "same-family perturbation replay passed"
        return False, True, f"same-family perturbation replay failed: {result['stderr'] or result['stdout']}"

    def validate(self, code: str, card: SkillCard) -> GatekeeperResult:
        checks: dict[str, bool] = {}
        reasons: list[str] = []

        passed, reason = self._unit_test_replay(code, card)
        checks["fresh_world_source_replay"] = passed
        if not passed:
            reasons.append(reason)

        passed, supported, reason = self._same_family_perturbation_replay(code, card)
        checks["same_family_perturbation_replay"] = passed
        checks["same_family_perturbation_replay_supported"] = supported
        checks["unsupported_family_replay"] = not supported
        if supported and not passed:
            reasons.append(reason)

        passed, reason = self._static_safety_scan(code)
        checks["static_safety_scan"] = passed
        if not passed:
            reasons.append(reason)

        passed, reason = self._trivial_solution_check(code, card)
        checks["trivial_solution_check"] = passed
        if not passed:
            reasons.append(reason)

        passed, reason = self._negative_precondition_adequacy_check(card)
        checks["negative_precondition_adequacy"] = passed
        if not passed:
            reasons.append(reason)

        seed_replay_attempt_count, seed_replay_pass_count, seed_replay_reason_histogram = self._source_oracle_seed_replay(
            code,
            card,
        )

        enforced_checks = {
            key: value
            for key, value in checks.items()
            if key not in {"same_family_perturbation_replay_supported", "unsupported_family_replay"}
        }
        metrics = {
            "seed_replay_attempt_count": seed_replay_attempt_count,
            "seed_replay_pass_count": seed_replay_pass_count,
            "seed_replay_pass_rate": round(seed_replay_pass_count / seed_replay_attempt_count, 4)
            if seed_replay_attempt_count
            else 0.0,
            "seed_replay_reason_histogram": seed_replay_reason_histogram,
            "same_family_replay_diagnostic_only": self._humaneval_seedable_family(card)
            and not bool(checks.get("same_family_perturbation_replay_supported", False)),
        }
        return GatekeeperResult(allowed=all(enforced_checks.values()), checks=checks, reasons=reasons, metrics=metrics)
