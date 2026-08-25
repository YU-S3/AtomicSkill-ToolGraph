"""Common helpers for FlowEvo."""

from __future__ import annotations

import ast
from pathlib import Path
import re
from typing import Any

from core.schemas import BenchmarkTaskInstance

MBPP_SCORE_PROTOCOLS = {"table1_compatible", "stress_plus_challenge"}


def ensure_project_dirs() -> None:
    """Ensure runtime data directories exist."""
    for p in ("data/traces", "data/skills", "data/tasks"):
        Path(p).mkdir(parents=True, exist_ok=True)


def visible_public_tests(task: BenchmarkTaskInstance) -> list[str]:
    """Return the prompt-visible tests for a task."""
    tests = list(getattr(task, "visible_public_tests", []) or [])
    if tests:
        return tests
    return list(getattr(task, "test_list", []) or [])


def resolve_mbpp_score_protocol(score_protocol: str) -> str:
    cleaned = str(score_protocol or "table1_compatible").strip() or "table1_compatible"
    if cleaned not in MBPP_SCORE_PROTOCOLS:
        raise ValueError(f"Unsupported MBPP score protocol: {cleaned}")
    return cleaned


def hidden_table1_tests(task: BenchmarkTaskInstance) -> list[str]:
    tests = list(getattr(task, "hidden_table1_tests", []) or [])
    if tests:
        return [str(test).strip() for test in tests if str(test).strip()]
    legacy = list(getattr(task, "table1_hidden_tests", []) or [])
    if legacy:
        return [str(test).strip() for test in legacy if str(test).strip()]
    visible = visible_public_tests(task)
    baseline = list(getattr(task, "test_list", []) or [])
    hidden = baseline[len(visible):] if baseline[: len(visible)] == visible else baseline[1:]
    return [str(test).strip() for test in hidden if str(test).strip()]


def hidden_stress_tests(task: BenchmarkTaskInstance) -> list[str]:
    tests = list(getattr(task, "hidden_stress_tests", []) or [])
    if tests:
        return [str(test).strip() for test in tests if str(test).strip()]
    legacy = list(getattr(task, "stress_plus_challenge_hidden_tests", []) or [])
    if legacy:
        return [str(test).strip() for test in legacy if str(test).strip()]
    hidden = hidden_table1_tests(task)
    hidden.extend(str(test).strip() for test in list(getattr(task, "challenge_test_list", []) or []))
    deduped: list[str] = []
    seen: set[str] = set()
    for test in hidden:
        cleaned = str(test).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped


def hidden_eval_tests(task: BenchmarkTaskInstance, *, score_protocol: str = "table1_compatible") -> list[str]:
    protocol = resolve_mbpp_score_protocol(score_protocol)
    if protocol == "table1_compatible":
        return hidden_table1_tests(task)
    tests = list(getattr(task, "hidden_eval_tests", []) or [])
    if tests and not list(getattr(task, "hidden_stress_tests", []) or []) and not list(
        getattr(task, "stress_plus_challenge_hidden_tests", []) or []
    ):
        return [str(test).strip() for test in tests if str(test).strip()]
    return hidden_stress_tests(task)


def all_function_eval_tests(task: BenchmarkTaskInstance, *, score_protocol: str = "table1_compatible") -> list[str]:
    tests = visible_public_tests(task)
    tests.extend(hidden_eval_tests(task, score_protocol=score_protocol))
    deduped: list[str] = []
    seen: set[str] = set()
    for test in tests:
        cleaned = str(test).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped


def infer_task_entry_point(task: BenchmarkTaskInstance) -> str:
    """Infer the callable name expected by a benchmark task."""
    if task.entry_point:
        return task.entry_point
    if task.benchmark == "humaneval":
        matches = re.findall(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", task.prompt)
        if matches:
            return matches[-1]
    for test_case in visible_public_tests(task):
        match = re.search(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", test_case)
        if match:
            return match.group(1)
    return "solve"


def infer_task_arg_count(task: BenchmarkTaskInstance) -> int | None:
    """Infer the expected positional arity from benchmark tests."""
    entry_point = infer_task_entry_point(task)
    for test_case in visible_public_tests(task):
        match = re.search(rf"{re.escape(entry_point)}\((.*?)\)", test_case)
        if not match:
            continue
        raw_args = match.group(1).strip()
        if not raw_args:
            return 0
        return raw_args.count(",") + 1
    return None


def _extract_assert_cases(task: BenchmarkTaskInstance) -> list[tuple[list[Any], Any]]:
    entry_point = infer_task_entry_point(task)
    cases: list[tuple[list[Any], Any]] = []
    for test_case in visible_public_tests(task):
        try:
            tree = ast.parse(test_case)
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


def infer_task_pattern(task: BenchmarkTaskInstance) -> str:
    """Infer a coarse task-pattern label for routing and analysis."""
    entry_point = infer_task_entry_point(task)
    text = "\n".join([task.prompt, task.text, task.test, "\n".join(visible_public_tests(task))]).lower()
    replay_cases = _extract_assert_cases(task)
    arity = infer_task_arg_count(task) or max((len(args) for args, _expected in replay_cases), default=0)
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
    if task.benchmark == "humaneval":
        return "function_synthesis"
    if task.benchmark == "gsm8k":
        if any(token in text for token in ("percent", "%", "discount", "tax", "interest", "tip")):
            return "gsm8k_percentage"
        if any(token in text for token in ("combination", "combinations", "permutation", "arrangement", "ways", "choose")):
            return "gsm8k_counting_combinatorics"
        if any(token in text for token in ("ratio", "rate", "per ", "each hour", "each minute", "mph")):
            return "gsm8k_rate_ratio"
        if any(token in text for token in ("dollar", "dollars", "cent", "cents", "price", "cost", "money", "hour", "minute", "day", "week")):
            return "gsm8k_time_money"
        if any(token in text for token in ("remaining", "left", "twice", "half", "sum of", "difference", "equation")):
            return "gsm8k_algebraic_word_problem"
        return "gsm8k_arithmetic"
    if any(token in text for token in ("sort", "ascending", "descending", "ordered", "rearrange")) or entry_point.startswith("sort"):
        return "sort"
    if any(token in text for token in ("count", "number of", "digits", "digit", "frequency", "occurrence", "ways")) or entry_point.startswith("count"):
        return "count"
    if replay_cases and all(isinstance(expected, bool) for _args, expected in replay_cases):
        if arity <= 1:
            return "scalar_guard_predicate"
        return "multiarg_relation_or_classifier"
    if replay_cases and all(
        all(not isinstance(arg, (list, tuple, dict, set, str, bytes)) for arg in args)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        for args, expected in replay_cases
    ):
        if numeric_formula_hint:
            if arity <= 1:
                return "numeric_formula_single_arg"
            return "numeric_formula_multi_arg"
    if any(
        token in text
        for token in ("check", "find", "search", "whether", "valid", "sum of", "minimum", "maximum", "divisor", "angle", "area", "volume")
    ) or entry_point.startswith("find"):
        return "predicate_or_search"
    return "mbpp_general"
