"""Deterministic verifiers for code benchmarks."""

from __future__ import annotations

import json

from core.schemas import BenchmarkTaskInstance, CodeTaskInstance, VerifierFeedback
from core.utils import (
    all_function_eval_tests,
    hidden_eval_tests,
    resolve_mbpp_score_protocol,
    visible_public_tests,
)
from env.sandbox import Sandbox

_HARNESS_MARKER = "__IDEA_V7_ASSERT_HARNESS__"


def _task_dataset_split(task: CodeTaskInstance) -> str:
    return str(task.metadata.get("dataset_split", "")) if getattr(task, "metadata", None) else ""


def _build_humaneval_program(task: CodeTaskInstance, candidate: str) -> str:
    """Build an executable HumanEval verification program.

    HumanEval test blobs already define a `check(...)` function, so the runner
    should call that function instead of overriding it.
    """
    return "\n\n".join(
        part
        for part in [
            task.prompt.rstrip(),
            candidate.rstrip(),
            task.test.rstrip(),
            f"check({task.entry_point})",
        ]
        if part
    )


def verify_humaneval(task: CodeTaskInstance, candidate: str, sandbox: Sandbox) -> VerifierFeedback:
    """Run deterministic verification for one HumanEval candidate."""
    program = _build_humaneval_program(task, candidate)
    result = sandbox.run(program)
    return VerifierFeedback(
        benchmark="humaneval",
        task_id=task.task_id,
        passed=result["passed"],
        dataset_split=_task_dataset_split(task),
        stdout=result["stdout"],
        stderr=result["stderr"],
        returncode=result["returncode"],
        timeout=result["timeout"],
        executed_code=program,
        failed_tests=[] if result["passed"] else ["HumanEval deterministic test program failed."],
        summary="passed" if result["passed"] else "failed",
    )


def _resolved_function_tests(
    task: CodeTaskInstance,
    *,
    scope: str = "default",
    tests: list[str] | None = None,
    score_protocol: str = "table1_compatible",
) -> list[str]:
    if tests is not None:
        return [str(test).strip() for test in tests if str(test).strip()]
    protocol = resolve_mbpp_score_protocol(score_protocol)
    if scope == "visible_public":
        return visible_public_tests(task)
    if scope == "hidden_eval":
        return hidden_eval_tests(task, score_protocol=protocol)
    if scope in {"all", "full"}:
        return all_function_eval_tests(task, score_protocol=protocol)
    return list(getattr(task, "test_list", []) or [])


def _build_assert_harness(task: CodeTaskInstance, candidate: str, tests: list[str]) -> str:
    parts = [
        "import json",
        candidate.rstrip(),
        task.test_setup_code.rstrip(),
        f"__IDEA_TESTS = {json.dumps(list(tests), ensure_ascii=False)}",
        "__idea_results = []",
        "__idea_passed = 0",
        "for __idea_index, __idea_test in enumerate(__IDEA_TESTS):",
        "    try:",
        "        exec(__idea_test, globals(), globals())",
        "        __idea_results.append({'index': __idea_index, 'passed': True, 'test': __idea_test, 'error': ''})",
        "        __idea_passed += 1",
        "    except Exception as __idea_exc:  # noqa: BLE001",
        "        __idea_results.append({",
        "            'index': __idea_index,",
        "            'passed': False,",
        "            'test': __idea_test,",
        "            'error': f'{type(__idea_exc).__name__}: {__idea_exc}',",
        "        })",
        f"print('{_HARNESS_MARKER}' + json.dumps({{'passed_test_count': __idea_passed, 'total_test_count': len(__IDEA_TESTS), 'results': __idea_results}}, ensure_ascii=False))",
    ]
    return "\n\n".join(part for part in parts if part)


def _parse_harness_stdout(stdout: str) -> tuple[str, dict[str, object] | None]:
    cleaned_lines: list[str] = []
    payload: dict[str, object] | None = None
    for raw_line in str(stdout or "").splitlines():
        if raw_line.startswith(_HARNESS_MARKER):
            try:
                payload = json.loads(raw_line[len(_HARNESS_MARKER) :])
            except json.JSONDecodeError:
                payload = None
            continue
        cleaned_lines.append(raw_line)
    cleaned_stdout = "\n".join(cleaned_lines).strip()
    if cleaned_stdout:
        cleaned_stdout += "\n"
    return cleaned_stdout, payload


def verify_function_task(
    task: CodeTaskInstance,
    candidate: str,
    sandbox: Sandbox,
    *,
    scope: str = "default",
    tests: list[str] | None = None,
    score_protocol: str = "table1_compatible",
) -> VerifierFeedback:
    protocol = resolve_mbpp_score_protocol(score_protocol)
    selected_tests = _resolved_function_tests(task, scope=scope, tests=tests, score_protocol=protocol)
    if not selected_tests:
        return VerifierFeedback(
            benchmark=task.benchmark,
            task_id=task.task_id,
            passed=True,
            dataset_split=_task_dataset_split(task),
            stdout="",
            stderr="",
            returncode=0,
            timeout=False,
            executed_code=candidate,
            failed_tests=[],
            summary="passed",
            metrics={
                "scope": scope,
                "score_protocol": protocol,
                "passed_test_count": 0,
                "total_test_count": 0,
                "test_results": [],
            },
        )
    program = _build_assert_harness(task, candidate, selected_tests)
    result = sandbox.run(program)
    cleaned_stdout, harness_payload = _parse_harness_stdout(result["stdout"])
    passed_test_count = 0
    total_test_count = len(selected_tests)
    failed_tests: list[str] = []
    if harness_payload is not None:
        passed_test_count = int(harness_payload.get("passed_test_count", 0) or 0)
        total_test_count = int(harness_payload.get("total_test_count", len(selected_tests)) or len(selected_tests))
        failed_tests = [
            str(item.get("test", "")).strip()
            for item in list(harness_payload.get("results", []) or [])
            if not bool(item.get("passed", False)) and str(item.get("test", "")).strip()
        ]
    elif not result["passed"]:
        failed_tests = list(selected_tests)
    passed = bool(harness_payload is not None and passed_test_count == total_test_count and not result["timeout"])
    return VerifierFeedback(
        benchmark=task.benchmark,
        task_id=task.task_id,
        passed=passed,
        dataset_split=_task_dataset_split(task),
        stdout=cleaned_stdout,
        stderr=result["stderr"],
        returncode=result["returncode"],
        timeout=result["timeout"],
        executed_code=program,
        failed_tests=failed_tests,
        summary="passed" if passed else "failed",
        metrics={
            "scope": scope,
            "score_protocol": protocol,
            "passed_test_count": passed_test_count,
            "total_test_count": total_test_count,
            "test_results": list(harness_payload.get("results", []) if harness_payload is not None else []),
            "candidate_runtime_error": bool(harness_payload is None and not result["passed"]),
        },
    )


def verify_code_task(
    task: CodeTaskInstance,
    candidate: str,
    sandbox: Sandbox,
    *,
    scope: str = "default",
    tests: list[str] | None = None,
    score_protocol: str = "table1_compatible",
) -> VerifierFeedback:
    """Dispatch deterministic verification by benchmark name."""
    if task.benchmark == "mbpp":
        return verify_function_task(task, candidate, sandbox, scope=scope, tests=tests, score_protocol=score_protocol)
    if task.benchmark == "humaneval":
        return verify_humaneval(task, candidate, sandbox)
    raise ValueError(f"Unsupported benchmark: {task.benchmark}")


def verify_task(
    task: BenchmarkTaskInstance,
    candidate: str,
    sandbox: Sandbox | None = None,
    *,
    scope: str = "default",
    tests: list[str] | None = None,
    score_protocol: str = "table1_compatible",
) -> VerifierFeedback:
    if sandbox is None:
        raise ValueError("A sandbox is required for code-benchmark verification.")
    return verify_code_task(
        task=task,
        candidate=candidate,
        sandbox=sandbox,
        scope=scope,
        tests=tests,
        score_protocol=score_protocol,
    )
