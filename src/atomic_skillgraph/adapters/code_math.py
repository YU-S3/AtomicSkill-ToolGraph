"""Code/Math Benchmark Adapter（HumanEval / GSM8K）。

复用 vendored FlowEvo 的 loader 与 verifier（§52 原则：相同 Harness/Evaluator 设置）：
- humaneval: env.humaneval_loader.load_humaneval_tasks + eval.verifier
- gsm8k:     env.gsm8k_loader.load_gsm8k_tasks + code_math 风格答案提取比对
"""

from __future__ import annotations

import re
from typing import Any

from ..core.llm import LLM
from .benchmark import (
    BenchmarkAdapter,
    CodeAttempt,
    CodeRunResult,
    Task,
    VerifyResult,
)

_FLOWEVO_SRC = None  # 由 ensure_flowevo_path() 设置


def ensure_flowevo_path() -> None:
    """确保 vendored FlowEvo 的 src 目录在 sys.path（其内部使用绝对导入）。"""
    import sys
    from pathlib import Path
    global _FLOWEVO_SRC
    if _FLOWEVO_SRC is not None:
        return
    here = Path(__file__).resolve()
    flowevo_src = here.parents[2] / "flowevo"
    _FLOWEVO_SRC = str(flowevo_src)
    if _FLOWEVO_SRC not in sys.path:
        sys.path.insert(0, _FLOWEVO_SRC)


class CodeMathAdapter:
    """HumanEval / GSM8K 的规范任务适配。"""

    name = "code_math"

    def __init__(self, benchmark: str, *, limit: int = 0,
                 timeout_seconds: float = 10.0) -> None:
        if benchmark not in ("humaneval", "gsm8k"):
            raise ValueError(f"CodeMathAdapter 不支持 benchmark={benchmark}")
        self.benchmark = benchmark
        self.limit = limit
        self.timeout_seconds = timeout_seconds
        self._flowevo_tasks: dict[str, Any] = {}

    # ------------------------------------------------------------------
    def load_tasks(self, limit: int = 0, task_type: str | None = None) -> list[Task]:
        ensure_flowevo_path()
        from env.gsm8k_loader import load_gsm8k_tasks
        from env.humaneval_loader import load_humaneval_tasks

        effective_limit = limit or self.limit
        if self.benchmark == "humaneval":
            raw = load_humaneval_tasks(profile="full", limit=effective_limit or None,
                                       dataset_split="")
            return [self._to_task(t) for t in raw]
        raw = load_gsm8k_tasks(limit=effective_limit or None, dataset_split="")
        tasks = [self._to_task(t) for t in raw]
        if task_type:
            tasks = [t for t in tasks if t.task_type == task_type]
        return tasks

    def parse_task_type(self, task: Task) -> str:
        return task.task_type

    def _to_task(self, flowevo_task) -> Task:
        from core.utils import infer_task_pattern
        task_type = infer_task_pattern(flowevo_task)
        context: dict[str, Any] = {
            "kind": "code",
            "entry": flowevo_task.entry_point or "solve",
            "flowevo_task": flowevo_task,
        }
        if self.benchmark == "gsm8k":
            context["kind"] = "math"
            context["answer"] = flowevo_task.metadata.get("gold_answer", "")
            context["tests"] = list(flowevo_task.test_list or [])
        else:
            from core.utils import visible_public_tests
            context["tests"] = [str(t) for t in visible_public_tests(flowevo_task)]
        goal = (flowevo_task.text or flowevo_task.prompt or flowevo_task.task_id)
        self._flowevo_tasks[flowevo_task.task_id] = flowevo_task
        return Task(
            task_id=flowevo_task.task_id,
            benchmark=self.benchmark,
            task_type=task_type,
            goal=str(goal)[:500],
            context=context,
            state={"facts": [], "text": str(goal)[:500]},
            target_effects=[{"predicate": "callable.returns_expected",
                             "args": {"entry_point": context["entry"]}}],
            metadata={"flowevo_task_id": flowevo_task.task_id},
        )

    # ------------------------------------------------------------------
    def verify_task(self, task: Task, candidate: str) -> VerifyResult:
        if self.benchmark == "gsm8k":
            return self._verify_gsm8k(task, candidate)
        return self._verify_humaneval(task, candidate)

    def _verify_humaneval(self, task: Task, candidate: str) -> VerifyResult:
        ensure_flowevo_path()
        from eval.verifier import verify_task as flowevo_verify
        from env.sandbox import Sandbox as FlowEvoSandbox
        flowevo_task = task.context.get("flowevo_task")
        if flowevo_task is None:
            return VerifyResult(passed=False, failure_type="missing_flowevo_task")
        sandbox = FlowEvoSandbox(timeout_seconds=self.timeout_seconds)
        try:
            feedback = flowevo_verify(flowevo_task, candidate, sandbox=sandbox)
        except Exception as exc:  # noqa: BLE001
            return VerifyResult(passed=False, failure_type="verifier_error",
                                stderr=str(exc)[:500])
        return VerifyResult(
            passed=bool(feedback.passed),
            tests=list(task.context.get("tests") or []),
            executed_code=str(feedback.executed_code or candidate),
            stdout=str(feedback.stdout or ""),
            stderr=str(feedback.stderr or ""),
            timeout=bool(feedback.timeout),
            feedback={"flowevo_failed_tests": list(feedback.failed_tests or [])},
            failure_type="" if feedback.passed else _failure_of(feedback),
        )

    def _verify_gsm8k(self, task: Task, candidate: str) -> VerifyResult:
        ensure_flowevo_path()
        from env.sandbox import Sandbox as FlowEvoSandbox
        expected = str(task.context.get("answer", ""))
        harness = candidate.rstrip() + "\n\nprint(solve())\n"
        run = FlowEvoSandbox(timeout_seconds=self.timeout_seconds).run(harness)
        actual = extract_math_answer(run.get("stdout", ""))
        passed = run["passed"] and _normalize_answer(actual) == _normalize_answer(expected) \
            and actual != ""
        return VerifyResult(
            passed=passed,
            tests=[f"assert _gsm8k_equal(solve(), {expected!r})"],
            executed_code=candidate,
            stdout=run.get("stdout", ""),
            stderr=run.get("stderr", ""),
            timeout=bool(run.get("timeout")),
            feedback={"extracted_answer": actual, "gold_answer": expected},
            failure_type="" if passed else ("timeout" if run.get("timeout") else "answer_mismatch"),
        )

    # ------------------------------------------------------------------
    def execute_tool(self, task: Task, tool, parameters: dict[str, Any],
                     state: dict[str, Any]) -> dict[str, Any]:
        candidate = tool.artifact_body()
        verify = self.verify_task(task, candidate)
        entry = str(task.context.get("entry", "solve"))
        return {
            "passed": verify.passed,
            "after": {"facts": [f"callable_returns_expected({entry})"] if verify.passed else [],
                      "text": verify.stdout or ""},
            "observation": verify.stdout or verify.stderr,
            "feedback": verify.to_dict(),
        }

    def replay_tool(self, tool, bindings: dict[str, Any],
                    before: dict[str, Any]) -> dict[str, Any]:
        ensure_flowevo_path()
        from env.sandbox import Sandbox as FlowEvoSandbox
        test_cases: list[str] = []
        for case in tool.replay_cases():
            test_cases.extend(case.get("tests") or [])
        if not test_cases:
            return {"passed": False, "after": {}, "reason": "no_replay_tests"}
        run = FlowEvoSandbox(timeout_seconds=self.timeout_seconds).run_tests(
            tool.artifact_body(), test_cases)
        return {"passed": run["passed"], "after": {},
                "reason": f"{run['passed_count']}/{run['total']}"}

    # ------------------------------------------------------------------
    def generate_code(self, task: Task, llm: LLM, *,
                      seed_context: str = "", max_repairs: int = 2) -> CodeRunResult:
        result = CodeRunResult()
        instructions = (
            "You are an expert Python programmer. Output ONLY raw Python code "
            "(no markdown fences, no explanation) that satisfies the task."
        )
        prompt = self._build_prompt(task, seed_context)
        for index in range(max_repairs + 1):
            try:
                response = llm.generate(instructions=instructions, input_text=prompt)
            except Exception as exc:  # noqa: BLE001
                result.failure_type = f"generation_error:{type(exc).__name__}"
                return result
            code = _extract_code(response.text)
            verify = self.verify_task(task, code)
            attempt = CodeAttempt(index=index,
                                  stage="draft" if index == 0 else "repair",
                                  code=code, verify=verify,
                                  usage=llm.usage)
            result.attempts.append(attempt)
            if verify.passed:
                result.success = True
                result.candidate_code = code
                result.feedback = verify
                result.first_attempt_success = index == 0
                result.retry_count = index
                return result
            feedback_text = verify.stderr or verify.failure_type or "tests failed"
            prompt = (f"{prompt}\n\nPrevious attempt failed ({feedback_text[:400]}). "
                      f"Output ONLY the corrected raw Python code.")
        result.failure_type = "max_repairs_exceeded"
        if result.attempts:
            result.candidate_code = result.attempts[-1].code
            result.feedback = result.attempts[-1].verify
            result.retry_count = len(result.attempts) - 1
        return result

    def _build_prompt(self, task: Task, seed_context: str) -> str:
        if self.benchmark == "gsm8k":
            prompt = (f"Question: {task.goal}\n"
                      f"Write raw Python only. Implement `def solve():`. "
                      f"Return only the final numeric answer.")
        else:
            flowevo_task = task.context.get("flowevo_task")
            if flowevo_task is not None and getattr(flowevo_task, "prompt", ""):
                prompt = f"{flowevo_task.prompt}"
            else:
                prompt = f"Task: {task.goal}"
        if seed_context:
            prompt = f"{seed_context}\n\nNow solve this task:\n{prompt}"
        return prompt


# ---------------------------------------------------------------------------
# 答案提取（与 vendored FlowEvo code_math/runner.py 的行为对齐）
# ---------------------------------------------------------------------------

def extract_math_answer(text: str) -> str:
    text = str(text or "")
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed[-1].strip()
    answer_match = re.search(r"(?i)the answer is[:\s]*([^\n.]+)", text)
    if answer_match:
        return answer_match.group(1).strip()
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return numbers[-1].strip() if numbers else ""


def _normalize_answer(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r",", "", text)
    if text.endswith(".0"):
        text = text[:-2]
    if "/" in text:
        try:
            numerator, denominator = text.split("/", 1)
            return f"{float(numerator) / float(denominator):.6f}".rstrip("0").rstrip(".")
        except (ValueError, ZeroDivisionError):
            return text
    return text


def _extract_code(text: str) -> str:
    text = str(text).strip()
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text


def _failure_of(feedback) -> str:
    if getattr(feedback, "timeout", False):
        return "timeout"
    if getattr(feedback, "returncode", 0) != 0 and getattr(feedback, "passed", False) is False:
        return "runtime_error"
    return "test_failure"
