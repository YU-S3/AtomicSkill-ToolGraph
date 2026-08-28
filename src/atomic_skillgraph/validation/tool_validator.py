"""Tool-level Test（三级验证体系第 1 层，设计文档 v2.0 §35.1）。

回答：该 executable artifact 本身是否满足接口与局部行为？
- action template 参数能否实例化
- Python function 单测 / replay 是否通过
- safety 是否通过
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.binding_ir import is_concrete_binding
from ..core.status import ArtifactKind, ToolLifecycle, ValidationLevel
from ..core.tool_ir import ToolAsset
from ..tools import sandbox as _sandbox_mod


@dataclass
class ValidationResult:
    level: ValidationLevel = ValidationLevel.TOOL
    ref: str = ""
    passed: bool = False
    checks: dict[str, bool] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "ref": self.ref,
            "passed": self.passed,
            "checks": self.checks,
            "messages": self.messages,
            "result": self.result,
        }


class ToolValidator:
    """对 Tool 执行接口/局部行为验证（不含 admission 的安全扫描）。"""

    def __init__(self, sandbox=None,
                 replay_fn: Callable[[ToolAsset, dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None) -> None:
        self.sandbox = sandbox or _sandbox_mod.Sandbox()
        self.replay_fn = replay_fn

    def validate_tool(self, tool: ToolAsset, inputs: dict[str, Any] | None = None) -> ValidationResult:
        result = ValidationResult(ref=str(tool.ref), level=ValidationLevel.TOOL)
        checks: dict[str, bool] = {}

        # 接口：所有必填参数可实例化
        required = [p.get("name") for p in tool.parameters() if p.get("required")]
        values = dict(inputs or {})
        provided = {key for key, value in values.items()
                    if is_concrete_binding(value)}
        missing = [name for name in required if name not in provided]
        checks["parameters_bindable"] = not missing
        if missing:
            result.messages.append(f"缺少必填参数：{missing}")

        certificate_present = bool(
            (tool.safety or {}).get("admission_certificate"))
        certificate_valid = tool.admission_certificate_valid()
        certificate_required = tool.status in {
            ToolLifecycle.CANDIDATE,
            ToolLifecycle.ACTIVE,
            ToolLifecycle.PREFERRED,
        }
        if certificate_present or certificate_required:
            checks["admission_certificate"] = certificate_valid
            if not certificate_valid:
                result.messages.append(
                    "可用状态 Tool 缺少有效 Admission certificate"
                    if not certificate_present else
                    "Admission certificate 与当前 Tool 内容不一致")

        if tool.artifact_kind == ArtifactKind.ACTION_TEMPLATE:
            # 模板可实例化：所有 slot 都有值
            steps = tool.artifact.get("steps") or []
            slots = {s for step in steps for s in _extract_slots(str(step))}
            unbound = slots - provided
            checks["template_instantiable"] = not unbound
            if unbound:
                result.messages.append(f"模板 slot 未绑定：{sorted(unbound)}")
            live_replay_passed = False
            if self.replay_fn is not None:
                replay = self.replay_fn(tool, inputs or {}, {})
                live_replay_passed = bool(replay.get("passed"))
                checks["template_executed"] = live_replay_passed
                result.result["after"] = replay.get("after") or {}
                if not replay.get("passed"):
                    result.messages.append(f"模板执行失败：{replay.get('reason', 'unknown')}")
            # Merely possessing a replay payload is not proof that this
            # validator executed it.  Action templates therefore require
            # either a successful live replay or the immutable Admission
            # certificate.  This makes stripped/tampered frozen banks fail
            # closed just like Python callables.
            checks["evidence_or_certificate"] = bool(
                live_replay_passed or certificate_valid)
            if not checks["evidence_or_certificate"]:
                if tool.has_unresolved_test_evidence():
                    result.messages.append(
                        "本地 replay evidence 不可用且 Admission certificate 无效")
                else:
                    result.messages.append(
                        "未执行环境 replay 且无有效 Admission certificate")
        else:
            # python_callable：单测/replay 执行
            test_cases: list[str] = []
            available_cases = tool.all_test_cases()
            for case in available_cases:
                if case.get("kind") == "parameterized_replay":
                    continue
                test_cases.extend(case.get("tests") or [])
            parameterized_cases = [
                case for case in available_cases
                if case.get("kind") == "parameterized_replay"]
            executed_any = bool(test_cases or parameterized_cases)
            replay_passed = True
            if test_cases:
                run = self.sandbox.run_tests(tool.artifact_body(), test_cases)
                replay_passed = replay_passed and bool(run["passed"])
                result.result["run"] = {
                    "passed_count": run["passed_count"],
                    "total": run["total"],
                    "failures": run["failures"],
                    "timeout": run["timeout"],
                }
                if not run["passed"]:
                    result.messages.append(
                        f"replay 测试未通过：{run['passed_count']}/{run['total']}")
            parameterized_result = _run_parameterized_replay(
                self.sandbox, tool.artifact_body(), parameterized_cases)
            if parameterized_cases:
                replay_passed = replay_passed and parameterized_result["passed"]
                result.result["parameterized_replay"] = parameterized_result
                if not parameterized_result["passed"]:
                    result.messages.append(
                        "parameterized replay 未通过："
                        f"{parameterized_result['passed_count']}/"
                        f"{parameterized_result['total']}")
            if executed_any:
                checks["replay_tests"] = replay_passed
            elif tool.has_unresolved_test_evidence() and certificate_valid:
                # Frozen/exported banks intentionally omit private replay
                # payloads.  The certificate binds the passed Admission to the
                # exact artifact, signature, interface and evidence hashes.
                checks["replay_tests"] = True
                checks["evidence_or_certificate"] = True
                result.messages.append(
                    "本地 replay evidence 不可用；使用有效 Admission certificate")
            else:
                checks["replay_tests"] = False
                result.messages.append("无可用 replay 测试")

        # safety（admission 已通过的记录）
        safety = tool.safety or {}
        checks["safety"] = bool(safety.get("direct_execution_allowed", False))

        result.checks = checks
        result.passed = all(checks.values())
        return result


def _extract_slots(step: str) -> set[str]:
    import re
    return set(re.findall(r"\{([a-z_][a-z0-9_]*)\}", str(step)))


def _run_parameterized_replay(sandbox, code: str,
                              cases: list[dict[str, Any]]) -> dict[str, Any]:
    passed_count = 0
    total = 0
    failures: list[dict[str, Any]] = []
    for case in cases:
        for binding in (case.get("bindings") or []):
            total += 1
            assignments = "\n".join(
                f"{key} = {value!r}"
                for key, value in (binding.get("value") or {}).items())
            harness = code.rstrip() + "\n" + assignments + "\n"
            run = sandbox.run_tests(harness, binding.get("tests") or [])
            if run["passed"]:
                passed_count += 1
            else:
                failures.append({
                    "value": dict(binding.get("value") or {}),
                    "failures": list(run.get("failures") or []),
                    "timeout": bool(run.get("timeout")),
                })
    return {
        "passed": bool(cases) and total > 0 and passed_count == total,
        "passed_count": passed_count,
        "total": total,
        "failures": failures,
    }
