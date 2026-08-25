"""Tool-level Test（三级验证体系第 1 层，设计文档 v2.0 §35.1）。

回答：该 executable artifact 本身是否满足接口与局部行为？
- action template 参数能否实例化
- Python function 单测 / replay 是否通过
- safety 是否通过
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.status import ArtifactKind, ValidationLevel
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
        provided = set((inputs or {}).keys())
        missing = [name for name in required if name not in provided]
        checks["parameters_bindable"] = not missing
        if missing:
            result.messages.append(f"缺少必填参数：{missing}")

        if tool.artifact_kind == ArtifactKind.ACTION_TEMPLATE:
            # 模板可实例化：所有 slot 都有值
            steps = tool.artifact.get("steps") or []
            slots = {s for step in steps for s in _extract_slots(str(step))}
            unbound = slots - provided
            checks["template_instantiable"] = not unbound
            if unbound:
                result.messages.append(f"模板 slot 未绑定：{sorted(unbound)}")
            if self.replay_fn is not None:
                replay = self.replay_fn(tool, inputs or {}, {})
                checks["template_executed"] = bool(replay.get("passed"))
                result.result["after"] = replay.get("after") or {}
                if not replay.get("passed"):
                    result.messages.append(f"模板执行失败：{replay.get('reason', 'unknown')}")
        else:
            # python_callable：单测/replay 执行
            test_cases: list[str] = []
            for case in tool.replay_cases():
                test_cases.extend(case.get("tests") or [])
            if test_cases:
                run = self.sandbox.run_tests(tool.artifact_body(), test_cases)
                checks["replay_tests"] = run["passed"]
                result.result["run"] = {
                    "passed_count": run["passed_count"],
                    "total": run["total"],
                    "failures": run["failures"],
                    "timeout": run["timeout"],
                }
                if not run["passed"]:
                    result.messages.append(
                        f"replay 测试未通过：{run['passed_count']}/{run['total']}")
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
