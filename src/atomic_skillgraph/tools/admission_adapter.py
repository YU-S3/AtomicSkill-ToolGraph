"""Tool Admission（设计文档 v2.0 §28）。

- Code Tool：syntax → 静态安全扫描 → trivial check → 单元/replay 测试 →
  扰动 replay（确定性复查）→ dedup/hash
- Interactive Tool（action_template）：接口完整 → 动作合法 → source trace replay →
  终端 Effect 验证（经 replay 回调）
只有「成功 Trace 提取 + Admission Passed」才能进入 status=candidate（§28.3）。
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.status import ArtifactKind, ToolLifecycle
from ..core.tool_ir import ToolAsset
from .sandbox import Sandbox

# 静态安全扫描：禁用 import 与危险调用（与 FlowEvo GateKeeper 对齐并略扩展）
_FORBIDDEN_IMPORTS = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "requests",
    "urllib", "http", "ctypes", "importlib", "pickle", "marshal", "multiprocessing",
    "threading", "pty", "fcntl", "resource", "signal", "tempfile", "glob",
}
_FORBIDDEN_CALLS = {
    "eval", "exec", "compile", "__import__", "open", "input", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "breakpoint",
    "memoryview", "bytearray",
}


@dataclass
class AdmissionResult:
    passed: bool = False
    checks: dict[str, bool] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    tool: ToolAsset | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": self.checks,
            "reasons": self.reasons,
        }


class AdmissionEngine:
    """Tool Skeleton → candidate/shadow 的 gate。

    replay_fn: 交互环境 replay 回调 `fn(tool, bindings, before) -> dict(passed, after)`。
    existing_hashes: 已有 artifact 结构哈希集合（dedup）。
    """

    def __init__(self, sandbox: Sandbox | None = None,
                 replay_fn: Callable[[ToolAsset, dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
                 existing_hashes: set[str] | None = None,
                 timeout_seconds: float = 10.0) -> None:
        self.sandbox = sandbox or Sandbox(timeout_seconds=timeout_seconds)
        self.replay_fn = replay_fn
        self.existing_hashes = existing_hashes or set()

    def admit(self, tool: ToolAsset) -> AdmissionResult:
        """执行 admission。通过 → status=candidate；失败 → status=shadow。"""
        tool.status = ToolLifecycle.ADMISSION_PENDING
        checks: dict[str, bool] = {}
        reasons: list[str] = []

        if tool.artifact_kind == ArtifactKind.ACTION_TEMPLATE:
            result = self._admit_action_template(tool, checks, reasons)
        else:
            result = self._admit_code_tool(tool, checks, reasons)

        if not reasons:
            tool.status = ToolLifecycle.CANDIDATE
            checks["admitted"] = True
        else:
            tool.status = ToolLifecycle.SHADOW
            checks["admitted"] = False
        return AdmissionResult(passed=not reasons, checks=checks, reasons=reasons, tool=tool)

    # ------------------------------------------------------------------
    def _admit_code_tool(self, tool: ToolAsset, checks: dict[str, bool],
                         reasons: list[str]) -> None:
        code = tool.artifact_body()

        # 1. syntax
        try:
            tree = ast.parse(code)
            checks["syntax"] = True
        except SyntaxError as exc:
            checks["syntax"] = False
            reasons.append(f"syntax_error: {exc}")
            return

        # 2. static safety scan
        unsafe = self._static_safety_scan(tree, code)
        checks["static_safety_scan"] = not unsafe
        if unsafe:
            reasons.append(f"unsafe_code: {unsafe[:3]}")

        # 3. trivial solution check
        trivial = self._trivial_check(tree, tool.entry_point())
        checks["trivial_solution_check"] = not trivial
        if trivial:
            reasons.append(f"trivial_solution: {trivial}")

        # 4. entry point 存在
        entry = tool.entry_point()
        if entry:
            names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
            checks["entry_point_exists"] = entry in names
            if entry not in names:
                reasons.append(f"entry_point_missing: {entry}")
        else:
            checks["entry_point_exists"] = False
            reasons.append("no entry point specified")

        # 5. unit / replay tests（沙箱执行）+ parameterized_replay（泛化 replay 门禁）
        plain_cases: list[str] = []
        for test in tool.tests:
            if test.get("kind") in ("parameterized_replay", "env_replay"):
                continue
            plain_cases.extend(test.get("tests") or [])
        param_cases = [t for t in tool.tests if t.get("kind") == "parameterized_replay"]

        if plain_cases:
            run = self.sandbox.run_tests(code, plain_cases)
            checks["unit_replay_tests"] = run["passed"]
            if not run["passed"]:
                reasons.append(f"unit_replay_failed: {len(run['failures'])}/{run['total']} failures")
                if run.get("timeout"):
                    reasons.append("unit_replay_timeout")
        elif param_cases:
            checks["unit_replay_tests"] = True  # 由 parameterized_replay 覆盖
        else:
            checks["unit_replay_tests"] = False
            reasons.append("no_replay_tests: 缺少可执行 replay case")

        param_ok = True
        if param_cases:
            for case in param_cases:
                bindings = case.get("bindings") or []
                if not bindings:
                    param_ok = False
                    reasons.append("parameterized_replay_empty")
                    continue
                for binding in bindings:
                    assignments = "\n".join(
                        f"{key} = {value!r}"
                        for key, value in (binding.get("value") or {}).items())
                    harness_code = code.rstrip() + "\n" + assignments + "\n"
                    run = self.sandbox.run_tests(harness_code, binding.get("tests") or [])
                    if not run["passed"]:
                        param_ok = False
                        reasons.append(f"parameterized_replay_failed: {binding.get('value')}")
            checks["parameterized_replay"] = param_ok

        # 6. perturbation replay：二次执行确定性复查（不同临时目录）
        if checks.get("unit_replay_tests") and plain_cases:
            rerun = self.sandbox.run_tests(code, plain_cases)
            checks["perturbation_replay"] = rerun["passed"]
            if not rerun["passed"]:
                reasons.append("perturbation_replay_nondeterministic")
        elif param_ok and param_cases:
            rerun_ok = True
            for case in param_cases:
                for binding in (case.get("bindings") or []):
                    assignments = "\n".join(
                        f"{key} = {value!r}"
                        for key, value in (binding.get("value") or {}).items())
                    harness_code = code.rstrip() + "\n" + assignments + "\n"
                    rerun = self.sandbox.run_tests(harness_code, binding.get("tests") or [])
                    rerun_ok = rerun_ok and rerun["passed"]
            checks["perturbation_replay"] = rerun_ok
            if not rerun_ok:
                reasons.append("perturbation_replay_nondeterministic")

        # 7. dedup
        struct_hash = tool.structural_hash()
        checks["dedup"] = struct_hash not in self.existing_hashes
        if struct_hash in self.existing_hashes:
            reasons.append("duplicate_artifact_hash")

    def _admit_action_template(self, tool: ToolAsset, checks: dict[str, bool],
                               reasons: list[str]) -> None:
        steps = list(tool.artifact.get("steps") or [])
        # 1. interface：参数完整
        params = tool.param_names()
        checks["interface_complete"] = bool(params)
        if not params:
            reasons.append("no_parameters")
        # 2. 模板合法性：slot 声明完整
        declared = set(params)
        used: set[str] = set()
        for step in steps:
            used.update(_extract_slots(str(step)))
        unknown = used - declared
        checks["slots_declared"] = not unknown
        if unknown:
            reasons.append(f"unknown_slots: {sorted(unknown)}")
        unused = declared - used
        checks["declared_slots_used"] = not unused
        if unused:
            reasons.append(f"unused_declared_parameters: {sorted(unused)}")
        # Action Tool 必须是可复用模板。诸如 ``mug 1``/``mug_1``、
        # ``cabinet 3`` 等具体 ALFWorld 实例不能绕过参数化后进入 Admission。
        concrete_literals = sorted({literal for step in steps
                                    for literal in _concrete_action_literals(str(step))})
        checks["instance_free_template"] = not concrete_literals
        if concrete_literals:
            reasons.append(f"concrete_instance_literals: {concrete_literals}")
        # 3. 动作合法：每步以已知动词开头且长度受限
        legal = all(
            len(str(step)) <= 300 and str(step).strip() for step in steps
        )
        checks["actions_legal"] = legal
        if not legal:
            reasons.append("illegal_action_lines")
        # 无效模板不应启动昂贵的环境 replay，更不能在 replay 时才偶然暴露。
        if unknown or unused or concrete_literals or not legal:
            checks["env_replay"] = False
            return
        # 4. source trace replay（环境回调）
        if self.replay_fn is None:
            checks["env_replay"] = False
            reasons.append("no_env_replay_callback")
        else:
            replay_case = next((t for t in tool.replay_cases() if t.get("kind") == "replay"), {})
            bindings = dict(replay_case.get("bindings") or {})
            before = dict(replay_case.get("before") or {})
            try:
                replay_result = self.replay_fn(tool, bindings, before)
                checks["env_replay"] = bool(replay_result.get("passed"))
                if not replay_result.get("passed"):
                    reasons.append("env_replay_failed")
            except Exception as exc:  # noqa: BLE001
                checks["env_replay"] = False
                reasons.append(f"env_replay_error: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    @staticmethod
    def _static_safety_scan(tree: ast.Module, code: str) -> list[str]:
        findings: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in _FORBIDDEN_IMPORTS:
                        findings.append(f"import:{root}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    if root in _FORBIDDEN_IMPORTS:
                        findings.append(f"import:{root}")
            elif isinstance(node, ast.Call):
                name = ""
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                if name in _FORBIDDEN_CALLS:
                    findings.append(f"call:{name}")
            elif isinstance(node, ast.Attribute):
                if node.attr in ("__globals__", "__builtins__", "__class__", "__subclasses__", "__mro__"):
                    findings.append(f"attr:{node.attr}")
        return findings

    @staticmethod
    def _trivial_check(tree: ast.Module, entry_point: str) -> str:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == entry_point:
                statements = [s for s in node.body if not isinstance(s, ast.Pass)]
                if not statements:
                    return "empty_body"
                if len(statements) == 1 and isinstance(statements[0], ast.Return):
                    ret = statements[0].value
                    if ret is None or isinstance(ret, ast.Constant):
                        return "constant_return"
        return ""


def _extract_slots(step: str) -> set[str]:
    return set(re.findall(r"\{([a-z_][a-z0-9_]*)\}", str(step)))


def _concrete_action_literals(step: str) -> set[str]:
    """返回模板正文中未被占位符保护的 ``entity 1``/``entity_1``。"""
    without_slots = re.sub(r"\{[a-z_][a-z0-9_]*\}", "", str(step),
                           flags=re.IGNORECASE)
    return {match.group(0).strip().lower()
            for match in re.finditer(
                r"(?<![a-z0-9])(?:[a-z][a-z0-9]*)(?:_|\s+)\d+(?![a-z0-9])",
                without_slots, flags=re.IGNORECASE)}
