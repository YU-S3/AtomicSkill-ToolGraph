"""最小 Python 沙箱（subprocess + 临时目录）。

设计参考 FlowEvo `env/sandbox.py`（Apache-2.0，vendored 于 src/flowevo/）的行为：
临时目录 + 子进程 + 墙钟超时。此处独立实现以保持 atomic_skillgraph 层
对 FlowEvo import 的零依赖（stage-1 smoke 不需要 FlowEvo）。
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


class Sandbox:
    """在独立临时文件中执行 Python 片段。"""

    def __init__(self, python_executable: str | None = None, timeout_seconds: float = 10.0,
                 tmp_root: str | None = None) -> None:
        # Reuse the interpreter that launched the experiment.  A bare
        # ``python`` is not guaranteed to exist in WSL/venv/CI environments.
        self.python_executable = python_executable or sys.executable
        self.timeout_seconds = timeout_seconds
        # tmp_root：自定义临时目录基座（受控环境/工作区）；None = 系统临时目录
        self.tmp_root = tmp_root
        if tmp_root:
            Path(tmp_root).mkdir(parents=True, exist_ok=True)

    def run(self, code: str) -> dict[str, Any]:
        """执行完整程序，返回 {stdout, stderr, returncode, passed, timeout}。"""
        import shutil
        import uuid
        tmp_dir = Path(tempfile.gettempdir() if self.tmp_root is None else self.tmp_root) \
            / f"asg_sandbox_{uuid.uuid4().hex[:10]}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            script_path = tmp_dir / "main.py"
            script_path.write_text(code, encoding="utf-8")
            try:
                result = subprocess.run(
                    [self.python_executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
                return {
                    "stdout": result.stdout or "",
                    "stderr": result.stderr or "",
                    "returncode": result.returncode,
                    "passed": result.returncode == 0,
                    "timeout": False,
                }
            except subprocess.TimeoutExpired as exc:
                return {
                    "stdout": exc.stdout or "",
                    "stderr": exc.stderr or "",
                    "returncode": -1,
                    "passed": False,
                    "timeout": True,
                }
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def run_tests(self, code: str, tests: list[str]) -> dict[str, Any]:
        """执行 `code` 并在其后逐条 `exec` 每个 test 断言。

        返回 {passed, total, passed_count, failures, stdout, stderr, timeout}。
        """
        if not tests:
            return {"passed": True, "total": 0, "passed_count": 0,
                    "failures": [], "stdout": "", "stderr": "", "timeout": False}
        harness_lines = [code.rstrip(), ""]
        for index, test in enumerate(tests):
            harness_lines.append(
                f"try:\n"
                f"    exec({test!r})\n"
                f"    print('ASG_TEST_RESULT:{index}:PASS')\n"
                f"except Exception as _e:\n"
                f"    print('ASG_TEST_RESULT:{index}:FAIL:' + repr(_e))\n"
            )
        result = self.run("\n".join(harness_lines))
        failures: list[str] = []
        passed_count = 0
        for line in (result.get("stdout") or "").splitlines():
            if line.startswith("ASG_TEST_RESULT:"):
                parts = line.split(":", 3)
                if len(parts) >= 3 and parts[2] == "PASS":
                    passed_count += 1
                else:
                    failures.append(line)
        total = len(tests)
        return {
            "passed": passed_count == total and not result.get("timeout"),
            "total": total,
            "passed_count": passed_count,
            "failures": failures,
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "timeout": bool(result.get("timeout")),
        }


def build_call_harness(code: str, entry_point: str, call: str) -> str:
    """组装「code + 调用入口 + 打印结果」的最小 harness。"""
    return (
        code.rstrip()
        + "\n\n"
        + call.strip()
        + "\n"
    )
