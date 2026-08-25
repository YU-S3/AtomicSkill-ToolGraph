"""Strict one-command small experiment suite for all three real benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from experiments.common import PROJECT_ROOT


CONDITIONS = ["baseline_dynamic", "flowevo", "atomic_graph_only",
              "tool_repo_only", "atomic_skillgraph_full"]
OURS = ["atomic_graph_only", "tool_repo_only", "atomic_skillgraph_full"]


class _BottomProgress:
    """不吞日志的双层底部动态进度区。

    普通日志打印在上方；子进程的逐任务进度行只用于更新底部状态，不再重复
    输出。TTY 使用 ANSI 原位刷新，总进度会包含当前阶段的内部进度。
    """

    _COUNT_RE = re.compile(r"(?:\[\s*)?(\d+)\s*/\s*(\d+)(?:\s*\])?")
    _MONITOR_RE = re.compile(
        r"(?P<pct>\d+(?:\.\d+)?)%\s*\|\s*条件\s*(?P<cond>\d+)/(?P<conds>\d+)\s*"
        r"(?P<name>.*?)(?:\s+任务\s*(?P<task>\d+)/(?P<tasks>\d+))?\s*\|")
    _EPISODE_RE = re.compile(
        r"\[\s*(?P<task>\d+)\s*/\s*(?P<tasks>\d+)\s*\]\s*"
        r"(?P<result>PASS|FAIL|ERROR)\b(?P<rest>.*)", re.IGNORECASE)
    _ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

    def __init__(self, total_stages: int) -> None:
        self.total = max(1, int(total_stages))
        self.completed = 0
        self.stage_fraction = 0.0
        self.label = "准备"
        self.detail = ""
        self.condition = ""
        self.task_done = 0
        self.task_total = 0
        self.last_result = ""
        self.started = time.monotonic()
        self.enabled = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self._rendered = False

    def start_stage(self, label: str) -> None:
        self.label = label
        self.detail = "启动中"
        self.condition = ""
        self.task_done = 0
        self.task_total = 0
        self.last_result = ""
        self.stage_fraction = 0.0
        self._draw()

    def observe(self, line: str) -> bool:
        """吸收子进程进度行；返回 True 表示该行不应再逐行打印。"""
        clean = self._ANSI_RE.sub("", str(line)).replace("\r", "").strip()
        monitor = self._MONITOR_RE.search(clean)
        if monitor:
            self.stage_fraction = max(
                self.stage_fraction, min(1.0, float(monitor.group("pct")) / 100.0))
            self.condition = monitor.group("name").strip()
            if monitor.group("task") and monitor.group("tasks"):
                self.task_done = int(monitor.group("task"))
                self.task_total = int(monitor.group("tasks"))
            self.detail = "运行中"
            self._draw()
            return True
        episode = self._EPISODE_RE.search(clean)
        if episode:
            self.task_done = int(episode.group("task"))
            self.task_total = max(1, int(episode.group("tasks")))
            self.last_result = episode.group("result").upper()
            # FlowEvo 子进程有时只有 episode 行，没有统一 monitor 行。
            if not self.condition:
                self.stage_fraction = max(self.stage_fraction,
                                          self.task_done / self.task_total)
            rest = episode.group("rest").strip()
            mode = re.search(r"mode=([^\s]+)", rest)
            self.detail = ((self.last_result + " · " + mode.group(1))
                           if mode else self.last_result)
            self._draw()
            # 正常 PASS/FAIL 只动态更新；代码异常仍保留为上方审计日志。
            return self.last_result != "ERROR"
        return False

    def finish_stage(self) -> None:
        self.completed = min(self.total, self.completed + 1)
        self.stage_fraction = 0.0
        self.detail = "阶段完成"
        self._draw()

    def log(self, line: str = "") -> None:
        text = str(line).rstrip("\r\n")
        if self.enabled:
            self._clear()
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
        self._draw()

    def close(self, *, passed: bool) -> None:
        if passed:
            self.completed = self.total
            self.stage_fraction = 0.0
            self.label = "全部完成"
            self.detail = ""
        else:
            self.detail = "异常中止"
        self._draw()
        if self.enabled:
            sys.stdout.write("\n")
            sys.stdout.flush()

    @property
    def fraction(self) -> float:
        return min(1.0, (self.completed + self.stage_fraction) / self.total)

    @staticmethod
    def _bar(ratio: float, width: int) -> str:
        filled = min(width, int(width * min(1.0, max(0.0, ratio))))
        return "█" * filled + "░" * (width - filled)

    def _status_lines(self) -> tuple[str, str]:
        ratio = self.fraction
        columns = shutil.get_terminal_size(fallback=(120, 30)).columns
        bar_width = max(12, min(34, columns - 78))
        elapsed = max(0, int(time.monotonic() - self.started))
        clock = f"{elapsed // 60:02d}:{elapsed % 60:02d}"
        eta_seconds = int(elapsed / ratio - elapsed) if ratio > 0 else 0
        eta = f"{eta_seconds // 60:02d}:{eta_seconds % 60:02d}"
        overall = (f"  总体 {self._bar(ratio, bar_width)} {ratio * 100:5.1f}%  "
                   f"阶段 {min(self.completed + 1, self.total)}/{self.total}  "
                   f"已用 {clock}  剩余 {eta}")
        current_ratio = self.stage_fraction
        task = (f"任务 {self.task_done}/{self.task_total}"
                if self.task_total else "等待任务信息")
        condition = f" · {self.condition}" if self.condition else ""
        detail = f" · {self.detail}" if self.detail else ""
        current = (f"  当前 {self._bar(current_ratio, bar_width)} "
                   f"{current_ratio * 100:5.1f}%  {self.label}{condition}  "
                   f"{task}{detail}")
        return overall[:max(1, columns - 1)], current[:max(1, columns - 1)]

    def _clear(self) -> None:
        if not self.enabled or not self._rendered:
            return
        # 光标位于第二行末尾：先清第二行，再上移清第一行。
        sys.stdout.write("\r\033[2K\033[1A\r\033[2K")
        self._rendered = False

    def _draw(self) -> None:
        if not self.enabled:
            return
        self._clear()
        overall, current = self._status_lines()
        sys.stdout.write("\r\033[2K\033[1;36m" + overall +
                         "\033[0m\n\r\033[2K\033[1;34m" + current + "\033[0m")
        self._rendered = True
        sys.stdout.flush()


def _run(command: list[str], progress: _BottomProgress, label: str) -> None:
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    progress.start_stage(label)
    progress.log("\n[all-small] " + " ".join(command))
    process = subprocess.Popen(
        command, cwd=PROJECT_ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            if not progress.observe(line):
                progress.log(line)
        return_code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise
    if return_code != 0:
        raise SystemExit(return_code)
    progress.finish_stage()


def _created(root: Path, pattern: str, before: set[Path]) -> Path:
    after = {path.resolve() for path in root.glob(pattern) if path.is_dir()}
    candidates = sorted(after - before, key=lambda path: path.stat().st_mtime,
                        reverse=True)
    if not candidates:
        raise RuntimeError(f"命令成功退出但没有产生新目录：{root / pattern}")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="依次运行 ALFWorld/HumanEval/GSM8K 五条件小实验及 ours 冻结 replay")
    parser.add_argument("--config-path", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--alfworld-data", default=None)
    parser.add_argument("--task-type", default="pick_heat_then_place_in_recep")
    parser.add_argument("--alfworld-limit", type=int, default=10)
    parser.add_argument("--humaneval-limit", type=int, default=10)
    parser.add_argument("--gsm8k-limit", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--online-output", default=str(
        PROJECT_ROOT / "runs" / "all_small_post_planner_fix"))
    parser.add_argument("--eval-output", default=str(
        PROJECT_ROOT / "runs" / "all_small_frozen_post_planner_fix"))
    parser.add_argument("--skip-frozen", action="store_true")
    args = parser.parse_args()

    online_root = Path(args.online_output)
    eval_root = Path(args.eval_output)
    online_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)
    specs = [
        ("alfworld", args.alfworld_limit),
        ("humaneval", args.humaneval_limit),
        ("gsm8k", args.gsm8k_limit),
    ]
    progress = _BottomProgress(total_stages=len(specs) * (1 if args.skip_frozen else 2))
    manifest: dict[str, object] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "conditions": CONDITIONS,
        "benchmarks": {},
    }
    try:
      for benchmark, limit in specs:
        before = {path.resolve() for path in online_root.glob(f"{benchmark}_*")
                  if path.is_dir()}
        command = [
            sys.executable, "-m", "experiments.run_small",
            "--benchmark", benchmark, "--limit", str(limit),
            "--conditions", *CONDITIONS,
            "--config-path", args.config_path,
            "--output-dir", str(online_root),
        ]
        if benchmark == "alfworld":
            command += ["--task-type", args.task_type,
                        "--max-steps", str(args.max_steps)]
            if args.alfworld_data:
                command += ["--alfworld-data", args.alfworld_data]
        _run(command, progress, f"{benchmark} 在线进化")
        run_dir = _created(online_root, f"{benchmark}_*", before)
        entry: dict[str, object] = {"online_run_dir": str(run_dir),
                                   "limit": limit}
        results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
        missing = [condition for condition in CONDITIONS if condition not in results]
        if missing:
            raise RuntimeError(f"{benchmark} 在线结果缺少条件：{missing}")

        if not args.skip_frozen:
            eval_before = {path.resolve() for path in eval_root.glob(
                f"{run_dir.name}_train_*") if path.is_dir()}
            eval_command = [
                sys.executable, "-m", "experiments.run_evolve_eval",
                "--run-dir", str(run_dir), "--benchmark", benchmark,
                "--conditions", *OURS, "--limit", str(limit),
                "--split", "train", "--max-steps", str(args.max_steps),
                "--config-path", args.config_path,
                "--output-dir", str(eval_root),
            ]
            if benchmark == "alfworld":
                eval_command += ["--task-type", args.task_type]
                if args.alfworld_data:
                    eval_command += ["--alfworld-data", args.alfworld_data]
            _run(eval_command, progress, f"{benchmark} 冻结 replay")
            eval_dir = _created(eval_root, f"{run_dir.name}_train_*", eval_before)
            aggregated = json.loads((eval_dir / "aggregated.json").read_text(
                encoding="utf-8"))
            failed = [condition for condition in OURS if not (
                aggregated.get(condition, {}).get("bank_unchanged_after_eval")
                and aggregated.get(condition, {}).get("frozen_bank_sha256")
                and not aggregated.get(condition, {}).get(
                    "graph_validation", {}).get("errors"))]
            if failed:
                raise RuntimeError(f"{benchmark} 冻结审计失败：{failed}")
            entry["frozen_eval_dir"] = str(eval_dir)
        manifest["benchmarks"][benchmark] = entry
        (online_root / "all_small_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    except BaseException:
        progress.close(passed=False)
        raise

    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["passed"] = True
    manifest_path = online_root / "all_small_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    progress.log(f"\n[all-small] 全部完成：{manifest_path}")
    progress.close(passed=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
