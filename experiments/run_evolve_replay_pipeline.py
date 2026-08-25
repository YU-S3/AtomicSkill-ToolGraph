"""One-command online evolution followed by strict frozen train replay."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from experiments.common import OUR_CONDITIONS, PROJECT_ROOT


DEFAULT_CONDITIONS = ["atomic_graph_only", "tool_repo_only",
                      "atomic_skillgraph_full"]


def _newest(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime,
                     reverse=True)
    return matches[0] if matches else None


def _run(command: list[str]) -> None:
    print("\n[pipeline] " + " ".join(command), flush=True)
    # Sandbox and benchmark helpers invoke the interpreter by the generic name
    # ``python``.  A caller may launch this module via an absolute venv Python
    # without activating that venv, so make the same interpreter discoverable
    # to every descendant process as well.
    child_env = os.environ.copy()
    # Do not resolve the venv's ``python`` symlink to /usr/bin/python3: the
    # sibling ``python`` command we need is in the original venv bin directory.
    interpreter_dir = str(Path(sys.executable).parent)
    child_env["PATH"] = interpreter_dir + os.pathsep + child_env.get("PATH", "")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False,
                               env=child_env)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="在线进化完成后立即执行严格冻结 train replay")
    parser.add_argument("--benchmark", default="alfworld",
                        choices=["alfworld", "humaneval", "gsm8k", "toy"])
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--task-type", default="pick_heat_then_place_in_recep")
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--alfworld-data", default=None)
    parser.add_argument("--config-path", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--online-output", default=str(
        PROJECT_ROOT / "runs" / "small_post_fix"))
    parser.add_argument("--eval-output", default=str(
        PROJECT_ROOT / "runs" / "evolve_eval_post_fix"))
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--run-dir", default=None,
                        help="跳过在线阶段，冻结 replay 指定的已完成在线目录")
    parser.add_argument("--rerun-online", action="store_true",
                        help="与 --run-dir 同用：仅将所选 ours 从零重跑，保留 baseline")
    args = parser.parse_args()

    invalid = [condition for condition in args.conditions
               if condition not in OUR_CONDITIONS]
    if invalid:
        parser.error(f"冻结 pipeline 只接受 ours conditions：{invalid}")

    online_root = Path(args.online_output)
    eval_root = Path(args.eval_output)
    online_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    if args.rerun_online and not args.run_dir:
        parser.error("--rerun-online 必须与 --run-dir 同时使用")

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        if args.rerun_online:
            online_command = [
                sys.executable, "-m", "experiments.run_small",
                "--benchmark", args.benchmark,
                "--limit", str(args.limit),
                "--conditions", *args.conditions,
                "--max-steps", str(args.max_steps),
                "--config-path", args.config_path,
                "--run-dir", str(run_dir),
                "--fresh-conditions",
            ]
            if args.benchmark == "alfworld":
                online_command += ["--task-type", args.task_type]
                if args.alfworld_data:
                    online_command += ["--alfworld-data", args.alfworld_data]
            if args.mock:
                online_command.append("--mock")
            _run(online_command)
    else:
        before = {path.resolve() for path in online_root.glob(f"{args.benchmark}_*")}
        online_command = [
            sys.executable, "-m", "experiments.run_small",
            "--benchmark", args.benchmark,
            "--limit", str(args.limit),
            "--conditions", *args.conditions,
            "--max-steps", str(args.max_steps),
            "--config-path", args.config_path,
            "--output-dir", str(online_root),
        ]
        if args.benchmark == "alfworld":
            online_command += ["--task-type", args.task_type]
            if args.alfworld_data:
                online_command += ["--alfworld-data", args.alfworld_data]
        if args.mock:
            online_command.append("--mock")
        _run(online_command)
        after = {path.resolve() for path in online_root.glob(f"{args.benchmark}_*")}
        created = sorted(after - before, key=lambda path: path.stat().st_mtime,
                         reverse=True)
        run_dir = created[0] if created else _newest(online_root,
                                                     f"{args.benchmark}_*")
        if run_dir is None:
            raise RuntimeError("在线阶段成功退出，但未找到新运行目录")

    online_results = run_dir / "results.json"
    if not online_results.exists():
        raise FileNotFoundError(f"在线结果不完整：{online_results}")
    online_data = json.loads(online_results.read_text(encoding="utf-8"))
    missing = [condition for condition in args.conditions
               if condition not in online_data]
    if missing:
        raise RuntimeError(f"在线结果缺少条件，拒绝冻结：{missing}")

    eval_before = {path.resolve() for path in eval_root.glob(
        f"{run_dir.name}_train_*")}
    eval_command = [
        sys.executable, "-m", "experiments.run_evolve_eval",
        "--run-dir", str(run_dir), "--benchmark", args.benchmark,
        "--conditions", *args.conditions,
        "--limit", str(args.limit), "--split", "train",
        "--max-steps", str(args.max_steps),
        "--config-path", args.config_path,
        "--output-dir", str(eval_root),
    ]
    if args.benchmark == "alfworld":
        eval_command += ["--task-type", args.task_type]
        if args.alfworld_data:
            eval_command += ["--alfworld-data", args.alfworld_data]
    if args.mock:
        eval_command.append("--mock")
    _run(eval_command)

    eval_after = {path.resolve() for path in eval_root.glob(
        f"{run_dir.name}_train_*")}
    created_eval = sorted(eval_after - eval_before,
                          key=lambda path: path.stat().st_mtime, reverse=True)
    eval_dir = created_eval[0] if created_eval else _newest(
        eval_root, f"{run_dir.name}_train_*")
    if eval_dir is None or not (eval_dir / "aggregated.json").exists():
        raise RuntimeError("冻结 replay 成功退出，但结果目录不完整")
    aggregated = json.loads((eval_dir / "aggregated.json").read_text(
        encoding="utf-8"))
    invalid_frozen = [condition for condition in args.conditions if not (
        aggregated.get(condition, {}).get("bank_unchanged_after_eval")
        and aggregated.get(condition, {}).get("frozen_bank_sha256")
        and not aggregated.get(condition, {}).get("graph_validation", {}).get("errors"))]
    if invalid_frozen:
        raise RuntimeError(f"冻结审计失败：{invalid_frozen}")

    frozen_audit = {
        condition: {
            "bank_unchanged_after_eval": aggregated[condition][
                "bank_unchanged_after_eval"],
            "frozen_bank_sha256": aggregated[condition][
                "frozen_bank_sha256"],
            "graph_validation_error_count": len(
                aggregated[condition]["graph_validation"]["errors"]),
        }
        for condition in args.conditions
    }

    manifest = {
        "started_at": started_at,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmark": args.benchmark,
        "conditions": args.conditions,
        "limit": args.limit,
        "task_type": args.task_type,
        "max_steps": args.max_steps,
        "online_run_dir": str(run_dir),
        "frozen_eval_dir": str(eval_dir),
        "frozen_audit_passed": True,
        "frozen_audit": frozen_audit,
    }
    (eval_dir / "pipeline_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[pipeline] 完成\n在线结果：{run_dir}\n冻结结果：{eval_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
