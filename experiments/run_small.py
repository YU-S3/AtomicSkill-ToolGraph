"""Stage 3：小规模数据结果测试（先看有没有效果）。

- ALFWorld：同一类任务 10 个（--task-type 指定，默认 pick_heat_then_place_in_recep）
- HumanEval：10 个任务
- GSM8K：50 个问题
- toy（可选）：无 API 小规模联调（同 Stage-2 任务集）

baseline 条件（baseline_dynamic / flowevo）走 vendored FlowEvo 原版 runner（真实 API）。
ours 条件走 v2.0 runtime；--mock 时用 MockLLM 走完整链路（验证管线、不看效果）。

用法：
    # 无 API、按有 API 方式走完整实验链路（HumanEval 10）
    python -m experiments.run_small --benchmark humaneval --limit 10 --mock

    # 真实 API 小规模结果测试（5 个核心条件）
    python -m experiments.run_small --benchmark humaneval --limit 10 --config-path configs/default.yaml
    python -m experiments.run_small --benchmark gsm8k --limit 50 --config-path configs/default.yaml
    python -m experiments.run_small --benchmark alfworld --limit 10 \
        --task-type pick_heat_then_place_in_recep --config-path configs/default.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.common import (  # noqa: E402
    ALL_CONDITIONS,
    load_conda_config,
    make_adapter,
    run_conditions,
    balanced_task_subset,
)
from experiments.report import (  # noqa: E402
    aggregate_results,
    save_aggregated,
    write_markdown_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CORE_CONDITIONS = ["baseline_dynamic", "flowevo", "atomic_graph_only",
                   "tool_repo_only", "atomic_skillgraph_full"]
DEFAULT_LIMITS = {"alfworld": 10, "humaneval": 10, "gsm8k": 50, "toy": 12}


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.exists():
        return ""
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _condition_bank_digests(run_dir: Path,
                            conditions: list[str]) -> dict[str, str]:
    """Hash only mutable knowledge banks, excluding reports/progress files."""
    return {
        condition: _tree_digest(run_dir / condition / "data")
        for condition in conditions
        if (run_dir / condition / "data").exists()
    }


def _clone_online_run(source_run: Path, destination_run: Path,
                      conditions: list[str]) -> dict[str, str]:
    """Clone an immutable milestone into an independently writable branch."""
    if not source_run.is_dir():
        raise ValueError(f"在线来源不存在或不是目录：{source_run}")
    if source_run.resolve() == destination_run.resolve():
        raise ValueError("扩展目标必须是新目录，不能原地修改来源里程碑")
    missing = [condition for condition in conditions
               if not (source_run / condition / "data").is_dir()
               or not (source_run / condition / "online_progress.json").is_file()]
    if missing:
        raise ValueError(f"来源里程碑缺少完整 condition bank/progress：{missing}")
    if not (source_run / "task_manifest.json").is_file():
        raise ValueError("来源里程碑缺少 task_manifest.json")
    source_manifest = json.loads(
        (source_run / "task_manifest.json").read_text(encoding="utf-8"))
    source_tasks = list(source_manifest.get("tasks") or [])
    source_signature = [{
        "task_id": str(item.get("task_id") or ""),
        "task_type": str(item.get("task_type") or ""),
        "game_file": str(item.get("game_file") or ""),
    } for item in source_tasks]
    for condition in conditions:
        progress = json.loads((source_run / condition /
                               "online_progress.json").read_text(encoding="utf-8"))
        if (int(progress.get("completed", -1)) != len(source_tasks)
                or list(progress.get("task_signature") or []) != source_signature
                or len(progress.get("episodes") or []) != len(source_tasks)):
            raise ValueError(
                f"来源里程碑尚未完整完成：{condition}，"
                f"要求 {len(source_tasks)} 个已记录 episode")

    source_digests = _condition_bank_digests(source_run, conditions)
    lineage_path = destination_run / "online_lineage.json"
    if destination_run.exists():
        if not lineage_path.is_file():
            raise ValueError(f"扩展目标已存在且不是可续跑分支：{destination_run}")
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        if (str(lineage.get("source_run_dir") or "")
                != str(source_run.resolve())):
            raise ValueError("扩展目标的来源里程碑与本次 --extend-from-run 不一致")
        recorded = dict(lineage.get("source_condition_bank_sha256") or {})
        if recorded != source_digests:
            raise ValueError("来源 120 里程碑在分支建立后发生变化，拒绝混合续跑")
        return source_digests

    destination_run.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_run, destination_run)
    copied_digests = _condition_bank_digests(destination_run, conditions)
    if source_digests != copied_digests:
        raise RuntimeError("在线里程碑克隆后 bank 哈希不一致，拒绝继续")
    lineage_path.write_text(json.dumps({
        "mode": "writable_online_fork",
        "source_run_dir": str(source_run.resolve()),
        "source_condition_bank_sha256": source_digests,
        "source_remains_unchanged": True,
        "destination_freeze_skills": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return source_digests


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 3: 小规模数据结果测试")
    parser.add_argument("--benchmark", required=True,
                        choices=["alfworld", "humaneval", "gsm8k", "toy"])
    parser.add_argument("--limit", type=int, default=None,
                        help="任务数（默认：alfworld=10, humaneval=10, gsm8k=50）")
    parser.add_argument("--task-type", default=None,
                        help="ALFWorld：同一类任务过滤（默认 pick_heat_then_place_in_recep）")
    parser.add_argument("--task-types", nargs="+", default=None,
                        help="ALFWorld：多个官方 label，按 --per-type-limit 均衡交错")
    parser.add_argument("--per-type-limit", type=int, default=None,
                        help="与 --task-types 同用：每类任务数")
    parser.add_argument("--conditions", nargs="+", default=CORE_CONDITIONS)
    parser.add_argument("--config-path", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "runs" / "small"))
    parser.add_argument("--mock", action="store_true",
                        help="无 API 走完整实验链路（ours 条件用 MockLLM；baseline 条件跳过）")
    parser.add_argument("--alfworld-data", default=None,
                        help="ALFWorld 数据目录（默认 ~/.cache/alfworld 或 ALFWORLD_DATA）")
    parser.add_argument(
        "--alfworld-split",
        choices=["train", "eval_in_distribution", "eval_out_of_distribution"],
        default="eval_out_of_distribution",
        help="ALFWorld split；valid_unseen 对应 eval_out_of_distribution",
    )
    parser.add_argument("--max-steps", type=int, default=None,
                        help="交互环境步数预算（默认 50；ours 与 baseline 对称生效）")
    parser.add_argument("--resume", action="store_true",
                        help="复用 output-dir 下最近的运行目录，跳过已完成条件")
    parser.add_argument("--run-dir", default=None,
                        help="精确复用指定运行目录（不新建时间戳目录）")
    parser.add_argument("--fresh-conditions", action="store_true",
                        help="重跑所选条件前将旧 condition data 移入可恢复备份")
    parser.add_argument(
        "--extend-online", action="store_true",
        help=("从 --extend-from-run 克隆一个在线里程碑到新的 --run-dir，"
              "再以前缀追加任务继续进化"),
    )
    parser.add_argument(
        "--extend-from-run", default=None,
        help="在线扩展的只读来源里程碑；来源保持不变，新的 run-dir 可正常写入",
    )
    args = parser.parse_args()

    if args.task_type and args.task_types:
        parser.error("--task-type 与 --task-types 不能同时使用")
    if args.task_types and args.benchmark != "alfworld":
        parser.error("--task-types 目前只用于 ALFWorld label 均衡取样")
    if args.task_types and (args.per_type_limit is None or args.per_type_limit <= 0):
        parser.error("--task-types 需要正整数 --per-type-limit")
    if args.extend_online:
        if args.benchmark != "alfworld" or args.alfworld_split != "train":
            parser.error("--extend-online 只支持 ALFWorld train 在线进化")
        if not args.run_dir:
            parser.error("--extend-online 必须显式指定新的 --run-dir")
        if not args.extend_from_run:
            parser.error("--extend-online 必须指定 --extend-from-run 来源里程碑")
        if args.fresh_conditions:
            parser.error("--extend-online 不能与 --fresh-conditions 同用")
        invalid = [name for name in args.conditions
                   if name in ("baseline_dynamic", "flowevo")]
        if invalid:
            parser.error("--extend-online 只追加 ours 条件，不追加 baseline："
                         f"{invalid}")
    limit = args.limit if args.limit is not None else DEFAULT_LIMITS[args.benchmark]
    if args.task_types:
        limit = int(args.per_type_limit) * len(args.task_types)
    task_type = (None if args.task_types else args.task_type or (
        "pick_heat_then_place_in_recep" if args.benchmark == "alfworld" else None))
    config = load_conda_config(args.config_path)
    if args.mock:
        config.llm.mock = True
    if args.max_steps and args.max_steps > 0:
        import dataclasses
        config = dataclasses.replace(config, max_steps=args.max_steps)

    output_dir = Path(args.output_dir)
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        if args.extend_online:
            print(f"[extend-online] 新的可写运行目录：{run_dir}")
        else:
            print(f"[rerun] 精确复用运行目录：{run_dir}")
    elif args.resume:
        existing = sorted(output_dir.glob(f"{args.benchmark}_*"), reverse=True)
        if existing:
            run_dir = existing[0]
            print(f"[resume] 复用运行目录：{run_dir}")
        else:
            run_dir = output_dir / f"{args.benchmark}_{timestamp}"
    else:
        run_dir = output_dir / f"{args.benchmark}_{timestamp}"
    if args.extend_online:
        source_run = Path(args.extend_from_run).resolve()
        try:
            _clone_online_run(source_run, run_dir, list(args.conditions))
        except ValueError as exc:
            parser.error(str(exc))
        print(f"[extend-online] 已克隆只读来源：{source_run}")
        print("[extend-online] 新副本 freeze_skills=False，可正常更新 Skill/Tool/utility")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

    if args.fresh_conditions:
        invalid_fresh = [name for name in args.conditions
                         if name in ("baseline_dynamic", "flowevo")]
        if invalid_fresh:
            parser.error("--fresh-conditions 仅支持 ours，不重置 baseline："
                         f"{invalid_fresh}")
        backup_root = run_dir / "_rerun_backups" / timestamp
        for condition in args.conditions:
            old_data = run_dir / condition
            if old_data.exists():
                backup_root.mkdir(parents=True, exist_ok=True)
                target = backup_root / condition
                old_data.rename(target)
                print(f"[rerun] 旧数据已备份：{old_data} -> {target}")

    print("=" * 78)
    print(f"Stage 3 小规模测试：{args.benchmark}（limit={limit}"
          + (f", task_type={task_type}" if task_type else "")
          + (f", balanced_types={args.task_types}, per_type={args.per_type_limit}"
             if args.task_types else "")
          + (f", max_steps={config.max_steps}" if args.benchmark == "alfworld" else "")
          + "）")
    print(f"条件：{args.conditions}   mock={args.mock}")
    print("=" * 78)

    adapter = make_adapter(args.benchmark, config,
                           task_type=task_type, split=args.alfworld_split,
                           alfworld_data=args.alfworld_data,
                           max_steps=config.max_steps,
                           kinds=("code", "math", "env"))
    if args.task_types:
        if args.benchmark == "alfworld" and hasattr(adapter, "load_balanced_tasks"):
            tasks = adapter.load_balanced_tasks(
                list(args.task_types), int(args.per_type_limit))
        else:
            tasks = balanced_task_subset(
                adapter.load_tasks(limit=0, task_type=task_type),
                list(args.task_types), int(args.per_type_limit))
        limit = len(tasks)
    else:
        tasks = adapter.load_tasks(limit=limit, task_type=task_type)
    if not tasks:
        print(f"[错误] 未加载到任务。请检查数据可用性（HF 网络 / ALFWorld 数据目录）。")
        return 1
    print(f"已加载任务 {len(tasks)} 个：{tasks[0].task_id} ... {tasks[-1].task_id}")
    if args.benchmark == "alfworld":
        manifest = {
            "benchmark": "alfworld",
            "split": args.alfworld_split,
            "selection": {
                "task_types": list(args.task_types or ([task_type] if task_type else [])),
                "per_type_limit": args.per_type_limit,
                "task_count": len(tasks),
            },
            "tasks": [{
                "task_id": task.task_id,
                "task_type": task.task_type,
                "game_file": str(task.context.get("game_file") or ""),
            } for task in tasks],
        }
        manifest_path = run_dir / "task_manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            prior_core = {key: value for key, value in previous.items()
                          if key != "extension_history"}
            if prior_core == manifest:
                manifest["extension_history"] = list(
                    previous.get("extension_history") or [])
            elif args.extend_online:
                prior_tasks = list(previous.get("tasks") or [])
                prior_selection = dict(previous.get("selection") or {})
                same_protocol = (
                    previous.get("benchmark") == manifest.get("benchmark")
                    and previous.get("split") == manifest.get("split")
                    and list(prior_selection.get("task_types") or [])
                    == list(manifest["selection"].get("task_types") or [])
                )
                strict_prefix = (
                    len(prior_tasks) < len(manifest["tasks"])
                    and prior_tasks == manifest["tasks"][:len(prior_tasks)]
                )
                prior_per_type = int(prior_selection.get("per_type_limit") or 0)
                next_per_type = int(manifest["selection"].get(
                    "per_type_limit") or 0)
                if not (same_protocol and strict_prefix
                        and next_per_type > prior_per_type):
                    raise RuntimeError(
                        "--extend-online 只允许同一 train split、相同 task types、"
                        "严格前缀且 per-type-limit 增大的任务清单")
                history = list(previous.get("extension_history") or [])
                history.append({
                    "task_count": len(prior_tasks),
                    "per_type_limit": prior_per_type,
                    "last_task_id": (prior_tasks[-1].get("task_id")
                                     if prior_tasks else ""),
                })
                manifest["extension_history"] = history
                print(f"[extend-online] {len(prior_tasks)} -> {len(manifest['tasks'])} "
                      "tasks；保留三个 ours bank 并继续进化")
            else:
                raise RuntimeError(
                    "run-dir 已绑定另一组 ALFWorld 任务；扩展同一 train bank "
                    "必须显式使用 --extend-online")
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # mock 模式下跳过 baseline（FlowEvo 原版无 mock LLM）；toy 无 baseline 映射
    skip_baseline = args.mock or args.benchmark == "toy"
    conditions = [c for c in args.conditions
                  if not (skip_baseline and c in ("baseline_dynamic", "flowevo"))]
    if skip_baseline and conditions != args.conditions:
        print(f"[提示] {'mock 模式' if args.mock else 'toy benchmark'} 跳过 baseline 条件"
              f"（原版 FlowEvo 需要真实 API / 不支持 toy）。")

    # --resume：跳过该运行目录中已完成的条件
    results_path = run_dir / "results.json"
    if args.resume and not args.extend_online and results_path.exists():
        done = set(json.loads(results_path.read_text(encoding="utf-8")).keys())
        skipped = [c for c in conditions if c in done]
        conditions = [c for c in conditions if c not in done]
        if skipped:
            print(f"[resume] 跳过已完成条件：{skipped}")
        if not conditions:
            print("[resume] 所有条件均已完成，无需运行。")
            return 0

    from atomic_skillgraph.adapters.toy_benchmarks import build_mock_script
    mock_script = build_mock_script() if args.benchmark == "toy" else None

    # 捕获历史结果（run_conditions 会增量覆写 results.json）
    prior_results: dict = {}
    if (args.resume or args.run_dir) and results_path.exists():
        prior_results = json.loads(results_path.read_text(encoding="utf-8"))

    results = run_conditions(
        conditions=conditions,
        benchmark=args.benchmark,
        config=config,
        adapter=adapter,
        tasks=tasks,
        output_dir=run_dir,
        limit=limit,
        config_path=str(PROJECT_ROOT / "configs" / "flowevo_default.yaml"),
        task_type=task_type,
        alfworld_split=args.alfworld_split,
        alfworld_data=args.alfworld_data,
        max_steps=config.max_steps if args.benchmark == "alfworld" else None,
        mock_script=mock_script,
        initial_results=prior_results,
        allow_task_extension=args.extend_online,
    )
    # --resume：合并历史结果（report 覆盖全部条件，而非仅本次新跑的）
    if prior_results:
        results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                encoding="utf-8")

    aggregated = aggregate_results(results)
    save_aggregated(aggregated, run_dir / "aggregated.json")
    write_markdown_report(aggregated, run_dir / "report.md",
                          title=f"Stage 3 小规模测试报告（{args.benchmark}）")

    print("\n" + "-" * 78)
    print("结果速览（success_rate）：")
    for name, value in aggregated.items():
        rate = value.get("success_rate")
        print(f"  {name:<28} {rate * 100:6.1f}%" if rate is not None else
              f"  {name:<28}  N/A")
    print(f"\n报告：{run_dir / 'report.md'}")
    print(f"汇总：{run_dir / 'aggregated.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
