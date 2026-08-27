"""冻结 Skill 进化效果评估（Train-Evolve-Test 两阶段，第二阶段）。

研究目标：单独评估"进化出的技能本身的质量"（而非在线适应过程）。

第一阶段（在线进化）：由既有实验入口完成（run_small / run_full），
产物位于 runs/<...>/<condition>/data/（SkillGraph + Tool Repository）。

第二阶段（本模块）：将各条件进化后的技能库**冻结**（只读，禁止任何
进化/统计写入），在 train（或 hold-out test）任务上重放，测量：
  - 冻结技能的复用覆盖率（direct/seeded 解决的任务比例）
  - 成功率 / tokens / direct-seeded-dynamic 分布
  - 与在线运行后半段（Late-run）的对照

用法示例（WSL 内）：
    # train 重放（与在线同一批任务）
    python -m experiments.run_evolve_eval \
        --run-dir runs/small/alfworld_20260822T102405 \
        --benchmark alfworld --task-type pick_heat_then_place_in_recep \
        --max-steps 100 --config-path configs/default.yaml \
        --alfworld-data ~/.cache/alfworld --split train

    # hold-out test 泛化（跳过在线训练的前 10 个任务）
    python -m experiments.run_evolve_eval \
        --run-dir runs/small/alfworld_20260822T102405 \
        --benchmark alfworld --task-type pick_heat_then_place_in_recep \
        --max-steps 100 --config-path configs/default.yaml \
        --alfworld-data ~/.cache/alfworld --split test --train-limit 10

    # 同时冻结评估 FlowEvo 完整库（train 重放；test 模式自动 --start-index）
    python -m experiments.run_evolve_eval \
        --run-dir runs/small/alfworld_20260822T102405 \
        --benchmark alfworld --task-type pick_heat_then_place_in_recep \
        --max-steps 100 --config-path configs/default.yaml \
        --alfworld-data ~/.cache/alfworld --split test --train-limit 10 \
        --eval-flowevo
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atomic_skillgraph.core.config import SystemConfig  # noqa: E402
from atomic_skillgraph.system import AtomicSkillGraphSystem  # noqa: E402
from atomic_skillgraph.graph.validator import validate_graph  # noqa: E402
from experiments.common import (  # noqa: E402
    BASELINE_FLOWEVO_CONDITIONS,
    PROJECT_ROOT,
    apply_condition,
    balanced_task_subset,
    load_conda_config,
    make_adapter,
    make_llm,
    run_balanced_baseline_condition,
    run_baseline_condition,
)
from experiments.report import (  # noqa: E402
    aggregate_results,
    save_aggregated,
    summarize_episodes,
    write_markdown_report,
)

OUR_CONDITIONS = ["atomic_graph_only", "tool_repo_only", "atomic_skillgraph_full"]


def _infer_online_limit(run_dir: Path, condition: str) -> int | None:
    """从在线运行的结果推断任务数。"""
    results_path = run_dir / "results.json"
    if not results_path.exists():
        return None
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
        entry = data.get(condition) or {}
        episodes = entry.get("episodes") or []
        if episodes:
            return len(episodes)
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _snapshot_frozen_bank(src_data: Path, dst_data: Path) -> str:
    """原子地建立不可变里程碑快照，拒绝混用不同在线 bank。"""
    source_digest = _bank_digest(src_data)
    marker_path = dst_data.parent / "frozen_snapshot.json"
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        recorded = str(marker.get("source_bank_sha256") or "")
        if recorded != source_digest:
            raise RuntimeError(
                f"eval-dir 已冻结另一在线里程碑：recorded={recorded[:12]}, "
                f"current={source_digest[:12]}；请为新里程碑使用新的 --eval-dir")
        if not dst_data.exists():
            raise RuntimeError("冻结快照标记存在但 data 目录缺失，拒绝继续")
        return source_digest
    if dst_data.exists() and any(dst_data.iterdir()):
        raise RuntimeError(
            f"冻结 data 已存在但缺少快照标记：{dst_data}；请使用新的 --eval-dir")

    dst_data.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix="frozen_snapshot_", dir=str(dst_data.parent)) as temp_name:
        temporary = Path(temp_name) / "data"
        temporary.mkdir()
        for sub in ("skill_graph", "tools"):
            src = src_data / sub
            if src.exists():
                shutil.copytree(src, temporary / sub)
        if dst_data.exists():
            # Only an empty directory is accepted above. rmdir is recoverable
            # in the sense that it cannot remove user data or non-empty trees.
            dst_data.rmdir()
        temporary.replace(dst_data)
    marker_tmp = marker_path.with_suffix(".json.tmp")
    marker_tmp.write_text(json.dumps({
        "source_data": str(src_data.resolve()),
        "source_bank_sha256": source_digest,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    marker_tmp.replace(marker_path)
    return source_digest


def _bank_digest(data_dir: Path) -> str:
    """对 SkillGraph 与 Tool Repository 做逐文件内容哈希。"""
    digest = hashlib.sha256()
    for sub in ("skill_graph", "tools"):
        root = data_dir / sub
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            digest.update(path.relative_to(data_dir).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def run_frozen_condition(condition: str, run_dir: Path, eval_dir: Path,
                         config: SystemConfig, adapter, tasks: list,
                         mock_script=None, monitor=None) -> dict:
    """冻结评估单个 ours 条件。"""
    src_data = run_dir / condition / "data"
    if not (src_data / "skill_graph").exists() and not (src_data / "tools").exists():
        raise FileNotFoundError(f"未找到在线进化产物：{src_data}（请先运行在线实验）")
    dst_data = eval_dir / condition / "data"
    source_digest = _snapshot_frozen_bank(src_data, dst_data)

    # 冻结 replay 必须恢复该 condition 在线训练时的 feature flags，不能让所有
    # 条件都按 Full 配置运行。
    conditioned = apply_condition(config, condition)
    cfg = dataclasses.replace(conditioned, data_dir=dst_data, freeze_skills=True)
    llm = make_llm(cfg, mock_script=mock_script)
    system = AtomicSkillGraphSystem(cfg, adapter, llm)
    graph_report = validate_graph(system.registry, system.tool_registry)
    if not graph_report.passed:
        preview = "; ".join(graph_report.errors[:3])
        raise RuntimeError(
            f"冻结库未通过当前代码的 SkillGraph 校验，不能作为正式 replay："
            f"{condition}: {preview}")
    digest_before = _bank_digest(dst_data)
    if digest_before != source_digest:
        raise RuntimeError(f"冻结快照内容哈希与在线源不一致：{condition}")
    frozen_before = {
        "skills": system.registry.stats(),
        "tools": system.tool_registry.stats(),
    }
    task_signature = [{
        "task_id": str(task.task_id),
        "task_type": str(task.task_type),
        "game_file": str(task.context.get("game_file") or ""),
    } for task in tasks]
    progress_path = eval_dir / condition / "frozen_progress.json"
    episodes: list[dict] = []
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("task_signature") != task_signature:
            raise RuntimeError(
                f"{condition} 冻结评估断点与本次任务清单不一致；请使用新输出目录")
        episodes = list(progress.get("episodes") or [])
    completed = len(episodes)
    if monitor is not None and completed:
        monitor.task_update(completed, note=f"resume {completed}/{len(tasks)}")
    for index, task in enumerate(tasks[completed:], start=completed + 1):
        episodes.append(system.run_task(task))
        temporary = progress_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({
            "condition": condition,
            "task_signature": task_signature,
            "completed": len(episodes),
            "episodes": episodes,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(progress_path)
        if monitor is not None:
            monitor.task_update(index, note=str(task.task_id))
    frozen_after = {
        "skills": system.registry.stats(),
        "tools": system.tool_registry.stats(),
    }
    digest_after = _bank_digest(dst_data)
    unchanged = digest_before == digest_after
    if not unchanged:
        raise RuntimeError(f"冻结评估期间技能库发生写入：{condition}")
    return {
        "condition": condition,
        "kind": "frozen_eval",
        "episodes": episodes,
        "frozen_bank": frozen_before,
        "frozen_bank_sha256": digest_before,
        "graph_validation": graph_report.to_dict(),
        "condition_features": dataclasses.asdict(cfg.features),
        "bank_unchanged_after_eval": unchanged,
    }


def run_flowevo_frozen(run_dir: Path, eval_dir: Path, config_path: str, *,
                       limit: int, max_steps: int, task_type: str | None,
                       alfworld_split: str = "eval_out_of_distribution",
                       alfworld_data: str | None = None,
                       start_index: int = 0,
                       on_progress=None,
                       flowevo_run_dir: Path | None = None) -> dict:
    """冻结评估 FlowEvo 完整库（加载在线 checkpoint 的 library，关闭 compile）。"""
    from atomic_skillgraph.adapters.flowevo import run_flowevo_baseline
    library_dir = flowevo_run_dir or (run_dir / "flowevo_flowevo")
    result = run_flowevo_baseline(
        project_root=PROJECT_ROOT,
        benchmark="alfworld",
        output_dir=eval_dir / "flowevo_flowevo",
        conditions=["full_library"],
        config_path=config_path,
        limit=limit,
        task_type=task_type,
        alfworld_split=alfworld_split,
        alfworld_data=alfworld_data,
        max_steps=max_steps,
        start_index=start_index,
        on_progress=on_progress,
        extra_env={"ALFWORLD_FREEZE_LIBRARY_DIR": str(library_dir)},
    )
    from experiments.common import parse_flowevo_results
    episodes = parse_flowevo_results(eval_dir / "flowevo_flowevo",
                                     "alfworld", "full_library")
    return {"condition": "flowevo", "kind": "flowevo_frozen_eval",
            "library_dir": str(library_dir),
            "subprocess": result, "episodes": episodes}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="冻结 Skill 进化效果评估（Train-Evolve-Test 第二阶段）")
    parser.add_argument("--run-dir", required=True,
                        help="在线进化运行的输出目录（含 <condition>/data 与 results.json）")
    parser.add_argument("--benchmark", required=True,
                        choices=["alfworld", "humaneval", "gsm8k", "toy"])
    parser.add_argument("--conditions", nargs="+", default=OUR_CONDITIONS)
    parser.add_argument("--task-type", default=None,
                        help="ALFWorld 任务类型（需与在线运行一致）")
    parser.add_argument("--task-types", nargs="+", default=None,
                        help="ALFWorld 多 label 均衡冻结 replay")
    parser.add_argument("--per-type-limit", type=int, default=None,
                        help="与 --task-types 同用：每类任务数")
    parser.add_argument("--limit", type=int, default=None,
                        help="评估任务数（默认取在线运行的任务数）")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="步数预算（需与在线运行一致）")
    parser.add_argument("--split", choices=["train", "test", "heldout"],
                        default="train",
                        help=("train=同批重放；test=同一数据 split 内按索引留出；"
                              "heldout=在显式 --alfworld-split 上完整评估"))
    parser.add_argument("--train-limit", type=int, default=None,
                        help="test 模式下在线训练使用的任务数（跳过前 train-limit 个）")
    parser.add_argument("--alfworld-data", default=None)
    parser.add_argument(
        "--alfworld-split",
        choices=["train", "eval_in_distribution", "eval_out_of_distribution"],
        default="eval_out_of_distribution",
        help="评估数据 split；valid_unseen 对应 eval_out_of_distribution",
    )
    parser.add_argument(
        "--baseline-conditions", nargs="*", default=[],
        choices=list(BASELINE_FLOWEVO_CONDITIONS),
        help="在评估 split 上从原始入口运行的非冻结 baseline",
    )
    parser.add_argument("--expected-train-count", type=int, default=None,
                        help="heldout 协议门禁：在线训练任务数必须等于该值")
    parser.add_argument("--expected-heldout-count", type=int, default=None,
                        help="heldout 协议门禁：评估任务数必须等于该值")
    parser.add_argument("--config-path", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "runs" / "evolve_eval"))
    parser.add_argument("--eval-dir", default=None,
                        help="精确复用冻结评估目录（支持逐题断点续跑）")
    parser.add_argument("--eval-flowevo", action="store_true",
                        help="同时冻结评估 FlowEvo 完整库（加载在线 checkpoint）")
    parser.add_argument("--flowevo-run-dir", default=None,
                        help="FlowEvo checkpoint 所在目录（默认 <run-dir>/flowevo_flowevo；"
                             "独立补跑 baseline 时指向新目录）")
    parser.add_argument("--mock", action="store_true", help="toy 基准联调用 MockLLM")
    args = parser.parse_args()

    if args.task_type and args.task_types:
        parser.error("--task-type 与 --task-types 不能同时使用")
    if args.task_types and (args.per_type_limit is None or args.per_type_limit <= 0):
        parser.error("--task-types 需要正整数 --per-type-limit")
    if args.eval_flowevo and "flowevo" in args.baseline_conditions:
        parser.error("--eval-flowevo 与 --baseline-conditions flowevo 不能同时使用")

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"[错误] 运行目录不存在：{run_dir}")
        return 1
    # 只评估在线运行实际产出过的条件（<condition>/data 存在）
    conditions = [c for c in args.conditions if (run_dir / c / "data").exists()]
    missing = [c for c in args.conditions if c not in conditions]
    if missing:
        print(f"[提示] 以下条件无在线进化产物，跳过：{missing}")
    if not conditions:
        print("[错误] 没有任何条件存在在线进化产物")
        return 1
    config = load_conda_config(args.config_path)
    config = dataclasses.replace(config, max_steps=args.max_steps)
    if args.mock:
        config = dataclasses.replace(config, llm=dataclasses.replace(config.llm, mock=True))

    output_dir = Path(args.output_dir)
    eval_dir = (Path(args.eval_dir).resolve() if args.eval_dir else
                output_dir / f"{run_dir.name}_{args.split}_{time.strftime('%Y%m%dT%H%M%S')}")
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"冻结 Skill 进化效果评估：{run_dir.name}  split={args.split}  "
          f"benchmark={args.benchmark}  max_steps={args.max_steps}")
    print(f"条件：{conditions}" + (" + flowevo(冻结库)" if args.eval_flowevo else ""))
    print("=" * 78)

    # 加载全部任务（供 train/test 切分）
    adapter = make_adapter(args.benchmark, config, task_type=args.task_type,
                           split=args.alfworld_split,
                           alfworld_data=args.alfworld_data, max_steps=args.max_steps,
                           kinds=("code", "math", "env"))
    load_limit = (int(args.limit) if args.split == "heldout"
                  and args.limit and not args.task_types else 0)
    all_tasks = adapter.load_tasks(limit=load_limit, task_type=args.task_type)
    if args.task_types:
        all_tasks = balanced_task_subset(
            all_tasks, list(args.task_types), int(args.per_type_limit))
    if args.split == "test":
        if args.train_limit is None:
            print("[错误] test 切分需要 --train-limit（在线训练任务数）")
            return 1
        tasks = all_tasks[args.train_limit:]
    else:
        limit = args.limit or _infer_online_limit(run_dir, conditions[0]) or len(all_tasks)
        tasks = all_tasks[:limit]
    if not tasks:
        print("[错误] 评估任务为空")
        return 1
    print(f"评估任务 {len(tasks)} 个：{tasks[0].task_id} ... {tasks[-1].task_id}")

    eval_manifest = {
        "benchmark": args.benchmark,
        "split": args.alfworld_split if args.benchmark == "alfworld" else args.split,
        "task_count": len(tasks),
        "tasks": [{
            "task_id": task.task_id,
            "task_type": task.task_type,
            "game_file": str(task.context.get("game_file") or ""),
        } for task in tasks],
    }
    if args.benchmark == "alfworld":
        eval_manifest_path = eval_dir / "task_manifest.json"
        if eval_manifest_path.exists():
            previous = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
            if previous != eval_manifest:
                print("[错误] eval-dir 已绑定另一组任务，拒绝跨 split 续跑")
                return 1
        eval_manifest_path.write_text(
            json.dumps(eval_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.split == "heldout":
        source_manifest_path = run_dir / "task_manifest.json"
        if not source_manifest_path.exists():
            print("[错误] heldout 评估要求在线目录含 task_manifest.json，"
                  "无法证明训练/测试隔离")
            return 1
        source_manifest = json.loads(
            source_manifest_path.read_text(encoding="utf-8"))
        if args.benchmark == "alfworld":
            source_split = str(source_manifest.get("split"))
            if source_split != "train":
                print(f"[错误] 正式 heldout 的在线 bank 必须来自 train，实际为 {source_split}")
                return 1
            if source_split == args.alfworld_split:
                print("[错误] heldout split 与在线进化 split 相同，拒绝数据泄漏")
                return 1
        source_task_count = int(source_manifest.get("selection", {}).get(
            "task_count", len(source_manifest.get("tasks") or [])))
        if (args.expected_train_count is not None
                and source_task_count != args.expected_train_count):
            print(f"[错误] 在线训练任务数应为 {args.expected_train_count}，"
                  f"实际为 {source_task_count}")
            return 1
        if (args.expected_heldout_count is not None
                and len(tasks) != args.expected_heldout_count):
            print(f"[错误] heldout 任务数应为 {args.expected_heldout_count}，"
                  f"实际为 {len(tasks)}")
            return 1
        train_files = {str(item.get("game_file") or "")
                       for item in source_manifest.get("tasks") or []
                       if str(item.get("game_file") or "")}
        eval_files = {str(item.get("game_file") or "")
                      for item in eval_manifest["tasks"]
                      if str(item.get("game_file") or "")}
        overlap = sorted(train_files & eval_files)
        if overlap:
            print(f"[错误] train/heldout game_file 重叠 {len(overlap)} 项，拒绝评估")
            return 1
        (eval_dir / "protocol_manifest.json").write_text(json.dumps({
            "protocol": "train_evolve_freeze_heldout",
            "source_run_dir": str(run_dir),
            "train_split": source_manifest.get("split"),
            "train_task_count": source_task_count,
            "heldout_split": eval_manifest["split"],
            "heldout_task_count": len(tasks),
            "game_file_overlap_count": 0,
            "frozen_ours_conditions": conditions,
            "heldout_baseline_conditions": list(args.baseline_conditions),
            "heldout_baseline_modes": {
                "baseline_dynamic": (
                    "stateless_dynamic_balanced_per_task_type"
                    if args.task_types else "stateless_dynamic"),
                "flowevo": (
                    "original_online_full_library_balanced_per_task_type"
                    if args.task_types else "original_online_full_library"),
            },
            "task_types": list(args.task_types or []),
            "per_type_limit": args.per_type_limit,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    from atomic_skillgraph.adapters.toy_benchmarks import build_mock_script
    from experiments.progress import ProgressMonitor
    mock_script = build_mock_script() if args.benchmark == "toy" else None

    monitor = ProgressMonitor()
    monitor.set_total_conditions(
        len(conditions) + len(args.baseline_conditions)
        + (1 if args.eval_flowevo else 0))
    results: dict = {}
    for condition in conditions:
        print(f"[frozen_eval] condition={condition} ...")
        monitor.condition_start(condition, task_total=len(tasks))
        results[condition] = run_frozen_condition(
            condition, run_dir, eval_dir, config, adapter, tasks,
            mock_script=mock_script, monitor=monitor)
        monitor.condition_finish()

    for condition in args.baseline_conditions:
        print(f"[heldout_baseline] condition={condition} split={args.alfworld_split} ...")
        monitor.condition_start(condition, task_total=len(tasks))
        if args.task_types:
            results[condition] = run_balanced_baseline_condition(
                condition, args.benchmark,
                config_path=str(PROJECT_ROOT / "configs" / "flowevo_default.yaml"),
                output_dir=eval_dir / f"{condition}_flowevo",
                task_types=list(args.task_types),
                per_type_limit=int(args.per_type_limit),
                alfworld_split=args.alfworld_split,
                alfworld_data=args.alfworld_data,
                max_steps=args.max_steps, monitor=monitor)
        else:
            results[condition] = run_baseline_condition(
                condition, args.benchmark,
                config_path=str(PROJECT_ROOT / "configs" / "flowevo_default.yaml"),
                output_dir=eval_dir / f"{condition}_flowevo",
                limit=len(tasks), task_type=args.task_type,
                alfworld_split=args.alfworld_split,
                alfworld_data=args.alfworld_data,
                max_steps=args.max_steps, monitor=monitor)
        monitor.condition_finish()

    if args.eval_flowevo:
        print("[frozen_eval] condition=flowevo（冻结库）...")
        # FlowEvo 的 results.json 条目是单条 summary（parse_flowevo_results），
        # 不能按 episode 数推断任务数；直接与 ours 冻结评估使用同一任务列表。
        limit = args.limit or len(tasks)
        start_index = args.train_limit if args.split == "test" else 0
        monitor.condition_start("flowevo(冻结库)", task_total=limit)
        flowevo_run_dir = Path(args.flowevo_run_dir) if args.flowevo_run_dir else None
        def _flowevo_progress(done: int, total: int, note: str) -> None:
            monitor.task_update(done, note=note.strip()[:80])
        results["flowevo"] = run_flowevo_frozen(
            run_dir, eval_dir,
            str(PROJECT_ROOT / "configs" / "flowevo_default.yaml"),
            limit=limit, start_index=start_index,
            max_steps=args.max_steps, task_type=args.task_type,
            alfworld_split=args.alfworld_split,
            alfworld_data=args.alfworld_data,
            on_progress=_flowevo_progress,
            flowevo_run_dir=flowevo_run_dir)
        monitor.condition_finish()
    monitor.finish()

    aggregated = aggregate_results(results)
    for condition, result in results.items():
        if condition not in aggregated or result.get("kind") != "frozen_eval":
            continue
        aggregated[condition].update({
            "frozen_bank": result.get("frozen_bank", {}),
            "frozen_bank_sha256": result.get("frozen_bank_sha256", ""),
            "bank_unchanged_after_eval": bool(
                result.get("bank_unchanged_after_eval", False)),
            "graph_validation": result.get("graph_validation", {}),
            "condition_features": result.get("condition_features", {}),
        })
    (eval_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    save_aggregated(aggregated, eval_dir / "aggregated.json")
    write_markdown_report(aggregated, eval_dir / "report.md",
                          title=f"冻结 Skill 进化效果评估报告（{run_dir.name}, {args.split}）")

    # 与在线运行对照
    online_results_path = run_dir / "results.json"
    print("\n" + "-" * 78)
    print("冻结评估 vs 在线运行对照（success_rate / Late-run / 复用 / tokens）：")
    online: dict = {}
    if online_results_path.exists():
        online = json.loads(online_results_path.read_text(encoding="utf-8"))
    for name, value in aggregated.items():
        s = value
        if value.get("kind") == "flowevo_baseline" or value.get("kind") == "flowevo_frozen_eval":
            print(f"  {name:<26} frozen={_pct(s.get('success_rate'))}"
                  f"  tokens={s.get('avg_tokens_per_task', 0):.0f}")
            continue
        online_entry = (online.get(name) or {}).get("episodes") or []
        online_s = summarize_episodes(online_entry) if online_entry else {}
        print(f"  {name:<26} frozen={_pct(s.get('success_rate'))}"
              f"  late_run={_pct(s.get('late_run_success_rate'))}"
              f"  reuse={_pct(s.get('atomic_reuse_rate'))}"
              f"  tokens={s.get('avg_tokens_per_task', 0):.0f}"
              f"   | 在线 late_run={_pct(online_s.get('late_run_success_rate'))}"
              f" 在线 tokens={online_s.get('avg_tokens_per_task', 0):.0f}")
    print(f"\n报告：{eval_dir / 'report.md'}")
    print(f"汇总：{eval_dir / 'aggregated.json'}")
    return 0


def _pct(value) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return "—"


if __name__ == "__main__":
    raise SystemExit(main())
