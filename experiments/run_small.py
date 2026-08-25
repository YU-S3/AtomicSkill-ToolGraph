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
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.common import (  # noqa: E402
    ALL_CONDITIONS,
    load_conda_config,
    make_adapter,
    run_conditions,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 3: 小规模数据结果测试")
    parser.add_argument("--benchmark", required=True,
                        choices=["alfworld", "humaneval", "gsm8k", "toy"])
    parser.add_argument("--limit", type=int, default=None,
                        help="任务数（默认：alfworld=10, humaneval=10, gsm8k=50）")
    parser.add_argument("--task-type", default=None,
                        help="ALFWorld：同一类任务过滤（默认 pick_heat_then_place_in_recep）")
    parser.add_argument("--conditions", nargs="+", default=CORE_CONDITIONS)
    parser.add_argument("--config-path", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "runs" / "small"))
    parser.add_argument("--mock", action="store_true",
                        help="无 API 走完整实验链路（ours 条件用 MockLLM；baseline 条件跳过）")
    parser.add_argument("--alfworld-data", default=None,
                        help="ALFWorld 数据目录（默认 ~/.cache/alfworld 或 ALFWORLD_DATA）")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="交互环境步数预算（默认 50；ours 与 baseline 对称生效）")
    parser.add_argument("--resume", action="store_true",
                        help="复用 output-dir 下最近的运行目录，跳过已完成条件")
    parser.add_argument("--run-dir", default=None,
                        help="精确复用指定运行目录（不新建时间戳目录）")
    parser.add_argument("--fresh-conditions", action="store_true",
                        help="重跑所选条件前将旧 condition data 移入可恢复备份")
    args = parser.parse_args()

    limit = args.limit if args.limit is not None else DEFAULT_LIMITS[args.benchmark]
    task_type = args.task_type or (
        "pick_heat_then_place_in_recep" if args.benchmark == "alfworld" else None)
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
          + (f", max_steps={config.max_steps}" if args.benchmark == "alfworld" else "")
          + "）")
    print(f"条件：{args.conditions}   mock={args.mock}")
    print("=" * 78)

    adapter = make_adapter(args.benchmark, config,
                           task_type=task_type, alfworld_data=args.alfworld_data,
                           max_steps=config.max_steps,
                           kinds=("code", "math", "env"))
    tasks = adapter.load_tasks(limit=limit, task_type=task_type)
    if not tasks:
        print(f"[错误] 未加载到任务。请检查数据可用性（HF 网络 / ALFWorld 数据目录）。")
        return 1
    print(f"已加载任务 {len(tasks)} 个：{tasks[0].task_id} ... {tasks[-1].task_id}")

    # mock 模式下跳过 baseline（FlowEvo 原版无 mock LLM）；toy 无 baseline 映射
    skip_baseline = args.mock or args.benchmark == "toy"
    conditions = [c for c in args.conditions
                  if not (skip_baseline and c in ("baseline_dynamic", "flowevo"))]
    if skip_baseline and conditions != args.conditions:
        print(f"[提示] {'mock 模式' if args.mock else 'toy benchmark'} 跳过 baseline 条件"
              f"（原版 FlowEvo 需要真实 API / 不支持 toy）。")

    # --resume：跳过该运行目录中已完成的条件
    results_path = run_dir / "results.json"
    if args.resume and results_path.exists():
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
        max_steps=config.max_steps if args.benchmark == "alfworld" else None,
        mock_script=mock_script,
        initial_results=prior_results,
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
