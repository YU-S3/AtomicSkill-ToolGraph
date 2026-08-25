"""Stage 4：完整实验（全量数据 + 核心条件 + 消融）。

    python -m experiments.run_full --benchmark humaneval --config-path configs/default.yaml
    python -m experiments.run_full --benchmark gsm8k --config-path configs/default.yaml
    python -m experiments.run_full --benchmark alfworld --config-path configs/default.yaml
    python -m experiments.run_full --all --conditions baseline_dynamic flowevo atomic_skillgraph_full ...

--resume：跳过已有结果的 benchmark/条件（基于各 benchmark 目录的 results.json）。
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
    OUR_CONDITIONS,
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

BENCHMARKS = {"humaneval": 0, "gsm8k": 0, "alfworld": 0}  # limit=0 → 全量
DEFAULT_FULL_CONDITIONS = ["baseline_dynamic", "flowevo", "atomic_graph_only",
                           "tool_repo_only", "atomic_skillgraph_full"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 4: 完整实验")
    parser.add_argument("--benchmark", choices=list(BENCHMARKS.keys()), default=None,
                        help="单个 benchmark；不指定且 --all 时跑全部")
    parser.add_argument("--all", action="store_true", help="运行全部 benchmark")
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_FULL_CONDITIONS)
    parser.add_argument("--ablations", action="store_true",
                        help="追加全部消融条件（§57.3）")
    parser.add_argument("--config-path", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "runs" / "full"))
    parser.add_argument("--task-type", default=None, help="ALFWorld 任务类型过滤")
    parser.add_argument("--limit", type=int, default=0, help="覆盖默认全量（调试用）")
    parser.add_argument("--resume", action="store_true", help="跳过已有结果的 benchmark/条件")
    parser.add_argument("--alfworld-data", default=None)
    args = parser.parse_args()

    if args.benchmark:
        benchmarks = [args.benchmark]
    elif args.all:
        benchmarks = list(BENCHMARKS.keys())
    else:
        parser.error("需要 --benchmark 或 --all")

    conditions = list(args.conditions)
    if args.ablations:
        for name in OUR_CONDITIONS:
            if name not in conditions and not name.startswith(("full", "task_type")):
                conditions.append(name)
        conditions.extend([c for c in OUR_CONDITIONS
                           if c.startswith(("full", "task_type")) and c not in conditions])

    config = load_conda_config(args.config_path)
    output_dir = Path(args.output_dir)
    combined: dict[str, dict] = {}

    for benchmark in benchmarks:
        benchmark_dir = output_dir / benchmark
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        results_path = benchmark_dir / "results.json"
        done_conditions: set[str] = set()
        if args.resume and results_path.exists():
            done_conditions = set(json.loads(results_path.read_text(encoding="utf-8")).keys())
        remaining = [c for c in conditions if c not in done_conditions]
        if not remaining:
            print(f"[skip] {benchmark}：所有条件已完成（--resume）")
            continue

        print("=" * 78)
        print(f"Stage 4 完整实验：{benchmark}，条件：{remaining}")
        print("=" * 78)
        limit = args.limit if args.limit > 0 else BENCHMARKS[benchmark]
        adapter = make_adapter(benchmark, config,
                               task_type=args.task_type, alfworld_data=args.alfworld_data)
        tasks = adapter.load_tasks(limit=limit, task_type=args.task_type)
        print(f"已加载任务 {len(tasks)} 个")
        if not tasks:
            print("[错误] 未加载到任务")
            continue

        results = run_conditions(
            conditions=remaining,
            benchmark=benchmark,
            config=config,
            adapter=adapter,
            tasks=tasks,
            output_dir=benchmark_dir,
            limit=limit,
            config_path=str(PROJECT_ROOT / "configs" / "flowevo_default.yaml"),
            task_type=args.task_type,
        )
        if args.resume and results_path.exists():
            prior = json.loads(results_path.read_text(encoding="utf-8"))
            prior.update({k: v for k, v in results.items()})
            results = prior
        aggregated = aggregate_results(results)
        save_aggregated(aggregated, benchmark_dir / "aggregated.json")
        write_markdown_report(aggregated, benchmark_dir / "report.md",
                              title=f"Stage 4 完整实验报告（{benchmark}）")
        combined[benchmark] = aggregated

    if combined:
        save_aggregated(combined, output_dir / "combined_aggregated.json")
        print("\n" + "-" * 78)
        print("汇总（success_rate）：")
        for benchmark, aggregated in combined.items():
            for name, value in aggregated.items():
                rate = value.get("success_rate")
                print(f"  {benchmark:<12} {name:<30} {rate * 100:6.1f}%" if rate is not None else
                      f"  {benchmark:<12} {name:<30}  N/A")
        print(f"\n产物目录：{output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
