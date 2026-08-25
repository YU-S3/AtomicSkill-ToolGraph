"""Stage 2：无 API 但走完整实验链路 smoke（Mock LLM + 合成任务）。

覆盖完整闭环：任务 → 检索/规划 → direct/seeded/dynamic 路由 → 验证 →
Trace → 原子化 → Tool admission → SkillGraph/ToolRepo 更新 → Composite →
Layer-3 insight → 全局泛化 → 生命周期治理 → 指标 → 报告。

用法：
    python -m experiments.run_smoke_fullchain
    python -m experiments.run_smoke_fullchain --output-dir runs/smoke_demo
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atomic_skillgraph.adapters.mock_llm import MockLLM  # noqa: E402
from atomic_skillgraph.adapters.toy_benchmarks import (  # noqa: E402
    ToyAdapter,
    build_mock_script,
    toy_tasks,
)
from atomic_skillgraph.core.config import SystemConfig  # noqa: E402
from atomic_skillgraph.system import AtomicSkillGraphSystem  # noqa: E402
from experiments.report import (  # noqa: E402
    aggregate_results,
    save_aggregated,
    write_markdown_report,
)

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "runs" / "smoke_fullchain"


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2: full-chain smoke (no API)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 任务序列：先冷启动学习，再同类任务验证复用/泛化（覆盖 code/math/env）
    tasks = toy_tasks(("code", "math", "env"))
    order = ["toy_code_double", "toy_code_triple", "toy_code_double_variant",
             "toy_code_square", "toy_math_add", "toy_math_mul", "toy_math_div",
             "toy_env_pick_place", "toy_env_heat", "toy_env_clean", "toy_env_heat2",
             "toy_env_heat3"]
    by_id = {t.task_id: t for t in tasks}
    ordered = [by_id[tid] for tid in order if tid in by_id]

    config = SystemConfig(
        data_dir=run_dir / "data",
        seed=args.seed,
        maintenance_interval=1,     # smoke 中每次成功都触发维护以覆盖全部生命周期路径
    )
    config = dataclasses.replace(config, llm=dataclasses.replace(config.llm, mock=True))
    # smoke 放宽 direct 门槛，保证 direct 路由被覆盖（阈值均配置化）
    config = dataclasses.replace(config, thresholds=dataclasses.replace(
        config.thresholds,
        direct_min_utility=0.3, direct_min_success=1,
        candidate_min_support=2, insight_min_samples=2,
    ))
    from atomic_skillgraph.tools.sandbox import Sandbox
    sandbox = Sandbox(tmp_root=str(run_dir / "sandbox_tmp"))
    llm = MockLLM(script=build_mock_script())
    adapter = ToyAdapter(kinds=("code", "math", "env"), sandbox=sandbox)
    system = AtomicSkillGraphSystem(config, adapter, llm)

    print("=" * 78)
    print("Stage 2 无 API 完整实验链路 smoke（Mock LLM + 合成任务）")
    print("=" * 78)
    episodes = []
    for task in ordered:
        started = time.perf_counter()
        episode = system.run_task(task)
        mode = ("DIRECT" if episode["direct_reuse_count"] else
                "SEEDED" if episode["seeded_generation_count"] else "DYNAMIC")
        print(f"  ep{episode['episode']:>2} {task.task_id:<26} "
              f"{'OK ' if episode['success'] else 'FAIL'} "
              f"mode={mode:<8} start={episode['start_mode']:<5} "
              f"tokens={episode['tokens']:>4} "
              f"skills={episode['skill_graph']['nodes']:>2} "
              f"tools={episode['tool_repo']['tools']:>2} "
              f"({(time.perf_counter() - started) * 1000:.0f}ms)")
        episodes.append(episode)

    aggregated = aggregate_results({"atomic_skillgraph_full": {"kind": "ours",
                                                               "episodes": episodes}})
    save_aggregated(aggregated, run_dir / "aggregated.json")
    write_markdown_report(aggregated, run_dir / "report.md",
                          title="Stage 2 无 API 完整链路 smoke 报告")

    summary = aggregated["atomic_skillgraph_full"]
    print("-" * 78)
    print(f"成功率 {summary['success_rate'] * 100:.0f}% "
          f"（{summary['num_passed']}/{summary['num_tasks']}），"
          f"direct {summary['direct_episode_rate'] * 100:.0f}% / "
          f"seeded {summary['seeded_episode_rate'] * 100:.0f}% / "
          f"dynamic {summary['dynamic_episode_rate'] * 100:.0f}%")
    print(f"原子复用率 {summary['atomic_reuse_rate'] * 100:.0f}%，"
          f"跨类型复用 {summary['cross_task_type_reuse_episodes']} 次，"
          f"admission 通过 {summary['admission_pass']} / shadow {summary['admission_shadow']}")
    print(f"最终 SkillGraph：{summary['final_skill_graph']}")
    print(f"最终 Tool 库：{summary['final_tool_repo']}")
    print(f"产物目录：{run_dir}（data/ 含 skill_graph、tools、traces、metrics）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
