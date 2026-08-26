"""指标汇总与报告生成（设计文档 v2.0 §58）。

Task-level: Success Rate / Late-run Success / First-attempt Success / Retry / 负迁移
Cost: Tokens per task / LLM calls / Direct / Seeded / Dynamic 频率
Atomic SkillGraph: Atomic Reuse Rate / Cross-Task-Type Reuse / Composite 复用
Tool Repository: 状态分布 / Admission Pass Rate / Candidate→Active / Generalize / Duplicate
Knowledge Growth: 节点/Tool 随 episode 增长曲线
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def summarize_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """我方条件的 episode 列表 → §58 指标。"""
    n = len(episodes)
    if n == 0:
        return {"num_tasks": 0}
    successes = sum(1 for e in episodes if e.get("success"))
    late_start = n // 2
    late = episodes[late_start:]
    late_successes = sum(1 for e in late if e.get("success"))
    tokens = [int(e.get("tokens", 0)) for e in episodes]
    direct_episodes = sum(1 for e in episodes if int(e.get("direct_reuse_count", 0)) > 0)
    seeded_episodes = sum(1 for e in episodes if int(e.get("seeded_generation_count", 0)) > 0)
    dynamic_episodes = sum(1 for e in episodes if int(e.get("dynamic_generation_count", 0)) > 0)
    reuse_episodes = sum(1 for e in episodes if e.get("reused_skill_refs"))
    cross_type = sum(1 for e in episodes if e.get("cross_task_type_reuse"))
    retries = sum(int(e.get("retries", 0)) for e in episodes)
    first_attempt = sum(1 for e in episodes
                        if e.get("success") and int(e.get("retries", 0)) == 0)
    graph_growth = [(e.get("episode"), (e.get("skill_graph") or {}).get("nodes", 0))
                    for e in episodes]
    tool_growth = [(e.get("episode"), (e.get("tool_repo") or {}).get("tools", 0))
                   for e in episodes]
    admission_pass = sum(
        int((e.get("evolution") or {}).get("success_processing", {}).get("admitted_tools", 0))
        for e in episodes)
    admission_shadow = sum(
        int((e.get("evolution") or {}).get("success_processing", {}).get("shadowed_tools", 0))
        for e in episodes)
    maintenance = sum(
        len((e.get("evolution") or {}).get("success_processing", {}).get("maintenance", []))
        for e in episodes)
    total_executed_nodes = sum(
        sum(int(value) for value in (e.get("node_mode_counts") or {}).values())
        for e in episodes)
    direct_nodes = sum(int((e.get("node_mode_counts") or {}).get("direct", 0))
                       for e in episodes)
    seeded_nodes = sum(int((e.get("node_mode_counts") or {}).get("seeded", 0))
                       for e in episodes)
    dynamic_nodes = sum(int((e.get("node_mode_counts") or {}).get("dynamic", 0))
                        for e in episodes)
    all_direct_episodes = sum(
        1 for e in episodes
        if int(e.get("planned_node_count", 0)) > 0
        and int(e.get("executed_node_count", 0)) == int(e.get("planned_node_count", 0))
        and int((e.get("node_mode_counts") or {}).get("direct", 0))
        == int(e.get("planned_node_count", 0)))
    all_executed_direct_episodes = sum(
        1 for e in episodes
        if int(e.get("executed_node_count", 0)) > 0
        and int((e.get("node_mode_counts") or {}).get("direct", 0))
        == int(e.get("executed_node_count", 0)))
    goal_early_terminal_episodes = sum(
        1 for e in episodes
        if bool(e.get("goal_terminal_before_plan_complete"))
        or (
            bool(e.get("success"))
            and int(e.get("planned_node_count", 0)) > 0
            and int(e.get("executed_node_count", 0))
            < int(e.get("planned_node_count", 0))
        )
    )
    goal_terminal_skipped_nodes = sum(
        max(0, int(e.get("planned_node_count", 0))
            - int(e.get("executed_node_count", 0)))
        for e in episodes
        if bool(e.get("goal_terminal_before_plan_complete"))
        or (
            bool(e.get("success"))
            and int(e.get("planned_node_count", 0)) > 0
            and int(e.get("executed_node_count", 0))
            < int(e.get("planned_node_count", 0))
        )
    )
    return {
        "num_tasks": n,
        "num_passed": successes,
        "success_rate": round(successes / n, 4),
        "late_run_success_rate": round(late_successes / max(len(late), 1), 4),
        "first_attempt_success_rate": round(first_attempt / max(successes, 1), 4),
        "avg_retries": round(retries / n, 3),
        "avg_tokens_per_task": round(sum(tokens) / n, 1),
        "total_tokens": sum(tokens),
        "direct_episode_rate": round(direct_episodes / n, 4),
        "any_direct_episode_rate": round(direct_episodes / n, 4),
        "all_nodes_direct_episode_rate": round(all_direct_episodes / n, 4),
        # Two deliberately different Direct denominators:
        # - executed: every node the benchmark allowed us to execute was Direct;
        # - full plan: every planned node was executed and Direct (strictest).
        "all_executed_nodes_direct_episode_rate": round(
            all_executed_direct_episodes / n, 4),
        "full_plan_direct_episode_rate": round(all_direct_episodes / n, 4),
        "goal_early_terminal_episode_count": goal_early_terminal_episodes,
        "goal_early_terminal_episode_rate": round(goal_early_terminal_episodes / n, 4),
        "goal_terminal_skipped_node_count": goal_terminal_skipped_nodes,
        "direct_node_rate": round(direct_nodes / max(total_executed_nodes, 1), 4),
        "seeded_node_rate": round(seeded_nodes / max(total_executed_nodes, 1), 4),
        "dynamic_node_rate": round(dynamic_nodes / max(total_executed_nodes, 1), 4),
        "executed_node_count": total_executed_nodes,
        "direct_node_count": direct_nodes,
        "seeded_node_count": seeded_nodes,
        "dynamic_node_count": dynamic_nodes,
        "seeded_episode_rate": round(seeded_episodes / n, 4),
        "dynamic_episode_rate": round(dynamic_episodes / n, 4),
        "atomic_reuse_rate": round(reuse_episodes / n, 4),
        "cross_task_type_reuse_rate": round(cross_type / max(reuse_episodes, 1), 4)
        if reuse_episodes else 0.0,
        "cross_task_type_reuse_episodes": cross_type,
        "admission_pass": admission_pass,
        "admission_shadow": admission_shadow,
        "admission_pass_rate": round(admission_pass / max(admission_pass + admission_shadow, 1), 4),
        "maintenance_actions": maintenance,
        "knowledge_growth_skills": graph_growth,
        "knowledge_growth_tools": tool_growth,
        "final_skill_graph": (episodes[-1].get("skill_graph") if episodes else {}),
        "final_tool_repo": (episodes[-1].get("tool_repo") if episodes else {}),
    }


def summarize_baseline(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        return {"num_tasks": 0}
    summary = episodes[0].get("flowevo_summary") or {}
    first = episodes[0]
    return {
        "num_tasks": int(first.get("num_tasks", 0)),
        "num_passed": int(first.get("num_passed", 0)),
        "success_rate": round(float(first.get("success_rate", 0.0)), 4),
        "avg_tokens_per_task": float(first.get("avg_tokens", 0.0)),
        "flowevo_summary": summary,
    }


def aggregate_results(results: dict[str, Any]) -> dict[str, Any]:
    aggregated: dict[str, Any] = {}
    for condition, result in results.items():
        episodes = result.get("episodes") or []
        if result.get("kind") == "flowevo_baseline":
            aggregated[condition] = {"kind": "flowevo_baseline",
                                     **summarize_baseline(episodes)}
        else:
            aggregated[condition] = {"kind": "ours",
                                     **summarize_episodes(episodes)}
    return aggregated


def write_markdown_report(aggregated: dict[str, Any], output_path: str | Path,
                          *, title: str = "AtomicSkillGraph v2.0 实验报告") -> None:
    lines = [f"# {title}", ""]
    our_conditions = [c for c, v in aggregated.items() if v.get("kind") == "ours"]
    baseline_conditions = [c for c, v in aggregated.items() if v.get("kind") == "flowevo_baseline"]
    table = ["| 条件 | 成功率 | Late-run | 首试成功 | Avg Tokens | Direct节点 | "
             "执行节点全Direct | 完整计划全Direct | 目标提前终止 | 含Seeded任务 | "
             "含Dynamic任务 | 原子复用率 | 跨类型复用率 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in baseline_conditions + our_conditions:
        v = aggregated[name]
        if v.get("kind") == "flowevo_baseline":
            table.append(
                f"| {name} (FlowEvo 原版) | {_pct(v.get('success_rate'))} | — | — | "
                f"{v.get('avg_tokens_per_task', 0):.0f} | — | — | — | — | — | — | — | — |")
        else:
            table.append(
                f"| {name} | {_pct(v.get('success_rate'))} | {_pct(v.get('late_run_success_rate'))} | "
                f"{_pct(v.get('first_attempt_success_rate'))} | {v.get('avg_tokens_per_task', 0):.0f} | "
                f"{_pct(v.get('direct_node_rate'))} | "
                f"{_pct(v.get('all_executed_nodes_direct_episode_rate'))} | "
                f"{_pct(v.get('full_plan_direct_episode_rate'))} | "
                f"{_pct(v.get('goal_early_terminal_episode_rate'))} | "
                f"{_pct(v.get('seeded_episode_rate'))} | "
                f"{_pct(v.get('dynamic_episode_rate'))} | {_pct(v.get('atomic_reuse_rate'))} | "
                f"{_pct(v.get('cross_task_type_reuse_rate'))} |")
    lines += table + [""]

    for name in our_conditions:
        v = aggregated[name]
        lines += [f"## {name}", "",
                  f"- 任务数：{v.get('num_tasks', 0)}，通过：{v.get('num_passed', 0)}"
                  f"（{_pct(v.get('success_rate'))}）",
                  f"- 平均重试：{v.get('avg_retries', 0)}，平均 tokens/task：{v.get('avg_tokens_per_task', 0)}",
                  f"- Direct 严格口径：节点 {v.get('direct_node_count', 0)} / "
                  f"{v.get('executed_node_count', 0)}（{_pct(v.get('direct_node_rate'))}）；"
                  f"实际执行节点全 Direct 任务 "
                  f"{_pct(v.get('all_executed_nodes_direct_episode_rate'))}；"
                  f"完整计划全 Direct 任务 {_pct(v.get('full_plan_direct_episode_rate'))}",
                  f"- Benchmark 目标提前终止："
                  f"{v.get('goal_early_terminal_episode_count', 0)} 个 episode "
                  f"（{_pct(v.get('goal_early_terminal_episode_rate'))}），"
                  f"因此跳过 {v.get('goal_terminal_skipped_node_count', 0)} 个规划节点",
                  f"- Admission：通过 {v.get('admission_pass', 0)} / shadow {v.get('admission_shadow', 0)}"
                  f"（通过率 {_pct(v.get('admission_pass_rate'))}）",
                  f"- 维护动作数：{v.get('maintenance_actions', 0)}",
                  f"- 最终 SkillGraph：{v.get('final_skill_graph', {})}",
                  f"- 最终 Tool Repository：{v.get('final_tool_repo', {})}",
                  *( [f"- 冻结库 SHA-256：`{v.get('frozen_bank_sha256')}`",
                      f"- 冻结前后完全一致：{v.get('bank_unchanged_after_eval')}",
                      f"- 冻结库图验证：{v.get('graph_validation', {})}"]
                     if v.get('frozen_bank_sha256') else [] ),
                  f"- 跨 task type 复用 episode 数：{v.get('cross_task_type_reuse_episodes', 0)}",
                  f"- 知识增长（skill 节点 / tool 数随 episode）：",
                  f"  `{v.get('knowledge_growth_skills')}`",
                  f"  `{v.get('knowledge_growth_tools')}`",
                  ""]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def _pct(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def save_aggregated(aggregated: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(aggregated, ensure_ascii=False, indent=2),
                          encoding="utf-8")
