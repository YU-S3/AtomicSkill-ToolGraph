"""Stage-2 完整链路 smoke（Mock LLM，无 API）的 pytest 版本。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dataclasses  # noqa: E402

import pytest  # noqa: E402

from atomic_skillgraph.adapters.mock_llm import MockLLM  # noqa: E402
from atomic_skillgraph.adapters.toy_benchmarks import (  # noqa: E402
    ToyAdapter,
    build_mock_script,
    toy_tasks,
)
from atomic_skillgraph.core.config import SystemConfig  # noqa: E402
from atomic_skillgraph.system import AtomicSkillGraphSystem  # noqa: E402
from atomic_skillgraph.tools.sandbox import Sandbox  # noqa: E402

TASK_ORDER = ["toy_code_double", "toy_code_triple", "toy_code_double_variant",
              "toy_code_square", "toy_math_add", "toy_math_mul", "toy_math_div",
              "toy_env_pick_place", "toy_env_heat", "toy_env_clean",
              "toy_env_heat2", "toy_env_heat3"]


def _build_system(data_dir: Path) -> AtomicSkillGraphSystem:
    config = SystemConfig(data_dir=data_dir, seed=42, maintenance_interval=1)
    config = dataclasses.replace(config, llm=dataclasses.replace(config.llm, mock=True))
    config = dataclasses.replace(config, thresholds=dataclasses.replace(
        config.thresholds, direct_min_utility=0.3, direct_min_success=1,
        candidate_min_support=2, insight_min_samples=2))
    sandbox = Sandbox(tmp_root=str(data_dir / "sandbox_tmp"))
    llm = MockLLM(script=build_mock_script())
    adapter = ToyAdapter(kinds=("code", "math", "env"), sandbox=sandbox)
    return AtomicSkillGraphSystem(config, adapter, llm)


def test_fullchain_smoke(workspace_tmp):
    tasks = toy_tasks(("code", "math", "env"))
    by_id = {t.task_id: t for t in tasks}
    system = _build_system(workspace_tmp / "data")
    episodes = [system.run_task(by_id[tid]) for tid in TASK_ORDER]
    assert all(e["success"] for e in episodes), \
        [e["task_id"] for e in episodes if not e["success"]]
    # 路由覆盖：cold dynamic / warm seeded / warm direct
    modes = {e["direct_reuse_count"] for e in episodes}
    route_audit = [
        (e["task_id"], e["start_mode"], e["planning_mode"],
         e["direct_reuse_count"], e["seeded_generation_count"],
         e["dynamic_generation_count"], e["planned_node_count"],
         e["reused_skill_refs"], e["used_tool_refs"],
         [
             {
                 "ref": node.get("ref"),
                 "params": node.get("params"),
                 "attempts": node.get("attempts"),
                 "fallback_reason": node.get("fallback_reason"),
             }
             for node in (system.trace_store.load(e["trace_id"]).realized_atomic_nodes or [])
         ])
        for e in episodes
    ]
    route_report = "\n".join(map(str, route_audit))
    assert any(system.trace_store.load(e["trace_id"]).metrics.get(
        "direct_attempt_count", 0) > 0 for e in episodes), route_report
    assert any(e["seeded_generation_count"] > 0 for e in episodes), route_report
    # 知识增长
    final = system.stats()
    assert final["skill_graph"]["nodes"] >= 10
    assert final["tool_repo"]["tools"] >= 5
    # 三类节点齐备
    by_kind = final["skill_graph"]["by_kind"]
    assert by_kind.get("abstract_atomic", 0) >= 4
    assert by_kind.get("implementation_atomic", 0) >= 3
    assert by_kind.get("composite", 0) >= 2

    # Every online LLM call is attributed to a stable agent bucket.  Composite
    # construction itself is deterministic; composite_agent is specifically
    # the semantic graph-proposal call that precedes code validation/building.
    required_buckets = {
        "planner_agent", "runtime_agent", "extractor_agent",
        "composite_agent", "evolution_repair_agent", "episode_total",
    }
    for episode in episodes:
        usage = episode["llm_usage_by_agent"]
        assert required_buckets <= set(usage)
        attributed = sum(
            int(details.get("total_tokens", 0) or 0)
            for name, details in usage.items()
            if name not in {"episode_total"})
        assert attributed == int(usage["episode_total"]["total_tokens"])
        assert episode["tokens"] == int(usage["episode_total"]["total_tokens"])
    assert any(e["llm_usage_by_agent"]["runtime_agent"]["call_count"] > 0
               for e in episodes)
    assert any(e["llm_usage_by_agent"]["extractor_agent"]["call_count"] > 0
               for e in episodes)
    assert any(e["llm_usage_by_agent"]["composite_agent"]["call_count"] > 0
               for e in episodes)


def test_toy_world_protocol():
    tasks = toy_tasks(("env",))
    task = tasks[0]
    from atomic_skillgraph.adapters.toy_benchmarks import ToyWorld
    world = ToyWorld(dict(task.context))
    result = world.step("go to countertop 1")
    assert result.accepted
    result = world.step("take mug 1 from countertop 1")
    assert result.accepted and "mug_1" in result.state["inventory"]
    result = world.step("go to shelf 1")
    result = world.step("put mug 1 in/on shelf 1")
    assert result.won and result.done
