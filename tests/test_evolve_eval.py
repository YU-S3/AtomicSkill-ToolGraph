"""冻结 Skill 进化效果评估测试（toy，无 API）。"""

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atomic_skillgraph.adapters.mock_llm import MockLLM  # noqa: E402
from atomic_skillgraph.adapters.toy_benchmarks import (  # noqa: E402
    ToyAdapter,
    build_mock_script,
    toy_tasks,
)
from atomic_skillgraph.core.config import SystemConfig  # noqa: E402
from atomic_skillgraph.system import AtomicSkillGraphSystem  # noqa: E402
from atomic_skillgraph.tools.sandbox import Sandbox  # noqa: E402
from experiments.common import balanced_task_subset  # noqa: E402

TASK_ORDER = ["toy_code_double", "toy_code_triple", "toy_code_double_variant",
              "toy_code_square", "toy_math_add", "toy_math_mul", "toy_math_div",
              "toy_env_pick_place", "toy_env_heat", "toy_env_clean",
              "toy_env_heat2", "toy_env_heat3"]


def test_balanced_task_subset_is_equal_and_round_robin():
    tasks = [SimpleNamespace(task_id=f"a{i}", task_type="a") for i in range(3)]
    tasks += [SimpleNamespace(task_id=f"b{i}", task_type="b") for i in range(3)]
    selected = balanced_task_subset(tasks, ["a", "b"], 2)
    assert [task.task_id for task in selected] == ["a0", "b0", "a1", "b1"]


def _build_system(data_dir: Path, freeze: bool) -> AtomicSkillGraphSystem:
    config = SystemConfig(data_dir=data_dir, seed=42, maintenance_interval=1,
                          freeze_skills=freeze)
    config = dataclasses.replace(config, llm=dataclasses.replace(config.llm, mock=True))
    config = dataclasses.replace(config, thresholds=dataclasses.replace(
        config.thresholds, direct_min_utility=0.3, direct_min_success=1,
        candidate_min_support=2, insight_min_samples=2))
    sandbox = Sandbox(tmp_root=str(data_dir / "sandbox_tmp"))
    llm = MockLLM(script=build_mock_script())
    adapter = ToyAdapter(kinds=("code", "math", "env"), sandbox=sandbox)
    return AtomicSkillGraphSystem(config, adapter, llm)


def test_frozen_eval(workspace_tmp):
    tasks = toy_tasks(("code", "math", "env"))
    by_id = {t.task_id: t for t in tasks}
    ordered = [by_id[tid] for tid in TASK_ORDER]

    # 第一阶段：在线进化
    system = _build_system(workspace_tmp / "online", freeze=False)
    for task in ordered:
        system.run_task(task)
    frozen_skill_stats = system.registry.stats()
    frozen_tool_stats = system.tool_registry.stats()
    assert frozen_skill_stats["nodes"] >= 10, "在线进化应产出技能"

    # 第二阶段：冻结重放（同一数据目录，只读模式）
    frozen_system = _build_system(workspace_tmp / "online", freeze=True)
    episodes = [frozen_system.run_task(task) for task in ordered]
    assert all(e["success"] for e in episodes), \
        [e["task_id"] for e in episodes if not e["success"]]
    # 冻结技能被复用
    assert any(e["direct_reuse_count"] > 0 or e["seeded_generation_count"] > 0
               for e in episodes), episodes
    # 所有 episode 的进化字段必须是 frozen_eval（无写入）
    for e in episodes:
        assert e["evolution"] == {"frozen_eval": True}, e["evolution"]
    # 技能库在评估后完全不变
    assert frozen_system.registry.stats() == frozen_skill_stats
    assert frozen_system.tool_registry.stats() == frozen_tool_stats


def test_frozen_bank_unchanged_after_second_eval(workspace_tmp):
    """连续两次冻结评估之间技能库也不应变化。"""
    tasks = toy_tasks(("code", "math", "env"))
    by_id = {t.task_id: t for t in tasks}
    ordered = [by_id[tid] for tid in TASK_ORDER]

    online = _build_system(workspace_tmp / "online", freeze=False)
    for task in ordered:
        online.run_task(task)

    eval1 = _build_system(workspace_tmp / "online", freeze=True)
    for task in ordered:
        eval1.run_task(task)
    after_first = (eval1.registry.stats(), eval1.tool_registry.stats())

    eval2 = _build_system(workspace_tmp / "online", freeze=True)
    for task in ordered:
        eval2.run_task(task)
    after_second = (eval2.registry.stats(), eval2.tool_registry.stats())
    assert after_first == after_second
