"""冻结 Skill 进化效果评估测试（toy，无 API）。"""

import dataclasses
import json
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
from experiments import common as experiment_common  # noqa: E402
from experiments.common import (  # noqa: E402
    balanced_task_subset,
    run_our_condition,
    run_balanced_baseline_condition,
)
from experiments.run_evolve_eval import _snapshot_frozen_bank  # noqa: E402
from experiments.run_small import (  # noqa: E402
    _clone_online_run,
    _condition_bank_digests,
)

TASK_ORDER = ["toy_code_double", "toy_code_triple", "toy_code_double_variant",
              "toy_code_square", "toy_math_add", "toy_math_mul", "toy_math_div",
              "toy_env_pick_place", "toy_env_heat", "toy_env_clean",
              "toy_env_heat2", "toy_env_heat3"]


def test_balanced_task_subset_is_equal_and_round_robin():
    tasks = [SimpleNamespace(task_id=f"a{i}", task_type="a") for i in range(3)]
    tasks += [SimpleNamespace(task_id=f"b{i}", task_type="b") for i in range(3)]
    selected = balanced_task_subset(tasks, ["a", "b"], 2)
    assert [task.task_id for task in selected] == ["a0", "b0", "a1", "b1"]


def test_balanced_baseline_runs_each_type_and_aggregates(monkeypatch, tmp_path):
    calls = []

    def fake_run(condition, benchmark, **kwargs):
        label = kwargs["task_type"]
        calls.append((label, kwargs["limit"], Path(kwargs["output_dir"])))
        passed = {"a": 1, "b": 2}[label]
        return {
            "subprocess": {"returncode": 0},
            "episodes": [{
                "num_tasks": 2,
                "num_passed": passed,
                "success_rate": passed / 2,
                "avg_tokens": {"a": 10.0, "b": 30.0}[label],
                "flowevo_summary": {"total": 2, "success": passed},
            }],
        }

    monkeypatch.setattr(experiment_common, "run_baseline_condition", fake_run)
    result = run_balanced_baseline_condition(
        "baseline_dynamic", "alfworld", config_path="unused.yaml",
        output_dir=tmp_path, task_types=["a", "b"], per_type_limit=2)
    assert [(label, limit) for label, limit, _ in calls] == [("a", 2), ("b", 2)]
    assert calls[0][2] != calls[1][2]
    episode = result["episodes"][0]
    assert episode["num_tasks"] == 4
    assert episode["num_passed"] == 3
    assert episode["success_rate"] == 0.75
    assert episode["avg_tokens"] == 20.0


def test_online_progress_can_extend_only_by_exact_prefix(workspace_tmp):
    tasks = toy_tasks(("code", "math", "env"))
    config = SystemConfig(data_dir=workspace_tmp / "unused", seed=42)
    config = dataclasses.replace(
        config, llm=dataclasses.replace(config.llm, mock=True))
    adapter = ToyAdapter(kinds=("code", "math", "env"),
                         sandbox=Sandbox(tmp_root=str(workspace_tmp / "sandbox")))
    output = workspace_tmp / "online_extension"
    first = run_our_condition(
        "atomic_graph_only", adapter, config, tasks[:2],
        mock_script=build_mock_script(), output_dir=output)
    assert len(first["episodes"]) == 2
    extended = run_our_condition(
        "atomic_graph_only", adapter, config, tasks[:3],
        mock_script=build_mock_script(), output_dir=output,
        allow_task_extension=True)
    assert len(extended["episodes"]) == 3
    progress = json.loads((output / "atomic_graph_only" /
                           "online_progress.json").read_text(encoding="utf-8"))
    assert progress["completed"] == 3
    assert len(progress["task_signature"]) == 3


def test_frozen_snapshot_rejects_later_online_milestone(workspace_tmp):
    source = workspace_tmp / "source"
    (source / "skill_graph").mkdir(parents=True)
    (source / "skill_graph" / "node.json").write_text(
        '{"version": 1}', encoding="utf-8")
    target = workspace_tmp / "eval" / "condition" / "data"
    first_digest = _snapshot_frozen_bank(source, target)
    assert first_digest
    assert (target / "skill_graph" / "node.json").exists()
    (source / "skill_graph" / "node.json").write_text(
        '{"version": 2}', encoding="utf-8")
    import pytest
    with pytest.raises(RuntimeError, match="另一在线里程碑"):
        _snapshot_frozen_bank(source, target)


def test_online_milestone_clone_is_independent_and_writable(workspace_tmp):
    source = workspace_tmp / "online_120"
    condition = "atomic_graph_only"
    data = source / condition / "data" / "skill_graph"
    data.mkdir(parents=True)
    (data / "node.json").write_text('{"utility": 0.5}', encoding="utf-8")
    signature = [{"task_id": "task_1", "task_type": "type_a",
                  "game_file": "game_1"}]
    (source / condition / "online_progress.json").write_text(json.dumps({
        "completed": 1, "task_signature": signature, "episodes": [{}],
    }), encoding="utf-8")
    (source / "task_manifest.json").write_text(json.dumps({
        "selection": {"task_count": 1}, "tasks": signature,
    }), encoding="utf-8")
    before = _condition_bank_digests(source, [condition])

    destination = workspace_tmp / "online_300"
    recorded = _clone_online_run(source, destination, [condition])
    assert recorded == before
    (destination / condition / "data" / "skill_graph" /
     "node.json").write_text('{"utility": 0.9}', encoding="utf-8")
    assert _condition_bank_digests(source, [condition]) == before
    assert _condition_bank_digests(destination, [condition]) != before
    lineage = json.loads((destination / "online_lineage.json").read_text(
        encoding="utf-8"))
    assert lineage["destination_freeze_skills"] is False


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
