import json
from types import SimpleNamespace

from atomic_skillgraph.adapters.benchmark import Task
from atomic_skillgraph.core.config import SystemConfig
from atomic_skillgraph.core.refs import SkillRef, ToolRef
from atomic_skillgraph.core.status import ExecutionMode
from atomic_skillgraph.core.trace_ir import (
    NodeExecutionStatus,
    NodeValidationResult,
    TraceRecord,
)
from atomic_skillgraph.runtime.runtime_graph import PlannedNode, RuntimeGraph, RuntimePlan
from atomic_skillgraph.system import (
    AtomicSkillGraphSystem,
    _append_planned_runtime_spans,
    _runtime_node_metrics,
)
from experiments import common as experiment_common
from experiments import run_evolve_eval
from experiments.common import restore_system_episode_counters
from experiments.report import summarize_episodes


def _plan(count: int = 1) -> RuntimePlan:
    return RuntimePlan(
        start_mode="warm",
        nodes=[
            PlannedNode(
                ref=SkillRef(f"generic.capability_{index}", "1.0.0"),
                step_id=f"step_{index:03d}",
                occurrence_id=f"occ_{index:03d}",
            )
            for index in range(count)
        ],
    )


def test_occurrence_span_unions_all_started_fallback_attempts():
    graph = RuntimeGraph("task", _plan())
    node = graph.nodes[0]
    node.execution_status = NodeExecutionStatus.EXECUTED_SUCCESS
    node.executed_action_count = 7
    node.attempts = [
        {
            "mode": "direct", "started": True, "passed": False,
            "action_start": 1, "action_end": 4, "action_count": 3,
            "failure_type": "effect_not_met",
        },
        {
            "mode": "seeded", "started": True, "passed": True,
            "action_start": 4, "action_end": 8, "action_count": 4,
        },
        {
            "mode": "dynamic", "started": False, "passed": False,
            "action_start": 8, "action_end": 8, "action_count": 0,
            "failure_type": "attempt_not_started",
        },
    ]
    trace = TraceRecord(task_id="task")

    _append_planned_runtime_spans(trace, graph)

    assert len(trace.runtime_spans) == 1
    span = trace.runtime_spans[0]
    assert (span.action_start, span.action_end) == (1, 8)
    assert [item["mode"] for item in span.metadata["attempt_spans"]] == [
        "direct", "seeded"]
    assert [item["passed"] for item in span.metadata["attempt_spans"]] == [
        False, True]


def test_runtime_metrics_require_real_actions_and_count_satisfied_as_completed():
    graph = RuntimeGraph("task", _plan(3))
    not_started, satisfied, executed = graph.nodes
    not_started.impl_ref = "skill://impl.selected_only@1.0.0"
    not_started.attempts = [{
        "mode": "direct", "started": False, "passed": False,
        "action_start": 0, "action_end": 0, "action_count": 0,
    }]
    satisfied.execution_status = NodeExecutionStatus.ALREADY_SATISFIED
    satisfied.passed = True
    executed.execution_status = NodeExecutionStatus.EXECUTED_SUCCESS
    executed.executed_action_count = 2
    executed.mode = ExecutionMode.SEEDED
    executed.attempts = [{
        "mode": "seeded", "started": True, "passed": True,
        "action_start": 0, "action_end": 2, "action_count": 2,
    }]

    metrics = _runtime_node_metrics(graph)

    assert metrics["executed_node_count"] == 1
    assert metrics["already_satisfied_node_count"] == 1
    assert metrics["completed_node_count"] == 2
    assert metrics["not_started_node_count"] == 1
    assert metrics["node_mode_counts"] == {"seeded": 1}
    assert metrics["selected_skill_refs"] == [
        str(item.ref) for item in graph.plan.nodes]
    assert metrics["executed_skill_refs"] == [str(graph.plan.nodes[2].ref)]
    assert metrics["successful_reused_skill_refs"] == [
        str(graph.plan.nodes[2].ref)]
    assert metrics["successful_tool_refs"] == []


def test_successful_tool_reuse_excludes_failed_direct_attempts():
    graph = RuntimeGraph("task", _plan(2))
    failed_then_seeded, direct_success = graph.nodes
    failed_then_seeded.execution_status = NodeExecutionStatus.EXECUTED_SUCCESS
    failed_then_seeded.executed_action_count = 2
    failed_then_seeded.attempts = [
        {"mode": "direct", "started": True, "passed": False,
         "tool_refs": ["tool://generic.failed@1.0.0"]},
        {"mode": "seeded", "started": True, "passed": True,
         "tool_refs": []},
    ]
    direct_success.execution_status = NodeExecutionStatus.EXECUTED_SUCCESS
    direct_success.executed_action_count = 1
    direct_success.attempts = [
        {"mode": "direct", "started": True, "passed": True,
         "tool_refs": ["tool://generic.success@1.0.0"]},
    ]

    metrics = _runtime_node_metrics(graph)
    assert metrics["successful_tool_refs"] == [
        "tool://generic.success@1.0.0"]


def test_code_direct_execution_sets_explicit_execution_status():
    atomic_ref = SkillRef("generic.code_capability", "1.0.0")
    tool_ref = ToolRef("generic.code_tool", "1.0.0")
    atomic = SimpleNamespace(ref=atomic_ref, outputs=[])
    implementation = SimpleNamespace(
        ref=SkillRef("impl.generic.code_capability", "1.0.0"),
        tool_bindings=[SimpleNamespace(tool_ref=tool_ref)])
    gate = SimpleNamespace(eligible=True, reason="")
    bridge = SimpleNamespace(
        direct_gate=lambda *_a, **_k: gate,
        execute_direct=lambda *_a, **_k: {
            "passed": True, "feedback": {}, "after": {}},
    )
    system = object.__new__(AtomicSkillGraphSystem)
    system.registry = SimpleNamespace(get=lambda _ref: atomic)
    system.selector = SimpleNamespace(select=lambda *_a, **_k:
                                      SimpleNamespace(implementation=implementation))
    system.resolver = SimpleNamespace(resolve=lambda *_a, **_k: [])
    system.bridge = bridge
    system.node_validator = SimpleNamespace(validate_atomic=lambda *_a, **_k:
                                             NodeValidationResult(
                                                 level="atomic", passed=True))
    system._record_tool_feedback = lambda *_a, **_k: None
    system._record_impl_feedback = lambda *_a, **_k: None
    task = Task(
        task_id="code_task", benchmark="humaneval", task_type="code",
        goal="return the expected value", state={})
    plan = RuntimePlan(
        start_mode="warm",
        nodes=[PlannedNode(ref=atomic_ref, step_id="step_000",
                           occurrence_id="occ_000")])

    trace = system._run_code_task(task, plan)

    node = system._last_runtime_graph.nodes[0]
    assert trace.success is True
    assert node.execution_status == NodeExecutionStatus.EXECUTED_SUCCESS
    assert _runtime_node_metrics(system._last_runtime_graph)[
        "executed_node_count"] == 1


def test_report_uses_all_tasks_for_first_attempt_and_realized_reuse():
    episodes = [
        {
            "episode": 1, "success": True, "retries": 0,
            "planned_node_count": 2, "executed_node_count": 1,
            "already_satisfied_node_count": 1, "completed_node_count": 2,
            "goal_terminal_before_plan_complete": True,
            "node_mode_counts": {"direct": 1},
            "successful_reused_skill_refs": ["skill://generic.a@1.0.0"],
            "successful_atomic_reuse_count": 1,
        },
        {
            "episode": 2, "success": True, "retries": 1,
            "retrieved_skill_refs": ["skill://generic.unused@1.0.0"],
            "successful_reused_skill_refs": [],
        },
        {
            "episode": 3, "success": False, "retries": 0,
            "successful_reused_skill_refs": [],
            "successful_tool_refs": ["tool://generic.failed@1.0.0"],
            "cross_task_type_tool_reuse": True,
        },
    ]

    summary = summarize_episodes(episodes)

    assert summary["first_attempt_success_rate"] == 0.3333
    assert summary["first_attempt_success_given_success"] == 0.5
    assert summary["atomic_reuse_rate"] == 0.3333
    assert summary["successful_atomic_reuse_count"] == 1
    assert summary["goal_early_terminal_episode_count"] == 0
    assert summary["goal_terminal_skipped_node_count"] == 0
    assert summary["successful_tool_reuse_episode_rate"] == 0.3333
    assert summary["cross_task_type_tool_reuse_rate"] == 1.0


class _StatsStore:
    def stats(self):
        return {}


class _CounterSystem:
    instances = []

    def __init__(self, *_args, **_kwargs):
        self.episode_count = 0
        self.success_count = 0
        self.registry = _StatsStore()
        self.tool_registry = _StatsStore()
        self.__class__.instances.append(self)

    def run_task(self, task):
        self.episode_count += 1
        success = bool(task.context.get("success", False))
        if success:
            self.success_count += 1
        return {
            "episode": self.episode_count,
            "task_id": task.task_id,
            "success": success,
        }


def _task(task_id: str, *, success: bool) -> Task:
    return Task(
        task_id=task_id, benchmark="toy", task_type="kind", goal="goal",
        context={"game_file": f"{task_id}.json", "success": success})


def test_restore_system_episode_counters():
    system = SimpleNamespace(episode_count=0, success_count=0)
    restore_system_episode_counters(
        system, [{"success": True}, {"success": False}, {"success": True}])
    assert system.episode_count == 3
    assert system.success_count == 2


def test_online_resume_appends_episode_numbers(monkeypatch, tmp_path):
    tasks = [_task("task_1", success=True), _task("task_2", success=False)]
    condition_dir = tmp_path / "online" / "atomic_graph_only"
    condition_dir.mkdir(parents=True)
    signature = [{
        "task_id": task.task_id,
        "task_type": task.task_type,
        "game_file": task.context["game_file"],
    } for task in tasks]
    (condition_dir / "online_progress.json").write_text(json.dumps({
        "condition": "atomic_graph_only",
        "task_signature": signature,
        "completed": 1,
        "episodes": [{"episode": 1, "task_id": "task_1", "success": True}],
    }), encoding="utf-8")
    _CounterSystem.instances.clear()
    monkeypatch.setattr(experiment_common, "AtomicSkillGraphSystem", _CounterSystem)
    monkeypatch.setattr(experiment_common, "make_llm", lambda *_a, **_k: object())

    result = experiment_common.run_our_condition(
        "atomic_graph_only", object(), SystemConfig(), tasks,
        output_dir=tmp_path / "online")

    assert [item["episode"] for item in result["episodes"]] == [1, 2]
    assert _CounterSystem.instances[-1].episode_count == 2
    assert _CounterSystem.instances[-1].success_count == 1


def test_frozen_resume_restores_system_counters(monkeypatch, tmp_path):
    condition = "atomic_graph_only"
    run_dir = tmp_path / "train"
    source_graph = run_dir / condition / "data" / "skill_graph"
    source_graph.mkdir(parents=True)
    (source_graph / "node.json").write_text("{}", encoding="utf-8")
    eval_dir = tmp_path / "eval"
    progress_path = eval_dir / condition / "frozen_progress.json"
    progress_path.parent.mkdir(parents=True)
    tasks = [_task("task_1", success=True), _task("task_2", success=False)]
    signature = [{
        "task_id": task.task_id,
        "task_type": task.task_type,
        "game_file": task.context["game_file"],
    } for task in tasks]
    progress_path.write_text(json.dumps({
        "condition": condition,
        "task_signature": signature,
        "completed": 1,
        "episodes": [{"episode": 1, "task_id": "task_1", "success": True}],
    }), encoding="utf-8")
    _CounterSystem.instances.clear()
    monkeypatch.setattr(run_evolve_eval, "AtomicSkillGraphSystem", _CounterSystem)
    monkeypatch.setattr(run_evolve_eval, "make_llm", lambda *_a, **_k: object())
    monkeypatch.setattr(run_evolve_eval, "validate_graph", lambda *_a, **_k:
                        SimpleNamespace(passed=True, errors=[],
                                        to_dict=lambda: {"passed": True}))

    result = run_evolve_eval.run_frozen_condition(
        condition, run_dir, eval_dir, SystemConfig(), object(), tasks)

    assert [item["episode"] for item in result["episodes"]] == [1, 2]
    assert _CounterSystem.instances[-1].episode_count == 2
    assert _CounterSystem.instances[-1].success_count == 1
