from __future__ import annotations

from pathlib import Path

from atomic_skillgraph.adapters.benchmark import EnvRunResult, Task
from atomic_skillgraph.core.config import SystemConfig
from atomic_skillgraph.core.refs import SkillRef, ToolRef
from atomic_skillgraph.core.skill_ir import AbstractAtomicSkill, ImplementationAtom, ToolBinding
from atomic_skillgraph.core.status import ArtifactKind, ExecutionMode, SkillStatus, ToolLifecycle
from atomic_skillgraph.core.tool_ir import ToolAsset
from atomic_skillgraph.core.trace_ir import ActionRecord, TraceRecord
from atomic_skillgraph.evolution.branch_repair import FailureBranchManager
from atomic_skillgraph.evolution.failure_processor import FailureProcessor
from atomic_skillgraph.graph.registry import SkillGraphRegistry
from atomic_skillgraph.system import AtomicSkillGraphSystem
from atomic_skillgraph.tools.registry import ToolRegistry


class _ReplayAdapter:
    def replay_tool(self, tool, bindings, before):
        steps = list(tool.artifact.get("steps") or [])
        return {"passed": any("repair" in step for step in steps),
                "after": {"facts": ["object.done(apple_1)"]}}

    def run_env_episode(self, task, llm, **kwargs):
        node_ref = kwargs.get("node_ref", "")
        result = EnvRunResult(success=False, actions=[
            {"name": "open microwave 1", "mode": "seeded", "accepted": True,
             "node_ref": node_ref},
            {"name": "repair apple 1 with microwave 1", "mode": "seeded",
             "accepted": True, "node_ref": node_ref},
            {"name": "repair apple 1", "mode": "seeded", "accepted": True,
             "node_ref": node_ref},
        ], steps=3, atomic_complete=True)
        result.seeded_used = True
        result.dynamic_used = False
        return result


class _RejectedTemplateReplayAdapter(_ReplayAdapter):
    def run_env_episode(self, task, llm, **kwargs):
        node_ref = kwargs.get("node_ref", "")
        result = EnvRunResult(success=False, actions=[
            {"name": "repair apple 1", "mode": "seeded", "accepted": False,
             "node_ref": node_ref},
            {"name": "look", "mode": "seeded", "accepted": True,
             "node_ref": node_ref},
        ], steps=2, atomic_complete=True)
        result.seeded_used = True
        result.dynamic_used = False
        return result


class _CapturingReplayAdapter(_ReplayAdapter):
    def __init__(self):
        self.env_calls = []

    def run_env_episode(self, task, llm, **kwargs):
        self.env_calls.append(dict(kwargs))
        return super().run_env_episode(task, llm, **kwargs)


def _bank(root: Path, *, effect_predicate: str = "object.done"):
    registry = SkillGraphRegistry(root / "skill_graph")
    tools = ToolRegistry(root / "tools")
    atomic = AbstractAtomicSkill(
        ref=SkillRef("toy-env.repairable", "1.0.0"),
        summary="repairable atomic",
        inputs=[{"name": "object", "semantic_type": "object"}],
        effects=[{"predicate": effect_predicate,
                  "args": {"object": "$inputs.object"}}],
        metadata={"statistics": {"failure_count": 0, "utility": 0.5}},
        status=SkillStatus.ACTIVE,
    )
    registry.register(atomic)
    tool = ToolAsset(
        ref=ToolRef("toy-env.repairable", "1.0.0"),
        artifact_kind=ArtifactKind.ACTION_TEMPLATE,
        summary="bad original",
        signature={"parameters": [{"name": "object", "required": True}]},
        interface={"inputs": {"object": "object"}, "outputs": {}},
        artifact={"template": "bad {object}", "steps": ["bad {object}"]},
        tests=[{"kind": "replay", "bindings": {"object": "apple 1"},
                "before": {}, "after": {"facts": ["object.done(apple_1)"]}}],
        safety={"direct_execution_allowed": True},
        statistics={"support_count": 2, "call_count": 0, "success_count": 0,
                    "failure_count": 0, "utility": 0.5},
        status=ToolLifecycle.CANDIDATE,
    )
    tools.register(tool)
    tools.set_status(tool.ref, ToolLifecycle.ACTIVE)
    impl = ImplementationAtom(
        ref=SkillRef("impl.toy-env.repairable", "1.0.0"),
        abstract_ref=atomic.ref,
        tool_bindings=[ToolBinding(tool_ref=tool.ref, role="primary",
                                   parameter_mapping={"object": "$inputs.object"})],
        compatibility={"harness": "env"},
        status=SkillStatus.ACTIVE,
    )
    registry.register(impl)
    return registry, tools, atomic, tool


def _state_snapshots(*fact_sets: list[str]) -> list[dict]:
    return [{"step": index, "state": {"facts": list(facts)}}
            for index, facts in enumerate(fact_sets)]


def test_rescued_direct_failure_repairs_in_branch_without_crediting_old_tool(workspace_tmp):
    root = workspace_tmp / "data"
    registry, tools, atomic, old_tool = _bank(root)
    # The failed executable receives its own failure before any LLM rescue result.
    tools.record_feedback(old_tool.ref, False, usage_mode="direct")
    trace = TraceRecord(
        trace_id="trace_branch_repair", task_id="task_repair", task_type="toy",
        task_goal="repair apple", benchmark="toy_env", success=True,
        provenance={"env_index": 0},
        actions=[
            ActionRecord(step=0, name="bad apple 1", mode=ExecutionMode.DIRECT,
                         node_ref=str(atomic.ref), tool_ref=str(old_tool.ref)),
            ActionRecord(step=1, name="repair apple 1", mode=ExecutionMode.SEEDED,
                         node_ref=str(atomic.ref)),
        ],
        realized_atomic_nodes=[{
            "ref": str(atomic.ref), "params": {"object": "apple 1"},
            "tool_refs": [str(old_tool.ref)], "passed": True,
            "attempts": [
                {"mode": "direct", "passed": False,
                 "failure_type": "effect_not_met", "tool_refs": [str(old_tool.ref)],
                 "action_start": 0, "action_end": 1, "before": {}, "after": {}},
                {"mode": "seeded", "passed": True, "failure_type": "",
                 "tool_refs": [], "action_start": 1, "action_end": 2,
                 "before": {}, "after": {"facts": ["object.done(apple_1)"]}},
            ],
        }],
    )
    task = Task(task_id="task_repair", benchmark="toy_env", task_type="toy",
                goal="repair apple", context={"env_index": 0})
    manager = FailureBranchManager(root, registry, tools, _ReplayAdapter(),
                                   SystemConfig(data_dir=root))
    events = manager.process(trace, task)
    assert len(events) == 1
    assert events[0]["status"] == "merged"
    assert events[0]["merge_audit"]["seeded_fallback_allowed"] is False
    old_after = tools.get(old_tool.ref)
    assert old_after.statistics["failure_count"] == 1
    assert old_after.statistics.get("success_count", 0) == 0
    repaired = tools.get_recommended(old_tool.tool_id)
    assert repaired.ref != old_tool.ref
    assert repaired.statistics["direct_success_count"] == 1
    assert "repair {object}" in repaired.artifact.get("steps", [])
    impl = registry.get_recommended("impl.toy-env.repairable")
    assert impl.tool_bindings[0].tool_ref == repaired.ref
    assert (root / "evolution" / "branches" / "trace_branch_repair-0-0-direct"
            / "manifest.json").exists()


def test_atomic_failure_is_not_double_counted_as_task_evidence(workspace_tmp):
    root = workspace_tmp / "data"
    registry, tools, atomic, _tool = _bank(root)
    config = SystemConfig(data_dir=root)
    trace = TraceRecord(
        trace_id="trace_failure_once", task_id="failed", task_type="toy",
        benchmark="toy_env", success=False,
        retrieved_skill_refs=[str(atomic.ref), str(atomic.ref)],
        realized_atomic_nodes=[{
            "ref": str(atomic.ref), "passed": False,
            "validation": {"messages": ["effect missing"]},
        }],
    )
    FailureProcessor(registry, tools, config).process_failure(trace)
    shell = AtomicSkillGraphSystem.__new__(AtomicSkillGraphSystem)
    shell.config = config
    shell.registry = registry
    shell._update_skill_evidence(trace, success=False)
    saved = registry.get(atomic.ref)
    stats = saved.metadata["statistics"]
    assert stats["failure_count"] == 1
    assert stats["task_failure_count"] == 1
    assert stats["task_use_count"] == 1


def test_successful_atomic_is_not_penalized_by_downstream_task_failure(workspace_tmp):
    root = workspace_tmp / "node_local_evidence"
    registry, _tools, atomic, _tool = _bank(root)
    config = SystemConfig(data_dir=root)
    trace = TraceRecord(
        trace_id="trace_upstream_passed", task_id="failed_later",
        task_type="toy", benchmark="toy_env", success=False,
        realized_atomic_nodes=[{
            "ref": str(atomic.ref), "passed": True,
            "validation": {"messages": []},
        }],
    )
    shell = AtomicSkillGraphSystem.__new__(AtomicSkillGraphSystem)
    shell.config = config
    shell.registry = registry
    shell._update_skill_evidence(trace, success=False)
    stats = registry.get(atomic.ref).metadata["statistics"]
    assert stats["execution_success_count"] == 1
    assert stats.get("execution_failure_count", 0) == 0
    assert stats["task_failure_count"] == 1


def test_repeated_generator_failures_do_not_suppress_verified_contract(
        workspace_tmp):
    root = workspace_tmp / "contract_governance"
    registry, _tools, atomic, _tool = _bank(root)
    config = SystemConfig(data_dir=root)
    shell = AtomicSkillGraphSystem.__new__(AtomicSkillGraphSystem)
    shell.config = config
    shell.registry = registry

    for index in range(5):
        trace = TraceRecord(
            trace_id=f"generator_failure_{index}", task_id=f"failed_{index}",
            task_type="toy", benchmark="toy_env", success=False,
            realized_atomic_nodes=[{
                "ref": str(atomic.ref), "passed": False,
                "attempts": [{"mode": "seeded", "passed": False},
                             {"mode": "dynamic", "passed": False}],
            }],
        )
        shell._update_skill_evidence(trace, success=False)

    saved = registry.get(atomic.ref)
    assert saved.status == SkillStatus.ACTIVE
    assert saved.metadata["statistics"]["execution_failure_count"] == 5
    reviews = saved.metadata.get("governance_reviews") or []
    assert reviews
    assert reviews[-1]["decision"] == "retain_abstract_contract"


def test_seeded_failure_dynamic_rescue_evolves_atomic_in_isolated_branch(workspace_tmp):
    root = workspace_tmp / "data"
    registry, tools, atomic, _tool = _bank(root)
    trace = TraceRecord(
        trace_id="trace_atomic_repair", task_id="task_atomic", task_type="toy",
        task_goal="repair apple", benchmark="toy_env", success=True,
        actions=[ActionRecord(step=0, name="repair apple 1",
                              mode=ExecutionMode.DYNAMIC,
                              node_ref=str(atomic.ref))],
        realized_atomic_nodes=[{
            "ref": str(atomic.ref), "params": {"object": "apple 1"},
            "passed": True, "attempts": [
                {"mode": "seeded", "passed": False, "failure_type": "effect_not_met",
                 "action_start": 0, "action_end": 0, "before": {}, "after": {}},
                {"mode": "dynamic", "passed": True, "failure_type": "",
                 "action_start": 0, "action_end": 1, "before": {},
                 "after": {"facts": ["object.done(apple_1)"]}},
            ],
        }],
    )
    task = Task(task_id="task_atomic", benchmark="toy_env", task_type="toy",
                goal="repair apple")
    manager = FailureBranchManager(root, registry, tools, _ReplayAdapter(),
                                   SystemConfig(data_dir=root), llm=object())
    events = manager.process(trace, task)
    assert events[0]["status"] == "merged"
    assert events[0]["kind"] == "atomic_guideline_repair"
    evolved = registry.get_recommended(atomic.ref.logical_id)
    assert evolved.ref != atomic.ref
    assert any("参数化补救模板" in rule for rule in evolved.guideline_rules())
    assert any("$inputs.object" in rule for rule in evolved.guideline_rules())
    assert all("apple 1" not in rule for rule in evolved.guideline_rules())
    assert evolved.metadata["repair_generalized"] is True
    assert events[0]["merge_audit"]["dynamic_fallback_allowed"] is False
    assert evolved.metadata["repair_replay_trace_ids"] == ["trace_atomic_repair"]

    versions_after_first_repair = registry.list_versions(atomic.ref.logical_id)
    trace.trace_id = "trace_atomic_repair_second_evidence"
    repeated = manager.process(trace, task)[0]
    assert repeated["status"] == "evidence_updated"
    assert repeated["reason"] == "existing_parameterized_repair_revalidated"
    assert registry.list_versions(atomic.ref.logical_id) == versions_after_first_repair
    reused = registry.get_recommended(atomic.ref.logical_id)
    assert reused.ref == evolved.ref
    assert reused.metadata["repair_replay_count"] == 2
    assert reused.metadata["repair_replay_trace_ids"] == [
        "trace_atomic_repair", "trace_atomic_repair_second_evidence"]


def test_atomic_repair_guard_excludes_partially_bound_bystanders(workspace_tmp):
    """Run the formal repair pipeline, not only the guard helper in isolation."""
    root = workspace_tmp / "repair_guard_grounding"
    registry, tools, atomic, _tool = _bank(root)
    before = {"facts": [
        "object_at(apple_1, workbench_1)",
        "object_at(cup_2, workbench_1)",
        "object_exists(apple_1)",
        "object_exists(cup_2)",
    ]}
    trace = TraceRecord(
        trace_id="trace_grounded_guard", task_id="task_grounded_guard",
        task_type="toy", task_goal="repair apple", benchmark="toy_env",
        success=True,
        actions=[ActionRecord(step=0, name="repair apple 1",
                              mode=ExecutionMode.DYNAMIC,
                              node_ref=str(atomic.ref))],
        realized_atomic_nodes=[{
            "ref": str(atomic.ref),
            "params": {"object": "apple 1", "work_location": "workbench 1"},
            "passed": True,
            "attempts": [
                {"mode": "seeded", "passed": False,
                 "failure_type": "effect_not_met", "action_start": 0,
                 "action_end": 0, "before": before, "after": before},
                {"mode": "dynamic", "passed": True, "failure_type": "",
                 "action_start": 0, "action_end": 1, "before": before,
                 "after": {"facts": ["object.done(apple_1)"]}},
            ],
        }],
    )
    task = Task(task_id="task_grounded_guard", benchmark="toy_env",
                task_type="toy", goal="repair apple")
    manager = FailureBranchManager(root, registry, tools, _ReplayAdapter(),
                                   SystemConfig(data_dir=root), llm=object())

    event = manager.process(trace, task)[0]

    assert event["status"] == "merged"
    evolved = registry.get_recommended(atomic.ref.logical_id)
    guard = evolved.metadata["repair_guard"]["preconditions"]
    assert guard == [
        {"predicate": "object.at_location",
         "args": {"object": "$inputs.object",
                  "location": "$inputs.work_location"}},
        {"predicate": "object.exists",
         "args": {"object": "$inputs.object"}},
    ]
    assert "cup_2" not in str(guard)


def test_atomic_repair_parameterizes_entities_and_removes_state_cycles_from_evidence(
        workspace_tmp):
    root = workspace_tmp / "generalized_repair"
    registry, tools, atomic, _tool = _bank(
        root, effect_predicate="object.heated")
    actions = [
        "open microwave 1", "close microwave 1", "open microwave 1",
        "repair apple 1 with microwave 1",
    ]
    trace = TraceRecord(
        trace_id="trace_generalized_repair", task_id="task_generalized",
        task_type="toy", task_goal="repair apple", benchmark="toy_env", success=True,
        actions=[ActionRecord(step=index, name=name, mode=ExecutionMode.DYNAMIC,
                              node_ref=str(atomic.ref))
                 for index, name in enumerate(actions)],
        state_snapshots=_state_snapshots(
            [],
            ["container_open(microwave_1)"],
            [],
            ["container_open(microwave_1)"],
            ["container_open(microwave_1)", "object_heated(apple_1)"],
        ),
        realized_atomic_nodes=[{
            "ref": str(atomic.ref),
            "params": {"object": "apple 1", "heating_station": "microwave 1"},
            "passed": True,
            "attempts": [
                {"mode": "seeded", "passed": False,
                 "failure_type": "effect_not_met", "action_start": 0,
                 "action_end": 0, "before": {}, "after": {}},
                {"mode": "dynamic", "passed": True, "failure_type": "",
                 "action_start": 0, "action_end": 4, "before": {},
                 "after": {"facts": ["object_heated(apple_1)"]}},
            ],
        }],
    )
    task = Task(task_id="task_generalized", benchmark="toy_env",
                task_type="toy", goal="repair apple")
    manager = FailureBranchManager(root, registry, tools, _ReplayAdapter(),
                                   SystemConfig(data_dir=root), llm=object())
    event = manager.process(trace, task)[0]
    assert event["status"] == "merged"
    evolved = registry.get_recommended(atomic.ref.logical_id)
    template = evolved.metadata["repair_action_template"]
    assert template == [
        "open {heating_station}",
        "repair {object} with {heating_station}",
    ]
    rules = "\n".join(evolved.guideline_rules())
    assert "apple 1" not in rules and "microwave 1" not in rules
    assert "$inputs.object" in rules and "$inputs.heating_station" in rules
    assert "仅当常规执行失败且核心 Effect 尚未满足时" in rules


def test_atomic_repair_rejects_ambiguous_generic_entity_bindings(workspace_tmp):
    root = workspace_tmp / "ambiguous_repair"
    registry, tools, atomic, _tool = _bank(
        root, effect_predicate="object.heated")
    actions = ["repair apple 1", "repair apple 2"]
    trace = TraceRecord(
        trace_id="trace_ambiguous_repair", task_id="task_ambiguous",
        task_type="toy", task_goal="repair apple", benchmark="toy_env",
        success=True,
        actions=[ActionRecord(step=index, name=name,
                              params={"object": f"apple {index + 1}"},
                              mode=ExecutionMode.DYNAMIC,
                              node_ref=str(atomic.ref))
                 for index, name in enumerate(actions)],
        state_snapshots=_state_snapshots(
            [], ["object_heated(apple_1)"],
            ["object_heated(apple_1)", "object_heated(apple_2)"]),
        realized_atomic_nodes=[{
            "ref": str(atomic.ref), "params": {"object": "apple"},
            "passed": True,
            "attempts": [
                {"mode": "seeded", "passed": False,
                 "failure_type": "effect_not_met", "action_start": 0,
                 "action_end": 0, "before": {}, "after": {}},
                {"mode": "dynamic", "passed": True, "failure_type": "",
                 "action_start": 0, "action_end": 2, "before": {},
                 "after": {"facts": ["object_heated(apple_2)"]}},
            ],
        }],
    )
    task = Task(task_id="task_ambiguous", benchmark="toy_env",
                task_type="toy", goal="repair apple")
    adapter = _CapturingReplayAdapter()
    manager = FailureBranchManager(root, registry, tools, adapter,
                                   SystemConfig(data_dir=root), llm=object())
    event = manager.process(trace, task)[0]
    assert event["status"] == "awaiting_success_evidence"
    assert event["reason"] == "ambiguous_binding"
    assert event["ambiguous_parameter_roles"] == ["object"]
    assert event["binding_candidates"]["object"] == ["apple 1", "apple 2"]
    assert event["repair_event_slice"]["reason"] == "ambiguous_binding"
    assert event["repair_event_slice"]["event_slice_validated"] is False
    assert adapter.env_calls == []
    assert registry.list_versions(atomic.ref.logical_id) == [atomic.ref.version]


def test_atomic_repair_resolves_unique_occurrence_before_binding_core_effect(workspace_tmp):
    root = workspace_tmp / "unique_occurrence_atomic"
    registry, tools, atomic, _tool = _bank(
        root, effect_predicate="object.heated")
    trace = TraceRecord(
        trace_id="trace_unique_occurrence_atomic", task_id="task_unique_atomic",
        task_type="toy", task_goal="heat apple", benchmark="toy_env",
        success=True,
        actions=[ActionRecord(
            step=0, name="repair apple 1", params={"object": "apple 1"},
            mode=ExecutionMode.DYNAMIC, node_ref=str(atomic.ref))],
        state_snapshots=_state_snapshots([], ["object_heated(apple_1)"]),
        realized_atomic_nodes=[{
            "ref": str(atomic.ref), "params": {"object": "apple"},
            "passed": True,
            "attempts": [
                {"mode": "seeded", "passed": False,
                 "failure_type": "effect_not_met", "action_start": 0,
                 "action_end": 0, "before": {}, "after": {}},
                {"mode": "dynamic", "passed": True, "failure_type": "",
                 "action_start": 0, "action_end": 1, "before": {},
                 "after": {"facts": ["object_heated(apple_1)"]}},
            ],
        }],
    )
    task = Task(task_id="task_unique_atomic", benchmark="toy_env",
                task_type="toy", goal="heat apple")
    adapter = _CapturingReplayAdapter()
    manager = FailureBranchManager(root, registry, tools, adapter,
                                   SystemConfig(data_dir=root), llm=object())
    event = manager.process(trace, task)[0]
    assert event["status"] == "merged"
    assert event["replay_bindings"]["object"] == "apple 1"
    assert adapter.env_calls[0]["effect_inputs"]["object"] == "apple 1"
    assert event["repair_event_slice"]["event_slice_validated"] is True
    assert event["repair_event_slice"]["effect_producer_indices"] == [0]
    assert event["repair_event_slice"]["retained_event_indices"] == [0]
    evolved = registry.get_recommended(atomic.ref.logical_id)
    assert evolved.metadata["repair_action_template"] == ["repair {object}"]
    assert all("apple 1" not in rule for rule in evolved.guideline_rules())


def test_direct_tool_repair_rejects_ambiguous_rescue_before_slicing(workspace_tmp):
    root = workspace_tmp / "ambiguous_occurrence_tool"
    registry, tools, atomic, old_tool = _bank(
        root, effect_predicate="object.heated")
    trace = TraceRecord(
        trace_id="trace_ambiguous_occurrence_tool", task_id="task_tool_ambiguous",
        task_type="toy", task_goal="heat apple", benchmark="toy_env",
        success=True,
        actions=[
            ActionRecord(step=0, name="bad apple", mode=ExecutionMode.DIRECT,
                         node_ref=str(atomic.ref), tool_ref=str(old_tool.ref)),
            ActionRecord(step=1, name="repair apple 1",
                         params={"object": "apple 1"},
                         mode=ExecutionMode.DYNAMIC, node_ref=str(atomic.ref)),
            ActionRecord(step=2, name="repair apple 2",
                         params={"object": "apple 2"},
                         mode=ExecutionMode.DYNAMIC, node_ref=str(atomic.ref)),
        ],
        state_snapshots=_state_snapshots(
            [], [], ["object_heated(apple_1)"],
            ["object_heated(apple_1)", "object_heated(apple_2)"]),
        realized_atomic_nodes=[{
            "ref": str(atomic.ref), "params": {"object": "apple"},
            "passed": True,
            "attempts": [
                {"mode": "direct", "passed": False,
                 "failure_type": "effect_not_met",
                 "tool_refs": [str(old_tool.ref)],
                 "action_start": 0, "action_end": 1,
                 "before": {}, "after": {}},
                {"mode": "dynamic", "passed": True, "failure_type": "",
                 "tool_refs": [], "action_start": 1, "action_end": 3,
                 "before": {},
                 "after": {"facts": ["object_heated(apple_2)"]}},
            ],
        }],
    )
    task = Task(task_id="task_tool_ambiguous", benchmark="toy_env",
                task_type="toy", goal="heat apple")
    manager = FailureBranchManager(root, registry, tools, _ReplayAdapter(),
                                   SystemConfig(data_dir=root))
    event = manager.process(trace, task)[0]
    assert event["status"] == "awaiting_success_evidence"
    assert event["reason"] == "ambiguous_binding"
    assert event["ambiguous_parameter_roles"] == ["object"]
    assert event["binding_candidates"]["object"] == ["apple 1", "apple 2"]
    assert event["repair_event_slice"]["event_slice_validated"] is False
    assert tools.get_recommended(old_tool.tool_id).ref == old_tool.ref


def test_direct_tool_repair_uses_shared_slice_and_retains_effect_producer(workspace_tmp):
    root = workspace_tmp / "unique_occurrence_tool"
    registry, tools, atomic, old_tool = _bank(
        root, effect_predicate="object.heated")
    trace = TraceRecord(
        trace_id="trace_unique_occurrence_tool", task_id="task_tool_unique",
        task_type="toy", task_goal="heat apple", benchmark="toy_env",
        success=True,
        actions=[
            ActionRecord(step=0, name="bad apple", mode=ExecutionMode.DIRECT,
                         node_ref=str(atomic.ref), tool_ref=str(old_tool.ref)),
            ActionRecord(step=1, name="repair apple 1",
                         params={"object": "apple 1"},
                         mode=ExecutionMode.DYNAMIC, node_ref=str(atomic.ref)),
        ],
        state_snapshots=_state_snapshots(
            [], [], ["object_heated(apple_1)"]),
        realized_atomic_nodes=[{
            "ref": str(atomic.ref), "params": {"object": "apple"},
            "passed": True,
            "attempts": [
                {"mode": "direct", "passed": False,
                 "failure_type": "effect_not_met",
                 "tool_refs": [str(old_tool.ref)],
                 "action_start": 0, "action_end": 1,
                 "before": {}, "after": {}},
                {"mode": "dynamic", "passed": True, "failure_type": "",
                 "tool_refs": [], "action_start": 1, "action_end": 2,
                 "before": {},
                 "after": {"facts": ["object_heated(apple_1)"]}},
            ],
        }],
    )
    task = Task(task_id="task_tool_unique", benchmark="toy_env",
                task_type="toy", goal="heat apple")
    manager = FailureBranchManager(root, registry, tools, _ReplayAdapter(),
                                   SystemConfig(data_dir=root))
    event = manager.process(trace, task)[0]
    assert event["status"] == "merged"
    assert event["replay_bindings"]["object"] == "apple 1"
    assert event["repair_event_slice"]["event_slice_validated"] is True
    assert event["repair_event_slice"]["effect_producer_indices"] == [1]
    assert event["repair_event_slice"]["retained_event_indices"] == [1]
    repaired = tools.get_recommended(old_tool.tool_id)
    assert repaired.artifact["steps"] == ["repair {object}"]


def test_atomic_repair_replay_does_not_credit_rejected_template_action(workspace_tmp):
    root = workspace_tmp / "rejected_replay_action"
    registry, tools, atomic, _tool = _bank(root)
    trace = TraceRecord(
        trace_id="trace_rejected_template", task_id="task_rejected",
        task_type="toy", task_goal="repair apple", benchmark="toy_env",
        success=True,
        actions=[ActionRecord(step=0, name="repair apple 1",
                              mode=ExecutionMode.DYNAMIC,
                              node_ref=str(atomic.ref))],
        realized_atomic_nodes=[{
            "ref": str(atomic.ref), "params": {"object": "apple 1"},
            "passed": True,
            "attempts": [
                {"mode": "seeded", "passed": False,
                 "failure_type": "effect_not_met", "action_start": 0,
                 "action_end": 0, "before": {}, "after": {}},
                {"mode": "dynamic", "passed": True, "failure_type": "",
                 "action_start": 0, "action_end": 1, "before": {},
                 "after": {"facts": ["object.done(apple_1)"]}},
            ],
        }],
    )
    task = Task(task_id="task_rejected", benchmark="toy_env",
                task_type="toy", goal="repair apple")
    manager = FailureBranchManager(
        root, registry, tools, _RejectedTemplateReplayAdapter(),
        SystemConfig(data_dir=root), llm=object())
    event = manager.process(trace, task)[0]
    assert event["status"] == "rejected"
    assert event["reason"] == "strict_seeded_replay_failed"
    assert event["strict_seeded_replay"]["atomic_effect_reached"] is True
    assert event["strict_seeded_replay"]["repair_template_executed"] is False
    assert registry.list_versions(atomic.ref.logical_id) == [atomic.ref.version]


def test_failure_without_rescue_creates_shadow_copies_but_never_merges(workspace_tmp):
    root = workspace_tmp / "data"
    registry, tools, atomic, old_tool = _bank(root)
    trace = TraceRecord(
        trace_id="trace_no_rescue", task_id="task_no_rescue", task_type="toy",
        task_goal="repair apple", benchmark="toy_env", success=False,
        realized_atomic_nodes=[{
            "ref": str(atomic.ref), "params": {"object": "apple 1"},
            "tool_refs": [str(old_tool.ref)], "passed": False,
            "attempts": [{
                "mode": "direct", "passed": False,
                "failure_type": "effect_not_met",
                "tool_refs": [str(old_tool.ref)],
                "action_start": 0, "action_end": 1,
                "before": {}, "after": {},
            }],
        }],
    )
    task = Task(task_id="task_no_rescue", benchmark="toy_env", task_type="toy",
                goal="repair apple")
    manager = FailureBranchManager(root, registry, tools, _ReplayAdapter(),
                                   SystemConfig(data_dir=root))
    event = manager.process(trace, task)[0]
    assert event["status"] == "awaiting_success_evidence"
    assert event["merge_allowed"] is False
    assert {item["kind"] for item in event["shadow_candidates"]} == {
        "abstract_atomic", "tool"}
    assert registry.get_recommended(atomic.ref.logical_id).ref == atomic.ref
    assert tools.get_recommended(old_tool.tool_id).ref == old_tool.ref
    branch = root / "evolution" / "branches" / "trace_no_rescue-0-0-direct"
    branch_registry = SkillGraphRegistry(branch / "bank" / "skill_graph")
    branch_tools = ToolRegistry(branch / "bank" / "tools")
    assert branch_registry.get_recommended(atomic.ref.logical_id).status == SkillStatus.SHADOW
    assert branch_tools.get_latest(old_tool.tool_id).status == ToolLifecycle.SHADOW
    assert branch_tools.get_recommended(old_tool.tool_id).ref == old_tool.ref
