from __future__ import annotations

from types import SimpleNamespace

from atomic_skillgraph.adapters.benchmark import Task
from atomic_skillgraph.core.config import SystemConfig
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.skill_ir import AbstractAtomicSkill, CompositeSkill
from atomic_skillgraph.core.status import SkillStatus
from atomic_skillgraph.graph.registry import SkillGraphRegistry
from atomic_skillgraph.evolution.success_processor import SuccessProcessor
from atomic_skillgraph.runtime.atomic_planner import AtomicPlanner
from atomic_skillgraph.core.trace_ir import TraceRecord


TASK_TYPE = "pick_heat_then_place_in_recep"


def _atomic(logical_id: str, predicate: str, inputs: list[str]):
    return AbstractAtomicSkill(
        ref=SkillRef(logical_id, "1.0.0"),
        summary=logical_id,
        inputs=[{"name": name} for name in inputs],
        effects=[{
            "predicate": predicate,
            "args": {name: f"$inputs.{name}" for name in inputs
                     if name in {"object", "target_location"}},
        }],
        metadata={"task_type_labels": [TASK_TYPE],
                  "statistics": {"utility": 0.5}},
        status=SkillStatus.ACTIVE,
    )


def _composite(logical_id: str, refs: list[str], summary: str, utility: float,
               *, status: SkillStatus = SkillStatus.ACTIVE,
               target_effects: list[dict] | None = None):
    return CompositeSkill(
        ref=SkillRef(logical_id, "1.0.0"), summary=summary,
        task_type_labels=[TASK_TYPE],
        graph={"nodes": refs},
        validator={"target_effects": target_effects or []},
        metadata={"statistics": {"utility": utility}},
        status=status,
    )


def _registry(root, *, include_complete: bool = True):
    registry = SkillGraphRegistry(root / "skill_graph")
    acquire = _atomic("alfworld.acquire_object", "agent.holds", ["object"])
    heat = _atomic("alfworld.heat_object", "object.heated", ["object"])
    place = _atomic("alfworld.place_object", "object.at_location",
                    ["object", "target_location"])
    for atomic in (acquire, heat, place):
        registry.register(atomic)
    partial = _composite(
        "composite.alfworld.acquire-heat",
        [str(acquire.ref), str(heat.ref)],
        "heat mug and put it in cabinet", 1.0,
    )
    registry.register(partial)
    if include_complete:
        registry.register(_composite(
            "composite.alfworld.acquire-heat-place",
            [str(acquire.ref), str(heat.ref), str(place.ref)],
            "unrelated complete chain", 0.1,
        ))
    return registry


def _task(*, goal: str = "heat some mug and put it in cabinet.",
          params: dict | None = None):
    return Task(
        task_id="alfworld_test", benchmark="alfworld", task_type=TASK_TYPE,
        goal=goal,
        context={"params": params or {}}, state={"facts": []},
        target_effects=[
            {"predicate": "object.heated", "args": {"object": "$object"}},
            {"predicate": "object.at_location",
             "args": {"object": "$object", "location": "$target_location"}},
        ],
    )


def test_complete_composite_beats_higher_scoring_partial(workspace_tmp):
    registry = _registry(workspace_tmp, include_complete=True)
    planner = AtomicPlanner(registry, SystemConfig())
    plan = planner.compile_runtime_graph(_task(params={
        "object": "mug", "target_location": "cabinet 1"}))
    assert plan.composite_ref.endswith(
        "composite.alfworld.acquire-heat-place@1.0.0")
    assert not any(node.dynamic for node in plan.nodes)
    assert "composite_effect_complete" in plan.notes


def test_partial_composite_dynamic_gap_binds_task_params(workspace_tmp):
    registry = _registry(workspace_tmp, include_complete=False)
    planner = AtomicPlanner(registry, SystemConfig())
    plan = planner.compile_runtime_graph(_task(params={
        "object": "mug", "target_location": "cabinet 1",
        "heating_station": "microwave 1"}))
    gaps = [node for node in plan.nodes if node.dynamic]
    assert len(gaps) == 1
    assert gaps[0].params["object"] == "mug"
    assert gaps[0].params["target_location"] == "cabinet 1"
    assert gaps[0].params["heating_station"] == "microwave 1"
    assert "composite_partial_with_bound_gap" in plan.notes


def test_irrelevant_broader_composite_is_not_selected_for_simple_goal(workspace_tmp):
    registry = _registry(workspace_tmp, include_complete=False)
    place = registry.get_recommended("alfworld.place_object")
    cool = _atomic("alfworld.cool_object", "object.cooled", ["object"])
    registry.register(cool)
    registry.register(_composite(
        "composite.alfworld.cool-place",
        [str(cool.ref), str(place.ref)], "cool then place", 1.0,
        target_effects=[
            {"predicate": "object.cooled", "args": {"object": "$object"}},
            {"predicate": "object.at_location",
             "args": {"object": "$object", "location": "$target_location"}},
        ],
    ))
    simple = _task(params={"object": "mug", "target_location": "cabinet 1"})
    simple.target_effects = [simple.target_effects[-1]]
    plan = AtomicPlanner(registry, SystemConfig()).compile_runtime_graph(simple)
    assert "cool-place" not in plan.composite_ref
    assert all(node.ref.logical_id != "alfworld.cool_object" for node in plan.nodes)


def test_frozen_replay_never_selects_single_trace_draft_composite(workspace_tmp):
    registry = _registry(workspace_tmp, include_complete=False)
    acquire = registry.get_recommended("alfworld.acquire_object")
    heat = registry.get_recommended("alfworld.heat_object")
    place = registry.get_recommended("alfworld.place_object")
    registry.register(_composite(
        "composite.alfworld.unconfirmed",
        [str(acquire.ref), str(heat.ref), str(place.ref)], "complete draft", 1.0,
        status=SkillStatus.DRAFT,
        target_effects=_task().target_effects,
    ))
    config = SystemConfig(freeze_skills=True)
    plan = AtomicPlanner(registry, config).compile_runtime_graph(_task(params={
        "object": "mug", "target_location": "cabinet 1"}))
    assert "unconfirmed" not in plan.composite_ref
    assert "controlled_candidate_exploration" not in plan.notes


def test_dynamic_transformation_gap_is_inserted_before_delivery(workspace_tmp):
    registry = _registry(workspace_tmp, include_complete=False)
    acquire = registry.get_recommended("alfworld.acquire_object")
    place = registry.get_recommended("alfworld.place_object")
    registry.register(_composite(
        "composite.alfworld.acquire-place",
        [str(acquire.ref), str(place.ref)], "acquire then place", 0.99,
        target_effects=[_task().target_effects[-1]],
    ))
    plan = AtomicPlanner(registry, SystemConfig()).compile_runtime_graph(
        _task(params={"object": "mug", "target_location": "cabinet 1",
                      "heating_station": "microwave 1"}))
    predicates = [
        next(iter(node.target_effects), {}).get("predicate")
        for node in plan.nodes
    ]
    assert predicates.index("object.heated") < predicates.index("object.at_location")


def test_unbound_partial_composite_falls_back_to_atomic_plan(workspace_tmp):
    registry = _registry(workspace_tmp, include_complete=False)
    planner = AtomicPlanner(registry, SystemConfig())
    plan = planner.compile_runtime_graph(_task(
        goal="heat an object", params={"object": "mug"}))
    assert plan.composite_ref == ""
    assert "atomic_only_plan" in plan.notes
    assert all(node.source != "dynamic_gap" for node in plan.nodes)


def test_atomic_plan_closes_dependencies_and_orders_acquire_heat_place(workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "ordered_graph")
    acquire = _atomic("alfworld.acquire_object", "agent.holds", ["object"])
    heat = _atomic("alfworld.heat_object", "object.heated", ["object"])
    place = _atomic("alfworld.place_object", "object.at_location",
                    ["object", "target_location"])
    heat.preconditions = [{"predicate": "agent.holds",
                           "args": {"object": "$inputs.object"}}]
    place.preconditions = [{"predicate": "agent.holds",
                            "args": {"object": "$inputs.object"}}]
    # Place 的分数故意最高，确认排序来自依赖/目标数据流而不是 retrieval score。
    acquire.metadata["statistics"]["utility"] = 0.7
    heat.metadata["statistics"]["utility"] = 0.8
    place.metadata["statistics"]["utility"] = 0.99
    for atomic in (acquire, heat, place):
        registry.register(atomic)
    config = SystemConfig()
    config.features.enable_composite = False
    planner = AtomicPlanner(registry, config)
    plan = planner.compile_runtime_graph(_task(params={
        "object": "mug", "target_location": "cabinet 1"}))
    assert [node.ref.logical_id for node in plan.nodes] == [
        "alfworld.acquire_object", "alfworld.heat_object", "alfworld.place_object"]


def test_pick_two_atomic_plan_repeats_closed_acquire_place_branch(workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "pick_two_graph")
    acquire = _atomic("alfworld.acquire_object", "agent.holds", ["object"])
    place = _atomic("alfworld.place_object", "object.at_location",
                    ["object", "target_location"])
    acquire.metadata["task_type_labels"] = ["pick_two_obj_and_place"]
    place.metadata["task_type_labels"] = ["pick_two_obj_and_place"]
    place.preconditions = [{"predicate": "agent.holds",
                            "args": {"object": "$inputs.object"}}]
    registry.register(acquire)
    registry.register(place)
    config = SystemConfig()
    config.features.enable_composite = False
    planner = AtomicPlanner(registry, config)
    task = Task(
        task_id="pick_two", benchmark="alfworld",
        task_type="pick_two_obj_and_place",
        goal="put two cd in safe.", state={"facts": []},
        context={"params": {"object": "cd", "object_type": "cd",
                            "target_location": "safe 1"}},
        target_effects=[{
            "predicate": "object.at_location",
            "args": {"object": "$object_type", "location": "$target_location"},
            "cardinality": 2, "distinct_by": "object",
        }],
    )
    plan = planner.compile_runtime_graph(task)
    assert [node.ref.logical_id for node in plan.nodes] == [
        "alfworld.acquire_object", "alfworld.place_object",
        "alfworld.acquire_object", "alfworld.place_object",
    ]


def test_legacy_composite_cannot_bind_destination_as_acquire_source():
    resolved = AtomicPlanner._resolve_composite_params(
        {"object": "$task.object",
         "object_location": "$task.target_location"},
        _task(params={"object": "mug", "target_location": "cabinet 1"}),
    )
    assert resolved == {"object": "mug"}


def test_learned_parameter_family_binds_unstated_resource_without_task_type():
    atomic = _atomic(
        "generic.state_change", "object.changed",
        ["object", "transformation_resource"])
    atomic.metadata["observed_parameter_families"] = {
        "object": ["apple"],
        "transformation_resource": ["fixture"],
    }
    task = Task(
        task_id="unseen", benchmark="generic_env", task_type="secret_label",
        goal="make an apple ready", state={"facts": []},
        context={
            "params": {"object": "apple"},
            "goal_roles": {"theme": "apple"},
            "exposed_entities": ["fixture_7", "bay_2"],
        },
    )
    planner = AtomicPlanner(SkillGraphRegistry.__new__(SkillGraphRegistry),
                            SystemConfig())
    assert planner._bind_params(task, atomic) == {
        "object": "apple",
        "transformation_resource": "fixture_7",
    }


def test_lifecycle_review_runs_even_when_heavy_maintenance_is_deferred():
    processor = object.__new__(SuccessProcessor)
    processor.config = SystemConfig(maintenance_interval=100)
    processor.base_compiler = None
    processor.atomicizer = SimpleNamespace(apply=lambda trace: SimpleNamespace(
        candidates=[], segments=[]))
    processor.registry = SimpleNamespace()
    calls: list[object] = []
    processor.lifecycle = SimpleNamespace(
        review=lambda **kwargs: calls.append(kwargs) or [])
    processor.generalizer = SimpleNamespace(
        run_maintenance=lambda: (_ for _ in ()).throw(
            AssertionError("heavy maintenance must stay deferred")))
    result = processor.process_success(
        TraceRecord(trace_id="trace_lifecycle", success=True),
        run_maintenance=False,
    )
    assert len(calls) == 1
    assert result.maintenance == []
