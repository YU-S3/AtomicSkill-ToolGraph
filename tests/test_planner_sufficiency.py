from __future__ import annotations

from types import SimpleNamespace

from atomic_skillgraph.adapters.benchmark import Task
from atomic_skillgraph.core.config import SystemConfig
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.edge_ir import GraphEdge
from atomic_skillgraph.core.skill_ir import AbstractAtomicSkill, CompositeSkill
from atomic_skillgraph.core.status import EdgeType, SkillNodeKind, SkillStatus
from atomic_skillgraph.graph.registry import RetrievalHit, SkillGraphRegistry
from atomic_skillgraph.evolution.success_processor import SuccessProcessor
from atomic_skillgraph.runtime.atomic_planner import AtomicPlanner
from atomic_skillgraph.runtime.runtime_graph import RuntimeGraph
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
    steps = []
    for index, ref in enumerate(refs):
        params = {"object": "$task.object"}
        if "place" in ref:
            params["target_location"] = "$task.target_location"
        steps.append({"step_id": f"step_{index:03d}",
                      "node_ref": ref, "params": params})
    control = [GraphEdge(
        source=refs[index], target=refs[index + 1], type=EdgeType.NEXT,
        scope="composite", source_step=f"step_{index:03d}",
        target_step=f"step_{index + 1:03d}").to_dict()
        for index in range(len(refs) - 1)]
    return CompositeSkill(
        ref=SkillRef(logical_id, "1.0.0"), summary=summary,
        task_type_labels=[TASK_TYPE],
        graph={"nodes": refs, "steps": steps, "control": control},
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
        target_effects=[_task().target_effects[0]],
    )
    registry.register(partial)
    if include_complete:
        registry.register(_composite(
            "composite.alfworld.acquire-heat-place",
            [str(acquire.ref), str(heat.ref), str(place.ref)],
            "unrelated complete chain", 0.1,
            target_effects=_task().target_effects,
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


def test_complete_composite_retains_unbound_causal_producer(workspace_tmp):
    """未知来源位置必须留给 Runtime 发现，不能删除 Acquire 生产者。"""
    registry = SkillGraphRegistry(workspace_tmp / "causal_unbound_graph")
    acquire = _atomic(
        "alfworld.acquire_object", "agent.holds",
        ["object", "object_location"])
    heat = _atomic("alfworld.heat_object", "object.heated", ["object"])
    place = _atomic("alfworld.place_object", "object.at_location",
                    ["object", "target_location"])
    heat.preconditions = [{
        "predicate": "agent.holds",
        "args": {"object": "$inputs.object"},
    }]
    for atomic in (acquire, heat, place):
        registry.register(atomic)
    registry.register(_composite(
        "composite.alfworld.causal-unbound",
        [str(acquire.ref), str(heat.ref), str(place.ref)],
        "acquire, heat, place", 1.0,
        target_effects=_task().target_effects))

    plan = AtomicPlanner(registry, SystemConfig()).compile_runtime_graph(
        _task(params={"object": "mug", "target_location": "cabinet 1"}))
    assert [node.ref.logical_id for node in plan.nodes] == [
        "alfworld.acquire_object", "alfworld.heat_object",
        "alfworld.place_object"]
    assert "object_location" not in plan.nodes[0].params


def test_partial_composite_defers_missing_effect_to_explicit_task_gap(
        workspace_tmp):
    registry = _registry(workspace_tmp, include_complete=False)
    planner = AtomicPlanner(registry, SystemConfig())
    plan = planner.compile_runtime_graph(_task(params={
        "object": "mug", "target_location": "cabinet 1",
        "heating_station": "microwave 1"}))
    assert plan.composite_ref.endswith(
        "composite.alfworld.acquire-heat@1.0.0")
    assert not any(node.dynamic for node in plan.nodes)
    assert [node.ref.logical_id for node in plan.nodes] == [
        "alfworld.acquire_object", "alfworld.heat_object"]
    assert "composite_partial_explicit_task_gap_pending" in plan.notes


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


def test_partial_composite_never_preinserts_anonymous_dynamic_gap(workspace_tmp):
    registry = _registry(workspace_tmp, include_complete=False)
    acquire = registry.get_recommended("alfworld.acquire_object")
    place = registry.get_recommended("alfworld.place_object")
    registry.register(_composite(
        "composite.alfworld.acquire-place",
        [str(acquire.ref), str(place.ref)], "acquire then place", 2.0,
        target_effects=[_task().target_effects[-1]],
    ))
    plan = AtomicPlanner(registry, SystemConfig()).compile_runtime_graph(
        _task(params={"object": "mug", "target_location": "cabinet 1",
                      "heating_station": "microwave 1"}))
    assert plan.composite_ref
    assert not any(node.dynamic for node in plan.nodes)
    predicates = [next(iter(node.target_effects), {}).get("predicate")
                  for node in plan.nodes]
    task_predicates = {
        str(item.get("predicate") or "") for item in _task().target_effects}
    # Whichever relevant partial Composite wins retrieval, at least one formal
    # target remains outside the pre-gap graph and no anonymous effect node is
    # allowed to stand in for it.
    assert task_predicates - set(predicates)
    assert all(not node.ref.logical_id.startswith("runtime.dynamic")
               for node in plan.nodes)
    assert "composite_partial_explicit_task_gap_pending" in plan.notes


def test_no_target_producer_never_executes_unrelated_top_hits(workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "irrelevant_atomic_graph")
    for atomic in (
        _atomic("generic.cool", "object.cooled", ["object"]),
        _atomic("generic.heat", "object.heated", ["object"]),
        _atomic("generic.observe", "object.observed_with", ["object"]),
    ):
        registry.register(atomic)
    simple = _task(params={"object": "mug", "target_location": "cabinet 1"})
    simple.target_effects = [simple.target_effects[-1]]
    plan = AtomicPlanner(registry, SystemConfig()).compile_runtime_graph(simple)
    assert len(plan.nodes) == 1
    assert plan.nodes[0].dynamic is True
    assert plan.nodes[0].target_effects[0]["predicate"] == "object.at_location"


def test_unbound_target_atomic_falls_back_to_one_task_dynamic_node(workspace_tmp):
    registry = _registry(workspace_tmp, include_complete=False)
    planner = AtomicPlanner(registry, SystemConfig())
    plan = planner.compile_runtime_graph(_task(
        goal="heat an object", params={"object": "mug"}))
    assert plan.composite_ref == ""
    assert "task_level_dynamic_fallback" in plan.notes
    assert len(plan.nodes) == 1
    assert plan.nodes[0].source == "task_dynamic"
    assert plan.nodes[0].target_effects == _task().target_effects


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
    assert [node.branch_id for node in plan.nodes] == [
        "branch_000", "branch_000", "branch_001", "branch_001"]
    assert [bool(node.distinct_bindings.get("object"))
            for node in plan.nodes] == [True, True, True, True]


def test_different_cardinality_targets_expand_exact_variable_width_branches(
        workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "different_cardinality")
    clean = AbstractAtomicSkill(
        ref=SkillRef("generic.clean", "1.0.0"), summary="clean",
        inputs=[{"name": "clean_object"}],
        effects=[{
            "predicate": "object.cleaned",
            "args": {"object": "$inputs.clean_object"},
        }], status=SkillStatus.ACTIVE)
    toggle = AbstractAtomicSkill(
        ref=SkillRef("generic.toggle", "1.0.0"), summary="toggle",
        inputs=[{"name": "toggle_object"}],
        effects=[{
            "predicate": "object.toggled",
            "args": {"object": "$inputs.toggle_object"},
        }], status=SkillStatus.ACTIVE)
    registry.register(clean)
    registry.register(toggle)
    targets = [{
        "predicate": "object.cleaned",
        "args": {"object": "$clean_object"},
        "cardinality": 2, "distinct_by": "object",
    }, {
        "predicate": "object.toggled",
        "args": {"object": "$toggle_object"},
        "cardinality": 3, "distinct_by": "object",
    }]
    task = Task(
        task_id="two_plus_three", benchmark="generic_env",
        goal="clean two widgets and toggle three lamps",
        context={"params": {"clean_object": "widget",
                            "toggle_object": "desklamp"}},
        state={"facts": []}, target_effects=targets)
    hits = [
        RetrievalHit(ref=clean.ref, kind=SkillNodeKind.ABSTRACT_ATOMIC,
                     obj=clean, score=1.0),
        RetrievalHit(ref=toggle.ref, kind=SkillNodeKind.ABSTRACT_ATOMIC,
                     obj=toggle, score=0.9),
    ]
    plan = AtomicPlanner(registry, SystemConfig())._compile_atomic_runtime_plan(
        task, hits, hits)

    assert [node.ref.logical_id for node in plan.nodes] == [
        "generic.clean", "generic.toggle",
        "generic.clean", "generic.toggle",
        "generic.toggle",
    ]
    assert [node.branch_id for node in plan.nodes] == [
        "branch_000", "branch_000", "branch_001", "branch_001",
        "branch_002",
    ]
    clean_group = "target_000:object.cleaned:object"
    toggle_group = "target_001:object.toggled:object"
    assert sum(clean_group in node.distinct_bindings.get("clean_object", [])
               for node in plan.nodes) == 2
    assert sum(toggle_group in node.distinct_bindings.get("toggle_object", [])
               for node in plan.nodes) == 3


def test_same_predicate_cardinality_keeps_argument_families_separate(
        workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "same_predicate_cardinality")
    place_mug = AbstractAtomicSkill(
        ref=SkillRef("generic.place_mug", "1.0.0"), summary="place mug",
        inputs=[{"name": "mug"}, {"name": "cabinet"}],
        effects=[{
            "predicate": "object.at_location",
            "args": {"object": "$inputs.mug",
                     "location": "$inputs.cabinet"},
        }], status=SkillStatus.ACTIVE)
    place_book = AbstractAtomicSkill(
        ref=SkillRef("generic.place_book", "1.0.0"), summary="place book",
        inputs=[{"name": "book"}, {"name": "shelf"}],
        effects=[{
            "predicate": "object.at_location",
            "args": {"object": "$inputs.book",
                     "location": "$inputs.shelf"},
        }], status=SkillStatus.ACTIVE)
    registry.register(place_mug)
    registry.register(place_book)
    targets = [{
        "predicate": "object.at_location",
        "args": {"object": "$mug", "location": "$cabinet"},
        "cardinality": 2, "distinct_by": "object",
    }, {
        "predicate": "object.at_location",
        "args": {"object": "$book", "location": "$shelf"},
        "cardinality": 3, "distinct_by": "object",
    }]
    task = Task(
        task_id="two_mugs_three_books", benchmark="generic_env",
        goal="place two mugs in a cabinet and three books on a shelf",
        context={"params": {
            "mug": "mug", "cabinet": "cabinet_1",
            "book": "book", "shelf": "shelf_1",
        }}, state={"facts": []}, target_effects=targets)
    hits = [
        RetrievalHit(ref=place_mug.ref,
                     kind=SkillNodeKind.ABSTRACT_ATOMIC,
                     obj=place_mug, score=1.0),
        RetrievalHit(ref=place_book.ref,
                     kind=SkillNodeKind.ABSTRACT_ATOMIC,
                     obj=place_book, score=0.9),
    ]

    plan = AtomicPlanner(registry, SystemConfig())._compile_atomic_runtime_plan(
        task, hits, hits)

    assert [node.ref.logical_id for node in plan.nodes] == [
        "generic.place_mug", "generic.place_book",
        "generic.place_mug", "generic.place_book",
        "generic.place_book",
    ]
    assert [node.branch_id for node in plan.nodes] == [
        "branch_000", "branch_000", "branch_001", "branch_001",
        "branch_002",
    ]


def test_pick_two_composite_without_data_edges_projects_distinct_to_acquire(
        workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "pick_two_composite_graph")
    acquire = _atomic("generic.acquire", "agent.holds", ["object"])
    acquire.outputs = [{
        "name": "object", "semantic_type": "object_ref",
        "materializer": {"kind": "input_role", "role": "object"},
    }]
    place = _atomic(
        "generic.place", "object.at_location",
        ["object", "target_location"])
    # This is a valid legacy Composite whose occurrence ordering is explicit,
    # but which has neither DATA_FLOW nor REQUIRES_SKILL edges.
    place.preconditions = []
    registry.register(acquire)
    registry.register(place)
    steps = [
        {"step_id": "a0", "node_ref": str(acquire.ref),
         "params": {"object": "$task.object"}},
        {"step_id": "p0", "node_ref": str(place.ref),
         "params": {"object": "$task.object",
                    "target_location": "$task.target_location"}},
        {"step_id": "a1", "node_ref": str(acquire.ref),
         "params": {"object": "$task.object"}},
        {"step_id": "p1", "node_ref": str(place.ref),
         "params": {"object": "$task.object",
                    "target_location": "$task.target_location"}},
    ]
    control = [GraphEdge(
        source=steps[index]["node_ref"],
        target=steps[index + 1]["node_ref"], type=EdgeType.NEXT,
        scope="composite", source_step=steps[index]["step_id"],
        target_step=steps[index + 1]["step_id"]).to_dict()
        for index in range(len(steps) - 1)]
    target = [{
        "predicate": "object.at_location",
        "args": {"object": "$object_type", "location": "$target_location"},
        "cardinality": 2, "distinct_by": "object",
    }]
    composite = CompositeSkill(
        ref=SkillRef("composite.generic.pick-two", "1.0.0"),
        summary="acquire and place two distinct objects",
        graph={"nodes": [str(acquire.ref), str(place.ref)],
               "steps": steps, "control": control, "data": []},
        validator={"target_effects": target},
        metadata={"statistics": {"utility": 1.0}},
        status=SkillStatus.ACTIVE,
    )
    registry.register(composite)
    task = Task(
        task_id="pick_two_composite", benchmark="generic_env",
        task_type="unseen_batch_task", goal="place two widgets",
        state={"facts": []},
        context={"params": {"object": "widget", "object_type": "widget",
                            "target_location": "bay_1"}},
        target_effects=target,
    )

    planner = AtomicPlanner(registry, SystemConfig())
    hit = RetrievalHit(
        ref=composite.ref, kind=SkillNodeKind.COMPOSITE,
        obj=composite, score=1.0)
    plan = planner._plan_from_composite(task, hit, [hit])

    assert plan.composite_ref == str(composite.ref)
    assert [node.branch_id for node in plan.nodes] == [
        "branch_000", "branch_000", "branch_001", "branch_001"]
    assert all(node.distinct_bindings.get("object") for node in plan.nodes)
    group = plan.nodes[0].distinct_bindings["object"][0]
    assert [node.distinct_branch_ids[group] for node in plan.nodes] == [
        "occ_000", "occ_000", "occ_001", "occ_001"]
    runtime = RuntimeGraph(task.task_id, plan)
    runtime.nodes[0].passed = True
    runtime.nodes[0].params["object"] = "widget_1"
    runtime.nodes[0].outputs["object"] = "widget_1"
    assert runtime.distinct_exclusions(1) == {}
    assert runtime.distinct_exclusions(2) == {"object": {"widget_1"}}


def test_legacy_composite_cannot_bind_destination_as_acquire_source():
    resolved = AtomicPlanner._resolve_composite_params(
        {"object": "$task.object",
         "object_location": "$task.target_location"},
        _task(params={"object": "mug", "target_location": "cabinet 1"}),
    )
    assert resolved == {"object": "mug"}


def test_planner_rejects_legacy_unsafe_composite_and_uses_atomics(
        workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "legacy_unsafe_composite")
    acquire = _atomic(
        "generic.acquire", "agent.holds", ["object", "object_location"])
    place = _atomic(
        "generic.place", "object.at_location", ["object", "target_location"])
    place.preconditions = [{
        "predicate": "agent.holds",
        "args": {"object": "$inputs.object"},
    }]
    registry.register(acquire)
    registry.register(place)
    target = [{
        "predicate": "object.at_location",
        "args": {"object": "$object", "location": "$target_location"},
    }]
    legacy = _composite(
        "composite.generic.legacy-unsafe-source",
        [str(acquire.ref), str(place.ref)],
        "acquire and place", 2.0, target_effects=target)
    legacy.graph["steps"][0]["params"]["object_location"] = (
        "$task.target_location")
    registry.register(legacy)
    task = Task(
        task_id="legacy_unsafe", benchmark="generic_env",
        task_type="unseen_task", goal="put mug in cabinet",
        state={"facts": []},
        context={"params": {"object": "mug",
                            "target_location": "cabinet_1"}},
        target_effects=target,
    )

    plan = AtomicPlanner(registry, SystemConfig()).compile_runtime_graph(task)

    assert plan.composite_ref == ""
    assert [node.ref.logical_id for node in plan.nodes] == [
        "generic.acquire", "generic.place"]
    assert plan.nodes[0].params.get("object_location") != "cabinet_1"


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
