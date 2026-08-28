from types import SimpleNamespace

from atomic_skillgraph.adapters.benchmark import Task
from atomic_skillgraph.core.config import SystemConfig
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.skill_ir import AbstractAtomicSkill
from atomic_skillgraph.core.status import EdgeType, SkillStatus
from atomic_skillgraph.graph.registry import SkillGraphRegistry
from atomic_skillgraph.runtime.atomic_planner import AtomicPlanner
from atomic_skillgraph.runtime.runtime_graph import PlannedNode


def _atomic(logical_id, predicate, *, inputs, preconditions=(), outputs=True):
    effect_args = {
        role: f"$inputs.{role}"
        for role in inputs
        if role in {"object", "container", "target_location"}
    }
    declarations = [{"name": role, "semantic_type": (
        "location_ref" if "location" in role else "entity_ref")}
        for role in inputs]
    output_declarations = []
    if outputs and effect_args:
        output_role = next(iter(effect_args))
        output_declarations.append({
            "name": output_role,
            "semantic_type": ("location_ref" if "location" in output_role
                              else "entity_ref"),
            "materializer": {
                "kind": "effect_arg", "predicate": predicate,
                "arg": output_role,
            },
        })
    return AbstractAtomicSkill(
        ref=SkillRef(logical_id, "1.0.0"), summary=logical_id,
        inputs=declarations, outputs=output_declarations,
        preconditions=[dict(item) for item in preconditions],
        effects=[{"predicate": predicate, "args": effect_args}],
        status=SkillStatus.ACTIVE,
    )


def _planner(workspace_tmp, atomics):
    registry = SkillGraphRegistry(workspace_tmp / "skill_graph")
    for atomic in atomics:
        registry.register(atomic)
    config = SystemConfig()
    config.features.enable_composite = False
    return AtomicPlanner(registry, config), registry


def _task(*, target_effects, params=None, goal="complete the task"):
    return Task(
        task_id="temporary_graph", benchmark="toy_env", task_type="generic",
        goal=goal, state={"facts": []}, context={"params": params or {}},
        target_effects=target_effects,
    )


def test_atomic_compilation_builds_occurrence_dependency_and_data_flow(
        workspace_tmp):
    acquire = _atomic("generic.acquire", "agent.holds", inputs=["object"])
    transform = _atomic(
        "generic.transform", "object.changed", inputs=["object"],
        preconditions=[{
            "predicate": "agent.holds",
            "args": {"object": "$inputs.object"},
        }],
    )
    deliver = _atomic(
        "generic.deliver", "object.at_location",
        inputs=["object", "target_location"], outputs=False,
        preconditions=[{
            "predicate": "object.changed",
            "args": {"object": "$inputs.object"},
        }],
    )
    planner, _ = _planner(workspace_tmp, [acquire, transform, deliver])
    task = _task(
        params={"object": "item_1", "target_location": "bay_1"},
        target_effects=[
            {"predicate": "object.changed",
             "args": {"object": "$object"}},
            {"predicate": "object.at_location",
             "args": {"object": "$object",
                      "target_location": "$target_location"}},
        ],
    )

    plan = planner.compile_runtime_graph(task)

    assert [node.ref.logical_id for node in plan.nodes] == [
        "generic.acquire", "generic.transform", "generic.deliver"]
    assert all(node.source == "atomic_compilation" for node in plan.nodes)
    assert len({node.occurrence_id for node in plan.nodes}) == 3
    edges = {(edge.type, edge.source_step, edge.target_step)
             for edge in plan.edges}
    assert (EdgeType.NEXT, "step_000", "step_001") in edges
    assert (EdgeType.NEXT, "step_001", "step_002") in edges
    assert (EdgeType.REQUIRES_SKILL, "step_000", "step_001") in edges
    assert (EdgeType.REQUIRES_SKILL, "step_001", "step_002") in edges
    data = [edge for edge in plan.edges if edge.type == EdgeType.DATA_FLOW]
    assert [(edge.source_step, edge.target_step,
             edge.mapping["source_output"], edge.mapping["target_input"])
            for edge in data] == [
        ("step_000", "step_001", "object", "object"),
        ("step_001", "step_002", "object", "object"),
    ]
    assert plan.audit["source_closure"]["passed"] is True


def test_cardinality_branches_have_no_cross_branch_data_flow(workspace_tmp):
    acquire = _atomic("generic.acquire", "agent.holds", inputs=["object"])
    deliver = _atomic(
        "generic.deliver", "object.at_location",
        inputs=["object", "target_location"], outputs=False,
        preconditions=[{
            "predicate": "agent.holds",
            "args": {"object": "$inputs.object"},
        }],
    )
    planner, _ = _planner(workspace_tmp, [acquire, deliver])
    task = _task(
        params={"object": "item", "target_location": "bay_1"},
        target_effects=[{
            "predicate": "object.at_location",
            "args": {"object": "$object",
                     "target_location": "$target_location"},
            "cardinality": 2, "distinct_by": "object",
        }],
    )

    plan = planner.compile_runtime_graph(task)

    assert [node.branch_id for node in plan.nodes] == [
        "branch_000", "branch_000", "branch_001", "branch_001"]
    data_pairs = [(edge.source_step, edge.target_step)
                  for edge in plan.edges if edge.type == EdgeType.DATA_FLOW]
    assert data_pairs == [("step_000", "step_001"),
                          ("step_002", "step_003")]
    requires_pairs = [(edge.source_step, edge.target_step)
                      for edge in plan.edges
                      if edge.type == EdgeType.REQUIRES_SKILL]
    assert requires_pairs == [("step_000", "step_001"),
                              ("step_002", "step_003")]
    distinct_groups = {
        group
        for node in plan.nodes
        for group in node.distinct_bindings.get("object", [])
    }
    assert len(distinct_groups) == 1
    assert all(node.distinct_bindings.get("object")
               for node in plan.nodes)
    assert plan.to_dict()["nodes"][2]["distinct_bindings"] == {
        "object": list(distinct_groups)}


def test_no_capability_or_incomplete_target_uses_one_task_dynamic_node(
        workspace_tmp):
    target = [{"predicate": "object.at_location",
               "args": {"object": "$object",
                        "target_location": "$target_location"}}]
    empty_planner, _ = _planner(workspace_tmp / "empty", [])
    empty = empty_planner.compile_runtime_graph(_task(
        params={"object": "item", "target_location": "bay"},
        target_effects=target))
    assert len(empty.nodes) == 1
    assert empty.nodes[0].dynamic is True
    assert empty.nodes[0].source == "task_dynamic"
    assert empty.nodes[0].target_effects == target

    unrelated = _atomic(
        "generic.observe", "object.observed", inputs=["object"])
    partial_planner, _ = _planner(workspace_tmp / "partial", [unrelated])
    partial = partial_planner.compile_runtime_graph(_task(
        params={"object": "item", "target_location": "bay"},
        target_effects=target))
    assert len(partial.nodes) == 1
    assert partial.nodes[0].source == "task_dynamic"
    assert "task_level_dynamic_fallback" in partial.notes


def test_unanchored_auxiliary_is_removable_but_target_producer_is_not(
        workspace_tmp):
    helper = _atomic(
        "generic.open", "container.open", inputs=["container"])
    target = _atomic("generic.acquire", "agent.holds", inputs=["object"])
    planner, _ = _planner(workspace_tmp, [helper, target])
    # Exercise the repair rule directly with explicit occurrence evidence;
    # dependency selection itself is tested by the full compilation cases.
    helper_node = PlannedNode(
        ref=helper.ref, step_id="step_000", target_effects=helper.effects)
    target_node = PlannedNode(
        ref=target.ref, step_id="step_001", target_effects=target.effects,
        params={"object": "item"})
    report = SimpleNamespace(
        node_reports=[
            SimpleNamespace(step_id="step_000",
                            unresolved_semantic_slots=["container"],
                            resolution_states={"container": "unresolvable"}),
            SimpleNamespace(step_id="step_001",
                            unresolved_semantic_slots=["object"],
                            resolution_states={"object": "unresolvable"}),
        ], errors=[],
    )
    task = _task(params={"object": "item"}, target_effects=[{
        "predicate": "agent.holds", "args": {"object": "$object"},
    }])

    kept, removed = planner._remove_unanchored_auxiliary_nodes(
        task, [helper_node, target_node], report)

    assert removed == ["step_000"]
    assert [node.step_id for node in kept] == ["step_001"]


def test_atomic_compilation_removes_unanchored_setup_and_revalidates(
        workspace_tmp):
    helper = _atomic(
        "generic.open", "container.open", inputs=["container"])
    target = _atomic(
        "generic.acquire", "agent.holds", inputs=["object", "container"],
        preconditions=[{
            "predicate": "container.open",
            "args": {"container": "$inputs.container"},
        }],
    )
    # The container is execution setup, not part of the Acquire core Effect.
    target.effects[0]["args"] = {"object": "$inputs.object"}
    planner, _ = _planner(workspace_tmp, [helper, target])
    task = _task(
        params={"object": "item_1"},
        target_effects=[{
            "predicate": "agent.holds", "args": {"object": "$object"},
        }],
    )

    plan = planner.compile_runtime_graph(task)

    assert [node.ref.logical_id for node in plan.nodes] == ["generic.acquire"]
    assert plan.nodes[0].dynamic is False
    assert "unanchored_auxiliary_removed" in plan.notes
    assert plan.audit["removed_unanchored_auxiliary_steps"] == ["step_000"]
    assert plan.audit["source_closure"]["passed"] is True
