from types import SimpleNamespace

from atomic_skillgraph.core.binding_ir import (
    BindingKind,
    BindingResolutionState,
    BindingSpec,
    binding_slot_name,
)
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.skill_ir import AbstractAtomicSkill
from atomic_skillgraph.core.status import EdgeType, SkillStatus
from atomic_skillgraph.graph.registry import SkillGraphRegistry
from atomic_skillgraph.runtime.data_flow_synthesizer import RuntimeDataFlowSynthesizer
from atomic_skillgraph.runtime.output_materializer import (
    materialize_atomic_outputs,
    validate_output_materializer,
)
from atomic_skillgraph.runtime.plan_validator import validate_plan_source_closure
from atomic_skillgraph.runtime.runtime_graph import PlannedNode, RuntimePlan


def _task(params=None):
    return SimpleNamespace(task_id="task_1", context={"params": params or {}}, state={})


def _atomic(logical_id, *, inputs, outputs, preconditions, effects):
    return AbstractAtomicSkill(
        ref=SkillRef(logical_id, "1.0.0"), summary=logical_id,
        inputs=inputs, outputs=outputs, preconditions=preconditions,
        effects=effects, status=SkillStatus.ACTIVE)


def _registry(workspace_tmp, *atomics):
    registry = SkillGraphRegistry(workspace_tmp / "second-round-graph")
    for atomic in atomics:
        registry.register(atomic)
    return registry


def test_bare_placeholder_is_a_semantic_slot():
    assert binding_slot_name("$object") == "object"
    assert binding_slot_name("$inputs.object") == "object"
    assert binding_slot_name("$task.object") == "object"
    assert binding_slot_name("$flow.object") == "object"
    assert BindingSpec.from_value("$object").kind == BindingKind.UNRESOLVED


def test_runtime_resolvable_tool_only_slot_keeps_seeded_route(workspace_tmp):
    acquire = _atomic(
        "generic.acquire", inputs=[
            {"name": "object", "semantic_type": "object_ref"},
            {"name": "object_location", "semantic_type": "location_ref",
             "direct_required": True, "runtime_resolvable": True,
             "anchor_roles": ["object"]},
        ], outputs=[], preconditions=[], effects=[
            {"predicate": "agent.holds", "args": {"object": "$inputs.object"}}
        ])
    registry = _registry(workspace_tmp, acquire)
    node = PlannedNode(
        ref=acquire.ref, step_id="step_000",
        params={"object": "apple", "object_location": "$object_location"},
        target_effects=list(acquire.effects))
    report = validate_plan_source_closure(
        RuntimePlan(nodes=[node]), registry, _task({"object": "apple"}))
    assert report.passed
    node_report = report.node_reports[0]
    assert node_report.resolution_states["object_location"] == (
        BindingResolutionState.RUNTIME_RESOLVABLE.value)
    assert "direct" not in node_report.executable_routes
    assert {"seeded", "dynamic"}.issubset(node_report.executable_routes)


def test_unanchored_semantic_slot_is_rejected(workspace_tmp):
    toggle = _atomic(
        "generic.toggle", inputs=[
            {"name": "device", "semantic_type": "object_ref",
             "semantic_required": True, "runtime_resolvable": True},
        ], outputs=[], preconditions=[], effects=[
            {"predicate": "object.toggled", "args": {"object": "$device"}}
        ])
    registry = _registry(workspace_tmp, toggle)
    plan = RuntimePlan(nodes=[PlannedNode(
        ref=toggle.ref, step_id="step_000", params={"device": "$device"},
        target_effects=list(toggle.effects))])
    report = validate_plan_source_closure(plan, registry, _task())
    assert not report.passed
    assert report.node_reports[0].resolution_states["device"] == (
        BindingResolutionState.UNRESOLVABLE.value)


def test_atomic_compilation_builds_exact_data_flow_and_source_closure(workspace_tmp):
    acquire = _atomic(
        "generic.acquire", inputs=[{"name": "object", "semantic_type": "object_ref"}],
        outputs=[{
            "name": "object", "semantic_type": "object_ref",
            "materializer": {"kind": "effect_arg", "predicate": "agent.holds",
                             "arg": "object"},
        }], preconditions=[], effects=[
            {"predicate": "agent.holds", "args": {"object": "$inputs.object"}}
        ])
    place = _atomic(
        "generic.place", inputs=[{"name": "object", "semantic_type": "object_ref"}],
        outputs=[], preconditions=[
            {"predicate": "agent.holds", "args": {"object": "$inputs.object"}}
        ], effects=[
            {"predicate": "object.at_location",
             "args": {"object": "$inputs.object", "location": "counter"}}
        ])
    registry = _registry(workspace_tmp, acquire, place)
    nodes = [
        PlannedNode(ref=acquire.ref, step_id="step_000", branch_id="branch_000",
                    params={"object": "apple"}, target_effects=list(acquire.effects)),
        PlannedNode(ref=place.ref, step_id="step_001", branch_id="branch_000",
                    params={"object": "$flow.object"}, target_effects=list(place.effects)),
    ]
    edges = RuntimeDataFlowSynthesizer().synthesize(_task(), nodes, registry)
    assert len(edges) == 1
    assert edges[0].type == EdgeType.DATA_FLOW
    assert edges[0].source_step == "step_000"
    assert edges[0].target_step == "step_001"
    assert edges[0].mapping["source_output"] == "object"
    assert edges[0].mapping["target_input"] == "object"
    report = validate_plan_source_closure(
        RuntimePlan(nodes=nodes, edges=edges), registry, _task())
    assert report.passed
    assert report.node_reports[1].resolution_states["object"] == (
        BindingResolutionState.PENDING_DATA_FLOW.value)


def test_two_object_branches_do_not_cross_bind(workspace_tmp):
    acquire = _atomic(
        "generic.acquire", inputs=[{"name": "object", "semantic_type": "object_ref"}],
        outputs=[{"name": "object", "semantic_type": "object_ref",
                  "materializer": {"kind": "input_role", "role": "object"}}],
        preconditions=[], effects=[
            {"predicate": "agent.holds", "args": {"object": "$inputs.object"}}
        ])
    place = _atomic(
        "generic.place", inputs=[{"name": "object", "semantic_type": "object_ref"}],
        outputs=[], preconditions=[
            {"predicate": "agent.holds", "args": {"object": "$inputs.object"}}
        ], effects=[
            {"predicate": "object.at_location",
             "args": {"object": "$inputs.object", "location": "counter"}}
        ])
    registry = _registry(workspace_tmp, acquire, place)
    nodes = [
        PlannedNode(ref=acquire.ref, step_id="a1", branch_id="branch_001",
                    params={"object": "apple_1"}),
        PlannedNode(ref=place.ref, step_id="p1", branch_id="branch_001",
                    params={"object": "$flow.object"}),
        PlannedNode(ref=acquire.ref, step_id="a2", branch_id="branch_002",
                    params={"object": "apple_2"}),
        PlannedNode(ref=place.ref, step_id="p2", branch_id="branch_002",
                    params={"object": "$flow.object"}),
    ]
    edges = RuntimeDataFlowSynthesizer().synthesize(_task(), nodes, registry)
    assert {(edge.source_step, edge.target_step) for edge in edges} == {
        ("a1", "p1"), ("a2", "p2")}


def test_nearest_compatible_producer_wins(workspace_tmp):
    producer = _atomic(
        "generic.producer", inputs=[{"name": "object", "semantic_type": "object_ref"}],
        outputs=[{"name": "object", "semantic_type": "object_ref",
                  "materializer": {"kind": "input_role", "role": "object"}}],
        preconditions=[], effects=[
            {"predicate": "agent.holds", "args": {"object": "$inputs.object"}}
        ])
    consumer = _atomic(
        "generic.consumer", inputs=[{"name": "object", "semantic_type": "object_ref"}],
        outputs=[], preconditions=[
            {"predicate": "agent.holds", "args": {"object": "$inputs.object"}}
        ], effects=[{"predicate": "object.heated",
                     "args": {"object": "$inputs.object"}}])
    registry = _registry(workspace_tmp, producer, consumer)
    nodes = [
        PlannedNode(ref=producer.ref, step_id="old", params={"object": "apple"}),
        PlannedNode(ref=producer.ref, step_id="new", params={"object": "apple"}),
        PlannedNode(ref=consumer.ref, step_id="use", params={"object": "apple"}),
    ]
    edges = RuntimeDataFlowSynthesizer().synthesize(_task(), nodes, registry)
    assert len(edges) == 1
    assert edges[0].source_step == "new"


def test_unmaterializable_output_is_rejected(workspace_tmp):
    producer = _atomic(
        "generic.bad-producer",
        inputs=[{"name": "left", "semantic_type": "object_ref"},
                {"name": "right", "semantic_type": "object_ref"}],
        outputs=[{"name": "result", "semantic_type": "object_ref"}],
        preconditions=[], effects=[
            {"predicate": "object.at_location",
             "args": {"object": "$inputs.left", "location": "$inputs.right"}}
        ])
    consumer = _atomic(
        "generic.consumer", inputs=[{"name": "object", "semantic_type": "object_ref"}],
        outputs=[], preconditions=[
            {"predicate": "agent.holds", "args": {"object": "$inputs.object"}}
        ], effects=[{"predicate": "object.heated",
                     "args": {"object": "$inputs.object"}}])
    registry = _registry(workspace_tmp, producer, consumer)
    validation = validate_output_materializer(producer, "result")
    assert not validation.passed
    nodes = [PlannedNode(ref=producer.ref, step_id="p"),
             PlannedNode(ref=consumer.ref, step_id="c")]
    assert RuntimeDataFlowSynthesizer().synthesize(_task(), nodes, registry) == []


def test_effect_arg_output_materializes_only_after_effect_is_true():
    atomic = _atomic(
        "generic.acquire", inputs=[{"name": "object", "semantic_type": "object_ref"}],
        outputs=[{"name": "object", "semantic_type": "object_ref",
                  "materializer": {"kind": "effect_arg", "predicate": "agent.holds",
                                   "arg": "object"}}],
        preconditions=[], effects=[
            {"predicate": "agent.holds", "args": {"object": "$inputs.object"}}
        ])
    assert materialize_atomic_outputs(
        atomic, {"object": "apple_1"}, {}, {"facts": []}) == {}
    assert materialize_atomic_outputs(
        atomic, {"object": "apple_1"}, {},
        {"facts": ["agent_holds(apple_1)"]}) == {"object": "apple_1"}
