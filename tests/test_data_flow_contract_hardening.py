from types import SimpleNamespace

import pytest

from atomic_skillgraph.atomicizer.effect_extractor import extract_effect
from atomic_skillgraph.core.edge_ir import GraphEdge
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.skill_ir import AbstractAtomicSkill, CompositeSkill
from atomic_skillgraph.core.status import EdgeType, SkillStatus
from atomic_skillgraph.graph.registry import SkillGraphRegistry
from atomic_skillgraph.runtime.output_materializer import (
    validate_output_materializer,
)
from atomic_skillgraph.runtime.plan_validator import (
    validate_composite_binding_closure,
    validate_plan_source_closure,
)
from atomic_skillgraph.runtime.runtime_graph import PlannedNode, RuntimePlan


def _producer(logical_id: str, *, semantic_type: str = "entity_ref"):
    return AbstractAtomicSkill(
        ref=SkillRef(logical_id, "1.0.0"),
        summary=logical_id,
        inputs=[{"name": "object", "semantic_type": semantic_type}],
        outputs=[{
            "name": "object",
            "semantic_type": semantic_type,
            "materializer": {
                "kind": "effect_arg",
                "predicate": "agent.holds",
                "arg": "object",
            },
        }],
        effects=[{
            "predicate": "agent.holds",
            "args": {"object": "$inputs.object"},
        }],
        status=SkillStatus.ACTIVE,
    )


def _consumer(*, semantic_type: str = "entity_ref"):
    return AbstractAtomicSkill(
        ref=SkillRef("generic.consume", "1.0.0"),
        summary="consume",
        inputs=[{"name": "object", "semantic_type": semantic_type}],
        outputs=[],
        preconditions=[{
            "predicate": "agent.holds",
            "args": {"object": "$inputs.object"},
        }],
        effects=[{
            "predicate": "object.changed",
            "args": {"object": "$inputs.object"},
        }],
        status=SkillStatus.ACTIVE,
    )


def _edge(source, target, *, source_step: str, target_input: str = "object",
          transform: str = "identity", scope: str = "composite"):
    return GraphEdge(
        source=str(source.ref),
        target=str(target.ref),
        type=EdgeType.DATA_FLOW,
        scope=scope,
        source_step=source_step,
        target_step="consume",
        mapping={
            "source_output": "object",
            "target_input": target_input,
            "transform": transform,
        },
    )


def _validate(workspace_tmp, path: str, *, duplicate: bool = False,
              target_input: str = "object", source_type: str = "entity_ref",
              target_type: str = "entity_ref", transform: str = "identity"):
    registry = SkillGraphRegistry(workspace_tmp / f"data-flow-{path}")
    first = _producer("generic.produce-first", semantic_type=source_type)
    second = _producer("generic.produce-second", semantic_type=source_type)
    target = _consumer(semantic_type=target_type)
    for atomic in (first, second, target):
        registry.register(atomic)

    edges = [_edge(
        first, target, source_step="first", target_input=target_input,
        transform=transform, scope=path)]
    if duplicate:
        edges.append(_edge(
            second, target, source_step="second", target_input=target_input,
            transform=transform, scope=path))

    if path == "composite":
        # Bind the real input from the task when the tested edge deliberately
        # targets a nonexistent slot.  That isolates edge-schema rejection
        # from the ordinary unresolved-semantic-input check.
        target_binding = ("$task.object" if target_input != "object"
                          else "$flow.object")
        composite = CompositeSkill(
            ref=SkillRef("composite.data-flow-contract", "1.0.0"),
            graph={
                "steps": [
                    {"step_id": "first", "node_ref": str(first.ref),
                     "params": {"object": "$task.object"}},
                    {"step_id": "second", "node_ref": str(second.ref),
                     "params": {"object": "$task.object"}},
                    {"step_id": "consume", "node_ref": str(target.ref),
                     "params": {"object": target_binding}},
                ],
                "nodes": [str(first.ref), str(second.ref), str(target.ref)],
                "data": [edge.to_dict() for edge in edges],
            },
            status=SkillStatus.DRAFT,
        )
        return validate_composite_binding_closure(composite, registry)

    target_binding = "apple_1" if target_input != "object" else "$flow.object"
    plan = RuntimePlan(
        nodes=[
            PlannedNode(ref=first.ref, step_id="first",
                        params={"object": "apple_1"}),
            PlannedNode(ref=second.ref, step_id="second",
                        params={"object": "apple_1"}),
            PlannedNode(ref=target.ref, step_id="consume",
                        params={"object": target_binding},
                        target_effects=list(target.effects)),
        ],
        edges=edges,
    )
    task = SimpleNamespace(context={"params": {"object": "apple_1"}})
    return validate_plan_source_closure(plan, registry, task)


@pytest.mark.parametrize("path", ["composite", "runtime"])
def test_conflicting_data_flow_sources_are_rejected(workspace_tmp, path):
    report = _validate(workspace_tmp, path, duplicate=True)

    assert not report.passed
    assert "conflicting_data_flow_sources:consume:object" in report.errors


@pytest.mark.parametrize("path", ["composite", "runtime"])
def test_data_flow_target_must_be_a_declared_atomic_input(workspace_tmp, path):
    report = _validate(
        workspace_tmp, path, target_input="nonexistent_slot")

    assert not report.passed
    assert "data_flow_target_input_missing:consume:nonexistent_slot" in report.errors


@pytest.mark.parametrize("path", ["composite", "runtime"])
def test_data_flow_semantic_type_mismatch_is_rejected(workspace_tmp, path):
    report = _validate(
        workspace_tmp, path, source_type="entity_ref",
        target_type="location_ref")

    assert not report.passed
    assert any(error.startswith("data_flow_schema_mismatch:")
               for error in report.errors)


@pytest.mark.parametrize("path", ["composite", "runtime"])
@pytest.mark.parametrize(("source_type", "target_type"), [
    ("", "location_ref"),
    ("entity_ref", ""),
])
def test_data_flow_missing_semantic_type_fails_closed(
        workspace_tmp, path, source_type, target_type):
    report = _validate(
        workspace_tmp, path, source_type=source_type,
        target_type=target_type)

    assert not report.passed
    assert any(error.startswith("data_flow_schema_missing:")
               for error in report.errors)


@pytest.mark.parametrize("path", ["composite", "runtime"])
def test_unimplemented_data_flow_transform_is_rejected(workspace_tmp, path):
    report = _validate(
        workspace_tmp, path, transform="extract_field")

    assert not report.passed
    assert any(error.startswith("unsupported_data_flow_transform:")
               for error in report.errors)


def test_effect_arg_materializer_cannot_relabel_its_input_type():
    atomic = _producer("generic.mislabeled-output")
    atomic.outputs[0]["semantic_type"] = "location_ref"

    validation = validate_output_materializer(atomic, "object")

    assert not validation.passed
    assert "materializer_semantic_type_mismatch:entity_ref->location_ref" in (
        validation.errors)


def test_extract_effect_declares_an_explicit_effect_arg_materializer():
    effect = extract_effect(
        {}, {}, {"object": "apple_1"},
        positive_effects=[{
            "predicate": "agent.holds",
            "args": {"object": "apple_1"},
        }],
        negative_effects=[],
    )

    assert effect.outputs == [{
        "name": "object",
        "semantic_type": "entity_ref",
        "materializer": {
            "kind": "effect_arg",
            "predicate": "agent.holds",
            "arg": "object",
        },
    }]


def test_extract_effect_omits_same_role_with_conflicting_values():
    effect = extract_effect(
        {}, {}, {"first": "apple_1", "second": "mug_1"},
        positive_effects=[
            {"predicate": "first.selected", "args": {"object": "apple_1"}},
            {"predicate": "second.selected", "args": {"object": "mug_1"}},
        ],
        negative_effects=[],
    )

    assert all(output.get("name") != "object" for output in effect.outputs)
