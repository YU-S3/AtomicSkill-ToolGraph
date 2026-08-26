from __future__ import annotations

import copy

import pytest

from atomic_skillgraph.core.trace_ir import ActionRecord, TraceRecord
from atomic_skillgraph.core.refs import ToolRef
from atomic_skillgraph.core.status import ArtifactKind
from atomic_skillgraph.core.tool_ir import ToolAsset
from atomic_skillgraph.tools.admission_adapter import AdmissionEngine
from atomic_skillgraph.tools.compiler_adapter import (
    AtomicSegment,
    mine_action_template_tools,
)
from atomic_skillgraph.tools.registry import ToolRegistry


def _trace(trace_id: str, action_names: list[str]) -> TraceRecord:
    return TraceRecord(
        trace_id=trace_id,
        task_id=trace_id,
        task_type="pick_heat_then_place_in_recep",
        benchmark="alfworld",
        success=True,
        actions=[ActionRecord(step=index, name=name)
                 for index, name in enumerate(action_names)],
    )


def _acquire_segment(object_name: str, source: str,
                     actions: list[str]) -> AtomicSegment:
    return AtomicSegment(
        name="acquire_object",
        kind="env",
        actions=[{"step": index, "name": name, "params": {}}
                 for index, name in enumerate(actions)],
        params={"object": object_name, "object_location": source},
        effect=[{"predicate": "agent.holds",
                 "args": {"object": object_name}}],
        entry_event_index=0,
        causal_event_indices=list(range(len(actions))),
        effect_producer_indices=[len(actions) - 1],
        event_slice_validated=True,
        replay_safe=True,
    )


def test_action_tool_identity_is_instance_free_but_shape_specific():
    surface_actions = [
        "go to countertop 1",
        "take mug 1 from countertop 1",
    ]
    same_shape_actions = [
        "go to countertop 2",
        "take cup 1 from countertop 2",
    ]
    closed_actions = [
        "go to cabinet 1",
        "open cabinet 1",
        "take mug 1 from cabinet 1",
    ]
    surface = mine_action_template_tools(
        _trace("surface", surface_actions),
        [_acquire_segment("mug 1", "countertop 1", surface_actions)],
    )[0]
    same_shape = mine_action_template_tools(
        _trace("same_shape", same_shape_actions),
        [_acquire_segment("cup 1", "countertop 2", same_shape_actions)],
    )[0]
    closed = mine_action_template_tools(
        _trace("closed", closed_actions),
        [_acquire_segment("mug 1", "cabinet 1", closed_actions)],
    )[0]

    assert surface.tool_id == same_shape.tool_id
    assert surface.structural_hash() == same_shape.structural_hash()
    assert closed.tool_id != surface.tool_id
    assert closed.artifact["steps"] == [
        "go to {object_location}",
        "open {object_location}",
        "take {object} from {object_location}",
    ]


def test_action_parameterization_accepts_underscore_binding_for_space_action():
    actions = ["heat mug 1 with microwave 1"]
    segment = AtomicSegment(
        name="heat_object", kind="env",
        actions=[{"step": 0, "name": actions[0], "params": {}}],
        params={"object": "mug_1", "heating_station": "microwave_1"},
        effect=[{"predicate": "object.heated", "args": {"object": "mug_1"}}],
        entry_event_index=0, causal_event_indices=[0], effect_producer_indices=[0],
        event_slice_validated=True, replay_safe=True,
    )
    tool = mine_action_template_tools(_trace("underscores", actions), [segment])[0]
    assert tool.artifact["steps"] == ["heat {object} with {heating_station}"]
    assert set(tool.param_names()) == {"object", "heating_station"}


def test_compiler_restores_state_grounded_location_without_capability_catalogue():
    trace = TraceRecord(
        trace_id="generic_location_contract", task_id="generic",
        task_type="unseen_task", benchmark="generic_env", success=True,
        actions=[
            ActionRecord(step=0, name="move near bay 7", accepted=True),
            ActionRecord(step=1, name="engage fixture 3", accepted=True),
        ],
        state_snapshots=[
            {"step": 0, "state": {"facts": [
                "agent_at(bay_1)", "object_at(fixture_3, bay_7)"]}},
            {"step": 1, "state": {"facts": [
                "agent_at(bay_7)", "object_at(fixture_3, bay_7)"]}},
            {"step": 2, "state": {"facts": [
                "agent_at(bay_7)", "object_at(fixture_3, bay_7)",
                "fixture_engaged(fixture_3)"]}},
        ],
    )
    segment = AtomicSegment(
        name="operate_fixture", kind="env",
        actions=[{"step": 1, "name": "engage fixture 3", "params": {}}],
        params={"fixture": "fixture_3"},
        before={"facts": [
            "agent_at(bay_7)", "object_at(fixture_3, bay_7)"]},
        after={"facts": [
            "agent_at(bay_7)", "object_at(fixture_3, bay_7)",
            "fixture_engaged(fixture_3)"]},
        effect=[{"predicate": "fixture.engaged",
                 "args": {"object": "fixture_3"}}],
        entry_event_index=1, causal_event_indices=[1],
        effect_producer_indices=[1], event_slice_validated=True,
        replay_safe=True,
    )

    tool = mine_action_template_tools(trace, [segment])[0]

    assert tool.artifact["steps"] == [
        "move near {fixture_location}", "engage {fixture}"]
    assert set(tool.param_names()) == {"fixture", "fixture_location"}
    assert tool.tests[0]["bindings"]["fixture_location"] == "bay_7"
    assert tool.tests[0]["prefix"] == []


def test_action_admission_rejects_concrete_instance_residue_before_replay():
    replay_called = False

    def replay(*_args):
        nonlocal replay_called
        replay_called = True
        return {"passed": True}

    tool = ToolAsset(
        ref=ToolRef("alfworld.bad_concrete", "1.0.0"),
        artifact_kind=ArtifactKind.ACTION_TEMPLATE,
        summary="bad concrete heat",
        signature={"parameters": [{"name": "object"},
                                   {"name": "heating_station"}]},
        artifact={"steps": ["heat mug 1 with {heating_station}"]},
        tests=[{"kind": "replay", "bindings": {
            "object": "mug_1", "heating_station": "microwave_1"}}],
    )
    result = AdmissionEngine(replay_fn=replay).admit(tool)
    assert not result.passed
    assert not result.checks["instance_free_template"]
    assert any(reason.startswith("concrete_instance_literals:")
               for reason in result.reasons)
    assert "unused_declared_parameters: ['object']" in result.reasons
    assert not replay_called


def test_tool_registry_rejects_same_version_with_different_executable(
        workspace_tmp):
    actions = ["go to countertop 1", "take mug 1 from countertop 1"]
    tool = mine_action_template_tools(
        _trace("immutable_tool", actions),
        [_acquire_segment("mug 1", "countertop 1", actions)],
    )[0]
    registry = ToolRegistry(workspace_tmp / "tools")
    registry.register(tool)
    replacement = copy.deepcopy(tool)
    replacement.artifact = {
        "template": "go to {object_location}\nopen {object_location}\n"
                    "take {object} from {object_location}",
        "steps": ["go to {object_location}", "open {object_location}",
                  "take {object} from {object_location}"],
    }

    with pytest.raises(ValueError, match="immutable_tool_version_collision"):
        registry.register(replacement)
    assert registry.get(tool.ref).artifact == tool.artifact
