from types import SimpleNamespace

from atomic_skillgraph.core.edge_ir import GraphEdge
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.skill_ir import AbstractAtomicSkill, CompositeSkill
from atomic_skillgraph.core.status import EdgeType, SkillStatus
from atomic_skillgraph.core.trace_ir import (
    NodeValidationResult,
    RuntimeSpan,
    TaskGapAnalysis,
    TraceRecord,
)
from atomic_skillgraph.evolution.success_processor import _revision_build_inputs
from atomic_skillgraph.evolution.trace_graph_reconstructor import (
    TraceGraphReconstructor,
)
from atomic_skillgraph.graph.registry import SkillGraphRegistry


def _atomic(name, *, preconditions=(), effects=(), negative=()):
    inputs = [
        {"name": "object", "semantic_type": "entity_ref"},
        {"name": "target_location", "semantic_type": "location_ref"},
    ]
    outputs = []
    for effect in effects:
        for arg in (effect.get("args") or {}):
            if not any(item["name"] == arg for item in outputs):
                outputs.append({
                    "name": arg,
                    "semantic_type": (
                        "location_ref" if "location" in arg else "entity_ref"),
                    "materializer": {
                        "kind": "effect_arg",
                        "predicate": effect["predicate"],
                        "arg": arg,
                    },
                })
    return AbstractAtomicSkill(
        ref=SkillRef(name, "1.0.0"), summary=name, inputs=inputs,
        outputs=outputs, preconditions=list(preconditions), effects=list(effects),
        metadata={"observed_negative_effects": list(negative)},
        status=SkillStatus.ACTIVE,
    )


def _effect(predicate, **args):
    return {"predicate": predicate,
            "args": {key: f"$inputs.{value}" for key, value in args.items()}}


def test_task_gap_recovery_is_canonicalized_to_causal_subsequence(workspace_tmp):
    exists = _effect("object.exists", object="object")
    holds = _effect("agent.holds", object="object")
    heated = _effect("object.heated", object="object")
    placed = _effect(
        "object.at_location", object="object", location="target_location")
    not_holds = {"not": holds}
    not_placed = {"not": placed}

    acquire = _atomic(
        "cap.acquire", preconditions=[exists], effects=[holds],
        negative=[not_placed])
    place = _atomic(
        "cap.place", preconditions=[holds], effects=[placed],
        negative=[not_holds])
    heat = _atomic(
        "cap.heat", preconditions=[holds], effects=[heated])
    registry = SkillGraphRegistry(workspace_tmp / "canonical_graph")
    for atomic in (acquire, place, heat):
        registry.register(atomic)

    parent = CompositeSkill(
        ref=SkillRef("composite.incomplete", "1.0.0"), summary="parent",
        graph={
            "nodes": [str(acquire.ref), str(place.ref)],
            "steps": [
                {"step_id": "old_acquire", "node_ref": str(acquire.ref),
                 "params": {"object": "$task.object"}},
                {"step_id": "old_place", "node_ref": str(place.ref),
                 "params": {"object": "$flow.object",
                            "target_location": "$task.target_location"}},
            ],
            "control": [GraphEdge(
                source=str(acquire.ref), target=str(place.ref),
                type=EdgeType.NEXT, scope="composite",
                source_step="old_acquire", target_step="old_place").to_dict()],
            "data": [GraphEdge(
                source=str(acquire.ref), target=str(place.ref),
                type=EdgeType.DATA_FLOW, scope="composite",
                source_step="old_acquire", target_step="old_place",
                mapping={"source_output": "object",
                         "target_input": "object",
                         "transform": "identity"}).to_dict()],
        },
        status=SkillStatus.ACTIVE,
    )
    registry.register(parent)

    params = {"object": "mug_1", "target_location": "cabinet_1"}
    before = {"facts": ["object_exists(mug_1)"]}
    trace = TraceRecord(
        trace_id="trace_canonical", success=True,
        selected_composite=str(parent.ref),
        provenance={"params": params, "realized_params": params,
                    "target_effects": [heated, placed]},
        task_gap_analysis=TaskGapAnalysis(missing_effects=[heated]),
        runtime_spans=[
            RuntimeSpan(kind="planned_node", occurrence_id="old_a",
                        action_start=0, action_end=1),
            RuntimeSpan(kind="planned_node", occurrence_id="old_p",
                        action_start=1, action_end=2),
            RuntimeSpan(kind="task_gap", occurrence_id="gap",
                        action_start=2, action_end=5,
                        missing_effects=[heated]),
        ],
        realized_atomic_nodes=[
            {"ref": str(acquire.ref), "origin_step_id": "old_acquire",
             "occurrence_id": "old_a", "passed": True, "params": params,
             "before": before, "after": {}},
            {"ref": str(place.ref), "origin_step_id": "old_place",
             "occurrence_id": "old_p", "passed": True, "params": params,
             "before": {}, "after": {}},
            {"ref": "skill://runtime.dynamic.task_gap@0.0.0",
             "occurrence_id": "gap", "passed": True},
        ],
        node_validators=[NodeValidationResult(
            node_ref="skill://runtime.dynamic.task_gap@0.0.0",
            level="task_gap", passed=True, occurrence_id="gap")],
    )

    gap_segments = [
        {"phase_id": "again_acquire", "source_kind": "task_gap",
         "runtime_occurrence_id": "gap", "params": params,
         "before": before, "effect": [holds],
         "negative_effect": [not_placed]},
        {"phase_id": "heat", "source_kind": "task_gap",
         "runtime_occurrence_id": "gap", "params": params,
         "before": {"facts": ["agent_holds(mug_1)"]},
         "preconditions": [holds], "effect": [heated]},
        {"phase_id": "final_place", "source_kind": "task_gap",
         "runtime_occurrence_id": "gap", "params": params,
         "before": {"facts": ["agent_holds(mug_1)"]},
         "preconditions": [holds], "effect": [placed],
         "negative_effect": [not_holds]},
    ]
    nodes = [acquire, heat, place]
    atomic_result = SimpleNamespace(
        candidates=[SimpleNamespace(
            skill=node, segment=segment,
            alignment=SimpleNamespace(matched=True))
            for node, segment in zip(nodes, gap_segments)],
        segments=gap_segments, decisions=["reuse", "reuse", "reuse"],
    )

    revision = TraceGraphReconstructor(registry).reconstruct(
        trace=trace, atomic_result=atomic_result,
        selected_composite=parent)

    assert revision.revision_kind == "repeated_occurrence_insert"
    assert [item["phase_id"] for item in revision.realized_occurrences] == [
        "parent_old_acquire", "parent_old_place",
        "again_acquire", "heat", "final_place"]
    assert [item["phase_id"] for item in revision.canonical_occurrences] == [
        "again_acquire", "heat", "final_place"]
    refs, segments = _revision_build_inputs(revision, registry)
    assert refs == [acquire.ref, heat.ref, place.ref]
    assert [item["phase_id"] for item in segments] == [
        "again_acquire", "heat", "final_place"]
