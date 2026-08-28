from atomic_skillgraph.core.edge_ir import GraphEdge
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.skill_ir import AbstractAtomicSkill, CompositeSkill
from atomic_skillgraph.core.status import EdgeType, SkillStatus
from atomic_skillgraph.graph.registry import SkillGraphRegistry
from atomic_skillgraph.runtime.plan_validator import validate_composite_binding_closure
from atomic_skillgraph.evolution.composite_lifecycle import evaluate_composite


def _atomic(ref: str, effect: str, *, inputs, outputs):
    return AbstractAtomicSkill(
        ref=SkillRef(ref, "1.0.0"), summary=ref,
        inputs=[{"name": name, "semantic_type": "object_ref"} for name in inputs],
        outputs=[{"name": name, "semantic_type": "object_ref"} for name in outputs],
        effects=[{"predicate": effect,
                  "args": {"object": "$inputs.object"}}],
        status=SkillStatus.ACTIVE)


def _composite(acquire, place, *, flow=True, status=SkillStatus.DRAFT):
    steps = [
        {"step_id": "acquire_1", "node_ref": str(acquire.ref),
         "params": {"object": "$task.object"}},
        {"step_id": "place_1", "node_ref": str(place.ref),
         "params": {"object": "$flow.held_object"}},
    ]
    edges = [GraphEdge(
        source=str(acquire.ref), target=str(place.ref), type=EdgeType.NEXT,
        scope="composite", source_step="acquire_1", target_step="place_1").to_dict()]
    data = ([GraphEdge(
        source=str(acquire.ref), target=str(place.ref), type=EdgeType.DATA_FLOW,
        scope="composite", source_step="acquire_1", target_step="place_1",
        mapping={"source_output": "held_object", "target_input": "object"},
    ).to_dict()] if flow else [])
    return CompositeSkill(
        ref=SkillRef("composite.acquire-place", "1.0.0"),
        graph={"steps": steps, "nodes": [str(acquire.ref), str(place.ref)],
               "control": edges, "data": data},
        validator={"target_effects": list(place.effects)},
        metadata={"statistics": {"support_count": 2}}, status=status)


def test_flow_binding_requires_exact_existing_output(workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "graph")
    acquire = _atomic("generic.acquire", "agent.holds",
                      inputs=["object"], outputs=["held_object"])
    place = _atomic("generic.place", "object.at_location",
                    inputs=["object"], outputs=[])
    registry.register(acquire)
    registry.register(place)
    valid = _composite(acquire, place)
    assert validate_composite_binding_closure(valid, registry).passed
    valid.graph["data"][0]["mapping"]["source_output"] = "missing_output"
    assert not validate_composite_binding_closure(valid, registry).passed


def test_draft_promotes_only_after_closure_and_active_children(workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "lifecycle")
    acquire = _atomic("generic.acquire", "agent.holds",
                      inputs=["object"], outputs=["held_object"])
    place = _atomic("generic.place", "object.at_location",
                    inputs=["object"], outputs=[])
    registry.register(acquire)
    registry.register(place)
    candidate = _composite(acquire, place)
    assert evaluate_composite(candidate, registry, min_support=2).status == SkillStatus.ACTIVE
    registry.set_status(acquire.ref, SkillStatus.SUPPRESSED)
    assert evaluate_composite(candidate, registry, min_support=2).status == SkillStatus.SHADOW


def test_draft_registration_does_not_replace_active_recommendation(workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "recommendation")
    active = _atomic("generic.acquire", "agent.holds",
                     inputs=["object"], outputs=["held_object"])
    registry.register(active)
    draft = _atomic("generic.acquire", "agent.holds",
                    inputs=["object"], outputs=["held_object"])
    draft.ref = SkillRef("generic.acquire", "1.1.0")
    draft.status = SkillStatus.DRAFT
    registry.register(draft)
    assert registry.get_latest("generic.acquire").ref == draft.ref
    assert registry.get_recommended("generic.acquire").ref == active.ref
