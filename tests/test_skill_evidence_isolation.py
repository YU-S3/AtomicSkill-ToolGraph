import json

import pytest

from atomic_skillgraph.core.config import SystemConfig
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.skill_ir import AbstractAtomicSkill
from atomic_skillgraph.core.status import SkillStatus
from atomic_skillgraph.core.trace_ir import TraceRecord
from atomic_skillgraph.atomicizer.effect_extractor import ExtractedEffect
from atomic_skillgraph.atomicizer.split_score import SplitScoreResult
from atomic_skillgraph.atomicizer.trace_atomicizer import TraceAtomicizer
from atomic_skillgraph.evolution.composite_builder import CompositeBuilder
from atomic_skillgraph.evolution.composite_builder import _portable_target_effects
from atomic_skillgraph.evolution.trace_graph_reconstructor import (
    TraceGraphRevision,
)
from atomic_skillgraph.evidence_store import EvidenceStore
from atomic_skillgraph.graph.registry import SkillGraphRegistry


def _atomic(name, *, inputs, effects, output=""):
    outputs = []
    if output:
        producer = effects[0]
        outputs = [{
            "name": output, "semantic_type": "entity_ref",
            "materializer": {
                "kind": "effect_arg",
                "predicate": producer["predicate"], "arg": output,
            },
        }]
    return AbstractAtomicSkill(
        ref=SkillRef(name, "1.0.0"), summary=name,
        inputs=[{"name": item,
                 "semantic_type": ("location_ref" if "location" in item
                                   else "entity_ref")}
                for item in inputs],
        outputs=outputs, effects=effects, status=SkillStatus.ACTIVE,
    )


def test_composite_persists_only_opaque_trace_evidence_refs(workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "data" / "skill_graph")
    acquire_effect = [{
        "predicate": "agent.holds", "args": {"object": "$inputs.object"}}]
    place_effect = [{
        "predicate": "object.at_location",
        "args": {"object": "$inputs.object",
                 "location": "$inputs.target_location"},
    }]
    acquire = _atomic(
        "generic.acquire", inputs=["object"], effects=acquire_effect,
        output="object")
    place = _atomic(
        "generic.place", inputs=["object", "target_location"],
        effects=place_effect)
    registry.register(acquire)
    registry.register(place)

    private_path = "/home/private/customer_a/report.xlsx"
    private_url = "https://internal.example/customer_a"
    trace = TraceRecord(
        trace_id="trace_private", task_id="customer_a_task_1",
        benchmark="generic", task_type="workflow",
        provenance={
            "params": {"object": "mug", "target_location": "cabinet"},
            "semantic_params": {
                "object": "mug", "target_location": "cabinet"},
            "target_effects": place_effect,
            "environment_instance": private_path,
        },
        success=True,
    )
    segments = [
        {"phase_id": "acquire", "event_start": 1, "event_end": 2,
         "params": {"object": "mug_1"},
         "before": {"facts": ["object_exists(mug_1)"],
                    "workspace": private_path},
         "after": {"facts": ["agent_holds(mug_1)"]},
         "effect": acquire_effect,
         "extraction_method": "llm_proposal_code_validated"},
        {"phase_id": "place", "event_start": 2, "event_end": 3,
         "params": {"object": "mug_1", "target_location": "cabinet_2"},
         "before": {"facts": ["agent_holds(mug_1)"]},
         "after": {"facts": ["object_at(mug_1,cabinet_2)"]},
         "effect": place_effect,
         "extraction_method": "llm_proposal_code_validated"},
    ]
    revision = TraceGraphRevision(
        realized_occurrences=[{
            "skill_ref": str(acquire.ref), "params": {"object": "mug_1"},
            "before": {"path": private_path}, "after": {"url": private_url},
        }],
        canonical_occurrences=[{
            "skill_ref": str(acquire.ref), "params": {"object": "mug_1"},
        }],
        revision_kind="new_capability_insert",
    )
    result = CompositeBuilder(
        registry, SystemConfig(data_dir=workspace_tmp / "data")
    ).build_or_align(
        [acquire.ref, place.ref], trace, segments=segments,
        graph_proposal={
            "validated": True, "summary": f"read {private_path}",
            "implicit_dependencies": [{
                "source_phase_id": "acquire",
                "target_phase_id": "place",
                "reason": f"see {private_url}",
            }],
        },
        revision=revision,
    )
    assert result.composite is not None

    path = (registry.root / "composite" /
            result.composite.ref.logical_id /
            f"{result.composite.ref.version}.json")
    persisted = path.read_text(encoding="utf-8")
    payload = json.loads(persisted)
    for forbidden in (
            "mug_1", "cabinet_2", private_path, private_url,
            '"grounded_params"', '"source_value"', '"target_value"',
            '"source_graph_revision"', '"summary_raw"',
            '"implicit_dependencies_raw"', '"llm_reason_raw"'):
        assert forbidden not in persisted
    assert payload["metadata"]["source_graph_revision_ref"]["evidence_ref"].startswith(
        "evidence://")
    assert payload["metadata"]["independent_support_keys"][0].startswith(
        "support:")
    assert payload["graph"]["steps"][0]["metadata"]["evidence_ref"][
        "evidence_ref"].startswith("evidence://")

    evidence_text = "\n".join(
        item.read_text(encoding="utf-8")
        for item in (workspace_tmp / "data" / "evidence").rglob("*.json"))
    assert "mug_1" in evidence_text
    assert private_path in evidence_text
    assert private_url in evidence_text


@pytest.mark.parametrize("metadata", [
    {"grounded_params": {"object": "mug"}},
    {"note": "/home/private/customer/report.xlsx"},
    {"note": "https://internal.example/private"},
    {"note": "owner@example.com"},
    {"api_key": "sk-secret-secret-secret"},
    {"note": "mug_17"},
    {"customer_ref": "/home/private/customer/report.xlsx"},
    {"occurrence_evidence": {
        "evidence_ref": "evidence://atomic_occurrence/" + "a" * 64,
        "evidence_hash": "b" * 64,
    }},
])
def test_skill_registry_rejects_inline_private_or_grounded_evidence(
        workspace_tmp, metadata):
    registry = SkillGraphRegistry(workspace_tmp / "guard" / "skill_graph")
    skill = AbstractAtomicSkill(
        ref=SkillRef("generic.guarded", "1.0.0"), summary="guarded",
        inputs=[{"name": "object", "semantic_type": "entity_ref"}],
        effects=[{"predicate": "object.changed",
                  "args": {"object": "$inputs.object"}}],
        metadata=metadata, status=SkillStatus.DRAFT,
    )
    with pytest.raises(ValueError, match="long_term_asset_guard_failed"):
        registry.register(skill)


def test_atomic_llm_alias_and_unnumbered_customer_stay_in_evidence(
        workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "atomic_private" / "skill_graph")
    atomicizer = TraceAtomicizer(registry)
    trace = TraceRecord(
        trace_id="trace_customer", benchmark="generic", task_type="workflow",
        success=True)
    segment = {
        "phase_id": "process_customer_a",
        "proposed_intent": "process_customer_a",
        "event_start": 1, "event_end": 2,
        "params": {"object": "customer-a_7"},
        "before": {"path": "/home/private/customer_a/input.json"},
        "after": {"url": "https://internal.example/customer_a"},
    }
    effect = ExtractedEffect(
        positive=[{"predicate": "entity.processed",
                   "args": {"object": "customer-a_7"}}],
        negative=[],
        inputs=[{"name": "object", "semantic_type": "entity_ref"}],
        outputs=[], preconditions=[],
        validator={"pre_checks": [], "post_checks": ["entity.processed"]},
        primary_family="entity_processed",
        suggested_name="entity_processed",
        summary="Verified transition: entity.processed",
    )
    skill = atomicizer._build_atomic_skill(
        trace, segment, effect, SplitScoreResult())
    registry.register(skill)

    path = (registry.root / "abstract_atomic" / skill.ref.logical_id /
            f"{skill.ref.version}.json")
    portable = path.read_text(encoding="utf-8")
    assert "customer_a" not in portable
    assert "customer-a" not in portable
    assert "internal.example" not in portable
    assert skill.metadata["observed_parameter_families"] == {}
    evidence = "\n".join(
        item.read_text(encoding="utf-8")
        for item in (workspace_tmp / "atomic_private" / "evidence").rglob("*.json"))
    assert "process_customer_a" in evidence
    assert "customer-a_7" in evidence


def test_atomic_outputs_are_recomputed_after_unsafe_effect_is_removed(
        workspace_tmp):
    registry = SkillGraphRegistry(
        workspace_tmp / "filtered_effect_output" / "skill_graph")
    atomicizer = TraceAtomicizer(registry)
    trace = TraceRecord(
        trace_id="trace_filtered_effect", benchmark="generic",
        task_type="workflow", success=True)
    segment = {
        "phase_id": "change_object", "event_start": 1, "event_end": 2,
        "params": {"object": "mug_1"},
    }
    effect = ExtractedEffect(
        positive=[{"predicate": "object.changed",
                   "args": {"object": "mug_2"}}],
        inputs=[{"name": "object", "semantic_type": "entity_ref"}],
        outputs=[{
            "name": "object", "semantic_type": "entity_ref",
            "materializer": {"kind": "effect_arg",
                             "predicate": "object.changed",
                             "arg": "object"},
        }],
        validator={"pre_checks": [], "post_checks": ["object.changed"]},
        primary_family="object_changed", suggested_name="object_changed",
    )

    skill = atomicizer._build_atomic_skill(
        trace, segment, effect, SplitScoreResult())

    assert skill.effects == []
    assert skill.outputs == []


def test_composite_target_contract_merges_concrete_and_semantic_params():
    trace = TraceRecord(success=True, provenance={
        "params": {"object": "mug_1", "target_location": "cabinet_2"},
        "semantic_params": {"object": "mug"},
        "target_effects": [{
            "predicate": "object.at_location",
            "args": {"object": "mug_1", "location": "cabinet_2"},
        }],
    })
    effects, errors = _portable_target_effects(trace)
    assert errors == []
    assert effects == [{
        "predicate": "object.at_location",
        "args": {"object": "$inputs.object",
                 "location": "$inputs.target_location"},
    }]


def test_composite_target_contract_canonicalizes_task_placeholder_syntax():
    trace = TraceRecord(success=True, provenance={
        "params": {"object": "bowl", "associated_entity": "lamp"},
        "target_effects": [{
            "predicate": "object.observed_with",
            "args": {"object": "$object",
                     "associated_entity": "$task.associated_entity"},
        }],
    })

    effects, errors = _portable_target_effects(trace)

    assert errors == []
    assert effects == [{
        "predicate": "object.observed_with",
        "args": {"object": "$inputs.object",
                 "associated_entity": "$inputs.associated_entity"},
    }]


def test_composite_target_contract_never_silently_drops_unbound_effect():
    trace = TraceRecord(success=True, provenance={
        "semantic_params": {"object": "mug"},
        "target_effects": [{
            "predicate": "object.at_location",
            "args": {"object": "mug_1", "location": "cabinet_2"},
        }],
    })
    effects, errors = _portable_target_effects(trace)
    assert effects == []
    assert "target_effect_unbound:0" in errors


def test_identical_evidence_retains_every_trace_span_association(workspace_tmp):
    store = EvidenceStore(workspace_tmp / "evidence_associations")
    payload = {"transition": {"before": "ready", "after": "done"}}
    first = store.put(
        "atomic_occurrence", payload, trace_id="trace_a",
        event_start=2, event_end=4)
    second = store.put(
        "atomic_occurrence", payload, trace_id="trace_b",
        event_start=7, event_end=9)

    assert first == second
    assert store.get(first) == payload
    assert store.get_associations(first) == [
        {"trace_id": "trace_a", "event_start": 2, "event_end": 4},
        {"trace_id": "trace_b", "event_start": 7, "event_end": 9},
    ]


def test_aligned_composite_keeps_occurrence_evidence_from_both_traces(
        workspace_tmp):
    root = workspace_tmp / "composite_multi_evidence"
    registry = SkillGraphRegistry(root / "skill_graph")
    acquire = _atomic(
        "generic.acquire", inputs=["object"], output="object",
        effects=[{"predicate": "agent.holds",
                  "args": {"object": "$inputs.object"}}])
    place = _atomic(
        "generic.place", inputs=["object", "target_location"],
        effects=[{"predicate": "object.at_location", "args": {
            "object": "$inputs.object",
            "location": "$inputs.target_location",
        }}])
    registry.register(acquire)
    registry.register(place)
    builder = CompositeBuilder(registry, SystemConfig(data_dir=root))

    def build(trace_id, task_id, object_value, location_value):
        trace = TraceRecord(
            trace_id=trace_id, task_id=task_id, benchmark="generic",
            task_type="workflow", success=True,
            provenance={
                "params": {"object": object_value,
                           "target_location": location_value},
                "target_effects": [{
                    "predicate": "object.at_location", "args": {
                        "object": object_value, "location": location_value,
                    },
                }],
            },
        )
        segments = [
            {
                "phase_id": "phase_a", "event_start": 1, "event_end": 2,
                "params": {"object": object_value},
                "before": {"seen": object_value},
                "after": {"held": object_value},
                "effect": list(acquire.effects),
                "extraction_method": "llm_proposal_code_validated",
            },
            {
                "phase_id": "phase_b", "event_start": 3, "event_end": 4,
                "params": {"object": object_value,
                           "target_location": location_value},
                "before": {"held": object_value},
                "after": {"placed": location_value},
                "effect": list(place.effects),
                "extraction_method": "llm_proposal_code_validated",
            },
        ]
        return builder.build_or_align(
            [acquire.ref, place.ref], trace, segments=segments,
            graph_proposal={"validated": True})

    first = build("trace_a", "task_a", "mug_1", "cabinet_1")
    second = build("trace_b", "task_b", "plate_2", "shelf_3")
    assert first.composite is not None
    assert second.composite is not None
    assert second.decision == "reuse"

    groups = second.composite.metadata["occurrence_evidence_refs"]
    assert {item["source_trace_id"] for item in groups} == {
        "trace_a", "trace_b"}
    pointers = [
        step["evidence_ref"]
        for group in groups for step in group["steps"]
    ]
    assert len(pointers) == 4
    assert all(registry.evidence_store.get(pointer) is not None
               for pointer in pointers)
