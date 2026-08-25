import copy
import json

import pytest

from atomic_skillgraph.atomicizer.semantic_extractor import (
    EXTRACTOR_PROMPT,
    SemanticExtractorAgent,
    build_structured_events,
    causal_slice,
    slice_event_occurrence,
    validate_phase_proposal,
)
from atomic_skillgraph.atomicizer.trace_atomicizer import TraceAtomicizer
from atomic_skillgraph.core.config import SystemConfig
from atomic_skillgraph.core.llm import LLMUsage
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.skill_ir import AbstractAtomicSkill
from atomic_skillgraph.core.status import SkillStatus
from atomic_skillgraph.core.trace_ir import ActionRecord, TraceRecord
from atomic_skillgraph.evolution.composite_builder import CompositeBuilder, _role_params
from atomic_skillgraph.evolution.insight_updater import InsightUpdater
from atomic_skillgraph.graph.registry import SkillGraphRegistry


class _ExtractorLLM:
    usage = LLMUsage()

    def generate(self, **_kwargs):
        payload = {
            "phases": [
                {"phase_id": "p0", "intent": "acquire_object", "event_start": 0,
                 "event_end": 0, "parameter_roles": {"object": "mug_1",
                 "object_location": "countertop_1"},
                 "effect_predicates": ["agent.holds"]},
                {"phase_id": "p1", "intent": "heat_object", "event_start": 1,
                 "event_end": 3, "parameter_roles": {"object": "mug_1",
                 "heating_station": "microwave_1"},
                 "effect_predicates": ["object.heated"]},
                {"phase_id": "p2", "intent": "place_object", "event_start": 4,
                 "event_end": 4, "parameter_roles": {"object": "mug_1",
                 "target_location": "cabinet_1"},
                 "effect_predicates": ["object.at_location"]},
            ],
            "discarded_event_indices": [], "workflow_summary": "acquire, heat, place",
        }
        return type("Response", (), {"text": json.dumps(payload)})()


class _ExtractorLLMWithNoncanonicalEffect(_ExtractorLLM):
    def generate(self, **kwargs):
        response = super().generate(**kwargs)
        payload = json.loads(response.text)
        payload["phases"][0]["effect_predicates"] = ["object.acquired"]
        return type("Response", (), {"text": json.dumps(payload)})()


def _trace(trace_id="trace_one"):
    states = [
        {"facts": ["object_at(mug_1, countertop_1)"], "inventory": []},
        {"facts": ["agent_holds(mug_1)"], "inventory": ["mug_1"]},
        {"facts": ["object_at(mug_1, microwave_1)"], "inventory": []},
        {"facts": ["agent_holds(mug_1)"], "inventory": ["mug_1"]},
        {"facts": ["agent_holds(mug_1)", "object_heated(mug_1)"], "inventory": ["mug_1"]},
        {"facts": ["object_heated(mug_1)", "object_at(mug_1, cabinet_1)"], "inventory": []},
    ]
    actions = [
        ("take mug 1 from countertop 1", {"object": "mug_1", "object_location": "countertop_1"}),
        ("move mug 1 to microwave 1", {"object": "mug_1", "target_location": "microwave_1"}),
        ("take mug 1 from microwave 1", {"object": "mug_1", "object_location": "microwave_1"}),
        ("heat mug 1 with microwave 1", {"object": "mug_1", "heating_station": "microwave_1"}),
        ("move mug 1 to cabinet 1", {"object": "mug_1", "target_location": "cabinet_1"}),
    ]
    return TraceRecord(
        trace_id=trace_id, task_id="t", task_type="heat_place", task_goal="heat and place mug",
        benchmark="env", success=True,
        actions=[ActionRecord(step=i, name=name, params=params) for i, (name, params) in enumerate(actions)],
        state_snapshots=[{"step": i, "state": state} for i, state in enumerate(states)],
        provenance={"params": {"object": "mug", "object_location": "countertop_1",
                               "heating_station": "microwave_1", "target_location": "cabinet_1"},
                    "target_effects": [
                        {"predicate": "object.heated", "args": {"object": "$inputs.object"}},
                        {"predicate": "object.at_location", "args": {"object": "$inputs.object",
                                                                        "location": "$inputs.target_location"}},
                    ]})


def _atomic(logical_id, preconditions, effects, inputs, outputs):
    return AbstractAtomicSkill(
        ref=SkillRef(logical_id, "1.0.0"), summary=logical_id, inputs=inputs, outputs=outputs,
        preconditions=preconditions, effects=effects,
        validator={"post_checks": [item["predicate"] for item in effects]},
        status=SkillStatus.ACTIVE)


def test_llm_macro_phase_hides_internal_place_acquire():
    result = SemanticExtractorAgent(_ExtractorLLM()).extract(_trace())
    assert result.method == "llm_proposal_code_validated"
    assert [phase["name"] for phase in result.phases] == [
        "acquire_object", "heat_object", "place_object"]
    assert (result.phases[1]["event_start"], result.phases[1]["event_end"]) == (3, 3)
    assert result.phases[1]["causal_event_indices"] == [3]
    assert result.phases[1]["removed_internal_event_indices"] == [1, 2]
    assert result.phases[1]["params"]["heating_station"] == "microwave_1"
    assert "target_location" not in result.phases[1]["params"]
    assert result.proposal["workflow_summary"] == "acquire, heat, place"
    assert len(result.validated_phases) == 3
    assert result.slice_diagnostics["unresolved_requirements"] == []
    assert result.to_dict()["phases"][1]["retained_actions"] == [
        "heat mug 1 with microwave 1"]


def test_causal_slice_treats_observed_object_origin_as_exogenous():
    phases = [
        {
            "phase_id": "acquire", "params": {"object": "mug_1",
                                                  "object_location": "cabinet_4"},
            "before": {"facts": ["object_exists(mug_1)",
                                   "object_at(mug_1, cabinet_4)"], "inventory": []},
            "after": {"facts": ["object_exists(mug_1)", "agent_holds(mug_1)"],
                      "inventory": ["mug_1"]},
            "preconditions": [
                {"predicate": "object.exists", "args": {"object": "$inputs.object"}},
                {"predicate": "object.at_location",
                 "args": {"object": "$inputs.object",
                          "location": "$inputs.object_location"}},
            ],
            "effect": [{"predicate": "agent.holds", "args": {"object": "mug_1"}}],
            "validation": {},
        },
        {
            "phase_id": "heat", "params": {"object": "mug_1",
                                               "heating_station": "microwave_1"},
            "before": {"facts": ["agent_holds(mug_1)"], "inventory": ["mug_1"]},
            "after": {"facts": ["agent_holds(mug_1)", "object_heated(mug_1)"],
                      "inventory": ["mug_1"]},
            "preconditions": [
                {"predicate": "agent.holds", "args": {"object": "$inputs.object"}}],
            "effect": [{"predicate": "object.heated", "args": {"object": "mug_1"}}],
            "validation": {},
        },
        {
            "phase_id": "place", "params": {"object": "mug_1",
                                                "target_location": "cabinet_5"},
            "before": {"facts": ["agent_holds(mug_1)", "object_heated(mug_1)"],
                       "inventory": ["mug_1"]},
            "after": {"facts": ["object_heated(mug_1)",
                                  "object_at(mug_1, cabinet_5)"], "inventory": []},
            "preconditions": [
                {"predicate": "agent.holds", "args": {"object": "$inputs.object"}}],
            "effect": [{"predicate": "object.at_location",
                        "args": {"object": "mug_1", "location": "cabinet_5"}}],
            "validation": {},
        },
    ]
    selected, diagnostics = causal_slice(
        phases,
        [{"predicate": "object.heated", "args": {"object": "$inputs.object"}},
         {"predicate": "object.at_location",
          "args": {"object": "$inputs.object", "location": "$inputs.target_location"}}],
        {"facts": [], "inventory": []},
        # Goal contract is a receptacle family; the validated occurrence keeps
        # the concrete successful instance cabinet_5.
        task_params={"object": "mug", "target_location": "cabinet"},
    )

    assert [phase["phase_id"] for phase in selected] == ["acquire", "heat", "place"]
    assert diagnostics["unresolved_requirements"] == []
    assert {item["predicate"]["predicate"]
            for item in diagnostics["external_preconditions"]} == {
                "object.exists", "object.at_location"}
    assert all(item["source"] == "observed_before_phase"
               for item in diagnostics["external_preconditions"])
    assert {(item["source_phase_id"], item["target_phase_id"])
            for item in diagnostics["dependencies"]} == {
                ("acquire", "heat"), ("acquire", "place")}


def test_state_effect_evidence_is_not_erased_by_llm_label_mismatch():
    result = SemanticExtractorAgent(_ExtractorLLMWithNoncanonicalEffect()).extract(_trace())
    assert result.method == "llm_proposal_code_validated"
    assert result.phases[0]["effect"] == [
        {"predicate": "agent.holds", "args": {"object": "mug_1"}}]
    assert result.phases[0]["validation"]["declared_effect_aligned"] is False
    assert result.phases[0]["validation"]["state_evidence_precedence"] is True


class _PersonalizedExtractorLLM:
    usage = LLMUsage()

    def __init__(self):
        self.inputs = []

    def generate(self, **kwargs):
        self.inputs.append(json.loads(kwargs["input_text"]))
        payload = {
            "phases": [
                {"phase_id": "p0", "intent": "acquire_and_transport_mug_to_microwave",
                 "event_start": 0, "event_end": 1,
                 "parameter_roles": {"object": "mug_1", "source_location": "countertop_1",
                                     "heating_station": "microwave_1"},
                 "effect_predicates": ["agent.holds(mug_1)"]},
                {"phase_id": "p1", "intent": "heat_this_mug", "event_start": 2,
                 "event_end": 2,
                 "parameter_roles": {"object": "mug_1", "heating_station": "microwave_1"},
                 "effect_predicates": ["object.heated(mug_1)"]},
                {"phase_id": "p2", "intent": "place_mug_in_cabinet", "event_start": 3,
                 "event_end": 4,
                 # LLM only names the class; executed phase evidence must refine
                 # it to mug_1 and reject the same-family bystander mug_2.
                 "parameter_roles": {"object": "mug", "target_location": "cabinet_1"},
                 "effect_predicates": ["object.at_location(mug_1,cabinet_1)"]},
            ],
            "discarded_event_indices": [], "workflow_summary": "personalized wording",
        }
        return type("Response", (), {"text": json.dumps(payload)})()


def _trace_with_internal_effects(trace_id="internal_one"):
    states = [
        {"facts": ["object_at(mug_1, countertop_1)"], "inventory": []},
        {"facts": ["agent_holds(mug_1)"], "inventory": ["mug_1"]},
        {"facts": ["agent_holds(mug_1)", "agent_at(microwave_1)"], "inventory": ["mug_1"]},
        {"facts": ["agent_holds(mug_1)", "object_heated(mug_1)"], "inventory": ["mug_1"]},
        {"facts": ["agent_holds(mug_1)", "object_heated(mug_1)",
                   "container_open(cabinet_1)"], "inventory": ["mug_1"]},
        {"facts": ["object_heated(mug_1)", "container_open(cabinet_1)",
                   "object_at(mug_1, cabinet_1)",
                   # Same-family bystander discovered in the final snapshot;
                   # it is not an Effect of placing mug_1.
                   "object_at(mug_2, cabinet_1)"], "inventory": []},
    ]
    actions = [
        ("take mug", {"object": "mug_1", "source_location": "countertop_1"}),
        ("go microwave", {"target_location": "microwave_1"}),
        ("heat mug", {"object": "mug_1", "heating_station": "microwave_1"}),
        ("open cabinet", {"target_location": "cabinet_1"}),
        ("put mug", {"object": "mug_1", "target_location": "cabinet_1"}),
    ]
    return TraceRecord(
        trace_id=trace_id, task_id="t", task_type="heat_place",
        task_goal="heat and place mug", benchmark="env", success=True,
        actions=[ActionRecord(step=i, name=name, params=params)
                 for i, (name, params) in enumerate(actions)],
        state_snapshots=[{"step": i, "state": state} for i, state in enumerate(states)],
        provenance={"params": {"object": "mug", "object_location": "countertop_1",
                               "heating_station": "microwave_1",
                               "target_location": "cabinet_1"},
                    "target_effects": [
                        {"predicate": "object.heated", "args": {"object": "$inputs.object"}},
                        {"predicate": "object.at_location",
                         "args": {"object": "$inputs.object",
                                  "location": "$inputs.target_location"}},
                    ]})


def test_personalized_names_and_internal_effects_are_canonicalized():
    result = SemanticExtractorAgent(_PersonalizedExtractorLLM()).extract(
        _trace_with_internal_effects())
    assert [phase["name"] for phase in result.phases] == [
        "acquire_object", "heat_object", "place_object"]
    acquire, _, place = result.phases
    assert acquire["event_end"] == 0
    assert acquire["params"] == {"object": "mug_1", "object_location": "countertop_1"}
    assert acquire["proposed_intent"] == "acquire_and_transport_mug_to_microwave"
    assert place["effect"] == [{"predicate": "object.at_location",
                                "args": {"object": "mug_1", "location": "cabinet_1"}}]
    assert place["params"] == {"object": "mug_1", "target_location": "cabinet_1"}
    assert place["validation"]["declared_effect_aligned"] is True


def test_occurrence_slice_drops_same_location_bystander_dependency():
    """A location match cannot make another object causally required."""
    initial = {
        "facts": [
            "agent_at(cabinet_1)",
            "object_at(mug_1, cabinet_1)",
        ],
        "inventory": [],
    }
    opened = {
        "facts": [
            *initial["facts"],
            "container_open(cabinet_1)",
        ],
        "inventory": [],
    }
    observed_bystander = {
        "facts": [
            *opened["facts"],
            "object_at(mug_2, cabinet_1)",
        ],
        "inventory": [],
    }
    held = {
        "facts": [
            "agent_at(cabinet_1)",
            "container_open(cabinet_1)",
            "object_at(mug_2, cabinet_1)",
            "agent_holds(mug_1)",
        ],
        "inventory": ["mug_1"],
    }
    events = [
        {
            "event_index": 0, "action": "open cabinet 1", "accepted": True,
            "before": initial, "after": opened,
            "positive_effects": [{"predicate": "container.open",
                                  "args": {"container": "cabinet_1"}}],
            "negative_effects": [],
        },
        {
            "event_index": 1, "action": "examine cabinet 1", "accepted": True,
            "before": opened, "after": observed_bystander,
            "positive_effects": [{"predicate": "object.at_location",
                                  "args": {"object": "mug_2",
                                           "location": "cabinet_1"}}],
            "negative_effects": [],
        },
        {
            "event_index": 2,
            "action": "take mug 1 from cabinet 1", "accepted": True,
            "before": observed_bystander, "after": held,
            "positive_effects": [{"predicate": "agent.holds",
                                  "args": {"object": "mug_1"}}],
            "negative_effects": [],
        },
    ]

    sliced = slice_event_occurrence(
        events, window_start=0, window_end=2,
        core_effects=[{"predicate": "agent.holds",
                       "args": {"object": "mug_1"}}],
        params={"object": "mug_1", "object_location": "cabinet_1"},
    )

    # Location-only container.open remains a real dependency, while the
    # observation of mug_2 at the same cabinet is excluded.
    assert sliced["retained_event_indices"] == [0, 2]
    assert sliced["replay_safe"] is True


def test_first_trace_cannot_persist_same_family_bystander_literal(workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "first_trace_clean_graph")
    result = TraceAtomicizer(
        registry,
        extractor_agent=SemanticExtractorAgent(_PersonalizedExtractorLLM()),
    ).apply(_trace_with_internal_effects("first_trace_with_bystander"))

    assert [candidate.skill.ref.logical_id for candidate in result.candidates] == [
        "env.acquire_object", "env.heat_object", "env.place_object"]
    place = registry.get_recommended("env.place_object")
    assert place is not None
    assert place.effects == [{
        "predicate": "object.at_location",
        "args": {"object": "$inputs.object",
                 "location": "$inputs.target_location"},
    }]
    assert "mug_2" not in json.dumps(place.to_dict(), ensure_ascii=False)


def test_later_extractor_receives_catalog_and_node_accumulates_generalization(workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "generalized_graph")
    llm = _PersonalizedExtractorLLM()
    atomicizer = TraceAtomicizer(registry, extractor_agent=SemanticExtractorAgent(llm))
    first = atomicizer.apply(_trace_with_internal_effects("independent_one"))
    atomicizer.apply(_trace_with_internal_effects("independent_two"))
    assert [candidate.skill.ref.logical_id for candidate in first.candidates] == [
        "env.acquire_object", "env.heat_object", "env.place_object"]
    assert registry.get_recommended("env.open_container") is None
    node = registry.get_recommended("env.acquire_object")
    assert node is not None
    assert node.metadata["statistics"]["support_count"] == 2
    assert node.metadata["generalization"]["status"] == "cross_trace_validated"
    assert "acquire_and_transport_mug_to_microwave" in node.metadata["semantic_alias_counts"]
    assert llm.inputs[0]["known_atomic_contracts"] == []
    assert any(item["canonical_name"] == "acquire_object"
               for item in llm.inputs[1]["known_atomic_contracts"])


def test_extractor_prompt_requires_entity_independent_capability_names():
    assert "Generalization is mandatory" in EXTRACTOR_PROMPT
    assert "take_out_a_banana" in EXTRACTOR_PROMPT
    assert "acquire_object (拿取物品)" in EXTRACTOR_PROMPT


def test_extractor_failure_cannot_fall_back_and_mutate_skillgraph(workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "fail_closed_graph")
    atomicizer = TraceAtomicizer(
        registry, extractor_agent=SemanticExtractorAgent(llm=None))
    result = atomicizer.apply(_trace("extractor_unavailable"))
    assert result.candidates == []
    assert result.segments == []
    assert "no_events_or_llm" in result.semantic_extraction["errors"]
    assert registry.list_all() == []


def test_event_slice_removes_navigation_loop_and_preserves_core_roles():
    states = [
        {"facts": ["agent_at(cabinet_1)", "agent_holds(cup_1)"],
         "inventory": ["cup_1"]},
        {"facts": ["agent_at(microwave_1)", "agent_holds(cup_1)"],
         "inventory": ["cup_1"]},
        {"facts": ["agent_at(microwave_1)", "agent_holds(cup_1)",
                   "container_open(microwave_1)"], "inventory": ["cup_1"]},
        {"facts": ["agent_at(cabinet_4)", "agent_holds(cup_1)",
                   "container_open(microwave_1)"], "inventory": ["cup_1"]},
        {"facts": ["agent_at(microwave_1)", "agent_holds(cup_1)",
                   "container_open(microwave_1)"], "inventory": ["cup_1"]},
        {"facts": ["agent_at(microwave_1)", "agent_holds(cup_1)",
                   "container_open(microwave_1)", "object_heated(cup_1)"],
         "inventory": ["cup_1"]},
        {"facts": ["agent_at(microwave_1)", "agent_holds(cup_1)",
                   "container_open(microwave_1)", "object_heated(cup_1)"],
         "inventory": ["cup_1"]},
    ]
    actions = [
        ("go to microwave 1", {}),
        ("open microwave 1", {}),
        ("go to cabinet 4", {}),
        ("go to microwave 1", {}),
        ("heat cup 1 with microwave 1",
         {"object": "cup 1", "heating_station": "microwave 1"}),
        ("heat cup 1 with microwave 1",
         {"object": "cup 1", "heating_station": "microwave 1"}),
    ]
    trace = TraceRecord(
        trace_id="looping_heat", task_id="looping_heat", task_type="heat",
        task_goal="heat cup", benchmark="env", success=True,
        actions=[ActionRecord(step=index, name=name, params=params)
                 for index, (name, params) in enumerate(actions)],
        state_snapshots=[{"step": index, "state": state}
                         for index, state in enumerate(states)],
        provenance={"params": {"object": "cup", "target_location": "cabinet 1",
                               "heating_station": "microwave 1"},
                    "target_effects": [{"predicate": "object.heated",
                                        "args": {"object": "$inputs.object"}}]},
    )
    events = build_structured_events(trace)
    phases, errors = validate_phase_proposal(trace, events, {"phases": [{
        "phase_id": "heat", "intent": "heat_object", "event_start": 0,
        "event_end": 5, "parameter_roles": dict(trace.provenance["params"]),
        "effect_predicates": ["object.heated"],
    }]})
    assert errors == []
    assert len(phases) == 1
    phase = phases[0]
    assert phase["causal_event_indices"] == [0, 1, 4]
    assert [action["action"] for action in phase["actions"]] == [
        "go to microwave 1", "open microwave 1",
        "heat cup 1 with microwave 1"]
    assert phase["params"] == {"object": "cup 1",
                               "heating_station": "microwave 1"}
    assert phase["effect"] == [
        {"predicate": "object.heated", "args": {"object": "cup_1"}}]
    assert phase["replay_safe"] is True
    assert phase["event_slice_diagnostics"][
        "counterfactual_forward_validated"] is True


def test_task_target_role_cannot_hide_observed_acquire_source_role():
    states = [
        {"facts": [], "inventory": []},
        {"facts": ["agent_at(cabinet_2)"], "inventory": []},
        {"facts": ["agent_at(cabinet_2)", "container_open(cabinet_2)",
                   "object_exists(cup_1)", "object_at(cup_1, cabinet_2)"],
         "inventory": []},
        {"facts": ["agent_at(cabinet_2)", "container_open(cabinet_2)",
                   "object_exists(cup_1)", "agent_holds(cup_1)"],
         "inventory": ["cup_1"]},
    ]
    actions = [
        ActionRecord(step=0, name="go to cabinet 2"),
        ActionRecord(step=1, name="open cabinet 2"),
        ActionRecord(step=2, name="take cup 1 from cabinet 2",
                     params={"object": "cup 1", "object_location": "cabinet 2"}),
    ]
    trace = TraceRecord(
        trace_id="acquire_role", task_id="acquire_role", task_type="acquire",
        task_goal="take cup", benchmark="env", success=True, actions=actions,
        state_snapshots=[{"step": index, "state": state}
                         for index, state in enumerate(states)],
        provenance={"params": {"object": "cup",
                               "target_location": "cabinet 1"}},
    )
    phases, errors = validate_phase_proposal(
        trace, build_structured_events(trace), {"phases": [{
            "phase_id": "acquire", "intent": "acquire_object",
            "event_start": 0, "event_end": 2,
            "parameter_roles": dict(trace.provenance["params"]),
            "effect_predicates": ["agent.holds"],
        }]})
    assert errors == []
    assert phases[0]["params"] == {
        "object": "cup 1", "object_location": "cabinet 2"}
    assert phases[0]["causal_event_indices"] == [0, 1, 2]
    assert phases[0]["replay_safe"] is True


def test_composite_is_draft_then_promoted_by_independent_trace(workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "graph")
    acquire = _atomic("env.acquire", [], [{"predicate": "agent.holds", "args": {"object": "$inputs.object"}}],
                      [{"name": "object", "semantic_type": "object_ref"}],
                      [{"name": "object", "semantic_type": "object_ref"}])
    heat = _atomic("env.heat", [{"predicate": "agent.holds", "args": {"object": "$inputs.object"}}],
                   [{"predicate": "object.heated", "args": {"object": "$inputs.object"}}],
                   [{"name": "object", "semantic_type": "object_ref"}],
                   [{"name": "object", "semantic_type": "object_ref"}])
    place = _atomic("env.place", [{"predicate": "agent.holds", "args": {"object": "$inputs.object"}}],
                    [{"predicate": "object.at_location", "args": {"object": "$inputs.object",
                                                                     "location": "$inputs.target_location"}}],
                    [{"name": "object", "semantic_type": "object_ref"}],
                    [{"name": "object", "semantic_type": "object_ref"}])
    for atomic in (acquire, heat, place):
        registry.register(atomic)
    config = SystemConfig(data_dir=workspace_tmp)
    config.thresholds.composite_min_support = 2
    builder = CompositeBuilder(registry, config)
    extraction = SemanticExtractorAgent(_ExtractorLLM()).extract(_trace())
    refs = [acquire.ref, heat.ref, place.ref]
    first = builder.build_or_align(refs, _trace("trace_one"), segments=extraction.phases)
    assert first.composite.status == SkillStatus.DRAFT
    second = builder.build_or_align(refs, _trace("trace_two"), segments=extraction.phases)
    assert second.composite.status == SkillStatus.ACTIVE
    assert second.composite.step_instances()[1]["params"]["heating_station"] == "$task.heating_station"


def test_composite_alignment_ignores_llm_wording_and_optional_hint(workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "canonical_composite_graph")
    acquire = _atomic(
        "env.acquire", [],
        [{"predicate": "agent.holds", "args": {"object": "$inputs.object"}}],
        [{"name": "object"}], [{"name": "object"}])
    heat = _atomic(
        "env.heat",
        [{"predicate": "agent.holds", "args": {"object": "$inputs.object"}}],
        [{"predicate": "object.heated", "args": {"object": "$inputs.object"}}],
        [{"name": "object"}], [{"name": "object"}])
    place = _atomic(
        "env.place",
        [{"predicate": "agent.holds", "args": {"object": "$inputs.object"}}],
        [{"predicate": "object.at_location",
          "args": {"object": "$inputs.object",
                   "location": "$inputs.target_location"}}],
        [{"name": "object"}, {"name": "target_location"}],
        [{"name": "object"}])
    for atomic in (acquire, heat, place):
        registry.register(atomic)
    config = SystemConfig(data_dir=workspace_tmp)
    config.thresholds.composite_min_support = 2
    builder = CompositeBuilder(registry, config)
    phases = SemanticExtractorAgent(_ExtractorLLM()).extract(_trace()).phases
    refs = [acquire.ref, heat.ref, place.ref]
    first = builder.build_or_align(
        refs, _trace("composite_wording_one"), segments=phases,
        graph_proposal={
            "validated": True,
            "summary": "Acquire a mug and put it into this coffeemachine",
            "implicit_dependencies": [{
                "source_phase_id": "p1", "target_phase_id": "p2",
                "reason": "the mug must remain heated",
            }],
        })
    second = builder.build_or_align(
        refs, _trace("composite_wording_two"), segments=phases,
        graph_proposal={"validated": True,
                        "summary": "A completely different sentence"})

    assert first.decision == "new"
    assert second.decision == "reuse"
    assert registry.list_versions(first.composite.ref.logical_id) == ["1.0.0"]
    assert second.composite.status == SkillStatus.ACTIVE
    assert "mug" not in second.composite.summary.lower()
    assert (second.composite.metadata["statistics"]["support_count"] == 2)


def test_composite_does_not_bind_unknown_source_to_same_family_target(workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "source_role_graph")
    acquire = _atomic(
        "env.acquire", [],
        [{"predicate": "agent.holds", "args": {"object": "$inputs.object"}}],
        [{"name": "object"}, {"name": "object_location"}],
        [{"name": "object"}])
    heat = _atomic(
        "env.heat",
        [{"predicate": "agent.holds", "args": {"object": "$inputs.object"}}],
        [{"predicate": "object.heated", "args": {"object": "$inputs.object"}}],
        [{"name": "object"}], [{"name": "object"}])
    for atomic in (acquire, heat):
        registry.register(atomic)
    trace = _trace("unknown_source")
    trace.provenance["params"].pop("object_location", None)
    phases = SemanticExtractorAgent(_ExtractorLLM()).extract(_trace()).phases[:2]
    phases = copy.deepcopy(phases)
    phases[0]["params"]["object_location"] = "cabinet_2"
    built = CompositeBuilder(registry, SystemConfig(data_dir=workspace_tmp)).build_or_align(
        [acquire.ref, heat.ref], trace, segments=phases)

    acquire_params = built.composite.step_instances()[0]["params"]
    assert acquire_params["object"] == "$task.object"
    assert "object_location" not in acquire_params


def test_composite_role_mapping_uses_semantic_task_params():
    trace = _trace("semantic_role")
    trace.provenance["params"]["target_location"] = "cabinet_1"
    trace.provenance["semantic_params"] = {
        "object": "mug", "target_location": "cabinet",
        "heating_station": "microwave",
    }
    mapped = _role_params({"object": "mug_5",
                           "target_location": "cabinet_5"}, trace)
    assert mapped == {"object": "$task.object",
                      "target_location": "$task.target_location"}
    # target 的 semantic family 不能被借给 source role。
    assert _role_params({"object_location": "cabinet_5"}, trace) == {}


def test_skill_registry_rejects_immutable_composite_overwrite(workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "immutable_graph")
    atomic = _atomic(
        "env.acquire", [],
        [{"predicate": "agent.holds", "args": {"object": "$inputs.object"}}],
        [{"name": "object"}], [{"name": "object"}])
    registry.register(atomic)
    builder = CompositeBuilder(registry, SystemConfig(data_dir=workspace_tmp))
    # Need two occurrences for a valid Composite.
    built = builder.build_or_align(
        [atomic.ref, atomic.ref], _trace("immutable_one"),
        segments=[{"phase_id": "a", "params": {"object": "mug_1"}},
                  {"phase_id": "b", "params": {"object": "mug_1"}}])
    overwritten = copy.deepcopy(built.composite)
    overwritten.summary = "instance-specific replacement"
    with pytest.raises(ValueError, match="immutable_version_collision"):
        registry.register(overwritten)


def test_unchanged_composite_insight_updates_evidence_without_version_churn(
        workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "insight_graph")
    atomic = _atomic(
        "env.acquire", [],
        [{"predicate": "agent.holds", "args": {"object": "$inputs.object"}}],
        [{"name": "object"}], [{"name": "object"}])
    registry.register(atomic)
    config = SystemConfig(data_dir=workspace_tmp)
    config.thresholds.insight_min_samples = 3
    built = CompositeBuilder(registry, config).build_or_align(
        [atomic.ref, atomic.ref], _trace("insight_source"),
        segments=[{"phase_id": "a", "params": {"object": "mug_1"}},
                  {"phase_id": "b", "params": {"object": "mug_1"}}])
    traces = [TraceRecord(trace_id=f"insight_{index}", task_type="heat_place",
                          task_goal="perform reusable operation")
              for index in range(3)]
    store = type("Store", (), {
        "by_task_type": lambda self, *_args, **_kwargs: traces,
    })()

    result = InsightUpdater(registry, store, config).update_if_ready(
        built.composite.ref, "heat_place")

    assert result["reason"] == "evidence_only"
    assert registry.list_versions(built.composite.ref.logical_id) == ["1.0.0"]
    assert registry.get_recommended(
        built.composite.ref.logical_id).insight["sample_count"] == 3


def test_insight_version_records_layer3_reason(workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "insight_reason_graph")
    atomic = _atomic(
        "env.acquire", [],
        [{"predicate": "agent.holds", "args": {"object": "$inputs.object"}}],
        [{"name": "object"}], [{"name": "object"}])
    registry.register(atomic)
    config = SystemConfig(data_dir=workspace_tmp)
    config.thresholds.insight_min_samples = 3
    built = CompositeBuilder(registry, config).build_or_align(
        [atomic.ref, atomic.ref], _trace("insight_reason_source"),
        segments=[{"phase_id": "a", "params": {"object": "mug_1"}},
                  {"phase_id": "b", "params": {"object": "mug_1"}}])
    traces = [TraceRecord(trace_id=f"reason_{index}", task_type="heat_place",
                          task_goal="use microwave and cabinet")
              for index in range(3)]
    store = type("Store", (), {
        "by_task_type": lambda self, *_args, **_kwargs: traces,
    })()
    result = InsightUpdater(registry, store, config).update_if_ready(
        built.composite.ref, "heat_place")
    assert result["updated"]
    latest = registry.get_latest(built.composite.ref.logical_id)
    assert latest.metadata["version_reason"] == "layer3_insight_update"
