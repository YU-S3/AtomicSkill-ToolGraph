from atomic_skillgraph.adapters.benchmark import parse_goal_params
from atomic_skillgraph.atomicizer.effect_extractor import (
    extract_effect,
    parameterize_predicates,
)
from atomic_skillgraph.core.predicates import StateSnapshot, check_effects
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.skill_ir import AbstractAtomicSkill
from atomic_skillgraph.core.status import SkillStatus
from atomic_skillgraph.runtime.atomic_planner import _canonical_predicate
from atomic_skillgraph.system import _refine_env_object_binding
from atomic_skillgraph.validation.node_validator import NodeValidator


def test_goal_parser_does_not_treat_some_as_source_location():
    params = parse_goal_params(
        "heat some apple and put it in garbagecan.",
        ["object", "object_location", "target_location"],
    )
    assert params.get("object") == "apple"
    assert "object_location" not in params


def test_alfworld_class_name_matches_numbered_instance():
    passed, missing = check_effects(
        StateSnapshot({"facts": ["agent_holds(apple_1)"],
                       "inventory": ["apple_1"]}),
        {"object": "apple"},
        [{"predicate": "agent.holds",
          "args": {"object": "$inputs.object"}}],
    )
    assert passed, missing


def test_concrete_alfworld_instance_does_not_match_another_instance():
    passed, _missing = check_effects(
        StateSnapshot({"facts": ["object_heated(mug_2)"]}),
        {"object": "mug_1"},
        [{"predicate": "object.heated",
          "args": {"object": "$inputs.object"}}],
    )
    assert passed is False


def test_generic_trace_binding_parameterizes_concrete_instance():
    parameterized = parameterize_predicates(
        [{"predicate": "object.heated", "args": {"object": "mug_2"}}],
        {"object": "mug"},
    )
    assert parameterized[0]["args"]["object"] == "$inputs.object"


def test_parameterization_prefers_exact_slot_over_earlier_family_slot():
    parameterized = parameterize_predicates(
        [{"predicate": "object.at_location",
          "args": {"object": "mug_1", "location": "cabinet_2"}}],
        {
            "object": "mug_1",
            # Deliberately inserted before the exact target binding.  A
            # family-only match must never steal cabinet_2 from this slot.
            "object_location": "cabinet",
            "target_location": "cabinet_2",
        },
    )
    assert parameterized == [{
        "predicate": "object.at_location",
        "args": {"object": "$inputs.object",
                 "location": "$inputs.target_location"},
    }]


def test_parameterization_deduplicates_instances_collapsed_to_generic_slot():
    parameterized = parameterize_predicates(
        [
            {"predicate": "object.heated", "args": {"object": "mug_1"}},
            {"predicate": "object.heated", "args": {"object": "mug_2"}},
        ],
        {"object": "mug"},
    )
    assert parameterized == [
        {"predicate": "object.heated",
         "args": {"object": "$inputs.object"}},
    ]


def test_runtime_carries_acquired_instance_to_downstream_nodes():
    params = _refine_env_object_binding(
        {"object": "mug", "heating_station": "microwave 1"},
        {"inventory": ["mug_1"], "facts": ["agent_holds(mug_1)"]},
    )
    assert params["object"] == "mug_1"


def test_effect_extractor_keeps_only_target_object_preconditions():
    before = {"facts": [
        "object_at(apple_1, countertop_1)",
        "object_at(bowl_1, countertop_1)",
        "object_exists(apple_1)", "object_exists(bowl_1)",
    ]}
    after = {"facts": ["agent_holds(apple_1)",
                       "object_at(bowl_1, countertop_1)",
                       "object_exists(apple_1)", "object_exists(bowl_1)"],
             "inventory": ["apple_1"]}
    effect = extract_effect(before, after, {
        "object": "apple_1", "object_location": "countertop_1"})
    assert effect.primary_family == "agent_holds"
    assert all("bowl" not in str(item) for item in effect.preconditions)
    assert {item["predicate"] for item in effect.preconditions} == {
        "object.at_location", "object.exists"}


def test_post_effect_can_pass_when_precondition_was_unobserved():
    atomic = AbstractAtomicSkill(
        ref=SkillRef("alfworld.acquire_object", "1.0.0"),
        summary="acquire", inputs=[{"name": "object"}], outputs=[],
        preconditions=[{"predicate": "object.exists",
                        "args": {"object": "$inputs.object"}}],
        effects=[{"predicate": "agent.holds",
                  "args": {"object": "$inputs.object"}}],
        validator={"post_checks": ["agent.holds"]},
        status=SkillStatus.ACTIVE,
    )
    result = NodeValidator().validate_atomic(
        atomic, {"facts": []}, {"facts": ["agent_holds(apple_1)"]},
        inputs={"object": "apple"})
    assert result.checks["preconditions"] is False
    assert result.checks["effects"] is True
    assert result.passed is True


def test_receptacle_and_location_effects_share_coverage_key():
    assert _canonical_predicate("object.in_receptacle") == "object.at_location"
