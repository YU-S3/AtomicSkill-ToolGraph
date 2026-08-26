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
from atomic_skillgraph.system import (
    _ground_env_runtime_params,
    _realized_task_bindings,
    _refine_env_object_binding,
)
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


def test_seeded_action_backfills_runtime_slot_before_atomic_validation():
    from types import SimpleNamespace

    atomic = AbstractAtomicSkill(
        ref=SkillRef("env.toggle", "1.0.0"), summary="toggle",
        inputs=[{"name": "light_source"}], outputs=[],
        preconditions=[{"predicate": "object.exists",
                        "args": {"object": "$inputs.light_source"}}],
        effects=[{"predicate": "object.toggled",
                  "args": {"object": "$inputs.light_source"}}],
        validator={"post_checks": ["object.toggled"]},
        status=SkillStatus.ACTIVE,
    )
    before = {"facts": ["object_exists(desklamp_1)"]}
    after = {"facts": ["object_exists(desklamp_1)",
                        "object_toggled(desklamp_1)"]}
    result = SimpleNamespace(actions=[
        {"step": 0, "name": "unrelated", "params": {"object": "pencil 1"},
         "accepted": True, "node_ref": "skill://env.acquire@1.0.0"},
        {"step": 1, "name": "exploratory use",
         "params": {"light_source": "desklamp 2"}, "accepted": True,
         "node_ref": "skill://env.toggle@1.0.0"},
        {"step": 2, "name": "use desklamp 1",
         "params": {"light_source": "desklamp 1"}, "accepted": True,
         "node_ref": "skill://env.toggle@1.0.0"},
    ])

    grounded, evidence = _ground_env_runtime_params(
        {}, atomic, atomic.effects, result, action_start=1,
        before=before, after=after, node_ref="skill://env.toggle@1.0.0")

    assert grounded == {"light_source": "desklamp_1"}
    assert evidence[0]["source"] == "accepted_action_param"
    validation = NodeValidator().validate_atomic(
        atomic, before, after, inputs=grounded)
    assert validation.passed is True


def test_positive_state_effect_can_ground_slot_missing_from_action_params():
    from types import SimpleNamespace

    atomic = AbstractAtomicSkill(
        ref=SkillRef("env.transform", "1.0.0"), summary="transform",
        inputs=[{"name": "object"}], outputs=[], preconditions=[],
        effects=[{"predicate": "object.cooled",
                  "args": {"object": "$inputs.object"}}],
        status=SkillStatus.ACTIVE,
    )
    before = {"facts": ["agent_holds(mug_1)"]}
    after = {"facts": ["agent_holds(mug_1)", "object_cooled(mug_1)"]}
    result = SimpleNamespace(actions=[{
        "step": 0, "params": {}, "accepted": True,
        "node_ref": "skill://env.transform@1.0.0",
    }])

    grounded, evidence = _ground_env_runtime_params(
        {"object": "mug"}, atomic, atomic.effects, result,
        action_start=0, before=before, after=after,
        node_ref="skill://env.transform@1.0.0")

    assert grounded["object"] == "mug_1"
    assert evidence == [{
        "parameter": "object", "value": "mug_1",
        "source": "observed_positive_effect", "predicate": "object.cooled",
    }]


def test_runtime_grounding_never_replaces_an_existing_concrete_identity():
    from types import SimpleNamespace

    atomic = AbstractAtomicSkill(
        ref=SkillRef("env.acquire", "1.0.0"), summary="acquire",
        inputs=[{"name": "object"}], outputs=[], preconditions=[],
        effects=[{"predicate": "agent.holds",
                  "args": {"object": "$inputs.object"}}],
        status=SkillStatus.ACTIVE,
    )
    result = SimpleNamespace(actions=[{
        "step": 0, "params": {"object": "mug 2"}, "accepted": True,
        "node_ref": "skill://env.acquire@1.0.0",
    }])
    grounded, evidence = _ground_env_runtime_params(
        {"object": "mug_1"}, atomic, atomic.effects, result,
        action_start=0, before={"facts": []},
        after={"facts": ["agent_holds(mug_2)"]},
        node_ref="skill://env.acquire@1.0.0")
    assert grounded == {"object": "mug_1"}
    assert evidence == []


def test_composite_validation_fills_only_task_roles_missing_at_plan_time():
    realized = _realized_task_bindings(
        {"target_location": "cabinet", "object_type": "mug"},
        [
            {"params": {"object": "pencil_1"}},
            {"params": {"light_source": "desklamp_1",
                        "target_location": "shelf_3"}},
        ],
    )
    assert realized == {
        "target_location": "cabinet",
        "object_type": "mug",
        "object": "pencil_1",
        "light_source": "desklamp_1",
    }
