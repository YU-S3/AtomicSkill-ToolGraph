from atomic_skillgraph.adapters.benchmark import parse_goal_params
from atomic_skillgraph.adapters.benchmark import EnvRunResult, Task
from atomic_skillgraph.adapters.mock_llm import MockLLM
from atomic_skillgraph.atomicizer.effect_extractor import (
    extract_effect,
    parameterize_predicates,
)
from atomic_skillgraph.core.predicates import StateSnapshot, check_effects
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.config import SystemConfig
from atomic_skillgraph.core.skill_ir import AbstractAtomicSkill
from atomic_skillgraph.core.status import SkillStatus
from atomic_skillgraph.runtime.atomic_planner import _canonical_predicate
from atomic_skillgraph.runtime.runtime_graph import (
    PlannedNode,
    RuntimeGraph,
    RuntimePlan,
)
from atomic_skillgraph.system import (
    AtomicSkillGraphSystem,
    _bind_known_location_slots,
    _ground_env_runtime_params,
    _realized_task_bindings,
    _refine_env_object_binding,
)
from atomic_skillgraph.core.trace_ir import TraceRecord
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


def test_known_state_binds_generic_source_location_alias():
    atomic = AbstractAtomicSkill(
        ref=SkillRef("env.acquire", "1.0.0"), summary="acquire",
        inputs=[{"name": "object"}, {"name": "source_location"}],
        preconditions=[{
            "predicate": "object.at_location",
            "args": {"object": "$inputs.object",
                     "location": "$inputs.source_location"},
        }],
        effects=[{"predicate": "agent.holds",
                  "args": {"object": "$inputs.object"}}],
        status=SkillStatus.ACTIVE,
    )

    bound = _bind_known_location_slots(
        {"object": "newspaper",
         "source_location": "$flow.source_location"},
        atomic,
        {"facts": ["object_at(newspaper_2, drawer_7)"], "inventory": []},
    )

    assert bound == {"object": "newspaper_2",
                     "source_location": "drawer_7"}


def test_atomic_only_runs_controlled_location_discovery(workspace_tmp):
    class _Adapter:
        supports_in_place_resume = True

        def __init__(self):
            self.discovery_calls = 0

        def discover_object_location(self, task, object_name, **kwargs):
            self.discovery_calls += 1
            state = {
                "facts": ["object_exists(mug_1)",
                          "object_at(mug_1, countertop_1)"],
                "inventory": [], "meta": {},
            }
            result = EnvRunResult(
                actions=[{"step": 0, "name": "go to countertop 1",
                          "params": {"location": "countertop 1"},
                          "accepted": True,
                          "node_ref": kwargs.get("node_ref", ""),
                          "origin": "framework_discovery"}],
                states=[{"step": 0, "state": state}],
                current_observation="a mug is on countertop 1",
                current_admissible=["take mug 1 from countertop 1"],
            )
            return ({"object": "mug_1",
                     "object_location": "countertop_1"}, result)

        def run_env_episode(self, task, llm, **kwargs):
            state = {
                "facts": ["object_exists(mug_1)",
                          "agent_holds(mug_1)"],
                "inventory": ["mug_1"], "meta": {},
            }
            return EnvRunResult(
                success=True, atomic_complete=True,
                actions=[{"step": 0, "name": "go to countertop 1",
                          "params": {"location": "countertop 1"},
                          "accepted": True, "origin": "framework_discovery"},
                         {"step": 1,
                          "name": "take mug 1 from countertop 1",
                          "params": {"object": "mug 1",
                                     "object_location": "countertop 1"},
                          "accepted": True,
                          "node_ref": kwargs.get("node_ref", "")}],
                states=[{"step": 0, "state": state}],
                current_observation="you take the mug",
                current_admissible=[], final_observation="you take the mug")

    config = SystemConfig(data_dir=workspace_tmp / "atomic_only")
    config.llm.mock = True
    config.features.enable_tool_evolution = False
    # Atomic contract resolution is mandatory even when optional framework
    # discovery is disabled for the experiment condition.
    config.features.enable_framework_discovery = False
    adapter = _Adapter()
    system = AtomicSkillGraphSystem(config, adapter, MockLLM(script={}))
    acquire = AbstractAtomicSkill(
        ref=SkillRef("env.acquire", "1.0.0"), summary="acquire object",
        inputs=[{"name": "object"}, {"name": "object_location"}],
        outputs=[], preconditions=[],
        effects=[{"predicate": "agent.holds",
                  "args": {"object": "$inputs.object"}}],
        guideline={"rules": ["obtain the requested object"]},
        status=SkillStatus.ACTIVE)
    system.registry.register(acquire)
    task = Task(
        task_id="atomic_only_discovery", benchmark="alfworld",
        goal="obtain a mug", context={"params": {"object": "mug"}},
        state={"facts": [], "inventory": [], "meta": {}},
        target_effects=list(acquire.effects))
    plan = RuntimePlan(nodes=[PlannedNode(
        ref=acquire.ref, step_id="step_000",
        # A symbolic source is unresolved even though the string is non-empty.
        # It must trigger bounded discovery instead of being sent to a Tool.
        params={"object": "mug",
                "object_location": "$flow.object_location"},
        target_effects=list(acquire.effects))])
    trace = TraceRecord(task_id=task.task_id, benchmark="alfworld")
    runtime = RuntimeGraph(task.task_id, plan)

    system._run_env_nodes(task, plan, trace, runtime)

    assert adapter.discovery_calls == 1
    assert runtime.nodes[0].params["object_location"] == "countertop_1"
    assert trace.metrics["controlled_location_discovery"][0]["found"] is True
    assert trace.metrics["execution_routing"][0][
        "mandatory_location_slots"] == ["object_location"]


def test_in_place_fallback_validates_preconditions_at_node_boundary(
        workspace_tmp):
    class _Adapter:
        supports_in_place_resume = True

        def __init__(self):
            self.calls = 0

        def run_env_episode(self, task, llm, **kwargs):
            self.calls += 1
            resume = dict(kwargs.get("resume") or {})
            actions = [dict(item) for item in resume.get("actions", [])]
            states = [dict(item) for item in resume.get("states", [])]
            if self.calls == 1:
                after = {"facts": ["object_at(mug_1, microwave_1)"],
                         "inventory": [], "meta": {}}
                actions.append({"step": len(actions),
                                "name": "move mug 1 to microwave 1",
                                "params": {"object": "mug_1"},
                                "accepted": True})
                states.append({"step": len(actions), "state": after})
                return EnvRunResult(
                    actions=actions, states=states,
                    failure_type="effect_not_met",
                    current_observation="mug is in microwave",
                    current_admissible=["take mug 1 from microwave 1"],
                    final_observation="mug is in microwave")
            after = {"facts": ["agent_holds(mug_1)",
                                "object_heated(mug_1)"],
                     "inventory": ["mug_1"], "meta": {}}
            actions.append({"step": len(actions),
                            "name": "heat mug 1 with microwave 1",
                            "params": {"object": "mug_1"},
                            "accepted": True})
            states.append({"step": len(actions), "state": after})
            # Reproduce the historical boundary bug: adapter-local completion
            # is absent, while formal Effect validation can still prove success.
            return EnvRunResult(
                actions=actions, states=states,
                current_observation="mug is hot",
                current_admissible=["continue"],
                final_observation="mug is hot")

    config = SystemConfig(data_dir=workspace_tmp / "fallback_boundary")
    config.llm.mock = True
    adapter = _Adapter()
    system = AtomicSkillGraphSystem(config, adapter, MockLLM(script={}))
    transform = AbstractAtomicSkill(
        ref=SkillRef("env.object_heated", "1.0.0"), summary="heat object",
        inputs=[{"name": "object"}],
        preconditions=[{"predicate": "agent.holds",
                        "args": {"object": "$inputs.object"}}],
        effects=[{"predicate": "object.heated",
                  "args": {"object": "$inputs.object"}}],
        guideline={"rules": ["produce the declared state"]},
        status=SkillStatus.ACTIVE)
    system.registry.register(transform)
    initial = {"facts": ["agent_holds(mug_1)"],
               "inventory": ["mug_1"], "meta": {}}
    task = Task(
        task_id="fallback_boundary", benchmark="alfworld", goal="heat mug",
        context={"params": {"object": "mug_1"}}, state=initial,
        target_effects=list(transform.effects))
    plan = RuntimePlan(nodes=[PlannedNode(
        ref=transform.ref, step_id="step_000", occurrence_id="occ_000",
        params={"object": "mug_1"}, target_effects=list(transform.effects))])
    trace = TraceRecord(task_id=task.task_id, benchmark=task.benchmark)
    runtime = RuntimeGraph(task.task_id, plan)

    system._run_env_nodes(task, plan, trace, runtime)

    # A later benchmark-only finalization call is allowed after the Atomic
    # node succeeds; the first two calls are the in-place node attempts.
    assert adapter.calls >= 2
    assert runtime.nodes[0].passed is True
    assert runtime.nodes[0].execution_status.value == "executed_success"
    assert trace.node_validators[-1].before["inventory"] == ["mug_1"]


def test_failed_mandatory_discovery_does_not_start_seeded_and_owns_node_budget(
        workspace_tmp):
    class _Adapter:
        supports_in_place_resume = True

        def __init__(self):
            self.agent_calls = 0

        def discover_object_location(self, task, object_name, **kwargs):
            state = {"facts": [], "inventory": [], "meta": {}}
            actions = [{
                "step": index, "name": f"go to drawer {index + 1}",
                "params": {}, "accepted": True,
                "origin": "framework_discovery",
            } for index in range(30)]
            return {}, EnvRunResult(
                actions=actions,
                states=[{"step": 30, "state": state}],
                failure_type="discovery_budget_exhausted",
                current_observation="not found",
                current_admissible=["look"],
            )

        def run_env_episode(self, task, llm, **kwargs):
            self.agent_calls += 1
            raise AssertionError("Seeded/Dynamic must not run without source binding")

    config = SystemConfig(data_dir=workspace_tmp / "discovery_failure")
    config.llm.mock = True
    config.max_steps = 100
    config.features.enable_framework_discovery = False
    adapter = _Adapter()
    system = AtomicSkillGraphSystem(config, adapter, MockLLM(script={}))
    acquire = AbstractAtomicSkill(
        ref=SkillRef("env.acquire", "1.0.0"), summary="acquire object",
        inputs=[{"name": "object"}, {"name": "object_location"}],
        preconditions=[{
            "predicate": "object.at_location",
            "args": {"object": "$inputs.object",
                     "location": "$inputs.object_location"},
        }],
        effects=[{"predicate": "agent.holds",
                  "args": {"object": "$inputs.object"}}],
        guideline={"rules": ["obtain the requested object"]},
        status=SkillStatus.ACTIVE,
    )
    system.registry.register(acquire)
    task = Task(
        task_id="discovery_failure", benchmark="alfworld",
        goal="obtain a newspaper",
        context={"params": {"object": "newspaper"}},
        state={"facts": [], "inventory": [], "meta": {}},
        target_effects=list(acquire.effects),
    )
    plan = RuntimePlan(nodes=[PlannedNode(
        ref=acquire.ref, step_id="step_000", occurrence_id="occ_000",
        params={"object": "newspaper",
                "object_location": "$flow.object_location"},
        target_effects=list(acquire.effects),
    )])
    trace = TraceRecord(task_id=task.task_id, benchmark=task.benchmark)
    runtime = RuntimeGraph(task.task_id, plan)

    system._run_env_nodes(task, plan, trace, runtime)

    node = runtime.nodes[0]
    assert adapter.agent_calls == 0
    assert node.attempt_started is False
    assert node.executed_action_count == 30
    assert node.attempts[0]["failure_type"] == "node_budget_exhausted"
    assert node.attempts[0]["failure_stage"] == "budget"
    assert node.attempts[0]["action_count"] == 30
    assert trace.metrics["execution_routing"][0]["candidate_modes"] == []


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


def test_post_effect_cannot_hide_known_false_possession_precondition():
    atomic = AbstractAtomicSkill(
        ref=SkillRef("env.heat", "1.0.0"), summary="heat",
        inputs=[{"name": "object"}], outputs=[],
        preconditions=[{"predicate": "agent.holds",
                        "args": {"object": "$inputs.object"}}],
        effects=[{"predicate": "object.heated",
                  "args": {"object": "$inputs.object"}}],
        status=SkillStatus.ACTIVE,
    )
    result = NodeValidator().validate_atomic(
        atomic,
        {"facts": [], "inventory": []},
        {"facts": ["object_heated(apple_1)"], "inventory": []},
        inputs={"object": "apple"})
    assert result.checks["preconditions"] is False
    assert result.checks["preconditions_not_known_false"] is False
    assert result.checks["effects"] is True
    assert result.passed is False


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


def test_wrong_entity_action_cannot_backfill_relational_source_location():
    from types import SimpleNamespace

    atomic = AbstractAtomicSkill(
        ref=SkillRef("env.acquire", "1.0.0"), summary="acquire",
        inputs=[{"name": "object"}, {"name": "object_location"}],
        preconditions=[{
            "predicate": "object.at_location",
            "args": {"object": "$inputs.object",
                     "location": "$inputs.object_location"},
        }],
        effects=[{"predicate": "agent.holds",
                  "args": {"object": "$inputs.object"}}],
        status=SkillStatus.ACTIVE,
    )
    result = SimpleNamespace(actions=[{
        "step": 0, "name": "take keychain 1 from sofa 1",
        "params": {"object": "keychain 1", "object_location": "sofa 1"},
        "accepted": True, "node_ref": str(atomic.ref),
    }])
    params = {"object": "newspaper",
              "object_location": "$flow.object_location"}

    grounded, evidence = _ground_env_runtime_params(
        params, atomic, atomic.effects, result, action_start=0,
        before={"facts": [], "inventory": []},
        after={"facts": ["agent_holds(keychain_1)"],
               "inventory": ["keychain_1"]},
        node_ref=str(atomic.ref))

    assert grounded == params
    assert evidence == []


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
