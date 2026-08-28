from atomic_skillgraph.adapters.alfworld import (
    _effects_met, _ground_effect_inputs_from_action,
    _hard_cycle_block_reason, _meaningful_env_state_changed)
from atomic_skillgraph.adapters.benchmark import EnvRunResult, Task
from atomic_skillgraph.adapters.mock_llm import MockLLM
from atomic_skillgraph.core.config import SystemConfig
from atomic_skillgraph.core.predicates import StateSnapshot, check_effects
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.refs import ToolRef
from atomic_skillgraph.core.skill_ir import (
    AbstractAtomicSkill, ImplementationAtom, ToolBinding)
from atomic_skillgraph.core.status import (
    ArtifactKind, ExecutionMode, SkillStatus, ToolLifecycle)
from atomic_skillgraph.core.tool_ir import ToolAsset
from atomic_skillgraph.core.trace_ir import TraceRecord
from atomic_skillgraph.runtime.runtime_graph import (
    PlannedNode, RuntimeGraph, RuntimePlan, distinct_values_conflict)
from atomic_skillgraph.system import (
    AtomicSkillGraphSystem, _bind_known_location_slots,
    _cardinality_allocation_covers_node,
    _refine_env_object_binding, _reused_distinct_bindings,
    _runtime_distinct_allocations, _runtime_distinct_exclusions)
from atomic_skillgraph.tools.admission_adapter import AdmissionEngine


class _DeadlineAdapter:
    supports_in_place_resume = True

    def __init__(self):
        self.deadlines = []

    def run_env_episode(self, task, llm, **kwargs):
        deadline = int(kwargs["max_steps"])
        self.deadlines.append(deadline)
        resume = kwargs.get("resume") or {}
        actions = [dict(item) for item in (resume.get("actions") or [])]
        states = [dict(item) for item in (resume.get("states") or [])]
        state = dict((resume.get("state") or task.state) or {})
        while len(actions) < deadline:
            actions.append({
                "step": len(actions), "name": "wait",
                "params": {}, "accepted": True,
                "node_ref": str(kwargs.get("node_ref") or ""),
            })
            states.append({"step": len(actions), "state": state})
        return EnvRunResult(
            actions=actions, states=states, steps=len(actions),
            failure_type="max_steps", current_observation="unchanged",
            current_admissible=["continue"], final_observation="unchanged",
        )


def _system(workspace_tmp, adapter):
    config = SystemConfig(data_dir=workspace_tmp)
    config.llm.mock = True
    config.max_steps = 100
    config.features.enable_tool_evolution = False
    return AtomicSkillGraphSystem(config, adapter, MockLLM(script={}))


def _task():
    return Task(
        task_id="budget", benchmark="alfworld", goal="finish",
        context={"params": {"object": "item"}},
        state={"facts": [], "inventory": [], "meta": {}},
        target_effects=[{
            "predicate": "object.ready",
            "args": {"object": "$inputs.object"},
        }],
    )


def test_task_dynamic_owns_global_remaining_budget_and_is_audited(workspace_tmp):
    adapter = _DeadlineAdapter()
    system = _system(workspace_tmp / "task", adapter)
    task = _task()
    plan = RuntimePlan(nodes=[PlannedNode(
        ref=SkillRef("runtime.dynamic.task_level", "0.0.0"),
        step_id="step_000", occurrence_id="task_dynamic_000",
        params={"object": "item"}, source="task_dynamic", dynamic=True,
        budget_scope="task", target_effects=list(task.target_effects),
    )])
    trace = TraceRecord(task_id=task.task_id, benchmark=task.benchmark)
    runtime = RuntimeGraph(task.task_id, plan)

    system._run_env_nodes(task, plan, trace, runtime)

    assert adapter.deadlines == [100]
    audit = trace.metrics["execution_routing"][0]
    assert {key: audit[key] for key in (
        "source", "budget_scope", "global_limit", "node_limit",
        "attempt_limit", "absolute_deadline",
    )} == {
        "source": "task_dynamic", "budget_scope": "task",
        "global_limit": 100, "node_limit": 100,
        "attempt_limit": 100, "absolute_deadline": 100,
    }
    assert runtime.nodes[0].attempts[0]["failure_type"] == (
        "episode_budget_exhausted")
    assert trace.failure_type == "episode_budget_exhausted"


def test_atomic_attempt_and_node_exhaustion_are_distinct(workspace_tmp):
    adapter = _DeadlineAdapter()
    system = _system(workspace_tmp / "atomic", adapter)
    task = _task()
    atomic = AbstractAtomicSkill(
        ref=SkillRef("generic.ready", "1.0.0"), summary="make ready",
        inputs=[{"name": "object"}], effects=list(task.target_effects),
        guideline={"rules": ["make the requested entity ready"]},
        status=SkillStatus.ACTIVE,
    )
    system.registry.register(atomic)
    plan = RuntimePlan(nodes=[PlannedNode(
        ref=atomic.ref, step_id="step_000", occurrence_id="occ_000",
        params={"object": "item"}, source="retrieval",
        budget_scope="atomic", target_effects=list(atomic.effects),
    )])
    trace = TraceRecord(task_id=task.task_id, benchmark=task.benchmark)
    runtime = RuntimeGraph(task.task_id, plan)

    system._run_env_nodes(task, plan, trace, runtime)

    assert adapter.deadlines == [15, 30]
    assert [item["failure_type"] for item in runtime.nodes[0].attempts] == [
        "attempt_budget_exhausted", "node_budget_exhausted"]


def test_tight_action_cycles_are_blocked_before_environment_execution():
    assert _hard_cycle_block_reason(["look", "look"], "look") == (
        "single_action_repeat")
    assert _hard_cycle_block_reason(
        ["go to shelf 1", "examine shelf 1",
         "go to shelf 1", "examine shelf 1"],
        "go to shelf 1") == "two_action_cycle"
    assert _hard_cycle_block_reason(
        ["go to cabinet 1", "open cabinet 1", "go to shelf 1",
         "go to cabinet 1", "open shelf 1"],
        "go to cabinet 1", ["cabinet_1"]) == "repeated_location_action"
    assert _hard_cycle_block_reason(
        ["go to cabinet 1", "go to shelf 1"], "go to cabinet 1",
        location_action_counts={"go to cabinet 1": 2}) == (
            "repeated_location_action")
    assert _hard_cycle_block_reason(
        ["go to microwave 1", "open microwave 1",
         "move item to microwave 1", "close microwave 1"],
        "open microwave 1", ["microwave_1"]) == ""
    assert _hard_cycle_block_reason(
        ["go to cabinet 2"], "examine cabinet 2", ["cabinet_2"]) == ""
    assert _meaningful_env_state_changed(
        {"facts": ["agent_at(a)"]},
        {"facts": ["agent_at(b)"]}) is False
    assert _meaningful_env_state_changed(
        {"facts": ["agent_at(a)"]},
        {"facts": ["agent_at(a)", "container_open(box_1)"]}) is True


def test_budget_scope_is_persisted_in_runtime_ir():
    node = PlannedNode(
        ref=SkillRef("runtime.dynamic.task_gap", "0.0.0"),
        dynamic=True, budget_scope="gap")
    assert node.to_dict()["budget_scope"] == "gap"
    runtime = RuntimeGraph("task", RuntimePlan(nodes=[node]))
    assert runtime.nodes[0].budget_scope == "gap"


def test_validated_distinct_instance_is_excluded_only_across_branches():
    group = "target_000:object.at_location:object"
    ref = SkillRef("generic.acquire", "1.0.0")
    nodes = [
        PlannedNode(ref=ref, step_id="a0", branch_id="branch_000",
                    distinct_bindings={"object": [group]}),
        PlannedNode(ref=ref, step_id="p0", branch_id="branch_000",
                    distinct_bindings={"object": [group]}),
        PlannedNode(ref=ref, step_id="a1", branch_id="branch_001",
                    params={"object": "remotecontrol"},
                    distinct_bindings={"object": [group]}),
    ]
    runtime = RuntimeGraph("two", RuntimePlan(nodes=nodes))
    runtime.nodes[0].passed = True
    runtime.nodes[0].params = {"object": "remotecontrol_4"}
    runtime.nodes[0].outputs = {"object": "remotecontrol_4"}

    # Acquire -> Place in one branch must keep sharing the same instance.
    assert runtime.distinct_exclusions(1) == {}
    # The next branch receives only validated claims from other branches.
    assert runtime.distinct_exclusions(2) == {
        "object": {"remotecontrol_4"}}

    acquire = AbstractAtomicSkill(
        ref=ref, summary="acquire an object",
        inputs=[{"name": "object"}, {"name": "object_location"}],
        effects=[{"predicate": "agent.holds",
                  "args": {"object": "$inputs.object"}}],
        status=SkillStatus.ACTIVE,
    )
    grounded = _bind_known_location_slots(
        {"object": "remotecontrol"}, acquire,
        {"facts": ["object_at(remotecontrol_4, armchair_1)"]},
        runtime.distinct_exclusions(2))
    assert grounded == {"object": "remotecontrol"}
    assert _reused_distinct_bindings(
        {"object": "remotecontrol"}, runtime.distinct_exclusions(2)) == {
            "object": "remotecontrol"}


def test_distinct_branch_identity_is_scoped_per_group():
    first = "target_000:object.at_location:object"
    second = "target_001:object.inspected:object"
    ref = SkillRef("generic.inspect", "1.0.0")
    shared = {"object": [first, second]}
    plan = RuntimePlan(nodes=[
        PlannedNode(
            ref=ref, step_id="n0", branch_id="legacy_shared",
            distinct_bindings=shared,
            distinct_branch_ids={first: "occ_000", second: "occ_000"}),
        PlannedNode(
            ref=ref, step_id="n1", branch_id="legacy_shared",
            distinct_bindings=shared,
            # Still the first placement occurrence, but already the second
            # inspection occurrence. A scalar branch_id cannot express this.
            distinct_branch_ids={first: "occ_000", second: "occ_001"}),
    ])
    runtime = RuntimeGraph("overlap", plan)
    runtime.nodes[0].passed = True
    runtime.nodes[0].params = {"object": "mug_1"}
    runtime.nodes[0].outputs = {"object": "mug_1"}

    assert runtime.distinct_exclusions(1) == {"object": {"mug_1"}}
    assert plan.to_dict()["nodes"][1]["distinct_branch_ids"] == {
        first: "occ_000", second: "occ_001"}
    assert runtime.nodes[1].to_dict()["distinct_branch_ids"] == {
        first: "occ_000", second: "occ_001"}


def test_distinct_exclusion_materializes_validated_state_witness():
    group = "target_000:object.at_location:object"
    ref = SkillRef("generic.place", "1.0.0")
    plan = RuntimePlan(nodes=[
        PlannedNode(
            ref=ref, step_id="a0",
            target_effects=[{
                "predicate": "object.at_location",
                # Deliberately reverse JSON field order. Facts always use the
                # predicate schema order: object, location.
                "args": {"location": "$inputs.target_location",
                         "object": "$inputs.object"},
            }],
            distinct_bindings={"object": [group]},
            distinct_branch_ids={group: "occ_000"}),
        PlannedNode(
            ref=ref, step_id="a1",
            distinct_bindings={"object": [group]},
            distinct_branch_ids={group: "occ_001"}),
    ])
    runtime = RuntimeGraph("materialized", plan)
    runtime.nodes[0].passed = True
    # The executor initially claimed only the class. The validated post-state
    # supplies the concrete instance that must be reserved.
    runtime.nodes[0].params = {
        "object": "mug", "target_location": "cabinet"}
    runtime.nodes[0].outputs = {"object": "mug"}
    runtime.nodes[0].after = {
        "facts": ["object_at(mug_1, cabinet_1)"]}

    assert runtime.distinct_exclusions(1) == {"object": {"mug_1"}}


def test_initial_distinct_witness_uses_schema_order_not_dict_order():
    group = "target_000:object.at_location:object"
    target = [{
        "predicate": "object.at_location",
        "args": {"location": "$target_location",
                 "object": "$object_type"},
        "cardinality": 2,
        "distinct_by": "object",
    }]
    task = Task(
        task_id="reverse_args", benchmark="generic_env",
        context={"params": {
            "object_type": "mug", "target_location": "cabinet"}},
        state={"facts": ["object_at(mug_1, cabinet_1)"]},
        target_effects=target,
    )
    planned = PlannedNode(
        ref=SkillRef("generic.place", "1.0.0"), step_id="p1",
        params={"object": "mug", "target_location": "cabinet"},
        distinct_bindings={"object": [group]},
        distinct_branch_ids={group: "occ_001"})
    runtime = RuntimeGraph(task.task_id, RuntimePlan(nodes=[planned]))

    assert _runtime_distinct_exclusions(
        task, planned, runtime, 0, task.state) == {
            "object": {"mug_1"}}


def test_relation_alias_witnesses_use_schema_order_and_object_at_fact():
    for predicate, location_arg in (
            ("object.in.container", "container"),
            ("object.in.receptacle", "receptacle")):
        group = f"target_000:object.at_location:object"
        target = [{
            "predicate": predicate,
            "args": {location_arg: "$target_location",
                     "object": "$object_type"},
            "cardinality": 2,
            "distinct_by": "object",
        }]
        task = Task(
            task_id=predicate, benchmark="generic_env",
            context={"params": {
                "object_type": "mug", "target_location": "cabinet"}},
            state={"facts": ["object_at(mug_1, cabinet_1)"]},
            target_effects=target,
        )
        planned = PlannedNode(
            ref=SkillRef("generic.place", "1.0.0"), step_id="p1",
            params={"object": "mug", "target_location": "cabinet"},
            distinct_bindings={"object": [group]},
            distinct_branch_ids={group: "occ_001"})
        runtime = RuntimeGraph(task.task_id, RuntimePlan(nodes=[planned]))

        single_alias_effect = [{
            **target[0], "cardinality": 1, "distinct_by": ""}]
        passed, missing = check_effects(
            StateSnapshot(task.state), task.context["params"],
            single_alias_effect)
        assert passed is True
        assert missing == []
        assert _runtime_distinct_exclusions(
            task, planned, runtime, 0, task.state) == {
                "object": {"mug_1"}}


def _cardinality_allocation_case(cardinality, witnesses):
    group = "target_000:object.at_location:object"
    target = [{
        "predicate": "object.at_location",
        "args": {"object": "$object", "location": "$target_location"},
        "cardinality": cardinality,
        "distinct_by": "object",
    }]
    task = Task(
        task_id=f"allocation_{cardinality}_{len(witnesses)}",
        benchmark="generic_env",
        context={"params": {"object": "widget",
                            "target_location": "bay"}},
        state={"facts": [f"object_at({item}, bay_1)" for item in witnesses]},
        target_effects=target,
    )
    nodes = [PlannedNode(
        ref=SkillRef("generic.place", "1.0.0"), step_id=f"p{index}",
        params={"object": "widget", "target_location": "bay"},
        target_effects=[{
            "predicate": "object.at_location",
            "args": {"object": "$inputs.object",
                     "location": "$inputs.target_location"},
        }],
        distinct_bindings={"object": [group]},
        distinct_branch_ids={group: f"occ_{index:03d}"})
        for index in range(cardinality)]
    runtime = RuntimeGraph(task.task_id, RuntimePlan(nodes=nodes))
    return task, runtime


def test_initial_k_of_n_witnesses_allocate_before_searching_remaining():
    task, runtime = _cardinality_allocation_case(2, ["widget_1"])
    first, first_groups = _runtime_distinct_allocations(
        task, runtime.plan.nodes[0], runtime, 0, task.state)
    second, second_groups = _runtime_distinct_allocations(
        task, runtime.plan.nodes[1], runtime, 1, task.state)

    assert first == {"object": "widget_1"}
    assert _cardinality_allocation_covers_node(
        runtime.plan.nodes[0], first, first_groups)
    assert second == {}
    assert second_groups == {}
    assert _runtime_distinct_exclusions(
        task, runtime.plan.nodes[0], runtime, 0, task.state,
        first, first_groups) == {}
    assert _runtime_distinct_exclusions(
        task, runtime.plan.nodes[1], runtime, 1, task.state,
        second, second_groups) == {"object": {"widget_1"}}


def test_initial_n_of_n_witnesses_allocate_every_occurrence():
    task, runtime = _cardinality_allocation_case(
        2, ["widget_2", "widget_1"])
    allocations = [
        _runtime_distinct_allocations(
            task, node, runtime, index, task.state)[0]
        for index, node in enumerate(runtime.plan.nodes)
    ]
    assert allocations == [
        {"object": "widget_1"}, {"object": "widget_2"}]


def test_multiple_initial_witnesses_allocate_stably_before_new_branch():
    task, runtime = _cardinality_allocation_case(
        3, ["widget_2", "widget_1"])
    allocations = [
        _runtime_distinct_allocations(
            task, node, runtime, index, task.state)[0]
        for index, node in enumerate(runtime.plan.nodes)
    ]
    assert allocations == [
        {"object": "widget_1"}, {"object": "widget_2"}, {}]


def test_same_role_multiple_groups_requires_every_group_allocation():
    place_group = "target_000:object.at_location:object"
    clean_group = "target_001:object.cleaned:object"
    targets = [{
        "predicate": "object.at_location",
        "args": {"object": "$object", "location": "$target_location"},
        "cardinality": 2, "distinct_by": "object",
    }, {
        "predicate": "object.cleaned",
        "args": {"object": "$object"},
        "cardinality": 2, "distinct_by": "object",
    }]
    task = Task(
        task_id="two_groups", benchmark="generic_env",
        context={"params": {"object": "mug",
                            "target_location": "counter"}},
        state={"facts": ["object_at(mug_1, counter_1)"]},
        target_effects=targets,
    )
    planned = PlannedNode(
        ref=SkillRef("generic.clean", "1.0.0"), step_id="c0",
        params={"object": "mug"},
        distinct_bindings={"object": [place_group, clean_group]},
        distinct_branch_ids={place_group: "occ_000",
                             clean_group: "occ_000"})
    runtime = RuntimeGraph(task.task_id, RuntimePlan(nodes=[planned]))

    bindings, groups = _runtime_distinct_allocations(
        task, planned, runtime, 0, task.state)
    assert set(groups) == {place_group}
    assert bindings == {}
    assert not _cardinality_allocation_covers_node(
        planned, bindings, groups)


def test_ambiguous_class_claim_is_not_erased_by_other_group_concrete():
    ambiguous = "target_000:object.inspected:object"
    concrete = "target_001:object.at_location:object"
    ref = SkillRef("generic.batch", "1.0.0")
    nodes = [
        PlannedNode(
            ref=ref, step_id="a0",
            distinct_bindings={"object": [ambiguous]},
            distinct_branch_ids={ambiguous: "occ_000"}),
        PlannedNode(
            ref=ref, step_id="b0",
            target_effects=[{
                "predicate": "object.at_location",
                "args": {"object": "$inputs.object",
                         "location": "$inputs.target_location"},
            }],
            distinct_bindings={"object": [concrete]},
            distinct_branch_ids={concrete: "occ_000"}),
        PlannedNode(
            ref=ref, step_id="current",
            distinct_bindings={"object": [ambiguous, concrete]},
            distinct_branch_ids={ambiguous: "occ_001",
                                 concrete: "occ_001"}),
    ]
    runtime = RuntimeGraph("ambiguous", RuntimePlan(nodes=nodes))
    runtime.nodes[0].passed = True
    runtime.nodes[0].params = {"object": "widget"}
    runtime.nodes[1].passed = True
    runtime.nodes[1].params = {
        "object": "widget", "target_location": "bay"}
    runtime.nodes[1].after = {
        "facts": ["object_at(widget_1, bay_1)"]}

    assert runtime.distinct_exclusions(2) == {
        "object": {"widget", "widget_1"}}


def test_distinct_class_and_instance_claims_conflict_symmetrically():
    assert distinct_values_conflict("mug", "mug_1") is True
    assert distinct_values_conflict("mug_1", "mug") is True
    assert distinct_values_conflict("mug_1", "mug_2") is False
    assert _reused_distinct_bindings(
        {"object": "mug_1"}, {"object": {"mug"}}) == {
            "object": "mug_1"}
    assert _reused_distinct_bindings(
        {"object": "mug"}, {"object": {"mug_1"}}) == {
            "object": "mug"}


def test_inventory_refinement_respects_distinct_exclusions():
    state = {"inventory": ["mug_1", "mug_2"]}

    assert _refine_env_object_binding(
        {"object": "mug"}, state,
        {"object": {"mug_1"}}) == {"object": "mug_2"}
    # A class-valued validated claim is fail-closed: it may denote either
    # instance, so refinement cannot silently select one of them.
    assert _refine_env_object_binding(
        {"object": "mug"}, state,
        {"object": {"mug"}}) == {"object": "mug"}


def test_non_placement_cardinality_uses_initial_state_witness():
    group = "target_000:object.toggled:object"
    target = [{
        "predicate": "object.toggled",
        "args": {"object": "$object_type"},
        "cardinality": 2,
        "distinct_by": "object",
    }]
    task = Task(
        task_id="toggle_two", benchmark="generic_env",
        goal="toggle two lamps",
        context={"params": {"object_type": "desklamp"}},
        state={"facts": ["object_toggled(desklamp_1)"]},
        target_effects=target,
    )
    planned = PlannedNode(
        ref=SkillRef("generic.toggle", "1.0.0"), step_id="t1",
        params={"object": "desklamp"},
        distinct_bindings={"object": [group]},
        distinct_branch_ids={group: "occ_001"})
    runtime = RuntimeGraph(task.task_id, RuntimePlan(nodes=[planned]))

    assert _runtime_distinct_exclusions(
        task, planned, runtime, 0, task.state) == {
            "object": {"desklamp_1"}}


def test_alfworld_stop_and_action_grounding_respect_toggle_exclusion():
    effect = [{
        "predicate": "object.toggled",
        "args": {"object": "$inputs.object"},
    }]
    inputs = {"object": "desklamp"}
    excluded = {"object": {"desklamp_1"}}

    # Existing lamp_1 cannot existentially complete occurrence two at zero
    # actions. A newly toggled lamp_2 can.
    assert _effects_met(
        {"facts": ["object_toggled(desklamp_1)"]},
        effect, inputs, excluded) is False
    assert _effects_met(
        {"facts": ["object_toggled(desklamp_1)",
                   "object_toggled(desklamp_2)"]},
        effect, inputs, excluded) is True
    assert _ground_effect_inputs_from_action(
        inputs, {"object": "desklamp_1"}, excluded) == inputs
    assert _ground_effect_inputs_from_action(
        inputs, {"object": "desklamp_2"}, excluded) == {
            "object": "desklamp_2"}


def test_shared_effect_role_is_grounded_to_one_executed_instance():
    effects = [
        {"predicate": "object.heated", "args": {"object": "$object"}},
        {"predicate": "object.at_location",
         "args": {"object": "$object", "location": "$target_location"}},
    ]
    state = {"facts": ["object_heated(egg_1)",
                        "object_at(egg_2, diningtable_1)"],
             "inventory": ["egg_1"]}
    generic = {"object": "egg", "target_location": "diningtable_1"}
    # Class-level matching alone is existential per predicate.
    assert _effects_met(state, effects, generic) is True
    grounded = _ground_effect_inputs_from_action(
        generic, {"object": "egg 1", "heating_station": "microwave 1"})
    assert grounded["object"] == "egg_1"
    assert grounded["target_location"] == "diningtable_1"
    assert _effects_met(state, effects, grounded) is False


def test_action_grounding_keeps_entity_and_source_location_relational():
    params = {"object": "newspaper",
              "object_location": "$flow.object_location"}

    wrong = _ground_effect_inputs_from_action(params, {
        "object": "keychain 1", "object_location": "sofa 1"})
    assert wrong == params

    matching = _ground_effect_inputs_from_action(params, {
        "object": "newspaper 2", "object_location": "drawer 7"})
    assert matching == {
        "object": "newspaper_2", "object_location": "drawer_7"}

    aliased = _ground_effect_inputs_from_action(
        {"object": "newspaper", "source_location": "$flow.source_location"},
        {"object": "keychain 1", "source_location": "sofa 1"})
    assert aliased == {
        "object": "newspaper", "source_location": "$flow.source_location"}


class _AcquireDirectAdapter:
    supports_in_place_resume = True

    def __init__(self):
        self.direct_steps = None
        self.calls = []

    def discover_object_location(self, task, object_name, **kwargs):
        state = {
            "facts": ["object_exists(mug_1)",
                      "object_at(mug_1, countertop_1)",
                      "agent_at(countertop_1)"],
            "inventory": [], "meta": {},
        }
        return ({"object": "mug_1", "object_location": "countertop_1"},
                EnvRunResult(
                    actions=[{"step": 0, "name": "go to countertop 1",
                              "params": {}, "accepted": True,
                              "origin": "framework_discovery"}],
                    states=[{"step": 1, "state": state}],
                    current_observation="a mug is on the countertop",
                    current_admissible=["take mug 1 from countertop 1"]))

    def run_env_episode(self, task, llm, **kwargs):
        self.calls.append(dict(kwargs))
        self.direct_steps = kwargs.get("direct_steps")
        if self.direct_steps is not None:
            assert self.direct_steps[0]["steps"] == [{
                "template": "take {object} from {object_location}",
                "params": {"object": "mug_1",
                           "object_location": "countertop_1"},
            }]
        state = {"facts": ["object_exists(mug_1)",
                            "agent_holds(mug_1)"],
                 "inventory": ["mug_1"], "meta": {}}
        return EnvRunResult(
            atomic_complete=True,
            actions=[{"step": 0, "name": "go to countertop 1",
                      "params": {}, "accepted": True,
                      "origin": "framework_discovery"},
                     {"step": 1, "name": "take mug 1 from countertop 1",
                      "params": {"object": "mug_1",
                                 "object_location": "countertop_1"},
                      "accepted": True}],
            states=[{"step": 2, "state": state}],
            current_observation="taken", current_admissible=["continue"],
            final_observation="taken")


def test_acquire_discovery_rebinds_source_before_direct_tool_selection(
        workspace_tmp):
    adapter = _AcquireDirectAdapter()
    system = _system(workspace_tmp / "acquire_direct", adapter)
    system.config.features.enable_tool_evolution = True
    system.config.features.enable_framework_discovery = True
    acquire = AbstractAtomicSkill(
        ref=SkillRef("generic.acquire", "1.0.0"), summary="obtain entity",
        inputs=[{"name": "object"}, {"name": "object_location"}],
        effects=[{"predicate": "agent.holds",
                  "args": {"object": "$inputs.object"}}],
        status=SkillStatus.ACTIVE,
    )
    tool = ToolAsset(
        ref=ToolRef("generic.acquire.take", "1.0.0"),
        artifact_kind=ArtifactKind.ACTION_TEMPLATE,
        summary="obtain an entity from its verified source",
        signature={"parameters": [
            {"name": "object"}, {"name": "object_location"}]},
        artifact={"steps": ["take {object} from {object_location}"]},
        safety={"direct_execution_allowed": True},
        status=ToolLifecycle.CANDIDATE,
    )
    assert AdmissionEngine(
        replay_fn=lambda *_args: {"passed": True}).admit(tool).passed
    tool.statistics.update({
        "utility": 1.0, "success_count": 3,
        "direct_success_count": 3,
        "admission_replay_success_count": 1,
    })
    impl = ImplementationAtom(
        ref=SkillRef("impl.generic.acquire.take", "1.0.0"),
        abstract_ref=acquire.ref,
        tool_bindings=[ToolBinding(tool.ref)], status=SkillStatus.ACTIVE,
        quality={"utility": 1.0, "success_count": 2},
    )
    system.registry.register(acquire)
    system.registry.register(impl)
    system.tool_registry.register(tool)
    system.tool_registry.set_status(tool.ref, ToolLifecycle.ACTIVE)
    task = Task(
        task_id="acquire_direct", benchmark="alfworld", goal="obtain mug",
        context={"params": {"object": "mug",
                            "target_location": "cabinet_1"}},
        state={"facts": [], "inventory": [], "meta": {}},
        target_effects=list(acquire.effects),
    )
    plan = RuntimePlan(nodes=[PlannedNode(
        ref=acquire.ref, step_id="step_000", occurrence_id="occ_000",
        params={"object": "mug",
                "object_location": "$flow.object_location"},
        source="composite", target_effects=list(acquire.effects),
    )])
    trace = TraceRecord(task_id=task.task_id, benchmark=task.benchmark)
    runtime = RuntimeGraph(task.task_id, plan)

    system._run_env_nodes(task, plan, trace, runtime)

    node = runtime.nodes[0]
    assert node.mode == ExecutionMode.DIRECT, trace.metrics[
        "execution_routing"][0]
    assert node.passed is True
    assert node.params["object"] == "mug_1"
    assert node.params["object_location"] == "countertop_1"
    assert node.params["object_location"] != task.context["params"][
        "target_location"]
    assert node.impl_ref == str(impl.ref)
    assert node.tool_refs == [str(tool.ref)]
    assert trace.metrics["execution_routing"][0]["location_discovered"] is True
