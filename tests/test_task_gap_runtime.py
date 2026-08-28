from atomic_skillgraph.adapters.benchmark import EnvRunResult, Task
from atomic_skillgraph.adapters.mock_llm import MockLLM
from atomic_skillgraph.atomicizer.semantic_extractor import build_structured_events
from atomic_skillgraph.core.config import SystemConfig
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.skill_ir import AbstractAtomicSkill, CompositeSkill
from atomic_skillgraph.core.status import SkillStatus
from atomic_skillgraph.core.trace_ir import (
    NodeExecutionStatus, NodeValidationResult, TraceRecord)
from atomic_skillgraph.runtime.runtime_graph import PlannedNode, RuntimeGraph, RuntimePlan
from atomic_skillgraph.system import (
    AtomicSkillGraphSystem, _run_env_episode_with_optional_exclusions,
    _runtime_learning_eligible, analyze_task_gap)


def _state(*facts):
    inventory = ["apple_1"] if "agent_holds(apple_1)" in facts else []
    return {"facts": list(facts), "inventory": inventory, "meta": {}}


def _result(resume, action, after, *, success=False, node_ref=""):
    prior_actions = [dict(item) for item in ((resume or {}).get("actions") or [])]
    prior_states = [dict(item) for item in ((resume or {}).get("states") or [])]
    if not prior_states:
        prior_states = [{"step": 0, "state": _state()}]
    actions = prior_actions + ([{
        "step": len(prior_actions), "name": action, "params": {"object": "apple_1"},
        "accepted": True, "node_ref": node_ref,
    }] if action else [])
    states = prior_states + ([{"step": len(actions), "state": after}]
                             if action else [])
    return EnvRunResult(
        success=success, atomic_complete=True, actions=actions, states=states,
        current_observation=action, final_observation=action,
        current_admissible=["continue"] if not success else [],
    )


class _GapAdapter:
    supports_in_place_resume = True

    def __init__(self, finalization_only=False):
        self.calls = []
        self.finalization_only = finalization_only

    def run_env_episode(self, task, llm, **kwargs):
        self.calls.append(dict(kwargs))
        node_ref = str(kwargs.get("node_ref") or "")
        resume = kwargs.get("resume")
        if "task_gap" in node_ref:
            return _result(resume, "heat apple_1", _state(
                "agent_holds(apple_1)", "object_heated(apple_1)"),
                success=True, node_ref=node_ref)
        if kwargs.get("stop_effects"):
            return _result(resume, "take apple_1", _state(
                "agent_holds(apple_1)"), success=False,
                node_ref=node_ref)
        return _result(resume, "finish protocol", _state(
            "agent_holds(apple_1)", "protocol_complete(done)"),
            success=True, node_ref=node_ref)


class _RuntimeResolveAdapter(_GapAdapter):
    def run_env_episode(self, task, llm, **kwargs):
        node_ref = str(kwargs.get("node_ref") or "")
        resume = kwargs.get("resume") or {}
        actions = [dict(item) for item in (resume.get("actions") or [])]
        states = [dict(item) for item in (resume.get("states") or [])]
        if not states:
            states = [{"step": 0, "state": _state()}]
        actions.append({
            "step": len(actions), "name": "put apple 1 in/on shelf 1",
            "params": {"object": "apple_1", "target_location": "shelf_1"},
            "accepted": True, "mode": "dynamic", "node_ref": node_ref,
        })
        states.append({"step": len(actions), "state": _state(
            "object_at(apple_1,shelf_1)")})
        return EnvRunResult(
            actions=actions, states=states, success=True,
            atomic_complete=True, steps=len(actions),
            current_observation="done", current_admissible=[],
            final_observation="done",
        )


class _CardinalityGapAdapter:
    supports_in_place_resume = True

    def __init__(self, *, batch_size=1, gap_reuses=False,
                 won_unchanged=False, gap_batch_size=1):
        self.batch_size = batch_size
        self.gap_reuses = gap_reuses
        self.won_unchanged = won_unchanged
        self.gap_batch_size = gap_batch_size
        self.calls = []

    def run_env_episode(self, task, llm, **kwargs):
        self.calls.append(dict(kwargs))
        resume = kwargs.get("resume") or {}
        actions = [dict(item) for item in (resume.get("actions") or [])]
        states = [dict(item) for item in (resume.get("states") or [])]
        before = dict(resume.get("state") or task.state or {})
        if not states:
            states = [{"step": 0, "state": before}]
        node_ref = str(kwargs.get("node_ref") or "")
        if not kwargs.get("stop_effects"):
            after = before
            params = {}
            success = True
        elif self.won_unchanged:
            after = before
            params = {}
            success = True
        elif "task_gap" in node_ref:
            prior = sorted(str(item) for item in before.get("facts", []) or [])
            if self.gap_reuses:
                after = before
                params = {"object": "widget_1",
                          "target_location": "bay_1"}
            else:
                start_index = self.batch_size + 1
                new_facts = [
                    f"object_at(widget_{index}, bay_1)"
                    for index in range(
                        start_index, start_index + self.gap_batch_size)
                ]
                after = {"facts": prior + new_facts,
                    "inventory": [], "meta": {}}
                params = ({"target_location": "bay_1"}
                          if self.gap_batch_size > 1 else
                          {"object": f"widget_{start_index}",
                           "target_location": "bay_1"})
            success = True
        else:
            prior = sorted(str(item) for item in before.get("facts", []) or [])
            start = len(prior) + 1
            new_facts = [
                f"object_at(widget_{index}, bay_1)"
                for index in range(start, start + self.batch_size)]
            after = {
                "facts": prior + new_facts,
                "inventory": [], "meta": {},
            }
            # A batch occurrence deliberately has no single object action
            # claim; its validated state supplies every concrete witness.
            params = ({"target_location": "bay_1"}
                      if self.batch_size > 1 else
                      {"object": f"widget_{start}",
                       "target_location": "bay_1"})
            success = False
        actions.append({
            "step": len(actions), "name": "apply placement",
            "params": params, "accepted": True, "node_ref": node_ref,
        })
        states.append({"step": len(actions), "state": after})
        return EnvRunResult(
            success=success, atomic_complete=True,
            actions=actions, states=states, steps=len(actions),
            current_observation="done", final_observation="done",
            current_admissible=[] if success else ["continue"],
        )


def _system(workspace_tmp, adapter):
    config = SystemConfig(data_dir=workspace_tmp)
    config.llm.mock = True
    config.features.enable_tool_evolution = False
    return AtomicSkillGraphSystem(config, adapter, MockLLM(script={}))


def _acquire():
    return AbstractAtomicSkill(
        ref=SkillRef("env.acquire", "1.0.0"), summary="acquire entity",
        inputs=[{"name": "object", "semantic_type": "object_ref"}],
        outputs=[{
            "name": "object", "semantic_type": "object_ref",
            "materializer": {"kind": "effect_arg",
                             "predicate": "agent.holds", "arg": "object"},
        }],
        effects=[{"predicate": "agent.holds",
                  "args": {"object": "$inputs.object"}}],
        status=SkillStatus.ACTIVE,
    )


def _place(*, cardinality=1):
    effect = {
        "predicate": "object.at_location",
        "args": {"object": "$inputs.object",
                 "location": "$inputs.target_location"},
    }
    if cardinality > 1:
        effect.update({"cardinality": cardinality,
                       "distinct_by": "object"})
    return AbstractAtomicSkill(
        ref=SkillRef(f"env.place-{cardinality}", "1.0.0"),
        summary="place entities",
        inputs=[{"name": "object", "semantic_type": "object_ref"},
                {"name": "target_location",
                 "semantic_type": "location_ref"}],
        effects=[effect], status=SkillStatus.ACTIVE,
    )


def test_analyze_task_gap_uses_formal_effects_not_benchmark_label():
    task = Task(
        task_id="gap", benchmark="toy_env", context={"params": {"object": "apple"}},
        target_effects=[{"predicate": "object.heated",
                         "args": {"object": "$inputs.object"}}],
    )
    analysis = analyze_task_gap(task, _state("agent_holds(apple_1)"),
                                {"object": "apple"})
    assert [item["predicate"] for item in analysis.missing_effects] == [
        "object.heated"]
    assert analysis.benchmark_only_finalization is False


def test_missing_target_creates_explicit_task_gap_and_pre_gap_validation(
        workspace_tmp):
    adapter = _GapAdapter()
    system = _system(workspace_tmp, adapter)
    acquire = _acquire()
    system.registry.register(acquire)
    composite = CompositeSkill(
        ref=SkillRef("composite.incomplete", "1.0.0"), summary="incomplete",
        graph={"steps": [{"step_id": "old_0", "node_ref": str(acquire.ref),
                           "params": {"object": "$task.object"}}]},
        validator={"target_effects": [{
            "predicate": "object.heated",
            "args": {"object": "$inputs.object"}}]},
        status=SkillStatus.ACTIVE,
    )
    system.registry.register(composite)
    task = Task(
        task_id="gap", benchmark="toy_env", goal="transform the entity",
        context={"params": {"object": "apple"}}, state=_state(),
        target_effects=list(composite.validator["target_effects"]),
    )
    plan = RuntimePlan(
        composite_ref=str(composite.ref),
        nodes=[PlannedNode(
            ref=acquire.ref, step_id="step_000", occurrence_id="occ_000",
            origin_step_id="old_0",
            params={"object": "apple"}, target_effects=list(acquire.effects))],
    )
    trace = TraceRecord(task_id=task.task_id, benchmark=task.benchmark)
    runtime = RuntimeGraph(task.task_id, plan)

    system._run_env_nodes(task, plan, trace, runtime)
    system._finalize_validation(trace, task)

    assert runtime.plan.nodes[-1].source == "task_gap"
    assert runtime.nodes[-1].ref == "skill://runtime.dynamic.task_gap@0.0.0"
    assert runtime.edges[-1].metadata["reason"] == "explicit_task_gap"
    assert trace.task_gap_analysis.missing_effects
    assert [(span.kind, span.action_start, span.action_end)
            for span in trace.runtime_spans] == [
                ("planned_node", 0, 1), ("task_gap", 1, 2)]
    assert trace.actions[1].origin == "task_gap_agent"
    # The first Atomic used one action; the explicit gap owns 20 more, not the
    # whole remaining episode.
    assert adapter.calls[1]["max_steps"] == 21
    gap_audit = trace.metrics["execution_routing"][-1]
    assert gap_audit["budget_scope"] == "gap"
    assert gap_audit["node_limit"] == 20
    assert gap_audit["absolute_deadline"] == 21
    assert trace.validation_layers["selected_composite_self"]["passed"] is False
    assert trace.validation_layers["full_runtime_graph"]["passed"] is True, (
        trace.validation_layers["full_runtime_graph"])


def _cardinality_gap_case(workspace_tmp, adapter, *, target_cardinality,
                          planned_cardinality=1):
    system = _system(workspace_tmp, adapter)
    place = _place(cardinality=planned_cardinality)
    system.registry.register(place)
    target = [{
        "predicate": "object.at_location",
        "args": {"object": "$inputs.object",
                 "location": "$inputs.target_location"},
        "cardinality": target_cardinality,
        "distinct_by": "object",
    }]
    task = Task(
        task_id="cardinality_gap", benchmark="toy_env",
        goal=f"place {target_cardinality} widgets",
        context={"params": {"object": "widget",
                            "target_location": "bay"}},
        state={"facts": [], "inventory": [], "meta": {}},
        target_effects=target,
    )
    group = "target_000:object.at_location:object"
    plan = RuntimePlan(nodes=[PlannedNode(
        ref=place.ref, step_id="step_000", occurrence_id="occ_000",
        branch_id="branch_000",
        params={"object": "widget", "target_location": "bay"},
        target_effects=list(place.effects),
        distinct_bindings={"object": [group]},
        distinct_branch_ids={group: "occ_000"},
    )])
    trace = TraceRecord(task_id=task.task_id, benchmark=task.benchmark)
    runtime = RuntimeGraph(task.task_id, plan)
    system._run_env_nodes(task, plan, trace, runtime)
    system._finalize_validation(trace, task)
    return system, task, trace, runtime


def test_task_gap_inherits_distinct_group_and_hard_rejects_reuse(
        workspace_tmp):
    adapter = _CardinalityGapAdapter(gap_reuses=True)
    system, task, trace, runtime = _cardinality_gap_case(
        workspace_tmp / "reuse", adapter, target_cardinality=2)

    gap_plan = runtime.plan.nodes[-1]
    gap_node = runtime.nodes[-1]
    group = "target_000:object.at_location:object"
    assert gap_plan.source == "task_gap"
    assert gap_plan.distinct_bindings == {"object": [group]}
    assert gap_plan.distinct_branch_ids == {group: "task_gap_000"}
    gap_call = adapter.calls[-1]
    assert gap_call["excluded_effect_bindings"] == {
        "object": {"widget_1"}}
    assert gap_call["effect_inputs"]["__distinct_exclusions__"] == {
        "object": ["widget_1"]}
    assert "widget_1" in gap_call["phase_goal"]
    assert gap_node.passed is False
    assert "distinct_binding_reused" in gap_node.validation.failure_codes
    assert gap_node.fallback_reason == "distinct_binding_reused"
    # Benchmark won remains authoritative, but this mismatch is neutral for
    # all evolution/evidence writers.
    assert trace.success is True
    trace.metrics["runtime_contract_valid"] = _runtime_learning_eligible(
        trace, task)
    trace.metrics["learning_eligible"] = False
    result = system._process_trace(trace, task)
    assert result["neutral_contract_mismatch"] is True
    assert "success_processing" not in result
    assert "failure_processing" not in result


def test_batch_partial_composite_gap_keeps_two_concrete_and_adds_third(
        workspace_tmp):
    adapter = _CardinalityGapAdapter(batch_size=2, gap_reuses=False)
    _system_obj, _task_obj, trace, runtime = _cardinality_gap_case(
        workspace_tmp / "batch", adapter, target_cardinality=3,
        planned_cardinality=2)

    gap_call = adapter.calls[-1]
    assert gap_call["excluded_effect_bindings"] == {
        "object": {"widget_1", "widget_2"}}
    assert "widget" not in gap_call["excluded_effect_bindings"]["object"]
    assert runtime.nodes[-1].passed is True
    progress = runtime.nodes[-1].attempts[0]["distinct_progress"][0]
    assert progress["before_witnesses"] == ["widget_1", "widget_2"]
    assert progress["accepted_new_witnesses"] == ["widget_3"]
    assert trace.success is True


def test_task_gap_materializes_multiple_new_witnesses_as_collection_output(
        workspace_tmp):
    adapter = _CardinalityGapAdapter(batch_size=1, gap_batch_size=2)
    _system_obj, task, trace, runtime = _cardinality_gap_case(
        workspace_tmp / "multi_witness_gap", adapter,
        target_cardinality=3, planned_cardinality=1)

    gap_node = runtime.nodes[-1]
    progress = gap_node.attempts[0]["distinct_progress"][0]
    assert progress["before_witnesses"] == ["widget_1"]
    assert progress["accepted_new_witnesses"] == ["widget_2", "widget_3"]
    assert gap_node.passed is True
    # Collection-valued evidence never masquerades as one arbitrary scalar.
    assert gap_node.outputs == {}
    assert gap_node.distinct_witness_outputs == {
        "object": ["widget_2", "widget_3"]}
    serialized = runtime.to_dict()["nodes"][-1]
    assert serialized["outputs"] == {}
    assert serialized["distinct_witness_outputs"] == {
        "object": ["widget_2", "widget_3"]}
    assert trace.realized_atomic_nodes[-1]["distinct_witness_outputs"] == {
        "object": ["widget_2", "widget_3"]}

    # Distinctness-aware downstream governance explicitly consumes the
    # collection even when no fact snapshot is available. Scalar DATA_FLOW
    # remains untouched because ``outputs`` is still empty.
    gap_node.after = {"facts": [], "inventory": [], "meta": {}}
    future_index = runtime.append_dynamic_gap(
        list(task.target_effects),
        {"object": "widget", "target_location": "bay"},
        task_target_effects=list(task.target_effects),
        occurrence_id="task_gap_001")
    assert runtime.distinct_exclusions(future_index) == {
        "object": {"widget_1", "widget_2", "widget_3"}}


def test_initial_witness_branches_are_already_satisfied_before_new_search(
        workspace_tmp):
    adapter = _CardinalityGapAdapter()
    system = _system(workspace_tmp / "initial_allocation", adapter)
    place = _place()
    system.registry.register(place)
    target = [{
        "predicate": "object.at_location",
        "args": {"object": "$inputs.object",
                 "location": "$inputs.target_location"},
        "cardinality": 2, "distinct_by": "object",
    }]
    task = Task(
        task_id="initial_one_of_two", benchmark="toy_env",
        context={"params": {"object": "widget",
                            "target_location": "bay"}},
        state={"facts": ["object_at(widget_1, bay_1)"],
               "inventory": [], "meta": {}},
        target_effects=target)
    group = "target_000:object.at_location:object"
    nodes = [PlannedNode(
        ref=place.ref, step_id=f"step_{index:03d}",
        occurrence_id=f"occ_{index:03d}", branch_id=f"branch_{index:03d}",
        params={"object": "widget", "target_location": "bay"},
        target_effects=list(place.effects),
        distinct_bindings={"object": [group]},
        distinct_branch_ids={group: f"occ_{index:03d}"})
        for index in range(2)]
    plan = RuntimePlan(nodes=nodes)
    trace = TraceRecord(task_id=task.task_id, benchmark=task.benchmark)
    runtime = RuntimeGraph(task.task_id, plan)

    system._run_env_nodes(task, plan, trace, runtime)

    assert runtime.nodes[0].execution_status == (
        NodeExecutionStatus.ALREADY_SATISFIED)
    assert runtime.nodes[0].params["object"] == "widget_1"
    assert runtime.nodes[1].execution_status == (
        NodeExecutionStatus.EXECUTED_SUCCESS)
    assert runtime.nodes[1].params["object"] == "widget_2"
    planned_call = next(
        call for call in adapter.calls if call.get("stop_effects"))
    assert planned_call["excluded_effect_bindings"] == {
        "object": {"widget_1"}}


def test_gap_group_keeps_original_target_index_after_filtered_missing_set():
    satisfied = {
        "predicate": "agent.holds",
        "args": {"object": "$inputs.object"},
    }
    missing = {
        "predicate": "object.at_location",
        "args": {"object": "$inputs.object",
                 "location": "$inputs.target_location"},
        "cardinality": 2, "distinct_by": "object",
    }
    runtime = RuntimeGraph("filtered", RuntimePlan())
    index = runtime.append_dynamic_gap(
        [missing], {"object": "widget", "target_location": "bay"},
        task_target_effects=[satisfied, missing])

    assert runtime.plan.nodes[index].distinct_bindings == {
        "object": ["target_001:object.at_location:object"]}


def test_benchmark_won_with_unchanged_node_state_is_neutral_for_learning(
        workspace_tmp):
    adapter = _CardinalityGapAdapter(won_unchanged=True)
    system = _system(workspace_tmp / "won_mismatch", adapter)
    place = _place()
    system.registry.register(place)
    task = Task(
        task_id="won_mismatch", benchmark="toy_env", goal="place widget",
        context={"params": {"object": "widget",
                            "target_location": "bay"}},
        state={"facts": [], "inventory": [], "meta": {}},
        target_effects=list(place.effects),
    )
    plan = RuntimePlan(nodes=[PlannedNode(
        ref=place.ref, step_id="step_000", occurrence_id="occ_000",
        params={"object": "widget", "target_location": "bay"},
        target_effects=list(place.effects))])
    trace = TraceRecord(task_id=task.task_id, benchmark=task.benchmark)
    runtime = RuntimeGraph(task.task_id, plan)

    system._run_env_nodes(task, plan, trace, runtime)
    system._finalize_validation(trace, task)
    contract_valid = _runtime_learning_eligible(trace, task)
    trace.metrics.update({
        "benchmark_won": trace.success,
        "runtime_contract_valid": contract_valid,
        "learning_eligible": (not trace.success) or contract_valid,
    })
    stats_before = dict(
        system.registry.get(place.ref).metadata.get("statistics") or {})
    evolution = system._process_trace(trace, task)

    assert trace.success is True
    assert contract_valid is False
    assert evolution["neutral_contract_mismatch"] is True
    assert dict(system.registry.get(place.ref).metadata.get(
        "statistics") or {}) == stats_before


def test_legacy_adapter_signature_remains_compatible_with_exclusion_dispatch():
    class _LegacyAdapter:
        def run_env_episode(
                self, task, llm, *, seed_context="", direct_steps=None,
                max_steps=5, resume=None, stop_effects=None,
                effect_inputs=None, node_ref="", phase_goal=""):
            return {"effect_inputs": effect_inputs, "phase_goal": phase_goal}

    result = _run_env_episode_with_optional_exclusions(
        _LegacyAdapter(), object(), object(), effect_inputs={"object": "mug"},
        excluded_effect_bindings={"object": {"mug_1"}},
        phase_goal="choose another mug")
    assert result == {
        "effect_inputs": {"object": "mug"},
        "phase_goal": "choose another mug"}


def test_unsegmented_seeded_win_without_node_validators_is_not_learnable():
    task = Task(
        task_id="unsegmented", benchmark="toy_env",
        context={"params": {"object": "apple"}},
        target_effects=[{
            "predicate": "agent.holds",
            "args": {"object": "$inputs.object"},
        }])
    trace = TraceRecord(task_id=task.task_id, benchmark=task.benchmark,
                        success=True)
    trace.actions.append(object())
    trace.state_snapshots = [{
        "step": 1, "state": {"facts": ["agent_holds(apple_1)"],
                              "inventory": ["apple_1"]}}]
    trace.realized_atomic_nodes = [{
        "ref": "skill://env.acquire@1.0.0",
        "execution_status": NodeExecutionStatus.NOT_STARTED.value,
        "attempt_started": False, "passed": False,
    }]

    assert _runtime_learning_eligible(trace, task) is False


def test_finalization_that_destroys_target_invalidates_learning_not_won():
    task = Task(
        task_id="destructive_finalization", benchmark="toy_env",
        context={"params": {"object": "apple"}},
        target_effects=[{
            "predicate": "agent.holds",
            "args": {"object": "$inputs.object"},
        }])
    trace = TraceRecord(task_id=task.task_id, benchmark=task.benchmark,
                        success=True)
    trace.state_snapshots = [
        {"step": 1, "state": {
            "facts": ["agent_holds(apple_1)"],
            "inventory": ["apple_1"]}},
        {"step": 2, "state": {"facts": [], "inventory": []}},
    ]
    trace.realized_atomic_nodes = [{
        "ref": "skill://env.acquire@1.0.0",
        "execution_status": NodeExecutionStatus.EXECUTED_SUCCESS.value,
        "attempt_started": True, "passed": True,
        "params": {"object": "apple_1"},
    }]
    trace.node_validators = [NodeValidationResult(
        node_ref="skill://env.acquire@1.0.0", level="atomic",
        passed=True)]

    assert trace.success is True
    assert _runtime_learning_eligible(trace, task) is False


def test_terminal_certificate_accepts_class_instance_but_not_wrong_family():
    task = Task(
        task_id="latent_look", benchmark="toy_env",
        context={"params": {
            "object": "bowl", "associated_entity": "desklamp"}},
        target_effects=[{
            "predicate": "object.observed_with",
            "args": {
                "object": "$inputs.object",
                "associated_entity": "$inputs.associated_entity",
            },
        }])
    trace = TraceRecord(task_id=task.task_id, benchmark=task.benchmark,
                        success=True)
    trace.state_snapshots = [{"step": 4, "state": {
        "facts": [
            "agent_holds(bowl_2)",
            "object_toggled(desklamp_1)",
            "object_at(desklamp_1, desk_1)",
            "agent_at(desk_1)",
        ],
        "inventory": ["bowl_2"],
    }}]
    trace.realized_atomic_nodes = [{
        "ref": "skill://env.observe@1.0.0",
        "execution_status": NodeExecutionStatus.EXECUTED_SUCCESS.value,
        "attempt_started": True, "passed": True,
        "params": {"object": "bowl_2",
                   "associated_entity": "desklamp_1"},
    }]
    trace.node_validators = [NodeValidationResult(
        node_ref="skill://env.observe@1.0.0", level="atomic",
        passed=True)]
    certificate = {
        "effect": {
            "predicate": "object.observed_with",
            "args": {"object": "bowl_2",
                     "associated_entity": "desklamp_1"},
        },
        "source": "benchmark_terminal_certificate_v1",
        "benchmark_won": True,
    }
    trace.metrics = {"runtime_diagnostics": {
        "terminal_verified_effects": [certificate]}}

    assert _runtime_learning_eligible(trace, task) is True

    certificate["effect"]["args"]["object"] = "plate_2"
    assert _runtime_learning_eligible(trace, task) is False


def test_benchmark_finalization_is_non_learnable(workspace_tmp):
    adapter = _GapAdapter(finalization_only=True)
    system = _system(workspace_tmp, adapter)
    acquire = _acquire()
    system.registry.register(acquire)
    task = Task(
        task_id="finalize", benchmark="toy_env",
        context={"params": {"object": "apple"}}, state=_state(),
        target_effects=list(acquire.effects),
    )
    plan = RuntimePlan(nodes=[PlannedNode(
        ref=acquire.ref, step_id="step_000", occurrence_id="occ_000",
        params={"object": "apple"}, target_effects=list(acquire.effects))])
    trace = TraceRecord(task_id=task.task_id, benchmark=task.benchmark)
    runtime = RuntimeGraph(task.task_id, plan)

    system._run_env_nodes(task, plan, trace, runtime)
    events = build_structured_events(trace)

    assert trace.runtime_spans[-1].kind == "benchmark_finalization"
    assert trace.runtime_spans[-1].learnable is False
    final_event = events[-1]
    assert final_event["origin"] == "benchmark_finalization"
    assert final_event["state_positive_effects"]
    assert final_event["positive_effects"] == []


def test_already_satisfied_node_has_no_execution_credit(workspace_tmp):
    adapter = _GapAdapter(finalization_only=True)
    system = _system(workspace_tmp, adapter)
    acquire = _acquire()
    system.registry.register(acquire)
    task = Task(
        task_id="already", benchmark="toy_env",
        context={"params": {"object": "apple"}},
        state=_state("agent_holds(apple_1)"),
        target_effects=list(acquire.effects),
    )
    plan = RuntimePlan(nodes=[PlannedNode(
        ref=acquire.ref, step_id="step_000", occurrence_id="occ_000",
        params={"object": "apple"}, target_effects=list(acquire.effects))])
    trace = TraceRecord(task_id=task.task_id, benchmark=task.benchmark)
    runtime = RuntimeGraph(task.task_id, plan)

    system._run_env_nodes(task, plan, trace, runtime)

    node = runtime.nodes[0]
    assert node.execution_status == NodeExecutionStatus.ALREADY_SATISFIED
    assert node.satisfied_without_execution is True
    assert node.attempts == []
    assert runtime.metrics["direct_started_count"] == 0
    assert runtime.metrics["seeded_generation_count"] == 0
    assert runtime.metrics["dynamic_generation_count"] == 0


def test_runtime_resolvable_semantic_slot_uses_agent_and_records_provenance(
        workspace_tmp):
    system = _system(workspace_tmp, _RuntimeResolveAdapter())
    place = AbstractAtomicSkill(
        ref=SkillRef("generic.place.runtime-resolved", "1.0.0"),
        summary="place an anchored object at a runtime-resolved destination",
        inputs=[
            {"name": "object", "semantic_type": "object_ref",
             "semantic_required": True},
            {"name": "target_location", "semantic_type": "location_ref",
             "semantic_required": True, "runtime_resolvable": True,
             "anchor_roles": ["object"]},
        ],
        effects=[{"predicate": "object.at_location", "args": {
            "object": "$object", "location": "$target_location"}}],
        guideline={"rules": ["resolve the destination while executing"]},
        status=SkillStatus.ACTIVE,
    )
    system.registry.register(place)
    task = Task(
        task_id="runtime_resolve", benchmark="toy_env",
        context={"params": {"object": "apple"}}, state=_state(),
        target_effects=list(place.effects),
    )
    plan = RuntimePlan(nodes=[PlannedNode(
        ref=place.ref, step_id="step_000", occurrence_id="occ_000",
        params={"object": "apple",
                "target_location": "$inputs.target_location"},
        target_effects=list(place.effects))])
    trace = TraceRecord(task_id=task.task_id, benchmark=task.benchmark)
    runtime = RuntimeGraph(task.task_id, plan)

    system._run_env_nodes(task, plan, trace, runtime)

    assert trace.failure_stage != "planning"
    assert runtime.nodes[0].passed is True
    assert runtime.nodes[0].params["target_location"] == "shelf_1"
    assert runtime.nodes[0].binding_provenance["target_location"][
        "source"] == "runtime"
    assert runtime.metrics["direct_started_count"] == 0
    assert (runtime.metrics["seeded_generation_count"]
            + runtime.metrics["dynamic_generation_count"]) == 1
