from atomic_skillgraph.adapters.benchmark import EnvRunResult, Task
from atomic_skillgraph.adapters.mock_llm import MockLLM
from atomic_skillgraph.atomicizer.semantic_extractor import build_structured_events
from atomic_skillgraph.core.config import SystemConfig
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.skill_ir import AbstractAtomicSkill, CompositeSkill
from atomic_skillgraph.core.status import SkillStatus
from atomic_skillgraph.core.trace_ir import NodeExecutionStatus, TraceRecord
from atomic_skillgraph.runtime.runtime_graph import PlannedNode, RuntimeGraph, RuntimePlan
from atomic_skillgraph.system import AtomicSkillGraphSystem, analyze_task_gap


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
    assert trace.validation_layers["selected_composite_self"]["passed"] is False
    assert trace.validation_layers["full_runtime_graph"]["passed"] is True, (
        trace.validation_layers["full_runtime_graph"])


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
