from atomic_skillgraph.adapters.alfworld import (
    _effects_met, _ground_effect_inputs_from_action,
    _hard_cycle_block_reason, _meaningful_env_state_changed)
from atomic_skillgraph.adapters.benchmark import EnvRunResult, Task
from atomic_skillgraph.adapters.mock_llm import MockLLM
from atomic_skillgraph.core.config import SystemConfig
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.refs import ToolRef
from atomic_skillgraph.core.skill_ir import (
    AbstractAtomicSkill, ImplementationAtom, ToolBinding)
from atomic_skillgraph.core.status import (
    ArtifactKind, ExecutionMode, SkillStatus, ToolLifecycle)
from atomic_skillgraph.core.tool_ir import ToolAsset
from atomic_skillgraph.core.trace_ir import TraceRecord
from atomic_skillgraph.runtime.runtime_graph import PlannedNode, RuntimeGraph, RuntimePlan
from atomic_skillgraph.system import AtomicSkillGraphSystem
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
