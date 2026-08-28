from atomic_skillgraph.runtime.budget import BudgetLedger
from atomic_skillgraph.runtime.runtime_graph import RuntimeGraph, RuntimePlan
from atomic_skillgraph.runtime.runtime_graph import PlannedNode
from atomic_skillgraph.core.config import SystemConfig
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.skill_ir import AbstractAtomicSkill
from atomic_skillgraph.core.status import ExecutionMode, SkillStatus
from atomic_skillgraph.core.trace_ir import TraceRecord
from atomic_skillgraph.adapters.benchmark import Task
from atomic_skillgraph.adapters.mock_llm import MockLLM
from atomic_skillgraph.system import AtomicSkillGraphSystem


def test_budget_ledger_uses_global_node_and_attempt_minimum():
    ledger = BudgetLedger(global_limit=100, node_limit=20, attempt_limit=7,
                          node_start=30, attempt_start=35, actions_used=39)
    assert ledger.global_remaining() == 61
    assert ledger.node_remaining() == 11
    assert ledger.attempt_remaining() == 3
    assert ledger.absolute_deadline() == 42


def test_zero_action_attempt_is_not_mode_usage():
    runtime = RuntimeGraph("task", RuntimePlan())
    assert runtime.metrics["direct_attempt_count"] == 0
    assert runtime.metrics["seeded_generation_count"] == 0
    # Usage is recorded only immediately before a real adapter call.
    runtime.record_usage(ExecutionMode.DIRECT)
    assert runtime.metrics["direct_attempt_count"] == 1
    assert runtime.metrics["direct_reuse_count"] == 0
    runtime.record_direct_success()
    assert runtime.metrics["direct_success_count"] == 1
    assert runtime.metrics["direct_reuse_count"] == 1


def test_zero_node_budget_does_not_call_adapter_or_count_dynamic(workspace_tmp):
    class Adapter:
        supports_in_place_resume = True

        def __init__(self):
            self.calls = 0

        def run_env_episode(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError("zero budget must not call adapter")

    config = SystemConfig(data_dir=workspace_tmp / "zero_budget")
    config.llm.mock = True
    config.thresholds.env_node_max_steps = 0
    config.thresholds.env_attempt_max_steps = 0
    adapter = Adapter()
    system = AtomicSkillGraphSystem(config, adapter, MockLLM(script={}))
    atomic = AbstractAtomicSkill(
        ref=SkillRef("generic.hold", "1.0.0"), summary="hold object",
        inputs=[{"name": "object"}],
        effects=[{"predicate": "agent.holds",
                  "args": {"object": "$inputs.object"}}],
        guideline={"rules": ["obtain the object"]}, status=SkillStatus.ACTIVE)
    system.registry.register(atomic)
    task = Task(task_id="zero", benchmark="toy_env", goal="hold object",
                context={"params": {"object": "mug_1"}},
                target_effects=list(atomic.effects))
    plan = RuntimePlan(nodes=[PlannedNode(
        ref=atomic.ref, step_id="step_000", params={"object": "mug_1"},
        target_effects=list(atomic.effects))])
    runtime = RuntimeGraph(task.task_id, plan)
    trace = TraceRecord(task_id=task.task_id, benchmark=task.benchmark)
    system._run_env_nodes(task, plan, trace, runtime)
    assert adapter.calls == 0
    assert runtime.metrics["dynamic_generation_count"] == 0
    assert runtime.nodes[0].attempts[0]["started"] is False
    assert trace.failure_stage == "budget"
