"""AtomicSkillGraphSystem：v2.0 运行时总装（设计文档 v2.0 §43 伪代码实现）。

Known Capability → Plan Before Execution（warm）
Unknown Capability → Learn After Successful Execution（cold）
"""

from __future__ import annotations

import copy
import time
from typing import Any

from .adapters.benchmark import BenchmarkAdapter, EnvRunResult, Task
from .core.binding_ir import (BindingKind, BindingProvenance,
                              BindingResolutionState, binding_slot_name,
                              is_concrete_binding)
from .atomicizer.trace_atomicizer import TraceAtomicizer
from .core.config import SystemConfig
from .core.llm import (LLM, LLM_USAGE_FIELDS, diff_llm_usage,
                       record_llm_usage_bucket, snapshot_llm_usage)
from .core.status import EdgeType, ExecutionMode, SkillNodeKind, SkillStatus, ToolLifecycle
from .core.trace_ir import (
    ActionRecord,
    AttemptRecord,
    NodeExecutionStatus,
    NodeValidationResult,
    RuntimeSpan,
    TaskGapAnalysis,
    TaskExecutionInstance,
    TraceRecord,
)
from .evolution.failure_processor import FailureProcessor
from .evolution.branch_repair import FailureBranchManager
from .evolution.composite_lifecycle import apply_self_sufficient_evidence
from .evolution.proposal_replayer import ProposalReplayer
from .evolution.success_processor import SuccessProcessor
from .graph.registry import SkillGraphRegistry
from .graph.validator import validate_graph
from .persistence import MetricsStore, RunStore, TraceStore
from .runtime.atomic_planner import AtomicPlanner
from .runtime.execution_bridge import ExecutionBridge
from .runtime.implementation_selector import ImplementationSelector
from .runtime.runtime_graph import RuntimeGraph, distinct_values_conflict
from .runtime.budget import BudgetLedger
from .runtime.plan_validator import (semantic_required_slots,
                                     slot_requirements_for)
from .runtime.output_materializer import materialize_atomic_outputs
from .tools.resolver import ToolResolver
from .tools.registry import ToolRegistry
from .validation.composite_validator import CompositeValidator
from .validation.node_validator import NodeValidator
from .validation.tool_validator import ToolValidator


class AtomicSkillGraphSystem:
    """集中式 Skill/Tool 联合自进化系统。"""

    def __init__(self, config: SystemConfig, adapter: BenchmarkAdapter, llm: LLM) -> None:
        self.config = config
        self.adapter = adapter
        self.llm = llm
        fork = getattr(llm, "fork", None)
        self.extractor_llm = (fork(
            temperature=config.llm.extractor_temperature,
            max_output_tokens=config.llm.extractor_max_output_tokens,
            request_timeout_seconds=config.llm.extractor_read_timeout_seconds,
            fallback_disable_thinking_on_length=True,
            reasoning_effort=config.llm.extractor_reasoning_effort,
            stream_response=True,
            app_name="atomic-skillgraph-extractor") if callable(fork) else llm)
        setattr(self.adapter, "llm_max_consecutive_errors",
                int(config.thresholds.llm_max_consecutive_errors))

        data_dir = config.data_dir
        self.registry = SkillGraphRegistry(data_dir / "skill_graph")
        self.tool_registry = ToolRegistry(data_dir / "tools")
        self.trace_store = TraceStore(data_dir / "traces")
        self.run_store = RunStore(data_dir / "runtime_runs")
        self.metrics_store = MetricsStore(data_dir / "metrics")

        self.planner = AtomicPlanner(self.registry, config, llm=llm)
        self.resolver = ToolResolver(self.tool_registry)
        self.selector = ImplementationSelector(self.registry, self.resolver, config)
        self.bridge = ExecutionBridge(adapter, config)
        self.bridge.set_llm(llm)
        self.node_validator = NodeValidator(enabled=config.features.enable_node_validator)
        self.tool_validator = ToolValidator()
        self.composite_validator = CompositeValidator(
            enabled=config.features.enable_node_validator)

        self.success_processor = SuccessProcessor(
            self.registry, self.tool_registry, self.trace_store, config, llm=llm,
            replay_fn=getattr(adapter, "replay_tool", None),
            extractor_llm=self.extractor_llm)
        self.failure_processor = FailureProcessor(self.registry, self.tool_registry, config)
        self.proposal_replayer = ProposalReplayer(data_dir)
        self.failure_branch_manager = FailureBranchManager(
            data_dir, self.registry, self.tool_registry, adapter, config, llm=llm)

        self.episode_count = 0
        self.success_count = 0
        self._feedback_buffer: list[tuple[str, list[str], bool, str]] | None = None

    # ------------------------------------------------------------------
    def run_task(self, task: Task) -> dict[str, Any]:
        """运行单个任务（§43）。返回 episode 摘要 dict。"""
        started = time.perf_counter()
        episode_usage_before = self._combined_llm_usage_snapshot()
        planner_usage_before = snapshot_llm_usage(self.llm)
        plan = self.planner.compile_runtime_graph(task)
        planner_usage_after = snapshot_llm_usage(self.llm)
        runtime_usage_before = planner_usage_after

        is_env = _is_env_task(task)
        pending_feedback: list[tuple[str, list[str], bool, str]] = []
        if is_env:
            infrastructure_attempts: list[dict[str, Any]] = []
            retry_limit = max(0, int(self.config.thresholds.infrastructure_episode_retries))
            trace = None
            final_feedback: list[tuple[str, list[str], bool, str]] = []
            for attempt_index in range(retry_limit + 1):
                # feedback 与环境一样按 episode 事务处理：基础设施失败的尝试不会
                # 污染 Tool/Implementation 统计；最终保留的尝试才提交。
                self._feedback_buffer = []
                trace = self._run_env_task(task, copy.deepcopy(plan))
                attempt_feedback = list(self._feedback_buffer)
                errors = list(trace.metrics.get("infrastructure_errors") or [])
                if trace.failure_type != "llm_error":
                    final_feedback = attempt_feedback
                    break
                infrastructure_attempts.append({
                    "attempt": attempt_index + 1,
                    "failure_type": trace.failure_type,
                    "errors": errors,
                })
                if attempt_index < retry_limit:
                    print("[infra-retry] task=%s reset from initial state (%d/%d)" % (
                        task.task_id, attempt_index + 1, retry_limit), flush=True)
                if attempt_index == retry_limit:
                    # 重试耗尽仍是基础设施失败，不提交该尝试产生的能力统计。
                    final_feedback = []
            self._feedback_buffer = None
            pending_feedback = final_feedback
            assert trace is not None
            trace.metrics["infrastructure_episode_retries"] = max(
                0, len(infrastructure_attempts) - int(trace.failure_type == "llm_error"))
            if infrastructure_attempts:
                trace.metrics["infrastructure_attempts"] = infrastructure_attempts
        else:
            trace = self._run_code_task(task, plan)

        runtime_usage_after = snapshot_llm_usage(self.llm)
        record_llm_usage_bucket(
            trace.metrics, "planner_agent",
            diff_llm_usage(planner_usage_before, planner_usage_after))
        record_llm_usage_bucket(
            trace.metrics, "runtime_agent",
            diff_llm_usage(runtime_usage_before, runtime_usage_after))

        trace.latency_ms = (time.perf_counter() - started) * 1000.0
        trace.token_cost = float(diff_llm_usage(
            episode_usage_before,
            self._combined_llm_usage_snapshot())["total_tokens"])
        if trace.success:
            trace.metrics.setdefault("task_success", 1)
        trace.provenance.setdefault("target_effects", copy.deepcopy(task.target_effects))
        trace.provenance.setdefault("params", copy.deepcopy(task.context.get("params") or {}))
        trace.provenance.setdefault(
            "semantic_params",
            copy.deepcopy(task.context.get("semantic_params")
                          or task.context.get("params") or {}))
        self._finalize_validation(trace, task)
        runtime_contract_valid = _runtime_learning_eligible(trace, task)
        learning_eligible = bool(
            not trace.success or runtime_contract_valid)
        trace.metrics["benchmark_won"] = bool(trace.success)
        trace.metrics["runtime_contract_valid"] = runtime_contract_valid
        trace.metrics["learning_eligible"] = learning_eligible
        if learning_eligible:
            for kind, refs, passed, mode in pending_feedback:
                if kind == "tool":
                    self._record_tool_feedback(refs, passed, mode)
                else:
                    self._record_impl_feedback(refs, passed)
        elif pending_feedback:
            trace.metrics["feedback_discarded_for_contract_mismatch"] = len(
                pending_feedback)
        # 保存 trace（成功与失败都保存）
        self.trace_store.save(trace)
        evolution_runtime_usage_before = snapshot_llm_usage(self.llm)
        evolution = self._process_trace(trace, task)
        evolution_runtime_usage_after = snapshot_llm_usage(self.llm)
        evolution_runtime_delta = diff_llm_usage(
            evolution_runtime_usage_before, evolution_runtime_usage_after)
        if self.extractor_llm is self.llm:
            # A compatibility client without fork() serves all agents.  The
            # enclosing evolution delta therefore contains the two explicitly
            # metered Extractor/Composite calls; subtract them to avoid double
            # attribution while retaining any branch-repair Runtime calls.
            usage_by_agent = trace.metrics.get("llm_usage_by_agent") or {}
            for field_name in LLM_USAGE_FIELDS:
                nested = sum(float((usage_by_agent.get(bucket) or {}).get(
                    field_name, 0) or 0)
                    for bucket in ("extractor_agent", "composite_agent"))
                remaining = max(0.0, float(evolution_runtime_delta.get(
                    field_name, 0) or 0) - nested)
                evolution_runtime_delta[field_name] = (
                    remaining if field_name == "latency_ms" else int(remaining))
        record_llm_usage_bucket(
            trace.metrics, "evolution_repair_agent",
            evolution_runtime_delta)
        # Failure-branch strict replay may itself call the LLM. Include that
        # learning cost in the online episode rather than hiding it.
        trace.latency_ms = (time.perf_counter() - started) * 1000.0
        episode_usage_after = self._combined_llm_usage_snapshot()
        episode_usage = diff_llm_usage(
            episode_usage_before, episode_usage_after)
        trace.token_cost = float(episode_usage["total_tokens"])
        usage_by_agent = trace.metrics.setdefault("llm_usage_by_agent", {})
        for bucket in (
                "planner_agent", "runtime_agent", "extractor_agent",
                "composite_agent", "evolution_repair_agent"):
            if bucket not in usage_by_agent:
                record_llm_usage_bucket(trace.metrics, bucket, {})
        accounted: dict[str, int | float] = {}
        unattributed: dict[str, int | float] = {}
        for field_name in LLM_USAGE_FIELDS:
            accounted_value = sum(
                float((usage or {}).get(field_name, 0) or 0)
                for name, usage in usage_by_agent.items()
                if name not in {"episode_total", "unattributed"}
                and isinstance(usage, dict))
            total_value = float(episode_usage.get(field_name, 0) or 0)
            missing_value = max(0.0, total_value - accounted_value)
            accounted[field_name] = (accounted_value
                                      if field_name == "latency_ms"
                                      else int(accounted_value))
            unattributed[field_name] = (missing_value
                                        if field_name == "latency_ms"
                                        else int(missing_value))
        if any(float(value or 0) > 0 for value in unattributed.values()):
            record_llm_usage_bucket(
                trace.metrics, "unattributed", unattributed)
        usage_by_agent["episode_total"] = {
            **episode_usage,
            "accounted_total_tokens": int(accounted["total_tokens"]),
            "unattributed_total_tokens": int(unattributed["total_tokens"]),
        }
        graph_report = validate_graph(self.registry, self.tool_registry)
        trace.metrics["skill_graph_valid"] = int(graph_report.passed)
        trace.metrics["skill_graph_error_count"] = len(graph_report.errors)
        # 进化后再次保存，使验证门禁结果与该 episode 的 Trace 一起可审计。
        self.trace_store.save(trace)

        # runtime instance + metrics
        runtime_graph = self._last_runtime_graph or RuntimeGraph(task.task_id, plan)
        instance = runtime_graph.build_execution_instance(
            task, trace.trace_id, trace.benchmark_result, trace.success)
        self.run_store.save(instance)

        self.episode_count += 1
        if trace.success:
            self.success_count += 1
        node_metrics = _runtime_node_metrics(runtime_graph)
        node_mode_counts = dict(node_metrics["node_mode_counts"])
        executed_node_count = int(node_metrics["executed_node_count"])
        already_satisfied_node_count = int(
            node_metrics["already_satisfied_node_count"])
        completed_node_count = int(node_metrics["completed_node_count"])
        validation_summary = {
            "tool_passed": sum(bool(item.get("passed"))
                               for item in trace.validation_layers.get("tool", [])),
            "tool_total": len(trace.validation_layers.get("tool", [])),
            "atomic_passed": sum(bool(item.get("passed"))
                                 for item in trace.validation_layers.get("atomic", [])),
            "atomic_total": len(trace.validation_layers.get("atomic", [])),
            "composite_passed": bool(
                (trace.validation_layers.get("composite") or {}).get("passed", True)),
            "benchmark_passed": bool(trace.success),
        }
        goal_terminal_before_plan_complete = bool(
            trace.success
            and len(runtime_graph.nodes) > 0
            and completed_node_count < len(runtime_graph.nodes)
        )
        episode = {
            "episode": self.episode_count,
            "task_id": task.task_id,
            "benchmark": task.benchmark,
            "task_type": task.task_type,
            "success": trace.success,
            "benchmark_won": bool(trace.metrics.get(
                "benchmark_won", trace.success)),
            "runtime_contract_valid": bool(trace.metrics.get(
                "runtime_contract_valid", True)),
            "learning_eligible": bool(trace.metrics.get(
                "learning_eligible", True)),
            "start_mode": trace.start_mode,
            "planning_mode": trace.planning_mode,
            "failure_type": trace.failure_type,
            "infrastructure_failure": trace.failure_type == "llm_error",
            "task_failure": bool(not trace.success and trace.failure_type != "llm_error"),
            "infrastructure_episode_retries": int(
                trace.metrics.get("infrastructure_episode_retries") or 0),
            "infrastructure_attempts": list(
                trace.metrics.get("infrastructure_attempts") or []),
            "infrastructure_errors": list(
                trace.metrics.get("infrastructure_errors") or []),
            "retries": trace.retries,
            "tokens": int(trace.token_cost),
            "llm_usage_by_agent": copy.deepcopy(
                trace.metrics.get("llm_usage_by_agent") or {}),
            "latency_ms": round(trace.latency_ms, 1),
            "direct_reuse_count": trace.direct_use_count(),
            "seeded_generation_count": trace.seeded_use_count(),
            "dynamic_generation_count": trace.dynamic_use_count(),
            "evolution": evolution,
            "graph_validation": graph_report.to_dict(),
            "node_mode_counts": node_mode_counts,
            "planned_node_count": len(runtime_graph.nodes),
            "executed_node_count": executed_node_count,
            "already_satisfied_node_count": already_satisfied_node_count,
            "completed_node_count": completed_node_count,
            "not_started_node_count": int(node_metrics["not_started_node_count"]),
            # The benchmark may legally terminate before every retrieved node is
            # needed.  Keep this separate from a runtime/tool failure.  This is
            # observable in ALFWorld games whose initial PDDL state already
            # satisfies one component of the natural-language goal.
            "goal_terminal_before_plan_complete": goal_terminal_before_plan_complete,
            "goal_terminal_skipped_node_count": (
                len(runtime_graph.nodes) - completed_node_count
                if goal_terminal_before_plan_complete else 0
            ),
            "fallback_node_count": sum(bool(node.fallback_reason)
                                       for node in runtime_graph.nodes),
            "validation_summary": validation_summary,
            "proposal_replay_count": len(evolution.get("proposal_replays") or []),
            "trace_id": trace.trace_id,
            # Retrieval is only candidate generation.  Formal reuse evidence
            # requires a selected registered Atomic to run and pass validation.
            "retrieved_skill_refs": list(trace.retrieved_skill_refs),
            "selected_skill_refs": list(node_metrics["selected_skill_refs"]),
            "executed_skill_refs": list(node_metrics["executed_skill_refs"]),
            "successful_reused_skill_refs": list(
                node_metrics["successful_reused_skill_refs"]),
            "successful_atomic_reuse_count": int(
                node_metrics["successful_atomic_reuse_count"]),
            "successful_tool_refs": list(
                node_metrics["successful_tool_refs"]),
            # Backward-compatible key, with corrected realized semantics.
            "reused_skill_refs": list(
                node_metrics["successful_reused_skill_refs"]),
            "used_tool_refs": list(trace.tool_refs),
            "cross_task_type_reuse": self._cross_task_type_reuse(
                task, trace,
                list(node_metrics["successful_reused_skill_refs"])),
            "cross_task_type_tool_reuse": self._cross_task_type_tool_reuse(
                task, list(node_metrics["successful_tool_refs"])),
            "skill_graph": self.registry.stats(),
            "tool_repo": self.tool_registry.stats(),
        }
        self.metrics_store.save_episode(self.episode_count, episode)
        return episode

    def _combined_llm_tokens(self) -> int:
        """Runtime and Extractor are independent clients but one episode cost."""
        return int(self._combined_llm_usage_snapshot()["total_tokens"])

    def _combined_llm_usage_snapshot(self) -> dict[str, int | float]:
        """Sum distinct Runtime and Extractor client usage counters."""
        clients = {id(self.llm): self.llm, id(self.extractor_llm): self.extractor_llm}
        combined: dict[str, int | float] = {
            field_name: 0.0 if field_name == "latency_ms" else 0
            for field_name in LLM_USAGE_FIELDS
        }
        for client in clients.values():
            snapshot = snapshot_llm_usage(client)
            for field_name in LLM_USAGE_FIELDS:
                combined[field_name] = (float(combined[field_name])
                                        + float(snapshot[field_name]))
        for field_name in LLM_USAGE_FIELDS:
            if field_name != "latency_ms":
                combined[field_name] = int(combined[field_name])
        return combined

    def _cross_task_type_reuse(
            self, task: Task, trace: TraceRecord,
            successful_skill_refs: list[str] | None = None) -> bool:
        """是否发生了跨 task_type 的能力复用（§58.3 指标）。"""
        # Merely retrieving or planning a node is not reuse evidence.  Formal
        # metrics pass the Atomics that actually executed and validated.  Keep
        # the old realized-node fallback only for external callers.
        used_refs = (list(successful_skill_refs)
                     if successful_skill_refs is not None else
                     [str(node["ref"]) for node in trace.realized_atomic_nodes
                      if node.get("ref")])
        seen_task_skills: set[str] = set()
        for ref_text in used_refs:
            try:
                from .core.refs import SkillRef
                ref = SkillRef.parse(ref_text)
            except ValueError:
                continue
            # Cross-task reuse is a task-level boolean; duplicate refs do not
            # need to be inspected repeatedly.
            if ref.logical_id in seen_task_skills:
                continue
            seen_task_skills.add(ref.logical_id)
            obj = self.registry.get(ref)
            if obj is None or not hasattr(obj, "metadata"):
                continue
            labels = set(obj.metadata.get("task_type_labels") or [])
            if task.task_type and labels and not labels.issubset({task.task_type}):
                return True
        return False

    def _cross_task_type_tool_reuse(
            self, task: Task, successful_tool_refs: list[str]) -> bool:
        """Successful Direct Tool reuse uses its own denominator and field."""
        source_types: set[str] = set()
        for tool_ref_text in successful_tool_refs:
            try:
                from .core.refs import ToolRef
                ref = ToolRef.parse(tool_ref_text)
            except ValueError:
                continue
            tool = self.tool_registry.get(ref)
            if tool is not None:
                source_types |= set(
                    tool.provenance.get("source_task_types") or [])
        return bool(task.task_type and source_types
                    and not source_types.issubset({task.task_type}))

    # ------------------------------------------------------------------
    # Code / Math 任务
    # ------------------------------------------------------------------
    def _run_code_task(self, task: Task, plan) -> TraceRecord:
        trace = TraceRecord(
            task_id=task.task_id, task_type=task.task_type, task_goal=task.goal,
            benchmark=task.benchmark, start_mode=plan.start_mode,
        )
        runtime_graph = RuntimeGraph(task.task_id, plan)
        self._last_runtime_graph = runtime_graph

        if not plan.nodes:
            trace.planning_mode = "pure_dynamic"
            runtime_graph.record_usage(ExecutionMode.DYNAMIC)
            result = self.bridge.generate_dynamic(task)
            self._fill_code_trace(trace, result, runtime_graph)
            return trace

        # warm：尝试每个节点的 direct → seeded → dynamic
        seed_parts: list[str] = []
        for node_index, planned in enumerate(plan.nodes):
            node = runtime_graph.nodes[node_index]
            atomic = self.registry.get(planned.ref)
            if atomic is None:
                runtime_graph.mark_fallback(node_index, ExecutionMode.DYNAMIC,
                                            "atomic_missing")
                node_index += 1
                continue

            choice = self.selector.select(atomic.ref, {"inputs": node.params,
                                                       "harness": "code_math"})
            if choice.implementation is None:
                runtime_graph.mark_fallback(node_index, ExecutionMode.DYNAMIC,
                                            choice.reason)
                node_index += 1
                continue
            node.impl_ref = str(choice.implementation.ref)
            resolved = self.resolver.resolve(choice.implementation,
                                             {"inputs": node.params})
            gate = self.bridge.direct_gate(atomic, choice.implementation, resolved,
                                           task.state, {"inputs": node.params})
            if gate.eligible:
                node.mode = ExecutionMode.DIRECT
                node.tool_refs = [str(r.binding.tool_ref) for r in resolved]
                runtime_graph.record_usage(ExecutionMode.DIRECT)
                direct_result = self.bridge.execute_direct(task, atomic,
                                                           choice.implementation,
                                                           resolved, task.state)
                feedback = direct_result.get("feedback") or {}
                attempt = AttemptRecord(
                    index=0, stage="direct_tool",
                    candidate=choice.implementation.tool_bindings[0].tool_ref.tool_id,
                    passed=bool(direct_result.get("passed")),
                    feedback=feedback, failure_type="" if direct_result.get("passed") else "tool_execution_error",
                )
                trace.attempts.append(attempt)
                after = direct_result.get("after") or {}
                validation = self.node_validator.validate_atomic(
                    atomic, task.state, after, inputs=node.params,
                    context={"harness": "code_math"})
                validation.node_ref = f"skill://{atomic.ref.logical_id}@{atomic.ref.version}"
                node.validation = validation
                node.passed = validation.passed
                node.after = after
                node.attempt_started = True
                node.executed_action_count = 1
                node.execution_status = (
                    NodeExecutionStatus.EXECUTED_SUCCESS
                    if validation.passed else
                    NodeExecutionStatus.EXECUTED_FAILURE)
                node.outputs = (materialize_atomic_outputs(
                    atomic, node.params, task.state, after,
                    tool_result=_execution_output_payload(direct_result))
                    if validation.passed else {})
                node.attempts.append({
                    "mode": ExecutionMode.DIRECT.value, "started": True,
                    "passed": validation.passed,
                    "failure_type": "" if validation.passed else
                                    "tool_execution_error",
                    "failure_stage": "validation",
                    "failure_cause": "" if validation.passed else
                                     "effect_violation",
                    "tool_refs": list(node.tool_refs),
                    "action_start": 0, "action_end": 1, "action_count": 1,
                    "step_id": node.step_id,
                    "occurrence_id": node.occurrence_id,
                })
                trace.node_validators.append(validation)
                self._record_tool_feedback(node.tool_refs, validation.passed, "direct")
                self._record_impl_feedback([node.impl_ref], validation.passed)
                if validation.passed:
                    runtime_graph.record_direct_success()
                    trace.success = True
                    trace.candidate_code = attempt.candidate
                    trace.benchmark_result = {"passed": True,
                                              "tests": feedback.get("tests", [])}
                    runtime_graph.apply_to_trace(trace)
                    trace.state_snapshots = [
                        {"step": 0, "state": task.state},
                        {"step": 1, "state": after},
                    ]
                    return trace
                runtime_graph.mark_fallback(node_index, ExecutionMode.SEEDED,
                                            "direct_failed")
            else:
                runtime_graph.mark_fallback(node_index, ExecutionMode.SEEDED,
                                            gate.reason)
            node.tool_refs = [str(r.binding.tool_ref) for r in resolved]
            seed_parts.append(self.bridge.seed_context(atomic, choice.implementation,
                                                       resolved, self.tool_registry))
            node_index += 1

        # seeded（所有规划节点 guideline + 工具上下文）
        seed_context = "\n\n".join(seed_parts)
        if not seed_context:
            runtime_graph.record_usage(ExecutionMode.DYNAMIC)
            result = self.bridge.generate_dynamic(task)
            self._fill_code_trace(trace, result, runtime_graph,
                                  stage_prefix="dynamic")
            return trace
        runtime_graph.record_usage(ExecutionMode.SEEDED)
        result = self.bridge.generate_seeded(task, seed_context)
        self._fill_code_trace(trace, result, runtime_graph, stage_prefix="seeded")
        # Seeded 只把 Tool 当作提示上下文，并没有执行该 executable。LLM 即使
        # 补救成功，也不能给此前 Direct 失败的 Tool 记成功证据。
        self._record_impl_feedback(
            [n.impl_ref for n in runtime_graph.nodes if n.impl_ref], trace.success)
        if trace.success:
            return trace
        # seeded 失败 → pure dynamic fallback
        trace.metrics["seeded_fallback_to_dynamic"] = 1
        runtime_graph.record_usage(ExecutionMode.DYNAMIC)
        dynamic_result = self.bridge.generate_dynamic(task)
        self._fill_code_trace(trace, dynamic_result, runtime_graph,
                              stage_prefix="dynamic")
        return trace

    def _fill_code_trace(self, trace: TraceRecord, result: dict[str, Any],
                         runtime_graph: RuntimeGraph,
                         stage_prefix: str = "dynamic") -> None:
        trace.success = bool(result.get("success"))
        trace.failure_type = str(result.get("failure_type") or "")
        trace.retries = int(result.get("retry_count") or 0)
        trace.candidate_code = str(result.get("candidate_code") or "")
        feedback = result.get("feedback") or {}
        trace.benchmark_result = {"passed": trace.success, **feedback}
        attempts = result.get("attempts") or []
        offset = len(trace.attempts)
        trace.attempts.extend(AttemptRecord(
            index=offset + local_index,
            stage=f"{stage_prefix}_{a.get('stage', 'draft')}",
            candidate=a.get("code", ""), passed=bool(a.get("verify", {}).get("passed")),
            feedback=a.get("verify", {}),
            failure_type=a.get("verify", {}).get("failure_type", ""),
        ) for local_index, a in enumerate(attempts))
        entry = _entry_of(trace)
        final_facts = [f"callable_returns_expected({entry})"] if trace.success else []
        trace.state_snapshots = [
            {"step": 0, "state": {"facts": [], "text": trace.task_goal}},
            {"step": len(trace.attempts), "state": {"facts": final_facts,
                                                    "text": "passed" if trace.success else "failed"}},
        ]
        runtime_graph.apply_to_trace(trace)

    # ------------------------------------------------------------------
    # 交互环境任务
    # ------------------------------------------------------------------
    def _run_env_task(self, task: Task, plan) -> TraceRecord:
        trace = TraceRecord(
            task_id=task.task_id, task_type=task.task_type, task_goal=task.goal,
            benchmark=task.benchmark, start_mode=plan.start_mode,
        )
        # 记录源任务定位信息（交互环境 Tool admission 的 replay 需要）
        trace.provenance["env_index"] = task.context.get("env_index")
        trace.provenance["params"] = dict(task.context.get("params") or {})
        trace.provenance["semantic_params"] = dict(
            task.context.get("semantic_params") or task.context.get("params") or {})
        trace.metrics["planning_audit"] = _build_planning_audit(
            task, plan, self.registry)
        runtime_graph = RuntimeGraph(task.task_id, plan)
        self._last_runtime_graph = runtime_graph

        if not plan.nodes:
            trace.planning_mode = "pure_dynamic"
            runtime_graph.record_usage(ExecutionMode.DYNAMIC)
            result = self.adapter.run_env_episode(task, self.llm, seed_context="",
                                                  max_steps=self.config.max_steps,
                                                  stop_effects=task.target_effects,
                                                  effect_inputs=dict(
                                                      task.context.get("params") or {}))
            self._fill_env_trace(trace, result, runtime_graph)
            return trace

        # 支持原地续跑的状态环境采用逐节点 mixed execution。每个节点可以独立
        # direct→seeded→dynamic，前一节点成功产生的环境状态直接交给下一节点。
        if bool(getattr(self.adapter, "supports_in_place_resume", False)):
            return self._run_env_nodes(task, plan, trace, runtime_graph)

        # 尝试为所有节点准备 direct 执行
        direct_steps: list[dict[str, Any]] = []
        seed_parts: list[str] = []
        all_direct = True
        for index, planned in enumerate(plan.nodes):
            atomic = self.registry.get(planned.ref)
            node = runtime_graph.nodes[index]
            if atomic is None:
                all_direct = False
                runtime_graph.mark_fallback(index, ExecutionMode.DYNAMIC, "atomic_missing")
                continue
            choice = self.selector.select(atomic.ref, {"inputs": planned.params,
                                                       "harness": "env"})
            if choice.implementation is None:
                all_direct = False
                runtime_graph.mark_fallback(index, ExecutionMode.DYNAMIC, choice.reason)
                continue
            node.impl_ref = str(choice.implementation.ref)
            resolved = self.resolver.resolve(choice.implementation,
                                             {"inputs": planned.params})
            gate = self.bridge.direct_gate(atomic, choice.implementation, resolved,
                                           task.state, {"inputs": planned.params})
            if not gate.eligible:
                all_direct = False
                runtime_graph.mark_fallback(index, ExecutionMode.SEEDED, gate.reason)
                node.tool_refs = [str(r.binding.tool_ref) for r in resolved]
                seed_parts.append(self.bridge.seed_context(atomic, choice.implementation,
                                                           resolved, self.tool_registry))
                continue
            steps = []
            for resolved_tool in resolved:
                tool = resolved_tool.tool
                for step in (tool.artifact.get("steps") or []):
                    steps.append({"template": str(step),
                                  "params": dict(resolved_tool.parameters)})
            if not steps:
                all_direct = False
                runtime_graph.mark_fallback(index, ExecutionMode.SEEDED, "empty_template")
                seed_parts.append(self.bridge.seed_context(atomic, choice.implementation,
                                                           resolved, self.tool_registry))
                continue
            node.mode = ExecutionMode.DIRECT
            node.tool_refs = [str(r.binding.tool_ref) for r in resolved]
            direct_steps.append({
                "node_ref": f"skill://{atomic.ref.logical_id}@{atomic.ref.version}",
                "step_id": planned.step_id,
                "steps": steps,
                "atomic": atomic,
                "inputs": planned.params,
                "graph_index": index,
            })

        if all_direct and direct_steps:
            result = self.adapter.run_env_episode(
                task, self.llm, seed_context="",
                direct_steps=[{"node_ref": d["node_ref"],
                               "step_id": d["step_id"],
                               "steps": d["steps"],
                               "tool_ref": ""}
                              for d in direct_steps],
                max_steps=self.config.max_steps)
            # 节点级验证：按 direct_steps 边界映射前后状态
            self._validate_env_direct_nodes(trace, result, direct_steps, runtime_graph)
            for node in runtime_graph.nodes:
                if node.attempt_started:
                    runtime_graph.record_usage(ExecutionMode.DIRECT)
                    if node.passed:
                        runtime_graph.record_direct_success()
                if node.tool_refs and node.attempt_started:
                    self._record_tool_feedback(node.tool_refs, node.passed, "direct")
                if node.impl_ref and node.attempt_started:
                    self._record_impl_feedback([node.impl_ref], node.passed)
            if result.success:
                self._fill_env_trace(trace, result, runtime_graph)
                return trace
            # direct 失败 → seeded 原地降级（从失败点继续，不重开 episode）
            trace.metrics["direct_fallback_to_seeded"] = 1
            direct_resume = _env_resume_payload(result)
        else:
            direct_resume = None

        # seeded 模式
        runtime_graph.record_usage(ExecutionMode.SEEDED)
        seed_context = "\n\n".join(seed_parts) if seed_parts else _guidelines_of(plan, self.registry)
        can_resume = bool(getattr(self.adapter, "supports_in_place_resume", False))
        result = self.adapter.run_env_episode(
            task, self.llm, seed_context=seed_context,
            max_steps=self.config.max_steps,
            **({"resume": direct_resume} if (can_resume and direct_resume) else {}))
        self._fill_env_trace(trace, result, runtime_graph)
        # Seeded LLM 成功不等于绑定 Tool 成功；Tool 仅在实际 Direct 执行时记账。
        self._record_impl_feedback(
            [n.impl_ref for n in runtime_graph.nodes if n.impl_ref], trace.success)
        if trace.success or not seed_context:
            return trace
        # seeded 失败 → pure dynamic 原地降级（从 seeded 失败点继续）
        trace.metrics["seeded_fallback_to_dynamic"] = 1
        runtime_graph.record_usage(ExecutionMode.DYNAMIC)
        # resume 后 result 包含合并的完整轨迹；清空已填入的 direct+seeded 动作，
        # 避免 _fill_env_trace 重复追加
        trace.actions = []
        trace.state_snapshots = []
        trace.observations = []
        dynamic_result = self.adapter.run_env_episode(
            task, self.llm, seed_context="",
            max_steps=self.config.max_steps,
            **({"resume": _env_resume_payload(result)} if can_resume else {}))
        self._fill_env_trace(trace, dynamic_result, runtime_graph)
        return trace

    def _run_env_nodes(self, task: Task, plan, trace: TraceRecord,
                       runtime_graph: RuntimeGraph) -> TraceRecord:
        """在同一环境 episode 中逐原子节点执行并局部降级。"""
        resume: dict[str, Any] | None = None
        last_result: Any = None
        shared_bindings = dict(task.context.get("params") or {})
        for index, planned in enumerate(plan.nodes):
            node = runtime_graph.nodes[index]
            node_start_actions = len((resume or {}).get("actions") or [])
            global_limit = int(self.config.max_steps)
            budget_scope = str(getattr(planned, "budget_scope", "atomic")
                               or "atomic")
            if budget_scope not in {"task", "atomic", "gap"}:
                raise ValueError(f"invalid_planned_node_budget_scope:{budget_scope}")
            if budget_scope == "task":
                node_budget = max(0, global_limit - node_start_actions)
                configured_attempt_limit = node_budget
            elif budget_scope == "gap":
                node_budget = int(
                    self.config.thresholds.env_dynamic_node_max_steps)
                configured_attempt_limit = node_budget
            else:
                node_budget = int(self.config.thresholds.env_node_max_steps)
                configured_attempt_limit = int(
                    self.config.thresholds.env_attempt_max_steps)
            node_deadline = min(global_limit,
                                node_start_actions + max(0, node_budget))
            routing_audit: dict[str, Any] = {
                "step_id": planned.step_id,
                "node_ref": str(planned.ref),
                "source": planned.source,
                "dynamic": bool(planned.dynamic),
                "budget_scope": budget_scope,
                "global_limit": global_limit,
                "node_limit": node_budget,
                "attempt_limit": configured_attempt_limit,
                "absolute_deadline": node_deadline,
                "initial_params": dict(planned.params),
                "target_effects": list(planned.target_effects or []),
                "implementation_candidates": [],
            }
            trace.metrics.setdefault("execution_routing", []).append(
                routing_audit)
            before = dict((resume or {}).get("state") or task.state or {})
            distinct_allocations, distinct_allocation_groups = (
                _runtime_distinct_allocations(
                task, planned, runtime_graph, index, before)
            )
            distinct_exclusions = _runtime_distinct_exclusions(
                task, planned, runtime_graph, index, before,
                distinct_allocations, distinct_allocation_groups)
            routing_audit["distinct_binding_allocations"] = dict(
                distinct_allocations)
            routing_audit["distinct_group_allocations"] = copy.deepcopy(
                distinct_allocation_groups)
            routing_audit["distinct_binding_exclusions"] = {
                role: sorted(str(value) for value in values)
                for role, values in distinct_exclusions.items()
            }
            planned.params = _apply_runtime_data_bindings(
                planned.params, planned.step_id, runtime_graph, shared_bindings,
                distinct_exclusions)
            # A plan may initially contain a class-valued entity slot because a
            # concrete instance is still hidden. Once execution establishes an
            # instance identity, carry it through later DATA_FLOW edges.
            planned.params = _refine_env_object_binding(
                planned.params, before, distinct_exclusions)
            # Existing cardinality witnesses own the first K stable
            # occurrences. Apply that concrete allocation after shared/state
            # refinement so a class-level task binding cannot overwrite it.
            planned.params.update(distinct_allocations)
            # RuntimeNodeState is created before execution.  Keep its persisted
            # view synchronized with parameters refined by prior-node data flow
            # or by framework discovery; otherwise node_results records the
            # stale class-valued planning input rather than the executed one.
            node.params = dict(planned.params)
            node.binding_provenance = _runtime_binding_provenance(
                task, planned, runtime_graph, before)
            atomic = None if planned.dynamic else self.registry.get(planned.ref)
            effects = list(planned.target_effects or getattr(atomic, "effects", []) or [])
            candidates: list[tuple[ExecutionMode, str, list[dict[str, Any]]]] = []
            location_discovered = False

            # A state that already satisfies this exact, grounded occurrence
            # advances control flow without pretending that a Tool/LLM ran.
            # Cardinality branches require a concrete unused instance; a
            # class-valued slot may not let occurrence two reuse occurrence one.
            allocated_cardinality_occurrence = (
                _cardinality_allocation_covers_node(
                    planned, distinct_allocations,
                    distinct_allocation_groups))
            if (effects and (
                    allocated_cardinality_occurrence
                    or _can_mark_already_satisfied(
                        task, planned, effects, before, runtime_graph, index,
                        distinct_exclusions))):
                already_check = (
                    "cardinality_occurrence_already_satisfied"
                    if allocated_cardinality_occurrence
                    else "effects_already_satisfied")
                validation = NodeValidationResult(
                    node_ref=str(planned.ref), level="atomic", passed=True,
                    step_id=node.step_id, occurrence_id=node.occurrence_id,
                    attempt_index=-1, mode=NodeExecutionStatus.ALREADY_SATISFIED.value,
                    checks={already_check: True},
                    before=before, after=before,
                    messages=[
                        "cardinality occurrence allocated to an existing "
                        "distinct goal witness"
                        if allocated_cardinality_occurrence else
                        "occurrence effect already satisfied before execution"],
                )
                node.validation = validation
                node.before = dict(before)
                node.after = dict(before)
                node.passed = True
                node.execution_status = NodeExecutionStatus.ALREADY_SATISFIED
                node.satisfied_without_execution = True
                node.outputs = materialize_atomic_outputs(
                    atomic, planned.params, before, before)
                trace.node_validators.append(validation)
                _update_verified_runtime_bindings(
                    task, planned, runtime_graph, shared_bindings, index)
                routing_audit["candidate_modes"] = []
                routing_audit["already_satisfied"] = True
                routing_audit["final_passed"] = True
                routing_audit["attempts"] = []
                continue

            if atomic is not None:
                produces_possession = any(
                    str(effect.get("predicate") or "").replace("_", ".") == "agent.holds"
                    for effect in effects if isinstance(effect, dict))
                discover = getattr(self.adapter, "discover_object_location", None)
                # Location slots come from the learned Atomic/Tool interface,
                # never from task_type or a fixed operation list. Bind anything
                # already witnessed in state before opening the bounded search.
                planned.params = _bind_known_location_slots(
                    planned.params, atomic, before, distinct_exclusions)
                node.params = dict(planned.params)
                node.binding_provenance = _runtime_binding_provenance(
                    task, planned, runtime_graph, before)
                tool_location_slots = self.selector.discoverable_location_slots(
                    atomic.ref, {"inputs": planned.params, "harness": "env"})
                contract_location_slots = {
                    str(item.get("name")) for item in (atomic.inputs or [])
                    if isinstance(item, dict)
                    and str(item.get("name") or "").endswith("_location")
                    and not is_concrete_binding(
                        planned.params.get(str(item.get("name"))))
                }
                # Resolving a declared Atomic source slot is part of executing
                # that contract, not an optional framework-search ablation.
                # The feature flag only opts additional Tool-only parameters
                # into discovery; it may never let an unresolved Atomic input
                # fall through to Seeded/Dynamic execution.
                mandatory_location_roles = {
                    slot: _source_location_entity_role(
                        atomic, slot, planned.params)
                    for slot in contract_location_slots
                }
                mandatory_location_roles = {
                    slot: role for slot, role in mandatory_location_roles.items()
                    if role and is_concrete_binding(planned.params.get(role))
                }
                location_slots = set(contract_location_slots)
                if self.config.features.enable_framework_discovery:
                    location_slots |= set(tool_location_slots)
                routing_audit["mandatory_location_slots"] = sorted(
                    mandatory_location_roles)
                discovery_result = None
                if callable(discover) and location_slots:
                    # Parameter discovery belongs to execution of the learned
                    # Atomic contract, not to Tool evolution.  Atomic-only must
                    # therefore receive the same bounded binding opportunity.
                    # A Tool ref is optional audit context when a usable
                    # implementation happens to exist.
                    discovery_tool_refs: list[str] = []
                    if self.config.features.enable_tool_evolution:
                        partial = self.selector.select_allowing_missing(
                            atomic.ref,
                            {"inputs": planned.params, "harness": "env"},
                            set(location_slots))
                        if partial.implementation is not None:
                            partial_resolved = self.resolver.resolve(
                                partial.implementation, {"inputs": planned.params})
                            discovery_tool_refs = [str(item.binding.tool_ref)
                                                   for item in partial_resolved]
                    for location_slot in sorted(location_slots):
                        if is_concrete_binding(planned.params.get(location_slot)):
                            continue
                        entity_role = (_source_location_entity_role(
                            atomic, location_slot, planned.params)
                            or location_slot[:-len("_location")])
                        entity_value = planned.params.get(entity_role)
                        if entity_value in (None, ""):
                            continue
                        excluded_objects = set(
                            distinct_exclusions.get(entity_role) or set())
                        binding, discovery_result = discover(
                            task, str(entity_value), resume=resume,
                            max_locations=self.config.thresholds.acquire_discovery_max_locations,
                            action_deadline=node_deadline,
                            node_ref=str(planned.ref),
                            tool_ref=discovery_tool_refs[0]
                            if discovery_tool_refs else "",
                            excluded_objects=excluded_objects,
                            allow_passive_navigable=not produces_possession)
                        resume = _env_resume_payload(discovery_result)
                        before = dict(resume.get("state") or before)
                        remapped = _remap_location_binding(
                            binding, entity_role, location_slot)
                        reused_discovery = _reused_distinct_bindings(
                            remapped, distinct_exclusions) if remapped else {}
                        if reused_discovery:
                            trace.metrics.setdefault(
                                "distinct_discovery_rejections", []).append({
                                    "node_ref": str(planned.ref),
                                    "bindings": dict(reused_discovery),
                                })
                            remapped = {}
                        discovery_start = node_start_actions
                        discovery_end = len(discovery_result.actions)
                        metric = {
                            "node_ref": str(planned.ref),
                            "entity_role": entity_role,
                            "location_role": location_slot,
                            "found": bool(remapped),
                            "binding": dict(remapped),
                            "excluded_objects": sorted(excluded_objects),
                            "checked_locations": list(
                                (before.get("meta") or {}).get(
                                    "checked_locations") or []),
                            "action_start": discovery_start,
                            "action_end": discovery_end,
                            "search_actions": max(0, discovery_end - discovery_start),
                        }
                        trace.metrics.setdefault(
                            "controlled_location_discovery", []).append(metric)
                        if produces_possession and entity_role == "object":
                            trace.metrics.setdefault(
                                "acquire_location_discovery", []).append(metric)
                        if remapped:
                            planned.params.update(remapped)
                            node.params = dict(planned.params)
                            for role, value in remapped.items():
                                node.binding_provenance[str(role)] = (
                                    BindingProvenance(
                                        source="runtime",
                                        role=str(role),
                                        evidence=(
                                            "framework_discovery",
                                            f"value={value}",
                                        ),
                                    ).to_dict())
                            location_discovered = True
                        else:
                            node.fallback_reason = (
                                f"location_discovery_failed:{location_slot}")
                            break
                discovery_action_end = len(
                    (resume or {}).get("actions") or [])
                discovery_action_count = max(
                    0, discovery_action_end - node_start_actions)
                node.executed_action_count += discovery_action_count
                routing_audit["preparation_action_count"] = (
                    discovery_action_count)
                unresolved_mandatory_locations = sorted(
                    slot for slot in mandatory_location_roles
                    if not is_concrete_binding(planned.params.get(slot)))
                if unresolved_mandatory_locations:
                    # Do not hand an unresolved relational source to a Seeded
                    # agent.  Besides duplicating bounded search, that permits
                    # an unrelated accepted action to manufacture a seemingly
                    # concrete source and poison failure/evolution statistics.
                    slot = unresolved_mandatory_locations[0]
                    cause = (str(getattr(discovery_result, "failure_type", "") or "")
                             if callable(discover)
                             else "location_discovery_unavailable")
                    cause = _attribute_env_budget_failure(
                        cause, action_end=discovery_action_end,
                        global_limit=global_limit,
                        node_deadline=node_deadline,
                        attempt_deadline=node_deadline,
                        budget_scope=budget_scope)
                    cause = cause or f"location_discovery_failed:{slot}"
                    if discovery_result is not None:
                        discovery_result.failure_type = cause
                    failure_stage = (
                        "budget" if cause in {
                            "episode_budget_exhausted",
                            "node_budget_exhausted",
                            "attempt_budget_exhausted",
                        } else "preparation")
                    node.fallback_reason = cause
                    node.before = dict(before)
                    node.after = dict(before)
                    node.execution_status = NodeExecutionStatus.NOT_STARTED
                    node.attempts.append({
                        "mode": "", "started": False, "passed": False,
                        "failure_type": cause,
                        "failure_stage": failure_stage,
                        "failure_cause": cause, "params": dict(planned.params),
                        "tool_refs": [], "action_start": node_start_actions,
                        "action_end": discovery_action_end,
                        "action_count": discovery_action_count,
                        "step_id": node.step_id,
                        "occurrence_id": node.occurrence_id,
                    })
                    routing_audit["candidate_modes"] = []
                    routing_audit["params_after_discovery"] = dict(planned.params)
                    routing_audit["location_discovered"] = False
                    routing_audit["unresolved_location_slots"] = (
                        unresolved_mandatory_locations)
                    routing_audit["attempts"] = [dict(item)
                                                 for item in node.attempts]
                    routing_audit["final_passed"] = False
                    routing_audit["fallback_reason"] = cause
                    trace.failure_stage = failure_stage
                    trace.failure_cause = cause
                    last_result = (discovery_result if discovery_result is not None
                                   else _not_started_env_result(resume, cause))
                    break
                selection_context = {
                    "inputs": planned.params, "harness": "env",
                    "prefer_minimal_after_preparation": location_discovered,
                }
                ranked_choices = self.selector.rank(atomic.ref, selection_context)
                fallback_payload = None
                direct_payload = None
                rejected: list[dict[str, Any]] = []
                for choice in ranked_choices:
                    implementation = choice.implementation
                    if implementation is None:
                        continue
                    resolved = self.resolver.resolve(
                        implementation, {"inputs": planned.params})
                    tool_checks = [self.tool_validator.validate_tool(
                        item.tool, item.parameters).to_dict()
                        for item in resolved if item.tool is not None]
                    payload = (implementation, resolved, tool_checks)
                    if fallback_payload is None:
                        fallback_payload = payload
                    gate = self.bridge.direct_gate(
                        atomic, implementation, resolved, before,
                        {"inputs": planned.params})
                    if tool_checks and not all(check["passed"] for check in tool_checks):
                        gate.eligible = False
                        gate.reason = (gate.reason + ";tool_validation_failed").strip(";")
                    steps: list[dict[str, Any]] = []
                    if gate.eligible:
                        for item in resolved:
                            for template in item.tool.artifact.get("steps") or []:
                                steps.append({"template": str(template),
                                              "params": dict(item.parameters)})
                    routing_audit["implementation_candidates"].append({
                        "implementation_ref": str(implementation.ref),
                        "implementation_status": implementation.status.value,
                        "resolved_tools": [{
                            **item.to_dict(),
                            "tool_status": (item.tool.status.value
                                            if item.tool is not None else "missing"),
                            "tool_usable": bool(item.tool is not None
                                                and item.tool.is_usable()),
                        } for item in resolved],
                        "direct_gate": gate.to_dict(),
                        "tool_validation": tool_checks,
                        "direct_step_count": len(steps),
                    })
                    if steps:
                        direct_payload = payload
                        node.impl_ref = str(implementation.ref)
                        node.tool_refs = [str(item.binding.tool_ref)
                                          for item in resolved]
                        trace.validation_layers.setdefault("tool", []).extend(tool_checks)
                        candidates.append((ExecutionMode.DIRECT, "", steps))
                        break
                    rejected.append({"implementation": str(implementation.ref),
                                     "reason": gate.reason or "empty_template"})
                if rejected:
                    trace.metrics.setdefault(
                        "direct_implementation_rejections", []).extend(rejected)
                seed_payload = direct_payload or fallback_payload
                if seed_payload is not None:
                    implementation, resolved, tool_checks = seed_payload
                    if direct_payload is None:
                        node.impl_ref = str(implementation.ref)
                        node.tool_refs = [str(item.binding.tool_ref)
                                          for item in resolved]
                        trace.validation_layers.setdefault("tool", []).extend(tool_checks)
                        node.fallback_reason = (rejected[-1]["reason"] if rejected
                                                else "no_direct_implementation")
                    seed = self.bridge.seed_context(
                        atomic, implementation, resolved, self.tool_registry)
                    if seed:
                        candidates.append((ExecutionMode.SEEDED, seed, []))
                elif not ranked_choices:
                    node.fallback_reason = "no_bindable_implementation"
                # 即使尚无 Implementation，Abstract Atomic 的契约/指南仍可作为种子。
                if not any(mode == ExecutionMode.SEEDED for mode, _, _ in candidates):
                    guideline = _guideline_of(atomic)
                    if guideline:
                        candidates.append((ExecutionMode.SEEDED, guideline, []))
            # dynamic 是每个节点最终兜底；动态 gap 直接从这里开始。
            candidates.append((ExecutionMode.DYNAMIC, "", []))
            routing_audit["candidate_modes"] = [mode.value
                                                for mode, _, _ in candidates]
            routing_audit["params_after_discovery"] = dict(planned.params)
            routing_audit["location_discovered"] = bool(location_discovered)

            required_slots = semantic_required_slots(effects)
            runtime_resolvable = _runtime_resolvable_semantic_slots(
                atomic, effects, planned.params)
            unresolved_core = sorted(
                slot for slot in required_slots
                if (not is_concrete_binding(planned.params.get(slot))
                    and slot not in runtime_resolvable))
            routing_audit["semantic_required_slots"] = sorted(required_slots)
            routing_audit["runtime_resolvable_semantic_slots"] = sorted(
                runtime_resolvable)
            routing_audit["unresolved_semantic_slots"] = unresolved_core
            if runtime_resolvable:
                # Runtime-resolvable semantic values are a valid source-closed
                # Seeded/Dynamic route, but cannot be sent to Direct as an
                # unresolved Tool argument.  The executed action/state Effect
                # must ground them before node validation and output materialization.
                candidates = [item for item in candidates
                              if item[0] != ExecutionMode.DIRECT]
                routing_audit["candidate_modes"] = [
                    mode.value for mode, _, _ in candidates]
            if unresolved_core:
                cause = "plan_binding_unresolved"
                node.fallback_reason = cause
                node.attempts.append({
                    "mode": "", "started": False, "passed": False,
                    "failure_type": cause, "failure_stage": "planning",
                    "failure_cause": cause, "params": dict(planned.params),
                    "tool_refs": [], "action_start": node_start_actions,
                    "action_end": node_start_actions, "action_count": 0,
                    "step_id": node.step_id,
                    "occurrence_id": node.occurrence_id,
                })
                routing_audit["attempts"] = [dict(item) for item in node.attempts]
                routing_audit["final_passed"] = False
                routing_audit["fallback_reason"] = cause
                trace.failure_stage = "planning"
                trace.failure_cause = cause
                last_result = _not_started_env_result(resume, cause)
                break

            node_succeeded = False
            for attempt_index, (mode, seed_context, steps) in enumerate(candidates):
                attempt_before = dict((resume or {}).get("state") or before)
                action_start = len((resume or {}).get("actions") or [])
                attempt_limit = configured_attempt_limit
                ledger = BudgetLedger(
                    global_limit=int(self.config.max_steps),
                    node_limit=max(0, node_budget),
                    attempt_limit=max(0, attempt_limit),
                    node_start=node_start_actions,
                    attempt_start=action_start,
                    actions_used=action_start,
                )
                remaining = ledger.attempt_remaining()
                attempt_deadline = ledger.absolute_deadline()
                if remaining <= 0:
                    if ledger.global_remaining() <= 0:
                        cause = "episode_budget_exhausted"
                    elif ledger.node_remaining() <= 0:
                        cause = "node_budget_exhausted"
                    else:
                        cause = "attempt_budget_exhausted"
                    node.attempts.append({
                        "mode": mode.value, "started": False, "passed": False,
                        "failure_type": "attempt_not_started",
                        "failure_stage": "budget", "failure_cause": cause,
                        "params": dict(planned.params), "tool_refs": [],
                        "action_start": action_start, "action_end": action_start,
                        "action_count": 0, "step_id": node.step_id,
                        "occurrence_id": node.occurrence_id,
                    })
                    node.fallback_reason = cause
                    trace.failure_stage = "budget"
                    trace.failure_cause = cause
                    last_result = _not_started_env_result(resume, cause)
                    break
                runtime_graph.record_usage(mode)
                node.attempt_started = True
                node.mode = mode
                direct_steps = None
                if mode == ExecutionMode.DIRECT:
                    direct_steps = [{
                        "node_ref": str(planned.ref), "tool_ref": node.tool_refs[0]
                        if node.tool_refs else "", "step_id": node.step_id,
                        "steps": steps,
                    }]
                result = _run_env_episode_with_optional_exclusions(
                    self.adapter,
                    task, self.llm, seed_context=seed_context,
                    direct_steps=direct_steps, max_steps=attempt_deadline,
                    resume=resume, stop_effects=effects,
                    effect_inputs=dict(planned.params),
                    excluded_effect_bindings=distinct_exclusions,
                    node_ref=str(planned.ref),
                    phase_goal=_phase_goal_of(
                        atomic, effects, planned.params,
                        distinct_exclusions),
                )
                last_result = result
                if getattr(result, "diagnostics", None):
                    routing_audit.setdefault("runtime_diagnostics", []).append({
                        "mode": mode.value,
                        "diagnostics": copy.deepcopy(result.diagnostics),
                    })
                after_payload = _env_resume_payload(result)
                after = dict(after_payload.get("state") or attempt_before)
                # Seeded/Dynamic agents can discover an initially unbound
                # entity while executing the node (for example the concrete
                # light used by a generic toggle capability).  Ground only
                # contract input slots from accepted action parameters and
                # observed state Effects, then validate with those realized
                # bindings.  Benchmark success is never used as a substitute
                # for this evidence.
                grounded, binding_evidence = _ground_env_runtime_params(
                    planned.params, atomic, effects, result,
                    action_start=action_start, before=attempt_before,
                    after=after, node_ref=str(planned.ref))
                planned.params = grounded
                node.params = dict(grounded)
                if binding_evidence:
                    for item in binding_evidence:
                        role = str(item.get("parameter") or "")
                        if not role:
                            continue
                        node.binding_provenance[role] = BindingProvenance(
                            source="runtime", role=role,
                            evidence=(str(item.get("source") or "runtime"),),
                        ).to_dict()
                    trace.metrics.setdefault("runtime_param_bindings", []).append({
                        "node_ref": str(planned.ref),
                        "mode": mode.value,
                        "bindings": binding_evidence,
                    })
                passed = bool(getattr(result, "atomic_complete", False))
                validation = None
                if atomic is not None:
                    validation = self.node_validator.validate_atomic(
                        atomic, attempt_before, after, inputs=planned.params,
                        context={"harness": "env"})
                    validation.node_ref = str(planned.ref)
                    validation.step_id = node.step_id
                    validation.occurrence_id = node.occurrence_id
                    validation.attempt_index = attempt_index
                    validation.mode = mode.value
                    validation.failure_stage = "validation"
                    passed = validation.passed or (bool(result.success) and not effects)
                    trace.node_validators.append(validation)
                elif effects:
                    from .core.predicates import StateSnapshot, check_effects
                    passed, missing = check_effects(StateSnapshot(after), planned.params,
                                                    effects, {"harness": "env"})
                    validation = NodeValidationResult(
                        node_ref=str(planned.ref), level="atomic", passed=passed,
                        step_id=node.step_id,
                        occurrence_id=node.occurrence_id,
                        attempt_index=attempt_index, mode=mode.value,
                        failure_stage="validation",
                        checks={"effects": passed}, before=before, after=after,
                        messages=[] if passed else [f"动态节点效果未发生：{missing}"],
                    )
                    trace.node_validators.append(validation)
                reused_distinct = _reused_distinct_bindings(
                    planned.params, distinct_exclusions)
                if reused_distinct:
                    passed = False
                    if validation is not None:
                        validation.passed = False
                        validation.checks["distinct_bindings"] = False
                        validation.messages.append(
                            "cross-branch distinct binding reused: "
                            f"{reused_distinct}")
                        if "distinct_binding_reused" not in validation.failure_codes:
                            validation.failure_codes.append(
                                "distinct_binding_reused")
                    if not result.failure_type:
                        result.failure_type = "distinct_binding_reused"
                    routing_audit.setdefault(
                        "distinct_binding_rejections", []).append({
                            "mode": mode.value,
                            "bindings": reused_distinct,
                        })
                action_end = len(after_payload.get("actions") or [])
                action_count = max(0, action_end - action_start)
                if result.success and not passed:
                    # Preserve benchmark-authoritative won for metrics, while
                    # the independent runtime-contract gate prevents this
                    # mismatch from writing positive or negative evolution
                    # evidence.
                    result.diagnostics.setdefault(
                        "contract_mismatch", []).append({
                            "node_ref": str(planned.ref),
                            "benchmark_reported_success": True,
                            "formal_node_validation_passed": False,
                        })
                    if not result.failure_type:
                        result.failure_type = (
                            "benchmark_goal_contract_mismatch")
                result.failure_type = _attribute_env_budget_failure(
                    str(result.failure_type or ""),
                    action_end=action_end,
                    global_limit=global_limit,
                    node_deadline=node_deadline,
                    attempt_deadline=attempt_deadline,
                    budget_scope=budget_scope,
                )
                node.executed_action_count += action_count
                node.attempts.append({"mode": mode.value, "started": True,
                                      "passed": passed,
                                      "failure_type": str(result.failure_type or ""),
                                      "failure_stage": "execution",
                                      "failure_cause": str(result.failure_type or ""),
                                      "params": dict(planned.params),
                                      "tool_refs": list(node.tool_refs)
                                      if mode == ExecutionMode.DIRECT else [],
                                      "action_start": action_start,
                                      "action_end": action_end,
                                      "action_count": action_count,
                                      "step_id": node.step_id,
                                      "occurrence_id": node.occurrence_id,
                                      "before": attempt_before,
                                      "after": after})
                node.validation = validation
                node.before, node.after, node.passed = before, after, passed
                node.execution_status = (
                    NodeExecutionStatus.EXECUTED_SUCCESS if passed
                    else NodeExecutionStatus.EXECUTED_FAILURE)
                resume = after_payload
                # Executable 证据必须按实际 attempt 记账。后续 Seeded/Dynamic
                # 是否救回节点，都不能覆盖本次 Direct Tool 的真实结果。
                if mode == ExecutionMode.DIRECT and node.tool_refs:
                    self._record_tool_feedback(node.tool_refs, passed, "direct")
                    if passed:
                        runtime_graph.record_direct_success()
                if mode == ExecutionMode.DIRECT and node.impl_ref:
                    self._record_impl_feedback([node.impl_ref], passed)
                if passed:
                    node.outputs = materialize_atomic_outputs(
                        atomic, planned.params, attempt_before, after,
                        tool_result=(_execution_output_payload(result)
                                     if mode == ExecutionMode.DIRECT else None))
                    _update_verified_runtime_bindings(
                        task, planned, runtime_graph, shared_bindings, index)
                    node_succeeded = True
                    break
                if result.success:
                    # Benchmark won 不能替代当前节点 Effect；保留不一致供审计。
                    break
                node.fallback_reason = str(result.failure_type or "effect_not_met")
                if not resume.get("admissible"):
                    break

            routing_audit["attempts"] = [dict(item) for item in node.attempts]
            routing_audit["final_passed"] = bool(node_succeeded)
            routing_audit["fallback_reason"] = str(node.fallback_reason or "")

            if last_result is not None and last_result.success:
                break
            if not node_succeeded:
                break

        # Freeze the self-sufficiency boundary before any task-level rescue.
        # A selected Composite is validated only against this state.  Missing
        # formal target Effects become one explicit task-gap occurrence;
        # protocol completion after all Effects hold is non-learnable.
        pre_gap_payload = dict(resume or {})
        pre_gap_state = dict(pre_gap_payload.get("state") or task.state or {})
        pre_gap_action_index = len(pre_gap_payload.get("actions") or [])
        trace.pre_gap_state = copy.deepcopy(pre_gap_state)
        trace.pre_gap_action_index = pre_gap_action_index
        trace.provenance["pre_gap_bindings"] = copy.deepcopy(shared_bindings)
        gap_analysis = analyze_task_gap(task, pre_gap_state, shared_bindings)
        trace.task_gap_analysis = gap_analysis

        planned_node_count = len(runtime_graph.nodes)
        all_nodes_passed = bool(planned_node_count) and all(
            node.passed for node in runtime_graph.nodes[:planned_node_count])
        benchmark_pending = last_result is None or not last_result.success
        if benchmark_pending and all_nodes_passed:
            if gap_analysis.missing_effects:
                final_actions = pre_gap_action_index
                if final_actions < int(self.config.max_steps):
                    runtime_graph.record_usage(ExecutionMode.DYNAMIC)
                    gap_index = runtime_graph.append_dynamic_gap(
                        gap_analysis.missing_effects, shared_bindings,
                        task_target_effects=list(task.target_effects or []))
                    gap_node = runtime_graph.nodes[gap_index]
                    gap_plan = runtime_graph.plan.nodes[gap_index]
                    gap_exclusions = _runtime_distinct_exclusions(
                        task, gap_plan, runtime_graph, gap_index,
                        pre_gap_state)
                    gap_params = _constrain_task_gap_params(
                        shared_bindings, gap_exclusions)
                    gap_plan.params = dict(gap_params)
                    gap_node.params = dict(gap_params)
                    gap_budget = int(
                        self.config.thresholds.env_dynamic_node_max_steps)
                    gap_deadline = min(
                        int(self.config.max_steps),
                        final_actions + max(0, gap_budget))
                    gap_routing_audit = {
                        "step_id": gap_plan.step_id,
                        "node_ref": str(gap_plan.ref),
                        "source": gap_plan.source,
                        "dynamic": True,
                        "budget_scope": gap_plan.budget_scope,
                        "global_limit": int(self.config.max_steps),
                        "node_limit": gap_budget,
                        "attempt_limit": gap_budget,
                        "absolute_deadline": gap_deadline,
                        "initial_params": dict(gap_params),
                        "distinct_binding_exclusions": {
                            role: sorted(str(value) for value in values)
                            for role, values in gap_exclusions.items()
                        },
                        "target_effects": list(
                            gap_analysis.missing_effects),
                        "candidate_modes": [ExecutionMode.DYNAMIC.value],
                        "implementation_candidates": [],
                    }
                    trace.metrics.setdefault("execution_routing", []).append(
                        gap_routing_audit)
                    gap_result = _run_env_episode_with_optional_exclusions(
                        self.adapter,
                        task, self.llm, seed_context="",
                        max_steps=gap_deadline, resume=resume,
                        stop_effects=gap_analysis.missing_effects,
                        effect_inputs=dict(gap_params),
                        excluded_effect_bindings=gap_exclusions,
                        node_ref=str(gap_plan.ref),
                        phase_goal=_phase_goal_of(
                            None, gap_analysis.missing_effects,
                            gap_params, gap_exclusions),
                    )
                    gap_end = len(gap_result.actions or [])
                    gap_result.failure_type = _attribute_env_budget_failure(
                        str(gap_result.failure_type or ""),
                        action_end=gap_end,
                        global_limit=int(self.config.max_steps),
                        node_deadline=gap_deadline,
                        attempt_deadline=gap_deadline,
                        budget_scope="gap",
                    )
                    _mark_result_action_origin(
                        gap_result, final_actions, gap_end,
                        origin="task_gap_agent", node_ref=str(gap_plan.ref))
                    gap_after_payload = _env_resume_payload(gap_result)
                    gap_after = dict(gap_after_payload.get("state") or pre_gap_state)
                    from .core.predicates import StateSnapshot, check_effects
                    gap_passed, gap_missing = check_effects(
                        StateSnapshot(gap_after), gap_params,
                        gap_analysis.missing_effects, {"harness": "env"})
                    distinct_gap_passed, distinct_gap_details = (
                        _validate_task_gap_distinct_progress(
                            task, gap_analysis.missing_effects,
                            pre_gap_state, gap_after, gap_params,
                            gap_exclusions))
                    gap_passed = bool(gap_passed and distinct_gap_passed)
                    gap_failure_codes: list[str] = []
                    gap_messages = ([] if not gap_missing else [
                        f"task gap effects not satisfied: {gap_missing}"])
                    if not distinct_gap_passed:
                        gap_result.failure_type = "distinct_binding_reused"
                        gap_failure_codes.append("distinct_binding_reused")
                        gap_messages.append(
                            "task gap did not add the required new distinct "
                            "target witness")
                    if not gap_passed:
                        # Preserve benchmark/harness won, but mark the formal
                        # mismatch so the runtime learning gate stays neutral.
                        if not str(gap_result.failure_type or ""):
                            gap_result.failure_type = (
                                "benchmark_goal_contract_mismatch"
                                if not gap_missing else "effect_not_met")
                        if (gap_missing
                                and "task_gap_effect_not_met"
                                not in gap_failure_codes):
                            gap_failure_codes.append(
                                "task_gap_effect_not_met")
                    gap_validation = NodeValidationResult(
                        node_ref=str(gap_plan.ref), level="task_gap",
                        passed=gap_passed, step_id=gap_node.step_id,
                        occurrence_id=gap_node.occurrence_id,
                        attempt_index=0, mode=ExecutionMode.DYNAMIC.value,
                        failure_stage="validation",
                        checks={
                            "missing_target_effects": not bool(gap_missing),
                            "distinct_target_progress": distinct_gap_passed,
                        },
                        before=pre_gap_state, after=gap_after,
                        messages=gap_messages,
                        failure_codes=gap_failure_codes,
                    )
                    gap_node.validation = gap_validation
                    gap_node.before, gap_node.after = pre_gap_state, gap_after
                    gap_node.passed = gap_passed
                    gap_node.distinct_witness_outputs = (
                        _materialize_task_gap_distinct_witness_outputs(
                            distinct_gap_details)
                        if gap_passed else {})
                    gap_node.attempt_started = gap_end > final_actions
                    gap_node.executed_action_count = max(
                        0, gap_end - final_actions)
                    gap_node.execution_status = (
                        NodeExecutionStatus.EXECUTED_SUCCESS if gap_passed
                        else NodeExecutionStatus.EXECUTED_FAILURE)
                    gap_node.fallback_reason = (
                        "" if gap_passed else
                        str(gap_result.failure_type or "effect_not_met"))
                    gap_node.attempts.append({
                        "mode": ExecutionMode.DYNAMIC.value,
                        "started": gap_node.attempt_started,
                        "passed": gap_passed,
                        "failure_type": str(gap_result.failure_type or ""),
                        "failure_stage": "execution",
                        "failure_cause": str(gap_result.failure_type or ""),
                        "params": dict(gap_params), "tool_refs": [],
                        "action_start": final_actions, "action_end": gap_end,
                        "action_count": gap_node.executed_action_count,
                        "step_id": gap_node.step_id,
                        "occurrence_id": gap_node.occurrence_id,
                        "before": pre_gap_state, "after": gap_after,
                        "distinct_progress": copy.deepcopy(
                            distinct_gap_details),
                    })
                    gap_routing_audit["distinct_progress"] = copy.deepcopy(
                        distinct_gap_details)
                    gap_routing_audit["attempts"] = [
                        dict(item) for item in gap_node.attempts]
                    gap_routing_audit["final_passed"] = bool(gap_passed)
                    gap_routing_audit["fallback_reason"] = str(
                        gap_node.fallback_reason or "")
                    trace.node_validators.append(gap_validation)
                    trace.runtime_spans.append(RuntimeSpan(
                        kind="task_gap", occurrence_id=gap_node.occurrence_id,
                        action_start=final_actions, action_end=gap_end,
                        node_ref=str(gap_plan.ref),
                        missing_effects=copy.deepcopy(
                            gap_analysis.missing_effects), learnable=True,
                    ))
                    trace.metrics["task_gap_required_count"] = 1
                    resume = gap_after_payload
                    last_result = gap_result
                    # Some harnesses require a final protocol action after the
                    # formal target state is reached.  Keep that separate from
                    # the learnable task-gap occurrence.
                    if (gap_passed and not gap_result.success
                            and gap_end < int(self.config.max_steps)):
                        last_result = self._run_benchmark_finalization(
                            task, resume, trace, gap_end)
                        resume = _env_resume_payload(last_result)
                else:
                    trace.failure_stage = "budget"
                    trace.failure_cause = "episode_budget_exhausted"
            elif gap_analysis.benchmark_only_finalization:
                final_actions = pre_gap_action_index
                if final_actions < int(self.config.max_steps):
                    last_result = self._run_benchmark_finalization(
                        task, resume, trace, final_actions)
                    resume = _env_resume_payload(last_result)
                else:
                    trace.failure_stage = "budget"
                    trace.failure_cause = "episode_budget_exhausted"
        if last_result is None:
            last_result = self.adapter.run_env_episode(task, self.llm,
                                                       max_steps=self.config.max_steps)
        _append_planned_runtime_spans(trace, runtime_graph)
        self._fill_env_trace(trace, last_result, runtime_graph)
        if trace.failure_cause and trace.failure_stage in {"planning", "budget"}:
            trace.failure_type = trace.failure_cause
        return trace

    def _run_benchmark_finalization(
            self, task: Task, resume: dict[str, Any] | None,
            trace: TraceRecord, action_start: int) -> Any:
        """Finish a benchmark protocol without generating learnable evidence."""
        result = self.adapter.run_env_episode(
            task, self.llm, seed_context="", max_steps=self.config.max_steps,
            resume=resume)
        action_end = len(result.actions or [])
        _mark_result_action_origin(
            result, action_start, action_end,
            origin="benchmark_finalization",
            node_ref="runtime.benchmark_finalization@0.0.0")
        trace.runtime_spans.append(RuntimeSpan(
            kind="benchmark_finalization",
            occurrence_id="benchmark_finalization_000",
            action_start=action_start, action_end=action_end,
            node_ref="runtime.benchmark_finalization@0.0.0",
            learnable=False,
        ))
        trace.metrics["benchmark_finalization_count"] = 1
        return result

    def _finalize_validation(self, trace: TraceRecord, task: Task) -> None:
        """持久化四层验证结果；各层独立记录，不用上层成功掩盖下层失败。"""
        realized_inputs = _realized_task_bindings(
            dict(task.context.get("params") or {}),
            trace.realized_atomic_nodes)
        # Persist the exact validation bindings separately from the semantic
        # task parameters.  This makes instance grounding auditable without
        # leaking concrete instances into learned Skill/Tool identity.
        trace.provenance["realized_params"] = copy.deepcopy(realized_inputs)
        trace.validation_layers["atomic"] = [item.to_dict()
                                               for item in trace.node_validators
                                               if item.level == "atomic"]
        if trace.selected_composite:
            try:
                from .core.refs import SkillRef
                composite = self.registry.get(SkillRef.parse(trace.selected_composite))
            except (ValueError, TypeError):
                composite = None
            if composite is not None:
                runtime_graph = getattr(self, "_last_runtime_graph", None)
                runtime_edges = ([edge.to_dict() for edge in runtime_graph.edges]
                                 if runtime_graph is not None else None)
                common_context = {
                    "harness": "env" if _is_env_task(task) else "code_math",
                    "realized_nodes": copy.deepcopy(trace.realized_atomic_nodes),
                    "runtime_edges": runtime_edges,
                    "require_realized_data_flow": runtime_edges is not None,
                }
                planned_results = [
                    item for item in trace.node_validators
                    if item.level == "atomic"
                    and not str(item.node_ref).startswith(
                        "runtime.dynamic.task_gap")
                ]
                self_result = self.composite_validator.validate_composite(
                    composite,
                    planned_results,
                    trace.pre_gap_state or trace.final_state(),
                    inputs=realized_inputs,
                    context={
                        **common_context,
                        "task_gap_required": bool(
                            trace.task_gap_analysis is not None
                            and trace.task_gap_analysis.missing_effects),
                        "task_gap_analysis": trace.task_gap_analysis,
                    },
                    registry=self.registry,
                )
                if (trace.task_gap_analysis is not None
                        and trace.task_gap_analysis.missing_effects):
                    self_result.checks["task_gap_not_required"] = False
                    if "task_gap_required" not in self_result.failure_codes:
                        self_result.failure_codes.append("task_gap_required")
                    self_result.messages.append(
                        "selected Composite required a task-gap rescue")
                    self_result.passed = False
                full_result = self.composite_validator.validate_composite(
                    composite, planned_results, trace.final_state(),
                    inputs=realized_inputs,
                    context=common_context,
                    registry=self.registry,
                )
                gap_results = [item for item in trace.node_validators
                               if item.level == "task_gap"]
                if gap_results:
                    full_result.checks["task_gap_effects_passed"] = all(
                        item.passed for item in gap_results)
                    full_result.passed = all(full_result.checks.values())
                trace.validation_layers["selected_composite_self"] = (
                    self_result.to_dict())
                trace.validation_layers["full_runtime_graph"] = (
                    full_result.to_dict())
                # Backward-compatible key now deliberately means self-
                # sufficient validation, never post-gap task success.
                trace.validation_layers["composite"] = self_result.to_dict()
        trace.validation_layers["benchmark"] = {
            "passed": bool(trace.success),
            "result": dict(trace.benchmark_result),
            "failure_type": trace.failure_type,
        }

    def _validate_env_direct_nodes(self, trace: TraceRecord, result: Any,
                                   direct_steps: list[dict[str, Any]],
                                   runtime_graph: RuntimeGraph) -> None:
        """按节点边界验证 direct 执行后的核心 Effect（§35.2）。"""
        states = [dict(s.get("state") or {}) for s in (result.states or [])]
        if not states:
            return
        spans = {str(item.get("step_id") or ""): dict(item)
                 for item in (getattr(result, "node_spans", []) or [])}
        action_index = 0
        for spec in direct_steps:
            step_count = len(spec["steps"])
            span = spans.get(str(spec.get("step_id") or ""), {})
            action_start = int(span.get("action_start", action_index))
            before = states[action_start] if action_start < len(states) else {}
            after_index = min(int(span.get("action_end",
                                           action_start + step_count)),
                              len(states) - 1)
            after = states[after_index]
            action_index = after_index
            atomic = spec["atomic"]
            node = runtime_graph.nodes[spec["graph_index"]]
            grounded, binding_evidence = _ground_env_runtime_params(
                spec["inputs"], atomic, list(atomic.effects or []), result,
                action_start=action_start, action_end=after_index,
                before=before, after=after, node_ref=spec["node_ref"])
            spec["inputs"].clear()
            spec["inputs"].update(grounded)
            node.params = dict(grounded)
            if binding_evidence:
                for item in binding_evidence:
                    role = str(item.get("parameter") or "")
                    if role:
                        node.binding_provenance[role] = BindingProvenance(
                            source="runtime", role=role,
                            evidence=(str(item.get("source") or "runtime"),),
                        ).to_dict()
                trace.metrics.setdefault("runtime_param_bindings", []).append({
                    "node_ref": spec["node_ref"],
                    "mode": ExecutionMode.DIRECT.value,
                    "bindings": binding_evidence,
                })
            validation = self.node_validator.validate_atomic(
                atomic, before, after, inputs=grounded,
                context={"harness": "env"})
            validation.node_ref = spec["node_ref"]
            validation.step_id = node.step_id
            validation.occurrence_id = node.occurrence_id
            validation.attempt_index = 0
            validation.mode = ExecutionMode.DIRECT.value
            validation.failure_stage = "validation"
            node.validation = validation
            node.before = before
            node.after = after
            node.passed = validation.passed
            node.attempt_started = after_index > action_start
            node.executed_action_count = max(0, after_index - action_start)
            node.satisfied_without_execution = bool(
                validation.passed and not node.attempt_started)
            node.execution_status = (
                NodeExecutionStatus.ALREADY_SATISFIED
                if node.satisfied_without_execution else
                (NodeExecutionStatus.EXECUTED_SUCCESS if validation.passed
                 else NodeExecutionStatus.EXECUTED_FAILURE))
            node.attempts.append({
                "mode": ExecutionMode.DIRECT.value,
                "started": node.attempt_started,
                "passed": validation.passed,
                "failure_type": "" if validation.passed else
                                "tool_execution_error",
                "failure_stage": "validation",
                "failure_cause": "" if validation.passed else
                                 "effect_violation",
                "tool_refs": list(node.tool_refs),
                "action_start": action_start, "action_end": after_index,
                "action_count": node.executed_action_count,
                "step_id": node.step_id,
                "occurrence_id": node.occurrence_id,
            })
            node.outputs = (materialize_atomic_outputs(
                                atomic, grounded, before, after,
                                tool_result=_execution_output_payload(result))
                            if validation.passed else {})
            trace.node_validators.append(validation)

    def _fill_env_trace(self, trace: TraceRecord, result: Any,
                        runtime_graph: RuntimeGraph) -> None:
        trace.success = bool(result.success)
        trace.failure_type = str(result.failure_type or "")
        if getattr(result, "infrastructure_errors", None):
            trace.metrics["infrastructure_errors"] = [
                dict(item) for item in result.infrastructure_errors]
        if getattr(result, "diagnostics", None):
            runtime_diagnostics = trace.metrics.setdefault(
                "runtime_diagnostics", {})
            for key, value in dict(result.diagnostics).items():
                if isinstance(value, list):
                    runtime_diagnostics.setdefault(key, []).extend(
                        copy.deepcopy(value))
                else:
                    runtime_diagnostics[key] = copy.deepcopy(value)
        trace.retries = sum(max(0, len(node.attempts) - 1)
                            for node in runtime_graph.nodes)
        trace.benchmark_result = {"passed": trace.success}
        if trace.failure_type == "llm_error":
            trace.benchmark_result["infrastructure_failure"] = True
        mode_map = {"direct": ExecutionMode.DIRECT,
                    "seeded": ExecutionMode.SEEDED,
                    "dynamic": ExecutionMode.DYNAMIC}
        for action in result.actions or []:
            trace.actions.append(ActionRecord(
                step=int(action.get("step", 0)),
                name=str(action.get("name", "")),
                params=dict(action.get("params") or {}),
                observation=str(action.get("observation", "")),
                accepted=bool(action.get("accepted", True)),
                mode=mode_map.get(str(action.get("mode", "dynamic")), ExecutionMode.DYNAMIC),
                node_ref=str(action.get("node_ref", "")),
                tool_ref=str(action.get("tool_ref", "")),
                origin=str(action.get("origin") or
                           ("tool" if action.get("tool_ref") else "agent")),
            ))
        for snapshot in result.states or []:
            trace.state_snapshots.append({"step": int(snapshot.get("step", 0)),
                                          "state": dict(snapshot.get("state") or {})})
        trace.observations = [
            {"step": i, "text": str(a.observation)} for i, a in enumerate(trace.actions)
        ]
        runtime_graph.apply_to_trace(trace)

    # ------------------------------------------------------------------
    # 事后进化（成功/失败都处理）
    # ------------------------------------------------------------------
    def _process_trace(self, trace: TraceRecord, task: Task) -> dict[str, Any]:
        # 冻结评估模式：技能库只读，不做任何进化/证据写入（Train-Evolve-Test 第二阶段）
        if self.config.freeze_skills:
            return {"frozen_eval": True}
        if trace.failure_type == "llm_error":
            # API/网络/超时失败不产生 Skill、Tool、Composite 的正负证据，也不
            # 触发 failure branch；否则基础设施质量会被误写成方法质量。
            return {"infrastructure_failure": True,
                    "learning_skipped": True,
                    "errors": list(trace.metrics.get("infrastructure_errors") or [])}
        if not bool(trace.metrics.get("learning_eligible", True)):
            return {
                "learning_skipped": True,
                "neutral_contract_mismatch": True,
                "benchmark_won": bool(trace.success),
                "runtime_contract_valid": False,
            }
        if trace.success:
            run_maintenance = (
                self.config.features.enable_tool_evolution
                and (self.success_count + 1) % max(self.config.maintenance_interval, 1) == 0
            )
            result = self.success_processor.process_success(trace,
                                                            run_maintenance=run_maintenance)
            self._update_skill_evidence(trace, success=True)
            replayed = self.proposal_replayer.consume_success(trace, result.atomic_refs)
            branches = self.failure_branch_manager.process(trace, task)
            return {"success_processing": result.to_dict(),
                    "proposal_replays": replayed,
                    "failure_branches": branches}
        failure = self.failure_processor.process_failure(trace)
        self._update_skill_evidence(trace, success=False)
        branches = self.failure_branch_manager.process(trace, task)
        return {"failure_processing": failure.to_dict(),
                "failure_branches": branches}

    def _record_tool_feedback(self, refs: list[str], success: bool, mode: str) -> None:
        """Tool 使用反馈（与 Skill 证据分开记录，§39.3）。"""
        if self.config.freeze_skills:
            return
        if self._feedback_buffer is not None:
            self._feedback_buffer.append(("tool", list(refs), bool(success), str(mode)))
            return
        from .core.refs import ToolRef
        for ref_text in refs:
            try:
                ref = ToolRef.parse(ref_text)
            except ValueError:
                continue
            self.tool_registry.record_feedback(ref, success, usage_mode=mode)

    def _record_impl_feedback(self, refs: list[str], success: bool) -> None:
        """Implementation Atom 质量统计更新（§17 quality / §39.3）。"""
        if self.config.freeze_skills:
            return
        if self._feedback_buffer is not None:
            self._feedback_buffer.append(("impl", list(refs), bool(success), ""))
            return
        from .core.refs import SkillRef
        for ref_text in refs:
            try:
                ref = SkillRef.parse(ref_text)
            except ValueError:
                continue
            impl = self.registry.get(ref)
            if impl is None or not hasattr(impl, "quality"):
                continue
            quality = dict(impl.quality or {})
            quality["use_count"] = int(quality.get("use_count", 0)) + 1
            if success:
                quality["success_count"] = int(quality.get("success_count", 0)) + 1
            else:
                quality["failure_count"] = int(quality.get("failure_count", 0)) + 1
            empirical = int(quality.get("success_count", 0)) / max(int(quality.get("use_count", 0)), 1)
            old = float(quality.get("utility", 0.5))
            quality["utility"] = round(0.5 * old + 0.5 * empirical, 4)
            impl.quality = quality
            self.registry.update_runtime_state(impl)

    def _update_skill_evidence(self, trace: TraceRecord, success: bool) -> None:
        """Skill 层证据更新（§39.3 与 Tool 分开记统计；负迁移治理 §39.2）。"""
        if self.config.freeze_skills:
            return
        if not self.config.features.enable_governance:
            return
        # Atomic utility is occurrence-local. A later node or the benchmark may
        # fail without invalidating an upstream transition whose own Effect was
        # verified. Composite utility, handled below, remains task/path-level.
        seen_task_atomics: set[str] = set()
        for realized in trace.realized_atomic_nodes:
            ref_text = str(realized.get("ref") or "")
            if not ref_text:
                continue
            try:
                from .core.refs import SkillRef
                ref = SkillRef.parse(ref_text)
            except ValueError:
                continue
            obj = self.registry.get(ref)
            if obj is None or not hasattr(obj, "metadata"):
                continue  # ImplementationAtom 不承载 Skill 层证据
            stats = dict(obj.metadata.get("statistics") or {})
            attempts = list(realized.get("attempts") or [])
            started = bool(realized.get("attempt_started")) or any(
                bool(item.get("started")) for item in attempts)
            action_count = int(realized.get("executed_action_count") or 0)
            required_slots = semantic_required_slots(
                list(getattr(obj, "effects", []) or []))
            core_bound = all(is_concrete_binding(
                dict(realized.get("params") or {}).get(slot))
                for slot in required_slots)
            if (str(realized.get("execution_status") or "")
                    == NodeExecutionStatus.ALREADY_SATISFIED.value):
                stats["selection_count"] = int(
                    stats.get("selection_count", 0)) + 1
                stats["already_satisfied_count"] = int(
                    stats.get("already_satisfied_count", 0)) + 1
                obj.metadata["statistics"] = stats
                self.registry.update_runtime_state(obj)
                continue
            if not started or not core_bound:
                stats["selection_count"] = int(
                    stats.get("selection_count", 0)) + 1
                reason = str(realized.get("fallback_reason") or "")
                if not core_bound or "binding" in reason:
                    stats["binding_failure_count"] = int(
                        stats.get("binding_failure_count", 0)) + 1
                elif "budget" in reason:
                    stats["budget_skip_count"] = int(
                        stats.get("budget_skip_count", 0)) + 1
                obj.metadata["statistics"] = stats
                self.registry.update_runtime_state(obj)
                continue
            stats["selection_count"] = int(
                stats.get("selection_count", 0)) + 1
            stats["use_count"] = int(stats.get("use_count", 0)) + 1
            stats["executed_action_count"] = int(
                stats.get("executed_action_count", 0)) + action_count
            node_passed = bool(realized.get("passed"))
            if node_passed:
                stats["execution_success_count"] = int(
                    stats.get("execution_success_count", 0)) + 1
                stats["effect_validation_success_count"] = int(
                    stats.get("effect_validation_success_count", 0)) + 1
                stats["contract_support_count"] = int(
                    stats.get("contract_support_count", 0)) + 1
            else:
                stats["execution_failure_count"] = int(
                    stats.get("execution_failure_count", 0)) + 1
                stats["effect_validation_failure_count"] = int(
                    stats.get("effect_validation_failure_count", 0)) + 1
            if ref.logical_id not in seen_task_atomics:
                seen_task_atomics.add(ref.logical_id)
                stats["task_use_count"] = int(stats.get("task_use_count", 0)) + 1
                task_key = "task_success_count" if success else "task_failure_count"
                stats[task_key] = int(stats.get(task_key, 0)) + 1
            total = int(stats.get("use_count", 0))
            empirical = int(stats.get("execution_success_count", 0)) / max(total, 1)
            old = float(stats.get("utility", 0.5))
            stats["utility"] = round(0.5 * old + 0.5 * empirical, 4)
            obj.metadata["statistics"] = stats
            failures = int(stats.get("execution_failure_count", 0))
            failure_threshold = int(
                self.config.thresholds.suppress_failure_threshold)
            if (total >= max(3, failure_threshold + 1)
                    and failures >= failure_threshold
                    and stats["utility"] < float(
                        self.config.thresholds.retirement_utility)):
                # A failed Seeded/Dynamic attempt is evidence about the current
                # executor, bindings or plan, not evidence that the verified
                # declarative Atomic contract is harmful.  Suppressing the
                # Abstract node here removes the only target producer from
                # frozen planning and creates a self-reinforcing all-Dynamic
                # fallback.  Keep the contract retrievable and record a
                # governance review; executable Tool/Implementation failures
                # are governed independently from actual Direct calls.
                reviews = list(obj.metadata.get("governance_reviews") or [])
                reviews.append({
                    "trace_id": trace.trace_id,
                    "decision": "retain_abstract_contract",
                    "reason": "execution_failure_is_not_contract_invalidation",
                    "use_count": total,
                    "execution_failure_count": failures,
                    "utility": stats["utility"],
                })
                obj.metadata["governance_reviews"] = reviews[-50:]
            self.registry.update_runtime_state(obj)

        if trace.selected_composite:
            try:
                from .core.refs import SkillRef
                ref = SkillRef.parse(trace.selected_composite)
            except ValueError:
                ref = None
            obj = self.registry.get(ref) if ref is not None else None
            if obj is not None and hasattr(obj, "metadata"):
                stats = dict(obj.metadata.get("statistics") or {})
                stats["selection_count"] = int(
                    stats.get("selection_count", 0)) + 1
                composite_validation = dict(
                    trace.validation_layers.get("selected_composite_self")
                    or trace.validation_layers.get("composite") or {})
                task_gap_missing = bool(
                    trace.task_gap_analysis is not None
                    and trace.task_gap_analysis.missing_effects)
                strong_task_gap_proof = _task_gap_is_strong_proof(trace)
                inconclusive_gap = bool(
                    task_gap_missing and not strong_task_gap_proof)
                validated = bool(composite_validation) and not inconclusive_gap
                composite_passed = bool(
                    composite_validation.get("passed")) if validated else False
                stats["validation_count"] = int(
                    stats.get("validation_count", 0)) + int(validated)
                stats["use_count"] = int(stats.get("use_count", 0)) + 1
                if validated:
                    key = ("execution_success_count" if composite_passed
                           else "execution_failure_count")
                    stats[key] = int(stats.get(key, 0)) + 1
                elif inconclusive_gap:
                    stats["unproven_task_gap_count"] = int(
                        stats.get("unproven_task_gap_count", 0)) + 1
                if int(trace.metrics.get("benchmark_finalization_count", 0)):
                    stats["benchmark_finalization_count"] = int(
                        stats.get("benchmark_finalization_count", 0)) + 1
                benchmark_key = ("benchmark_success_count" if success
                                 else "benchmark_failure_count")
                stats[benchmark_key] = int(stats.get(benchmark_key, 0)) + 1
                execution_total = (
                    int(stats.get("execution_success_count", 0))
                    + int(stats.get("execution_failure_count", 0)))
                if validated and execution_total:
                    empirical = int(stats.get(
                        "execution_success_count", 0)) / execution_total
                    stats["utility"] = round(
                        0.5 * float(stats.get("utility", 0.5))
                        + 0.5 * empirical, 4)
                obj.metadata["statistics"] = stats
                failure_codes = {
                    str(value) for value in
                    (composite_validation.get("failure_codes") or [])}
                self.registry.update_runtime_state(obj)
                if validated:
                    apply_self_sufficient_evidence(
                        obj, self.registry, passed=composite_passed,
                        failure_codes=failure_codes,
                        task_gap_proved_missing_effect=strong_task_gap_proof,
                    )

    # ------------------------------------------------------------------
    def _get_recommended(self, logical_id: str):
        obj = self.registry.get_recommended(logical_id)
        return obj

    def stats(self) -> dict[str, Any]:
        return {
            "episodes": self.episode_count,
            "successes": self.success_count,
            "skill_graph": self.registry.stats(),
            "tool_repo": self.tool_registry.stats(),
        }


def _build_planning_audit(task: Task, plan: Any,
                          registry: SkillGraphRegistry) -> dict[str, Any]:
    """Persist why a target was reusable, excluded, or made Dynamic."""
    aliases = {
        "object.in_receptacle": "object.at_location",
        "object.in_container": "object.at_location",
    }

    def predicates(items: list[dict[str, Any]]) -> list[str]:
        return sorted({
            aliases.get(str(item.get("predicate") or ""),
                        str(item.get("predicate") or ""))
            for item in items if isinstance(item, dict) and item.get("predicate")
        })

    targets = predicates(list(task.target_effects or []))
    producers: list[dict[str, Any]] = []
    for obj in registry.list_by_kind(SkillNodeKind.ABSTRACT_ATOMIC):
        effects = predicates(list(getattr(obj, "effects", []) or []))
        overlap = sorted(set(targets) & set(effects))
        if not overlap:
            continue
        producers.append({
            "ref": str(obj.ref),
            "status": obj.status.value,
            "effects": effects,
            "target_overlap": overlap,
            "support_count": int(
                (obj.metadata.get("statistics") or {}).get("support_count", 0)),
        })

    nodes: list[dict[str, Any]] = []
    for planned in plan.nodes:
        atomic = (None if planned.dynamic else
                  registry.get(planned.ref)
                  or registry.get_recommended(planned.ref.logical_id))
        nodes.append({
            "step_id": planned.step_id,
            "ref": str(planned.ref),
            "source": planned.source,
            "dynamic": bool(planned.dynamic),
            "params": dict(planned.params),
            "target_effects": list(planned.target_effects or []),
            "atomic_status": (atomic.status.value if atomic is not None else ""),
            "inputs": list(getattr(atomic, "inputs", []) or []),
            "preconditions": list(getattr(atomic, "preconditions", []) or []),
            "effects": list(getattr(atomic, "effects", []) or []),
        })
    return {
        "target_effects": list(task.target_effects or []),
        "target_predicates": targets,
        "start_mode": plan.start_mode,
        "notes": list(plan.notes or []),
        "selected_composite": str(plan.composite_ref or ""),
        "planned_nodes": nodes,
        "registered_target_producers_all_statuses": producers,
        "candidate_rejections": list((getattr(plan, "audit", {}) or {}).get(
            "candidate_rejections") or []),
        "selected_plan": str((getattr(plan, "audit", {}) or {}).get(
            "selected_plan") or plan.composite_ref or "atomic_or_dynamic"),
        "fallback_reason": str((getattr(plan, "audit", {}) or {}).get(
            "fallback_reason") or ""),
    }


def analyze_task_gap(task: Task, state: dict[str, Any],
                     bindings: dict[str, Any]) -> TaskGapAnalysis:
    """Compute the formal target delta at the exact planned-graph boundary."""
    from .core.predicates import StateSnapshot, check_effects

    missing: list[dict[str, Any]] = []
    snapshot = StateSnapshot(state or {})
    for raw in task.target_effects or []:
        if not isinstance(raw, dict) or not raw.get("predicate"):
            continue
        effect = copy.deepcopy(raw)
        passed, _messages = check_effects(
            snapshot, bindings, [effect], {"harness": "env"})
        if not passed:
            missing.append(effect)
    targets_satisfied = not missing
    reasons = (["formal_target_effects_missing_at_pre_gap_boundary"]
               if missing else
               ["formal_target_effects_satisfied_but_benchmark_not_terminal"])
    return TaskGapAnalysis(
        missing_effects=missing,
        targets_already_satisfied=targets_satisfied,
        benchmark_only_finalization=targets_satisfied,
        reasons=reasons,
    )


def _runtime_learning_eligible(
        trace: TraceRecord, task: Task | None = None) -> bool:
    """Validate runtime evidence independently from benchmark-authoritative won."""
    realized = list(trace.realized_atomic_nodes or [])
    selected = [
        node for node in realized
        if (bool(node.get("attempt_started"))
            or str(node.get("execution_status") or "")
            == NodeExecutionStatus.ALREADY_SATISFIED.value)
    ]
    selected_valid = all(bool(node.get("passed")) for node in selected)
    validators = [
        item for item in trace.node_validators
        if item.level in {"atomic", "task_gap"}
    ]
    validators_valid = all(bool(item.passed) for item in validators)
    gap_required = bool(
        trace.task_gap_analysis is not None
        and trace.task_gap_analysis.missing_effects)
    gap_validators = [item for item in validators if item.level == "task_gap"]
    gap_valid = (not gap_required) or (
        bool(gap_validators) and all(item.passed for item in gap_validators))
    # Legacy/non-resume whole-plan Seeded execution cannot be credited when it
    # produced actions but no formal per-node boundary/validator at all.
    segmented_evidence = not (
        realized and trace.actions and not validators)
    final_target_valid = True
    if task is not None and _is_env_task(task) and task.target_effects:
        final_target_valid = _final_target_contract_valid(trace, task)
    return bool(selected_valid and validators_valid and gap_valid
                and segmented_evidence and final_target_valid)


def _final_target_contract_valid(trace: TraceRecord, task: Task) -> bool:
    """Recheck formal targets after benchmark finalization/protocol actions."""
    from .core.predicates import StateSnapshot, check_effects

    bindings = _realized_task_bindings(
        dict(task.context.get("params") or {}),
        list(trace.realized_atomic_nodes or []))
    snapshot = StateSnapshot(trace.final_state())
    certificates = list(
        (trace.metrics.get("runtime_diagnostics") or {}).get(
            "terminal_verified_effects") or [])
    for target in task.target_effects or []:
        if not isinstance(target, dict):
            return False
        passed, _missing = check_effects(
            snapshot, bindings, [target], {"harness": "env"})
        if passed:
            continue
        cardinality = max(1, int(target.get("cardinality", 1) or 1))
        if cardinality != 1:
            return False
        single = {**target, "cardinality": 1, "distinct_by": ""}
        certified = any(
            isinstance(item, dict)
            and bool(item.get("benchmark_won"))
            and str(item.get("source") or "")
            == "benchmark_terminal_certificate_v1"
            and isinstance(item.get("effect"), dict)
            and _terminal_certificate_matches_target(
                dict(item["effect"]), single, bindings)
            for item in certificates)
        if not certified:
            return False
    return True


def _terminal_certificate_matches_target(
        certificate_effect: dict[str, Any], target_effect: dict[str, Any],
        bindings: dict[str, Any]) -> bool:
    """Match a concrete terminal certificate to a bound target contract.

    Task goals commonly retain a class binding (``bowl``), while terminal
    evidence necessarily names the concrete instance (``bowl_2``). Matching is
    directional: a class may accept one of its instances, but a concrete task
    identity never accepts a different instance.
    """
    from .core.predicates import (
        ordered_predicate_args, predicate_fact_signature)

    certificate_name = str(certificate_effect.get("predicate") or "")
    target_name = str(target_effect.get("predicate") or "")
    certificate_fact, _certificate_schema = predicate_fact_signature(
        certificate_name)
    target_fact, _target_schema = predicate_fact_signature(target_name)
    if not certificate_fact or certificate_fact != target_fact:
        return False

    certificate_args = ordered_predicate_args(
        certificate_name, dict(certificate_effect.get("args") or {}))
    target_args = ordered_predicate_args(
        target_name, dict(target_effect.get("args") or {}))
    if len(certificate_args) != len(target_args):
        return False
    for (_certificate_arg, actual), (_target_arg, raw_expected) in zip(
            certificate_args, target_args):
        slot = binding_slot_name(raw_expected)
        expected = bindings.get(slot, raw_expected) if slot else raw_expected
        if (not is_concrete_binding(actual)
                or not is_concrete_binding(expected)
                or not _runtime_values_compatible(expected, actual)):
            return False
    return True


def _task_gap_is_strong_proof(trace: TraceRecord) -> bool:
    """Require executed, code-validated gap evidence before suppression."""
    analysis = trace.task_gap_analysis
    if (not trace.success or not trace.selected_composite
            or analysis is None or not analysis.missing_effects):
        return False
    proof = dict((trace.provenance or {}).get("task_gap_effect_proof") or {})
    if (not bool(proof.get("passed"))
            or not list(proof.get("action_caused_effects") or [])):
        return False
    spans = [span for span in trace.runtime_spans if span.kind == "task_gap"]
    if not spans or not any(span.action_end > span.action_start for span in spans):
        return False
    gap_ids = {span.occurrence_id for span in spans}
    proved_occurrences = {
        str(value) for value in (proof.get("inserted_occurrence_ids") or [])
        if str(value)}
    if not proved_occurrences or not proved_occurrences.issubset(gap_ids):
        return False
    planned = [
        node for node in trace.realized_atomic_nodes
        if str(node.get("occurrence_id") or "") not in gap_ids
        and not str(node.get("ref") or "").endswith(
            "runtime.dynamic.task_gap@0.0.0")
    ]
    if not planned or not all(bool(node.get("passed")) for node in planned):
        return False
    return any(
        item.level == "task_gap" and item.passed
        and item.occurrence_id in gap_ids
        for item in trace.node_validators
    )


def _mark_result_action_origin(result: Any, start: int, end: int, *,
                               origin: str, node_ref: str) -> None:
    actions = list(getattr(result, "actions", None) or [])
    for index in range(max(0, start), min(max(start, end), len(actions))):
        actions[index]["origin"] = origin
        actions[index]["node_ref"] = node_ref
        if origin != "tool":
            actions[index]["tool_ref"] = ""
    result.actions = actions


def _runtime_node_metrics(runtime_graph: RuntimeGraph) -> dict[str, Any]:
    """Derive execution/reuse metrics from realized occurrence state.

    Selection, execution and successful reuse are deliberately separate.  A
    planning/budget record with ``started=false`` and an Implementation chosen
    by the router are not evidence that the Atomic executed.
    """
    executed_statuses = {
        NodeExecutionStatus.EXECUTED_SUCCESS,
        NodeExecutionStatus.EXECUTED_FAILURE,
    }
    node_mode_counts: dict[str, int] = {}
    executed_node_count = 0
    already_satisfied_node_count = 0
    selected_refs: list[str] = []
    executed_refs: list[str] = []
    successful_refs: list[str] = []
    successful_tool_refs: list[str] = []
    successful_occurrences = 0

    def append_unique(target: list[str], value: str) -> None:
        if value and value not in target:
            target.append(value)

    for planned, node in zip(runtime_graph.plan.nodes, runtime_graph.nodes):
        reusable = not bool(planned.dynamic)
        if reusable:
            append_unique(selected_refs, str(planned.ref))
        if node.execution_status == NodeExecutionStatus.ALREADY_SATISFIED:
            already_satisfied_node_count += 1
            continue
        executed = (
            node.execution_status in executed_statuses
            and int(node.executed_action_count or 0) > 0
        )
        if not executed:
            continue
        executed_node_count += 1
        started_attempts = [
            item for item in node.attempts if bool(item.get("started"))]
        for attempt in started_attempts:
            if (str(attempt.get("mode") or "") == ExecutionMode.DIRECT.value
                    and bool(attempt.get("passed"))):
                for tool_ref in attempt.get("tool_refs") or []:
                    append_unique(successful_tool_refs, str(tool_ref))
        mode = (str(started_attempts[-1].get("mode") or "")
                if started_attempts else node.mode.value)
        if mode not in {item.value for item in ExecutionMode}:
            mode = node.mode.value
        node_mode_counts[mode] = node_mode_counts.get(mode, 0) + 1
        if reusable:
            append_unique(executed_refs, str(planned.ref))
        if (reusable
                and node.execution_status == NodeExecutionStatus.EXECUTED_SUCCESS):
            successful_occurrences += 1
            append_unique(successful_refs, str(planned.ref))

    planned_count = len(runtime_graph.nodes)
    completed_node_count = min(
        planned_count, executed_node_count + already_satisfied_node_count)
    return {
        "node_mode_counts": node_mode_counts,
        "executed_node_count": executed_node_count,
        "already_satisfied_node_count": already_satisfied_node_count,
        "completed_node_count": completed_node_count,
        "not_started_node_count": max(0, planned_count - completed_node_count),
        "selected_skill_refs": selected_refs,
        "executed_skill_refs": executed_refs,
        "successful_reused_skill_refs": successful_refs,
        "successful_atomic_reuse_count": successful_occurrences,
        "successful_tool_refs": successful_tool_refs,
    }


def _append_planned_runtime_spans(trace: TraceRecord,
                                  runtime_graph: RuntimeGraph) -> None:
    """Persist final occurrence boundaries without merging task-gap spans."""
    existing = {(span.kind, span.occurrence_id) for span in trace.runtime_spans}
    for planned, node in zip(runtime_graph.plan.nodes, runtime_graph.nodes):
        if planned.source == "task_gap":
            continue
        attempts = [dict(item) for item in node.attempts
                    if bool(item.get("started"))]
        if not attempts:
            continue
        action_start = min(int(item.get("action_start", 0)) for item in attempts)
        action_end = max(int(item.get("action_end", 0)) for item in attempts)
        attempt_spans = [{
            "mode": str(item.get("mode") or ""),
            "action_start": int(item.get("action_start", 0)),
            "action_end": int(item.get("action_end", 0)),
            "action_count": int(item.get("action_count", 0)),
            "passed": bool(item.get("passed")),
            "failure_type": str(item.get("failure_type") or ""),
            "failure_stage": str(item.get("failure_stage") or ""),
            "failure_cause": str(item.get("failure_cause") or ""),
        } for item in attempts]
        key = ("planned_node", node.occurrence_id)
        if key in existing:
            continue
        trace.runtime_spans.append(RuntimeSpan(
            kind="planned_node", occurrence_id=node.occurrence_id,
            action_start=action_start,
            action_end=action_end,
            node_ref=node.ref, learnable=True,
            metadata={"step_id": node.step_id,
                      "execution_status": node.execution_status.value,
                      "attempt_spans": attempt_spans},
        ))
    trace.runtime_spans.sort(
        key=lambda span: (span.action_start, span.action_end, span.kind))


def _can_mark_already_satisfied(
        task: Task, planned: Any, effects: list[dict[str, Any]],
        state: dict[str, Any], runtime_graph: RuntimeGraph, index: int,
        distinct_exclusions: dict[str, set[Any]] | None = None) -> bool:
    from .core.predicates import StateSnapshot, check_effects

    required = semantic_required_slots(effects)
    if any(not is_concrete_binding(planned.params.get(slot))
           for slot in required):
        return False
    passed, _missing = check_effects(
        StateSnapshot(state or {}), planned.params, effects,
        {"harness": "env"})
    if not passed:
        return False

    # A class-valued binding can existentially match an already consumed
    # witness.  It is not evidence that this occurrence owns a fresh instance.
    if _reused_distinct_bindings(
            planned.params, distinct_exclusions or {}):
        return False

    max_cardinality = max(
        [max(1, int(effect.get("cardinality", 1) or 1))
         for effect in task.target_effects if isinstance(effect, dict)] or [1])
    if max_cardinality <= 1:
        return True

    # In a repeated branch, a prior occurrence with the same contract and the
    # same semantic binding consumes that witness.  This is role/cardinality
    # based and contains no benchmark operation or entity vocabulary.
    current_values = tuple(sorted(
        (slot, str(planned.params.get(slot))) for slot in required))
    current_predicates = tuple(sorted(
        str(effect.get("predicate") or "") for effect in effects))
    for prior_index in range(index):
        prior_plan = runtime_graph.plan.nodes[prior_index]
        prior_state = runtime_graph.nodes[prior_index]
        prior_predicates = tuple(sorted(
            str(effect.get("predicate") or "")
            for effect in prior_plan.target_effects))
        prior_values = tuple(sorted(
            (slot, str(prior_plan.params.get(slot))) for slot in required))
        if (prior_state.passed and prior_predicates == current_predicates
                and prior_values == current_values):
            return False
    return True


def _env_resume_payload(result: Any) -> dict[str, Any]:
    """从 EnvRunResult 提取原地续跑载荷（observation/admissible/actions/states）。

    用于 direct→seeded、seeded→dynamic 的原地降级：下一阶段从失败点的
    当前状态继续同一 episode，不重开环境。
    """
    states = [dict(s) for s in (getattr(result, "states", None) or [])]
    return {
        "observation": str(getattr(result, "current_observation", "")
                           or getattr(result, "final_observation", "")),
        "admissible": list(getattr(result, "current_admissible", []) or []),
        "actions": [dict(a) for a in (getattr(result, "actions", None) or [])],
        "states": states,
        "state": (states[-1].get("state") or {}) if states else {},
    }


def _execution_output_payload(result: Any) -> Any:
    """Expose a Direct executor's structured return to output materializers."""
    if isinstance(result, dict):
        return (result.get("tool_result") or result.get("outputs") or
                result.get("result") or result)
    diagnostics = getattr(result, "diagnostics", None)
    if isinstance(diagnostics, dict):
        return (diagnostics.get("tool_result") or
                diagnostics.get("direct_tool_result") or
                diagnostics.get("outputs") or diagnostics)
    return result


def _not_started_env_result(resume: dict[str, Any] | None,
                            failure_type: str) -> EnvRunResult:
    payload = dict(resume or {})
    return EnvRunResult(
        actions=[dict(item) for item in (payload.get("actions") or [])],
        states=[dict(item) for item in (payload.get("states") or [])],
        steps=len(payload.get("actions") or []),
        failure_type=failure_type,
        current_observation=str(payload.get("observation") or ""),
        current_admissible=list(payload.get("admissible") or []),
        final_observation=str(payload.get("observation") or ""),
    )


def _is_env_task(task: Task) -> bool:
    kind = str(task.context.get("kind", ""))
    if kind == "env":
        return True
    return task.benchmark in ("alfworld", "toy_env")


def _entry_of(trace: TraceRecord) -> str:
    for node in trace.planned_atomic_nodes:
        params = node.get("params") or {}
        entry = params.get("entry_point")
        if entry:
            return str(entry)
    return "solve"


def _guidelines_of(plan, registry: SkillGraphRegistry) -> str:
    parts: list[str] = []
    for node in plan.nodes:
        atomic = registry.get(node.ref)
        if atomic is None:
            continue
        parts.append(f"[Atomic Skill] {atomic.summary}")
        for rule in atomic.guideline_rules():
            parts.append(f"  - {rule}")
    return "\n".join(parts)


def _guideline_of(atomic: Any) -> str:
    if atomic is None:
        return ""
    parts = [f"[Atomic Skill] {getattr(atomic, 'summary', '')}"]
    parts.extend(f"  - {rule}" for rule in atomic.guideline_rules())
    effects = getattr(atomic, "effects", []) or []
    if effects:
        parts.append(f"  Required effects: {effects}")
    return "\n".join(parts)


def _phase_goal_of(atomic: Any, effects: list[dict[str, Any]],
                   params: dict[str, Any],
                   excluded_bindings: dict[str, set[Any]] | None = None) -> str:
    """构造只覆盖当前 Atomic Effect 的阶段目标，防止一个 Seeded 节点包办整题。"""
    summary = str(getattr(atomic, "summary", "") or "complete the current atomic step")
    bound: list[str] = []
    from .core.predicates import bind_args
    for effect in effects:
        args = bind_args(dict(effect.get("args") or {}), params, {})
        detail = ", ".join(f"{key}={value}" for key, value in args.items())
        bound.append(f"{effect.get('predicate')}({detail})")
    target = "; ".join(bound) or summary
    exclusions = {
        str(role): sorted(str(value) for value in values)
        for role, values in (excluded_bindings or {}).items() if values
    }
    distinct_instruction = ""
    if exclusions:
        rendered = "; ".join(
            f"{role} excludes {', '.join(values)}"
            for role, values in exclusions.items())
        distinct_instruction = (
            " Cardinality constraint: select a DIFFERENT concrete instance; "
            f"do not take, move, or reuse already claimed instances ({rendered})."
        )
    return (f"Complete ONLY this atomic step: {summary}. Required state: {target}."
            f"{distinct_instruction} Do not continue to later task stages "
            "after this state is reached.")


def _run_env_episode_with_optional_exclusions(
        adapter: Any, task: Task, llm: Any, **kwargs: Any) -> Any:
    """Pass the new exclusion contract without breaking legacy adapters."""
    import inspect

    method = adapter.run_env_episode
    try:
        parameters = inspect.signature(method).parameters.values()
        supports = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            or parameter.name == "excluded_effect_bindings"
            for parameter in parameters)
    except (TypeError, ValueError):
        supports = False
    if not supports:
        kwargs.pop("excluded_effect_bindings", None)
    return method(task, llm, **kwargs)


def _refine_env_object_binding(
        params: dict[str, Any], state: dict[str, Any],
        excluded_bindings: dict[str, set[Any]] | None = None
        ) -> dict[str, Any]:
    """Carry a discovered environment entity instance into downstream nodes.

    Planning correctly starts from a class-level goal slot (for example
    an entity family). Once execution identifies the exact instance, the tracker
    carries it through downstream data flow. Refining only a class-valued slot
    keeps the choice stable and never replaces an already concrete binding.
    """
    import re
    from .core.predicates import normalize_value

    refined = dict(params or {})
    wanted = normalize_value(refined.get("object", ""))
    if not wanted or re.search(r"_\d+$", wanted):
        return refined
    family = re.sub(r"_\d+$", "", wanted)
    excluded = set((excluded_bindings or {}).get("object") or set())
    matches = []
    for item in state.get("inventory", []) or []:
        actual = normalize_value(item)
        if (re.sub(r"_\d+$", "", actual) == family
                and not any(distinct_values_conflict(actual, claimed)
                            for claimed in excluded)):
            matches.append(actual)
    if len(matches) == 1:
        refined["object"] = matches[0]
    return refined


def _source_location_entity_role(
        atomic: Any, location_slot: str,
        params: dict[str, Any]) -> str:
    """Return the entity input anchored by a learned source-location slot.

    The relation is inferred from the Atomic contract rather than an operation
    or task label.  A conventional ``object_location`` maps directly to
    ``object``.  A generic ``source_location`` is supported when a precondition
    relates it to exactly one other input (for example ``object.at_location``).
    Destination-only slots that occur solely in Effects are intentionally not
    treated as sources to discover.
    """
    slot = str(location_slot or "")
    if not slot.endswith("_location"):
        return ""
    declared = {
        str(item.get("name")) for item in (getattr(atomic, "inputs", []) or [])
        if isinstance(item, dict) and item.get("name")
    }
    prefix = slot[:-len("_location")]
    if prefix in declared and prefix in params:
        return prefix

    candidates: set[str] = set()
    for condition in (getattr(atomic, "preconditions", []) or []):
        if not isinstance(condition, dict):
            continue
        relation_slots: set[str] = set()
        _collect_input_slots(dict(condition.get("args") or {}), relation_slots)
        if slot not in relation_slots:
            continue
        candidates.update(
            role for role in relation_slots
            if role != slot and role in declared and not role.endswith("_location"))
    concrete = sorted(
        role for role in candidates
        if is_concrete_binding(dict(params or {}).get(role)))
    if len(concrete) == 1:
        return concrete[0]
    return ""


def _bind_known_location_slots(
        params: dict[str, Any], atomic: Any, state: dict[str, Any],
        excluded_bindings: dict[str, set[Any]] | None = None
        ) -> dict[str, Any]:
    """Fill learned ``<entity_role>_location`` slots from state evidence."""
    import re
    from .core.predicates import normalize_value

    refined = dict(params or {})
    slots = {
        str(item.get("name")) for item in (getattr(atomic, "inputs", []) or [])
        if isinstance(item, dict)
        and str(item.get("name") or "").endswith("_location")
    }
    facts = list((state or {}).get("facts") or [])
    excluded_by_role = {
        str(role): {normalize_value(value) for value in values
                    if value not in (None, "")}
        for role, values in (excluded_bindings or {}).items()
    }
    for slot in slots:
        if is_concrete_binding(refined.get(slot)):
            continue
        entity_role = (_source_location_entity_role(
            atomic, slot, refined) or slot[:-len("_location")])
        wanted = normalize_value(refined.get(entity_role, ""))
        if not wanted:
            continue
        excluded = excluded_by_role.get(entity_role, set())
        candidates: list[tuple[str, str]] = []
        for fact in facts:
            match = re.fullmatch(r"object_at\((.+?),\s*(.+?)\)", str(fact))
            if not match:
                continue
            actual, location = normalize_value(match.group(1)), normalize_value(match.group(2))
            if (not any(distinct_values_conflict(actual, claimed)
                        for claimed in excluded)
                    and _runtime_values_compatible(wanted, actual)):
                candidates.append((actual, location))
        if len(set(candidates)) != 1:
            continue
        actual, location = candidates[0]
        if _runtime_binding_can_refine(refined.get(entity_role), actual):
            refined[entity_role] = actual
        refined[slot] = location
    return refined


def _remap_location_binding(binding: dict[str, Any], entity_role: str,
                            location_role: str) -> dict[str, Any]:
    """Map the adapter's neutral entity/location evidence to learned roles."""
    if not binding:
        return {}
    entity = binding.get("object")
    location = binding.get("object_location")
    if entity in (None, "") or location in (None, ""):
        return {}
    return {str(entity_role): entity, str(location_role): location}


def _apply_runtime_data_bindings(params: dict[str, Any], step_id: str,
                                 runtime_graph: RuntimeGraph,
                                 shared: dict[str, Any],
                                 excluded_bindings: dict[str, set[Any]] | None = None
                                 ) -> dict[str, Any]:
    """Ground a node from verified upstream occurrence data.

    Explicit DATA_FLOW mappings take precedence.  Shared same-role bindings are
    a fallback for an ordinary single-entity task and never replace a concrete
    instance with another instance.
    """
    refined = dict(params or {})
    by_step = {node.step_id: node for node in runtime_graph.nodes}
    for edge in runtime_graph.edges:
        if edge.target_step != step_id or edge.type != EdgeType.DATA_FLOW:
            continue
        source = by_step.get(edge.source_step)
        if source is None or not source.passed:
            continue
        source_role = str((edge.mapping or {}).get("source_output") or "")
        target_role = str((edge.mapping or {}).get("target_input") or "")
        value = source.outputs.get(source_role)
        if _binding_value_is_excluded(
                target_role, value, excluded_bindings or {}):
            continue
        if target_role and value not in (None, "") and (
                refined.get(target_role) in (None, "")
                or _runtime_binding_can_refine(refined.get(target_role), value)):
            refined[target_role] = value
    for role, value in shared.items():
        if role not in refined or value in (None, ""):
            continue
        if _binding_value_is_excluded(
                role, value, excluded_bindings or {}):
            continue
        if _runtime_binding_can_refine(refined.get(role), value):
            refined[role] = value
    return refined


def _runtime_distinct_exclusions(
        task: Task, planned: Any, runtime_graph: RuntimeGraph, index: int,
        state: dict[str, Any],
        assigned_bindings: dict[str, Any] | None = None,
        assigned_groups: dict[str, dict[str, Any]] | None = None
        ) -> dict[str, set[Any]]:
    """Combine validated cross-branch claims with visible completed witnesses."""
    exclusions = {
        str(role): set(values)
        for role, values in runtime_graph.distinct_exclusions(index).items()
    }
    constrained_roles = set(
        dict(getattr(planned, "distinct_bindings", {}) or {}))
    if constrained_roles:
        for role, groups in dict(
                getattr(planned, "distinct_bindings", {}) or {}).items():
            for group in groups:
                contract = _distinct_group_target_effect(task, str(group))
                if contract is None:
                    continue
                completed = _completed_distinct_effect_instances(
                    task, state,
                    dict(getattr(planned, "params", {}) or {}),
                    contract=contract)
                group_assignment = dict(assigned_groups or {}).get(str(group), {})
                assigned = (group_assignment.get("witness")
                            if str(group_assignment.get("role") or "") == str(role)
                            else dict(assigned_bindings or {}).get(str(role)))
                if assigned not in (None, ""):
                    completed = {
                        witness for witness in completed
                        if str(witness) != str(assigned)
                    }
                exclusions.setdefault(str(role), set()).update(completed)
    return {role: values for role, values in exclusions.items() if values}


def _runtime_distinct_allocations(
        task: Task, planned: Any, runtime_graph: RuntimeGraph, index: int,
        state: dict[str, Any]
        ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Allocate existing concrete goal witnesses to early occurrences.

    For a cardinality ``N`` target with ``K`` already-satisfied witnesses, the
    first ``K`` stable occurrence branches own those witnesses. Later branches
    exclude all of them and search only for the remaining instances. Task Gap
    is always a remaining-work occurrence and therefore never receives an
    already-satisfied allocation.
    """
    if str(getattr(planned, "source", "") or "") == "task_gap":
        return {}, {}
    group_assignments: dict[str, dict[str, Any]] = {}
    for role, groups in dict(
            getattr(planned, "distinct_bindings", {}) or {}).items():
        for group in groups:
            contract = _distinct_group_target_effect(task, str(group))
            if contract is None:
                continue
            witnesses = sorted(_completed_distinct_effect_instances(
                task, state, dict(getattr(planned, "params", {}) or {}),
                contract=contract))
            ordinal = runtime_graph.distinct_occurrence_ordinal(
                index, str(group))
            if ordinal < 0 or ordinal >= len(witnesses):
                continue
            witness = witnesses[ordinal]
            current = dict(getattr(planned, "params", {}) or {}).get(str(role))
            if (current not in (None, "")
                    and not str(current).startswith("$")
                    and not distinct_values_conflict(current, witness)):
                continue
            group_assignments[str(group)] = {
                "role": str(role),
                "witness": witness,
                "occurrence_ordinal": ordinal,
                "target_effect": copy.deepcopy(contract),
            }

    assignments: dict[str, Any] = {}
    constrained = dict(getattr(planned, "distinct_bindings", {}) or {})
    for role, groups in constrained.items():
        records = [group_assignments.get(str(group)) for group in groups]
        if not records or any(not isinstance(record, dict)
                              for record in records):
            continue
        witnesses = {str(record.get("witness") or "") for record in records}
        if len(witnesses) == 1 and "" not in witnesses:
            assignments[str(role)] = next(iter(witnesses))
    return assignments, group_assignments


def _cardinality_allocation_covers_node(
        planned: Any, assignments: dict[str, Any],
        group_assignments: dict[str, dict[str, Any]]) -> bool:
    constrained = dict(getattr(planned, "distinct_bindings", {}) or {})
    if not constrained:
        return False
    for role, groups in constrained.items():
        if str(role) not in assignments:
            return False
        for group in groups:
            record = group_assignments.get(str(group))
            if (not isinstance(record, dict)
                    or str(record.get("role") or "") != str(role)
                    or str(record.get("witness") or "")
                    != str(assignments[str(role)])):
                return False
    return True


def _constrain_task_gap_params(
        params: dict[str, Any],
        excluded_bindings: dict[str, set[Any]]) -> dict[str, Any]:
    """Make Task Gap exclusions explicit without pinning it to an old instance."""
    import re
    from .core.predicates import normalize_value

    constrained = dict(params or {})
    metadata: dict[str, list[str]] = {}
    for role, raw_values in (excluded_bindings or {}).items():
        values = sorted({normalize_value(item) for item in raw_values if item})
        if not values:
            continue
        metadata[str(role)] = values
        current = constrained.get(str(role))
        normalized = normalize_value(current) if current not in (None, "") else ""
        if (normalized and re.search(r"_\d+$", normalized)
                and any(distinct_values_conflict(normalized, value)
                        for value in values)):
            # Keep the semantic family available to the agent, but never pass
            # the claimed concrete identity back as the requested input.
            constrained[str(role)] = re.sub(r"_\d+$", "", normalized)
    if metadata:
        constrained["__distinct_exclusions__"] = metadata
    return constrained


def _validate_task_gap_distinct_progress(
        task: Task, effects: list[dict[str, Any]],
        before: dict[str, Any], after: dict[str, Any],
        params: dict[str, Any],
        excluded_bindings: dict[str, set[Any]]) -> tuple[bool, list[dict[str, Any]]]:
    """Require new non-reserved witnesses for every cardinality Gap target."""
    details: list[dict[str, Any]] = []
    passed = True
    for effect in effects or []:
        if (not isinstance(effect, dict)
                or int(effect.get("cardinality", 1) or 1) <= 1):
            continue
        distinct_arg = str(effect.get("distinct_by") or "")
        if not distinct_arg:
            continue
        args = dict(effect.get("args") or {})
        role = binding_slot_name(args.get(distinct_arg)) or distinct_arg
        before_witnesses = _completed_distinct_effect_instances(
            task, before, params, contract=effect)
        after_witnesses = _completed_distinct_effect_instances(
            task, after, params, contract=effect)
        excluded = set(excluded_bindings.get(role) or set())
        required_new = max(
            1, int(effect.get("cardinality", 1) or 1)
            - len(before_witnesses))
        exact_new = {
            witness for witness in after_witnesses
            if witness not in before_witnesses
        }
        accepted_new = {
            witness for witness in exact_new
            if not any(distinct_values_conflict(witness, claimed)
                       for claimed in excluded)
        }
        current_passed = len(accepted_new) >= required_new
        passed &= current_passed
        details.append({
            "predicate": str(effect.get("predicate") or ""),
            "distinct_by": distinct_arg,
            "role": role,
            "required_new_count": required_new,
            "before_witnesses": sorted(before_witnesses),
            "after_witnesses": sorted(after_witnesses),
            "excluded_witnesses": sorted(str(item) for item in excluded),
            "new_witnesses": sorted(exact_new),
            "accepted_new_witnesses": sorted(accepted_new),
            "passed": current_passed,
        })
    return passed, details


def _materialize_task_gap_distinct_witness_outputs(
        details: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Materialize validated Gap witnesses without inventing a scalar output.

    One Task Gap occurrence can satisfy multiple cardinality witnesses.  A
    role therefore maps to a stable, de-duplicated list; callers that require a
    scalar must continue using the ordinary ``outputs`` channel and an
    explicit scalar materializer.
    """
    from .core.predicates import normalize_value

    collected: dict[str, set[str]] = {}
    for detail in details or []:
        if not isinstance(detail, dict) or not detail.get("passed"):
            continue
        role = str(detail.get("role") or "")
        if not role:
            continue
        values = {
            normalize_value(value)
            for value in (detail.get("accepted_new_witnesses") or [])
            if value not in (None, "")
        }
        if values:
            collected.setdefault(role, set()).update(values)
    return {role: sorted(values) for role, values in sorted(collected.items())}


def _binding_value_is_excluded(
        role: str, value: Any,
        excluded_bindings: dict[str, set[Any]]) -> bool:
    if value in (None, ""):
        return False
    return any(
        distinct_values_conflict(value, item)
        for item in excluded_bindings.get(str(role), set())
        if item not in (None, ""))


def _reused_distinct_bindings(
        params: dict[str, Any],
        excluded_bindings: dict[str, set[Any]]) -> dict[str, Any]:
    """Return roles that are exact or class-level aliases of claimed values."""
    reused: dict[str, Any] = {}
    for role, excluded_values in excluded_bindings.items():
        value = dict(params or {}).get(role)
        if value in (None, "") or str(value).startswith("$"):
            continue
        if any(distinct_values_conflict(value, excluded)
               for excluded in excluded_values):
            reused[str(role)] = value
    return reused


def _runtime_resolvable_semantic_slots(
        atomic: Any, effects: list[dict[str, Any]],
        params: dict[str, Any]) -> set[str]:
    """Return unresolved semantic slots with an explicit runtime anchor.

    This mirrors the plan-validator contract at execution time.  Merely adding
    ``runtime_resolvable`` is insufficient: every declared anchor role must be
    concrete, otherwise a node-level agent would receive an ungrounded local
    goal and the whole task must be recompiled/fail closed instead.
    """
    if atomic is None:
        return set()
    requirements = slot_requirements_for(atomic, effects)
    result: set[str] = set()
    for name, requirement in requirements.items():
        if (not requirement.semantic_required
                or not requirement.runtime_resolvable
                or is_concrete_binding(params.get(name))):
            continue
        if (requirement.anchor_roles and all(
                is_concrete_binding(params.get(role))
                for role in requirement.anchor_roles)):
            result.add(name)
    return result


def _runtime_binding_provenance(
        task: Task, planned: Any, runtime_graph: RuntimeGraph,
        state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Persist the concrete source used by each current occurrence input."""
    params = dict(getattr(planned, "params", {}) or {})
    specs = dict(getattr(planned, "binding_specs", {}) or {})
    task_params = dict(task.context.get("params") or {})
    by_step = {item.step_id: item for item in runtime_graph.nodes}
    result: dict[str, dict[str, Any]] = {}
    for role, value in params.items():
        if not is_concrete_binding(value):
            continue
        spec = specs.get(role)
        if spec is not None and not hasattr(spec, "kind"):
            from .core.binding_ir import BindingSpec
            spec = BindingSpec.from_value(spec)
        incoming = next((
            edge for edge in runtime_graph.edges
            if edge.type == EdgeType.DATA_FLOW
            and edge.target_step == getattr(planned, "step_id", "")
            and str((edge.mapping or {}).get("target_input") or "") == str(role)
            and (by_step.get(edge.source_step) is not None)
            and by_step[edge.source_step].passed
            and by_step[edge.source_step].outputs.get(str(
                (edge.mapping or {}).get("source_output") or "")) == value
        ), None)
        if incoming is not None:
            source_output = str((incoming.mapping or {}).get(
                "source_output") or "")
            provenance = BindingProvenance(
                source="data_flow", role=str(role),
                source_step=str(incoming.source_step),
                source_output=source_output,
                evidence=("validated_source_output",),
            )
        elif (spec is not None and spec.kind == BindingKind.TASK
              and task_params.get(spec.task_role) == value):
            provenance = BindingProvenance(
                source="task", role=str(role),
                evidence=(f"task_role={spec.task_role}",),
            )
        elif role in task_params and task_params.get(role) == value:
            provenance = BindingProvenance(
                source="task", role=str(role),
                evidence=(f"task_role={role}",),
            )
        elif spec is not None and spec.kind == BindingKind.LITERAL:
            provenance = BindingProvenance(
                source="literal", role=str(role),
                evidence=("persisted_literal",),
            )
        elif spec is not None and spec.kind == BindingKind.STATE:
            provenance = BindingProvenance(
                source="state", role=str(role),
                evidence=(spec.state_predicate or "structured_state",),
            )
        else:
            # The value was refined from the current structured state/shared
            # verified occurrence context.  This is auditable but deliberately
            # weaker than an explicit DATA_FLOW producer.
            provenance = BindingProvenance(
                source="state", role=str(role),
                evidence=("runtime_state_or_verified_shared_binding",),
            )
        result[str(role)] = provenance.to_dict()
    return result


def _update_verified_runtime_bindings(task: Task, planned: Any,
                                      runtime_graph: RuntimeGraph,
                                      shared: dict[str, Any], index: int) -> None:
    """Commit concrete bindings only after the current Atomic validates."""
    cardinality = max(
        [max(1, int(effect.get("cardinality", 1) or 1))
         for effect in task.target_effects if isinstance(effect, dict)] or [1])
    distinct_roles = set(
        dict(getattr(planned, "distinct_bindings", {}) or {}))
    for role, value in dict(planned.params or {}).items():
        if value in (None, "") or str(value).startswith("$"):
            continue
        current = shared.get(role)
        if current in (None, "") or _runtime_binding_can_refine(current, value):
            # A cardinality workflow deliberately selects a new entity on each
            # branch.  Its identity is carried by DATA_FLOW edges, not a global
            # task binding that would force all branches to reuse object one.
            if (role in distinct_roles
                    or (cardinality > 1 and role == "object_location")):
                continue
            shared[role] = value

    # Eagerly propagate mapped outputs so a later node is already concrete
    # before Tool selection and controlled location discovery.
    source_step = planned.step_id
    for edge in runtime_graph.edges:
        if edge.source_step != source_step or edge.type != EdgeType.DATA_FLOW:
            continue
        source_role = str((edge.mapping or {}).get("source_output") or "")
        target_role = str((edge.mapping or {}).get("target_input") or "")
        source_state = runtime_graph.nodes[index]
        value = source_state.outputs.get(source_role)
        if not target_role or value in (None, ""):
            continue
        for future in runtime_graph.plan.nodes[index + 1:]:
            if future.step_id != edge.target_step:
                continue
            if (future.params.get(target_role) in (None, "")
                    or _runtime_binding_can_refine(
                        future.params.get(target_role), value)):
                future.params[target_role] = value
            break


def _ground_env_runtime_params(
        params: dict[str, Any], atomic: Any,
        effects: list[dict[str, Any]], result: Any, *,
        action_start: int, action_end: int | None = None,
        before: dict[str, Any], after: dict[str, Any],
        node_ref: str = "",
        ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Ground a node's unresolved inputs from its actual execution evidence.

    Planning parameters are hypotheses.  An environment attempt can resolve a
    hypothesis to a concrete instance through an accepted action or through a
    real positive state transition.  This function only fills slots declared
    by the Atomic contract; it never invents a role from benchmark labels or
    treats task-level success as node evidence.
    """
    from .core.predicates import (
        StateSnapshot,
        check_effects,
        compute_effects,
    )

    refined = dict(params or {})
    declared_effects = [dict(item) for item in (effects or [])
                        if isinstance(item, dict)]
    contract_items = list(getattr(atomic, "preconditions", []) or [])
    contract_items += list(getattr(atomic, "effects", []) or [])
    contract_items += declared_effects
    slots = {
        str(item.get("name"))
        for item in (getattr(atomic, "inputs", []) or [])
        if isinstance(item, dict) and item.get("name")
    }
    for item in contract_items:
        _collect_input_slots(item, slots)
    evidence: list[dict[str, Any]] = []

    actions = list(getattr(result, "actions", None) or [])
    start = max(0, int(action_start))
    end = len(actions) if action_end is None else max(start, int(action_end))
    # Prefer the latest accepted grounding for each unresolved role: the action
    # nearest the observed Effect is causal evidence, while earlier actions in
    # the same Seeded attempt may merely be exploration or recovery.
    for action in reversed(actions[start:end]):
        if not bool(action.get("accepted", True)):
            continue
        action_node = str(action.get("node_ref") or "")
        if node_ref and action_node and action_node != node_ref:
            continue
        action_params = dict(action.get("params") or {})
        for slot, observed in action_params.items():
            if slot not in slots or observed in (None, ""):
                continue
            if (str(slot).endswith("_location")
                    and not _accepted_location_binding_matches_entity(
                        atomic, str(slot), refined, action_params)):
                continue
            normalized = _normalize_runtime_binding(observed)
            old = refined.get(slot)
            if not _runtime_binding_can_refine(old, normalized):
                continue
            refined[slot] = normalized
            evidence.append({
                "parameter": str(slot),
                "value": normalized,
                "source": "accepted_action_param",
                "action_step": action.get("step"),
            })

    # Action adapters need not expose every semantic role.  A newly added fact
    # that matches a declared Effect is independent state evidence for the
    # placeholder occupying that fact argument.
    positive, _negative = compute_effects(StateSnapshot(before),
                                           StateSnapshot(after))
    for expected in declared_effects:
        expected_name = _canonical_runtime_predicate(expected.get("predicate"))
        if not expected_name:
            continue
        expected_args = dict(expected.get("args") or {})
        candidates = [item for item in positive
                      if _canonical_runtime_predicate(item.get("predicate"))
                      == expected_name]
        for actual in candidates:
            actual_args = dict(actual.get("args") or {})
            proposed: dict[str, Any] = {}
            compatible = True
            for role, expected_value in expected_args.items():
                actual_value = actual_args.get(role)
                slot = _input_slot_name(expected_value)
                if slot:
                    if slot not in slots or actual_value in (None, ""):
                        continue
                    normalized = _normalize_runtime_binding(actual_value)
                    old = refined.get(slot)
                    if (_runtime_values_compatible(old, normalized)
                            or _runtime_binding_can_refine(old, normalized)):
                        proposed[slot] = normalized
                    else:
                        compatible = False
                        break
                elif actual_value not in (None, "") and not _runtime_values_compatible(
                        expected_value, actual_value):
                    compatible = False
                    break
            if not compatible or not proposed:
                continue
            trial = dict(refined)
            for slot, value in proposed.items():
                if _runtime_binding_can_refine(trial.get(slot), value):
                    trial[slot] = value
            effect_ok, _missing = check_effects(
                StateSnapshot(after), trial, [expected], {"harness": "env"})
            if not effect_ok:
                continue
            for slot, value in proposed.items():
                old = refined.get(slot)
                if not _runtime_binding_can_refine(old, value):
                    continue
                refined[slot] = value
                evidence.append({
                    "parameter": str(slot),
                    "value": value,
                    "source": "observed_positive_effect",
                    "predicate": expected_name,
                })
            break
    return refined, evidence


def _accepted_location_binding_matches_entity(
        atomic: Any, location_slot: str, current: dict[str, Any],
        action_params: dict[str, Any]) -> bool:
    """Require one accepted action to ground an entity/location pair jointly.

    A location parameter alone does not identify whose location it is.  When
    the Atomic precondition names the source relation, the same action must
    carry a compatible value for that entity role.  For destination-style
    location slots, any shared entity parameters must still be compatible.
    """
    entity_role = _source_location_entity_role(
        atomic, location_slot, current)
    if entity_role:
        observed_entity = dict(action_params or {}).get(entity_role)
        if observed_entity in (None, ""):
            return False
        existing_entity = dict(current or {}).get(entity_role)
        return (_runtime_values_compatible(existing_entity, observed_entity)
                or _runtime_binding_can_refine(
                    existing_entity, observed_entity))

    shared_roles = [
        role for role in action_params
        if role in current and not str(role).endswith("_location")
    ]
    return all(
        _runtime_values_compatible(current.get(role), action_params.get(role))
        or _runtime_binding_can_refine(
            current.get(role), action_params.get(role))
        for role in shared_roles)


def _collect_input_slots(value: Any, slots: set[str]) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            _collect_input_slots(nested, slots)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _collect_input_slots(nested, slots)
    else:
        slot = _input_slot_name(value)
        if slot:
            slots.add(slot)


def _input_slot_name(value: Any) -> str:
    text = str(value or "")
    if text.startswith("$inputs."):
        return text[len("$inputs."):]
    if text.startswith("$") and "." not in text[1:]:
        return text[1:]
    return ""


def _normalize_runtime_binding(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    from .core.predicates import normalize_value
    return normalize_value(value)


def _attribute_env_budget_failure(
        failure_type: str, *, action_end: int, global_limit: int,
        node_deadline: int, attempt_deadline: int,
        budget_scope: str) -> str:
    """Replace the adapter's ambiguous ``max_steps`` with its real owner.

    Adapter deadlines are absolute episode action indices.  Attribution must
    therefore use the same absolute indices; comparing a local action count to
    a global deadline silently mislabels resumed attempts.
    """
    if str(failure_type or "") not in {
            "max_steps", "discovery_budget_exhausted"}:
        return str(failure_type or "")
    used = int(action_end)
    if used >= int(global_limit):
        return "episode_budget_exhausted"
    # A full-task node owns the rest of the episode, so reaching its deadline
    # is necessarily an episode exhaustion even when invoked after a resume.
    if budget_scope == "task" and used >= int(node_deadline):
        return "episode_budget_exhausted"
    if used >= int(node_deadline):
        return "node_budget_exhausted"
    if used >= int(attempt_deadline):
        return "attempt_budget_exhausted"
    return str(failure_type or "")


def _runtime_values_compatible(left: Any, right: Any) -> bool:
    """Match a class-valued binding to an instance without conflating IDs."""
    import re
    from .core.predicates import normalize_value

    left_norm, right_norm = normalize_value(left), normalize_value(right)
    if not left_norm or left_norm.startswith("$"):
        return False
    if left_norm == right_norm:
        return True
    if re.search(r"_\d+$", left_norm):
        return False
    return re.sub(r"_\d+$", "", left_norm) == re.sub(r"_\d+$", "", right_norm)


def _runtime_binding_can_refine(current: Any, observed: Any) -> bool:
    """Whether observed evidence may make an existing slot more concrete."""
    import re
    from .core.predicates import normalize_value

    if observed in (None, ""):
        return False
    if current in (None, "") or str(current).startswith("$"):
        return True
    current_norm = normalize_value(current)
    observed_norm = normalize_value(observed)
    if current_norm == observed_norm:
        return False
    if re.search(r"_\d+$", current_norm):
        return False
    return (re.sub(r"_\d+$", "", current_norm)
            == re.sub(r"_\d+$", "", observed_norm)
            and bool(re.search(r"_\d+$", observed_norm)))


def _canonical_runtime_predicate(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", ".")


def _realized_task_bindings(task_params: dict[str, Any],
                            realized_nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Fill task roles absent at planning time from validated runtime nodes.

    Existing task-level values retain precedence.  In particular a generic
    cardinality slot must not be narrowed to one of several occurrences.
    """
    realized = dict(task_params or {})
    for node in realized_nodes or []:
        if not isinstance(node, dict):
            continue
        for slot, value in dict(node.get("params") or {}).items():
            if value in (None, ""):
                continue
            current = realized.get(slot)
            if current in (None, "") or str(current).startswith("$"):
                realized[slot] = value
    return realized


def _distinct_group_target_effect(
        task: Task, group_id: str) -> dict[str, Any] | None:
    import re

    matched = re.match(r"target_(\d+):", str(group_id or ""))
    if matched is None:
        return None
    index = int(matched.group(1))
    effects = list(task.target_effects or [])
    if index >= len(effects) or not isinstance(effects[index], dict):
        return None
    return effects[index]


def _completed_distinct_effect_instances(
        task: Task, state: dict[str, Any], params: dict[str, Any], *,
        contract: dict[str, Any] | None = None) -> set[str]:
    """Return witnesses already satisfying any cardinality goal contract.

    This is execution state, not a persisted instance-specific Skill/Tool
    identity. The trigger is the goal contract itself, never a benchmark task
    label or operation name. It covers placement as well as arbitrary state
    predicates such as toggled/heated/inspected.
    """
    import re
    from .core.predicates import (
        bind_args, normalize_value, ordered_predicate_args,
        predicate_fact_signature)

    contracts = ([contract] if isinstance(contract, dict) else [
        effect for effect in (task.target_effects or [])
        if isinstance(effect, dict)])
    context = dict(task.context.get("params") or {})
    bindings = {**context, **dict(params or {})}
    completed: set[str] = set()
    for current in contracts:
        if (int(current.get("cardinality", 1) or 1) <= 1
                or not str(current.get("distinct_by") or "")):
            continue
        distinct_arg = str(current.get("distinct_by") or "")
        predicate = str(current.get("predicate") or "")
        bound_args = bind_args(
            dict(current.get("args") or {}), bindings, bindings)
        ordered_args = ordered_predicate_args(predicate, bound_args)
        keys = [key for key, _value in ordered_args]
        args = dict(ordered_args)
        if distinct_arg not in keys:
            continue
        distinct_index = keys.index(distinct_arg)
        fact_name, _argument_order = predicate_fact_signature(predicate)
        expected_predicate = _runtime_state_predicate(fact_name)
        for fact in state.get("facts", []) or []:
            match = re.fullmatch(
                r"\s*([a-zA-Z0-9_.]+)\((.*)\)\s*", str(fact))
            if match is None or _runtime_state_predicate(
                    match.group(1)) != expected_predicate:
                continue
            values = [normalize_value(item.strip())
                      for item in match.group(2).split(",")]
            if len(values) != len(keys):
                continue
            compatible = True
            for index, key in enumerate(keys):
                expected = args.get(key)
                if expected in (None, "") or str(expected).startswith("$"):
                    continue
                if not distinct_values_conflict(expected, values[index]):
                    compatible = False
                    break
            if compatible:
                completed.add(values[distinct_index])
    return completed


def _runtime_state_predicate(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace(".", "_")
    return {
        "object_at_location": "object_at",
        "object_in_receptacle": "object_at",
        "object_in_container": "object_at",
    }.get(normalized, normalized)
