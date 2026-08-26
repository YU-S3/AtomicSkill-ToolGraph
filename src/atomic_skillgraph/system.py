"""AtomicSkillGraphSystem：v2.0 运行时总装（设计文档 v2.0 §43 伪代码实现）。

Known Capability → Plan Before Execution（warm）
Unknown Capability → Learn After Successful Execution（cold）
"""

from __future__ import annotations

import copy
import time
from typing import Any

from .adapters.benchmark import BenchmarkAdapter, Task
from .atomicizer.trace_atomicizer import TraceAtomicizer
from .core.config import SystemConfig
from .core.llm import LLM
from .core.status import EdgeType, ExecutionMode, SkillNodeKind, SkillStatus, ToolLifecycle
from .core.trace_ir import (
    ActionRecord,
    AttemptRecord,
    NodeValidationResult,
    TaskExecutionInstance,
    TraceRecord,
)
from .evolution.failure_processor import FailureProcessor
from .evolution.branch_repair import FailureBranchManager
from .evolution.proposal_replayer import ProposalReplayer
from .evolution.success_processor import SuccessProcessor
from .graph.registry import SkillGraphRegistry
from .graph.validator import validate_graph
from .persistence import MetricsStore, RunStore, TraceStore
from .runtime.atomic_planner import AtomicPlanner
from .runtime.execution_bridge import ExecutionBridge
from .runtime.implementation_selector import ImplementationSelector
from .runtime.runtime_graph import RuntimeGraph
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
        usage_baseline = self._combined_llm_tokens()
        plan = self.planner.compile_runtime_graph(task)

        is_env = _is_env_task(task)
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
            for kind, refs, passed, mode in final_feedback:
                if kind == "tool":
                    self._record_tool_feedback(refs, passed, mode)
                else:
                    self._record_impl_feedback(refs, passed)
            assert trace is not None
            trace.metrics["infrastructure_episode_retries"] = max(
                0, len(infrastructure_attempts) - int(trace.failure_type == "llm_error"))
            if infrastructure_attempts:
                trace.metrics["infrastructure_attempts"] = infrastructure_attempts
        else:
            trace = self._run_code_task(task, plan)

        trace.latency_ms = (time.perf_counter() - started) * 1000.0
        trace.token_cost = max(0, self._combined_llm_tokens() - usage_baseline)
        if trace.success:
            trace.metrics.setdefault("task_success", 1)
        trace.provenance.setdefault("target_effects", copy.deepcopy(task.target_effects))
        trace.provenance.setdefault("params", copy.deepcopy(task.context.get("params") or {}))
        trace.provenance.setdefault(
            "semantic_params",
            copy.deepcopy(task.context.get("semantic_params")
                          or task.context.get("params") or {}))
        self._finalize_validation(trace, task)
        # 保存 trace（成功与失败都保存）
        self.trace_store.save(trace)
        evolution = self._process_trace(trace, task)
        # Failure-branch strict replay may itself call the LLM. Include that
        # learning cost in the online episode rather than hiding it.
        trace.latency_ms = (time.perf_counter() - started) * 1000.0
        trace.token_cost = max(0, self._combined_llm_tokens() - usage_baseline)
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
        node_mode_counts: dict[str, int] = {}
        executed_node_count = 0
        for node in runtime_graph.nodes:
            if not node.attempts and node.validation is None and not node.impl_ref:
                continue
            executed_node_count += 1
            node_mode_counts[node.mode.value] = node_mode_counts.get(node.mode.value, 0) + 1
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
            and executed_node_count < len(runtime_graph.nodes)
        )
        episode = {
            "episode": self.episode_count,
            "task_id": task.task_id,
            "benchmark": task.benchmark,
            "task_type": task.task_type,
            "success": trace.success,
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
            "latency_ms": round(trace.latency_ms, 1),
            "direct_reuse_count": trace.direct_use_count(),
            "seeded_generation_count": trace.seeded_use_count(),
            "dynamic_generation_count": trace.dynamic_use_count(),
            "evolution": evolution,
            "graph_validation": graph_report.to_dict(),
            "node_mode_counts": node_mode_counts,
            "planned_node_count": len(runtime_graph.nodes),
            "executed_node_count": executed_node_count,
            # The benchmark may legally terminate before every retrieved node is
            # needed.  Keep this separate from a runtime/tool failure.  This is
            # observable in ALFWorld games whose initial PDDL state already
            # satisfies one component of the natural-language goal.
            "goal_terminal_before_plan_complete": goal_terminal_before_plan_complete,
            "goal_terminal_skipped_node_count": (
                len(runtime_graph.nodes) - executed_node_count
                if goal_terminal_before_plan_complete else 0
            ),
            "fallback_node_count": sum(bool(node.fallback_reason)
                                       for node in runtime_graph.nodes),
            "validation_summary": validation_summary,
            "proposal_replay_count": len(evolution.get("proposal_replays") or []),
            "trace_id": trace.trace_id,
            "reused_skill_refs": (trace.retrieved_skill_refs
                                  if trace.start_mode == "warm" else []),
            "used_tool_refs": list(trace.tool_refs),
            "cross_task_type_reuse": self._cross_task_type_reuse(task, trace),
            "skill_graph": self.registry.stats(),
            "tool_repo": self.tool_registry.stats(),
        }
        self.metrics_store.save_episode(self.episode_count, episode)
        return episode

    def _combined_llm_tokens(self) -> int:
        """Runtime and Extractor are independent clients but one episode cost."""
        clients = {id(self.llm): self.llm, id(self.extractor_llm): self.extractor_llm}
        return sum(int(getattr(client.usage, "total_tokens", 0) or 0)
                   for client in clients.values())

    def _cross_task_type_reuse(self, task: Task, trace: TraceRecord) -> bool:
        """是否发生了跨 task_type 的能力复用（§58.3 指标）。"""
        tool_types: set[str] = set()
        for tool_ref_text in trace.tool_refs:
            try:
                from .core.refs import ToolRef
                ref = ToolRef.parse(tool_ref_text)
            except ValueError:
                continue
            tool = self.tool_registry.get(ref) or self.tool_registry.get_recommended(ref.tool_id)
            if tool is None:
                continue
            tool_types |= set(tool.provenance.get("source_task_types") or [])
        if task.task_type and tool_types and not tool_types.issubset({task.task_type}):
            return True
        used_refs = [str(node["ref"]) for node in trace.realized_atomic_nodes
                     if node.get("ref")]
        if trace.selected_composite:
            used_refs.append(trace.selected_composite)
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
            obj = self.registry.get(ref) or self.registry.get_recommended(ref.logical_id)
            if obj is None or not hasattr(obj, "metadata"):
                continue
            labels = set(obj.metadata.get("task_type_labels") or [])
            if task.task_type and labels and not labels.issubset({task.task_type}):
                return True
        return False

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
            atomic = self._get_recommended(planned.ref.logical_id)
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
                trace.node_validators.append(validation)
                self._record_tool_feedback(node.tool_refs, validation.passed, "direct")
                self._record_impl_feedback([node.impl_ref], validation.passed)
                if validation.passed:
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
        runtime_graph = RuntimeGraph(task.task_id, plan)
        self._last_runtime_graph = runtime_graph

        if not plan.nodes:
            trace.planning_mode = "pure_dynamic"
            runtime_graph.record_usage(ExecutionMode.DYNAMIC)
            result = self.adapter.run_env_episode(task, self.llm, seed_context="",
                                                  max_steps=self.config.max_steps)
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
            atomic = self._get_recommended(planned.ref.logical_id)
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
                "steps": steps,
                "atomic": atomic,
                "inputs": planned.params,
                "graph_index": index,
            })

        if all_direct and direct_steps:
            runtime_graph.record_usage(ExecutionMode.DIRECT)
            result = self.adapter.run_env_episode(
                task, self.llm, seed_context="",
                direct_steps=[{"node_ref": d["node_ref"],
                               "steps": d["steps"],
                               "tool_ref": ""}
                              for d in direct_steps],
                max_steps=self.config.max_steps)
            # 节点级验证：按 direct_steps 边界映射前后状态
            self._validate_env_direct_nodes(trace, result, direct_steps, runtime_graph)
            for node in runtime_graph.nodes:
                if node.tool_refs:
                    self._record_tool_feedback(node.tool_refs, node.passed, "direct")
                if node.impl_ref:
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
            before = dict((resume or {}).get("state") or task.state or {})
            planned.params = _apply_runtime_data_bindings(
                planned.params, planned.step_id, runtime_graph, shared_bindings)
            # A plan may initially contain a class-valued entity slot because a
            # concrete instance is still hidden. Once execution establishes an
            # instance identity, carry it through later DATA_FLOW edges.
            planned.params = _refine_env_object_binding(planned.params, before)
            # RuntimeNodeState is created before execution.  Keep its persisted
            # view synchronized with parameters refined by prior-node data flow
            # or by framework discovery; otherwise node_results records the
            # stale class-valued planning input rather than the executed one.
            node.params = dict(planned.params)
            atomic = None if planned.dynamic else self._get_recommended(planned.ref.logical_id)
            effects = list(planned.target_effects or getattr(atomic, "effects", []) or [])
            candidates: list[tuple[ExecutionMode, str, list[dict[str, Any]]]] = []
            location_discovered = False

            if atomic is not None:
                produces_possession = any(
                    str(effect.get("predicate") or "").replace("_", ".") == "agent.holds"
                    for effect in effects if isinstance(effect, dict))
                discover = getattr(self.adapter, "discover_object_location", None)
                # Location slots come from the learned Atomic/Tool interface,
                # never from task_type or a fixed operation list. Bind anything
                # already witnessed in state before opening the bounded search.
                planned.params = _bind_known_location_slots(
                    planned.params, atomic, before)
                node.params = dict(planned.params)
                location_slots = self.selector.discoverable_location_slots(
                    atomic.ref, {"inputs": planned.params, "harness": "env"})
                location_slots |= {
                    str(item.get("name")) for item in (atomic.inputs or [])
                    if isinstance(item, dict)
                    and str(item.get("name") or "").endswith("_location")
                    and not planned.params.get(str(item.get("name")))
                }
                if (self.config.features.enable_tool_evolution
                        and callable(discover) and location_slots):
                    partial = self.selector.select_allowing_missing(
                        atomic.ref, {"inputs": planned.params, "harness": "env"},
                        set(location_slots))
                    if partial.implementation is not None:
                        partial_resolved = self.resolver.resolve(
                            partial.implementation, {"inputs": planned.params})
                        discovery_tool_refs = [str(item.binding.tool_ref)
                                               for item in partial_resolved]
                        for location_slot in sorted(location_slots):
                            if planned.params.get(location_slot):
                                continue
                            entity_role = location_slot[:-len("_location")]
                            entity_value = planned.params.get(entity_role)
                            if entity_value in (None, ""):
                                continue
                            excluded_objects = (
                                _completed_distinct_effect_instances(
                                    task, before, planned.params)
                                if entity_role == "object" else set())
                            binding, discovery_result = discover(
                                task, str(entity_value), resume=resume,
                                max_locations=self.config.thresholds.acquire_discovery_max_locations,
                                node_ref=str(planned.ref),
                                tool_ref=discovery_tool_refs[0]
                                if discovery_tool_refs else "",
                                excluded_objects=excluded_objects,
                                allow_passive_navigable=not produces_possession)
                            resume = _env_resume_payload(discovery_result)
                            before = dict(resume.get("state") or before)
                            remapped = _remap_location_binding(
                                binding, entity_role, location_slot)
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
                                "search_actions": len(discovery_result.actions),
                            }
                            trace.metrics.setdefault(
                                "controlled_location_discovery", []).append(metric)
                            if produces_possession and entity_role == "object":
                                trace.metrics.setdefault(
                                    "acquire_location_discovery", []).append(metric)
                            if remapped:
                                planned.params.update(remapped)
                                node.params = dict(planned.params)
                                location_discovered = True
                            else:
                                node.fallback_reason = (
                                    f"location_discovery_failed:{location_slot}")
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

            node_succeeded = False
            for mode, seed_context, steps in candidates:
                attempt_before = dict((resume or {}).get("state") or before)
                action_start = len((resume or {}).get("actions") or [])
                runtime_graph.record_usage(mode)
                node.mode = mode
                direct_steps = None
                if mode == ExecutionMode.DIRECT:
                    direct_steps = [{
                        "node_ref": str(planned.ref), "tool_ref": node.tool_refs[0]
                        if node.tool_refs else "", "steps": steps,
                    }]
                result = self.adapter.run_env_episode(
                    task, self.llm, seed_context=seed_context,
                    direct_steps=direct_steps, max_steps=self.config.max_steps,
                    resume=resume, stop_effects=effects,
                    effect_inputs=dict(planned.params), node_ref=str(planned.ref),
                    phase_goal=_phase_goal_of(atomic, effects, planned.params),
                )
                last_result = result
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
                    passed = validation.passed or (bool(result.success) and not effects)
                    trace.node_validators.append(validation)
                elif effects:
                    from .core.predicates import StateSnapshot, check_effects
                    passed, missing = check_effects(StateSnapshot(after), planned.params,
                                                    effects, {"harness": "env"})
                    validation = NodeValidationResult(
                        node_ref=str(planned.ref), level="atomic", passed=passed,
                        checks={"effects": passed}, before=before, after=after,
                        messages=[] if passed else [f"动态节点效果未发生：{missing}"],
                    )
                    trace.node_validators.append(validation)
                node.attempts.append({"mode": mode.value, "passed": passed,
                                      "failure_type": str(result.failure_type or ""),
                                      "params": dict(planned.params),
                                      "tool_refs": list(node.tool_refs)
                                      if mode == ExecutionMode.DIRECT else [],
                                      "action_start": action_start,
                                      "action_end": len(after_payload.get("actions") or []),
                                      "before": attempt_before,
                                      "after": after})
                node.validation = validation
                node.before, node.after, node.passed = before, after, passed
                resume = after_payload
                # Executable 证据必须按实际 attempt 记账。后续 Seeded/Dynamic
                # 是否救回节点，都不能覆盖本次 Direct Tool 的真实结果。
                if mode == ExecutionMode.DIRECT and node.tool_refs:
                    self._record_tool_feedback(node.tool_refs, passed, "direct")
                if mode == ExecutionMode.DIRECT and node.impl_ref:
                    self._record_impl_feedback([node.impl_ref], passed)
                if passed:
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

            if last_result is not None and last_result.success:
                break
            if not node_succeeded:
                break

        # 所有原子目标已满足但 benchmark 尚未发出 won 时，让动态 Agent 收尾；
        # 这是任务级验证，不改变已完成节点的归因。
        all_nodes_passed = bool(runtime_graph.nodes) and all(n.passed for n in runtime_graph.nodes)
        if last_result is not None and not last_result.success and all_nodes_passed:
            runtime_graph.record_usage(ExecutionMode.DYNAMIC)
            last_result = self.adapter.run_env_episode(
                task, self.llm, seed_context="", max_steps=self.config.max_steps,
                resume=resume,
            )
        if last_result is None:
            last_result = self.adapter.run_env_episode(task, self.llm,
                                                       max_steps=self.config.max_steps)
        self._fill_env_trace(trace, last_result, runtime_graph)
        return trace

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
                result = self.composite_validator.validate_composite(
                    composite,
                    [item for item in trace.node_validators if item.level == "atomic"],
                    trace.final_state(), inputs=realized_inputs,
                    context={"harness": "env" if _is_env_task(task) else "code_math"},
                )
                trace.validation_layers["composite"] = result.to_dict()
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
        action_index = 0
        for spec in direct_steps:
            step_count = len(spec["steps"])
            action_start = action_index
            before = states[action_start] if action_start < len(states) else {}
            after_index = min(action_start + step_count, len(states) - 1)
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
                trace.metrics.setdefault("runtime_param_bindings", []).append({
                    "node_ref": spec["node_ref"],
                    "mode": ExecutionMode.DIRECT.value,
                    "bindings": binding_evidence,
                })
            validation = self.node_validator.validate_atomic(
                atomic, before, after, inputs=grounded,
                context={"harness": "env"})
            validation.node_ref = spec["node_ref"]
            node.validation = validation
            node.before = before
            node.after = after
            node.passed = validation.passed
            trace.node_validators.append(validation)

    def _fill_env_trace(self, trace: TraceRecord, result: Any,
                        runtime_graph: RuntimeGraph) -> None:
        trace.success = bool(result.success)
        trace.failure_type = str(result.failure_type or "")
        if getattr(result, "infrastructure_errors", None):
            trace.metrics["infrastructure_errors"] = [
                dict(item) for item in result.infrastructure_errors]
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
            impl = self.registry.get(ref) or self.registry.get_recommended(ref.logical_id)
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
        # Attribute evidence only to the path that actually executed. Retrieval
        # is exposure, not use; an LLM rescue must not credit every candidate.
        used_refs = [str(node["ref"]) for node in trace.realized_atomic_nodes
                     if node.get("ref") and (not success or bool(node.get("passed")))]
        composite_validation = trace.validation_layers.get("composite") or {}
        if trace.selected_composite and (
                not success or bool(composite_validation.get("passed"))):
            used_refs.append(trace.selected_composite)
        seen_task_skills: set[str] = set()
        for ref_text in used_refs:
            try:
                from .core.refs import SkillRef
                ref = SkillRef.parse(ref_text)
            except ValueError:
                continue
            # A Composite may call the same Atomic more than once, and some
            # planners may return duplicate refs.  Task-level evidence is one
            # Bernoulli observation per logical Skill per task, not per call.
            if ref.logical_id in seen_task_skills:
                continue
            seen_task_skills.add(ref.logical_id)
            obj = self.registry.get(ref) or self.registry.get_recommended(ref.logical_id)
            if obj is None:
                continue
            if not hasattr(obj, "metadata"):
                continue  # ImplementationAtom 不承载 Skill 层证据
            stats = dict(obj.metadata.get("statistics") or {})
            # 这一层记录实际路径的 task-level 证据；节点级 attempt 统计仍独立。
            stats["task_use_count"] = int(stats.get("task_use_count", 0)) + 1
            stats["use_count"] = int(stats.get("use_count", 0)) + 1
            if success:
                stats["task_success_count"] = int(stats.get("task_success_count", 0)) + 1
                stats["execution_success_count"] = int(
                    stats.get("execution_success_count", 0)) + 1
            else:
                stats["task_failure_count"] = int(stats.get("task_failure_count", 0)) + 1
                stats["execution_failure_count"] = int(
                    stats.get("execution_failure_count", 0)) + 1
            total = int(stats.get("use_count", 0))
            empirical = int(stats.get("execution_success_count", 0)) / max(total, 1)
            old = float(stats.get("utility", 0.5))
            stats["utility"] = round(0.5 * old + 0.5 * empirical, 4)
            obj.metadata["statistics"] = stats
            # 负迁移抑制：复用后失败 >=2 且 utility 低 → suppressed
            if int(stats.get("task_failure_count", 0)) >= 2 and stats["utility"] < 0.35:
                obj.status = SkillStatus.SUPPRESSED
            self.registry.update_runtime_state(obj)

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
        atomic = registry.get_recommended(node.ref.logical_id)
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
                   params: dict[str, Any]) -> str:
    """构造只覆盖当前 Atomic Effect 的阶段目标，防止一个 Seeded 节点包办整题。"""
    summary = str(getattr(atomic, "summary", "") or "complete the current atomic step")
    bound: list[str] = []
    from .core.predicates import bind_args
    for effect in effects:
        args = bind_args(dict(effect.get("args") or {}), params, {})
        detail = ", ".join(f"{key}={value}" for key, value in args.items())
        bound.append(f"{effect.get('predicate')}({detail})")
    target = "; ".join(bound) or summary
    return (f"Complete ONLY this atomic step: {summary}. Required state: {target}. "
            "Do not continue to later task stages after this state is reached.")


def _refine_env_object_binding(params: dict[str, Any],
                               state: dict[str, Any]) -> dict[str, Any]:
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
    matches = []
    for item in state.get("inventory", []) or []:
        actual = normalize_value(item)
        if re.sub(r"_\d+$", "", actual) == family:
            matches.append(actual)
    if len(matches) == 1:
        refined["object"] = matches[0]
    return refined


def _bind_known_location_slots(params: dict[str, Any], atomic: Any,
                               state: dict[str, Any]) -> dict[str, Any]:
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
    for slot in slots:
        if refined.get(slot) not in (None, ""):
            continue
        entity_role = slot[:-len("_location")]
        wanted = normalize_value(refined.get(entity_role, ""))
        if not wanted:
            continue
        candidates: list[tuple[str, str]] = []
        for fact in facts:
            match = re.fullmatch(r"object_at\((.+?),\s*(.+?)\)", str(fact))
            if not match:
                continue
            actual, location = normalize_value(match.group(1)), normalize_value(match.group(2))
            if _runtime_values_compatible(wanted, actual):
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
                                 shared: dict[str, Any]) -> dict[str, Any]:
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
        value = source.params.get(source_role)
        if target_role and value not in (None, "") and (
                refined.get(target_role) in (None, "")
                or _runtime_binding_can_refine(refined.get(target_role), value)):
            refined[target_role] = value
    for role, value in shared.items():
        if role not in refined or value in (None, ""):
            continue
        if _runtime_binding_can_refine(refined.get(role), value):
            refined[role] = value
    return refined


def _update_verified_runtime_bindings(task: Task, planned: Any,
                                      runtime_graph: RuntimeGraph,
                                      shared: dict[str, Any], index: int) -> None:
    """Commit concrete bindings only after the current Atomic validates."""
    cardinality = max(
        [max(1, int(effect.get("cardinality", 1) or 1))
         for effect in task.target_effects if isinstance(effect, dict)] or [1])
    for role, value in dict(planned.params or {}).items():
        if value in (None, "") or str(value).startswith("$"):
            continue
        current = shared.get(role)
        if current in (None, "") or _runtime_binding_can_refine(current, value):
            # A cardinality workflow deliberately selects a new entity on each
            # branch.  Its identity is carried by DATA_FLOW edges, not a global
            # task binding that would force all branches to reuse object one.
            if cardinality > 1 and role in {"object", "object_location"}:
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
        value = planned.params.get(source_role)
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
        for slot, observed in dict(action.get("params") or {}).items():
            if slot not in slots or observed in (None, ""):
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


def _completed_distinct_effect_instances(task: Task, state: dict[str, Any],
                                         params: dict[str, Any]) -> set[str]:
    """Exclude instances already satisfying a cardinality placement goal.

    This is execution state, not a persisted instance-specific Skill/Tool
    identity. The trigger is the goal contract itself, never a benchmark task
    label. It prevents a repeated generic producer branch from selecting a
    completed distinct instance back out of the destination.
    """
    import re
    from .core.predicates import bind_args, normalize_value

    contract = next((effect for effect in (task.target_effects or [])
                     if isinstance(effect, dict)
                     and str(effect.get("predicate") or "") == "object.at_location"
                     and int(effect.get("cardinality", 1) or 1) > 1
                     and str(effect.get("distinct_by") or "") == "object"), None)
    if contract is None:
        return set()
    context = dict(task.context.get("params") or {})
    bindings = {**context, **dict(params or {})}
    args = bind_args(dict(contract.get("args") or {}), bindings, bindings)
    wanted = normalize_value(args.get("object") or "")
    target = normalize_value(args.get("location") or "")
    if not wanted or not target:
        return set()
    family = re.sub(r"_\d+$", "", wanted)
    completed: set[str] = set()
    for fact in state.get("facts", []) or []:
        match = re.fullmatch(r"object_at\((.+?),\s*(.+?)\)", str(fact))
        if not match:
            continue
        obj, location = normalize_value(match.group(1)), normalize_value(match.group(2))
        if re.sub(r"_\d+$", "", obj) == family and location == target:
            completed.add(obj)
    return completed
