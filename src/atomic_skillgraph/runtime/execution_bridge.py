"""Execution Bridge：节点执行路由（设计文档 v2.0 §22、§43）。

Route 1: Direct Skill/Tool Execution（满足可靠性硬门槛）
    ↓ 不够可靠或执行失败
Route 2: Skill-Conditioned Generation（seeded）
    ↓ 仍失败/无匹配能力
Route 3: Pure Dynamic Planning / Generation

硬门槛（§22.1）：admission passed、未 suppressed/retired、preconditions 满足、
Implementation compatible、Interface 可绑定、历史可靠性达阈值、无已知严重负迁移。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.config import SystemConfig
from ..core.skill_ir import AbstractAtomicSkill, ImplementationAtom
from ..core.status import ExecutionMode
from ..tools.resolver import ResolvedTool


@dataclass
class DirectGateResult:
    eligible: bool = False
    reason: str = ""
    checks: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"eligible": self.eligible, "reason": self.reason, "checks": self.checks}


class ExecutionBridge:
    """在 benchmark adapter 与 Runtime Graph 之间桥接节点执行。"""

    def __init__(self, adapter, config: SystemConfig) -> None:
        self.adapter = adapter
        self.config = config

    # ------------------------------------------------------------------
    # 硬门槛（§22.1）
    # ------------------------------------------------------------------
    def direct_gate(self, atomic: AbstractAtomicSkill, impl: ImplementationAtom,
                    resolved: list[ResolvedTool], state: dict[str, Any],
                    context: dict[str, Any]) -> DirectGateResult:
        checks: dict[str, bool] = {}
        reasons: list[str] = []

        checks["implementation_active"] = impl.status.value == "active"
        if not checks["implementation_active"]:
            reasons.append("implementation_not_active")

        checks["tools_admitted"] = bool(resolved) and all(
            r.ok and r.tool is not None and r.tool.is_usable() for r in resolved)
        if not checks["tools_admitted"]:
            reasons.append("tool_not_admitted_or_unbindable")

        for tool in (r.tool for r in resolved if r.tool is not None):
            if tool.status.value in ("suppressed", "retired"):
                checks.setdefault("tool_not_suppressed", False)
                reasons.append(f"tool_suppressed:{tool.ref}")
                break
        checks.setdefault("tool_not_suppressed", True)

        # 历史可靠性（admission replay / seeded 成功也可作为引导证据）
        reliability_ok = True
        for tool in (r.tool for r in resolved if r.tool is not None):
            stats = tool.statistics or {}
            utility = float(stats.get("utility", 0.0))
            direct_success = int(stats.get("direct_success_count", 0))
            success = int(stats.get("success_count", 0))
            admission_success = int(stats.get("admission_replay_success_count", 0))
            min_success = self.config.thresholds.direct_min_success
            if utility < self.config.thresholds.direct_min_utility:
                reliability_ok = False
                reasons.append(f"low_utility:{tool.ref}")
            admitted_active = (tool.status.value in ("active", "preferred")
                               and admission_success >= 1)
            if (direct_success < min_success and success < min_success
                    and not admitted_active):
                reliability_ok = False
                reasons.append(f"insufficient_success_evidence:{tool.ref}")
        checks["reliability_threshold"] = reliability_ok

        # 前置条件（谓词层）
        from ..core.predicates import StateSnapshot, evaluate_preconditions, bind_args
        snapshot = StateSnapshot(state)
        inputs = dict(context.get("inputs") or {})
        pre_ok, _missing = evaluate_preconditions(snapshot, inputs,
                                                  atomic.preconditions, context)
        checks["preconditions"] = pre_ok
        if not pre_ok:
            reasons.append("precondition_violation")

        eligible = all(checks.values())
        return DirectGateResult(
            eligible=eligible,
            reason=";".join(reasons) or "all_gates_passed",
            checks=checks,
        )

    # ------------------------------------------------------------------
    # Route 1: direct
    # ------------------------------------------------------------------
    def execute_direct(self, task, atomic: AbstractAtomicSkill,
                       impl: ImplementationAtom, resolved: list[ResolvedTool],
                       state: dict[str, Any]) -> dict[str, Any]:
        """直接执行已解析的 Tool（§43：execute_atomic_with_tools）。"""
        from ..core.trace_ir import new_id
        primary = resolved[0]
        result = self.adapter.execute_tool(task, primary.tool, primary.parameters, state)
        result.setdefault("node_ref", str(atomic.ref))
        result.setdefault("tool_refs", [str(r.binding.tool_ref) for r in resolved])
        result.setdefault("execution_id", new_id("direct"))
        return result

    # ------------------------------------------------------------------
    # Route 2: seeded generation
    # ------------------------------------------------------------------
    def seed_context(self, atomic: AbstractAtomicSkill,
                     impl: ImplementationAtom,
                     resolved: list[ResolvedTool],
                     tool_registry) -> str:
        """组装 skill-conditioned generation 的种子上下文。

        LLM 不看到整个 Tool Repository，只看到当前 Atomic Skill 的 guideline +
        已绑定 Tool 的摘要与代码（§13：Planner 选 Skill，Resolver 解析 Tool）。
        """
        parts: list[str] = []
        parts.append(f"[Atomic Skill] {atomic.summary}")
        for rule in atomic.guideline_rules():
            parts.append(f"  - {rule}")
        for resolved_tool in resolved[:2]:
            tool = resolved_tool.tool
            if tool is None:
                continue
            parts.append(f"[Tool] {tool.summary}")
            body = tool.artifact_body()
            if body:
                parts.append("```python" if tool.artifact_kind.value == "python_callable" else "```text")
                parts.append(body[:3000])
                parts.append("```")
        return "\n".join(parts)

    def generate_seeded(self, task, seed_context: str) -> dict[str, Any]:
        if hasattr(self.adapter, "generate_code"):
            return _code_result_to_dict(self.adapter.generate_code(
                task, self._llm, seed_context=seed_context,
                max_repairs=self.config.max_repairs))
        if hasattr(self.adapter, "run_env_episode"):
            return _env_result_to_dict(self.adapter.run_env_episode(
                task, self._llm, seed_context=seed_context,
                max_steps=self.config.max_steps))
        raise NotImplementedError("adapter 不支持 seeded generation")

    def generate_dynamic(self, task) -> dict[str, Any]:
        if hasattr(self.adapter, "generate_code"):
            return _code_result_to_dict(self.adapter.generate_code(
                task, self._llm, seed_context="",
                max_repairs=self.config.max_repairs))
        if hasattr(self.adapter, "run_env_episode"):
            return _env_result_to_dict(self.adapter.run_env_episode(
                task, self._llm, seed_context="",
                max_steps=self.config.max_steps))
        raise NotImplementedError("adapter 不支持 dynamic generation")

    def set_llm(self, llm) -> None:
        self._llm = llm


def _code_result_to_dict(result) -> dict[str, Any]:
    return {
        "mode": "code",
        "success": result.success,
        "attempts": [a.to_dict() for a in result.attempts],
        "candidate_code": result.candidate_code,
        "feedback": result.feedback.to_dict(),
        "failure_type": result.failure_type,
        "first_attempt_success": result.first_attempt_success,
        "retry_count": result.retry_count,
    }


def _env_result_to_dict(result) -> dict[str, Any]:
    return {
        "mode": "env",
        "success": result.success,
        "actions": result.actions,
        "states": result.states,
        "steps": result.steps,
        "failure_type": result.failure_type,
        "direct_used": result.direct_used,
        "seeded_used": result.seeded_used,
        "dynamic_used": result.dynamic_used,
        "final_observation": result.final_observation,
    }
