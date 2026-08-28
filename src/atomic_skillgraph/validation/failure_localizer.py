"""节点级错误归因（设计文档 v2.0 §34）。

定位流程：
    1. 找到第一个失败 Atomic Node / validator
    2. 检查上游输入是否已经污染
    3. 检查 Implementation Atom 是否选错
    4. 检查 tool_ref / parameter binding
    5. 检查 Tool-level execution/test
    6. 检查 Atomic Effect 是否满足
    7. 检查 control/data edge
    8. 检查 Composite 高层目标
    9. 输出 responsibility + confidence

原子 Tool 内部错误不默认继续细分（§34.3）；只有长期证据表明其内部形成独立
复用/失败簇，才触发 split_tool / split_skill。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.skill_ir import AbstractAtomicSkill
from ..core.status import ErrorKind, ExecutionMode
from ..core.trace_ir import NodeValidationResult, TraceRecord


@dataclass
class FailureAttribution:
    node_ref: str = ""
    step_id: str = ""
    occurrence_id: str = ""
    kind: ErrorKind = ErrorKind.UNKNOWN
    responsibility: str = ""      # atomic | implementation | tool | control_flow | benchmark | unknown
    confidence: float = 0.0
    message: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_ref": self.node_ref,
            "step_id": self.step_id,
            "occurrence_id": self.occurrence_id,
            "kind": self.kind.value,
            "responsibility": self.responsibility,
            "confidence": round(self.confidence, 3),
            "message": self.message,
            "evidence": self.evidence,
        }


class FailureLocalizer:
    """失败轨迹 → 归因列表。"""

    def localize(self, trace: TraceRecord, registry=None) -> list[FailureAttribution]:
        if trace.success:
            return []

        # Attempt-level validators may contain an early failed Direct attempt
        # which was later rescued by Seeded/Dynamic execution.  Attribute the
        # first *final failed occurrence*, never the first historical failure.
        failed_occurrences = [
            node for node in trace.realized_atomic_nodes
            if not bool(node.get("passed"))
            and (bool(node.get("attempt_started"))
                 or any(bool(a.get("started")) for a in node.get("attempts") or []))
        ]
        failed_node = failed_occurrences[0] if failed_occurrences else None
        first_failure = None
        if failed_node is not None:
            occurrence = str(failed_node.get("occurrence_id") or "")
            step_id = str(failed_node.get("step_id") or "")
            candidates = [v for v in trace.node_validators if not v.passed and (
                (occurrence and v.occurrence_id == occurrence)
                or (not occurrence and step_id and v.step_id == step_id))]
            if candidates:
                first_failure = candidates[-1]
        if first_failure is not None:
            return self._localize_node_failure(trace, first_failure)

        # Planning/budget failures did not execute a capability and therefore
        # must not be blamed on an Atomic/Tool contract.
        if trace.failure_stage in {"planning", "budget"}:
            kind = (ErrorKind.PLAN_BINDING_UNRESOLVED
                    if trace.failure_stage == "planning"
                    else ErrorKind.ATTEMPT_NOT_STARTED)
            return [FailureAttribution(
                kind=kind, responsibility=trace.failure_stage,
                confidence=1.0,
                message=trace.failure_cause or trace.failure_type,
                evidence={"failure_stage": trace.failure_stage,
                          "failure_cause": trace.failure_cause})]

        # 无节点验证器（cold start 纯动态失败）
        failure_type = trace.failure_type or ""
        if "generation" in failure_type or "llm" in failure_type:
            return [FailureAttribution(kind=ErrorKind.UNKNOWN, responsibility="unknown",
                                       confidence=0.4, message=f"生成失败：{failure_type}")]
        if trace.attempts and not any(a.passed for a in trace.attempts):
            return [FailureAttribution(kind=ErrorKind.BENCHMARK_FAILURE,
                                       responsibility="benchmark",
                                       confidence=0.8,
                                       message="所有尝试均未通过 benchmark 验证")]
        return [FailureAttribution(kind=ErrorKind.BENCHMARK_FAILURE,
                                   responsibility="benchmark", confidence=0.5,
                                   message=f"benchmark 失败：{failure_type or 'unknown'}")]

    def _localize_node_failure(self, trace: TraceRecord,
                               failure: NodeValidationResult) -> list[FailureAttribution]:
        checks = failure.checks or {}
        attributions: list[FailureAttribution] = []
        evidence = {"checks": checks, "messages": failure.messages,
                    "step_id": failure.step_id,
                    "occurrence_id": failure.occurrence_id,
                    "attempt_index": failure.attempt_index,
                    "mode": failure.mode}
        identity = {"node_ref": failure.node_ref,
                    "step_id": failure.step_id,
                    "occurrence_id": failure.occurrence_id}

        # 2. 上游输入污染：前一个节点失败
        upstream_failed = self._upstream_failed(
            trace, failure.occurrence_id or failure.step_id or failure.node_ref)
        if upstream_failed:
            attributions.append(FailureAttribution(
                node_ref=upstream_failed, kind=ErrorKind.DATA_FLOW_ERROR,
                responsibility="control_flow", confidence=0.7,
                message=f"上游节点 {upstream_failed} 失败，当前节点输入可能已被污染",
                evidence=evidence,
            ))

        # 3-6. 按检查项分类
        if not checks.get("preconditions", True):
            attributions.append(FailureAttribution(
                **identity, kind=ErrorKind.PRECONDITION_VIOLATION,
                responsibility="atomic", confidence=0.9,
                message="前置条件未满足（输入/状态不满足执行条件）",
                evidence=evidence,
            ))
        if not checks.get("effects", True):
            # 效果失败：区分 tool 执行失败 vs effect 判定失败
            matching = next((node for node in trace.realized_atomic_nodes
                             if str(node.get("occurrence_id") or "") ==
                             failure.occurrence_id), {})
            tool_failed = any(
                bool(attempt.get("started"))
                and str(attempt.get("mode") or "") == ExecutionMode.DIRECT.value
                and bool(attempt.get("tool_refs"))
                and not bool(attempt.get("passed"))
                for attempt in matching.get("attempts") or [])
            if tool_failed:
                attributions.append(FailureAttribution(
                    **identity, kind=ErrorKind.TOOL_EXECUTION_ERROR,
                    responsibility="tool", confidence=0.8,
                    message="Tool 执行未达成核心 Effect",
                    evidence=evidence,
                ))
            else:
                attributions.append(FailureAttribution(
                    **identity, kind=ErrorKind.EFFECT_VIOLATION,
                    responsibility="atomic", confidence=0.8,
                    message="核心 Effect 验证失败",
                    evidence=evidence,
                ))
        if not checks.get("parameters_bindable", True):
            attributions.append(FailureAttribution(
                **identity, kind=ErrorKind.TOOL_BINDING_ERROR,
                responsibility="implementation", confidence=0.9,
                message="Tool 参数绑定失败（缺参/映射错误）",
                evidence=evidence,
            ))
        if not checks.get("template_instantiable", True):
            attributions.append(FailureAttribution(
                **identity, kind=ErrorKind.TOOL_BINDING_ERROR,
                responsibility="tool", confidence=0.85,
                message="action template 无法实例化",
                evidence=evidence,
            ))
        if not checks.get("safety", True):
            attributions.append(FailureAttribution(
                **identity, kind=ErrorKind.TOOL_SAFETY_REJECTION,
                responsibility="tool", confidence=0.9,
                message="Tool 安全策略拒绝执行",
                evidence=evidence,
            ))
        if not attributions:
            attributions.append(FailureAttribution(
                **identity, kind=ErrorKind.UNKNOWN,
                responsibility="unknown", confidence=0.4,
                message=f"节点验证失败：{failure.messages[:2]}",
                evidence=evidence,
            ))
        return attributions

    @staticmethod
    def _upstream_failed(trace: TraceRecord, occurrence: str) -> str:
        identities = [str(n.get("occurrence_id") or n.get("step_id")
                          or n.get("ref") or n.get("node_ref") or "")
                      for n in trace.realized_atomic_nodes]
        if occurrence not in identities:
            return ""
        index = identities.index(occurrence)
        for realized in trace.realized_atomic_nodes[:index]:
            if not realized.get("passed", True):
                return str(realized.get("ref") or realized.get("node_ref") or "")
        return ""
