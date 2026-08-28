"""失败轨迹进化管线（设计文档 v2.0 §45）。

失败可产生：Failure Mode / guideline 更新候选 / validator 更新 / split candidate /
tool update candidate / tool specialize candidate / add_tool_test /
implementation fallback 调整。

失败不能：直接生成 recommended executable tool、直接把新代码设为 active/preferred、
绕过 admission、绕过成功 replay。

固定原则：Failure proposes; successful replay admits.（§33.2）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.config import SystemConfig
from ..core.binding_ir import is_concrete_binding
from ..core.refs import SkillRef, ToolRef
from ..core.status import ErrorKind
from ..core.trace_ir import TraceRecord
from ..persistence import ProposalStore
from ..tools.registry import ToolRegistry
from ..validation.failure_localizer import FailureAttribution, FailureLocalizer
from ..graph.registry import SkillGraphRegistry
from ..runtime.plan_validator import semantic_required_slots


@dataclass
class FailureProcessingResult:
    attributions: list[dict[str, Any]] = field(default_factory=list)
    proposals: list[dict[str, Any]] = field(default_factory=list)
    evidence_updates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attributions": self.attributions,
            "proposals": self.proposals,
            "evidence_updates": self.evidence_updates,
        }


class FailureProcessor:
    """失败轨迹 → 归因 + 提案（任何 executable repair 必须成功 replay 后才能入 candidate）。"""

    def __init__(self, registry: SkillGraphRegistry, tool_registry: ToolRegistry,
                 config: SystemConfig) -> None:
        self.registry = registry
        self.tool_registry = tool_registry
        self.config = config
        self.localizer = FailureLocalizer()
        self.proposals = ProposalStore(config.data_dir)

    def process_failure(self, trace: TraceRecord) -> FailureProcessingResult:
        result = FailureProcessingResult()
        attributions = self.localizer.localize(trace, self.registry)
        result.attributions = [a.to_dict() for a in attributions]

        # 1. 记录失败证据（Skill 与 Tool 分开记统计，§39.3）
        self._record_failure_evidence(trace, result)

        # 2. 按归因类型生成提案（shadow，不激活）
        for attribution in attributions:
            proposal = self._propose(trace, attribution)
            if proposal is not None:
                result.proposals.append(proposal)
        return result

    # ------------------------------------------------------------------
    def _record_failure_evidence(self, trace: TraceRecord,
                                 result: FailureProcessingResult) -> None:
        # Tool failure evidence is exact-attempt evidence: selected/seeded
        # references and rescued historical attempts are not counted here.
        failed_direct_refs = {
            str(tool_ref)
            for node in trace.realized_atomic_nodes
            if not bool(node.get("passed"))
            for attempt in (node.get("attempts") or [])
            if bool(attempt.get("started"))
            and str(attempt.get("mode") or "") == "direct"
            and not bool(attempt.get("passed"))
            for tool_ref in (attempt.get("tool_refs") or [])
            if tool_ref
        }
        for tool_ref_text in sorted(failed_direct_refs):
            try:
                ref = ToolRef.parse(tool_ref_text)
            except ValueError:
                continue
            tool = self.tool_registry.get(ref)
            if tool is None:
                continue
            # Runtime feedback already records call/failure/consecutive counts.
            # Failure processing only attaches trace provenance; do not double count.
            tool.statistics["last_failure_trace_id"] = trace.trace_id
            self.tool_registry.update_tool(tool)
            result.evidence_updates.append(f"tool_failure_evidence:{ref}")

        # Abstract Skill 失败证据
        for node in trace.realized_atomic_nodes:
            node_ref = str(node.get("ref", ""))
            if (not node_ref or node.get("passed", True)
                    or not bool(node.get("attempt_started"))
                    or int(node.get("executed_action_count") or 0) <= 0):
                continue
            attempts = list(node.get("attempts") or [])
            if attempts and all(str(item.get("failure_stage") or "") in {
                    "planning", "budget"} for item in attempts):
                continue
            try:
                ref = SkillRef.parse(node_ref)
            except ValueError:
                continue
            atomic = self.registry.get(ref)
            if atomic is None:
                continue
            required = semantic_required_slots(
                list(getattr(atomic, "effects", []) or []))
            if not all(is_concrete_binding(
                    dict(node.get("params") or {}).get(slot))
                       for slot in required):
                continue
            stats = dict(atomic.metadata.get("statistics") or {})
            stats["failure_count"] = int(stats.get("failure_count", 0)) + 1
            atomic.metadata["statistics"] = stats
            # The complete failed occurrence is private replay evidence.  A
            # portable Skill records only a categorical failure mode and an
            # opaque reference; validator messages often contain concrete
            # objects, paths, observations, or other task data.
            failure_ref = self.registry.evidence_store.put(
                "atomic_failure",
                {"node": node, "failure_type": trace.failure_type},
                trace_id=trace.trace_id,
            )
            # 失败模式记录（§33.1 add_failure_mode）
            failure_modes = list(atomic.failure_modes)
            mode_name = _failure_mode_of(node)
            matching = next(
                (item for item in failure_modes
                 if str(item.get("name") or "") == mode_name), None)
            if mode_name and matching is None:
                failure_modes.append({
                    "name": mode_name,
                    "source_trace_ids": [trace.trace_id],
                    "evidence_refs": [failure_ref],
                    "occurrence_count": 1,
                })
                atomic.failure_modes = failure_modes
            elif mode_name and matching is not None:
                # Migrate the former singular representation in memory and
                # retain every repeated occurrence.  A categorical failure
                # mode may recur, but each supporting trace must remain
                # reachable through its own private evidence pointer.
                trace_ids = list(matching.get("source_trace_ids") or [])
                legacy_trace = str(matching.pop("source_trace_id", "") or "")
                if legacy_trace and legacy_trace not in trace_ids:
                    trace_ids.append(legacy_trace)
                if trace.trace_id and trace.trace_id not in trace_ids:
                    trace_ids.append(trace.trace_id)
                refs = list(matching.get("evidence_refs") or [])
                legacy_ref = matching.pop("evidence_ref", None)
                if isinstance(legacy_ref, dict) and legacy_ref not in refs:
                    refs.append(legacy_ref)
                if failure_ref not in refs:
                    refs.append(failure_ref)
                matching["source_trace_ids"] = trace_ids
                matching["evidence_refs"] = refs
                matching["occurrence_count"] = int(
                    matching.get("occurrence_count") or 1) + 1
                atomic.failure_modes = failure_modes
            self.registry.update_runtime_state(atomic)
            result.evidence_updates.append(f"atomic_failure_evidence:{ref.logical_id}")

    # ------------------------------------------------------------------
    def _propose(self, trace: TraceRecord,
                 attribution: FailureAttribution) -> dict[str, Any] | None:
        kind = attribution.kind
        common = {"attribution": attribution.to_dict(),
                  "tool_refs": list(trace.tool_refs),
                  "step_id": attribution.step_id,
                  "occurrence_id": attribution.occurrence_id}
        if kind == ErrorKind.TOOL_EXECUTION_ERROR:
            return self.proposals.add(
                "tool_update", trace.trace_id, attribution.node_ref,
                attribution.message,
                payload={**common,
                         "note": "新 executable body 必须经成功 replay + admission 才能成为 candidate"})
        if kind == ErrorKind.TOOL_BINDING_ERROR:
            return self.proposals.add(
                "add_tool_test", trace.trace_id, attribution.node_ref,
                attribution.message,
                payload=common)
        if kind == ErrorKind.EFFECT_VIOLATION:
            return self.proposals.add(
                "contract_revision", trace.trace_id, attribution.node_ref,
                attribution.message,
                payload=common)
        if kind == ErrorKind.PRECONDITION_VIOLATION:
            return self.proposals.add(
                "contract_revision", trace.trace_id, attribution.node_ref,
                attribution.message,
                payload=common)
        if kind == ErrorKind.CONTROL_FLOW_ERROR or kind == ErrorKind.DATA_FLOW_ERROR:
            return self.proposals.add(
                "composite_revision", trace.trace_id, attribution.node_ref,
                attribution.message,
                payload=common)
        return None


def _failure_mode_of(node: dict[str, Any]) -> str:
    """Return an instance-free categorical failure identity."""
    import re
    attempts = [item for item in (node.get("attempts") or [])
                if item.get("started") and not item.get("passed")]
    stage = next((str(item.get("failure_stage") or "")
                  for item in attempts if item.get("failure_stage")), "")
    mode = next((str(item.get("mode") or "")
                 for item in attempts if item.get("mode")), "")
    status = str(node.get("execution_status") or "")

    def token(value: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
        # Status/stage/mode are framework enums.  Reject anything that looks
        # like a concrete environment instance rather than persisting it.
        return "" if re.search(r"(?:^|_)\w+_\d+(?:_|$)", normalized) else normalized

    parts = [item for item in (token(stage), token(mode), token(status)) if item]
    return ":".join(parts[:3]) or "effect_validation_failed"
