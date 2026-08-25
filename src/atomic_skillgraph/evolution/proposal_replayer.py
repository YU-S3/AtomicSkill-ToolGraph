"""Consume failure proposals only when later successful evidence can replay them."""

from __future__ import annotations

from typing import Any

from ..core.refs import SkillRef
from ..core.trace_ir import TraceRecord
from ..persistence import ProposalStore


class ProposalReplayer:
    """Evidence gate for the rule: failure proposes; success admits.

    Contract/composite proposals are evidence proposals, not executable patches.
    They are resolved only by a later successful trace covering the same node or
    task type. Executable Tool updates still have to pass Tool admission separately.
    """

    def __init__(self, data_dir) -> None:
        self.store = ProposalStore(data_dir)

    def consume_success(self, trace: TraceRecord,
                        learned_refs: list[str] | None = None) -> list[dict[str, Any]]:
        if not trace.success:
            return []
        covered = {_logical(str(node.get("ref") or node.get("node_ref") or ""))
                   for node in trace.realized_atomic_nodes}
        covered.update(_logical(ref) for ref in (learned_refs or []))
        covered.discard("")
        events: list[dict[str, Any]] = []
        for proposal in self.store.pending():
            payload = dict(proposal.get("payload") or {})
            same_task_type = payload.get("task_type") == trace.task_type
            target = _logical(str(proposal.get("target_ref") or ""))
            tool_overlap = bool(set(payload.get("tool_refs") or []) & set(trace.tool_refs))
            matched = target in covered or tool_overlap or (
                proposal.get("kind") == "composite_revision" and same_task_type)
            if not matched:
                continue
            result = {
                "successful_trace_id": trace.trace_id,
                "covered_refs": sorted(covered),
                "tool_overlap": tool_overlap,
                "decision": "evidence_replayed",
                "note": "Executable changes require an independently admitted Tool version.",
            }
            self.store.mark(proposal["proposal_id"], "replayed", result)
            events.append({"proposal_id": proposal["proposal_id"], **result})
        return events


def _logical(ref_text: str) -> str:
    if not ref_text:
        return ""
    try:
        return SkillRef.parse(ref_text).logical_id
    except ValueError:
        text = ref_text.removeprefix("skill://")
        return text.rsplit("@", 1)[0]
