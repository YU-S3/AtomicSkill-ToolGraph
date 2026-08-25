"""Tool 生命周期治理（设计文档 v2.0 §29、§39、§40）。

candidate → active（额外成功证据）
active   → preferred（同功能类中最优）
…        → suppressed（有害证据）→ retired（长期无效/被取代）
shadow   → admission_pending（修复后重新 admission）
rollback = 恢复推荐指针（历史 artifact 不被覆盖）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.config import Thresholds
from ..core.refs import ToolRef
from ..core.status import EdgeType, ToolLifecycle
from ..core.tool_ir import lifecycle_rank
from .registry import ToolRegistry


@dataclass
class LifecycleEvent:
    tool_ref: str = ""
    action: str = ""          # promote_active | promote_preferred | suppress | retire | rollback | revalidate
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"tool_ref": self.tool_ref, "action": self.action, "reason": self.reason}


class ToolLifecycleManager:
    """按历史统计驱动的生命周期维护器（不训练 Router，规则 + 阈值，§39.1）。"""

    def __init__(self, registry: ToolRegistry, thresholds: Thresholds | None = None) -> None:
        self.registry = registry
        self.thresholds = thresholds or Thresholds()

    # ------------------------------------------------------------------
    def review(self, skill_graph=None) -> list[LifecycleEvent]:
        """一轮周期性审查：candidate→active、preferred 选择、失败审查。"""
        events: list[LifecycleEvent] = []
        events.extend(self._promote_candidates())
        events.extend(self._select_preferred())
        events.extend(self._review_failures())
        if skill_graph is not None:
            events.extend(self._retire_superseded(skill_graph))
        return events

    def _promote_candidates(self) -> list[LifecycleEvent]:
        events: list[LifecycleEvent] = []
        for tool in self.registry.list_all(statuses={ToolLifecycle.CANDIDATE}):
            stats = tool.statistics
            support = int(stats.get("support_count", 0))
            success = int(stats.get("success_count", 0))
            admission_success = int(stats.get("admission_replay_success_count", 0))
            if (support >= self.thresholds.candidate_min_support
                    and (success >= 1 or admission_success >= 1)):
                try:
                    self.registry.set_status(tool.ref, ToolLifecycle.ACTIVE)
                    events.append(LifecycleEvent(str(tool.ref), "promote_active",
                                                 f"support={support}, success={success}"))
                except ValueError:
                    continue
        return events

    def _select_preferred(self) -> list[LifecycleEvent]:
        """同 tool_id 家族 + 跨家族同 entry_point：utility 最优 → preferred（§39.2）。"""
        events: list[LifecycleEvent] = []
        classes: dict[str, list] = {}
        for tool in self.registry.list_all(
                statuses={ToolLifecycle.CANDIDATE, ToolLifecycle.ACTIVE, ToolLifecycle.PREFERRED}):
            key = f"{tool.artifact_kind.value}:{tool.entry_point()}"
            classes.setdefault(key, []).append(tool)
        for key, tools in classes.items():
            if len(tools) < 2:
                continue
            ranked = sorted(tools, key=lambda t: (t.utility(), lifecycle_rank(t.status)), reverse=True)
            best = ranked[0]
            second = ranked[1] if len(ranked) > 1 else None
            margin = best.utility() - (second.utility() if second else 0.0)
            if best.status == ToolLifecycle.PREFERRED:
                continue
            if (best.status == ToolLifecycle.ACTIVE and margin >= self.thresholds.preferred_margin):
                try:
                    self.registry.set_status(best.ref, ToolLifecycle.PREFERRED)
                    events.append(LifecycleEvent(str(best.ref), "promote_preferred",
                                                 f"utility={best.utility():.3f} (class={key})"))
                except ValueError:
                    continue
        return events

    def _review_failures(self) -> list[LifecycleEvent]:
        events: list[LifecycleEvent] = []
        for tool in self.registry.list_all(
                statuses={ToolLifecycle.CANDIDATE, ToolLifecycle.ACTIVE, ToolLifecycle.PREFERRED}):
            stats = tool.statistics
            failures = int(stats.get("failure_count", 0))
            total = int(stats.get("call_count", 0))
            consecutive = int(stats.get("consecutive_failures", 0))
            if consecutive >= self.thresholds.direct_max_consecutive_failures and total >= 3:
                try:
                    self.registry.set_status(tool.ref, ToolLifecycle.SUPPRESSED)
                    events.append(LifecycleEvent(str(tool.ref), "suppress",
                                                 f"consecutive_failures={consecutive}"))
                except ValueError:
                    continue
                continue
            if total >= 5 and failures / max(total, 1) >= 0.8:
                try:
                    self.registry.set_status(tool.ref, ToolLifecycle.SUPPRESSED)
                    events.append(LifecycleEvent(str(tool.ref), "suppress",
                                                 f"failure_rate={failures / total:.2f}"))
                except ValueError:
                    continue
        # 长期无效 → retired
        for tool in self.registry.list_all(statuses={ToolLifecycle.SUPPRESSED}):
            if tool.utility() < self.thresholds.retirement_utility:
                try:
                    self.registry.set_status(tool.ref, ToolLifecycle.RETIRED)
                    events.append(LifecycleEvent(str(tool.ref), "retire",
                                                 f"utility={tool.utility():.3f}"))
                except ValueError:
                    continue
        return events

    def _retire_superseded(self, skill_graph) -> list[LifecycleEvent]:
        events: list[LifecycleEvent] = []
        for edge in skill_graph.iter_edges():
            if edge.get("type") != EdgeType.SUPERSEDES.value:
                continue
            try:
                old = ToolRef.parse(edge["source"])
            except ValueError:
                continue
            tool = self.registry.get(old)
            if tool is not None and tool.status not in (ToolLifecycle.RETIRED, ToolLifecycle.SHADOW):
                try:
                    self.registry.set_status(old, ToolLifecycle.RETIRED)
                    events.append(LifecycleEvent(str(old), "retire", "superseded"))
                except ValueError:
                    continue
        return events

    # ------------------------------------------------------------------
    def revalidate(self, ref: ToolRef) -> LifecycleEvent:
        """suppressed/shadow Tool 经新成功 replay 证据后重新进入 admission 通道。"""
        tool = self.registry.get(ref)
        if tool is None:
            return LifecycleEvent(str(ref), "revalidate", "tool_missing")
        if tool.status == ToolLifecycle.SUPPRESSED:
            try:
                self.registry.set_status(ref, ToolLifecycle.CANDIDATE)
                return LifecycleEvent(str(ref), "revalidate", "suppressed->candidate")
            except ValueError:
                return LifecycleEvent(str(ref), "revalidate", "transition_rejected")
        if tool.status == ToolLifecycle.SHADOW:
            try:
                self.registry.set_status(ref, ToolLifecycle.ADMISSION_PENDING)
                return LifecycleEvent(str(ref), "revalidate", "shadow->admission_pending")
            except ValueError:
                return LifecycleEvent(str(ref), "revalidate", "transition_rejected")
        return LifecycleEvent(str(ref), "revalidate", "status_unchanged")

    def rollback(self, tool_id: str, version: str) -> LifecycleEvent:
        ref = self.registry.rollback(tool_id, version)
        return LifecycleEvent(str(ref), "rollback", f"recommended_pointer={version}")
