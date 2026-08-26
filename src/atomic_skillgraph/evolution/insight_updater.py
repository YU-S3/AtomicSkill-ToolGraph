"""Composite Layer-3 Insight Updater（设计文档 v2.0 §10.3、§36.3）。

Evidence is grouped by the validated Composite occurrence contract, never by
an official benchmark task label.  Entity/role statistics are learned from
persisted structured actions rather than a benchmark receptacle whitelist.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from ..core.config import SystemConfig
from ..core.refs import SkillRef, bump_version
from ..core.status import EdgeType
from ..persistence import TraceStore
from ..graph.registry import SkillGraphRegistry

class InsightUpdater:
    """多轨迹 insight 聚合（规则版；可选 LLM 摘要，默认关闭）。"""

    def __init__(self, registry: SkillGraphRegistry, trace_store: TraceStore,
                 config: SystemConfig) -> None:
        self.registry = registry
        self.trace_store = trace_store
        self.config = config

    def update_if_ready(self, composite_ref: SkillRef) -> dict[str, Any] | None:
        features = self.config.features
        if not features.enable_composite or not features.enable_layer3_insight:
            return None
        composite = self.registry.get(composite_ref) or self.registry.get_recommended(composite_ref.logical_id)
        if composite is None:
            return None
        source_ids = list((composite.metadata or {}).get("source_trace_ids") or [])
        traces = [trace for trace_id in source_ids
                  if (trace := self.trace_store.load(str(trace_id))) is not None
                  and trace.success]
        if len(traces) < self.config.thresholds.insight_min_samples:
            return {"updated": False, "reason": "insufficient_samples",
                    "sample_count": len(traces)}
        insight = self._aggregate(traces)
        old_insight = dict(composite.insight or {})
        # sample_count is accumulated evidence, not a semantic graph change.
        # Do not manufacture one Composite version per successful episode when
        # the reusable locations/pitfalls/facts remain identical.
        semantic_old = {key: value for key, value in old_insight.items()
                        if key != "sample_count"}
        semantic_new = {key: value for key, value in insight.items()
                        if key != "sample_count"}
        if semantic_old == semantic_new:
            composite.insight = insight
            self.registry.update_runtime_state(composite)
            self.registry.recommend(composite.ref)
            return {"updated": False, "reason": "evidence_only",
                    "sample_count": len(traces), "version": composite.ref.version}
        old_ref = composite.ref
        composite.insight = insight
        # insight（图内容的一部分）变化 → 新版本（§38.1）
        latest = self.registry.get_latest(composite.ref.logical_id) or composite
        version = bump_version(latest.ref.version, "minor")
        existing_versions = set(self.registry.list_versions(composite.ref.logical_id))
        while version in existing_versions:
            version = bump_version(version, "minor")
        composite.ref = SkillRef(composite.ref.logical_id, version)
        composite.metadata = dict(composite.metadata or {})
        composite.metadata["version_reason"] = "layer3_insight_update"
        self.registry.register(composite)
        self.registry.add_edge(str(composite.ref), str(old_ref),
                               EdgeType.SUPERSEDES,
                               evidence=[trace.trace_id for trace in traces],
                               metadata={"reason": "layer3_insight_update"})
        return {"updated": True, "sample_count": len(traces),
                "insight": insight, "new_version": composite.ref.version}

    @staticmethod
    def _aggregate(traces: list) -> dict[str, Any]:
        location_counter: Counter = Counter()
        role_counter: Counter = Counter()
        pitfall_counter: Counter = Counter()
        for trace in traces:
            for action in trace.actions:
                payload = action.to_dict() if hasattr(action, "to_dict") else dict(action)
                for role, value in (payload.get("params") or {}).items():
                    family = _entity_family(value)
                    if not family:
                        continue
                    role_counter[(str(role), family)] += 1
                    if any(token in str(role).lower()
                           for token in ("location", "place", "station", "source", "destination")):
                        location_counter[family] += 1
            # 常见 pitfall：重复动作（机械重复 = 搜索低效信号）
            action_names = [a.name for a in trace.actions]
            for i in range(len(action_names) - 1):
                if action_names[i] == action_names[i + 1]:
                    pitfall_counter[action_names[i]] += 1

        common_locations = [name for name, count in location_counter.most_common(5) if count >= 1]
        search_priority = [name for name, _count in location_counter.most_common(8)]
        environment_facts = [
            f"role {role} repeatedly binds entity family {family}"
            for (role, family), count in role_counter.most_common(5) if count >= 2
        ]
        common_pitfalls = []
        if pitfall_counter:
            top_action = pitfall_counter.most_common(1)[0][0]
            common_pitfalls.append(f"重复执行 {top_action} 动作会增加步数开销，先确认目标状态再行动")
        return {
            "layer": 3,
            "sample_count": len(traces),
            "common_locations": common_locations,
            "search_priority": search_priority,
            "common_pitfalls": common_pitfalls,
            "environment_facts": environment_facts,
            "failure_distribution": {},
        }


def _entity_family(value: Any) -> str:
    import re
    normalized = re.sub(r"\s+", "_", str(value or "").strip().lower())
    return re.sub(r"_\d+$", "", normalized)
