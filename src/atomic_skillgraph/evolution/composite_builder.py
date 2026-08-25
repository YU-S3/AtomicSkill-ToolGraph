"""Composite Builder（设计文档 v2.0 §36）。

从成功 Atomic Chain 构建 Composite：高层目标 + 子图 + 控制关系 + 高层验证。
不做 Composite Tool（§36.4）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.config import SystemConfig
from ..core.edge_ir import GraphEdge
from ..core.refs import SkillRef, bump_version
from ..core.skill_ir import CompositeSkill
from ..core.status import EdgeType, SkillStatus
from ..core.trace_ir import TraceRecord
from ..graph.aligner import align_composite
from ..graph.registry import SkillGraphRegistry


@dataclass
class CompositeBuildResult:
    composite: CompositeSkill | None = None
    decision: str = ""        # new | reuse | skipped
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "composite": str(self.composite.ref) if self.composite else None,
            "decision": self.decision,
            "reason": self.reason,
        }


class CompositeBuilder:
    """成功原子链 → Composite Skill。"""

    def __init__(self, registry: SkillGraphRegistry, config: SystemConfig) -> None:
        self.registry = registry
        self.config = config

    def build_or_align(self, atomic_refs: list[SkillRef],
                       trace: TraceRecord, *,
                       segments: list[dict[str, Any]] | None = None,
                       graph_proposal: dict[str, Any] | None = None) -> CompositeBuildResult:
        features = self.config.features
        if not features.enable_composite:
            return CompositeBuildResult(decision="skipped",
                                        reason="composite_disabled")
        if len(atomic_refs) < 2:
            return CompositeBuildResult(decision="skipped",
                                        reason="fewer_than_two_atomic_nodes")

        segments = list(segments or [{} for _ in atomic_refs])
        graph_proposal = dict(graph_proposal or {})
        pairs = list(zip(atomic_refs, segments))
        proposed_order = list(graph_proposal.get("ordered_phase_ids") or [])
        observed_order = [str(segment.get("phase_id") or f"phase_{index:03d}")
                          for index, segment in enumerate(segments)]
        # LLM is advisory: a reordering unsupported by the executed occurrence
        # order is rejected rather than silently changing causal evidence.
        if proposed_order == observed_order:
            by_phase = {str(segment.get("phase_id") or f"phase_{index:03d}"): (ref, segment)
                        for index, (ref, segment) in enumerate(pairs)}
            pairs = [by_phase[phase_id] for phase_id in proposed_order]
        atomic_refs = [pair[0] for pair in pairs]
        segments = [pair[1] for pair in pairs]
        node_refs = [f"{ref.logical_id}@{ref.version}" for ref in atomic_refs]
        steps = [
            {"step_id": f"step_{index:03d}", "node_ref": node_ref,
             "params": _role_params(segment.get("params") or {}, trace),
             "metadata": {"source_trace_id": trace.trace_id,
                          "phase_id": str(segment.get("phase_id") or f"phase_{index:03d}"),
                          "event_start": segment.get("event_start"),
                          "event_end": segment.get("event_end")}}
            for index, (node_ref, segment) in enumerate(zip(node_refs, segments))
        ]
        control = [
            GraphEdge(
                source=steps[index]["node_ref"], target=steps[index + 1]["node_ref"],
                type=EdgeType.NEXT, scope="composite",
                source_step=steps[index]["step_id"],
                target_step=steps[index + 1]["step_id"],
                evidence=[trace.trace_id],
            ).to_dict()
            for index in range(len(steps) - 1)
        ]
        data = _infer_data_edges(steps, self.registry, trace.trace_id)
        dependencies = _infer_dependency_edges(
            steps, self.registry, trace.trace_id, graph_proposal)
        logical_id = _composite_id(atomic_refs)
        candidate = CompositeSkill(
            ref=SkillRef(logical_id=logical_id, version="1.0.0"),
            # Composite 的公开语义由已验证 Atomic 角色链确定。LLM 的原始
            # summary 只作审计证据，不能把 mug/microwave 等执行实例写入
            # 可复用能力身份。
            summary=_summary_from_refs(atomic_refs),
            task_type_labels=[trace.task_type] if trace.task_type else [],
            graph={
                "nodes": node_refs,
                "steps": steps,
                "control": control,
                "data": data,
                "dependencies": dependencies,
                "semantic": [],
                "evolution": [],
            },
            guideline={"layer": 2, "rules": [
                "按原子节点顺序执行；每个节点执行后必须通过节点验证器。",
            ]},
            insight={"layer": 3, "sample_count": 1,
                     "common_locations": [], "common_pitfalls": [],
                     "environment_facts": [], "search_priority": [],
                     "failure_distribution": {}},
            validator={"checks": _composite_checks(atomic_refs, self.registry),
                       "target_effects": list(trace.provenance.get("target_effects") or [])},
            metadata={
                "source_trace_ids": [trace.trace_id],
                "semantic_proposals": [{
                    "source_trace_id": trace.trace_id,
                    "summary_raw": str(graph_proposal.get("summary") or "").strip(),
                    "implicit_dependencies_raw": list(
                        graph_proposal.get("implicit_dependencies") or []),
                }],
                "candidate": {"admission": "multi_trace_or_strict_replay",
                              "graph_proposal_validated": bool(graph_proposal.get("validated")),
                              "semantic_extraction_validated": bool(segments) and all(
                                  str(segment.get("extraction_method") or "")
                                  == "llm_proposal_code_validated" for segment in segments),
                              "strict_replay_passed": False},
                "statistics": {"use_count": 0, "success_count": 1,
                               "failure_count": 0, "utility": 0.5,
                               "support_count": 1},
            },
            status=SkillStatus.DRAFT,
        )

        alignment = align_composite(candidate, self.registry)
        if alignment.matched:
            existing = self.registry.get(SkillRef.parse(alignment.matched_ref))
            if existing is None:
                existing = self.registry.get_recommended(
                    SkillRef.parse(alignment.matched_ref).logical_id)
            if existing is not None:
                self._merge_evidence(existing, candidate, trace)
                min_support = max(2, int(self.config.thresholds.composite_min_support))
                support = int((existing.metadata.get("statistics") or {}).get(
                    "support_count", 0))
                strict_replay = bool((existing.metadata.get("candidate") or {}).get(
                    "strict_replay_passed"))
                if support >= min_support or strict_replay:
                    existing.status = SkillStatus.ACTIVE
                self.registry.update_runtime_state(existing)
                # 对齐可能命中历史版本；让推荐指针跟随支持度最高、仍可用的
                # canonical artifact，避免 pointer 停在被覆盖/低支持版本。
                self.registry.recommend(existing.ref)
                return CompositeBuildResult(composite=existing, decision="reuse",
                                            reason="same_atomic_chain")
        # 同一逻辑链下确有不同的已验证 DAG 时分配新版本，绝不再次写入
        # logical_id@1.0.0。相同 DAG 已在 align_composite 中合并。
        existing_versions = self.registry.list_versions(logical_id)
        if existing_versions:
            latest = self.registry.get_latest(logical_id)
            base = latest.ref.version if latest is not None else existing_versions[-1]
            version = bump_version(base, "minor")
            while version in existing_versions:
                version = bump_version(version, "minor")
            candidate.ref = SkillRef(logical_id=logical_id, version=version)
            candidate.metadata["version_reason"] = "genuinely_distinct_occurrence_dag"
        self.registry.register(candidate)
        return CompositeBuildResult(composite=candidate, decision="new",
                                    reason="registered")

    @staticmethod
    def _merge_evidence(existing: CompositeSkill, incoming: CompositeSkill,
                        trace: TraceRecord) -> None:
        labels = list(existing.task_type_labels)
        for label in incoming.task_type_labels:
            if label not in labels:
                labels.append(label)
        stats = dict(existing.metadata.get("statistics") or {})
        sources = list(existing.metadata.get("source_trace_ids") or [])
        independent = trace.trace_id not in sources
        if independent:
            sources.append(trace.trace_id)
            stats["support_count"] = int(stats.get("support_count", 0)) + 1
            stats["success_count"] = int(stats.get("success_count", 0)) + 1
        existing.task_type_labels = labels
        existing.metadata["statistics"] = stats
        existing.metadata["source_trace_ids"] = sources
        proposals = list(existing.metadata.get("semantic_proposals") or [])
        for proposal in incoming.metadata.get("semantic_proposals") or []:
            source = str(proposal.get("source_trace_id") or "")
            if source and not any(str(item.get("source_trace_id") or "") == source
                                  for item in proposals):
                proposals.append(dict(proposal))
        existing.metadata["semantic_proposals"] = proposals
        if independent:
            existing.insight["sample_count"] = int(existing.insight.get("sample_count", 0)) + 1


def _composite_id(atomic_refs: list[SkillRef]) -> str:
    """composite.{benchmark}.{a}-{b}-{c}：由原子链构成，不含 task_type（跨类型复用）。"""
    parts = [ref.logical_id.split(".", 1)[-1] for ref in atomic_refs]
    benchmark = ""
    for ref in atomic_refs:
        if "." in ref.logical_id:
            benchmark = ref.logical_id.split(".", 1)[0]
            break
    prefix = f"composite.{benchmark}" if benchmark else "composite"
    return f"{prefix}.{'-'.join(parts)}"[:150]


def _summary_from_refs(atomic_refs: list[SkillRef]) -> str:
    """Generate an instance-free public description from Atomic identities."""
    names = []
    for ref in atomic_refs:
        leaf = ref.logical_id.rsplit(".", 1)[-1]
        names.append(leaf.replace("-", " ").replace("_", " "))
    return f"复合能力：{' → '.join(names)}"


def _composite_checks(atomic_refs: list[SkillRef],
                      registry: SkillGraphRegistry) -> list[str]:
    # 中间态效果不应被误写成 Composite 最终态不变量。例如 place 完成后
    # agent.holds 必然为假，但 acquire 节点仍然是正确完成的。
    transient = {"agent.holds", "container.open", "location.checked",
                 "object.is_accessible", "object.exists"}
    checks: list[str] = []
    for ref in atomic_refs:
        atomic = registry.get(ref) or registry.get_recommended(ref.logical_id)
        if atomic is None:
            continue
        for effect in atomic.effects:
            name = str(effect.get("predicate", ""))
            if name and name not in transient and name not in checks:
                checks.append(name)
    return checks


def _infer_data_edges(steps: list[dict[str, Any]], registry: SkillGraphRegistry,
                      trace_id: str) -> list[dict[str, Any]]:
    """Infer nearest-producer output→input mappings across the occurrence DAG."""
    edges: list[dict[str, Any]] = []
    for target_index in range(1, len(steps)):
        target_step = steps[target_index]
        try:
            target_ref = SkillRef.parse(target_step["node_ref"])
        except ValueError:
            continue
        target = registry.get(target_ref) or registry.get_recommended(target_ref.logical_id)
        if target is None:
            continue
        inputs = list(getattr(target, "inputs", None) or [])
        for input_spec in inputs:
            for source_index in range(target_index - 1, -1, -1):
                source_step = steps[source_index]
                try:
                    source_ref = SkillRef.parse(source_step["node_ref"])
                except ValueError:
                    continue
                source = registry.get(source_ref) or registry.get_recommended(source_ref.logical_id)
                if source is None:
                    continue
                matched = False
                for output in list(getattr(source, "outputs", None) or []):
                    same_name = output.get("name") == input_spec.get("name")
                    same_type = (output.get("semantic_type")
                                 and output.get("semantic_type") == input_spec.get("semantic_type"))
                    if not (same_name or same_type):
                        continue
                    edges.append(GraphEdge(
                        source=source_step["node_ref"], target=target_step["node_ref"],
                        type=EdgeType.DATA_FLOW, scope="composite",
                        source_step=source_step["step_id"],
                        target_step=target_step["step_id"],
                        mapping={
                            "source_output": str(output.get("name")),
                            "target_input": str(input_spec.get("name")),
                            "schema": str(input_spec.get("semantic_type") or "value"),
                            "transform": "identity",
                        },
                        evidence=[trace_id],
                    ).to_dict())
                    matched = True
                    break
                if matched:
                    break
    return edges


def _infer_dependency_edges(steps: list[dict[str, Any]], registry: SkillGraphRegistry,
                            trace_id: str,
                            graph_proposal: dict[str, Any]) -> list[dict[str, Any]]:
    """Build causal edges from the nearest Effect producer plus gated LLM hints."""
    edges: list[dict[str, Any]] = []
    phase_to_step = {str(step.get("metadata", {}).get("phase_id")): step for step in steps}
    for target_index, target_step in enumerate(steps):
        target_ref = SkillRef.parse(target_step["node_ref"])
        target = registry.get(target_ref) or registry.get_recommended(target_ref.logical_id)
        for precondition in list(getattr(target, "preconditions", []) or []):
            predicate = str(precondition.get("predicate") or "")
            for source_index in range(target_index - 1, -1, -1):
                source_step = steps[source_index]
                source_ref = SkillRef.parse(source_step["node_ref"])
                source = registry.get(source_ref) or registry.get_recommended(source_ref.logical_id)
                if any(str(effect.get("predicate") or "") == predicate
                       for effect in list(getattr(source, "effects", []) or [])):
                    edges.append(GraphEdge(
                        source=source_step["node_ref"], target=target_step["node_ref"],
                        type=EdgeType.REQUIRES_SKILL, scope="composite",
                        source_step=source_step["step_id"], target_step=target_step["step_id"],
                        evidence=[trace_id], metadata={"predicate": predicate,
                                                      "origin": "effect_precondition"},
                    ).to_dict())
                    break
    for hint in graph_proposal.get("implicit_dependencies") or []:
        source_step = phase_to_step.get(str(hint.get("source_phase_id") or ""))
        target_step = phase_to_step.get(str(hint.get("target_phase_id") or ""))
        if source_step is None or target_step is None:
            continue
        if steps.index(source_step) >= steps.index(target_step):
            continue
        edges.append(GraphEdge(
            source=source_step["node_ref"], target=target_step["node_ref"],
            type=EdgeType.REQUIRES_SKILL, scope="composite",
            source_step=source_step["step_id"], target_step=target_step["step_id"],
            evidence=[trace_id], metadata={
                "origin": "llm_semantic_proposal",
                "reason": "LLM 提议存在非显式依赖；执行结构仍由代码验证",
                "llm_reason_raw": str(hint.get("reason") or ""),
            },
        ).to_dict())
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for edge in edges:
        unique[(str(edge.get("source_step")), str(edge.get("target_step")),
                str((edge.get("metadata") or {}).get("predicate") or "semantic"))] = edge
    return list(unique.values())


def _role_params(params: dict[str, Any], trace: TraceRecord) -> dict[str, Any]:
    """Persist semantic task roles, never source-instance literals."""
    from ..core.predicates import normalize_value
    known = dict(trace.provenance.get("params") or {})
    semantic = dict(trace.provenance.get("semantic_params") or {})
    result: dict[str, Any] = {}
    compatible_roles = {
        "object": {"object", "object1", "object2"},
        "object_location": {"object_location", "source_location",
                            "source_receptacle"},
        "target_location": {"target_location", "target_receptacle",
                            "destination", "destination_location"},
        "heating_station": {"heating_station"},
        "cleaning_station": {"cleaning_station"},
        "cooling_station": {"cooling_station"},
    }
    for key, value in params.items():
        value_norm = normalize_value(value)
        matched_role = ""
        allowed = compatible_roles.get(str(key), {str(key)})
        # Exact identity always wins.  In particular cabinet_2 must not bind to
        # target_location=cabinet_1 merely because both share the cabinet class.
        for role, role_value in known.items():
            if str(role) not in allowed:
                continue
            role_norm = normalize_value(role_value)
            if value_norm == role_norm:
                matched_role = str(role)
                break
        # semantic_params 保存目标语义类型（cabinet），params 保存本题可执行
        # 实例（cabinet_1）。Occurrence 可以是 cabinet_5；角色映射应按前者的
        # family 对齐，但仍严格限制 source/target 等兼容角色，不能串位。
        if not matched_role:
            for role, role_value in semantic.items():
                if str(role) not in allowed:
                    continue
                role_norm = normalize_value(role_value)
                if (value_norm and role_norm
                        and __import__("re").sub(r"_\d+$", "", value_norm)
                        == __import__("re").sub(r"_\d+$", "", role_norm)):
                    matched_role = str(role)
                    break
        # 兼容没有 semantic_params 的旧 trace：只允许泛化的 executable role
        # 吸收其具体 occurrence，两个具体位置之间绝不按 family 强行绑定。
        if not matched_role:
            for role, role_value in known.items():
                if str(role) not in allowed:
                    continue
                role_norm = normalize_value(role_value)
                role_is_generic = not __import__("re").search(r"_\d+$", role_norm)
                if (value_norm and role_norm and role_is_generic
                        and __import__("re").sub(r"_\d+$", "", value_norm)
                        == role_norm):
                    matched_role = str(role)
                    break
        if matched_role:
            result[str(key)] = f"$task.{matched_role}"
    return result
