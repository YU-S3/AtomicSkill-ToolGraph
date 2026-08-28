"""Composite Builder（设计文档 v2.0 §36）。

从成功 Atomic Chain 构建 Composite：高层目标 + 子图 + 控制关系 + 高层验证。
不做 Composite Tool（§36.4）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..core.config import SystemConfig
from ..core.edge_ir import GraphEdge
from ..core.binding_ir import binding_slot_name
from ..core.refs import SkillRef, bump_version, content_hash
from ..core.skill_ir import CompositeSkill
from ..core.status import EdgeType, SkillStatus
from ..core.trace_ir import TraceRecord
from ..graph.aligner import align_composite
from ..graph.registry import SkillGraphRegistry
from .composite_lifecycle import evaluate_composite
from .composite_revision import CompositeRevisionBuilder
from .trace_graph_reconstructor import TraceGraphRevision
from ..runtime.contract_matcher import match_effect_contract
from ..runtime.output_materializer import validate_output_materializer
from ..atomicizer.effect_extractor import (
    is_fully_parameterized_predicate,
    parameterize_predicates,
)
from ..persistence_guard import validate_long_term_asset


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
        self.revision_builder = CompositeRevisionBuilder(registry)

    def build_or_align(self, atomic_refs: list[SkillRef],
                       trace: TraceRecord, *,
                       segments: list[dict[str, Any]] | None = None,
                       graph_proposal: dict[str, Any] | None = None,
                       revision: TraceGraphRevision | None = None
                       ) -> CompositeBuildResult:
        features = self.config.features
        if not features.enable_composite:
            return CompositeBuildResult(decision="skipped",
                                        reason="composite_disabled")
        if len(atomic_refs) < 2:
            return CompositeBuildResult(decision="skipped",
                                        reason="fewer_than_two_atomic_nodes")

        provided_segments = segments is not None
        segments = list(segments or [{} for _ in atomic_refs])
        if provided_segments and len(segments) != len(atomic_refs):
            return CompositeBuildResult(
                decision="skipped",
                reason=("occurrence_segment_count_mismatch:"
                        f"refs={len(atomic_refs)}:segments={len(segments)}"))
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
        target_effects, target_errors = _portable_target_effects(trace)
        if target_errors:
            return CompositeBuildResult(
                decision="skipped",
                reason="target_contract_not_portable:" + "|".join(target_errors))
        node_refs = [f"{ref.logical_id}@{ref.version}" for ref in atomic_refs]
        occurrence_params = _role_params_for_segments(segments, trace)
        occurrence_evidence = [
            self.registry.evidence_store.put(
                "composite_occurrence",
                {
                    "params": dict(segment.get("params") or {}),
                    "before": dict(segment.get("before") or {}),
                    "after": dict(segment.get("after") or {}),
                    "effect": [dict(item) for item in
                               (segment.get("effect") or [])
                               if isinstance(item, dict)],
                    "negative_effect": [dict(item) for item in
                                        (segment.get("negative_effect") or [])
                                        if isinstance(item, dict)],
                },
                trace_id=trace.trace_id,
                event_start=_optional_int(segment.get("event_start")),
                event_end=_optional_int(segment.get("event_end")),
            )
            for segment in segments
        ]
        source_phase_ids = [
            str(segment.get("phase_id") or f"phase_{index:03d}")
            for index, segment in enumerate(segments)
        ]
        steps = [
            {"step_id": f"step_{index:03d}", "node_ref": node_ref,
             "params": occurrence_params[index],
             "metadata": {"source_trace_id": trace.trace_id,
                          # The Extractor's raw phase id is occurrence evidence
                          # and may itself contain an entity instance.  The
                          # reusable graph uses a local canonical id.
                          "phase_id": f"phase_{index:03d}",
                          "event_start": segment.get("event_start"),
                          "event_end": segment.get("event_end"),
                          "evidence_ref": occurrence_evidence[index]}}
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
        proposal_evidence_ref = self.registry.evidence_store.put(
            "composite_semantic_proposal", graph_proposal,
            trace_id=trace.trace_id)
        data = _infer_data_edges(
            steps, segments, occurrence_evidence,
            self.registry, trace.trace_id)
        dependencies = _infer_dependency_edges(
            steps, self.registry, trace.trace_id, graph_proposal,
            proposal_evidence_ref, source_phase_ids=source_phase_ids)
        revision_evidence_ref = (
            self.registry.evidence_store.put(
                "graph_revision", revision.to_dict(),
                trace_id=trace.trace_id)
            if revision is not None else {})
        logical_id = _composite_id(atomic_refs)
        candidate = CompositeSkill(
            ref=SkillRef(logical_id=logical_id, version="1.0.0"),
            # Composite 的公开语义由已验证 Atomic 角色链确定。LLM 的原始
            # summary 只作审计证据，不能把任何具体执行实例写入
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
            validator={"checks": _composite_checks(
                           atomic_refs, self.registry, segments),
                       "check_semantics": "terminal_effect_closure_v2",
                       "target_effects": target_effects},
            metadata={
                "source_trace_ids": [trace.trace_id],
                "independent_support_keys": [_support_key(trace)],
                "semantic_proposals": [{
                    "source_trace_id": trace.trace_id,
                    "evidence_ref": proposal_evidence_ref,
                }],
                # The canonical graph remains instance-free and immutable;
                # occurrence evidence from every aligned trace is retained in
                # this append-only side index instead of replacing graph.steps.
                "occurrence_evidence_refs": [{
                    "source_trace_id": trace.trace_id,
                    "support_key": _support_key(trace),
                    "steps": [
                        {"step_id": step["step_id"],
                         "evidence_ref": occurrence_evidence[index]}
                        for index, step in enumerate(steps)
                    ],
                }],
                "candidate": {"admission": "multi_trace",
                              "graph_proposal_validated": bool(graph_proposal.get("validated")),
                              "semantic_extraction_validated": bool(segments) and all(
                                  str(segment.get("extraction_method") or "")
                                  == "llm_proposal_code_validated" for segment in segments)},
                "statistics": {"use_count": 0, "success_count": 1,
                               "failure_count": 0, "utility": 0.5,
                               "support_count": 1},
                "source_graph_revision_ref": revision_evidence_ref,
            },
            status=SkillStatus.DRAFT,
        )
        decision = evaluate_composite(
            candidate, self.registry,
            min_support=self.config.thresholds.composite_min_support)
        candidate.status = decision.status
        candidate.metadata["candidate"]["lifecycle_reason"] = decision.reason

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
                lifecycle = evaluate_composite(
                    existing, self.registry, min_support=min_support)
                existing.status = lifecycle.status
                existing.metadata.setdefault("candidate", {})[
                    "lifecycle_reason"] = lifecycle.reason
                self.registry.update_runtime_state(existing)
                self.revision_builder.apply(
                    existing, revision, trace_id=trace.trace_id)
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
        self.revision_builder.apply(
            candidate, revision, trace_id=trace.trace_id)
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
        support_keys = list(existing.metadata.get("independent_support_keys") or [])
        key = _support_key(trace)
        independent = key not in support_keys
        if independent:
            sources.append(trace.trace_id)
            support_keys.append(key)
            stats["support_count"] = int(stats.get("support_count", 0)) + 1
            stats["success_count"] = int(stats.get("success_count", 0)) + 1
        existing.task_type_labels = labels
        existing.metadata["statistics"] = stats
        existing.metadata["source_trace_ids"] = sources
        existing.metadata["independent_support_keys"] = support_keys
        proposals = list(existing.metadata.get("semantic_proposals") or [])
        for proposal in incoming.metadata.get("semantic_proposals") or []:
            source = str(proposal.get("source_trace_id") or "")
            if source and not any(str(item.get("source_trace_id") or "") == source
                                  for item in proposals):
                proposals.append(dict(proposal))
        existing.metadata["semantic_proposals"] = proposals
        occurrence_groups = list(
            existing.metadata.get("occurrence_evidence_refs") or [])
        for group in incoming.metadata.get("occurrence_evidence_refs") or []:
            identity = (
                str(group.get("source_trace_id") or ""),
                str(group.get("support_key") or ""),
                tuple(
                    (str(item.get("step_id") or ""),
                     str((item.get("evidence_ref") or {}).get(
                         "evidence_ref") or ""))
                    for item in (group.get("steps") or [])
                    if isinstance(item, dict)
                ),
            )
            if not any(
                    identity == (
                        str(item.get("source_trace_id") or ""),
                        str(item.get("support_key") or ""),
                        tuple(
                            (str(step.get("step_id") or ""),
                             str((step.get("evidence_ref") or {}).get(
                                 "evidence_ref") or ""))
                            for step in (item.get("steps") or [])
                            if isinstance(step, dict)
                        ),
                    )
                    for item in occurrence_groups
                    if isinstance(item, dict)):
                occurrence_groups.append(dict(group))
        existing.metadata["occurrence_evidence_refs"] = occurrence_groups
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


def _portable_target_effects(
        trace: TraceRecord) -> tuple[list[dict[str, Any]], list[str]]:
    """Return a complete role-parameterized target contract or fail closed."""
    bindings = dict(trace.provenance.get("params") or {})
    # Semantic roles refine the concrete task params but must not replace
    # fields they omit.
    bindings.update(dict(trace.provenance.get("semantic_params") or {}))
    raw = [dict(item) for item in
           (trace.provenance.get("target_effects") or [])
           if isinstance(item, dict)]
    # Benchmark/task contracts historically use ``$role`` and ``$task.role``
    # while reusable Skill contracts require the explicit ``$inputs.role``
    # namespace.  Canonicalize only roles that the trace actually binds; an
    # unknown symbolic role remains unbound and is rejected below.
    canonical_raw: list[dict[str, Any]] = []
    for item in raw:
        candidate = dict(item)
        args: dict[str, Any] = {}
        for name, value in dict(item.get("args") or {}).items():
            role = binding_slot_name(value)
            args[str(name)] = (
                f"$inputs.{role}"
                if role and role in bindings else value
            )
        candidate["args"] = args
        canonical_raw.append(candidate)
    parameterized = parameterize_predicates(
        canonical_raw,
        bindings,
    )
    errors: list[str] = []
    if len(parameterized) != len(raw):
        errors.append("target_effect_count_changed")
    for index, item in enumerate(parameterized):
        args = dict(item.get("args") or {})
        if args and not is_fully_parameterized_predicate(item):
            errors.append(f"target_effect_unbound:{index}")
        findings = validate_long_term_asset(
            {"target_effect": item}, asset_kind="composite_target_contract")
        if findings:
            errors.append(f"target_effect_unsafe:{index}:{findings[0]}")
    return (parameterized if not errors else []), errors


def _composite_checks(atomic_refs: list[SkillRef],
                      registry: SkillGraphRegistry,
                      segments: list[dict[str, Any]] | None = None) -> list[str]:
    """Compute terminal predicates by replaying actual positive/negative deltas.

    No predicate is declared transient by name. An earlier Effect disappears
    from the Composite contract only when a later occurrence contains the
    matching negative state delta.
    """
    active: dict[tuple[str, tuple[tuple[str, str], ...]], dict[str, Any]] = {}
    segments = list(segments or [])
    for index, ref in enumerate(atomic_refs):
        segment = segments[index] if index < len(segments) else {}
        positives = [dict(item) for item in (segment.get("effect") or [])
                     if isinstance(item, dict)]
        negatives = [dict(item) for item in (segment.get("negative_effect") or [])
                     if isinstance(item, dict)]
        if not positives:
            atomic = registry.get(ref) or registry.get_recommended(ref.logical_id)
            if atomic is not None:
                positives = [dict(item) for item in atomic.effects
                             if isinstance(item, dict)]
                negatives = [dict(item) for item in
                             (atomic.metadata.get("observed_negative_effects") or [])
                             if isinstance(item, dict)]
        for negative in negatives:
            inner = negative.get("not") if isinstance(negative.get("not"), dict) else negative
            active.pop(_effect_key(inner), None)
        for positive in positives:
            active[_effect_key(positive)] = positive

    checks: list[str] = []
    for effect in active.values():
        name = str(effect.get("predicate") or "")
        if name and name not in checks:
            checks.append(name)
    return checks


def _effect_key(effect: dict[str, Any]) -> tuple[str, tuple[tuple[str, str], ...]]:
    args = tuple(sorted(
        (str(key), str(value).strip().lower().replace(" ", "_"))
        for key, value in (effect.get("args") or {}).items()))
    return str(effect.get("predicate") or ""), args


def _infer_data_edges(
        steps: list[dict[str, Any]], segments: list[dict[str, Any]],
        occurrence_evidence: list[dict[str, Any]],
        registry: SkillGraphRegistry, trace_id: str,
        ) -> list[dict[str, Any]]:
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
                    output_name = str(output.get("name") or "")
                    materializer = validate_output_materializer(
                        source, output_name)
                    if not output_name or not materializer.passed:
                        continue
                    same_type = bool(
                        output.get("semantic_type")
                        and output.get("semantic_type")
                        == input_spec.get("semantic_type"))
                    source_params = dict(
                        (segments[source_index] if source_index < len(segments)
                         else {}).get("params") or {})
                    target_params = dict(
                        (segments[target_index] if target_index < len(segments)
                         else {}).get("params") or {})
                    source_role = _materializer_source_role(
                        source, materializer.materializer)
                    source_value = source_params.get(
                        source_role or output_name)
                    target_value = target_params.get(
                        str(input_spec.get("name") or ""))
                    grounded_same_value = bool(
                        source_value not in (None, "")
                        and target_value not in (None, "")
                        and _same_grounded_value(source_value, target_value))
                    # Broad semantic types such as entity_ref are not enough
                    # on their own: two unrelated roles can share that type.
                    if not (same_type and grounded_same_value):
                        continue
                    edges.append(GraphEdge(
                        source=source_step["node_ref"], target=target_step["node_ref"],
                        type=EdgeType.DATA_FLOW, scope="composite",
                        source_step=source_step["step_id"],
                        target_step=target_step["step_id"],
                        mapping={
                            "source_output": output_name,
                            "target_input": str(input_spec.get("name")),
                            "schema": str(input_spec.get("semantic_type") or "value"),
                            "transform": "identity",
                            "evidence": {
                                "grounding_match": True,
                                "source_evidence_hash": str(
                                    occurrence_evidence[source_index].get(
                                        "evidence_hash") or ""),
                                "target_evidence_hash": str(
                                    occurrence_evidence[target_index].get(
                                        "evidence_hash") or ""),
                            },
                        },
                        evidence=[trace_id],
                    ).to_dict())
                    matched = True
                    break
                if matched:
                    break
    return edges


def _same_grounded_value(left: Any, right: Any) -> bool:
    normalize = lambda value: re.sub(
        r"\s+", "_", str(value or "").strip().lower())
    return bool(normalize(left) and normalize(left) == normalize(right))


def _support_key(trace: TraceRecord) -> str:
    instance = str((trace.provenance or {}).get("environment_instance") or
                   (trace.provenance or {}).get("game_file") or "")
    return "support:" + content_hash({
        "benchmark": trace.benchmark,
        "task_id": trace.task_id,
        "environment_instance": instance,
    })


def _infer_dependency_edges(steps: list[dict[str, Any]], registry: SkillGraphRegistry,
                            trace_id: str,
                            graph_proposal: dict[str, Any],
                            proposal_evidence_ref: dict[str, Any],
                            source_phase_ids: list[str] | None = None,
                            ) -> list[dict[str, Any]]:
    """Build causal edges from the nearest Effect producer plus gated LLM hints."""
    edges: list[dict[str, Any]] = []
    aliases = list(source_phase_ids or [
        str(step.get("metadata", {}).get("phase_id")) for step in steps])
    phase_to_step = {phase_id: step
                     for phase_id, step in zip(aliases, steps)}
    for target_index, target_step in enumerate(steps):
        target_ref = SkillRef.parse(target_step["node_ref"])
        target = registry.get(target_ref) or registry.get_recommended(target_ref.logical_id)
        for precondition in list(getattr(target, "preconditions", []) or []):
            predicate = str(precondition.get("predicate") or "")
            for source_index in range(target_index - 1, -1, -1):
                source_step = steps[source_index]
                source_ref = SkillRef.parse(source_step["node_ref"])
                source = registry.get(source_ref) or registry.get_recommended(source_ref.logical_id)
                if any(match_effect_contract(effect, precondition).passed
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
                "proposal_evidence_ref": proposal_evidence_ref,
            },
        ).to_dict())
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for edge in edges:
        unique[(str(edge.get("source_step")), str(edge.get("target_step")),
                str((edge.get("metadata") or {}).get("predicate") or "semantic"))] = edge
    return list(unique.values())


def _materializer_source_role(atomic: Any,
                              materializer: dict[str, Any]) -> str:
    kind = str(materializer.get("kind") or "")
    if kind == "input_role":
        return str(materializer.get("role") or "")
    if kind != "effect_arg":
        return ""
    predicate = str(materializer.get("predicate") or "")
    arg = str(materializer.get("arg") or "")
    roles = {
        binding_slot_name(dict(effect.get("args") or {}).get(arg))
        for effect in list(getattr(atomic, "effects", []) or [])
        if isinstance(effect, dict)
        and str(effect.get("predicate") or "") == predicate
        and arg in dict(effect.get("args") or {})
    }
    roles.discard("")
    return next(iter(roles)) if len(roles) == 1 else ""


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _role_params(params: dict[str, Any], trace: TraceRecord) -> dict[str, Any]:
    """Persist semantic task roles, never source-instance literals."""
    from ..core.predicates import normalize_value
    known = dict(trace.provenance.get("params") or {})
    semantic = dict(trace.provenance.get("semantic_params") or {})
    result: dict[str, Any] = {}

    def role_kind(role: str) -> str:
        lowered = str(role).lower()
        if (lowered == "object_location"
                or any(token in lowered for token in ("source", "origin"))):
            return "source_location"
        if any(token in lowered for token in ("target", "destination")):
            return "target_location"
        if any(token in lowered for token in (
                "location", "container",
                "receptacle", "station", "position", "place")):
            return "location"
        return "entity"

    for key, value in params.items():
        value_norm = normalize_value(value)
        matched_role = ""
        # Role identity comes from the validated occurrence/goal data flow.
        # Never translate through a benchmark-specific compatibility table.
        for source in (known, semantic):
            if str(key) not in source:
                continue
            role_norm = normalize_value(source[str(key)])
            same = value_norm == role_norm
            generic_family = (value_norm and role_norm
                              and not __import__("re").search(r"_\d+$", role_norm)
                              and __import__("re").sub(r"_\d+$", "", value_norm)
                              == role_norm)
            if same or generic_family:
                matched_role = str(key)
                break
        if not matched_role:
            candidates: list[str] = []
            for source in (known, semantic):
                for source_role, source_value in source.items():
                    role_norm = normalize_value(source_value)
                    same = value_norm == role_norm
                    generic_family = (value_norm and role_norm
                                      and not __import__("re").search(
                                          r"_\d+$", role_norm)
                                      and __import__("re").sub(
                                          r"_\d+$", "", value_norm)
                                      == role_norm)
                    key_kind = role_kind(str(key))
                    source_kind = role_kind(str(source_role))
                    compatible_kind = (key_kind == source_kind
                                       or key_kind == "location")
                    if ((same or generic_family) and compatible_kind):
                        candidates.append(str(source_role))
            if candidates:
                matched_role = sorted(set(candidates), key=lambda role: (
                    0 if str(key) in role or role in str(key) else 1,
                    len(role), role))[0]
        if matched_role:
            result[str(key)] = f"$task.{matched_role}"
    return result


def _role_params_for_segments(segments: list[dict[str, Any]],
                              trace: TraceRecord) -> list[dict[str, Any]]:
    """Parameterize a whole occurrence chain without losing local roles.

    A location can appear as ``target_location`` on a navigation occurrence
    and as ``object_location`` or ``transformation_resource`` on the following
    capability.  Parameterizing each phase independently used to discard the
    former because it was not the task's final destination.  This whole-chain
    pass aligns equal grounded values first, then emits either a task role or
    an occurrence-scoped ``$flow`` role.
    """
    from ..core.predicates import normalize_value

    known = dict(trace.provenance.get("params") or {})
    semantic = dict(trace.provenance.get("semantic_params") or {})
    def task_role(value: Any, preferred_role: str) -> str:
        value_norm = normalize_value(value)
        # A grounded executable parameter is unambiguous and wins first.
        # Example: the final cabinet_1 can safely bind target_location even
        # when the semantic goal merely says "cabinet".
        for role, task_value in known.items():
            role_norm = normalize_value(task_value)
            if not value_norm or not role_norm:
                continue
            same = value_norm == role_norm
            family = (str(role) == str(preferred_role)
                      and not re.search(r"_\d+$", role_norm)
                      and re.sub(r"_\d+$", "", value_norm) == role_norm)
            if same or family:
                return str(role)

        # A generic semantic value (for example "cabinet") may match several
        # concrete instances.  Only accept that family match when the role
        # discovered from this occurrence chain agrees with the task role;
        # otherwise an object source cabinet could be mistaken for the final
        # destination cabinet.
        for role, task_value in semantic.items():
            if str(role) != str(preferred_role):
                continue
            role_norm = normalize_value(task_value)
            if not value_norm or not role_norm:
                continue
            same = value_norm == role_norm
            family = (not re.search(r"_\d+$", role_norm)
                      and re.sub(r"_\d+$", "", value_norm) == role_norm)
            if same or family:
                return str(role)
        return ""

    roles_by_value: dict[str, set[str]] = {}
    for segment in segments:
        for raw_role, value in dict(segment.get("params") or {}).items():
            normalized = normalize_value(value)
            if normalized:
                roles_by_value.setdefault(normalized, set()).add(str(raw_role))

    def preferred_flow_role(value: Any, local_role: str) -> str:
        normalized = normalize_value(value)
        candidates = roles_by_value.get(normalized) or {str(local_role)}
        # A specific occurrence role carries more information than the generic
        # role used by a navigation node.  This is a naming preference only;
        # no operation, entity, or benchmark taxonomy is encoded.
        def rank(role: str) -> tuple[int, int, str]:
            generic = role in {"target_location", "execution_location", "location"}
            relational = role.endswith("_location") or role.endswith("_resource") \
                or role.endswith("_station")
            return (1 if generic else 0, 0 if relational else 1, role)
        return sorted(candidates, key=rank)[0]

    result: list[dict[str, Any]] = []
    for segment in segments:
        mapped: dict[str, Any] = {}
        for key, value in dict(segment.get("params") or {}).items():
            flow_role = preferred_flow_role(value, str(key))
            role = task_role(value, flow_role)
            if role:
                mapped[str(key)] = f"$task.{role}"
            elif value not in (None, ""):
                mapped[str(key)] = f"$flow.{flow_role}"
        result.append(mapped)
    return result
