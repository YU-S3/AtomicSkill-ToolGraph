"""Trace Atomicizer：从成功轨迹发现原子能力（设计文档 v2.0 §25）。

Causal Trace Normalization → Candidate Boundary Detection → Effect Extraction
→ I/O/Precondition Inference → Independent Validator Construction
→ Atomicity Check / SplitScore → Candidate Alignment → Add/Reuse/Merge/Split
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from ..core.refs import SkillRef
from ..core.skill_ir import AbstractAtomicSkill
from ..core.status import EdgeType, SkillNodeKind, SkillStatus
from ..core.trace_ir import TraceRecord
from ..graph.aligner import (
    AlignDecision,
    _atomic_contract_compatible,
    align_atomic,
)
from ..graph.registry import SkillGraphRegistry
from ..runtime.output_materializer import validate_output_materializer
from .boundary_detector import detect_boundaries
from .effect_extractor import (
    _FACT_FAMILY_NAMES,
    _family_of,
    _semantic_type_of,
    ExtractedEffect,
    extract_effect,
    output_declarations_from_effects,
    parameterize_predicates,
)
from .split_score import SplitScoreResult, compute_split_score
from .semantic_extractor import SemanticExtraction, SemanticExtractorAgent


@dataclass
class AtomicCandidate:
    segment: dict[str, Any]
    effect: ExtractedEffect
    skill: AbstractAtomicSkill
    split_score: SplitScoreResult
    alignment: AlignDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_name": self.segment.get("name", ""),
            "skill_ref": str(self.skill.ref),
            "effect": self.effect.to_dict(),
            "split_score": self.split_score.to_dict(),
            "alignment": self.alignment.to_dict(),
        }


@dataclass
class AtomicizationResult:
    candidates: list[AtomicCandidate] = field(default_factory=list)
    segments: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)   # add | reuse | merge | split
    semantic_extraction: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "segments": self.segments,
            "decisions": self.decisions,
            "semantic_extraction": self.semantic_extraction,
        }


class TraceAtomicizer:
    """成功轨迹 → 原子候选（Abstract Atomic Skill 层面；Tool 由 Miner 处理）。"""

    def __init__(self, registry: SkillGraphRegistry,
                 thresholds=None, extractor_agent: SemanticExtractorAgent | None = None,
                 *, allow_legacy_fallback: bool = False) -> None:
        self.registry = registry
        from ..core.config import Thresholds
        self.thresholds = thresholds or Thresholds()
        self.extractor_agent = extractor_agent
        self.allow_legacy_fallback = bool(allow_legacy_fallback)

    def atomicize_success(self, trace: TraceRecord) -> AtomicizationResult:
        if not trace.success:
            raise ValueError("atomicize_success 只接受成功轨迹")

        # 1-2. 因果归一化 + 边界检测
        extraction = (self.extractor_agent.extract(
                          trace, known_atomic_contracts=self._known_atomic_contracts())
                      if self.extractor_agent is not None else SemanticExtraction())
        # Once an Extractor Agent is configured, an invalid/empty proposal must
        # fail closed for evolution.  Falling back to the legacy string/range
        # detector here bypasses semantic de-duplication and the event-level
        # causal replay gate precisely when the Extractor API has an error,
        # allowing loops to enter the bank.  Task success is still persisted;
        # this trace simply contributes no new capability evidence.
        segments = (extraction.phases if self.extractor_agent is not None
                    and not self.allow_legacy_fallback
                    else extraction.phases or detect_boundaries(trace))
        result = AtomicizationResult(
            segments=segments, semantic_extraction=extraction.to_dict())

        # 3-6. Effect 提取 / I/O 推断 / Validator 构造 / SplitScore / 对齐
        for segment in segments:
            effect = _materialize_segment_effect(segment)
            if not effect.positive:
                continue  # 纯机械片段（无核心状态转移）不构成原子能力
            # 原子性检查：单一效果 → keep；多效果 → 依据 SplitScore
            evidence = _build_evidence(segment, self.registry)
            split_score = compute_split_score(segment, evidence, self.thresholds)

            effects = (_split_effect(effect) if split_score.verdict == "split"
                       and len(effect.positive) > 1 else [effect])
            for child_index, child_effect in enumerate(effects):
                child_segment = dict(segment)
                if len(effects) > 1:
                    child_segment["name"] = (
                        f"{segment.get('name') or 'atomic'}-{child_effect.primary_family}")
                    child_segment["effect"] = list(child_effect.positive)
                    child_segment["split_parent"] = str(segment.get("name") or "atomic")
                    child_segment["split_index"] = child_index
                skill = self._build_atomic_skill(
                    trace, child_segment, child_effect, split_score)
                if not skill.effects:
                    continue
                alignment = align_atomic(skill, self.registry)

                if alignment.matched:
                    result.decisions.append("reuse")
                elif len(effects) > 1:
                    result.decisions.append("split")
                else:
                    result.decisions.append("add")
                result.candidates.append(AtomicCandidate(
                    segment=child_segment, effect=child_effect, skill=skill,
                    split_score=split_score, alignment=alignment,
                ))
        # Keep a one-to-one candidate/segment relation for Tool mining/binding.
        result.segments = [candidate.segment for candidate in result.candidates]
        return result

    # ------------------------------------------------------------------
    def _build_atomic_skill(self, trace: TraceRecord, segment: dict[str, Any],
                            effect: ExtractedEffect,
                            split_score: SplitScoreResult) -> AbstractAtomicSkill:
        # The immutable identity comes from code-verified Effect semantics.
        # LLM wording is evidence/alias only and may evolve across traces.
        segment_name = str(effect.suggested_name or segment.get("name") or "atomic")
        benchmark = _safe_prefix(trace.benchmark)
        logical_id = f"{benchmark}.{segment_name}"
        # 多效果且 split 判定为 split 时，逻辑 id 加主效果族后缀避免污染
        if split_score.verdict == "split" and effect.primary_family:
            logical_id = f"{benchmark}.{segment_name}-{effect.primary_family}"

        guideline_rules = []
        if effect.summary:
            guideline_rules.append(f"核心效果：{effect.summary}。")
        guideline_rules.append(
            "执行前先确认前置条件满足；执行后必须通过节点验证器确认效果真实发生。")

        # 谓词参数化：实例字面量 → $inputs.<slot>（跨实例复用）
        bound_params = segment.get("params") or {}
        parameterized_effects = _safe_parameterized_effects(
            parameterize_predicates(effect.positive, bound_params), bound_params)
        parameterized_preconditions = parameterize_predicates(effect.preconditions,
                                                              bound_params)
        parameterized_negative_effects = _parameterize_negative_effects(
            effect.negative, bound_params)

        # Store the grounded occurrence in the local Evidence Store.  The
        # portable Atomic node contains only generalized contracts plus an
        # opaque content-addressed reference.
        occurrence_ref = self.registry.evidence_store.put(
            "atomic_occurrence",
            {
                "segment": segment,
                "positive_effects": effect.positive,
                "negative_effects": effect.negative,
                "preconditions": effect.preconditions,
            },
            trace_id=trace.trace_id,
            event_start=_optional_int(segment.get("event_start")),
            event_end=_optional_int(segment.get("event_end")),
        )
        # Raw LLM aliases and entity values are occurrence evidence even when
        # they do not carry a numeric suffix (for example a customer name).
        # The portable alias is therefore the code-derived Effect identity.
        safe_aliases = {segment_name: 1} if segment_name else {}
        portable_negative_effects = [
            item for item in parameterized_negative_effects
            if _fully_portable_predicate(item)
        ]
        portable_outputs = output_declarations_from_effects(
            parameterized_effects)
        skill = AbstractAtomicSkill(
            ref=SkillRef(logical_id=logical_id, version="1.0.0"),
            summary=effect.summary or f"原子能力：{segment_name}",
            inputs=effect.inputs,
            outputs=portable_outputs,
            preconditions=parameterized_preconditions,
            effects=parameterized_effects,
            validator=effect.validator,
            failure_modes=[],
            guideline={"layer": 2, "rules": guideline_rules},
            metadata={
                "task_type_labels": [trace.task_type] if trace.task_type else [],
                "source_trace_ids": [trace.trace_id],
                "source_kinds": [str(segment.get("source_kind") or "unscoped")],
                "runtime_occurrence_ids": [str(
                    segment.get("runtime_occurrence_id") or "")],
                "task_gap_ids": ([str(segment.get("task_gap_id"))]
                                 if segment.get("task_gap_id") else []),
                "statistics": {
                    "use_count": 0,
                    "success_count": 1,
                    "failure_count": 0,
                    "utility": 0.5,
                    "support_count": 1,
                },
                "benchmark": trace.benchmark,
                "semantic_alias_counts": safe_aliases,
                "observed_parameter_families": {},
                # Negative deltas are occurrence evidence used for causal and
                # terminal-effect closure. They are not inferred from the
                # capability name and remain auditable after registration.
                "observed_negative_effects": portable_negative_effects,
                "occurrence_evidence_refs": [occurrence_ref],
                "generalization": {
                    "canonical_name": segment_name,
                    "identity_source": "code_validated_core_effect",
                    "status": "canonicalized_single_trace",
                    "independent_trace_count": 1,
                },
            },
            # One successful occurrence is a candidate, not yet reusable
            # frozen knowledge. Independent trace support promotes it below.
            status=SkillStatus.DRAFT,
        )
        # The final Effect contract may be smaller than the Extractor proposal
        # after parameterization/privacy filtering.  Revalidate materializers
        # against that exact contract so no output can point at a removed
        # producer or silently relabel its semantic type.
        skill.outputs = [
            output for output in skill.outputs
            if validate_output_materializer(skill, output).passed
        ]
        return skill

    # ------------------------------------------------------------------
    def apply(self, trace: TraceRecord) -> AtomicizationResult:
        """atomicize + 注册（对齐时合并证据；新候选注册）。"""
        result = self.atomicize_success(trace)
        for candidate_index, candidate in enumerate(result.candidates):
            decision = "reuse" if candidate.alignment.matched else "add"
            if decision == "reuse":
                # 复用：更新证据（跨实例复用计数 + 来源轨迹）
                existing_ref = SkillRef.parse(candidate.alignment.matched_ref)
                existing = self.registry.get(existing_ref)
                if existing is None:
                    existing = self.registry.get_recommended(existing_ref.logical_id)
                if existing is not None:
                    self._merge_evidence(existing, candidate.skill, trace)
                    self.registry.update_runtime_state(existing)
                    candidate.skill = existing
                else:
                    self.registry.register(candidate.skill)
            else:
                # A non-equivalent extraction must not overwrite an existing
                # immutable ref merely because the generated logical_id and
                # default version collide.  This previously let a trace with a
                # concrete ``mug_2`` effect replace the canonical
                # ``$inputs.object`` contract in Full runs.
                collision = self.registry.get(candidate.skill.ref)
                if collision is not None:
                    if _atomic_contract_compatible(collision, candidate.skill):
                        self._merge_evidence(collision, candidate.skill, trace)
                        self.registry.update_runtime_state(collision)
                        candidate.skill = collision
                        result.decisions[candidate_index] = (
                            "reuse_contract_collision")
                        continue
                    # Same generated name with an incompatible I/O/Effect
                    # contract must remain a separate capability instead of
                    # corrupting the first immutable node.
                    signature = json.dumps({
                        "inputs": candidate.skill.inputs,
                        "outputs": candidate.skill.outputs,
                        "effects": candidate.skill.effects,
                    }, sort_keys=True, ensure_ascii=False)
                    suffix = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:8]
                    candidate.skill.ref = SkillRef(
                        f"{candidate.skill.ref.logical_id}__{suffix}",
                        candidate.skill.ref.version)
                    variant = self.registry.get(candidate.skill.ref)
                    if variant is not None:
                        if not _atomic_contract_compatible(
                                variant, candidate.skill):
                            raise ValueError(
                                "atomic_contract_hash_collision:"
                                f"{candidate.skill.ref}")
                        self._merge_evidence(variant, candidate.skill, trace)
                        self.registry.update_runtime_state(variant)
                        candidate.skill = variant
                        result.decisions[candidate_index] = (
                            "reuse_contract_variant")
                        continue
                    result.decisions[candidate_index] = "add_contract_variant"
                self.registry.register(candidate.skill)
                score = float(candidate.alignment.evidence.get("align_score", 0.0))
                if candidate.alignment.matched_ref and score >= 0.3:
                    self.registry.add_edge(
                        str(candidate.skill.ref), candidate.alignment.matched_ref,
                        EdgeType.SIMILAR,
                        metadata={"confidence": score, "reason": "atomic_aligner"},
                        evidence=[trace.trace_id],
                    )
        return result

    @staticmethod
    def _merge_evidence(existing: AbstractAtomicSkill, incoming: AbstractAtomicSkill,
                        trace: TraceRecord) -> None:
        """复用对齐：合并 task_type 标签、来源轨迹、支持数；必要时提升版本。"""
        labels = list(existing.metadata.get("task_type_labels") or [])
        for label in incoming.metadata.get("task_type_labels") or []:
            if label not in labels:
                labels.append(label)
        stats = dict(existing.metadata.get("statistics") or {})
        existing.metadata["task_type_labels"] = labels
        existing.metadata["statistics"] = stats
        sources = list(existing.metadata.get("source_trace_ids") or [])
        independent = trace.trace_id not in sources
        if independent:
            sources.append(trace.trace_id)
            stats["support_count"] = int(stats.get("support_count", 0)) + 1
            stats["success_count"] = int(stats.get("success_count", 0)) + 1
        existing.metadata["source_trace_ids"] = sources
        existing.metadata["statistics"] = stats
        alias_counts = dict(existing.metadata.get("semantic_alias_counts") or {})
        for alias, count in (incoming.metadata.get("semantic_alias_counts") or {}).items():
            alias_counts[str(alias)] = int(alias_counts.get(str(alias), 0)) + int(count)
        existing.metadata["semantic_alias_counts"] = alias_counts
        # Grounded entity families remain in occurrence evidence.  Retain the
        # field only for schema compatibility with older planners/banks.
        existing.metadata["observed_parameter_families"] = {}
        negative_evidence = list(
            existing.metadata.get("observed_negative_effects") or [])
        for effect in incoming.metadata.get("observed_negative_effects") or []:
            if effect not in negative_evidence:
                negative_evidence.append(effect)
        existing.metadata["observed_negative_effects"] = negative_evidence
        occurrence_refs = list(
            existing.metadata.get("occurrence_evidence_refs") or [])
        for evidence_ref in incoming.metadata.get("occurrence_evidence_refs") or []:
            if evidence_ref not in occurrence_refs:
                occurrence_refs.append(evidence_ref)
        existing.metadata["occurrence_evidence_refs"] = occurrence_refs
        # Provenance is occurrence evidence as well.  Reusing a generalized
        # Atomic must not erase the fact that one support occurrence came from
        # an explicit TaskGap (or which Runtime occurrence supplied it).
        for key in ("source_kinds", "runtime_occurrence_ids", "task_gap_ids"):
            merged = [str(value) for value in
                      (existing.metadata.get(key) or []) if str(value)]
            for value in incoming.metadata.get(key) or []:
                text = str(value)
                if text and text not in merged:
                    merged.append(text)
            existing.metadata[key] = merged
        if independent:
            # A necessary Precondition should recur.  Intersecting grounded,
            # parameterized contracts removes incidental earlier state such as
            # an unrelated transformation that happened to be true once.
            incoming_pre = {repr(item): item for item in incoming.preconditions}
            existing.preconditions = [
                item for item in existing.preconditions
                if repr(item) in incoming_pre
            ]
            existing.validator["pre_checks"] = sorted({
                str(item.get("predicate") or "")
                for item in existing.preconditions if item.get("predicate")
            })
        trace_count = len(sources)
        existing.metadata["generalization"] = {
            "canonical_name": existing.ref.logical_id.rsplit(".", 1)[-1],
            "identity_source": "code_validated_core_effect",
            "status": ("cross_trace_validated" if trace_count >= 2
                       else "canonicalized_single_trace"),
            "independent_trace_count": trace_count,
            "semantic_alias_count": len(alias_counts),
        }
        if trace_count >= 2 and existing.status == SkillStatus.DRAFT:
            existing.status = SkillStatus.ACTIVE

    def _known_atomic_contracts(self) -> list[dict[str, Any]]:
        """Small auditable catalog supplied to each independent Extractor call."""
        catalog: list[dict[str, Any]] = []
        for node in self.registry.list_all_versions(SkillNodeKind.ABSTRACT_ATOMIC):
            if node.status in {SkillStatus.SUPPRESSED, SkillStatus.RETIRED,
                               SkillStatus.SHADOW}:
                continue
            if not isinstance(node, AbstractAtomicSkill):
                continue
            catalog.append({
                "canonical_name": node.ref.logical_id.rsplit(".", 1)[-1],
                "logical_id": node.ref.logical_id,
                "summary": node.summary,
                "inputs": [item.get("name") for item in node.inputs],
                "effects": [item.get("predicate") for item in node.effects],
                "semantic_aliases": sorted(
                    (node.metadata.get("semantic_alias_counts") or {}).keys()),
                "support_count": int(
                    (node.metadata.get("statistics") or {}).get("support_count", 0)),
            })
        return catalog[:100]


def _materialize_segment_effect(segment: dict[str, Any]) -> ExtractedEffect:
    """Preserve the Effect contract already validated against event evidence.

    The semantic validator may remove helper transitions from a proposed
    occurrence. Re-diffing the whole phase here used to reintroduce those
    transitions and could create a false extra Atomic node.
    """
    params = dict(segment.get("params") or {})
    effect = extract_effect(segment.get("before") or {},
                            segment.get("after") or {}, bound_params=params)
    validated = segment.get("effect")
    if not isinstance(validated, list) or not validated:
        return effect

    effect.positive = [dict(item) for item in validated if isinstance(item, dict)]
    effect.negative = [dict(item) for item in (segment.get("negative_effect") or [])
                       if isinstance(item, dict)]
    effect.preconditions = [dict(item) for item in (segment.get("preconditions") or [])
                            if isinstance(item, dict)]
    families = sorted({_family_of(item) for item in effect.positive if _family_of(item)})
    effect.primary_family = families[0] if families else ""
    if effect.primary_family:
        name, summary = _FACT_FAMILY_NAMES.get(
            effect.primary_family, (effect.primary_family, effect.primary_family))
        effect.suggested_name = name
        effect.summary = summary
    effect.outputs = output_declarations_from_effects(effect.positive)
    effect.validator = {
        "pre_checks": sorted({str(item.get("predicate"))
                              for item in effect.preconditions if item.get("predicate")}),
        "post_checks": sorted({str(item.get("predicate"))
                               for item in effect.positive if item.get("predicate")}),
    }
    return effect


def _split_effect(effect: ExtractedEffect) -> list[ExtractedEffect]:
    """Materialize one independently verifiable contract per positive Effect."""
    children: list[ExtractedEffect] = []
    for positive in effect.positive:
        predicate = str(positive.get("predicate") or "effect")
        family = predicate.replace(".", "_")
        output_names = list((positive.get("args") or {}).keys())
        children.append(ExtractedEffect(
            positive=[positive], negative=[], inputs=list(effect.inputs),
            outputs=output_declarations_from_effects([positive]),
            preconditions=list(effect.preconditions),
            validator={
                "pre_checks": list(effect.validator.get("pre_checks") or []),
                "post_checks": [predicate],
            },
            primary_family=family,
            suggested_name=family,
            summary=f"实现单一状态效果：{predicate}",
        ))
    return children


def _build_evidence(segment: dict[str, Any], registry: SkillGraphRegistry) -> dict[str, Any]:
    """从 SkillGraph 历史构建 SplitScore 证据。"""
    evidence: dict[str, Any] = {
        "reuse_evidence": {},
        "validation_evidence": 0,
        "failure_clusters": {},
        "io_evidence": 0,
        "retry_evidence": 0,
        "executor_evidence": 0,
    }
    effects = segment.get("effect") or []
    for obj in registry.list_all_versions():
        if obj.status in {SkillStatus.SUPPRESSED, SkillStatus.RETIRED,
                          SkillStatus.SHADOW}:
            continue
        from ..core.skill_ir import CompositeSkill
        if isinstance(obj, CompositeSkill):
            continue
        obj_effects = getattr(obj, "effects", None) or []
        overlap = _effect_overlap(effects, obj_effects)
        if overlap > 0:
            stats = (obj.metadata or {}).get("statistics") or {}
            support = int(stats.get("support_count", 0))
            if support > 1:
                evidence["reuse_evidence"][obj.ref.logical_id] = support
                evidence["validation_evidence"] += 1
                evidence["io_evidence"] += 1
        # 替代实现证据：不同 logical_id 但 effect 相同
    return evidence


def _effect_overlap(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> int:
    def key(effect: dict[str, Any]) -> str:
        return f"{effect.get('predicate')}:{sorted((effect.get('args') or {}).items())}"
    return len({key(e) for e in a} & {key(e) for e in b})


def _safe_prefix(benchmark: str) -> str:
    import re
    cleaned = re.sub(r"[^a-z0-9]+", "-", str(benchmark).lower()).strip("-")
    return cleaned or "generic"


def _entity_family(value: Any) -> str:
    """Collapse a conventional numeric instance suffix for generalization audit."""
    import re
    normalized = str(value).strip().lower().replace(" ", "_")
    return re.sub(r"_\d+$", "", normalized)


def _fully_portable_predicate(value: Any) -> bool:
    """Negative closure may persist only role-parameterized contracts."""
    predicates = [value.get("not")] if isinstance(value, dict) \
        and isinstance(value.get("not"), dict) else [value]
    return all(isinstance(item, dict)
               and all(isinstance(arg, str) and arg.startswith("$inputs.")
                       for arg in dict(item.get("args") or {}).values())
               for item in predicates)


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_parameterized_effects(effects: list[dict[str, Any]],
                                bound_params: dict[str, Any]) -> list[dict[str, Any]]:
    """Reject a concrete occurrence literal that escaped generic grounding."""
    import re
    safe: list[dict[str, Any]] = []
    for effect in effects:
        args = effect.get("args") or {}
        # A concrete occurrence literal may not enter an Abstract Effect when
        # the same semantic family is already represented by an input role.
        # Placeholder names themselves are intentionally unconstrained.
        input_families = {_entity_family(value) for value in bound_params.values()}
        leaked = any(isinstance(value, str) and re.search(r"_\d+$", value)
                     and _entity_family(value) in input_families
                     for value in args.values())
        if not leaked:
            safe.append(effect)
    return safe


def _parameterize_negative_effects(
        effects: list[dict[str, Any]],
        bound_params: dict[str, Any]) -> list[dict[str, Any]]:
    """Parameterize the predicate nested in a negative Effect wrapper."""
    result: list[dict[str, Any]] = []
    for effect in effects:
        if not isinstance(effect, dict):
            continue
        inner = effect.get("not")
        if isinstance(inner, dict):
            parameterized = parameterize_predicates([inner], bound_params)
            if parameterized:
                result.append({"not": parameterized[0]})
        else:
            parameterized = parameterize_predicates([effect], bound_params)
            result.extend(parameterized)
    return result
