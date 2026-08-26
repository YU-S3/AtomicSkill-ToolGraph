"""候选对齐器（设计文档 v2.0 §25.3 第 7 步、§37）。

- align_atomic：SemanticMatch ∧ IOCompatible ∧ CoreEffectEquivalent ∧ ValidatorConsistent
- align_implementation：同 Abstract + 等价 Tool binding + 参数映射 + compatibility
- align_composite：同原子链
规则 + 可选 LLM 混合；嵌入仅用于候选召回，不作为最终 merge 判据（§31.2）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..core.skill_ir import AbstractAtomicSkill, CompositeSkill, ImplementationAtom, ToolBinding
from ..core.status import SkillNodeKind
from .registry import SkillGraphRegistry

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


@dataclass
class AlignDecision:
    matched: bool
    matched_ref: str = ""            # 命中的现有引用
    evidence: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched": self.matched,
            "matched_ref": self.matched_ref,
            "evidence": self.evidence,
            "reason": self.reason,
        }


def _tokens(*texts: str) -> set[str]:
    result: set[str] = set()
    for text in texts:
        for token in _TOKEN_RE.findall(str(text).lower()):
            if len(token) >= 3:
                result.add(token)
    return result


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a), len(b))


def _norm_effects(effects: list[dict[str, Any]]) -> set[str]:
    result: set[str] = set()
    for effect in effects:
        if not isinstance(effect, dict):
            result.add(str(effect))
            continue
        name = str(effect.get("predicate", ""))
        args = {str(k): str(v).replace("$", "") for k, v in (effect.get("args") or {}).items()}
        result.add(f"{name}:{sorted(args.items())}")
    return result


def _io_names(io_list: list[dict[str, Any]]) -> set[str]:
    return {str(item.get("name", "")).lower() for item in io_list if item.get("name")}


def _atomic_contract_compatible(left: AbstractAtomicSkill,
                                right: AbstractAtomicSkill) -> bool:
    """Hard merge gate for occurrence semantics.

    Sharing a predicate name is insufficient: observing an existing relation
    and producing that relation can expose different required roles.  Only
    identical parameterized Effects and the same declared input/output role
    interface may accumulate support under one immutable Atomic identity.
    Preconditions are evidence refined across occurrences and are therefore
    deliberately not part of this immutable gate.
    """
    return (
        _norm_effects(left.effects) == _norm_effects(right.effects)
        and _io_names(left.inputs) == _io_names(right.inputs)
        and _io_names(left.outputs) == _io_names(right.outputs)
    )


def align_atomic(candidate: AbstractAtomicSkill, registry: SkillGraphRegistry) -> AlignDecision:
    """把 Atomic 候选对齐到已有 Abstract Atomic Skill（§37.1）。"""
    candidates = registry.list_by_kind(SkillNodeKind.ABSTRACT_ATOMIC)
    best: tuple[float, Any] | None = None
    for existing in candidates:
        compatible = _atomic_contract_compatible(candidate, existing)
        same_identity_family = (
            existing.ref.logical_id == candidate.ref.logical_id
            or existing.ref.logical_id.startswith(
                f"{candidate.ref.logical_id}__")
        )
        if same_identity_family and compatible:
            best = (1.0, existing)
            break
        semantic = _overlap(_tokens(candidate.summary, *(candidate.guideline_rules())),
                            _tokens(existing.summary, *(existing.guideline_rules())))
        io_ok = _overlap(_io_names(candidate.inputs) | _io_names(candidate.outputs),
                         _io_names(existing.inputs) | _io_names(existing.outputs))
        effect_eq = _norm_effects(candidate.effects) == _norm_effects(existing.effects)
        validator_ok = _overlap(_tokens(str(candidate.validator)),
                                _tokens(str(existing.validator)))
        score = 0.3 * semantic + 0.2 * io_ok + 0.3 * float(effect_eq) + 0.2 * validator_ok
        if best is None or score > best[0]:
            best = (score, existing)
    if best is None:
        return AlignDecision(matched=False, reason="no_existing_abstract_atomic")
    score, existing = best
    evidence = {
        "semantic_overlap": round(_overlap(_tokens(candidate.summary), _tokens(existing.summary)), 3),
        "io_overlap": round(_overlap(_io_names(candidate.inputs) | _io_names(candidate.outputs),
                                     _io_names(existing.inputs) | _io_names(existing.outputs)), 3),
        "effect_equivalent": _norm_effects(candidate.effects) == _norm_effects(existing.effects),
        "align_score": round(score, 3),
    }
    evidence["contract_compatible"] = _atomic_contract_compatible(
        candidate, existing)
    matched = (score >= 0.6 and evidence["effect_equivalent"]
               and evidence["contract_compatible"])
    return AlignDecision(matched=matched, matched_ref=str(existing.ref),
                         evidence=evidence, reason="semantic_plus_effect_match" if matched else "below_threshold")


def align_implementation(candidate: ImplementationAtom, registry: SkillGraphRegistry) -> AlignDecision:
    """Implementation merge（§37.2）：implements 同一 Abstract + Tool binding 等价。"""
    for existing in registry.list_by_kind(SkillNodeKind.IMPLEMENTATION_ATOMIC):
        if existing.abstract_ref.logical_id != candidate.abstract_ref.logical_id:
            continue
        if (existing.compatibility or {}) != (candidate.compatibility or {}):
            continue
        existing_bindings = sorted((b.to_dict()["tool_ref"], b.role, str(b.parameter_mapping))
                                   for b in existing.tool_bindings)
        candidate_bindings = sorted((b.to_dict()["tool_ref"], b.role, str(b.parameter_mapping))
                                    for b in candidate.tool_bindings)
        if existing_bindings == candidate_bindings:
            return AlignDecision(matched=True, matched_ref=str(existing.ref),
                                 evidence={"abstract": candidate.abstract_ref.logical_id},
                                 reason="equivalent_implementation")
    return AlignDecision(matched=False, reason="no_equivalent_implementation")


def align_composite(candidate: CompositeSkill, registry: SkillGraphRegistry) -> AlignDecision:
    """Align by the executable occurrence DAG, not by LLM wording.

    LLM-only implicit dependencies are advisory annotations.  Their presence,
    absence, or natural-language reason must not create a new Composite version
    when the ordered Atomic calls, role bindings, code-derived dependencies and
    data flow are unchanged.  Search every historical version so an insight
    update cannot hide the canonical Composite from later occurrences.
    """
    candidate_signature = _composite_occurrence_signature(candidate)
    matches = [
        existing
        for existing in registry.list_all_versions(SkillNodeKind.COMPOSITE)
        if _composite_occurrence_signature(existing) == candidate_signature
    ]
    if matches:
        status_rank = {"active": 2, "draft": 1}

        def rank(existing: CompositeSkill) -> tuple[int, int, tuple[int, int, int]]:
            stats = dict(existing.metadata.get("statistics") or {})
            return (
                status_rank.get(existing.status.value, 0),
                int(stats.get("support_count", 0)),
                _version_key(existing.ref.version),
            )

        existing = max(matches, key=rank)
        return AlignDecision(matched=True, matched_ref=str(existing.ref),
                             evidence={"occurrence_signature": candidate_signature[0]},
                             reason="same_validated_occurrence_dag")
    return AlignDecision(matched=False, reason="no_same_chain_composite")


def _composite_occurrence_signature(composite: CompositeSkill) -> tuple[Any, ...]:
    """Canonical identity of a reusable Composite occurrence topology."""
    steps = tuple(
        (str(step["node_ref"]).rsplit("@", 1)[0],
         tuple(sorted((str(k), str(v))
                      for k, v in step.get("params", {}).items())))
        for step in composite.step_instances()
    )
    control = tuple(sorted(
        (edge.source_step, edge.target_step, edge.type.value)
        for edge in composite.edge_objects() if edge.category == "control"
    ))
    data = tuple(sorted(
        (edge.source_step, edge.target_step,
         str((edge.mapping or {}).get("source_output") or ""),
         str((edge.mapping or {}).get("target_input") or ""),
         str((edge.mapping or {}).get("transform") or "identity"))
        for edge in composite.edge_objects() if edge.category == "data"
    ))
    dependencies = tuple(sorted(
        (edge.source_step, edge.target_step,
         str((edge.metadata or {}).get("predicate") or ""))
        for edge in composite.edge_objects()
        if edge.category == "dependency"
        and str((edge.metadata or {}).get("origin") or "")
        != "llm_semantic_proposal"
    ))
    return steps, control, data, dependencies


def _version_key(version: str) -> tuple[int, int, int]:
    try:
        major, minor, patch = str(version).split(".")
        return int(major), int(minor), int(patch)
    except (TypeError, ValueError):
        return 0, 0, 0


def bindings_equal(a: list[ToolBinding], b: list[ToolBinding]) -> bool:
    def key(binding: ToolBinding):
        return (binding.tool_ref.tool_id, binding.role, str(sorted(binding.parameter_mapping.items())))
    return sorted(key(x) for x in a) == sorted(key(x) for x in b)
