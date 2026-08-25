"""Typed SkillGraph edge IR.

The v2 design names six edge families.  This module freezes the executable
schema that was previously missing: versioned endpoints, runtime step
instances, conditions, data mappings, policies, evidence and scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .refs import content_hash
from .status import EdgeType


STRUCTURAL = {EdgeType.CONTAINS, EdgeType.IMPLEMENTS}
CONTROL = {
    EdgeType.NEXT, EdgeType.BRANCH, EdgeType.PARALLEL,
    EdgeType.RETRY, EdgeType.FALLBACK, EdgeType.LOOP,
}
DATA = {EdgeType.DATA_FLOW}
DEPENDENCY = {
    EdgeType.REQUIRES_SKILL, EdgeType.REQUIRES_PERMISSION,
    EdgeType.REQUIRES_ENVIRONMENT, EdgeType.REQUIRES_SCHEMA,
}
SEMANTIC = {
    EdgeType.EQUIVALENT, EdgeType.SIMILAR,
    EdgeType.ALTERNATIVE, EdgeType.CONFLICT,
}
EVOLUTION = {
    EdgeType.DERIVED_FROM, EdgeType.SUPERSEDES, EdgeType.SPLIT_FROM,
    EdgeType.MERGED_FROM, EdgeType.GENERALIZED_FROM,
    EdgeType.SPECIALIZED_FROM,
}


def edge_category(edge_type: EdgeType) -> str:
    for name, family in (
        ("structural", STRUCTURAL), ("control", CONTROL), ("data", DATA),
        ("dependency", DEPENDENCY), ("semantic", SEMANTIC),
        ("evolution", EVOLUTION),
    ):
        if edge_type in family:
            return name
    raise ValueError(f"unknown edge type: {edge_type}")


@dataclass
class GraphEdge:
    """One persistent or runtime graph edge.

    ``source``/``target`` are fixed Skill refs for global/composite scope and
    step ids for runtime scope.  ``source_step``/``target_step`` preserve the
    instance identity when a Composite calls the same Skill more than once.
    """

    source: str
    target: str
    type: EdgeType
    subtype: str = ""
    scope: str = "global"              # global | composite | runtime
    source_step: str = ""
    target_step: str = ""
    condition: dict[str, Any] = field(default_factory=dict)
    mapping: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    edge_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.type, EdgeType):
            self.type = EdgeType(str(self.type))
        if not self.edge_id:
            digest = content_hash({
                "source": self.source,
                "target": self.target,
                "type": self.type.value,
                "subtype": self.subtype,
                "scope": self.scope,
                "source_step": self.source_step,
                "target_step": self.target_step,
                "condition": self.condition,
                "mapping": self.mapping,
                "policy": self.policy,
                "owner_ref": (self.metadata.get("owner_ref")
                              or self.metadata.get("composite_ref") or ""),
            })[:16]
            self.edge_id = f"edge_{digest}"

    @property
    def category(self) -> str:
        return edge_category(self.type)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.source or not self.target:
            errors.append("edge source/target cannot be empty")
        if self.scope not in {"global", "composite", "runtime"}:
            errors.append(f"invalid edge scope: {self.scope}")
        if self.scope in {"composite", "runtime"} and (
                not self.source_step or not self.target_step):
            errors.append("composite/runtime edges require source_step and target_step")

        if self.type == EdgeType.BRANCH and not self.condition:
            errors.append("branch edge requires condition")
        if self.type == EdgeType.RETRY:
            if int(self.policy.get("max_attempts", 0)) < 1:
                errors.append("retry edge requires policy.max_attempts >= 1")
        if self.type == EdgeType.FALLBACK and not self.policy.get("on"):
            errors.append("fallback edge requires policy.on")
        if self.type == EdgeType.LOOP:
            if not self.condition:
                errors.append("loop edge requires condition")
            if int(self.policy.get("max_iterations", 0)) < 1:
                errors.append("loop edge requires policy.max_iterations >= 1")
        if self.type == EdgeType.PARALLEL and self.policy.get("join") not in {
                "all", "any", "first_success"}:
            errors.append("parallel edge requires policy.join=all|any|first_success")
        if self.type == EdgeType.DATA_FLOW:
            if not self.mapping.get("source_output"):
                errors.append("data_flow requires mapping.source_output")
            if not self.mapping.get("target_input"):
                errors.append("data_flow requires mapping.target_input")
        if self.type in DEPENDENCY and not (
                self.metadata.get("requirement") or self.target):
            errors.append("dependency edge requires a target or metadata.requirement")
        if self.type in SEMANTIC:
            confidence = self.metadata.get("confidence")
            if confidence is None or not 0.0 <= float(confidence) <= 1.0:
                errors.append("semantic edge requires metadata.confidence in [0, 1]")
            if not self.evidence:
                errors.append("semantic edge requires evidence")
        if self.type in EVOLUTION and not (
                self.evidence or self.metadata.get("reason")):
            errors.append("evolution edge requires evidence or metadata.reason")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source": self.source,
            "target": self.target,
            "type": self.type.value,
            "category": self.category,
            "subtype": self.subtype,
            "scope": self.scope,
            "source_step": self.source_step,
            "target_step": self.target_step,
            "condition": self.condition,
            "mapping": self.mapping,
            "policy": self.policy,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphEdge":
        return cls(
            edge_id=str(data.get("edge_id") or ""),
            source=str(data.get("source") or ""),
            target=str(data.get("target") or ""),
            type=EdgeType(str(data.get("type"))),
            subtype=str(data.get("subtype") or ""),
            scope=str(data.get("scope") or "global"),
            source_step=str(data.get("source_step") or ""),
            target_step=str(data.get("target_step") or ""),
            condition=dict(data.get("condition") or {}),
            mapping=dict(data.get("mapping") or {}),
            policy=dict(data.get("policy") or {}),
            evidence=[str(v) for v in (data.get("evidence") or [])],
            metadata=dict(data.get("metadata") or {}),
        )


def validate_edges(edges: Iterable[GraphEdge]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for edge in edges:
        if edge.edge_id in seen:
            errors.append(f"duplicate edge id: {edge.edge_id}")
        seen.add(edge.edge_id)
        errors.extend(f"{edge.edge_id}: {message}" for message in edge.validate())
    return errors
