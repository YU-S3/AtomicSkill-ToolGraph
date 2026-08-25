"""SkillGraph 结构验证（设计文档 v2.0 §56.2）。

- 所有 Skill refs 存在
- contains / implements 合法
- Composite 有入口出口、除 loop 外无非法控制环
- retired 节点不进入新默认路径
- N:M Tool binding 合法（绑定到可用 Tool，1:1 模式时检查）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.edge_ir import DEPENDENCY, GraphEdge
from ..core.refs import SkillRef, ToolRef
from ..core.skill_ir import CompositeSkill, ImplementationAtom
from ..core.status import EdgeType, SkillNodeKind, SkillStatus
from .graph import GraphCheckReport, composite_node_order
from .registry import SkillGraphRegistry


@dataclass
class GraphValidationReport:
    checks: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {"checks": self.checks, "errors": self.errors, "warnings": self.warnings}


def validate_graph(registry: SkillGraphRegistry, tool_registry=None) -> GraphValidationReport:
    report = GraphValidationReport()

    # 1. implements 边
    implements_ok = True
    for edge in registry.iter_edges():
        if edge.get("type") != EdgeType.IMPLEMENTS.value:
            continue
        try:
            source = SkillRef.parse(edge["source"])
        except ValueError:
            report.errors.append(f"implements 边 source 非法：{edge['source']}")
            implements_ok = False
            continue
        impl = registry.get(source)
        if impl is None:
            report.errors.append(f"implements 边 source 不存在：{edge['source']}")
            implements_ok = False
            continue
        if not isinstance(impl, ImplementationAtom):
            report.errors.append(f"implements 边 source 不是 ImplementationAtom：{edge['source']}")
            implements_ok = False
            continue
        abstract = registry.get_recommended(impl.abstract_ref.logical_id)
        if abstract is None:
            report.errors.append(f"Implementation {impl.ref} 的 Abstract 目标不存在：{impl.abstract_ref}")
            implements_ok = False
    report.checks["implements_valid"] = implements_ok

    # Typed edge schema + endpoint integrity.
    edge_schema_ok = True
    for edge in registry.edge_objects():
        edge_errors = edge.validate()
        if edge_errors:
            edge_schema_ok = False
            report.errors.extend(f"Edge {edge.edge_id}: {error}" for error in edge_errors)
        if edge.type in DEPENDENCY:
            continue  # dependency targets may be environment/schema/permission URIs
        if edge.scope == "runtime":
            continue
        for endpoint_name, endpoint in (("source", edge.source), ("target", edge.target)):
            try:
                ref = SkillRef.parse(endpoint)
            except ValueError:
                edge_schema_ok = False
                report.errors.append(
                    f"Edge {edge.edge_id} {endpoint_name} 不是固定 SkillRef：{endpoint}")
                continue
            if registry.get(ref) is None:
                edge_schema_ok = False
                report.errors.append(
                    f"Edge {edge.edge_id} {endpoint_name} 不存在：{endpoint}")
    report.checks["edge_schema_valid"] = edge_schema_ok

    # 2. contains 边 + Composite 结构
    for composite in registry.list_by_kind(SkillNodeKind.COMPOSITE):
        report = _check_composite(composite, registry, report)

    # 3. Tool binding 合法性（N:M 绑定检查）
    if tool_registry is not None:
        for impl in registry.list_by_kind(SkillNodeKind.IMPLEMENTATION_ATOMIC):
            for binding in impl.tool_bindings:
                tool = tool_registry.get(binding.tool_ref)
                if tool is None:
                    report.errors.append(
                        f"Implementation {impl.ref} 绑定到不存在的 Tool：{binding.tool_ref}")
                    continue
                if tool.status.value in ("suppressed", "retired", "shadow", "draft"):
                    report.warnings.append(
                        f"Implementation {impl.ref} 绑定了不可直接调用的 Tool 状态"
                        f" {tool.status.value}：{binding.tool_ref}")

    report.checks["graph_valid"] = report.passed
    return report


def _check_composite(composite: CompositeSkill, registry: SkillGraphRegistry,
                     report: GraphValidationReport) -> GraphValidationReport:
    node_refs = composite.nodes()
    for ref_text in node_refs:
        try:
            ref = SkillRef.parse(str(ref_text))
            obj = registry.get(ref) or registry.get_recommended(ref.logical_id)
        except ValueError:
            obj = None
        if obj is None:
            report.errors.append(f"Composite {composite.ref} 引用不存在节点：{ref_text}")
            continue
        if obj.status == SkillStatus.RETIRED:
            report.errors.append(f"Composite {composite.ref} 引用已退役节点：{ref_text}")

    order, sub_report = composite_node_order(composite, registry)
    for err in sub_report.errors:
        report.errors.append(f"Composite {composite.ref}: {err}")
    for warn in sub_report.warnings:
        report.warnings.append(f"Composite {composite.ref}: {warn}")
    report.checks[f"composite_{composite.ref.logical_id}_order"] = not sub_report.errors
    return report
