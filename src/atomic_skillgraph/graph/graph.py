"""SkillGraph 图结构操作（设计文档 v2.0 §14、§15、§56.2）。

静态图语义：拓扑排序、环检测（除 loop 外非法）、Composite 子图展开、入口/出口识别。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.refs import SkillRef
from ..core.skill_ir import CompositeSkill
from ..core.status import EdgeType

# 参与 Composite 控制流的边类型
_CONTROL_EDGES = {
    EdgeType.NEXT,
    EdgeType.BRANCH,
    EdgeType.PARALLEL,
    EdgeType.FALLBACK,
    EdgeType.RETRY,
    EdgeType.LOOP,
}


@dataclass
class GraphCheck:
    name: str
    passed: bool
    message: str = ""


@dataclass
class GraphCheckReport:
    checks: list[GraphCheck] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors

    def add(self, name: str, passed: bool, message: str = "", *, warning: bool = False) -> None:
        self.checks.append(GraphCheck(name=name, passed=passed, message=message))
        if not passed:
            if warning:
                self.warnings.append(f"[{name}] {message}")
            else:
                self.errors.append(f"[{name}] {message}")


def topo_sort(nodes: list[str], edges: list[list[str]]) -> tuple[list[str] | None, list[list[str]]]:
    """拓扑排序（Kahn）。返回 (有序节点 | 环时 None, 检测到的环)。"""
    indeg: dict[str, int] = {n: 0 for n in nodes}
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    for src, dst in edges:
        if src not in adj or dst not in adj:
            continue
        adj[src].append(dst)
        indeg[dst] += 1
    queue = [n for n in nodes if indeg[n] == 0]
    order: list[str] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for nxt in adj[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    if len(order) != len(nodes):
        cycle_nodes = [n for n in nodes if indeg[n] > 0]
        cycles = _find_cycle(adj, cycle_nodes)
        return None, cycles
    return order, []


def _find_cycle(adj: dict[str, list[str]], candidates: list[str]) -> list[list[str]]:
    """DFS 找环（用于错误报告）。"""
    cycles: list[list[str]] = []
    for start in candidates:
        path: list[str] = []
        visited: set[str] = set()

        def dfs(node: str) -> None:
            if node in path:
                idx = path.index(node)
                cycles.append(path[idx:] + [node])
                return
            if node in visited:
                return
            visited.add(node)
            path.append(node)
            for nxt in adj.get(node, []):
                dfs(nxt)
            path.pop()

        dfs(start)
    # 去重
    unique: list[list[str]] = []
    seen: set[frozenset] = set()
    for cycle in cycles:
        key = frozenset(cycle)
        if key not in seen:
            seen.add(key)
            unique.append(cycle)
    return unique


def composite_node_order(composite: CompositeSkill, registry) -> tuple[list[str], GraphCheckReport]:
    """展开 Composite 的原子节点顺序（按 control 边拓扑排序）。

    返回 (ordered atomic logical_ids, report)。引用中的版本号被剥离用于图结构，
    版本解析由 registry.get_recommended 完成。
    """
    report = GraphCheckReport()
    steps = composite.step_instances()
    step_ids = [step["step_id"] for step in steps]
    by_step = {step["step_id"]: step["node_ref"] for step in steps}
    non_loop = [
        [edge.source_step, edge.target_step]
        for edge in composite.edge_objects()
        if edge.category == "control" and edge.type not in {EdgeType.LOOP, EdgeType.RETRY}
    ]

    order, cycles = topo_sort(step_ids, non_loop)
    if order is None:
        report.add("composite_dag", False, f"control 边存在环：{cycles}")
        # 兜底：按节点声明顺序
        report.add("composite_order_fallback", True, "使用声明顺序（存在控制环）", warning=True)
        return [_logical_of(step["node_ref"]) for step in steps], report
    report.add("composite_dag", True, "")
    return [_logical_of(by_step[step_id]) for step_id in order], report


def composite_step_order(composite: CompositeSkill, registry, *,
                         allow_draft_children: bool = False
                         ) -> tuple[list[dict[str, Any]], GraphCheckReport]:
    """Return exact versioned occurrence steps in executable order.

    The main runtime is sequential today, so complex control edges fail closed.
    Missing exact child versions and ambiguous edge endpoints invalidate the
    whole candidate instead of silently returning a truncated workflow.
    """
    report = GraphCheckReport()
    steps = composite.step_instances()
    by_id = {str(step["step_id"]): dict(step) for step in steps}
    if len(by_id) != len(steps):
        report.add("unique_step_ids", False, "duplicate occurrence step_id")
        return [], report
    for step in steps:
        try:
            ref = SkillRef.parse(str(step.get("node_ref") or ""))
        except ValueError as exc:
            report.add("exact_child_ref", False, str(exc))
            continue
        child = registry.get(ref)
        if child is None:
            report.add("exact_child_ref", False, f"missing exact child: {ref}")
        elif (str(getattr(child, "status", "").value) != "active"
              and not (allow_draft_children
                       and str(getattr(child, "status", "").value) == "draft")):
            report.add("exact_child_status", False,
                       f"child not active: {ref} status={child.status.value}")
    unsupported = [edge for edge in composite.edge_objects()
                   if edge.type in {EdgeType.BRANCH, EdgeType.PARALLEL,
                                    EdgeType.LOOP, EdgeType.RETRY,
                                    EdgeType.FALLBACK}]
    if unsupported:
        report.add("supported_runtime_control", False,
                   "unsupported_runtime_control_edge:" +
                   ",".join(edge.type.value for edge in unsupported))
    ordering = []
    for edge in composite.edge_objects():
        if not edge.source_step or not edge.target_step:
            report.add("occurrence_edge_endpoint", False,
                       f"edge lacks occurrence endpoints: {edge.edge_id}")
            continue
        if edge.source_step not in by_id or edge.target_step not in by_id:
            report.add("occurrence_edge_endpoint", False,
                       f"edge endpoint missing: {edge.source_step}->{edge.target_step}")
            continue
        if edge.type == EdgeType.NEXT:
            ordering.append([edge.source_step, edge.target_step])
    if report.errors:
        return [], report
    order, cycles = topo_sort(list(by_id), ordering)
    if order is None:
        report.add("composite_dag", False, f"control cycle: {cycles}")
        return [], report
    report.add("composite_dag", True)
    return [by_id[step_id] for step_id in order], report


def entry_exit(nodes: list[str], edges: list[list[str]]) -> tuple[list[str], list[str]]:
    """识别入口/出口节点。"""
    indeg: dict[str, int] = {n: 0 for n in nodes}
    outdeg: dict[str, int] = {n: 0 for n in nodes}
    for src, dst in edges:
        if src in indeg and dst in indeg:
            indeg[dst] += 1
            outdeg[src] += 1
    entries = [n for n in nodes if indeg[n] == 0]
    exits = [n for n in nodes if outdeg[n] == 0]
    if not entries:
        entries = nodes[:1]
    if not exits:
        exits = nodes[-1:]
    return entries, exits


def _logical_of(ref: str) -> str:
    """`skill://logical_id@version` / `logical_id@version` -> logical_id。"""
    text = str(ref)
    if text.startswith("skill://"):
        text = text[len("skill://"):]
    return text.rsplit("@", 1)[0]


def strip_version(ref: str) -> str:
    return _logical_of(ref)


def build_control_edges(node_order: list[str]) -> list[list[str]]:
    """顺序链 -> next 边。"""
    return [[node_order[i], node_order[i + 1]] for i in range(len(node_order) - 1)]


def resolve_node_refs(composite: CompositeSkill, registry) -> dict[str, SkillRef]:
    """把 Composite 节点引用解析为 registry 中的推荐版本引用。"""
    resolved: dict[str, SkillRef] = {}
    for ref_text in composite.nodes():
        logical = _logical_of(ref_text)
        obj = registry.get_recommended(logical)
        if obj is not None:
            resolved[logical] = obj.ref
    return resolved
