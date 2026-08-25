"""Global Tool Generalizer（设计文档 v2.0 §46、附录 C）。

运行时不按 task_type 分仓；`task_type` 只作为 provenance feature（§46.4）。

判断动作（§46.3）：
    完全行为等价           -> merge_tools
    仅实例常量不同         -> generalize_tool
    特殊输入需要单独逻辑   -> specialize_tool
    一个 Tool 多个核心效果 -> split_tool（由 failure cluster / effect 证据触发）
    没有足够证据           -> keep separate

判据（§31.3）：InterfaceCompatible ∧ EffectCompatible ∧ Parameterizable ∧ ReplayConsistent；
新 generalized Tool 必须在来源实例 replay 通过后才能进入 admission（§31.3、附录 C）。
"""

from __future__ import annotations

import ast
import copy
import re
from dataclasses import dataclass, field
from typing import Any

from ..core.refs import ToolRef, bump_version, content_hash
from ..core.status import ArtifactKind, ToolLifecycle
from ..core.tool_ir import ToolAsset
from .admission_adapter import AdmissionEngine
from .registry import ToolRegistry


@dataclass
class ToolGroup:
    """结构相似 / 行为等价的 Tool 组。"""

    tools: list[ToolAsset] = field(default_factory=list)
    relation: str = "similar"          # equivalent | parameterizable | unrelated

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation": self.relation,
            "tools": [str(t.ref) for t in self.tools],
        }


@dataclass
class EvolutionAction:
    kind: str                            # merge | generalize | specialize | split | none
    tools: list[str] = field(default_factory=list)
    proposed: ToolAsset | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "tools": self.tools,
            "proposed": str(self.proposed.ref) if self.proposed else None,
            "reason": self.reason,
        }


class ToolGeneralizer:
    """跨 task type 的全局 Tool 泛化/合并维护器。"""

    def __init__(self, registry: ToolRegistry, admission: AdmissionEngine | None = None,
                 min_group: int = 2, sandbox=None) -> None:
        self.registry = registry
        self.admission = admission or AdmissionEngine(
            existing_hashes={t.structural_hash() for t in registry.list_all()})
        self.min_group = min_group
        from .sandbox import Sandbox
        self.sandbox = sandbox or Sandbox()

    # ------------------------------------------------------------------
    def find_groups(self) -> list[ToolGroup]:
        """按 (artifact_kind, 参数数, 入口点) 粗分组，再做结构相似性判断。"""
        usable = self.registry.list_usable()
        buckets: dict[tuple[str, int, str], list[ToolAsset]] = {}
        for tool in usable:
            key = (tool.artifact_kind.value, len(tool.param_names()), tool.entry_point())
            buckets.setdefault(key, []).append(tool)

        groups: list[ToolGroup] = []
        for bucket in buckets.values():
            if len(bucket) < self.min_group:
                continue
            # 两两比较
            assigned: set[str] = set()
            for i, base in enumerate(bucket):
                if str(base.ref) in assigned:
                    continue
                group = [base]
                for other in bucket[i + 1:]:
                    if str(other.ref) in assigned:
                        continue
                    relation = self._relation(base, other)
                    if relation != "unrelated":
                        group.append(other)
                        assigned.add(str(other.ref))
                if len(group) >= self.min_group:
                    groups.append(ToolGroup(tools=group, relation="similar"))
        return groups

    def _relation(self, a: ToolAsset, b: ToolAsset) -> str:
        if a.structural_hash() == b.structural_hash():
            return "equivalent"
        if a.artifact_kind != b.artifact_kind:
            return "unrelated"
        if a.artifact_kind == ArtifactKind.PYTHON_CALLABLE:
            if _ast_diff_score(a.artifact_body(), b.artifact_body()) >= 0.7:
                return "parameterizable"
        else:
            if _template_shape(a) == _template_shape(b):
                return "parameterizable"
        return "unrelated"

    # ------------------------------------------------------------------
    def merge_duplicates(self) -> list[EvolutionAction]:
        """行为等价 → merge：保留 utility 最优者，其余 shadow + lineage.merged_from。"""
        actions: list[EvolutionAction] = []
        for group in self.find_groups():
            equivalents = [t for t in group.tools
                           if all(self._relation(t, o) == "equivalent" for o in group.tools if o.ref != t.ref)]
            if len(equivalents) < 2:
                continue
            best = max(equivalents, key=lambda t: (t.utility(), t.status.value))
            for tool in equivalents:
                if tool.ref == best.ref:
                    continue
                try:
                    merged = self.registry.get(tool.ref)
                    if merged is None:
                        continue
                    merged.status = ToolLifecycle.SHADOW
                    merged.lineage["merged_from"] = list(merged.lineage.get("merged_from") or [])
                    self.registry.set_status(tool.ref, ToolLifecycle.SHADOW)
                    actions.append(EvolutionAction(kind="merge", tools=[str(tool.ref), str(best.ref)],
                                                   reason="duplicate_equivalence"))
                except ValueError:
                    continue
        return actions

    # ------------------------------------------------------------------
    def propose_generalized(self, group: ToolGroup) -> ToolAsset | None:
        """仅实例常量不同 → parameterize。

        规则版实现：AST 同构位置的常量 → 模板参数符号（如 K_FACTOR），
        replay 用例按来源实例绑定参数值（parameterized_replay）。
        真实实验可用 LLM 增强（保留接口，不改变 admission 门禁）。
        """
        tools = [t for t in group.tools if t.artifact_kind == ArtifactKind.PYTHON_CALLABLE]
        if len(tools) < self.min_group:
            return None
        base = tools[0]
        code, symbol, bindings = _parameterize_constants(tools)
        if code is None or not symbol or not bindings:
            return None
        entry = base.entry_point()
        parameterized_replay = {
            "kind": "parameterized_replay",
            "entry_point": entry,
            "template_parameter": symbol,
            "bindings": bindings,
        }
        tool = ToolAsset(
            ref=ToolRef(base.tool_id, bump_version(
                (self.registry.get_latest(base.tool_id) or base).ref.version, "minor")),
            artifact_kind=ArtifactKind.PYTHON_CALLABLE,
            summary=f"Generalized tool parameterized over: {symbol} (from {len(tools)} variants)",
            signature={"entry_point": entry,
                       "parameters": [{"name": p, "required": True}
                                      for p in base.param_names()],
                       "template_parameters": [symbol]},
            interface={"inputs": base.param_names(), "outputs": [{"name": "result"}]},
            artifact={"code": code, "template_parameters": [symbol]},
            tests=[parameterized_replay],
            safety={"direct_execution_allowed": False,
                    "checks_passed": ["parameterized_replay"],
                    "note": "direct 执行需先绑定 template_parameter（v2.0 经 seeded 注入使用）"},
            provenance={
                "source_trace_ids": [tid for t in tools for tid in t.provenance.get("source_trace_ids", [])],
                "source_task_types": list({tt for t in tools for tt in t.provenance.get("source_task_types", [])}),
                "extraction_method": "tool_generalizer_ast",
            },
            statistics={"support_count": len(tools), "call_count": 0, "success_count": 0,
                        "failure_count": 0, "utility": 0.5},
            lineage={"generalized_from": [str(t.ref) for t in tools],
                     "specialized_from": [], "supersedes": None},
            status=ToolLifecycle.DRAFT,
        )
        return tool

    def propose_specialized(self, source: ToolAsset,
                            constraints: dict[str, Any], reason: str) -> ToolAsset:
        """为稳定的特殊输入域创建独立候选；不直接修改来源 Tool。"""
        slug = content_hash(constraints)[:8]
        data = copy.deepcopy(source.to_dict())
        data["tool_id"] = f"{source.tool_id}.specialized.{slug}"
        data["version"] = "1.0.0"
        data["summary"] = f"{source.summary} [specialized: {reason}]"
        data["status"] = ToolLifecycle.DRAFT.value
        data["statistics"] = {"support_count": 1, "call_count": 0,
                              "success_count": 0, "failure_count": 0,
                              "utility": 0.5}
        data["lineage"] = {**dict(source.lineage),
                           "specialized_from": [str(source.ref)],
                           "specialization_constraints": constraints}
        data["provenance"] = {**dict(source.provenance),
                              "extraction_method": "tool_specializer_rules"}
        return ToolAsset.from_dict(data)

    def propose_split(self, source: ToolAsset,
                      partitions: list[dict[str, Any]]) -> list[ToolAsset]:
        """按已形成的独立效果/步骤簇拆分 Tool；每个子 Tool 单独 admission。"""
        children: list[ToolAsset] = []
        source_steps = list(source.artifact.get("steps") or [])
        for index, partition in enumerate(partitions):
            indices = [int(value) for value in partition.get("step_indices") or []]
            steps = [source_steps[i] for i in indices if 0 <= i < len(source_steps)]
            if source.artifact_kind == ArtifactKind.ACTION_TEMPLATE and not steps:
                continue
            data = copy.deepcopy(source.to_dict())
            suffix = re.sub(r"[^a-z0-9]+", "_",
                            str(partition.get("name") or f"part_{index}").lower()).strip("_")
            data["tool_id"] = f"{source.tool_id}.split.{suffix or index}"
            data["version"] = "1.0.0"
            data["summary"] = str(partition.get("summary") or
                                  f"Split part {index + 1} of {source.summary}")
            if steps:
                data["artifact"] = {**dict(source.artifact), "steps": steps}
            data["tests"] = list(partition.get("tests") or source.tests)
            data["status"] = ToolLifecycle.DRAFT.value
            data["statistics"] = {"support_count": 1, "call_count": 0,
                                  "success_count": 0, "failure_count": 0,
                                  "utility": 0.5}
            data["lineage"] = {**dict(source.lineage),
                               "split_from": str(source.ref),
                               "split_effects": list(partition.get("effects") or [])}
            children.append(ToolAsset.from_dict(data))
        return children

    def admit_evolution(self, candidates: list[ToolAsset], kind: str,
                        source_refs: list[str]) -> list[EvolutionAction]:
        """统一 admission 门：specialize/split 也不能绕过 replay 与安全检查。"""
        actions: list[EvolutionAction] = []
        for candidate in candidates:
            result = self.admission.admit(candidate)
            if result.passed:
                self.registry.register(candidate)
                actions.append(EvolutionAction(kind=kind, tools=source_refs,
                                               proposed=candidate,
                                               reason="admitted_candidate"))
            else:
                candidate.status = ToolLifecycle.SHADOW
                try:
                    self.registry.register(candidate)
                except ValueError:
                    pass
                actions.append(EvolutionAction(kind=kind, tools=source_refs,
                                               reason=f"admission_rejected: {result.reasons[:2]}"))
        return actions

    # ------------------------------------------------------------------
    def run_maintenance(self) -> list[EvolutionAction]:
        """执行一轮全局维护：merge → generalize（admission 内含 replay 门禁）。

        Replay 门禁（§31.3 ReplayConsistent）：parameterized_replay 用例在
        admission 中逐绑定执行；任何来源实例 replay 失败 → shadow，不激活。
        同一组工具已有成功泛化产物时跳过（避免重复版本堆积）。
        """
        actions: list[EvolutionAction] = []
        actions.extend(self.merge_duplicates())
        for group in self.find_groups():
            group_refs = {str(t.ref) for t in group.tools}
            # 已泛化 guard
            family_versions = [
                candidate
                for source in group.tools
                for candidate in self.registry.iter_all_versions(source.tool_id)
            ]
            already = any(
                set(t.lineage.get("generalized_from") or []) == group_refs
                and t.status in (ToolLifecycle.CANDIDATE, ToolLifecycle.ACTIVE,
                                 ToolLifecycle.PREFERRED)
                for t in family_versions
            )
            if already:
                actions.append(EvolutionAction(kind="generalize",
                                               tools=[str(t.ref) for t in group.tools],
                                               reason="already_generalized"))
                continue
            generalized = self.propose_generalized(group)
            if generalized is None:
                continue
            result = self.admission.admit(generalized)
            if result.passed:
                self.registry.register(generalized)
                actions.append(EvolutionAction(kind="generalize",
                                               tools=[str(t.ref) for t in group.tools],
                                               proposed=generalized,
                                               reason="admitted_candidate"))
            else:
                generalized.status = ToolLifecycle.SHADOW
                try:
                    self.registry.register(generalized)
                except ValueError:
                    pass
                actions.append(EvolutionAction(kind="generalize",
                                               tools=[str(t.ref) for t in group.tools],
                                               reason=f"replay_or_admission_rejected: {result.reasons[:2]}"))
        return actions


def _merge_replay_tests(tools: list[ToolAsset]) -> list[str]:
    tests: list[str] = []
    seen: set[str] = set()
    for tool in tools:
        for case in tool.replay_cases():
            for test in case.get("tests") or []:
                if test not in seen:
                    seen.add(test)
                    tests.append(test)
    return tests


# ---------------------------------------------------------------------------
# AST 同构参数化（规则版；真实实验可用 LLM 增强，此处保持无 API 可运行）
# ---------------------------------------------------------------------------

def _ast_diff_score(code_a: str, code_b: str) -> float:
    """规范化 AST 同构评分：在相同位置的子树中，可替换节点（字面量/名字）占比。"""
    try:
        tree_a, tree_b = ast.parse(code_a), ast.parse(code_b)
    except SyntaxError:
        return 0.0
    nodes_a = [n for n in ast.walk(tree_a) if isinstance(n, (ast.FunctionDef, ast.Call, ast.Return, ast.Assign, ast.Expr, ast.BinOp))]
    nodes_b = [n for n in ast.walk(tree_b) if isinstance(n, (ast.FunctionDef, ast.Call, ast.Return, ast.Assign, ast.Expr, ast.BinOp))]
    if not nodes_a or not nodes_b:
        return 0.0
    matched = 0
    for a, b in zip(nodes_a, nodes_b):
        if type(a) is type(b) and _shape(a) == _shape(b):
            matched += 1
    return matched / max(len(nodes_a), len(nodes_b))


def _shape(node: ast.AST) -> str:
    if isinstance(node, ast.FunctionDef):
        return f"def:{len(node.args.args)}"
    if isinstance(node, ast.Call):
        return f"call:{len(node.args)}"
    if isinstance(node, ast.BinOp):
        return f"binop:{type(node.op).__name__}"
    if isinstance(node, ast.Return):
        return "return"
    if isinstance(node, ast.Assign):
        return f"assign:{len(node.targets)}"
    if isinstance(node, ast.Expr):
        return "expr"
    return type(node).__name__


def _parameterize_constants(tools: list[ToolAsset]) -> tuple[str | None, str, list[dict[str, Any]]]:
    """AST 同构位置常量 → 模板参数符号 + 每来源绑定值。

    返回 (新代码, 参数符号, bindings=[{value, tests}...])；无法安全参数化时
    返回 (None, "", [])。只参数化常量（变量名差异不做——需 LLM 语义判断）。
    """
    codes = [t.artifact_body() for t in tools]
    if len(codes) < 2:
        return None, "", []
    try:
        trees = [ast.parse(code) for code in codes]
    except SyntaxError:
        return None, "", []
    functions = [_first_function(tree) for tree in trees]
    if any(fn is None for fn in functions):
        return None, "", []

    base_fn = functions[0]
    # path -> {tool_ref: literal_repr}
    constant_positions: dict[str, dict[str, str]] = {}
    for tool, fn in zip(tools, functions):
        _collect_constants(fn, [], constant_positions, str(tool.ref))

    # 位置参数化：至少两个来源取值不同
    parameterized: dict[str, str] = {}
    symbol_index = 0
    for path, per_source in sorted(constant_positions.items()):
        if len(per_source) < 2 or len(set(per_source.values())) <= 1:
            continue
        parameterized[path] = f"K_PARAM_{symbol_index}"
        symbol_index += 1
    if not parameterized:
        return None, "", []

    # base AST 对应位置替换为 Name(symbol)
    for path_text, symbol in parameterized.items():
        path = [int(x) for x in path_text.split(",")]
        replaced = _replace_constant_at(base_fn, path, symbol)
        if not replaced:
            return None, "", []
    new_code = ast.unparse(base_fn)

    # 每来源绑定：该来源在此位置的常量值 + 其 replay 测试
    bindings: list[dict[str, Any]] = []
    for tool in tools:
        per_source = {path: constant_positions.get(path, {}).get(str(tool.ref))
                      for path in parameterized}
        merged: dict[str, Any] = {}
        for path, symbol in parameterized.items():
            merged[symbol] = _parse_literal(per_source[path])
        tests: list[str] = []
        for case in tool.replay_cases():
            tests.extend(case.get("tests") or [])
        bindings.append({"value": merged, "tests": tests})
    symbol = next(iter(parameterized.values()))
    return new_code, symbol, bindings


def _collect_constants(node: ast.AST, path: list[int],
                       out: dict[str, dict[str, str]], source_key: str) -> None:
    """收集常量节点（str/int/float），path 为 ast.iter_child_nodes 索引路径。"""
    for index, child in enumerate(ast.iter_child_nodes(node)):
        child_path = path + [index]
        if isinstance(child, ast.Constant) and isinstance(child.value, (str, int, float)) \
                and not isinstance(child.value, bool):
            out.setdefault(_path_key(child_path), {})[source_key] = repr(child.value)
        _collect_constants(child, child_path, out, source_key)


def _replace_constant_at(root: ast.AST, path: list[int], symbol: str) -> bool:
    """沿索引路径定位父节点，把 path 末位的常量替换为 Name(symbol)。"""
    parent = root
    for index in path[:-1]:
        children = list(ast.iter_child_nodes(parent))
        if index >= len(children):
            return False
        parent = children[index]
    index = path[-1]
    replacement = ast.Name(id=symbol, ctx=ast.Load())
    seen = 0
    for field, value in ast.iter_fields(parent):
        if isinstance(value, list):
            if seen + len(value) > index:
                slot = index - seen
                if slot < len(value) and isinstance(value[slot], ast.Constant):
                    value[slot] = ast.copy_location(replacement, value[slot])
                    return True
                return False
            seen += len(value)
        elif isinstance(value, ast.AST):
            if seen == index:
                setattr(parent, field, ast.copy_location(replacement, value))
                return True
            seen += 1
    return False


def _path_key(path: list[int]) -> str:
    return ",".join(str(i) for i in path)


def _parse_literal(text: Any) -> Any:
    if text is None:
        return None
    try:
        return ast.literal_eval(str(text))
    except (ValueError, SyntaxError):
        return str(text)


def _first_function(tree: ast.Module) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node
    return None


def _template_shape(tool: ToolAsset) -> str:
    steps = tool.artifact.get("steps") or []
    return "|".join(re.sub(r"\{[^}]+\}", "{}", str(step)) for step in steps)
