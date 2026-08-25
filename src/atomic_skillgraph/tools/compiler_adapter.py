"""Tool Candidate Miner（设计文档 v2.0 §25.4、§27、§47.5）。

- Code 类：AST 提取 helper / 顶层函数 → python_callable Skeleton（FlowEvo
  PrimitiveCompiler 的角色，即 "Code Tool Candidate Miner"，而非 Atomic Skill 定义器）
- Interactive 类：动作子序列参数化 → action_template Skeleton
- 一次成功即可产生 Skeleton；Skeleton 必须经 admission 才能成为 candidate（§27.1）
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from ..core.refs import ToolRef, content_hash
from ..core.status import ArtifactKind, ToolLifecycle
from ..core.tool_ir import ToolAsset
from ..core.trace_ir import TraceRecord
from ..atomicizer.effect_extractor import parameterize_predicates

_SLOT_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class AtomicSegment:
    """原子化后的成功轨迹片段（Trace Atomicizer 输出）。"""

    name: str                       # 段名（如 acquire_object / normalize_values）
    kind: str                       # "code" | "env"
    task_type: str = ""
    trace_id: str = ""
    actions: list[dict[str, Any]] = field(default_factory=list)
    code: str = ""
    entry_point: str = ""
    tests: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)  # slot -> 本次绑定值
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)
    effect: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    entry_event_index: int | None = None
    causal_event_indices: list[int] = field(default_factory=list)
    effect_producer_indices: list[int] = field(default_factory=list)
    event_slice_validated: bool = False
    replay_safe: bool = True
    replay_prefix_actions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "task_type": self.task_type,
            "trace_id": self.trace_id,
            "actions": self.actions,
            "code": self.code,
            "entry_point": self.entry_point,
            "tests": self.tests,
            "params": self.params,
            "before": self.before,
            "after": self.after,
            "effect": self.effect,
            "summary": self.summary,
            "entry_event_index": self.entry_event_index,
            "causal_event_indices": self.causal_event_indices,
            "effect_producer_indices": self.effect_producer_indices,
            "event_slice_validated": self.event_slice_validated,
            "replay_safe": self.replay_safe,
            "replay_prefix_actions": self.replay_prefix_actions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AtomicSegment":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _safe_tool_id(prefix: str, name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_.-]+", "-", str(name).lower()).strip("-")
    return f"{prefix}.{cleaned or 'tool'}"[:120]


def _normalize_segments(segments: list[Any]) -> list[AtomicSegment]:
    return [s if isinstance(s, AtomicSegment) else AtomicSegment.from_dict(s)
            for s in segments]


def mine_code_tools(trace: TraceRecord, segments: list[AtomicSegment],
                    *, enable_primitive_reuse: bool = True) -> list[ToolAsset]:
    """从成功代码轨迹挖掘 python_callable Tool Skeleton。

    主入口函数 → 主 Tool；可达 helper（AST 调用图）→ helper 级 Tool 候选
    （PrimitiveCompiler 角色）。一个 Segment 可产出 1+N 个 Skeleton。
    """
    segments = _normalize_segments(segments)
    tools: list[ToolAsset] = []
    for segment in segments:
        if segment.kind != "code" or not segment.code.strip():
            continue
        try:
            tree = ast.parse(segment.code)
        except SyntaxError:
            continue
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if not functions:
            continue
        entry = segment.entry_point or functions[-1].name
        entry_node = next((f for f in functions if f.name == entry), functions[-1])

        # 主入口 Tool
        main_tool = _build_code_tool(segment, entry_node.name, trace)
        tools.append(main_tool)

        # helper 级 Tool 候选（AST 调用图可达 helper）
        if enable_primitive_reuse:
            reachable = _reachable_helpers(tree, entry_node)
            for helper in reachable[:2]:
                helper_code = ast.unparse(ast.Module(body=[helper], type_ignores=[]))
                helper_segment = AtomicSegment(
                    name=f"{segment.name}.{helper.name}",
                    kind="code",
                    task_type=segment.task_type,
                    trace_id=segment.trace_id,
                    code=helper_code,
                    entry_point=helper.name,
                    tests=[],
                    params=segment.params,
                    before=segment.before,
                    after=segment.after,
                    summary=f"helper-level executable primitive: {helper.name}",
                )
                tools.append(_build_code_tool(helper_segment, helper.name, trace))
    return tools


def _build_code_tool(segment: AtomicSegment, entry_point: str, trace: TraceRecord) -> ToolAsset:
    params = _signature_from_ast(segment.code, entry_point)
    replay_case = {
        "kind": "replay",
        "entry_point": entry_point,
        "source_trace_id": trace.trace_id,
        "tests": segment.tests,
    }
    name = segment.name
    if "." in name:
        name = name.replace(".", "-")
    # tool_id 带代码哈希：同构代码去重（重复实例 → 同一 Tool），不同常量 → 独立
    # specialised Tool（供 Global Generalizer 泛化，§30.6）。
    code_key = hashlib.sha256(segment.code.strip().encode("utf-8")).hexdigest()[:8]
    tool = ToolAsset(
        ref=ToolRef(
            tool_id=_safe_tool_id(_bench_prefix(trace.benchmark), f"{name}-{code_key}"),
            version="0.1.0",
        ),
        artifact_kind=ArtifactKind.PYTHON_CALLABLE,
        summary=segment.summary or f"Executable tool for atomic effect: {segment.name}",
        signature={"entry_point": entry_point, "parameters": params},
        interface={"inputs": params, "outputs": [{"name": "result"}]},
        artifact={"code": segment.code},
        tests=[replay_case] if segment.tests else [],
        safety={"direct_execution_allowed": True, "checks_passed": []},
        provenance={
            "source_trace_ids": [trace.trace_id],
            "source_task_types": [trace.task_type],
            "extraction_method": "code_atomicizer_ast",
        },
        statistics={"support_count": 1, "call_count": 0, "success_count": 0,
                    "failure_count": 0, "utility": 0.5},
        lineage={"generalized_from": [], "specialized_from": [], "supersedes": None},
        status=ToolLifecycle.DRAFT,
    )
    return tool


def _bench_prefix(benchmark: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", str(benchmark).lower()).strip("-")
    return cleaned or "generic"


def _signature_from_ast(code: str, entry_point: str) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            args = node.args
            names = [a.arg for a in args.args]
            if args.vararg:
                names.append(f"*{args.vararg.arg}")
            defaults_offset = len(names) - len(args.defaults) if args.defaults else len(names)
            return [
                {"name": name, "required": i < defaults_offset}
                for i, name in enumerate(names)
            ]
    return []


def _reachable_helpers(tree: ast.Module, entry: ast.FunctionDef) -> list[ast.FunctionDef]:
    """以入口函数为根的调用图 DFS，收集可达 helper。"""
    called = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    direct: set[str] = set()
    for node in ast.walk(entry):
        if isinstance(node, ast.Call):
            target = _call_name(node.func)
            if target and target in called and target != entry.name:
                direct.add(target)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    ordered: list[ast.FunctionDef] = []
    seen: set[str] = set()

    def dfs(name: str) -> None:
        if name in seen or name not in functions:
            return
        seen.add(name)
        node = functions[name]
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                target = _call_name(child.func)
                if target in functions and target != name:
                    dfs(target)
        ordered.append(node)

    for name in sorted(direct):
        dfs(name)
    return ordered


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


# ---------------------------------------------------------------------------
# Interactive：动作模板 Skeleton
# ---------------------------------------------------------------------------

def mine_action_template_tools(trace: TraceRecord, segments: list[AtomicSegment]) -> list[ToolAsset]:
    """把交互环境的原子段动作子序列参数化为 action_template Skeleton（§53）。"""
    segments = _normalize_segments(segments)
    tools: list[ToolAsset] = []
    for segment in segments:
        if segment.kind != "env" or not segment.actions:
            continue
        steps, slots = _parameterize_actions(segment.actions, segment.params)
        if not steps:
            continue
        # 清理：无槽位/无信息步骤是实例特有探索 → 移入 replay 前缀；
        # 连续重复去重。Tool 本体只保留符号化有效步骤。
        core_steps, extra_prefix = _clean_template_steps(steps)
        if not core_steps:
            continue
        parameters = [
            {"name": slot, "semantic_type": _guess_semantic_type(slot), "required": True}
            for slot in sorted(slots)
        ]
        # One Atomic capability can have multiple valid executable shapes
        # (Acquire from an open surface vs. go→open→take from a cabinet).  Give
        # each normalized shape a stable identity so a later occurrence cannot
        # overwrite or receive support for a different template body.
        shape_key = content_hash({
            "steps": core_steps,
            "parameters": parameters,
            "expected_effects": parameterize_predicates(
                segment.effect, segment.params),
        })[:10]
        # 源轨迹前缀动作（字面值）：admission replay 从任务初始状态开始，
        # 需先重放前缀以达成段落的前置状态（§28.2 source trace replay）
        prefix = _trace_prefix_actions(trace, segment) + extra_prefix
        replay_case = {
            "kind": "replay",
            "bindings": segment.params,
            "steps": core_steps,
            "before": segment.before,
            "after": segment.after,
            "expected_effects": parameterize_predicates(
                segment.effect, segment.params),
            "prefix": prefix,
            "source_trace_id": trace.trace_id,
            # 交互环境 replay 需要源任务定位（reset_to_task / 重建世界）
            "source": {
                "task_id": trace.task_id,
                "env_index": (trace.provenance or {}).get("env_index"),
            },
        }
        tool = ToolAsset(
            ref=ToolRef(
                tool_id=_safe_tool_id(
                    _bench_prefix(trace.benchmark), f"{segment.name}-{shape_key}"),
                version="0.1.0",
            ),
            artifact_kind=ArtifactKind.ACTION_TEMPLATE,
            # LLM rationale may mention the source object ("heat this cup").
            # Tool identity/summary must describe the reusable capability.
            summary=f"Action template for reusable atomic capability: {segment.name}",
            signature={"parameters": parameters},
            interface={
                "inputs": {p["name"]: p["semantic_type"] for p in parameters},
                "outputs": {"effect": [str(e) for e in parameterize_predicates(
                    segment.effect, segment.params)]},
            },
            # Persist the cleaned symbolic body.  ``steps`` still contains
            # source-task exploration that was deliberately moved to the
            # replay-only prefix above; writing it here defeats the cleaner and
            # makes Direct reuse execute irrelevant locations.
            artifact={"template": "\n".join(core_steps), "steps": core_steps},
            tests=[replay_case],
            safety={"direct_execution_allowed": True, "checks_passed": []},
            provenance={
                "source_trace_ids": [trace.trace_id],
                "source_task_types": [trace.task_type],
                "extraction_method": "trace_action_parameterizer",
            },
            statistics={"support_count": 1, "call_count": 0, "success_count": 0,
                        "failure_count": 0, "utility": 0.5},
            lineage={"generalized_from": [], "specialized_from": [], "supersedes": None},
            status=ToolLifecycle.DRAFT,
        )
        tools.append(tool)
    return tools


def _clean_template_steps(steps: list[str]) -> tuple[list[str], list[str]]:
    """模板步骤清理：返回 (核心符号化步骤, 探索前缀步骤)。

    - 不含 {槽位} 的步骤 / look / inventory：实例特有探索 → 前缀（仅供 replay）
    - 连续重复去重
    """
    core: list[str] = []
    prefix: list[str] = []
    for step in steps:
        stripped = str(step).strip()
        lowered = stripped.lower()
        if lowered in ("inventory", "look") or not re.search(r"\{[^}]+\}", stripped):
            prefix.append(stripped)
            continue
        if core and core[-1] == stripped:
            continue
        core.append(stripped)
    return core, prefix


def _trace_prefix_actions(trace: TraceRecord, segment: AtomicSegment) -> list[str]:
    """源轨迹中该段之前的所有动作文本（按执行顺序，字面值）。"""
    if segment.replay_prefix_actions:
        return [str(action.get("action") or action.get("name") or "").strip()
                for action in segment.replay_prefix_actions
                if str(action.get("action") or action.get("name") or "").strip()
                and str(action.get("action") or action.get("name") or "").lower()
                not in ("inventory", "look")]
    segment_steps = {int(a.get("step", 0)) for a in segment.actions}
    if not segment_steps:
        return []
    first_step = (int(segment.entry_event_index)
                  if segment.entry_event_index is not None else min(segment_steps))
    prefix: list[str] = []
    for action in trace.actions:
        action_dict = action.to_dict() if hasattr(action, "to_dict") else dict(action)
        step = int(action_dict.get("step", 0))
        if step < first_step:
            name = str(action_dict.get("name", "")).strip()
            if name and name.lower() not in ("inventory", "look"):
                prefix.append(name)
    return prefix


def _parameterize_actions(actions: list[dict[str, Any]],
                          params: dict[str, Any]) -> tuple[list[str], set[str]]:
    """把动作序列参数化：动作文本中的实例常量替换为 {slot} 占位符。

    动作文本（如 "take tomato 1 from countertop 1"）中的参数值按词边界替换；
    值相同的参数（如 go-to 与 take 的同一位置）归一为同一 slot。
    """
    value_to_slot: dict[str, str] = {}
    slots: set[str] = set()
    for slot, value in params.items():
        key = _norm_value(value)
        if key and key not in value_to_slot:
            value_to_slot[key] = slot

    steps: list[str] = []
    for action in actions:
        text = str(action.get("name", "")).strip()
        if not text:
            continue
        for value, slot in _slot_candidates(params, value_to_slot):
            replaced = _replace_value(text, value, slot)
            if replaced != text:
                slots.add(slot)
            text = replaced
        # 残留的参数追加（文本中未出现对应值时保留 key=value 提示）
        for key, value in (action.get("params") or {}).items():
            if _norm_value(value) not in value_to_slot:
                text = f"{text} {key}={value}".strip()
        steps.append(text)
    return steps, slots


def _slot_candidates(params: dict[str, Any],
                     value_to_slot: dict[str, str]) -> list[tuple[Any, str]]:
    """(value, slot) 候选，按值长度降序（先替换长值避免部分匹配）。"""
    candidates = [(value, slot) for slot, value in params.items()
                  if _norm_value(value) in value_to_slot]
    candidates.sort(key=lambda item: len(str(item[0])), reverse=True)
    return candidates


def _replace_value(text: str, value: Any, slot: str) -> str:
    raw = str(value).strip()
    # ALFWorld 在结构化参数中使用 mug_1，而动作文本使用 mug 1。两者必须
    # 命中同一个实例值，否则模板会表面声明 {object}，正文却残留 mug 1。
    tokens = [re.escape(token) for token in re.split(r"[_\s]+", raw) if token]
    flexible = r"(?:[_\s]+)".join(tokens)
    patterns = sorted({re.escape(raw), re.escape(_norm_value(value)), flexible},
                      key=len, reverse=True)
    for pattern in patterns:
        text = re.sub(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", f"{{{slot}}}", text,
                      flags=re.IGNORECASE)
    return text


def _norm_value(value: Any) -> str:
    return re.sub(r"\s+", "_", str(value).strip().lower())


def _guess_semantic_type(slot: str) -> str:
    lowered = slot.lower()
    if "location" in lowered or "recep" in lowered or lowered in ("sink", "shelf", "countertop"):
        return "location_ref"
    if "station" in lowered or "heating" in lowered or "cooling" in lowered:
        return "station_ref"
    return "object_ref"
