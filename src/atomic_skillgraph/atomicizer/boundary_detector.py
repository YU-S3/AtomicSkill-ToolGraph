"""候选边界检测（设计文档 v2.0 §25.2、§6）。

- Interactive Environment：成功 Action 边界 + 状态变化 + 目标谓词变化 →
  以「稳定状态转移」切分动作子序列
- Code/Math：顶层函数 + 可达 helper + AST 调用图（helper 由 Miner 进一步处理）
原子性权威标准 = stable Effect + stable I/O + independent validation + reusable
boundary（§25.1），而不是 helper 名 / task_type / action 数。
"""

from __future__ import annotations

import ast
from typing import Any

from ..core.predicates import StateSnapshot, compute_effects
from ..core.trace_ir import TraceRecord
from .effect_extractor import _family_of

# 交互环境：移动到无状态变化的动作（机械步骤）附着到下一个状态转移段
_MOVEMENT_VERBS = {"go", "goto", "look", "inventory", "examine", "open"}


def detect_env_boundaries(trace: TraceRecord) -> list[dict[str, Any]]:
    """按状态转移切分交互轨迹（快照在动作之后记录：snapshot[i+1] 对应 action[i] 之后）。

    返回 segment 列表：
      {name, kind:"env", actions, before, after, params, effect, summary}
    params 为本次实例化的参数绑定（slot -> value），用于 Tool 参数化。
    """
    snapshots = list(trace.state_snapshots)
    actions = [_as_dict(a) for a in trace.actions]
    if not snapshots or not actions:
        return []

    groups: list[dict[str, Any]] = []
    current_before = dict(snapshots[0].get("state") or {})
    current_actions: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        after_index = index + 1
        after = dict(snapshots[after_index].get("state") or {}) \
            if after_index < len(snapshots) else current_before
        current_actions.append(action)
        if _state_changed(current_before, after):
            groups.append({"before": current_before, "after": after,
                           "actions": current_actions})
            current_actions = []
            current_before = after
    # 收尾：无状态变化尾部（例如最后一步已由快照覆盖）丢弃；有未闭合动作则附着最后组
    if current_actions and groups and _state_changed(current_before, groups[-1]["after"]):
        groups[-1]["actions"].extend(current_actions)

    # 机械动作（无状态变化，如 go to）合并到下一个有状态转移的组
    merged: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for group in groups:
        if _is_mechanical(group["actions"]):
            pending.extend(group["actions"])
            group["actions"] = []
        if group["actions"]:
            group["actions"] = pending + group["actions"]
            pending = []
            merged.append(group)
    if pending and merged:
        merged[-1]["actions"].extend(pending)

    # 参数绑定提取（动作中的对象/位置值）
    segments: list[dict[str, Any]] = []
    for index, group in enumerate(merged):
        before, after = group["before"], group["after"]
        before_snap, after_snap = StateSnapshot(before), StateSnapshot(after)
        positive, _negative = compute_effects(before_snap, after_snap)
        # 过滤机械性噪声事实（location_checked / agent_at）
        positive = [e for e in positive
                    if _family_of(e) not in {"location_checked", "agent_at"}]
        params = _extract_action_params(group["actions"])
        effect_names = sorted({_family_of(e) for e in positive if _family_of(e)})
        # 语义命名：单一效果族 → 建议名（heat_object / place_object）
        if len(effect_names) == 1:
            from .effect_extractor import _FACT_FAMILY_NAMES
            name = _FACT_FAMILY_NAMES.get(effect_names[0], (effect_names[0], ""))[0]
        else:
            name = f"segment_{index + 1}"
        segment = {
            "name": name,
            "kind": "env",
            "actions": list(group["actions"]),
            "before": before,
            "after": after,
            "params": params,
            "effect": positive,
            "summary": _summary_from_effects(positive),
        }
        segments.append(segment)
    return segments


def detect_code_boundaries(trace: TraceRecord) -> list[dict[str, Any]]:
    """Code/Math：以入口函数为整体原子段；helper 交由 Miner 提取。"""
    code = trace.candidate_code or _last_passing_code(trace)
    if not code.strip():
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not functions:
        return []
    entry = _infer_entry(trace, functions)
    tests = _collect_tests(trace)
    return [{
        "name": _snake(entry),
        "kind": "code",
        "code": code,
        "entry_point": entry,
        "tests": tests,
        "params": {},
        "before": {"facts": [], "text": "code task start"},
        "after": {"facts": [f"callable_returns_expected({_snake(entry)})"],
                  "text": "tests pass"},
        "effect": [{"predicate": "callable.returns_expected",
                    "args": {"entry_point": _snake(entry)}}],
        "summary": f"函数 {entry} 对给定输入返回期望输出",
    }]


def detect_boundaries(trace: TraceRecord) -> list[dict[str, Any]]:
    if trace.benchmark in ("alfworld", "toy_env") or trace.actions:
        return detect_env_boundaries(trace)
    return detect_code_boundaries(trace)


# ---------------------------------------------------------------------------

_NOISE_PREFIXES = ("location_checked", "agent_at")


def _state_changed(before: dict[str, Any], after: dict[str, Any]) -> bool:
    def meaningful(facts):
        return {str(f) for f in (facts or [])
                if not str(f).startswith(_NOISE_PREFIXES)}
    before_facts = meaningful(before.get("facts") or [])
    after_facts = meaningful(after.get("facts") or [])
    if before_facts != after_facts:
        return True
    return (before.get("inventory") or []) != (after.get("inventory") or [])


def _is_mechanical(actions: list[dict[str, Any]]) -> bool:
    for action in actions:
        name = str(action.get("name", "")).split()[0].lower()
        if name not in _MOVEMENT_VERBS:
            return False
    return bool(actions)


def _extract_action_params(actions: list[dict[str, Any]]) -> dict[str, Any]:
    """从动作参数中提取实例绑定（object / object_location / ...）。"""
    params: dict[str, Any] = {}
    for action in actions:
        for key, value in (action.get("params") or {}).items():
            if key not in params and value not in (None, ""):
                params[key] = value
    return params


def _summary_from_effects(effects: list[dict[str, Any]]) -> str:
    if not effects:
        return "状态转移片段"
    parts = []
    for effect in effects:
        name = str(effect.get("predicate", ""))
        args = effect.get("args") or {}
        parts.append(f"{name}({', '.join(f'{k}={v}' for k, v in args.items())})")
    return "使状态满足：" + "；".join(parts)


def _last_passing_code(trace: TraceRecord) -> str:
    for attempt in reversed(trace.attempts):
        if attempt.passed and attempt.candidate:
            return attempt.candidate
    return trace.candidate_code


def _collect_tests(trace: TraceRecord) -> list[str]:
    tests: list[str] = []
    for attempt in reversed(trace.attempts):
        feedback = attempt.feedback or {}
        for test in (feedback.get("tests") or []):
            if test not in tests:
                tests.append(test)
        if attempt.passed:
            break
    benchmark_tests = (trace.benchmark_result or {}).get("tests") or []
    for test in benchmark_tests:
        if test not in tests:
            tests.append(test)
    return tests


def _infer_entry(trace: TraceRecord, functions: list[ast.FunctionDef]) -> str:
    """从 benchmark_result/attempts 的测试断言推断入口名，否则取最后一个函数。"""
    text = str((trace.benchmark_result or {}).get("executed_code") or "")
    for attempt in reversed(trace.attempts):
        text += "\n" + str((attempt.feedback or {}).get("executed_code") or "")
    import re
    match = re.search(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)
    if match:
        return match.group(1)
    names = {f.name for f in functions}
    for test in _collect_tests(trace):
        match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", test)
        if match and match.group(1) in names:
            return match.group(1)
    return functions[-1].name


def _snake(text: str) -> str:
    import re
    text = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", text)
    return re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_")


def _as_dict(action: Any) -> dict[str, Any]:
    if isinstance(action, dict):
        return action
    if hasattr(action, "to_dict"):
        return action.to_dict()
    return {"name": str(action), "params": {}}
