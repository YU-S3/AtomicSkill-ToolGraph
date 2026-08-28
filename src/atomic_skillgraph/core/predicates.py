"""状态谓词与 Effect 检查（设计文档 v2.0 §5、§16、§35）。

状态快照统一建模为：{"facts": [规范化事实字符串...], "inventory": [...],
"text": 观察文本, "meta": {...}}。事实字符串形如 `agent_holds(egg_1)`、
`object_at(egg_1, countertop_1)`、`tests_pass(3,3)` 等。

Effect 谓词形式（设计文档 §16）：
    {"predicate": "agent.holds", "args": {"object": "$object"}}
其中 `$name` 引用输入参数，`$context.<key>` 引用任务上下文。
"""

from __future__ import annotations

import re
from typing import Any

_FACT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*)\s*\((.*)\)$")

# 谓词名 -> 事实名映射（含参数顺序）
_PREDICATE_FACT_NAMES: dict[str, tuple[str, tuple[str, ...]]] = {
    "object.exists": ("object_exists", ("object",)),
    "object.is_accessible": ("object_is_accessible", ("object",)),
    "agent.holds": ("agent_holds", ("object",)),
    "object.at_location": ("object_at", ("object", "location")),
    "object.heated": ("object_heated", ("object",)),
    "object.cleaned": ("object_cleaned", ("object",)),
    "object.cooled": ("object_cooled", ("object",)),
    "object.lit": ("object_lit", ("object",)),
    "object.toggled": ("object_toggled", ("object",)),
    "object.observed_with": ("object_observed_with", ("object", "associated_entity")),
    "object.in_receptacle": ("object_in_receptacle", ("object", "receptacle")),
    "container.open": ("container_open", ("container",)),
    "location.checked": ("location_checked", ("location",)),
    "callable.returns_expected": ("callable_returns_expected", ("entry_point",)),
    "tests.pass": ("tests_pass", ("entry_point",)),
    "answer.correct": ("answer_correct", ()),
}


def predicate_fact_signature(name: Any) -> tuple[str, tuple[str, ...]]:
    """Return the canonical fact name and semantic argument order.

    Effect dictionaries are JSON objects and their insertion order is not a
    predicate schema.  Runtime witness extraction must therefore use this
    registry when aligning a fact tuple with named Effect arguments.  Both a
    semantic predicate name (``object.at_location``) and its canonical fact
    name (``object_at``) are accepted.
    """
    normalized = str(name or "").strip().lower()
    underscored = normalized.replace(".", "_")
    relation_aliases = {
        "object.in.container": ("object_at", ("object", "container")),
        "object.in_container": ("object_at", ("object", "container")),
        "object_in_container": ("object_at", ("object", "container")),
        "object.in.receptacle": ("object_at", ("object", "receptacle")),
        "object.in_receptacle": ("object_at", ("object", "receptacle")),
        "object_in_receptacle": ("object_at", ("object", "receptacle")),
    }
    if normalized in relation_aliases:
        return relation_aliases[normalized]
    if underscored in relation_aliases:
        return relation_aliases[underscored]
    for predicate, (fact_name, order) in _PREDICATE_FACT_NAMES.items():
        if normalized == predicate or underscored == predicate.replace(".", "_"):
            return fact_name, tuple(order)
        if normalized == fact_name or underscored == fact_name:
            return fact_name, tuple(order)
    return underscored, ()


def ordered_predicate_args(
        name: Any, args: dict[str, Any]) -> list[tuple[str, Any]]:
    """Order named arguments by the predicate schema, preserving extras."""
    raw = dict(args or {})
    _fact_name, order = predicate_fact_signature(name)
    result = [(key, raw[key]) for key in order if key in raw]
    result.extend((key, value) for key, value in raw.items()
                  if key not in order)
    return result


def distinct_claims_conflict(left: Any, right: Any) -> bool:
    """Whether two class/concrete claims may denote the same instance."""
    if left in (None, "") or right in (None, ""):
        return False
    left_value = normalize_value(left)
    right_value = normalize_value(right)
    if left_value == right_value:
        return True
    left_instance = bool(re.search(r"_\d+$", left_value))
    right_instance = bool(re.search(r"_\d+$", right_value))
    if left_instance and right_instance:
        return False
    family = lambda value: re.sub(r"_\d+$", "", value)
    return family(left_value) == family(right_value)


def matching_effect_witnesses(
        state: "StateSnapshot | dict[str, Any]", effect: dict[str, Any],
        inputs: dict[str, Any], *, distinct_arg: str,
        context: dict[str, Any] | None = None) -> set[str]:
    """Return grounded witnesses matching one named Effect argument."""
    snapshot = state if isinstance(state, StateSnapshot) else StateSnapshot(state)
    predicate = str(effect.get("predicate") or "")
    bound = bind_args(dict(effect.get("args") or {}), inputs, context or inputs)
    ordered = ordered_predicate_args(predicate, bound)
    names = [key for key, _value in ordered]
    if distinct_arg not in names:
        return set()
    distinct_index = names.index(distinct_arg)
    fact_name, _order = predicate_fact_signature(predicate)
    witnesses: set[str] = set()
    for fact in snapshot.facts:
        parsed = _FACT_RE.fullmatch(str(fact).strip())
        if parsed is None:
            continue
        actual_name, _actual_order = predicate_fact_signature(parsed.group(1))
        if actual_name != fact_name:
            continue
        actual_values = [normalize_value(item.strip())
                         for item in parsed.group(2).split(",")]
        if len(actual_values) != len(ordered):
            continue
        if all(
                expected in (None, "")
                or str(expected).startswith("$")
                or _value_matches(expected, actual_values[index])
                for index, (_key, expected) in enumerate(ordered)):
            witnesses.add(actual_values[distinct_index])
    return witnesses


class StateSnapshot:
    """轻量状态快照包装（内部仍是普通 dict，方便 JSON 持久化）。"""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        data = dict(data or {})
        self.facts: set[str] = {str(f) for f in data.get("facts", []) if str(f).strip()}
        self.inventory: list[str] = [str(x) for x in data.get("inventory", [])]
        self.text: str = str(data.get("text", ""))
        self.meta: dict[str, Any] = dict(data.get("meta", {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "facts": sorted(self.facts),
            "inventory": list(self.inventory),
            "text": self.text,
            "meta": dict(self.meta),
        }

    # -- 事实操作 ----------------------------------------------------------
    def has_fact(self, fact: str) -> bool:
        return fact in self.facts

    def add_fact(self, fact: str) -> None:
        self.facts.add(str(fact).strip())

    def remove_fact(self, fact: str) -> None:
        self.facts.discard(str(fact).strip())


def normalize_value(value: Any) -> str:
    """将任意值规范化为事实参数 token（下划线、去空格、小写）。"""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", "_", text)
    return text


def bind_args(args: dict[str, Any], inputs: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """解析 `$input.name` / `$context.key` 占位符，返回绑定后的参数。"""
    bound: dict[str, Any] = {}
    for key, value in args.items():
        if isinstance(value, str) and value.startswith("$"):
            path = value[1:].split(".")
            if len(path) == 2 and path[0] == "inputs":
                bound[key] = inputs.get(path[1], value)
            elif len(path) == 2 and path[0] == "context":
                bound[key] = context.get(path[1], value)
            elif len(path) == 1:
                bound[key] = inputs.get(path[0], context.get(path[0], value))
            else:
                bound[key] = value
        else:
            bound[key] = value
    return bound


def _fact_string(predicate_name: str, args: dict[str, Any]) -> str:
    # Formal Effect evaluation and witness extraction must use the same
    # predicate schema.  In particular, relation aliases such as
    # ``object.in.container`` are represented by the canonical ``object_at``
    # fact, and JSON argument insertion order is not semantic argument order.
    fact_name, _order = predicate_fact_signature(predicate_name)
    values = [
        normalize_value(value)
        for _key, value in ordered_predicate_args(predicate_name, args)
    ]
    return f"{fact_name}({', '.join(values)})"


def evaluate_predicate(state: StateSnapshot, predicate: dict[str, Any]) -> bool:
    """在状态快照上评估单个谓词。"""
    if not isinstance(predicate, dict):
        return False
    # 直接事实形式
    direct = predicate.get("fact")
    if direct:
        return state.has_fact(str(direct))
    name = str(predicate.get("predicate", "")).strip()
    if not name:
        return False
    args = predicate.get("args") or {}

    fact = _fact_string(name, args)

    cardinality = max(1, int(predicate.get("cardinality", 1) or 1))
    if cardinality > 1:
        # ``distinct_by`` names the identity whose cardinality matters. Two
        # tuples that differ only in another argument (for example one object
        # observed with two lamps) are not two distinct target objects.
        distinct_arg = str(predicate.get("distinct_by") or "")
        if distinct_arg:
            ordered = ordered_predicate_args(name, args)
            if (distinct_arg not in {key for key, _value in ordered}
                    or any(value in (None, "")
                           or (isinstance(value, str)
                               and value.startswith("$"))
                           for _key, value in ordered)):
                return False
            return len(matching_effect_witnesses(
                state, predicate, {}, distinct_arg=distinct_arg,
                context={})) >= cardinality
        # Backward-compatible semantics for legacy contracts without an
        # explicit identity role: count complete distinct fact tuples.
        return _matching_fact_count(state, fact) >= cardinality

    # 特定求值器
    if name == "object.exists":
        return _object_known(state, args.get("object"))
    if name == "agent.holds":
        obj = normalize_value(args.get("object", ""))
        return any(_value_matches(obj, normalize_value(x)) for x in state.inventory) \
            or _has_matching_fact(state, fact)
    if name == "tests.pass":
        entry = normalize_value(args.get("entry_point", ""))
        for f in state.facts:
            m = _FACT_RE.match(f)
            if m and m.group(1) in ("tests_pass", "callable_returns_expected") and entry in f:
                return True
        return False
    if name == "answer.correct":
        return state.has_fact("answer_correct()")

    return _has_matching_fact(state, fact)


def _has_matching_fact(state: StateSnapshot, expected_fact: str) -> bool:
    """匹配规范事实，并允许 ALFWorld 类名与实例编号等价（apple ↔ apple_1）。"""
    return _matching_fact_count(state, expected_fact) > 0


def _matching_fact_count(state: StateSnapshot, expected_fact: str) -> int:
    """Count distinct grounded facts matching a possibly class-valued fact."""
    if state.has_fact(expected_fact):
        return 1
    expected = _FACT_RE.match(expected_fact)
    if expected is None:
        return 0
    expected_name, _expected_order = predicate_fact_signature(
        expected.group(1))
    expected_args = [part.strip() for part in expected.group(2).split(",")]
    matches = 0
    for actual_fact in state.facts:
        actual = _FACT_RE.match(actual_fact)
        if actual is None:
            continue
        actual_name, _actual_order = predicate_fact_signature(actual.group(1))
        if actual_name != expected_name:
            continue
        actual_args = [part.strip() for part in actual.group(2).split(",")]
        if len(actual_args) != len(expected_args):
            continue
        if all(_value_matches(want, got)
               for want, got in zip(expected_args, actual_args)):
            matches += 1
    return matches


def _value_matches(expected: Any, actual: Any) -> bool:
    expected_norm, actual_norm = normalize_value(expected), normalize_value(actual)
    if not expected_norm or expected_norm.startswith("$"):
        return False
    if expected_norm == actual_norm:
        return True
    strip_instance = lambda value: re.sub(r"_\d+$", "", value)
    # A class-valued task slot (``mug``) may match an observed ALFWorld
    # instance (``mug_1``).  Once the runtime has resolved a concrete instance,
    # however, identity must be preserved: ``mug_1`` must never satisfy a
    # contract for ``mug_2``.  The old symmetric comparison made separate
    # atomic validators existentially choose different objects in one plan.
    if re.search(r"_\d+$", expected_norm):
        return False
    return strip_instance(expected_norm) == strip_instance(actual_norm)


def evaluate_preconditions(
    state: StateSnapshot,
    inputs: dict[str, Any],
    preconditions: list[dict[str, Any]],
    context: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """评估前置条件列表，返回 (是否全部满足, 未满足项描述)。"""
    context = context or {}
    missing: list[str] = []
    for precondition in preconditions:
        if not isinstance(precondition, dict):
            missing.append(str(precondition))
            continue
        bound_args = bind_args(precondition.get("args") or {}, inputs, context)
        predicate = dict(precondition)
        predicate["args"] = bound_args
        if not evaluate_predicate(state, predicate):
            missing.append(_describe(precondition, bound_args))
    return (not missing), missing


def check_effects(
    after: StateSnapshot,
    inputs: dict[str, Any],
    effects: list[dict[str, Any]],
    context: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """检查核心 Effect 是否在 `after` 状态中成立，返回 (是否全部成立, 未成立项)。"""
    context = context or {}
    missing: list[str] = []
    for effect in effects:
        if not isinstance(effect, dict):
            missing.append(str(effect))
            continue
        bound_args = bind_args(effect.get("args") or {}, inputs, context)
        predicate = dict(effect)
        predicate["args"] = bound_args
        if not evaluate_predicate(after, predicate):
            missing.append(_describe(effect, bound_args))
    return (not missing), missing


def compute_effects(before: StateSnapshot, after: StateSnapshot) -> tuple[list[dict], list[dict]]:
    """从前后状态差集推断 Effect（供 Trace Atomicizer 使用）。"""
    added = sorted(after.facts - before.facts)
    removed = sorted(before.facts - after.facts)
    positive: list[dict] = []
    for fact in added:
        pred = _fact_to_predicate(fact)
        if pred is not None:
            positive.append(pred)
    negative: list[dict] = []
    for fact in removed:
        pred = _fact_to_predicate(fact)
        if pred is not None:
            negative.append({"not": pred})
    return positive, negative


def _fact_to_predicate(fact: str) -> dict[str, Any] | None:
    m = _FACT_RE.match(fact)
    if not m:
        return None
    fact_name, raw_args = m.group(1), m.group(2)
    args: list[str] = [a.strip() for a in raw_args.split(",") if a.strip()]
    reverse = {v[0]: (k, v[1]) for k, v in _PREDICATE_FACT_NAMES.items()}
    if fact_name in reverse:
        pred_name, order = reverse[fact_name]
        arg_map = {order[i]: args[i] for i in range(min(len(order), len(args)))}
        return {"predicate": pred_name, "args": arg_map}
    return {"predicate": fact_name, "args": {f"arg{i}": a for i, a in enumerate(args)}}


def _object_known(state: StateSnapshot, obj: Any) -> bool:
    value = normalize_value(obj)
    if not value:
        return False
    for f in state.facts:
        if value in f:
            return True
    if value in {normalize_value(x) for x in state.inventory}:
        return True
    return value in state.text.lower() or value.replace("_", " ") in state.text.lower()


def _describe(predicate: dict[str, Any], args: dict[str, Any]) -> str:
    name = predicate.get("predicate", predicate.get("fact", "?"))
    arg_text = ", ".join(f"{k}={v}" for k, v in args.items())
    return f"{name}({arg_text})" if arg_text else str(name)
