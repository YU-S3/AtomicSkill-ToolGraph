"""Effect 提取（设计文档 v2.0 §25.3：Effect Extraction / I/O 推断 / Validator 构造）。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..core.predicates import StateSnapshot, compute_effects, _fact_to_predicate

# 事实族 → 语义命名（用于 atomic candidate 的 logical_id 命名）
_FACT_FAMILY_NAMES = {
    "agent_holds": ("acquire_object", "Agent 持有目标对象"),
    "object_heated": ("heat_object", "目标对象被加热"),
    "object_cleaned": ("clean_object", "目标对象被清洁"),
    "object_cooled": ("cool_object", "目标对象被冷却"),
    "object_at": ("place_object", "目标对象被放置到目标位置"),
    "object_lit": ("light_object", "目标对象被点亮"),
    "container_open": ("open_container", "容器被打开"),
    "callable_returns_expected": ("solve_function", "函数对给定输入返回期望输出"),
    "tests_pass": ("pass_tests", "测试用例全部通过"),
    "answer_correct": ("produce_answer", "产出正确答案"),
}


@dataclass
class ExtractedEffect:
    """一个片段的状态效果 + 推断出的 I/O 与验证器。"""

    positive: list[dict[str, Any]] = field(default_factory=list)
    negative: list[dict[str, Any]] = field(default_factory=list)
    inputs: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    preconditions: list[dict[str, Any]] = field(default_factory=list)
    validator: dict[str, Any] = field(default_factory=dict)
    primary_family: str = ""
    suggested_name: str = ""
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "positive": self.positive,
            "negative": self.negative,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "preconditions": self.preconditions,
            "validator": self.validator,
            "primary_family": self.primary_family,
            "suggested_name": self.suggested_name,
            "summary": self.summary,
        }


# 机械性事实族：移动/探索产生的位置变化，不构成核心 Effect（§6.2 封装原则）
_NOISE_FAMILIES = {"location_checked", "agent_at"}


def extract_effect(before: dict[str, Any], after: dict[str, Any],
                   bound_params: dict[str, Any] | None = None) -> ExtractedEffect:
    """从前后状态快照提取 Effect + I/O + 前置条件 + 节点验证器。"""
    bound_params = bound_params or {}
    before_snapshot = StateSnapshot(before)
    after_snapshot = StateSnapshot(after)
    positive, negative = compute_effects(before_snapshot, after_snapshot)
    # 过滤机械性噪声事实（移动/探索），只保留核心状态转移
    positive = [p for p in positive if _family_of(p) not in _NOISE_FAMILIES]
    positive = _causal_positive_effects(positive, bound_params)
    negative = [n for n in negative if not _is_noise_effect(n)]
    result = ExtractedEffect(positive=positive, negative=negative)

    # 主效果族（用于命名 / 归因）
    families = sorted({_family_of(p) for p in positive if _family_of(p)})
    result.primary_family = families[0] if families else ""
    if result.primary_family:
        name, summary = _FACT_FAMILY_NAMES.get(result.primary_family,
                                               (_snake(result.primary_family), result.primary_family))
        result.suggested_name = name
        result.summary = summary

    # 输出：从正向效果参数推断
    output_names: list[str] = []
    for effect in positive:
        for key, value in (effect.get("args") or {}).items():
            if key not in output_names and value not in output_names:
                output_names.append(key)
    result.outputs = [{"name": name, "semantic_type": _semantic_type_of(name)}
                      for name in output_names]

    # 输入：绑定参数
    result.inputs = [{"name": key, "semantic_type": _semantic_type_of(key)}
                     for key in sorted(bound_params.keys())]

    # 前置条件：从 before 状态推断；参数化为 $inputs.* 后只保留与输入绑定相关的
    # （避免实例字面量爆炸导致跨实例无法满足）
    raw_preconditions: list[dict[str, Any]] = []
    for fact in sorted(before_snapshot.facts):
        pred = _fact_to_predicate(fact)
        if pred is None:
            continue
        name = str(pred.get("predicate", ""))
        if _family_of(pred) in _NOISE_FAMILIES:
            continue
        if name in ("object.exists", "object.at_location", "agent.holds", "container.open",
                    "object.is_accessible", "callable.returns_expected"):
            raw_preconditions.append(pred)
    parameterized = parameterize_predicates(raw_preconditions, bound_params)
    relevant = [p for p in parameterized if _relevant_precondition(
        p, result.primary_family, bound_params)]
    # 去重并限制为最小、与目标对象有关的前置条件。
    seen: set[str] = set()
    result.preconditions = []
    for predicate in relevant:
        key = repr(predicate)
        if key in seen:
            continue
        seen.add(key)
        result.preconditions.append(predicate)
        if len(result.preconditions) >= 3:
            break

    # 节点验证器：pre_checks（前置） + post_checks（效果）
    result.validator = {
        "pre_checks": sorted({str(p.get("predicate")) for p in result.preconditions}),
        "post_checks": sorted({str(p.get("predicate")) for p in positive}),
    }
    return result


def _has_input_reference(predicate: dict[str, Any]) -> bool:
    """前置条件的参数是否引用 $inputs.*（跨实例可绑定）。"""
    args = predicate.get("args") or {}
    return any(isinstance(v, str) and v.startswith("$inputs.") for v in args.values())


def _relevant_precondition(predicate: dict[str, Any], primary_family: str,
                           bound_params: dict[str, Any]) -> bool:
    """只保留当前目标实体的必要条件，避免把场景中旁观对象写进 Contract。"""
    args = predicate.get("args") or {}
    name = str(predicate.get("predicate") or "")
    object_arg = args.get("object")
    if "object" in bound_params and object_arg != "$inputs.object":
        return False
    if primary_family == "agent_holds":
        if name == "object.exists":
            return object_arg == "$inputs.object"
        if name == "object.at_location":
            return (object_arg == "$inputs.object"
                    and args.get("location") == "$inputs.object_location")
        return False
    if primary_family in {"object_heated", "object_cleaned", "object_cooled",
                          "object_at"}:
        return name == "agent.holds" and object_arg == "$inputs.object"
    return _has_input_reference(predicate)


def _family_of(effect: dict[str, Any]) -> str:
    name = str(effect.get("predicate", ""))
    for family in _FACT_FAMILY_NAMES:
        if name.replace(".", "_").startswith(family):
            return family
    if name in _FACT_FAMILY_NAMES:
        return name
    return name.replace(".", "_") or ""


def _is_noise_effect(effect: dict[str, Any]) -> bool:
    """负向噪声过滤：`{"not": {...}}` 形式内的噪声谓词。"""
    inner = effect.get("not")
    if isinstance(inner, dict):
        return _family_of(inner) in _NOISE_FAMILIES
    return _family_of(effect) in _NOISE_FAMILIES


def _causal_positive_effects(effects: list[dict[str, Any]],
                             bound_params: dict[str, Any]) -> list[dict[str, Any]]:
    """排除探索时新观察到的旁观对象；只保留动作参数所指实体的状态变化。"""
    from ..core.predicates import normalize_value
    target_object = normalize_value(bound_params.get("object", ""))
    target_location = normalize_value(bound_params.get("target_location", ""))

    def same_entity(left: Any, right: str) -> bool:
        value = normalize_value(left)
        if not value or not right:
            return False
        strip_instance = lambda item: re.sub(r"_\d+$", "", item)
        # A concrete execution binding (mug_1) identifies exactly one object.
        # Family matching is allowed only when the binding itself is abstract
        # (mug).  Otherwise a newly observed mug_2 can be misattributed to the
        # action on mug_1.
        if re.search(r"_\d+$", right):
            return value == right
        return value == right or strip_instance(value) == strip_instance(right)

    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    for effect in effects:
        name = str(effect.get("predicate") or "")
        args = effect.get("args") or {}
        # exists / at_location 的新增常来自首次观察，不是动作产生的 Effect。
        if name == "object.exists":
            continue
        if name == "object.at_location":
            if not (same_entity(args.get("object"), target_object)
                    and target_location
                    and same_entity(args.get("location"), target_location)):
                continue
        elif "object" in args and target_object:
            if not same_entity(args.get("object"), target_object):
                continue
        key = repr(effect)
        if key not in seen:
            seen.add(key)
            kept.append(effect)
    return kept


def parameterize_predicates(effects: list[dict[str, Any]],
                            bound_params: dict[str, Any]) -> list[dict[str, Any]]:
    """把效果/前置条件中的实例字面量替换为 `$inputs.<slot>` 占位符。

    保证抽象原子技能可跨实例复用（§16：args 引用 $inputs 参数）。
    """
    from ..core.predicates import normalize_value
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for effect in effects:
        if not isinstance(effect, dict):
            out.append(effect)
            continue
        args: dict[str, Any] = {}
        for key, value in (effect.get("args") or {}).items():
            norm = normalize_value(value)
            replaced = False
            candidates: list[tuple[int, str]] = []
            for slot, param_value in bound_params.items():
                param_norm = normalize_value(param_value)
                # Goals normally bind an object class (``mug``), while the
                # successful trace exposes a concrete instance (``mug_2``).
                # Treat that one-way class→instance relation as the same slot,
                # otherwise a later success can write ``mug_2`` into the
                # reusable Abstract contract.
                same_value = param_norm == norm and bool(norm)
                # Family matching is one-way: an abstract slot (mug) may bind
                # a concrete observation (mug_1), never the reverse.
                same_alfworld_family = (
                    bool(param_norm and norm)
                    and re.sub(r"_\d+$", "", param_norm)
                    == re.sub(r"_\d+$", "", norm)
                    and not re.search(r"_\d+$", param_norm)
                )
                if same_value:
                    candidates.append((0, str(slot)))
                elif same_alfworld_family:
                    candidates.append((1, str(slot)))
            if candidates:
                best_rank = min(rank for rank, _slot in candidates)
                ranked = [slot for rank, slot in candidates if rank == best_rank]
                slot = _preferred_slot(str(key), ranked)
                args[key] = f"$inputs.{slot}"
                replaced = True
            if not replaced:
                args[key] = value
        parameterized = {**effect, "args": args}
        canonical = repr(parameterized)
        if canonical not in seen:
            seen.add(canonical)
            out.append(parameterized)
    return out


def _preferred_slot(argument: str, slots: list[str]) -> str:
    preferences = {
        "object": ["object"],
        "location": ["target_location", "object_location", "location"],
        "container": ["container", "target_location", "object_location",
                      "heating_station", "cleaning_station", "cooling_station"],
    }.get(argument, [argument])
    for preferred in preferences:
        if preferred in slots:
            return preferred
    return sorted(slots)[0]


def _snake(text: str) -> str:
    import re
    text = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", text)
    return re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_") or "effect"


def _semantic_type_of(name: str) -> str:
    lowered = str(name).lower()
    if "location" in lowered or "recep" in lowered:
        return "location_ref"
    if "station" in lowered or "heating" in lowered or "cooling" in lowered:
        return "station_ref"
    if "object" in lowered or "container" in lowered:
        return "object_ref"
    if "entry" in lowered:
        return "entry_point"
    return "value"
