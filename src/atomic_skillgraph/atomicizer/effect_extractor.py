"""Effect 提取（设计文档 v2.0 §25.3：Effect Extraction / I/O 推断 / Validator 构造）。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..core.predicates import StateSnapshot, compute_effects, _fact_to_predicate

# Kept as a compatibility hook for callers that import the symbol. Capability
# names are no longer enumerated: any verified predicate family is normalized
# mechanically below, while the Extractor Agent proposes its semantic alias.
_FACT_FAMILY_NAMES: dict[str, tuple[str, str]] = {}


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


# Observation-only facts are filtered below. Whether a location transition is
# noise or a real capability is decided relative to the goal and causal graph,
# never globally by predicate name.
_NOISE_FAMILIES = {"location_checked"}


def extract_effect(before: dict[str, Any], after: dict[str, Any],
                   bound_params: dict[str, Any] | None = None) -> ExtractedEffect:
    """从前后状态快照提取 Effect + I/O + 前置条件 + 节点验证器。"""
    bound_params = bound_params or {}
    before_snapshot = StateSnapshot(before)
    after_snapshot = StateSnapshot(after)
    positive, negative = compute_effects(before_snapshot, after_snapshot)
    # Keep genuine state transitions. Goal-relative slicing decides whether a
    # movement is a capability or merely setup/exploration.
    positive = [p for p in positive if _family_of(p) not in _NOISE_FAMILIES]
    positive = _causal_positive_effects(positive, bound_params)
    negative = [n for n in negative if not _is_noise_effect(n)]
    result = ExtractedEffect(positive=positive, negative=negative)

    # 主效果族（用于命名 / 归因）
    families = sorted({_family_of(p) for p in positive if _family_of(p)})
    result.primary_family = families[0] if families else ""
    if result.primary_family:
        name, summary = _FACT_FAMILY_NAMES.get(result.primary_family,
                                               (_snake(result.primary_family),
                                                f"Verified transition: "
                                                f"{str(positive[0].get('predicate') or '')}"))
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
        # Start from every structured fact the adapter can represent. The
        # generic grounding filter below retains only reads tied to this
        # occurrence's declared inputs; no benchmark predicate catalogue is
        # used to decide which preconditions are possible.
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
    """Retain only grounded input-dependent reads, independent of operation name."""
    if not _has_input_reference(predicate):
        return False
    # Replacing one shared argument is not enough to make the whole fact part
    # of this occurrence. A bystander can share the same location with the
    # target entity; its fact must not become a precondition
    # must not become a precondition. Entity-valued literals must therefore be
    # grounded to a declared input as well. Numeric/boolean literals remain
    # valid predicate constants.
    for value in (predicate.get("args") or {}).values():
        if not isinstance(value, str) or value.startswith("$inputs."):
            continue
        lowered = value.strip().lower()
        if lowered in {"true", "false", "none", "null"}:
            continue
        try:
            float(lowered)
            continue
        except ValueError:
            return False
    return True


def _family_of(effect: dict[str, Any]) -> str:
    name = str(effect.get("predicate", ""))
    return name.replace(".", "_") or ""


def _is_noise_effect(effect: dict[str, Any]) -> bool:
    """负向噪声过滤：`{"not": {...}}` 形式内的噪声谓词。"""
    inner = effect.get("not")
    if isinstance(inner, dict):
        return _family_of(inner) in _NOISE_FAMILIES
    return _family_of(effect) in _NOISE_FAMILIES


def _causal_positive_effects(effects: list[dict[str, Any]],
                             bound_params: dict[str, Any]) -> list[dict[str, Any]]:
    """Exclude observations unrelated to any grounded phase participant."""
    from ..core.predicates import normalize_value
    participants = {normalize_value(value) for value in bound_params.values()
                    if value not in (None, "")}
    participants.discard("")
    entity_participants = {
        normalize_value(value) for role, value in bound_params.items()
        if value not in (None, "")
        and not str(role).lower().endswith("_location")
        and not any(token in str(role).lower()
                    for token in ("station", "resource", "device", "instrument",
                                  "destination"))
    }
    entity_participants.discard("")

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
        # Existence is observational knowledge, not a capability Effect.
        if name == "object.exists":
            continue
        argument_values = [normalize_value(value) for value in args.values()]
        if ("object" in args and entity_participants
                and not any(same_entity(args.get("object"), participant)
                            for participant in entity_participants)):
            continue
        if participants and argument_values and not any(
                any(same_entity(value, participant) for participant in participants)
                for value in argument_values):
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
    if argument in slots:
        return argument
    lexical = [slot for slot in slots
               if argument in slot or slot in argument]
    if lexical:
        return sorted(lexical, key=lambda item: (len(item), item))[0]
    if argument in {"location", "container"}:
        relational = [slot for slot in slots
                      if any(token in slot for token in ("location", "place", "station"))]
        if relational:
            return sorted(relational, key=lambda item: (len(item), item))[0]
    return sorted(slots)[0]


def _snake(text: str) -> str:
    import re
    text = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", text)
    return re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_") or "effect"


def _semantic_type_of(name: str) -> str:
    lowered = str(name).lower()
    if any(token in lowered for token in (
            "location", "place", "destination", "source", "origin", "recep")):
        return "location_ref"
    if any(token in lowered for token in (
            "resource", "device", "instrument", "station")):
        return "resource_ref"
    if "entry" in lowered:
        return "entry_point"
    return "entity_ref"
