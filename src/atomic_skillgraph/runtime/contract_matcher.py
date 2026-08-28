"""Effect-contract matching with roles, cardinality and concrete bindings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.binding_ir import is_concrete_binding


@dataclass(frozen=True)
class ContractMatch:
    passed: bool
    reason: str = ""
    unified_roles: dict[str, str] = field(default_factory=dict)


def canonical_predicate(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("_", ".")
    return {
        "object.in.receptacle": "object.at.location",
        "object.in.container": "object.at.location",
    }.get(normalized, normalized)


def match_effect_contract(producer_effect: dict[str, Any],
                          target_effect: dict[str, Any],
                          available_bindings: dict[str, Any] | None = None
                          ) -> ContractMatch:
    if canonical_predicate(producer_effect.get("predicate", "")) != canonical_predicate(
            target_effect.get("predicate", "")):
        return ContractMatch(False, "predicate_mismatch")
    producer_args = dict(producer_effect.get("args") or {})
    target_args = dict(target_effect.get("args") or {})
    bindings = available_bindings or {}
    if len(producer_args) != len(target_args):
        return ContractMatch(False, "argument_role_mismatch")
    producer_values = list(producer_args.values())
    target_values = list(target_args.values())
    # Predicate schemas may call the same semantic argument ``location`` or
    # ``target_location``.  Placeholder roles/literals, not serialization key
    # spelling, are the binding contract.
    unmatched_targets = list(target_values)
    pairs: list[tuple[Any, Any]] = []
    for source in producer_values:
        source_role = _placeholder_role(source)
        match_index = next((index for index, target in enumerate(unmatched_targets)
                            if (source_role and source_role == _placeholder_role(target))
                            or (source_role and _placeholder_role(target)
                                and is_concrete_binding(bindings.get(source_role))
                                and bindings.get(source_role)
                                == bindings.get(_placeholder_role(target)))
                            or (is_concrete_binding(source)
                                and is_concrete_binding(target)
                                and source == target)), None)
        if match_index is None:
            return ContractMatch(False, "argument_role_mismatch")
        pairs.append((source, unmatched_targets.pop(match_index)))
    unified: dict[str, str] = {}
    for source, target in pairs:
        role = _placeholder_role(target) or "literal"
        if is_concrete_binding(source) and is_concrete_binding(target) and source != target:
            return ContractMatch(False, f"literal_mismatch:{role}")
        source_role = _placeholder_role(source)
        target_role = _placeholder_role(target)
        if source_role and target_role:
            unified[source_role] = target_role
        if target_role and target_role in bindings and not is_concrete_binding(bindings[target_role]):
            return ContractMatch(False, f"target_binding_unresolved:{target_role}")
    producer_count = max(1, int(producer_effect.get("cardinality", 1) or 1))
    target_count = max(1, int(target_effect.get("cardinality", 1) or 1))
    if producer_count < target_count:
        return ContractMatch(False, "cardinality_insufficient")
    producer_distinct = str(producer_effect.get("distinct_by") or "")
    target_distinct = str(target_effect.get("distinct_by") or "")
    if target_distinct and producer_distinct != target_distinct:
        return ContractMatch(False, "distinctness_mismatch")
    return ContractMatch(True, unified_roles=unified)


def _placeholder_role(value: Any) -> str:
    if not isinstance(value, str) or not value.startswith("$"):
        return ""
    return value.rsplit(".", 1)[-1].lstrip("$")
