"""Occurrence-aware DATA_FLOW synthesis for temporary Atomic plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.binding_ir import BindingKind, BindingSpec, binding_slot_name
from ..core.edge_ir import GraphEdge
from ..core.status import EdgeType
from .contract_matcher import match_effect_contract
from .output_materializer import validate_output_materializer


@dataclass(frozen=True)
class _Candidate:
    source_index: int
    source_node: Any
    source_output: str
    target_input: str
    materializer: dict[str, Any]
    score: int


class RuntimeDataFlowSynthesizer:
    """Infer exact occurrence-to-occurrence value transfer.

    The synthesizer uses only typed contracts and grounded bindings.  It never
    joins occurrences solely because their names are lexically similar.
    """

    def synthesize(self, task: Any, occurrences: list[Any], registry: Any
                   ) -> list[GraphEdge]:
        edges: list[GraphEdge] = []
        for target_index, target in enumerate(occurrences):
            target_atomic = registry.get(target.ref)
            if target_atomic is None:
                continue
            for target_decl in list(getattr(target_atomic, "inputs", []) or []):
                if not isinstance(target_decl, dict):
                    continue
                target_input = str(target_decl.get("name") or "")
                if not target_input:
                    continue
                candidate = self._nearest_candidate(
                    occurrences, target_index, target, target_atomic,
                    target_decl, registry)
                if candidate is None:
                    continue
                source = candidate.source_node
                source_step = _step_id(source, candidate.source_index)
                target_step = _step_id(target, target_index)
                edge = GraphEdge(
                    source=str(source.ref), target=str(target.ref),
                    type=EdgeType.DATA_FLOW, scope="runtime",
                    source_step=source_step, target_step=target_step,
                    mapping={
                        "source_output": candidate.source_output,
                        "target_input": target_input,
                        "source_semantic_type": _output_type(
                            registry.get(source.ref), candidate.source_output),
                        "target_semantic_type": str(
                            target_decl.get("semantic_type") or ""),
                        "materializer": dict(candidate.materializer),
                        "mode": "runtime_synthesized",
                    },
                    evidence=["contract_and_occurrence_order"],
                    metadata={"task_id": str(getattr(task, "task_id", "")),
                              "producer_index": candidate.source_index,
                              "consumer_index": target_index},
                )
                edges.append(edge)
                # Persist the exact producer for validation/runtime audit.  A
                # concrete task value remains in ``params``; DATA_FLOW may
                # refine its class-valued identity after source validation.
                target.binding_specs[target_input] = BindingSpec(
                    BindingKind.DATA_FLOW, source_step=source_step,
                    source_output=candidate.source_output,
                    symbol=f"$flow.{candidate.source_output}")
        return edges

    def _nearest_candidate(
            self, occurrences: list[Any], target_index: int, target: Any,
            target_atomic: Any, target_decl: dict[str, Any], registry: Any
            ) -> _Candidate | None:
        target_input = str(target_decl.get("name") or "")
        candidates: list[_Candidate] = []
        for source_index in range(target_index):
            source = occurrences[source_index]
            source_atomic = registry.get(source.ref)
            if source_atomic is None or not _same_branch(source, target):
                continue
            for output_decl in list(getattr(source_atomic, "outputs", []) or []):
                if not isinstance(output_decl, dict):
                    continue
                source_output = str(output_decl.get("name") or "")
                validation = validate_output_materializer(source_atomic, output_decl)
                if not source_output or not validation.passed:
                    continue
                if not _types_compatible(
                        str(output_decl.get("semantic_type") or ""),
                        str(target_decl.get("semantic_type") or "")):
                    continue
                materializer = dict(validation.materializer)
                _annotate_source_role(source_atomic, materializer)
                if not _grounded_values_compatible(
                        source, materializer, target, target_input):
                    continue
                relation_score = _contract_relation_score(
                    source_atomic, materializer,
                    target_atomic, target_input)
                if relation_score <= 0:
                    continue
                candidates.append(_Candidate(
                    source_index, source, source_output, target_input,
                    materializer, relation_score))
        if not candidates:
            return None
        # Most specific causal contract first, then nearest prior occurrence.
        return max(candidates, key=lambda item: (item.score, item.source_index))


def _step_id(node: Any, index: int) -> str:
    return str(getattr(node, "step_id", "") or
               getattr(node, "origin_step_id", "") or f"step_{index:03d}")


def _output_type(atomic: Any, output_name: str) -> str:
    for item in list(getattr(atomic, "outputs", []) or []):
        if isinstance(item, dict) and str(item.get("name") or "") == output_name:
            return str(item.get("semantic_type") or "")
    return ""


def _types_compatible(source: str, target: str) -> bool:
    return not source or not target or source == target or "value" in {source, target}


def _same_branch(source: Any, target: Any) -> bool:
    """Reject an explicitly identified cross-cardinality branch."""

    source_branch = str(getattr(source, "branch_id", "") or "")
    target_branch = str(getattr(target, "branch_id", "") or "")
    if source_branch and target_branch and source_branch != target_branch:
        return False
    branch_keys = ("branch_id", "cardinality_index", "occurrence_index")
    source_params = dict(getattr(source, "params", {}) or {})
    target_params = dict(getattr(target, "params", {}) or {})
    for key in branch_keys:
        left, right = source_params.get(key), target_params.get(key)
        if left not in (None, "") and right not in (None, "") and left != right:
            return False
    return True


def _grounded_values_compatible(source: Any, materializer: dict[str, Any],
                                target: Any, target_input: str) -> bool:
    source_params = dict(getattr(source, "params", {}) or {})
    target_params = dict(getattr(target, "params", {}) or {})
    source_role = ""
    if materializer.get("kind") == "input_role":
        source_role = str(materializer.get("role") or "")
    elif materializer.get("kind") == "effect_arg":
        # Effect arg selectors refer to a placeholder in the source contract.
        source_role = str(materializer.get("source_role") or "")
    source_value = source_params.get(source_role) if source_role else None
    target_value = target_params.get(target_input)
    if (_concrete(source_value) and _concrete(target_value)
            and str(source_value) != str(target_value)):
        return False
    return True


def _contract_relation_score(source_atomic: Any, materializer: dict[str, Any],
                             target_atomic: Any, target_input: str) -> int:
    source_effects = list(getattr(source_atomic, "effects", []) or [])
    predicate = str(materializer.get("predicate") or "")
    arg = str(materializer.get("arg") or "")
    selected_effects = [effect for effect in source_effects
                        if isinstance(effect, dict)
                        and (not predicate
                             or str(effect.get("predicate") or "") == predicate)
                        and (not arg or arg in dict(effect.get("args") or {}))]
    target_preconditions = [item for item in
                            list(getattr(target_atomic, "preconditions", []) or [])
                            if isinstance(item, dict)
                            and target_input in {
                                binding_slot_name(value)
                                for value in dict(item.get("args") or {}).values()
                            }]
    if any(match_effect_contract(effect, precondition).passed
           for effect in selected_effects for precondition in target_preconditions):
        return 4
    # A shared role name alone is not causal evidence.  In particular it must
    # not connect two repeated producer occurrences merely because both accept
    # an ``object`` input.
    return 0


def _annotate_source_role(source_atomic: Any,
                          materializer: dict[str, Any]) -> None:
    if materializer.get("kind") == "input_role":
        materializer.setdefault("source_role", materializer.get("role"))
        return
    if materializer.get("kind") != "effect_arg":
        return
    predicate = str(materializer.get("predicate") or "")
    arg = str(materializer.get("arg") or "")
    roles = {
        binding_slot_name(dict(effect.get("args") or {}).get(arg))
        for effect in list(getattr(source_atomic, "effects", []) or [])
        if isinstance(effect, dict)
        and str(effect.get("predicate") or "") == predicate
        and arg in dict(effect.get("args") or {})
    }
    roles.discard("")
    if len(roles) == 1:
        materializer["source_role"] = next(iter(roles))


def _concrete(value: Any) -> bool:
    return value not in (None, "") and not (
        isinstance(value, str) and value.startswith("$"))
