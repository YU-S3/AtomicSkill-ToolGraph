"""Validated materialization of Atomic occurrence outputs.

An output is useful to DATA_FLOW only when the runtime can state exactly where
its value comes from.  Natural-language names are never treated as executable
semantics.  A narrow legacy inference is retained for old banks when one and
only one Effect argument can safely provide the declared output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.binding_ir import binding_slot_name, is_concrete_binding
from ..core.predicates import StateSnapshot, _fact_to_predicate, evaluate_predicate


MATERIALIZER_KINDS = {"input_role", "effect_arg", "tool_result", "state_fact_arg"}


@dataclass(frozen=True)
class OutputMaterializerValidation:
    passed: bool
    output_name: str = ""
    materializer: dict[str, Any] = field(default_factory=dict)
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "output_name": self.output_name,
            "materializer": dict(self.materializer),
            "errors": list(self.errors),
        }


def get_output_declaration(atomic: Any, output_name: str) -> dict[str, Any] | None:
    for declaration in list(getattr(atomic, "outputs", []) or []):
        if (isinstance(declaration, dict)
                and str(declaration.get("name") or "") == str(output_name)):
            return declaration
    return None


def validate_output_materializer(
        atomic: Any, output: str | dict[str, Any]) -> OutputMaterializerValidation:
    declaration = (get_output_declaration(atomic, output)
                   if isinstance(output, str) else output)
    if not isinstance(declaration, dict):
        return OutputMaterializerValidation(
            False, str(output), errors=("output_declaration_missing",))
    name = str(declaration.get("name") or "")
    if not name:
        return OutputMaterializerValidation(
            False, errors=("output_name_missing",))
    raw = declaration.get("materializer")
    materializer = (dict(raw) if isinstance(raw, dict)
                    else _infer_legacy_materializer(atomic, declaration))
    if not materializer:
        return OutputMaterializerValidation(
            False, name, errors=("output_materializer_missing_or_ambiguous",))
    kind = str(materializer.get("kind") or "")
    errors: list[str] = []
    if kind not in MATERIALIZER_KINDS:
        errors.append(f"unsupported_materializer_kind:{kind or 'missing'}")
    elif kind == "input_role":
        role = str(materializer.get("role") or materializer.get("input") or "")
        input_declaration = _input_declaration(atomic, role)
        if not role or not input_declaration:
            errors.append(f"materializer_input_role_missing:{role}")
        else:
            materializer["role"] = role
            output_type = str(declaration.get("semantic_type") or "")
            input_type = str(input_declaration.get("semantic_type") or "")
            if output_type and input_type and output_type != input_type:
                errors.append(
                    f"materializer_semantic_type_mismatch:{input_type}->{output_type}")
    elif kind == "effect_arg":
        predicate = str(materializer.get("predicate") or "")
        arg = str(materializer.get("arg") or "")
        matches = [effect for effect in list(getattr(atomic, "effects", []) or [])
                   if isinstance(effect, dict)
                   and str(effect.get("predicate") or "") == predicate]
        if not predicate or not arg or not any(
                arg in dict(effect.get("args") or {}) for effect in matches):
            errors.append(f"materializer_effect_arg_missing:{predicate}:{arg}")
        else:
            # The value selected by an Effect argument normally originates
            # from one declared input role.  A declaration cannot relabel an
            # object-valued role as a location (or vice versa) merely by
            # changing output.semantic_type.
            roles = {
                binding_slot_name(dict(effect.get("args") or {}).get(arg))
                for effect in matches
                if arg in dict(effect.get("args") or {})
            }
            roles.discard("")
            output_type = str(declaration.get("semantic_type") or "")
            input_types = {
                str(item.get("semantic_type") or "")
                for role in roles
                for item in [_input_declaration(atomic, role)]
                if isinstance(item, dict)
            }
            if len(roles) > 1:
                errors.append(
                    f"materializer_effect_arg_ambiguous_roles:{predicate}:{arg}")
            elif any(output_type and input_type
                     and "value" not in {output_type, input_type}
                     and output_type != input_type
                     for input_type in input_types):
                source_type = next(iter(input_types), "")
                errors.append(
                    f"materializer_semantic_type_mismatch:"
                    f"{source_type}->{output_type}")
    elif kind == "tool_result":
        key = str(materializer.get("key") or "")
        if not key:
            errors.append("materializer_tool_result_key_missing")
    elif kind == "state_fact_arg":
        predicate = str(materializer.get("predicate") or "")
        arg = str(materializer.get("arg") or "")
        if not predicate or not arg:
            errors.append("materializer_state_fact_selector_missing")
    return OutputMaterializerValidation(
        not errors, name, materializer, tuple(errors))


def output_is_materializable(atomic: Any, output_name: str) -> bool:
    return validate_output_materializer(atomic, output_name).passed


def materialize_atomic_outputs(
        atomic: Any, inputs: dict[str, Any], before: dict[str, Any] | None,
        after: dict[str, Any] | None, tool_result: Any = None) -> dict[str, Any]:
    """Materialize only declared outputs whose value is concrete and proven."""

    result: dict[str, Any] = {}
    for output in list(getattr(atomic, "outputs", []) or []):
        validation = validate_output_materializer(atomic, output)
        if not validation.passed:
            continue
        materializer = validation.materializer
        kind = str(materializer["kind"])
        value: Any = None
        if kind == "input_role":
            value = inputs.get(str(materializer.get("role") or ""))
        elif kind == "effect_arg":
            value = _materialize_effect_arg(
                atomic, inputs, after or {}, str(materializer.get("predicate") or ""),
                str(materializer.get("arg") or ""))
        elif kind == "tool_result":
            key = str(materializer.get("key") or "")
            if isinstance(tool_result, dict):
                value = tool_result.get(key)
            else:
                value = getattr(tool_result, key, None)
        elif kind == "state_fact_arg":
            value = _materialize_state_fact_arg(
                after or {}, str(materializer.get("predicate") or ""),
                str(materializer.get("arg") or ""), inputs)
        if is_concrete_binding(value):
            result[validation.output_name] = value
    return result


def _input_declaration(atomic: Any, role: str) -> dict[str, Any] | None:
    return next((item for item in list(getattr(atomic, "inputs", []) or [])
                 if isinstance(item, dict)
                 and str(item.get("name") or "") == role), None)


def _infer_legacy_materializer(atomic: Any,
                               declaration: dict[str, Any]) -> dict[str, Any]:
    """Infer old output declarations only when the source is unambiguous."""

    name = str(declaration.get("name") or "")
    if _input_declaration(atomic, name):
        return {"kind": "input_role", "role": name, "legacy_inferred": True}
    candidates: list[tuple[str, str, str]] = []
    output_type = str(declaration.get("semantic_type") or "")
    input_types = {
        str(item.get("name") or ""): str(item.get("semantic_type") or "")
        for item in list(getattr(atomic, "inputs", []) or [])
        if isinstance(item, dict) and item.get("name")
    }
    for effect in list(getattr(atomic, "effects", []) or []):
        if not isinstance(effect, dict):
            continue
        predicate = str(effect.get("predicate") or "")
        for arg, value in dict(effect.get("args") or {}).items():
            role = binding_slot_name(value)
            if not role:
                continue
            if output_type and input_types.get(role) and input_types[role] != output_type:
                continue
            candidates.append((predicate, str(arg), role))
    unique = list(dict.fromkeys(candidates))
    if len(unique) != 1:
        return {}
    predicate, arg, _ = unique[0]
    return {"kind": "effect_arg", "predicate": predicate, "arg": arg,
            "legacy_inferred": True}


def _materialize_effect_arg(atomic: Any, inputs: dict[str, Any],
                            after: dict[str, Any], predicate: str,
                            arg: str) -> Any:
    for effect in list(getattr(atomic, "effects", []) or []):
        if (not isinstance(effect, dict)
                or str(effect.get("predicate") or "") != predicate):
            continue
        args = dict(effect.get("args") or {})
        if arg not in args:
            continue
        value = args[arg]
        role = binding_slot_name(value)
        concrete = inputs.get(role) if role else value
        if not is_concrete_binding(concrete):
            return None
        bound_args: dict[str, Any] = {}
        for key, raw in args.items():
            bound_role = binding_slot_name(raw)
            bound_args[key] = inputs.get(bound_role) if bound_role else raw
        candidate = dict(effect)
        candidate["args"] = bound_args
        # The output is promised only after the corresponding Effect holds.
        if evaluate_predicate(StateSnapshot(after), candidate):
            return concrete
    return None


def _materialize_state_fact_arg(after: dict[str, Any], predicate: str, arg: str,
                                inputs: dict[str, Any]) -> Any:
    for fact in sorted(StateSnapshot(after).facts):
        candidate = _fact_to_predicate(fact)
        if (not isinstance(candidate, dict)
                or str(candidate.get("predicate") or "") != predicate):
            continue
        args = dict(candidate.get("args") or {})
        value = args.get(arg)
        if not is_concrete_binding(value):
            continue
        # Other provided anchors must agree with the selected state fact.
        compatible = True
        for role, expected in inputs.items():
            if role in args and is_concrete_binding(expected) and args[role] != expected:
                compatible = False
                break
        if compatible:
            return value
    return None
