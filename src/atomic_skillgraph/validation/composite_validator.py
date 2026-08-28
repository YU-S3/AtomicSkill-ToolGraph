"""Composite Validator（三级验证体系第 3 层，设计文档 v2.0 §35.3）。

回答：高层 Skill 的组合目标是否满足？Benchmark Verifier 回答整个任务是否成功。
四层（Tool Test → Atomic Node Validator → Composite Validator → Benchmark Verifier）
不能互相替代。
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from ..core.binding_ir import is_concrete_binding
from ..core.predicates import StateSnapshot, _fact_to_predicate, bind_args, check_effects
from ..core.skill_ir import CompositeSkill
from ..core.trace_ir import NodeValidationResult


class CompositeFailureCode(str, Enum):
    """Stable machine-readable Composite failure reasons.

    Human-readable messages remain useful for audit reports, but governance and
    lifecycle decisions must consume these codes instead of parsing prose.
    """

    ATOMIC_OCCURRENCE_FAILED = "atomic_occurrence_failed"
    CONTROL_COVERAGE_FAILED = "control_coverage_failed"
    DATA_FLOW_FAILED = "data_flow_failed"
    BOUND_TARGET_FAILED = "bound_target_failed"
    TASK_GAP_REQUIRED = "task_gap_required"
    CHILD_VERSION_MISSING = "child_version_missing"


class CompositeValidator:
    """Composite 高层目标验证。"""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def validate_composite(self, composite: CompositeSkill,
                           node_results: list[NodeValidationResult],
                           final_state: dict[str, Any],
                           inputs: dict[str, Any] | None = None,
                           context: dict[str, Any] | None = None,
                           registry=None) -> NodeValidationResult:
        inputs = inputs or {}
        context = context or {}
        result = NodeValidationResult(node_ref=str(composite.ref), level="composite")
        if not self.enabled:
            result.passed = True
            result.checks["validator_disabled"] = True
            return result

        # 1. 所有子节点 occurrence 的最终结果通过。
        #
        # Runtime 会为同一个计划节点的每次 fallback attempt 都保留一条
        # NodeValidationResult。例如 Seeded 执行失败后由 Dynamic 原地救回时，
        # 序列中会连续出现同一节点的 ``False, True``。前一条仍是 failure
        # branch/Tool 归因所需的真实历史，但不能让已经成功完成的 occurrence
        # 永久污染 Composite 的最终成功语义。
        effective_results = _effective_occurrence_results(node_results)
        all_nodes_ok = all(node.passed for node in effective_results)
        result.checks["all_atomic_nodes_passed"] = all_nodes_ok
        if not all_nodes_ok:
            failed = [node.node_ref for node in effective_results if not node.passed]
            result.messages.append(f"子节点验证失败：{failed}")
            _add_failure_code(
                result, CompositeFailureCode.ATOMIC_OCCURRENCE_FAILED)

        # Exact child versions are part of the immutable Composite contract.
        # The runtime planner normally rejects a missing version before this
        # point; retaining the check here gives replay/audit a stable code too.
        missing_children: list[str] = []
        if registry is not None:
            from ..core.refs import SkillRef
            for step in composite.step_instances():
                try:
                    child = registry.get(SkillRef.parse(str(step["node_ref"])))
                except (KeyError, TypeError, ValueError):
                    child = None
                if child is None:
                    missing_children.append(str(step.get("node_ref") or ""))
        missing_children.extend(str(item) for item in
                                (context.get("missing_child_versions") or []))
        if missing_children:
            result.checks["exact_child_versions_present"] = False
            result.messages.append(
                f"Composite 子节点精确版本缺失：{sorted(set(missing_children))}")
            _add_failure_code(result, CompositeFailureCode.CHILD_VERSION_MISSING)

        # 2. 高层验证规则（composite.validator：谓词名列表，映射到最终状态事实）
        validator = composite.validator or {}
        final_snapshot = StateSnapshot(final_state)
        if isinstance(validator, list):
            check_names = validator
        elif isinstance(validator, dict):
            check_names = validator.get("checks") or validator.get("post_checks") or []
        else:
            check_names = []
        terminal_closure = (
            isinstance(validator, dict)
            and validator.get("check_semantics") == "terminal_effect_closure_v2")
        for check_name in check_names:
            observed_at_end = _check_name_in_facts(
                str(check_name), final_snapshot.facts)
            # New Composites carry code-computed terminal closure and therefore
            # enforce every check in the final state. Historical banks stored a
            # union of intermediate Effects; retain read-only compatibility
            # without a predicate-name whitelist and rely on their Atomic and
            # bound-target validators.
            ok = observed_at_end if terminal_closure else (
                observed_at_end or all_nodes_ok)
            result.checks[f"composite_check:{check_name}"] = ok
            if not ok:
                result.messages.append(f"高层目标未满足：{check_name}")

        # 3. Occurrence-aware control coverage. Membership-only checks let one
        # execution satisfy several repeated steps with the same logical ref.
        control_ok, missing_occurrences = _control_coverage(
            composite, effective_results, context)
        result.checks["control_flow_covered"] = control_ok
        if not control_ok:
            result.messages.append(
                f"Composite occurrence 未完整按序执行：{missing_occurrences}")
            _add_failure_code(
                result, CompositeFailureCode.CONTROL_COVERAGE_FAILED)

        # DATA_FLOW is an executable occurrence edge, not documentation.  New
        # runtimes pass their realized edges in context.  Legacy callers that
        # cannot provide runtime edges keep read compatibility unless they set
        # require_realized_data_flow explicitly.
        data_ok, missing_data = _realized_data_flow_coverage(
            composite, context)
        if data_ok is not None:
            result.checks["data_flow_realized_coverage"] = data_ok
            if not data_ok:
                result.messages.append(
                    f"Composite DATA_FLOW 未在 Runtime Graph 中实现：{missing_data}")
                _add_failure_code(result, CompositeFailureCode.DATA_FLOW_FAILED)

        # 4. Bound target predicates must hold for the requested entity/role;
        # predicate-name-only checks can confuse a bystander object with target.
        target_effects = list(validator.get("target_effects") or [])
        if target_effects:
            from ..core.predicates import check_effects
            target_ok, missing = check_effects(final_snapshot, inputs, target_effects, context)
            result.checks["bound_target_effects"] = target_ok
            if not target_ok:
                result.messages.append(f"绑定后的任务目标未满足：{missing}")
                _add_failure_code(
                    result, CompositeFailureCode.BOUND_TARGET_FAILED)

        task_gap_required = bool(context.get("task_gap_required"))
        gap_analysis = context.get("task_gap_analysis") or {}
        if hasattr(gap_analysis, "to_dict"):
            gap_analysis = gap_analysis.to_dict()
        task_gap_required = task_gap_required or bool(
            isinstance(gap_analysis, dict)
            and gap_analysis.get("missing_effects"))
        if task_gap_required:
            result.checks["self_sufficient_without_task_gap"] = False
            result.messages.append(
                "selected Composite 在自足边界尚需 task-gap 才能完成目标")
            _add_failure_code(result, CompositeFailureCode.TASK_GAP_REQUIRED)

        result.passed = all(result.checks.values())
        return result


def _add_failure_code(result: NodeValidationResult,
                      code: CompositeFailureCode) -> None:
    value = code.value
    if value not in result.failure_codes:
        result.failure_codes.append(value)


def _control_coverage(
        composite: CompositeSkill,
        effective_results: list[NodeValidationResult],
        context: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate exact occurrence order when runtime identity is available."""
    expected_steps = [dict(step) for step in composite.step_instances()]
    expected_ids = [str(step.get("step_id") or "") for step in expected_steps]
    expected_by_id = {str(step.get("step_id") or ""): str(
        step.get("node_ref") or "") for step in expected_steps}
    realized_nodes = list(context.get("realized_nodes") or [])
    realized_by_identity: dict[str, dict[str, Any]] = {}
    for node in realized_nodes:
        if not isinstance(node, dict):
            continue
        for identity in (node.get("occurrence_id"), node.get("step_id")):
            if identity:
                realized_by_identity[str(identity)] = node

    actual_ids: list[str] = []
    exact_identity_available = bool(expected_ids) and bool(
        realized_nodes or any(item.step_id in expected_by_id
                              for item in effective_results if item.step_id))
    if exact_identity_available:
        for result in effective_results:
            if not result.passed:
                continue
            realized = (realized_by_identity.get(str(result.occurrence_id or ""))
                        or realized_by_identity.get(str(result.step_id or ""))
                        or {})
            origin = str(realized.get("origin_step_id") or "")
            if not origin and result.step_id in expected_by_id:
                origin = str(result.step_id)
            if not origin:
                # An explicit Composite cannot be covered by an anonymous
                # occurrence once the runtime supplies occurrence identities.
                actual_ids.append("<unmapped>")
                continue
            expected_ref = expected_by_id.get(origin, "")
            actual_ref = str(realized.get("ref") or result.node_ref or "")
            if expected_ref and _normalized_ref(expected_ref) != _normalized_ref(actual_ref):
                actual_ids.append(f"<wrong-ref:{origin}>")
            else:
                actual_ids.append(origin)
        ok = actual_ids == expected_ids
        missing = expected_ids[len(actual_ids):] if actual_ids == expected_ids[:len(actual_ids)] \
            else [f"expected={expected_ids}", f"actual={actual_ids}"]
        return ok, missing

    # Historical graph compatibility: old artifacts/results did not persist
    # occurrence IDs.  Preserve their logical subsequence semantics, while all
    # newly compiled plans take the exact path above.
    node_refs = [_normalized_ref(node.node_ref)
                 for node in effective_results if node.passed]
    expected = [_normalized_ref(str(step.get("node_ref") or ""))
                for step in expected_steps]
    cursor = 0
    for actual in node_refs:
        if cursor < len(expected) and expected[cursor] == actual:
            cursor += 1
    return cursor == len(expected), expected[cursor:]


def _realized_data_flow_coverage(
        composite: CompositeSkill,
        context: dict[str, Any]) -> tuple[bool | None, list[dict[str, Any]]]:
    expected = [edge for edge in composite.edge_objects()
                if edge.category == "data"]
    if not expected:
        return True, []
    runtime_payloads = context.get("runtime_edges")
    require = bool(context.get("require_realized_data_flow"))
    if runtime_payloads is None:
        return (False, [edge.to_dict() for edge in expected]) if require else (None, [])

    from ..core.edge_ir import GraphEdge
    realized = []
    for payload in runtime_payloads or []:
        try:
            edge = payload if isinstance(payload, GraphEdge) else GraphEdge.from_dict(payload)
        except (TypeError, ValueError):
            continue
        if edge.category == "data":
            realized.append(edge)

    origin_by_runtime: dict[str, str] = {}
    for node in context.get("realized_nodes") or []:
        if not isinstance(node, dict):
            continue
        origin = str(node.get("origin_step_id") or "")
        if not origin:
            continue
        for runtime_id in (node.get("step_id"), node.get("occurrence_id")):
            if runtime_id:
                origin_by_runtime[str(runtime_id)] = origin

    def identity(edge, *, runtime: bool = False) -> tuple[str, str, str, str, str]:
        mapping = edge.mapping or {}
        source_step = str(edge.source_step or "")
        target_step = str(edge.target_step or "")
        if runtime:
            source_step = origin_by_runtime.get(source_step, source_step)
            target_step = origin_by_runtime.get(target_step, target_step)
        return (
            source_step, target_step,
            str(mapping.get("source_output") or ""),
            str(mapping.get("target_input") or ""),
            str(mapping.get("transform") or "identity"),
        )

    realized_by_identity = {identity(edge, runtime=True): edge
                            for edge in realized}
    nodes_by_runtime: dict[str, dict[str, Any]] = {}
    for node in context.get("realized_nodes") or []:
        if not isinstance(node, dict):
            continue
        for runtime_id in (node.get("step_id"), node.get("occurrence_id")):
            if runtime_id:
                nodes_by_runtime[str(runtime_id)] = node

    missing: list[dict[str, Any]] = []
    for edge in expected:
        expected_identity = identity(edge)
        runtime_edge = realized_by_identity.get(expected_identity)
        if runtime_edge is None:
            missing.append({**edge.to_dict(),
                            "realization_error": "runtime_edge_missing"})
            continue
        if not require:
            continue
        mapping = runtime_edge.mapping or {}
        source_output = str(mapping.get("source_output") or "")
        target_input = str(mapping.get("target_input") or "")
        source_node = nodes_by_runtime.get(str(runtime_edge.source_step or ""))
        target_node = nodes_by_runtime.get(str(runtime_edge.target_step or ""))
        if source_node is None or target_node is None:
            missing.append({**edge.to_dict(),
                            "realization_error": "runtime_endpoint_missing"})
            continue
        source_value = dict(source_node.get("outputs") or {}).get(source_output)
        if (not bool(source_node.get("passed"))
                or not is_concrete_binding(source_value)):
            missing.append({**edge.to_dict(),
                            "realization_error": "source_output_not_materialized"})
            continue
        provenance = dict(
            (target_node.get("binding_provenance") or {}).get(target_input)
            or {})
        provenance_source_step = str(provenance.get("source_step") or "")
        provenance_source_origin = origin_by_runtime.get(
            provenance_source_step, provenance_source_step)
        expected_source_origin = origin_by_runtime.get(
            str(runtime_edge.source_step or ""),
            str(runtime_edge.source_step or ""))
        if (str(provenance.get("source") or "") != "data_flow"
                or provenance_source_origin != expected_source_origin
                or str(provenance.get("source_output") or "") != source_output):
            missing.append({**edge.to_dict(),
                            "realization_error": "target_provenance_not_data_flow"})
            continue
        target_value = dict(target_node.get("params") or {}).get(target_input)
        if (not is_concrete_binding(target_value)
                or str(target_value) != str(source_value)):
            missing.append({**edge.to_dict(),
                            "realization_error": "transferred_value_mismatch"})
    return not missing, missing


def _normalized_ref(value: str) -> str:
    text = str(value or "")
    if text.startswith("skill://"):
        text = text[len("skill://"):]
    return text


def _check_name_in_facts(check_name: str, facts: set[str]) -> bool:
    return any(
        isinstance(predicate := _fact_to_predicate(str(fact)), dict)
        and str(predicate.get("predicate") or "") == check_name
        for fact in facts
    )


def _effective_occurrence_results(
        node_results: list[NodeValidationResult]) -> list[NodeValidationResult]:
    """Collapse retry attempts while preserving distinct plan occurrences.

    Attempts for one runtime node are emitted contiguously and execution stops
    retrying that node at its first success.  Therefore a result for the same
    logical node replaces the immediately preceding *failed* effective result.
    A result following a success starts a new occurrence, which is important
    when a Composite intentionally invokes the same Atomic more than once.

    The input objects are never mutated or removed; callers such as failure
    attribution continue to see the complete failed-attempt history.
    """
    effective: list[NodeValidationResult] = []
    occurrence_positions: dict[str, int] = {}
    for node in node_results:
        occurrence = str(node.occurrence_id or node.step_id or "")
        if occurrence and occurrence in occurrence_positions:
            effective[occurrence_positions[occurrence]] = node
        elif occurrence:
            occurrence_positions[occurrence] = len(effective)
            effective.append(node)
        elif (effective and not effective[-1].passed
                and _logical_node_ref(effective[-1].node_ref)
                and _logical_node_ref(effective[-1].node_ref)
                == _logical_node_ref(node.node_ref)):
            effective[-1] = node
        else:
            effective.append(node)
    return effective


def _logical_node_ref(value: str) -> str:
    """Normalize URI/version spelling for retry identity comparison."""
    text = str(value or "")
    if "://" in text:
        text = text.split("://", 1)[1]
    return text.rsplit("@", 1)[0]
