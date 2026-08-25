"""Composite Validator（三级验证体系第 3 层，设计文档 v2.0 §35.3）。

回答：高层 Skill 的组合目标是否满足？Benchmark Verifier 回答整个任务是否成功。
四层（Tool Test → Atomic Node Validator → Composite Validator → Benchmark Verifier）
不能互相替代。
"""

from __future__ import annotations

from typing import Any

from ..core.predicates import StateSnapshot, bind_args, check_effects
from ..core.skill_ir import CompositeSkill
from ..core.trace_ir import NodeValidationResult


class CompositeValidator:
    """Composite 高层目标验证。"""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def validate_composite(self, composite: CompositeSkill,
                           node_results: list[NodeValidationResult],
                           final_state: dict[str, Any],
                           inputs: dict[str, Any] | None = None,
                           context: dict[str, Any] | None = None) -> NodeValidationResult:
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
        # NodeValidationResult。例如 Seeded Heat 失败后由 Dynamic 原地救回时，
        # 序列中会连续出现 ``Heat(False), Heat(True)``。前一条仍是 failure
        # branch/Tool 归因所需的真实历史，但不能让已经成功完成的 occurrence
        # 永久污染 Composite 的最终成功语义。
        effective_results = _effective_occurrence_results(node_results)
        all_nodes_ok = all(node.passed for node in effective_results)
        result.checks["all_atomic_nodes_passed"] = all_nodes_ok
        if not all_nodes_ok:
            failed = [node.node_ref for node in effective_results if not node.passed]
            result.messages.append(f"子节点验证失败：{failed}")

        # 2. 高层验证规则（composite.validator：谓词名列表，映射到最终状态事实）
        validator = composite.validator or {}
        final_snapshot = StateSnapshot(final_state)
        if isinstance(validator, list):
            check_names = validator
        elif isinstance(validator, dict):
            check_names = validator.get("checks") or validator.get("post_checks") or []
        else:
            check_names = []
        for check_name in check_names:
            # 兼容历史 Composite 中曾保存的中间态检查；它们由对应 Atomic
            # Validator 证明“曾达成”，不要求在任务最终状态继续保持。
            transient = str(check_name) in {
                "agent.holds", "container.open", "location.checked",
                "object.is_accessible", "object.exists",
            }
            ok = ((transient and all_nodes_ok)
                  or _check_name_in_facts(str(check_name), final_snapshot.facts))
            result.checks[f"composite_check:{check_name}"] = ok
            if not ok:
                result.messages.append(f"高层目标未满足：{check_name}")

        # 3. Occurrence-aware control coverage. Membership-only checks let one
        # execution satisfy several repeated steps with the same logical ref.
        node_refs = [node.node_ref.rsplit("@", 1)[0].split(":")[-1]
                     for node in effective_results if node.passed]
        expected = [str(n).rsplit("@", 1)[0] for n in composite.nodes()]
        cursor = 0
        for actual in node_refs:
            if cursor < len(expected) and expected[cursor] in actual:
                cursor += 1
        result.checks["control_flow_covered"] = cursor == len(expected)
        if cursor != len(expected):
            result.messages.append(f"Composite occurrence 未完整按序执行：{expected[cursor:]}")

        # 4. Bound target predicates must hold for the requested entity/role;
        # predicate-name-only checks can confuse a bystander object with target.
        target_effects = list(validator.get("target_effects") or [])
        if target_effects:
            from ..core.predicates import check_effects
            target_ok, missing = check_effects(final_snapshot, inputs, target_effects, context)
            result.checks["bound_target_effects"] = target_ok
            if not target_ok:
                result.messages.append(f"绑定后的任务目标未满足：{missing}")

        result.passed = all(result.checks.values())
        return result


def _check_name_in_facts(check_name: str, facts: set[str]) -> bool:
    aliases = {
        "object.at_location": "object_at(",
        "agent.holds": "agent_holds(",
        "object.heated": "object_heated(",
        "object.cleaned": "object_cleaned(",
        "object.cooled": "object_cooled(",
        "container.open": "container_open(",
        "tests.pass": "tests_pass(",
        "callable.returns_expected": "callable_returns_expected(",
    }
    prefix = aliases.get(check_name, check_name.replace(".", "_") + "(")
    return any(str(fact).startswith(prefix) for fact in facts)


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
    for node in node_results:
        if (effective and not effective[-1].passed
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
