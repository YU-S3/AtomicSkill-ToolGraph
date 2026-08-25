"""Atomic Skill Node Validator（三级验证体系第 2 层，设计文档 v2.0 §35.2）。

回答：该 Tool/Implementation 执行后，Atomic Skill 的核心状态 Effect 是否真的发生？
例：AcquireObject → inventory contains object。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.predicates import (
    StateSnapshot,
    bind_args,
    check_effects,
    evaluate_preconditions,
)
from ..core.skill_ir import AbstractAtomicSkill
from ..core.status import ValidationLevel
from ..core.trace_ir import NodeValidationResult


class NodeValidator:
    """抽象原子技能的前置条件 / 核心 Effect 验证。"""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def validate_atomic(self, atomic: AbstractAtomicSkill,
                        before: dict[str, Any], after: dict[str, Any],
                        inputs: dict[str, Any] | None = None,
                        context: dict[str, Any] | None = None) -> NodeValidationResult:
        inputs = inputs or {}
        context = context or {}
        result = NodeValidationResult(node_ref=str(atomic.ref), level="atomic")
        if not self.enabled:
            result.passed = True
            result.checks["validator_disabled"] = True
            result.before = before
            result.after = after
            return result

        before_snapshot = StateSnapshot(before)
        after_snapshot = StateSnapshot(after)

        # 前置条件
        pre_ok, missing_pre = evaluate_preconditions(before_snapshot, inputs,
                                                      atomic.preconditions, context)
        result.checks["preconditions"] = pre_ok
        if not pre_ok:
            result.messages.append(f"前置条件未满足：{missing_pre}")

        # 核心 Effect
        effect_ok, missing_effect = check_effects(after_snapshot, inputs,
                                                  atomic.effects, context)
        result.checks["effects"] = effect_ok
        if not effect_ok:
            result.messages.append(f"核心 Effect 未发生：{missing_effect}")

        # 声明的 validator pre/post checks
        validator = atomic.validator or {}
        for check_name in validator.get("pre_checks") or []:
            # 已覆盖于 preconditions；在此只做占位检查记录
            result.checks[f"pre_check:{check_name}"] = True
        for check_name in validator.get("post_checks") or []:
            normalized_check = str(check_name).replace(".", "_")
            covered = any(
                normalized_check in str(effect.get("predicate", "")).replace(".", "_")
                for effect in atomic.effects
            )
            result.checks[f"post_check:{check_name}"] = covered
            if not covered:
                result.messages.append(f"post_check 未覆盖：{check_name}")

        result.before = before
        result.after = after
        # Preconditions describe eligibility before execution. In partially observed
        # environments an unobserved precondition must block Direct, but must not
        # invalidate an Effect that was actually achieved by Seeded/Dynamic execution.
        outcome_checks = [value for name, value in result.checks.items()
                          if name != "preconditions" and not name.startswith("pre_check:")]
        result.passed = all(outcome_checks) if outcome_checks else effect_ok
        return result
