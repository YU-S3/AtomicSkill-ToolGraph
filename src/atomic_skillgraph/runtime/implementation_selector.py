"""Implementation Selector（设计文档 v2.0 §13、§22）。

LLM 不直接选择 Tool：Planner 选定 Atomic Skill 后，本模块在已实现该 Abstract
的 Implementation Atom 中选择（兼容性、历史质量、1:1/N:M 约束）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.config import SystemConfig
from ..core.refs import SkillRef
from ..core.skill_ir import ImplementationAtom
from ..core.status import SkillNodeKind, SkillStatus
from ..graph.registry import SkillGraphRegistry


@dataclass
class ImplementationChoice:
    implementation: ImplementationAtom | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation": str(self.implementation.ref) if self.implementation else None,
            "reason": self.reason,
        }


class ImplementationSelector:
    """选择 Atomic Skill 的具体实现。"""

    def __init__(self, registry: SkillGraphRegistry, tool_resolver,
                 config: SystemConfig) -> None:
        self.registry = registry
        self.tool_resolver = tool_resolver
        self.config = config

    def select(self, atomic_ref: SkillRef, context: dict[str, Any]) -> ImplementationChoice:
        """返回 (implementation | None, reason)。None 表示无可用实现 → 上层走动态兜底。"""
        ranked = self.rank(atomic_ref, context)
        if not ranked:
            candidates = self._implementations_of(atomic_ref)
            return ImplementationChoice(
                reason="no_implementation" if not candidates else
                       "no_bindable_implementation")
        return ranked[0]

    def rank(self, atomic_ref: SkillRef,
             context: dict[str, Any]) -> list[ImplementationChoice]:
        """返回全部可绑定实现，供 Runtime 逐个通过 Direct Gate。

        排序只是偏好，不等于执行许可；可靠性、前置条件和验证仍由 Direct Gate
        决定。框架若已经建立了某些执行前置状态，会优先选择不重复这些准备动作
        的最小实现，但不会依据动作动词或预定义能力名称识别实现。
        """
        candidates = self._implementations_of(atomic_ref)
        if not candidates:
            return []

        features = self.config.features
        ranked: list[tuple[float, ImplementationAtom]] = []
        for impl in candidates:
            if impl.status != SkillStatus.ACTIVE:
                continue
            # 兼容性：compatibility.harness 与 context.harness 匹配（宽松）
            harness = (impl.compatibility or {}).get("harness", "")
            if harness and context.get("harness") and harness != context["harness"]:
                continue
            # 可绑定性：至少一个绑定能解析到可用 Tool
            resolved = self.tool_resolver.resolve(impl, context)
            usable = [r for r in resolved if r.ok and r.tool is not None and r.tool.is_usable()]
            if not usable:
                continue
            # 1:1 约束（消融：强制 Skill:Tool=1:1）
            if not features.enable_nm_binding and len(impl.tool_bindings) != 1:
                continue
            quality = impl.quality or {}
            score = float(quality.get("utility", 0.5))
            success = int(quality.get("success_count", 0))
            score += 0.02 * min(success, 10)
            # 外部准备已经建立了可验证的位置/可达性状态时，最短的符号化实现
            # 与当前状态最匹配。这里只计算 Tool 的步骤数，不检查动作词，也不
            # 假定当前 Atomic 属于任何预定义操作类别。
            if context.get("prefer_minimal_after_preparation"):
                score += _prepared_state_fit_score(usable)
            ranked.append((score, impl))

        if not ranked:
            return []
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [ImplementationChoice(implementation=impl,
                                     reason=f"ranked_by_utility:{score:.3f}")
                for score, impl in ranked]

    def select_allowing_missing(self, atomic_ref: SkillRef, context: dict[str, Any],
                                allowed_missing: set[str]) -> ImplementationChoice:
        """为受控参数发现选择实现；只放行明确列出的缺失参数。

        该方法不绕过 Tool 生命周期、harness、1:1/N:M 或可用性门禁。它只允许
        resolver 因 ``allowed_missing`` 中的参数暂时未绑定，发现完成后仍须再走
        标准 ``select`` 与 Direct gate。
        """
        ranked: list[tuple[float, ImplementationAtom]] = []
        for impl in self._implementations_of(atomic_ref):
            if impl.status != SkillStatus.ACTIVE:
                continue
            harness = (impl.compatibility or {}).get("harness", "")
            if harness and context.get("harness") and harness != context["harness"]:
                continue
            if not self.config.features.enable_nm_binding and len(impl.tool_bindings) != 1:
                continue
            resolved = self.tool_resolver.resolve(impl, context)
            if not resolved:
                continue
            acceptable = True
            for item in resolved:
                if item.tool is None or not item.tool.is_usable():
                    acceptable = False
                    break
                missing = set(getattr(item, "missing", []) or [])
                if not missing.issubset(allowed_missing):
                    acceptable = False
                    break
            if not acceptable:
                continue
            quality = impl.quality or {}
            score = float(quality.get("utility", 0.5))
            score += 0.02 * min(int(quality.get("success_count", 0)), 10)
            ranked.append((score, impl))
        if not ranked:
            return ImplementationChoice(reason="no_discoverable_implementation")
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ImplementationChoice(ranked[0][1],
                                    f"selected_for_parameter_discovery:{ranked[0][0]:.3f}")

    def discoverable_location_slots(self, atomic_ref: SkillRef,
                                    context: dict[str, Any]) -> set[str]:
        """Return location-only gaps of the best otherwise-bindable Tool.

        The parameter names are learned Tool contracts. No task label, action
        verb, entity class, or benchmark operation is consulted.
        """
        ranked: list[tuple[float, set[str]]] = []
        for impl in self._implementations_of(atomic_ref):
            if impl.status != SkillStatus.ACTIVE:
                continue
            harness = (impl.compatibility or {}).get("harness", "")
            if harness and context.get("harness") and harness != context["harness"]:
                continue
            if not self.config.features.enable_nm_binding and len(impl.tool_bindings) != 1:
                continue
            resolved = self.tool_resolver.resolve(impl, context)
            if not resolved or any(item.tool is None or not item.tool.is_usable()
                                   for item in resolved):
                continue
            missing = {str(value) for item in resolved
                       for value in (item.missing or [])}
            if not missing or not all(value.endswith("_location")
                                      for value in missing):
                continue
            quality = impl.quality or {}
            score = float(quality.get("utility", 0.5))
            score += 0.02 * min(int(quality.get("success_count", 0)), 10)
            ranked.append((score, missing))
        if not ranked:
            return set()
        ranked.sort(key=lambda item: item[0], reverse=True)
        return set(ranked[0][1])

    def _implementations_of(self, atomic_ref: SkillRef) -> list[ImplementationAtom]:
        result: list[ImplementationAtom] = []
        for impl in self.registry.list_all_versions(SkillNodeKind.IMPLEMENTATION_ATOMIC):
            if impl.abstract_ref.logical_id == atomic_ref.logical_id:
                result.append(impl)
        return result


def _prepared_state_fit_score(resolved_tools: list[Any]) -> float:
    """Prefer the least redundant executable shape after verified preparation.

    The score is deliberately structural. It has no benchmark verb list and
    therefore works for any learned Tool whose alternative implementations
    differ only by already-satisfied setup steps.
    """
    step_count = sum(
        1
        for item in resolved_tools
        for step in (item.tool.artifact.get("steps") or [])
        if str(step).strip()
    )
    # One redundant step must dominate the bounded historical utility bonus;
    # otherwise a high-support setup-heavy shape still wins after that setup
    # has already been executed by controlled discovery.
    return 10.0 - 2.0 * min(step_count, 50)
