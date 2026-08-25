"""Tool Resolver：在 Skill/Implementation 选定后完成参数绑定（设计文档 v2.0 §13、§23）。

LLM 不直接看到 Tool 列表：Planner 只规划 Atomic Skill，Implementation Selector
选择实现，Resolver 负责 tool_ref 与 parameter_mapping 的具体绑定。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.skill_ir import ImplementationAtom, ToolBinding
from .registry import ToolRegistry


@dataclass
class ResolvedTool:
    binding: ToolBinding
    tool: Any
    parameters: dict[str, Any] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_ref": str(self.binding.tool_ref),
            "role": self.binding.role,
            "parameters": self.parameters,
            "missing": self.missing,
        }


class ToolResolver:
    """解析 Implementation Atom 的 Tool 绑定。"""

    def __init__(self, tool_registry: ToolRegistry) -> None:
        self.tool_registry = tool_registry

    def resolve(self, implementation: ImplementationAtom,
                context: dict[str, Any]) -> list[ResolvedTool]:
        inputs = dict(context.get("inputs") or {})
        ctx = dict(context)
        resolved: list[ResolvedTool] = []
        for binding in implementation.tool_bindings:
            tool = self.tool_registry.get(binding.tool_ref)
            if tool is None:
                # 版本指针可能被推荐版本替代
                tool = self.tool_registry.get_recommended(binding.tool_ref.tool_id)
            if tool is None or not tool.is_usable():
                resolved.append(ResolvedTool(binding=binding, tool=tool,
                                             missing=["tool_unavailable"]))
                continue
            parameters, missing = self._bind(binding, tool, inputs, ctx)
            resolved.append(ResolvedTool(binding=binding, tool=tool,
                                         parameters=parameters, missing=missing))
        return resolved

    def _bind(self, binding: ToolBinding, tool, inputs: dict[str, Any],
              ctx: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        parameters: dict[str, Any] = {}
        missing: list[str] = []
        mapping = binding.parameter_mapping or {}
        # 未显式映射的参数：同名输入直通
        for param in tool.param_names():
            if param in mapping:
                value = _resolve_value(mapping[param], inputs, ctx)
                if value is _MISSING:
                    missing.append(param)
                else:
                    parameters[param] = value
            elif param in inputs:
                parameters[param] = inputs[param]
            else:
                missing.append(param)
        return parameters, missing


_MISSING = object()


def _resolve_value(spec: Any, inputs: dict[str, Any], ctx: dict[str, Any]) -> Any:
    """解析绑定值：`$inputs.x` / `$context.x.y` / 字面量。"""
    if not isinstance(spec, str) or not spec.startswith("$"):
        return spec
    path = spec[1:].split(".")
    if path and path[0] == "inputs":
        return _nested_get(inputs, path[1:])
    if path and path[0] == "context":
        return _nested_get(ctx, path[1:])
    return _MISSING


def _nested_get(data: dict[str, Any], path: list[str]) -> Any:
    current: Any = data
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return _MISSING
    return current
