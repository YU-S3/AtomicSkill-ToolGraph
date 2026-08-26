"""Global Tool Repository（设计文档 v2.0 §18、§40、§56.3）。

存储布局：
    data/tools/
    ├── registry.json
    └── <tool_id>/<version>/tool.json   （artifact 内联存储，第一版）

版本不可变；rollback = 恢复推荐指针；历史 artifact 不被物理覆盖（§38.4）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator

from ..core.refs import ToolRef
from ..core.status import ToolLifecycle, USABLE_TOOL_STATUSES, tool_transition_allowed
from ..core.tool_ir import ToolAsset, lifecycle_rank
from ..persistence import atomic_write_json


class ToolRegistry:
    """集中式 Tool Repository 的文件存储实现。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.registry_path = self.root / "registry.json"
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._write_index({})

    # ------------------------------------------------------------------
    def _read_index(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        return raw if isinstance(raw, dict) else {}

    def _write_index(self, index: dict[str, Any]) -> None:
        atomic_write_json(self.registry_path, index)

    def _tool_path(self, tool_id: str, version: str) -> Path:
        return self.root / tool_id / version / "tool.json"

    def _save_tool(self, tool: ToolAsset) -> None:
        path = self._tool_path(tool.tool_id, tool.ref.version)
        atomic_write_json(path, tool.to_dict())

    # ------------------------------------------------------------------
    def register(self, tool: ToolAsset) -> ToolRef:
        """注册新 Tool 版本。

        - 新 tool_id：初始状态必须是 draft / admission_pending / candidate / shadow
        - 已有 tool_id：新版本（状态由 admission 决定，此处不做强迁移检查）
        """
        errors = tool.validate()
        if errors:
            raise ValueError(f"{tool.tool_id} 校验失败：{errors}")
        index = self._read_index()
        existing = index.get(tool.tool_id)
        if existing is None:
            if tool.status not in (ToolLifecycle.DRAFT, ToolLifecycle.ADMISSION_PENDING,
                                   ToolLifecycle.CANDIDATE, ToolLifecycle.SHADOW):
                raise ValueError(
                    f"新 Tool {tool.tool_id} 初始状态非法：{tool.status.value}（须经 draft→admission→candidate）")
        else:
            # 同 ref 只允许同一 executable identity 的幂等证据更新。不同
            # 模板/签名必须使用新的 shape id 或版本，绝不能覆盖旧 artifact。
            prior = self.get(tool.ref)
            if prior is not None and prior.tool_id == tool.tool_id:
                if (prior.structural_hash() != tool.structural_hash()
                        or prior.interface != tool.interface):
                    raise ValueError(
                        f"immutable_tool_version_collision: {tool.ref} 已存在且"
                        " executable/interface 不同")
                tool.statistics = _merge_statistics(prior.statistics, tool.statistics)
                sources = list(prior.provenance.get("source_trace_ids") or [])
                for tid in tool.provenance.get("source_trace_ids") or []:
                    if tid not in sources:
                        sources.append(tid)
                # Independent successful source traces are the canonical Tool
                # support evidence.  A shadow executable can later pass
                # admission under the same immutable ref; max-merging two
                # freshly compiled ``support_count=1`` records used to leave it
                # at one even though provenance already contained two traces,
                # preventing lifecycle activation and Direct reuse.
                tool.statistics["support_count"] = max(
                    int(tool.statistics.get("support_count", 0)), len(sources))
                task_types = list(prior.provenance.get("source_task_types") or [])
                for task_type in tool.provenance.get("source_task_types") or []:
                    if task_type not in task_types:
                        task_types.append(task_type)
                # Preserve the already-admitted executable contract and replay
                # case; only evidence/statistics/lifecycle are mutable in place.
                tool.summary = prior.summary
                tool.signature = prior.signature
                tool.interface = prior.interface
                tool.artifact = prior.artifact
                tool.tests = prior.tests
                tool.safety = prior.safety
                tool.lineage = prior.lineage
                tool.provenance = {
                    **prior.provenance,
                    **tool.provenance,
                    "source_trace_ids": sources,
                    "source_task_types": task_types,
                }
        self._save_tool(tool)
        versions = sorted(set(list((existing or {}).get("versions") or []) + [tool.ref.version]),
                          key=_semver_key)
        index[tool.tool_id] = {
            "current_version": tool.ref.version,
            "recommended_version": (existing or {}).get("recommended_version") or versions[0],
            "latest_status": tool.status.value,
            "versions": versions,
        }
        self._write_index(index)
        return tool.ref

    def get(self, ref: ToolRef) -> ToolAsset | None:
        path = self._tool_path(ref.tool_id, ref.version)
        if not path.exists():
            return None
        try:
            return ToolAsset.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValueError):
            return None

    def get_latest(self, tool_id: str) -> ToolAsset | None:
        index = self._read_index().get(tool_id)
        if index is None:
            return None
        return self.get(ToolRef(tool_id, index["current_version"]))

    def get_recommended(self, tool_id: str) -> ToolAsset | None:
        index = self._read_index().get(tool_id)
        if index is None:
            return None
        version = index.get("recommended_version") or index["current_version"]
        return self.get(ToolRef(tool_id, version))

    def index_entry(self, tool_id: str) -> dict[str, Any] | None:
        return self._read_index().get(tool_id)

    def list_versions(self, tool_id: str) -> list[str]:
        index = self._read_index().get(tool_id)
        return list(index.get("versions") or []) if index else []

    def list_all(self, *, statuses: set[ToolLifecycle] | None = None) -> list[ToolAsset]:
        index = self._read_index()
        tools: list[ToolAsset] = []
        for tool_id, entry in index.items():
            tool = self.get(ToolRef(tool_id, entry.get("recommended_version") or entry["current_version"]))
            if tool is None:
                continue
            if statuses is not None and tool.status not in statuses:
                continue
            tools.append(tool)
        return tools

    def list_usable(self) -> list[ToolAsset]:
        """可直接调用（candidate/active/preferred）的 Tool。"""
        return self.list_all(statuses=USABLE_TOOL_STATUSES)

    def iter_all_versions(self, tool_id: str) -> Iterator[ToolAsset]:
        for version in self.list_versions(tool_id):
            tool = self.get(ToolRef(tool_id, version))
            if tool is not None:
                yield tool

    # ------------------------------------------------------------------
    def set_status(self, ref: ToolRef, status: ToolLifecycle) -> ToolAsset:
        tool = self.get(ref)
        if tool is None:
            raise KeyError(str(ref))
        current = tool.status
        if not tool_transition_allowed(current, status):
            raise ValueError(f"非法 Tool 生命周期迁移：{current.value} -> {status.value}")
        tool.status = status
        self._save_tool(tool)
        index = self._read_index()
        entry = index.get(ref.tool_id)
        if entry and entry.get("recommended_version") == ref.version:
            entry["latest_status"] = status.value
            self._write_index(index)
        return tool

    def recommend(self, ref: ToolRef) -> None:
        index = self._read_index()
        entry = index.get(ref.tool_id)
        if entry is None or ref.version not in (entry.get("versions") or []):
            raise KeyError(str(ref))
        entry["recommended_version"] = ref.version
        tool = self.get(ref)
        if tool is not None:
            entry["latest_status"] = tool.status.value
        self._write_index(index)

    def rollback(self, tool_id: str, version: str) -> ToolRef:
        ref = ToolRef(tool_id, version)
        self.recommend(ref)
        return ref

    def record_feedback(self, ref: ToolRef, success: bool, *, usage_mode: str = "direct") -> ToolAsset | None:
        """记录一次使用反馈并更新统计（§39.3：Skill 与 Tool 分开记统计）。"""
        tool = self.get(ref)
        if tool is None:
            return None
        tool.record_usage(success, usage_mode=usage_mode)
        self._save_tool(tool)
        return tool

    def update_tool(self, tool: ToolAsset) -> None:
        """原地更新已注册 Tool 的统计/元数据（不改变版本）。"""
        self._save_tool(tool)

    def find_duplicates(self, tool: ToolAsset) -> list[ToolAsset]:
        """按结构化哈希找重复实现（合并候选）。"""
        target_hash = tool.structural_hash()
        return [t for t in self.list_all() if t.structural_hash() == target_hash
                and t.ref != tool.ref]

    def stats(self) -> dict[str, Any]:
        index = self._read_index()
        by_status: dict[str, int] = {}
        total = 0
        for entry in index.values():
            by_status[entry.get("latest_status", "?")] = by_status.get(entry.get("latest_status", "?"), 0) + 1
            total += 1
        return {"tools": total, "by_status": by_status}


def _semver_key(version: str) -> tuple[int, int, int]:
    parts = version.split(".")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def _merge_statistics(prior: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """同 ref 重复注册时保留累计证据（call/success/failure/utility）。"""
    merged = dict(incoming)
    for key in ("call_count", "success_count", "failure_count",
                 "direct_use_count", "direct_success_count", "direct_failure_count",
                 "support_count", "consecutive_failures"):
        merged[key] = max(int(prior.get(key, 0)), int(incoming.get(key, 0)))
    if prior.get("utility") is not None:
        merged["utility"] = prior.get("utility")
    if prior.get("success_rate") is not None:
        merged["success_rate"] = prior.get("success_rate")
    return merged
