"""持久化基础：TraceStore / RunStore / ProposalStore / MetricsStore。

所有 JSON 文件存储，原子写（tmp + replace），与设计文档 v2.0 §40 布局一致。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterator

from .core.trace_ir import TaskExecutionInstance, TraceRecord


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically, tolerating short Windows/AV replace locks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.02 * (attempt + 1))


def _atomic_write(path: Path, payload: Any) -> None:
    atomic_write_json(path, payload)


def _atomic_read(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


class TraceStore:
    """v2.0 规范 Trace 的持久化（data/traces/{trace_id}.json）。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, trace: TraceRecord) -> Path:
        path = self.root / f"{trace.trace_id}.json"
        _atomic_write(path, trace.to_dict())
        return path

    def load(self, trace_id: str) -> TraceRecord | None:
        path = self.root / f"{trace_id}.json"
        data = _atomic_read(path, None)
        return TraceRecord.from_dict(data) if isinstance(data, dict) else None

    def load_path(self, path: str | Path) -> TraceRecord | None:
        data = _atomic_read(Path(path), None)
        return TraceRecord.from_dict(data) if isinstance(data, dict) else None

    def iter_traces(self) -> Iterator[TraceRecord]:
        for path in sorted(self.root.glob("*.json")):
            trace = self.load_path(path)
            if trace is not None:
                yield trace

    def by_task_type(self, task_type: str, *, success_only: bool = True) -> list[TraceRecord]:
        result = []
        for trace in self.iter_traces():
            if trace.task_type != task_type:
                continue
            if success_only and not trace.success:
                continue
            result.append(trace)
        return result


class RunStore:
    """Task Execution Instance 持久化（data/runtime_runs/）。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, instance: TaskExecutionInstance) -> Path:
        path = self.root / f"{instance.execution_id}.json"
        _atomic_write(path, instance.to_dict())
        return path


class ProposalStore:
    """失败轨迹产生的修复提案（shadow，不直接激活；data/evolution/proposals.json）。

    Failure proposes; successful replay admits.（§33.2）
    """

    def __init__(self, root: str | Path) -> None:
        self.path = Path(root) / "evolution" / "proposals.json"

    def add(self, kind: str, trace_id: str, target_ref: str, reason: str,
            payload: dict[str, Any] | None = None) -> dict[str, Any]:
        proposals = _atomic_read(self.path, [])
        proposal = {
            "proposal_id": f"prop_{int(time.time() * 1000)}_{len(proposals)}",
            "kind": kind,               # tool_update | add_tool_test | split_tool |
                                        # contract_revision | composite_revision
            "trace_id": trace_id,
            "target_ref": target_ref,
            "reason": reason,
            "payload": payload or {},
            "status": "pending_replay",  # pending_replay | replayed | rejected
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        proposals.append(proposal)
        _atomic_write(self.path, proposals)
        return proposal

    def list_all(self) -> list[dict[str, Any]]:
        return _atomic_read(self.path, [])

    def mark(self, proposal_id: str, status: str,
             result: dict[str, Any] | None = None) -> None:
        proposals = self.list_all()
        for proposal in proposals:
            if proposal["proposal_id"] == proposal_id:
                proposal["status"] = status
                proposal["replay_result"] = result or {}
                proposal["resolved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _atomic_write(self.path, proposals)

    def pending(self) -> list[dict[str, Any]]:
        return [proposal for proposal in self.list_all()
                if proposal.get("status") == "pending_replay"]


class MetricsStore:
    """实验指标累积（data/metrics/）。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_episode(self, episode_index: int, record: dict[str, Any]) -> None:
        path = self.root / f"episode_{episode_index:05d}.json"
        _atomic_write(path, record)

    def save_summary(self, key: str, payload: Any) -> None:
        _atomic_write(self.root / f"{key}.json", payload)
