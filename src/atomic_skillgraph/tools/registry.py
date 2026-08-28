"""Global Tool Repository（设计文档 v2.0 §18、§40、§56.3）。

存储布局：
    data/tools/
    ├── registry.json
    └── <tool_id>/<version>/tool.json   （artifact 内联存储，第一版）

版本不可变；rollback = 恢复推荐指针；历史 artifact 不被物理覆盖（§38.4）。
"""

from __future__ import annotations

import json
import inspect
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from ..core.refs import ToolRef
from ..core.status import ToolLifecycle, USABLE_TOOL_STATUSES, tool_transition_allowed
from ..core.tool_ir import ToolAsset, lifecycle_rank
from ..evidence_store import (
    EvidenceStore,
    evidence_hash,
    evidence_pointer_valid,
)
from ..persistence import atomic_write_json
from ..persistence_guard import validate_long_term_asset


class ToolRegistry:
    """集中式 Tool Repository 的文件存储实现。"""

    def __init__(self, root: str | Path, *,
                 evidence_root: str | Path | None = None) -> None:
        self.root = Path(root)
        self.registry_path = self.root / "registry.json"
        # Private replay evidence is a sibling of the portable Tool bank.
        self.evidence_store = EvidenceStore(
            evidence_root or (self.root.parent / "evidence" / "tool_tests"))
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
        self._externalize_tests(tool)
        payload = tool.to_dict()
        findings = validate_long_term_asset(
            payload, asset_kind=f"tool:{tool.artifact_kind.value}")
        if findings:
            raise ValueError(
                "long_term_asset_guard_failed:"
                f"{tool.ref}:" + ";".join(findings[:12]))
        path = self._tool_path(tool.tool_id, tool.ref.version)
        atomic_write_json(path, payload)

    def _externalize_tests(self, tool: ToolAsset) -> None:
        """Move raw Tool tests/replay cases into the private EvidenceStore."""
        persisted: list[dict[str, Any]] = []
        raw_cases: list[dict[str, Any]] = []
        unresolved: list[str] = []
        hydrated_by_hash: dict[str, list[dict[str, Any]]] = {}
        for hydrated in tool.all_test_cases():
            if isinstance(hydrated, dict):
                hydrated_by_hash.setdefault(
                    evidence_hash(hydrated), []).append(dict(hydrated))
        for raw in list(tool.tests or []):
            case = dict(raw)
            ref_text = str(case.get("evidence_ref") or "")
            if ref_text.startswith("evidence://"):
                if not evidence_pointer_valid(case, kind="tool_test"):
                    raise ValueError(
                        f"invalid_tool_evidence_pointer:{tool.ref}:{ref_text}")
                persisted.append({
                    "evidence_ref": ref_text,
                    "evidence_hash": str(case.get("evidence_hash") or ""),
                })
                payload = self.evidence_store.get(case)
                if not isinstance(payload, dict):
                    # Cross-repository branch merges carry refs issued by the
                    # branch EvidenceStore plus transient hydrated payloads.
                    # Materialize that verified payload in this repository;
                    # never persist a dangling ref in a mutable main bank.
                    candidates = hydrated_by_hash.get(
                        str(case.get("evidence_hash") or ""), [])
                    payload = candidates[0] if candidates else None
                    if isinstance(payload, dict):
                        source = dict(payload.get("source") or {})
                        local_ref = self.evidence_store.put(
                            "tool_test", payload,
                            trace_id=str(
                                payload.get("source_trace_id") or ""),
                            event_start=_optional_int(payload.get(
                                "event_start", source.get("event_start"))),
                            event_end=_optional_int(payload.get(
                                "event_end", source.get("event_end"))),
                        )
                        if local_ref != persisted[-1]:
                            raise ValueError(
                                f"tool_evidence_identity_mismatch:"
                                f"{tool.ref}:{ref_text}")
                if isinstance(payload, dict):
                    raw_cases.append(payload)
                else:
                    unresolved.append(ref_text)
                    raise ValueError(
                        f"missing_tool_evidence_payload:{tool.ref}:{ref_text}")
                continue
            source = dict(case.get("source") or {})
            evidence_ref = self.evidence_store.put(
                "tool_test", case,
                trace_id=str(case.get("source_trace_id") or ""),
                event_start=_optional_int(
                    case.get("event_start", source.get("event_start"))),
                event_end=_optional_int(
                    case.get("event_end", source.get("event_end"))),
            )
            persisted.append(evidence_ref)
            raw_cases.append(case)
        tool.tests = persisted
        if tool._resolved_tests is None:
            tool.set_resolved_tests(raw_cases, unresolved_refs=unresolved)

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
                if (prior.admission_contract_hash()
                        != tool.admission_contract_hash()):
                    raise ValueError(
                        f"immutable_tool_version_collision: {tool.ref} 已存在且"
                        " executable/interface/safety contract 不同")
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
                # Preserve the immutable executable contract.  Replay and
                # safety evidence require a more careful rule: a formerly
                # Shadow ref may later pass Admission with fresh successful
                # evidence.  Overwriting that incoming test set/certificate
                # with the old Shadow payload would make the just-admitted
                # Tool unverifiable after persistence.
                incoming_admitted = tool.admission_certificate_valid()
                tool.summary = prior.summary
                tool.signature = prior.signature
                tool.interface = prior.interface
                tool.artifact = prior.artifact
                if not incoming_admitted:
                    tool.tests = prior.tests
                    tool.set_resolved_tests(
                        prior.replay_cases(),
                        unresolved_refs=list(prior._unresolved_test_refs),
                    )
                    tool.safety = prior.safety
                tool.lineage = prior.lineage
                tool.provenance = {
                    **prior.provenance,
                    **tool.provenance,
                    "source_trace_ids": sources,
                    "source_task_types": task_types,
                }
        if (tool.status in USABLE_TOOL_STATUSES
                and not tool.admission_certificate_valid()):
            raise ValueError(
                f"usable_tool_requires_valid_admission_certificate:{tool.ref}")
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
            tool = ToolAsset.from_dict(
                json.loads(path.read_text(encoding="utf-8")))
            resolved: list[dict[str, Any]] = []
            unresolved: list[str] = []
            has_pointer = False
            for item in tool.tests:
                ref_text = str(item.get("evidence_ref") or "")
                if not ref_text.startswith("evidence://"):
                    # Compatibility for old banks.  A subsequent write
                    # automatically migrates this case out of tool.json.
                    resolved.append(dict(item))
                    continue
                has_pointer = True
                if not evidence_pointer_valid(item, kind="tool_test"):
                    unresolved.append(ref_text or "invalid_evidence_pointer")
                    continue
                payload = self.evidence_store.get(item)
                if isinstance(payload, dict):
                    resolved.append(payload)
                else:
                    unresolved.append(ref_text)
            if has_pointer:
                tool.set_resolved_tests(
                    resolved, unresolved_refs=unresolved)
            return tool
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
        return [tool for tool in self.list_all(statuses=USABLE_TOOL_STATUSES)
                if tool.is_usable()]

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
        if (status in USABLE_TOOL_STATUSES
                and not tool.admission_certificate_valid()):
            raise ValueError(
                f"usable_tool_requires_valid_admission_certificate:{tool.ref}")
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
        prior = self.get(tool.ref)
        if prior is None:
            raise KeyError(str(tool.ref))
        if prior.admission_contract_hash() != tool.admission_contract_hash():
            raise ValueError(
                f"immutable_tool_contract_update:{tool.ref};"
                "artifact/signature/interface/safety 变化必须新版本并重新 Admission")
        if prior.test_evidence_identities() != tool.test_evidence_identities():
            raise ValueError(
                f"immutable_tool_evidence_update:{tool.ref};"
                "tests/evidence 变化必须重新 Admission")
        self._save_tool(tool)

    def audit_frozen_readiness(self) -> dict[str, Any]:
        """Read-only audit for exporting this repository without evidence."""
        issues: list[dict[str, str]] = []
        scanned = 0
        for path in sorted(self.root.glob("*/*/tool.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                tool = ToolAsset.from_dict(payload)
            except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
                issues.append({"path": str(path), "code": "unreadable_tool",
                               "detail": f"{type(exc).__name__}:{exc}"})
                continue
            scanned += 1
            raw_tests = [item for item in tool.tests
                         if not str(item.get("evidence_ref") or "").startswith(
                             "evidence://")]
            if raw_tests:
                issues.append({"path": str(path), "code": "raw_tests_embedded",
                               "detail": f"count={len(raw_tests)}"})
            invalid_pointers = [
                item for item in tool.tests
                if str(item.get("evidence_ref") or "").startswith("evidence://")
                and not evidence_pointer_valid(item, kind="tool_test")]
            if invalid_pointers:
                issues.append({
                    "path": str(path), "code": "invalid_evidence_pointer",
                    "detail": f"count={len(invalid_pointers)}"})
            findings = validate_long_term_asset(
                payload, asset_kind=f"tool:{tool.artifact_kind.value}")
            if findings:
                issues.append({"path": str(path),
                               "code": "long_term_asset_guard_failed",
                               "detail": ";".join(findings[:12])})
            if (tool.status in USABLE_TOOL_STATUSES
                    and not tool.admission_certificate_valid()):
                issues.append({
                    "path": str(path),
                    "code": "usable_missing_valid_admission_certificate",
                    "detail": str(tool.ref),
                })
        return {"passed": not issues, "scanned": scanned, "issues": issues}

    def assert_frozen_ready(self) -> dict[str, Any]:
        """Raise before snapshotting a Tool bank that depends on private data."""
        audit = self.audit_frozen_readiness()
        if not audit["passed"]:
            summary = ";".join(
                f"{item['code']}:{item['path']}" for item in audit["issues"][:12])
            raise RuntimeError(f"tool_frozen_readiness_failed:{summary}")
        return audit

    def migrate_legacy_assets(
            self, admission_factory: Callable[..., Any] | None = None, *,
            replay_callback: Callable[..., dict[str, Any]] | None = None,
            demote_failed: bool = True) -> dict[str, Any]:
        """Explicitly externalize legacy tests and optionally re-admit Tools.

        This method never fabricates an Admission certificate.  A usable
        legacy Tool is certified only when a caller-supplied Admission engine
        or replay callback performs the real checks successfully.  Calling it
        without either option merely removes inline evidence; readiness still
        fails until re-admission occurs.
        """
        from .admission_adapter import AdmissionEngine

        report: dict[str, Any] = {
            "scanned": 0, "externalized": 0, "readmitted": 0,
            "demoted": 0, "errors": [],
        }
        index = self._read_index()
        for path in sorted(self.root.glob("*/*/tool.json")):
            report["scanned"] += 1
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                tool = ToolAsset.from_dict(payload)
            except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
                report["errors"].append(
                    f"unreadable:{path}:{type(exc).__name__}:{exc}")
                continue
            # Pointer-only assets need their local EvidenceStore hydrated
            # before an explicit re-admission can execute the original cases.
            hydrated = self.get(tool.ref)
            if hydrated is not None:
                tool = hydrated
            raw_tests = any(
                not str(item.get("evidence_ref") or "").startswith(
                    "evidence://") for item in (payload.get("tests") or []))
            needs_admission = bool(
                tool.status in USABLE_TOOL_STATUSES
                and not tool.admission_certificate_valid())
            if needs_admission and (admission_factory is not None
                                    or replay_callback is not None):
                original_status = tool.status
                try:
                    if admission_factory is not None:
                        parameters = inspect.signature(
                            admission_factory).parameters.values()
                        accepts_tool = any(
                            parameter.kind in {
                                inspect.Parameter.POSITIONAL_ONLY,
                                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                inspect.Parameter.VAR_POSITIONAL,
                            } for parameter in parameters)
                        engine = (admission_factory(tool) if accepts_tool
                                  else admission_factory())
                    else:
                        engine = AdmissionEngine(replay_fn=replay_callback)
                    result = engine.admit(tool)
                except Exception as exc:  # noqa: BLE001 - migration audit
                    result = None
                    report["errors"].append(
                        f"admission_error:{tool.ref}:{type(exc).__name__}:{exc}")
                if result is not None and bool(result.passed):
                    tool.status = original_status
                    report["readmitted"] += 1
                elif demote_failed:
                    tool.status = ToolLifecycle.SHADOW
                    report["demoted"] += 1
            try:
                self._save_tool(tool)
                if raw_tests:
                    report["externalized"] += 1
                entry = index.get(tool.tool_id)
                if (entry is not None
                        and entry.get("current_version") == tool.ref.version):
                    entry["latest_status"] = tool.status.value
            except Exception as exc:  # noqa: BLE001 - per-asset report
                report["errors"].append(
                    f"persist_error:{tool.ref}:{type(exc).__name__}:{exc}")
        self._write_index(index)
        report["frozen_readiness"] = self.audit_frozen_readiness()
        report["passed"] = bool(
            not report["errors"] and report["frozen_readiness"]["passed"])
        return report

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


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
