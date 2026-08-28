"""Tool Asset IR 与生命周期（设计文档 v2.0 §18、§27-§30、附录 A）。

Tool = <Σ, X, T_s, S_f, P_v, Q, L>：
- Σ signature/parameters；X executable artifact；T_s tests/replay cases；
- S_f safety；P_v provenance；Q statistics；L lifecycle/lineage。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .refs import ToolRef, artifact_hash, bump_version, content_hash
from .status import ArtifactKind, ToolLifecycle, tool_transition_allowed


class ToolIRError(ValueError):
    """Tool IR 校验错误。"""


# 生命周期可进入 Runtime 候选的状态（candidate 可用但优先级低于 active/preferred）
USABLE = {ToolLifecycle.CANDIDATE, ToolLifecycle.ACTIVE, ToolLifecycle.PREFERRED}

# 生命周期偏序排名（用于同功能选择）
_LIFECYCLE_RANK = {
    ToolLifecycle.PREFERRED: 4,
    ToolLifecycle.ACTIVE: 3,
    ToolLifecycle.CANDIDATE: 2,
    ToolLifecycle.SHADOW: 0,
    ToolLifecycle.DRAFT: -1,
    ToolLifecycle.ADMISSION_PENDING: -1,
    ToolLifecycle.SUPPRESSED: -2,
    ToolLifecycle.RETIRED: -3,
}


@dataclass
class ToolAsset:
    ref: ToolRef
    artifact_kind: ArtifactKind = ArtifactKind.PYTHON_CALLABLE
    summary: str = ""
    signature: dict[str, Any] = field(default_factory=dict)  # parameters: [...]
    interface: dict[str, Any] = field(default_factory=dict)   # inputs/outputs
    artifact: dict[str, Any] = field(default_factory=dict)    # {"code": ...} / {"template": ...} / content_ref
    tests: list[dict[str, Any]] = field(default_factory=list)
    safety: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)
    lineage: dict[str, Any] = field(default_factory=dict)
    status: ToolLifecycle = ToolLifecycle.DRAFT
    # Evidence payloads are hydrated by ToolRegistry only when the local
    # EvidenceStore is available.  They are deliberately excluded from
    # ``to_dict`` so a portable Tool bank cannot re-embed private replay data.
    _resolved_tests: list[dict[str, Any]] | None = field(
        default=None, repr=False, compare=False)
    _unresolved_test_refs: list[str] = field(
        default_factory=list, repr=False, compare=False)

    # -- 属性 ----------------------------------------------------------------
    @property
    def tool_id(self) -> str:
        return self.ref.tool_id

    def parameters(self) -> list[dict[str, Any]]:
        return list(self.signature.get("parameters") or [])

    def param_names(self) -> list[str]:
        return [str(p.get("name")) for p in self.parameters() if p.get("name")]

    def artifact_body(self) -> str:
        """可执行 artifact 文本（代码或模板）。"""
        artifact = self.artifact
        if "code" in artifact:
            return str(artifact["code"])
        if "template" in artifact:
            return str(artifact["template"])
        if "steps" in artifact:
            return "\n".join(str(s) for s in artifact["steps"])
        return ""

    def artifact_hash(self) -> str:
        return artifact_hash(self.artifact_body())

    def entry_point(self) -> str:
        return str(self.signature.get("entry_point") or self.artifact.get("entry_point") or "")

    def replay_cases(self) -> list[dict[str, Any]]:
        source = self.all_test_cases()
        cases = [t for t in source
                 if t.get("kind") in (None, "replay", "unit", "perturbation")]
        return cases or source

    def all_test_cases(self) -> list[dict[str, Any]]:
        """Return every locally available hydrated test, including parameterized cases."""
        return (list(self._resolved_tests)
                if self._resolved_tests is not None
                else [dict(item) for item in self.tests
                      if not _is_evidence_pointer(item)])

    def set_resolved_tests(self, tests: list[dict[str, Any]], *,
                           unresolved_refs: list[str] | None = None) -> None:
        self._resolved_tests = [dict(item) for item in tests]
        self._unresolved_test_refs = list(unresolved_refs or [])

    def has_unresolved_test_evidence(self) -> bool:
        return bool(self._unresolved_test_refs)

    def test_evidence_identities(self) -> list[dict[str, str]]:
        """Canonical evidence ref/hash pairs bound into Admission.

        Raw pre-registration tests deterministically map to the same
        ``evidence://tool_test/<sha256>`` URI that ToolRegistry later writes.
        Sorting is order-independent, while intentionally preserving duplicate
        occurrences so removal of one certified case is detectable.
        """
        identities: list[dict[str, str]] = []
        for item in self.tests:
            if _is_evidence_pointer(item):
                ref = str(item.get("evidence_ref") or "")
                digest = str(item.get("evidence_hash") or "")
            else:
                digest = _full_content_hash(item)
                ref = f"evidence://tool_test/{digest}"
            identities.append({"evidence_ref": ref,
                               "evidence_hash": digest})
        return sorted(identities, key=lambda value: (
            value["evidence_ref"], value["evidence_hash"]))

    def test_evidence_hashes(self) -> list[str]:
        """Compatibility view of certified hashes, preserving multiplicity."""
        return [item["evidence_hash"]
                for item in self.test_evidence_identities()]

    def issue_admission_certificate(
            self, *, checks: dict[str, bool] | None = None) -> dict[str, Any]:
        """Bind a passed Admission decision to executable and test hashes."""
        certificate = {
            "schema_version": 2,
            "passed": True,
            "artifact_hash": self.artifact_hash(),
            "contract_hash": self.admission_contract_hash(),
            "test_evidence": self.test_evidence_identities(),
            "test_evidence_hashes": self.test_evidence_hashes(),
            "checks": {str(key): bool(value)
                       for key, value in sorted((checks or {}).items())},
        }
        self.safety = dict(self.safety or {})
        self.safety["admission_certificate"] = certificate
        return certificate

    def admission_contract_hash(self) -> str:
        """Hash every executable/interface field covered by Admission."""
        return _full_content_hash({
            "artifact_kind": self.artifact_kind.value,
            "signature": self.signature,
            "interface": self.interface,
            "artifact": self.artifact,
            "safety_policy": {
                str(key): value for key, value in (self.safety or {}).items()
                if str(key) != "admission_certificate"
            },
        })

    def admission_certificate_valid(self) -> bool:
        certificate = dict(
            (self.safety or {}).get("admission_certificate") or {})
        checks = dict(certificate.get("checks") or {})
        if (not certificate
                or int(certificate.get("schema_version") or 0) != 2
                or not bool(certificate.get("passed"))
                or not bool(checks.get("admitted"))
                or not all(bool(value) for value in checks.values())):
            return False
        return bool(
            str(certificate.get("artifact_hash") or "")
            == self.artifact_hash()
            and str(certificate.get("contract_hash") or "")
            == self.admission_contract_hash()
            and list(certificate.get("test_evidence") or [])
            == self.test_evidence_identities()
            and list(certificate.get("test_evidence_hashes") or [])
            == self.test_evidence_hashes()
        )

    def is_usable(self) -> bool:
        return self.status in USABLE and self.admission_certificate_valid()

    def utility(self) -> float:
        return float(self.statistics.get("utility", 0.0))

    # -- 状态迁移 ------------------------------------------------------------
    def transition(self, to: ToolLifecycle) -> "ToolAsset":
        if not tool_transition_allowed(self.status, to):
            raise ToolIRError(f"非法 Tool 生命周期迁移：{self.status.value} -> {to.value}")
        self.status = to
        return self

    # -- 统计更新 ------------------------------------------------------------
    def record_usage(self, success: bool, *, usage_mode: str = "direct", alpha: float = 0.5) -> None:
        stats = self.statistics
        stats["call_count"] = int(stats.get("call_count", 0)) + 1
        if success:
            stats["success_count"] = int(stats.get("success_count", 0)) + 1
            stats["consecutive_failures"] = 0
            if usage_mode == "direct":
                stats["direct_use_count"] = int(stats.get("direct_use_count", 0)) + 1
                stats["direct_success_count"] = int(stats.get("direct_success_count", 0)) + 1
        else:
            stats["failure_count"] = int(stats.get("failure_count", 0)) + 1
            stats["consecutive_failures"] = int(stats.get("consecutive_failures", 0)) + 1
            if usage_mode == "direct":
                stats["direct_use_count"] = int(stats.get("direct_use_count", 0)) + 1
                stats["direct_failure_count"] = int(stats.get("direct_failure_count", 0)) + 1
        success_total = int(stats.get("success_count", 0))
        total = int(stats.get("call_count", 0))
        empirical = success_total / max(total, 1)
        old = float(stats.get("utility", 0.0))
        stats["utility"] = round((1 - alpha) * old + alpha * empirical, 4)
        stats["success_rate"] = round(empirical, 4)

    # -- 序列化 ----------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.ref.tool_id,
            "version": self.ref.version,
            "status": self.status.value,
            "artifact_kind": self.artifact_kind.value,
            "summary": self.summary,
            "signature": self.signature,
            "interface": self.interface,
            "artifact": self.artifact,
            "tests": self.tests,
            "safety": self.safety,
            "provenance": self.provenance,
            "statistics": self.statistics,
            "lineage": self.lineage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolAsset":
        return cls(
            ref=ToolRef(
                tool_id=str(data["tool_id"]),
                version=str(data.get("version") or "1.0.0"),
            ),
            artifact_kind=ArtifactKind(str(data.get("artifact_kind") or "python_callable")),
            summary=str(data.get("summary", "")),
            signature=dict(data.get("signature") or {}),
            interface=dict(data.get("interface") or {}),
            artifact=dict(data.get("artifact") or {}),
            tests=list(data.get("tests") or []),
            safety=dict(data.get("safety") or {}),
            provenance=dict(data.get("provenance") or {}),
            statistics=dict(data.get("statistics") or {}),
            lineage=dict(data.get("lineage") or {}),
            status=ToolLifecycle(str(data.get("status") or "draft")),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.summary.strip():
            errors.append("summary 不能为空")
        if "parameters" not in self.signature:
            errors.append("signature.parameters 至少需要一个参数（或无参函数需显式声明空参数表）")
        if not self.artifact_body():
            errors.append("artifact 缺少可执行内容（code/template/steps）")
        return errors

    def structural_hash(self) -> str:
        """artifact + signature 的结构哈希（去重/合并判据之一）。"""
        return content_hash(
            {
                "artifact_kind": self.artifact_kind.value,
                "signature": self.signature,
                "artifact": self.artifact_body(),
            }
        )

    def next_version(self, change: str = "patch") -> str:
        return bump_version(self.ref.version, change)


def lifecycle_rank(status: ToolLifecycle) -> int:
    return _LIFECYCLE_RANK.get(status, -4)


def _is_evidence_pointer(item: Any) -> bool:
    return bool(isinstance(item, dict)
                and str(item.get("evidence_ref") or "").startswith(
                    "evidence://"))


def _full_content_hash(payload: Any) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
