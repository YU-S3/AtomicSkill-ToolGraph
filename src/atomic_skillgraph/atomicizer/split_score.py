"""SplitScore：原子性判定（设计文档 v2.0 §6.4）。

SplitScore = 0.20*R_reuse + 0.20*R_validation + 0.15*R_failure + 0.15*R_io
           + 0.10*R_retry + 0.10*R_executor + 0.10*R_effect

默认（阈值必须配置化）：
    >= 0.70            强制形成 split candidate
    0.45 – 0.70        通过 replay/复用证据决定
    < 0.45             保持封装
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.config import Thresholds

_WEIGHTS = {
    "reuse": 0.20,
    "validation": 0.20,
    "failure": 0.15,
    "io": 0.15,
    "retry": 0.10,
    "executor": 0.10,
    "effect": 0.10,
}


@dataclass
class SplitScoreResult:
    score: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    verdict: str = "keep"          # split | evidence | keep
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "verdict": self.verdict,
            "reasons": self.reasons,
        }


def compute_split_score(segment: dict[str, Any],
                        evidence: dict[str, Any] | None = None,
                        thresholds: Thresholds | None = None) -> SplitScoreResult:
    """对原子化片段评估是否应继续拆分。

    evidence 提供历史统计：
      - reuse_evidence: {sub_name: reuse_count}    子过程被独立复用的证据
      - validation_evidence: 子过程可独立验证的证据（validator 覆盖）
      - failure_clusters: {family: count}          独立失败簇
      - io_evidence: 子过程独立输入输出
      - retry_evidence: 子过程独立重试需求
      - executor_evidence: 子过程存在替代实现
      - effect_evidence: 片段包含多个独立核心 Effect
    """
    thresholds = thresholds or Thresholds()
    evidence = evidence or {}
    effects = segment.get("effect") or []
    actions = segment.get("actions") or []

    effect_count = len(effects)
    has_multiple_effects = effect_count > 1

    components: dict[str, float] = {}
    reasons: list[str] = []

    # R_reuse：跨实例复用证据
    reuse = _norm(sum(evidence.get("reuse_evidence", {}).values()), 3)
    components["reuse"] = reuse
    if reuse > 0.5:
        reasons.append(f"子过程被独立复用 {int(sum(evidence.get('reuse_evidence', {}).values()))} 次")

    # R_validation：可独立验证
    validation = _norm(evidence.get("validation_evidence", 0), 2)
    components["validation"] = validation
    if validation >= 0.5:
        reasons.append("子过程拥有独立验证器")

    # R_failure：独立失败簇
    failures = evidence.get("failure_clusters", {})
    failure_score = _norm(len([c for c in failures.values() if c >= 2]), 2)
    components["failure"] = failure_score
    if failure_score >= 0.5:
        reasons.append(f"存在 {len(failures)} 个独立失败簇")

    # R_io：子过程独立 I/O
    io_score = _norm(evidence.get("io_evidence", 0), 2)
    components["io"] = io_score
    if io_score >= 0.5:
        reasons.append("子过程具有独立稳定 I/O")

    # R_retry：独立 retry/fallback
    retry = _norm(evidence.get("retry_evidence", 0), 1)
    components["retry"] = retry

    # R_executor：替代实现
    executor = _norm(evidence.get("executor_evidence", 0), 1)
    components["executor"] = executor

    # R_effect：多个核心 Effect + 动作跨度
    effect_score = 1.0 if has_multiple_effects else 0.0
    if not has_multiple_effects and len(actions) > 8:
        effect_score = 0.5  # 长跨度但单一效果 → 中等
    components["effect"] = effect_score
    if has_multiple_effects:
        reasons.append(f"片段包含 {effect_count} 个独立核心 Effect")

    score = sum(_WEIGHTS[k] * components.get(k, 0.0) for k in _WEIGHTS)

    if score >= thresholds.split_score_force:
        verdict = "split"
    elif score >= thresholds.split_score_consider:
        verdict = "evidence"
    else:
        verdict = "keep"
    return SplitScoreResult(score=round(score, 4), components=components,
                            verdict=verdict, reasons=reasons)


def _norm(value: Any, saturation: int) -> float:
    return max(0.0, min(1.0, float(value) / max(saturation, 1)))
