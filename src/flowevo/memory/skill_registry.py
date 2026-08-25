"""Minimal local registry for SkillCard objects and compiled skill files."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from agent.skill_compatibility import infer_skill_compatibility_metadata
from core.schemas import SkillCard, SkillFailureEvent


class SkillRegistry:
    """Persist and query skills from a single JSON file."""

    def __init__(
        self,
        file_path: str = "data/skills/registry.json",
        skills_root: str | None = None,
        suppression_utility_threshold: float = 0.35,
        suppression_failure_threshold: int = 2,
        prune_audit_fail_threshold: int = 2,
    ) -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.skills_root = Path(skills_root) if skills_root is not None else self.file_path.parent
        self.skills_root.mkdir(parents=True, exist_ok=True)
        self.suppression_utility_threshold = suppression_utility_threshold
        self.suppression_failure_threshold = suppression_failure_threshold
        self.prune_audit_fail_threshold = prune_audit_fail_threshold
        if not self.file_path.exists():
            self.file_path.write_text("[]", encoding="utf-8")

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _running_mean(self, previous: float, count: int, value: float) -> float:
        if count <= 0:
            return round(float(value), 4)
        return round(((float(previous) * float(count - 1)) + float(value)) / float(count), 4)

    def _card_benchmark(self, card: SkillCard) -> str:
        return str(card.preconditions.get("benchmark", card.family) or "")

    def _cluster_key(self, card: SkillCard) -> str:
        compile_metadata = dict(card.compile_metadata or {})
        cluster_key = str(compile_metadata.get("transfer_cluster_key", "") or "").strip()
        if cluster_key:
            return cluster_key
        signature = dict(card.signature or {})
        args = signature.get("args") or []
        arg_count = len(args) if isinstance(args, list) else 0
        task_pattern = str(card.preconditions.get("task_pattern", "") or "")
        helper_shape = str(compile_metadata.get("helper_shape_key", "") or "no_helpers")
        return f"{self._card_benchmark(card)}::{task_pattern}::{arg_count}::{helper_shape}"

    def _distillation_score(self, card: SkillCard) -> float:
        positive = int(card.positive_transfer_count or 0)
        negative = int(card.negative_transfer_count or 0)
        return round(
            float(card.utility)
            + 0.35 * positive
            - 0.45 * negative
            + 0.40 * max(float(card.avg_delta_first_attempt or 0.0), 0.0)
            + 0.20 * max(-float(card.avg_delta_retry_count or 0.0), 0.0)
            + 0.0005 * max(-float(card.avg_delta_tokens_per_solved or 0.0), 0.0)
            + 0.05 * float(card.direct_reuse_evidence_count or 0),
            4,
        )

    def _apply_status_rules(self, card: SkillCard) -> SkillCard:
        if card.status in {"draft", "shadow", "pruned"}:
            return card
        if card.status == "suppressed" and card.audit_fail_count >= self.prune_audit_fail_threshold:
            card.status = "pruned"
            return card
        if card.utility < self.suppression_utility_threshold or card.consecutive_failures >= self.suppression_failure_threshold:
            card.status = "suppressed"
        else:
            card.status = "active"
        return card

    def _load_payload(self) -> list[dict]:
        for attempt in range(2):
            raw = self.file_path.read_text(encoding="utf-8").strip()
            if not raw:
                return []
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                if attempt == 0:
                    time.sleep(0.05)
                    continue
                raise
            return payload if isinstance(payload, list) else []
        return []

    def _write_payload(self, payload: list[dict]) -> None:
        temp_path = self.file_path.with_suffix(self.file_path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self.file_path)

    def list_all(self) -> list[SkillCard]:
        data = self._load_payload()
        cards = [SkillCard.model_validate(item) for item in data]
        return [self._enrich_card_metadata(card) for card in cards]

    def list_active(self) -> list[SkillCard]:
        return [card for card in self.list_all() if card.status == "active"]

    def list_shadow(self) -> list[SkillCard]:
        return [card for card in self.list_all() if card.status == "shadow"]

    def get(self, skill_id: str) -> SkillCard | None:
        for card in self.list_all():
            if card.skill_id == skill_id:
                return card
        return None

    def upsert(self, card: SkillCard) -> None:
        cards = self.list_all()
        table = {c.skill_id: c for c in cards}
        table[card.skill_id] = self._apply_status_rules(self._enrich_card_metadata(card))
        payload = [c.model_dump(mode="json") for c in table.values()]
        self._write_payload(payload)

    def set_status(self, skill_id: str, status: str) -> SkillCard | None:
        cards = self.list_all()
        table = {c.skill_id: c for c in cards}
        card = table.get(skill_id)
        if card is None:
            return None
        card.status = status
        table[skill_id] = card
        payload = [c.model_dump(mode="json") for c in table.values()]
        self._write_payload(payload)
        return self.get(skill_id)

    def save_code(self, skill_id: str, code: str) -> Path:
        path = self.skills_root / f"{skill_id}.py"
        path.write_text(code, encoding="utf-8")
        return path

    def record_usage(
        self,
        skill_id: str,
        success: bool,
        *,
        usage_mode: str = "",
        routing_mode: str = "",
        first_attempt_success: bool | None = None,
        retry_count: int | None = None,
        token_cost_to_pass: float | None = None,
        transfer_positive: bool | None = None,
        transfer_negative: bool | None = None,
        smoothing_alpha: float = 0.5,
    ) -> SkillCard | None:
        card = self.get(skill_id)
        if card is None:
            return None
        card.call_count += 1
        card.last_used_at = self._now()
        if usage_mode == "direct":
            card.direct_use_count += 1
            if success:
                card.direct_success_count += 1
            else:
                card.direct_failure_count += 1
        elif usage_mode == "seeded":
            card.seeded_use_count += 1
            sample_count = max(card.seeded_use_count, 1)
            if routing_mode:
                card.last_used_routing_mode = str(routing_mode)
            if first_attempt_success is not None:
                card.avg_delta_first_attempt = self._running_mean(
                    card.avg_delta_first_attempt,
                    sample_count,
                    1.0 if bool(first_attempt_success) else 0.0,
                )
            if retry_count is not None:
                # Lower retry count is better, so store the utility-oriented sign.
                card.avg_delta_retry_count = self._running_mean(
                    card.avg_delta_retry_count,
                    sample_count,
                    -float(retry_count),
                )
            if token_cost_to_pass is not None:
                # Lower token cost is better, so store the utility-oriented sign.
                card.avg_delta_tokens_per_solved = self._running_mean(
                    card.avg_delta_tokens_per_solved,
                    sample_count,
                    -float(token_cost_to_pass),
                )
            positive = bool(success) if transfer_positive is None else bool(transfer_positive)
            negative = (not bool(success)) if transfer_negative is None else bool(transfer_negative)
            if positive:
                card.positive_transfer_count += 1
            if negative:
                card.negative_transfer_count += 1
            if success:
                card.seeded_success_count += 1
            else:
                card.seeded_failure_count += 1
        if success:
            card.success_count += 1
            card.consecutive_failures = 0
        else:
            card.consecutive_failures += 1
        empirical_success = card.success_count / max(card.call_count, 1)
        card.utility = round((1 - smoothing_alpha) * float(card.utility) + smoothing_alpha * empirical_success, 4)
        self.upsert(card)
        return self.get(skill_id)

    def utility_distilled_skill_ids(
        self,
        *,
        benchmark: str,
        retain_per_cluster: int = 1,
        second_margin: float = 0.25,
    ) -> set[str]:
        grouped: dict[str, list[SkillCard]] = {}
        for card in self.list_active():
            if self._card_benchmark(card) != benchmark:
                continue
            if not card.skill_id:
                continue
            if int(card.negative_transfer_count or 0) >= 2 and int(card.positive_transfer_count or 0) == 0:
                continue
            grouped.setdefault(self._cluster_key(card), []).append(card)

        kept: set[str] = set()
        for cards in grouped.values():
            ordered = sorted(
                cards,
                key=lambda card: (
                    self._distillation_score(card),
                    float(card.utility),
                    int(card.positive_transfer_count or 0),
                    -int(card.negative_transfer_count or 0),
                    str(card.skill_id),
                ),
                reverse=True,
            )
            if not ordered:
                continue
            keep_limit = max(1, retain_per_cluster)
            kept.add(str(ordered[0].skill_id))
            if keep_limit > 1 and len(ordered) > 1:
                best_score = self._distillation_score(ordered[0])
                for card in ordered[1:keep_limit]:
                    if best_score - self._distillation_score(card) <= second_margin:
                        kept.add(str(card.skill_id))
        return kept

    def append_failure(
        self,
        skill_id: str,
        *,
        task_id: str,
        benchmark: str,
        planning_mode: str,
        verifier_message: str,
        failure_type: str,
    ) -> SkillCard | None:
        card = self.get(skill_id)
        if card is None:
            return None
        card.failure_log.append(
            SkillFailureEvent(
                task_id=task_id,
                benchmark=benchmark,
                planning_mode=planning_mode,
                verifier_message=verifier_message[:500],
                failure_type=failure_type,
                timestamp=self._now(),
            )
        )
        self.upsert(card)
        return self.get(skill_id)

    def _enrich_card_metadata(self, card: SkillCard) -> SkillCard:
        metadata = infer_skill_compatibility_metadata(
            {
                "skill_id": card.skill_id,
                "name": card.name,
                "description": card.description,
                "family": card.family,
                "signature": dict(card.signature or {}),
                "preconditions": dict(card.preconditions or {}),
                "direct_eligible": bool(card.direct_eligible),
                "tests": list(card.tests or []),
                "compile_metadata": dict(card.compile_metadata or {}),
                "negative_transfer_risk": float(card.negative_transfer_count or 0),
            }
        )
        card.role = str(metadata.get("role") or card.role or "solver")
        card.task_family = str(metadata.get("task_family") or card.task_family or "")
        card.environment_affordance = [str(item) for item in (metadata.get("environment_affordance") or card.environment_affordance or [])]
        card.insertion_mode = str(metadata.get("insertion_mode") or card.insertion_mode or "soft_hint")
        card.negative_transfer_risk_label = str(
            metadata.get("negative_transfer_risk_label") or card.negative_transfer_risk_label or "low"
        )
        card.skill_family = str(metadata.get("skill_family") or card.skill_family or "")
        return card

    def update_audit_result(self, skill_id: str, passed: bool, utility_delta: float = 0.1) -> SkillCard | None:
        card = self.get(skill_id)
        if card is None:
            return None
        card.last_audit_at = self._now()
        if passed:
            card.audit_pass_count += 1
            card.consecutive_failures = 0
            card.utility = round(min(1.0, float(card.utility) + utility_delta), 4)
        else:
            card.audit_fail_count += 1
            card.consecutive_failures += 1
            card.utility = round(max(0.0, float(card.utility) - utility_delta), 4)
        self.upsert(card)
        return self.get(skill_id)

    def top_for_audit(self, top_k: int = 3) -> list[SkillCard]:
        cards = [card for card in self.list_all() if card.status in {"active", "suppressed"}]
        cards.sort(key=lambda card: (card.call_count, card.utility, -card.audit_fail_count), reverse=True)
        return cards[:top_k]
