"""Persistent registry for StrategyCard objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.strategy import StrategyCard


class StrategyRegistry:
    """Persist, query, and update strategy cards from a single JSON file."""

    def __init__(self, file_path: str = "data/strategies/registry.json") -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._cards: dict[str, StrategyCard] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self.file_path.exists():
            self.file_path.write_text("[]", encoding="utf-8")
            self._cards = {}
            return
        raw = self.file_path.read_text(encoding="utf-8").strip()
        if not raw:
            self._cards = {}
            return
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self._cards = {}
            return
        if not isinstance(data, list):
            self._cards = {}
            return
        self._cards = {}
        for item in data:
            if isinstance(item, dict):
                card = StrategyCard.from_dict(item)
                if card.strategy_id:
                    self._cards[card.strategy_id] = card

    def _save(self) -> None:
        payload = [card.to_dict() for card in self._cards.values()]
        temp_path = self.file_path.with_suffix(self.file_path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(self.file_path)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, card: StrategyCard) -> StrategyCard:
        """Add or update a strategy card.  Deduplicates by source_skill_id."""
        # If a strategy for the same source skill already exists, update it.
        for existing in self._cards.values():
            if existing.source_skill_id == card.source_skill_id:
                existing.description = card.description
                existing.applicability = card.applicability
                existing.approach_summary = card.approach_summary
                existing.task_pattern = card.task_pattern
                existing.source_benchmark = card.source_benchmark
                existing.source_task_id = card.source_task_id
                self._save()
                return existing
        self._cards[card.strategy_id] = card
        self._save()
        return card

    def get(self, strategy_id: str) -> StrategyCard | None:
        return self._cards.get(strategy_id)

    def list_all(self) -> list[StrategyCard]:
        return list(self._cards.values())

    def list_active(self) -> list[StrategyCard]:
        return [c for c in self._cards.values() if c.status == "active"]

    def match_by_pattern(self, task_pattern: str, top_k: int = 3) -> list[StrategyCard]:
        """Return active strategies matching a task pattern, ranked by utility."""
        matches = [
            c for c in self._cards.values()
            if c.status == "active" and c.task_pattern == task_pattern
        ]
        matches.sort(key=lambda c: c.utility, reverse=True)
        return matches[:top_k]

    # ------------------------------------------------------------------
    # ContrastiveEval tracking
    # ------------------------------------------------------------------

    def record_guided_usage(self, strategy_id: str, success: bool) -> None:
        card = self._cards.get(strategy_id)
        if card is None:
            return
        card.guided_use_count += 1
        if success:
            card.guided_success_count += 1
        self._check_contrastive_suppress(card)
        self._save()

    def record_unguided_usage(self, task_pattern: str, success: bool) -> None:
        """Update unguided stats for all active strategies matching the pattern."""
        changed = False
        for card in self._cards.values():
            if card.task_pattern == task_pattern and card.status == "active":
                card.unguided_use_count += 1
                if success:
                    card.unguided_success_count += 1
                changed = True
        if changed:
            self._save()

    def _check_contrastive_suppress(self, card: StrategyCard) -> None:
        if card.guided_use_count >= 5 and card.contrastive_delta < -0.1:
            card.status = "suppressed"
        elif card.guided_use_count >= 3 and card.guided_success_rate < 0.3:
            card.status = "suppressed"

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {"active": 0, "suppressed": 0, "retired": 0}
        for card in self._cards.values():
            if card.status in counts:
                counts[card.status] += 1
        return counts

    def snapshot(self) -> dict[str, Any]:
        return {
            "total": len(self._cards),
            "status_counts": self.status_counts(),
            "cards": [card.to_dict() for card in self._cards.values()],
        }
