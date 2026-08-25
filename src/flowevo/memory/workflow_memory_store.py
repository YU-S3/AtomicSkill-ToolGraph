"""Persistent store for textual workflow memory baselines."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from core.schemas import WorkflowMemoryCard


class WorkflowMemoryStore:
    """Persist and retrieve workflow-summary memories."""

    def __init__(self, file_path: str = "data/tasks/workflow_memory.json") -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.file_path.write_text("[]\n", encoding="utf-8")

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _sort_key(self, card: WorkflowMemoryCard) -> tuple[int, str, str]:
        return (-int(card.support_count), str(card.updated_at), str(card.memory_id))

    def list_all(self) -> list[WorkflowMemoryCard]:
        data = json.loads(self.file_path.read_text(encoding="utf-8"))
        cards = [WorkflowMemoryCard.model_validate(item) for item in data]
        cards.sort(key=self._sort_key)
        return cards

    def _write(self, cards: list[WorkflowMemoryCard]) -> None:
        payload = [card.model_dump(mode="json") for card in sorted(cards, key=self._sort_key)]
        self.file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def upsert(self, card: WorkflowMemoryCard) -> WorkflowMemoryCard:
        table = {item.memory_id: item for item in self.list_all()}
        current = table.get(card.memory_id)
        if current is None:
            current = card.model_copy(deep=True)
            current.support_count = max(1, int(current.support_count or 1))
        else:
            current.workflow_summary = card.workflow_summary or current.workflow_summary
            current.repair_summary = card.repair_summary or current.repair_summary
            current.applicability_notes = card.applicability_notes or current.applicability_notes
            current.support_count += max(1, int(card.support_count or 1))
        current.updated_at = self._now()
        table[current.memory_id] = current
        self._write(list(table.values()))
        return current

    def match(self, benchmark: str, task_pattern: str, limit: int = 2) -> list[WorkflowMemoryCard]:
        cards = [
            card
            for card in self.list_all()
            if card.benchmark == benchmark and card.task_pattern == task_pattern
        ]
        cards.sort(key=self._sort_key)
        return cards[:limit]
