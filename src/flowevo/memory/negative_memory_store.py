"""Persistent negative memory for toxic or failing artifacts."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from math import exp
from pathlib import Path

from core.schemas import NegativeMemoryEntry


class NegativeMemoryStore:
    """Persist failure evidence across runtime, audit, and prune events."""

    def __init__(self, file_path: str = "data/governance/negative_memory.json") -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.file_path.write_text("[]", encoding="utf-8")

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def list_all(self) -> list[NegativeMemoryEntry]:
        data = json.loads(self.file_path.read_text(encoding="utf-8"))
        return [NegativeMemoryEntry.model_validate(item) for item in data]

    def _write(self, entries: list[NegativeMemoryEntry]) -> None:
        payload = [entry.model_dump(mode="json") for entry in entries]
        self.file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(
        self,
        *,
        artifact_id: str,
        artifact_kind: str,
        benchmark: str,
        lifecycle_phase: str = "",
        task_pattern: str = "",
        planning_mode: str = "",
        usage_subtype: str = "",
        failure_type: str = "",
        source: str,
        summary: str,
        trace_id: str = "",
        failed_family: str = "",
        failed_api_pattern: str = "",
        misroute_source_skill: str = "",
        failure_after_seeded_use: bool = False,
        severity: float = 0.5,
    ) -> NegativeMemoryEntry:
        entry = NegativeMemoryEntry(
            entry_id=f"neg_{uuid.uuid4().hex[:10]}",
            artifact_id=artifact_id,
            artifact_kind=artifact_kind,
            benchmark=benchmark,
            lifecycle_phase=lifecycle_phase,
            task_pattern=task_pattern,
            planning_mode=planning_mode,
            usage_subtype=usage_subtype,
            failure_type=failure_type,
            source=source,
            summary=summary[:500],
            trace_id=trace_id,
            failed_family=failed_family,
            failed_api_pattern=failed_api_pattern[:300],
            misroute_source_skill=misroute_source_skill,
            failure_after_seeded_use=bool(failure_after_seeded_use),
            severity=round(float(severity), 4),
            timestamp=self._now(),
        )
        entries = self.list_all()
        entries.append(entry)
        self._write(entries)
        return entry

    def count_for_artifact(
        self,
        artifact_id: str,
        *,
        source: str | None = None,
        lifecycle_phase: str | None = None,
    ) -> int:
        return sum(
            1
            for entry in self.list_all()
            if entry.artifact_id == artifact_id
            and (source is None or entry.source == source)
            and (lifecycle_phase is None or entry.lifecycle_phase == lifecycle_phase)
        )

    def toxicity_score_for_artifact(
        self,
        artifact_id: str,
        *,
        source: str | None = None,
        sources: Iterable[str] | None = None,
        usage_subtype: str | None = None,
        lifecycle_phase: str | None = None,
        half_life_hours: float = 72.0,
    ) -> float:
        now = datetime.now(timezone.utc)
        score = 0.0
        source_filter = {str(item) for item in sources} if sources is not None else None
        for entry in self.list_all():
            if entry.artifact_id != artifact_id:
                continue
            if source is not None and entry.source != source:
                continue
            if source_filter is not None and entry.source not in source_filter:
                continue
            if usage_subtype is not None and entry.usage_subtype != usage_subtype:
                continue
            if lifecycle_phase is not None and entry.lifecycle_phase != lifecycle_phase:
                continue
            try:
                created_at = datetime.fromisoformat(entry.timestamp)
            except ValueError:
                created_at = now
            age_hours = max(0.0, (now - created_at).total_seconds() / 3600.0)
            score += float(entry.severity) * exp(-age_hours / max(half_life_hours, 1e-6))
        return round(score, 4)

    def clear_for_artifact(self, artifact_id: str, *, source: str | None = None, usage_subtype: str | None = None) -> int:
        entries = self.list_all()
        retained = [
            entry
            for entry in entries
            if not (
                entry.artifact_id == artifact_id
                and (source is None or entry.source == source)
                and (usage_subtype is None or entry.usage_subtype == usage_subtype)
            )
        ]
        removed = len(entries) - len(retained)
        if removed:
            self._write(retained)
        return removed
