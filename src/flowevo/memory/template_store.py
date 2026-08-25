"""Persistent store for runtime template priors."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from core.schemas import Template


class TemplateStore:
    """Persist, match, and update lightweight runtime templates."""

    def __init__(self, file_path: str = "data/tasks/templates.json") -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.file_path.write_text("[]", encoding="utf-8")

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _sort_key(self, template: Template) -> tuple[int, int, str]:
        return (-int(template.priority), -int(template.success_count), str(template.template_id))

    def _priority(self, template: Template) -> int:
        return max(0, int(template.support_count) + int(template.success_count) - int(template.failure_count))

    def _normalize(self, template: Template) -> Template:
        if not template.benchmark:
            template.benchmark = template.family
        if not template.family:
            template.family = template.benchmark
        if not template.preferred_skill_sequence and template.preferred_skill_ids:
            template.preferred_skill_sequence = list(template.preferred_skill_ids)
        if not template.preferred_skill_ids and template.preferred_skill_sequence:
            template.preferred_skill_ids = list(template.preferred_skill_sequence)
        template.priority = self._priority(template)
        return template

    def _write(self, templates: list[Template]) -> None:
        payload = [self._normalize(template).model_dump(mode="json") for template in sorted(templates, key=self._sort_key)]
        self.file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_all(self) -> list[Template]:
        data = json.loads(self.file_path.read_text(encoding="utf-8"))
        return [self._normalize(Template.model_validate(item)) for item in data]

    def get(self, template_id: str) -> Template | None:
        for template in self.list_all():
            if template.template_id == template_id:
                return template
        return None

    def upsert(self, template: Template) -> None:
        templates = self.list_all()
        table = {item.template_id: item for item in templates}
        normalized = self._normalize(template)
        if not normalized.updated_at:
            normalized.updated_at = self._now()
        table[normalized.template_id] = normalized
        self._write(list(table.values()))

    def status_counts(self) -> dict[str, int]:
        counts = {"draft": 0, "active": 0, "suppressed": 0}
        for template in self.list_all():
            if template.status in counts:
                counts[template.status] += 1
        return counts

    def match(
        self,
        benchmark: str,
        task_pattern: str,
        limit: int = 2,
        template_kinds: list[str] | None = None,
    ) -> list[Template]:
        templates = [
            template
            for template in self.list_all()
            if template.status == "active"
            and template.benchmark == benchmark
            and template.task_pattern == task_pattern
            and (not template_kinds or template.template_kind in template_kinds)
        ]
        templates.sort(key=self._sort_key)
        return templates[:limit]

    def observe_candidates(self, candidates: list[Template]) -> list[Template]:
        if not candidates:
            return []
        table = {template.template_id: template for template in self.list_all()}
        now = self._now()
        updated: list[Template] = []

        for candidate in candidates:
            normalized = self._normalize(candidate.model_copy(deep=True))
            current = table.get(normalized.template_id)
            if current is None:
                current = normalized
                current.support_count = max(1, int(normalized.support_count or 1))
                current.status = "active" if current.support_count >= 2 else "draft"
                current.updated_at = now
            else:
                current.description = normalized.description or current.description
                current.template_kind = normalized.template_kind or current.template_kind
                current.benchmark = normalized.benchmark or current.benchmark
                current.family = normalized.family or current.family
                current.task_pattern = normalized.task_pattern or current.task_pattern
                current.trigger_signature = normalized.trigger_signature or current.trigger_signature
                current.support_count += max(1, int(normalized.support_count or 1))
                for directive in normalized.prompt_directives:
                    if directive not in current.prompt_directives:
                        current.prompt_directives.append(directive)
                for skill_id in normalized.preferred_skill_ids:
                    if skill_id not in current.preferred_skill_ids:
                        current.preferred_skill_ids.append(skill_id)
                current.preferred_skill_sequence = list(current.preferred_skill_ids)
                for trace_id in normalized.exemplar_trace_ids:
                    if trace_id not in current.exemplar_trace_ids:
                        current.exemplar_trace_ids.append(trace_id)
                current.exemplar_trace_ids = current.exemplar_trace_ids[:5]
                if current.status != "suppressed" and current.support_count >= 2:
                    current.status = "active"
                current.updated_at = now
            current.priority = self._priority(current)
            table[current.template_id] = current
            updated.append(current.model_copy(deep=True))

        self._write(list(table.values()))
        updated.sort(key=self._sort_key)
        return updated

    def record_match_feedback(self, template_ids: list[str], success: bool) -> list[Template]:
        if not template_ids:
            return []
        table = {template.template_id: template for template in self.list_all()}
        now = self._now()
        updated: list[Template] = []

        for template_id in template_ids:
            current = table.get(template_id)
            if current is None:
                continue
            if success:
                current.success_count += 1
                if current.status != "suppressed" and current.support_count >= 2:
                    current.status = "active"
            else:
                current.failure_count += 1
                if current.failure_count >= 2 and current.failure_count >= current.success_count:
                    current.status = "suppressed"
            current.updated_at = now
            current.priority = self._priority(current)
            table[current.template_id] = current
            updated.append(current.model_copy(deep=True))

        self._write(list(table.values()))
        updated.sort(key=self._sort_key)
        return updated
