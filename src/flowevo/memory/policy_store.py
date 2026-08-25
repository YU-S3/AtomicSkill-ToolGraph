"""Persistent non-parametric routing and prompt policy state."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from core.schemas import ExecutionTrace, PolicyState, PolicyValue, SkillPolicyValue


class PolicyStore:
    """Persist benchmark/task-pattern/skill-level routing policy."""

    def __init__(self, file_path: str = "data/policy/state.json") -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.save(PolicyState())

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def load(self) -> PolicyState:
        data = json.loads(self.file_path.read_text(encoding="utf-8"))
        return PolicyState.model_validate(data)

    def save(self, state: PolicyState) -> PolicyState:
        self.file_path.write_text(state.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")
        return state

    def _ensure_policy(self, table: dict[str, PolicyValue], key: str) -> PolicyValue:
        current = table.get(key)
        if current is None:
            current = PolicyValue()
            table[key] = current
        return current

    def _ensure_skill_policy(self, table: dict[str, SkillPolicyValue], key: str) -> SkillPolicyValue:
        current = table.get(key)
        if current is None:
            current = SkillPolicyValue()
            table[key] = current
        return current

    def _clamp(self, value: float, bound: float = 1.0) -> float:
        return round(max(-bound, min(bound, value)), 4)

    def _append_directive(self, directives: list[str], directive: str) -> None:
        if directive and directive not in directives:
            directives.append(directive)

    def _append_subtype_directive(self, table: dict[str, list[str]], usage_subtype: str, directive: str) -> None:
        if not usage_subtype:
            return
        directives = table.setdefault(usage_subtype, [])
        if directive and directive not in directives:
            directives.append(directive)

    def _bump_subtype_delta(self, table: dict[str, float], usage_subtype: str, delta: float) -> None:
        if not usage_subtype:
            return
        current = float(table.get(usage_subtype, 0.0))
        table[usage_subtype] = self._clamp(current + delta)

    def _had_reuse_signal(self, trace: ExecutionTrace) -> bool:
        return bool(
            trace.selected_skill_id
            or trace.planner_visible_candidates
            or trace.hard_filter_survivors
            or trace.recall_candidates
        )

    def context(self, *, benchmark: str, task_pattern: str, selected_skill_id: str = "", usage_subtype: str = "") -> dict[str, object]:
        state = self.load()
        direct_delta = 0.0
        seeded_delta = 0.0
        directives: list[str] = []
        for table, key in (
            (state.benchmark_policies, benchmark),
            (state.task_pattern_policies, task_pattern),
        ):
            value = table.get(key)
            if value is None:
                continue
            direct_delta += float(value.direct_skill_delta)
            seeded_delta += float(value.seeded_delta)
            if usage_subtype:
                direct_delta += float(value.direct_skill_delta_by_subtype.get(usage_subtype, 0.0))
                seeded_delta += float(value.seeded_delta_by_subtype.get(usage_subtype, 0.0))
            for directive in value.prompt_directives:
                self._append_directive(directives, directive)
            for directive in value.prompt_directives_by_subtype.get(usage_subtype, []):
                self._append_directive(directives, directive)

        if selected_skill_id:
            value = state.skill_policies.get(selected_skill_id)
            if value is not None:
                direct_delta += float(value.direct_skill_delta)
                seeded_delta += float(value.seeded_delta)
                if usage_subtype:
                    direct_delta += float(value.direct_skill_delta_by_subtype.get(usage_subtype, 0.0))
                    seeded_delta += float(value.seeded_delta_by_subtype.get(usage_subtype, 0.0))
                for directive in value.prompt_directives:
                    self._append_directive(directives, directive)
                for directive in value.prompt_directives_by_subtype.get(usage_subtype, []):
                    self._append_directive(directives, directive)
                if value.suppressed:
                    self._append_directive(directives, "avoid suppressed skill reuse")

        return {
            "version": state.version,
            "direct_skill_delta": round(direct_delta, 4),
            "seeded_delta": round(seeded_delta, 4),
            "prompt_directives": directives,
        }

    def update_after_episode(
        self,
        *,
        trace: ExecutionTrace,
        selected_skill_status_after: str = "",
    ) -> tuple[int, str]:
        state = self.load()
        benchmark_policy = self._ensure_policy(state.benchmark_policies, trace.benchmark)
        pattern_policy = self._ensure_policy(state.task_pattern_policies, trace.task_pattern or "unknown")
        skill_policy = self._ensure_skill_policy(state.skill_policies, trace.selected_skill_id) if trace.selected_skill_id else None
        changes: list[str] = []
        now = self._now()
        usage_subtype = trace.realized_usage_subtype or trace.planned_usage_subtype or ""
        reuse_signal = self._had_reuse_signal(trace)

        if trace.episode_provenance == "pure_dynamic_success" and reuse_signal:
            # A fresh solve should only down-weight reuse when a concrete reuse
            # opportunity was actually present. Keep this local to direct
            # routing for the observed task pattern so shared-bank growth does
            # not learn an anti-seeded bias from unrelated fresh solves.
            pattern_policy.direct_skill_delta = self._clamp(pattern_policy.direct_skill_delta - 0.05)
            self._bump_subtype_delta(pattern_policy.direct_skill_delta_by_subtype, usage_subtype, -0.05)
            changes.append("pure_dynamic_success_bias")
        elif trace.episode_provenance == "seeded_dynamic_success":
            benchmark_policy.seeded_delta = self._clamp(benchmark_policy.seeded_delta + 0.1)
            pattern_policy.seeded_delta = self._clamp(pattern_policy.seeded_delta + 0.1)
            self._bump_subtype_delta(benchmark_policy.seeded_delta_by_subtype, usage_subtype, +0.1)
            self._bump_subtype_delta(pattern_policy.seeded_delta_by_subtype, usage_subtype, +0.1)
            self._append_directive(benchmark_policy.prompt_directives, "prefer selective seed reuse when prior fits")
            self._append_directive(pattern_policy.prompt_directives, "prefer selective seed reuse when prior fits")
            self._append_subtype_directive(benchmark_policy.prompt_directives_by_subtype, usage_subtype, "prefer selective seed reuse when prior fits")
            self._append_subtype_directive(pattern_policy.prompt_directives_by_subtype, usage_subtype, "prefer selective seed reuse when prior fits")
            if skill_policy is not None:
                skill_policy.seeded_delta = self._clamp(skill_policy.seeded_delta + 0.1)
                self._bump_subtype_delta(skill_policy.seeded_delta_by_subtype, usage_subtype, +0.1)
                self._append_directive(skill_policy.prompt_directives, "prefer reuse of this successful seed carefully")
                self._append_subtype_directive(skill_policy.prompt_directives_by_subtype, usage_subtype, "prefer reuse of this successful seed carefully")
            changes.append("seeded_dynamic_success_bias")
        elif trace.episode_provenance == "direct_skill_reuse" and trace.success:
            benchmark_policy.direct_skill_delta = self._clamp(benchmark_policy.direct_skill_delta + 0.05)
            pattern_policy.direct_skill_delta = self._clamp(pattern_policy.direct_skill_delta + 0.05)
            self._bump_subtype_delta(benchmark_policy.direct_skill_delta_by_subtype, usage_subtype, +0.05)
            self._bump_subtype_delta(pattern_policy.direct_skill_delta_by_subtype, usage_subtype, +0.05)
            if skill_policy is not None:
                skill_policy.direct_skill_delta = self._clamp(skill_policy.direct_skill_delta + 0.05)
                self._bump_subtype_delta(skill_policy.direct_skill_delta_by_subtype, usage_subtype, +0.05)
            changes.append("direct_skill_reuse_bias")

        if trace.planning_mode == "skill_seeded_dynamic" and not trace.success:
            benchmark_policy.seeded_delta = self._clamp(benchmark_policy.seeded_delta - 0.1)
            pattern_policy.seeded_delta = self._clamp(pattern_policy.seeded_delta - 0.1)
            self._bump_subtype_delta(benchmark_policy.seeded_delta_by_subtype, usage_subtype, -0.1)
            self._bump_subtype_delta(pattern_policy.seeded_delta_by_subtype, usage_subtype, -0.1)
            self._append_directive(pattern_policy.prompt_directives, "avoid over-relying on seed code")
            self._append_subtype_directive(pattern_policy.prompt_directives_by_subtype, usage_subtype, "avoid over-relying on seed code")
            if skill_policy is not None:
                skill_policy.seeded_delta = self._clamp(skill_policy.seeded_delta - 0.1)
                self._bump_subtype_delta(skill_policy.seeded_delta_by_subtype, usage_subtype, -0.1)
                self._append_directive(skill_policy.prompt_directives, "avoid over-relying on this seed code")
                self._append_subtype_directive(skill_policy.prompt_directives_by_subtype, usage_subtype, "avoid over-relying on this seed code")
            changes.append("seeded_failure_bias")

        if selected_skill_status_after in {"suppressed", "pruned"} and skill_policy is not None:
            skill_policy.suppressed = True
            skill_policy.direct_skill_delta = self._clamp(skill_policy.direct_skill_delta - 0.5)
            skill_policy.seeded_delta = self._clamp(skill_policy.seeded_delta - 0.5)
            self._append_directive(skill_policy.prompt_directives, f"avoid {selected_skill_status_after} skill reuse")
            changes.append(f"skill_{selected_skill_status_after}")

        if not changes:
            return state.version, "no_policy_change"

        for obj in (benchmark_policy, pattern_policy, skill_policy):
            if obj is not None:
                obj.updated_at = now

        state.version += 1
        state.last_updated_at = now
        self.save(state)
        summary = ", ".join(changes) if changes else "no_policy_change"
        return state.version, summary
