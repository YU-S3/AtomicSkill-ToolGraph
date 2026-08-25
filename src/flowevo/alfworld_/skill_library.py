"""Three-layer skill library for ALFWorld.

Stores three independent kinds of artifacts compiled from successful traces:

* **Layer 1 — Executable Templates** (``AlfWorldSkillTemplate``)
  Direct reuse via the executor's ``_execute_direct`` path. Zero LLM tokens
  per reuse. Matched on exact ``task_type``.

* **Layer 2 — Trajectory Exemplars** (``AlfWorldExemplar``)
  Denoised successful action sequences injected as few-shot examples into
  the generator prompt. Matched on exact ``task_type``.

* **Layer 3 — Structural Insights** (``AlfWorldInsight``)
  Cross-trace statistical knowledge (search priorities, common locations)
  injected as policy directives. Aggregated automatically from trace stats.

Retrieval priority for any task: Layer 1 > Layer 2 > Layer 3.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from alfworld_.schemas import (
    AlfWorldExemplar,
    AlfWorldInsight,
    AlfWorldSkillTemplate,
    AlfWorldTrace,
)


_TAKE_RE = re.compile(r"take\s+.+?\s+from\s+(.+)", re.IGNORECASE)


# Per-task-type environment hints injected into Layer 3 insights once we
# have enough samples.  These are stable facts about the simulator, not
# learned knowledge — but the insight only surfaces them when the agent has
# proven (via successes) that the task type is being attempted.
_ENVIRONMENT_FACTS: dict[str, list[str]] = {
    "pick_clean_then_place_in_recep": ["sinkbasin is usually numbered 1"],
    "pick_heat_then_place_in_recep": ["microwave is usually numbered 1"],
    "pick_cool_then_place_in_recep": ["fridge is usually numbered 1"],
}


class SkillLibrary:
    """Unified three-layer skill storage."""

    _REPLACE_THRESHOLD: float = 0.5
    _MAX_EXEMPLARS_PER_TYPE: int = 3
    _INSIGHT_MIN_SAMPLES: int = 3

    # Governance thresholds
    _AUDIT_UTILITY_THRESHOLD: float = 0.5
    _AUDIT_MIN_USES: int = 3
    _AUDIT_INACTIVITY_LIMIT: int = 50  # episodes since last_used

    # ContrastiveEval thresholds
    _CONTRASTIVE_MIN_GUIDED: int = 5
    _CONTRASTIVE_MIN_UNGUIDED: int = 3
    _CONTRASTIVE_HARM_THRESHOLD: float = -0.1

    def __init__(self, governance_enabled: bool = True) -> None:
        self._governance_enabled = governance_enabled

        # Layer 1
        self._templates: dict[str, AlfWorldSkillTemplate] = {}
        self._template_by_type: dict[str, str] = {}
        self._template_utility: dict[str, float] = {}
        self._template_status: dict[str, str] = {}
        self._template_use_count: dict[str, int] = {}
        self._template_success_count: dict[str, int] = {}
        self._template_failure_count: dict[str, int] = {}
        self._template_last_used: dict[str, int] = {}

        # Layer 2
        self._exemplars: dict[str, list[AlfWorldExemplar]] = {}

        # Layer 3
        self._insights: dict[str, AlfWorldInsight] = {}
        self._trace_stats: dict[str, list[dict[str, Any]]] = {}

        # ContrastiveEval (per task_type guided/unguided tracking)
        self._contrastive_stats: dict[str, dict[str, int]] = {}
        self._injection_suppressed: set[str] = set()

        # Episode counter for activity-based governance
        self._current_episode: int = 0

    # ==================================================================
    # Ingestion
    # ==================================================================

    def add_template(self, template: AlfWorldSkillTemplate) -> bool:
        """Add a Layer 1 template. One per task_type, replaceable on low
        utility. Returns True when ingested."""
        task_type = template.task_type
        existing_tid = self._template_by_type.get(task_type)
        if existing_tid:
            if self._governance_enabled:
                existing_status = self._template_status.get(existing_tid, "active")
                existing_utility = self._template_utility.get(existing_tid, 1.0)
                if (
                    existing_status == "active"
                    and existing_utility >= self._REPLACE_THRESHOLD
                ):
                    return False
            self._remove_template(existing_tid)

        tid = template.template_id
        self._templates[tid] = template
        self._template_by_type[task_type] = tid
        self._template_utility[tid] = 1.0
        self._template_status[tid] = "active"
        self._template_use_count[tid] = 0
        self._template_success_count[tid] = 0
        self._template_failure_count[tid] = 0
        return True

    def add_exemplar(self, exemplar: AlfWorldExemplar) -> None:
        """Add a Layer 2 exemplar. Keeps the most recent N per task_type."""
        task_type = exemplar.task_type
        bucket = self._exemplars.setdefault(task_type, [])
        bucket.append(exemplar)
        if len(bucket) > self._MAX_EXEMPLARS_PER_TYPE:
            self._exemplars[task_type] = bucket[-self._MAX_EXEMPLARS_PER_TYPE:]

    def record_trace_stats(self, trace: AlfWorldTrace) -> None:
        """Accumulate Layer 3 statistics from a trace (success OR failure).

        Successful traces contribute pickup-location frequencies.  Failed
        traces contribute failure-mode counts that get rendered into Layer 3
        ``common_pitfalls``.  Both flow into the same per-task-type bucket.
        """
        task_type = trace.task_type
        bucket = self._trace_stats.setdefault(task_type, [])

        object_locations: list[str] = []
        actions_used: list[str] = []
        for action_record in trace.actions:
            action = action_record.action
            m = _TAKE_RE.match(action)
            if m:
                object_locations.append(m.group(1).strip())
            head = action.split()[0].lower() if action else ""
            if head:
                actions_used.append(head)

        # Failure mode classification (only meaningful when not success).
        failure_reason = ""
        if not trace.success:
            if trace.total_steps >= trace.max_steps:
                failure_reason = "timeout"
            elif len(trace.actions) >= 10:
                last_10 = [a.action for a in trace.actions[-10:]]
                if len(set(last_10)) <= 3:
                    failure_reason = "action_loop"
                else:
                    failure_reason = "other"
            else:
                failure_reason = "other"

        bucket.append({
            "success": trace.success,
            "steps": trace.total_steps,
            "object_locations": object_locations,
            "actions_used": actions_used,
            "failure_reason": failure_reason,
        })

        if len(bucket) >= self._INSIGHT_MIN_SAMPLES:
            self._update_insight(task_type)

    def _update_insight(self, task_type: str) -> None:
        stats_list = self._trace_stats[task_type]

        # Locations are only meaningful from successful traces.
        location_counter: Counter[str] = Counter()
        for stats in stats_list:
            if not stats.get("success", True):
                continue
            for loc in stats.get("object_locations", []):
                loc_type = re.sub(r"\s+\d+$", "", loc).strip()
                if loc_type:
                    location_counter[loc_type] += 1

        if location_counter:
            sorted_locs = [loc for loc, _ in location_counter.most_common(5)]
            search_priority = " > ".join(sorted_locs)
            common_locations = sorted_locs
        else:
            search_priority = ""
            common_locations = []

        # Failure-mode aggregation -> common_pitfalls
        failure_reasons: Counter[str] = Counter()
        for stats in stats_list:
            if stats.get("success", True):
                continue
            reason = stats.get("failure_reason", "")
            if reason:
                failure_reasons[reason] += 1

        common_pitfalls: list[str] = []
        total_failures = sum(failure_reasons.values())
        if total_failures > 0:
            for reason, count in failure_reasons.most_common(3):
                pct = count / total_failures * 100
                if reason == "timeout":
                    common_pitfalls.append(
                        "%.0f%% of failures are timeouts -- explore more efficiently"
                        % pct
                    )
                elif reason == "action_loop":
                    common_pitfalls.append(
                        "%.0f%% of failures involve action loops -- try different approaches"
                        % pct
                    )
                else:
                    common_pitfalls.append(
                        "%.0f%% of failures had no clear pattern" % pct
                    )

        env_facts = list(_ENVIRONMENT_FACTS.get(task_type, []))

        self._insights[task_type] = AlfWorldInsight(
            task_type=task_type,
            search_priority=search_priority,
            common_locations=common_locations,
            common_pitfalls=common_pitfalls,
            environment_facts=env_facts,
            sample_count=len(stats_list),
        )

    # ==================================================================
    # Retrieval
    # ==================================================================

    def retrieve(self, task_type: str) -> dict[str, Any]:
        """Return all available layers for *task_type* plus the best layer.

        Output dict keys: ``template``, ``exemplar``, ``insight``,
        ``best_layer`` (one of 1/2/3 or ``None``).

        When ContrastiveEval has flagged this ``task_type`` as showing
        negative transfer for injection-based layers, Layer 2 (exemplar) and
        Layer 3 (insight) are masked out.  Layer 1 (template) is always
        returned because direct execution does not go through the LLM and
        therefore cannot suffer from prompt-injection-induced regressions.
        """
        template = self._get_active_template(task_type)
        injection_blocked = (
            self._governance_enabled and task_type in self._injection_suppressed
        )
        exemplar = None if injection_blocked else self._get_best_exemplar(task_type)
        insight = None if injection_blocked else self._insights.get(task_type)

        if template is not None:
            best = 1
        elif exemplar is not None:
            best = 2
        elif insight is not None:
            best = 3
        else:
            best = None

        return {
            "template": template,
            "exemplar": exemplar,
            "insight": insight,
            "best_layer": best,
        }

    def _get_active_template(self, task_type: str) -> AlfWorldSkillTemplate | None:
        tid = self._template_by_type.get(task_type)
        if tid and self._template_status.get(tid) == "active":
            return self._templates.get(tid)
        return None

    def _get_best_exemplar(self, task_type: str) -> AlfWorldExemplar | None:
        bucket = self._exemplars.get(task_type, [])
        return bucket[-1] if bucket else None

    # ==================================================================
    # Layer 1 utility tracking
    # ==================================================================

    def advance_episode(self) -> None:
        """Tick the internal episode counter.

        Callers should invoke this once at the start of every episode so
        ``periodic_audit`` can compute template inactivity in episode units.
        """
        self._current_episode += 1

    def record_template_usage(self, template_id: str, success: bool) -> None:
        if template_id not in self._templates:
            return
        self._template_use_count[template_id] = (
            self._template_use_count.get(template_id, 0) + 1
        )
        if success:
            self._template_success_count[template_id] = (
                self._template_success_count.get(template_id, 0) + 1
            )
        else:
            self._template_failure_count[template_id] = (
                self._template_failure_count.get(template_id, 0) + 1
            )

        total = self._template_use_count[template_id]
        succ = self._template_success_count.get(template_id, 0)
        self._template_utility[template_id] = succ / max(total, 1)
        self._template_last_used[template_id] = self._current_episode

        if (
            self._template_failure_count.get(template_id, 0) >= 3
            and self._template_utility[template_id] < 0.3
        ):
            self._template_status[template_id] = "suppressed"

    def _remove_template(self, tid: str) -> None:
        template = self._templates.pop(tid, None)
        self._template_utility.pop(tid, None)
        self._template_status.pop(tid, None)
        self._template_use_count.pop(tid, None)
        self._template_success_count.pop(tid, None)
        self._template_failure_count.pop(tid, None)
        if template is not None:
            tt = template.task_type
            if self._template_by_type.get(tt) == tid:
                del self._template_by_type[tt]

    # ==================================================================
    # ContrastiveEval (per task_type guided/unguided tracking)
    # ==================================================================

    def _new_contrastive_record(self) -> dict[str, int]:
        return {
            "guided_success": 0,
            "guided_total": 0,
            "unguided_success": 0,
            "unguided_total": 0,
        }

    def record_guided(self, task_type: str, success: bool) -> None:
        """Record a guided episode (Layer 2 or Layer 3 injected)."""
        stats = self._contrastive_stats.setdefault(
            task_type, self._new_contrastive_record(),
        )
        stats["guided_total"] += 1
        if success:
            stats["guided_success"] += 1
        self._check_contrastive_suppress(task_type)

    def record_unguided(self, task_type: str, success: bool) -> None:
        """Record a holdout episode that ran without skill injection."""
        stats = self._contrastive_stats.setdefault(
            task_type, self._new_contrastive_record(),
        )
        stats["unguided_total"] += 1
        if success:
            stats["unguided_success"] += 1
        self._check_contrastive_suppress(task_type)

    def get_contrastive_delta(self, task_type: str) -> float | None:
        """Return ``guided_rate - unguided_rate`` or ``None`` if too small.

        Requires at least ``_CONTRASTIVE_MIN_GUIDED`` guided samples and
        ``_CONTRASTIVE_MIN_UNGUIDED`` unguided samples to have any
        statistical meaning.
        """
        stats = self._contrastive_stats.get(task_type)
        if not stats:
            return None
        if (
            stats["guided_total"] < self._CONTRASTIVE_MIN_GUIDED
            or stats["unguided_total"] < self._CONTRASTIVE_MIN_UNGUIDED
        ):
            return None
        g_rate = stats["guided_success"] / max(stats["guided_total"], 1)
        u_rate = stats["unguided_success"] / max(stats["unguided_total"], 1)
        return g_rate - u_rate

    def _check_contrastive_suppress(self, task_type: str) -> bool:
        """If guided is materially worse than unguided, suppress injection.

        Note: this only suppresses Layer 2/3 retrieval for this task_type;
        Layer 1 (direct execute) is unaffected because direct execution
        does not go through the LLM and therefore cannot be harmed by a
        bad prompt injection.
        """
        delta = self.get_contrastive_delta(task_type)
        if delta is not None and delta < self._CONTRASTIVE_HARM_THRESHOLD:
            self._injection_suppressed.add(task_type)
            return True
        return False

    # ==================================================================
    # Governance
    # ==================================================================

    def periodic_audit(self) -> dict[str, list[tuple[str, float]]]:
        """Three-dimension audit of the active library.

        Returns a dict with three keys:

        * ``suppressed_low_utility``: ``(template_id, success_rate)`` pairs
          for templates whose success rate fell below the threshold.
        * ``suppressed_inactive``: ``(template_id, episodes_since_used)``
          pairs for templates that have not been used in too many episodes.
        * ``suppressed_negative_transfer``: ``(task_type, contrastive_delta)``
          pairs for task types whose injected layers were doing more harm
          than good.

        When governance is disabled, returns an empty result (no suppression).
        """
        result: dict[str, list[tuple[str, float]]] = {
            "suppressed_low_utility": [],
            "suppressed_inactive": [],
            "suppressed_negative_transfer": [],
        }
        if not self._governance_enabled:
            return result

        for tid in list(self._templates.keys()):
            if self._template_status.get(tid) != "active":
                continue

            # Dimension 1: success rate
            total = self._template_use_count.get(tid, 0)
            if total >= self._AUDIT_MIN_USES:
                rate = self._template_success_count.get(tid, 0) / max(total, 1)
                if rate < self._AUDIT_UTILITY_THRESHOLD:
                    self._template_status[tid] = "suppressed"
                    result["suppressed_low_utility"].append((tid, rate))
                    continue

            # Dimension 2: inactivity (only meaningful once we've started
            # ticking the episode counter via advance_episode).
            if self._current_episode > 0:
                last_used = self._template_last_used.get(tid, 0)
                gap = self._current_episode - last_used
                if gap > self._AUDIT_INACTIVITY_LIMIT:
                    self._template_status[tid] = "suppressed"
                    result["suppressed_inactive"].append((tid, float(gap)))
                    continue

        # Dimension 3: contrastive negative transfer (per task_type)
        for task_type in list(self._contrastive_stats.keys()):
            if task_type in self._injection_suppressed:
                continue
            if self._check_contrastive_suppress(task_type):
                delta = self.get_contrastive_delta(task_type)
                result["suppressed_negative_transfer"].append(
                    (task_type, float(delta) if delta is not None else 0.0)
                )

        return result

    # ==================================================================
    # Snapshot / serialization
    # ==================================================================

    def snapshot(self) -> dict[str, Any]:
        return {
            "templates_total": len(self._templates),
            "templates_active": sum(
                1 for s in self._template_status.values() if s == "active"
            ),
            "templates_suppressed": sum(
                1 for s in self._template_status.values() if s == "suppressed"
            ),
            "exemplars_total": sum(len(v) for v in self._exemplars.values()),
            "exemplars_types": len(self._exemplars),
            "insights_total": len(self._insights),
            "contrastive_types": len(self._contrastive_stats),
            "injection_suppressed_types": len(self._injection_suppressed),
            "current_episode": self._current_episode,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "templates": {tid: t.to_dict() for tid, t in self._templates.items()},
            "template_by_type": dict(self._template_by_type),
            "template_utility": dict(self._template_utility),
            "template_status": dict(self._template_status),
            "template_use_count": dict(self._template_use_count),
            "template_success_count": dict(self._template_success_count),
            "template_failure_count": dict(self._template_failure_count),
            "template_last_used": dict(self._template_last_used),
            "exemplars": {
                tt: [e.to_dict() for e in exms]
                for tt, exms in self._exemplars.items()
            },
            "insights": {tt: ins.to_dict() for tt, ins in self._insights.items()},
            "trace_stats": {tt: list(v) for tt, v in self._trace_stats.items()},
            "contrastive_stats": {
                tt: dict(s) for tt, s in self._contrastive_stats.items()
            },
            "injection_suppressed": sorted(self._injection_suppressed),
            "current_episode": self._current_episode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillLibrary":
        lib = cls()
        for tid, tdata in data.get("templates", {}).items():
            lib._templates[tid] = AlfWorldSkillTemplate.from_dict(tdata)
        lib._template_by_type = dict(data.get("template_by_type", {}))
        lib._template_utility = dict(data.get("template_utility", {}))
        lib._template_status = dict(data.get("template_status", {}))
        lib._template_use_count = dict(data.get("template_use_count", {}))
        lib._template_success_count = dict(data.get("template_success_count", {}))
        lib._template_failure_count = dict(data.get("template_failure_count", {}))
        lib._template_last_used = {
            tid: int(v) for tid, v in data.get("template_last_used", {}).items()
        }
        for tt, exm_list in data.get("exemplars", {}).items():
            lib._exemplars[tt] = [
                AlfWorldExemplar.from_dict(d) for d in exm_list
            ]
        for tt, ins_data in data.get("insights", {}).items():
            lib._insights[tt] = AlfWorldInsight.from_dict(ins_data)
        lib._trace_stats = {
            tt: list(v) for tt, v in data.get("trace_stats", {}).items()
        }
        lib._contrastive_stats = {
            tt: {k: int(v) for k, v in s.items()}
            for tt, s in data.get("contrastive_stats", {}).items()
        }
        lib._injection_suppressed = set(data.get("injection_suppressed", []))
        lib._current_episode = int(data.get("current_episode", 0))
        return lib

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        loaded = SkillLibrary.from_dict(data)
        self.__dict__.update(loaded.__dict__)

    @property
    def active_template_count(self) -> int:
        return sum(1 for s in self._template_status.values() if s == "active")

    @property
    def total_template_count(self) -> int:
        return len(self._templates)
