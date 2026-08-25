"""Compile successful episodes into lightweight runtime template priors."""

from __future__ import annotations

import hashlib
import json

from core.schemas import ExecutionTrace, Template


class TemplateCompiler:
    """Derive first-pass template priors from successful artifacts plus traces."""

    SEEDED_DIRECTIVE = "prefer selective seed reuse for this task pattern"
    SEEDED_REPAIR_DIRECTIVE = "preserve the seeded interface while repairing"
    REPAIR_DIRECTIVE = "repair against verifier feedback before rewriting structure"
    FRESH_REPAIR_DIRECTIVE = "prefer fresh structure with only local seed borrowing"

    def _template_id(self, kind: str, trigger_signature: dict[str, object]) -> str:
        digest = hashlib.sha256(json.dumps(trigger_signature, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        benchmark = str(trigger_signature.get("benchmark", "unknown"))
        task_pattern = str(trigger_signature.get("task_pattern", "unknown"))
        return f"tpl_{kind}_{benchmark}_{task_pattern}_{digest}"

    def _draft_conditioned_ids(self, trace: ExecutionTrace) -> list[str]:
        for event in trace.generation_events:
            if event.generation_stage == "draft" and event.conditioned_skill_ids:
                return [skill_id for skill_id in event.conditioned_skill_ids[:2] if skill_id]
        if trace.selected_skill_id:
            return [trace.selected_skill_id]
        if trace.retrieval_scores:
            ids = [str(item.get("skill_id", "")) for item in trace.retrieval_scores[:1]]
            return [skill_id for skill_id in ids if skill_id]
        return []

    def _first_failure_type(self, trace: ExecutionTrace) -> str:
        for attempt in trace.attempts:
            feedback = attempt.verifier_feedback
            if feedback is not None and not feedback.passed:
                return attempt.failure_type or trace.failure_type or "unknown"
        return trace.failure_type or "unknown"

    def _has_successful_repair(self, trace: ExecutionTrace) -> bool:
        return any(attempt.stage.startswith("repair") for attempt in trace.attempts)

    def _seeded_context_template(self, trace: ExecutionTrace) -> Template | None:
        if trace.planning_mode != "skill_seeded_dynamic" or not trace.success:
            return None
        skill_ids = self._draft_conditioned_ids(trace)
        if not skill_ids:
            return None
        trigger_signature = {
            "benchmark": trace.benchmark,
            "task_pattern": trace.task_pattern,
            "planning_mode": trace.planning_mode,
            "seed_skill_ids": skill_ids[:2],
        }
        directives = [self.SEEDED_DIRECTIVE]
        if self._has_successful_repair(trace):
            directives.append(self.SEEDED_REPAIR_DIRECTIVE)
        return Template(
            template_id=self._template_id("seeded_context", trigger_signature),
            benchmark=trace.benchmark,
            family=trace.benchmark,
            task_pattern=trace.task_pattern,
            template_kind="seeded_context",
            description=f"Seeded context prior for {trace.benchmark}/{trace.task_pattern}.",
            trigger_signature=trigger_signature,
            prompt_directives=directives,
            preferred_skill_ids=skill_ids[:2],
            preferred_skill_sequence=skill_ids[:2],
            support_count=1,
            exemplar_trace_ids=[trace.trace_id],
            status="draft",
            priority=1,
        )

    def _repair_motif_template(self, trace: ExecutionTrace) -> Template | None:
        if not trace.success or not self._has_successful_repair(trace):
            return None
        seeded = bool(trace.skill_seeded or trace.planning_mode == "skill_seeded_dynamic")
        failure_type = self._first_failure_type(trace)
        trigger_signature = {
            "benchmark": trace.benchmark,
            "task_pattern": trace.task_pattern,
            "seeded": seeded,
            "failure_type": failure_type,
        }
        directives = [self.REPAIR_DIRECTIVE]
        if seeded:
            directives.insert(0, self.SEEDED_REPAIR_DIRECTIVE)
        else:
            directives.append(self.FRESH_REPAIR_DIRECTIVE)
        skill_ids = self._draft_conditioned_ids(trace) if seeded else []
        return Template(
            template_id=self._template_id("repair_motif", trigger_signature),
            benchmark=trace.benchmark,
            family=trace.benchmark,
            task_pattern=trace.task_pattern,
            template_kind="repair_motif",
            description=f"Repair motif prior for {trace.benchmark}/{trace.task_pattern} after {failure_type}.",
            trigger_signature=trigger_signature,
            prompt_directives=directives,
            preferred_skill_ids=skill_ids[:2],
            preferred_skill_sequence=skill_ids[:2],
            support_count=1,
            exemplar_trace_ids=[trace.trace_id],
            status="draft",
            priority=1,
        )

    def compile_success_templates(self, trace: ExecutionTrace) -> list[Template]:
        if not trace.success:
            return []
        templates: list[Template] = []
        seeded_context = self._seeded_context_template(trace)
        if seeded_context is not None:
            templates.append(seeded_context)
        repair_motif = self._repair_motif_template(trace)
        if repair_motif is not None:
            templates.append(repair_motif)
        return templates
