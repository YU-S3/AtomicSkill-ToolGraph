"""Compile successful traces into textual workflow-memory summaries."""

from __future__ import annotations

import hashlib
import json

from core.schemas import ExecutionTrace, WorkflowMemoryCard


class WorkflowMemoryCompiler:
    """Build deterministic workflow summaries without LLM summarization."""

    def _memory_id(self, benchmark: str, task_pattern: str, signature: dict[str, object]) -> str:
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        return f"wm_{benchmark}_{task_pattern}_{digest}"

    def _attempt_stages(self, trace: ExecutionTrace) -> list[str]:
        return [attempt.stage for attempt in trace.attempts]

    def _first_failure_type(self, trace: ExecutionTrace) -> str:
        for attempt in trace.attempts:
            feedback = attempt.verifier_feedback
            if feedback is not None and not feedback.passed:
                return attempt.failure_type or trace.failure_type or "unknown"
        return trace.failure_type or "unknown"

    def compile_success_memory(self, trace: ExecutionTrace) -> WorkflowMemoryCard | None:
        if not trace.success:
            return None

        attempt_stages = self._attempt_stages(trace)
        repair_attempts = sum(1 for attempt in trace.attempts if attempt.stage.startswith("repair"))
        first_failure_type = self._first_failure_type(trace)
        signature = {
            "benchmark": trace.benchmark,
            "task_pattern": trace.task_pattern,
            "planning_mode": trace.planning_mode,
            "repair_attempts": repair_attempts,
            "first_failure_type": first_failure_type,
        }
        workflow_summary = (
            f"Planning mode `{trace.planning_mode}` succeeded with attempt stages "
            f"{' -> '.join(attempt_stages) if attempt_stages else 'draft_only'}. "
            "Keep the required signature fixed and follow verifier-driven iteration."
        )
        if repair_attempts:
            repair_summary = (
                f"First observed failure type: `{first_failure_type}`. "
                f"The successful run required {repair_attempts} repair step(s); "
                "inspect the verifier feedback before changing structure."
            )
        else:
            repair_summary = "This successful run passed without repair; prefer a clean first draft before broader rewrites."
        applicability_notes = (
            f"benchmark={trace.benchmark}; task_pattern={trace.task_pattern}; "
            f"selection_reason={trace.selection_reason[:160]}"
        )
        return WorkflowMemoryCard(
            memory_id=self._memory_id(trace.benchmark, trace.task_pattern or "unknown", signature),
            benchmark=trace.benchmark,
            task_pattern=trace.task_pattern or "unknown",
            workflow_summary=workflow_summary,
            repair_summary=repair_summary,
            applicability_notes=applicability_notes,
            support_count=1,
        )
