"""Trace-level audit heuristics for governance and negative memory."""

from __future__ import annotations

from core.schemas import ExecutionTrace


def audit_trace(trace: ExecutionTrace) -> list[str]:
    """Return lightweight audit warnings derived from runtime traces."""
    warnings: list[str] = []
    if trace.failure_type == "generation_error":
        warnings.append("generation_error")
    if trace.failure_type == "timeout":
        warnings.append("timeout")
    if len(trace.attempts) >= 3:
        warnings.append("heavy_repair_chain")
    if trace.llm_total_tokens_total >= 2000:
        warnings.append("high_token_cost")
    if trace.planning_mode == "skill_seeded_dynamic" and not trace.success:
        warnings.append("negative_seed_transfer")
    if trace.draft_injected_template_ids and not trace.success:
        warnings.append("template_assisted_failure")
    if trace.injected_primitive_ids and not trace.success:
        warnings.append("primitive_assisted_failure")
    return warnings
