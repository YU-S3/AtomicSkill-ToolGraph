"""Utility scoring heuristics for runtime governance."""

from __future__ import annotations

from core.schemas import ExecutionTrace


def score_episode(trace: ExecutionTrace) -> float:
    """Score one episode using success, reuse, growth, and efficiency signals."""
    score = 0.0
    if trace.success:
        score += 1.0
    else:
        score -= 0.5

    if trace.planning_mode == "direct_skill" and trace.skill_hit:
        score += 0.4
    if trace.planning_mode == "skill_seeded_dynamic" and trace.success:
        score += 0.25
    if trace.compiled_skill_admitted:
        score += 0.3
    if trace.compiled_primitive_admitted:
        score += 0.2

    attempts = max(1, len(trace.attempts))
    score -= 0.05 * max(0, attempts - 1)
    score -= min(0.5, float(trace.llm_total_tokens_total) / 5000.0)
    score -= min(0.25, float(trace.llm_latency_ms_total) / 20000.0)

    if trace.failure_type in {"generation_error", "timeout"}:
        score -= 0.4
    if trace.fallback_triggered:
        score -= 0.1

    return round(score, 4)
