"""v7 third equation: G_{t+1} = MetaUpdate(G_t, S_{t+1}, utility, failures).

Pure function that takes the current GeneratorState, an ExecutionTrace, and
the SkillRegistry, and returns a new GeneratorState.  It never mutates its
inputs and never writes to any store — the caller decides what to persist.
"""

from __future__ import annotations

import copy
from dataclasses import replace

from core.generator_state import (
    GeneratorState,
    RoutingBiases,
    SkillActionEntry,
    WorkflowMotif,
    _infer_insertion_mode,
    _infer_role,
    _safe_rate,
)
from core.schemas import ExecutionTrace


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DIRECT_DEMOTION_CONSECUTIVE_FAILURES = 2
_MACRO_PROMOTION_SEEDED_RATE = 0.7
_MACRO_PROMOTION_MIN_SUPPORT = 3
_MOTIF_REGISTRATION_THRESHOLD = 3
_UTILITY_DIRECTIVE_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return round(max(lo, min(hi, value)), 4)


def _collect_used_skill_ids(trace: ExecutionTrace) -> list[str]:
    """Extract all skill ids that were actively used in this episode."""
    ids: list[str] = []
    if trace.selected_skill_id:
        ids.append(trace.selected_skill_id)
    for sid in trace.used_skills:
        if sid not in ids:
            ids.append(sid)
    for sid in trace.compiled_skill_ids:
        if sid not in ids:
            ids.append(sid)
    return ids


# ---------------------------------------------------------------------------
# Layer 1: action-space update
# ---------------------------------------------------------------------------

def _update_action_layer(
    space: dict[str, SkillActionEntry],
    trace: ExecutionTrace,
    skill_registry,  # noqa: ANN001
) -> dict[str, SkillActionEntry]:
    """Return an updated copy of the skill action space."""
    space = {k: copy.copy(v) for k, v in space.items()}

    used_ids = _collect_used_skill_ids(trace)

    for skill_id in used_ids:
        card = skill_registry.get(skill_id)
        if card is None:
            continue

        # Refresh entry from latest card stats.
        direct_rate = _safe_rate(card.direct_success_count, card.direct_use_count)
        seeded_rate = _safe_rate(card.seeded_success_count, card.seeded_use_count)

        entry = space.get(skill_id)
        if entry is None:
            entry = SkillActionEntry(
                skill_id=skill_id,
                role=_infer_role(card),
                insertion_mode=_infer_insertion_mode(card),
                utility=float(card.utility),
                direct_success_rate=direct_rate,
                seeded_success_rate=seeded_rate,
                preferred_usage="either",
            )
        else:
            entry.utility = float(card.utility)
            entry.direct_success_rate = direct_rate
            entry.seeded_success_rate = seeded_rate

        # Rule 1: consecutive direct failures >= 2 → demote to seeded.
        if (
            card.consecutive_failures >= _DIRECT_DEMOTION_CONSECUTIVE_FAILURES
            and entry.preferred_usage == "direct"
        ):
            entry.preferred_usage = "seeded"
            entry.role = "planning_prior"

        # Rule 2: seeded success rate > 0.7 with enough support → promote.
        seeded_total = card.seeded_use_count
        if (
            seeded_rate > _MACRO_PROMOTION_SEEDED_RATE
            and seeded_total >= _MACRO_PROMOTION_MIN_SUPPORT
        ):
            entry.role = "macro_action"

        # Suppressed / pruned cards → disable.
        if card.status in ("suppressed", "pruned"):
            entry.role = "suppressed"
            entry.insertion_mode = "disabled"
            entry.preferred_usage = "none"

        space[skill_id] = entry

    return space


# ---------------------------------------------------------------------------
# Layer 2: structure (motif) update
# ---------------------------------------------------------------------------

def _update_structure_layer(
    motifs: list[WorkflowMotif],
    trace: ExecutionTrace,
) -> list[WorkflowMotif]:
    """Return an updated copy of the active motif list."""
    motifs = [copy.copy(m) for m in motifs]

    # Only record motifs from successful skill-seeded episodes.
    if not (trace.success and trace.planning_mode == "skill_seeded_dynamic"):
        return motifs

    used_ids = _collect_used_skill_ids(trace)
    if not used_ids:
        return motifs

    task_pattern = trace.task_pattern or "unknown"
    seq_key = tuple(sorted(used_ids))

    # Try to find an existing motif with the same pattern + sequence.
    matched: WorkflowMotif | None = None
    for motif in motifs:
        if motif.task_pattern == task_pattern and tuple(sorted(motif.skill_sequence)) == seq_key:
            matched = motif
            break

    if matched is not None:
        matched.support_count += 1
        total = matched.support_count or 1
        # Incrementally update success_rate assuming all supports were successes
        # up to now (we only enter this branch on success).
        matched.success_rate = _safe_rate(total, total)
    else:
        # Create a candidate; it only becomes "active" once support >= threshold.
        candidate = WorkflowMotif(
            motif_id=f"motif_{task_pattern}_{'_'.join(used_ids[:3])}",
            task_pattern=task_pattern,
            skill_sequence=used_ids,
            support_count=1,
            success_rate=1.0,
            avg_token_saving=0.0,
        )
        motifs.append(candidate)

    # Prune motifs that haven't reached the activation threshold — keep them
    # around until they do, but callers can filter on support_count.
    # (We keep all motifs in the list so support can accumulate.)
    return motifs


# ---------------------------------------------------------------------------
# Layer 3: meta-update (biases + directives)
# ---------------------------------------------------------------------------

def _update_meta_layer(
    routing_biases: RoutingBiases,
    directives: list[str],
    trace: ExecutionTrace,
    skill_registry,  # noqa: ANN001
) -> tuple[RoutingBiases, list[str]]:
    """Return updated routing biases and prompt directives."""
    biases = replace(routing_biases)
    directives = list(directives)  # shallow copy

    # --- bias adjustments mirroring PolicyStore.update_after_episode logic ---
    usage_subtype = trace.realized_usage_subtype or trace.planned_usage_subtype or ""

    if trace.episode_provenance == "seeded_dynamic_success":
        biases.seeded_delta = _clamp(biases.seeded_delta + 0.1)
    elif trace.episode_provenance == "direct_skill_reuse" and trace.success:
        biases.direct_skill_delta = _clamp(biases.direct_skill_delta + 0.05)

    if trace.planning_mode == "skill_seeded_dynamic" and not trace.success:
        biases.seeded_delta = _clamp(biases.seeded_delta - 0.1)

    # --- new v7 directives ---

    # Directive: if a task_pattern's active skills have avg utility > threshold,
    # prefer skill-conditioned generation.
    task_pattern = trace.task_pattern or ""
    if task_pattern:
        pattern_utilities: list[float] = []
        for card in skill_registry.list_active():
            card_pattern = str(card.preconditions.get("task_pattern", "") or "")
            if card_pattern == task_pattern:
                pattern_utilities.append(float(card.utility))
        if pattern_utilities:
            avg_utility = sum(pattern_utilities) / len(pattern_utilities)
            if avg_utility > _UTILITY_DIRECTIVE_THRESHOLD:
                directive = f"prefer skill-conditioned generation for {task_pattern}"
                if directive not in directives:
                    directives.append(directive)

    # Directive: if a skill has more negative than positive transfer, avoid it.
    for skill_id in _collect_used_skill_ids(trace):
        card = skill_registry.get(skill_id)
        if card is None:
            continue
        if card.negative_transfer_count > card.positive_transfer_count and card.negative_transfer_count > 0:
            directive = f"avoid {skill_id} for seeded generation"
            if directive not in directives:
                directives.append(directive)

    return biases, directives


# ---------------------------------------------------------------------------
# Layer 4: strategy directives from global skill bank statistics
# ---------------------------------------------------------------------------

_STRATEGY_PREFIX = "[strategy] "
_STRATEGY_MIN_EPISODES = 5


def _generate_strategy_directives(
    skill_registry,  # noqa: ANN001
    current_directives: list[str],
    episode_count: int,
) -> list[str]:
    """Extract strategy-level planning directives from the skill bank.

    This is the v7 "planning prior" layer: instead of injecting skill code,
    we inject decision hints derived from historical success/failure patterns.
    """
    if episode_count < _STRATEGY_MIN_EPISODES:
        return list(current_directives)

    # Keep non-strategy directives; rebuild strategy ones from scratch.
    directives = [d for d in current_directives if not d.startswith(_STRATEGY_PREFIX)]

    active_skills = skill_registry.list_active()
    if not active_skills:
        return directives

    # Aggregate per-task_pattern statistics.
    pattern_stats: dict[str, dict[str, int | float]] = {}
    for card in active_skills:
        pattern = str(card.preconditions.get("task_pattern", "unknown") or "unknown")
        if pattern not in pattern_stats:
            pattern_stats[pattern] = {
                "count": 0,
                "total_utility": 0.0,
                "direct_success": 0,
                "direct_failure": 0,
                "seeded_success": 0,
                "seeded_failure": 0,
                "total_positive_transfer": 0,
                "total_negative_transfer": 0,
            }
        s = pattern_stats[pattern]
        s["count"] += 1
        s["total_utility"] += float(card.utility)
        s["direct_success"] += int(card.direct_success_count or 0)
        s["direct_failure"] += int(card.direct_failure_count or 0)
        s["seeded_success"] += int(card.seeded_success_count or 0)
        s["seeded_failure"] += int(card.seeded_failure_count or 0)
        s["total_positive_transfer"] += int(card.positive_transfer_count or 0)
        s["total_negative_transfer"] += int(card.negative_transfer_count or 0)

    for pattern, s in pattern_stats.items():
        avg_utility = s["total_utility"] / max(s["count"], 1)
        direct_total = s["direct_success"] + s["direct_failure"]
        seeded_total = s["seeded_success"] + s["seeded_failure"]

        # High-utility pattern with reliable direct reuse.
        if avg_utility >= 0.8 and direct_total >= 2:
            direct_rate = s["direct_success"] / max(direct_total, 1)
            if direct_rate >= 0.7:
                directives.append(
                    _STRATEGY_PREFIX
                    + "pattern '%s': direct skill reuse is reliable "
                    "(%d/%d success, avg_utility=%.2f)"
                    % (pattern, s["direct_success"], direct_total, avg_utility)
                )

        # Seeded generation track record.
        if seeded_total >= 3:
            seeded_rate = s["seeded_success"] / max(seeded_total, 1)
            if seeded_rate < 0.4:
                directives.append(
                    _STRATEGY_PREFIX
                    + "pattern '%s': avoid seeded generation -- "
                    "historical seeded success only %d/%d"
                    % (pattern, s["seeded_success"], seeded_total)
                )
            elif seeded_rate >= 0.7:
                directives.append(
                    _STRATEGY_PREFIX
                    + "pattern '%s': seeded generation is effective "
                    "(%d/%d success)"
                    % (pattern, s["seeded_success"], seeded_total)
                )

        # High negative transfer → explicit warning.
        if (
            s["total_negative_transfer"] >= 3
            and s["total_negative_transfer"] > s["total_positive_transfer"]
        ):
            directives.append(
                _STRATEGY_PREFIX
                + "pattern '%s': high negative transfer risk -- "
                "consider fresh dynamic generation instead of skill reuse"
                % pattern
            )

    return directives


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def meta_update(
    state: GeneratorState,
    trace: ExecutionTrace,
    skill_registry,  # noqa: ANN001  memory.skill_registry.SkillRegistry
) -> GeneratorState:
    """v7 third equation: produce G_{t+1} from (G_t, Trace(W_t), S_{t+1}).

    Pure function — the input ``state`` is never mutated.
    """

    new_action_space = _update_action_layer(
        state.skill_action_space, trace, skill_registry,
    )

    new_motifs = _update_structure_layer(
        state.active_motifs, trace,
    )

    new_biases, new_directives = _update_meta_layer(
        state.routing_biases, state.prompt_directives, trace, skill_registry,
    )

    new_episode_count = state.episode_count + 1

    # Strategy directives from global skill bank statistics.
    new_directives = _generate_strategy_directives(
        skill_registry=skill_registry,
        current_directives=new_directives,
        episode_count=new_episode_count,
    )

    return GeneratorState(
        skill_action_space=new_action_space,
        active_motifs=new_motifs,
        routing_biases=new_biases,
        prompt_directives=new_directives,
        version=state.version + 1,
        episode_count=new_episode_count,
    )
