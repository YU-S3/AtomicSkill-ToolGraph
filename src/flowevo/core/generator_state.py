"""v7 third equation: unified generator state G_t.

Assembles the full mutable state of the generator from PolicyStore,
TemplateStore, and SkillRegistry so that meta_update can operate on a
single, coherent object.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Component dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SkillActionEntry:
    """Single skill's role inside the generator action space."""

    skill_id: str
    role: str  # "macro_action" | "planning_prior" | "local_structure" | "suppressed"
    insertion_mode: str  # "hard_skeleton" | "soft_hint" | "guard_only" | "disabled"
    utility: float
    direct_success_rate: float
    seeded_success_rate: float
    preferred_usage: str  # "direct" | "seeded" | "either" | "none"


@dataclass
class WorkflowMotif:
    """Validated high-frequency skill combination reusable as a template."""

    motif_id: str
    task_pattern: str
    skill_sequence: list[str]  # ordered skill_id list
    support_count: int  # number of successful episodes supporting this motif
    success_rate: float
    avg_token_saving: float  # token saving compared to no-motif baseline


@dataclass
class RoutingBiases:
    """Aggregated routing biases from PolicyStore."""

    direct_skill_delta: float = 0.0
    seeded_delta: float = 0.0


@dataclass
class GeneratorState:
    """v7 third-equation operand: complete mutable state of G_t."""

    # action layer: available skills and their roles
    skill_action_space: dict[str, SkillActionEntry] = field(default_factory=dict)
    # structure layer: validated workflow motifs
    active_motifs: list[WorkflowMotif] = field(default_factory=list)
    # meta-update layer: routing biases and prompt directives
    routing_biases: RoutingBiases = field(default_factory=RoutingBiases)
    prompt_directives: list[str] = field(default_factory=list)
    # version tracking
    version: int = 0
    episode_count: int = 0


# ---------------------------------------------------------------------------
# Role inference helpers
# ---------------------------------------------------------------------------

_ROLE_MAP = {
    "hard_skeleton": "macro_action",
    "soft_hint": "planning_prior",
    "guard_only": "local_structure",
    "disabled": "suppressed",
}


def _safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator > 0 else 0.0


def _infer_role(card) -> str:  # noqa: ANN001  (SkillCard)
    """Map a SkillCard's properties to an action-space role."""
    if card.status in ("suppressed", "pruned"):
        return "suppressed"
    if card.direct_eligible and card.utility >= 0.7 and card.direct_success_count >= 3:
        return "macro_action"
    return _ROLE_MAP.get(card.insertion_mode, "planning_prior")


def _infer_insertion_mode(card) -> str:  # noqa: ANN001
    """Normalise SkillCard.insertion_mode to the v7 vocabulary."""
    if card.status in ("suppressed", "pruned"):
        return "disabled"
    return card.insertion_mode if card.insertion_mode in _ROLE_MAP else "soft_hint"


def _infer_preferred_usage(card) -> str:  # noqa: ANN001
    """Decide direct / seeded / either / none from card statistics."""
    if card.status in ("suppressed", "pruned"):
        return "none"
    direct_rate = _safe_rate(card.direct_success_count, card.direct_use_count)
    seeded_rate = _safe_rate(card.seeded_success_count, card.seeded_use_count)
    has_direct = card.direct_use_count > 0
    has_seeded = card.seeded_use_count > 0
    if has_direct and direct_rate >= 0.5 and has_seeded and seeded_rate >= 0.5:
        return "either"
    if has_direct and direct_rate >= 0.5:
        return "direct"
    if has_seeded and seeded_rate >= 0.3:
        return "seeded"
    # Not enough history yet — default to either so the skill gets tried.
    if not has_direct and not has_seeded:
        return "either"
    return "none"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_generator_state(
    skill_registry,  # noqa: ANN001  memory.skill_registry.SkillRegistry
    template_store,  # noqa: ANN001  memory.template_store.TemplateStore
    policy_store,  # noqa: ANN001  memory.policy_store.PolicyStore
    *,
    benchmark: str = "",
    task_pattern: str = "",
) -> GeneratorState:
    """Reconstruct GeneratorState G_t from the three canonical stores.

    Called once at the start of each episode so the planner / generator
    can operate on a single, consistent snapshot.
    """

    # -- 1. action layer: skill_action_space --------------------------------
    skill_action_space: dict[str, SkillActionEntry] = {}
    for card in skill_registry.list_active():
        entry = SkillActionEntry(
            skill_id=card.skill_id,
            role=_infer_role(card),
            insertion_mode=_infer_insertion_mode(card),
            utility=float(card.utility),
            direct_success_rate=_safe_rate(card.direct_success_count, card.direct_use_count),
            seeded_success_rate=_safe_rate(card.seeded_success_count, card.seeded_use_count),
            preferred_usage=_infer_preferred_usage(card),
        )
        skill_action_space[card.skill_id] = entry

    # -- 2. structure layer: active_motifs ----------------------------------
    active_motifs: list[WorkflowMotif] = []
    if benchmark and task_pattern:
        matched_templates = template_store.match(benchmark, task_pattern, limit=5)
    else:
        matched_templates = [t for t in template_store.list_all() if t.status == "active"]

    for template in matched_templates:
        if not template.preferred_skill_sequence:
            continue
        total = template.support_count or 1
        motif = WorkflowMotif(
            motif_id=template.template_id,
            task_pattern=template.task_pattern,
            skill_sequence=list(template.preferred_skill_sequence),
            support_count=template.support_count,
            success_rate=_safe_rate(template.success_count, total),
            avg_token_saving=0.0,  # placeholder — no per-template token data yet
        )
        active_motifs.append(motif)

    # -- 3. meta-update layer: routing_biases + prompt_directives -----------
    ctx = policy_store.context(benchmark=benchmark, task_pattern=task_pattern)
    routing_biases = RoutingBiases(
        direct_skill_delta=float(ctx.get("direct_skill_delta", 0.0)),
        seeded_delta=float(ctx.get("seeded_delta", 0.0)),
    )
    prompt_directives: list[str] = list(ctx.get("prompt_directives", []))

    return GeneratorState(
        skill_action_space=skill_action_space,
        active_motifs=active_motifs,
        routing_biases=routing_biases,
        prompt_directives=prompt_directives,
        version=int(ctx.get("version", 0)),
        episode_count=0,
    )
