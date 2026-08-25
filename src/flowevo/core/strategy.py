"""v7 mainline 2: StrategyCard — strategy-level artifact compiled from skills.

A StrategyCard is a first-class citizen parallel to SkillCard.  It carries a
natural-language strategy description rather than executable code, enabling
the skill bank's "effective coverage" to expand from exact-match to
same-task-pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StrategyCard:
    """Strategy description extracted from a successfully compiled skill."""

    strategy_id: str
    source_skill_id: str
    task_pattern: str
    description: str
    applicability: str
    approach_summary: str

    # Lifecycle
    utility: float = 1.0
    status: str = "active"  # "active" | "suppressed" | "retired"

    # ContrastiveEval statistics
    guided_use_count: int = 0
    guided_success_count: int = 0
    unguided_use_count: int = 0
    unguided_success_count: int = 0

    # Provenance
    source_benchmark: str = ""
    source_task_id: str = ""
    created_at_episode: int = 0

    @property
    def guided_success_rate(self) -> float:
        if self.guided_use_count == 0:
            return 0.0
        return round(self.guided_success_count / self.guided_use_count, 4)

    @property
    def unguided_success_rate(self) -> float:
        if self.unguided_use_count == 0:
            return 0.0
        return round(self.unguided_success_count / self.unguided_use_count, 4)

    @property
    def contrastive_delta(self) -> float:
        """guided_rate - unguided_rate.  Positive = helpful, negative = harmful."""
        if self.guided_use_count < 3 or self.unguided_use_count < 3:
            return 0.0
        return round(self.guided_success_rate - self.unguided_success_rate, 4)

    @property
    def total_support(self) -> int:
        return self.guided_use_count + self.unguided_use_count

    def to_injection_text(self) -> str:
        """Compact text for prompt injection (~100 tokens)."""
        parts = [
            "For tasks matching pattern '%s':" % self.task_pattern,
            "  Approach: %s" % self.approach_summary,
            "  Details: %s" % self.description,
        ]
        if self.guided_use_count >= 3:
            parts.append(
                "  (Historical success: %d/%d when applied to similar tasks)"
                % (self.guided_success_count, self.guided_use_count)
            )
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "source_skill_id": self.source_skill_id,
            "task_pattern": self.task_pattern,
            "description": self.description,
            "applicability": self.applicability,
            "approach_summary": self.approach_summary,
            "utility": self.utility,
            "status": self.status,
            "guided_use_count": self.guided_use_count,
            "guided_success_count": self.guided_success_count,
            "unguided_use_count": self.unguided_use_count,
            "unguided_success_count": self.unguided_success_count,
            "source_benchmark": self.source_benchmark,
            "source_task_id": self.source_task_id,
            "created_at_episode": self.created_at_episode,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyCard":
        return cls(
            strategy_id=str(data.get("strategy_id", "")),
            source_skill_id=str(data.get("source_skill_id", "")),
            task_pattern=str(data.get("task_pattern", "")),
            description=str(data.get("description", "")),
            applicability=str(data.get("applicability", "")),
            approach_summary=str(data.get("approach_summary", "")),
            utility=float(data.get("utility", 1.0)),
            status=str(data.get("status", "active")),
            guided_use_count=int(data.get("guided_use_count", 0)),
            guided_success_count=int(data.get("guided_success_count", 0)),
            unguided_use_count=int(data.get("unguided_use_count", 0)),
            unguided_success_count=int(data.get("unguided_success_count", 0)),
            source_benchmark=str(data.get("source_benchmark", "")),
            source_task_id=str(data.get("source_task_id", "")),
            created_at_episode=int(data.get("created_at_episode", 0)),
        )
