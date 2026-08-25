"""ALFWorld-specific data structures.

Parallel to core/schemas.py but adapted for interactive text-world tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Task types extracted from ALFWorld game file paths
# ---------------------------------------------------------------------------

ALFWORLD_TASK_TYPES = {
    "pick_and_place_simple",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "look_at_obj_in_light",
    "pick_two_obj_and_place",
}

# Canonical short names for reporting
TASK_TYPE_SHORT = {
    "pick_and_place_simple": "pick_place",
    "pick_clean_then_place_in_recep": "clean_place",
    "pick_heat_then_place_in_recep": "heat_place",
    "pick_cool_then_place_in_recep": "cool_place",
    "look_at_obj_in_light": "examine_light",
    "pick_two_obj_and_place": "pick_two_place",
}


# ---------------------------------------------------------------------------
# Single interaction step
# ---------------------------------------------------------------------------

@dataclass
class AlfWorldAction:
    """One interaction step with the environment."""

    step: int
    action: str
    observation: str
    score: float
    done: bool
    admissible_commands: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Task descriptor
# ---------------------------------------------------------------------------

@dataclass
class AlfWorldTask:
    """Description of one ALFWorld task instance.

    Fields are named to shadow BenchmarkTaskInstance / CodeTaskInstance where
    possible, so that environment-agnostic modules (planner, retriever,
    meta_update) can consume them through duck typing.
    """

    task_id: str
    task_type: str  # one of ALFWORLD_TASK_TYPES
    goal: str
    initial_observation: str = ""
    initial_admissible: list[str] = field(default_factory=list)
    game_file: str = ""
    benchmark: str = "alfworld"
    metadata: dict[str, Any] = field(default_factory=dict)

    # --- Duck-typing surface for FlowEvo core modules ---

    @property
    def task_pattern(self) -> str:
        return self.task_type

    @property
    def text(self) -> str:
        return self.goal

    @property
    def prompt(self) -> str:
        return self.goal

    @property
    def query(self) -> str:
        return self.goal

    @property
    def entry_point(self) -> str:
        return self.task_type

    @property
    def family(self) -> str:
        return "alfworld"

    # Fields expected by some FlowEvo code paths (safe defaults)
    canonical_solution: str = ""
    test: str = ""
    test_list: list[str] = field(default_factory=list)
    visible_public_tests: list[str] = field(default_factory=list)
    hidden_table1_tests: list[str] = field(default_factory=list)
    hidden_eval_tests: list[str] = field(default_factory=list)
    test_setup_code: str = ""
    code: str = ""


# ---------------------------------------------------------------------------
# Execution trace
# ---------------------------------------------------------------------------

@dataclass
class AlfWorldTrace:
    """Full execution trace for one ALFWorld task.

    Mirrors core.schemas.ExecutionTrace in spirit but stores action histories
    instead of code histories.
    """

    trace_id: str = ""
    task_id: str = ""
    benchmark: str = "alfworld"
    task_type: str = ""
    task_pattern: str = ""
    goal: str = ""

    # Outcome
    success: bool = False
    first_attempt_success: bool = False
    total_steps: int = 0
    max_steps: int = 50
    final_score: float = 0.0

    # Action history
    actions: list[AlfWorldAction] = field(default_factory=list)
    action_history: list[str] = field(default_factory=list)

    # Planning
    planning_mode: str = ""
    selected_skill_id: str = ""
    skill_seeded: bool = False
    seed_source: str = ""
    selection_reason: str = ""
    episode_provenance: str = ""

    # Skill usage
    used_skills: list[str] = field(default_factory=list)
    skill_hit: bool = False
    fallback_triggered: bool = False
    direct_use_count: int = 0
    direct_success_count: int = 0
    direct_failure_count: int = 0
    seeded_use_count: int = 0
    seeded_success_count: int = 0
    seeded_failure_count: int = 0

    # Token accounting
    llm_prompt_tokens_total: int = 0
    llm_completion_tokens_total: int = 0
    llm_total_tokens_total: int = 0
    llm_call_count: int = 0
    llm_latency_ms_total: float = 0.0

    # Compilation
    compiled_skill_ids: list[str] = field(default_factory=list)
    compiled_skill_admitted: bool = False
    compiled_strategy_id: str = ""

    # Strategy
    injected_strategy_ids: list[str] = field(default_factory=list)
    strategy_guided: bool = False
    strategy_holdout: bool = False
    policy_directives: list[str] = field(default_factory=list)

    # GeneratorState
    generator_state_version: int = 0

    # Governance
    utility_score: float = 0.0
    utility_before: float | None = None
    utility_after: float | None = None
    failure_type: str = ""
    # 与任务失败分离保存的 HTTP/网络/超时错误证据。
    infrastructure_errors: list[dict[str, Any]] = field(default_factory=list)

    # Retrieval diagnostics
    retrieval_scores: list[dict[str, Any]] = field(default_factory=list)
    retrieved_skills: list[str] = field(default_factory=list)

    # Misc
    policy_version: int = 0
    setting_name: str = ""

    # --- Compatibility shims for FlowEvo core modules ---

    @property
    def code_history(self) -> list[str]:
        return self.action_history

    @property
    def retry_count(self) -> int:
        return max(0, self.llm_call_count - 1)

    @property
    def family(self) -> str:
        return "alfworld"

    @property
    def token_cost(self) -> float:
        return float(self.llm_total_tokens_total)

    @property
    def latency_ms(self) -> float:
        return self.llm_latency_ms_total

    # Fields expected by some FlowEvo code paths
    attempts: list[Any] = field(default_factory=list)
    verifier_feedback: list[Any] = field(default_factory=list)
    generation_events: list[Any] = field(default_factory=list)
    matched_template_ids: list[str] = field(default_factory=list)
    matched_primitive_ids: list[str] = field(default_factory=list)
    injected_primitive_ids: list[str] = field(default_factory=list)
    draft_injected_template_ids: list[str] = field(default_factory=list)
    repair_injected_template_ids: list[str] = field(default_factory=list)
    negative_memory_ids: list[str] = field(default_factory=list)
    audit_warnings: list[str] = field(default_factory=list)
    compile_gate_checks: dict[str, bool] = field(default_factory=dict)
    compile_rejection_stage: str = ""
    compile_rejection_reason: str = ""
    planned_usage_subtype: str = ""
    realized_usage_subtype: str = ""
    compact_seeded_prompt_used: bool = False
    scaffold_hint_injected: bool = False
    macro_call_executed: bool = False
    scaffold_function_inserted: bool = False
    guard_checker_emitted: bool = False
    guard_checker_context_injected: bool = False
    guard_checker_feedback_used: bool = False
    scaffold_suppression_reason: str = ""
    repair_seeded_demoted: bool = False
    repair_demotion_reason: str = ""
    runtime_skill_block_reason: str = ""
    artifact_mediated: bool = False
    compiled_primitive_ids: list[str] = field(default_factory=list)
    compiled_primitive_admitted: bool = False
    compiled_skill_status: str = ""
    compiled_skill_provenance: list[str] = field(default_factory=list)
    compiled_skill_admission_bank: str = ""
    compiled_skill_seed_replay_pass_rate: float = 0.0
    compiled_skill_seed_replay_attempt_count: int = 0
    compiled_skill_seed_replay_pass_count: int = 0
    replay_validation_count: int = 0
    replay_failure_count: int = 0
    seed_context_skill_count: int = 0
    seed_context_mode: str = ""
    seed_routing_mode: str = ""
    seed_context_char_count: int = 0
    seed_context_contains_code: bool = False
    seed_context_contains_tests: bool = False
    transfer_utility_score: float = 0.0
    estimated_seed_token_cost: float = 0.0
    historical_positive_transfer: float = 0.0
    negative_transfer_risk: float = 0.0
    planner_compatibility_score: float = 0.0
    planner_compatibility_veto_reasons: list[str] = field(default_factory=list)
    planner_soft_hint_reasons: list[str] = field(default_factory=list)
    positive_transfer_count: int = 0
    negative_transfer_count: int = 0


# ---------------------------------------------------------------------------
# ALFWorld skill template (executable artifact)
# ---------------------------------------------------------------------------

@dataclass
class AlfWorldSkillTemplate:
    """Parameterised action sequence extracted from a successful trace.

    This is the ALFWorld analogue of a compiled Python skill function.
    """

    template_id: str
    task_type: str

    # Parameterised action sequence
    action_template: list[str]  # e.g. ["go to {object_location}", "take {object} from {object_location}", ...]
    parameters: list[str]  # parameter names: ["object", "object_location", "target_location"]

    # Concrete example from source trace
    example_bindings: dict[str, str] = field(default_factory=dict)

    # Provenance
    source_task_id: str = ""
    source_trace_id: str = ""

    # Statistics
    success_count: int = 0
    failure_count: int = 0

    def instantiate(self, bindings: dict[str, str]) -> list[str]:
        """Fill template parameters with concrete values."""
        actions: list[str] = []
        for step in self.action_template:
            action = step
            for param, value in bindings.items():
                action = action.replace("{%s}" % param, value)
            actions.append(action)
        return actions

    @property
    def template_length(self) -> int:
        return len(self.action_template)

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "task_type": self.task_type,
            "action_template": self.action_template,
            "parameters": self.parameters,
            "example_bindings": self.example_bindings,
            "source_task_id": self.source_task_id,
            "source_trace_id": self.source_trace_id,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlfWorldSkillTemplate":
        return cls(
            template_id=str(data.get("template_id", "")),
            task_type=str(data.get("task_type", "")),
            action_template=list(data.get("action_template", [])),
            parameters=list(data.get("parameters", [])),
            example_bindings=dict(data.get("example_bindings", {})),
            source_task_id=str(data.get("source_task_id", "")),
            source_trace_id=str(data.get("source_trace_id", "")),
            success_count=int(data.get("success_count", 0)),
            failure_count=int(data.get("failure_count", 0)),
        )


# ---------------------------------------------------------------------------
# Layer 2: Trajectory exemplar (denoised successful trace)
# ---------------------------------------------------------------------------

@dataclass
class AlfWorldExemplar:
    """Layer 2 skill: procedural guideline extracted from a successful trace.

    Previously stored raw denoised trajectories (action sequences).
    Now stores abstract procedural rules extracted by LLM from the trace.
    The class name is kept to minimise cross-file diff; semantically it
    is now a "procedural guideline" rather than a "trajectory exemplar".
    """

    exemplar_id: str
    task_type: str
    goal: str
    rules: list[str]           # 3-5 procedural rules (abstract, no entity names)
    source_task_id: str = ""
    source_trace_id: str = ""
    extraction_tokens: int = 0  # LLM tokens used for extraction (compilation cost)

    def to_prompt_text(self) -> str:
        """Format as an injectable guideline for the generator prompt."""
        lines = ["Procedural guidelines for '%s' tasks:" % self.task_type]
        for i, rule in enumerate(self.rules, 1):
            lines.append("  %d. %s" % (i, rule))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "exemplar_id": self.exemplar_id,
            "task_type": self.task_type,
            "goal": self.goal,
            "rules": list(self.rules),
            "source_task_id": self.source_task_id,
            "source_trace_id": self.source_trace_id,
            "extraction_tokens": self.extraction_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlfWorldExemplar":
        # Backwards compat: old checkpoints stored `effective_actions` instead
        # of `rules`.  Fall back to them so we can load v0/v1/v2/v3 snapshots.
        rules = data.get("rules")
        if not rules:
            rules = data.get("effective_actions", [])
        return cls(
            exemplar_id=str(data.get("exemplar_id", "")),
            task_type=str(data.get("task_type", "")),
            goal=str(data.get("goal", "")),
            rules=list(rules),
            source_task_id=str(data.get("source_task_id", "")),
            source_trace_id=str(data.get("source_trace_id", "")),
            extraction_tokens=int(data.get("extraction_tokens", 0)),
        )


# ---------------------------------------------------------------------------
# Layer 3: Structural insight (cross-trace statistical knowledge)
# ---------------------------------------------------------------------------

@dataclass
class AlfWorldInsight:
    """Layer 3 skill: aggregated knowledge inducted from multiple traces.

    Used as a policy directive when neither a template nor a recent exemplar
    is available for the current task type.
    """

    task_type: str
    search_priority: str = ""
    common_objects: list[str] = field(default_factory=list)
    common_locations: list[str] = field(default_factory=list)
    common_pitfalls: list[str] = field(default_factory=list)
    environment_facts: list[str] = field(default_factory=list)
    sample_count: int = 0

    def to_prompt_text(self) -> str:
        """Format as a policy directive line block."""
        parts: list[str] = []
        if self.search_priority:
            parts.append("Search priority: %s" % self.search_priority)
        if self.common_locations:
            parts.append("Objects are often found in: %s"
                         % ", ".join(self.common_locations[:5]))
        if self.common_pitfalls:
            parts.append("Avoid: %s" % "; ".join(self.common_pitfalls[:3]))
        if self.environment_facts:
            parts.append("Known facts: %s" % "; ".join(self.environment_facts[:3]))
        if not parts:
            return ""
        return "Guidance from %d past experiences:\n%s" % (
            self.sample_count, "\n".join("- " + p for p in parts),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "search_priority": self.search_priority,
            "common_objects": list(self.common_objects),
            "common_locations": list(self.common_locations),
            "common_pitfalls": list(self.common_pitfalls),
            "environment_facts": list(self.environment_facts),
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlfWorldInsight":
        return cls(
            task_type=str(data.get("task_type", "")),
            search_priority=str(data.get("search_priority", "")),
            common_objects=list(data.get("common_objects", [])),
            common_locations=list(data.get("common_locations", [])),
            common_pitfalls=list(data.get("common_pitfalls", [])),
            environment_facts=list(data.get("environment_facts", [])),
            sample_count=int(data.get("sample_count", 0)),
        )
