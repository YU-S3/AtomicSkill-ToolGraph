from __future__ import annotations

from typing import Any, Optional, Union

from . import pydantic_compat  # noqa: F401

from pydantic import BaseModel, Field

from .types import SkillStatus, StepType


class TaskInstance(BaseModel):
    task_id: str
    family: str
    query: str
    table_paths: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    ground_truth: Optional[Union[dict[str, Any], str]] = None


class CodeTaskInstance(BaseModel):
    task_id: str
    benchmark: str
    prompt: str = ""
    canonical_solution: str = ""
    test: str = ""
    entry_point: Optional[str] = None
    text: str = ""
    code: str = ""
    test_list: list[str] = Field(default_factory=list)
    test_setup_code: str = ""
    visible_public_tests: list[str] = Field(default_factory=list)
    hidden_table1_tests: list[str] = Field(default_factory=list)
    hidden_stress_tests: list[str] = Field(default_factory=list)
    table1_hidden_tests: list[str] = Field(default_factory=list)
    stress_plus_challenge_hidden_tests: list[str] = Field(default_factory=list)
    hidden_eval_tests: list[str] = Field(default_factory=list)
    challenge_test_list: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


BenchmarkTaskInstance = CodeTaskInstance


class VerifierFeedback(BaseModel):
    benchmark: str
    task_id: str
    passed: bool
    dataset_split: str = ""
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    timeout: bool = False
    executed_code: str = ""
    failed_tests: list[str] = Field(default_factory=list)
    summary: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)


class ActionRecord(BaseModel):
    tool_name: str
    tool_input_summary: str
    tool_output_summary: str


class AttemptRecord(BaseModel):
    attempt_index: int
    parent_attempt: Optional[int] = None
    stage: str
    candidate_code: str
    verifier_feedback: Optional[VerifierFeedback] = None
    failure_type: str = ""
    repair_source: str = ""
    notes: str = ""


class GenerationEvent(BaseModel):
    generation_stage: str
    provider: str = ""
    model: str = ""
    conditioned_skill_ids: list[str] = Field(default_factory=list)
    planned_skill_nodes: list[dict[str, Any]] = Field(default_factory=list)
    planned_skill_node_count: int = 0
    source_structural_role: str = ""
    scaffold_hint_injected: bool = False
    guard_checker_emitted: bool = False
    guard_checker_context_injected: bool = False
    guard_checker_feedback_used: bool = False
    scaffold_suppression_reason: str = ""
    repair_seeded_demoted: bool = False
    repair_demotion_reason: str = ""
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
    compact_seeded_prompt_used: bool = False
    task_context_tokens: int = 0
    tests_context_tokens: int = 0
    feedback_context_tokens: int = 0
    previous_code_context_tokens: int = 0
    seed_summary_tokens: int = 0
    seed_context_tokens: int = 0
    scaffold_tokens: int = 0
    repair_strategy: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    error_summary: str = ""


class WorkflowStep(BaseModel):
    step_id: str
    step_type: StepType
    payload: dict[str, Any] = Field(default_factory=dict)


class WorkflowPlan(BaseModel):
    plan_id: str
    task_id: str
    retrieved_skills: list[str] = Field(default_factory=list)
    planning_mode: str = "pure_dynamic"
    task_pattern: str = ""
    selected_skill_id: str = ""
    seed_context_skill_ids: list[str] = Field(default_factory=list)
    seed_context_skill_modes: dict[str, str] = Field(default_factory=dict)
    planned_usage_subtype: str = ""
    seed_routing_mode: str = ""
    transfer_utility_score: float = 0.0
    estimated_seed_token_cost: float = 0.0
    historical_positive_transfer: float = 0.0
    negative_transfer_risk: float = 0.0
    planner_compatibility_score: float = 0.0
    planner_compatibility_veto_reasons: list[str] = Field(default_factory=list)
    planner_soft_hint_reasons: list[str] = Field(default_factory=list)
    skill_seeded: bool = False
    soft_aware_enabled: bool = False
    fallback_planning_mode: str = ""
    bifurcated_skill_router_enabled: bool = False
    bifurcated_exact_or_seeded_enabled: bool = False
    selective_integrated_enabled: bool = False
    seeded_disable_local_insertion: bool = False
    exact_direct_postcall_wrapper_enabled: bool = False
    direct_postcall_max_local_patch: int = 0
    seed_source: str = ""
    selection_reason: str = ""
    policy_version: int = 0
    policy_directives: list[str] = Field(default_factory=list)
    matched_workflow_memory_ids: list[str] = Field(default_factory=list)
    matched_template_ids: list[str] = Field(default_factory=list)
    matched_primitive_ids: list[str] = Field(default_factory=list)
    planner_visible_candidates: list[str] = Field(default_factory=list)
    planner_rejection_reasons: dict[str, list[str]] = Field(default_factory=dict)
    steps: list[WorkflowStep] = Field(default_factory=list)


class ExecutionTrace(BaseModel):
    trace_id: str
    task_id: str
    family: str
    query: str
    benchmark: str = ""
    setting_name: str = ""
    phase_a_bank_source: str = ""
    phase_b_usage_setting: str = ""
    task_order_seed: Optional[int] = None
    phase_a_seed: Optional[int] = None
    retrieved_skills: list[str] = Field(default_factory=list)
    retrieval_scores: list[dict[str, Any]] = Field(default_factory=list)
    planning_mode: str = "pure_dynamic"
    task_pattern: str = ""
    episode_provenance: str = ""
    selected_skill_id: str = ""
    compiled_skill_admission_bank: str = ""
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
    planner_compatibility_veto_reasons: list[str] = Field(default_factory=list)
    planner_soft_hint_reasons: list[str] = Field(default_factory=list)
    compact_seeded_prompt_used: bool = False
    planned_usage_subtype: str = ""
    realized_usage_subtype: str = ""
    realization_evidence_source: str = ""
    subtype_realization_gap_reason: str = ""
    planned_skill_nodes: list[dict[str, Any]] = Field(default_factory=list)
    realized_skill_nodes: list[dict[str, Any]] = Field(default_factory=list)
    planned_skill_node_count: int = 0
    realized_skill_node_count: int = 0
    structured_realization_rate: float = 0.0
    planned_realized_gap_count: int = 0
    planned_realized_gap_rate: float = 0.0
    macro_call_executed: bool = False
    scaffold_hint_injected: bool = False
    scaffold_function_inserted: bool = False
    guard_checker_emitted: bool = False
    guard_checker_context_injected: bool = False
    guard_checker_feedback_used: bool = False
    scaffold_suppression_reason: str = ""
    repair_seeded_demoted: bool = False
    repair_demotion_reason: str = ""
    active_skill_bank_at_episode_start: list[dict[str, Any]] = Field(default_factory=list)
    recall_candidates: list[dict[str, Any]] = Field(default_factory=list)
    hard_filter_survivors: list[dict[str, Any]] = Field(default_factory=list)
    hard_filter_drop_reasons: dict[str, list[str]] = Field(default_factory=dict)
    rerank_score_decomposition: dict[str, dict[str, Any]] = Field(default_factory=dict)
    planner_visible_candidates: list[str] = Field(default_factory=list)
    planner_rejection_reasons: dict[str, list[str]] = Field(default_factory=dict)
    causal_stage_outcome: str = ""
    unlock_pass: str = ""
    compile_unlock_failed: bool = False
    pass_b_frozen_bank: bool = False
    skill_seeded: bool = False
    seed_source: str = ""
    selection_reason: str = ""
    runtime_skill_block_reason: str = ""
    utility_before: Optional[float] = None
    utility_after: Optional[float] = None
    plan: Optional[WorkflowPlan] = None
    actions: list[ActionRecord] = Field(default_factory=list)
    attempts: list[AttemptRecord] = Field(default_factory=list)
    generation_events: list[GenerationEvent] = Field(default_factory=list)
    code_history: list[str] = Field(default_factory=list)
    verifier_feedback: list[VerifierFeedback] = Field(default_factory=list)
    final_result: dict[str, Any] = Field(default_factory=dict)
    execution_outputs: list[dict[str, Any]] = Field(default_factory=list)
    success: bool = False
    token_cost: float = 0.0
    latency_ms: float = 0.0
    artifact_execution_latency_ms: float = 0.0
    artifact_import_latency_ms: float = 0.0
    artifact_verification_latency_ms: float = 0.0
    llm_prompt_tokens_total: int = 0
    llm_completion_tokens_total: int = 0
    llm_total_tokens_total: int = 0
    llm_latency_ms_total: float = 0.0
    llm_call_count: int = 0
    compiled_skill_ids: list[str] = Field(default_factory=list)
    compiled_skill_provenance: list[str] = Field(default_factory=list)
    compiled_skill_admitted: bool = False
    compiled_skill_status: str = ""
    compile_gate_checks: dict[str, bool] = Field(default_factory=dict)
    compile_rejection_stage: str = ""
    compile_rejection_reason: str = ""
    compiled_skill_seed_replay_pass_rate: float = 0.0
    compiled_skill_seed_replay_attempt_count: int = 0
    compiled_skill_seed_replay_pass_count: int = 0
    policy_version: int = 0
    policy_update_summary: str = ""
    matched_workflow_memory_ids: list[str] = Field(default_factory=list)
    injected_workflow_memory_ids: list[str] = Field(default_factory=list)
    updated_workflow_memory_ids: list[str] = Field(default_factory=list)
    matched_template_ids: list[str] = Field(default_factory=list)
    draft_injected_template_ids: list[str] = Field(default_factory=list)
    repair_injected_template_ids: list[str] = Field(default_factory=list)
    updated_template_ids: list[str] = Field(default_factory=list)
    updated_template_kinds: list[str] = Field(default_factory=list)
    template_update_summary: str = ""
    matched_primitive_ids: list[str] = Field(default_factory=list)
    injected_primitive_ids: list[str] = Field(default_factory=list)
    compiled_primitive_ids: list[str] = Field(default_factory=list)
    compiled_primitive_admitted: bool = False
    primitive_update_summary: str = ""
    utility_score: float = 0.0
    audit_warnings: list[str] = Field(default_factory=list)
    negative_memory_ids: list[str] = Field(default_factory=list)
    failure_type: str = ""
    used_skills: list[str] = Field(default_factory=list)
    skill_hit: bool = False
    fallback_triggered: bool = False
    first_attempt_success: bool = False
    retry_count: int = 0
    operational_event_count: int = 0
    skill_mediated_event_count: int = 0
    skill_mediated_event_rate: float = 0.0
    draft_prompt_tokens: int = 0
    task_context_tokens: int = 0
    tests_context_tokens: int = 0
    feedback_context_tokens: int = 0
    previous_code_context_tokens: int = 0
    seed_summary_tokens: int = 0
    seed_context_tokens: int = 0
    scaffold_tokens: int = 0
    repair_prompt_tokens: int = 0
    num_repairs: int = 0
    repair_round_count: int = 0
    full_rewrite_repair_count: int = 0
    local_patch_repair_count: int = 0
    repair_failure_types: list[str] = Field(default_factory=list)
    repair_heavy: bool = False
    artifact_mediated: bool = False
    skill_node_insertion_count: int = 0
    direct_use_count: int = 0
    direct_success_count: int = 0
    direct_failure_count: int = 0
    seeded_use_count: int = 0
    seeded_success_count: int = 0
    seeded_failure_count: int = 0
    replay_validation_count: int = 0
    replay_failure_count: int = 0


class SkillFailureEvent(BaseModel):
    task_id: str
    benchmark: str
    planning_mode: str
    verifier_message: str = ""
    failure_type: str = ""
    timestamp: str


class SkillCard(BaseModel):
    skill_id: str
    name: str
    description: str
    family: str
    signature: dict[str, Any] = Field(default_factory=dict)
    preconditions: dict[str, Any] = Field(default_factory=dict)
    direct_eligible: bool = False
    preferred_usage_subtypes: list[str] = Field(default_factory=list)
    direct_reuse_evidence_count: int = 0
    code_path: str = ""
    tests: list[dict[str, Any]] = Field(default_factory=list)
    utility: float = 0.0
    call_count: int = 0
    success_count: int = 0
    consecutive_failures: int = 0
    last_used_at: str = ""
    last_audit_at: str = ""
    audit_pass_count: int = 0
    audit_fail_count: int = 0
    failure_log: list[SkillFailureEvent] = Field(default_factory=list)
    direct_use_count: int = 0
    direct_success_count: int = 0
    direct_failure_count: int = 0
    seeded_use_count: int = 0
    seeded_success_count: int = 0
    seeded_failure_count: int = 0
    positive_transfer_count: int = 0
    negative_transfer_count: int = 0
    avg_delta_first_attempt: float = 0.0
    avg_delta_retry_count: float = 0.0
    avg_delta_tokens_per_solved: float = 0.0
    last_used_routing_mode: str = ""
    role: str = "solver"
    task_family: str = ""
    environment_affordance: list[str] = Field(default_factory=list)
    insertion_mode: str = "soft_hint"
    negative_transfer_risk_label: str = "low"
    skill_family: str = ""
    replay_validation_count: int = 0
    replay_failure_count: int = 0
    lineage: list[str] = Field(default_factory=list)
    provenance: str = "dynamic_success"
    compile_metadata: dict[str, Any] = Field(default_factory=dict)
    status: SkillStatus = "draft"


class PolicyValue(BaseModel):
    direct_skill_delta: float = 0.0
    seeded_delta: float = 0.0
    prompt_directives: list[str] = Field(default_factory=list)
    direct_skill_delta_by_subtype: dict[str, float] = Field(default_factory=dict)
    seeded_delta_by_subtype: dict[str, float] = Field(default_factory=dict)
    prompt_directives_by_subtype: dict[str, list[str]] = Field(default_factory=dict)
    updated_at: str = ""


class SkillPolicyValue(BaseModel):
    direct_skill_delta: float = 0.0
    seeded_delta: float = 0.0
    prompt_directives: list[str] = Field(default_factory=list)
    direct_skill_delta_by_subtype: dict[str, float] = Field(default_factory=dict)
    seeded_delta_by_subtype: dict[str, float] = Field(default_factory=dict)
    prompt_directives_by_subtype: dict[str, list[str]] = Field(default_factory=dict)
    suppressed: bool = False
    updated_at: str = ""


class PolicyState(BaseModel):
    version: int = 0
    benchmark_policies: dict[str, PolicyValue] = Field(default_factory=dict)
    task_pattern_policies: dict[str, PolicyValue] = Field(default_factory=dict)
    skill_policies: dict[str, SkillPolicyValue] = Field(default_factory=dict)
    last_updated_at: str = ""


class Template(BaseModel):
    template_id: str
    benchmark: str = ""
    family: str = ""
    task_pattern: str = ""
    template_kind: str = ""
    description: str
    trigger_signature: dict[str, Any] = Field(default_factory=dict)
    prompt_directives: list[str] = Field(default_factory=list)
    preferred_skill_ids: list[str] = Field(default_factory=list)
    preferred_skill_sequence: list[str] = Field(default_factory=list)
    support_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    exemplar_trace_ids: list[str] = Field(default_factory=list)
    status: str = "draft"
    priority: int = 0
    updated_at: str = ""


class WorkflowMemoryCard(BaseModel):
    memory_id: str
    benchmark: str
    task_pattern: str
    workflow_summary: str
    repair_summary: str = ""
    applicability_notes: str = ""
    support_count: int = 0
    updated_at: str = ""


class PrimitiveCard(BaseModel):
    primitive_id: str
    name: str
    description: str
    benchmark: str
    task_pattern: str
    helper_name: str
    target_entry_point: str
    signature: dict[str, Any] = Field(default_factory=dict)
    code_path: str = ""
    source_trace_ids: list[str] = Field(default_factory=list)
    source_skill_ids: list[str] = Field(default_factory=list)
    support_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    utility: float = 0.0
    last_used_at: str = ""
    updated_at: str = ""
    provenance: str = "compiled_primitive"
    status: str = "draft"


class NegativeMemoryEntry(BaseModel):
    entry_id: str
    artifact_id: str
    artifact_kind: str
    benchmark: str
    lifecycle_phase: str = ""
    task_pattern: str = ""
    planning_mode: str = ""
    usage_subtype: str = ""
    failure_type: str = ""
    source: str = ""
    summary: str = ""
    trace_id: str = ""
    failed_family: str = ""
    failed_api_pattern: str = ""
    misroute_source_skill: str = ""
    failure_after_seeded_use: bool = False
    severity: float = 0.0
    timestamp: str
