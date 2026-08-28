from types import SimpleNamespace

from atomic_skillgraph.core.edge_ir import GraphEdge
from atomic_skillgraph.core.refs import SkillRef, ToolRef
from atomic_skillgraph.core.skill_ir import (
    AbstractAtomicSkill,
    CompositeSkill,
    ImplementationAtom,
    ToolBinding,
)
from atomic_skillgraph.core.status import EdgeType, SkillStatus
from atomic_skillgraph.core.trace_ir import (
    NodeValidationResult,
    RuntimeSpan,
    TaskGapAnalysis,
    TraceRecord,
)
from atomic_skillgraph.evolution.branch_repair import FailureBranchManager
from atomic_skillgraph.evolution.composite_builder import CompositeBuilder
from atomic_skillgraph.evolution.composite_lifecycle import (
    evaluate_composite,
    reevaluate_waiting_composites,
)
from atomic_skillgraph.evolution.trace_graph_reconstructor import (
    TraceGraphReconstructor,
)
from atomic_skillgraph.evolution.success_processor import _revision_build_inputs
from atomic_skillgraph.graph.aligner import align_composite
from atomic_skillgraph.graph.registry import SkillGraphRegistry
from atomic_skillgraph.validation.composite_validator import (
    CompositeFailureCode,
    CompositeValidator,
)
from atomic_skillgraph.core.config import SystemConfig
from atomic_skillgraph.system import _task_gap_is_strong_proof


def _atomic(name: str, status=SkillStatus.ACTIVE, *, utility=0.5):
    return AbstractAtomicSkill(
        ref=SkillRef(name, "1.0.0"), summary=name,
        effects=[{"predicate": f"{name}.done", "args": {}}],
        metadata={"statistics": {"support_count": 2, "utility": utility}},
        status=status,
    )


def _composite(ref: str, nodes, *, status=SkillStatus.DRAFT, support=2):
    steps = [{"step_id": f"s{index}", "node_ref": str(node.ref), "params": {}}
             for index, node in enumerate(nodes)]
    control = [GraphEdge(
        source=str(nodes[index].ref), target=str(nodes[index + 1].ref),
        type=EdgeType.NEXT, scope="composite",
        source_step=steps[index]["step_id"],
        target_step=steps[index + 1]["step_id"],
    ).to_dict() for index in range(len(nodes) - 1)]
    return CompositeSkill(
        ref=SkillRef(ref, "1.0.0"),
        summary=ref,
        graph={"nodes": [str(node.ref) for node in nodes],
               "steps": steps, "control": control, "data": []},
        validator={}, status=status,
        metadata={"statistics": {"support_count": support, "utility": 0.5}},
    )


def test_structured_codes_cover_exact_occurrence_and_data_flow():
    left, right = _atomic("cap.left"), _atomic("cap.right")
    composite = _composite("composite.left-right", [left, right])
    composite.graph["data"] = [GraphEdge(
        source=str(left.ref), target=str(right.ref), type=EdgeType.DATA_FLOW,
        scope="composite", source_step="s0", target_step="s1",
        mapping={"source_output": "object", "target_input": "object"},
    ).to_dict()]
    results = [
        NodeValidationResult(node_ref=str(right.ref), passed=True,
                             step_id="runtime_b", occurrence_id="occ_b"),
        NodeValidationResult(node_ref=str(left.ref), passed=True,
                             step_id="runtime_a", occurrence_id="occ_a"),
    ]
    result = CompositeValidator().validate_composite(
        composite, results, {"facts": []}, context={
            "realized_nodes": [
                {"occurrence_id": "occ_b", "origin_step_id": "s1",
                 "ref": str(right.ref)},
                {"occurrence_id": "occ_a", "origin_step_id": "s0",
                 "ref": str(left.ref)},
            ],
            "runtime_edges": [],
            "require_realized_data_flow": True,
        })
    assert result.passed is False
    assert CompositeFailureCode.CONTROL_COVERAGE_FAILED.value in result.failure_codes
    assert CompositeFailureCode.DATA_FLOW_FAILED.value in result.failure_codes


def test_data_flow_requires_materialized_value_and_exact_provenance():
    left, right = _atomic("cap.flow-left"), _atomic("cap.flow-right")
    composite = _composite("composite.real-flow", [left, right])
    flow = GraphEdge(
        source=str(left.ref), target=str(right.ref), type=EdgeType.DATA_FLOW,
        scope="runtime", source_step="runtime_0", target_step="runtime_1",
        mapping={"source_output": "object", "target_input": "object",
                 "transform": "identity"},
    )
    composite.graph["data"] = [GraphEdge(
        source=str(left.ref), target=str(right.ref), type=EdgeType.DATA_FLOW,
        scope="composite", source_step="s0", target_step="s1",
        mapping=dict(flow.mapping),
    ).to_dict()]
    results = [
        NodeValidationResult(node_ref=str(left.ref), passed=True,
                             step_id="runtime_0", occurrence_id="occ_0"),
        NodeValidationResult(node_ref=str(right.ref), passed=True,
                             step_id="runtime_1", occurrence_id="occ_1"),
    ]
    nodes = [
        {"step_id": "runtime_0", "occurrence_id": "occ_0",
         "origin_step_id": "s0", "ref": str(left.ref), "passed": True,
         "outputs": {}},
        {"step_id": "runtime_1", "occurrence_id": "occ_1",
         "origin_step_id": "s1", "ref": str(right.ref), "passed": True,
         "params": {"object": "item_1"},
         "binding_provenance": {"object": {"source": "task"}}},
    ]
    validator = CompositeValidator()
    failed = validator.validate_composite(
        composite, results, {}, context={
            "realized_nodes": nodes,
            "runtime_edges": [flow.to_dict()],
            "require_realized_data_flow": True,
        })
    assert CompositeFailureCode.DATA_FLOW_FAILED.value in failed.failure_codes
    assert any("source_output_not_materialized" in message
               for message in failed.messages)

    nodes[0]["outputs"] = {"object": "item_1"}
    nodes[1]["binding_provenance"] = {"object": {
        "source": "data_flow", "source_step": "runtime_0",
        "source_output": "object"}}
    passed = validator.validate_composite(
        composite, results, {}, context={
            "realized_nodes": nodes,
            "runtime_edges": [flow.to_dict()],
            "require_realized_data_flow": True,
        })
    assert CompositeFailureCode.DATA_FLOW_FAILED.value not in passed.failure_codes


def test_draft_child_blocks_without_structural_shadow(workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "draft_child")
    draft, active = _atomic("cap.draft", SkillStatus.DRAFT), _atomic("cap.active")
    registry.register(draft)
    registry.register(active)
    composite = _composite("composite.draft-active", [draft, active])
    decision = evaluate_composite(composite, registry, min_support=2)
    assert decision.status == SkillStatus.DRAFT
    assert decision.reason == "awaiting_child_activation"
    registry.set_status(draft.ref, SkillStatus.ACTIVE)
    promoted = evaluate_composite(composite, registry, min_support=2)
    assert promoted.status == SkillStatus.ACTIVE


def test_lifecycle_promotion_adds_supersedes_without_later_rebuild(
        workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "lifecycle_supersedes")
    old_left, old_right = _atomic("cap.old-left"), _atomic("cap.old-right")
    draft_child = _atomic("cap.revised-child", SkillStatus.DRAFT)
    for atomic in (old_left, old_right, draft_child):
        registry.register(atomic)
    parent = _composite(
        "composite.parent", [old_left, old_right],
        status=SkillStatus.ACTIVE)
    registry.register(parent)
    revised = _composite(
        "composite.revised", [old_left, draft_child],
        status=SkillStatus.DRAFT, support=2)
    revised.metadata["derived_from_refs"] = [str(parent.ref)]
    registry.register(revised)

    registry.set_status(draft_child.ref, SkillStatus.ACTIVE)
    events = reevaluate_waiting_composites(registry, min_support=2)
    assert any(item["composite_ref"] == str(revised.ref)
               and item["to"] == SkillStatus.ACTIVE.value for item in events)
    supersedes = [edge for edge in registry.edge_objects(EdgeType.SUPERSEDES)
                  if edge.source == str(revised.ref)]
    assert [edge.target for edge in supersedes] == [str(parent.ref)]


def test_composite_alignment_preserves_occurrence_order(workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "ordered_alignment")
    left, right = _atomic("cap.left"), _atomic("cap.right")
    registry.register(left)
    registry.register(right)
    forward = _composite(
        "composite.forward", [left, right], status=SkillStatus.ACTIVE)
    registry.register(forward)
    backward = _composite("composite.backward", [right, left])
    assert align_composite(backward, registry).matched is False


def test_composite_alignment_includes_control_dag_identity(workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "control_alignment")
    first, second, third = (
        _atomic("cap.first"), _atomic("cap.second"), _atomic("cap.third"))
    for atomic in (first, second, third):
        registry.register(atomic)
    linear = _composite(
        "composite.linear", [first, second, third],
        status=SkillStatus.ACTIVE)
    registry.register(linear)
    forked = _composite("composite.forked", [first, second, third])
    forked.graph["control"] = [
        GraphEdge(
            source=str(first.ref), target=str(second.ref),
            type=EdgeType.NEXT, scope="composite",
            source_step="s0", target_step="s1").to_dict(),
        GraphEdge(
            source=str(first.ref), target=str(third.ref),
            type=EdgeType.NEXT, scope="composite",
            source_step="s0", target_step="s2").to_dict(),
    ]
    assert align_composite(forked, registry).matched is False


def test_suppressed_recommendation_falls_back_to_best_active(workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "fallback")
    old = _atomic("cap.versioned", utility=0.9)
    registry.register(old)
    newer = _atomic("cap.versioned", utility=0.4)
    newer.ref = SkillRef("cap.versioned", "1.1.0")
    registry.register(newer)
    assert registry.get_recommended("cap.versioned").ref == newer.ref
    registry.set_status(newer.ref, SkillStatus.SUPPRESSED)
    assert registry.get_recommended("cap.versioned").ref == old.ref


def test_task_gap_revision_derives_new_graph_and_suppresses_exact_parent(
        workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "revision")
    first, second, inserted = (
        _atomic("cap.first"), _atomic("cap.second"), _atomic("cap.inserted"))
    for atomic in (first, second, inserted):
        registry.register(atomic)
    old = _composite(
        "composite.old", [first, second], status=SkillStatus.ACTIVE)
    registry.register(old)
    trace = TraceRecord(
        trace_id="trace_gap_revision", success=True,
        selected_composite=str(old.ref),
        provenance={"target_effects": list(inserted.effects)},
        task_gap_analysis=TaskGapAnalysis(
            missing_effects=list(inserted.effects)),
        runtime_spans=[
            RuntimeSpan(kind="planned_node", occurrence_id="p0",
                        action_start=0, action_end=1),
            RuntimeSpan(kind="planned_node", occurrence_id="p1",
                        action_start=1, action_end=2),
            RuntimeSpan(kind="task_gap", occurrence_id="gap0",
                        action_start=2, action_end=3,
                        missing_effects=list(inserted.effects)),
        ],
        realized_atomic_nodes=[
            {"ref": str(first.ref), "occurrence_id": "p0", "passed": True},
            {"ref": str(second.ref), "occurrence_id": "p1", "passed": True},
            {"ref": "skill://runtime.dynamic.task_gap@0.0.0",
             "occurrence_id": "gap0", "passed": True},
        ],
        node_validators=[NodeValidationResult(
            node_ref="skill://runtime.dynamic.task_gap@0.0.0",
            level="task_gap", passed=True, occurrence_id="gap0")],
    )
    segments = [
        {"phase_id": "p0", "source_kind": "planned_node",
         "runtime_occurrence_id": "p0", "params": {}, "effect": first.effects},
        {"phase_id": "p1", "source_kind": "planned_node",
         "runtime_occurrence_id": "p1", "params": {}, "effect": second.effects},
        {"phase_id": "gap", "source_kind": "task_gap",
         "runtime_occurrence_id": "gap0", "params": {}, "effect": inserted.effects},
    ]
    candidates = [SimpleNamespace(skill=node, segment=segment,
                                  alignment=SimpleNamespace(matched=True))
                  for node, segment in zip((first, second, inserted), segments)]
    atomic_result = SimpleNamespace(
        candidates=candidates, segments=segments,
        decisions=["reuse", "reuse", "reuse"])
    revision = TraceGraphReconstructor(registry).reconstruct(
        trace=trace, atomic_result=atomic_result, selected_composite=old)
    assert revision.revision_kind == "existing_capability_insert"
    assert revision.task_gap_proved_missing_effect is True

    result = CompositeBuilder(
        registry, SystemConfig(data_dir=workspace_tmp / "revision_data")
    ).build_or_align(
        [first.ref, second.ref, inserted.ref], trace,
        segments=segments, revision=revision)
    assert result.composite is not None
    assert registry.get(old.ref).status == SkillStatus.SUPPRESSED
    lineage = [edge for edge in registry.edge_objects(EdgeType.DERIVED_FROM)
               if edge.source == str(result.composite.ref)]
    assert [edge.target for edge in lineage] == [str(old.ref)]


def test_revision_preserves_zero_action_parent_missing_from_extractor(
        workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "zero_action_revision")
    first, second, inserted = (
        _atomic("cap.first"), _atomic("cap.second"), _atomic("cap.inserted"))
    for atomic in (first, second, inserted):
        registry.register(atomic)
    old = _composite(
        "composite.old-zero-action", [first, second],
        status=SkillStatus.ACTIVE)
    registry.register(old)
    trace = TraceRecord(
        trace_id="trace_zero_action_gap", success=True,
        selected_composite=str(old.ref),
        provenance={"target_effects": list(inserted.effects)},
        task_gap_analysis=TaskGapAnalysis(
            missing_effects=list(inserted.effects)),
        runtime_spans=[
            # s0 was ALREADY_SATISFIED and therefore intentionally has no
            # action span and no Extractor phase.
            RuntimeSpan(kind="planned_node", occurrence_id="p1",
                        action_start=0, action_end=1),
            RuntimeSpan(kind="task_gap", occurrence_id="gap0",
                        action_start=1, action_end=2,
                        missing_effects=list(inserted.effects)),
        ],
        realized_atomic_nodes=[
            {"ref": str(first.ref), "origin_step_id": "s0",
             "occurrence_id": "p0", "passed": True,
             "execution_status": "already_satisfied", "params": {},
             "before": {}, "after": {}},
            {"ref": str(second.ref), "origin_step_id": "s1",
             "occurrence_id": "p1", "passed": True,
             "execution_status": "executed_success", "params": {},
             "before": {}, "after": {}},
            {"ref": "skill://runtime.dynamic.task_gap@0.0.0",
             "occurrence_id": "gap0", "passed": True},
        ],
        node_validators=[NodeValidationResult(
            node_ref="skill://runtime.dynamic.task_gap@0.0.0",
            level="task_gap", passed=True, occurrence_id="gap0")],
    )
    gap_segment = {
        "phase_id": "gap", "source_kind": "task_gap",
        "runtime_occurrence_id": "gap0", "params": {},
        "effect": inserted.effects,
        "extraction_method": "llm_proposal_code_validated",
    }
    atomic_result = SimpleNamespace(
        candidates=[SimpleNamespace(
            skill=inserted, segment=gap_segment,
            alignment=SimpleNamespace(matched=True))],
        segments=[gap_segment], decisions=["reuse"])

    revision = TraceGraphReconstructor(registry).reconstruct(
        trace=trace, atomic_result=atomic_result, selected_composite=old)
    assert revision.revision_kind == "existing_capability_insert"
    assert revision.task_gap_proved_missing_effect is True
    assert [item["skill_ref"] for item in revision.realized_occurrences] == [
        str(first.ref), str(second.ref), str(inserted.ref)]
    assert revision.realized_occurrences[0]["execution_status"] == (
        "already_satisfied")

    build_inputs = _revision_build_inputs(revision, registry)
    assert build_inputs is not None
    refs, segments = build_inputs
    result = CompositeBuilder(
        registry, SystemConfig(data_dir=workspace_tmp / "zero_action_data")
    ).build_or_align(refs, trace, segments=segments, revision=revision)
    assert result.composite is not None
    assert [step["node_ref"] for step in result.composite.step_instances()] == [
        f"{first.ref.logical_id}@{first.ref.version}",
        f"{second.ref.logical_id}@{second.ref.version}",
        f"{inserted.ref.logical_id}@{inserted.ref.version}"]


def test_observation_only_gap_never_suppresses_parent(workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "observation_only_gap")
    first, second, missing, revealed_action = (
        _atomic("cap.first"), _atomic("cap.second"),
        _atomic("cap.missing-target"), _atomic("cap.open-for-observation"))
    same_role_contract = [{
        "predicate": "object.at_location",
        "args": {"object": "$inputs.object",
                 "location": "$inputs.target_location"},
    }]
    # Both contracts intentionally have the same predicate and placeholder
    # names.  Their concrete scopes differ and must never be merged.
    missing.effects = list(same_role_contract)
    revealed_action.effects = list(same_role_contract)
    for atomic in (first, second, missing, revealed_action):
        registry.register(atomic)
    parent = _composite(
        "composite.observation-parent", [first, second],
        status=SkillStatus.ACTIVE)
    registry.register(parent)
    trace = TraceRecord(
        trace_id="trace_observation_only", success=True,
        selected_composite=str(parent.ref),
        provenance={"target_effects": list(missing.effects),
                    "realized_params": {
                        "object": "mug_1",
                        "target_location": "cabinet_1"}},
        task_gap_analysis=TaskGapAnalysis(
            missing_effects=list(missing.effects)),
        runtime_spans=[
            RuntimeSpan(kind="planned_node", occurrence_id="p0",
                        action_start=0, action_end=1),
            RuntimeSpan(kind="planned_node", occurrence_id="p1",
                        action_start=1, action_end=2),
            RuntimeSpan(kind="task_gap", occurrence_id="gap0",
                        action_start=2, action_end=3,
                        missing_effects=list(missing.effects)),
        ],
        realized_atomic_nodes=[
            {"ref": str(first.ref), "origin_step_id": "s0",
             "occurrence_id": "p0", "passed": True},
            {"ref": str(second.ref), "origin_step_id": "s1",
             "occurrence_id": "p1", "passed": True},
            {"ref": "skill://runtime.dynamic.task_gap@0.0.0",
             "occurrence_id": "gap0", "passed": True},
        ],
        node_validators=[NodeValidationResult(
            node_ref="skill://runtime.dynamic.task_gap@0.0.0",
            level="task_gap", passed=True, occurrence_id="gap0")],
    )
    # The gap produced the same predicate for mug_2, while the formally missing
    # mug_1 fact merely became visible afterward.  Same placeholder names must
    # not collapse these two binding scopes.
    segment = {
        "phase_id": "gap_open", "source_kind": "task_gap",
        "runtime_occurrence_id": "gap0",
        "params": {"object": "mug_2",
                   "target_location": "cabinet_1"},
        "effect": revealed_action.effects,
        "extraction_method": "llm_proposal_code_validated",
    }
    atomic_result = SimpleNamespace(
        candidates=[SimpleNamespace(
            skill=revealed_action, segment=segment,
            alignment=SimpleNamespace(matched=True))],
        segments=[segment], decisions=["reuse"])

    revision = TraceGraphReconstructor(registry).reconstruct(
        trace=trace, atomic_result=atomic_result,
        selected_composite=parent)
    assert revision.revision_kind == "observation_only_gap"
    assert revision.task_gap_proved_missing_effect is False
    suppressed = CompositeBuilder(
        registry, SystemConfig(data_dir=workspace_tmp / "observation_data")
    ).revision_builder.suppress_proven_incomplete_parent(
        revision, trace_id=trace.trace_id)
    assert suppressed == ""
    assert registry.get(parent.ref).status == SkillStatus.ACTIVE

    trace.provenance["task_gap_effect_proof"] = {
        "passed": False, "inserted_occurrence_ids": ["gap0"],
        "action_caused_effects": list(revealed_action.effects),
    }
    assert _task_gap_is_strong_proof(trace) is False


def test_tool_repair_binds_only_exact_failed_implementation(workspace_tmp):
    registry = SkillGraphRegistry(workspace_tmp / "exact_impl")
    atomic = _atomic("cap.tool")
    registry.register(atomic)
    old_tool = ToolRef("tool.cap", "1.0.0")
    new_tool = ToolRef("tool.cap", "1.1.0")
    first = ImplementationAtom(
        ref=SkillRef("impl.cap.tool", "1.0.0"), abstract_ref=atomic.ref,
        tool_bindings=[ToolBinding(
            tool_ref=old_tool, role="primary", parameter_mapping={})],
        status=SkillStatus.ACTIVE,
    )
    second = ImplementationAtom(
        ref=SkillRef("impl.cap.tool", "1.1.0"), abstract_ref=atomic.ref,
        tool_bindings=[ToolBinding(
            tool_ref=old_tool, role="primary", parameter_mapping={})],
        status=SkillStatus.ACTIVE,
    )
    registry.register(first)
    registry.register(second)
    evolved = FailureBranchManager._bind_candidate_implementations(
        registry, old_tool, new_tool, "branch_exact",
        exact_impl_ref=str(first.ref))
    assert len(evolved) == 1
    assert evolved[0].execution_policy["repair_parent_impl"] == str(first.ref)
    assert evolved[0].tool_bindings[0].tool_ref == new_tool
