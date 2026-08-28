"""成功轨迹进化管线（设计文档 v2.0 §44）。

1. FlowEvo 原始编译产物（可选 base_compiler，作为 baseline 对照/证据）
2. 原子化真实成功因果链
3. 对齐/创建 Abstract Atomic Skill
4. 生成 Implementation Atom 与 Tool candidates（Skeleton → Admission → candidate）
5. 构造或更新 Composite Skill
6. 更新 Layer-2 guideline / evidence
7. 达到多轨迹门槛时更新 Layer-3 insight
8. 全局 Tool generalization / merge / specialization maintenance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..atomicizer.trace_atomicizer import TraceAtomicizer
from ..core.config import SystemConfig
from ..core.refs import SkillRef, ToolRef, bump_version
from ..core.skill_ir import ImplementationAtom, ToolBinding
from ..core.status import EdgeType, SkillNodeKind, SkillStatus, ToolLifecycle
from ..core.trace_ir import TraceRecord
from ..graph.aligner import align_implementation
from ..graph.registry import SkillGraphRegistry
from ..tools.admission_adapter import AdmissionEngine
from ..tools.compiler_adapter import mine_action_template_tools, mine_code_tools
from ..tools.generalizer import ToolGeneralizer
from ..tools.lifecycle import ToolLifecycleManager
from ..tools.registry import ToolRegistry
from .composite_builder import CompositeBuilder
from .composite_lifecycle import reevaluate_waiting_composites
from .insight_updater import InsightUpdater
from .trace_graph_reconstructor import TraceGraphReconstructor
from ..atomicizer.semantic_extractor import SemanticExtractorAgent


@dataclass
class SuccessProcessingResult:
    atomic_refs: list[str] = field(default_factory=list)
    tool_refs: list[str] = field(default_factory=list)
    admitted_tools: int = 0
    shadowed_tools: int = 0
    composite: dict[str, Any] = field(default_factory=dict)
    insight: dict[str, Any] = field(default_factory=dict)
    maintenance: list[dict[str, Any]] = field(default_factory=list)
    base_artifacts: int = 0
    notes: list[str] = field(default_factory=list)
    extraction: dict[str, Any] = field(default_factory=dict)
    graph_proposal: dict[str, Any] = field(default_factory=dict)
    graph_revision: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "atomic_refs": self.atomic_refs,
            "tool_refs": self.tool_refs,
            "admitted_tools": self.admitted_tools,
            "shadowed_tools": self.shadowed_tools,
            "composite": self.composite,
            "insight": self.insight,
            "maintenance": self.maintenance,
            "base_artifacts": self.base_artifacts,
            "notes": self.notes,
            "extraction": self.extraction,
            "graph_proposal": self.graph_proposal,
            "graph_revision": self.graph_revision,
        }


class SuccessProcessor:
    """成功轨迹 → SkillGraph + Tool Repository 联合进化。"""

    def __init__(self, registry: SkillGraphRegistry, tool_registry: ToolRegistry,
                 trace_store, config: SystemConfig,
                 llm=None, base_compiler: Callable | None = None,
                 sandbox=None, replay_fn: Callable | None = None,
                 extractor_llm=None) -> None:
        self.registry = registry
        self.tool_registry = tool_registry
        self.trace_store = trace_store
        self.config = config
        self.llm = llm
        self.base_compiler = base_compiler
        self.extractor_agent = SemanticExtractorAgent(
            extractor_llm, thinking=config.llm.extractor_thinking)
        self.atomicizer = TraceAtomicizer(
            registry, config.thresholds, extractor_agent=self.extractor_agent,
            # The deterministic mock suite predates the semantic Extractor and
            # intentionally exercises the legacy detector without an API.  A
            # real experiment (mock=False) always fails closed.
            allow_legacy_fallback=bool(config.llm.mock or extractor_llm is None))
        self.composite_builder = CompositeBuilder(registry, config)
        self.graph_reconstructor = TraceGraphReconstructor(registry)
        self.insight_updater = InsightUpdater(registry, trace_store, config)
        self.admission = AdmissionEngine(
            sandbox=sandbox,
            replay_fn=replay_fn,
            existing_hashes={t.structural_hash() for t in tool_registry.list_all()},
            timeout_seconds=config.thresholds.admission_timeout_seconds,
        )
        self.generalizer = ToolGeneralizer(tool_registry, self.admission,
                                           sandbox=sandbox)
        self.lifecycle = ToolLifecycleManager(tool_registry, config.thresholds)

    def process_success(self, trace: TraceRecord, *,
                        run_maintenance: bool = True) -> SuccessProcessingResult:
        result = SuccessProcessingResult()

        # 1. FlowEvo 原始编译产物（可选）
        if self.base_compiler is not None:
            try:
                base_artifacts = self.base_compiler(trace)
                result.base_artifacts = len(base_artifacts) if base_artifacts else 0
            except Exception:  # noqa: BLE001
                result.notes.append("base_compiler_failed")

        # 2-3. 原子化 + 对齐/注册 Abstract Atomic Skill
        atomic_result = self.atomicizer.apply(trace)
        result.extraction = dict(getattr(atomic_result, "semantic_extraction", {}) or {})
        atomic_refs = [candidate.skill.ref for candidate in atomic_result.candidates]
        result.atomic_refs = [str(ref) for ref in atomic_refs]
        selected_composite = None
        if trace.selected_composite:
            try:
                selected_composite = self.registry.get(
                    SkillRef.parse(trace.selected_composite))
            except ValueError:
                selected_composite = None
        graph_reconstructor = getattr(self, "graph_reconstructor", None)
        if graph_reconstructor is None:
            # Compatibility for lightweight test/embedding fixtures that
            # construct the processor without calling __init__.
            graph_reconstructor = TraceGraphReconstructor(self.registry)
        revision = graph_reconstructor.reconstruct(
            trace=trace, atomic_result=atomic_result,
            selected_composite=selected_composite)
        result.graph_revision = revision.to_dict()
        trace.provenance["task_gap_effect_proof"] = {
            "passed": bool(revision.task_gap_proved_missing_effect),
            "revision_kind": revision.revision_kind,
            "inserted_occurrence_ids": [
                str(item.get("runtime_occurrence_id") or "")
                for item in revision.inserted_occurrences],
            "action_caused_effects": [
                dict(effect)
                for item in revision.inserted_occurrences
                for effect in (item.get("effect") or [])
                if isinstance(effect, dict)],
            "source": "origin_aware_extractor_code_validation",
        }
        lifecycle_events = (reevaluate_waiting_composites(
            self.registry,
            min_support=max(2, int(
                self.config.thresholds.composite_min_support)))
            if hasattr(self.registry, "list_all_versions") else [])
        for event in lifecycle_events:
            result.notes.append(
                "composite_lifecycle_reevaluated:"
                f"{event['composite_ref']}:{event['from']}->{event['to']}")

        # 4. Tool Skeleton → admission → candidate → Implementation 绑定
        segments = atomic_result.segments
        tools_by_phase: dict[str, list[str]] = {}
        for segment, atomic_ref in zip(segments, atomic_refs):
            source_kind = str(segment.get("source_kind") or "")
            if source_kind == "benchmark_finalization":
                result.notes.append(
                    f"tool_mining_blocked_benchmark_finalization:{atomic_ref.logical_id}")
                continue
            is_env_segment = (str(segment.get("kind") or "") == "env"
                              or trace.benchmark in ("alfworld", "toy_env"))
            safe_event_slice = (bool(segment.get("event_slice_validated"))
                                and bool(segment.get("replay_safe")))
            # Real interactive Tools are executable claims.  Missing slice
            # evidence is unsafe, not equivalent to a clean legacy segment.
            # Mock compatibility is retained only for deterministic old tests.
            legacy_test_mode = bool(
                self.config.llm.mock
                or getattr(self.atomicizer, "allow_legacy_fallback", False))
            # A terminal certificate proves a multi-fact terminal relation,
            # not that its final event is a standalone executable template.
            # Keep the Abstract capability and Composite occurrence, but never
            # compile this slice into a misleading one-action Direct Tool.
            if bool(segment.get("terminal_effect_origin")):
                result.notes.append(
                    f"tool_mining_blocked_terminal_certificate:{atomic_ref.logical_id}")
                continue
            if (is_env_segment and not legacy_test_mode
                    and not safe_event_slice):
                result.notes.append(
                    f"tool_mining_blocked_unsafe_event_slice:{atomic_ref.logical_id}")
                continue
            # Direct 失败后由 Seeded/Dynamic 救回的动作不能按普通成功样本直接
            # 写入 main Tool Repository。它必须进入隔离 repair branch，并在
            # 原失败任务上通过 strict-direct replay 后才能合并。
            if (_contains_rescued_direct(trace, atomic_ref)
                    or _trace_contains_rescued_code_direct(trace)):
                result.notes.append(
                    f"tool_mining_deferred_to_failure_branch:{atomic_ref.logical_id}")
                continue
            tools = self._mine_tools(trace, [segment])
            admitted = self._admit_tools(tools, trace, result)
            tools_by_phase[str(segment.get("phase_id") or "")] = [
                str(tool.ref) for tool in admitted]
            if admitted:
                self._bind_or_update_implementation(atomic_ref, admitted, trace)
            else:
                result.notes.append(f"no_tool_admitted_for:{atomic_ref.logical_id}")

        # 5. Composite
        semantic_valid = (result.extraction.get("method")
                          == "llm_proposal_code_validated")
        allow_mock_fallback = bool(self.config.llm.mock)
        revision_blocks_composite = revision.revision_kind in {
            "benchmark_finalization_only", "observation_only_gap"}
        build_atomic_refs = list(atomic_refs)
        build_segments = list(segments)
        if revision.revision_kind in {
                "new_capability_insert", "existing_capability_insert",
                "repeated_occurrence_insert"}:
            revised_inputs = _revision_build_inputs(revision, self.registry)
            if revised_inputs is not None:
                build_atomic_refs, build_segments = revised_inputs
                result.notes.append(
                    "composite_built_from_complete_runtime_occurrence_sequence")
            else:
                revision_blocks_composite = True
                result.notes.append(
                    "composite_revision_blocked_incomplete_runtime_occurrences")
        if (self.config.features.enable_composite and atomic_refs
                and not revision_blocks_composite
                and (semantic_valid or allow_mock_fallback)):
            occurrences = []
            for index, (segment, atomic_ref) in enumerate(
                    zip(build_segments, build_atomic_refs)):
                atomic = self.registry.get(atomic_ref) or self.registry.get_recommended(
                    atomic_ref.logical_id)
                phase_id = str(segment.get("phase_id") or f"phase_{index:03d}")
                occurrences.append({
                    "phase_id": phase_id, "skill_ref": str(atomic_ref),
                    "event_start": int(segment.get("event_start", index)),
                    "event_end": int(segment.get("event_end", index)),
                    "params": dict(segment.get("params") or {}),
                    "preconditions": list(getattr(atomic, "preconditions", []) or []),
                    "effects": list(getattr(atomic, "effects", []) or []),
                    "tool_refs": tools_by_phase.get(phase_id, []),
                })
            graph_proposal = (self.extractor_agent.propose_graph(trace, occurrences)
                              if hasattr(self, "extractor_agent") else {})
            result.graph_proposal = graph_proposal
            build = self.composite_builder.build_or_align(
                build_atomic_refs, trace, segments=build_segments,
                graph_proposal=graph_proposal, revision=revision)
            result.composite = build.to_dict()
            # 7. Layer-3 insight
            if build.composite is not None:
                support = int((build.composite.metadata.get("statistics") or {}).get(
                    "support_count", 0))
                insight_due = (
                    support == int(self.config.thresholds.insight_min_samples)
                    or run_maintenance
                )
                if insight_due:
                    insight = self.insight_updater.update_if_ready(
                        build.composite.ref)
                    if insight:
                        result.insight = insight
                else:
                    result.insight = {
                        "updated": False,
                        "reason": "deferred_until_threshold_or_maintenance",
                        "sample_count": support,
                    }
        elif revision_blocks_composite:
            result.composite = {
                "composite": None, "decision": "skipped",
                "reason": revision.revision_kind,
            }
            result.notes.append(
                f"composite_revision_routed:{revision.revision_kind}")
        elif self.config.features.enable_composite and atomic_refs:
            result.notes.append("composite_skipped:semantic_extraction_not_validated")

        if (revision.task_gap_proved_missing_effect
                and not result.composite.get("composite")):
            # A proof that the selected parent was incomplete is not itself a
            # verified replacement.  Keep the parent available until a revised
            # candidate reaches Active through independent support/replay.
            result.notes.append(
                "selected_composite_retained_until_active_replacement")

        # 6. 语义证据已在 apply（Abstract statistics）与 record 中更新

        # 8. 生命周期审查很轻量，且是 candidate 获得 Admission/support 后进入
        # Direct 的唯一正常通道，因此每次成功都执行。若也按 maintenance_interval
        # 延迟，小规模实验可能在达到阈值后仍冻结为 candidate，形成 Direct 冷启动
        # 死锁。昂贵的跨 Tool 泛化仍保持周期触发。
        if (self.config.features.enable_tool_evolution
                and self.config.features.enable_governance):
            events = self.lifecycle.review(skill_graph=self.registry)
            result.maintenance = [e.to_dict() for e in events]
        if (run_maintenance and self.config.features.enable_tool_evolution
                and self.config.features.enable_generalization):
            for action in self.generalizer.run_maintenance():
                result.maintenance.append(action.to_dict())
                if action.proposed is not None:
                    bound = self._bind_evolved_tool(action.proposed,
                                                   action.tools, trace)
                    if bound:
                        result.notes.append(f"evolved_tool_bound:{bound}")
        return result

    # ------------------------------------------------------------------
    def _mine_tools(self, trace: TraceRecord, segments: list[dict[str, Any]]) -> list:
        from ..tools.compiler_adapter import AtomicSegment
        # 仅挖掘具有核心 Effect 的片段（纯机械片段不构成 Tool）
        segment_objs = [AtomicSegment.from_dict(segment) for segment in segments
                        if segment.get("effect")]
        if trace.benchmark in ("alfworld", "toy_env") or trace.actions:
            return mine_action_template_tools(trace, segment_objs)
        return mine_code_tools(trace, segment_objs,
                               enable_primitive_reuse=self.config.features.enable_primitive_reuse)

    def _admit_tools(self, tools: list, trace: TraceRecord,
                     result: SuccessProcessingResult) -> list[Any]:
        admitted: list = []
        for skeleton in tools:
            # Legacy/corrupt banks may already contain the unsuffixed ref with
            # another body.  Allocate a fresh immutable version instead of
            # crediting or overwriting that executable.
            existing = self.tool_registry.get(skeleton.ref)
            if (existing is not None
                    and existing.structural_hash() != skeleton.structural_hash()):
                latest = self.tool_registry.get_latest(skeleton.tool_id) or existing
                version = bump_version(latest.ref.version, "minor")
                while self.tool_registry.get(ToolRef(skeleton.tool_id, version)) is not None:
                    version = bump_version(version, "minor")
                skeleton.ref = ToolRef(skeleton.tool_id, version)
                result.notes.append(
                    f"tool_shape_collision_new_version:{skeleton.ref}")
                existing = None
            # 同 ref 且同 executable 的独立成功 occurrence → 证据 +1；
            # 同一 trace 重放不会重复计 support。
            if existing is not None and existing.is_usable():
                sources = list(existing.provenance.get("source_trace_ids") or [])
                independent = trace.trace_id not in sources
                if independent:
                    sources.append(trace.trace_id)
                    stats = existing.statistics
                    stats["support_count"] = int(stats.get("support_count", 0)) + 1
                    existing.provenance["source_trace_ids"] = sources
                    task_types = list(
                        existing.provenance.get("source_task_types") or [])
                    if trace.task_type and trace.task_type not in task_types:
                        task_types.append(trace.task_type)
                    existing.provenance["source_task_types"] = task_types
                self.tool_registry.update_tool(existing)
                result.notes.append(f"tool_reused:{existing.tool_id}")
                result.tool_refs.append(str(existing.ref))
                admitted.append(existing)
                continue
            # 去重复用：已有等价可用 Tool（不同 ref）→ 证据 +1，不重复注册
            duplicates = [d for d in self.tool_registry.find_duplicates(skeleton)
                          if d.ref != skeleton.ref]
            reused = None
            for dup in duplicates:
                if dup.is_usable():
                    stats = dup.statistics
                    stats["support_count"] = int(stats.get("support_count", 0)) + 1
                    self.tool_registry.update_tool(dup)
                    reused = dup
                    break
            if reused is not None:
                result.notes.append(f"tool_reused:{reused.tool_id}")
                result.tool_refs.append(str(reused.ref))
                admitted.append(reused)
                continue
            admission = self.admission.admit(skeleton)
            if admission.passed:
                # Admission replay actually executed this executable and verified
                # its Effect/tests, so it is valid Tool success evidence. This is
                # distinct from a Seeded/Dynamic LLM rescue, which never credits
                # the Tool.
                skeleton.statistics["admission_replay_success_count"] = (
                    int(skeleton.statistics.get("admission_replay_success_count", 0)) + 1)
                ref = self.tool_registry.register(skeleton)
                result.admitted_tools += 1
                result.tool_refs.append(str(ref))
                admitted.append(skeleton)
            else:
                # shadow：保留用于审计和后续修复，不直接调用（§27.1）
                try:
                    self.tool_registry.register(skeleton)
                except ValueError:
                    pass
                result.shadowed_tools += 1
                result.notes.append(f"tool_shadowed:{skeleton.tool_id}:{';'.join(admission.reasons[:2])}")
        return admitted

    def _bind_or_update_implementation(self, atomic_ref: SkillRef,
                                       admitted_tools: list,
                                       trace: TraceRecord) -> None:
        features = self.config.features
        bindings = []
        for tool in admitted_tools:
            mapping = {param: f"$inputs.{param}" for param in tool.param_names()}
            bindings.append(ToolBinding(tool_ref=tool.ref, role="primary",
                                        parameter_mapping=mapping))
        if not features.enable_nm_binding:
            bindings = bindings[:1]

        impl = ImplementationAtom(
            ref=SkillRef(f"impl.{atomic_ref.logical_id}",
                         self._next_impl_version(atomic_ref)),
            abstract_ref=atomic_ref,
            tool_bindings=bindings,
            execution_policy={
                "mode": "direct_if_eligible",
                "on_failure": "seeded_then_dynamic",
                "max_direct_retries": 0,
            },
            compatibility={"harness": _harness_of(trace.benchmark)},
            quality={"use_count": 0, "success_count": 1, "failure_count": 0,
                     "utility": 0.5},
            status=SkillStatus.ACTIVE,
        )
        alignment = align_implementation(impl, self.registry)
        if alignment.matched:
            existing = self.registry.get(SkillRef.parse(alignment.matched_ref))
            if existing is not None:
                quality = dict(existing.quality or {})
                quality["success_count"] = int(quality.get("success_count", 0)) + 1
                quality["utility"] = min(1.0, float(quality.get("utility", 0.5)) + 0.05)
                existing.quality = quality
                self.registry.update_runtime_state(existing)
                return
        self.registry.register(impl)
        self.registry.add_edge(
            str(impl.ref), f"environment://{impl.compatibility.get('harness', 'unknown')}",
            EdgeType.REQUIRES_ENVIRONMENT,
            metadata={"requirement": impl.compatibility.get("harness", "unknown")},
            evidence=[trace.trace_id],
        )

    def _next_impl_version(self, atomic_ref: SkillRef) -> str:
        """同一 Abstract 的不同 Tool 绑定 → Implementation 新版本（§38.1）。"""
        latest = self.registry.get_latest(f"impl.{atomic_ref.logical_id}")
        if latest is None:
            return "1.0.0"
        return bump_version(latest.ref.version, "minor")

    def _bind_evolved_tool(self, tool, source_tool_refs: list[str],
                           trace: TraceRecord) -> str:
        """让已 admission 的泛化 Tool 可被 Resolver 找到，而非成为孤立资产。"""
        sources = set(source_tool_refs)
        for impl in self.registry.list_by_kind(SkillNodeKind.IMPLEMENTATION_ATOMIC):
            if not any(str(binding.tool_ref) in sources for binding in impl.tool_bindings):
                continue
            latest = self.registry.get_latest(impl.ref.logical_id) or impl
            bindings: list[ToolBinding] = []
            replaced = False
            for binding in impl.tool_bindings:
                if str(binding.tool_ref) in sources and not replaced:
                    mapping = {name: f"$inputs.{name}" for name in tool.param_names()}
                    bindings.append(ToolBinding(tool_ref=tool.ref, role=binding.role,
                                                parameter_mapping=mapping))
                    replaced = True
                elif str(binding.tool_ref) not in sources:
                    bindings.append(binding)
            evolved = ImplementationAtom(
                ref=SkillRef(impl.ref.logical_id,
                             bump_version(latest.ref.version, "minor")),
                abstract_ref=impl.abstract_ref, tool_bindings=bindings,
                execution_policy={**impl.execution_policy,
                                  "evolved_tool": True},
                compatibility=dict(impl.compatibility),
                quality={"use_count": 0, "success_count": 0,
                         "failure_count": 0, "utility": 0.45},
                status=SkillStatus.ACTIVE,
            )
            self.registry.register(evolved)
            self.registry.add_edge(str(evolved.ref), str(impl.ref),
                                   EdgeType.GENERALIZED_FROM,
                                   evidence=[trace.trace_id],
                                   metadata={"reason": "tool_generalization_binding"})
            return str(evolved.ref)
        return ""

def _revision_build_inputs(revision, registry) \
        -> tuple[list[SkillRef], list[dict[str, Any]]] | None:
    """Materialize the complete validated occurrence sequence for revision.

    Parent steps may be absent from Extractor output when they required zero
    actions.  The reconstructor preserves their exact immutable child refs;
    this helper fails closed if any occurrence cannot be resolved.
    """
    refs: list[SkillRef] = []
    segments: list[dict[str, Any]] = []
    occurrences = list(
        getattr(revision, "canonical_occurrences", None)
        or getattr(revision, "realized_occurrences", None) or [])
    if len(occurrences) < 2:
        return None
    for index, occurrence in enumerate(occurrences):
        try:
            ref = SkillRef.parse(str(occurrence.get("skill_ref") or ""))
        except ValueError:
            return None
        if registry.get(ref) is None:
            return None
        refs.append(ref)
        segment = dict(occurrence)
        segment.setdefault("phase_id", f"phase_{index:03d}")
        segment.setdefault("source_kind", "planned_node")
        segment.setdefault("runtime_occurrence_id", "")
        segment.setdefault("params", {})
        segment.setdefault("effect", [])
        segment.setdefault("negative_effect", [])
        segment.setdefault("preconditions", [])
        segments.append(segment)
    return refs, segments


def _harness_of(benchmark: str) -> str:
    if benchmark in ("alfworld", "toy_env"):
        return "env"
    return "code_math"


def _contains_rescued_direct(trace: TraceRecord, atomic_ref: SkillRef) -> bool:
    """只根据节点真实 attempt 判定 Direct 失败后救回。

    轨迹中的位置发现、导航等框架动作不能作为 Tool 失败证据。
    """
    wanted = str(atomic_ref).split("@", 1)[0]
    for node in trace.realized_atomic_nodes or []:
        if str(node.get("ref") or "").split("@", 1)[0] != wanted:
            continue
        attempts = list(node.get("attempts") or [])
        for index, attempt in enumerate(attempts):
            if (str(attempt.get("mode") or "") == "direct"
                    and not bool(attempt.get("passed"))
                    and any(bool(later.get("passed")) and
                            str(later.get("mode") or "") in ("seeded", "dynamic")
                            for later in attempts[index + 1:])):
                return True
    return False


def _trace_contains_rescued_code_direct(trace: TraceRecord) -> bool:
    attempts = list(trace.attempts or [])
    return any(attempt.stage == "direct_tool" and not attempt.passed
               and any(later.passed for later in attempts[index + 1:])
               for index, attempt in enumerate(attempts))
