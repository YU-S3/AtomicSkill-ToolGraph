"""Atomic Planner（设计文档 v2.0 §21）。

检索顺序（§21.2）：
    解析目标状态 → capability/effect 召回 → Composite → Abstract Atomic
    → I/O / Preconditions / Effect 硬过滤 → 结构与历史 Utility 重排
    → 最小充分 Runtime Graph → Implementation 选择（另模块）→ Tool 解析（另模块）
task_type 禁止作为硬过滤（§21.3）；LLM 只规划 Atomic Skill（§13）。
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..adapters.benchmark import parse_goal_params
from ..core.binding_ir import (BindingKind, BindingSpec, binding_slot_name,
                               is_concrete_binding, resolve_binding)
from ..core.config import SystemConfig
from ..core.edge_ir import GraphEdge
from ..core.llm import LLM
from ..core.refs import SkillRef
from ..core.semantic_roles import unsafe_composite_task_role_binding
from ..core.skill_ir import AbstractAtomicSkill, CompositeSkill
from ..core.status import EdgeType, SkillNodeKind, SkillStatus
from ..graph.graph import composite_step_order
from ..graph.registry import RetrievalHit, SkillGraphRegistry
from .data_flow_synthesizer import RuntimeDataFlowSynthesizer
from .runtime_graph import PlannedNode, RuntimePlan
from .plan_validator import (semantic_required_slots, validate_plan_bindings,
                             validate_plan_source_closure)
from .contract_matcher import match_effect_contract


class PlanCompilationError(RuntimeError):
    pass

_PLAN_PROMPT = (
    "You are an atomic task planner. Given the task goal and a list of reusable "
    "atomic skills (each with summary and inputs), select the minimal sufficient "
    "ordered subset of skills that achieves the goal, and bind any known parameters "
    "from the goal text. Output ONLY a JSON object with the form "
    '{"skills": [{"logical_id": "...", "params": {...}}]} (no extra text).'
)


class AtomicPlanner:
    """任务 → 最小充分 Runtime Graph。"""

    def __init__(self, registry: SkillGraphRegistry, config: SystemConfig,
                 llm: LLM | None = None) -> None:
        self.registry = registry
        self.config = config
        self.llm = llm
        self.composite_min_score = 0.45
        self.data_flow_synthesizer = RuntimeDataFlowSynthesizer()

    def compile_runtime_graph(self, task) -> RuntimePlan:
        features = self.config.features
        query = {
            "goal_text": task.goal,
            "state": task.state,
            "available_inputs": list((task.context.get("params") or {}).keys())
            + _extract_entity_names(task.goal),
            "target_effects": task.target_effects,
        }
        hard_restrict = features.task_type_hard_restricted or not features.enable_cross_task_type_reuse
        # Official benchmark labels are retained for sampling/metrics only in
        # the default method.  They enter retrieval solely in the explicit
        # task-type-restricted ablation.
        if hard_restrict:
            query["task_type"] = task.task_type
        # Composite 的充分性判断必须看到完整候选集，不能让词面得分较低但
        # Effect 完整的 Composite 在 top-k 截断时先被丢掉。
        retrieval_limit = max(self.config.retrieval_top_k,
                              len(self.registry.list_all()))
        retrieved = self.registry.retrieve(
            query, top_k=retrieval_limit,
            hard_restrict_task_type=hard_restrict,
            task_type_bonus=(self.config.task_type_soft_bonus if hard_restrict else 0.0),
        )
        # A formal target contract is a stronger retrieval key than lexical
        # similarity. Keep the complete structural pool so a low-summary-score
        # producer can close an Effect→Precondition dependency. With no formal
        # target, retain the conservative score gate.
        if not task.target_effects:
            retrieved = [h for h in retrieved
                         if h.score >= self.config.planning_min_score]
        if not retrieved:
            return self._task_dynamic_plan(
                task, [], reason="no_capability_retrieved")

        # 1. Composite 优先，但先做目标 Effect 充分性硬过滤。summary/utility
        # 只能在同等充分的候选之间重排，不能让两步链压过完整三步链。
        if features.enable_composite:
            composites = [h for h in retrieved if h.kind == SkillNodeKind.COMPOSITE]
            if self.config.freeze_skills:
                # Frozen evaluation may only consume promoted knowledge.  A
                # single-trace draft is an online exploration candidate, not a
                # frozen reusable capability, even when its schema validates.
                composites = [h for h in composites
                              if h.obj.status.value == "active"]
            # Sufficiency is a hard gate before lexical/utility score. A fully
            # covering Composite cannot be discarded merely because its summary
            # is phrased differently from the new goal.
            complete = [h for h in composites
                        if (self._composite_covers_targets(
                                h.obj, task.target_effects)
                            and self._composite_goal_relevant(
                                h.obj, task.target_effects))]
            if complete:
                rejections: list[dict[str, Any]] = []
                for selected in sorted(
                        complete,
                        key=lambda h: self._composite_selection_rank(
                            h, task.target_effects), reverse=True):
                    if selected.obj.status != SkillStatus.ACTIVE:
                        rejections.append({"ref": str(selected.ref),
                                           "reason": "candidate_not_active"})
                        continue
                    try:
                        plan = self._plan_from_composite(task, selected, retrieved)
                    except PlanCompilationError as exc:
                        rejections.append({"ref": str(selected.ref),
                                           "reason": str(exc)})
                        continue
                    report = validate_plan_bindings(plan, self.registry, task)
                    if not report.passed:
                        rejections.append({"ref": str(selected.ref),
                                           "reason": "plan_binding_unresolved",
                                           "details": report.to_dict()})
                        continue
                    plan.notes.append("composite_effect_complete")
                    plan.audit = {
                        "retrieved_candidates": [str(h.ref) for h in composites],
                        "candidate_rejections": rejections,
                        "selected_plan": str(selected.ref),
                    }
                    return plan

            eligible = [h for h in composites
                        if (h.score >= self.composite_min_score
                            and h.obj.status == SkillStatus.ACTIVE
                            and self._composite_goal_relevant(
                                h.obj, task.target_effects))]
            if eligible:
                # A relevant partial Composite may execute its verified
                # occurrences, but missing task Effects are never pre-inserted
                # as anonymous planner nodes.  System.run_task freezes the
                # pre-gap boundary first and appends exactly one explicit
                # ``runtime.dynamic.task_gap`` occurrence from the measured
                # state delta.
                for partial in sorted(
                        eligible,
                        key=lambda h: (self._composite_target_coverage(
                            h.obj, task.target_effects), h.score),
                        reverse=True):
                    # The future explicit Task Gap must itself have a closed
                    # semantic source.  Otherwise executing the partial plan
                    # would merely postpone an ungrounded local goal.
                    if not self._partial_task_gap_source_closed(task, partial.obj):
                        continue
                    try:
                        partial_plan = self._plan_from_composite(
                            task, partial, retrieved)
                    except PlanCompilationError:
                        continue
                    report = validate_plan_bindings(
                        partial_plan, self.registry, task)
                    if partial_plan and report.passed:
                        partial_plan.notes.append(
                            "composite_partial_explicit_task_gap_pending")
                        partial_plan.audit = {
                            "retrieved_candidates": [str(h.ref)
                                                     for h in composites],
                            "selected_plan": str(partial.ref),
                            "partial_target_coverage": (
                                self._composite_target_coverage(
                                    partial.obj, task.target_effects)),
                        }
                        return partial_plan

        # 2. Abstract Atomic 直接规划
        atomics = [h for h in retrieved if h.kind == SkillNodeKind.ABSTRACT_ATOMIC]
        if self.config.freeze_skills:
            atomics = [h for h in atomics if h.obj.status == SkillStatus.ACTIVE]
        if not atomics:
            return self._task_dynamic_plan(
                task, retrieved,
                reason="no_reusable_target_producer_dynamic_only")

        return self._compile_atomic_runtime_plan(task, atomics, retrieved)

    # ------------------------------------------------------------------
    def _compile_atomic_runtime_plan(
            self, task, atomics: list[RetrievalHit],
            retrieved: list[RetrievalHit]) -> RuntimePlan:
        """Compile retrieved Atomics into an occurrence-aware temporary DAG.

        The persistent Composite ablation must remove only stored workflow
        knowledge.  Basic value transfer is a runtime invariant, so the Atomic
        fallback still receives explicit dependency and DATA_FLOW edges.
        """
        nodes = self._plan_from_atomics(task, atomics)
        if not nodes or not self._atomic_nodes_cover_targets(task, nodes):
            return self._task_dynamic_plan(
                task, retrieved, reason="atomic_target_closure_incomplete")

        plan = self._build_atomic_runtime_plan(task, nodes, retrieved)
        report = validate_plan_source_closure(plan, self.registry, task)
        plan.audit["source_closure"] = report.to_dict()
        if report.passed:
            return plan

        # A helper occurrence without a semantic anchor is not a meaningful
        # local LLM goal.  Remove it and let the anchored consumer's
        # Seeded/Dynamic implementation perform such setup internally.  A
        # target-producing occurrence is never removed this way.
        repaired_nodes, removed = self._remove_unanchored_auxiliary_nodes(
            task, nodes, report)
        if removed and repaired_nodes:
            repaired = self._build_atomic_runtime_plan(
                task, repaired_nodes, retrieved)
            repaired_report = validate_plan_source_closure(
                repaired, self.registry, task)
            repaired.audit.update({
                "source_closure": repaired_report.to_dict(),
                "initial_source_closure": report.to_dict(),
                "removed_unanchored_auxiliary_steps": removed,
            })
            if (repaired_report.passed
                    and self._atomic_nodes_cover_targets(task, repaired.nodes)):
                repaired.notes.append("unanchored_auxiliary_removed")
                return repaired

        fallback = self._task_dynamic_plan(
            task, retrieved, reason="atomic_source_closure_failed")
        fallback.audit["atomic_compilation_rejection"] = {
            "source_closure": report.to_dict(),
            "removed_unanchored_auxiliary_steps": removed,
        }
        return fallback

    def _build_atomic_runtime_plan(
            self, task, nodes: list[PlannedNode],
            retrieved: list[RetrievalHit]) -> RuntimePlan:
        self._renumber_atomic_occurrences(nodes)
        # Synthesis mutates target binding_specs with the exact producer.  A
        # repaired graph must not retain a source_step that was removed from an
        # earlier candidate graph.
        for node in nodes:
            if node.source != "atomic_compilation":
                continue
            for role, spec in list(node.binding_specs.items()):
                if spec.kind == BindingKind.DATA_FLOW:
                    task_value = dict(task.context.get("params") or {}).get(role)
                    node.binding_specs[role] = (
                        BindingSpec(BindingKind.TASK, task_role=role,
                                    symbol=f"$task.{role}")
                        if is_concrete_binding(task_value)
                        else BindingSpec.from_value(f"$inputs.{role}"))
        edges = self._runtime_edges(nodes)
        edges.extend(self._atomic_dependency_edges(nodes))
        edges.extend(self.data_flow_synthesizer.synthesize(
            task, nodes, self.registry))
        self._annotate_cardinality_constraints(task, nodes, edges)
        return RuntimePlan(
            start_mode="warm",
            nodes=nodes,
            edges=_deduplicate_runtime_edges(edges),
            retrieved=[hit.to_dict() for hit in retrieved],
            notes=[
                "atomic_only_plan",
                "atomic_occurrence_dag_compiled",
                "partial_composite_rejected_if_unbound",
            ],
            audit={"plan_source": "atomic_compilation"},
        )

    def _annotate_cardinality_constraints(
            self, task, nodes: list[PlannedNode],
            edges: list[GraphEdge]) -> None:
        """Project task ``distinct_by`` contracts onto occurrence inputs.

        The goal contract names a predicate argument, while an Atomic may use
        a different local input name.  Contract unification identifies the
        terminal input and DATA_FLOW materializers propagate the same group
        back through its causal branch.  Runtime can then reject cross-branch
        instance reuse without knowing a benchmark task type or entity name.
        """
        task_params = dict(task.context.get("params") or {})
        terminals_by_group: dict[str, list[PlannedNode]] = {}
        terminal_roles_by_group: dict[str, dict[str, str]] = {}
        target_roles_by_group: dict[str, str] = {}
        for target_index, target in enumerate(task.target_effects or []):
            if not isinstance(target, dict):
                continue
            cardinality = max(1, int(target.get("cardinality", 1) or 1))
            distinct_arg = str(target.get("distinct_by") or "")
            if cardinality <= 1 or not distinct_arg:
                continue
            target_args = dict(target.get("args") or {})
            target_role = binding_slot_name(target_args.get(distinct_arg))
            if not target_role:
                target_role = distinct_arg
            predicate = _canonical_predicate(str(target.get("predicate") or ""))
            group_id = (
                f"target_{target_index:03d}:{predicate}:{distinct_arg}")
            target_roles_by_group[group_id] = target_role
            single = {**target, "cardinality": 1, "distinct_by": ""}
            matched_occurrences = 0
            for node in nodes:
                if matched_occurrences >= cardinality:
                    break
                atomic = self.registry.get(node.ref)
                if atomic is None:
                    continue
                bindings = {**task_params, **dict(node.params or {})}
                for effect in getattr(atomic, "effects", []) or []:
                    if not isinstance(effect, dict):
                        continue
                    matched = match_effect_contract(effect, single, bindings)
                    if not matched.passed:
                        continue
                    local_role = next((
                        source_role
                        for source_role, mapped_role in
                        matched.unified_roles.items()
                        if mapped_role == target_role
                    ), "")
                    if not local_role:
                        local_role = binding_slot_name(
                            dict(effect.get("args") or {}).get(distinct_arg))
                    if local_role:
                        _add_distinct_group(
                            node, local_role, group_id)
                        terminals = terminals_by_group.setdefault(group_id, [])
                        if node not in terminals:
                            terminals.append(node)
                        terminal_roles_by_group.setdefault(group_id, {})[
                            node.step_id] = local_role
                        matched_occurrences += 1
                        break

        # Stored Composites predate branch_id in some repositories.  Their
        # ordered target-producing occurrences still provide an unambiguous
        # cardinality branch boundary; Atomic compilation already supplies the
        # same branch ids and is left untouched.
        for group_id, terminals in terminals_by_group.items():
            for ordinal, node in enumerate(terminals):
                if not node.branch_id:
                    node.branch_id = f"branch_{ordinal:03d}"
                _set_distinct_branch_id(
                    node, group_id, f"occ_{ordinal:03d}")

        # Walk causal DATA_FLOW edges backwards until every producer role that
        # materializes a distinct consumer value carries the same constraint.
        by_step = {node.step_id: node for node in nodes}
        changed = True
        while changed:
            changed = False
            for edge in edges:
                if edge.type not in {
                        EdgeType.DATA_FLOW, EdgeType.REQUIRES_SKILL}:
                    continue
                source = by_step.get(str(edge.source_step or ""))
                target = by_step.get(str(edge.target_step or ""))
                if source is None or target is None:
                    continue
                role_pairs: list[tuple[str, str]] = []
                if edge.type == EdgeType.DATA_FLOW:
                    mapping = dict(edge.mapping or {})
                    target_role = str(mapping.get("target_input") or "")
                    materializer = dict(mapping.get("materializer") or {})
                    source_role = str(
                        materializer.get("source_role")
                        or mapping.get("source_output") or "")
                    if source_role and target_role:
                        role_pairs.append((source_role, target_role))
                else:
                    source_atomic = self.registry.get(source.ref)
                    target_atomic = self.registry.get(target.ref)
                    for effect in (
                            getattr(source_atomic, "effects", []) or []):
                        for precondition in (
                                getattr(target_atomic, "preconditions", []) or []):
                            matched = match_effect_contract(
                                effect, precondition)
                            if matched.passed:
                                role_pairs.extend(
                                    matched.unified_roles.items())
                propagated = False
                for source_role, target_role in role_pairs:
                    groups = list(
                        target.distinct_bindings.get(target_role) or [])
                    for group_id in groups:
                        added = _add_distinct_group(
                            source, source_role, group_id)
                        branch_added = _set_distinct_branch_id(
                            source, group_id,
                            target.distinct_branch_ids.get(group_id)
                            or target.branch_id)
                        propagated |= added or branch_added
                changed |= propagated
                if (propagated and target.branch_id
                        and not source.branch_id):
                    source.branch_id = target.branch_id
                    changed = True

        # Some valid stored Composites encode only ordered occurrences with
        # task bindings; there is no DATA_FLOW/REQUIRES edge to traverse.  Each
        # terminal closes one contiguous occurrence branch. Project the group
        # to preceding nodes that carry the same semantic/task-bound value.
        index_by_step = {node.step_id: index
                         for index, node in enumerate(nodes)}
        for group_id, terminals in terminals_by_group.items():
            previous_terminal = -1
            for terminal in terminals:
                terminal_index = index_by_step.get(terminal.step_id, -1)
                if terminal_index < 0:
                    continue
                terminal_role = terminal_roles_by_group[group_id].get(
                    terminal.step_id, "")
                branch_identity = (
                    terminal.distinct_branch_ids.get(group_id)
                    or terminal.branch_id)
                for candidate in nodes[
                        previous_terminal + 1:terminal_index + 1]:
                    roles = self._contiguous_distinct_roles(
                        candidate, terminal, terminal_role,
                        target_roles_by_group.get(group_id, ""), task_params)
                    for role in roles:
                        _add_distinct_group(candidate, role, group_id)
                        _set_distinct_branch_id(
                            candidate, group_id, branch_identity)
                    if roles and not candidate.branch_id:
                        candidate.branch_id = terminal.branch_id
                previous_terminal = terminal_index

    def _contiguous_distinct_roles(
            self, node: PlannedNode, terminal: PlannedNode,
            terminal_role: str, target_role: str,
            task_params: dict[str, Any]) -> set[str]:
        """Find inputs carrying a terminal's distinct semantic entity."""
        atomic = self.registry.get(node.ref)
        terminal_atomic = self.registry.get(terminal.ref)
        if atomic is None or terminal_atomic is None or not terminal_role:
            return set()

        def declaration_type(owner: Any, role: str) -> str:
            return next((
                str(item.get("semantic_type") or "")
                for item in (getattr(owner, "inputs", []) or [])
                if isinstance(item, dict)
                and str(item.get("name") or "") == role
            ), "")

        wanted = (task_params.get(target_role)
                  or terminal.params.get(terminal_role))
        terminal_type = declaration_type(terminal_atomic, terminal_role)
        roles: set[str] = set()
        for declaration in getattr(atomic, "inputs", []) or []:
            if not isinstance(declaration, dict):
                continue
            role = str(declaration.get("name") or "")
            if not role:
                continue
            spec = node.binding_specs.get(role)
            task_role = (str(spec.task_role)
                         if spec is not None
                         and spec.kind == BindingKind.TASK else "")
            candidate_value = (node.params.get(role)
                               or task_params.get(task_role))
            same_value = bool(
                wanted not in (None, "")
                and candidate_value not in (None, "")
                and _same_entity_family(wanted, candidate_value))
            candidate_type = str(
                declaration.get("semantic_type") or "")
            same_type = bool(
                terminal_type and candidate_type == terminal_type)
            task_alias = bool(
                task_role and target_role
                and task_params.get(task_role) not in (None, "")
                and task_params.get(target_role) not in (None, "")
                and _same_entity_family(
                    task_params[task_role], task_params[target_role]))
            if same_value and (
                    role == terminal_role or same_type or task_alias):
                roles.add(role)
        return roles

    def _task_dynamic_plan(self, task, retrieved: list[RetrievalHit], *,
                           reason: str) -> RuntimePlan:
        """Return one full-task Dynamic node, never an unanchored local gap."""
        effects = [dict(item) for item in (task.target_effects or [])
                   if isinstance(item, dict)]
        params = dict(task.context.get("params") or {})
        for effect in effects:
            discovered = self._bind_dynamic_effect_params(task, effect)
            for role, value in discovered.items():
                if is_concrete_binding(value):
                    params.setdefault(role, value)
        binding_specs = {
            str(role): BindingSpec.from_value(value)
            for role, value in params.items()
        }
        node = PlannedNode(
            ref=SkillRef("runtime.dynamic.task_level", "0.0.0"),
            step_id="step_000", occurrence_id="task_dynamic_000",
            branch_id="task", binding_specs=binding_specs,
            params=params, source="task_dynamic", target_effects=effects,
            dynamic=True,
            budget_scope="task",
        )
        return RuntimePlan(
            start_mode="cold", nodes=[node], edges=[],
            retrieved=[hit.to_dict() for hit in retrieved],
            notes=[reason, "task_level_dynamic_fallback"],
            audit={"plan_source": "task_dynamic", "fallback_reason": reason},
        )

    def _remove_unanchored_auxiliary_nodes(
            self, task, nodes: list[PlannedNode], report
            ) -> tuple[list[PlannedNode], list[str]]:
        unclosed_steps: set[str] = set()
        for node_report in list(getattr(report, "node_reports", []) or []):
            unresolved = list(getattr(
                node_report, "unresolved_semantic_slots", []) or [])
            states = dict(getattr(node_report, "resolution_states", {}) or {})
            if unresolved or any(str(value).lower().endswith("unresolvable")
                                 for value in states.values()):
                unclosed_steps.add(str(getattr(node_report, "step_id", "")))
        for error in list(getattr(report, "errors", []) or []):
            match = re.search(r"(?:step|step_id)=([^:;]+)", str(error))
            if match:
                unclosed_steps.add(match.group(1))

        removed: list[str] = []
        kept: list[PlannedNode] = []
        for node in nodes:
            if (node.step_id in unclosed_steps
                    and not self._node_produces_task_target(task, node)):
                removed.append(node.step_id)
                continue
            kept.append(node)
        return kept, removed

    def _atomic_nodes_cover_targets(self, task,
                                    nodes: list[PlannedNode]) -> bool:
        for target in list(task.target_effects or []):
            if not isinstance(target, dict):
                continue
            required = max(1, int(target.get("cardinality", 1) or 1))
            single = {**target, "cardinality": 1, "distinct_by": ""}
            produced = 0
            for node in nodes:
                atomic = self.registry.get(node.ref)
                if atomic is None:
                    continue
                bindings = {
                    **dict(task.context.get("params") or {}),
                    **dict(node.params or {}),
                }
                if any(match_effect_contract(effect, single, bindings).passed
                       for effect in (getattr(atomic, "effects", []) or [])):
                    produced += 1
            if produced < required:
                return False
        return True

    def _node_produces_task_target(self, task, node: PlannedNode) -> bool:
        atomic = self.registry.get(node.ref)
        if atomic is None:
            return False
        bindings = {
            **dict(task.context.get("params") or {}),
            **dict(node.params or {}),
        }
        return any(
            match_effect_contract(
                effect, {**target, "cardinality": 1, "distinct_by": ""},
                bindings).passed
            for effect in (getattr(atomic, "effects", []) or [])
            for target in (task.target_effects or [])
            if isinstance(target, dict)
        )

    def _atomic_dependency_edges(
            self, nodes: list[PlannedNode]) -> list[GraphEdge]:
        """Materialize nearest Effect→Precondition dependencies per branch."""
        edges: list[GraphEdge] = []
        for target_index, target_node in enumerate(nodes):
            target = self.registry.get(target_node.ref)
            if target is None:
                continue
            for precondition in (getattr(target, "preconditions", []) or []):
                producer: PlannedNode | None = None
                for source_node in reversed(nodes[:target_index]):
                    if (target_node.branch_id and source_node.branch_id
                            and target_node.branch_id != source_node.branch_id):
                        continue
                    source = self.registry.get(source_node.ref)
                    if source is None:
                        continue
                    bindings = {**dict(source_node.params or {}),
                                **dict(target_node.params or {})}
                    if any(match_effect_contract(
                            effect, precondition, bindings).passed
                           for effect in (getattr(source, "effects", []) or [])):
                        producer = source_node
                        break
                if producer is None:
                    continue
                edges.append(GraphEdge(
                    source=str(producer.ref), target=str(target_node.ref),
                    type=EdgeType.REQUIRES_SKILL, scope="runtime",
                    source_step=producer.step_id,
                    target_step=target_node.step_id,
                    metadata={"requirement": dict(precondition)},
                ))
        return edges

    @staticmethod
    def _renumber_atomic_occurrences(nodes: list[PlannedNode]) -> None:
        per_branch: dict[str, int] = {}
        for index, node in enumerate(nodes):
            branch = node.branch_id or "branch_000"
            node.branch_id = branch
            ordinal = per_branch.get(branch, 0)
            per_branch[branch] = ordinal + 1
            node.step_id = f"step_{index:03d}"
            node.occurrence_id = f"{branch}_occ_{ordinal:03d}"

    # ------------------------------------------------------------------
    def _plan_from_composite(self, task, hit: RetrievalHit,
                             retrieved: list[RetrievalHit]) -> RuntimePlan:
        composite: CompositeSkill = hit.obj
        steps, report = composite_step_order(composite, self.registry)
        if not report.passed:
            raise PlanCompilationError("plan_graph_invalid:" + ";".join(report.errors))
        nodes: list[PlannedNode] = []
        incoming: dict[tuple[str, str], BindingSpec] = {}
        for edge in composite.edge_objects():
            if edge.type != EdgeType.DATA_FLOW:
                continue
            target_input = str(edge.mapping.get("target_input") or "")
            source_output = str(edge.mapping.get("source_output") or "")
            if target_input and source_output:
                incoming[(edge.target_step, target_input)] = BindingSpec(
                    BindingKind.DATA_FLOW, source_step=edge.source_step,
                    source_output=source_output)
        task_params = dict(task.context.get("params") or {})
        for index, step in enumerate(steps):
            try:
                exact_ref = SkillRef.parse(str(step.get("node_ref") or ""))
            except ValueError as exc:
                raise PlanCompilationError(f"exact_child_ref_invalid:{exc}") from exc
            obj = self.registry.get(exact_ref)
            if obj is None or obj.status != SkillStatus.ACTIVE:
                raise PlanCompilationError(f"exact_child_unavailable:{exact_ref}")
            stored = dict(step.get("params") or {})
            for slot_role, binding in stored.items():
                if unsafe_composite_task_role_binding(
                        str(slot_role), binding):
                    raise PlanCompilationError(
                        "unsafe_composite_task_role_binding:"
                        f"step={step['step_id']}:slot={slot_role}:"
                        f"binding={binding}")
            binding_specs: dict[str, BindingSpec] = {}
            params: dict[str, Any] = {}
            for key, value in stored.items():
                spec = incoming.get((str(step["step_id"]), str(key)),
                                    BindingSpec.from_value(value))
                binding_specs[str(key)] = spec
                resolved = resolve_binding(spec, task_params)
                if is_concrete_binding(resolved):
                    params[str(key)] = resolved
                elif spec.kind == BindingKind.UNRESOLVED:
                    # Keep the symbolic value visible for audit/closure checks.
                    params[str(key)] = spec.symbol or value
            nodes.append(PlannedNode(ref=obj.ref, step_id=f"step_{index:03d}",
                                     occurrence_id=f"occ_{index:03d}",
                                     origin_step_id=str(step["step_id"]),
                                     binding_specs=binding_specs,
                                     params=params, source="composite",
                                     target_effects=list(getattr(obj, "effects", []))))
        # A Composite is a validated causal workflow.  In particular, an
        # intermediate Effect consumed as a later Precondition identifies a
        # required producer; it is the strongest reason to retain the node,
        # never a reason to drop it.  Unbound occurrence parameters are
        # resolved from state/data-flow or bounded runtime discovery.
        edges = self._runtime_edges(nodes, composite.edge_objects())
        self._annotate_cardinality_constraints(task, nodes, edges)
        return RuntimePlan(
            start_mode="warm",
            composite_ref=str(composite.ref),
            nodes=nodes,
            edges=edges,
            retrieved=[h.to_dict() for h in retrieved],
            notes=[f"composite_plan:{composite.ref.logical_id}"],
        )

    @staticmethod
    def _resolve_composite_params(stored: dict[str, Any], task) -> dict[str, Any]:
        """Resolve persisted semantic roles without rebinding every same-kind step globally."""
        context = dict(task.context.get("params") or {})
        resolved: dict[str, Any] = {}
        for key, value in stored.items():
            if unsafe_composite_task_role_binding(str(key), value):
                # Historical banks may contain value-equality guesses that
                # confused an occurrence source with the task destination.
                # Keep this compatibility helper fail-closed; the actual
                # Composite planning path rejects the whole unsafe asset.
                continue
            if isinstance(value, str) and value.startswith("$task."):
                role = value[len("$task."):]
                if context.get(role) not in (None, ""):
                    resolved[str(key)] = context[role]
            elif isinstance(value, str) and value.startswith("$flow."):
                # Preserve an occurrence-scoped role until runtime evidence or
                # an incoming DATA_FLOW edge grounds it.  Never replace it with
                # a semantically unrelated task-level location.
                resolved[str(key)] = value
            elif value not in (None, ""):
                resolved[str(key)] = value
        return resolved

    def _plan_from_atomics(self, task, hits: list[RetrievalHit]) -> list[PlannedNode]:
        # 最小充分：greedy 覆盖 target_effects（若提供），否则取 top-k 中得分 > 阈值的
        if task.target_effects:
            base_branch = self._cover_target_effects(
                task.target_effects, hits,
                dict(task.context.get("params") or {}))
            base_branch = self._close_and_order_dependencies(
                base_branch, hits, task.target_effects)
            expanded = self._expand_cardinality_workflow(
                base_branch, task.target_effects,
                dict(task.context.get("params") or {}))
        else:
            base_branch = self._llm_plan(task, hits) or hits[:3]
            expanded = [
                (hit, "branch_000") for hit in base_branch]
        nodes: list[PlannedNode] = []
        branch_offsets: dict[str, int] = {}
        for index, (hit, branch_id) in enumerate(expanded):
            atomic = hit.obj
            params = self._bind_params(task, atomic)
            branch_offset = branch_offsets.get(branch_id, 0)
            branch_offsets[branch_id] = branch_offset + 1
            task_params = dict(task.context.get("params") or {})
            binding_specs: dict[str, BindingSpec] = {}
            for declaration in (getattr(atomic, "inputs", []) or []):
                role = str(declaration.get("name") or "")
                if not role:
                    continue
                if role in task_params and is_concrete_binding(task_params[role]):
                    binding_specs[role] = BindingSpec(
                        BindingKind.TASK, task_role=role,
                        symbol=f"$task.{role}")
                elif is_concrete_binding(params.get(role)):
                    binding_specs[role] = BindingSpec.from_value(params[role])
                else:
                    binding_specs[role] = BindingSpec.from_value(
                        f"$inputs.{role}")
            nodes.append(PlannedNode(
                ref=atomic.ref, step_id=f"step_{index:03d}",
                occurrence_id=f"{branch_id}_occ_{branch_offset:03d}",
                branch_id=branch_id, binding_specs=binding_specs,
                params=params, source="atomic_compilation",
                target_effects=list(getattr(atomic, "effects", []))))
        return nodes

    @staticmethod
    def _expand_cardinality_workflow(
            selected: list[RetrievalHit],
            target_effects: list[dict[str, Any]],
            bindings: dict[str, Any] | None = None,
            ) -> list[tuple[RetrievalHit, str]]:
        """Expand variable-width causal branches by target cardinality.

        At occurrence ordinal ``j`` only targets with ``cardinality > j`` are
        active; their backwards dependency closures are unioned in original
        topological order. Thus a 2+3 goal yields ``A+B, A+B, B`` rather than
        repeating the whole union three times.
        """
        if not selected:
            return []
        cardinalities = [
            max(1, int(effect.get("cardinality", 1) or 1))
            for effect in target_effects if isinstance(effect, dict)]
        if max(cardinalities or [1]) <= 1:
            return [(hit, "branch_000") for hit in selected]

        def produces(producer: RetrievalHit, consumer: RetrievalHit) -> bool:
            return any(
                match_effect_contract(effect, precondition).passed
                for effect in (getattr(producer.obj, "effects", []) or [])
                for precondition in (
                    getattr(consumer.obj, "preconditions", []) or []))

        expanded: list[tuple[RetrievalHit, str]] = []
        covered_ids: set[str] = set()
        max_cardinality = max(cardinalities or [1])
        for ordinal in range(max_cardinality):
            active_targets = [
                target for target in target_effects
                if isinstance(target, dict)
                and max(1, int(target.get("cardinality", 1) or 1)) > ordinal
            ]
            terminal_ids: set[str] = set()
            for target in active_targets:
                single = {**target, "cardinality": 1, "distinct_by": ""}
                terminal_ids.update(
                    hit.obj.ref.logical_id for hit in selected
                    if any(
                        match_effect_contract(
                            effect, single, bindings).passed
                           for effect in (
                               getattr(hit.obj, "effects", []) or [])))
            if not terminal_ids:
                continue
            closure_ids = set(terminal_ids)
            changed = True
            while changed:
                changed = False
                consumers = [
                    hit for hit in selected
                    if hit.obj.ref.logical_id in closure_ids]
                for producer in selected:
                    logical_id = producer.obj.ref.logical_id
                    if logical_id in closure_ids:
                        continue
                    if any(produces(producer, consumer)
                           for consumer in consumers):
                        closure_ids.add(logical_id)
                        changed = True
            closure = [
                hit for hit in selected
                if hit.obj.ref.logical_id in closure_ids]
            covered_ids.update(closure_ids)
            branch_id = f"branch_{ordinal:03d}"
            expanded.extend((hit, branch_id) for hit in closure)

        # Preserve a non-target helper exactly once if closure inference could
        # not associate it with a formal target; never multiply it by max(N).
        leftovers = [
            hit for hit in selected
            if hit.obj.ref.logical_id not in covered_ids]
        if leftovers:
            branch_id = f"branch_{max_cardinality:03d}"
            expanded.extend((hit, branch_id) for hit in leftovers)
        return expanded

    @staticmethod
    def _close_and_order_dependencies(selected: list[RetrievalHit],
                                      hits: list[RetrievalHit],
                                      target_effects: list[dict]) -> list[RetrievalHit]:
        """补齐前置能力并按 Effect→Precondition 数据流拓扑排序。

        目标 Effect 的覆盖只决定“需要哪些终点能力”；执行顺序不能继续沿用
        retrieval score。共享前置状态的节点必须先纳入对应 producer；同层节点
        再按任务目标 Effect 的声明顺序稳定排序。
        """
        def produces(producer: RetrievalHit, consumer: RetrievalHit) -> bool:
            return any(match_effect_contract(effect, precondition).passed
                       for effect in getattr(producer.obj, "effects", []) or []
                       for precondition in getattr(consumer.obj, "preconditions", []) or [])

        def keys(items: list[dict]) -> set[str]:
            return {_canonical_predicate(str(item.get("predicate") or ""))
                    for item in items if isinstance(item, dict) and item.get("predicate")}

        pool = list(hits)
        chosen = list(selected)
        chosen_ids = {hit.obj.ref.logical_id for hit in chosen}
        # 递归加入能满足已选节点前置条件的最高分 producer。环境初始事实等
        # 外部条件可能没有 producer，保留给运行时验证/位置发现处理。
        changed = True
        while changed:
            changed = False
            produced = set().union(*(keys(getattr(hit.obj, "effects", []))
                                     for hit in chosen)) if chosen else set()
            required = set().union(*(keys(getattr(hit.obj, "preconditions", []))
                                     for hit in chosen)) if chosen else set()
            for predicate in sorted(required - produced):
                consumers = [hit for hit in chosen
                             if predicate in keys(getattr(hit.obj, "preconditions", []))]
                producers = [hit for hit in pool
                             if any(produces(hit, consumer) for consumer in consumers)]
                if not producers:
                    continue
                producer = max(producers, key=lambda hit: hit.score)
                logical = producer.obj.ref.logical_id
                if logical not in chosen_ids:
                    chosen.append(producer)
                    chosen_ids.add(logical)
                    changed = True

        target_rank = {
            _canonical_predicate(str(effect.get("predicate") or "")): index
            for index, effect in enumerate(target_effects)
            if isinstance(effect, dict) and effect.get("predicate")
        }
        effects_by_id = {hit.obj.ref.logical_id: keys(getattr(hit.obj, "effects", []))
                         for hit in chosen}
        pre_by_id = {hit.obj.ref.logical_id: keys(getattr(hit.obj, "preconditions", []))
                     for hit in chosen}
        by_id = {hit.obj.ref.logical_id: hit for hit in chosen}
        edges: dict[str, set[str]] = {logical: set() for logical in by_id}
        indegree = {logical: 0 for logical in by_id}
        for producer in by_id:
            for consumer in by_id:
                if producer == consumer:
                    continue
                if produces(by_id[producer], by_id[consumer]):
                    if consumer not in edges[producer]:
                        edges[producer].add(consumer)
                        indegree[consumer] += 1

        def rank(logical: str) -> tuple[int, float, str]:
            ranks = [target_rank[p] for p in effects_by_id[logical] if p in target_rank]
            # 非目标 producer 优先；目标节点遵循 target_effects 的语义顺序。
            return (min(ranks) + 1 if ranks else 0, -by_id[logical].score, logical)

        ready = sorted((logical for logical, degree in indegree.items() if degree == 0),
                       key=rank)
        ordered: list[RetrievalHit] = []
        while ready:
            logical = ready.pop(0)
            ordered.append(by_id[logical])
            for consumer in sorted(edges[logical]):
                indegree[consumer] -= 1
                if indegree[consumer] == 0:
                    ready.append(consumer)
                    ready.sort(key=rank)
        # 异常契约形成环时保持确定性，不静默丢节点。
        if len(ordered) != len(chosen):
            seen = {hit.obj.ref.logical_id for hit in ordered}
            ordered.extend(sorted((hit for hit in chosen
                                   if hit.obj.ref.logical_id not in seen),
                                  key=lambda hit: rank(hit.obj.ref.logical_id)))
        return ordered

    def _composite_effect_keys(self, composite: CompositeSkill) -> set[str]:
        """Composite Effect 闭包：由有序子 Atomic 的核心 Effect 合并得到。"""
        keys: set[str] = set()
        for step in composite.step_instances():
            atomic = self.registry.get(SkillRef.parse(str(step["node_ref"])))
            if atomic is None:
                continue
            for effect in list(getattr(atomic, "effects", []) or []):
                if isinstance(effect, dict) and effect.get("predicate"):
                    keys.add(_canonical_predicate(str(effect["predicate"])))
        return keys

    def _composite_effect_counts(self, composite: CompositeSkill) -> dict[str, int]:
        counts: dict[str, int] = {}
        for step in composite.step_instances():
            atomic = self.registry.get(SkillRef.parse(str(step["node_ref"])))
            if atomic is None:
                continue
            for effect in list(getattr(atomic, "effects", []) or []):
                if isinstance(effect, dict) and effect.get("predicate"):
                    key = _canonical_predicate(str(effect["predicate"]))
                    counts[key] = counts.get(key, 0) + 1
        return counts

    @staticmethod
    def _target_effect_counts(target_effects: list[dict]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for effect in target_effects:
            if not isinstance(effect, dict) or not effect.get("predicate"):
                continue
            key = _canonical_predicate(str(effect["predicate"]))
            counts[key] = counts.get(key, 0) + max(
                1, int(effect.get("cardinality", 1) or 1))
        return counts

    @staticmethod
    def _target_effect_keys(target_effects: list[dict]) -> set[str]:
        return {
            _canonical_predicate(str(effect.get("predicate") or ""))
            for effect in target_effects
            if isinstance(effect, dict) and effect.get("predicate")
        }

    def _composite_covers_targets(self, composite: CompositeSkill,
                                  target_effects: list[dict]) -> bool:
        learned = list((composite.validator or {}).get("target_effects") or [])
        if not learned:
            return False
        return all(any(match_effect_contract(candidate, target).passed
                       for candidate in learned)
                   for target in target_effects)

    def _composite_goal_relevant(self, composite: CompositeSkill,
                                 target_effects: list[dict]) -> bool:
        """Require the learned goal contract to be no broader than this task.

        Structural helper Effects (navigation, open state, possession) are not
        inferred from names here.  The comparison uses the goal contract saved
        when the successful Composite occurrence was built.  Thus a workflow
        learned for ``changed + delivered`` cannot be selected for a task that
        asks only for ``delivered``.
        """
        learned = list((composite.validator or {}).get("target_effects") or [])
        if not learned:
            # Legacy Composite versions without a declared goal contract are
            # auditable history, not safe normal-planner candidates.
            return False
        wanted = self._target_effect_counts(target_effects)
        declared = self._target_effect_counts(learned)
        return all(predicate in wanted and required <= wanted[predicate]
                   for predicate, required in declared.items())

    def _composite_selection_rank(self, hit: RetrievalHit,
                                  target_effects: list[dict]) -> tuple:
        """Prefer exact/minimal verified contracts before lexical utility."""
        declared = self._target_effect_counts(
            list((hit.obj.validator or {}).get("target_effects") or []))
        wanted = self._target_effect_counts(target_effects)
        mismatch = sum(abs(declared.get(key, 0) - wanted.get(key, 0))
                       for key in set(declared) | set(wanted))
        return (-mismatch, -len(hit.obj.step_instances()), hit.score)

    def _composite_target_coverage(self, composite: CompositeSkill,
                                   target_effects: list[dict]) -> float:
        target = self._target_effect_counts(target_effects)
        if not target:
            return 1.0
        actual = self._composite_effect_counts(composite)
        covered = sum(min(required, actual.get(predicate, 0))
                      for predicate, required in target.items())
        return covered / max(sum(target.values()), 1)

    def _partial_task_gap_source_closed(self, task,
                                        composite: CompositeSkill) -> bool:
        """Require every future TaskGap target to have concrete task anchors.

        A partial Composite is useful only when the missing terminal contract
        can be stated as a grounded task-level goal.  We deliberately do not
        invent a local ``runtime.dynamic.<effect>`` occurrence here: the
        system measures the real pre-gap state and appends one explicit TaskGap
        occurrence after the verified parent occurrences finish.
        """
        actual = self._composite_effect_counts(composite)
        consumed: dict[str, int] = {}
        for target in list(task.target_effects or []):
            if not isinstance(target, dict):
                continue
            predicate = _canonical_predicate(
                str(target.get("predicate") or ""))
            required = max(1, int(target.get("cardinality", 1) or 1))
            available = max(0, int(actual.get(predicate, 0)))
            already_consumed = consumed.get(predicate, 0)
            covered = max(0, min(required, available - already_consumed))
            consumed[predicate] = already_consumed + covered
            for _ in range(required - covered):
                missing = dict(target)
                missing["cardinality"] = 1
                params = self._bind_dynamic_effect_params(task, missing)
                if any(not is_concrete_binding(params.get(slot))
                       for slot in semantic_required_slots([missing])):
                    return False
        return True

    @staticmethod
    def _bind_dynamic_effect_params(task, effect: dict[str, Any]) -> dict[str, Any]:
        """用规范任务参数绑定 dynamic gap，不把 `$name` 留给 Runtime。"""
        params = dict(task.context.get("params") or {})
        placeholder_names: list[str] = []
        for value in dict(effect.get("args") or {}).values():
            if not isinstance(value, str) or not value.startswith("$"):
                continue
            path = value[1:].split(".")
            name = path[-1]
            if name and name not in placeholder_names:
                placeholder_names.append(name)
        for name, value in parse_goal_params(task.goal, placeholder_names).items():
            params.setdefault(name, value)
        return params

    @staticmethod
    def _runtime_edges(nodes: list[PlannedNode],
                       composite_edges: list[GraphEdge] | None = None) -> list[GraphEdge]:
        """生成可执行运行时边；Composite 边按实例顺序映射后保留类型语义。"""
        edges: list[GraphEdge] = []
        by_origin_step = {node.origin_step_id: node for node in nodes
                          if node.origin_step_id}
        if composite_edges:
            for edge in composite_edges:
                source = by_origin_step.get(edge.source_step)
                target = by_origin_step.get(edge.target_step)
                if source is None or target is None:
                    raise PlanCompilationError(
                        f"runtime_edge_endpoint_missing:{edge.source_step}->{edge.target_step}")
                edges.append(GraphEdge(
                    source=str(source.ref), target=str(target.ref), type=edge.type,
                    subtype=edge.subtype, scope="runtime",
                    source_step=source.step_id, target_step=target.step_id,
                    condition=dict(edge.condition), mapping=dict(edge.mapping),
                    policy=dict(edge.policy), evidence=list(edge.evidence),
                    metadata={**edge.metadata, "composite_edge_id": edge.edge_id},
                ))
        # 只在没有控制连接的位置补 NEXT；动态 gap 总是接到前一节点。
        connected = {(e.source_step, e.target_step) for e in edges
                     if e.category == "control"}
        for left, right in zip(nodes, nodes[1:]):
            if (left.step_id, right.step_id) in connected:
                continue
            edges.append(GraphEdge(
                source=str(left.ref), target=str(right.ref), type=EdgeType.NEXT,
                scope="runtime", source_step=left.step_id, target_step=right.step_id,
            ))
        return edges

    def _llm_plan(self, task, hits: list[RetrievalHit]) -> list[RetrievalHit] | None:
        """LLM 只规划 Atomic Skill 子集与顺序（§13）；失败时回退规则排序。"""
        if self.llm is None:
            return None
        skill_lines = []
        for hit in hits:
            atomic = hit.obj
            summary = getattr(atomic, "summary", "")
            inputs = [i.get("name", "") for i in getattr(atomic, "inputs", [])]
            skill_lines.append(
                f"- logical_id: {atomic.ref.logical_id} | inputs: {inputs} | summary: {summary}"
            )
        prompt = (
            f"Task goal: {task.goal}\n"
            "Available atomic skills:\n" + "\n".join(skill_lines)
        )
        try:
            response = self.llm.generate(instructions=_PLAN_PROMPT, input_text=prompt)
            data = _parse_plan_json(response.text)
        except Exception:  # noqa: BLE001
            return None
        if not data:
            return None
        by_id = {h.obj.ref.logical_id: h for h in hits}
        selected = [by_id[i] for i in data if i in by_id]
        if not selected:
            return None
        # LLM 规划结果只提供顺序；参数绑定仍由 _bind_params 完成
        return selected

    @staticmethod
    def _cover_target_effects(target_effects: list[dict], hits: list[RetrievalHit],
                              bindings: dict[str, Any] | None = None
                              ) -> list[RetrievalHit]:
        unmatched = list(target_effects)
        selected: list[RetrievalHit] = []
        for hit in sorted(hits, key=lambda h: h.score, reverse=True):
            matched = [target for target in unmatched
                       if any(match_effect_contract(
                           effect, {**target, "cardinality": 1,
                                    "distinct_by": ""}, bindings).passed
                              for effect in getattr(hit.obj, "effects", []) or [])]
            if matched:
                selected.append(hit)
                unmatched = [target for target in unmatched if target not in matched]
                if not unmatched:
                    break
        if selected:
            return selected
        # No matching target producer is safer than arbitrary high-scoring
        # capabilities. The caller materializes explicit Dynamic goal gaps.
        return []

    # ------------------------------------------------------------------
    def _bind_params(self, task, atomic: AbstractAtomicSkill) -> dict[str, Any]:
        """Bind from explicit goal roles and learned contract evidence only."""
        context_params = dict(task.context.get("params") or {})
        input_names = [str(i.get("name", "")) for i in atomic.inputs]
        params: dict[str, Any] = {}
        for name in input_names:
            if name in context_params:
                params[name] = context_params[name]
        goal_params = parse_goal_params(task.goal, input_names)
        for name, value in goal_params.items():
            params.setdefault(name, value)
        learned_families = dict(
            (getattr(atomic, "metadata", {}) or {}).get(
                "observed_parameter_families") or {})
        candidates = _task_entity_candidates(task)
        for name in input_names:
            if params.get(name) not in (None, ""):
                continue
            families = [str(value) for value in learned_families.get(name, [])
                        if value not in (None, "")]
            matches = [candidate for candidate in candidates
                       if any(_same_entity_family(candidate, family)
                              for family in families)]
            if len(matches) == 1:
                params[name] = matches[0]

        # A position may already be known in the current structured state. This
        # is evidence binding, not benchmark-specific navigation advice.
        for name in input_names:
            if params.get(name) not in (None, "") or not name.endswith("_location"):
                continue
            entity_role = name[:-len("_location")]
            entity = params.get(entity_role)
            location = _state_location_of(task.state, entity)
            if location:
                params[name] = location
        return params


def _task_entity_candidates(task: Any) -> list[str]:
    values: list[Any] = []
    context = dict(getattr(task, "context", {}) or {})
    roles = dict(context.get("goal_roles") or {})
    values.extend(roles.values())
    values.extend(context.get("goal_entities") or [])
    values.extend(context.get("exposed_entities") or [])
    values.extend(dict(context.get("params") or {}).values())
    candidates: list[str] = []
    for value in values:
        normalized = _normalize_entity(value)
        if normalized and not normalized.isdigit() and normalized not in candidates:
            candidates.append(normalized)
    return candidates


def _normalize_entity(value: Any) -> str:
    return re.sub(r"\s+", "_", str(value or "").strip().lower())


def _same_entity_family(left: Any, right: Any) -> bool:
    left_value, right_value = _normalize_entity(left), _normalize_entity(right)
    if not left_value or not right_value:
        return False
    strip_instance = lambda value: re.sub(r"_\d+$", "", value)
    return strip_instance(left_value) == strip_instance(right_value)


def _state_location_of(state: dict[str, Any], entity: Any) -> str:
    wanted = _normalize_entity(entity)
    if not wanted:
        return ""
    matches: list[str] = []
    for fact in (state or {}).get("facts", []) or []:
        match = re.fullmatch(r"object_at\((.+?),\s*(.+?)\)", str(fact))
        if match and _same_entity_family(match.group(1), wanted):
            matches.append(_normalize_entity(match.group(2)))
    return matches[0] if len(set(matches)) == 1 else ""


def _extract_entity_names(text: str) -> list[str]:
    """从 goal 文本提取实体词（作为检索的 available_inputs 弱信号）。"""
    words = re.findall(r"[a-z0-9_ ]{2,30}", str(text).lower())
    entities: list[str] = []
    for phrase in words:
        phrase = phrase.strip()
        if 2 <= len(phrase) <= 30 and phrase not in entities:
            entities.append(phrase)
    return entities[:20]


def _parse_plan_json(text: str) -> list[str] | None:
    """从 LLM 输出解析 {skills: [{logical_id,...}]} 并返回 logical_id 列表。"""
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        data = json.loads(text[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    skills = data.get("skills") if isinstance(data, dict) else None
    if not isinstance(skills, list):
        return None
    ids = [str(s.get("logical_id")) for s in skills if isinstance(s, dict) and s.get("logical_id")]
    return ids or None


def _add_distinct_group(node: PlannedNode, role: str, group_id: str) -> bool:
    """Attach one normalized group once and report whether IR changed."""
    role_name = str(role or "")
    group_name = str(group_id or "")
    if not role_name or not group_name:
        return False
    groups = node.distinct_bindings.setdefault(role_name, [])
    if group_name in groups:
        return False
    groups.append(group_name)
    return True


def _set_distinct_branch_id(
        node: PlannedNode, group_id: str, branch_id: str) -> bool:
    group_name = str(group_id or "")
    branch_name = str(branch_id or "")
    if not group_name or not branch_name:
        return False
    current = node.distinct_branch_ids.get(group_name)
    if current == branch_name:
        return False
    if current:
        # Conflicting projections are kept explicit instead of silently
        # replacing one cardinality occurrence with another.
        return False
    node.distinct_branch_ids[group_name] = branch_name
    return True


def _deduplicate_runtime_edges(edges: list[GraphEdge]) -> list[GraphEdge]:
    """Keep edge semantics exact while removing duplicate synthesis results."""
    unique: list[GraphEdge] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for edge in edges:
        requirement = dict((edge.metadata or {}).get("requirement") or {})
        key = (
            edge.type.value,
            str(edge.source_step),
            str(edge.target_step),
            json.dumps(edge.mapping or {}, sort_keys=True, ensure_ascii=False),
            json.dumps(requirement, sort_keys=True, ensure_ascii=False),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(edge)
    return unique


def _canonical_predicate(name: str) -> str:
    aliases = {
        "object.in_receptacle": "object.at_location",
        "object.in.receptacle": "object.at_location",
        "object_in_receptacle": "object.at_location",
        "object.in_container": "object.at_location",
        "object.in.container": "object.at_location",
        "object_in_container": "object.at_location",
    }
    return aliases.get(str(name), str(name))
