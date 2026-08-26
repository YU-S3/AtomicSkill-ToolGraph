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
from ..core.config import SystemConfig
from ..core.edge_ir import GraphEdge
from ..core.llm import LLM
from ..core.refs import SkillRef
from ..core.skill_ir import AbstractAtomicSkill, CompositeSkill
from ..core.status import EdgeType, SkillNodeKind, SkillStatus
from ..graph.graph import composite_node_order
from ..graph.registry import RetrievalHit, SkillGraphRegistry
from .runtime_graph import PlannedNode, RuntimePlan

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
            return RuntimePlan(start_mode="cold", notes=["no_capability_retrieved"])

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
                active = [h for h in complete
                          if h.obj.status.value == "active"]
                # Online may perform one controlled candidate exploration to
                # obtain independent support. Frozen replay stays active-only.
                selectable = active or [
                    h for h in complete if h.obj.status.value == "draft"]
                if selectable:
                    selected = max(
                        selectable,
                        key=lambda h: self._composite_selection_rank(
                            h, task.target_effects),
                    )
                    plan = self._plan_from_composite(task, selected, retrieved)
                    plan.notes.append("composite_effect_complete")
                    if selected.obj.status.value == "draft":
                        plan.notes.append("controlled_candidate_exploration")
                        if self.config.freeze_skills:
                            plan.notes.append("strict_frozen_candidate_replay")
                    return plan

            eligible = [h for h in composites
                        if (h.score >= self.composite_min_score
                            and self._composite_goal_relevant(
                                h.obj, task.target_effects))]
            if eligible:
                # 没有完整 Composite 时才允许用 partial + dynamic gap。若 gap
                # 参数无法完全绑定，则继续走 Atomic greedy，而不是执行占位符。
                partial = max(
                    eligible,
                    key=lambda h: (self._composite_target_coverage(
                        h.obj, task.target_effects), h.score),
                )
                partial_plan = self._plan_from_composite(task, partial, retrieved)
                if not self._has_unbound_dynamic_gap(partial_plan.nodes):
                    partial_plan.notes.append("composite_partial_with_bound_gap")
                    return partial_plan

        # 2. Abstract Atomic 直接规划
        atomics = [h for h in retrieved if h.kind == SkillNodeKind.ABSTRACT_ATOMIC]
        if self.config.freeze_skills:
            atomics = [h for h in atomics if h.obj.status == SkillStatus.ACTIVE]
        if not atomics:
            dynamic = self._append_dynamic_gaps(task, [])
            return RuntimePlan(
                start_mode="cold", nodes=dynamic,
                edges=self._runtime_edges(dynamic),
                retrieved=[h.to_dict() for h in retrieved],
                notes=["no_reusable_target_producer_dynamic_only"],
            )

        nodes = self._plan_from_atomics(task, atomics)
        nodes = self._append_dynamic_gaps(task, nodes)
        return RuntimePlan(
            start_mode="warm",
            nodes=nodes,
            edges=self._runtime_edges(nodes),
            retrieved=[h.to_dict() for h in retrieved],
            notes=["atomic_only_plan", "partial_composite_rejected_if_unbound"],
        )

    # ------------------------------------------------------------------
    def _plan_from_composite(self, task, hit: RetrievalHit,
                             retrieved: list[RetrievalHit]) -> RuntimePlan:
        composite: CompositeSkill = hit.obj
        order, _report = composite_node_order(composite, self.registry)
        nodes: list[PlannedNode] = []
        steps_by_logical: dict[str, list[dict[str, Any]]] = {}
        for step in composite.step_instances():
            logical_id = str(step.get("node_ref") or "").rsplit("@", 1)[0]
            steps_by_logical.setdefault(logical_id, []).append(step)
        occurrences: dict[str, int] = {}
        for index, logical in enumerate(order):
            obj = self.registry.get_recommended(logical)
            if obj is None:
                continue
            occurrence = occurrences.get(logical, 0)
            occurrences[logical] = occurrence + 1
            stored_steps = steps_by_logical.get(logical, [])
            if occurrence < len(stored_steps):
                # The occurrence role map is authoritative.  Falling back to
                # global task bindings here used to bind every navigation node
                # to the final destination and erased source/station roles.
                stored = dict(stored_steps[occurrence].get("params") or {})
                params = self._resolve_composite_params(stored, task)
                fallback = self._bind_params(task, obj)
                for key, value in list(params.items()):
                    if not (isinstance(value, str)
                            and value.startswith("$flow.")):
                        continue
                    flow_role = value[len("$flow."):]
                    replacement = ((task.context.get("params") or {}).get(flow_role)
                                   or fallback.get(key) or fallback.get(flow_role))
                    if replacement not in (None, ""):
                        params[key] = replacement
                    else:
                        params.pop(key, None)
            else:
                params = self._bind_params(task, obj)
            nodes.append(PlannedNode(ref=obj.ref, step_id=f"step_{index:03d}",
                                     params=params, source="composite",
                                     target_effects=list(getattr(obj, "effects", []))))
        nodes = self._prune_unbound_support_nodes(
            task, self._append_dynamic_gaps(task, nodes))
        return RuntimePlan(
            start_mode="warm",
            composite_ref=str(composite.ref),
            nodes=nodes,
            edges=self._runtime_edges(nodes, composite.edge_objects()),
            retrieved=[h.to_dict() for h in retrieved],
            notes=[f"composite_plan:{composite.ref.logical_id}"],
        )

    def _prune_unbound_support_nodes(self, task,
                                     nodes: list[PlannedNode]) -> list[PlannedNode]:
        """Drop only unexecutable helper occurrences subsumed downstream.

        A hidden source/resource location can make a standalone helper
        occurrence unbound, while the later state-changing capability can
        discover that location itself.  The decision is contract-driven: the
        node must have missing declared inputs, produce no requested goal, and
        all of its Effects must be consumed as later Preconditions.
        """
        target_keys = self._target_effect_keys(task.target_effects)
        kept: list[PlannedNode] = []
        for index, node in enumerate(nodes):
            atomic = None if node.dynamic else self.registry.get_recommended(
                node.ref.logical_id)
            if atomic is None:
                kept.append(node)
                continue
            required_inputs = {str(item.get("name") or "")
                               for item in (atomic.inputs or [])
                               if isinstance(item, dict) and item.get("name")}
            missing = {name for name in required_inputs
                       if node.params.get(name) in (None, "")}
            effects = {
                _canonical_predicate(str(item.get("predicate") or ""))
                for item in (atomic.effects or []) if isinstance(item, dict)
            }
            later_preconditions = {
                _canonical_predicate(str(item.get("predicate") or ""))
                for later in nodes[index + 1:]
                for later_obj in ([self.registry.get_recommended(
                    later.ref.logical_id)] if not later.dynamic else [])
                if later_obj is not None
                for item in (later_obj.preconditions or [])
                if isinstance(item, dict)
            }
            removable = (bool(missing) and bool(effects)
                         and not (effects & target_keys)
                         and effects.issubset(later_preconditions))
            if not removable:
                kept.append(node)
        for index, node in enumerate(kept):
            node.step_id = f"step_{index:03d}"
        return kept

    @staticmethod
    def _resolve_composite_params(stored: dict[str, Any], task) -> dict[str, Any]:
        """Resolve persisted semantic roles without rebinding every same-kind step globally."""
        context = dict(task.context.get("params") or {})
        resolved: dict[str, Any] = {}
        for key, value in stored.items():
            if isinstance(value, str) and value.startswith("$task."):
                role = value[len("$task."):]
                # A hidden source location must be discovered, not rebound from
                # a semantically different destination role. This also guards
                # replay of older banks containing an unsafe role mapping.
                if (str(key) in {"object_location", "source_location"}
                        and role in {"target_location", "target_receptacle",
                                     "destination", "destination_location"}):
                    continue
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
            selected = self._cover_target_effects(task.target_effects, hits)
            selected = self._close_and_order_dependencies(selected, hits,
                                                          task.target_effects)
            selected = self._expand_cardinality_workflow(
                selected, task.target_effects)
        else:
            selected = self._llm_plan(task, hits) or hits[:3]
        nodes: list[PlannedNode] = []
        for index, hit in enumerate(selected):
            atomic = hit.obj
            params = self._bind_params(task, atomic)
            nodes.append(PlannedNode(ref=atomic.ref, step_id=f"step_{index:03d}",
                                     params=params, source="retrieval",
                                     target_effects=list(getattr(atomic, "effects", []))))
        return nodes

    @staticmethod
    def _expand_cardinality_workflow(
            selected: list[RetrievalHit],
            target_effects: list[dict[str, Any]]) -> list[RetrievalHit]:
        """Repeat a minimal causal branch for an explicit N-object goal.

        Repeating only a terminal node can violate its producer dependencies;
        repeating the already dependency-closed ordered branch preserves the
        executable data flow for any explicit cardinality target.
        """
        repeat = max(
            [max(1, int(effect.get("cardinality", 1) or 1))
             for effect in target_effects if isinstance(effect, dict)] or [1])
        if repeat <= 1 or not selected:
            return selected
        return [hit for _ in range(repeat) for hit in selected]

    @staticmethod
    def _close_and_order_dependencies(selected: list[RetrievalHit],
                                      hits: list[RetrievalHit],
                                      target_effects: list[dict]) -> list[RetrievalHit]:
        """补齐前置能力并按 Effect→Precondition 数据流拓扑排序。

        目标 Effect 的覆盖只决定“需要哪些终点能力”；执行顺序不能继续沿用
        retrieval score。共享前置状态的节点必须先纳入对应 producer；同层节点
        再按任务目标 Effect 的声明顺序稳定排序。
        """
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
                producers = [hit for hit in pool
                             if predicate in keys(getattr(hit.obj, "effects", []))]
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
                if effects_by_id[producer] & pre_by_id[consumer]:
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

    def _append_dynamic_gaps(self, task, nodes: list[PlannedNode]) -> list[PlannedNode]:
        """把未被检索能力覆盖的目标效果显式化，供逐节点动态执行。"""
        if not task.target_effects:
            return nodes
        covered = {_canonical_predicate(str(effect.get("predicate") or ""))
                   for node in nodes for effect in node.target_effects
                   if isinstance(effect, dict) and effect.get("predicate")}
        for effect in task.target_effects:
            predicate = str(effect.get("predicate") or "unknown")
            canonical = _canonical_predicate(predicate)
            if canonical in covered:
                continue
            slug = re.sub(r"[^a-z0-9_.]+", "_", predicate.lower()).strip("_") or "effect"
            gap = PlannedNode(
                ref=SkillRef(f"runtime.dynamic.{slug}", "0.0.0"),
                source="dynamic_gap",
                params=self._bind_dynamic_effect_params(task, effect),
                target_effects=[dict(effect)], dynamic=True,
            )
            # Missing target Effects participate in the same declared goal
            # order as learned producers.  Appending every gap at the end made
            # transformation goals execute after final delivery.
            target_order = {
                _canonical_predicate(str(item.get("predicate") or "")): rank
                for rank, item in enumerate(task.target_effects)
                if isinstance(item, dict) and item.get("predicate")
            }
            gap_rank = target_order.get(canonical, len(target_order))
            insert_at = len(nodes)
            for index, node in enumerate(nodes):
                ranks = [target_order.get(
                    _canonical_predicate(str(item.get("predicate") or "")),
                    -1)
                    for item in node.target_effects if isinstance(item, dict)]
                ranks = [rank for rank in ranks if rank >= 0]
                if ranks and min(ranks) > gap_rank:
                    insert_at = index
                    break
            nodes.insert(insert_at, gap)
            covered.add(canonical)
        for index, node in enumerate(nodes):
            node.step_id = f"step_{index:03d}"
        return nodes

    def _composite_effect_keys(self, composite: CompositeSkill) -> set[str]:
        """Composite Effect 闭包：由有序子 Atomic 的核心 Effect 合并得到。"""
        keys: set[str] = set()
        for step in composite.step_instances():
            logical = SkillRef.parse(str(step["node_ref"])).logical_id
            atomic = self.registry.get_recommended(logical)
            for effect in list(getattr(atomic, "effects", []) or []):
                if isinstance(effect, dict) and effect.get("predicate"):
                    keys.add(_canonical_predicate(str(effect["predicate"])))
        return keys

    def _composite_effect_counts(self, composite: CompositeSkill) -> dict[str, int]:
        counts: dict[str, int] = {}
        for step in composite.step_instances():
            logical = SkillRef.parse(str(step["node_ref"])).logical_id
            atomic = self.registry.get_recommended(logical)
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
        target = self._target_effect_counts(target_effects)
        actual = self._composite_effect_counts(composite)
        return all(actual.get(predicate, 0) >= required
                   for predicate, required in target.items())

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
            return True  # compatibility for historical non-v2 artifacts
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
    def _has_unbound_dynamic_gap(nodes: list[PlannedNode]) -> bool:
        from ..core.predicates import bind_args
        for node in nodes:
            if not node.dynamic:
                continue
            for effect in node.target_effects:
                bound = bind_args(dict(effect.get("args") or {}), node.params,
                                  node.params)
                if any(isinstance(value, str) and value.startswith("$")
                       for value in bound.values()):
                    return True
        return False

    @staticmethod
    def _runtime_edges(nodes: list[PlannedNode],
                       composite_edges: list[GraphEdge] | None = None) -> list[GraphEdge]:
        """生成可执行运行时边；Composite 边按实例顺序映射后保留类型语义。"""
        edges: list[GraphEdge] = []
        by_ref: dict[str, list[PlannedNode]] = {}
        for node in nodes:
            by_ref.setdefault(str(node.ref), []).append(node)
        if composite_edges:
            cursors: dict[str, int] = {}
            for edge in composite_edges:
                source_nodes = by_ref.get(edge.source, [])
                target_nodes = by_ref.get(edge.target, [])
                if not source_nodes or not target_nodes:
                    continue
                si = min(cursors.get(f"s:{edge.source}", 0), len(source_nodes) - 1)
                ti = min(cursors.get(f"t:{edge.target}", 0), len(target_nodes) - 1)
                source, target = source_nodes[si], target_nodes[ti]
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
    def _cover_target_effects(target_effects: list[dict], hits: list[RetrievalHit]) -> list[RetrievalHit]:
        def effect_keys(effects: list[dict]) -> set[str]:
            return {
                _canonical_predicate(str(e.get("predicate") or ""))
                for e in effects if isinstance(e, dict)
            }
        target = effect_keys(target_effects)
        covered: set[str] = set()
        selected: list[RetrievalHit] = []
        for hit in sorted(hits, key=lambda h: h.score, reverse=True):
            keys = effect_keys(getattr(hit.obj, "effects", []))
            if keys & target and not keys.issubset(covered):
                selected.append(hit)
                covered |= keys
                if covered >= target:
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


def _canonical_predicate(name: str) -> str:
    aliases = {
        "object.in_receptacle": "object.at_location",
        "object.in_container": "object.at_location",
    }
    return aliases.get(str(name), str(name))
