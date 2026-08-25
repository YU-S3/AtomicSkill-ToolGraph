"""Atomic Planner（设计文档 v2.0 §21）。

检索顺序（§21.2）：
    解析目标状态 → task_type 弱召回 → Composite → Abstract Atomic
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
from ..core.status import EdgeType, SkillNodeKind
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
            "task_type": task.task_type,
            "state": task.state,
            "available_inputs": list((task.context.get("params") or {}).keys())
            + _extract_entity_names(task.goal),
            "target_effects": task.target_effects,
        }
        hard_restrict = features.task_type_hard_restricted or not features.enable_cross_task_type_reuse
        # Composite 的充分性判断必须看到完整候选集，不能让词面得分较低但
        # Effect 完整的 Composite 在 top-k 截断时先被丢掉。
        retrieval_limit = max(self.config.retrieval_top_k,
                              len(self.registry.list_all()))
        retrieved = self.registry.retrieve(
            query, top_k=retrieval_limit,
            hard_restrict_task_type=hard_restrict,
            task_type_bonus=self.config.task_type_soft_bonus,
        )
        # 得分低于阈值 → 视为无可用能力（cold start，避免弱匹配负迁移）
        retrieved = [h for h in retrieved if h.score >= self.config.planning_min_score]
        if not retrieved:
            return RuntimePlan(start_mode="cold", notes=["no_capability_retrieved"])

        # 1. Composite 优先，但先做目标 Effect 充分性硬过滤。summary/utility
        # 只能在同等充分的候选之间重排，不能让两步链压过完整三步链。
        if features.enable_composite:
            composites = [h for h in retrieved if h.kind == SkillNodeKind.COMPOSITE]
            eligible = [h for h in composites if h.score >= self.composite_min_score]
            if self.config.freeze_skills:
                eligible = [h for h in eligible if (
                    h.obj.status.value == "active"
                    or (h.obj.status.value == "draft" and (
                        self.config.llm.mock
                        or bool((h.obj.metadata.get("candidate") or {}).get(
                            "semantic_extraction_validated")))))]
            if eligible:
                complete = [h for h in eligible
                            if self._composite_covers_targets(h.obj,
                                                              task.target_effects)]
                if complete:
                    active = [h for h in complete
                              if h.obj.status.value == "active"]
                    # Online may perform one controlled candidate exploration
                    # to obtain independent support. Frozen replay is strictly
                    # active-only and therefore cannot evaluate one-trace drafts.
                    selectable = active or [
                        h for h in complete if h.obj.status.value == "draft"]
                    if not selectable:
                        complete = []
                    else:
                        selected = max(selectable, key=lambda h: h.score)
                        plan = self._plan_from_composite(task, selected, retrieved)
                        plan.notes.append("composite_effect_complete")
                        if selected.obj.status.value == "draft":
                            plan.notes.append("controlled_candidate_exploration")
                            if self.config.freeze_skills:
                                plan.notes.append("strict_frozen_candidate_replay")
                        return plan

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
        if not atomics:
            return RuntimePlan(
                start_mode="cold",
                retrieved=[h.to_dict() for h in retrieved],
                notes=["retrieved_but_no_safe_atomic_plan"],
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
            params = self._bind_params(task, obj)
            occurrence = occurrences.get(logical, 0)
            occurrences[logical] = occurrence + 1
            stored_steps = steps_by_logical.get(logical, [])
            if occurrence < len(stored_steps):
                params.update(self._resolve_composite_params(
                    stored_steps[occurrence].get("params") or {}, task))
            nodes.append(PlannedNode(ref=obj.ref, step_id=f"step_{index:03d}",
                                     params=params, source="composite",
                                     target_effects=list(getattr(obj, "effects", []))))
        nodes = self._append_dynamic_gaps(task, nodes)
        return RuntimePlan(
            start_mode="warm",
            composite_ref=str(composite.ref),
            nodes=nodes,
            edges=self._runtime_edges(nodes, composite.edge_objects()),
            retrieved=[h.to_dict() for h in retrieved],
            notes=[f"composite_plan:{composite.ref.logical_id}"],
        )

    @staticmethod
    def _resolve_composite_params(stored: dict[str, Any], task) -> dict[str, Any]:
        """Resolve persisted semantic roles without rebinding every same-kind step globally."""
        context = dict(task.context.get("params") or {})
        resolved: dict[str, Any] = {}
        for key, value in stored.items():
            if isinstance(value, str) and value.startswith("$task."):
                role = value[len("$task."):]
                # Source location is hidden in ALFWorld and must be discovered.
                # Never reinterpret the final destination as an Acquire source,
                # including when replaying an older bank that persisted this
                # erroneous role mapping.
                if (str(key) in {"object_location", "source_location"}
                        and role in {"target_location", "target_receptacle",
                                     "destination", "destination_location"}):
                    continue
                if context.get(role) not in (None, ""):
                    resolved[str(key)] = context[role]
            elif value not in (None, ""):
                resolved[str(key)] = value
        return resolved

    def _plan_from_atomics(self, task, hits: list[RetrievalHit]) -> list[PlannedNode]:
        # 最小充分：greedy 覆盖 target_effects（若提供），否则取 top-k 中得分 > 阈值的
        if task.target_effects:
            selected = self._cover_target_effects(task.target_effects, hits)
            selected = self._close_and_order_dependencies(selected, hits,
                                                          task.target_effects)
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
    def _close_and_order_dependencies(selected: list[RetrievalHit],
                                      hits: list[RetrievalHit],
                                      target_effects: list[dict]) -> list[RetrievalHit]:
        """补齐前置能力并按 Effect→Precondition 数据流拓扑排序。

        目标 Effect 的覆盖只决定“需要哪些终点能力”；执行顺序不能继续沿用
        retrieval score。比如 Heat 与 Place 都依赖 ``agent.holds``，因此必须先
        纳入 Acquire；同层节点再按任务目标 Effect 的声明顺序稳定排序，得到
        Acquire→Heat→Place，而不是 Place→Heat。
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
        if not task.target_effects or not nodes:
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
            nodes.append(PlannedNode(
                ref=SkillRef(f"runtime.dynamic.{slug}", "0.0.0"),
                step_id=f"step_{len(nodes):03d}", source="dynamic_gap",
                params=self._bind_dynamic_effect_params(task, effect),
                target_effects=[dict(effect)], dynamic=True,
            ))
            covered.add(canonical)
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

    @staticmethod
    def _target_effect_keys(target_effects: list[dict]) -> set[str]:
        return {
            _canonical_predicate(str(effect.get("predicate") or ""))
            for effect in target_effects
            if isinstance(effect, dict) and effect.get("predicate")
        }

    def _composite_covers_targets(self, composite: CompositeSkill,
                                  target_effects: list[dict]) -> bool:
        target = self._target_effect_keys(target_effects)
        return not target or target.issubset(self._composite_effect_keys(composite))

    def _composite_target_coverage(self, composite: CompositeSkill,
                                   target_effects: list[dict]) -> float:
        target = self._target_effect_keys(target_effects)
        if not target:
            return 1.0
        return len(target & self._composite_effect_keys(composite)) / len(target)

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
            f"Task type: {task.task_type}\n"
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
        return hits[:3]

    # ------------------------------------------------------------------
    def _bind_params(self, task, atomic: AbstractAtomicSkill) -> dict[str, Any]:
        """参数绑定：context.params 优先 → goal 文本弱解析。"""
        context_params = dict(task.context.get("params") or {})
        input_names = [str(i.get("name", "")) for i in atomic.inputs]
        params: dict[str, Any] = {}
        for name in input_names:
            if name in context_params:
                params[name] = context_params[name]
        goal_params = parse_goal_params(task.goal, input_names)
        for name, value in goal_params.items():
            params.setdefault(name, value)
        return params


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
