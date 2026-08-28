"""SkillGraph Registry：集中式长期图存储（设计文档 v2.0 §14、§40）。

存储布局（JSON 文件存储，第一版）：
    data/skill_graph/
    ├── graph.json                     # 节点索引 + 边
    ├── abstract_atomic/<id>/<ver>.json
    ├── implementation_atomic/<id>/<ver>.json
    └── composite/<id>/<ver>.json

版本不可变：注册即写新版本文件，历史版本不被物理覆盖（§38）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ..core.edge_ir import GraphEdge
from ..core.refs import SkillRef
from ..core.skill_ir import (
    AbstractAtomicSkill,
    CompositeSkill,
    ImplementationAtom,
    load_skill_from_dict,
)
from ..core.status import EdgeType, SkillNodeKind, SkillStatus
from ..persistence import atomic_write_json

_SKILL_KIND_DIR = {
    SkillNodeKind.ABSTRACT_ATOMIC: "abstract_atomic",
    SkillNodeKind.IMPLEMENTATION_ATOMIC: "implementation_atomic",
    SkillNodeKind.COMPOSITE: "composite",
}


@dataclass
class RetrievalHit:
    ref: SkillRef
    kind: SkillNodeKind
    obj: Any
    score: float = 0.0
    breakdown: dict[str, float] = field(default_factory=dict)
    matched_task_type: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": str(self.ref),
            "kind": self.kind.value,
            "score": round(self.score, 4),
            "breakdown": {k: round(v, 4) for k, v in self.breakdown.items()},
            "matched_task_type": self.matched_task_type,
        }


class SkillGraphRegistry:
    """集中式 Global SkillGraph 的文件存储实现。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.graph_path = self.root / "graph.json"
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.graph_path.exists():
            self._write_graph({"nodes": {}, "edges": []})

    # ------------------------------------------------------------------
    # 底层持久化
    # ------------------------------------------------------------------
    def _read_graph(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.graph_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        raw.setdefault("nodes", {})
        raw.setdefault("edges", [])
        return raw

    def _write_graph(self, graph: dict[str, Any]) -> None:
        atomic_write_json(self.graph_path, graph)

    def _version_path(self, kind: SkillNodeKind, logical_id: str, version: str) -> Path:
        return self.root / _SKILL_KIND_DIR[kind] / logical_id / f"{version}.json"

    def _load_obj(self, kind: SkillNodeKind, logical_id: str, version: str):
        path = self._version_path(kind, logical_id, version)
        if not path.exists():
            return None
        try:
            return load_skill_from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValueError):
            return None

    def _save_obj(self, obj) -> SkillRef:
        path = self._version_path(obj.kind, obj.ref.logical_id, obj.ref.version)
        atomic_write_json(path, obj.to_dict())
        return obj.ref

    def _touch(self) -> None:
        graph = self._read_graph()
        graph["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._write_graph(graph)

    # ------------------------------------------------------------------
    # 注册 / 读取
    # ------------------------------------------------------------------
    def register(self, obj) -> SkillRef:
        """注册新版本（不可变；同版本已存在时幂等返回）。"""
        errors = obj.validate()
        if errors:
            raise ValueError(f"{obj.ref.logical_id} 校验失败：{errors}")
        prior = self._load_obj(obj.kind, obj.ref.logical_id, obj.ref.version)
        if prior is not None:
            if prior.to_dict() == obj.to_dict():
                return obj.ref
            raise ValueError(
                f"immutable_version_collision: {obj.ref} 已存在且内容不同；"
                "请对齐复用或分配新版本")
        self._save_obj(obj)
        graph = self._read_graph()
        nodes = graph["nodes"]
        entry = nodes.get(obj.ref.logical_id) or {}
        versions = sorted(set(list(entry.get("versions") or []) + [obj.ref.version]),
                          key=_semver_key)
        entry.update({
            "kind": obj.kind.value,
            "current_version": obj.ref.version,
            "latest_version": obj.ref.version,
            "status": obj.status.value,
            "versions": versions,
        })
        version_statuses = dict(entry.get("version_statuses") or {})
        version_statuses[obj.ref.version] = obj.status.value
        entry["version_statuses"] = version_statuses
        # Draft/Shadow merely advance latest.  They must never displace an
        # executable Active recommendation.
        if obj.status == SkillStatus.ACTIVE:
            entry["recommended_version"] = obj.ref.version
        nodes[obj.ref.logical_id] = entry
        # 结构边
        if isinstance(obj, ImplementationAtom):
            self._upsert_edge(graph, GraphEdge(
                source=str(SkillRef(obj.ref.logical_id, obj.ref.version)),
                target=str(obj.abstract_ref), type=EdgeType.IMPLEMENTS,
                evidence=[str(obj.ref)],
            ))
        if isinstance(obj, CompositeSkill):
            for step in obj.step_instances():
                self._upsert_edge(graph, GraphEdge(
                    source=str(obj.ref), target=step["node_ref"],
                    type=EdgeType.CONTAINS, scope="global",
                    source_step=str(obj.ref), target_step=step["step_id"],
                    evidence=[str(obj.ref)],
                    metadata={"step_id": step["step_id"]},
                ))
            for edge in obj.edge_objects():
                payload = edge.to_dict()
                payload["edge_id"] = ""
                payload["metadata"] = {
                    **edge.metadata, "owner_ref": str(obj.ref),
                }
                self._upsert_edge(graph, GraphEdge.from_dict(payload))
        self._write_graph(graph)
        return obj.ref

    @staticmethod
    def _upsert_edge(graph: dict[str, Any], edge: GraphEdge) -> None:
        errors = edge.validate()
        if errors:
            raise ValueError(f"edge validation failed: {errors}")
        edges = graph["edges"]
        payload = edge.to_dict()
        for index, existing in enumerate(edges):
            same_identity = (
                existing.get("source") == edge.source
                and existing.get("target") == edge.target
                and existing.get("type") == edge.type.value
                and str(existing.get("source_step") or "") == edge.source_step
                and str(existing.get("target_step") or "") == edge.target_step
            )
            if existing.get("edge_id") == edge.edge_id or same_identity:
                edges[index] = payload
                return
        edges.append(payload)

    def add_edge(self, source: str, target: str, etype: EdgeType,
                 subtype: str = "", metadata: dict[str, Any] | None = None,
                 *, scope: str = "global", source_step: str = "",
                 target_step: str = "", condition: dict[str, Any] | None = None,
                 mapping: dict[str, Any] | None = None,
                 policy: dict[str, Any] | None = None,
                 evidence: list[str] | None = None) -> str:
        graph = self._read_graph()
        edge = GraphEdge(
            source=source, target=target, type=etype, subtype=subtype,
            scope=scope, source_step=source_step, target_step=target_step,
            condition=condition or {}, mapping=mapping or {}, policy=policy or {},
            evidence=evidence or [], metadata=metadata or {},
        )
        self._upsert_edge(graph, edge)
        self._write_graph(graph)
        return edge.edge_id

    def add_edge_object(self, edge: GraphEdge) -> str:
        graph = self._read_graph()
        self._upsert_edge(graph, edge)
        self._write_graph(graph)
        return edge.edge_id

    def get(self, ref: SkillRef):
        index = self.index_entry(ref.logical_id)
        if index is None:
            return None
        return self._load_obj(SkillNodeKind(index["kind"]), ref.logical_id, ref.version)

    def get_latest(self, logical_id: str):
        index = self.index_entry(logical_id)
        if index is None:
            return None
        return self._load_obj(SkillNodeKind(index["kind"]), logical_id,
                              index.get("latest_version") or index["current_version"])

    def get_recommended(self, logical_id: str):
        index = self.index_entry(logical_id)
        if index is None:
            return None
        version = str(index.get("recommended_version") or "")
        if not version:
            return None
        obj = self._load_obj(SkillNodeKind(index["kind"]), logical_id, version)
        return obj if obj is not None and obj.status == SkillStatus.ACTIVE else None

    def index_entry(self, logical_id: str) -> dict[str, Any] | None:
        return self._read_graph()["nodes"].get(logical_id)

    def list_versions(self, logical_id: str) -> list[str]:
        index = self.index_entry(logical_id)
        return list(index.get("versions") or []) if index else []

    def list_by_kind(self, kind: SkillNodeKind, *, statuses: set[SkillStatus] | None = None):
        graph = self._read_graph()
        result = []
        for logical_id, entry in graph["nodes"].items():
            if entry.get("kind") != kind.value:
                continue
            version = str(entry.get("recommended_version") or "")
            obj = self._load_obj(kind, logical_id, version) if version else None
            if obj is not None and statuses is not None and obj.status not in statuses:
                continue
            if obj is not None:
                result.append(obj)
        return result

    def list_all_versions(self, kind: SkillNodeKind | None = None):
        """所有逻辑节点的所有历史版本（用于实现选择/审计；版本不可变）。"""
        graph = self._read_graph()
        result = []
        for logical_id, entry in graph["nodes"].items():
            node_kind = SkillNodeKind(entry.get("kind"))
            if kind is not None and node_kind != kind:
                continue
            for version in entry.get("versions") or []:
                obj = self._load_obj(node_kind, logical_id, version)
                if obj is not None:
                    result.append(obj)
        return result

    def list_all(self):
        graph = self._read_graph()
        result = []
        for logical_id, entry in graph["nodes"].items():
            version = str(entry.get("recommended_version") or "")
            obj = (self._load_obj(SkillNodeKind(entry["kind"]), logical_id, version)
                   if version else None)
            if obj is not None:
                result.append(obj)
        return result

    def iter_edges(self) -> Iterator[dict[str, Any]]:
        for edge in self._read_graph()["edges"]:
            try:
                yield GraphEdge.from_dict(edge).to_dict()
            except (TypeError, ValueError):
                yield dict(edge)

    def edge_objects(self, edge_type: EdgeType | None = None) -> list[GraphEdge]:
        result: list[GraphEdge] = []
        for payload in self._read_graph()["edges"]:
            try:
                edge = GraphEdge.from_dict(payload)
            except (TypeError, ValueError):
                continue
            if edge_type is None or edge.type == edge_type:
                result.append(edge)
        return result

    # ------------------------------------------------------------------
    # 状态 / 推荐指针（回滚 = 恢复推荐指针，不覆盖历史 artifact，§38.4）
    # ------------------------------------------------------------------
    def set_status(self, ref: SkillRef, status: SkillStatus) -> None:
        graph = self._read_graph()
        entry = graph["nodes"].get(ref.logical_id)
        if entry is None:
            raise KeyError(ref.logical_id)
        obj = self.get(ref)
        if obj is None:
            raise KeyError(str(ref))
        obj.status = status
        self._save_obj(obj)
        entry.setdefault("version_statuses", {})[ref.version] = status.value
        if ref.version == (entry.get("latest_version") or entry.get("current_version")):
            entry["status"] = status.value
        if status == SkillStatus.ACTIVE:
            entry["recommended_version"] = ref.version
        elif entry.get("recommended_version") == ref.version:
            entry.pop("recommended_version", None)
        self._write_graph(graph)

    def update_runtime_state(self, obj) -> SkillRef:
        """Persist mutable evidence/quality/status without a semantic version bump."""
        entry = self.index_entry(obj.ref.logical_id)
        if entry is None or obj.ref.version not in (entry.get("versions") or []):
            raise KeyError(str(obj.ref))
        errors = obj.validate()
        if errors:
            raise ValueError(f"{obj.ref.logical_id} 校验失败：{errors}")
        self._save_obj(obj)
        graph = self._read_graph()
        entry = graph["nodes"][obj.ref.logical_id]
        entry.setdefault("version_statuses", {})[obj.ref.version] = obj.status.value
        if obj.ref.version == (entry.get("latest_version") or entry.get("current_version")):
            entry["status"] = obj.status.value
        if obj.status == SkillStatus.ACTIVE:
            entry["recommended_version"] = obj.ref.version
        elif entry.get("recommended_version") == obj.ref.version:
            entry.pop("recommended_version", None)
        self._write_graph(graph)
        return obj.ref

    def recommend(self, ref: SkillRef) -> None:
        graph = self._read_graph()
        entry = graph["nodes"].get(ref.logical_id)
        if entry is None:
            raise KeyError(ref.logical_id)
        if ref.version not in (entry.get("versions") or []):
            raise KeyError(ref.version)
        obj = self.get(ref)
        if obj is None or obj.status != SkillStatus.ACTIVE:
            raise ValueError(f"only_active_may_be_recommended:{ref}")
        entry["recommended_version"] = ref.version
        self._write_graph(graph)

    def rollback(self, logical_id: str, version: str) -> SkillRef:
        """将推荐指针恢复到历史版本（物理上不删除任何文件）。"""
        ref = SkillRef(logical_id, version)
        self.recommend(ref)
        return ref

    # ------------------------------------------------------------------
    # 检索（附录 D：capability-based，task_type 仅弱召回加分）
    # ------------------------------------------------------------------
    def retrieve(self, query: dict[str, Any], *, top_k: int = 5,
                 hard_restrict_task_type: bool = False,
                 task_type_bonus: float = 0.3,
                 statuses: set[SkillStatus] | None = None) -> list[RetrievalHit]:
        goal_text = str(query.get("goal_text", "")).lower()
        task_type = str(query.get("task_type", ""))
        state = query.get("state") or {}
        available_inputs = {str(x).lower() for x in (query.get("available_inputs") or [])}
        target_effects = query.get("target_effects") or []

        goal_tokens = _tokenize(goal_text)
        hits: list[RetrievalHit] = []
        allowed = statuses or {SkillStatus.ACTIVE}
        pool = (self.list_all() if allowed == {SkillStatus.ACTIVE}
                else [obj for obj in self.list_all_versions()
                      if obj.status in allowed])
        for obj in pool:
            if obj.status not in allowed:
                continue
            labels = {str(t) for t in (getattr(obj, "task_type_labels", None) or [])}
            labels |= {str(t) for t in ((getattr(obj, "metadata", None) or {}).get("task_type_labels") or [])}
            matched_type = bool(task_type and task_type in labels)
            if hard_restrict_task_type and task_type and not matched_type:
                continue
            breakdown: dict[str, float] = {}

            # semantic（summary + guideline）
            guideline = getattr(obj, "guideline", {}) or {}
            rules = guideline.get("rules") or []
            semantic_text = " ".join([getattr(obj, "summary", "") or "", *[str(r) for r in rules]]).lower()
            semantic_tokens = _tokenize(semantic_text)
            overlap = len(goal_tokens & semantic_tokens) / max(len(goal_tokens), 1)
            breakdown["semantic"] = overlap

            # effect
            effect_score = 0.0
            if target_effects and getattr(obj, "effects", None):
                effect_set = {_norm_effect(e) for e in obj.effects}
                target_set = {_norm_effect(e) for e in target_effects}
                effect_score = len(effect_set & target_set) / max(len(target_set), 1)
            breakdown["effect"] = effect_score

            # I/O fit
            io_score = 0.0
            if available_inputs:
                input_names = {str(i.get("name", "")).lower() for i in getattr(obj, "inputs", [])}
                io_score = len(available_inputs & input_names) / max(len(available_inputs), 1)
            breakdown["io_fit"] = io_score

            # precondition fit（对 state 的软匹配：state 事实覆盖预条件）
            pre_score = 1.0
            preconditions = getattr(obj, "preconditions", None)
            if preconditions:
                facts = {str(f) for f in (state.get("facts") or [])}
                covered = 0
                for p in preconditions:
                    pred_name = str(p.get("predicate", ""))
                    if not pred_name or any(pred_name.replace(".", "_") in f for f in facts):
                        covered += 1
                pre_score = covered / max(len(preconditions), 1)
            breakdown["precondition_fit"] = pre_score

            # 历史 utility
            meta = getattr(obj, "metadata", {}) or {}
            stats = meta.get("statistics") or {}
            utility = float(stats.get("utility", 0.5))
            breakdown["historical_utility"] = utility

            score = (
                0.30 * overlap
                + 0.20 * effect_score
                + 0.15 * io_score
                + 0.15 * pre_score
                + 0.20 * utility
            )
            if matched_type:
                score += task_type_bonus
            hits.append(RetrievalHit(ref=obj.ref, kind=obj.kind, obj=obj, score=score,
                                     breakdown=breakdown, matched_task_type=matched_type))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def stats(self) -> dict[str, Any]:
        graph = self._read_graph()
        by_kind: dict[str, int] = {}
        by_edge_type: dict[str, int] = {}
        by_edge_category: dict[str, int] = {}
        for entry in graph["nodes"].values():
            by_kind[entry.get("kind", "?")] = by_kind.get(entry.get("kind", "?"), 0) + 1
        for payload in graph["edges"]:
            edge_type = str(payload.get("type") or "?")
            category = str(payload.get("category") or "?")
            by_edge_type[edge_type] = by_edge_type.get(edge_type, 0) + 1
            by_edge_category[category] = by_edge_category.get(category, 0) + 1
        return {"nodes": len(graph["nodes"]), "edges": len(graph["edges"]),
                "by_kind": by_kind, "by_edge_type": by_edge_type,
                "by_edge_category": by_edge_category}


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for word in text.split():
        word = word.strip(".,;:()[]{}!?\"'")
        if len(word) >= 3:
            tokens.add(word)
    return tokens


def _norm_effect(effect: dict[str, Any]) -> str:
    if not isinstance(effect, dict):
        return str(effect)
    name = str(effect.get("predicate", ""))
    # Placeholder paths belong to different scopes in stored Skills and task
    # goals (for example $inputs.object versus $object). Retrieval compares the
    # verified predicate/argument contract, not those serialization paths.
    args = {
        str(key): ("$" if str(value).startswith("$") else str(value))
        for key, value in (effect.get("args") or {}).items()
    }
    return f"{name}:{json.dumps(args, sort_keys=True)}"


def _semver_key(version: str) -> tuple[int, int, int]:
    parts = version.split(".")
    return (int(parts[0]), int(parts[1]), int(parts[2]))
