"""Skill 三类持久化节点的 IR（设计文档 v2.0 §5、§16、§17、§19、§42）。

- AbstractAtomicSkill：做什么（稳定核心 Effect）
- ImplementationAtom：怎么执行（tool_ref + bindings + execution_policy，不保存大段代码）
- CompositeSkill：高层组合 + Layer-3 insight

所有对象支持 dict 往返与内容哈希；版本不可变（变更即新版本）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .edge_ir import GraphEdge
from .refs import SkillRef, ToolRef, bump_version, content_hash
from .status import EdgeType, SkillNodeKind, SkillStatus


class SkillIRError(ValueError):
    """Skill IR 校验错误。"""


@dataclass
class AbstractAtomicSkill:
    ref: SkillRef
    summary: str = ""
    inputs: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    preconditions: list[dict[str, Any]] = field(default_factory=list)
    effects: list[dict[str, Any]] = field(default_factory=list)
    validator: dict[str, Any] = field(default_factory=dict)
    failure_modes: list[dict[str, Any]] = field(default_factory=list)
    guideline: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    status: SkillStatus = SkillStatus.ACTIVE

    # -- 属性 ----------------------------------------------------------------
    @property
    def kind(self) -> SkillNodeKind:
        return SkillNodeKind.ABSTRACT_ATOMIC

    @property
    def logical_id(self) -> str:
        return self.ref.logical_id

    def semantic_hash(self) -> str:
        """语义/契约/Effect 内容哈希（不含可变统计）。"""
        return content_hash(
            self.to_dict(),
            exclude=("ref", "status", "metadata"),
        )

    def next_version(self, change: str = "patch") -> str:
        return bump_version(self.ref.version, change)

    # -- 校验 ----------------------------------------------------------------
    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.summary.strip():
            errors.append("summary 不能为空")
        if not self.effects:
            errors.append("AbstractAtomicSkill 必须至少有一个核心 Effect")
        for effect in self.effects:
            if not isinstance(effect, dict) or not effect.get("predicate"):
                errors.append(f"非法 Effect 定义：{effect!r}")
        for precondition in self.preconditions:
            if not isinstance(precondition, dict):
                errors.append(f"非法前置条件定义：{precondition!r}")
        for item in self.inputs + self.outputs:
            if not isinstance(item, dict) or not item.get("name"):
                errors.append(f"I/O 定义缺少 name：{item!r}")
        return errors

    # -- 序列化 ----------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "id": self.ref.logical_id,
            "version": self.ref.version,
            "status": self.status.value,
            "summary": self.summary,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "preconditions": self.preconditions,
            "effects": self.effects,
            "validator": self.validator,
            "failure_modes": self.failure_modes,
            "guideline": self.guideline,
            "implementation_refs": [
                f"impl.{self.ref.logical_id}@{self.ref.version}"
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AbstractAtomicSkill":
        logical_id = str(data.get("id") or data.get("logical_id") or "")
        version = str(data.get("version") or "1.0.0")
        return cls(
            ref=SkillRef(logical_id=logical_id, version=version),
            summary=str(data.get("summary", "")),
            inputs=list(data.get("inputs") or []),
            outputs=list(data.get("outputs") or []),
            preconditions=list(data.get("preconditions") or []),
            effects=list(data.get("effects") or []),
            validator=dict(data.get("validator") or {}),
            failure_modes=list(data.get("failure_modes") or []),
            guideline=dict(data.get("guideline") or {}),
            metadata=dict(data.get("metadata") or {}),
            status=SkillStatus(str(data.get("status") or "active")),
        )

    # -- 高层 guideline 的便捷访问 -------------------------------------------
    def guideline_rules(self) -> list[str]:
        rules = self.guideline.get("rules") or []
        return [str(r) for r in rules]


@dataclass
class ToolBinding:
    tool_ref: ToolRef
    role: str = "primary"
    parameter_mapping: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_ref": f"tool://{self.tool_ref.tool_id}@{self.tool_ref.version}",
            "role": self.role,
            "parameter_mapping": self.parameter_mapping,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolBinding":
        return cls(
            tool_ref=ToolRef.parse(str(data["tool_ref"])),
            role=str(data.get("role", "primary")),
            parameter_mapping=dict(data.get("parameter_mapping") or {}),
        )


@dataclass
class ImplementationAtom:
    ref: SkillRef
    abstract_ref: SkillRef
    tool_bindings: list[ToolBinding] = field(default_factory=list)
    execution_policy: dict[str, Any] = field(default_factory=dict)
    compatibility: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    status: SkillStatus = SkillStatus.ACTIVE

    @property
    def kind(self) -> SkillNodeKind:
        return SkillNodeKind.IMPLEMENTATION_ATOMIC

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.tool_bindings:
            errors.append("ImplementationAtom 至少需要一个 Tool Binding（|B| >= 1）")
        for binding in self.tool_bindings:
            if not binding.tool_ref.tool_id:
                errors.append("Tool Binding 缺少 tool_ref")
        return errors

    def semantic_hash(self) -> str:
        return content_hash(self.to_dict(), exclude=("ref", "status", "quality"))

    def next_version(self, change: str = "patch") -> str:
        return bump_version(self.ref.version, change)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "id": self.ref.logical_id,
            "version": self.ref.version,
            "status": self.status.value,
            "implements": {
                "id": self.abstract_ref.logical_id,
                "version": self.abstract_ref.version,
            },
            "compatibility": self.compatibility,
            "tool_bindings": [b.to_dict() for b in self.tool_bindings],
            "execution_policy": self.execution_policy,
            "validator_overrides": [],
            "quality": self.quality,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImplementationAtom":
        implements = data.get("implements") or {}
        return cls(
            ref=SkillRef(
                logical_id=str(data.get("id") or data.get("logical_id") or ""),
                version=str(data.get("version") or "1.0.0"),
            ),
            abstract_ref=SkillRef(
                logical_id=str(implements.get("id") or ""),
                version=str(implements.get("version") or "1.0.0"),
            ),
            tool_bindings=[ToolBinding.from_dict(b) for b in (data.get("tool_bindings") or [])],
            execution_policy=dict(data.get("execution_policy") or {}),
            compatibility=dict(data.get("compatibility") or {}),
            quality=dict(data.get("quality") or {}),
            status=SkillStatus(str(data.get("status") or "active")),
        )


@dataclass
class CompositeSkill:
    ref: SkillRef
    summary: str = ""
    task_type_labels: list[str] = field(default_factory=list)
    graph: dict[str, Any] = field(default_factory=dict)  # nodes + control + data
    guideline: dict[str, Any] = field(default_factory=dict)
    insight: dict[str, Any] = field(default_factory=dict)  # Layer-3
    validator: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    status: SkillStatus = SkillStatus.ACTIVE

    @property
    def kind(self) -> SkillNodeKind:
        return SkillNodeKind.COMPOSITE

    def nodes(self) -> list[str]:
        """Atomic Skill refs in step-instance order (backward compatible)."""
        return [step["node_ref"] for step in self.step_instances()]

    def step_instances(self) -> list[dict[str, Any]]:
        """Return stable call instances so one Skill may appear more than once."""
        raw_steps = self.graph.get("steps") or []
        if raw_steps:
            return [
                {
                    "step_id": str(step.get("step_id") or f"step_{index:03d}"),
                    "node_ref": str(step.get("node_ref") or step.get("ref") or ""),
                    "params": dict(step.get("params") or {}),
                    "metadata": dict(step.get("metadata") or {}),
                }
                for index, step in enumerate(raw_steps)
            ]
        return [
            {"step_id": f"step_{index:03d}", "node_ref": str(ref),
             "params": {}, "metadata": {}}
            for index, ref in enumerate(self.graph.get("nodes") or [])
        ]

    def control_edges(self) -> list[list[str]]:
        """Legacy source/target node-ref pairs used by older callers."""
        pairs: list[list[str]] = []
        for edge in self.edge_objects():
            if edge.category == "control":
                pairs.append([edge.source, edge.target])
        return pairs

    def edge_objects(self) -> list[GraphEdge]:
        steps = self.step_instances()
        by_id = {step["step_id"]: step for step in steps}
        by_ref: dict[str, list[dict[str, Any]]] = {}
        for step in steps:
            by_ref.setdefault(_logical_ref(step["node_ref"]), []).append(step)

        def resolve(value: str, occurrence: int = 0) -> dict[str, Any] | None:
            if value in by_id:
                return by_id[value]
            matches = by_ref.get(_logical_ref(value), [])
            return matches[min(occurrence, len(matches) - 1)] if matches else None

        result: list[GraphEdge] = []
        raw_groups = [
            (self.graph.get("control") or [], EdgeType.NEXT),
            (self.graph.get("data") or [], EdgeType.DATA_FLOW),
            (self.graph.get("dependencies") or [], EdgeType.REQUIRES_SKILL),
            (self.graph.get("semantic") or [], EdgeType.SIMILAR),
            (self.graph.get("evolution") or [], EdgeType.DERIVED_FROM),
        ]
        for raw_edges, default_type in raw_groups:
            for raw in raw_edges:
                if isinstance(raw, dict):
                    payload = dict(raw)
                    payload.setdefault("type", default_type.value)
                    payload.setdefault("scope", "composite")
                    source_value = str(payload.get("source_step") or payload.get("source") or "")
                    target_value = str(payload.get("target_step") or payload.get("target") or "")
                    source_step = resolve(source_value)
                    target_step = resolve(target_value)
                    if source_step:
                        payload["source"] = source_step["node_ref"]
                        payload["source_step"] = source_step["step_id"]
                    if target_step:
                        payload["target"] = target_step["node_ref"]
                        payload["target_step"] = target_step["step_id"]
                    result.append(GraphEdge.from_dict(payload))
                    continue
                if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                    continue
                source_step = resolve(str(raw[0]))
                target_step = resolve(str(raw[1]))
                if source_step is None or target_step is None:
                    continue
                kwargs: dict[str, Any] = {}
                if default_type == EdgeType.DATA_FLOW:
                    kwargs["mapping"] = {
                        "source_output": "*", "target_input": "*",
                        "mode": "legacy_inferred",
                    }
                result.append(GraphEdge(
                    source=source_step["node_ref"], target=target_step["node_ref"],
                    type=default_type, scope="composite",
                    source_step=source_step["step_id"],
                    target_step=target_step["step_id"],
                    evidence=[str(self.ref)], **kwargs,
                ))
        return result

    def normalized_graph(self) -> dict[str, Any]:
        steps = self.step_instances()
        edge_objects = self.edge_objects()
        return {
            **self.graph,
            "nodes": [step["node_ref"] for step in steps],
            "steps": steps,
            "control": [edge.to_dict() for edge in edge_objects
                        if edge.category == "control"],
            "data": [edge.to_dict() for edge in edge_objects
                     if edge.category == "data"],
            "dependencies": [edge.to_dict() for edge in edge_objects
                             if edge.category == "dependency"],
            "semantic": [edge.to_dict() for edge in edge_objects
                         if edge.category == "semantic"],
            "evolution": [edge.to_dict() for edge in edge_objects
                          if edge.category == "evolution"],
        }

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.summary.strip():
            errors.append("summary 不能为空")
        nodes = self.nodes()
        if not nodes:
            errors.append("CompositeSkill.graph.nodes 不能为空")
        steps = self.step_instances()
        step_ids = [step["step_id"] for step in steps]
        if len(step_ids) != len(set(step_ids)):
            errors.append("CompositeSkill step_id 必须唯一")
        if any(not step["node_ref"] for step in steps):
            errors.append("CompositeSkill step node_ref 不能为空")
        valid_steps = set(step_ids)
        for edge in self.edge_objects():
            errors.extend(edge.validate())
            if edge.scope == "composite" and (
                    edge.source_step not in valid_steps or edge.target_step not in valid_steps):
                errors.append(f"edge 引用了未知 step：{edge.edge_id}")
        return errors

    def semantic_hash(self) -> str:
        return content_hash(self.to_dict(), exclude=("ref", "status", "metadata", "insight"))

    def next_version(self, change: str = "patch") -> str:
        return bump_version(self.ref.version, change)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "id": self.ref.logical_id,
            "version": self.ref.version,
            "status": self.status.value,
            "summary": self.summary,
            "task_type_labels": self.task_type_labels,
            "graph": self.normalized_graph(),
            "guideline": self.guideline,
            "insight": self.insight,
            "validator": self.validator,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompositeSkill":
        return cls(
            ref=SkillRef(
                logical_id=str(data.get("id") or data.get("logical_id") or ""),
                version=str(data.get("version") or "1.0.0"),
            ),
            summary=str(data.get("summary", "")),
            task_type_labels=list(data.get("task_type_labels") or []),
            graph=dict(data.get("graph") or {}),
            guideline=dict(data.get("guideline") or {}),
            insight=dict(data.get("insight") or {}),
            validator=dict(data.get("validator") or {}),
            metadata=dict(data.get("metadata") or {}),
            status=SkillStatus(str(data.get("status") or "active")),
        )


def load_skill_from_dict(data: dict[str, Any]) -> AbstractAtomicSkill | ImplementationAtom | CompositeSkill:
    """按 kind 反序列化。"""
    kind = str(data.get("kind") or "")
    if kind == SkillNodeKind.ABSTRACT_ATOMIC.value:
        return AbstractAtomicSkill.from_dict(data)
    if kind == SkillNodeKind.IMPLEMENTATION_ATOMIC.value:
        return ImplementationAtom.from_dict(data)
    if kind == SkillNodeKind.COMPOSITE.value:
        return CompositeSkill.from_dict(data)
    raise SkillIRError(f"未知 Skill 节点类型：{kind!r}")


def _logical_ref(ref: str) -> str:
    text = str(ref)
    if text.startswith("skill://"):
        text = text[len("skill://"):]
    return text.rsplit("@", 1)[0]
