"""Stage-1 无 API smoke 检查集：IR / Graph / Tools / Atomicizer / Evolution。

纯规则与合成数据，无 LLM、无网络、无环境依赖。供 `experiments.run_smoke_ir`
与 pytest（tests/test_smoke_ir.py）复用。
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from atomic_skillgraph.core.config import SystemConfig, Thresholds  # noqa: E402
from atomic_skillgraph.core.refs import (  # noqa: E402
    SkillRef,
    ToolRef,
    bump_version,
    check_version,
    content_hash,
)
from atomic_skillgraph.core.skill_ir import (  # noqa: E402
    AbstractAtomicSkill,
    CompositeSkill,
    ImplementationAtom,
    ToolBinding,
    load_skill_from_dict,
)
from atomic_skillgraph.core.status import (  # noqa: E402
    EdgeType,
    SkillNodeKind,
    SkillStatus,
    ToolLifecycle,
    tool_transition_allowed,
)
from atomic_skillgraph.core.tool_ir import ToolAsset  # noqa: E402
from atomic_skillgraph.core.trace_ir import (  # noqa: E402
    ActionRecord,
    AttemptRecord,
    NodeValidationResult,
    TraceRecord,
    new_id,
)
from atomic_skillgraph.graph.aligner import (  # noqa: E402
    align_atomic,
    align_composite,
    align_implementation,
)
from atomic_skillgraph.graph.graph import (  # noqa: E402
    build_control_edges,
    composite_node_order,
    topo_sort,
)
from atomic_skillgraph.graph.registry import SkillGraphRegistry  # noqa: E402
from atomic_skillgraph.graph.validator import validate_graph  # noqa: E402


class CheckError(AssertionError):
    pass


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise CheckError(message)


def check_refs_and_semver() -> None:
    ref = SkillRef.parse("skill://alfworld.acquire-object@1.0.0")
    expect(str(ref) == "skill://alfworld.acquire-object@1.0.0", f"SkillRef 往返失败：{ref}")
    tool = ToolRef.parse("tool://alfworld/acquire@1.2.0")
    expect(str(tool) == "tool://alfworld/acquire@1.2.0", f"ToolRef 往返失败：{tool}")
    expect(check_version("1.0.0") and not check_version("1.0"), "semver 校验错误")
    expect(bump_version("1.0.0") == "1.0.1", "patch bump 错误")
    expect(bump_version("1.0.0", "minor") == "1.1.0", "minor bump 错误")
    expect(bump_version("1.0.0", "major") == "2.0.0", "major bump 错误")
    expect(content_hash({"a": 1, "b": [2]}) == content_hash({"b": [2], "a": 1}),
           "content_hash 应排序无关")
    expect(content_hash({"a": 1}) != content_hash({"a": 2}), "content_hash 冲突")


def _sample_atomic(logical_id: str = "test.acquire-object") -> AbstractAtomicSkill:
    return AbstractAtomicSkill(
        ref=SkillRef(logical_id, "1.0.0"),
        summary="获取目标对象，使 Agent 持有该对象",
        inputs=[{"name": "object", "semantic_type": "object_ref"},
                {"name": "object_location", "semantic_type": "location_ref"}],
        outputs=[{
            "name": "held_object", "semantic_type": "object_ref",
            "materializer": {
                "kind": "effect_arg", "predicate": "agent.holds",
                "arg": "object",
            },
        }],
        preconditions=[{"predicate": "object.exists", "args": {"object": "$object"}}],
        effects=[{"predicate": "agent.holds", "args": {"object": "$object"}}],
        validator={"pre_checks": ["object_exists"], "post_checks": ["agent_holds"]},
        failure_modes=[{"name": "object_not_found"}],
        guideline={"layer": 2, "rules": ["先确认对象当前位置"]},
        metadata={"task_type_labels": ["pick_and_place_simple"]},
    )


def _sample_tool(tool_id: str = "test.acquire-template", code: str = "") -> ToolAsset:
    body = code or "def acquire(object, object_location):\n    return object\n"
    return ToolAsset(
        ref=ToolRef(tool_id, "0.1.0"),
        summary="在已知位置获取对象",
        signature={"entry_point": "acquire",
                   "parameters": [{"name": "object", "required": True},
                                  {"name": "object_location", "required": True}]},
        interface={"inputs": {"object": "object_ref", "object_location": "location_ref"},
                   "outputs": {"held_object": "object_ref"}},
        artifact={"code": body},
        tests=[{"kind": "replay", "entry_point": "acquire",
                "tests": ["assert acquire('egg_1', 'countertop_1') == 'egg_1'"]}],
        safety={"direct_execution_allowed": True, "checks_passed": []},
        provenance={"source_trace_ids": ["trace_x"]},
        statistics={"support_count": 1, "call_count": 0, "success_count": 0,
                    "failure_count": 0, "utility": 0.5},
        lineage={"generalized_from": [], "specialized_from": [], "supersedes": None},
        status=ToolLifecycle.DRAFT,
    )


def check_skill_ir_roundtrip() -> None:
    atomic = _sample_atomic()
    expect(not atomic.validate(), f"atomic 校验错误：{atomic.validate()}")
    data = atomic.to_dict()
    loaded = load_skill_from_dict(data)
    expect(isinstance(loaded, AbstractAtomicSkill), "roundtrip kind 错误")
    expect(str(loaded.ref) == str(atomic.ref), "roundtrip ref 错误")
    expect(atomic.semantic_hash() == loaded.semantic_hash(), "语义哈希不稳定")
    # 无 effect 的 atomic 非法
    broken = _sample_atomic()
    broken.effects = []
    expect(bool(broken.validate()), "缺少 Effect 的 Atomic 应校验失败")

    impl = ImplementationAtom(
        ref=SkillRef("impl.test.acquire-object", "1.0.0"),
        abstract_ref=atomic.ref,
        tool_bindings=[ToolBinding(tool_ref=ToolRef("test.acquire-template", "0.1.0"),
                                   role="primary",
                                   parameter_mapping={"object": "$inputs.object"})],
        execution_policy={"mode": "direct_if_eligible"},
        compatibility={"harness": "env"},
        quality={"utility": 0.5},
    )
    expect(not impl.validate(), f"impl 校验错误：{impl.validate()}")
    broken_impl = ImplementationAtom(ref=impl.ref, abstract_ref=atomic.ref,
                                     tool_bindings=[])
    expect(bool(broken_impl.validate()), "无 Binding 的 Implementation 应校验失败")

    composite = CompositeSkill(
        ref=SkillRef("composite.test.pick-place", "1.0.0"),
        summary="获取后放置",
        graph={"nodes": ["test.acquire-object@1.0.0", "test.place-object@1.0.0"],
               "control": build_control_edges(["test.acquire-object", "test.place-object"])},
        validator={"checks": ["object.at_location"]},
        insight={"layer": 3, "sample_count": 0},
    )
    expect(not composite.validate(), f"composite 校验错误：{composite.validate()}")


def check_tool_ir_lifecycle() -> None:
    tool = _sample_tool()
    expect(tool.status == ToolLifecycle.DRAFT, "初始状态应为 draft")
    expect(tool_transition_allowed(ToolLifecycle.DRAFT, ToolLifecycle.ADMISSION_PENDING),
           "draft->admission_pending 应合法")
    expect(not tool_transition_allowed(ToolLifecycle.DRAFT, ToolLifecycle.ACTIVE),
           "draft->active 应非法（必须经 admission）")
    tool.record_usage(True, usage_mode="direct")
    tool.record_usage(False)
    expect(tool.statistics["call_count"] == 2, "call_count 统计错误")
    expect(0 < tool.statistics["utility"] < 1, "utility EMA 错误")
    expect(tool.artifact_hash() == ToolAsset.from_dict(tool.to_dict()).artifact_hash(),
           "artifact hash 应稳定")


def check_registry_and_graph(tmp_path: Path) -> None:
    registry = SkillGraphRegistry(tmp_path / "graph")
    atomic = _sample_atomic()
    registry.register(atomic)
    impl = ImplementationAtom(
        ref=SkillRef("impl.test.acquire-object", "1.0.0"),
        abstract_ref=atomic.ref,
        tool_bindings=[ToolBinding(tool_ref=ToolRef("test.acquire-template", "0.1.0"))],
    )
    registry.register(impl)
    place = AbstractAtomicSkill(
        ref=SkillRef("test.place-object", "1.0.0"),
        summary="将对象放置到目标位置",
        inputs=[{"name": "object", "semantic_type": "object_ref"},
                {"name": "location", "semantic_type": "location_ref"}],
        outputs=[],
        effects=[{"predicate": "object.at_location", "args": {"object": "$object",
                                                              "location": "$location"}}],
        validator={}, metadata={"task_type_labels": ["pick_and_place_simple"]},
    )
    registry.register(place)
    composite = CompositeSkill(
        ref=SkillRef("composite.test.pick-place", "1.0.0"),
        summary="获取后放置",
        graph={
            "nodes": ["test.acquire-object@1.0.0",
                      "test.place-object@1.0.0"],
            "steps": [
                {"step_id": "acquire_000",
                 "node_ref": "test.acquire-object@1.0.0",
                 "params": {"object": "$task.object"}},
                {"step_id": "place_000",
                 "node_ref": "test.place-object@1.0.0",
                 "params": {"object": "$flow.held_object",
                            "location": "$task.location"}},
            ],
            "control": [{
                "source": "test.acquire-object@1.0.0",
                "target": "test.place-object@1.0.0",
                "source_step": "acquire_000", "target_step": "place_000",
                "type": EdgeType.NEXT.value, "scope": "composite",
            }],
            "data": [{
                "source": "test.acquire-object@1.0.0",
                "target": "test.place-object@1.0.0",
                "source_step": "acquire_000", "target_step": "place_000",
                "type": EdgeType.DATA_FLOW.value, "scope": "composite",
                "mapping": {"source_output": "held_object",
                            "target_input": "object",
                            "source_semantic_type": "object_ref",
                            "target_semantic_type": "object_ref",
                            "transform": "identity"},
            }],
        },
        validator={}, metadata={"task_type_labels": ["pick_and_place_simple"]},
    )
    registry.register(composite)

    expect(len(registry.list_by_kind(SkillNodeKind.ABSTRACT_ATOMIC)) == 2, "atomic 数量错误")
    expect(len(registry.list_by_kind(SkillNodeKind.IMPLEMENTATION_ATOMIC)) == 1, "impl 数量错误")
    # implements 边存在
    edges = [e for e in registry.iter_edges() if e.get("type") == EdgeType.IMPLEMENTS.value]
    expect(bool(edges), "缺少 implements 边")
    # contains 边
    contains = [e for e in registry.iter_edges() if e.get("type") == EdgeType.CONTAINS.value]
    expect(len(contains) == 2, f"contains 边数量错误：{len(contains)}")

    # 检索：task_type 弱召回
    hits = registry.retrieve({"goal_text": "take object and place it",
                              "task_type": "pick_and_place_simple",
                              "state": {}, "available_inputs": ["object"]})
    expect(any(h.obj.ref.logical_id == "test.acquire-object" for h in hits),
           "检索应召回 acquire-object")
    expect(any(h.matched_task_type for h in hits), "task_type 弱召回信号缺失")
    # 硬过滤模式：不同 task_type 应被过滤
    hits_hard = registry.retrieve({"goal_text": "take object", "task_type": "other_type",
                                   "state": {}, "available_inputs": []},
                                  hard_restrict_task_type=True)
    expect(not hits_hard, "task_type 硬过滤模式应过滤全部")

    # 拓扑 / 环
    order, cycles = topo_sort(["a", "b", "c"], [["a", "b"], ["b", "c"]])
    expect(order == ["a", "b", "c"], f"拓扑排序错误：{order}")
    order2, cycles2 = topo_sort(["a", "b", "c"], [["a", "b"], ["b", "c"], ["c", "a"]])
    expect(order2 is None and cycles2, "应检测到环")

    # composite 展开顺序
    loaded_composite = registry.get_recommended("composite.test.pick-place")
    node_order, report = composite_node_order(loaded_composite, registry)
    expect(node_order == ["test.acquire-object", "test.place-object"],
           f"composite 顺序错误：{node_order}")
    expect(report.passed, f"composite 图检查失败：{report.errors}")

    # 回滚（推荐指针恢复）
    atomic_v2 = AbstractAtomicSkill(
        ref=SkillRef("test.acquire-object", "1.1.0"),
        summary=atomic.summary + "（修订）",
        inputs=atomic.inputs, outputs=atomic.outputs,
        preconditions=atomic.preconditions, effects=atomic.effects,
        validator=atomic.validator, failure_modes=atomic.failure_modes,
        guideline=atomic.guideline, metadata=atomic.metadata,
    )
    registry.register(atomic_v2)
    expect(registry.get_recommended("test.acquire-object").ref.version == "1.1.0",
           "推荐指针应指向最新")
    registry.rollback("test.acquire-object", "1.0.0")
    expect(registry.get_recommended("test.acquire-object").ref.version == "1.0.0",
           "rollback 应恢复推荐指针")

    # 图验证
    report = validate_graph(registry, tool_registry=None)
    expect(report.passed, f"图验证失败：{report.errors}")


def check_aligner(tmp_path: Path) -> None:
    registry = SkillGraphRegistry(tmp_path / "graph")
    a1 = _sample_atomic("test.acquire-object")
    registry.register(a1)
    a2 = _sample_atomic("other.acquire-object")  # 语义相同
    decision = align_atomic(a2, registry)
    expect(decision.matched, f"语义相同的 Atomic 应对齐：{decision.evidence}")
    a3 = AbstractAtomicSkill(
        ref=SkillRef("other.heat-object", "1.0.0"),
        summary="加热对象", inputs=[{"name": "object"}], outputs=[],
        effects=[{"predicate": "object.heated", "args": {"object": "$object"}}],
        validator={}, metadata={},
    )
    decision3 = align_atomic(a3, registry)
    expect(not decision3.matched, "不同 Effect 的 Atomic 不应合并")

    impl = ImplementationAtom(
        ref=SkillRef("impl.other.acquire-object", "1.0.0"),
        abstract_ref=a1.ref,
        tool_bindings=[ToolBinding(tool_ref=ToolRef("t", "1.0.0"))],
        compatibility={"harness": "env"},
    )
    registry.register(impl)
    impl2 = ImplementationAtom(
        ref=SkillRef("impl.other2.acquire-object", "1.0.0"),
        abstract_ref=a1.ref,
        tool_bindings=[ToolBinding(tool_ref=ToolRef("t", "1.0.0"))],
        compatibility={"harness": "env"},
    )
    decision_impl = align_implementation(impl2, registry)
    expect(decision_impl.matched, "等价 Implementation 应合并")

    composite = CompositeSkill(
        ref=SkillRef("composite.test.pick-place", "1.0.0"),
        summary="获取后放置",
        graph={"nodes": ["test.acquire-object@1.0.0", "test.place-object@1.0.0"],
               "control": build_control_edges(["test.acquire-object", "test.place-object"])},
        validator={}, metadata={},
    )
    registry.register(composite)
    composite2 = CompositeSkill(
        ref=SkillRef("composite.other.pick-place", "1.0.0"),
        summary="获取后放置（另一来源）",
        graph={"nodes": ["test.acquire-object@1.0.0", "test.place-object@1.0.0"],
               "control": build_control_edges(["test.acquire-object", "test.place-object"])},
        validator={}, metadata={},
    )
    decision_comp = align_composite(composite2, registry)
    expect(decision_comp.matched, "同原子链 Composite 应对齐")


def check_tool_registry_and_resolver(tmp_path: Path) -> None:
    from atomic_skillgraph.tools.admission_adapter import AdmissionEngine
    from atomic_skillgraph.tools.registry import ToolRegistry
    from atomic_skillgraph.tools.resolver import ToolResolver
    registry = ToolRegistry(tmp_path / "tools")
    tool = _sample_tool()
    expect(AdmissionEngine().admit(tool).passed,
           "Tool 必须经 Admission 后注册为 candidate")
    registry.register(tool)
    expect(registry.get(tool.ref) is not None, "Tool 注册失败")
    expect(registry.get_recommended("test.acquire-template") is not None, "推荐版本缺失")

    impl = ImplementationAtom(
        ref=SkillRef("impl.test.acquire-object", "1.0.0"),
        abstract_ref=SkillRef("test.acquire-object", "1.0.0"),
        tool_bindings=[ToolBinding(tool_ref=tool.ref, role="primary",
                                   parameter_mapping={
                                       "object": "$inputs.object",
                                       "object_location": "$inputs.object_location"})],
    )
    resolver = ToolResolver(registry)
    resolved = resolver.resolve(impl, {"inputs": {"object": "egg_1",
                                                  "object_location": "countertop_1"}})
    expect(len(resolved) == 1 and resolved[0].ok, f"解析失败：{resolved[0].missing}")
    expect(resolved[0].parameters["object"] == "egg_1", "参数绑定错误")

    missing = resolver.resolve(impl, {"inputs": {}})
    expect(missing and missing[0].missing, "缺少参数应被检出")

    # 状态迁移合法性
    tool2 = registry.get(tool.ref)
    try:
        registry.set_status(tool.ref, ToolLifecycle.ACTIVE)
        active_ok = True
    except ValueError:
        active_ok = False
    expect(active_ok, "candidate->active 应合法")
    try:
        registry.set_status(tool.ref, ToolLifecycle.DRAFT)
        back_ok = True
    except ValueError:
        back_ok = False
    expect(not back_ok, "active->draft 应非法")

    # feedback
    registry.record_feedback(tool.ref, True, usage_mode="direct")
    updated = registry.get(tool.ref)
    expect(updated.statistics["call_count"] == 1, "feedback 统计错误")


def check_admission(tmp_path: Path) -> None:
    from atomic_skillgraph.tools.admission_adapter import AdmissionEngine
    from atomic_skillgraph.tools.sandbox import Sandbox
    sandbox = Sandbox(tmp_root=str(tmp_path / "sandbox"))
    # 代码 Tool：通过
    good = _sample_tool()
    admission = AdmissionEngine(sandbox=sandbox, existing_hashes=set())
    result = admission.admit(good)
    expect(result.passed, f"安全代码应通过 admission：{result.reasons}")
    expect(good.status == ToolLifecycle.CANDIDATE, "admission 通过后应为 candidate")

    # 危险代码：拒绝
    bad = _sample_tool(tool_id="bad.tool",
                       code="import os\n\ndef f(x):\n    os.system('echo hi')\n    return x\n")
    result_bad = admission.admit(bad)
    expect(not result_bad.passed, "危险 import 应被拒绝")
    expect(bad.status == ToolLifecycle.SHADOW, "admission 失败后应为 shadow")

    # trivial：拒绝
    trivial = _sample_tool(tool_id="trivial.tool",
                           code="def f(x):\n    return 1\n")
    result_trivial = admission.admit(trivial)
    expect(not result_trivial.passed, "平凡解应被拒绝")

    # 重复哈希：拒绝
    dup = _sample_tool(tool_id="dup.tool")
    admission2 = AdmissionEngine(sandbox=sandbox, existing_hashes={dup.structural_hash()})
    result_dup = admission2.admit(dup)
    expect(not result_dup.passed and any("duplicate" in r for r in result_dup.reasons),
           "重复 artifact 应被 dedup 拒绝")

    # action template：replay 回调
    template = ToolAsset(
        ref=ToolRef("test.env-template", "0.1.0"),
        summary="获取对象模板",
        signature={"parameters": [{"name": "object", "required": True}]},
        interface={},
        artifact={"template": "go to {object}", "steps": ["go to {object}"]},
        tests=[{"kind": "replay", "bindings": {"object": "egg_1"}}],
        safety={"direct_execution_allowed": True, "checks_passed": []},
        provenance={}, statistics={}, lineage={}, status=ToolLifecycle.DRAFT,
    )
    from atomic_skillgraph.core.status import ArtifactKind
    template.artifact_kind = ArtifactKind.ACTION_TEMPLATE

    def replay_fn(tool, bindings, before):
        return {"passed": bindings.get("object") == "egg_1", "after": {}}

    admission_env = AdmissionEngine(sandbox=sandbox, replay_fn=replay_fn)
    result_env = admission_env.admit(template)
    expect(result_env.passed, f"合法模板应通过 admission：{result_env.reasons}")


def _env_trace() -> TraceRecord:
    """合成一条成功交互轨迹（pick_and_place 风格）。"""
    trace = TraceRecord(
        trace_id=new_id("trace"), task_id="env_task_1",
        task_type="pick_and_place_simple",
        task_goal="Put a mug on the shelf.",
        benchmark="toy_env", start_mode="cold",
    )
    # 状态快照：每个动作之后一条（snapshot[i+1] 对应 action[i] 之后）
    trace.state_snapshots = [
        {"step": 0, "state": {"facts": ["object_at(mug_1, countertop_1)",
                                        "object_exists(mug_1)"],
                              "inventory": [], "text": "see mug"}},
        {"step": 1, "state": {"facts": ["object_at(mug_1, countertop_1)",
                                        "object_exists(mug_1)"],
                              "inventory": [], "text": "at countertop"}},
        {"step": 2, "state": {"facts": ["object_exists(mug_1)", "agent_holds(mug_1)"],
                              "inventory": ["mug_1"], "text": "holding mug"}},
        {"step": 3, "state": {"facts": ["object_exists(mug_1)", "agent_holds(mug_1)"],
                              "inventory": ["mug_1"], "text": "at shelf"}},
        {"step": 4, "state": {"facts": ["object_exists(mug_1)", "object_at(mug_1, shelf_1)"],
                              "inventory": [], "text": "placed mug"}},
    ]
    trace.actions = [
        ActionRecord(step=0, name="go to countertop 1", params={"location": "countertop 1"},
                     observation="You see mug 1.", accepted=True),
        ActionRecord(step=1, name="take mug 1 from countertop 1", params={"object": "mug 1",
                                                  "object_location": "countertop 1"},
                     observation="You pick up mug 1.", accepted=True),
        ActionRecord(step=2, name="go to shelf 1", params={"location": "shelf 1"},
                     observation="shelf", accepted=True),
        ActionRecord(step=3, name="put mug 1 in/on shelf 1", params={"object": "mug 1",
                                                 "target_location": "shelf 1"},
                     observation="You place mug 1.", accepted=True),
    ]
    trace.success = True
    trace.benchmark_result = {"passed": True}
    return trace


def check_atomicizer_env(tmp_path: Path) -> None:
    from atomic_skillgraph.atomicizer.trace_atomicizer import TraceAtomicizer
    from atomic_skillgraph.graph.registry import SkillGraphRegistry
    registry = SkillGraphRegistry(tmp_path / "graph")
    atomicizer = TraceAtomicizer(registry)
    trace = _env_trace()
    result = atomicizer.atomicize_success(trace)
    expect(bool(result.candidates), "应产生原子候选")
    predicates = {
        str(effect.get("predicate") or "")
        for candidate in result.candidates
        for effect in candidate.skill.effects
    }
    expect("agent.holds" in predicates, f"应保留第一个真实状态转移：{predicates}")
    expect("object.at_location" in predicates,
           f"应保留第二个真实状态转移：{predicates}")
    for candidate in result.candidates:
        expect(bool(candidate.skill.effects), f"候选 {candidate.skill.ref} 缺少 Effect")
        expect(candidate.skill.validator.get("post_checks"),
               f"候选 {candidate.skill.ref} 缺少节点验证器")
    # apply：注册
    applied = atomicizer.apply(trace)
    expect(len(registry.list_all_versions(SkillNodeKind.ABSTRACT_ATOMIC)) >= 2,
           "apply 后应注册原子候选版本")


def check_atomicizer_code(tmp_path: Path) -> None:
    from atomic_skillgraph.atomicizer.trace_atomicizer import TraceAtomicizer
    from atomic_skillgraph.graph.registry import SkillGraphRegistry
    from atomic_skillgraph.tools.compiler_adapter import mine_code_tools
    registry = SkillGraphRegistry(tmp_path / "graph")
    atomicizer = TraceAtomicizer(registry)
    trace = TraceRecord(
        trace_id=new_id("trace"), task_id="code_1", task_type="toy_code_arithmetic",
        task_goal="Write a function solve(x) that returns 2*x.",
        benchmark="toy_code", start_mode="cold",
        candidate_code="def double(x):\n    return 2 * x\n\n"
                       "def solve(x):\n    return double(x)\n",
        success=True,
        benchmark_result={"passed": True,
                          "tests": ["assert solve(3) == 6"]},
    )
    trace.attempts = [AttemptRecord(index=0, stage="draft",
                                    candidate=trace.candidate_code, passed=True,
                                    feedback={"tests": ["assert solve(3) == 6"]})]
    result = atomicizer.atomicize_success(trace)
    expect(bool(result.candidates), "代码轨迹应产生原子候选")
    atomicizer.apply(trace)
    # Tool 挖掘：主入口 + helper
    tools = mine_code_tools(trace, result.segments)
    expect(len(tools) >= 1, "应挖掘出代码 Tool")
    names = {t.tool_id for t in tools}
    expect(any("solve" in n for n in names), f"应包含主入口 Tool：{names}")
    expect(any("double" in n for n in names), f"应包含 helper 级 Tool（primitive 角色）：{names}")


def check_generalizer(tmp_path: Path) -> None:
    from atomic_skillgraph.tools.admission_adapter import AdmissionEngine
    from atomic_skillgraph.tools.generalizer import ToolGeneralizer
    from atomic_skillgraph.tools.registry import ToolRegistry
    from atomic_skillgraph.tools.sandbox import Sandbox
    registry = ToolRegistry(tmp_path / "tools")
    sandbox = Sandbox(tmp_root=str(tmp_path / "sandbox"))

    def make(tool_id: str, constant: int, test: str) -> ToolAsset:
        tool = _sample_tool(tool_id=tool_id,
                            code=f"def solve(x):\n    return {constant} * x\n")
        tool.signature["entry_point"] = "solve"
        tool.tests = [{"kind": "replay", "entry_point": "solve", "tests": [test]}]
        expect(AdmissionEngine(sandbox=sandbox).admit(tool).passed,
               "泛化来源 Tool 必须先通过 Admission")
        return tool

    t2 = make("arith.double", 2, "assert solve(3) == 6")
    t3 = make("arith.triple", 3, "assert solve(3) == 9")
    registry.register(t2)
    registry.register(t3)
    generalizer = ToolGeneralizer(
        registry, min_group=2,
        admission=AdmissionEngine(sandbox=sandbox), sandbox=sandbox)
    groups = generalizer.find_groups()
    expect(bool(groups), "应发现参数化候选组")
    generalized = generalizer.propose_generalized(groups[0])
    expect(generalized is not None, "应生成 generalized Tool")
    body = generalized.artifact_body()
    expect("def solve(x)" in body or "solve" in body, f"泛化代码非法：{body}")
    actions = generalizer.run_maintenance()
    expect(bool(actions), "维护应产生动作")
    expect(any(a.kind == "generalize" for a in actions), "应有 generalize 动作")


def check_evolution(tmp_path: Path) -> None:
    """成功轨迹 → 全进化管线（无 LLM）；失败轨迹 → 提案。"""
    from atomic_skillgraph.evolution.failure_processor import FailureProcessor
    from atomic_skillgraph.evolution.success_processor import SuccessProcessor
    from atomic_skillgraph.graph.registry import SkillGraphRegistry
    from atomic_skillgraph.persistence import TraceStore
    from atomic_skillgraph.tools.registry import ToolRegistry
    from atomic_skillgraph.tools.sandbox import Sandbox
    import dataclasses
    cfg = SystemConfig(data_dir=Path(tmp_path))
    cfg = dataclasses.replace(cfg, thresholds=Thresholds(admission_timeout_seconds=20.0))
    registry = SkillGraphRegistry(Path(tmp_path) / "skill_graph")
    tool_registry = ToolRegistry(Path(tmp_path) / "tools")
    trace_store = TraceStore(Path(tmp_path) / "traces")

    sandbox = Sandbox(tmp_root=str(Path(tmp_path) / "sandbox"))
    success_processor = SuccessProcessor(registry, tool_registry, trace_store, cfg,
                                         sandbox=sandbox,
                                         replay_fn=lambda tool, bindings, before: {
                                             "passed": True, "after": dict(before or {})})
    trace = _env_trace()
    trace_store.save(trace)
    result = success_processor.process_success(trace)
    expect(bool(result.atomic_refs), "应注册原子技能")
    expect(result.admitted_tools > 0, f"应有 admission 通过的 Tool：{result.notes}")
    expect(bool(result.tool_refs), "应有 Tool 引用")
    expect(registry.list_by_kind(SkillNodeKind.IMPLEMENTATION_ATOMIC),
           "应生成 Implementation Atom")

    # 失败轨迹：归因 + 提案（不激活任何 executable）
    failure_processor = FailureProcessor(registry, tool_registry, cfg)
    fail_trace = _env_trace()
    fail_trace.trace_id = new_id("trace")
    fail_trace.success = False
    fail_trace.failure_type = "benchmark_failure"
    failed_ref = str(result.atomic_refs[0])
    fail_trace.realized_atomic_nodes = [{
        "ref": failed_ref, "step_id": "step_000",
        "occurrence_id": "occ_000", "passed": False,
        "attempt_started": True, "executed_action_count": 1,
        "params": {"object": "mug_1"},
        "attempts": [{"mode": "dynamic", "started": True,
                      "passed": False, "action_start": 0,
                      "action_end": 1, "action_count": 1}],
    }]
    fail_trace.node_validators = [
        NodeValidationResult(
            node_ref=failed_ref, step_id="step_000",
            occurrence_id="occ_000", mode="dynamic",
            level="atomic", passed=False,
            checks={"preconditions": True, "effects": False},
            messages=["核心 Effect 未发生"],
        )
    ]
    fail_result = failure_processor.process_failure(fail_trace)
    expect(bool(fail_result.attributions), "应有归因结果")
    expect(bool(fail_result.proposals), "应有修复提案（shadow）")
    usable = tool_registry.list_usable()
    for tool in usable:
        expect(tool.status != ToolLifecycle.DRAFT, "失败轨迹不得产生新 active 工具")


def check_validators(tmp_path: Path) -> None:
    from atomic_skillgraph.core.predicates import StateSnapshot, check_effects, evaluate_preconditions
    from atomic_skillgraph.validation.node_validator import NodeValidator
    from atomic_skillgraph.validation.failure_localizer import FailureLocalizer
    atomic = _sample_atomic()
    before = {"facts": ["object_exists(egg_1)"], "inventory": [], "text": ""}
    after = {"facts": ["object_exists(egg_1)", "agent_holds(egg_1)"],
             "inventory": ["egg_1"], "text": ""}
    validator = NodeValidator(enabled=True)
    result = validator.validate_atomic(atomic, before, after,
                                       inputs={"object": "egg_1"})
    expect(result.passed, f"节点验证应通过：{result.messages}")
    bad_after = {"facts": ["object_exists(egg_1)"], "inventory": [], "text": ""}
    result_bad = validator.validate_atomic(atomic, before, bad_after,
                                           inputs={"object": "egg_1"})
    expect(not result_bad.passed, "Effect 未发生时验证应失败")

    # 失败定位
    trace = _env_trace()
    trace.success = False
    trace.failure_type = "benchmark_failure"
    trace.realized_atomic_nodes = [{
        "ref": str(atomic.ref), "step_id": "step_000",
        "occurrence_id": "occ_000", "passed": False,
        "attempt_started": True, "executed_action_count": 1,
        "params": {"object": "egg_1"},
        "attempts": [{"mode": "dynamic", "started": True,
                      "passed": False, "action_count": 1}],
    }]
    result_bad.step_id = "step_000"
    result_bad.occurrence_id = "occ_000"
    result_bad.mode = "dynamic"
    trace.node_validators = [result_bad]
    localizer = FailureLocalizer()
    attributions = localizer.localize(trace)
    expect(bool(attributions), "应有归因")
    expect(attributions[0].kind.value == "effect_violation"
           or attributions[0].kind.value == "precondition_violation",
           f"归因类别错误：{attributions[0].kind}")


ALL_CHECKS = [
    ("refs_and_semver", check_refs_and_semver),
    ("skill_ir_roundtrip", check_skill_ir_roundtrip),
    ("tool_ir_lifecycle", check_tool_ir_lifecycle),
    ("registry_and_graph", check_registry_and_graph),
    ("aligner", check_aligner),
    ("tool_registry_and_resolver", check_tool_registry_and_resolver),
    ("admission", check_admission),
    ("atomicizer_env", check_atomicizer_env),
    ("atomicizer_code", check_atomicizer_code),
    ("generalizer", check_generalizer),
    ("evolution", check_evolution),
    ("validators", check_validators),
]


def run_all_checks(tmp_root: Path | None = None) -> dict[str, str]:
    """运行全部检查，返回 {check_name: 'PASS'|'FAIL: ...'}。

    默认在项目 runs/ 下建临时目录（受控环境友好，避免系统临时目录权限问题）。
    """
    import shutil
    import time
    import uuid
    results: dict[str, str] = {}
    if tmp_root is not None:
        root = Path(tmp_root)
        root.mkdir(parents=True, exist_ok=True)
        cleanup_root = None
    else:
        base = Path(__file__).resolve().parents[1] / "runs" / ".smoke_tmp"
        base.mkdir(parents=True, exist_ok=True)
        root = base / f"asg_smoke_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        root.mkdir(parents=True, exist_ok=True)
        cleanup_root = root
    try:
        for name, func in ALL_CHECKS:
            try:
                if func.__code__.co_argcount == 0:
                    func()
                else:
                    func(root / name)
                results[name] = "PASS"
            except Exception as exc:  # noqa: BLE001
                results[name] = f"FAIL: {type(exc).__name__}: {exc}"
    finally:
        if cleanup_root is not None:
            shutil.rmtree(cleanup_root, ignore_errors=True)
    return results
