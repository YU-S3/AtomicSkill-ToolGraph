from atomic_skillgraph.core.config import SystemConfig
from atomic_skillgraph.core.refs import SkillRef, ToolRef
from atomic_skillgraph.core.skill_ir import (
    AbstractAtomicSkill,
    ImplementationAtom,
    ToolBinding,
)
from atomic_skillgraph.core.status import ArtifactKind, SkillStatus, ToolLifecycle
from atomic_skillgraph.core.tool_ir import ToolAsset
from atomic_skillgraph.graph.registry import SkillGraphRegistry
from atomic_skillgraph.runtime.implementation_selector import ImplementationSelector
from atomic_skillgraph.tools.registry import ToolRegistry
from atomic_skillgraph.tools.resolver import ToolResolver
from experiments.report import summarize_episodes


def _acquire_tool(tool_id: str, steps: list[str]) -> ToolAsset:
    return ToolAsset(
        ref=ToolRef(tool_id, "1.0.0"),
        artifact_kind=ArtifactKind.ACTION_TEMPLATE,
        summary="generic acquire",
        signature={"parameters": [{"name": "object"},
                                   {"name": "object_location"}]},
        artifact={"steps": steps},
        status=ToolLifecycle.CANDIDATE,
    )


def test_discovery_context_prefers_take_only_acquire(workspace_tmp):
    graph = SkillGraphRegistry(workspace_tmp / "selector_graph")
    tools = ToolRegistry(workspace_tmp / "selector_tools")
    atomic = AbstractAtomicSkill(
        ref=SkillRef("alfworld.acquire_object", "1.0.0"),
        summary="acquire object",
        inputs=[{"name": "object"}, {"name": "object_location"}],
        outputs=[{"name": "object"}],
        effects=[{"predicate": "agent.holds",
                  "args": {"object": "$inputs.object"}}],
        status=SkillStatus.ACTIVE,
    )
    graph.register(atomic)
    go_take = _acquire_tool(
        "alfworld.acquire.go_take",
        ["go to {object_location}", "take {object} from {object_location}"],
    )
    take_only = _acquire_tool(
        "alfworld.acquire.take_only",
        ["take {object} from {object_location}"],
    )
    tools.register(go_take)
    tools.register(take_only)
    go_impl = ImplementationAtom(
        ref=SkillRef("impl.acquire.go_take", "1.0.0"),
        abstract_ref=atomic.ref,
        tool_bindings=[ToolBinding(go_take.ref)],
        quality={"utility": 0.9, "success_count": 10},
    )
    take_impl = ImplementationAtom(
        ref=SkillRef("impl.acquire.take_only", "1.0.0"),
        abstract_ref=atomic.ref,
        tool_bindings=[ToolBinding(take_only.ref)],
        quality={"utility": 0.4, "success_count": 0},
    )
    graph.register(go_impl)
    graph.register(take_impl)
    selector = ImplementationSelector(graph, ToolResolver(tools), SystemConfig())
    context = {"harness": "env", "inputs": {
        "object": "mug_1", "object_location": "countertop_1"}}
    assert selector.select(atomic.ref, context).implementation.ref == go_impl.ref
    context["prefer_take_only_acquire"] = True
    assert selector.select(atomic.ref, context).implementation.ref == take_impl.ref
    ranked = selector.rank(atomic.ref, context)
    assert [choice.implementation.ref for choice in ranked] == [
        take_impl.ref, go_impl.ref]


def test_report_separates_any_direct_node_rate_and_all_direct_episodes():
    episodes = [
        {"episode": 1, "success": True, "retries": 0,
         "planned_node_count": 3, "executed_node_count": 3,
         "node_mode_counts": {"direct": 3}, "direct_reuse_count": 3,
         "seeded_generation_count": 0, "dynamic_generation_count": 0},
        {"episode": 2, "success": True, "retries": 0,
         "planned_node_count": 3, "executed_node_count": 3,
         "node_mode_counts": {"direct": 2, "seeded": 1},
         "direct_reuse_count": 2, "seeded_generation_count": 1,
         "dynamic_generation_count": 0},
    ]
    summary = summarize_episodes(episodes)
    assert summary["any_direct_episode_rate"] == 1.0
    assert summary["direct_node_rate"] == 0.8333
    assert summary["all_nodes_direct_episode_rate"] == 0.5
    assert summary["direct_node_count"] == 5
    assert summary["seeded_node_count"] == 1
