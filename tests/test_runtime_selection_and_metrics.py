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
from atomic_skillgraph.tools.admission_adapter import AdmissionEngine
from atomic_skillgraph.tools.registry import ToolRegistry
from atomic_skillgraph.tools.resolver import ToolResolver
from experiments.report import summarize_episodes


def _action_tool(tool_id: str, steps: list[str]) -> ToolAsset:
    parameters = [{"name": "fixture"}]
    if any("{fixture_location}" in step for step in steps):
        parameters.append({"name": "fixture_location"})
    tool = ToolAsset(
        ref=ToolRef(tool_id, "1.0.0"),
        artifact_kind=ArtifactKind.ACTION_TEMPLATE,
        summary="generic verified transition",
        signature={"parameters": parameters},
        artifact={"steps": steps},
        status=ToolLifecycle.CANDIDATE,
    )
    assert AdmissionEngine(
        replay_fn=lambda *_args: {"passed": True}).admit(tool).passed
    return tool


def test_prepared_context_prefers_structurally_minimal_tool(workspace_tmp):
    graph = SkillGraphRegistry(workspace_tmp / "selector_graph")
    tools = ToolRegistry(workspace_tmp / "selector_tools")
    atomic = AbstractAtomicSkill(
        ref=SkillRef("generic.fixture_engaged", "1.0.0"),
        summary="engage fixture",
        inputs=[{"name": "fixture"}, {"name": "fixture_location"}],
        outputs=[{"name": "fixture"}],
        effects=[{"predicate": "fixture.engaged",
                  "args": {"object": "$inputs.fixture"}}],
        status=SkillStatus.ACTIVE,
    )
    graph.register(atomic)
    prepared_and_core = _action_tool(
        "generic.fixture.move_engage",
        ["move near {fixture_location}", "engage {fixture}"],
    )
    core_only = _action_tool(
        "generic.fixture.engage_only",
        ["engage {fixture}"],
    )
    tools.register(prepared_and_core)
    tools.register(core_only)
    prepared_impl = ImplementationAtom(
        ref=SkillRef("impl.fixture.move_engage", "1.0.0"),
        abstract_ref=atomic.ref,
        tool_bindings=[ToolBinding(prepared_and_core.ref)],
        quality={"utility": 0.9, "success_count": 10},
    )
    core_impl = ImplementationAtom(
        ref=SkillRef("impl.fixture.engage_only", "1.0.0"),
        abstract_ref=atomic.ref,
        tool_bindings=[ToolBinding(core_only.ref)],
        quality={"utility": 0.4, "success_count": 0},
    )
    graph.register(prepared_impl)
    graph.register(core_impl)
    selector = ImplementationSelector(graph, ToolResolver(tools), SystemConfig())
    context = {"harness": "env", "inputs": {
        "fixture": "fixture_1", "fixture_location": "bay_1"}}
    assert selector.select(atomic.ref, context).implementation.ref == prepared_impl.ref
    context["prefer_minimal_after_preparation"] = True
    assert selector.select(atomic.ref, context).implementation.ref == core_impl.ref
    ranked = selector.rank(atomic.ref, context)
    assert [choice.implementation.ref for choice in ranked] == [
        core_impl.ref, prepared_impl.ref]


def test_selector_discovers_location_slot_from_learned_tool_contract(workspace_tmp):
    graph = SkillGraphRegistry(workspace_tmp / "generic_location_graph")
    tools = ToolRegistry(workspace_tmp / "generic_location_tools")
    atomic = AbstractAtomicSkill(
        ref=SkillRef("generic.fixture_engaged", "1.0.0"),
        summary="verified transition", inputs=[{"name": "fixture"}],
        effects=[{"predicate": "fixture.engaged",
                  "args": {"object": "$inputs.fixture"}}],
        status=SkillStatus.ACTIVE)
    graph.register(atomic)
    tool = ToolAsset(
        ref=ToolRef("generic.engage", "1.0.0"),
        artifact_kind=ArtifactKind.ACTION_TEMPLATE,
        summary="generic executable",
        signature={"parameters": [
            {"name": "fixture"}, {"name": "fixture_location"}]},
        artifact={"steps": [
            "move near {fixture_location}", "engage {fixture}"]},
        status=ToolLifecycle.CANDIDATE)
    assert AdmissionEngine(
        replay_fn=lambda *_args: {"passed": True}).admit(tool).passed
    tools.register(tool)
    impl = ImplementationAtom(
        ref=SkillRef("impl.generic.fixture_engaged", "1.0.0"),
        abstract_ref=atomic.ref,
        tool_bindings=[ToolBinding(tool.ref)],
        quality={"utility": 0.7, "success_count": 2},
        status=SkillStatus.ACTIVE)
    graph.register(impl)
    selector = ImplementationSelector(graph, ToolResolver(tools), SystemConfig())
    context = {"harness": "env", "inputs": {"fixture": "fixture_3"}}
    assert selector.discoverable_location_slots(atomic.ref, context) == {
        "fixture_location"}
    assert selector.select_allowing_missing(
        atomic.ref, context, {"fixture_location"}).implementation.ref == impl.ref


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
    assert summary["all_executed_nodes_direct_episode_rate"] == 0.5
    assert summary["full_plan_direct_episode_rate"] == 0.5
    assert summary["goal_early_terminal_episode_rate"] == 0.0
    assert summary["direct_node_count"] == 5
    assert summary["seeded_node_count"] == 1


def test_report_separates_goal_early_terminal_from_direct_failure():
    episodes = [
        {"episode": 1, "success": True, "retries": 0,
         "planned_node_count": 3, "executed_node_count": 2,
         "goal_terminal_before_plan_complete": True,
         "goal_terminal_skipped_node_count": 1,
         "node_mode_counts": {"direct": 2}, "direct_reuse_count": 2,
         "seeded_generation_count": 0, "dynamic_generation_count": 0},
    ]
    summary = summarize_episodes(episodes)
    assert summary["direct_node_rate"] == 1.0
    assert summary["all_executed_nodes_direct_episode_rate"] == 1.0
    assert summary["full_plan_direct_episode_rate"] == 0.0
    # Backward-compatible strict metric retains its original meaning.
    assert summary["all_nodes_direct_episode_rate"] == 0.0
    assert summary["goal_early_terminal_episode_count"] == 1
    assert summary["goal_early_terminal_episode_rate"] == 1.0
    assert summary["goal_terminal_skipped_node_count"] == 1
