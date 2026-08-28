from atomic_skillgraph.core.binding_ir import is_concrete_binding
from atomic_skillgraph.core.refs import SkillRef, ToolRef
from atomic_skillgraph.core.skill_ir import ImplementationAtom, ToolBinding
from atomic_skillgraph.core.status import ArtifactKind, SkillStatus, ToolLifecycle
from atomic_skillgraph.core.tool_ir import ToolAsset
from atomic_skillgraph.tools.admission_adapter import AdmissionEngine
from atomic_skillgraph.tools.registry import ToolRegistry
from atomic_skillgraph.tools.resolver import ToolResolver


def test_nested_symbolic_values_are_not_concrete():
    for value in ("$flow.container", "$task.unknown", "$inputs.object",
                  {"path": "$flow.file"}, ["literal", "$flow.x"], None, ""):
        assert not is_concrete_binding(value)


def test_resolver_requires_exact_tool_version_and_concrete_values(workspace_tmp):
    tools = ToolRegistry(workspace_tmp / "tools")
    tool = ToolAsset(
        ref=ToolRef("generic.take", "1.0.0"),
        artifact_kind=ArtifactKind.ACTION_TEMPLATE,
        summary="take an object",
        signature={"parameters": [{"name": "object"}]},
        artifact={"steps": ["take {object}"]}, status=ToolLifecycle.CANDIDATE)
    assert AdmissionEngine(
        replay_fn=lambda *_args: {"passed": True}).admit(tool).passed
    tools.register(tool)
    impl = ImplementationAtom(
        ref=SkillRef("impl.take", "1.0.0"),
        abstract_ref=SkillRef("generic.take", "1.0.0"),
        tool_bindings=[ToolBinding(
            ToolRef("generic.take", "1.0.0"),
            parameter_mapping={"object": "$inputs.object"})],
        status=SkillStatus.ACTIVE)
    resolved = ToolResolver(tools).resolve(
        impl, {"inputs": {"object": "$flow.object"}})[0]
    assert not resolved.ok and resolved.missing == ["object"]
    impl.tool_bindings[0].tool_ref = ToolRef("generic.take", "9.9.9")
    missing_version = ToolResolver(tools).resolve(
        impl, {"inputs": {"object": "mug_1"}})[0]
    assert missing_version.missing == ["tool_version_missing"]
