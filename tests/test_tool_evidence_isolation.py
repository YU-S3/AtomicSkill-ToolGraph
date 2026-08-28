from __future__ import annotations

import json
import shutil

import pytest

from atomic_skillgraph.core.refs import ToolRef
from atomic_skillgraph.core.status import ArtifactKind, ToolLifecycle
from atomic_skillgraph.core.tool_ir import ToolAsset
from atomic_skillgraph.evidence_store import EvidenceStore
from atomic_skillgraph.persistence_guard import validate_long_term_asset
from atomic_skillgraph.tools.admission_adapter import AdmissionEngine
from atomic_skillgraph.tools.generalizer import ToolGeneralizer
from atomic_skillgraph.tools.registry import ToolRegistry
from atomic_skillgraph.validation.tool_validator import ToolValidator


def _admitted_code_tool() -> tuple[ToolAsset, dict]:
    replay_case = {
        "kind": "replay",
        "source_trace_id": "trace_private_case",
        "tests": ["assert solve() == 2"],
        "bindings": {"customer": "customer_7"},
        "before": {"path": r"D:\private\customer_7\report.xlsx"},
        "after": {"uploaded_to": "https://internal.example/report"},
        "prefix": ["contact analyst@example.com"],
        "source": {"task_id": "customer_7_task", "env_index": 7},
    }
    tool = ToolAsset(
        ref=ToolRef("test.portable-solver", "0.1.0"),
        artifact_kind=ArtifactKind.PYTHON_CALLABLE,
        summary="Portable arithmetic callable",
        signature={"entry_point": "solve", "parameters": []},
        interface={"inputs": [], "outputs": [{"name": "result"}]},
        artifact={"code": "def solve(value=1):\n    return value + 1\n"},
        tests=[replay_case],
        safety={"direct_execution_allowed": True, "checks_passed": []},
        status=ToolLifecycle.DRAFT,
    )
    result = AdmissionEngine().admit(tool)
    assert result.passed
    assert tool.admission_certificate_valid()
    return tool, replay_case


def _admitted_action_tool() -> tuple[ToolAsset, dict]:
    replay_case = {
        "kind": "replay",
        "source_trace_id": "trace_private_action_case",
        "bindings": {"object": "mug_7"},
        "before": {"location": "countertop_3"},
        "after": {"held": "mug_7"},
    }
    tool = ToolAsset(
        ref=ToolRef("test.portable-action", "0.1.0"),
        artifact_kind=ArtifactKind.ACTION_TEMPLATE,
        summary="Portable object interaction",
        signature={"parameters": [
            {"name": "object", "semantic_type": "object_ref",
             "required": True},
        ]},
        interface={"inputs": {"object": "object_ref"},
                   "outputs": {"held": "object_ref"}},
        artifact={"template": "take {object}",
                  "steps": ["take {object}"]},
        tests=[replay_case],
        safety={"direct_execution_allowed": True, "checks_passed": []},
        status=ToolLifecycle.DRAFT,
    )
    result = AdmissionEngine(
        replay_fn=lambda _tool, _bindings, _before: {
            "passed": True, "after": {"held": "mug_7"},
        }).admit(tool)
    assert result.passed
    assert tool.admission_certificate_valid()
    return tool, replay_case


def test_tool_registry_externalizes_and_hydrates_private_replay_case(
        workspace_tmp):
    data = workspace_tmp / "data"
    registry = ToolRegistry(data / "tools")
    tool, replay_case = _admitted_code_tool()

    registry.register(tool)
    persisted_path = data / "tools" / tool.tool_id / "0.1.0" / "tool.json"
    persisted = json.loads(persisted_path.read_text(encoding="utf-8"))

    assert len(persisted["tests"]) == 1
    assert set(persisted["tests"][0]) == {"evidence_ref", "evidence_hash"}
    serialized = json.dumps(persisted, ensure_ascii=False)
    for private_literal in (
            "customer_7", "report.xlsx", "internal.example",
            "analyst@example.com", "env_index", "bindings", "prefix"):
        assert private_literal not in serialized

    evidence_files = list((data / "evidence" / "tool_tests").rglob("*.json"))
    assert len(evidence_files) == 1
    assert "customer_7" in evidence_files[0].read_text(encoding="utf-8")

    loaded = registry.get(tool.ref)
    assert loaded is not None
    assert loaded.replay_cases() == [replay_case]
    assert not loaded.has_unresolved_test_evidence()
    assert loaded.admission_certificate_valid()


def test_shadow_ref_can_be_readmitted_without_losing_fresh_certificate(
        workspace_tmp):
    registry = ToolRegistry(workspace_tmp / "readmission" / "tools")
    admitted, replay_case = _admitted_code_tool()
    shadow = ToolAsset.from_dict(admitted.to_dict())
    shadow.status = ToolLifecycle.SHADOW
    shadow.safety = {"direct_execution_allowed": True,
                     "checks_passed": []}
    registry.register(shadow)

    fresh = ToolAsset(
        ref=admitted.ref,
        artifact_kind=admitted.artifact_kind,
        summary=admitted.summary,
        signature=dict(admitted.signature),
        interface=dict(admitted.interface),
        artifact=dict(admitted.artifact),
        tests=[replay_case],
        safety={"direct_execution_allowed": True, "checks_passed": []},
        status=ToolLifecycle.DRAFT,
    )
    result = AdmissionEngine().admit(fresh)
    assert result.passed
    registry.register(fresh)

    loaded = registry.get(fresh.ref)
    assert loaded is not None
    assert loaded.status == ToolLifecycle.CANDIDATE
    assert loaded.admission_certificate_valid()
    assert loaded.replay_cases() == [replay_case]


def test_frozen_tool_without_private_evidence_uses_admission_certificate(
        workspace_tmp):
    online = workspace_tmp / "online"
    registry = ToolRegistry(online / "tools")
    tool, _case = _admitted_code_tool()
    registry.register(tool)

    frozen = workspace_tmp / "frozen"
    shutil.copytree(online / "tools", frozen / "tools")
    assert not (frozen / "evidence").exists()

    frozen_registry = ToolRegistry(frozen / "tools")
    loaded = frozen_registry.get(tool.ref)
    assert loaded is not None
    assert loaded.replay_cases() == []
    assert loaded.has_unresolved_test_evidence()
    assert loaded.admission_certificate_valid()

    validation = ToolValidator().validate_tool(loaded, {})
    assert validation.passed
    assert validation.checks["admission_certificate"]
    assert validation.checks["evidence_or_certificate"]
    assert validation.checks["replay_tests"]


def test_tampered_frozen_tool_cannot_use_admission_certificate(workspace_tmp):
    online = workspace_tmp / "online_tamper"
    registry = ToolRegistry(online / "tools")
    tool, _case = _admitted_code_tool()
    registry.register(tool)

    frozen = workspace_tmp / "frozen_tamper"
    shutil.copytree(online / "tools", frozen / "tools")
    path = frozen / "tools" / tool.tool_id / "0.1.0" / "tool.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["artifact"]["code"] = "def solve(value=1):\n    return 999\n"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = ToolRegistry(frozen / "tools").get(tool.ref)
    assert loaded is not None
    assert not loaded.admission_certificate_valid()
    validation = ToolValidator().validate_tool(loaded, {})
    assert not validation.passed
    assert not validation.checks["admission_certificate"]


def test_frozen_action_tool_without_evidence_uses_bound_certificate(
        workspace_tmp):
    online = workspace_tmp / "online_action"
    registry = ToolRegistry(online / "tools")
    tool, _case = _admitted_action_tool()
    certificate_before = dict(tool.safety["admission_certificate"])
    registry.register(tool)

    # Externalization must not change the test hashes covered by Admission.
    persisted = registry.get(tool.ref)
    assert persisted is not None
    assert persisted.safety["admission_certificate"] == certificate_before
    assert persisted.admission_certificate_valid()

    frozen = workspace_tmp / "frozen_action"
    shutil.copytree(online / "tools", frozen / "tools")
    loaded = ToolRegistry(frozen / "tools").get(tool.ref)
    assert loaded is not None
    assert loaded.has_unresolved_test_evidence()
    assert loaded.replay_cases() == []

    validation = ToolValidator().validate_tool(
        loaded, {"object": "runtime_object"})
    assert validation.passed
    assert validation.checks["admission_certificate"]
    assert validation.checks["evidence_or_certificate"]


def test_tampered_frozen_action_tool_fails_closed(workspace_tmp):
    online = workspace_tmp / "online_action_tamper"
    registry = ToolRegistry(online / "tools")
    tool, _case = _admitted_action_tool()
    registry.register(tool)

    frozen = workspace_tmp / "frozen_action_tamper"
    shutil.copytree(online / "tools", frozen / "tools")
    path = frozen / "tools" / tool.tool_id / "0.1.0" / "tool.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["artifact"]["steps"] = ["place {object}"]
    payload["artifact"]["template"] = "place {object}"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = ToolRegistry(frozen / "tools").get(tool.ref)
    assert loaded is not None
    assert not loaded.admission_certificate_valid()
    validation = ToolValidator().validate_tool(
        loaded, {"object": "runtime_object"})
    assert not validation.passed
    assert not validation.checks["admission_certificate"]
    assert not validation.checks["evidence_or_certificate"]


def test_frozen_action_tool_with_certificate_removed_fails_closed(
        workspace_tmp):
    online = workspace_tmp / "online_action_stripped"
    registry = ToolRegistry(online / "tools")
    tool, _case = _admitted_action_tool()
    registry.register(tool)

    frozen = workspace_tmp / "frozen_action_stripped"
    shutil.copytree(online / "tools", frozen / "tools")
    path = frozen / "tools" / tool.tool_id / "0.1.0" / "tool.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["safety"].pop("admission_certificate")
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = ToolRegistry(frozen / "tools").get(tool.ref)
    assert loaded is not None
    validation = ToolValidator().validate_tool(
        loaded, {"object": "runtime_object"})
    assert not validation.passed
    assert not validation.checks["evidence_or_certificate"]


@pytest.mark.parametrize(("payload", "code"), [
    ({"artifact": {"steps": ["take mug_1"]}}, "concrete_instance"),
    ({"summary": r"read D:\private\customer\report.xlsx"},
     "absolute_windows_path"),
    ({"summary": "read /home/user/private/report.xlsx"},
     "absolute_posix_path"),
    ({"summary": "open https://internal.example/report"}, "url"),
    ({"summary": "contact analyst@example.com"}, "email"),
    ({"artifact": {"code": 'API_KEY="sk-abcdefghijklmnopqrstuvwxyz"'}},
     "secret_literal"),
])
def test_long_term_asset_guard_rejects_private_or_instance_literals(
        payload, code):
    findings = validate_long_term_asset(
        payload, asset_kind="tool:action_template")
    assert any(item.startswith(code + ":") for item in findings)


def test_long_term_asset_guard_allows_structural_numbered_symbols():
    findings = validate_long_term_asset({
        "summary": "generalized from 2 variants",
        "signature": {"parameters": [
            {"name": "PARAM_0"}, {"name": "ARG_1"}, {"name": "INPUT_2"},
        ]},
        "artifact": {"code": "def helper_1(PARAM_0):\n    return PARAM_0\n"},
    }, asset_kind="tool:python_callable")
    assert findings == []


def test_tool_registry_refuses_instance_residue_in_portable_artifact(
        workspace_tmp):
    registry = ToolRegistry(workspace_tmp / "guarded" / "tools")
    tool = ToolAsset(
        ref=ToolRef("test.bad-template", "0.1.0"),
        artifact_kind=ArtifactKind.ACTION_TEMPLATE,
        summary="Bad unparameterized template",
        signature={"parameters": [
            {"name": "object", "semantic_type": "object_ref", "required": True},
        ]},
        interface={"inputs": {"object": "object_ref"}, "outputs": {}},
        artifact={"template": "take mug_1", "steps": ["take mug_1"]},
        tests=[],
        status=ToolLifecycle.DRAFT,
    )
    with pytest.raises(ValueError, match="long_term_asset_guard_failed"):
        registry.register(tool)
    assert not (registry.root / tool.tool_id / "0.1.0" / "tool.json").exists()


@pytest.mark.parametrize("artifact_literal", ["customer_a", "customer a"])
def test_action_admission_rejects_unnumbered_grounded_binding_residue(
        artifact_literal):
    tool = ToolAsset(
        ref=ToolRef("test.grounded-template", "0.1.0"),
        artifact_kind=ArtifactKind.ACTION_TEMPLATE,
        summary="A supposedly portable action",
        signature={"parameters": [
            {"name": "object", "semantic_type": "object_ref",
             "required": True},
            {"name": "target", "semantic_type": "location_ref",
             "required": True},
        ]},
        interface={"inputs": {"object": "object_ref",
                              "target": "location_ref"}, "outputs": {}},
        artifact={
            "template": f"move {artifact_literal} to {{target}} using {{object}}",
            "steps": [
                f"move {artifact_literal} to {{target}} using {{object}}"],
        },
        tests=[{
            "kind": "replay",
            "bindings": {"object": "customer_a", "target": "archive"},
            "before": {}, "after": {},
        }],
        status=ToolLifecycle.DRAFT,
    )
    replay_called = False

    def replay(*_args):
        nonlocal replay_called
        replay_called = True
        return {"passed": True}

    result = AdmissionEngine(replay_fn=replay).admit(tool)
    assert not result.passed
    assert not result.checks["instance_free_template"]
    assert any(reason.startswith("unparameterized_grounded_bindings:")
               for reason in result.reasons)
    assert not replay_called


def test_action_admission_rejects_grounded_binding_in_public_summary():
    tool = ToolAsset(
        ref=ToolRef("test.grounded-summary", "0.1.0"),
        artifact_kind=ArtifactKind.ACTION_TEMPLATE,
        summary="Move customer_a into the archive",
        signature={"parameters": [
            {"name": "object", "semantic_type": "object_ref",
             "required": True},
            {"name": "target", "semantic_type": "location_ref",
             "required": True},
        ]},
        interface={"inputs": {"object": "object_ref",
                              "target": "location_ref"}, "outputs": {}},
        artifact={"template": "move {object} to {target}",
                  "steps": ["move {object} to {target}"]},
        tests=[{"kind": "replay", "bindings": {
            "object": "customer_a", "target": "archive"}}],
        status=ToolLifecycle.DRAFT,
    )

    result = AdmissionEngine(
        replay_fn=lambda *_args: {"passed": True}).admit(tool)

    assert not result.passed
    assert any("unparameterized_grounded_bindings" in reason
               for reason in result.reasons)


def test_evidence_store_detects_corruption(workspace_tmp):
    store = EvidenceStore(workspace_tmp / "evidence")
    payload = {"before": {"object": "mug_1"}, "after": {"held": True}}
    ref = store.put(
        "tool_test", payload, trace_id="trace_one",
        event_start=2, event_end=3)
    assert store.get(ref) == payload

    digest = ref["evidence_hash"]
    path = store.root / "tool_test" / digest[:2] / f"{digest}.json"
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["after"]["held"] = False
    path.write_text(json.dumps(envelope), encoding="utf-8")
    assert store.get(ref) is None


def test_python_callable_can_be_readmitted_after_test_externalization(
        workspace_tmp):
    registry = ToolRegistry(workspace_tmp / "readmit" / "tools")
    tool, _case = _admitted_code_tool()
    registry.register(tool)

    loaded = registry.get(tool.ref)
    assert loaded is not None and loaded.all_test_cases()
    result = AdmissionEngine().admit(loaded)
    assert result.passed
    assert loaded.admission_certificate_valid()
    registry.register(loaded)
    assert registry.get(tool.ref).admission_certificate_valid()


def test_update_tool_rejects_contract_safety_and_evidence_mutation(
        workspace_tmp):
    registry = ToolRegistry(workspace_tmp / "immutable_updates" / "tools")
    tool, _case = _admitted_code_tool()
    registry.register(tool)

    statistics_only = registry.get(tool.ref)
    statistics_only.statistics["support_count"] = 4
    registry.update_tool(statistics_only)
    assert registry.get(tool.ref).statistics["support_count"] == 4

    artifact_change = registry.get(tool.ref)
    artifact_change.artifact["code"] = "def solve(value=1):\n    return 99\n"
    with pytest.raises(ValueError, match="immutable_tool_contract_update"):
        registry.update_tool(artifact_change)

    safety_change = registry.get(tool.ref)
    safety_change.safety["direct_execution_allowed"] = False
    assert not safety_change.admission_certificate_valid()
    with pytest.raises(ValueError, match="immutable_tool_contract_update"):
        registry.update_tool(safety_change)

    evidence_change = registry.get(tool.ref)
    evidence_change.tests = []
    with pytest.raises(ValueError, match="immutable_tool_evidence_update"):
        registry.update_tool(evidence_change)


def test_parameterized_python_replay_is_consistent_online_and_frozen(
        workspace_tmp):
    tool = ToolAsset(
        ref=ToolRef("test.parameterized", "0.1.0"),
        artifact_kind=ArtifactKind.PYTHON_CALLABLE,
        summary="Parameterized arithmetic callable",
        signature={"entry_point": "solve", "parameters": []},
        interface={"inputs": [], "outputs": [{"name": "result"}]},
        artifact={"code": "def solve(value=1):\n    return value + OFFSET\n",
                  "template_parameters": ["OFFSET"]},
        tests=[{"kind": "parameterized_replay", "bindings": [
            {"value": {"OFFSET": 1}, "tests": ["assert solve() == 2"]},
            {"value": {"OFFSET": 2}, "tests": ["assert solve() == 3"]},
        ]}],
        safety={"direct_execution_allowed": True},
        status=ToolLifecycle.DRAFT,
    )
    assert AdmissionEngine().admit(tool).passed
    online = workspace_tmp / "parameterized_online"
    registry = ToolRegistry(online / "tools")
    registry.register(tool)
    assert ToolValidator().validate_tool(registry.get(tool.ref), {}).passed

    frozen = workspace_tmp / "parameterized_frozen"
    shutil.copytree(online / "tools", frozen / "tools")
    loaded = ToolRegistry(frozen / "tools").get(tool.ref)
    assert loaded.has_unresolved_test_evidence()
    assert ToolValidator().validate_tool(loaded, {}).passed


def test_certificate_binds_evidence_ref_hash_pair_and_multiplicity(
        workspace_tmp):
    registry = ToolRegistry(workspace_tmp / "evidence_identity" / "tools")
    tool, replay_case = _admitted_code_tool()
    tool.tests = [replay_case, dict(replay_case)]
    assert AdmissionEngine().admit(tool).passed
    registry.register(tool)
    certificate = tool.safety["admission_certificate"]
    assert len(certificate["test_evidence"]) == 2
    path = registry.root / tool.tool_id / "0.1.0" / "tool.json"

    removed = json.loads(path.read_text(encoding="utf-8"))
    removed["tests"] = removed["tests"][:1]
    path.write_text(json.dumps(removed), encoding="utf-8")
    assert not registry.get(tool.ref).admission_certificate_valid()

    # Restore, then alter only the URI while retaining the certified hash.
    registry.register(tool)
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["tests"][0]["evidence_ref"] = (
        "evidence://tool_test/" + "0" * 64)
    path.write_text(json.dumps(tampered), encoding="utf-8")
    loaded = registry.get(tool.ref)
    assert loaded.has_unresolved_test_evidence()
    assert not loaded.admission_certificate_valid()
    assert not ToolValidator().validate_tool(loaded, {}).passed


def test_manual_usable_tool_without_admission_certificate_is_rejected(
        workspace_tmp):
    tool, _case = _admitted_code_tool()
    tool.safety.pop("admission_certificate")
    tool.status = ToolLifecycle.CANDIDATE
    registry = ToolRegistry(workspace_tmp / "manual_candidate" / "tools")
    with pytest.raises(
            ValueError, match="usable_tool_requires_valid_admission_certificate"):
        registry.register(tool)


def _install_legacy_tool(registry: ToolRegistry, tool: ToolAsset) -> None:
    path = registry.root / tool.tool_id / tool.ref.version / "tool.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tool.to_dict()), encoding="utf-8")
    registry.registry_path.write_text(json.dumps({
        tool.tool_id: {
            "current_version": tool.ref.version,
            "recommended_version": tool.ref.version,
            "latest_status": tool.status.value,
            "versions": [tool.ref.version],
        },
    }), encoding="utf-8")


@pytest.mark.parametrize("kind", [
    ArtifactKind.PYTHON_CALLABLE,
    ArtifactKind.ACTION_TEMPLATE,
])
def test_explicit_legacy_migration_externalizes_then_replays_before_certificate(
        workspace_tmp, kind):
    if kind == ArtifactKind.PYTHON_CALLABLE:
        tool, _case = _admitted_code_tool()
    else:
        tool, _case = _admitted_action_tool()
    tool.status = ToolLifecycle.ACTIVE
    tool.safety.pop("admission_certificate")
    registry = ToolRegistry(
        workspace_tmp / f"legacy_{kind.value}" / "tools")
    _install_legacy_tool(registry, tool)

    before = registry.audit_frozen_readiness()
    assert not before["passed"]
    assert {item["code"] for item in before["issues"]} >= {
        "raw_tests_embedded", "usable_missing_valid_admission_certificate"}

    # Explicit migration without a replay authority removes private payloads
    # but must not silently sign the old usable Tool.
    first = registry.migrate_legacy_assets()
    assert first["externalized"] == 1
    assert first["readmitted"] == 0
    assert not first["passed"]
    assert not registry.get(tool.ref).admission_certificate_valid()

    if kind == ArtifactKind.PYTHON_CALLABLE:
        second = registry.migrate_legacy_assets(
            admission_factory=lambda _tool: AdmissionEngine())
    else:
        second = registry.migrate_legacy_assets(
            replay_callback=lambda _tool, bindings, _before: {
                "passed": bool(bindings.get("object")), "after": {}})
    assert second["readmitted"] == 1
    assert second["passed"]
    migrated = registry.get(tool.ref)
    assert migrated.status == ToolLifecycle.ACTIVE
    assert migrated.admission_certificate_valid()
    assert registry.assert_frozen_ready()["passed"]


def test_frozen_snapshot_entry_rejects_legacy_raw_tool(workspace_tmp):
    from experiments.run_evolve_eval import _snapshot_frozen_bank

    src_data = workspace_tmp / "legacy_snapshot_source"
    registry = ToolRegistry(src_data / "tools")
    tool, _case = _admitted_code_tool()
    tool.status = ToolLifecycle.ACTIVE
    tool.safety.pop("admission_certificate")
    _install_legacy_tool(registry, tool)

    with pytest.raises(RuntimeError, match="tool_frozen_readiness_failed"):
        _snapshot_frozen_bank(
            src_data, workspace_tmp / "legacy_snapshot_destination" / "data")


def test_branch_candidate_copies_hydrated_evidence_into_main_registry(
        workspace_tmp):
    branch = ToolRegistry(workspace_tmp / "branch_bank" / "tools")
    main = ToolRegistry(workspace_tmp / "main_bank" / "tools")
    candidate, replay_case = _admitted_action_tool()

    branch.register(candidate)
    assert candidate.tests[0]["evidence_ref"].startswith("evidence://")
    assert candidate.all_test_cases() == [replay_case]
    assert not list((workspace_tmp / "main_bank" / "evidence").rglob("*.json"))

    main.register(candidate)
    copied = main.get(candidate.ref)
    assert copied is not None
    assert copied.all_test_cases() == [replay_case]
    assert not copied.has_unresolved_test_evidence()
    assert copied.admission_certificate_valid()
    assert len(list((workspace_tmp / "main_bank" / "evidence").rglob(
        "*.json"))) == 1


def test_cross_registry_write_rejects_dangling_evidence_ref(workspace_tmp):
    branch = ToolRegistry(workspace_tmp / "dangling_branch" / "tools")
    candidate, _case = _admitted_action_tool()
    branch.register(candidate)

    # Simulate a serialized branch asset without either its private store or
    # the transient hydrated payload kept on the original object.
    detached = ToolAsset.from_dict(candidate.to_dict())
    main = ToolRegistry(workspace_tmp / "dangling_main" / "tools")
    with pytest.raises(ValueError, match="missing_tool_evidence_payload"):
        main.register(detached)


def test_specialize_and_split_preserve_hydrated_tests_for_readmission(
        workspace_tmp):
    registry = ToolRegistry(workspace_tmp / "evolution_hydration" / "tools")
    code, _ = _admitted_code_tool()
    action, _ = _admitted_action_tool()
    registry.register(code)
    registry.register(action)
    generalizer = ToolGeneralizer(registry)

    specialized = generalizer.propose_specialized(
        registry.get(code.ref), {"domain": "positive"}, "bounded domain")
    assert specialized.all_test_cases()
    assert AdmissionEngine().admit(specialized).passed
    registry.register(specialized)

    children = generalizer.propose_split(
        registry.get(action.ref), [{"name": "core", "step_indices": [0]}])
    assert len(children) == 1 and children[0].all_test_cases()
    assert AdmissionEngine(
        replay_fn=lambda _tool, bindings, _before: {
            "passed": bool(bindings.get("object")), "after": {}}).admit(
                children[0]).passed
    registry.register(children[0])
