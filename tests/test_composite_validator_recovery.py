from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.skill_ir import CompositeSkill
from atomic_skillgraph.core.trace_ir import NodeValidationResult
from atomic_skillgraph.validation.composite_validator import CompositeValidator


def _node(logical_id: str, passed: bool, message: str = "") -> NodeValidationResult:
    return NodeValidationResult(
        node_ref=f"skill://{logical_id}@1.0.0",
        level="atomic",
        passed=passed,
        checks={"effects": passed},
        messages=[message] if message else [],
    )


def _heat_place_composite() -> CompositeSkill:
    return CompositeSkill(
        ref=SkillRef(
            "composite.alfworld.acquire_object-heat_object-place_object",
            "1.0.0",
        ),
        graph={
            "nodes": [
                "alfworld.acquire_object@1.0.0",
                "alfworld.heat_object@1.0.0",
                "alfworld.place_object@1.0.0",
            ],
        },
        validator={
            "checks": ["object.heated", "object.at_location"],
            "target_effects": [
                {
                    "predicate": "object.heated",
                    "args": {"object": "$inputs.object"},
                },
                {
                    "predicate": "object.at_location",
                    "args": {
                        "object": "$inputs.object",
                        "location": "$inputs.target_location",
                    },
                },
            ],
        },
    )


def test_seeded_heat_failure_recovered_by_dynamic_passes_composite():
    acquire_succeeded = _node("alfworld.acquire_object", True)
    seeded_heat_failed = _node(
        "alfworld.heat_object", False, "seeded action_cycle")
    dynamic_heat_succeeded = _node("alfworld.heat_object", True)
    place_succeeded = _node("alfworld.place_object", True)
    attempt_history = [
        acquire_succeeded,
        seeded_heat_failed,
        dynamic_heat_succeeded,
        place_succeeded,
    ]

    result = CompositeValidator().validate_composite(
        _heat_place_composite(),
        attempt_history,
        {
            "facts": [
                "object_heated(potato_1)",
                "object_at(potato_1, garbagecan_1)",
            ],
            "inventory": [],
        },
        inputs={"object": "potato_1", "target_location": "garbagecan_1"},
    )

    assert result.passed is True
    assert result.checks["all_atomic_nodes_passed"] is True
    assert result.checks["control_flow_covered"] is True
    assert result.checks["bound_target_effects"] is True
    assert not any("子节点验证失败" in message for message in result.messages)
    # Composite validation reads a final occurrence view; failure attribution
    # still receives the complete, unchanged attempt history.
    assert attempt_history[1] is seeded_heat_failed
    assert seeded_heat_failed.passed is False
    assert len(attempt_history) == 4


def test_recovery_collapse_does_not_merge_two_successful_same_skill_occurrences():
    composite = CompositeSkill(
        ref=SkillRef("composite.repeat_heat", "1.0.0"),
        graph={"nodes": [
            "alfworld.heat_object@1.0.0",
            "alfworld.heat_object@1.0.0",
        ]},
        validator={},
    )
    attempts = [
        _node("alfworld.heat_object", False, "seeded failed"),
        _node("alfworld.heat_object", True),
        _node("alfworld.heat_object", True),
    ]

    result = CompositeValidator().validate_composite(
        composite, attempts, {"facts": ["object_heated(potato_1)"]})

    assert result.passed is True
    assert result.checks["all_atomic_nodes_passed"] is True
    assert result.checks["control_flow_covered"] is True


def test_unrecovered_final_attempt_still_fails_composite():
    attempts = [
        _node("alfworld.acquire_object", True),
        _node("alfworld.heat_object", False, "dynamic effect missing"),
    ]

    result = CompositeValidator().validate_composite(
        _heat_place_composite(),
        attempts,
        {
            "facts": [
                "object_heated(potato_1)",
                "object_at(potato_1, garbagecan_1)",
            ],
        },
        inputs={"object": "potato_1", "target_location": "garbagecan_1"},
    )

    assert result.passed is False
    assert result.checks["all_atomic_nodes_passed"] is False
    assert result.checks["control_flow_covered"] is False
