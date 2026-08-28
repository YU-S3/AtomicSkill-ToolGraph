from atomic_skillgraph.core.trace_ir import NodeValidationResult, TraceRecord
from atomic_skillgraph.validation.failure_localizer import FailureLocalizer


def test_rescued_earlier_occurrence_is_not_localized_as_failure():
    ref = "skill://generic.acquire@1.0.0"
    trace = TraceRecord(success=False, realized_atomic_nodes=[
        {"ref": ref, "step_id": "step_000", "occurrence_id": "acquire_1",
         "passed": True, "attempt_started": True,
         "attempts": [{"mode": "seeded", "started": True, "passed": False},
                      {"mode": "dynamic", "started": True, "passed": True}]},
        {"ref": "skill://generic.place@1.0.0", "step_id": "step_001",
         "occurrence_id": "place_1", "passed": True, "attempt_started": True},
        {"ref": ref, "step_id": "step_002", "occurrence_id": "acquire_2",
         "passed": False, "attempt_started": True,
         "attempts": [{"mode": "seeded", "started": True, "passed": False},
                      {"mode": "dynamic", "started": True, "passed": False}]},
    ], node_validators=[
        NodeValidationResult(node_ref=ref, step_id="step_000",
                             occurrence_id="acquire_1", mode="seeded",
                             passed=False, checks={"effects": False}),
        NodeValidationResult(node_ref=ref, step_id="step_000",
                             occurrence_id="acquire_1", mode="dynamic",
                             passed=True, checks={"effects": True}),
        NodeValidationResult(node_ref=ref, step_id="step_002",
                             occurrence_id="acquire_2", mode="dynamic",
                             passed=False, checks={"effects": False}),
    ])
    attribution = FailureLocalizer().localize(trace)[0]
    assert attribution.step_id == "step_002"
    assert attribution.occurrence_id == "acquire_2"


def test_planning_failure_is_not_blame_on_atomic_or_tool():
    trace = TraceRecord(success=False, failure_stage="planning",
                        failure_cause="plan_binding_unresolved")
    attribution = FailureLocalizer().localize(trace)[0]
    assert attribution.responsibility == "planning"
    assert attribution.node_ref == ""
