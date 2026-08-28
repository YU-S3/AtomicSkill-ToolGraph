from atomic_skillgraph.atomicizer.semantic_extractor import build_structured_events
from atomic_skillgraph.core.status import ExecutionMode
from atomic_skillgraph.core.trace_ir import ActionRecord, TraceRecord


def test_framework_action_is_state_evidence_but_not_capability_effect():
    trace = TraceRecord(
        actions=[ActionRecord(step=0, name="open cabinet 1",
                              mode=ExecutionMode.DYNAMIC,
                              origin="framework_discovery")],
        state_snapshots=[
            {"step": 0, "state": {"facts": []}},
            {"step": 1, "state": {"facts": ["container_open(cabinet_1)"]}},
        ])
    event = build_structured_events(trace)[0]
    assert event["origin"] == "framework_discovery"
    assert "container_open(cabinet_1)" in event["after"]["facts"]
    assert event["positive_effects"] == []
