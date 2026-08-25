from atomic_skillgraph.core.edge_ir import GraphEdge, edge_category
from atomic_skillgraph.core.status import EdgeType
from atomic_skillgraph.runtime.graph_executor import (
    GraphExecutionState,
    RuntimeGraphExecutor,
    evaluate_condition,
)
from atomic_skillgraph.core.trace_ir import TraceRecord
from atomic_skillgraph.evolution.proposal_replayer import ProposalReplayer
from atomic_skillgraph.persistence import ProposalStore


def test_six_edge_families_have_typed_valid_ir():
    edges = [
        GraphEdge("skill://a@1.0.0", "skill://b@1.0.0", EdgeType.CONTAINS),
        GraphEdge("a", "b", EdgeType.BRANCH, scope="runtime",
                  source_step="a", target_step="b",
                  condition={"field": "ok", "equals": True}),
        GraphEdge("a", "b", EdgeType.DATA_FLOW, scope="runtime",
                  source_step="a", target_step="b",
                  mapping={"source_output": "x", "target_input": "y"}),
        GraphEdge("skill://a@1.0.0", "environment://env",
                  EdgeType.REQUIRES_ENVIRONMENT,
                  metadata={"requirement": "env"}),
        GraphEdge("skill://a@1.0.0", "skill://b@1.0.0", EdgeType.SIMILAR,
                  evidence=["trace_1"], metadata={"confidence": 0.8}),
        GraphEdge("skill://b@2.0.0", "skill://b@1.0.0", EdgeType.SUPERSEDES,
                  evidence=["trace_2"], metadata={"reason": "contract_change"}),
    ]
    assert [edge_category(edge.type) for edge in edges] == [
        "structural", "control", "data", "dependency", "semantic", "evolution"
    ]
    assert all(not edge.validate() for edge in edges)
    assert len({edge.edge_id for edge in edges}) == len(edges)


def test_runtime_graph_executes_branch_retry_and_data_mapping():
    edges = [
        GraphEdge("a", "b", EdgeType.BRANCH, scope="runtime",
                  source_step="a", target_step="b",
                  condition={"field": "route", "equals": "b"}),
        GraphEdge("a", "c", EdgeType.BRANCH, scope="runtime",
                  source_step="a", target_step="c",
                  condition={"field": "route", "equals": "c"}),
        GraphEdge("b", "c", EdgeType.DATA_FLOW, scope="runtime",
                  source_step="b", target_step="c",
                  mapping={"source_output": "value", "target_input": "input"}),
        GraphEdge("b", "b", EdgeType.RETRY, scope="runtime",
                  source_step="b", target_step="b",
                  policy={"max_attempts": 2}),
    ]
    executor = RuntimeGraphExecutor(["a", "b", "c"], edges)
    state = GraphExecutionState(values={"b": {"value": 7}})
    assert executor.initial_steps() == ["a"]
    assert executor.next_steps("a", True, state, {"route": "b"}) == ["b"]
    assert executor.next_steps("b", False, state, {"failure_type": "x"}) == ["b"]
    assert executor.next_steps("b", False, state, {"failure_type": "x"}) == []
    executor.apply_data_flow(edges[2], state)
    assert state.values["c"]["input"] == 7
    assert evaluate_condition({"all": [
        {"field": "x", "equals": 1}, {"field": "y", "truthy": True}
    ]}, {"x": 1, "y": "yes"})


def test_failure_proposal_requires_matching_success_evidence(tmp_path):
    store = ProposalStore(tmp_path)
    proposal = store.add(
        "contract_revision", "trace_failed", "skill://env.acquire@1.0.0",
        "effect missing", payload={"task_type": "pick", "tool_refs": []},
    )
    replayer = ProposalReplayer(tmp_path)
    unrelated = TraceRecord(trace_id="trace_other", task_type="other", success=True)
    assert replayer.consume_success(unrelated, ["skill://env.place@1.0.0"]) == []
    assert store.pending()[0]["proposal_id"] == proposal["proposal_id"]

    success = TraceRecord(trace_id="trace_success", task_type="pick", success=True)
    events = replayer.consume_success(success, ["skill://env.acquire@1.1.0"])
    assert events[0]["successful_trace_id"] == "trace_success"
    assert store.pending() == []
    saved = store.list_all()[0]
    assert saved["status"] == "replayed"
    assert saved["replay_result"]["decision"] == "evidence_replayed"
