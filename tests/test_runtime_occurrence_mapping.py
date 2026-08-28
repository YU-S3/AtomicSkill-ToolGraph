from atomic_skillgraph.core.edge_ir import GraphEdge
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.status import EdgeType
from atomic_skillgraph.runtime.atomic_planner import AtomicPlanner
from atomic_skillgraph.runtime.runtime_graph import PlannedNode


def test_repeated_logical_refs_map_edges_by_occurrence_not_ref():
    acquire = SkillRef("generic.acquire", "1.0.0")
    place = SkillRef("generic.place", "1.0.0")
    nodes = [
        PlannedNode(acquire, step_id="runtime_0", origin_step_id="acquire_1"),
        PlannedNode(place, step_id="runtime_1", origin_step_id="place_1"),
        PlannedNode(acquire, step_id="runtime_2", origin_step_id="acquire_2"),
        PlannedNode(place, step_id="runtime_3", origin_step_id="place_2"),
    ]
    source_edges = [
        GraphEdge(source=str(acquire), target=str(place), type=EdgeType.DATA_FLOW,
                  scope="composite", source_step="acquire_1", target_step="place_1",
                  mapping={"source_output": "held", "target_input": "object"}),
        GraphEdge(source=str(acquire), target=str(place), type=EdgeType.DATA_FLOW,
                  scope="composite", source_step="acquire_2", target_step="place_2",
                  mapping={"source_output": "held", "target_input": "object"}),
    ]
    runtime = AtomicPlanner._runtime_edges(nodes, source_edges)
    data = [edge for edge in runtime if edge.type == EdgeType.DATA_FLOW]
    assert [(edge.source_step, edge.target_step) for edge in data] == [
        ("runtime_0", "runtime_1"), ("runtime_2", "runtime_3")]
