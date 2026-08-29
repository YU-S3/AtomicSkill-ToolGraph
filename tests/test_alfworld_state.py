"""ALFWorld PDDL 状态解析与原子化管线测试（真实观察语料，无 API）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atomic_skillgraph.adapters.alfworld import (  # noqa: E402
    AlfWorldAdapter,
    _AlfStateTracker,
    _compact_seed_context,
    _controlled_acquire_replay_setup,
    _executable_goal_params,
    _goal_roles_from_text,
    _parse_action_params,
    _parse_alfworld_state,
    _record_location_inspection,
    _semantic_goal_params,
    _target_effects_of,
)
from atomic_skillgraph.atomicizer.trace_atomicizer import TraceAtomicizer  # noqa: E402
from atomic_skillgraph.atomicizer.semantic_extractor import build_structured_events  # noqa: E402
from atomic_skillgraph.adapters.benchmark import Task  # noqa: E402
from atomic_skillgraph.core.trace_ir import ActionRecord, TraceRecord  # noqa: E402
from atomic_skillgraph.core.refs import ToolRef  # noqa: E402
from atomic_skillgraph.core.status import ArtifactKind, ToolLifecycle  # noqa: E402
from atomic_skillgraph.core.tool_ir import ToolAsset  # noqa: E402
from atomic_skillgraph.core.predicates import StateSnapshot, check_effects  # noqa: E402
from atomic_skillgraph.graph.registry import SkillGraphRegistry  # noqa: E402
from atomic_skillgraph.tools.compiler_adapter import (  # noqa: E402
    _infer_execution_context_preconditions,
    mine_action_template_tools,
)
from atomic_skillgraph.system import _completed_distinct_effect_instances  # noqa: E402

# 真实 ALFWorld PDDL 环境（0.4.2）观察语料（来自一次成功 heat 任务）
ACTIONS = [
    ("inventory", "You are not carrying anything."),
    ("look", "You are in the middle of a room. Looking quickly around you, you see nothing."),
    ("go to cabinet 1", "You arrive at cabinet 1. On the cabinet 1, you see a peppershaker 2."),
    ("go to cabinet 2", "You arrive at cabinet 2. The cabinet 2 is closed."),
    ("open cabinet 2", "You open the cabinet 2. The cabinet 2 is open. In it, you see nothing."),
    ("go to countertop 1", "You arrive at countertop 1. On the countertop 1, you see a bowl 1, "
                           "a glassbottle 1, a mug 2, and a papertowelroll 1."),
    ("take mug 2 from countertop 1", "You pick up the mug 2 from the countertop 1."),
    ("go to microwave 1", "You arrive at microwave 1. The microwave 1 is closed."),
    ("open microwave 1", "You open the microwave 1. The microwave 1 is open. "
                         "In it, you see a egg 1, and a potato 1."),
    ("heat mug 2 with microwave 1", "You heat the mug 2 using the microwave 1."),
    ("go to coffeemachine 1", "You arrive at coffeemachine 1. On the coffeemachine 1, you see nothing."),
    ("move mug 2 to coffeemachine 1", "You move the mug 2 to the coffeemachine 1."),
]


def _build_trace():
    trace = TraceRecord(
        task_id="alfworld_test", task_type="pick_heat_then_place_in_recep",
        task_goal="heat some mug and put it in coffeemachine.",
        benchmark="alfworld", start_mode="cold", success=True,
        benchmark_result={"passed": True},
        provenance={"env_index": 0, "params": {"object": "mug 2",
                                                "object_location": "countertop 1",
                                                "heating_station": "microwave 1",
                                                "target_location": "coffeemachine 1"}},
    )
    tracker = _AlfStateTracker()
    trace.state_snapshots.append({"step": 0, "state": tracker.state()})
    for i, (name, obs) in enumerate(ACTIONS):
        tracker.update(obs)
        trace.actions.append(ActionRecord(step=i, name=name,
                                          params=_parse_action_params(name),
                                          observation=obs,
                                          accepted=True))
        trace.state_snapshots.append({"step": i + 1, "state": tracker.state()})
    return trace


def test_pddl_state_parser():
    single = _parse_alfworld_state(
        "You arrive at countertop 1. On the countertop 1, you see a mug 2, and a bowl 1.")
    facts = single["facts"]
    assert "agent_at(countertop_1)" in facts
    assert "object_at(mug_2, countertop_1)" in facts
    assert "object_at(bowl_1, countertop_1)" in facts
    heat = _parse_alfworld_state("You heat the mug 2 using the microwave 1.")
    assert "object_heated(mug_2)" in heat["facts"]
    move = _parse_alfworld_state("You move the mug 2 to the coffeemachine 1.")
    assert "object_at(mug_2, coffeemachine_1)" in move["facts"]
    light = _parse_alfworld_state("You turn on the desklamp 1.")
    assert "object_toggled(desklamp_1)" in light["facts"]


def test_official_task_type_is_ignored_by_goal_contract():
    assert _target_effects_of("pick_heat_then_place_in_recep") == []
    simple = _target_effects_of(
        "pick_and_place_simple", "put a mug in cabinet.",
        {"object": "mug", "target_location": "cabinet"})
    assert [item["predicate"] for item in simple] == ["object.at_location"]
    look = _target_effects_of(
        "look_at_obj_in_light", "examine the book with the lamp.",
        {"object": "book", "associated_entity": "lamp"})
    assert [item["predicate"] for item in look] == ["object.observed_with"]
    pick_two = _target_effects_of(
        "pick_two_obj_and_place", "put two cd in safe.",
        {"object_type": "cd", "target_location": "safe"})
    assert pick_two[0]["cardinality"] == 2
    assert pick_two[0]["distinct_by"] == "object"


def test_goal_contract_is_compositional_not_task_type_enumerated():
    effects = _target_effects_of(
        "unseen_household_task",
        "heat and place three apple in a cabinet.",
        {"object": "apple", "object_type": "apple",
         "target_location": "cabinet 1"},
    )
    assert [item["predicate"] for item in effects] == [
        "object.heated", "object.at_location"]
    assert effects[-1]["cardinality"] == 3
    assert effects[-1]["distinct_by"] == "object"

    light = _target_effects_of(
        "unseen_inspection_task", "examine the book with the lamp",
        {"object": "book", "associated_entity": "lamp 1"})
    assert [item["predicate"] for item in light] == ["object.observed_with"]


def test_goal_roles_do_not_inject_unstated_station_or_workflow():
    roles = _goal_roles_from_text(
        "heat some apple and put it in fridge.")
    params = _executable_goal_params(
        roles, ["fridge_1", "microwave_1", "sinkbasin_1"])
    assert roles == {"theme": "apple", "destination": "fridge"}
    assert params == {"object": "apple", "target_location": "fridge_1"}
    assert not any("station" in key for key in params)

    inspection = _goal_roles_from_text(
        "look at pencil under the desklamp.")
    assert inspection == {
        "theme": "pencil", "associated_entity": "desklamp"}


def test_pick_two_cardinality_requires_distinct_grounded_objects():
    effect = _target_effects_of(
        "pick_two_obj_and_place", "put two cd in safe.",
        {"object_type": "cd", "target_location": "safe 1"})
    inputs = {"object_type": "cd", "target_location": "safe 1"}
    one = StateSnapshot({"facts": ["object_at(cd_1, safe_1)"]})
    two = StateSnapshot({"facts": [
        "object_at(cd_1, safe_1)", "object_at(cd_2, safe_1)"]})
    assert check_effects(one, inputs, effect)[0] is False
    assert check_effects(two, inputs, effect)[0] is True


def test_cardinality_counts_distinct_by_role_not_complete_fact_tuples():
    effect = [{
        "predicate": "object.observed_with",
        # Reverse insertion order to ensure the predicate schema, rather than
        # JSON object ordering, locates the distinct object argument.
        "args": {
            "associated_entity": "$associated_entity",
            "object": "$object",
        },
        "cardinality": 2,
        "distinct_by": "object",
    }]
    inputs = {"object": "mug", "associated_entity": "desklamp"}
    repeated_object = StateSnapshot({"facts": [
        "object_observed_with(mug_1, desklamp_1)",
        "object_observed_with(mug_1, desklamp_2)",
    ]})
    distinct_objects = StateSnapshot({"facts": [
        "object_observed_with(mug_1, desklamp_1)",
        "object_observed_with(mug_2, desklamp_2)",
    ]})

    assert check_effects(repeated_object, inputs, effect)[0] is False
    assert check_effects(distinct_objects, inputs, effect)[0] is True


def test_pick_two_excludes_instances_already_placed_at_target():
    task = Task(
        task_id="two", benchmark="generic_env", task_type="arbitrary_batch_delivery",
        goal="put two cd in safe.",
        context={"params": {"object": "cd", "object_type": "cd",
                            "target_location": "safe 1"}},
        target_effects=[{
            "predicate": "object.at_location",
            "args": {"object": "$object_type", "location": "$target_location"},
            "cardinality": 2, "distinct_by": "object"}],
    )
    excluded = _completed_distinct_effect_instances(
        task,
        {"facts": ["object_at(cd_1, safe_1)",
                   "object_at(cd_2, desk_1)", "object_at(book_1, safe_1)"]},
        {"object": "cd"},
    )
    assert excluded == {"cd_1"}


def test_goal_semantic_location_is_separate_from_executable_instance():
    executable = {"object": "mug", "target_location": "cabinet 1",
                  "heating_station": "microwave 1"}
    semantic = _semantic_goal_params(
        "heat some mug and put it in cabinet.", executable)
    assert executable["target_location"] == "cabinet 1"
    assert semantic["target_location"] == "cabinet"
    assert _semantic_goal_params(
        "heat some mug and put it in cabinet 1.", executable
    )["target_location"] == "cabinet 1"


def test_tracker_accumulates():
    tracker = _AlfStateTracker()
    tracker.update("You arrive at countertop 1. On the countertop 1, you see a mug 2.")
    tracker.update("You pick up the mug 2 from the countertop 1.")
    state = tracker.state()
    assert "agent_holds(mug_2)" in state["facts"], state["facts"]
    assert "mug_2" in state["inventory"]
    assert "object_at(mug_2, countertop_1)" not in state["facts"]
    tracker.update("You heat the mug 2 using the microwave 1.")
    assert "object_heated(mug_2)" in tracker.state()["facts"]
    tracker.update("You move the mug 2 to the coffeemachine 1.")
    facts = tracker.state()["facts"]
    assert "object_at(mug_2, coffeemachine_1)" in facts
    assert not any(f.startswith("object_at(mug_2,") and "coffeemachine_1" not in f
                   for f in facts), "旧位置应被移除"
    assert not any(f.startswith("agent_holds(mug_2)") for f in facts), "持有应被移除"
    tracker.update("You turn on the desklamp 1.")
    assert "object_toggled(desklamp_1)" in tracker.state()["facts"]
    tracker.update("You turn off the desklamp 1.")
    assert "object_toggled(desklamp_1)" not in tracker.state()["facts"]


def test_structured_events_expose_take_heat_place_effects():
    events = build_structured_events(_build_trace())
    by_action = {event["action"]: event for event in events}
    take = by_action["take mug 2 from countertop 1"]
    heat = by_action["heat mug 2 with microwave 1"]
    place = by_action["move mug 2 to coffeemachine 1"]
    visit = by_action["go to countertop 1"]

    assert {item["predicate"] for item in visit["positive_effects"]} == {"agent_at"}
    assert {item["predicate"] for item in visit["observed_effects"]} == {
        "object.exists", "object.at_location"}
    assert {item["predicate"] for item in take["positive_effects"]} == {"agent.holds"}
    assert any(item.get("not", {}).get("predicate") == "object.at_location"
               for item in take["negative_effects"])
    assert {item["predicate"] for item in heat["positive_effects"]} == {"object.heated"}
    assert {item["predicate"] for item in place["positive_effects"]} == {
        "object.at_location"}
    assert any(item.get("not", {}).get("predicate") == "agent.holds"
               for item in place["negative_effects"])


def test_structured_event_exposes_light_toggle_effect_and_parameter():
    trace = TraceRecord(
        task_id="look", task_type="look_at_obj_in_light",
        task_goal="examine the cd with the desklamp.", benchmark="alfworld",
        success=True, provenance={"params": {
            "object": "cd", "associated_entity": "desklamp 1"}},
    )
    tracker = _AlfStateTracker()
    tracker.update("You pick up the cd 1 from the desk 1.")
    trace.state_snapshots.append({"step": 0, "state": tracker.state()})
    observation = "You turn on the desklamp 1."
    tracker.update(observation, action="use desklamp 1", accepted=True)
    trace.actions.append(ActionRecord(
        step=0, name="use desklamp 1",
        params=_parse_action_params("use desklamp 1"),
        observation=observation, accepted=True))
    trace.state_snapshots.append({"step": 1, "state": tracker.state()})
    event = build_structured_events(trace)[0]
    assert event["params"] == {"associated_entity": "desklamp 1"}
    assert {item["predicate"] for item in event["positive_effects"]} == {
        "object.toggled", "object.observed_with"}
    assert {"predicate": "object.observed_with", "args": {
        "object": "cd_1", "associated_entity": "desklamp_1"}} \
        in event["positive_effects"]


def test_atomicizer_on_pddl_trace(workspace_tmp):
    trace = _build_trace()
    registry = SkillGraphRegistry(workspace_tmp / "graph")
    atomicizer = TraceAtomicizer(registry)
    result = atomicizer.atomicize_success(trace)
    names = {c.skill.ref.logical_id for c in result.candidates}
    print("原子候选:", names)
    assert names, "应从成功 PDDL 轨迹原子化出技能"
    assert any("acquire" in n or "holds" in n for n in names), names
    assert any("heat" in n for n in names), names
    assert any("place" in n or "object_at" in n for n in names), names
    # Tool 挖掘：模板应含参数槽位
    tools = mine_action_template_tools(trace, result.segments)
    assert tools, "应挖掘出 action template 工具"
    bodies = "\n".join(t.artifact_body() for t in tools)
    assert "{object}" in bodies and ("{object_location}" in bodies
                                     or "{heating_station}" in bodies
                                     or "{target_location}" in bodies), bodies
    acquire = next(t for t in tools if "agent_holds" in t.tool_id)
    assert acquire.artifact.get("steps") == [
        "go to {object_location}",
        "take {object} from {object_location}",
    ], "源任务的无槽位探索动作不应写入 Direct Tool 本体"


def test_minimal_action_template_declares_consumed_location_context():
    preconditions = _infer_execution_context_preconditions(
        ["take {object} from {object_location}"],
        {"object": "mug_1", "object_location": "countertop_1"},
        {"facts": ["agent_at(countertop_1)",
                   "object_at(mug_1, countertop_1)"]},
    )
    assert preconditions == [{
        "predicate": "agent_at",
        "args": {"location": "$inputs.object_location"},
    }]
    assert _infer_execution_context_preconditions(
        ["go to {object_location}",
         "take {object} from {object_location}"],
        {"object": "mug_1", "object_location": "countertop_1"},
        {"facts": ["agent_at(countertop_1)"]},
    ) == []


def _take_only_tool(*, declare_context: bool) -> ToolAsset:
    interface = {
        "inputs": {"object": "object_ref", "object_location": "location_ref"},
        "outputs": {"effect": []},
    }
    if declare_context:
        interface["preconditions"] = [{
            "predicate": "agent_at",
            "args": {"location": "$inputs.object_location"},
        }]
    return ToolAsset(
        ref=ToolRef("alfworld.test_take_only", "0.1.0"),
        artifact_kind=ArtifactKind.ACTION_TEMPLATE,
        signature={"parameters": [
            {"name": "object"}, {"name": "object_location"}]},
        interface=interface,
        artifact={"steps": ["take {object} from {object_location}"]},
        tests=[{"kind": "replay", "bindings": {
            "object": "mug_1", "object_location": "countertop_1"}}],
        status=ToolLifecycle.DRAFT,
    )


def test_action_template_admission_rejects_undeclared_location_context():
    adapter = AlfWorldAdapter()
    result = adapter.replay_tool(
        _take_only_tool(declare_context=False),
        {"object": "mug_1", "object_location": "countertop_1"},
        {"facts": ["agent_at(countertop_1)"]},
    )
    assert not result["passed"]
    assert result["reason"].startswith("undeclared_execution_context:")


def test_direct_tool_context_is_checked_against_live_agent_location():
    adapter = AlfWorldAdapter()
    tool = _take_only_tool(declare_context=True)
    params = {"object": "mug_1", "object_location": "countertop_1"}
    rejected = adapter.validate_tool_context(
        tool, params, {"facts": ["agent_at(desk_1)"]})
    accepted = adapter.validate_tool_context(
        tool, params, {"facts": ["agent_at(countertop_1)"]})
    assert not rejected["passed"]
    assert "declared_context_missing" in rejected["reason"]
    assert accepted["passed"]


def test_rejected_direct_template_action_stops_immediately():
    class _RejectingEnv:
        def __init__(self):
            self.calls = []

        def step(self, action):
            self.calls.append(action)
            return type("_StepResult", (), {
                "observation": "Nothing happens.", "score": 0.0,
                "done": False, "won": False, "accepted": False,
                "admissible_commands": ["go to countertop 1"],
            })()

    adapter = AlfWorldAdapter()
    env = _RejectingEnv()
    adapter._current_env = env
    task = Task(task_id="direct_reject", benchmark="alfworld",
                task_type="unseen", goal="obtain a mug", context={})
    result = adapter.run_env_episode(
        task, _StubLLM(),
        direct_steps=[{
            "step_id": "step_000", "node_ref": "skill://acquire@1.0.0",
            "steps": [{"template": "take {object} from {object_location}",
                       "params": {"object": "mug_1",
                                  "object_location": "countertop_1"}}],
        }],
        resume={
            "observation": "at desk", "admissible": ["go to countertop 1"],
            "actions": [], "states": [],
            "state": {"facts": ["agent_at(desk_1)"],
                      "inventory": [], "meta": {}},
        },
        stop_effects=[{"predicate": "agent.holds",
                       "args": {"object": "$inputs.object"}}],
        effect_inputs={"object": "mug_1"}, max_steps=15,
    )
    assert env.calls == ["take mug 1 from countertop 1"]
    assert result.failure_type == "direct_template_action_rejected"
    assert result.steps == 1


# ---------------------------------------------------------------------------
# 原地降级（in-place fallback）+ 轻量注入
# ---------------------------------------------------------------------------

class _FakeEnv:
    """最小 ALFWorld 环境替身：run_env_episode resume 路径不创建真实环境。"""

    def __init__(self, actions: list[str]):
        self.actions = list(actions)
        self.calls: list[str] = []

    def step(self, action: str):
        self.calls.append(action)
        obs = f"You arrive at {action.split()[-1]}." if self.calls else "ok"
        done = not self.actions
        won = done and self.calls[-1] == "win"
        return type("_StepResult", (), {
            "observation": obs, "score": 0.0, "done": done, "won": won,
            "admissible_commands": self.actions,
        })()


class _StubLLM:
    """总是回答首条可行动作的 LLM 替身。"""

    def __init__(self, scripted: dict[str, str] | None = None):
        self.scripted = scripted or {}

    def generate(self, *, instructions, input_text, temperature=None, thinking=None):
        return type("_Resp", (), {"text": "Think: continue.\nAct: %s" % self.scripted.get("act", ""),
                                  "prompt_tokens": 0, "completion_tokens": 0,
                                  "total_tokens": 0, "latency_ms": 0.0,
                                  "provider": "stub", "model": "stub"})()


def test_compact_seed_context():
    context = (
        "[Atomic Skill] Agent 持有目标物体\n"
        "  - 先找物体再拿起\n"
        "```text\ntake {object} from {object_location}\n```\n"
        "[Tool] 使状态满足 agent.holds\n"
    )
    compact = _compact_seed_context(context)
    assert "[Atomic Skill]" in compact
    assert "take {object}" not in compact, "围栏块内的 Tool 体应被移除"
    assert len(compact) < len(context)


def test_tracker_from_state():
    tracker = _AlfStateTracker(from_state={"facts": ["object_at(mug_1, countertop_1)"],
                                           "inventory": [],
                                           "meta": {"checked_locations": ["cabinet_1"]}})
    assert "object_at(mug_1, countertop_1)" in tracker.state()["facts"]
    assert tracker.state()["meta"]["checked_locations"] == ["cabinet_1"]


def test_rejected_dynamic_visit_does_not_pollute_checked_locations():
    tracker = _AlfStateTracker(from_state={
        "facts": ["agent_at(table_1)"], "inventory": [], "meta": {}})
    _record_location_inspection(
        tracker, "go to countertop 1", "Nothing happens.", accepted=False)
    assert tracker.state()["meta"].get("checked_locations") in (None, [])

    tracker.update("You arrive at countertop 1. You see nothing.")
    _record_location_inspection(
        tracker, "go to countertop 1",
        "You arrive at countertop 1. You see nothing.", accepted=True)
    assert tracker.state()["meta"]["checked_locations"] == ["countertop_1"]


def test_acquire_location_discovery_is_bounded_and_deduplicated():
    class _SearchEnv:
        def __init__(self):
            self.calls = []

        def step(self, action: str):
            self.calls.append(action)
            observations = {
                "go to cabinet 1": "You arrive at cabinet 1. The cabinet 1 is closed.",
                "open cabinet 1": "You open the cabinet 1. The cabinet 1 is open. In it, you see nothing.",
                "go to countertop 1": ("You arrive at countertop 1. On the countertop 1, "
                                         "you see a mug 2."),
            }
            admissible = ["go to cabinet 1", "go to countertop 1"]
            if action == "go to cabinet 1":
                admissible.append("open cabinet 1")
            if action == "go to countertop 1":
                admissible.append("take mug 2 from countertop 1")
            return type("_StepResult", (), {
                "observation": observations[action], "score": 0.0,
                "done": False, "won": False,
                "admissible_commands": admissible,
            })()

    adapter = AlfWorldAdapter()
    env = _SearchEnv()
    adapter._current_env = env
    task = Task(task_id="search", benchmark="alfworld",
                task_type="pick_heat_then_place_in_recep", goal="heat a mug",
                context={"env_index": 0})
    resume = {
        "observation": "You are in the middle of a room.",
        "admissible": ["go to cabinet 1", "go to countertop 1"],
        "actions": [], "states": [],
        "state": {"facts": [], "inventory": [], "meta": {}},
    }
    binding, result = adapter.discover_object_location(
        task, "mug", resume=resume, max_locations=5,
        node_ref="skill://acquire@1", tool_ref="tool://acquire@1")
    assert binding == {"object": "mug_2", "object_location": "countertop_1"}
    assert env.calls == ["go to cabinet 1", "open cabinet 1",
                         "go to countertop 1"]
    assert len(env.calls) == len(set(env.calls)), "位置搜索不应重复动作/循环"
    checked = result.states[-1]["state"]["meta"]["checked_locations"]
    assert checked == ["cabinet_1", "countertop_1"]
    assert not result.direct_used
    assert all(action["origin"] == "framework_discovery"
               and not action["tool_ref"] for action in result.actions)


def test_discovery_retries_one_rejected_visit_before_marking_checked():
    class _TransientVisitEnv:
        def __init__(self):
            self.calls = []

        def step(self, action: str):
            self.calls.append(action)
            if len(self.calls) == 1:
                observation = "Nothing happens."
                accepted = False
                commands = ["go to countertop 1"]
            else:
                observation = ("You arrive at countertop 1. On the countertop 1, "
                               "you see a mug 2.")
                accepted = True
                commands = ["take mug 2 from countertop 1"]
            return type("_StepResult", (), {
                "observation": observation, "score": 0.0,
                "done": False, "won": False, "accepted": accepted,
                "admissible_commands": commands,
            })()

    adapter = AlfWorldAdapter()
    env = _TransientVisitEnv()
    adapter._current_env = env
    task = Task(task_id="transient", benchmark="alfworld",
                task_type="unseen", goal="find an object", context={})
    resume = {
        "observation": "room", "admissible": ["go to countertop 1"],
        "actions": [], "states": [],
        "state": {"facts": [], "inventory": [], "meta": {}},
    }

    binding, result = adapter.discover_object_location(
        task, "mug", resume=resume, max_locations=1)

    assert binding == {"object": "mug_2", "object_location": "countertop_1"}
    assert env.calls == ["go to countertop 1", "go to countertop 1"]
    assert [item["accepted"] for item in result.actions] == [False, True]
    assert result.states[-1]["state"]["meta"]["checked_locations"] == [
        "countertop_1"]


def test_discovery_never_marks_twice_rejected_location_as_checked():
    class _RejectedThenValidEnv:
        def __init__(self):
            self.calls = []

        def step(self, action: str):
            self.calls.append(action)
            if action == "go to cabinet 1":
                observation = "Nothing happens."
                accepted = False
                commands = ["go to cabinet 1", "go to countertop 1"]
            else:
                observation = ("You arrive at countertop 1. On the countertop 1, "
                               "you see a mug 2.")
                accepted = True
                commands = ["take mug 2 from countertop 1"]
            return type("_StepResult", (), {
                "observation": observation, "score": 0.0,
                "done": False, "won": False, "accepted": accepted,
                "admissible_commands": commands,
            })()

    adapter = AlfWorldAdapter()
    env = _RejectedThenValidEnv()
    adapter._current_env = env
    task = Task(task_id="rejected", benchmark="alfworld",
                task_type="unseen", goal="find an object", context={})
    resume = {
        "observation": "room",
        "admissible": ["go to cabinet 1", "go to countertop 1"],
        "actions": [], "states": [],
        "state": {"facts": [], "inventory": [], "meta": {}},
    }

    binding, result = adapter.discover_object_location(
        task, "mug", resume=resume, max_locations=2)

    assert binding == {"object": "mug_2", "object_location": "countertop_1"}
    assert env.calls == ["go to cabinet 1", "go to cabinet 1",
                         "go to countertop 1"]
    checked = result.states[-1]["state"]["meta"]["checked_locations"]
    assert checked == ["countertop_1"]


def test_acquire_admission_setup_discovers_then_exposes_take():
    class _ReplaySearchEnv:
        def __init__(self):
            self.calls = []

        def step(self, action):
            self.calls.append(action)
            if action == "go to cabinet 1":
                obs = "You arrive at cabinet 1. The cabinet 1 is closed."
                commands = ["open cabinet 1", "go to countertop 1"]
            elif action == "open cabinet 1":
                obs = "You open the cabinet 1. In it, you see a mug 2."
                commands = ["take mug 2 from cabinet 1", "go to countertop 1"]
            else:
                raise AssertionError(action)
            return type("_StepResult", (), {
                "observation": obs, "done": False, "won": False,
                "admissible_commands": commands,
            })()

    env = _ReplaySearchEnv()
    tracker = _AlfStateTracker(initial_observation="room")
    bindings, commands, passed = _controlled_acquire_replay_setup(
        env, tracker, ["go to cabinet 1", "go to countertop 1"],
        {"object": "mug_1", "object_location": "cabinet_1"})
    assert passed
    assert env.calls == ["go to cabinet 1", "open cabinet 1"]
    assert bindings == {"object": "mug_2", "object_location": "cabinet_1"}
    assert "take mug 2 from cabinet 1" in commands


def test_runtime_discovery_materializes_remote_take_binding():
    class _RemoteBindingEnv:
        def __init__(self):
            self.calls = []

        def step(self, action):
            self.calls.append(action)
            assert action == "go to countertop 1"
            return type("_StepResult", (), {
                "observation": ("You arrive at countertop 1. On the countertop 1, "
                                "you see a mug 2."),
                "done": False, "won": False,
                "admissible_commands": ["take mug 2 from countertop 1"],
            })()

    adapter = AlfWorldAdapter()
    env = _RemoteBindingEnv()
    adapter._current_env = env
    task = Task(task_id="remote", benchmark="alfworld",
                task_type="pick_heat_then_place_in_recep", goal="heat a mug",
                context={"env_index": 0,
                         "initial_observation": "you see a countertop 1"})
    resume = {
        "observation": "You arrive at coffeemachine 1.",
        "admissible": ["take mug 2 from countertop 1"],
        "actions": [], "states": [],
        "state": {"facts": ["agent_at(coffeemachine_1)"],
                  "inventory": [], "meta": {}},
    }
    binding, result = adapter.discover_object_location(
        task, "mug", resume=resume, max_locations=5)
    assert binding == {"object": "mug_2", "object_location": "countertop_1"}
    assert env.calls == ["go to countertop 1"]
    assert "object_at(mug_2, countertop_1)" in result.states[-1]["state"]["facts"]


def test_discovery_keeps_current_observed_instance_over_remote_take_hint():
    class _NoStepEnv:
        def step(self, action):
            raise AssertionError(f"已在当前状态观察到对象，不应再导航：{action}")

    adapter = AlfWorldAdapter()
    adapter._current_env = _NoStepEnv()
    task = Task(task_id="multi_instance", benchmark="alfworld",
                task_type="pick_heat_then_place_in_recep", goal="heat a mug",
                context={"env_index": 0})
    resume = {
        "observation": ("You arrive at countertop 1. On the countertop 1, "
                        "you see a mug 2."),
        # 另一个同族实例的远端命令不能覆盖当前真实 observation。
        "admissible": ["take mug 1 from coffeemachine 1"],
        "actions": [], "states": [],
        "state": {"facts": ["agent_at(countertop_1)",
                              "object_at(mug_2, countertop_1)",
                              "object_exists(mug_2)"],
                  "inventory": [], "meta": {}},
    }
    binding, result = adapter.discover_object_location(
        task, "mug", resume=resume, max_locations=5)
    assert binding == {"object": "mug_2", "object_location": "countertop_1"}
    assert result.actions == []


def test_non_acquire_navigable_entity_binds_without_consuming_tool_step():
    class _NoStepEnv:
        def step(self, action):
            raise AssertionError(f"passive binding must leave navigation to Tool: {action}")

    adapter = AlfWorldAdapter()
    adapter._current_env = _NoStepEnv()
    task = Task(task_id="fixture", benchmark="generic_env",
                task_type="hidden_label", goal="inspect a fixture", context={})
    resume = {
        "observation": "room", "admissible": ["go to fixture 3"],
        "actions": [], "states": [],
        "state": {"facts": [], "inventory": [], "meta": {}},
    }
    binding, result = adapter.discover_object_location(
        task, "fixture", resume=resume, max_locations=2,
        allow_passive_navigable=True)
    assert binding == {"object": "fixture_3",
                       "object_location": "fixture_3"}
    assert result.actions == []
    assert result.states == []
    assert result.current_admissible == ["go to fixture 3"]


def test_discovery_preserves_its_budget_exhaustion_reason():
    class _NoStepEnv:
        def step(self, action):
            raise AssertionError(
                f"zero action deadline must stop before env.step: {action}")

    adapter = AlfWorldAdapter()
    adapter._current_env = _NoStepEnv()
    task = Task(task_id="discovery_budget", benchmark="alfworld",
                task_type="hidden_label", goal="find a newspaper", context={})
    resume = {
        "observation": "room", "admissible": ["go to shelf 1"],
        "actions": [], "states": [],
        "state": {"facts": [], "inventory": [], "meta": {}},
    }

    binding, result = adapter.discover_object_location(
        task, "newspaper", resume=resume, max_locations=2,
        action_deadline=0)

    assert binding == {}
    assert result.failure_type == "discovery_budget_exhausted"
    assert result.actions == []


def test_llm_errors_do_not_advance_environment_and_are_persisted():
    class _CountingEnv:
        def __init__(self):
            self.calls = []

        def step(self, action):
            self.calls.append(action)
            raise AssertionError("API 异常期间不应调用 env.step")

    class _FailingLLM:
        def generate(self, **kwargs):
            raise TimeoutError("provider timed out")

    adapter = AlfWorldAdapter()
    adapter.llm_max_consecutive_errors = 1
    env = _CountingEnv()
    adapter._current_env = env
    task = Task(task_id="llm_error", benchmark="alfworld",
                task_type="pick_heat_then_place_in_recep", goal="heat a mug",
                context={"env_index": 0})
    resume = {
        "observation": "room", "admissible": ["look"], "actions": [],
        "states": [{"step": 0, "state": {"facts": [], "meta": {}}}],
        "state": {"facts": [], "meta": {}},
    }
    result = adapter.run_env_episode(task, _FailingLLM(), resume=resume,
                                     max_steps=1)
    assert result.failure_type == "llm_error"
    assert env.calls == []
    assert len(result.infrastructure_errors) == 1
    assert result.infrastructure_errors[-1]["exception_type"] == "TimeoutError"


def test_env_episode_resume_in_place(workspace_tmp):
    """direct 失败后 seeded 原地续跑：动作/状态合并、预算扣除、轻量注入。"""
    adapter = AlfWorldAdapter()
    adapter._current_env = _FakeEnv(["go to cabinet 1", "take mug 1 from cabinet 1"])
    task = Task(task_id="alfworld_resume_test", benchmark="alfworld",
                task_type="pick_heat_then_place_in_recep",
                goal="heat some mug and put it in cabinet.",
                context={"env_index": 0})
    llm = _StubLLM(scripted={"act": "go to cabinet 1"})
    resume = {
        "observation": "You arrive at countertop 1.",
        "admissible": ["go to cabinet 1", "take mug 1 from cabinet 1"],
        "actions": [{"step": 0, "name": "go to countertop 1",
                     "params": {"location": "countertop 1"},
                     "observation": "You arrive at countertop 1.",
                     "accepted": True, "mode": "direct",
                     "node_ref": "", "tool_ref": ""}],
        "states": [{"step": 0, "state": {"facts": ["agent_at(countertop_1)"],
                                         "inventory": []}}],
        "state": {"facts": ["agent_at(countertop_1)"], "inventory": []},
    }
    seed_context = "[Atomic Skill] hold\n```text\ntake {object}\n```"
    result = adapter.run_env_episode(task, llm, seed_context=seed_context,
                                     max_steps=5, resume=resume)
    # 合并轨迹：原有 1 步 + 续跑动作
    assert result.actions[0]["name"] == "go to countertop 1"
    assert len(result.actions) > 1, "应原地续跑出新动作"
    assert result.steps == len(result.actions)
    assert result.current_observation, "应携带当前观察供下一阶段续跑"
    assert result.current_admissible, "应携带当前可行动作"
    # 预算扣除：max_steps=5，原有 1 步 → 至多再走 4 步
    assert len(result.actions) <= 5
    # 状态追踪器从 from_state 恢复
    assert any("agent_at(countertop_1)" in (s.get("state") or {}).get("facts", [])
               for s in result.states)


def test_env_episode_preserves_won_when_final_effect_is_met():
    """最终放置同时满足节点 Effect 和 benchmark won，必须优先保留 won。"""
    class _WinningEnv:
        def step(self, action: str):
            return type("_StepResult", (), {
                "observation": "You move the mug 1 to the cabinet 1.",
                "score": 1.0, "done": True, "won": True,
                "admissible_commands": [],
            })()

    adapter = AlfWorldAdapter()
    adapter._current_env = _WinningEnv()
    task = Task(task_id="alfworld_won_test", benchmark="alfworld",
                task_type="pick_heat_then_place_in_recep",
                goal="put a hot mug in cabinet.", context={"env_index": 0})
    resume = {
        "observation": "You are carrying: a mug 1.",
        "admissible": ["move mug 1 to cabinet 1"],
        "actions": [],
        "states": [{"step": 0, "state": {
            "facts": ["agent_holds(mug_1)", "object_heated(mug_1)"],
            "inventory": ["mug_1"]}}],
        "state": {"facts": ["agent_holds(mug_1)", "object_heated(mug_1)"],
                  "inventory": ["mug_1"]},
    }
    result = adapter.run_env_episode(
        task, _StubLLM(scripted={"act": "move mug 1 to cabinet 1"}),
        seed_context="place the held object", max_steps=5, resume=resume,
        stop_effects=[{"predicate": "object.at_location",
                       "args": {"object": "$inputs.object",
                                "location": "$inputs.target_location"}}],
        effect_inputs={"object": "mug_1", "target_location": "cabinet 1"},
    )
    assert result.success is True
    assert result.atomic_complete is True
    assert result.steps == 1


def test_action_cycle_gets_bounded_agent_recovery_before_failure():
    class _CycleEnv:
        def step(self, action: str):
            won = action == "go to shelf 1"
            return type("_StepResult", (), {
                "observation": "You arrive at shelf 1." if won else "You see the desk 1.",
                "score": 1.0 if won else 0.0,
                "done": won,
                "won": won,
                "admissible_commands": ["examine desk 1", "go to shelf 1"],
            })()

    class _CycleAwareLLM:
        def generate(self, *, input_text, **_kwargs):
            action = ("go to shelf 1" if "Runtime cycle recovery" in input_text
                      else "examine desk 1")
            return type("_Resp", (), {
                "text": f"Act: {action}", "prompt_tokens": 0,
                "completion_tokens": 0, "total_tokens": 0,
                "latency_ms": 0.0, "provider": "stub", "model": "stub",
            })()

    adapter = AlfWorldAdapter()
    adapter._current_env = _CycleEnv()
    task = Task(task_id="cycle_recovery", benchmark="alfworld",
                task_type="generic", goal="reach the useful location",
                context={"env_index": 0})
    resume = {
        "observation": "You see the desk 1.",
        "admissible": ["examine desk 1", "go to shelf 1"],
        "actions": [], "states": [{"step": 0, "state": {
            "facts": [], "inventory": [], "meta": {}}}],
        "state": {"facts": [], "inventory": [], "meta": {}},
    }

    result = adapter.run_env_episode(
        task, _CycleAwareLLM(), resume=resume, max_steps=20)

    assert result.success is True
    assert result.failure_type == ""
    assert result.actions[-1]["name"] == "go to shelf 1"
    events = result.diagnostics.get("action_cycle_events") or []
    assert len(events) == 1
    assert events[0]["recovery_allowed"] is True


def test_latent_terminal_relation_requires_complete_auditable_evidence():
    """真实 episode 路径签发终局证书；错误实例不能靠 won 混入。"""
    script = {
        "go to desk 1": (
            "You arrive at desk 1. On the desk 1, you see a desklamp 1.",
            ["use desklamp 1", "go to desk 2"], False),
        "use desklamp 1": (
            "You turn on the desklamp 1.", ["go to desk 2"], False),
        "go to desk 2": (
            "You arrive at desk 2. On the desk 2, you see a bowl 2.",
            ["take bowl 2 from desk 2"], False),
        "take bowl 2 from desk 2": (
            "You pick up the bowl 2 from the desk 2.",
            ["go to desk 1"], False),
        "return to desk 1": (
            "You arrive at desk 1.", [], True),
    }

    class _LookEnv:
        def step(self, action: str):
            key = ("return to desk 1" if action == "go to desk 1"
                   and getattr(self, "visited", False) else action)
            if action == "go to desk 1":
                self.visited = True
            observation, admissible, won = script[key]
            return type("_StepResult", (), {
                "observation": observation, "score": float(won),
                "done": won, "won": won,
                "admissible_commands": admissible, "accepted": True,
            })()

    class _SequenceLLM:
        def __init__(self):
            self.actions = iter([
                "go to desk 1", "use desklamp 1", "go to desk 2",
                "take bowl 2 from desk 2", "go to desk 1"])

        def generate(self, **_kwargs):
            return type("_Resp", (), {"text": f"Act: {next(self.actions)}"})()

    adapter = AlfWorldAdapter()
    adapter._current_env = _LookEnv()
    task = Task(task_id="latent_look", benchmark="alfworld", task_type="hidden",
                goal="examine the bowl with the desklamp", context={"env_index": 0})
    resume = {
        "observation": "You are in a room.",
        "admissible": ["go to desk 1"], "actions": [],
        "states": [{"step": 0, "state": {"facts": [], "inventory": []}}],
        "state": {"facts": [], "inventory": []},
    }
    target = [{"predicate": "object.observed_with", "args": {
        "object": "$object", "associated_entity": "$associated_entity"}}]
    result = adapter.run_env_episode(
        task, _SequenceLLM(), resume=resume, max_steps=10,
        stop_effects=target,
        effect_inputs={"object": "bowl", "associated_entity": "desklamp"})

    assert result.success is True
    certificates = result.diagnostics["terminal_verified_effects"]
    assert certificates == [{
        "effect": {"predicate": "object.observed_with", "args": {
            "object": "bowl_2", "associated_entity": "desklamp_1"}},
        "action_index": 4,
        "source": "benchmark_terminal_certificate_v1",
        "benchmark_won": True,
        "evidence_facts": [
            "agent_holds(bowl_2)", "object_toggled(desklamp_1)",
            "object_at(desklamp_1, desk_1)", "agent_at(desk_1)"],
        "evidence_rule": (
            "target_held_and_associated_entity_toggled_and_colocated"),
        "standalone_action_effect": False,
    }]

    # 同一终态不能为错误主题签发证书，即使调用方声称 benchmark won。
    from atomic_skillgraph.adapters.alfworld import _terminal_effect_certificates
    wrong = _terminal_effect_certificates(
        result.states[-1]["state"], target,
        {"object": "pencil", "associated_entity": "desklamp"},
        action_index=4)
    assert wrong == []
