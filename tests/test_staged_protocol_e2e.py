"""No-API integration test for the real staged ALFWorld experiment entrypoints.

The environment is deterministic and local, but orchestration is not mocked:
``run_small.main`` and ``run_evolve_eval.main`` perform task selection, three
online conditions, persistence, frozen snapshots, milestone cloning, extension,
and a second held-out evaluation through the production code paths.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from atomic_skillgraph.adapters.benchmark import EnvRunResult, Task
from atomic_skillgraph.adapters.mock_llm import MockLLM
from atomic_skillgraph.adapters.toy_benchmarks import (
    ToyAdapter,
    ToyWorld,
    _fill_template,
    _norm,
    _parse_action_params,
    _split_step,
    _toy_effects_met,
)
from experiments import common as experiment_common
from experiments import run_evolve_eval, run_small


TASK_TYPES = [
    "pick_and_place_simple",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "look_at_obj_in_light",
    "pick_two_obj_and_place",
]
OURS = ["atomic_graph_only", "tool_repo_only", "atomic_skillgraph_full"]


def _tasks(split: str, per_type: int) -> list[Task]:
    tasks: list[Task] = []
    for type_index, task_type in enumerate(TASK_TYPES):
        for item_index in range(per_type):
            object_name = f"item_{type_index}_{item_index}_1"
            task_id = f"protocol_{split}_{type_index}_{item_index}"
            params = {
                "object": object_name,
                # Deliberately omit object_location.  A warm Acquire must be
                # retained and bind it through controlled discovery.
                "heating_station": "microwave_1",
                "target_location": "cabinet_1",
            }
            tasks.append(Task(
                task_id=task_id, benchmark="alfworld", task_type=task_type,
                goal=f"Heat {object_name} and put it in cabinet 1.",
                context={
                    "task_id": task_id,
                    "goal": f"Heat {object_name} and put it in cabinet 1.",
                    "checks": [f"object_heated({object_name})",
                               f"object_at({object_name}, cabinet_1)"],
                    "objects": {object_name: "countertop_1"},
                    "params": params,
                    "semantic_params": params,
                    "game_file": f"/{split}/{task_type}/{item_index}/game.tw-pddl",
                },
                state={"facts": [], "inventory": [], "meta": {}},
                target_effects=[
                    {"predicate": "object.heated",
                     "args": {"object": "$object"}},
                    {"predicate": "object.at_location",
                     "args": {"object": "$object",
                              "location": "$target_location"}},
                ],
            ))
    return tasks


class _ProtocolAdapter(ToyAdapter):
    """Deterministic environment; phase execution never repairs prerequisites."""

    name = "alfworld_protocol_fixture"

    def __init__(self, split: str) -> None:
        super().__init__(kinds=("env",))
        self.split = split
        self._tasks = _tasks(split, 2 if split == "train" else 1)

    def load_tasks(self, limit: int = 0,
                   task_type: str | None = None) -> list[Task]:
        selected = [task for task in self._tasks
                    if not task_type or task.task_type == task_type]
        return selected[:limit] if limit and limit > 0 else selected

    def load_balanced_tasks(self, task_types: list[str],
                            per_type_limit: int) -> list[Task]:
        return experiment_common.balanced_task_subset(
            self._tasks, task_types, per_type_limit)

    def discover_object_location(self, task: Task, object_name: str, *,
                                 resume=None, **_kwargs):
        if resume and getattr(self, "_current_world", None) is not None:
            world = self._current_world
            actions = [dict(item) for item in (resume.get("actions") or [])]
            states = [dict(item) for item in (resume.get("states") or [])]
        else:
            world = ToyWorld(dict(task.context))
            self._current_world = world
            actions, states = [], [{"step": 0, "state": world.state()}]
        wanted = re_family(object_name)
        match = next((item for item in world.objects
                      if re_family(item) == wanted), "")
        location = world.objects.get(match, "")
        result = EnvRunResult(
            actions=actions, states=states,
            current_observation=world.observation(),
            current_admissible=world.admissible(),
            final_observation=world.observation(),
        )
        return ({"object": match, "object_location": location}
                if match and location else {}), result

    def run_env_episode(self, task: Task, llm, *, seed_context="",
                        direct_steps=None, max_steps=50, resume=None,
                        stop_effects=None, effect_inputs=None, node_ref="",
                        phase_goal="") -> EnvRunResult:
        if resume and getattr(self, "_current_world", None) is not None:
            world = self._current_world
            actions = [dict(item) for item in (resume.get("actions") or [])]
            states = [dict(item) for item in (resume.get("states") or [])]
        else:
            world = ToyWorld(dict(task.context))
            self._current_world = world
            actions, states = [], [{"step": 0, "state": world.state()}]

        if direct_steps:
            commands = []
            for specification in direct_steps:
                for step in specification.get("steps") or []:
                    template, params = _split_step(
                        step, specification.get("params") or {})
                    commands.append((_fill_template(template, params), params,
                                     "direct", specification.get("tool_ref", "")))
        else:
            commands = [(command, _parse_action_params(command),
                         "seeded" if seed_context else "dynamic", "")
                        for command in self._commands_for_phase(
                            task, world, stop_effects, effect_inputs)]

        result = EnvRunResult(actions=actions, states=states,
                              direct_used=bool(direct_steps),
                              seeded_used=bool(seed_context and not direct_steps),
                              dynamic_used=bool(not seed_context and not direct_steps))
        for command, params, mode, tool_ref in commands:
            if len(result.actions) >= max_steps:
                break
            step = world.step(command)
            result.actions.append({
                "step": len(result.actions), "name": command,
                "params": params, "observation": step.observation,
                "accepted": step.accepted, "mode": mode,
                "node_ref": node_ref, "tool_ref": tool_ref,
                "origin": "tool" if mode == "direct" else "agent",
            })
            result.states.append({"step": world.steps, "state": world.state()})
            if not step.accepted:
                result.failure_type = "phase_prerequisite_missing"
                break
            if _toy_effects_met(world.state(), stop_effects, effect_inputs):
                result.atomic_complete = True
                break
            if step.done:
                break
        result.success = world.won
        result.steps = len(result.actions)
        result.current_observation = world.observation()
        result.current_admissible = world.admissible()
        result.final_observation = world.observation()
        if not result.success and not result.atomic_complete and not result.failure_type:
            result.failure_type = "phase_effect_missing"
        return result

    @staticmethod
    def _commands_for_phase(task: Task, world: ToyWorld,
                            stop_effects, effect_inputs) -> list[str]:
        obj = _norm((effect_inputs or {}).get("object")
                    or task.context["params"]["object"])
        source = next((location for item, location in world.objects.items()
                       if re_family(item) == re_family(obj)), "countertop_1")
        predicates = {str(item.get("predicate") or "").replace("_", ".")
                      for item in (stop_effects or [])}
        if (not predicates or len(predicates) > 1):  # cold whole-task execution
            return [
                f"go to {source.replace('_', ' ')}",
                f"take {obj.replace('_', ' ')} from {source.replace('_', ' ')}",
                "go to microwave 1",
                f"heat {obj.replace('_', ' ')} with microwave 1",
                "go to cabinet 1",
                f"put {obj.replace('_', ' ')} in/on cabinet 1",
            ]
        if "agent.holds" in predicates:
            return [f"go to {source.replace('_', ' ')}",
                    f"take {obj.replace('_', ' ')} from {source.replace('_', ' ')}"]
        if "object.heated" in predicates:
            # Intentionally no take fallback: if Planner prunes Acquire this
            # phase fails, reproducing the historical real-run regression.
            return ["go to microwave 1",
                    f"heat {obj.replace('_', ' ')} with microwave 1"]
        if "object.at.location" in predicates:
            return ["go to cabinet 1",
                    f"put {obj.replace('_', ' ')} in/on cabinet 1"]
        return []


def re_family(value: str) -> str:
    import re
    return re.sub(r"_\d+$", "", _norm(value))


def _adapter_factory(_benchmark, _config, **kwargs):
    return _ProtocolAdapter(str(kwargs.get("split") or "train"))


def _llm_factory(_config, mock_script=None):
    return MockLLM(script=mock_script or {})


def _invoke(monkeypatch, entrypoint, arguments: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", [entrypoint.__module__, *arguments])
    return entrypoint()


def _common_online_args(run_dir: Path, per_type: int) -> list[str]:
    return [
        "--benchmark", "alfworld", "--alfworld-split", "train",
        "--task-types", *TASK_TYPES,
        "--per-type-limit", str(per_type),
        "--conditions", *OURS,
        "--max-steps", "30", "--mock", "--run-dir", str(run_dir),
    ]


def _common_eval_args(run_dir: Path, eval_dir: Path,
                      train_count: int) -> list[str]:
    return [
        "--run-dir", str(run_dir), "--benchmark", "alfworld",
        "--conditions", *OURS, "--split", "heldout",
        "--alfworld-split", "eval_out_of_distribution",
        "--task-types", *TASK_TYPES, "--per-type-limit", "1",
        "--limit", "6", "--expected-train-count", str(train_count),
        "--expected-heldout-count", "6", "--max-steps", "30",
        "--mock", "--eval-dir", str(eval_dir),
    ]


def test_frozen_snapshot_is_bound_to_source_milestone_even_if_banks_match(
        workspace_tmp):
    source_a = workspace_tmp / "source_a" / "data"
    source_b = workspace_tmp / "source_b" / "data"
    for source in (source_a, source_b):
        (source / "skill_graph").mkdir(parents=True)
        (source / "skill_graph" / "graph.json").write_text(
            '{"nodes": {}, "edges": []}', encoding="utf-8")
    destination = workspace_tmp / "eval" / "condition" / "data"
    run_evolve_eval._snapshot_frozen_bank(source_a, destination)
    with pytest.raises(RuntimeError, match="另一在线里程碑"):
        run_evolve_eval._snapshot_frozen_bank(source_b, destination)


def test_real_entrypoints_stage_freeze_extend_and_regressions(
        monkeypatch, workspace_tmp):
    monkeypatch.setattr(run_small, "make_adapter", _adapter_factory)
    monkeypatch.setattr(run_evolve_eval, "make_adapter", _adapter_factory)
    monkeypatch.setattr(experiment_common, "make_llm", _llm_factory)
    monkeypatch.setattr(run_evolve_eval, "make_llm", _llm_factory)

    online_6 = workspace_tmp / "online_6"
    eval_6 = workspace_tmp / "eval_from_6"
    online_12 = workspace_tmp / "online_12"
    eval_12 = workspace_tmp / "eval_from_12"

    assert _invoke(monkeypatch, run_small.main,
                   _common_online_args(online_6, 1)) == 0
    source_before = run_small._condition_bank_digests(online_6, OURS)
    assert _invoke(monkeypatch, run_evolve_eval.main,
                   _common_eval_args(online_6, eval_6, 6)) == 0

    extend_args = _common_online_args(online_12, 2) + [
        "--extend-online", "--extend-from-run", str(online_6)]
    assert _invoke(monkeypatch, run_small.main, extend_args) == 0
    assert run_small._condition_bank_digests(online_6, OURS) == source_before
    assert _invoke(monkeypatch, run_evolve_eval.main,
                   _common_eval_args(online_12, eval_12, 12)) == 0

    source_results = json.loads((online_6 / "results.json").read_text())
    extended_results = json.loads((online_12 / "results.json").read_text())
    assert all(len(source_results[name]["episodes"]) == 6 for name in OURS)
    assert all(len(extended_results[name]["episodes"]) == 12 for name in OURS)
    # This is an orchestration/immutability test, not a performance claim about
    # the finite MockLLM script. Every condition must finish every episode and
    # preserve a specific non-infrastructure outcome.
    for name in OURS:
        assert all("success" in item and item.get("failure_type") != "llm_error"
                   for item in extended_results[name]["episodes"])
    for eval_path in (eval_6 / "results.json", eval_12 / "results.json"):
        frozen_results = json.loads(eval_path.read_text())
        assert all(len(frozen_results[name]["episodes"]) == 6
                   for name in OURS)
        assert all(frozen_results[name]["bank_unchanged_after_eval"]
                   for name in OURS)
    lineage = json.loads((online_12 / "online_lineage.json").read_text())
    assert lineage["source_condition_bank_sha256"] == source_before
    assert lineage["destination_freeze_skills"] is False

    traces = []
    for condition in OURS:
        for path in (eval_12 / condition / "data" / "traces").glob("*.json"):
            traces.append(json.loads(path.read_text()))
    assert all(not trace.get("metrics", {}).get(
        "controlled_location_discovery") for trace in traces)
    assert not any(action.get("origin") == "framework_discovery"
                   for trace in traces for action in trace.get("actions", []))
    assert not any(
        validation.get("passed")
        and validation.get("checks", {}).get(
            "preconditions_not_known_false") is False
        for trace in traces
        for validation in trace.get("validation_layers", {}).get("atomic", []))

    # A frozen directory is tied to one exact bank milestone.
    with pytest.raises(RuntimeError, match="另一在线里程碑"):
        _invoke(monkeypatch, run_evolve_eval.main,
                _common_eval_args(online_12, eval_6, 12))
