"""ALFWorld environment wrapper for FlowEvo.

Provides a unified interface for initialising the TextWorld-backed ALFWorld
environment, resetting to new tasks, and stepping through actions.
"""

from __future__ import annotations

import os
import re
import yaml
from pathlib import Path
from typing import Any

from alfworld_.schemas import AlfWorldAction, AlfWorldTask, ALFWORLD_TASK_TYPES


# ---------------------------------------------------------------------------
# Task type extraction from game file paths
# ---------------------------------------------------------------------------

_GAME_FILE_TYPE_RE = re.compile(
    r"(pick_and_place_simple|"
    r"pick_clean_then_place_in_recep|"
    r"pick_heat_then_place_in_recep|"
    r"pick_cool_then_place_in_recep|"
    r"look_at_obj_in_light|"
    r"pick_two_obj_and_place)"
)


def _task_type_from_game_file(game_file: str) -> str:
    """Extract the canonical task type from an ALFWorld game file path."""
    match = _GAME_FILE_TYPE_RE.search(game_file)
    if match:
        return match.group(1)
    return "unknown"


# task_type 名称 -> ALFWorld 任务类型 ID（用于环境级过滤，baseline 与 ours 共用）
TASK_TYPE_IDS = {
    "pick_and_place_simple": 1,
    "look_at_obj_in_light": 2,
    "pick_clean_then_place_in_recep": 3,
    "pick_heat_then_place_in_recep": 4,
    "pick_cool_then_place_in_recep": 5,
    "pick_two_obj_and_place": 6,
}


def _goal_from_observation(observation: str) -> str:
    """Extract the goal sentence from an ALFWorld initial observation."""
    marker = "your task is to:"
    lower = observation.lower()
    idx = lower.find(marker)
    if idx >= 0:
        return observation[idx + len(marker) :].strip()
    return observation[:200].strip()


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

class AlfWorldStepResult:
    """Result of a single environment step.

    Note: ``done`` is OR'd with the timeout from textworld's Limit wrapper,
    so ``done=True`` does NOT mean task success.  Use ``won`` (mirrored from
    the ``infos`` dict) for true task completion.
    """

    __slots__ = ("observation", "score", "done", "won", "admissible_commands")

    def __init__(
        self,
        observation: str,
        score: float,
        done: bool,
        won: bool,
        admissible_commands: list[str],
    ) -> None:
        self.observation = observation
        self.score = score
        self.done = done
        self.won = won
        self.admissible_commands = admissible_commands


# ---------------------------------------------------------------------------
# Environment wrapper
# ---------------------------------------------------------------------------

class AlfWorldEnv:
    """Wraps the AlfredTWEnv to provide a clean reset / step API.

    Usage::

        env = AlfWorldEnv(split="eval_out_of_distribution")
        count = env.initialize()
        for _ in range(count):
            task, obs, admissible = env.reset()
            while True:
                result = env.step(action)
                if result.done:
                    break
    """

    def __init__(
        self,
        split: str = "eval_out_of_distribution",
        max_steps: int = 50,
        alfworld_data: str | None = None,
        task_type: str | None = None,
    ) -> None:
        self.split = split
        self.max_steps = max_steps
        self.task_type = task_type
        self.alfworld_data = alfworld_data or os.environ.get(
            "ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"),
        )
        self._env: Any = None
        self._task_index = 0
        self._total_tasks = 0
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> int:
        """Boot the TextWorld environment.  Returns the number of tasks."""
        import alfworld.agents.environment as alf_env

        config = self._build_config()
        EnvClass = alf_env.get_environment("AlfredTWEnv")
        tw_env = EnvClass(config, train_eval=self.split)
        self._env = tw_env.init_env(batch_size=1)
        self._initialized = True

        # Read total task count from the internal game list
        game_files = getattr(tw_env, "gamefiles", None) or getattr(tw_env, "game_files", None)
        if game_files is not None:
            self._total_tasks = len(game_files)
        else:
            # Fallback: known counts
            self._total_tasks = {"eval_out_of_distribution": 134, "eval_in_distribution": 140}.get(self.split, 134)

        self._task_index = 0
        return self._total_tasks

    def reset(self) -> tuple[AlfWorldTask, str, list[str]]:
        """Reset to the next task.

        Returns ``(task, initial_observation, admissible_commands)``.
        """
        if not self._initialized:
            self.initialize()

        obs, info = self._env.reset()
        observation: str = obs[0]
        admissible: list[str] = info.get("admissible_commands", [[]])[0]
        game_file: str = (info.get("extra.gamefile") or [""])[0]

        task_type = _task_type_from_game_file(game_file)
        goal = _goal_from_observation(observation)

        task = AlfWorldTask(
            task_id="%s_%d" % (self.split, self._task_index),
            task_type=task_type,
            goal=goal,
            initial_observation=observation,
            initial_admissible=list(admissible),
            game_file=game_file,
            metadata={"task_index": self._task_index},
        )

        self._task_index += 1
        return task, observation, admissible

    def reset_to_task(self, target_index: int) -> tuple[AlfWorldTask, str, list[str]]:
        """Re-initialize and fast-forward to replay the task at *target_index*.

        Used to rerun a task from scratch: after a failed episode, the env is
        fully re-created and advanced to the same game so a fresh attempt can
        start from the initial state.
        """
        self._initialized = False
        self.initialize()
        for _ in range(target_index):
            self._env.reset()
        self._task_index = target_index
        return self.reset()

    def step(self, action: str) -> AlfWorldStepResult:
        """Execute one action in the current task."""
        obs, scores, dones, infos = self._env.step([action])
        # `won` is set by ALFWorld's GameProgression and reflects true task
        # completion.  `done` is OR'd with the Limit wrapper timeout, so we
        # cannot trust it as a success signal.
        won_list = infos.get("won", [False])
        won = bool(won_list[0]) if won_list else False
        return AlfWorldStepResult(
            observation=obs[0],
            score=float(scores[0]),
            done=bool(dones[0]),
            won=won,
            admissible_commands=infos.get("admissible_commands", [[]])[0],
        )

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _build_config(self) -> dict[str, Any]:
        data = self.alfworld_data

        split_map = {
            "eval_out_of_distribution": "valid_unseen",
            "eval_in_distribution": "valid_seen",
            "train": "train",
        }
        split_dir = split_map.get(self.split, "valid_unseen")

        # 环境级任务类型过滤（同一类任务；默认全部 6 类）
        if self.task_type:
            type_id = TASK_TYPE_IDS.get(self.task_type)
            if type_id is None:
                raise ValueError(
                    "未知 ALFWorld task_type: %r（可选：%s）"
                    % (self.task_type, ", ".join(sorted(TASK_TYPE_IDS))))
            task_types = [type_id]
        else:
            task_types = [1, 2, 3, 4, 5, 6]

        config: dict[str, Any] = {
            "env": {
                "type": "AlfredTWEnv",
                "regen_game_files": False,
                "domain_randomization": False,
                "task_types": task_types,
                "expert_type": "handcoded",
                "goal_desc_human_anns_prob": 0.0,
                "data_path": data,
                "logic": {
                    "domain": os.path.join(data, "logic", "alfred.pddl"),
                    "grammar": os.path.join(data, "logic", "alfred.twl2"),
                },
                "json_game": {
                    "data_path": os.path.join(data, "json_2.1.1"),
                },
            },
            "dataset": {
                "data_path": os.path.join(data, "json_2.1.1"),
                "eval_id_data_path": os.path.join(data, "json_2.1.1", "valid_seen"),
                "eval_ood_data_path": os.path.join(data, "json_2.1.1", "valid_unseen"),
                "num_train_games": -1,
                "num_eval_games": -1,
            },
            "general": {
                "random_seed": 42,
                "training_method": "dagger",
            },
            "dagger": {
                "training": {
                    "batch_size": 10,
                    "max_nb_steps_per_episode": self.max_steps,
                    "nb_epochs": 50,
                },
            },
            "controller": {"type": "oracle", "debug": False},
        }
        return config

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def task_count(self) -> int:
        return self._total_tasks

    @property
    def initialized(self) -> bool:
        return self._initialized


# ---------------------------------------------------------------------------
# Convenience loader (pre-loads task metadata without running episodes)
# ---------------------------------------------------------------------------

def load_alfworld_tasks(
    split: str = "eval_out_of_distribution",
    limit: int | None = None,
    task_type: str | None = None,
    alfworld_data: str | None = None,
    max_steps: int = 50,
) -> list[AlfWorldTask]:
    """Pre-load ALFWorld task metadata by iterating env.reset().

    The returned tasks carry ``initial_observation`` and ``game_file`` so
    that downstream code can inspect them without holding the environment
    open.
    """
    env = AlfWorldEnv(split=split, task_type=task_type,
                      alfworld_data=alfworld_data, max_steps=max_steps)
    count = env.initialize()
    if limit is not None:
        count = min(count, limit)

    tasks: list[AlfWorldTask] = []
    for _ in range(count):
        task, _obs, _admissible = env.reset()
        tasks.append(task)
    return tasks
