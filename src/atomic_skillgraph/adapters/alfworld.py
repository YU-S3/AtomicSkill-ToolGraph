"""ALFWorld Adapter（设计文档 v2.0 §52、§53；复用 vendored FlowEvo 的 env 协议）。

真实环境：vendored `flowevo.alfworld_.env.AlfWorldEnv`（需要 `pip install alfworld`
与数据目录）。任务按 task_type 过滤可得到「同一类任务 10 个」。

won 是唯一成功信号（与 FlowEvo 一致）。
"""

from __future__ import annotations

import re
from typing import Any

from ..core.llm import LLM
from ..core.predicates import StateSnapshot, check_effects
from .benchmark import (
    BenchmarkAdapter,
    EnvRunResult,
    Task,
)
from .code_math import ensure_flowevo_path

class AlfWorldAdapter:
    """真实 ALFWorld（textworld 后端）适配。"""

    name = "alfworld"
    # 支持 direct→seeded→dynamic 原地降级（run_env_episode 接受 resume 载荷）
    supports_in_place_resume = True

    def __init__(self, split: str = "eval_out_of_distribution", max_steps: int = 50,
                 task_type: str | None = None, alfworld_data: str | None = None) -> None:
        self.split = split
        self.max_steps = max_steps
        self.task_type_filter = task_type
        self.alfworld_data = alfworld_data
        self._env = None
        self._task_indices: dict[str, int] = {}   # task_id -> env index（供 reset_to_task）
        self._current_env = None

    # ------------------------------------------------------------------
    def _get_env(self):
        ensure_flowevo_path()
        if self._env is None:
            from alfworld_.env import AlfWorldEnv
            env = AlfWorldEnv(split=self.split, max_steps=self.max_steps,
                              alfworld_data=self.alfworld_data,
                              task_type=self.task_type_filter)
            env.initialize()
            self._env = env
        return self._env

    def load_tasks(self, limit: int = 0, task_type: str | None = None) -> list[Task]:
        """枚举 split 内任务；task_type 在环境层过滤（与 baseline 一致）。"""
        ensure_flowevo_path()
        from alfworld_.env import load_alfworld_tasks
        wanted_type = task_type or self.task_type_filter
        raw_tasks = load_alfworld_tasks(split=self.split, limit=None,
                                        task_type=wanted_type,
                                        alfworld_data=self.alfworld_data,
                                        max_steps=self.max_steps)
        if wanted_type:
            raw_tasks = [t for t in raw_tasks if t.task_type == wanted_type]
        if limit and limit > 0:
            raw_tasks = raw_tasks[:limit]
        return [self._task_from_raw(raw, index)
                for index, raw in enumerate(raw_tasks)]

    def load_balanced_tasks(self, task_types: list[str],
                            per_type_limit: int) -> list[Task]:
        """Select a deterministic balanced prefix without loading all train games.

        ALFWorld train expands to roughly ten thousand game instances.  Loading
        all of them merely to keep 50 per label is both slow and unnecessary.
        This scans the deterministic environment order only until every bucket
        is full while preserving the global environment index used by replay.
        """
        ensure_flowevo_path()
        from alfworld_.env import AlfWorldEnv, TASK_TYPE_IDS
        labels = [str(item) for item in task_types]
        unknown = sorted(set(labels) - set(TASK_TYPE_IDS))
        if unknown:
            raise ValueError(f"未知 ALFWorld task_type: {unknown}")
        env = AlfWorldEnv(split=self.split, max_steps=self.max_steps,
                          alfworld_data=self.alfworld_data, task_type=None)
        total = env.initialize()
        buckets: dict[str, list[Task]] = {label: [] for label in labels}
        for index in range(total):
            raw, _observation, _admissible = env.reset()
            label = str(raw.task_type)
            game_file = str(getattr(raw, "game_file", "")).replace("\\", "/")
            # AlfredTWEnv's ``train`` iterator also contains valid_train.
            # The formal protocol requested here uses physical train only.
            if (self.split == "train"
                    and "/json_2.1.1/train/" not in game_file):
                continue
            if label in buckets and len(buckets[label]) < per_type_limit:
                buckets[label].append(self._task_from_raw(raw, index))
            if all(len(bucket) >= per_type_limit for bucket in buckets.values()):
                break
        missing = {label: len(bucket) for label, bucket in buckets.items()
                   if len(bucket) < per_type_limit}
        if missing:
            raise ValueError(
                f"均衡任务不足：要求每类 {per_type_limit} 个，实际不足 {missing}")
        selected = [task for label in labels for task in buckets[label]]
        # Keep the randomized ALFWorld environment order.  Besides avoiding a
        # label-blocked curriculum, monotonic indices let the adapter advance
        # the same environment instead of reinitializing it for every task.
        return sorted(selected, key=lambda task: int(task.context["env_index"]))

    def _task_from_raw(self, raw, env_index: int) -> Task:
        task_id = f"alfworld_{env_index}_{raw.task_type}"
        self._task_indices[task_id] = env_index
        state = _parse_alfworld_state(raw.initial_observation)
        # Goal roles come only from the task sentence and entities exposed by
        # the environment; the official label never injects an operation.
        goal_roles = _goal_roles_from_text(str(raw.goal))
        exposed_entities = _entities_from_admissible(raw.initial_admissible)
        params = _executable_goal_params(goal_roles, exposed_entities)
        if params.get("object_type"):
            params.setdefault("object", params["object_type"])
        semantic_params = _semantic_goal_params(str(raw.goal), params)
        return Task(
            task_id=task_id, benchmark="alfworld",
            task_type=str(raw.task_type), goal=str(raw.goal),
            context={
                "kind": "env", "env_index": env_index,
                "initial_observation": str(raw.initial_observation),
                "initial_admissible": list(raw.initial_admissible),
                "goal_roles": goal_roles,
                "goal_entities": sorted(set(goal_roles.values())),
                "exposed_entities": exposed_entities,
                "game_file": str(getattr(raw, "game_file", "")),
                "params": params, "semantic_params": semantic_params,
            },
            state=state,
            target_effects=_target_effects_of("", str(raw.goal), params),
            metadata={},
        )

    def parse_task_type(self, task: Task) -> str:
        return task.task_type

    def verify_task(self, task: Task, candidate: str) -> Any:
        raise NotImplementedError("ALFWorld 不使用代码验证")

    # ------------------------------------------------------------------
    def _start_task(self, task: Task):
        """把环境定位到指定任务并返回 (observation, admissible)。"""
        env = self._get_env()
        index = int(task.context.get("env_index", 0))
        current = int(getattr(env, "_task_index", 0))
        if getattr(env, "initialized", False) and current <= index:
            while current < index:
                env.reset()
                current += 1
            _task, obs, admissible = env.reset()
        else:
            _task, obs, admissible = env.reset_to_task(index)
        expected_file = str(task.context.get("game_file") or "").replace("\\", "/")
        actual_file = str(getattr(_task, "game_file", "") or "").replace("\\", "/")
        if expected_file and actual_file and expected_file != actual_file:
            raise RuntimeError(
                "ALFWorld deterministic task mapping changed: "
                f"index={index}, expected={expected_file}, actual={actual_file}")
        self._current_env = env
        return obs, admissible

    def execute_tool(self, task: Task, tool, parameters: dict[str, Any],
                     state: dict[str, Any]) -> dict[str, Any]:
        """在当前 episode 环境上执行 action template。"""
        env = self._current_env or self._get_env()
        steps = list(tool.artifact.get("steps") or [])
        for step in steps:
            filled = _fill_template(step, parameters)
            result = env.step(filled)
            if result.done and not result.won:
                return {"passed": False,
                        "after": _parse_alfworld_state(result.observation),
                        "observation": result.observation, "reason": "env_done_failed"}
        # 模板执行后：效果验证由上层 node validator 完成；此处给基础信息
        return {"passed": True, "after": _parse_alfworld_state(result.observation),
                "observation": result.observation, "reason": "template_executed"}

    def replay_tool(self, tool, bindings: dict[str, Any],
                    before: dict[str, Any]) -> dict[str, Any]:
        """source trace replay：在源任务上重放模板，验证段落效果状态重现
        （interactive admission §28.2）。

        单个原子模板执行后任务不会 won，因此判定标准是：重放后累积状态
        覆盖源段的**新增效果事实**（after - before 差集，过滤噪声）。
        """
        replay_case = next((t for t in tool.replay_cases() if t.get("kind") == "replay"), None)
        if replay_case is None:
            return {"passed": False, "reason": "no_replay_case"}
        env_index = _replay_env_index(replay_case)
        if env_index is None:
            return {"passed": False, "reason": "source_task_unknown"}
        ensure_flowevo_path()
        from alfworld_.env import AlfWorldEnv
        env = AlfWorldEnv(split=self.split, max_steps=self.max_steps,
                          alfworld_data=self.alfworld_data,
                          task_type=self.task_type_filter)
        env.initialize()
        _task, obs, _adm = env.reset_to_task(env_index)
        admissible = list(_adm)

        expected_effects = [dict(item) for item in
                            (replay_case.get("expected_effects") or [])
                            if isinstance(item, dict)]
        before_facts = set((replay_case.get("before") or {}).get("facts") or [])
        after_facts = set((replay_case.get("after") or {}).get("facts") or [])
        expected = {f for f in (after_facts - before_facts)
                    if not f.startswith(("location_checked", "agent_at"))}

        tracker = _AlfStateTracker(initial_observation=obs)
        # 先重放源轨迹前缀（达成段落前置状态，§28.2 source trace replay）
        for prefix_step in (replay_case.get("prefix") or []):
            if _is_noop_step(prefix_step):
                continue
            prefix_result = env.step(str(prefix_step))
            tracker.update(prefix_result.observation, action=str(prefix_step),
                           accepted=_env_step_accepted(prefix_result))
            admissible = list(prefix_result.admissible_commands)
            if prefix_result.done and not prefix_result.won:
                return {"passed": False, "after": tracker.state(),
                        "reason": "prefix_failed"}
        # 再执行模板步骤
        steps = list(tool.artifact.get("steps") or [])
        replay_bindings = dict(bindings)
        is_acquire = any(str(effect.get("predicate") or "") == "agent.holds"
                         for effect in expected_effects)
        template_navigates = any(str(step).strip().lower().startswith("go to ")
                                 for step in steps)
        # take-only Acquire Tool 的 source trace replay 必须先恢复“对象可取”的
        # 环境前置状态。该发现过程是 Admission 框架行为，不计作 Tool 本体。
        if is_acquire and not template_navigates:
            replay_bindings, admissible, setup_ok = _controlled_acquire_replay_setup(
                env, tracker, admissible, replay_bindings)
            if not setup_ok:
                return {"passed": False, "after": tracker.state(),
                        "reason": "acquire_location_discovery_failed"}
        for step in steps:
            if _is_noop_step(step):
                continue
            filled = _fill_template(step, replay_bindings)
            result = env.step(filled)
            tracker.update(result.observation, action=filled,
                           accepted=_env_step_accepted(result))
            admissible = list(result.admissible_commands)
            if result.done:
                break
        replay_state = tracker.state()
        replay_facts = set(replay_state["facts"])
        if expected_effects:
            covered, missing = check_effects(
                StateSnapshot(replay_state), replay_bindings, expected_effects)
            expected_present = True
        else:
            covered = expected.issubset(replay_facts)
            missing = sorted(expected - replay_facts)[:3]
            expected_present = bool(expected)
        return {"passed": covered and expected_present,
                 "after": tracker.state(),
                 "reason": "replay_effect_covered" if covered else
                          f"replay_effect_missing: {missing}"}

    def discover_object_location(self, task: Task, object_name: str, *,
                                 resume: dict[str, Any] | None = None,
                                 max_locations: int = 30,
                                 node_ref: str = "", tool_ref: str = "",
                                 excluded_objects: set[str] | None = None,
                                 allow_passive_navigable: bool = False,
                                 ) -> tuple[dict[str, str], EnvRunResult]:
        """Bounded location discovery for a learned entity/location contract.

        Only environment-exposed navigation/open actions are used. A location
        becomes checked only after an accepted arrival and valid observation;
        a rejected visit gets one retry. For a non-acquisition Tool whose target
        is itself an exposed navigable entity, the command is sufficient to bind
        its navigation parameter passively; the Tool retains and executes the
        actual navigation step.
        """
        # 位置发现是代码框架的有界前置步骤，不是 Direct Tool 调用。
        result = EnvRunResult()
        excluded = {_norm(item) for item in (excluded_objects or set()) if item}
        if resume:
            obs = str(resume.get("observation") or "")
            admissible = list(resume.get("admissible") or [])
            result.actions = [dict(a) for a in (resume.get("actions") or [])]
            result.states = [dict(s) for s in (resume.get("states") or [])]
            tracker = _AlfStateTracker(from_state=resume.get("state") or {})
        else:
            obs, admissible = self._start_task(task)
            tracker = _AlfStateTracker(initial_observation=obs)
            result.states.append({"step": 0, "state": tracker.state()})

        def finish(binding: dict[str, str]) -> tuple[dict[str, str], EnvRunResult]:
            result.steps = len(result.actions)
            result.current_observation = obs
            result.current_admissible = list(admissible)
            result.final_observation = obs
            if not binding:
                result.failure_type = "object_location_not_found"
            return binding, result

        def observed_binding(preferred_location: str = "") -> dict[str, str]:
            """从已验证状态取绑定；当前/指定位置优先于历史观察。"""
            current_locations = {
                match.group(1) for fact in tracker.facts
                if (match := re.fullmatch(r"agent_at\((.+?)\)", fact))}
            candidates: list[tuple[int, str, str]] = []
            for fact in tracker.facts:
                match = re.fullmatch(r"object_at\((.+?),\s*(.+?)\)", fact)
                if not match or not _same_object_family(match.group(1), object_name):
                    continue
                if _norm(match.group(1)) in excluded:
                    continue
                location = _norm(match.group(2))
                priority = (0 if preferred_location and location == preferred_location
                            else 1 if location in current_locations else 2)
                candidates.append((priority, _norm(match.group(1)), location))
            if not candidates:
                return {}
            _priority, obj, location = sorted(candidates)[0]
            return {"object": obj, "object_location": location}

        def locate() -> dict[str, str]:
            # 已观察的状态事实强于命令提示。admissible 中的远端 take 只提出
            # 待物化候选，不能覆盖当前 observation 已确认的对象实例与位置。
            observed = observed_binding()
            if observed:
                return observed
            for command in admissible:
                match = re.match(r"take (.+?) from (.+)$", command, re.IGNORECASE)
                if (match and _same_object_family(match.group(1), object_name)
                        and _norm(match.group(1)) not in excluded):
                    return {"object": _norm(match.group(1)),
                            "object_location": _norm(match.group(2))}
            return {}

        if allow_passive_navigable:
            navigable = []
            for command in admissible:
                match = re.fullmatch(r"go to\s+(.+)", command,
                                     flags=re.IGNORECASE)
                if (match and _same_object_family(match.group(1), object_name)
                        and _norm(match.group(1)) not in excluded):
                    navigable.append(_norm(match.group(1)))
            if len(set(navigable)) == 1:
                target = navigable[0]
                return finish({"object": target, "object_location": target})

        # A repeated same-family Acquire may need to revisit the first source:
        # both objects can start in one receptacle. The local set still keeps
        # each location bounded to one visit in this discovery call.
        checked = (set() if excluded else
                   set(tracker.meta.get("checked_locations") or []))
        exhausted: set[str] = set()
        queue: list[tuple[str, str]] = []

        def enqueue(commands: list[str], observation: str = "") -> None:
            candidates: dict[str, str] = {}
            # Candidate locations come only from actions exposed by the
            # environment. The initial admissible set is retained because some
            # wrappers expose only a local subset after the first movement.
            exposed = [*commands,
                       *list(task.context.get("initial_admissible") or [])]
            for command in exposed:
                match = re.match(r"go to (.+)$", command, re.IGNORECASE)
                if not match:
                    continue
                location = _norm(match.group(1))
                candidates.setdefault(location, command)
            existing = {item[0] for item in queue}
            additions = [(location, command) for location, command in candidates.items()
                         if (location not in checked and location not in exhausted
                             and location not in existing)]
            additions.sort(key=lambda item: item[0])
            queue.extend(additions)

        def execute(action: str) -> Any:
            nonlocal obs, admissible
            env_result = self._current_env.step(action)
            obs = env_result.observation
            admissible = list(env_result.admissible_commands)
            accepted = _env_step_accepted(env_result)
            result.actions.append({
                "step": len(result.actions), "name": action,
                "params": _parse_action_params(action),
                "observation": obs, "accepted": accepted,
                "mode": "dynamic", "node_ref": node_ref, "tool_ref": "",
                "origin": "framework_discovery",
            })
            tracker.update(obs, action=action, accepted=accepted)
            result.states.append({"step": len(result.actions), "state": tracker.state()})
            return env_result

        def materialize(binding: dict[str, str]) -> dict[str, str]:
            """把从 admissible 推断出的远端位置变成真实可验证的当前状态。"""
            nonlocal obs, admissible
            source = _norm(binding.get("object_location") or "")
            obj = _norm(binding.get("object") or "")
            at_source = f"agent_at({source})" in tracker.facts
            observed = f"object_at({obj}, {source})" in tracker.facts
            if source and (not at_source or not observed):
                go_action = next((command for command in admissible
                                  if re.fullmatch(
                                      rf"go to\s+{re.escape(source.replace('_', ' '))}",
                                      command, flags=re.IGNORECASE)),
                                 f"go to {source.replace('_', ' ')}")
                arrived = False
                for _attempt in range(2):
                    env_result = execute(go_action)
                    if env_result.done and not env_result.won:
                        return {}
                    arrived = (_env_step_accepted(env_result)
                               and f"agent_at({source})" in tracker.facts)
                    if arrived:
                        break
                if not arrived:
                    exhausted.add(source)
                    return {}
                open_action = next((command for command in admissible
                                    if re.fullmatch(
                                        rf"open\s+{re.escape(source.replace('_', ' '))}",
                                        command, flags=re.IGNORECASE)), None)
                if open_action:
                    opened = False
                    for _attempt in range(2):
                        env_result = execute(open_action)
                        if env_result.done and not env_result.won:
                            return {}
                        if _env_step_accepted(env_result):
                            opened = True
                            break
                    if not opened:
                        exhausted.add(source)
                        return {}
                checked.add(source)
                tracker.meta["checked_locations"] = sorted(checked)
                result.states[-1]["state"] = tracker.state()
            verified = observed_binding(source)
            if (verified and verified.get("object_location") == source
                    and f"agent_at({source})" in tracker.facts):
                return verified
            return {}

        found = locate()
        if found:
            found = materialize(found)
            if found:
                return finish(found)

        enqueue(admissible, obs)
        inspected: set[str] = set()
        location_limit = max(0, int(max_locations))
        while queue:
            location, go_action = queue.pop(0)
            if location in checked or location in exhausted:
                continue
            if location not in inspected:
                if len(inspected) >= location_limit:
                    break
                inspected.add(location)
            arrived = False
            for _attempt in range(2):
                env_result = execute(go_action)
                if env_result.done:
                    return finish({})
                arrived = (_env_step_accepted(env_result)
                           and f"agent_at({location})" in tracker.facts)
                if arrived:
                    break
            if not arrived:
                exhausted.add(location)
                enqueue(admissible, obs)
                continue
            # 封闭容器必须打开后才算检查完成。
            open_action = next((cmd for cmd in admissible
                                if re.fullmatch(rf"open\s+{re.escape(location.replace('_', ' '))}",
                                                cmd, re.IGNORECASE)), None)
            if open_action:
                opened = False
                for _attempt in range(2):
                    env_result = execute(open_action)
                    if env_result.done:
                        return finish({})
                    if _env_step_accepted(env_result):
                        opened = True
                        break
                if not opened:
                    exhausted.add(location)
                    enqueue(admissible, obs)
                    continue
            checked.add(location)
            tracker.meta["checked_locations"] = sorted(checked)
            # 保存加入 checked 后的结构化状态，而不是只保存在 Python 局部变量。
            result.states[-1]["state"] = tracker.state()
            found = locate()
            if found:
                found = materialize(found)
                if found:
                    return finish(found)
            enqueue(admissible, obs)
        return finish({})

    # ------------------------------------------------------------------
    def run_env_episode(self, task: Task, llm: LLM, *,
                        seed_context: str = "",
                        direct_steps: list[dict[str, Any]] | None = None,
                        max_steps: int = 50,
                        resume: dict[str, Any] | None = None,
                        stop_effects: list[dict[str, Any]] | None = None,
                        effect_inputs: dict[str, Any] | None = None,
                        node_ref: str = "",
                        phase_goal: str = "") -> EnvRunResult:
        result = EnvRunResult()
        if resume:
            # 原地降级：从上一阶段失败点继续同一 episode（不 reset 环境）。
            # 环境对象 self._current_env 仍停在该 episode 的中间态。
            obs = str(resume.get("observation") or "")
            admissible = list(resume.get("admissible") or [])
            result.actions = [dict(a) for a in (resume.get("actions") or [])]
            result.states = [dict(s) for s in (resume.get("states") or [])]
            result.steps = len(result.actions)
            tracker = _AlfStateTracker(from_state=resume.get("state") or {})
        else:
            obs, admissible = self._start_task(task)
            tracker = _AlfStateTracker(initial_observation=obs)
            result.states.append({"step": 0, "state": tracker.state()})
        result.current_observation = obs
        result.current_admissible = list(admissible)
        if _effects_met(tracker.state(), stop_effects, effect_inputs):
            result.atomic_complete = True
            result.final_observation = obs
            return result

        if direct_steps:
            result.direct_used = True
            for step_spec in direct_steps:
                for step in step_spec.get("steps") or []:
                    template, params = _split_step(step, step_spec.get("params") or {})
                    filled = _fill_template(template, params)
                    # Location discovery may already have established a Tool's
                    # navigation precondition. Treating an idempotent movement
                    # as satisfied avoids sending an invalid same-location
                    # command while keeping that step in the reusable artifact.
                    if _navigation_already_satisfied(filled, tracker):
                        continue
                    env_result = self._current_env.step(filled)
                    obs = env_result.observation
                    admissible = env_result.admissible_commands
                    result.current_observation = obs
                    result.current_admissible = list(admissible)
                    result.actions.append({
                        "step": len(result.actions),
                        "name": filled,
                        "params": params,
                        "observation": env_result.observation,
                        "accepted": _env_step_accepted(env_result),
                        "mode": "direct",
                        "node_ref": step_spec.get("node_ref", ""),
                        "tool_ref": step_spec.get("tool_ref", ""),
                    })
                    tracker.update(env_result.observation, action=filled,
                                   accepted=_env_step_accepted(env_result))
                    result.states.append({"step": len(result.actions),
                                          "state": tracker.state()})
                    # The action that establishes the final atomic Effect may
                    # simultaneously finish the benchmark.  Preserve the
                    # authoritative ALFWorld signal before returning at the
                    # local Effect boundary.
                    if env_result.won:
                        result.success = True
                        _record_terminal_effect_certificates(
                            result, tracker.state(), stop_effects, effect_inputs,
                            action_index=len(result.actions) - 1)
                        result.atomic_complete = _effects_met(
                            tracker.state(), stop_effects, effect_inputs)
                        result.steps = len(result.actions)
                        result.final_observation = env_result.observation
                        return result
                    if _effects_met(tracker.state(), stop_effects, effect_inputs):
                        result.atomic_complete = True
                        result.steps = len(result.actions)
                        result.final_observation = obs
                        return result
                    if env_result.done:
                        result.failure_type = "direct_template_failed"
                        result.steps = len(result.actions)
                        result.final_observation = env_result.observation
                        return result
            result.failure_type = "direct_template_goal_missed"
            result.steps = len(result.actions)
            result.final_observation = obs
            return result

        # seeded / dynamic LLM 循环（与 FlowEvo AlfWorldGenerator 的 ReAct prompt 对齐，
        # §57.1：baseline 与 ours 保持相同 harness/prompt/model，只改变 Skill 表示）
        result.seeded_used = bool(seed_context)
        result.dynamic_used = not seed_context
        instructions = _ALFWORLD_SYSTEM_PROMPT
        active_goal = phase_goal or task.goal
        prompt = _build_step_user(active_goal, obs, admissible,
                                  action_history=[], observation_history=[],
                                  skill_context=seed_context,
                                  checked_locations=_checked_locations(tracker))
        nothing_count = 0
        last_actions: list[str] = []
        action_history: list[str] = []
        observation_history: list[str] = []
        # 连续 LLM 失败上限：API 失速/连续空响应时快速结束本任务并跳到
        # 下一个（与 FlowEvo executor 的 3 连败保护对齐），避免空烧步数
        llm_error_streak = 0
        max_llm_errors = max(1, int(getattr(self, "llm_max_consecutive_errors", 1)))
        cycle_recoveries = 0
        max_cycle_recoveries = max(0, int(getattr(
            self, "max_action_cycle_recoveries", 2)))
        # 原地降级：步数预算扣除上一阶段已消耗的步数
        remaining_steps = max(0, int(max_steps) - result.steps)
        local_env_steps = 0
        while local_env_steps < remaining_steps:
            try:
                # 贪心解码：与 FlowEvo AlfWorldGenerator 一致（temperature=0.0）
                response = llm.generate(instructions=instructions, input_text=prompt,
                                        temperature=0.0, thinking="disabled")
                action = _parse_single_action(response.text, admissible)
                if any(str(effect.get("predicate") or "").replace("_", ".") == "agent.holds"
                       for effect in (stop_effects or []) if isinstance(effect, dict)):
                    action = _avoid_checked_location(action, admissible,
                                                     _checked_locations(tracker))
                llm_error_streak = 0
            except Exception as exc:  # noqa: BLE001
                llm_error_streak += 1
                result.infrastructure_errors.append({
                    "attempt": llm_error_streak,
                    "exception_type": type(exc).__name__,
                    "message": str(exc)[:500],
                    "action_count": len(result.actions),
                })
                if llm_error_streak >= max_llm_errors:
                    result.failure_type = "llm_error"
                    result.steps = len(result.actions)
                    result.final_observation = obs
                    return result
                # API 错误不是环境动作：保持同一 observation 原地重试，不能用
                # admissible[0] 污染轨迹或让任务在“重试”期间悄悄前进。
                continue
            env_result = self._current_env.step(action)
            local_env_steps += 1
            accepted = _env_step_accepted(env_result)
            result.actions.append({
                "step": len(result.actions),
                "name": action,
                "params": _parse_action_params(action),
                "observation": env_result.observation,
                "accepted": accepted,
                "mode": "seeded" if seed_context else "dynamic",
                "node_ref": node_ref,
                "tool_ref": "",
            })
            tracker.update(env_result.observation, action=action,
                           accepted=accepted)
            _record_location_inspection(
                tracker, action, env_result.observation, accepted=accepted)
            result.states.append({"step": len(result.actions),
                                  "state": tracker.state()})
            action_history.append(action)
            observation_history.append(env_result.observation)
            obs = env_result.observation
            admissible = env_result.admissible_commands
            result.current_observation = obs
            result.current_admissible = list(admissible)
            # ``won`` is the benchmark-level source of truth.  Check it before
            # the atomic stop condition so the final placement is not mistaken
            # for an unfinished episode that needs a second Dynamic call.
            if env_result.won:
                result.success = True
                _record_terminal_effect_certificates(
                    result, tracker.state(), stop_effects, effect_inputs,
                    action_index=len(result.actions) - 1)
                result.atomic_complete = _effects_met(
                    tracker.state(), stop_effects, effect_inputs)
                result.steps = len(result.actions)
                result.final_observation = obs
                return result
            if _effects_met(tracker.state(), stop_effects, effect_inputs):
                result.atomic_complete = True
                result.steps = len(result.actions)
                result.final_observation = obs
                return result
            if env_result.done:
                result.failure_type = "timeout_or_done"
                result.steps = len(result.actions)
                result.final_observation = obs
                return result
            if not accepted:
                nothing_count += 1
                if nothing_count >= 3:
                    result.failure_type = "nothing_happens"
                    break
            else:
                nothing_count = 0
            # 循环检测（与 FlowEvo executor 完全一致）：ABCABCABC（3 次重复的
            # 3 元组，共 9 步），且 12 步后才检查——避免误杀正常探索
            last_actions.append(action)
            if len(last_actions) >= 12 and _has_cycle(last_actions,
                                                      cycle_len=3, repeats=3):
                cycle_window = list(last_actions[-9:])
                result.diagnostics.setdefault("action_cycle_events", []).append({
                    "action_count": len(result.actions),
                    "cycle_actions": cycle_window,
                    "recovery_index": cycle_recoveries + 1,
                    "recovery_allowed": cycle_recoveries < max_cycle_recoveries,
                })
                if cycle_recoveries < max_cycle_recoveries:
                    cycle_recoveries += 1
                    # Preserve the real environment state and ask the Agent to
                    # choose a different admissible branch.  The framework does
                    # not invent an action or use task-type knowledge.
                    prompt = _build_step_user(
                        active_goal, obs, admissible,
                        action_history=action_history,
                        observation_history=observation_history,
                        skill_context=_compact_seed_context(seed_context),
                        checked_locations=_checked_locations(tracker),
                    ) + (
                        "\n\nRuntime cycle recovery: the following recent action "
                        "sequence repeated without satisfying the current goal: "
                        f"{cycle_window}. Choose a DIFFERENT admissible action or "
                        "a different object/location branch. Do not repeat this "
                        "sequence."
                    )
                    last_actions.clear()
                    continue
                result.failure_type = "action_cycle"
                break
            # 重建 prompt（带最近 10 步历史，与 FlowEvo 一致）；
            # 首步之后用轻量技能上下文（去掉 Tool 体，仅 summary/guideline）
            step_context = (seed_context if local_env_steps == 1
                            else _compact_seed_context(seed_context))
            prompt = _build_step_user(active_goal, obs, admissible,
                                      action_history=action_history,
                                      observation_history=observation_history,
                                      skill_context=step_context,
                                      checked_locations=_checked_locations(tracker))
        result.success = False
        result.steps = len(result.actions)
        result.failure_type = result.failure_type or "max_steps"
        result.final_observation = obs
        return result


def _effects_met(state: dict[str, Any], effects: list[dict[str, Any]] | None,
                 inputs: dict[str, Any] | None) -> bool:
    if not effects:
        return False
    passed, _missing = check_effects(StateSnapshot(state), inputs or {}, effects,
                                     {"harness": "env"})
    return passed


# ---------------------------------------------------------------------------

def _parse_alfworld_state(observation: str) -> dict[str, Any]:
    """把 ALFWorld PDDL 观察文本解析为单步状态快照（局部事实）。"""
    facts: list[str] = []
    inventory: list[str] = []
    text = str(observation or "").lower()

    arrive = re.search(r"you arrive at (.+?)\.", text)
    if arrive:
        facts.append(f"agent_at({_norm(arrive.group(1))})")

    for match in re.finditer(r"on the (.+?), you see (.+?)\.", text):
        loc = _norm(match.group(1))
        for obj in _extract_objects(match.group(2)):
            facts.append(f"object_at({obj}, {loc})")
            facts.append(f"object_exists({obj})")

    open_match = re.search(r"the ([a-z0-9 ]+?) is open", text)
    in_it = re.search(r"in it, you see (.+?)\.", text)
    if in_it and open_match:
        loc = _norm(open_match.group(1))
        for obj in _extract_objects(in_it.group(1)):
            facts.append(f"object_at({obj}, {loc})")
            facts.append(f"object_exists({obj})")

    pickup = re.search(r"you pick up the (.+?) from the (.+?)\.", text)
    if pickup:
        obj = _norm(pickup.group(1))
        inventory.append(obj)
        facts.append(f"agent_holds({obj})")

    move = re.search(r"you move the (.+?) to the (.+?)\.", text)
    if move:
        obj, loc = _norm(move.group(1)), _norm(move.group(2))
        facts.append(f"object_at({obj}, {loc})")

    put = re.search(r"you put the (.+?) (?:in|on) the (.+?)\.", text)
    if put:
        obj, loc = _norm(put.group(1)), _norm(put.group(2))
        facts.append(f"object_at({obj}, {loc})")

    heat = re.search(r"you heat the (.+?) using the (.+?)\.", text)
    if heat:
        facts.append(f"object_heated({_norm(heat.group(1))})")
    clean = re.search(r"you clean the (.+?) using the (.+?)\.", text)
    if clean:
        facts.append(f"object_cleaned({_norm(clean.group(1))})")
    cool = re.search(r"you cool the (.+?) using the (.+?)\.", text)
    if cool:
        facts.append(f"object_cooled({_norm(cool.group(1))})")
    toggled_on = re.search(r"you turn on the (.+?)\.", text)
    if toggled_on:
        facts.append(f"object_toggled({_norm(toggled_on.group(1))})")

    for open_obj in re.findall(r"the ([a-z0-9 ]+?) is open", text):
        facts.append(f"container_open({_norm(open_obj)})")

    carrying = re.search(r"you are carrying (.+)\.?", text)
    if carrying:
        for obj in _extract_objects(carrying.group(1)):
            if obj != "nothing":
                inventory.append(obj)
                facts.append(f"agent_holds({obj})")

    return {"facts": sorted(set(facts)), "inventory": inventory, "text": str(observation or "")[:2000],
            "meta": {}}


def _extract_objects(phrase: str) -> list[str]:
    """从 'a mug 2, and a bowl 1' 类短语提取对象名（下划线形式）。"""
    phrase = re.sub(r"\s+and\s+", ", ", str(phrase).lower())
    objs: list[str] = []
    for part in phrase.split(","):
        part = re.sub(r"^(a |an |some )", "", part.strip()).strip().rstrip(".")
        if part and part not in ("nothing", "anything"):
            objs.append(_norm(part))
    return [o for o in objs if o]


class _AlfStateTracker:
    """累积式 ALFWorld 状态跟踪：每步观察叠加增量事实。

    用于轨迹状态快照（atomicizer 的 before/after 与节点验证器）。
    """

    def __init__(self, initial_observation: str = "",
                 from_state: dict[str, Any] | None = None) -> None:
        self.facts: set[str] = set()
        self.inventory: list[str] = []
        self.meta: dict[str, Any] = {}
        if from_state:
            self.facts = set((from_state or {}).get("facts") or [])
            self.inventory = list((from_state or {}).get("inventory") or [])
            self.meta = dict((from_state or {}).get("meta") or {})
        if initial_observation:
            self.update(initial_observation)

    def update(self, observation: str, *, action: str = "",
               accepted: bool = True) -> None:
        text = str(observation or "").lower()
        # Per-step epistemic additions are persisted separately from world
        # transitions.  The facts remain available to later Preconditions,
        # but opening/visiting a place must not be learned as an ability that
        # physically moves every newly revealed object there.
        observed_facts: set[str] = set()
        self.meta["last_observed_facts"] = []

        arrive = re.search(r"you arrive at (.+?)\.", text)
        if arrive:
            self.facts = {f for f in self.facts if not f.startswith("agent_at(")}
            self.facts.add(f"agent_at({_norm(arrive.group(1))})")

        def add_objects(phrase: str, loc: str) -> None:
            for obj in _extract_objects(phrase):
                exists_fact = f"object_exists({obj})"
                location_fact = f"object_at({obj}, {_norm(loc)})"
                if exists_fact not in self.facts:
                    observed_facts.add(exists_fact)
                if location_fact not in self.facts:
                    observed_facts.add(location_fact)
                self.facts.add(exists_fact)
                self.facts.add(location_fact)

        for match in re.finditer(r"on the (.+?), you see (.+?)\.", text):
            add_objects(match.group(2), match.group(1))

        open_match = re.search(r"the ([a-z0-9 ]+?) is open", text)
        in_it = re.search(r"in it, you see (.+?)\.", text)
        if in_it and open_match:
            add_objects(in_it.group(1), open_match.group(1))
        if open_match:
            self.facts.add(f"container_open({_norm(open_match.group(1))})")
        closed = re.search(r"the ([a-z0-9 ]+?) is closed", text)
        if closed:
            self.facts.discard(f"container_open({_norm(closed.group(1))})")

        pickup = re.search(r"you pick up the (.+?) from the (.+?)\.", text)
        if pickup:
            obj = _norm(pickup.group(1))
            # take 成功后对象已不在原 receptacle。旧实现只增加
            # agent_holds，保留了 object_at，使后续 Effect 差分缺少
            # 真实的负 Effect，且会形成“同时手持且仍在容器”的矛盾状态。
            self.facts = {f for f in self.facts
                          if not f.startswith(f"object_at({obj},")}
            if obj not in self.inventory:
                self.inventory.append(obj)
            self.facts.add(f"agent_holds({obj})")

        move = re.search(r"you move the (.+?) to the (.+?)\.", text)
        put = re.search(r"you put the (.+?) (?:in|on) the (.+?)\.", text)
        for m in (move, put):
            if m:
                obj, loc = _norm(m.group(1)), _norm(m.group(2))
                self.facts = {f for f in self.facts
                              if not f.startswith(f"object_at({obj},")
                              and f != f"agent_holds({obj})"}
                if obj in self.inventory:
                    self.inventory.remove(obj)
                self.facts.add(f"object_at({obj}, {loc})")

        heat = re.search(r"you heat the (.+?) using the (.+?)\.", text)
        if heat:
            self.facts.add(f"object_heated({_norm(heat.group(1))})")
        clean = re.search(r"you clean the (.+?) using the (.+?)\.", text)
        if clean:
            self.facts.add(f"object_cleaned({_norm(clean.group(1))})")
        cool = re.search(r"you cool the (.+?) using the (.+?)\.", text)
        if cool:
            self.facts.add(f"object_cooled({_norm(cool.group(1))})")
        toggled_on = re.search(r"you turn on the (.+?)\.", text)
        if toggled_on:
            self.facts.add(f"object_toggled({_norm(toggled_on.group(1))})")
        toggled_off = re.search(r"you turn off the (.+?)\.", text)
        if toggled_off:
            self.facts.discard(f"object_toggled({_norm(toggled_off.group(1))})")

        carrying = re.search(r"you are carrying (.+)\.?", text)
        if carrying:
            self.inventory = []
            for obj in _extract_objects(carrying.group(1)):
                if obj != "nothing":
                    self.inventory.append(obj)
                    self.facts.add(f"agent_holds({obj})")

        # Record the achieved relation only when the environment accepts the
        # interaction and its observation confirms activation while an object
        # is held.  This uses protocol evidence, never the official task label.
        used = re.fullmatch(r"use\s+(.+)", str(action or "").strip(),
                            flags=re.IGNORECASE)
        if accepted and used and re.search(r"\byou turn on\b", text):
            associated = _norm(used.group(1))
            for obj in self.inventory:
                self.facts.add(
                    f"object_observed_with({_norm(obj)}, {associated})")
        self.meta["last_observed_facts"] = sorted(observed_facts)

    def state(self) -> dict[str, Any]:
        return {"facts": sorted(self.facts), "inventory": list(self.inventory),
                "text": "", "meta": dict(self.meta)}


def _norm(text: str) -> str:
    """规范化实体名：小写 + 空格转下划线（mug 2 -> mug_2）。"""
    return re.sub(r"\s+", "_", str(text).strip().lower())


def _env_step_accepted(result: Any) -> bool:
    """Use an explicit adapter verdict when present, with text fallback."""
    observation = str(getattr(result, "observation", "") or "")
    explicit = bool(getattr(result, "accepted", True))
    return explicit and "nothing happens" not in observation.lower()


def _navigation_already_satisfied(action: str, tracker: _AlfStateTracker) -> bool:
    match = re.fullmatch(r"go to\s+(.+)", str(action).strip(),
                         flags=re.IGNORECASE)
    return bool(match and f"agent_at({_norm(match.group(1))})" in tracker.facts)


def _same_object_family(candidate: str, requested: str) -> bool:
    """``apple`` 与具体实例 ``apple 2``/``apple_2`` 属于同一对象族。"""
    left = re.sub(r"_\d+$", "", _norm(candidate))
    right = re.sub(r"_\d+$", "", _norm(requested))
    return bool(left and left == right)


def _record_terminal_effect_certificates(
        result: EnvRunResult, state: dict[str, Any],
        target_effects: list[dict[str, Any]] | None,
        effect_inputs: dict[str, Any] | None, *, action_index: int) -> None:
    """Persist latent terminal relations only when their evidence is complete.

    ALFWorld may report ``won`` after a final navigation while its observation
    does not spell out the achieved relation.  ``won`` alone is deliberately
    insufficient: each supported predicate has a predicate-level certificate
    that binds concrete participants and cites all terminal state facts.  The
    certificate is extraction evidence, not a mutation of the tracked world
    state and not evidence that the last action is a standalone Tool.
    """
    certificates = _terminal_effect_certificates(
        state, target_effects, effect_inputs, action_index=action_index)
    if certificates:
        result.diagnostics.setdefault(
            "terminal_verified_effects", []).extend(certificates)


def _terminal_effect_certificates(
        state: dict[str, Any], target_effects: list[dict[str, Any]] | None,
        effect_inputs: dict[str, Any] | None, *, action_index: int
        ) -> list[dict[str, Any]]:
    """Return auditable certificates for latent goal relations.

    This dispatch is predicate-driven and never reads an ALFWorld task label.
    A new latent predicate must provide its own evidence rule; unknown
    predicates fail closed.
    """
    facts = {str(item) for item in (state.get("facts") or [])}
    inventory = {_norm(item) for item in (state.get("inventory") or [])}
    inputs = dict(effect_inputs or {})

    def fact_values(name: str, arity: int) -> list[tuple[str, ...]]:
        values: list[tuple[str, ...]] = []
        for fact in facts:
            match = re.fullmatch(rf"{re.escape(name)}\((.*)\)", fact)
            if not match:
                continue
            args = tuple(_norm(part) for part in match.group(1).split(","))
            if len(args) == arity:
                values.append(args)
        return values

    def requested(value: Any) -> str:
        text = str(value or "")
        if text.startswith("$"):
            key = text[1:].split(".")[-1]
            return _norm(inputs.get(key, ""))
        return _norm(text)

    certificates: list[dict[str, Any]] = []
    held = inventory | {item[0] for item in fact_values("agent_holds", 1)}
    toggled = {item[0] for item in fact_values("object_toggled", 1)}
    agent_locations = {item[0] for item in fact_values("agent_at", 1)}
    object_locations = fact_values("object_at", 2)
    for target in target_effects or []:
        if not isinstance(target, dict):
            continue
        # Explicit state deltas remain the primary evidence path.  This
        # certificate is only for a relation ALFWorld keeps latent at success.
        if str(target.get("predicate") or "") != "object.observed_with":
            continue
        args = dict(target.get("args") or {})
        wanted_object = requested(args.get("object"))
        wanted_associated = requested(args.get("associated_entity"))
        objects = sorted(item for item in held
                         if _same_object_family(item, wanted_object))
        associated = sorted(item for item in toggled
                            if _same_object_family(item, wanted_associated))
        for concrete_object in objects:
            for concrete_associated in associated:
                locations = sorted(
                    location for entity, location in object_locations
                    if entity == concrete_associated and location in agent_locations)
                if not locations:
                    continue
                location = locations[0]
                evidence = [
                    f"agent_holds({concrete_object})",
                    f"object_toggled({concrete_associated})",
                    f"object_at({concrete_associated}, {location})",
                    f"agent_at({location})",
                ]
                if not all(item in facts or (
                        item == f"agent_holds({concrete_object})"
                        and concrete_object in inventory) for item in evidence):
                    continue
                certificates.append({
                    "effect": {
                        "predicate": "object.observed_with",
                        "args": {"object": concrete_object,
                                 "associated_entity": concrete_associated},
                    },
                    "action_index": int(action_index),
                    "source": "benchmark_terminal_certificate_v1",
                    "benchmark_won": True,
                    "evidence_facts": evidence,
                    "evidence_rule": (
                        "target_held_and_associated_entity_toggled_and_colocated"),
                    # The relation is certified at the terminal boundary.  It
                    # is not attributed to the final navigation action alone.
                    "standalone_action_effect": False,
                })
                break
            if certificates:
                break
    return certificates


def _goal_roles_from_text(goal: str) -> dict[str, str]:
    """Parse semantic participants from the goal sentence, without a task label.

    This is an environment-language adapter, not a solution template: it names
    the theme, final relation target, and an explicitly mentioned associated
    entity.  It never supplies an intermediate resource or an action sequence.
    """
    text = re.sub(r"\s+", " ", str(goal or "").strip().lower()).rstrip(". !?")
    if not text:
        return {}

    roles: dict[str, str] = {}
    relation = re.search(
        r"\b(in|on|into|onto|under|with|using)\s+(?:the\s+)?"
        r"([a-z][a-z0-9]*(?:\s+\d+)?)\s*$", text)
    prefix = text
    if relation:
        relation_name, value = relation.group(1), _norm(relation.group(2))
        if relation_name in {"in", "on", "into", "onto"}:
            roles["destination"] = value
        else:
            roles["associated_entity"] = value
        prefix = text[:relation.start()].strip()

    # A coordinated goal may contain both a final destination and an earlier
    # clause ("... and put it in ...").  The theme is the noun phrase of the
    # first clause, or the phrase immediately before the final relation.
    first_clause = re.split(r"\b(?:and then|then|and)\b", prefix, maxsplit=1)[0]
    tokens = re.findall(r"[a-z][a-z0-9]*", first_clause)
    determiners = {"a", "an", "the", "some", "one", "two", "three", "four", "five"}
    pronouns = {"it", "them", "this", "that"}
    # ALFWorld goal language places the theme at the end of the first clause.
    # Choosing the final content token avoids a benchmark object whitelist and
    # does not reveal how that entity should be manipulated.
    content = [token for token in tokens if token not in determiners and token not in pronouns]
    if content:
        roles["theme"] = _norm(content[-1])

    count = _goal_cardinality(text)
    if count > 1:
        roles["cardinality"] = str(count)
    return roles


def _entities_from_admissible(commands: list[str]) -> list[str]:
    """Collect grounded entities exposed by the environment action protocol."""
    entities: list[str] = []
    for command in commands or []:
        values = list(_parse_action_params(str(command)).values())
        unary = re.fullmatch(
            r"(?:go to|open|close|use|examine)\s+(.+)",
            str(command).strip(), flags=re.IGNORECASE)
        if unary:
            values.append(unary.group(1))
        for value in values:
            normalized = _norm(value)
            if normalized and normalized not in entities:
                entities.append(normalized)
    return entities


def _resolve_exposed_entity(value: str, exposed: list[str]) -> str:
    matches = [item for item in exposed if _same_object_family(item, value)]
    # A unique exposed instance is executable evidence.  Ambiguous families
    # remain abstract until runtime observation resolves the exact instance.
    return matches[0] if len(matches) == 1 else _norm(value)


def _executable_goal_params(goal_roles: dict[str, str],
                            exposed_entities: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    theme = str(goal_roles.get("theme") or "")
    destination = str(goal_roles.get("destination") or "")
    associated = str(goal_roles.get("associated_entity") or "")
    if theme:
        params["object"] = _resolve_exposed_entity(theme, exposed_entities)
    if destination:
        params["target_location"] = _resolve_exposed_entity(
            destination, exposed_entities)
    if associated:
        params["associated_entity"] = _resolve_exposed_entity(
            associated, exposed_entities)
    cardinality = int(goal_roles.get("cardinality") or 1)
    if cardinality > 1 and theme:
        params["object_type"] = _norm(theme)
        params["object"] = _norm(theme)
    return params


def _semantic_goal_params(goal: str, executable_params: dict[str, Any]) -> dict[str, Any]:
    """Separate goal-level entity families from executable instances."""
    semantic = dict(executable_params or {})
    text = str(goal or "").strip().lower()
    for role, raw_value in list(semantic.items()):
        if not isinstance(raw_value, str):
            continue
        normalized = _norm(raw_value)
        family = re.sub(r"_\d+$", "", normalized)
        family_text = family.replace("_", " ")
        explicit_instance = re.search(
            rf"\b{re.escape(family_text)}\s+\d+\b", text)
        if (not explicit_instance
                and re.search(rf"\b{re.escape(family_text)}\b", text)):
            semantic[str(role)] = family_text
    return semantic


def _is_noop_step(step: str) -> bool:
    """无信息动作（inventory/look）在模板回放时跳过。"""
    return str(step).strip().lower() in ("inventory", "look")


def _controlled_acquire_replay_setup(
        env: Any, tracker: _AlfStateTracker, admissible: list[str],
        bindings: dict[str, Any], max_locations: int = 30,
        ) -> tuple[dict[str, Any], list[str], bool]:
    """Admission replay 专用的有界位置发现；不属于 Action Tool artifact。"""
    resolved = dict(bindings)
    commands = list(admissible)
    wanted = str(resolved.get("object") or "")
    preferred = _norm(resolved.get("object_location") or "")

    def take_binding(items: list[str]) -> dict[str, str]:
        for command in items:
            match = re.fullmatch(r"take (.+?) from (.+)", command,
                                 flags=re.IGNORECASE)
            if match and (not wanted or _same_object_family(match.group(1), wanted)):
                return {"object": _norm(match.group(1)),
                        "object_location": _norm(match.group(2))}
        return {}

    found = take_binding(commands)
    if found:
        resolved.update(found)
        return resolved, commands, True

    checked: set[str] = set()
    exhausted: set[str] = set()
    inspected: set[str] = set()
    queue: list[tuple[str, str]] = []

    def enqueue(items: list[str]) -> None:
        candidates: list[tuple[str, str]] = []
        for command in items:
            match = re.fullmatch(r"go to (.+)", command, flags=re.IGNORECASE)
            if match:
                location = _norm(match.group(1))
                if (location not in checked and location not in exhausted
                        and all(location != item[0] for item in queue)):
                    candidates.append((location, command))
        candidates.sort(key=lambda item: (item[0] != preferred, item[0]))
        queue.extend(candidates)

    enqueue(commands)
    while queue and len(inspected) < max(0, int(max_locations)):
        location, go_action = queue.pop(0)
        if location in checked or location in exhausted:
            continue
        inspected.add(location)
        arrived = False
        for _attempt in range(2):
            step_result = env.step(go_action)
            tracker.update(step_result.observation, action=go_action,
                           accepted=_env_step_accepted(step_result))
            commands = list(step_result.admissible_commands)
            if step_result.done and not step_result.won:
                return resolved, commands, False
            arrived = (_env_step_accepted(step_result)
                       and f"agent_at({location})" in tracker.facts)
            if arrived:
                break
        if not arrived:
            exhausted.add(location)
            enqueue(commands)
            continue
        open_action = next((command for command in commands
                            if re.fullmatch(
                                rf"open\s+{re.escape(location.replace('_', ' '))}",
                                command, flags=re.IGNORECASE)), None)
        if open_action:
            opened = False
            for _attempt in range(2):
                step_result = env.step(open_action)
                tracker.update(step_result.observation, action=open_action,
                               accepted=_env_step_accepted(step_result))
                commands = list(step_result.admissible_commands)
                if step_result.done and not step_result.won:
                    return resolved, commands, False
                if _env_step_accepted(step_result):
                    opened = True
                    break
            if not opened:
                exhausted.add(location)
                enqueue(commands)
                continue
        checked.add(location)
        found = take_binding(commands)
        if found:
            resolved.update(found)
            return resolved, commands, True
        enqueue(commands)
    return resolved, commands, False


def _target_effects_of(task_type: str, goal: str = "",
                       params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Compose a goal-state contract from the user-visible sentence only.

    This parser declares *what* must hold, never an action sequence. It does
    not enumerate benchmark task types, objects, receptacles, or solutions.
    The learned SkillGraph/Tool Repository remains responsible for discovering
    *how* to achieve the contract.

    ``task_type`` remains in the signature for compatibility with stored-run
    tooling, but is intentionally ignored.  Labels are sampling/report fields.
    """
    params = dict(params or {})
    text = str(goal or "").lower()
    effects: list[dict[str, Any]] = []

    # These words formalize an explicitly stated desired state. They do not
    # prescribe an action, resource, boundary, or workflow to the agent.
    if re.search(r"\b(?:heat|heated|hot)\b", text):
        effects.append({"predicate": "object.heated",
                        "args": {"object": "$object"}})
    if re.search(r"\b(?:clean|cleaned)\b", text):
        effects.append({"predicate": "object.cleaned",
                        "args": {"object": "$object"}})
    if re.search(r"\b(?:cool|cooled|cold)\b", text):
        effects.append({"predicate": "object.cooled",
                        "args": {"object": "$object"}})

    associated = params.get("associated_entity")
    is_observation_goal = bool(
        associated and re.search(r"\b(?:examine|look at)\b", text))
    if is_observation_goal:
        return [
            {"predicate": "object.observed_with",
             "args": {"object": "$object",
                      "associated_entity": "$associated_entity"}},
        ]

    has_destination = bool(
        params.get("target_location")
        or re.search(r"\b(?:put|place)\b", text))
    if has_destination:
        count = _goal_cardinality(text)
        object_slot = "$object_type" if count > 1 else "$object"
        placement: dict[str, Any] = {
            "predicate": "object.at_location",
            "args": {"object": object_slot, "location": "$target_location"},
        }
        if count > 1:
            placement.update({"cardinality": count, "distinct_by": "object"})
        effects.append(placement)
    return effects


def _goal_cardinality(text: str) -> int:
    """Parse an explicit count from the visible goal sentence; default one."""
    words = {"two": 2, "three": 3, "four": 4, "five": 5}
    match = re.search(
        r"\b(?:put|place|pick|find)\s+(two|three|four|five|[2-9])\b", text)
    if match:
        token = match.group(1)
        return words.get(token, int(token) if token.isdigit() else 1)
    return 1


def _fill_template(step: str, params: dict[str, Any]) -> str:
    try:
        return step.format(**{k: str(v).replace("_", " ") for k, v in params.items()})
    except (KeyError, ValueError):
        return step


def _split_step(step: Any, default_params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if isinstance(step, dict):
        return str(step.get("template", "")), dict(step.get("params") or default_params)
    return str(step), dict(default_params)


# ---------------------------------------------------------------------------
# Runtime action-selection prompt.  It deliberately contains no benchmark
# taxonomy, solution pattern, operation catalogue, or hand-written workflow.
# ---------------------------------------------------------------------------

_ALFWORLD_SYSTEM_PROMPT = (
    "You control an agent in an interactive environment. You receive only the "
    "task goal, current observation, prior interaction, optional learned capability "
    "evidence, and actions currently declared valid by the environment. Infer the next "
    "step from that evidence. Do not assume a benchmark task taxonomy or a predefined "
    "workflow. Choose exactly one action from the current valid-actions list.\n\n"
    "Format your response as:\n"
    "Act: <the exact action from the valid actions list>\n"
)

_ACT_RE = re.compile(r"^act\s*:\s*", re.IGNORECASE)
_PREFIX_RE = re.compile(r"^(\d+[\.\)]\s*|-\s*|>\s*)")


def _build_step_user(task_goal: str, observation: str, admissible_commands: list[str],
                     action_history: list[str], observation_history: list[str],
                     skill_context: str,
                     checked_locations: list[str] | None = None) -> str:
    parts: list[str] = ["Task: %s" % task_goal]
    if skill_context:
        parts.append("\nRelevant experience:\n%s" % skill_context)
    if checked_locations:
        pretty = ", ".join(item.replace("_", " ") for item in checked_locations)
        parts.append("\nStructured search state:\nAlready checked: %s. "
                     "Do not navigate to these receptacles again while searching."
                     % pretty)
    if action_history:
        recent_n = min(len(action_history), 10)
        start = len(action_history) - recent_n
        parts.append("\nRecent actions:")
        for i in range(start, len(action_history)):
            parts.append("  > %s" % action_history[i])
            if i < len(observation_history):
                parts.append("    %s" % observation_history[i][:120])
    parts.append("\nCurrent observation:\n%s" % observation)
    parts.append("\nValid actions (%d):" % len(admissible_commands))
    for cmd in admissible_commands:
        parts.append("  %s" % cmd)
    parts.append("\nChoose ONE action from the valid actions list above:")
    return "\n".join(parts)


def _checked_locations(tracker: _AlfStateTracker) -> list[str]:
    return sorted(set(tracker.meta.get("checked_locations") or []))


def _record_location_inspection(tracker: _AlfStateTracker, action: str,
                                observation: str, *, accepted: bool = True) -> None:
    """把完成观察的位置写入可续跑的结构化状态。"""
    if not accepted or "nothing happens" in str(observation).lower():
        return
    go = re.match(r"go to (.+)$", action, re.IGNORECASE)
    opened = re.match(r"open (.+)$", action, re.IGNORECASE)
    checked = set(_checked_locations(tracker))
    if go:
        location = _norm(go.group(1))
        if (" is closed" not in str(observation).lower()
                and f"agent_at({location})" in tracker.facts):
            checked.add(location)
    if opened:
        checked.add(_norm(opened.group(1)))
    tracker.meta["checked_locations"] = sorted(checked)


def _avoid_checked_location(action: str, admissible: list[str],
                            checked_locations: list[str]) -> str:
    """Acquire 搜索阶段阻止 LLM 在已检查 receptacle 之间循环。"""
    match = re.match(r"go to (.+)$", action, re.IGNORECASE)
    checked = set(checked_locations)
    if not match or _norm(match.group(1)) not in checked:
        return action
    for command in admissible:
        candidate = re.match(r"go to (.+)$", command, re.IGNORECASE)
        if candidate and _norm(candidate.group(1)) not in checked:
            return command
    return action


_FENCE_BLOCK_RE = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)


def _compact_seed_context(seed_context: str, max_chars: int = 900) -> str:
    """轻量注入：去掉 Tool 体围栏块，只保留技能 summary/guideline 文本并限长。

    Tool 模板已在首步全量注入；后续步骤只需高层指引（借鉴 FlowEvo 的
    Layer-2 guideline 压缩策略，~120 token 级别）。
    """
    text = _FENCE_BLOCK_RE.sub("", seed_context or "")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + " …"
    return text


def _parse_single_action(text: str, admissible_commands: list[str]) -> str:
    """Think/Act 输出解析（与 FlowEvo 相同优先级）。"""
    text = text.strip()
    # 1. Act: 行
    for line in text.split("\n"):
        line = line.strip()
        m = _ACT_RE.match(line)
        if m:
            action = _PREFIX_RE.sub("", line[m.end():].strip()).strip()
            for cmd in admissible_commands:
                if cmd.lower() == action.lower():
                    return cmd
            matches = [c for c in admissible_commands if c.lower() in action.lower()]
            if matches:
                return max(matches, key=len)
    # 2. 逐行精确匹配
    for line in text.split("\n"):
        line = line.strip()
        if line.lower().startswith("think"):
            continue
        cleaned = _PREFIX_RE.sub("", line).strip()
        for cmd in admissible_commands:
            if cmd.lower() == cleaned.lower():
                return cmd
    # 3. 全文子串
    lower = text.lower()
    matches = [c for c in admissible_commands if c.lower() in lower]
    if matches:
        return max(matches, key=len)
    # 4. 兜底
    return admissible_commands[0] if admissible_commands else "look"


def _extract_action(text: str, admissible: list[str]) -> str:
    return _parse_single_action(text, admissible)


def _has_cycle(history: list[str], cycle_len: int = 3, repeats: int = 2) -> bool:
    """确定性循环检测（与 FlowEvo executor._has_cycle 相同语义）。

    最近 cycle_len * repeats 步 = repeats 份相同的 cycle_len 元组时返回 True。
    """
    needed = cycle_len * repeats
    if len(history) < needed:
        return False
    window = history[-needed:]
    first = window[:cycle_len]
    for i in range(1, repeats):
        if window[i * cycle_len:(i + 1) * cycle_len] != first:
            return False
    return True


def _parse_action_params(action: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    # go-to 不单独记录 location：由同段内 take/heat/move 的
    # object_location / station / target_location 承担槽位（避免冗余槽）
    take_match = re.match(r"take (.+?) from (.+)", action)
    if take_match:
        params["object"] = take_match.group(1).strip()
        params["object_location"] = take_match.group(2).strip()
    put_match = re.match(r"put (.+?) in/on (.+)", action)
    if put_match:
        params["object"] = put_match.group(1).strip()
        params["target_location"] = put_match.group(2).strip()
    move_match = re.match(r"move (.+?) to (.+)", action)
    if move_match:
        params["object"] = move_match.group(1).strip()
        params["target_location"] = move_match.group(2).strip()
    heat_match = re.match(r"heat (.+?) with (.+)", action)
    if heat_match:
        params["object"] = heat_match.group(1).strip()
        params["heating_station"] = heat_match.group(2).strip()
    clean_match = re.match(r"clean (.+?) with (.+)", action)
    if clean_match:
        params["object"] = clean_match.group(1).strip()
        params["cleaning_station"] = clean_match.group(2).strip()
    cool_match = re.match(r"cool (.+?) with (.+)", action)
    if cool_match:
        params["object"] = cool_match.group(1).strip()
        params["cooling_station"] = cool_match.group(2).strip()
    use_match = re.match(r"use (.+)", action)
    if use_match:
        params["associated_entity"] = use_match.group(1).strip()
    return params


def _replay_env_index(replay_case: dict[str, Any]) -> int | None:
    source = replay_case.get("source") or {}
    index = source.get("env_index")
    if index is None:
        # 尝试从 bindings 元数据恢复
        bindings = replay_case.get("bindings") or {}
        index = bindings.get("__env_index__")
    try:
        return int(index)
    except (TypeError, ValueError):
        return None
