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

_SUPPORTED_TASK_TYPES = {
    "pick_and_place_simple",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "look_at_obj_in_light",
    "pick_two_obj_and_place",
}

_RECEPTACLE_TYPES = {
    "armchair", "bathtub", "bathtubbasin", "bed", "box", "cabinet", "cart",
    "chair", "coffeemachine", "countertop", "desk", "diningtable", "drawer",
    "dresser", "fridge", "garbagecan", "laundryhamper", "microwave", "ottoman",
    "safe", "shelf", "sidetable", "sink", "sinkbasin", "sofa", "stoveburner",
    "toaster", "toilet",
}
_DISCOVERY_SOURCE_PRIORITY = {
    "countertop": 0, "diningtable": 1, "sidetable": 2, "shelf": 3,
    "coffeemachine": 4, "garbagecan": 5, "cabinet": 6, "drawer": 7,
    "fridge": 8, "microwave": 9,
}


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
                                        task_type=wanted_type)
        if wanted_type:
            raw_tasks = [t for t in raw_tasks if t.task_type == wanted_type]
        if limit and limit > 0:
            raw_tasks = raw_tasks[:limit]
        tasks: list[Task] = []
        param_extractor = _get_param_extractor()
        for index, raw in enumerate(raw_tasks):
            task_id = f"alfworld_{index}_{raw.task_type}"
            self._task_indices[task_id] = index
            state = _parse_alfworld_state(raw.initial_observation)
            # 复用 vendored FlowEvo ParamExtractor（§23：goal/task 参数解析）
            params: dict[str, Any] = {}
            if param_extractor is not None and raw.task_type != "pick_two_obj_and_place":
                try:
                    params = param_extractor.extract(
                        str(raw.task_type), str(raw.goal),
                        str(raw.initial_observation), list(raw.initial_admissible))
                    params = {str(k): v for k, v in (params or {}).items()}
                except Exception:  # noqa: BLE001
                    params = {}
            semantic_params = _semantic_goal_params(str(raw.goal), params)
            tasks.append(Task(
                task_id=task_id,
                benchmark="alfworld",
                task_type=str(raw.task_type),
                goal=str(raw.goal),
                context={
                    "kind": "env",
                    "env_index": index,
                    "initial_observation": str(raw.initial_observation),
                    "initial_admissible": list(raw.initial_admissible),
                    "game_file": str(getattr(raw, "game_file", "")),
                    "params": params,
                    # Goal contract keeps a receptacle family (cabinet), while
                    # executable params keep a concrete admissible instance.
                    "semantic_params": semantic_params,
                },
                state=state,
                target_effects=_target_effects_of(raw.task_type),
                metadata={},
            ))
        return tasks

    def parse_task_type(self, task: Task) -> str:
        return task.task_type

    def verify_task(self, task: Task, candidate: str) -> Any:
        raise NotImplementedError("ALFWorld 不使用代码验证")

    # ------------------------------------------------------------------
    def _start_task(self, task: Task):
        """把环境定位到指定任务并返回 (observation, admissible)。"""
        env = self._get_env()
        index = int(task.context.get("env_index", 0))
        _task, obs, admissible = env.reset_to_task(index)
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
            tracker.update(prefix_result.observation)
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
            tracker.update(result.observation)
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
                                 node_ref: str = "", tool_ref: str = ""
                                 ) -> tuple[dict[str, str], EnvRunResult]:
        """为 Acquire Tool 做确定性、有界且不重复的位置发现。

        只执行 admissible 中的 ``go to``/``open`` 探索动作；每个 receptacle
        最多检查一次。找到具体对象实例后返回 object/object_location 绑定，但不
        代替 Acquire 本身，随后仍由标准 Direct Tool 执行 take。
        """
        # 位置发现是代码框架的有界前置步骤，不是 Direct Tool 调用。
        result = EnvRunResult()
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
                if match and _same_object_family(match.group(1), object_name):
                    return {"object": _norm(match.group(1)),
                            "object_location": _norm(match.group(2))}
            return {}

        checked = set(tracker.meta.get("checked_locations") or [])
        queue: list[tuple[str, str]] = []

        def enqueue(commands: list[str], observation: str = "") -> None:
            candidates: dict[str, str] = {}
            for command in commands:
                match = re.match(r"go to (.+)$", command, re.IGNORECASE)
                if not match:
                    continue
                location = _norm(match.group(1))
                candidates.setdefault(location, command)
            # 有些 wrapper 在中途 observation 中不再完整暴露所有 ``go to``；
            # 初始房间描述仍是可信候选来源。只解析已知 receptacle，绝不把物体
            # 实例拼成探索命令。
            text = " ".join((observation,
                             str(task.context.get("initial_observation") or "")))
            for match in re.finditer(r"\b([a-z]+)\s+(\d+)\b", text,
                                     flags=re.IGNORECASE):
                family = match.group(1).lower()
                if family in _RECEPTACLE_TYPES:
                    location = _norm(match.group(0))
                    candidates.setdefault(location,
                                          f"go to {match.group(0).lower()}")
            existing = {item[0] for item in queue}
            additions = [(location, command) for location, command in candidates.items()
                         if location not in checked and location not in existing]
            additions.sort(key=lambda item: _discovery_location_priority(item[0]))
            queue.extend(additions)

        def execute(action: str) -> Any:
            nonlocal obs, admissible
            env_result = self._current_env.step(action)
            obs = env_result.observation
            admissible = list(env_result.admissible_commands)
            result.actions.append({
                "step": len(result.actions), "name": action,
                "params": _parse_action_params(action),
                "observation": obs, "accepted": "nothing happens" not in obs.lower(),
                "mode": "dynamic", "node_ref": node_ref, "tool_ref": "",
                "origin": "framework_discovery",
            })
            tracker.update(obs)
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
                env_result = execute(go_action)
                if env_result.done and not env_result.won:
                    return {}
                open_action = next((command for command in admissible
                                    if re.fullmatch(
                                        rf"open\s+{re.escape(source.replace('_', ' '))}",
                                        command, flags=re.IGNORECASE)), None)
                if open_action:
                    env_result = execute(open_action)
                    if env_result.done and not env_result.won:
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
        inspected = 0
        while queue and inspected < max(0, int(max_locations)):
            location, go_action = queue.pop(0)
            if location in checked:
                continue
            env_result = execute(go_action)
            if env_result.done:
                return finish({})
            # 封闭容器必须打开后才算检查完成。
            open_action = next((cmd for cmd in admissible
                                if re.fullmatch(rf"open\s+{re.escape(location.replace('_', ' '))}",
                                                cmd, re.IGNORECASE)), None)
            if open_action:
                env_result = execute(open_action)
                if env_result.done:
                    return finish({})
            checked.add(location)
            inspected += 1
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
                        "accepted": True,
                        "mode": "direct",
                        "node_ref": step_spec.get("node_ref", ""),
                        "tool_ref": step_spec.get("tool_ref", ""),
                    })
                    tracker.update(env_result.observation)
                    result.states.append({"step": len(result.actions),
                                          "state": tracker.state()})
                    # The action that establishes the final atomic Effect may
                    # simultaneously finish the benchmark.  Preserve the
                    # authoritative ALFWorld signal before returning at the
                    # local Effect boundary.
                    if env_result.won:
                        result.success = True
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
            accepted = "Nothing happens" not in env_result.observation
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
            tracker.update(env_result.observation)
            _record_location_inspection(tracker, action, env_result.observation)
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

    def update(self, observation: str) -> None:
        text = str(observation or "").lower()

        arrive = re.search(r"you arrive at (.+?)\.", text)
        if arrive:
            self.facts = {f for f in self.facts if not f.startswith("agent_at(")}
            self.facts.add(f"agent_at({_norm(arrive.group(1))})")

        def add_objects(phrase: str, loc: str) -> None:
            for obj in _extract_objects(phrase):
                self.facts.add(f"object_exists({obj})")
                self.facts.add(f"object_at({obj}, {_norm(loc)})")

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

        carrying = re.search(r"you are carrying (.+)\.?", text)
        if carrying:
            self.inventory = []
            for obj in _extract_objects(carrying.group(1)):
                if obj != "nothing":
                    self.inventory.append(obj)
                    self.facts.add(f"agent_holds({obj})")

    def state(self) -> dict[str, Any]:
        return {"facts": sorted(self.facts), "inventory": list(self.inventory),
                "text": "", "meta": dict(self.meta)}


def _norm(text: str) -> str:
    """规范化实体名：小写 + 空格转下划线（mug 2 -> mug_2）。"""
    return re.sub(r"\s+", "_", str(text).strip().lower())


def _same_object_family(candidate: str, requested: str) -> bool:
    """``apple`` 与具体实例 ``apple 2``/``apple_2`` 属于同一对象族。"""
    left = re.sub(r"_\d+$", "", _norm(candidate))
    right = re.sub(r"_\d+$", "", _norm(requested))
    return bool(left and left == right)


def _discovery_location_priority(location: str) -> tuple[int, str]:
    """稳定且实例无关的位置搜索顺序；常见开放表面优先。"""
    normalized = _norm(location)
    family = re.sub(r"_\d+$", "", normalized)
    return (_DISCOVERY_SOURCE_PRIORITY.get(family, 50), normalized)


def _semantic_goal_params(goal: str, executable_params: dict[str, Any]) -> dict[str, Any]:
    """Separate goal-level types from concrete executable ALFWorld bindings."""
    semantic = dict(executable_params or {})
    text = str(goal or "").strip().lower()
    target = str(semantic.get("target_location") or "").strip()
    if target:
        normalized = _norm(target)
        family = re.sub(r"_\d+$", "", normalized)
        family_text = family.replace("_", " ")
        explicit_instance = re.search(
            rf"\b{re.escape(family_text)}\s+\d+\b", text)
        if (not explicit_instance
                and re.search(rf"\b{re.escape(family_text)}\b", text)):
            semantic["target_location"] = family_text
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
    queue: list[tuple[str, str]] = []

    def enqueue(items: list[str]) -> None:
        candidates: list[tuple[str, str]] = []
        for command in items:
            match = re.fullmatch(r"go to (.+)", command, flags=re.IGNORECASE)
            if match:
                location = _norm(match.group(1))
                if location not in checked and all(location != item[0] for item in queue):
                    candidates.append((location, command))
        candidates.sort(key=lambda item: (item[0] != preferred, item[0]))
        queue.extend(candidates)

    enqueue(commands)
    while queue and len(checked) < max(0, int(max_locations)):
        location, go_action = queue.pop(0)
        if location in checked:
            continue
        step_result = env.step(go_action)
        tracker.update(step_result.observation)
        commands = list(step_result.admissible_commands)
        if step_result.done and not step_result.won:
            return resolved, commands, False
        open_action = next((command for command in commands
                            if re.fullmatch(
                                rf"open\s+{re.escape(location.replace('_', ' '))}",
                                command, flags=re.IGNORECASE)), None)
        if open_action:
            step_result = env.step(open_action)
            tracker.update(step_result.observation)
            commands = list(step_result.admissible_commands)
            if step_result.done and not step_result.won:
                return resolved, commands, False
        checked.add(location)
        found = take_binding(commands)
        if found:
            resolved.update(found)
            return resolved, commands, True
        enqueue(commands)
    return resolved, commands, False


def _target_effects_of(task_type: str) -> list[dict[str, Any]]:
    mapping = {
        "pick_heat_then_place_in_recep": [
            {"predicate": "object.heated", "args": {"object": "$object"}},
            {"predicate": "object.at_location", "args": {"object": "$object", "location": "$target_location"}},
        ],
        "pick_clean_then_place_in_recep": [
            {"predicate": "object.cleaned", "args": {"object": "$object"}},
            {"predicate": "object.at_location", "args": {"object": "$object", "location": "$target_location"}},
        ],
        "pick_cool_then_place_in_recep": [
            {"predicate": "object.cooled", "args": {"object": "$object"}},
            {"predicate": "object.at_location", "args": {"object": "$object", "location": "$target_location"}},
        ],
        "look_at_obj_in_light": [
            {"predicate": "object.lit", "args": {"object": "$object"}},
        ],
    }
    return mapping.get(task_type, [])


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
# ReAct prompt（与 vendored FlowEvo alfworld_/generator.py 对齐，§57.1 公平对照）
# ---------------------------------------------------------------------------

_ALFWORLD_SYSTEM_PROMPT = (
    "You are an expert household robot completing tasks in a virtual home. "
    "You will be given a task goal, the current observation, and a list of valid actions.\n\n"
    "At each step:\n"
    "1. Think about what you need to do next and why.\n"
    "2. Choose exactly ONE action from the valid actions list.\n\n"
    "Common task patterns:\n"
    "- pick_and_place: go to object location -> take it -> go to destination -> put it\n"
    "- pick_clean_then_place: go to object -> take -> go to sinkbasin -> clean -> go to dest -> put\n"
    "- pick_heat_then_place: go to object -> take -> go to microwave -> heat -> go to dest -> put\n"
    "- pick_cool_then_place: go to object -> take -> go to fridge -> cool -> go to dest -> put\n"
    "- examine_in_light: go to object -> take -> go to lamp -> use lamp\n"
    "- pick_two: find first object -> take -> go to dest -> put -> find second -> take -> go to dest -> put\n\n"
    "Format your response as:\n"
    "Think: <your step-by-step reasoning>\n"
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
                                observation: str) -> None:
    """把完成观察的位置写入可续跑的结构化状态。"""
    go = re.match(r"go to (.+)$", action, re.IGNORECASE)
    opened = re.match(r"open (.+)$", action, re.IGNORECASE)
    checked = set(_checked_locations(tracker))
    if go and " is closed" not in str(observation).lower():
        checked.add(_norm(go.group(1)))
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


def _get_param_extractor():
    """复用 vendored FlowEvo ParamExtractor（§23：goal/task 参数解析）。"""
    try:
        ensure_flowevo_path()
        from alfworld_.param_extractor import ParamExtractor
        return ParamExtractor()
    except Exception:  # noqa: BLE001
        return None
