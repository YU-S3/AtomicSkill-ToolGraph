"""Toy Benchmarks：无 API smoke 用的合成任务 + 微型文本世界。

- toy_code / toy_math：合成代码与算术任务（mock LLM 或真实 LLM 均可驱动）
- toy_env：ALFWorld 风格的微型文本环境（与 vendored FlowEvo 的 duck-typing
  协议兼容：initialize/reset/step(won 唯一成功信号)）
覆盖：cold start 学习、warm start 复用、direct/seeded/dynamic 路由、
Tool admission、泛化（double/triple → 参数化）、Composite 与 Layer-3。
"""

from __future__ import annotations

import re
from typing import Any

from ..core.llm import LLM
from ..core.predicates import StateSnapshot, check_effects
from ..tools.sandbox import Sandbox
from .benchmark import (
    BenchmarkAdapter,
    CodeAttempt,
    CodeRunResult,
    EnvRunResult,
    EnvStepResult,
    Task,
    VerifyResult,
)

# ---------------------------------------------------------------------------
# 任务定义
# ---------------------------------------------------------------------------

TOY_CODE_TASKS: list[dict[str, Any]] = [
    {"task_id": "toy_code_double", "task_type": "toy_code_arithmetic",
     "goal": "Write a function solve(x) that returns 2*x.",
     "entry": "solve", "tests": ["assert solve(3) == 6", "assert solve(5) == 10"],
     "code": "def solve(x):\n    return 2 * x\n"},
    {"task_id": "toy_code_triple", "task_type": "toy_code_arithmetic",
     "goal": "Write a function solve(x) that returns 3*x.",
     "entry": "solve", "tests": ["assert solve(3) == 9", "assert solve(4) == 12"],
     "code": "def solve(x):\n    return 3 * x\n"},
    {"task_id": "toy_code_double_variant", "task_type": "toy_code_arithmetic",
     "goal": "Implement solve(x) returning twice the input.",
     "entry": "solve", "tests": ["assert solve(6) == 12", "assert solve(7) == 14"],
     "code": "def solve(x):\n    return 2 * x\n"},
    {"task_id": "toy_code_square", "task_type": "toy_code_arithmetic",
     "goal": "Write a function solve(x) that returns the square of x.",
     "entry": "solve", "tests": ["assert solve(4) == 16"],
     "code": "def solve(x):\n    return x * x\n"},
]

TOY_MATH_TASKS: list[dict[str, Any]] = [
    {"task_id": "toy_math_add", "task_type": "toy_math_arithmetic",
     "goal": "What is 2+3?", "answer": "5"},
    {"task_id": "toy_math_mul", "task_type": "toy_math_arithmetic",
     "goal": "What is 4*5?", "answer": "20"},
    {"task_id": "toy_math_div", "task_type": "toy_math_arithmetic",
     "goal": "What is 21/3?", "answer": "7"},
]

# 微型文本世界任务（goal 谓词检查；context.params 供参数绑定/检索）
TOY_ENV_TASKS: list[dict[str, Any]] = [
    {"task_id": "toy_env_pick_place", "task_type": "pick_and_place_simple",
     "goal": "Put a mug on the shelf.",
     "checks": ["object_at(mug_1, shelf_1)"],
     "objects": {"mug_1": "countertop_1"},
     "params": {"object": "mug 1", "object_location": "countertop 1",
                "target_location": "shelf 1"}},
    {"task_id": "toy_env_heat", "task_type": "pick_heat_then_place_in_recep",
     "goal": "Heat an egg and put it in the fridge.",
     "checks": ["object_heated(egg_1)", "object_at(egg_1, fridge_1)"],
     "objects": {"egg_1": "countertop_1"},
     "params": {"object": "egg 1", "object_location": "countertop 1",
                "heating_station": "microwave 1", "target_location": "fridge 1"}},
    {"task_id": "toy_env_clean", "task_type": "pick_clean_then_place_in_recep",
     "goal": "Clean an apple and put it in the fridge.",
     "checks": ["object_cleaned(apple_1)", "object_at(apple_1, fridge_1)"],
     "objects": {"apple_1": "countertop_1"},
     "params": {"object": "apple 1", "object_location": "countertop 1",
                "cleaning_station": "sinkbasin 1", "target_location": "fridge 1"}},
    {"task_id": "toy_env_heat2", "task_type": "pick_heat_then_place_in_recep",
     "goal": "Heat a potato and put it in the fridge.",
     "checks": ["object_heated(potato_1)", "object_at(potato_1, fridge_1)"],
     "objects": {"potato_1": "countertop_1"},
     "params": {"object": "potato 1", "object_location": "countertop 1",
                "heating_station": "microwave 1", "target_location": "fridge 1"}},
    {"task_id": "toy_env_heat3", "task_type": "pick_heat_then_place_in_recep",
     "goal": "Heat a tomato and put it in the fridge.",
     "checks": ["object_heated(tomato_1)", "object_at(tomato_1, fridge_1)"],
     "objects": {"tomato_1": "countertop_1"},
     "params": {"object": "tomato 1", "object_location": "countertop 1",
                "heating_station": "microwave 1", "target_location": "fridge 1"}},
]

# MockLLM 脚本：任务 id → 正确解
TOY_MOCK_SCRIPT: dict[str, dict[str, Any]] = {
    "toy_code_double": {"code": "def solve(x):\n    return 2 * x\n"},
    "toy_code_triple": {"code": "def solve(x):\n    return 3 * x\n"},
    "toy_code_double_variant": {"code": "def solve(x):\n    return 2 * x\n"},
    "toy_code_square": {"code": "def solve(x):\n    return x * x\n"},
    "toy_math_add": {"answer": "5"},
    "toy_math_mul": {"answer": "20"},
    "toy_math_div": {"answer": "7"},
    "toy_env_pick_place": {"actions": [
        "go to countertop 1", "take mug 1 from countertop 1",
        "go to shelf 1", "put mug 1 in/on shelf 1"]},
    "toy_env_heat": {"actions": [
        "go to countertop 1", "take egg 1 from countertop 1",
        "go to microwave 1", "heat egg 1 with microwave 1",
        "go to fridge 1", "put egg 1 in/on fridge 1"]},
    "toy_env_clean": {"actions": [
        "go to countertop 1", "take apple 1 from countertop 1",
        "go to sinkbasin 1", "clean apple 1 with sinkbasin 1",
        "go to fridge 1", "put apple 1 in/on fridge 1"]},
    "toy_env_heat2": {"actions": [
        "go to countertop 1", "take potato 1 from countertop 1",
        "go to microwave 1", "heat potato 1 with microwave 1",
        "go to fridge 1", "put potato 1 in/on fridge 1"]},
    "toy_env_heat3": {"actions": [
        "go to countertop 1", "take tomato 1 from countertop 1",
        "go to microwave 1", "heat tomato 1 with microwave 1",
        "go to fridge 1", "put tomato 1 in/on fridge 1"]},
}


def toy_tasks(kinds: tuple[str, ...] = ("code", "math", "env")) -> list[Task]:
    tasks: list[Task] = []
    if "code" in kinds:
        for spec in TOY_CODE_TASKS:
            tasks.append(Task(
                task_id=spec["task_id"], benchmark="toy_code",
                task_type=spec["task_type"], goal=spec["goal"],
                context={"entry": spec["entry"], "tests": spec["tests"],
                         "kind": "code"},
                state={"facts": [], "text": spec["goal"]},
                target_effects=[{"predicate": "callable.returns_expected",
                                 "args": {"entry_point": spec["entry"]}}],
                metadata={"solution": spec["code"]},
            ))
    if "math" in kinds:
        for spec in TOY_MATH_TASKS:
            tasks.append(Task(
                task_id=spec["task_id"], benchmark="toy_math",
                task_type=spec["task_type"], goal=spec["goal"],
                context={"entry": "solve", "tests": [],
                         "kind": "math", "answer": spec["answer"]},
                state={"facts": [], "text": spec["goal"]},
                target_effects=[{"predicate": "answer.correct"}],
                metadata={"solution": f"def solve():\n    return {spec['answer']!r}\n"},
            ))
    if "env" in kinds:
        for spec in TOY_ENV_TASKS:
            initial_world = ToyWorld(spec)
            tasks.append(Task(
                task_id=spec["task_id"], benchmark="toy_env",
                task_type=spec["task_type"], goal=spec["goal"],
                context={"kind": "env", "checks": spec["checks"],
                         "objects": spec["objects"],
                         "params": spec.get("params", {})},
                state=initial_world.state(),
                target_effects=[],
                metadata={},
            ))
    return tasks


def build_mock_script() -> dict[str, dict[str, Any]]:
    return {k: dict(v) for k, v in TOY_MOCK_SCRIPT.items()}


# ---------------------------------------------------------------------------
# 微型文本世界
# ---------------------------------------------------------------------------

_LOCATIONS = ["countertop_1", "shelf_1", "fridge_1", "microwave_1",
              "sinkbasin_1", "cabinet_1"]


class ToyWorld:
    """ALFWorld 风格的微型确定性文本世界。

    内部状态统一使用下划线命名（countertop_1 / mug_1）；动作文本与观察
    使用空格形式（go to countertop 1），由解析函数双向归一。
    """

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.reset()

    def reset(self) -> None:
        self.position = "countertop_1"
        self.inventory: list[str] = []
        self.objects: dict[str, str] = {
            _norm(obj): _resolve_location(loc)
            for obj, loc in (self.spec.get("objects") or {}).items()
        }
        self.attrs: dict[str, set[str]] = {}      # object -> {heated, cleaned}
        self.checked: set[str] = set()
        self.steps = 0
        self.done = False
        self.won = False

    def state(self) -> dict[str, Any]:
        facts: list[str] = [f"agent_at({self.position})"]
        for obj, loc in self.objects.items():
            facts.append(f"object_at({obj}, {loc})")
        for obj in self.inventory:
            facts.append(f"agent_holds({obj})")
        for obj, attrs in self.attrs.items():
            for attr in attrs:
                facts.append(f"object_{attr}({obj})")
        for loc in self.checked:
            facts.append(f"location_checked({loc})")
        return {"facts": sorted(facts), "inventory": list(self.inventory),
                "text": self.observation(), "meta": {}}

    def observation(self) -> str:
        pretty = lambda text: str(text).replace("_", " ")  # noqa: E731
        here = [pretty(obj) for obj, loc in self.objects.items() if loc == self.position]
        if here:
            return (f"You are in {pretty(self.position)}. You see: {', '.join(here)}. "
                    f"Inventory: {', '.join(pretty(o) for o in self.inventory) or 'empty'}.")
        return (f"You are in {pretty(self.position)}. Nothing here. "
                f"Inventory: {', '.join(pretty(o) for o in self.inventory) or 'empty'}.")

    def admissible(self) -> list[str]:
        pretty = lambda text: str(text).replace("_", " ")  # noqa: E731
        commands = [f"go to {pretty(loc)}" for loc in _LOCATIONS if loc != self.position]
        for obj, loc in self.objects.items():
            if loc == self.position and obj not in self.inventory:
                commands.append(f"take {pretty(obj)} from {pretty(loc)}")
        for obj in self.inventory:
            for loc in _LOCATIONS:
                commands.append(f"put {pretty(obj)} in/on {pretty(loc)}")
            commands.append(f"heat {pretty(obj)} with microwave 1")
            commands.append(f"clean {pretty(obj)} with sinkbasin 1")
        return commands

    def step(self, action: str) -> EnvStepResult:
        self.steps += 1
        action = str(action).strip()
        self.checked.add(self.position)
        if self.steps > 30:
            self.done = True
            return EnvStepResult(observation=self.observation() + " [timeout]",
                                 done=True, won=False,
                                 admissible_commands=self.admissible(),
                                 state=self.state(), accepted=False)
        moved = self._apply(action)
        self.won = self._goal_reached()
        if self.won:
            self.done = True
        result = EnvStepResult(
            observation=self.observation(),
            score=1.0 if self.won else 0.0,
            done=self.done,
            won=self.won,
            admissible_commands=self.admissible(),
            state=self.state(),
            accepted=moved,
        )
        if not moved:
            result.observation = "Nothing happens. " + result.observation
        return result

    def _apply(self, action: str) -> bool:
        go_match = re.match(r"go to (.+)", action)
        if go_match:
            loc = _resolve_location(go_match.group(1).strip())
            if loc in _LOCATIONS:
                self.position = loc
                return True
            return False
        take_match = re.match(r"take (.+?) from (.+)", action)
        if take_match:
            obj = _norm(take_match.group(1).strip())
            loc = _resolve_location(take_match.group(2).strip())
            if obj in self.objects and self.objects[obj] == loc \
                    and obj not in self.inventory and self.position == loc:
                self.inventory.append(obj)
                self.objects.pop(obj)
                return True
            return False
        put_match = re.match(r"put (.+?) in/on (.+)", action)
        if put_match:
            obj = _norm(put_match.group(1).strip())
            loc = _resolve_location(put_match.group(2).strip())
            if obj in self.inventory and loc in _LOCATIONS and self.position == loc:
                self.inventory.remove(obj)
                self.objects[obj] = loc
                return True
            return False
        heat_match = re.match(r"heat (.+?) with (.+)", action)
        if heat_match:
            obj = _norm(heat_match.group(1).strip())
            if obj in self.inventory and self.position == "microwave_1":
                self.attrs.setdefault(obj, set()).add("heated")
                return True
            return False
        clean_match = re.match(r"clean (.+?) with (.+)", action)
        if clean_match:
            obj = _norm(clean_match.group(1).strip())
            if obj in self.inventory and self.position == "sinkbasin_1":
                self.attrs.setdefault(obj, set()).add("cleaned")
                return True
            return False
        return False

    def _goal_reached(self) -> bool:
        checks = self.spec.get("checks") or []
        facts = self.state()["facts"]
        return all(any(_check_matches(check, fact) for fact in facts)
                   for check in checks)


def _check_matches(check: str, fact: str) -> bool:
    return _norm(check).replace("_", "") == _norm(fact).replace("_", "")


def _resolve_location(text: str) -> str:
    norm = _norm(text)
    for loc in _LOCATIONS:
        if _norm(loc) == norm or _norm(loc).replace("_", "") == norm.replace("_", ""):
            return loc
    return norm


def _norm(text: str) -> str:
    return re.sub(r"\s+", "_", str(text).strip().lower())


# ---------------------------------------------------------------------------
# Toy Adapter
# ---------------------------------------------------------------------------

class ToyAdapter:
    """实现 BenchmarkAdapter 协议的合成适配器（stage-2 smoke 与无 API 联调）。"""

    name = "toy"
    supports_in_place_resume = True

    def __init__(self, kinds: tuple[str, ...] = ("code", "math", "env"),
                 sandbox: Sandbox | None = None) -> None:
        self.kinds = kinds
        self.sandbox = sandbox or Sandbox()
        self.world: ToyWorld | None = None

    def load_tasks(self, limit: int = 0, task_type: str | None = None) -> list[Task]:
        tasks = toy_tasks(self.kinds)
        if task_type:
            tasks = [t for t in tasks if t.task_type == task_type]
        if limit and limit > 0:
            tasks = tasks[:limit]
        return tasks

    def parse_task_type(self, task: Task) -> str:
        return task.task_type

    # -- 验证 ----------------------------------------------------------------
    def verify_task(self, task: Task, candidate: str) -> VerifyResult:
        kind = task.context.get("kind", "code")
        if kind == "math":
            return self._verify_math(task, candidate)
        return self._verify_code(task, candidate)

    def _verify_code(self, task: Task, candidate: str) -> VerifyResult:
        tests = list(task.context.get("tests") or [])
        run = self.sandbox.run_tests(candidate, tests)
        result = VerifyResult(
            passed=run["passed"],
            tests=tests,
            executed_code=candidate,
            stdout=run.get("stdout", ""),
            stderr=run.get("stderr", ""),
            timeout=run.get("timeout", False),
        )
        result.failure_type = "test_failure" if not run["passed"] else ""
        return result

    def _verify_math(self, task: Task, candidate: str) -> VerifyResult:
        expected = str(task.context.get("answer", ""))
        # 执行 candidate 中的 solve() 并比对
        harness = candidate.rstrip() + "\n\nprint(solve())\n"
        run = self.sandbox.run(harness)
        actual = (run.get("stdout") or "").strip().splitlines()
        actual_answer = actual[-1].strip() if actual else ""
        passed = run["passed"] and actual_answer == expected
        return VerifyResult(
            passed=passed,
            tests=[f"assert solve() == {expected!r}"],
            executed_code=candidate,
            stdout=run.get("stdout", ""),
            stderr=run.get("stderr", ""),
            timeout=run.get("timeout", False),
            failure_type="" if passed else ("timeout" if run.get("timeout") else "test_failure"),
        )

    # -- Tool 执行 / replay ---------------------------------------------------
    def execute_tool(self, task: Task, tool, parameters: dict[str, Any],
                     state: dict[str, Any]) -> dict[str, Any]:
        from ..core.status import ArtifactKind
        if tool.artifact_kind == ArtifactKind.ACTION_TEMPLATE:
            return self._execute_template(tool, parameters)
        candidate = tool.artifact_body()
        verify = self.verify_task(task, candidate)
        return {
            "passed": verify.passed,
            "after": {"facts": _success_facts(task) if verify.passed else [], "text": ""},
            "observation": verify.stdout or verify.stderr,
            "feedback": verify.to_dict(),
        }

    def replay_tool(self, tool, bindings: dict[str, Any],
                    before: dict[str, Any]) -> dict[str, Any]:
        from ..core.status import ArtifactKind
        if tool.artifact_kind == ArtifactKind.ACTION_TEMPLATE:
            return self._replay_template(tool, bindings, before)
        # 代码 tool：运行其 replay 测试
        test_cases: list[str] = []
        for case in tool.replay_cases():
            if case.get("kind") in ("parameterized_replay",):
                continue
            test_cases.extend(case.get("tests") or [])
        run = self.sandbox.run_tests(tool.artifact_body(), test_cases)
        return {"passed": run["passed"], "after": {}, "reason":
                f"{run['passed_count']}/{run['total']}" if test_cases else "no_tests"}

    def _replay_template(self, tool, bindings: dict[str, Any],
                         before: dict[str, Any]) -> dict[str, Any]:
        """从 replay case 的 before/after 重建来源世界并重放（source trace replay）。"""
        case = next((t for t in (tool.tests or []) if t.get("kind") == "replay"), None)
        if case is None:
            return {"passed": False, "after": {}, "reason": "no_replay_case"}
        before_state = dict(case.get("before") or before or {})
        after_state = dict(case.get("after") or {})
        world = ToyWorld({"task_id": "replay", "goal": "", "checks": [],
                          "objects": {}})
        _restore_world(world, before_state)
        checks = _facts_to_checks((after_state.get("facts") or []))
        world.spec["checks"] = checks
        steps = list(tool.artifact.get("steps") or [])
        merged_bindings = dict(bindings)
        merged_bindings.update(case.get("bindings") or {})
        for step in steps:
            filled = _fill_template(step, merged_bindings)
            result = world.step(filled)
            if result.done and not result.won:
                return {"passed": False, "after": world.state(), "reason": "replay_env_failed"}
        expected_effects = [dict(item) for item in (case.get("expected_effects") or [])
                            if isinstance(item, dict)]
        if expected_effects:
            passed, missing = check_effects(
                StateSnapshot(world.state()), merged_bindings, expected_effects)
            return {"passed": passed, "after": world.state(),
                    "reason": ("replay_effect_covered" if passed else
                               f"replay_effect_missing: {missing}")}
        return {"passed": world.won, "after": world.state(),
                "reason": "replay_won" if world.won else "replay_goal_missed"}

    def _execute_template(self, tool, parameters: dict[str, Any]) -> dict[str, Any]:
        steps = list(tool.artifact.get("steps") or [])
        world_spec = _world_from_bindings(parameters)
        if world_spec is None:
            return {"passed": False, "after": {}, "reason": "bindings_incomplete"}
        world = ToyWorld(world_spec)
        for step in steps:
            filled = step.format(**{k: str(v) for k, v in parameters.items()})
            result = world.step(filled)
            if result.done and not result.won:
                return {"passed": False, "after": world.state(), "reason": "env_failed"}
        return {"passed": world.won, "after": world.state(),
                "reason": "ok" if world.won else "goal_not_reached"}

    # -- 生成 ----------------------------------------------------------------
    def generate_code(self, task: Task, llm: LLM, *,
                      seed_context: str = "", max_repairs: int = 2) -> CodeRunResult:
        result = CodeRunResult()
        instructions = (
            "You are an expert Python programmer. Output ONLY raw Python code "
            "(no markdown fences, no explanation) that satisfies the task."
        )
        prompt = task.goal
        if task.context.get("kind") == "math":
            prompt = (f"{task.goal}\nWrite raw Python only: implement `def solve():` "
                      f"that returns the final numeric answer.")
        prompt = f"[task_id: {task.task_id}]\n{prompt}"
        if seed_context:
            prompt = f"{seed_context}\n\nNow solve this task:\n{prompt}"
        for index in range(max_repairs + 1):
            response = llm.generate(instructions=instructions, input_text=prompt)
            code = _extract_code(response.text)
            verify = self.verify_task(task, code)
            attempt = CodeAttempt(index=index,
                                  stage="draft" if index == 0 else "repair",
                                  code=code, verify=verify,
                                  usage=llm.usage)
            result.attempts.append(attempt)
            if verify.passed:
                result.success = True
                result.candidate_code = code
                result.feedback = verify
                result.first_attempt_success = index == 0
                result.retry_count = index
                return result
            prompt = (f"{prompt}\n\nPrevious attempt failed: {verify.stderr or verify.failure_type}\n"
                      f"Fix the code and output ONLY the corrected Python.")
        result.failure_type = "max_repairs_exceeded"
        if result.attempts:
            result.candidate_code = result.attempts[-1].code
            result.feedback = result.attempts[-1].verify
            result.retry_count = len(result.attempts) - 1
        return result

    def run_env_episode(self, task: Task, llm: LLM, *,
                        seed_context: str = "",
                        direct_steps: list[dict[str, Any]] | None = None,
                        max_steps: int = 30,
                        resume: dict[str, Any] | None = None,
                        stop_effects: list[dict[str, Any]] | None = None,
                        effect_inputs: dict[str, Any] | None = None,
                        node_ref: str = "",
                        phase_goal: str = "") -> EnvRunResult:
        result = EnvRunResult()
        if resume and getattr(self, "_current_world", None) is not None:
            world = self._current_world
            result.actions = [dict(item) for item in (resume.get("actions") or [])]
            result.states = [dict(item) for item in (resume.get("states") or [])]
        else:
            world = ToyWorld(dict(task.context))
            self._current_world = world
            result.states.append({"step": 0, "state": world.state()})
        result.current_observation = world.observation()
        result.current_admissible = world.admissible()
        if _toy_effects_met(world.state(), stop_effects, effect_inputs):
            result.atomic_complete = True
            result.final_observation = world.observation()
            result.steps = len(result.actions)
            return result

        if direct_steps:
            result.direct_used = True
            for index, step_spec in enumerate(direct_steps):
                for step in step_spec.get("steps") or []:
                    template, params = _split_step(step, step_spec.get("params") or {})
                    filled = _fill_template(template, params)
                    env_result = world.step(filled)
                    result.actions.append({
                        "step": len(result.actions),
                        "name": filled,
                        "params": params,
                        "observation": env_result.observation,
                        "accepted": env_result.accepted,
                        "mode": "direct",
                        "node_ref": step_spec.get("node_ref", ""),
                        "tool_ref": step_spec.get("tool_ref", ""),
                    })
                    result.states.append({"step": world.steps, "state": world.state()})
                    result.current_observation = world.observation()
                    result.current_admissible = world.admissible()
                    if _toy_effects_met(world.state(), stop_effects, effect_inputs):
                        result.atomic_complete = True
                        result.steps = len(result.actions)
                        result.final_observation = env_result.observation
                        return result
                    if env_result.done:
                        result.success = env_result.won
                        result.steps = world.steps
                        result.final_observation = env_result.observation
                        result.failure_type = "" if env_result.won else "direct_template_failed"
                        return result
            result.success = world.won
            result.steps = world.steps
            result.final_observation = world.observation()
            result.failure_type = "" if world.won else "direct_template_goal_missed"
            return result

        # seeded / dynamic LLM 循环
        result.seeded_used = bool(seed_context)
        result.dynamic_used = not seed_context
        instructions = (
            "You are an agent in a text world. Choose the single best action from "
            "the admissible commands. Output ONLY the action text."
        )
        history: list[str] = []
        active_goal = phase_goal or task.goal
        prompt = f"[task_id: {task.task_id}]\n"
        prompt += (f"Your task is to: {active_goal}\n" if not seed_context else
                   f"{seed_context}\n\nYour task is to: {active_goal}\n")
        remaining_steps = max(0, int(max_steps) - len(result.actions))
        for _ in range(remaining_steps):
            admissible = world.admissible()
            prompt += (f"Observation: {world.observation()}\n"
                       f"Admissible: {', '.join(admissible[:12])}\nAction:")
            response = llm.generate(instructions=instructions, input_text=prompt)
            action = _extract_action(response.text, admissible)
            env_result = world.step(action)
            result.actions.append({
                "step": len(result.actions),
                "name": action,
                "params": _parse_action_params(action),
                "observation": env_result.observation,
                "accepted": env_result.accepted,
                "mode": "seeded" if seed_context else "dynamic",
                "node_ref": node_ref,
                "tool_ref": "",
            })
            result.states.append({"step": world.steps, "state": world.state()})
            result.current_observation = world.observation()
            result.current_admissible = world.admissible()
            if _toy_effects_met(world.state(), stop_effects, effect_inputs):
                result.atomic_complete = True
                result.steps = len(result.actions)
                result.final_observation = env_result.observation
                return result
            history.append(f"Action: {action}\nObservation: {env_result.observation}")
            if env_result.done:
                result.success = env_result.won
                result.steps = world.steps
                result.final_observation = env_result.observation
                result.failure_type = "" if env_result.won else "timeout_or_stuck"
                return result
            if not env_result.accepted and len(result.actions) >= 3 \
                    and all(not a["accepted"] for a in result.actions[-3:]):
                result.failure_type = "nothing_happens"
                break
            prompt += f"\nAction: {action}\nObservation: {env_result.observation}\n"
        result.success = False
        result.steps = world.steps
        result.failure_type = result.failure_type or "max_steps"
        result.final_observation = world.observation()
        return result


def _toy_effects_met(state: dict[str, Any], effects: list[dict[str, Any]] | None,
                     inputs: dict[str, Any] | None) -> bool:
    if not effects:
        return False
    passed, _missing = check_effects(StateSnapshot(state), inputs or {}, effects,
                                     {"harness": "env"})
    return passed


def _extract_code(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text


def _extract_action(text: str, admissible: list[str]) -> str:
    candidate = text.strip().splitlines()[0].strip()
    for command in admissible:
        if candidate.lower() in command.lower() or command.lower() in candidate.lower():
            return command
    # 无匹配时原样执行（ToyWorld 会返回 accepted=False，触发 nothing-happens 检测）
    return candidate or (admissible[0] if admissible else "look")


def _parse_action_params(action: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    # go-to 的 location 不单独记录：由同段内 take/put/heat/clean 的
    # object_location / target_location / station 参数承担槽位（避免冗余槽）
    take_match = re.match(r"take (.+?) from (.+)", action)
    if take_match:
        params["object"] = take_match.group(1).strip()
        params["object_location"] = take_match.group(2).strip()
    put_match = re.match(r"put (.+?) in/on (.+)", action)
    if put_match:
        params["object"] = put_match.group(1).strip()
        params["target_location"] = put_match.group(2).strip()
    heat_match = re.match(r"heat (.+?) with (.+)", action)
    if heat_match:
        params["object"] = heat_match.group(1).strip()
        params["heating_station"] = heat_match.group(2).strip()
    clean_match = re.match(r"clean (.+?) with (.+)", action)
    if clean_match:
        params["object"] = clean_match.group(1).strip()
        params["cleaning_station"] = clean_match.group(2).strip()
    return params


def _success_facts(task: Task) -> list[str]:
    entry = str(task.context.get("entry", "solve"))
    return [f"callable_returns_expected({entry})"]


def _world_from_bindings(bindings: dict[str, Any]) -> dict[str, Any] | None:
    obj = str(bindings.get("object", "")).strip()
    if not obj:
        return None
    return {
        "task_id": "replay",
        "goal": "",
        "checks": [f"agent_holds({obj})"],
        "objects": {obj: "countertop 1"},
    }


def _restore_world(world: "ToyWorld", before_state: dict[str, Any]) -> None:
    """从 before 事实恢复 ToyWorld 状态（source trace replay 用）。"""
    facts = before_state.get("facts") or []
    for fact in facts:
        hold = re.match(r"agent_holds\(([^)]+)\)", str(fact))
        if hold:
            obj = _norm(hold.group(1))
            if obj not in world.inventory:
                world.inventory.append(obj)
            continue
        at = re.match(r"object_at\(([^,]+),\s*([^)]+)\)", str(fact))
        if at:
            world.objects[_norm(at.group(1))] = _resolve_location(at.group(2))
            continue
        pos = re.match(r"agent_at\(([^)]+)\)", str(fact))
        if pos:
            world.position = _resolve_location(pos.group(1))
            continue
        attr = re.match(r"object_(heated|cleaned|cooled|lit)\(([^)]+)\)", str(fact))
        if attr:
            obj = _norm(attr.group(2))
            world.attrs.setdefault(obj, set()).add(attr.group(1))
    world.inventory = [o for o in world.inventory if o not in world.objects]


def _facts_to_checks(facts: list[str]) -> list[str]:
    """after 事实 → goal 检查（只保留 ToyWorld 可产生的事实类型）。"""
    checks: list[str] = []
    for fact in facts:
        text = str(fact)
        if text.startswith(("object_at(", "agent_holds(", "object_heated(",
                            "object_cleaned(")):
            checks.append(text)
    return checks


def _denorm(text: str) -> str:
    return str(text).strip().replace("_", " ")


def _fill_template(step: str, params: dict[str, Any]) -> str:
    try:
        return step.format(**{k: str(v).replace("_", " ") for k, v in params.items()})
    except KeyError:
        return step


def _split_step(step: Any, default_params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """step 条目可为字符串或 {template, params}。"""
    if isinstance(step, dict):
        return str(step.get("template", "")), dict(step.get("params") or default_params)
    return str(step), dict(default_params)
