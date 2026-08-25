"""Mock LLM：无 API 阶段（stage-2 smoke）的确定性替身。

实现与真实 LLM 相同的协议（core.llm.LLM）。响应按提示词特征规则生成：
- 计划类提示 → 固定 JSON 计划
- 代码类提示 → 从内置任务脚本表返回已知正确解（toy 任务）
- 环境类提示 → 脚本化动作序列
- 泛化/其他 → 保守占位
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from ..core.llm import LLM, LLMResponse, LLMUsage


class MockLLM:
    """规则驱动的确定性 LLM 替身。

    script: {task_key -> {"code": str | None, "answer": str | None, "actions": [...]}}
    按 prompt 中的任务 id / goal 特征匹配脚本；未命中时返回保守占位。
    """

    def __init__(self, script: dict[str, dict[str, Any]] | None = None) -> None:
        self.script = script or {}
        self._usage = LLMUsage()

    @property
    def usage(self) -> LLMUsage:
        return self._usage

    def reset_usage(self) -> None:
        self._usage = LLMUsage()

    def fork(self, **_kwargs) -> "MockLLM":
        """Mirror the real provider's independent-agent interface for tests."""
        return MockLLM(script=self.script)

    # ------------------------------------------------------------------
    def generate(self, instructions: str, input_text: str,
                 *, temperature: float | None = None,
                 thinking: str | None = None,
                 structured_output: bool = False) -> LLMResponse:
        text = self._respond(instructions, input_text)
        tokens = max(1, len(text) // 4)
        prompt_tokens = max(1, len(instructions) // 4 + len(input_text) // 4)
        response = LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=tokens,
            total_tokens=prompt_tokens + tokens,
            latency_ms=1.0,
            provider="mock",
            model="mock-llm",
        )
        self._usage.add(response)
        return response

    # ------------------------------------------------------------------
    def _respond(self, instructions: str, input_text: str) -> str:
        combined = (instructions or "") + "\n" + (input_text or "")

        # 1. 计划类（Atomic Planner）
        if "atomic task planner" in combined.lower() or '"skills"' in combined:
            return self._plan_response(combined)

        # 2. 环境动作类（先于代码生成判断：seed 上下文可能含 python 代码）
        if _looks_like_env_step(combined):
            for key, entry in self.script.items():
                if _task_match(key, combined):
                    actions = entry.get("actions") or []
                    if actions:
                        return _pick_next_action(actions, combined)
            return "look"

        # 3. 代码生成类
        if _looks_like_code_generation(combined):
            for key, entry in self.script.items():
                if _task_match(key, combined):
                    code = entry.get("code")
                    if code:
                        return code
                    answer = entry.get("answer")
                    if answer is not None:
                        return self._answer_code(answer)
            return self._fallback_code(combined)

        # 4. 默认：回显最后一个问题
        return "OK"

    # ------------------------------------------------------------------
    def _plan_response(self, prompt: str) -> str:
        """从 prompt 的 Available atomic skills 中选前 2 个，构造计划 JSON。"""
        ids = re.findall(r"logical_id:\s*([A-Za-z0-9_.\-]+)", prompt)
        skills = [{"logical_id": i, "params": {}} for i in ids[:2]]
        return json.dumps({"skills": skills}, ensure_ascii=False)

    @staticmethod
    def _answer_code(answer: str) -> str:
        return (
            "def solve():\n"
            f"    return {answer!r}\n"
        )

    @staticmethod
    def _fallback_code(prompt: str) -> str:
        # 保守：返回入口函数桩（验证会失败 → 触发 repair/fallback 路径，正是 smoke 要覆盖的）
        match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", prompt)
        name = match.group(1) if match else "solve"
        return f"def {name}(*args, **kwargs):\n    return None\n"


def _slug(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(key).lower())


def _task_match(key: str, combined: str) -> bool:
    """按任务 id 精确匹配（`[task_id: key]` 标记），避免子串误匹配。"""
    if re.search(rf"task_id:\s*{re.escape(key)}\b", combined):
        return True
    slug = _slug(key)
    if slug:
        match = re.search(r"task_id:\s*([a-z0-9_]+)", combined, re.I)
        if match and _slug(match.group(1)) == slug:
            return True
    return False


def _looks_like_code_generation(text: str) -> bool:
    lowered = text.lower()
    return ("write" in lowered and "python" in lowered) or "def solve" in lowered \
        or "code task" in lowered or "implement" in lowered


def _looks_like_env_step(text: str) -> bool:
    lowered = text.lower()
    return "admissible" in lowered or "your task is to" in lowered \
        or "observation" in lowered and "action" in lowered


def _pick_next_action(actions: list[str], prompt: str) -> str:
    """按已执行步数选择下一个脚本动作。

    每步迭代在 prompt 中留下 2 个 "Action:" 标记（历史行 + 当前指令后缀），
    已执行步数 = (总数 - 1) // 2。
    """
    total = len(re.findall(r"(?:^|\n)Action:", prompt))
    done = max(0, (total - 1) // 2)
    if done < len(actions):
        return actions[done]
    return actions[-1]
