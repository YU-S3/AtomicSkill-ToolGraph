"""Lightweight strategy storage with ContrastiveEval for ALFWorld."""

from __future__ import annotations

from typing import Any

from runtime.config import GenerationSettings


# ---------------------------------------------------------------------------
# Strategy bank
# ---------------------------------------------------------------------------

class AlfWorldStrategyBank:
    """In-memory strategy store with guided/unguided tracking."""

    def __init__(self) -> None:
        self._strategies: dict[str, dict[str, Any]] = {}

    def add(self, strategy_id: str, task_type: str, description: str) -> None:
        self._strategies[strategy_id] = {
            "strategy_id": strategy_id,
            "task_type": task_type,
            "description": description,
            "guided_success": 0,
            "guided_total": 0,
            "unguided_success": 0,
            "unguided_total": 0,
            "status": "active",
        }

    def get_active_for_type(self, task_type: str) -> list[dict[str, Any]]:
        return [
            s for s in self._strategies.values()
            if s["task_type"] == task_type and s["status"] == "active"
        ]

    def record_guided(self, strategy_id: str, success: bool) -> None:
        s = self._strategies.get(strategy_id)
        if s is None:
            return
        s["guided_total"] += 1
        if success:
            s["guided_success"] += 1
        self._check_suppress(strategy_id)

    def record_unguided(self, task_type: str, success: bool) -> None:
        for s in self._strategies.values():
            if s["task_type"] == task_type and s["status"] == "active":
                s["unguided_total"] += 1
                if success:
                    s["unguided_success"] += 1

    def _check_suppress(self, strategy_id: str) -> None:
        s = self._strategies[strategy_id]
        if s["guided_total"] >= 5 and s["unguided_total"] >= 3:
            g_rate = s["guided_success"] / max(s["guided_total"], 1)
            u_rate = s["unguided_success"] / max(s["unguided_total"], 1)
            if g_rate < u_rate - 0.1:
                s["status"] = "suppressed"

    @property
    def active_count(self) -> int:
        return sum(1 for s in self._strategies.values() if s["status"] == "active")

    @property
    def total_count(self) -> int:
        return len(self._strategies)

    def snapshot(self) -> dict[str, Any]:
        return {
            "total": self.total_count,
            "active": self.active_count,
            "suppressed": sum(
                1 for s in self._strategies.values() if s["status"] == "suppressed"
            ),
        }

    # ------------------------------------------------------------------
    # Resume support — full state serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {"strategies": {sid: dict(s) for sid, s in self._strategies.items()}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlfWorldStrategyBank":
        bank = cls()
        bank._strategies = {
            sid: dict(s) for sid, s in data.get("strategies", {}).items()
        }
        return bank


# ---------------------------------------------------------------------------
# Strategy extraction from traces
# ---------------------------------------------------------------------------

_HEURISTIC_STRATEGIES: dict[str, str] = {
    "pick_and_place_simple": (
        "Find the target object by systematically checking receptacles "
        "(countertops, shelves, drawers, cabinets). Pick it up, navigate "
        "to the destination, and place it using 'move'."
    ),
    "pick_and_place_with_movable_recep": (
        "Find the target object, pick it up, navigate to the destination, "
        "and place it. The object may be inside a movable receptacle."
    ),
    "pick_clean_then_place_in_recep": (
        "Find the target object, pick it up, go to the sinkbasin and "
        "clean it, then navigate to the destination and place it."
    ),
    "pick_heat_then_place_in_recep": (
        "Find the target object, pick it up, go to the microwave and "
        "heat it, then navigate to the destination and place it."
    ),
    "pick_cool_then_place_in_recep": (
        "Find the target object, pick it up, go to the fridge and "
        "cool it, then navigate to the destination and place it."
    ),
    "look_at_obj_in_light": (
        "Find the target object, pick it up, go to a light source "
        "(desklamp or floorlamp) and use it to illuminate the object."
    ),
    "pick_two_obj_and_place": (
        "Find the first instance of the target object, pick it up and "
        "place it at the destination. Then find a second instance and "
        "place it at the destination too."
    ),
}

_STRATEGY_SETTINGS = GenerationSettings(temperature=0.2, max_output_tokens=200)


def extract_strategy(
    trace: Any,
    llm_client: Any | None = None,
) -> tuple[dict[str, str] | None, int]:
    """Extract a reusable strategy description from a successful trace.

    Returns ``(strategy_dict_or_None, tokens_used)``.  The second element
    lets the caller attribute the strategy-extraction LLM cost back to the
    episode that triggered it (otherwise the experiment metrics would
    silently undercount tokens for the strategy-enabled conditions).

    Uses LLM if available; falls back to a heuristic.
    """
    if not trace.success:
        return None, 0

    strategy_id = "strat_%s_%s" % (trace.task_type, trace.trace_id)
    tokens_used = 0

    # Try LLM-enhanced extraction
    if llm_client is not None:
        action_lines = "\n".join(
            "  %d. %s" % (i + 1, a)
            for i, a in enumerate(trace.action_history[:20])
        )
        prompt = (
            "Analyze this successful task execution and extract a reusable strategy.\n\n"
            "Task type: %s\nGoal: %s\n"
            "Actions taken (effective only):\n%s\n\n"
            "Respond with a single paragraph describing the key approach for this "
            "type of task. Focus on WHAT to do and in WHAT ORDER, not specific "
            "object names. Start with 'STRATEGY:'"
        ) % (trace.task_type, trace.goal, action_lines)
        try:
            resp = llm_client.generate(
                instructions="Extract reusable task strategies from execution traces.",
                input_text=prompt,
                settings=_STRATEGY_SETTINGS,
            )
            tokens_used = (resp.prompt_tokens or 0) + (resp.completion_tokens or 0)
            text = resp.text.strip()
            desc = text.split("STRATEGY:")[-1].strip() if "STRATEGY:" in text else text
            if len(desc) > 20:
                return (
                    {
                        "strategy_id": strategy_id,
                        "task_type": trace.task_type,
                        "description": desc,
                    },
                    tokens_used,
                )
        except Exception:
            pass

    # Heuristic fallback
    desc = _HEURISTIC_STRATEGIES.get(
        trace.task_type, "Complete the household task step by step.",
    )
    return (
        {
            "strategy_id": strategy_id,
            "task_type": trace.task_type,
            "description": desc,
        },
        tokens_used,
    )
