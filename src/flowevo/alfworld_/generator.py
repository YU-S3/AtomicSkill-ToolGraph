"""ALFWorld ReAct-style step-by-step action generator."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from runtime.config import GenerationSettings


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

@dataclass
class AlfWorldStepOutput:
    """Result of a single generator step."""

    action: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Prefer the provider's authoritative total because it may include token
    # classes that are not reconstructible from the two legacy counters.
    total_tokens: int = 0
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
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

_PREFIX_RE = re.compile(r"^(\d+[\.\)]\s*|-\s*|>\s*)")

# temperature=0.0: empirical testing showed 0.3 introduced too much variance.
# max_output_tokens=256: Think/Act format needs room for reasoning (~100-150
# tokens for Think + ~20 tokens for Act).  Previous 64 prevented any reasoning.
_STEP_SETTINGS = GenerationSettings(
    temperature=0.0, max_output_tokens=256, thinking="disabled")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class AlfWorldGenerator:
    """ReAct-style step-by-step generator for ALFWorld.

    Each call to :meth:`step` produces exactly one action chosen from the
    current admissible commands list.
    """

    def __init__(self, llm_client: Any) -> None:
        self._llm = llm_client

    def step(
        self,
        task_goal: str,
        observation: str,
        admissible_commands: list[str],
        action_history: list[str] | None = None,
        observation_history: list[str] | None = None,
        skill_context: str = "",
        policy_directives: list[str] | None = None,
    ) -> AlfWorldStepOutput:
        """Generate the next action.

        Returns an :class:`AlfWorldStepOutput` with the chosen action and
        token counts.
        """
        system = _build_system(policy_directives)
        user = _build_step_user(
            task_goal=task_goal,
            observation=observation,
            admissible_commands=admissible_commands,
            action_history=action_history or [],
            observation_history=observation_history or [],
            skill_context=skill_context,
        )
        resp = self._llm.generate(
            instructions=system,
            input_text=user,
            settings=_STEP_SETTINGS,
        )
        action = _parse_single_action(resp.text, admissible_commands)
        return AlfWorldStepOutput(
            action=action,
            prompt_tokens=int(getattr(resp, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(resp, "completion_tokens", 0) or 0),
            total_tokens=int(
                getattr(resp, "total_tokens", 0)
                or (getattr(resp, "prompt_tokens", 0) or 0)
                + (getattr(resp, "completion_tokens", 0) or 0)
            ),
            latency_ms=float(getattr(resp, "latency_ms", 0.0) or 0.0),
        )


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_system(policy_directives: list[str] | None) -> str:
    if not policy_directives:
        return _SYSTEM_PROMPT
    parts = [d for d in policy_directives if d.strip()]
    if not parts:
        return _SYSTEM_PROMPT
    directive_text = "\n".join("- %s" % d for d in parts)
    return "%s\n\nAdditional guidance:\n%s" % (_SYSTEM_PROMPT, directive_text)


def _build_step_user(
    task_goal: str,
    observation: str,
    admissible_commands: list[str],
    action_history: list[str],
    observation_history: list[str],
    skill_context: str,
) -> str:
    parts: list[str] = ["Task: %s" % task_goal]

    if skill_context:
        parts.append("\nRelevant experience:\n%s" % skill_context)

    # Show recent history (last 10 steps max to stay within context)
    if action_history:
        recent_n = min(len(action_history), 10)
        start = len(action_history) - recent_n
        parts.append("\nRecent actions:")
        for i in range(start, len(action_history)):
            parts.append("  > %s" % action_history[i])
            if i < len(observation_history):
                parts.append("    %s" % observation_history[i][:120])

    parts.append("\nCurrent observation:\n%s" % observation)

    # Show ALL admissible commands so the LLM can pick the right one
    parts.append("\nValid actions (%d):" % len(admissible_commands))
    for cmd in admissible_commands:
        parts.append("  %s" % cmd)

    parts.append("\nChoose ONE action from the valid actions list above:")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Action parser — extract Act: line from Think/Act formatted output
# ---------------------------------------------------------------------------

_ACT_RE = re.compile(r"^act\s*:\s*", re.IGNORECASE)


def _parse_single_action(text: str, admissible_commands: list[str]) -> str:
    """Parse the action from Think/Act formatted LLM output.

    Extraction priority:
    1. Find the ``Act:`` line and match against admissible commands.
    2. Fall back to scanning all lines for an admissible command match.
    3. Substring containment (longest match wins).
    4. Return first admissible command as last resort.
    """
    text = text.strip()

    # --- 1. Extract Act: line ---
    for line in text.split("\n"):
        line = line.strip()
        m = _ACT_RE.match(line)
        if m:
            action = line[m.end():].strip()
            action = _PREFIX_RE.sub("", action).strip()
            # Exact match (case-insensitive)
            for cmd in admissible_commands:
                if cmd.lower() == action.lower():
                    return cmd
            # Substring match within the Act: value
            matches = [c for c in admissible_commands if c.lower() in action.lower()]
            if matches:
                return max(matches, key=len)

    # --- 2. Line-by-line exact match (handles old format / no Act: prefix) ---
    for line in text.split("\n"):
        line = line.strip()
        if line.lower().startswith("think"):
            continue
        cleaned = _PREFIX_RE.sub("", line).strip()
        for cmd in admissible_commands:
            if cmd.lower() == cleaned.lower():
                return cmd

    # --- 3. Substring containment across full output ---
    lower = text.lower()
    matches = [c for c in admissible_commands if c.lower() in lower]
    if matches:
        return max(matches, key=len)

    # --- 4. Last resort: first admissible command ---
    return admissible_commands[0] if admissible_commands else "look"
