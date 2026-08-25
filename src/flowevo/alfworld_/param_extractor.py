"""Extract action template parameters from ALFWorld observations.

Parameter names match the blueprint slots in ``compiler._TASK_BLUEPRINTS``.
"""

from __future__ import annotations

import re
from typing import Any


class ParamExtractor:
    """Rule-based parameter extraction for direct skill reuse.

    Returns *partial* bindings when ``object_location`` cannot be inferred
    from the initial observation (which is the common case — most objects
    are inside closed containers).  The executor is expected to run a
    zero-LLM exploration phase to fill in the missing location.
    """

    def extract(
        self,
        task_type: str,
        goal: str,
        observation: str,
        admissible_commands: list[str],
    ) -> dict[str, str] | None:
        """Return parameter bindings or ``None`` on hard failure.

        Missing ``object_location`` is *not* a hard failure — the dict will
        simply omit that key, and the caller decides what to do.
        """

        object_name = self._extract_object_from_goal(goal)
        if not object_name:
            return None

        target_location = self._extract_target_from_goal(goal, admissible_commands)

        if task_type == "pick_two_obj_and_place":
            return self._extract_pick_two(object_name, target_location,
                                          observation, admissible_commands)

        # --- Single-object tasks ---
        bindings: dict[str, str] = {}

        # Try to find object with numeric ID in observation
        object_with_id = self._find_object_with_id(object_name, observation)
        object_location = self._find_object_location(object_name, observation)

        if object_with_id:
            bindings["object"] = object_with_id
        # If not visible yet, store the base name — exploration will find the ID
        else:
            bindings["object"] = object_name

        if object_location:
            bindings["object_location"] = object_location
        # object_location missing is OK — caller does exploration

        if target_location:
            bindings["target_location"] = target_location

        # Task-specific stations.  _find_receptacle returns "" if absent;
        # we fall back to a canonical sentinel only as a last resort.
        if "clean" in task_type:
            station = self._find_receptacle("sinkbasin", admissible_commands)
            bindings["cleaning_station"] = station or "sinkbasin 1"
        elif "heat" in task_type:
            station = self._find_receptacle("microwave", admissible_commands)
            bindings["heating_station"] = station or "microwave 1"
        elif "cool" in task_type:
            station = self._find_receptacle("fridge", admissible_commands)
            bindings["cooling_station"] = station or "fridge 1"
        elif "light" in task_type or "look_at" in task_type:
            lamp = self._find_receptacle("desklamp", admissible_commands)
            if not lamp:
                lamp = self._find_receptacle("floorlamp", admissible_commands)
            bindings["light_source"] = lamp or "desklamp 1"

        return bindings

    def _extract_pick_two(
        self, object_name: str, target_location: str,
        observation: str, admissible_commands: list[str],
    ) -> dict[str, str] | None:
        # pick_two needs exploration to find both objects — return partial
        if not target_location:
            return None
        return {
            "object_type": object_name,  # base name without ID
            "target_location": target_location,
        }

    # ------------------------------------------------------------------
    # Goal parsing
    # ------------------------------------------------------------------

    _OBJECT_PATTERNS = [
        re.compile(
            r"(?:put|place)\s+(?:a|an|some|the|two)\s+"
            r"(?:clean|hot|cold|cool|heated|cooled)?\s*(\w+)"
        ),
        re.compile(r"(?:examine|look at)\s+(?:a|an|some|the)\s+(\w+)"),
        re.compile(r"(?:heat|cool|clean)\s+(?:a|an|some|the)\s+(\w+)"),
        re.compile(r"(?:find)\s+(?:a|an|some|the|two)\s+(\w+)"),
    ]

    _TARGET_PATTERNS = [
        # "put it in/on X"
        re.compile(
            r"(?:put|place)\s+it\s+(?:in|on|into)\s+(?:a|an|the)?\s*(\w+)"
        ),
        # "put ... in/on X" (last "in/on WORD" in the sentence)
        re.compile(r"\b(?:in|on)\s+(\w+)\s*\.?\s*$"),
    ]

    def _extract_object_from_goal(self, goal: str) -> str:
        lower = goal.lower()
        for pat in self._OBJECT_PATTERNS:
            m = pat.search(lower)
            if m:
                return self._singularize(m.group(1).strip())
        return ""

    @staticmethod
    def _singularize(word: str) -> str:
        """Best-effort English singularisation for ALFWorld object names."""
        if not word or len(word) < 3:
            return word
        if word.endswith("ives"):
            return word[:-3] + "fe"  # knives -> knife, wives -> wife
        if word.endswith("ves"):
            return word[:-3] + "f"   # leaves -> leaf, wolves -> wolf
        if word.endswith("ies"):
            return word[:-3] + "y"  # batteries -> battery
        if word.endswith("ses") or word.endswith("xes") or word.endswith("zes"):
            return word[:-2]
        if word.endswith("ss"):
            return word  # glass / mass — already singular-looking
        if word.endswith("s"):
            return word[:-1]  # pillows -> pillow
        return word

    def _extract_target_from_goal(
        self, goal: str, admissible_commands: list[str],
    ) -> str:
        lower = goal.lower()
        for pat in self._TARGET_PATTERNS:
            m = pat.search(lower)
            if m:
                target_type = m.group(1).strip()
                found = self._find_receptacle(target_type, admissible_commands)
                # Last-resort sentinel only when nothing else matches
                return found or ("%s 1" % target_type)
        return ""

    # ------------------------------------------------------------------
    # Object identification
    # ------------------------------------------------------------------

    def _find_object_with_id(self, object_type: str, observation: str) -> str:
        pattern = r"\b(%s\s+\d+)\b" % re.escape(object_type.lower())
        m = re.search(pattern, observation.lower())
        return m.group(1) if m else ""

    def _find_object_location(self, object_type: str, observation: str) -> str:
        obs_lower = observation.lower()
        obj_lower = object_type.lower()
        for prefix in ("on the", "in the"):
            pattern = r"%s (\w+\s+\d+),\s+you see[^.]*\b%s" % (
                prefix, re.escape(obj_lower),
            )
            m = re.search(pattern, obs_lower)
            if m:
                return m.group(1)
        return ""

    # ------------------------------------------------------------------
    # Receptacle lookup
    # ------------------------------------------------------------------

    def _find_receptacle(
        self, receptacle_type: str, admissible_commands: list[str],
    ) -> str:
        """Find a receptacle of the given type.

        Returns the empty string if not found.  Callers must distinguish
        "found" from "not found" themselves and supply a sentinel/fallback
        if needed.
        """
        for cmd in admissible_commands:
            if cmd.startswith("go to ") and receptacle_type in cmd.lower():
                return cmd[len("go to "):]
        return ""
