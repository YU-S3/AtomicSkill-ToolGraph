"""Compile successful ALFWorld traces into reusable skill templates."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

from alfworld_.schemas import (
    AlfWorldExemplar,
    AlfWorldSkillTemplate,
    AlfWorldTrace,
)


@dataclass
class CompileResult:
    """Output of one compile() call: a Layer 1 template + a Layer 2 exemplar.

    Either field may be ``None`` independently — e.g. a successful trace whose
    structure does not match a known blueprint still produces an exemplar.
    """

    template: AlfWorldSkillTemplate | None
    exemplar: AlfWorldExemplar | None


class AlfWorldCompiler:
    """Extract parameterised action templates from successful traces.

    The key challenge is that ReAct-style agents produce *noisy* traces full
    of exploration, backtracking, and repeated actions.  The compiler must
    distil the trace down to the **minimal causal chain** that actually
    accomplished the task.

    Strategy: instead of trying to clean up the raw trace, we reconstruct the
    *ideal* action sequence from the identified entities and the known task
    structure.  This gives short, clean templates every time.

    Layer 2 (exemplar / guideline) can optionally use an LLM client to extract
    abstract procedural rules from the trace.  Pass ``llm_client`` to enable
    the LLM-based path; otherwise a rule-based fallback from the task blueprint
    is used.
    """

    def __init__(self, llm_client: Any = None) -> None:
        self.llm_client = llm_client

    # Canonical action sequences per task type (parameterised)
    # ALFWorld uses "move X to Y" for placement, NOT "put X in/on Y".
    # clean/heat/cool are direct commands (no put required).
    _TASK_BLUEPRINTS: dict[str, list[str]] = {
        "pick_and_place_simple": [
            "go to {object_location}",
            "take {object} from {object_location}",
            "go to {target_location}",
            "move {object} to {target_location}",
        ],
        "pick_and_place_with_movable_recep": [
            "go to {object_location}",
            "take {object} from {object_location}",
            "go to {target_location}",
            "move {object} to {target_location}",
        ],
        "pick_clean_then_place_in_recep": [
            "go to {object_location}",
            "take {object} from {object_location}",
            "go to {cleaning_station}",
            "clean {object} with {cleaning_station}",
            "go to {target_location}",
            "move {object} to {target_location}",
        ],
        "pick_heat_then_place_in_recep": [
            "go to {object_location}",
            "take {object} from {object_location}",
            "go to {heating_station}",
            "heat {object} with {heating_station}",
            "go to {target_location}",
            "move {object} to {target_location}",
        ],
        "pick_cool_then_place_in_recep": [
            "go to {object_location}",
            "take {object} from {object_location}",
            "go to {cooling_station}",
            "cool {object} with {cooling_station}",
            "go to {target_location}",
            "move {object} to {target_location}",
        ],
        "look_at_obj_in_light": [
            "go to {object_location}",
            "take {object} from {object_location}",
            "go to {light_source}",
            "use {light_source}",
        ],
        "pick_two_obj_and_place": [
            "go to {object1_location}",
            "take {object1} from {object1_location}",
            "go to {target_location}",
            "move {object1} to {target_location}",
            "go to {object2_location}",
            "take {object2} from {object2_location}",
            "go to {target_location}",
            "move {object2} to {target_location}",
        ],
    }

    def compile(self, trace: AlfWorldTrace) -> CompileResult:
        """Compile a successful trace into Layer 1 + Layer 2 artifacts.

        Returns a :class:`CompileResult` whose ``template`` and ``exemplar``
        fields may independently be ``None``.  Failed traces yield an empty
        result.
        """
        if not trace.success or not trace.actions:
            return CompileResult(template=None, exemplar=None)

        template = self._compile_template(trace)
        exemplar = self._compile_guideline(trace)
        return CompileResult(template=template, exemplar=exemplar)

    # ------------------------------------------------------------------
    # Layer 1: parameterised executable template (blueprint-based)
    # ------------------------------------------------------------------

    def _compile_template(self, trace: AlfWorldTrace) -> AlfWorldSkillTemplate | None:
        """Build a parameterised :class:`AlfWorldSkillTemplate` or ``None``."""
        # Extract concrete entities from the trace
        bindings = self._extract_entities(trace)
        if not bindings:
            return None

        blueprint = self._TASK_BLUEPRINTS.get(trace.task_type)
        if not blueprint:
            return None

        # Check all required params are present; fill stations from
        # admissible commands if not found in the trace itself.
        required = set()
        for step in blueprint:
            for m in re.finditer(r"\{(\w+)\}", step):
                required.add(m.group(1))

        # Infer stations from known defaults when trace didn't contain the
        # explicit clean/heat/cool/use action.
        _STATION_DEFAULTS = {
            "cleaning_station": "sinkbasin 1",
            "heating_station": "microwave 1",
            "cooling_station": "fridge 1",
            "light_source": "desklamp 1",
        }
        for param in list(required - set(bindings.keys())):
            if param in _STATION_DEFAULTS:
                bindings[param] = _STATION_DEFAULTS[param]

        missing = required - set(bindings.keys())
        if missing:
            return None

        # Instantiate blueprint with concrete values to get example,
        # then store the parameterised blueprint as the template
        parameters = sorted(required)

        return AlfWorldSkillTemplate(
            template_id="alf_skill_%s_%s" % (trace.task_type, uuid.uuid4().hex[:8]),
            task_type=trace.task_type,
            action_template=list(blueprint),
            parameters=parameters,
            example_bindings={k: bindings[k] for k in parameters},
            source_task_id=trace.task_id,
            source_trace_id=trace.trace_id,
            success_count=1,
        )

    # ------------------------------------------------------------------
    # Layer 2: LLM-extracted procedural guideline (v4)
    # ------------------------------------------------------------------

    _GUIDELINE_EXTRACTION_PROMPT = """You are analyzing a successful task execution trace to extract reusable procedural rules.

Task type: {task_type}
Task goal: {goal}

Successful action sequence:
{actions}

Extract 3-5 reusable procedural rules that would help an agent solve similar tasks in the future.

Requirements:
- Each rule must be ABSTRACT: do NOT include specific object names, location names, or numbers
- Use generic terms like "target object", "cleaning station", "destination receptacle"
- Each rule should be one sentence in the format: "When [situation], [do what], because [why]"
- Focus on STRATEGY (search approach, action ordering) not exact steps
- Include at least one rule about what to AVOID (common failure patterns)
- 3-5 rules only, keep each concise

Rules:"""

    def _compile_guideline(self, trace: AlfWorldTrace) -> AlfWorldExemplar | None:
        """Extract a procedural guideline from a successful trace.

        Uses the configured LLM client when available; otherwise falls back
        to a rule-based extraction from the blueprint.
        """
        if self.llm_client is None:
            return self._compile_guideline_rule_based(trace)

        # Build a denoised action listing for the prompt.
        actions_text: list[str] = []
        for a in trace.actions:
            obs_lower = a.observation.lower() if a.observation else ""
            if "nothing happens" in obs_lower:
                continue
            if a.action.lower() in ("inventory", "look"):
                continue
            obs_short = a.observation.split("\n")[0][:60] if a.observation else ""
            actions_text.append("%s -> %s" % (a.action, obs_short))

        if not actions_text:
            return None

        prompt = self._GUIDELINE_EXTRACTION_PROMPT.format(
            task_type=trace.task_type,
            goal=trace.goal,
            actions="\n".join(
                "  %d. %s" % (i + 1, a) for i, a in enumerate(actions_text)
            ),
        )

        try:
            from runtime.config import GenerationSettings  # local import to avoid cycles
            settings = GenerationSettings(temperature=0.0, max_output_tokens=500)
            response = self.llm_client.generate(
                instructions="You are a task analysis assistant. Extract concise procedural rules.",
                input_text=prompt,
                settings=settings,
            )

            # Parse LLM output into a flat list of rules
            rules: list[str] = []
            _LEAD_NUM = re.compile(r"^[\d]+[\.\)]\s*")
            _LEAD_BULLET = re.compile(r"^[-\*\u2022]\s*")
            for line in response.text.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                line = _LEAD_NUM.sub("", line)
                line = _LEAD_BULLET.sub("", line)
                line = line.strip()
                if len(line) > 10:
                    rules.append(line)

            if not rules:
                return None

            rules = rules[:5]
            extraction_tokens = (response.prompt_tokens or 0) + (response.completion_tokens or 0)

            return AlfWorldExemplar(
                exemplar_id="guide_%s_%s" % (trace.task_type, uuid.uuid4().hex[:8]),
                task_type=trace.task_type,
                goal=trace.goal,
                rules=rules,
                source_task_id=trace.task_id,
                source_trace_id=trace.trace_id,
                extraction_tokens=extraction_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                "  [compiler] guideline extraction failed: %s, using rule-based fallback"
                % str(exc)[:200]
            )
            return self._compile_guideline_rule_based(trace)

    def _compile_guideline_rule_based(
        self, trace: AlfWorldTrace,
    ) -> AlfWorldExemplar | None:
        """Deterministic fallback: derive abstract rules from the blueprint."""
        blueprint = self._TASK_BLUEPRINTS.get(trace.task_type)
        if not blueprint:
            return None

        rules: list[str] = []
        for step in blueprint:
            abstract = step.replace(
                "{object_location}", "the location where the target object is found",
            )
            abstract = abstract.replace("{object}", "the target object")
            abstract = abstract.replace(
                "{target_location}", "the destination receptacle",
            )
            abstract = abstract.replace(
                "{cleaning_station}", "the cleaning station",
            )
            abstract = abstract.replace(
                "{heating_station}", "the heating station",
            )
            abstract = abstract.replace(
                "{cooling_station}", "the cooling station",
            )
            abstract = abstract.replace("{light_source}", "the light source")
            # pick_two-specific placeholders
            abstract = abstract.replace("{object1_location}", "the first object location")
            abstract = abstract.replace("{object1}", "the first target object")
            abstract = abstract.replace("{object2_location}", "the second object location")
            abstract = abstract.replace("{object2}", "the second target object")
            rules.append("Step: %s" % abstract)

        rules.append(
            "Avoid: Do not revisit locations already checked; explore systematically"
        )

        return AlfWorldExemplar(
            exemplar_id="guide_rb_%s_%s" % (trace.task_type, uuid.uuid4().hex[:8]),
            task_type=trace.task_type,
            goal=trace.goal,
            rules=rules,
            source_task_id=trace.task_id,
            source_trace_id=trace.trace_id,
            extraction_tokens=0,
        )

    # ------------------------------------------------------------------
    # Entity extraction from trace actions
    # ------------------------------------------------------------------

    _TAKE_RE = re.compile(r"take\s+(.+?)\s+from\s+(.+)", re.IGNORECASE)
    _PUT_RE = re.compile(r"put\s+(.+?)\s+(?:in|on|in/on)\s+(.+)", re.IGNORECASE)
    _MOVE_RE = re.compile(r"move\s+(.+?)\s+to\s+(.+)", re.IGNORECASE)
    _CLEAN_RE = re.compile(r"clean\s+(.+?)\s+with\s+(.+)", re.IGNORECASE)
    _HEAT_RE = re.compile(r"heat\s+(.+?)\s+with\s+(.+)", re.IGNORECASE)
    _COOL_RE = re.compile(r"cool\s+(.+?)\s+with\s+(.+)", re.IGNORECASE)
    _USE_RE = re.compile(r"use\s+(.+)", re.IGNORECASE)

    def _extract_entities(self, trace: AlfWorldTrace) -> dict[str, str]:
        """Walk the trace and extract concrete entity names + locations."""
        bindings: dict[str, str] = {}
        task_type = trace.task_type

        # Only consider actions that the environment accepted
        effective = [
            rec for rec in trace.actions
            if "nothing happens" not in rec.observation.lower()
        ]

        # --- Object, object_location, target_location ---
        # The FIRST successful "take X from Y" tells us object + object_location.
        # The LAST successful "put X in/on Y" (or "move X to Y") tells us
        # target_location.
        objects_taken: list[tuple[str, str]] = []  # (object, location)
        targets_placed: list[tuple[str, str]] = []  # (object, location)

        for rec in effective:
            action = rec.action

            m = self._TAKE_RE.match(action)
            if m:
                objects_taken.append((m.group(1).strip(), m.group(2).strip()))
                continue

            m = self._PUT_RE.match(action)
            if m:
                targets_placed.append((m.group(1).strip(), m.group(2).strip()))
                continue

            m = self._MOVE_RE.match(action)
            if m:
                targets_placed.append((m.group(1).strip(), m.group(2).strip()))
                continue

            m = self._CLEAN_RE.match(action)
            if m:
                bindings["cleaning_station"] = m.group(2).strip()
                continue

            m = self._HEAT_RE.match(action)
            if m:
                bindings["heating_station"] = m.group(2).strip()
                continue

            m = self._COOL_RE.match(action)
            if m:
                bindings["cooling_station"] = m.group(2).strip()
                continue

            m = self._USE_RE.match(action)
            if m:
                bindings["light_source"] = m.group(1).strip()
                continue

        # --- Assign object / object_location / target_location ---
        if task_type == "pick_two_obj_and_place":
            # Need two distinct objects
            if len(objects_taken) >= 2 and len(targets_placed) >= 1:
                bindings["object1"] = objects_taken[0][0]
                bindings["object1_location"] = objects_taken[0][1]
                # Find a second distinct object
                for obj, loc in objects_taken[1:]:
                    if obj != objects_taken[0][0]:
                        bindings["object2"] = obj
                        bindings["object2_location"] = loc
                        break
                else:
                    # Same object taken twice from different locations
                    bindings["object2"] = objects_taken[1][0]
                    bindings["object2_location"] = objects_taken[1][1]
                bindings["target_location"] = targets_placed[0][1]
        else:
            if objects_taken:
                bindings["object"] = objects_taken[0][0]
                bindings["object_location"] = objects_taken[0][1]
            if targets_placed:
                # Last put/move is the final placement
                bindings["target_location"] = targets_placed[-1][1]

        # For light tasks, the light_source location IS the go-to target
        # (e.g., "desklamp 1" is both the use target and navigation target)
        # No target_location needed.

        return bindings


# ---------------------------------------------------------------------------
# SkillCard bridge
# ---------------------------------------------------------------------------

def template_to_skill_card(template: AlfWorldSkillTemplate) -> dict[str, Any]:
    """Convert an :class:`AlfWorldSkillTemplate` to a SkillCard-compatible dict."""
    return {
        "skill_id": template.template_id,
        "benchmark": "alfworld",
        "task_pattern": template.task_type,
        "signature": {
            "entry_point": template.task_type,
            "param_count": len(template.parameters),
            "params": template.parameters,
        },
        "code": json.dumps({
            "action_template": template.action_template,
            "parameters": template.parameters,
            "example_bindings": template.example_bindings,
        }),
        "preconditions": {
            "task_pattern": template.task_type,
            "task_type": template.task_type,
        },
        "source_task_id": template.source_task_id,
        "utility": 1.0,
        "status": "active",
        "direct_eligible": True,
    }
