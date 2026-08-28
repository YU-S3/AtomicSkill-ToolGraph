"""ALFWorld task executor with step-by-step LLM interaction."""

from __future__ import annotations

import re
import uuid
from typing import Any

from alfworld_.env import AlfWorldEnv
from alfworld_.generator import AlfWorldGenerator
from alfworld_.param_extractor import ParamExtractor
from alfworld_.schemas import AlfWorldAction, AlfWorldSkillTemplate, AlfWorldTrace


class AlfWorldExecutor:
    """Execute ALFWorld tasks step-by-step.

    Supports three planning modes:

    * ``direct_skill`` -- instantiate a skill template (no LLM); includes a
      zero-LLM exploration phase to locate the target object if needed.
    * ``strategy_guided`` -- step-by-step LLM with strategy hints.
    * ``pure_dynamic`` -- step-by-step LLM without hints.
    """

    def __init__(
        self,
        env: AlfWorldEnv,
        generator: AlfWorldGenerator,
        max_steps: int = 50,
    ) -> None:
        self.env = env
        self.generator = generator
        self.max_steps = max_steps

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        task: Any,
        initial_observation: str,
        admissible_commands: list[str],
        planning_mode: str = "pure_dynamic",
        skill_template: AlfWorldSkillTemplate | None = None,
        skill_bindings: dict[str, str] | None = None,
        skill_context: str = "",
        policy_directives: list[str] | None = None,
        skill_library_result: dict[str, Any] | None = None,
    ) -> AlfWorldTrace:
        """Run a full episode and return the execution trace.

        ``skill_library_result`` is the dict produced by
        :meth:`SkillLibrary.retrieve` and overrides the legacy
        ``skill_template`` / ``skill_context`` / ``policy_directives`` fields
        when present.  Layer 1 (template) → direct reuse with fallback;
        Layer 2 (exemplar) → few-shot prompt; Layer 3 (insight) → policy
        directive.  Layers compose: an exemplar still augments the LLM
        fallback after a Layer 1 failure, and Layer 3 insights are always
        added to ``policy_directives``.
        """

        # ---- Fold skill_library_result into the legacy parameters ----
        layer_template: AlfWorldSkillTemplate | None = skill_template
        layer_exemplar = None
        layer_insight = None
        best_layer: int | None = None
        if skill_library_result:
            best_layer = skill_library_result.get("best_layer")
            lib_template = skill_library_result.get("template")
            if lib_template is not None:
                layer_template = lib_template
            layer_exemplar = skill_library_result.get("exemplar")
            layer_insight = skill_library_result.get("insight")

        # Determine planning_mode for the new path
        if skill_library_result is not None:
            if best_layer == 1 and layer_template is not None and skill_bindings is not None:
                planning_mode = "direct_skill"
            elif layer_exemplar is not None:
                planning_mode = "exemplar_guided"
            elif layer_insight is not None:
                planning_mode = "insight_guided"
            else:
                planning_mode = "pure_dynamic"

        # Compose skill_context: exemplar (Layer 2) takes precedence over the
        # legacy free-text skill_context.  Even when Layer 1 is selected, the
        # exemplar is still useful for the fallback LLM path.
        effective_skill_context = skill_context
        if layer_exemplar is not None:
            effective_skill_context = layer_exemplar.to_prompt_text()

        # Compose policy directives: insight text gets appended.
        effective_directives: list[str] = list(policy_directives or [])
        if layer_insight is not None:
            insight_text = layer_insight.to_prompt_text()
            if insight_text:
                effective_directives.append(insight_text)

        trace = AlfWorldTrace(
            trace_id="alf_%s" % uuid.uuid4().hex[:8],
            task_id=task.task_id,
            task_type=task.task_type,
            task_pattern=task.task_type,
            goal=task.goal,
            planning_mode=planning_mode,
            max_steps=self.max_steps,
        )

        if (
            planning_mode == "direct_skill"
            and layer_template is not None
            and skill_bindings is not None
        ):
            self._execute_direct(
                trace, task, layer_template, skill_bindings,
                initial_observation, admissible_commands,
            )
            if trace.success:
                self._finalize(trace)
                return trace
            # Direct failed -> fall back to step-by-step LLM.
            # Use the LATEST environment state as the LLM's starting point,
            # because direct reuse already executed several actions and the
            # initial observation is now stale.
            if trace.actions:
                last = trace.actions[-1]
                fallback_obs = last.observation
                fallback_admissible = last.admissible_commands or admissible_commands
            else:
                fallback_obs = initial_observation
                fallback_admissible = admissible_commands
            # When falling back, downgrade the planning_mode tag so the
            # episode record reflects what actually drove the LLM steps.
            if layer_exemplar is not None:
                trace.planning_mode = "direct_then_exemplar"
            elif effective_directives:
                trace.planning_mode = "direct_then_insight"
            else:
                trace.planning_mode = "direct_then_dynamic"
        else:
            fallback_obs = initial_observation
            fallback_admissible = admissible_commands

        # Step-by-step LLM execution
        self._execute_step_by_step(
            trace, task, fallback_obs, fallback_admissible,
            skill_context=effective_skill_context,
            policy_directives=effective_directives if effective_directives else None,
        )

        self._finalize(trace)
        return trace

    # ------------------------------------------------------------------
    # Direct reuse with exploration
    # ------------------------------------------------------------------

    def _execute_direct(
        self,
        trace: AlfWorldTrace,
        task: Any,
        template: AlfWorldSkillTemplate,
        bindings: dict[str, str],
        initial_observation: str,
        admissible_commands: list[str],
    ) -> None:
        trace.direct_use_count = 1
        current_location = ""  # track where we are after exploration

        # Reserve budget so the template has room to execute after exploration.
        # template_reserved = template length + 5-step safety buffer; the
        # remainder of max_steps is what exploration gets.  Using max(5, ...)
        # guarantees at least a few exploration steps even for very long
        # templates so we still get a chance to find the object.
        template_reserved = len(template.action_template) + 5
        explore_budget = max(5, self.max_steps - template_reserved)

        # Fill in missing object_location via zero-LLM exploration
        if "object_location" in template.parameters and "object_location" not in bindings:
            obj_name = bindings.get("object", "")
            found = self._explore_for_object(
                trace, obj_name, admissible_commands,
                max_explore_steps=explore_budget,
            )
            if found:
                bindings = dict(bindings)
                bindings["object"] = found[0]
                bindings["object_location"] = found[1]
                current_location = found[1]  # we're AT this location
            else:
                trace.direct_failure_count = 1
                return

        # For pick_two, exploration is too complex -- skip direct
        if task.task_type == "pick_two_obj_and_place":
            trace.direct_failure_count = 1
            return

        # Instantiate and execute the template
        try:
            actions = template.instantiate(bindings)
        except Exception:
            trace.direct_failure_count = 1
            return

        if not actions:
            trace.direct_failure_count = 1
            return

        for i, action in enumerate(actions):
            if len(trace.actions) >= self.max_steps:
                break

            # Skip "go to X" if we're already at X
            if (
                current_location
                and action.lower() == ("go to %s" % current_location).lower()
            ):
                current_location = ""  # consumed
                continue

            result = self.env.step(action)
            trace.actions.append(AlfWorldAction(
                step=len(trace.actions), action=action,
                observation=result.observation, score=result.score,
                done=result.done,
                admissible_commands=result.admissible_commands,
            ))
            trace.action_history.append(action)

            # Track navigation for future skip
            if action.lower().startswith("go to "):
                current_location = action[len("go to "):]
            elif action.lower().startswith("take "):
                current_location = ""  # state changed

            if result.won:
                trace.success = True
                trace.first_attempt_success = True
                trace.direct_success_count = 1
                return
            if result.done:
                # done without won = timeout
                trace.direct_failure_count = 1
                return
            if "nothing happens" in result.observation.lower():
                trace.direct_failure_count = 1
                return

        # Plan ran to completion without done
        trace.direct_failure_count = 1

    def _explore_for_object(
        self,
        trace: AlfWorldTrace,
        object_name: str,
        admissible_commands: list[str],
        max_explore_steps: int | None = None,
    ) -> tuple[str, str] | None:
        """Visit receptacles to find *object_name*. No LLM calls.

        Returns ``(object_with_id, receptacle)`` or ``None``.
        Exploration steps are recorded in *trace*.
        """
        if max_explore_steps is None:
            max_explore_steps = self.max_steps // 2

        obj_lower = object_name.lower()
        # Pattern to detect "object N" in observation text
        obj_id_re = re.compile(r"\b(%s\s+\d+)\b" % re.escape(obj_lower))

        go_commands = [c for c in admissible_commands if c.startswith("go to ")]
        explore_start = len(trace.actions)

        for cmd in go_commands:
            if (
                len(trace.actions) >= self.max_steps
                or len(trace.actions) - explore_start >= max_explore_steps
            ):
                break
            receptacle = cmd[len("go to "):]

            # Navigate
            result = self.env.step(cmd)
            trace.actions.append(AlfWorldAction(
                step=len(trace.actions), action=cmd,
                observation=result.observation, score=result.score,
                done=result.done,
                admissible_commands=result.admissible_commands,
            ))
            trace.action_history.append(cmd)

            # Check if object is visible
            m = obj_id_re.search(result.observation.lower())
            if m:
                return m.group(1), receptacle

            # If receptacle is closed, try opening it
            obs_lower = result.observation.lower()
            if "closed" in obs_lower:
                open_cmd = "open %s" % receptacle
                if open_cmd in result.admissible_commands:
                    if len(trace.actions) >= self.max_steps:
                        break
                    result2 = self.env.step(open_cmd)
                    trace.actions.append(AlfWorldAction(
                        step=len(trace.actions), action=open_cmd,
                        observation=result2.observation, score=result2.score,
                        done=result2.done,
                        admissible_commands=result2.admissible_commands,
                    ))
                    trace.action_history.append(open_cmd)

                    m2 = obj_id_re.search(result2.observation.lower())
                    if m2:
                        return m2.group(1), receptacle

        return None

    # ------------------------------------------------------------------
    # Step-by-step LLM execution
    # ------------------------------------------------------------------

    @staticmethod
    def _has_cycle(history: list[str], cycle_len: int = 3, repeats: int = 2) -> bool:
        """Detect a deterministic cycle of length *cycle_len* in *history*.

        Returns True when the last ``cycle_len * repeats`` entries are
        ``repeats`` identical copies of the same ``cycle_len`` actions.
        Catches the classic ``A -> B -> C -> A -> B -> C`` deadlock that
        temperature=0 ReAct agents fall into.
        """
        needed = cycle_len * repeats
        if len(history) < needed:
            return False
        window = history[-needed:]
        first = window[:cycle_len]
        for i in range(1, repeats):
            if window[i * cycle_len : (i + 1) * cycle_len] != first:
                return False
        return True

    def _execute_step_by_step(
        self,
        trace: AlfWorldTrace,
        task: Any,
        initial_observation: str,
        admissible_commands: list[str],
        skill_context: str = "",
        policy_directives: list[str] | None = None,
    ) -> None:
        action_history: list[str] = list(trace.action_history)
        observation_history: list[str] = [a.observation for a in trace.actions]
        current_obs = initial_observation
        current_admissible = admissible_commands

        nothing_count = 0
        # API 瞬时失败时保持环境不动并原地重试；基础设施错误不能伪装成
        # admissible[0] 的任务动作，否则各条件会得到被污染的轨迹。
        llm_error_streak = 0
        # LLMClient 内部已完成 provider retry；这里不再重复同一决策，
        # 直接交给外层从任务初始状态做 episode retry。
        MAX_CONSECUTIVE_LLM_ERRORS = 1

        while len(trace.actions) < self.max_steps:
            step_idx = len(trace.actions)
            try:
                step_out = self.generator.step(
                    task_goal=task.goal,
                    observation=current_obs,
                    admissible_commands=current_admissible,
                    action_history=action_history,
                    observation_history=observation_history,
                    # Layer 2 (v4 guideline redesign): inject the skill_context
                    # on EVERY step, not just the first.  Guidelines are ~120
                    # tokens (abstract procedural rules), so per-step overhead
                    # is small, and the LLM benefits from repeated exposure
                    # when a trajectory stretches to 30-50 steps.
                    skill_context=skill_context,
                    policy_directives=policy_directives,
                )
                llm_error_streak = 0
            except Exception as exc:  # noqa: BLE001 — best-effort recovery
                llm_error_streak += 1
                trace.infrastructure_errors.append({
                    "attempt": llm_error_streak,
                    "exception_type": type(exc).__name__,
                    "message": str(exc)[:500],
                    "action_count": len(trace.actions),
                })
                if llm_error_streak >= MAX_CONSECUTIVE_LLM_ERRORS:
                    trace.failure_type = "llm_error"
                    return
                continue

            trace.llm_prompt_tokens_total += step_out.prompt_tokens
            trace.llm_completion_tokens_total += step_out.completion_tokens
            trace.llm_total_tokens_total += (
                step_out.total_tokens
                or step_out.prompt_tokens + step_out.completion_tokens
            )
            trace.llm_call_count += 1
            trace.llm_latency_ms_total += step_out.latency_ms

            result = self.env.step(step_out.action)

            trace.actions.append(AlfWorldAction(
                step=step_idx, action=step_out.action,
                observation=result.observation, score=result.score,
                done=result.done,
                admissible_commands=result.admissible_commands,
            ))
            trace.action_history.append(step_out.action)
            action_history.append(step_out.action)
            observation_history.append(result.observation)

            current_obs = result.observation
            current_admissible = result.admissible_commands

            if result.won:
                trace.success = True
                if len(trace.actions) == 1:
                    trace.first_attempt_success = True
                return
            if result.done:
                # done without won = timeout from Limit wrapper
                return

            if "nothing happens" in result.observation.lower():
                nothing_count += 1
                if nothing_count >= 3:
                    return
            else:
                nothing_count = 0

            # Cycle detection: bail when the LLM is stuck in a deterministic
            # A->B->C->A->B->C loop (no "nothing happens" to trigger above).
            # Require 3 full repeats (9 actions) to avoid killing legitimate
            # exploration that briefly looks repetitive.  Only check after
            # we have at least 12 actions so early progress isn't flagged.
            if (
                len(trace.action_history) >= 12
                and self._has_cycle(trace.action_history, cycle_len=3, repeats=3)
            ):
                trace.failure_type = "action_cycle_detected"
                return

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _finalize(trace: AlfWorldTrace) -> None:
        trace.total_steps = len(trace.actions)
        if trace.actions:
            trace.final_score = trace.actions[-1].score
