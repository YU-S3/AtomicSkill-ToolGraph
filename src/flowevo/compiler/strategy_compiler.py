"""Compile strategy descriptions from successful traces and skills.

Supports two modes:
- Heuristic (default): code-pattern detection, no LLM calls.
- LLM-enhanced: uses the runtime LLM to produce high-quality strategy
  descriptions from successful code.  Falls back to heuristic on error.
"""

from __future__ import annotations

import re
from typing import Any

from core.schemas import ExecutionTrace, SkillCard
from core.strategy import StrategyCard


# Common code patterns and their human-readable labels.
_PATTERN_DETECTORS: list[tuple[str, str]] = [
    ("sorting", "sorted("),
    ("sorting", ".sort("),
    ("frequency counting", "Counter("),
    ("frequency counting", "collections.Counter"),
    ("set operations", "set("),
    ("set operations", "frozenset("),
    ("regex matching", "re."),
    ("dictionary mapping", "defaultdict("),
    ("mathematical computation", "math."),
    ("string manipulation", ".split("),
    ("string manipulation", ".join("),
    ("string manipulation", ".replace("),
    ("list comprehension", " for "),
    ("recursive approach", "def _"),
]

_NESTED_LOOP = re.compile(r"for\s+\w+.*:\s*\n(?:\s+.*\n)*?\s+for\s+\w+.*:")
_LAMBDA = re.compile(r"\blambda\b")

_LLM_SYSTEM = (
    "You are a programming strategy analyst. "
    "Extract reusable problem-solving strategies from successful code. "
    "Be concise and specific."
)

_LLM_PROMPT_TEMPLATE = """\
Analyze this Python function that successfully solved a coding task.
Extract a reusable problem-solving strategy (NOT the code itself).

Task pattern: {task_pattern}
Task description: {task_text}

Successful code:
```python
{code}
```

Respond in exactly this format:
APPROACH: [One sentence describing the core algorithmic approach]
STRATEGY: [2-3 sentences describing the reusable problem-solving strategy. \
Focus on WHEN to use this approach and WHY it works, not HOW to implement it. \
Be specific about the problem characteristics that make this approach effective.]
APPLICABILITY: [One sentence describing what types of problems this strategy applies to]"""


class StrategyCompiler:
    """Extract a StrategyCard from a successful trace + compiled skill.

    If *llm_client* is provided, strategy descriptions are generated via
    an LLM call.  Otherwise a fast heuristic is used.
    """

    def __init__(self, llm_client: Any = None, llm_settings: Any = None) -> None:
        self._llm_client = llm_client
        self._llm_settings = llm_settings

    def compile_strategy(
        self,
        trace: ExecutionTrace,
        skill_card: SkillCard,
        skill_code: str,
    ) -> StrategyCard | None:
        if not trace.success:
            return None
        if not skill_code.strip():
            return None

        task_pattern = trace.task_pattern or "unknown"

        if self._llm_client is not None and self._llm_settings is not None:
            card = self._compile_with_llm(trace, skill_card, skill_code, task_pattern)
            if card is not None:
                return card
            # LLM failed — fall through to heuristic

        return self._compile_heuristic(trace, skill_card, skill_code, task_pattern)

    # ------------------------------------------------------------------
    # Heuristic mode (no LLM)
    # ------------------------------------------------------------------

    def _compile_heuristic(
        self,
        trace: ExecutionTrace,
        skill_card: SkillCard,
        skill_code: str,
        task_pattern: str,
    ) -> StrategyCard:
        approach_summary = self._extract_approach(skill_code, task_pattern)
        description = self._build_description(trace, skill_card, skill_code)
        applicability = self._infer_applicability(skill_card, task_pattern)

        return StrategyCard(
            strategy_id="strategy_%s" % skill_card.skill_id,
            source_skill_id=skill_card.skill_id,
            task_pattern=task_pattern,
            description=description,
            applicability=applicability,
            approach_summary=approach_summary,
            source_benchmark=trace.benchmark,
            source_task_id=trace.task_id,
        )

    # ------------------------------------------------------------------
    # LLM-enhanced mode
    # ------------------------------------------------------------------

    def _compile_with_llm(
        self,
        trace: ExecutionTrace,
        skill_card: SkillCard,
        skill_code: str,
        task_pattern: str,
    ) -> StrategyCard | None:
        task_text = str(getattr(trace, "query", "") or "").strip()
        if not task_text:
            task_text = task_pattern

        prompt = _LLM_PROMPT_TEMPLATE.format(
            task_pattern=task_pattern,
            task_text=task_text[:500],
            code=skill_code[:1500],
        )

        try:
            response = self._llm_client.generate(
                instructions=_LLM_SYSTEM,
                input_text=prompt,
                settings=self._llm_settings,
            )
            text = response.text if hasattr(response, "text") else str(response)
        except Exception:
            return None

        approach = ""
        description = ""
        applicability = ""

        for line in text.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("APPROACH:"):
                approach = line[len("APPROACH:"):].strip()
            elif line.upper().startswith("STRATEGY:"):
                description = line[len("STRATEGY:"):].strip()
            elif line.upper().startswith("APPLICABILITY:"):
                applicability = line[len("APPLICABILITY:"):].strip()

        if not approach or not description:
            return None

        return StrategyCard(
            strategy_id="strategy_%s" % skill_card.skill_id,
            source_skill_id=skill_card.skill_id,
            task_pattern=task_pattern,
            description=description,
            applicability=applicability or self._infer_applicability(skill_card, task_pattern),
            approach_summary=approach,
            source_benchmark=trace.benchmark,
            source_task_id=trace.task_id,
        )

    # ------------------------------------------------------------------
    # Heuristic helpers
    # ------------------------------------------------------------------

    def _extract_approach(self, code: str, task_pattern: str) -> str:
        detected: list[str] = []
        seen: set[str] = set()
        for label, marker in _PATTERN_DETECTORS:
            if label in seen:
                continue
            if isinstance(marker, str) and marker in code:
                detected.append(label)
                seen.add(label)

        if _NESTED_LOOP.search(code):
            if "nested iteration" not in seen:
                detected.append("nested iteration")
        if _LAMBDA.search(code):
            if "lambda-based transformation" not in seen:
                detected.append("lambda-based transformation")
        if code.count("def ") > 1:
            if "helper function decomposition" not in seen:
                detected.append("helper function decomposition")

        if detected:
            return "Uses %s for %s tasks" % (", ".join(detected[:3]), task_pattern)
        return "Direct implementation for %s tasks" % task_pattern

    def _build_description(
        self,
        trace: ExecutionTrace,
        skill_card: SkillCard,
        code: str,
    ) -> str:
        lines = code.strip().split("\n")
        parts: list[str] = []

        sig_line = next((l for l in lines if l.strip().startswith("def ")), "")
        if sig_line:
            parts.append("Function signature: %s" % sig_line.strip())

        docstring = self._extract_docstring(lines)
        if docstring:
            parts.append("Purpose: %s" % docstring)

        code_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]
        parts.append("Implementation complexity: %d lines" % len(code_lines))

        return ". ".join(parts)

    @staticmethod
    def _extract_docstring(lines: list[str]) -> str:
        in_docstring = False
        docstring_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if '"""' in stripped or "'''" in stripped:
                if in_docstring:
                    break
                in_docstring = True
                content = stripped.replace('"""', "").replace("'''", "").strip()
                if content:
                    docstring_lines.append(content)
                continue
            if in_docstring:
                if stripped:
                    docstring_lines.append(stripped)
        return " ".join(docstring_lines[:2])

    def _infer_applicability(self, skill_card: SkillCard, task_pattern: str) -> str:
        conditions = ["Task pattern: %s" % task_pattern]

        sig = skill_card.signature if isinstance(skill_card.signature, dict) else {}
        args = sig.get("args") or []
        if isinstance(args, list) and args:
            conditions.append("Functions taking %d parameter(s)" % len(args))
        entry = str(sig.get("entry_point", "") or "")
        if entry:
            conditions.append("Entry point style: %s" % entry)

        return "; ".join(conditions)
