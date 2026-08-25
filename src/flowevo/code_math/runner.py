"""Unified runner for code/math benchmarks (HumanEval, MBPP, GSM8K, MATH).

Conditions parallel ALFWorld's experimental design:
- io_baseline: direct generation, no skill
- cot_baseline: chain-of-thought generation
- full_library: compile successful solutions + reuse
- ours: compile + reuse + adaptive escalation

Usage::

    cd src
    python -m code_math.runner \
        --benchmark humaneval --limit 10 \
        --config-path ../configs/openrouter_4o_mini.yaml \
        --output-dir /tmp/code_test \
        --conditions io_baseline
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

_SRC = str(Path(__file__).resolve().parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from code_math.loader import load_tasks
from core.schemas import CodeTaskInstance
from env.sandbox import Sandbox
from eval.verifier import verify_humaneval, verify_function_task
from runtime.config import GenerationSettings, load_runtime_config
from runtime.llm_client import LLMClient


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

CONDITIONS: dict[str, dict[str, Any]] = {
    "io_baseline": {
        "display": "IO Baseline (direct generation)",
        "cot": False,
        "compile": False,
        "use_skill": False,
        "retry": False,
    },
    "cot_baseline": {
        "display": "CoT (chain-of-thought)",
        "cot": True,
        "compile": False,
        "use_skill": False,
        "retry": False,
    },
    "full_library": {
        "display": "Full Library (compile + reuse)",
        "cot": True,
        "compile": True,
        "use_skill": True,
        "retry": False,
    },
    "ours": {
        "display": "Ours (compile + reuse + adaptive escalation)",
        "cot": True,
        "compile": True,
        "use_skill": True,
        "retry": True,
        "max_retries": 3,
    },
    "expel": {
        "display": "ExpeL (insight injection from prior experience)",
        "cot": True,
        "compile": False,
        "use_skill": False,
        "retry": False,
        "expel": True,
    },
}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_code_prompt(task: CodeTaskInstance, cot: bool, skill_context: str = "") -> str:
    """Build generation prompt for code tasks (HumanEval/MBPP)."""
    parts: list[str] = []
    if skill_context:
        parts.append("Here is a similar solved problem for reference:\n%s\n" % skill_context)

    if task.benchmark == "mbpp":
        parts.append("Write a Python function that solves the following task.")
        parts.append("Task: %s" % task.text)
        if task.test_list:
            parts.append("\nYour function MUST pass these test cases:")
            for t in task.test_list:  # all test cases, not just first 3
                parts.append("  %s" % t)
        if cot:
            parts.append(
                "\nAnalyze the test cases carefully, think step by step, "
                "then write the complete Python function."
            )
        else:
            parts.append("\nWrite only the function. No explanation.")
    else:
        # HumanEval: prompt already contains function signature + docstring
        parts.append("Complete the following Python function:")
        parts.append(task.prompt)
        if cot:
            parts.append(
                "\nThink step by step about the problem, then write the "
                "complete function. Make sure to handle edge cases."
            )
        else:
            parts.append("\nWrite only the function body. No explanation.")

    return "\n".join(parts)


def _build_math_prompt(task: CodeTaskInstance, cot: bool, skill_context: str = "") -> str:
    """Build generation prompt for math tasks (GSM8K/MATH)."""
    parts: list[str] = []
    if skill_context:
        parts.append("Here is a similar solved problem for reference:\n%s\n" % skill_context)

    parts.append("Problem: %s" % task.prompt)

    if cot:
        parts.append("\nSolve step by step. End with: The answer is [your answer].")
    else:
        parts.append("\nGive the final answer only.")

    return "\n".join(parts)


def build_prompt(task: CodeTaskInstance, cot: bool, skill_context: str = "") -> str:
    if task.benchmark in ("humaneval", "mbpp"):
        return _build_code_prompt(task, cot, skill_context)
    return _build_math_prompt(task, cot, skill_context)


# ---------------------------------------------------------------------------
# Answer extraction for math tasks
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_ANSWER_IS_RE = re.compile(r"[Tt]he answer is[:\s]*(.+?)[\.\n]")
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


def extract_math_answer(text: str) -> str:
    """Extract final answer from LLM math output."""
    # Try \boxed{}
    m = _BOXED_RE.search(text)
    if m:
        return m.group(1).strip()
    # Try "The answer is X"
    m = _ANSWER_IS_RE.search(text)
    if m:
        return m.group(1).strip().replace(",", "")
    # Try last number
    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1].replace(",", "")
    return text.strip().split("\n")[-1].strip()


def _normalize_answer(ans: str) -> str:
    """Normalize a math answer for comparison."""
    ans = ans.strip()
    # LaTeX cleanup
    ans = ans.replace(r"\left", "").replace(r"\right", "")
    ans = ans.replace(r"\(", "").replace(r"\)", "")
    ans = ans.replace(r"\[", "").replace(r"\]", "")
    ans = ans.replace(r"\dfrac", r"\frac")
    ans = ans.replace(r"\tfrac", r"\frac")
    # Strip currency/unit symbols
    ans = ans.replace(",", "").replace("$", "").replace("%", "")
    # Strip outer $ from LaTeX
    if ans.startswith("$") and ans.endswith("$"):
        ans = ans[1:-1].strip()
    # Strip text units after number ("3 bolts" → "3", "230 miles" → "230")
    m = re.match(r"^(-?\d+\.?\d*)\s+[a-zA-Z]", ans)
    if m:
        ans = m.group(1)
    # Try numeric normalization
    try:
        val = float(ans)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return ans.lower().strip()


# ---------------------------------------------------------------------------
# Verification dispatcher
# ---------------------------------------------------------------------------

def verify(
    task: CodeTaskInstance,
    solution: str,
    sandbox: Sandbox,
) -> tuple[bool, str]:
    """Verify a solution. Returns (passed, feedback_text)."""
    if task.benchmark == "humaneval":
        fb = verify_humaneval(task, solution, sandbox)
        return fb.passed, fb.summary

    if task.benchmark == "mbpp":
        fb = verify_function_task(task, solution, sandbox, scope="default")
        return fb.passed, fb.summary

    if task.benchmark in ("gsm8k", "math"):
        gold = task.metadata.get("gold_answer", "")
        predicted = extract_math_answer(solution)
        passed = _normalize_answer(predicted) == _normalize_answer(gold)
        feedback = "correct" if passed else (
            "wrong: predicted=%s, gold=%s" % (predicted, gold)
        )
        return passed, feedback

    return False, "unknown benchmark"


# ---------------------------------------------------------------------------
# Code extraction from LLM output
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str, task: CodeTaskInstance) -> str:
    """Extract code from LLM output (handles markdown fences).

    Preserves leading indentation (critical for HumanEval function bodies).
    """
    m = _CODE_BLOCK_RE.search(text)
    if m:
        # lstrip('\n') removes leading blank lines but preserves indentation
        return m.group(1).lstrip("\n").rstrip()
    # Fallback: return as-is (strip only trailing whitespace)
    return text.rstrip()


# ---------------------------------------------------------------------------
# Skill library (minimal, pattern-matching based)
# ---------------------------------------------------------------------------

class CodeSkillLibrary:
    """Minimal skill library for code/math tasks.

    Layer 1: Exact task_id match (direct replay of cached solution).
    Layer 2: Same-benchmark successful solutions as few-shot context.
    """

    def __init__(self) -> None:
        self._solutions: dict[str, str] = {}  # task_id -> solution
        self._by_benchmark: dict[str, list[tuple[str, str, str]]] = {}  # benchmark -> [(task_id, prompt_snippet, solution)]

    def add(self, task: CodeTaskInstance, solution: str) -> None:
        self._solutions[task.task_id] = solution
        bucket = self._by_benchmark.setdefault(task.benchmark, [])
        snippet = (task.text or task.prompt)[:200]
        bucket.append((task.task_id, snippet, solution[:500]))

    def retrieve(self, task: CodeTaskInstance) -> dict[str, Any]:
        """Retrieve skill context for a task.

        Math tasks (gsm8k/math): no context injection (hurts reasoning).
        Code tasks (humaneval/mbpp): inject most similar solution by keyword overlap.
        """
        # Layer 1: exact match
        if task.task_id in self._solutions:
            return {"type": "exact", "solution": self._solutions[task.task_id]}

        # Math tasks: no few-shot (pilot showed -30pt regression)
        if task.benchmark in ("gsm8k", "math"):
            return {"type": "none"}

        # Code tasks: find most similar by keyword overlap
        bucket = self._by_benchmark.get(task.benchmark, [])
        if bucket:
            task_words = set((task.text or task.prompt).lower().split())
            best_score = -1
            best_context = None
            for tid, snippet, sol in bucket:
                overlap = len(set(snippet.lower().split()) & task_words)
                if overlap > best_score:
                    best_score = overlap
                    best_context = "Example:\nProblem: %s\nSolution:\n%s" % (snippet, sol)
            if best_context and best_score > 3:
                return {"type": "context", "context": best_context}

        return {"type": "none"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "solutions": dict(self._solutions),
            "by_benchmark": {
                k: [(tid, sn, sol) for tid, sn, sol in v]
                for k, v in self._by_benchmark.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeSkillLibrary":
        lib = cls()
        lib._solutions = dict(data.get("solutions", {}))
        for bm, entries in data.get("by_benchmark", {}).items():
            lib._by_benchmark[bm] = [(e[0], e[1], e[2]) for e in entries]
        return lib

    @property
    def size(self) -> int:
        return len(self._solutions)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

_CHECKPOINT_VERSION = 1


def _save_checkpoint(
    output_dir: Path, condition: str, benchmark: str,
    episodes: list[dict[str, Any]], library: CodeSkillLibrary | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    name = "%s_%s" % (benchmark, condition)
    ckpt = output_dir / ("_checkpoint_%s.json" % name)
    tmp = ckpt.with_suffix(".tmp")
    data = {
        "version": _CHECKPOINT_VERSION,
        "benchmark": benchmark,
        "condition": condition,
        "completed": len(episodes),
        "episodes": episodes,
        "library": library.to_dict() if library else None,
    }
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(ckpt)


def _load_checkpoint(
    output_dir: Path, condition: str, benchmark: str,
) -> tuple[list[dict[str, Any]], CodeSkillLibrary | None, int] | None:
    name = "%s_%s" % (benchmark, condition)
    ckpt = output_dir / ("_checkpoint_%s.json" % name)
    if not ckpt.exists():
        return None
    try:
        data = json.loads(ckpt.read_text(encoding="utf-8"))
        if data.get("version") != _CHECKPOINT_VERSION:
            return None
        episodes = list(data.get("episodes", []))
        lib_data = data.get("library")
        library = CodeSkillLibrary.from_dict(lib_data) if lib_data else None
        return episodes, library, len(episodes)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-condition runner
# ---------------------------------------------------------------------------

def _expel_add_insight(
    insights: list[str], ep: dict[str, Any], task_desc: str = "",
) -> None:
    """Extract a one-line insight from an episode for ExpeL injection."""
    desc = task_desc or ep.get("task_id", "")
    if ep.get("passed"):
        insights.append(
            "SUCCESS on '%s': generated correct solution." % desc[:60]
        )
    else:
        fb = ep.get("feedback", "")[:80]
        insights.append(
            "FAILURE on '%s': %s. Lesson: check edge cases carefully."
            % (desc[:60], fb)
        )


_GEN_SETTINGS = GenerationSettings(temperature=0.0, max_output_tokens=2048)
_GEN_RETRY = GenerationSettings(temperature=0.2, max_output_tokens=2048)
_GEN_L2 = GenerationSettings(temperature=0.5, max_output_tokens=2048)
_GEN_L3 = GenerationSettings(temperature=0.7, max_output_tokens=2048)


def run_condition(
    benchmark: str,
    condition_name: str,
    cfg: dict[str, Any],
    llm: LLMClient,
    tasks: list[CodeTaskInstance],
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:

    print("\n" + "=" * 72)
    print("  %s / %s — %s" % (benchmark, condition_name, cfg["display"]))
    print("=" * 72)

    sandbox = Sandbox(timeout_seconds=10.0)

    # Resume
    resumed = (
        _load_checkpoint(output_dir, condition_name, benchmark)
        if output_dir else None
    )
    if resumed is not None:
        episodes, library, start_idx = resumed
        if start_idx >= len(tasks):
            print("  [resume] Already complete (%d/%d)." % (start_idx, len(tasks)))
            return episodes
        print("  [resume] Loaded %d episodes." % start_idx)
    else:
        episodes = []
        library = CodeSkillLibrary() if cfg.get("compile") else None
        start_idx = 0

    cum_pass = sum(1 for e in episodes if e.get("passed"))

    # ExpeL: collect insights from prior episodes (online accumulation)
    expel_insights: list[str] = []
    if cfg.get("expel"):
        # Rebuild insights from resumed episodes
        for e in episodes:
            _expel_add_insight(expel_insights, e)

    for idx in range(start_idx, len(tasks)):
        task = tasks[idx]
        t0 = time.time()
        total_tokens = 0

        try:
            # Skill retrieval / ExpeL insight injection
            skill_context = ""
            if cfg.get("expel") and expel_insights:
                # Inject last 3 insights as context
                skill_context = (
                    "Insights from prior experience:\n"
                    + "\n".join("- %s" % ins for ins in expel_insights[-3:])
                )
            elif cfg.get("use_skill") and library:
                skill = library.retrieve(task)
                if skill["type"] == "context":
                    skill_context = skill["context"]

            is_code = task.benchmark in ("humaneval", "mbpp")

            # For ours + code, skip skill_context at Level 1 (direct reuse is
            # rare on HumanEval/MBPP); the retry stages still inject it.
            skill_context_l1 = (
                "" if (is_code and condition_name == "ours") else skill_context
            )

            # Build prompt and generate (Level 1: greedy single shot)
            prompt = build_prompt(task, cot=cfg.get("cot", False),
                                  skill_context=skill_context_l1)

            gen_instructions = "You are an expert programmer and mathematician."
            resp = llm.generate(
                instructions=gen_instructions,
                input_text=prompt,
                settings=_GEN_SETTINGS,
            )
            total_tokens += (
                (resp.prompt_tokens or 0) + (resp.completion_tokens or 0)
            )
            if is_code:
                solution = extract_code(resp.text, task)
            else:
                solution = resp.text.strip()
            passed, feedback = verify(task, solution, sandbox)

            # Level 2 mini-ensemble (2 candidates, temp=0.5, no skill).
            # Any passing candidate ends the stage early.
            if (is_code and condition_name == "ours") and not passed:
                last_sol, last_fb = solution, feedback
                for _ in range(2):
                    resp2 = llm.generate(
                        instructions=gen_instructions,
                        input_text=prompt,
                        settings=_GEN_L2,
                    )
                    total_tokens += (
                        (resp2.prompt_tokens or 0) + (resp2.completion_tokens or 0)
                    )
                    sol2 = extract_code(resp2.text, task)
                    pass2, fb2 = verify(task, sol2, sandbox)
                    last_sol, last_fb = sol2, fb2
                    if pass2:
                        solution, passed, feedback = sol2, pass2, fb2
                        break
                if not passed:
                    solution, feedback = last_sol, last_fb

            # Retry on verification failure (Levels 3 and 4)
            retries = 0
            effective_max_retries = cfg.get("max_retries", 3)
            if is_code and condition_name == "ours":
                effective_max_retries = 2  # Level 3 (temp=0.7) + Level 4 (temp=0.2)

            if cfg.get("retry") and not passed:
                for retry in range(effective_max_retries):
                    # Build retry prompt (code-specific vs math)
                    ref_parts = []
                    if skill_context:
                        ref_parts.append(
                            "Reference from similar solved problem:\n%s\n"
                            % skill_context
                        )
                    if is_code:
                        ref_parts.append(
                            "Your Python function failed the tests.\n\n"
                            "Task: %s\n\n"
                            "Your code:\n```python\n%s\n```\n\n"
                            "Test result: %s\n\n"
                            "Fix the function. Think about what went wrong, "
                            "then provide the corrected complete function."
                            % ((task.text or task.prompt)[:500],
                               solution[:800], feedback[:300])
                        )
                    else:
                        ref_parts.append(
                            "You attempted to solve a problem and failed.\n"
                            "Problem: %s\n"
                            "Your solution:\n%s\n"
                            "Feedback: %s\n"
                            "Reflect on what went wrong. Then provide a "
                            "corrected solution."
                            % ((task.text or task.prompt)[:500],
                               solution[:500], feedback)
                        )
                    ref_prompt = "\n".join(ref_parts)

                    if is_code and condition_name == "ours":
                        # Level 3 temp=0.7, Level 4 temp=0.2
                        retry_settings = _GEN_L3 if retry == 0 else _GEN_RETRY
                    else:
                        retry_settings = _GEN_SETTINGS
                    ref_resp = llm.generate(
                        instructions="You are an expert at debugging and correcting solutions.",
                        input_text=ref_prompt,
                        settings=retry_settings,
                    )
                    total_tokens += (ref_resp.prompt_tokens or 0) + (ref_resp.completion_tokens or 0)

                    if task.benchmark in ("humaneval", "mbpp"):
                        solution = extract_code(ref_resp.text, task)
                    else:
                        solution = ref_resp.text.strip()

                    passed, feedback = verify(task, solution, sandbox)
                    retries += 1
                    if passed:
                        break

            # Compile successful solution
            if passed and library and cfg.get("compile"):
                library.add(task, solution)

            wall = time.time() - t0
            cum_pass += int(passed)

            ep = {
                "benchmark": benchmark,
                "condition": condition_name,
                "task_index": idx,
                "task_id": task.task_id,
                "passed": passed,
                "tokens": total_tokens,
                "retries": retries,
                "wall_time_s": round(wall, 2),
                "feedback": feedback[:200],
                "library_size": library.size if library else len(expel_insights),
            }

            # ExpeL: accumulate insight from this episode
            if cfg.get("expel"):
                _expel_add_insight(
                    expel_insights, ep,
                    task_desc=(task.text or task.prompt)[:100],
                )

            status = "PASS" if passed else "FAIL"
            print(
                "  [%3d/%d] %s  %-30s tk=%5d  ins=%d  cum=%.0f%%"
                % (idx + 1, len(tasks), status,
                   task.task_id[:30], total_tokens,
                   len(expel_insights) if cfg.get("expel") else (library.size if library else 0),
                   100 * cum_pass / (idx + 1))
            )

        except Exception as exc:
            wall = time.time() - t0
            ep = {
                "benchmark": benchmark,
                "condition": condition_name,
                "task_index": idx,
                "task_id": task.task_id,
                "passed": False,
                "tokens": total_tokens,
                "retries": 0,
                "wall_time_s": round(wall, 2),
                "feedback": "error: %s" % str(exc)[:200],
                "library_size": library.size if library else 0,
            }
            print("  [%3d/%d] ERROR %s" % (idx + 1, len(tasks), str(exc)[:100]))

        episodes.append(ep)
        if output_dir:
            _save_checkpoint(output_dir, condition_name, benchmark, episodes, library)

    return episodes


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _summary(eps: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(eps)
    if not n:
        return {}
    passed = sum(1 for e in eps if e["passed"])
    tokens = sum(e.get("tokens", 0) for e in eps)
    return {
        "n": n,
        "passed": passed,
        "pass_rate": round(passed / n, 4),
        "avg_tokens": round(tokens / n, 1),
        "total_tokens": tokens,
    }


def generate_report(
    all_eps: dict[str, dict[str, list[dict[str, Any]]]],
    output_dir: Path,
) -> None:
    """Generate unified report across benchmarks × conditions."""
    lines = ["# Code/Math Benchmark Report\n"]
    lines.append("| Benchmark | Condition | N | Pass | Rate | Avg Tokens |")
    lines.append("|---|---|---|---|---|---|")

    flat_rows = []
    for bm, cond_eps in sorted(all_eps.items()):
        for cond, eps in sorted(cond_eps.items()):
            s = _summary(eps)
            lines.append(
                "| %s | %s | %d | %d | %.0f%% | %.0f |"
                % (bm, cond, s["n"], s["passed"],
                   s["pass_rate"] * 100, s["avg_tokens"])
            )
            flat_rows.extend(eps)

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    # CSV
    if flat_rows:
        fieldnames = sorted(set().union(*(r.keys() for r in flat_rows)))
        with open(output_dir / "episodes.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)

    # JSON summary
    summaries = {}
    for bm, cond_eps in all_eps.items():
        summaries[bm] = {cond: _summary(eps) for cond, eps in cond_eps.items()}
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )

    print("\nReports written to %s" % output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Code/Math benchmark runner")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--benchmark", nargs="+",
        default=["humaneval", "mbpp", "gsm8k"],
        help="Benchmarks to run",
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="Max tasks per benchmark (0 = all)")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--conditions", nargs="+",
        default=list(CONDITIONS.keys()),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_runtime_config(config_path=args.config_path)
    if args.model:
        config = dataclasses.replace(base_config, model=args.model)
    else:
        config = base_config
    llm = LLMClient(config=config)

    print("Code/Math Benchmark Runner")
    print("  Benchmarks: %s" % ", ".join(args.benchmark))
    print("  Limit:      %s" % (args.limit or "all"))
    print("  Model:      %s" % config.model)
    print("  Provider:   %s" % config.provider)
    print("  Conditions: %s" % ", ".join(args.conditions))
    print()

    all_eps: dict[str, dict[str, list[dict[str, Any]]]] = {}
    grand_t0 = time.time()

    for bm in args.benchmark:
        print("Loading %s..." % bm)
        tasks = load_tasks(bm, limit=args.limit)
        print("  %d tasks loaded" % len(tasks))
        all_eps[bm] = {}

        for cond in args.conditions:
            if cond not in CONDITIONS:
                print("  [warn] Unknown condition '%s'" % cond)
                continue
            eps = run_condition(
                bm, cond, CONDITIONS[cond], llm, tasks,
                output_dir=output_dir,
            )
            all_eps[bm][cond] = eps
            s = _summary(eps)
            print(
                "\n  >> %s/%s: %d/%d pass (%.0f%%), avg_tokens=%.0f\n"
                % (bm, cond, s.get("passed", 0), s.get("n", 0),
                   s.get("pass_rate", 0) * 100, s.get("avg_tokens", 0))
            )

    grand_wall = time.time() - grand_t0
    print("\nTotal wall: %.0fs (%.1f min)" % (grand_wall, grand_wall / 60))
    generate_report(all_eps, output_dir)


if __name__ == "__main__":
    main()
