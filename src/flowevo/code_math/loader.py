"""Unified loader for code/math benchmarks.

Loads HumanEval, MBPP, GSM8K, and MATH into a common CodeTaskInstance format.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

_SRC = str(Path(__file__).resolve().parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from core.schemas import CodeTaskInstance


# ---------------------------------------------------------------------------
# MATH answer extraction
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


def _extract_math_answer(solution: str) -> str:
    """Extract the answer from a MATH solution's \\boxed{} markup."""
    m = _BOXED_RE.search(solution)
    if m:
        return m.group(1).strip()
    # Fallback: last line
    lines = solution.strip().split("\n")
    return lines[-1].strip() if lines else ""


# ---------------------------------------------------------------------------
# GSM8K answer extraction
# ---------------------------------------------------------------------------

_GSM8K_ANSWER_RE = re.compile(r"####\s*([^\n]+)")


def _extract_gsm8k_answer(answer_text: str) -> str:
    m = _GSM8K_ANSWER_RE.search(answer_text)
    if m:
        return m.group(1).strip().replace(",", "")
    return answer_text.strip()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_humaneval(limit: int = 0) -> list[CodeTaskInstance]:
    """Load HumanEval tasks."""
    from env.local_datasets import load_dataset_local_first
    ds, dataset_source = load_dataset_local_first(
        "openai/openai_humaneval", split="test")
    tasks = []
    for row in ds:
        tasks.append(CodeTaskInstance(
            task_id=row["task_id"],
            benchmark="humaneval",
            prompt=row["prompt"],
            canonical_solution=row["canonical_solution"],
            test=row["test"],
            entry_point=row["entry_point"],
            metadata={"dataset_source": dataset_source},
        ))
    if limit > 0:
        tasks = tasks[:limit]
    return tasks


def load_mbpp(limit: int = 0) -> list[CodeTaskInstance]:
    """Load MBPP tasks (sanitized split)."""
    from datasets import load_dataset
    ds = load_dataset("mbpp", split="test")
    tasks = []
    for i, row in enumerate(ds):
        tasks.append(CodeTaskInstance(
            task_id="mbpp_%d" % row.get("task_id", i),
            benchmark="mbpp",
            prompt="",
            text=row["text"],
            code=row["code"],
            test_list=list(row.get("test_list", [])),
            entry_point="",
        ))
    if limit > 0:
        tasks = tasks[:limit]
    return tasks


def load_gsm8k(limit: int = 0) -> list[CodeTaskInstance]:
    """Load GSM8K tasks."""
    from env.local_datasets import load_dataset_local_first
    ds, dataset_source = load_dataset_local_first(
        "openai/gsm8k", config="main", split="test")
    tasks = []
    for i, row in enumerate(ds):
        gold_answer = _extract_gsm8k_answer(row["answer"])
        tasks.append(CodeTaskInstance(
            task_id="gsm8k_%d" % i,
            benchmark="gsm8k",
            prompt=row["question"],
            canonical_solution=row["answer"],
            text=row["question"],
            metadata={"gold_answer": gold_answer,
                      "dataset_source": dataset_source},
        ))
    if limit > 0:
        tasks = tasks[:limit]
    return tasks


_MATH_CONFIGS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]


def load_math(limit: int = 0) -> list[CodeTaskInstance]:
    """Load MATH tasks (all categories)."""
    from datasets import load_dataset
    tasks = []
    for config in _MATH_CONFIGS:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", config, split="test")
        except Exception:
            continue
        for i, row in enumerate(ds):
            gold_answer = _extract_math_answer(row["solution"])
            tasks.append(CodeTaskInstance(
                task_id="math_%s_%d" % (config, i),
                benchmark="math",
                prompt=row["problem"],
                canonical_solution=row["solution"],
                text=row["problem"],
                metadata={
                    "gold_answer": gold_answer,
                    "level": row.get("level", ""),
                    "type": row.get("type", config),
                },
            ))
    if limit > 0:
        tasks = tasks[:limit]
    return tasks


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

BENCHMARKS = {
    "humaneval": load_humaneval,
    "mbpp": load_mbpp,
    "gsm8k": load_gsm8k,
    "math": load_math,
}


def load_tasks(benchmark: str, limit: int = 0) -> list[CodeTaskInstance]:
    """Load tasks for a given benchmark."""
    loader = BENCHMARKS.get(benchmark)
    if loader is None:
        raise ValueError("Unknown benchmark: %s (available: %s)"
                         % (benchmark, ", ".join(BENCHMARKS)))
    return loader(limit=limit)
