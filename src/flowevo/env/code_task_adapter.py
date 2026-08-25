"""Legacy compatibility wrapper for code-generation benchmark loading."""

from __future__ import annotations

from core.schemas import CodeTaskInstance
from env.benchmark_adapter import SUPPORTED_BENCHMARKS, load_benchmark_tasks


def load_code_tasks(
    benchmark: str,
    profile: str = "full",
    limit: int | None = None,
    dataset_split: str | None = None,
) -> list[CodeTaskInstance]:
    tasks = load_benchmark_tasks(benchmark=benchmark, profile=profile, limit=limit, dataset_split=dataset_split)
    return [task for task in tasks if isinstance(task, CodeTaskInstance)]
