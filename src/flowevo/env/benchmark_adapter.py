"""Unified benchmark adapter helpers for executable benchmarks."""

from __future__ import annotations

from core.schemas import BenchmarkTaskInstance, CodeTaskInstance


SUPPORTED_BENCHMARKS = ("mbpp", "humaneval")


def load_benchmark_tasks(
    benchmark: str,
    *,
    profile: str = "full",
    limit: int | None = None,
    dataset_split: str | None = None,
) -> list[BenchmarkTaskInstance]:
    if benchmark == "mbpp":
        from env.mbpp_loader import load_mbpp_tasks

        return load_mbpp_tasks(profile=profile, limit=limit, dataset_split=dataset_split or "")
    if benchmark == "humaneval":
        from env.humaneval_loader import load_humaneval_tasks

        return load_humaneval_tasks(profile=profile, limit=limit, dataset_split=dataset_split or "")
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def load_code_tasks(
    benchmark: str,
    profile: str = "full",
    limit: int | None = None,
    dataset_split: str | None = None,
) -> list[CodeTaskInstance]:
    tasks = load_benchmark_tasks(benchmark=benchmark, profile=profile, limit=limit, dataset_split=dataset_split)
    return [task for task in tasks if isinstance(task, CodeTaskInstance)]
