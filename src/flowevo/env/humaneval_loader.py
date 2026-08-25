"""Load HumanEval tasks via Hugging Face datasets."""

from __future__ import annotations

import json
from pathlib import Path

from core.schemas import CodeTaskInstance
from env.local_datasets import load_dataset_local_first


HUMANEVAL_DEFAULT_SPLIT = "test"


def _read_manifest_task_ids(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("task_ids", "ids", "selected_task_ids"):
                values = data.get(key)
                if isinstance(values, list):
                    return [str(item).strip() for item in values if str(item).strip()]
            raise ValueError(f"Unsupported HumanEval manifest JSON structure: {path}")
        if isinstance(data, list):
            task_ids: list[str] = []
            for item in data:
                if isinstance(item, dict):
                    task_ids.append(str(item.get("task_id", "")).strip())
                else:
                    task_ids.append(str(item).strip())
            return [task_id for task_id in task_ids if task_id]
        raise ValueError(f"Unsupported HumanEval manifest JSON structure: {path}")

    task_ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if cleaned and not cleaned.startswith("#"):
            task_ids.append(cleaned)
    return task_ids


def _resolve_dataset_split(dataset_split: str) -> tuple[str, set[str] | None]:
    cleaned = (dataset_split or "").strip()
    if not cleaned:
        return HUMANEVAL_DEFAULT_SPLIT, None

    split_path = Path(cleaned).expanduser()
    if split_path.is_file():
        task_ids = set(_read_manifest_task_ids(split_path))
        if not task_ids:
            raise ValueError(f"HumanEval manifest {split_path} did not contain any task ids")
        return HUMANEVAL_DEFAULT_SPLIT, task_ids

    if split_path.suffix and not split_path.exists():
        raise FileNotFoundError(f"HumanEval dataset split or manifest not found: {dataset_split}")

    return cleaned, None


def _task_allowed(task_id: str, allowed_task_ids: set[str] | None) -> bool:
    if allowed_task_ids is None:
        return True
    return task_id in allowed_task_ids


def load_humaneval_tasks(
    profile: str = "full",
    limit: int | None = None,
    dataset_split: str = "",
) -> list[CodeTaskInstance]:
    """Load HumanEval and convert rows into `CodeTaskInstance` objects.

    Args:
        profile: HumanEval only supports `full`.
        limit: Optional maximum number of tasks.
        dataset_split: A Hugging Face split name, or a path to a manifest file
            containing task ids to keep.

    Returns:
        A list of unified code task instances.
    """
    if limit is not None and limit <= 0:
        limit = None

    if profile != "full":
        raise ValueError(f"Unsupported HumanEval profile: {profile}")

    source_split, allowed_task_ids = _resolve_dataset_split(dataset_split)
    dataset, dataset_source = load_dataset_local_first(
        "openai/openai_humaneval", split=source_split)
    resolved_dataset_split = dataset_split.strip() or source_split
    tasks: list[CodeTaskInstance] = []

    for row in dataset:
        task_id = str(row["task_id"])
        if not _task_allowed(task_id, allowed_task_ids):
            continue
        tasks.append(
            CodeTaskInstance(
                task_id=task_id,
                benchmark="humaneval",
                prompt=row["prompt"],
                canonical_solution=row["canonical_solution"],
                test=row["test"],
                entry_point=row["entry_point"],
                metadata={
                    "dataset_split": resolved_dataset_split,
                    "dataset_source": dataset_source,
                    "profile": profile,
                },
            )
        )
        if limit is not None and len(tasks) >= limit:
            break
    return tasks
