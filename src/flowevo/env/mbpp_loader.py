"""Load MBPP tasks via Hugging Face datasets."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from datasets import load_dataset

from core.schemas import CodeTaskInstance


MBPP_STANDARD500_START = 11
MBPP_STANDARD500_END = 510
MBPP_DEFAULT_SPLIT = "test"
MBPP_DEFAULT_SCORE_PROTOCOL = "table1_compatible"


def _matches_profile(task_id: int, profile: str) -> bool:
    """Check whether one MBPP task belongs to the selected profile."""
    if profile == "full":
        return True
    if profile == "standard500":
        return MBPP_STANDARD500_START <= task_id <= MBPP_STANDARD500_END
    raise ValueError(f"Unsupported MBPP profile: {profile}")


def _read_manifest_task_ids(path: Path) -> list[str]:
    """Read task ids from a manifest file."""
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            records = data.get("records")
            if isinstance(records, list):
                task_ids = [str((item or {}).get("task_id", "")).strip() for item in records if isinstance(item, dict)]
                return [task_id for task_id in task_ids if task_id]
            for key in ("task_ids", "ids", "selected_task_ids"):
                values = data.get(key)
                if isinstance(values, list):
                    return [str(item).strip() for item in values if str(item).strip()]
            raise ValueError(f"Unsupported MBPP manifest JSON structure: {path}")
        if isinstance(data, list):
            task_ids: list[str] = []
            for item in data:
                if isinstance(item, dict):
                    task_ids.append(str(item.get("task_id", "")).strip())
                else:
                    task_ids.append(str(item).strip())
            return [task_id for task_id in task_ids if task_id]
        raise ValueError(f"Unsupported MBPP manifest JSON structure: {path}")

    task_ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if cleaned and not cleaned.startswith("#"):
            task_ids.append(cleaned)
    return task_ids


def _resolve_dataset_split(dataset_split: str) -> tuple[str, set[str] | None]:
    """Resolve a split name or manifest path into a usable MBPP selection."""
    cleaned = (dataset_split or "").strip()
    if not cleaned:
        return MBPP_DEFAULT_SPLIT, None

    split_path = Path(cleaned).expanduser()
    if split_path.is_file():
        task_ids = set(_read_manifest_task_ids(split_path))
        if not task_ids:
            raise ValueError(f"MBPP manifest {split_path} did not contain any task ids")
        return MBPP_DEFAULT_SPLIT, task_ids

    if split_path.suffix and not split_path.exists():
        raise FileNotFoundError(f"MBPP dataset split or manifest not found: {dataset_split}")

    return cleaned, None


def _task_allowed(task_id: str, allowed_task_ids: set[str] | None) -> bool:
    if allowed_task_ids is None:
        return True
    return task_id in allowed_task_ids


def _normalize_tests(raw_tests: list[object] | None) -> list[str]:
    return [str(test).strip() for test in list(raw_tests or []) if str(test).strip()]


def split_mbpp_tests(
    *,
    test_list: list[object] | None,
    challenge_test_list: list[object] | None,
) -> tuple[list[str], list[str], list[str]]:
    """Split MBPP tests into prompt-visible and hidden evaluation subsets."""
    normalized_public = _normalize_tests(test_list)
    visible = normalized_public[:1]
    hidden_table1 = normalized_public[1:]
    challenge_tests = _normalize_tests(challenge_test_list)
    hidden_stress = [*hidden_table1, *challenge_tests]
    return visible, hidden_table1, hidden_stress


def _challenge_only_tests(hidden_table1: list[str], hidden_stress: list[str]) -> list[str]:
    remaining = list(hidden_stress)
    for test in hidden_table1:
        if test in remaining:
            remaining.remove(test)
    return remaining


def _task_from_manifest_record(
    *,
    record: dict[str, Any],
    dataset_split: str,
    profile: str,
) -> CodeTaskInstance:
    visible = _normalize_tests(record.get("visible_public_tests"))
    hidden_table1 = _normalize_tests(record.get("hidden_table1_tests"))
    hidden_stress = _normalize_tests(record.get("hidden_stress_tests"))
    if not hidden_stress:
        hidden_stress = list(hidden_table1)
    challenge_only = _challenge_only_tests(hidden_table1, hidden_stress)
    prompt = str(record.get("question_with_signature") or record.get("prompt") or record.get("text") or "")
    text = str(record.get("text") or record.get("prompt") or "")
    return CodeTaskInstance(
        task_id=str(record.get("task_id", "")).strip(),
        benchmark="mbpp",
        prompt=prompt,
        text=text,
        canonical_solution=str(record.get("canonical_solution") or record.get("code") or ""),
        code="",
        entry_point=str(record.get("entry_point") or "").strip() or None,
        test_list=list(visible),
        visible_public_tests=list(visible),
        hidden_table1_tests=list(hidden_table1),
        hidden_stress_tests=list(hidden_stress),
        table1_hidden_tests=list(hidden_table1),
        stress_plus_challenge_hidden_tests=list(hidden_stress),
        hidden_eval_tests=list(hidden_table1),
        challenge_test_list=list(challenge_only),
        test_setup_code=str(record.get("test_setup_code") or ""),
        metadata={
            "dataset_split": dataset_split,
            "profile": profile,
            "source_manifest_task_id": str(record.get("task_id", "")).strip(),
            "subset_name": str(record.get("subset_name", "") or ""),
            "mbpp_score_protocol": MBPP_DEFAULT_SCORE_PROTOCOL,
            "public_test_count": len(visible),
            "table1_hidden_test_count": len(hidden_table1),
            "challenge_test_count": len(challenge_only),
            "hidden_test_count": len(hidden_stress),
        },
    )


def _load_manifest_records(path: Path) -> list[dict[str, Any]] | None:
    if path.suffix.lower() != ".json" or not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    records = data.get("records")
    if not isinstance(records, list):
        return None
    return [item for item in records if isinstance(item, dict)]


def load_mbpp_tasks(
    profile: str = "full",
    limit: int | None = None,
    dataset_split: str = "",
) -> list[CodeTaskInstance]:
    """Load MBPP tasks and convert rows into `CodeTaskInstance` objects.

    Args:
        profile: Either `full` or `standard500`.
        limit: Optional maximum number of tasks after filtering.
        dataset_split: A Hugging Face split name, or a path to a manifest file
            containing task ids to keep.

    Returns:
        A list of unified code task instances.
    """
    if limit is not None and limit <= 0:
        limit = None

    manifest_records = None
    split_path = Path((dataset_split or "").strip()).expanduser() if str(dataset_split or "").strip() else None
    if split_path is not None and split_path.is_file():
        manifest_records = _load_manifest_records(split_path)
    if manifest_records is not None:
        resolved_dataset_split = dataset_split.strip() or os.fspath(split_path)
        tasks: list[CodeTaskInstance] = []
        for record in manifest_records:
            task_id = int(str(record.get("task_id", "") or 0))
            if not _matches_profile(task_id, profile):
                continue
            tasks.append(
                _task_from_manifest_record(
                    record=record,
                    dataset_split=resolved_dataset_split,
                    profile=profile,
                )
            )
            if limit is not None and len(tasks) >= limit:
                break
        return tasks

    source_split, allowed_task_ids = _resolve_dataset_split(dataset_split)
    dataset = load_dataset("mbpp", split=source_split)
    resolved_dataset_split = dataset_split.strip() or source_split
    tasks: list[CodeTaskInstance] = []

    for row in dataset:
        task_id = int(row["task_id"])
        if not _matches_profile(task_id, profile):
            continue
        if not _task_allowed(str(task_id), allowed_task_ids):
            continue
        visible_public_tests, hidden_table1_tests, hidden_stress_tests = split_mbpp_tests(
            test_list=row.get("test_list", []),
            challenge_test_list=row.get("challenge_test_list", []),
        )
        challenge_only = _challenge_only_tests(hidden_table1_tests, hidden_stress_tests)
        tasks.append(
            CodeTaskInstance(
                task_id=str(task_id),
                benchmark="mbpp",
                text=row.get("text", ""),
                prompt=row.get("text", ""),
                canonical_solution=row.get("code", ""),
                code="",
                test_list=visible_public_tests,
                test_setup_code=row.get("test_setup_code", ""),
                visible_public_tests=visible_public_tests,
                hidden_table1_tests=hidden_table1_tests,
                hidden_stress_tests=hidden_stress_tests,
                table1_hidden_tests=hidden_table1_tests,
                stress_plus_challenge_hidden_tests=hidden_stress_tests,
                hidden_eval_tests=hidden_table1_tests,
                challenge_test_list=challenge_only,
                metadata={
                    "dataset_split": resolved_dataset_split,
                    "profile": profile,
                    "reference_code_present": bool(str(row.get("code", "")).strip()),
                    "mbpp_score_protocol": MBPP_DEFAULT_SCORE_PROTOCOL,
                    "public_test_count": len(visible_public_tests),
                    "table1_hidden_test_count": len(hidden_table1_tests),
                    "challenge_test_count": len(challenge_only),
                    "hidden_test_count": len(hidden_stress_tests),
                },
            )
        )
        if limit is not None and len(tasks) >= limit:
            break
    return tasks
