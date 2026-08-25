"""Load GSM8K tasks via Hugging Face datasets and wrap them as code tasks."""

from __future__ import annotations

import json
import re
from pathlib import Path

from core.schemas import CodeTaskInstance
from env.local_datasets import load_dataset_local_first


GSM8K_DEFAULT_CONFIG = "main"
GSM8K_DEFAULT_SPLIT = "test"
_ANSWER_PATTERN = re.compile(r"####\s*([^\n]+)")


def _read_manifest(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return {"task_ids": data}
    raise ValueError(f"Unsupported GSM8K manifest JSON structure: {path}")


def _read_manifest_task_ids(path: Path) -> tuple[str, str, list[str]]:
    if path.suffix.lower() != ".json":
        task_ids = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        return GSM8K_DEFAULT_CONFIG, GSM8K_DEFAULT_SPLIT, task_ids

    payload = _read_manifest(path)
    hf_config = str(payload.get("hf_config") or payload.get("config") or GSM8K_DEFAULT_CONFIG).strip() or GSM8K_DEFAULT_CONFIG
    hf_split = str(payload.get("hf_split") or payload.get("split") or GSM8K_DEFAULT_SPLIT).strip() or GSM8K_DEFAULT_SPLIT
    values = payload.get("task_ids") or payload.get("ids") or payload.get("selected_task_ids") or []
    task_ids: list[str] = []
    if isinstance(values, list):
        for item in values:
            if isinstance(item, dict):
                task_id = str(item.get("task_id", "")).strip()
            else:
                task_id = str(item).strip()
            if task_id:
                task_ids.append(task_id)
    if not task_ids:
        raise ValueError(f"GSM8K manifest {path} did not contain any task ids")
    return hf_config, hf_split, task_ids


def _resolve_dataset_split(dataset_split: str) -> tuple[str, str, set[str] | None, str]:
    cleaned = (dataset_split or "").strip()
    if not cleaned:
        resolved = f"{GSM8K_DEFAULT_CONFIG}:{GSM8K_DEFAULT_SPLIT}"
        return GSM8K_DEFAULT_CONFIG, GSM8K_DEFAULT_SPLIT, None, resolved

    split_path = Path(cleaned).expanduser()
    if split_path.is_file():
        hf_config, hf_split, task_ids = _read_manifest_task_ids(split_path)
        return hf_config, hf_split, set(task_ids), cleaned

    if cleaned in {"train", "test"}:
        return GSM8K_DEFAULT_CONFIG, cleaned, None, f"{GSM8K_DEFAULT_CONFIG}:{cleaned}"

    if ":" in cleaned:
        hf_config, hf_split = [part.strip() for part in cleaned.split(":", 1)]
        if not hf_config or not hf_split:
            raise ValueError(f"Unsupported GSM8K dataset split: {dataset_split}")
        return hf_config, hf_split, None, f"{hf_config}:{hf_split}"

    if split_path.suffix and not split_path.exists():
        raise FileNotFoundError(f"GSM8K dataset split or manifest not found: {dataset_split}")
    raise ValueError(f"Unsupported GSM8K dataset split: {dataset_split}")


def _extract_gold_answer(answer: str) -> str:
    match = _ANSWER_PATTERN.search(answer or "")
    if not match:
        raise ValueError("GSM8K answer field is missing the `#### final_answer` marker")
    return match.group(1).strip()


def _task_prompt(question: str) -> str:
    return "\n".join(
        [
            "Write raw Python only.",
            "Implement `def solve():`.",
            "Return only the final numeric answer.",
            "Do not print anything.",
            "",
            "Question:",
            question.strip(),
        ]
    ).strip()


def _gsm8k_test_setup() -> str:
    return "\n".join(
        [
            "from decimal import Decimal, InvalidOperation",
            "",
            "def _gsm8k_normalize(value):",
            "    text = str(value).strip()",
            "    if not text:",
            "        raise AssertionError('solve() returned an empty answer')",
            "    cleaned = text.replace(',', '').replace('$', '').replace(' ', '')",
            "    return Decimal(cleaned)",
            "",
            "def _gsm8k_equal(actual, expected):",
            "    try:",
            "        actual_value = _gsm8k_normalize(actual)",
            "        expected_value = _gsm8k_normalize(expected)",
            "    except InvalidOperation as exc:",
            "        raise AssertionError(f'Non-numeric GSM8K answer: {actual!r}') from exc",
            "    return abs(actual_value - expected_value) <= Decimal('1e-9')",
        ]
    )


def load_gsm8k_tasks(
    *,
    profile: str = "full",
    limit: int | None = None,
    dataset_split: str = "",
) -> list[CodeTaskInstance]:
    """Load GSM8K and convert rows into `CodeTaskInstance` objects."""
    del profile
    if limit is not None and limit <= 0:
        limit = None

    hf_config, hf_split, allowed_task_ids, resolved_dataset_split = _resolve_dataset_split(dataset_split)
    dataset, dataset_source = load_dataset_local_first(
        "openai/gsm8k", config=hf_config, split=hf_split)
    tasks: list[CodeTaskInstance] = []

    for index, row in enumerate(dataset):
        task_id = f"gsm8k_{hf_config}_{hf_split}_{index}"
        if allowed_task_ids is not None and task_id not in allowed_task_ids:
            continue
        question = str(row.get("question", "")).strip()
        gold_answer = _extract_gold_answer(str(row.get("answer", "")))
        tasks.append(
            CodeTaskInstance(
                task_id=task_id,
                benchmark="gsm8k",
                prompt=_task_prompt(question),
                entry_point="solve",
                text=question,
                test_setup_code=_gsm8k_test_setup(),
                test_list=[f"assert _gsm8k_equal(solve(), {gold_answer!r})"],
                metadata={
                    "dataset_split": resolved_dataset_split,
                    "dataset_source": dataset_source,
                    "hf_config": hf_config,
                    "hf_split": hf_split,
                    "question": question,
                    "gold_answer": gold_answer,
                    "disable_direct_skill_reuse": True,
                },
            )
        )
        if limit is not None and len(tasks) >= limit:
            break
    return tasks
