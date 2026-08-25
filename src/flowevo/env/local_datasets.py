"""Shared local-first Hugging Face dataset resolution for every condition."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOCAL_HF_ROOT = PROJECT_ROOT / "data" / "hf"

_LOCAL_FILES = {
    ("openai/openai_humaneval", "", "test"):
        LOCAL_HF_ROOT / "openai_humaneval" / "test-00000-of-00001.parquet",
    ("openai/gsm8k", "main", "test"):
        LOCAL_HF_ROOT / "gsm8k" / "main" / "test-00000-of-00001.parquet",
}

_ENV_OVERRIDES = {
    "openai/openai_humaneval": "ASG_HUMANEVAL_PARQUET",
    "openai/gsm8k": "ASG_GSM8K_PARQUET",
}


def local_parquet_path(dataset_id: str, config: str = "",
                       split: str = "test") -> Path | None:
    """Return a configured local Parquet file when one is available."""
    override_name = _ENV_OVERRIDES.get(dataset_id, "")
    override = os.getenv(override_name, "").strip() if override_name else ""
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"{override_name} points to a missing dataset file: {path}")
        return path
    path = _LOCAL_FILES.get((dataset_id, config, split))
    return path if path is not None and path.is_file() else None


def load_dataset_local_first(dataset_id: str, *, config: str = "",
                             split: str = "test") -> tuple[Any, str]:
    """Load the shared local Parquet without any network request, else use HF."""
    from datasets import load_dataset

    local = local_parquet_path(dataset_id, config, split)
    if local is not None:
        dataset = load_dataset(
            "parquet", data_files={split: str(local)}, split=split)
        return dataset, str(local)
    if config:
        return load_dataset(dataset_id, config, split=split), \
            f"hf://{dataset_id}/{config}/{split}"
    return load_dataset(dataset_id, split=split), f"hf://{dataset_id}/{split}"
