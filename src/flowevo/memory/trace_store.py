"""Persistence layer for execution traces.

MVP keeps a minimal JSON-file based store.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.schemas import ExecutionTrace


class TraceStore:
    """Save and load execution traces from local disk."""

    def __init__(self, root: str = "data/traces") -> None:
        """Initialize store.

        Args:
            root: Directory for trace JSON files.
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, trace: ExecutionTrace) -> Path:
        """Save one trace as JSON.

        Args:
            trace: Trace object to persist.

        Returns:
            Saved file path.
        """
        path = self.root / f"{trace.trace_id}.json"
        path.write_text(
            json.dumps(trace.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def load(self, trace_id: str) -> ExecutionTrace:
        """Load one trace by trace id or json filename stem."""
        path = self.root / f"{trace_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return ExecutionTrace.model_validate(data)

    def load_path(self, path: str) -> ExecutionTrace:
        """Load one trace from an explicit JSON path."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return ExecutionTrace.model_validate(data)
