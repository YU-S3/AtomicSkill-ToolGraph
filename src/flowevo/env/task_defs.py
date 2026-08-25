"""Task definitions for FlowEvo.

This module defines small factory helpers for creating `TaskInstance` objects.
"""

from __future__ import annotations

from core.schemas import TaskInstance


def build_dummy_task(task_id: str = "task_dummy_001") -> TaskInstance:
    """Build a minimal dummy task used for local smoke tests.

    Args:
        task_id: Unique id of the task.

    Returns:
        A ready-to-run `TaskInstance`.
    """
    return TaskInstance(
        task_id=task_id,
        family="filter_select",
        query="Return rows where age > 30",
        table_paths=["data/tasks/manual_people.csv"],
        metadata={
            "difficulty": "easy",
            "source": "dummy",
            "manual_table_preview": [
                {"name": "Alice", "age": 28},
                {"name": "Bob", "age": 35},
                {"name": "Carol", "age": 41},
            ],
        },
        ground_truth={"matched_names": ["Bob", "Carol"]},
    )
