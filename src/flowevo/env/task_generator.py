"""Task generator interface.

For MVP this module only exposes a tiny deterministic generator.
"""

from __future__ import annotations

from core.schemas import TaskInstance
from env.task_defs import build_dummy_task


def generate_task() -> TaskInstance:
    """Generate one task instance.

    Returns:
        A deterministic dummy `TaskInstance`.
    """
    return build_dummy_task()
