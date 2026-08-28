"""Action-budget ledger.  All limits are absolute action counts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BudgetLedger:
    global_limit: int
    node_limit: int
    attempt_limit: int
    node_start: int = 0
    attempt_start: int = 0
    actions_used: int = 0

    def global_remaining(self) -> int:
        return max(0, int(self.global_limit) - int(self.actions_used))

    def node_remaining(self) -> int:
        return max(0, int(self.node_limit)
                   - (int(self.actions_used) - int(self.node_start)))

    def attempt_remaining(self) -> int:
        return max(0, min(
            self.global_remaining(), self.node_remaining(),
            int(self.attempt_limit)
            - (int(self.actions_used) - int(self.attempt_start))))

    def absolute_deadline(self) -> int:
        return int(self.actions_used) + self.attempt_remaining()
