"""Compatibility wrapper for the refactored tactical task module."""

from tactical_task_impl import (
    ENEMY_BASELINE_MODEL,
    FRIEND_BASELINE_MODEL,
    TacticalTask,
    TacticalTermination,
)

__all__ = [
    "ENEMY_BASELINE_MODEL",
    "FRIEND_BASELINE_MODEL",
    "TacticalTask",
    "TacticalTermination",
]
