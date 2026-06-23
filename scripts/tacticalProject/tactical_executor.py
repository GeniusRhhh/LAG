"""Compatibility wrapper for the refactored tactical executor."""

from tactical_executor_impl import (
    TacticalExecutor,
)
from tactical_executor_logging import (
    log_enemy_behavior,
    log_maneuver_execution,
    log_missile_launch_check,
    log_phase_transition,
)

__all__ = [
    "TacticalExecutor",
    "log_enemy_behavior",
    "log_maneuver_execution",
    "log_missile_launch_check",
    "log_phase_transition",
]
