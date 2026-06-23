"""Compatibility wrapper for the refactored unified enemy tactical AI."""

from enemy_ai_types import (
    ActionParameters,
    ActionType,
    EnemyTacticalPhase,
    RadarMode,
    SituationData,
    TacticalMode,
    ThreatAssessment,
    ThreatLevel,
)
from unified_enemy_tactical_ai_impl import UnifiedEnemyTacticalAI

__all__ = [
    "ActionParameters",
    "ActionType",
    "EnemyTacticalPhase",
    "RadarMode",
    "SituationData",
    "TacticalMode",
    "ThreatAssessment",
    "ThreatLevel",
    "UnifiedEnemyTacticalAI",
]
