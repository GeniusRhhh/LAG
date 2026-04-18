"""
协同探测V4验证测试模块
"""
from .scenario_generator import (
    ScenarioGenerator, EnemyScenario,
    FormationType, AltitudeProfile, ApproachDirection,
    SpeedProfile, ManeuverType, AwacsStatus,
    SPEED_NORMAL, SPEED_HIGH, SPEED_LOW
)
from .verification_collector import (
    VerificationCollector, VerificationReporter, VerificationMetrics,
    LockEvent, PhaseTransition, ModeTransition, RecoveryEvent
)

__all__ = [
    'ScenarioGenerator', 'EnemyScenario',
    'FormationType', 'AltitudeProfile', 'ApproachDirection',
    'SpeedProfile', 'ManeuverType', 'AwacsStatus',
    'SPEED_NORMAL', 'SPEED_HIGH', 'SPEED_LOW',
    'VerificationCollector', 'VerificationReporter', 'VerificationMetrics',
    'LockEvent', 'PhaseTransition', 'ModeTransition', 'RecoveryEvent',
]
