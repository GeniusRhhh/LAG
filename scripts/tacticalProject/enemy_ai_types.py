"""Shared enums and dataclasses for enemy tactical AI modules."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class TacticalMode(Enum):
    """战术模式枚举 - 核心框架保持不变"""
    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"
    NEUTRAL = "neutral"


class EnemyTacticalPhase(Enum):
    """敌方战术阶段 - 五阶段框架保持不变"""
    NLT_MELD = "NLT_MELD"
    MELD_MTR = "MELD_MTR"
    MTR_TR = "MTR_TR"
    TR_DOR = "TR_DOR"
    DOR_DR = "DOR_DR"


class ThreatLevel(Enum):
    """威胁等级枚举 - 扩展威胁评估"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    SEVERE = 5


class RadarMode(Enum):
    """雷达工作模式 - 保持现有系统"""
    SEARCH = "search"
    TRACK = "track"
    LOCK = "lock"
    STANDBY = "standby"


class ActionType(Enum):
    """基础动作类型枚举"""
    MAINTAIN_HEADING = "maintain_heading"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    CLIMB = "climb"
    DESCEND = "descend"
    ACCELERATE = "accelerate"
    DECELERATE = "decelerate"
    CRANK_LEFT = "crank_left"
    CRANK_RIGHT = "crank_right"
    NOTCH_MANEUVER = "notch_maneuver"
    BEAM_MANEUVER = "beam_maneuver"
    DIVE_ESCAPE = "dive_escape"
    CHAFF_FLARE_MANEUVER = "chaff_flare_maneuver"
    SPIRAL_DIVE = "spiral_dive"
    SHORT_SKATE = "short_skate"
    DEFENSIVE_SPLIT = "defensive_split"
    AGGRESSIVE_APPROACH = "aggressive_approach"
    RETURN_TO_BASE = "return_to_base"


@dataclass
class SituationData:
    """态势数据结构"""
    min_enemy_distance: float
    closest_enemy_bearing: float
    missile_threats: List[Dict]
    radar_locked: bool
    lock_duration: float
    teammate_alive: bool
    teammate_distance: float
    current_altitude: float
    current_velocity: float
    current_heading: float


@dataclass
class ThreatAssessment:
    """威胁评估结果"""
    threat_level: ThreatLevel
    threat_score: float
    primary_threat_id: Optional[str]
    missile_threat_count: int
    lock_threat: bool
    distance_threat: bool


@dataclass
class ActionParameters:
    """动作参数结构"""
    duration: float
    turn_angle: Optional[float] = None
    turn_rate: Optional[float] = None
    altitude_change: Optional[float] = None
    velocity_change: Optional[float] = None
    target_heading: Optional[float] = None
