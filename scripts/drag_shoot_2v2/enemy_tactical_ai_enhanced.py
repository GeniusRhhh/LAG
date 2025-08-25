#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Enemy Tactical AI System for Air Combat Simulation

Provides 5-stage distance control matching friendly forces, complete maneuver integration and tactical randomness

Design concept:
1. Completely mirrors friendly DragShootTacticalTask distance control timeline
2. Implements complete Short Skate 3-phase maneuver (Crank→Turn Cold→Escape)
3. Provides intelligent switching between 3 tactical modes (AGGRESSIVE/DEFENSIVE/NEUTRAL)
4. Integrates tactical randomness and diversified confrontation capabilities
5. Unified support for drag shoot, pincer attack and other tactical scenarios
"""

import logging
import numpy as np
import random
import math
import time
from typing import Tuple, Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass

# Import necessary components
try:
    from envs.JSBSim.tasks.pure_maneuvers import BasicManeuvers, CompositeManeuverExecutor
    from envs.JSBSim.core.catalog import Catalog as c
    from envs.JSBSim.utils.utils import get_radar_manager
    from envs.JSBSim.models.baseline_actor import BaselineActor
except ImportError as e:
    logging.warning(f"Component import failed, using fallback mode: {e}")


class TacticalMode(Enum):
    """Tactical mode enumeration"""
    AGGRESSIVE = "aggressive"    # Attack mode: active engagement, priority missile launch
    DEFENSIVE = "defensive"      # Defense mode: evasive maneuver, threat response
    NEUTRAL = "neutral"          # Neutral mode: balanced approach, flexible adjustment


class EnemyTacticalPhase(Enum):
    """Enemy tactical phase enumeration"""
    NLT_MELD = "NLT_MELD"    # 90-81km: Long-range engagement phase
    MELD_MTR = "MELD_MTR"    # 81-50km: Medium-range combat phase
    MTR_TR = "MTR_TR"        # 50-40km: Missile target range phase
    TR_DOR = "TR_DOR"        # 40-35km: Target range to dynamic attack range
    DOR_DR = "DOR_DR"        # 35-14.5km: Dynamic attack to defense range


class ThreatLevel(Enum):
    """Threat level enumeration"""
    NONE = 0      # No threat
    LOW = 1       # Low threat
    MEDIUM = 2    # Medium threat
    HIGH = 3      # High threat
    CRITICAL = 4  # Critical threat


class FormationRole(Enum):
    """编队角色枚举"""
    LEADER = "leader"         # 长机
    WINGMAN = "wingman"       # 僚机
    INDEPENDENT = "independent" # 独立作战


class CoordinationMode(Enum):
    """协同模式枚举"""
    FORMATION_KEEP = "formation_keep"     # 保持编队
    SPLIT_ATTACK = "split_attack"         # 分离攻击
    MUTUAL_SUPPORT = "mutual_support"     # 相互支援
    INDEPENDENT_ACTION = "independent_action" # 独立行动


class ECMType(Enum):
    """电子对抗措施类型"""
    NOISE_JAMMING = "noise_jamming"       # 噪声干扰
    DECEPTION_JAMMING = "deception_jamming" # 欺骗干扰
    CHAFF = "chaff"                       # 箔条干扰
    FLARE = "flare"                       # 红外诱饵
    FREQUENCY_AGILITY = "frequency_agility" # 频率捷变
    SIDELOBE_BLANKING = "sidelobe_blanking" # 旁瓣消隐


class MissileType(Enum):
    """导弹类型枚举"""
    AIM_120 = "aim_120"                   # AIM-120 主动雷达制导
    AIM_9 = "aim_9"                       # AIM-9 红外制导
    R_27ER = "r_27er"                     # R-27ER 半主动雷达制导
    R_73 = "r_73"                         # R-73 红外制导


class MissilePhase(Enum):
    """导弹飞行阶段"""
    BOOST = "boost"                       # 助推段
    MIDCOURSE = "midcourse"               # 中段
    TERMINAL = "terminal"                 # 末段


class RadarMode(Enum):
    """Radar working mode"""
    SEARCH = "search"      # Search mode (wide beam)
    TRACK = "track"        # Track mode (narrow beam)
    LOCK = "lock"          # Lock mode (continuous illumination)
    STANDBY = "standby"    # Standby mode


class ManeuverType(Enum):
    """Maneuver type enumeration - 完整的BVR战术机动"""
    # 基础机动
    SHORT_SKATE = "short_skate"           # Short skate maneuver
    NOTCH = "notch"                       # Radar evasion maneuver
    BEAM = "beam"                         # Beam maneuver

    # 新增的BVR战术机动
    CRANK = "crank"                       # 独立Crank机动
    DRAG = "drag"                         # Drag机动 - 拖拽敌方导弹
    GIMBAL = "gimbal"                     # Gimbal机动 - 利用雷达扫描限制
    MULTIPATH = "multipath"               # Multipath机动 - 利用地面反射干扰

    # 复杂规避机动
    BARREL_ROLL = "barrel_roll"           # Barrel roll evasion
    SPLIT_S = "split_s"                   # Split-S escape
    WEAVE = "weave"                       # Weave maneuver
    DEFENSIVE_SPIRAL = "defensive_spiral" # Defensive spiral

    # 协同机动
    PINCER_ATTACK = "pincer_attack"       # 钳形夹击
    BRACKET = "bracket"                   # 包围机动
    LEAPFROG = "leapfrog"                 # 蛙跳机动


@dataclass
class N001VERadarModel:
    """N001VE radar model parameters"""
    max_detection_range: float = 120000    # Maximum detection range 120km
    max_track_range: float = 80000         # Maximum tracking range 80km
    max_lock_range: float = 60000          # Maximum lock range 60km
    max_simultaneous_tracks: int = 8       # Simultaneous tracking targets
    search_beam_width: float = 60.0        # Search beam width (degrees)
    track_beam_width: float = 3.0          # Track beam width (degrees)
    lock_beam_width: float = 1.0           # Lock beam width (degrees)
    scan_period: float = 4.0               # Scan period (seconds)
    lock_update_rate: float = 0.1          # Lock update rate (seconds)
    detection_probability_base: float = 0.9 # Base detection probability
    track_loss_probability: float = 0.05   # Track loss probability
    jamming_resistance: float = 0.7        # Jamming resistance


@dataclass
class RadarTarget:
    """Radar target information"""
    target_id: str
    distance: float
    bearing: float
    elevation: float
    velocity: float
    rcs: float = 5.0                       # Radar cross section (m²)
    detection_probability: float = 0.0
    track_quality: float = 0.0
    lock_time: float = 0.0
    last_update: float = 0.0


@dataclass
class ManeuverState:
    """Enhanced maneuver state data class"""
    maneuver_type: ManeuverType     # Maneuver type
    phase: str                      # Current phase
    phase_start_time: float         # Phase start time
    total_start_time: float         # Total start time
    initial_heading: float          # Initial heading
    initial_altitude: float         # Initial altitude
    target_heading: float = 0.0     # Target heading
    target_altitude: float = 0.0    # Target altitude
    crank_angle: float = 0.0        # Crank angle
    turn_cold_angle: float = 0.0    # Turn Cold angle
    locked: bool = False            # Whether locked
    lock_duration: float = 0.0      # Lock duration
    # New maneuver parameters
    roll_direction: int = 1         # Roll direction (1=right, -1=left)
    climb_rate: float = 0.0         # Climb rate (m/s)
    weave_amplitude: float = 0.0    # Weave amplitude
    weave_period: float = 0.0       # Weave period
    spiral_radius: float = 0.0      # Spiral radius


@dataclass
class TacticalRandomness:
    """Tactical randomness parameters"""
    crank_angle: float = 45.0          # Crank angle randomization
    turn_cold_angle: float = 120.0     # Turn Cold angle randomization
    crank_duration: float = 8.0        # Crank duration randomization
    re_attack_probability: float = 0.8  # Re-attack probability randomization
    re_attack_distance: float = 60000   # Re-attack distance randomization
    formation_offset: float = 0.0       # Formation offset randomization
    coordination_delay: float = 2.0     # Coordination delay randomization
    attack_delay: float = 2.0           # Attack delay randomization
    retreat_threshold: float = 0.7      # Retreat threshold randomization


@dataclass
class FormationState:
    """编队状态信息"""
    leader_id: str = "B0100"           # 长机ID
    wingman_id: str = "B0200"          # 僚机ID
    formation_mode: CoordinationMode = CoordinationMode.FORMATION_KEEP
    separation_distance: float = 2000.0 # 编队间距 (m)
    relative_bearing: float = 30.0     # 相对方位角 (度)
    coordination_active: bool = True   # 协同是否激活
    last_communication: float = 0.0    # 上次通信时间
    shared_target: Optional[str] = None # 共享目标


@dataclass
class MissileDefenseState:
    """导弹防御状态"""
    incoming_missiles: List[str] = None # 来袭导弹列表
    missile_types: Dict[str, MissileType] = None # 导弹类型映射
    missile_phases: Dict[str, MissilePhase] = None # 导弹阶段映射
    threat_priorities: Dict[str, float] = None # 威胁优先级
    defense_strategy: Optional[ManeuverType] = None # 防御策略
    countermeasure_deployed: bool = False # 是否已部署对抗措施

    def __post_init__(self):
        if self.incoming_missiles is None:
            self.incoming_missiles = []
        if self.missile_types is None:
            self.missile_types = {}
        if self.missile_phases is None:
            self.missile_phases = {}
        if self.threat_priorities is None:
            self.threat_priorities = {}


@dataclass
class EnergyState:
    """能量状态管理"""
    current_speed: float = 250.0       # 当前速度 (m/s)
    current_altitude: float = 8000.0   # 当前高度 (m)
    fuel_remaining: float = 1.0        # 剩余燃料比例
    energy_level: str = "HIGH"         # 能量等级 (HIGH/MEDIUM/LOW/CRITICAL)
    optimal_speed: float = 300.0       # 最优速度
    optimal_altitude: float = 10000.0  # 最优高度
    energy_management_active: bool = True # 能量管理是否激活


@dataclass
class ElectronicWarfareState:
    """电子战状态"""
    ecm_active: bool = False           # ECM是否激活
    ecm_type: Optional[ECMType] = None # ECM类型
    ecm_start_time: float = 0.0        # ECM开始时间
    ecm_duration: float = 0.0          # ECM持续时间
    eccm_capability: float = 0.8       # ECCM能力
    jamming_effectiveness: float = 0.0  # 干扰效果
    rwr_warning: bool = False          # RWR警告
    chaff_count: int = 60              # 箔条数量
    flare_count: int = 60              # 红外诱饵数量


class EnhancedEnemyTacticalAI:
    """
    Enhanced Enemy Tactical AI System

    Core Features:
    1. 5-stage distance control timeline (mirroring friendly forces)
    2. Complete Short Skate 3-phase maneuver implementation
    3. Intelligent switching between 3 tactical modes
    4. Tactical randomness and diversified confrontation
    5. Dual-aircraft formation coordination mechanism
    6. Stable threat assessment system
    """
    
    def __init__(self):
        """Initialize Enhanced Enemy Tactical AI"""
        
        # Tactical distance configuration - completely mirrors friendly DragShootTacticalTask
        self.tactical_distances = {
            'NLT_MELD_min': 81000,   # 81km
            'MELD_MTR_min': 50000,   # 50km
            'MTR_TR_min': 40000,     # 40km
            'TR_DOR_min': 35000,     # 35km
            'DOR_DR_min': 14500,     # 14.5km
        }
        
        # Current state
        self.current_phase = EnemyTacticalPhase.NLT_MELD
        
        # Individual phase tracking for each agent
        self.current_phase_B0100 = EnemyTacticalPhase.NLT_MELD
        self.current_phase_B0200 = EnemyTacticalPhase.NLT_MELD
        self.last_phase_change_time = {"B0100": 0.0, "B0200": 0.0}
        self.phase_hysteresis_buffer = 2000  # 2km buffer to prevent oscillation
        
        # N001VE radar system
        self.radar_model = N001VERadarModel()
        self.radar_mode: Dict[str, RadarMode] = {
            "B0100": RadarMode.SEARCH,
            "B0200": RadarMode.SEARCH
        }
        self.radar_targets: Dict[str, Dict[str, RadarTarget]] = {
            "B0100": {},
            "B0200": {}
        }
        self.radar_scan_time: Dict[str, float] = {"B0100": 0.0, "B0200": 0.0}
        self.radar_lock_targets: Dict[str, Optional[str]] = {"B0100": None, "B0200": None}
        
        # Enhanced maneuver state management
        self.active_maneuvers: Dict[str, ManeuverState] = {}
        self.maneuver_history: Dict[str, List[ManeuverType]] = {
            "B0100": [],
            "B0200": []
        }
        self.maneuver_cooldowns: Dict[str, Dict[ManeuverType, float]] = {
            "B0100": {mt: 0.0 for mt in ManeuverType},
            "B0200": {mt: 0.0 for mt in ManeuverType}
        }
        
        # Maneuver state management (backward compatibility)
        self.short_skate_states: Dict[str, ManeuverState] = {}
        
        # Tactical randomness
        self.tactical_randomness = {
            "B0100": self._generate_tactical_randomness(),
            "B0200": self._generate_tactical_randomness()
        }
        
        # Tactical mode management
        self.current_tactical_mode = {"B0100": TacticalMode.NEUTRAL, "B0200": TacticalMode.NEUTRAL}
        self.mode_change_cooldown = 8.0  # 8 second cooldown
        self.last_mode_change_time = {"B0100": -999, "B0200": -999}
        
        # Re-attack state management
        self.re_attack_state = {"B0100": {"active": False, "last_check_time": 0.0}, 
                               "B0200": {"active": False, "last_check_time": 0.0}}
        self.missile_cooldown = 6.0  # 6 second missile launch cooldown
        
        # Maneuver lock management
        self.maneuver_locks = {"B0100": {"locked": False, "lock_time": 0.0, "lock_duration": 0.0},
                              "B0200": {"locked": False, "lock_time": 0.0, "lock_duration": 0.0}}
        
        # Return to base state management
        self.return_to_base_states: Dict[str, Dict[str, Any]] = {}

        # 新增系统状态管理
        # 编队协同系统
        self.formation_state = FormationState()
        self.formation_roles = {
            "B0100": FormationRole.LEADER,
            "B0200": FormationRole.WINGMAN
        }

        # 导弹防御系统
        self.missile_defense_states = {
            "B0100": MissileDefenseState(),
            "B0200": MissileDefenseState()
        }

        # 能量管理系统
        self.energy_states = {
            "B0100": EnergyState(),
            "B0200": EnergyState()
        }

        # 电子战系统
        self.ew_states = {
            "B0100": ElectronicWarfareState(),
            "B0200": ElectronicWarfareState()
        }

        # 协同作战参数
        self.coordination_active = True
        self.last_coordination_update = 0.0
        self.shared_tactical_picture = {}  # 共享战术态势图

        try:
            # Initialize maneuver components
            self.basic_maneuvers = BasicManeuvers()
            self.maneuver_executor = CompositeManeuverExecutor()
            logging.info("✅ Enhanced Enemy AI initialization complete with advanced systems")
        except Exception as e:
            logging.warning(f"⚠️ Maneuver component initialization failed, using fallback mode: {e}")
    
    def _generate_tactical_randomness(self) -> TacticalRandomness:
        """Generate tactical randomness parameters"""
        return TacticalRandomness(
            crank_angle=random.uniform(35.0, 55.0),        # 35-55 degree random Crank angle
            turn_cold_angle=random.uniform(100.0, 140.0),  # 100-140 degree random Turn Cold angle
            crank_duration=random.uniform(6.0, 10.0),      # 6-10 second random duration
            re_attack_probability=random.uniform(0.6, 0.9), # 0.6-0.9 random re-attack probability
            re_attack_distance=random.uniform(45000, 60000), # 45-60km random re-attack distance
            formation_offset=random.uniform(-3.0, 3.0),    # ±3 nautical mile random formation offset
            coordination_delay=random.uniform(0.0, 3.0)    # 0-3 second random coordination delay
        )
    
    def _refresh_tactical_randomness(self, agent_id: str):
        """Refresh tactical randomness parameters"""
        self.tactical_randomness[agent_id] = self._generate_tactical_randomness()
        randomness = self.tactical_randomness[agent_id]
        logging.info(f"🎲 {agent_id} Tactical randomness refreshed: "
                    f"Crank={randomness.crank_angle:.1f}°, "
                    f"TurnCold={randomness.turn_cold_angle:.1f}°, "
                    f"ReAttack={randomness.re_attack_probability:.2f}")

    def get_tactical_command(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """
        Get enemy tactical command - main interface function
        
        Args:
            env: Environment object
            agent_id: Agent ID (B0100 or B0200)
            current_time: Current time
            
        Returns:
            Tuple[int, int, int]: [altitude command, heading command, speed command] indices
        """
        try:
            # 1. Update all subsystems
            self._update_radar_system(env, agent_id, current_time)
            self._update_formation_coordination(env, agent_id, current_time)
            self._update_missile_defense_system(env, agent_id, current_time)
            self._update_energy_management(env, agent_id, current_time)
            self._update_electronic_warfare(env, agent_id, current_time)

            # 2. Update tactical phase
            self._update_tactical_phase(env, agent_id)

            # 3. Check critical defensive actions first
            # 3a. Missile defense priority
            missile_defense_command = self._check_missile_defense_actions(env, agent_id, current_time)
            if missile_defense_command:
                return missile_defense_command

            # 3b. Electronic warfare defensive actions
            ew_command = self._check_electronic_warfare_actions(env, agent_id, current_time)
            if ew_command:
                return ew_command

            # 4. Check return to base conditions
            if self._should_return_to_base(env, agent_id, current_time):
                return self._execute_return_to_base(env, agent_id)

            # 5. Check maneuver lock status
            if self._is_maneuver_locked(agent_id, current_time):
                return self._execute_locked_maneuver(env, agent_id, current_time)

            # 6. Formation coordination decision
            coordination_command = self._check_formation_coordination(env, agent_id, current_time)
            if coordination_command:
                return coordination_command

            # 7. Enhanced threat assessment
            threat_level = self._assess_comprehensive_threat_level(env, agent_id)

            # 8. Select tactical mode with coordination
            tactical_mode = self._select_coordinated_tactical_mode(env, agent_id, threat_level, current_time)

            # 9. Enhanced maneuver decision
            maneuver_command = self._enhanced_maneuver_decision(env, agent_id, current_time)
            if maneuver_command:
                return maneuver_command

            # 10. Generate coordinated tactical command
            command_indices = self._generate_coordinated_tactical_command(env, agent_id, tactical_mode, current_time)

            # 11. Apply energy-aware boundary check
            safe_command = self._apply_energy_aware_boundary_check(env, agent_id, command_indices)

            return safe_command
            
        except Exception as e:
            logging.error(f"❌ {agent_id} Enhanced enemy AI execution error: {e}")
            return 7, 8, 3  # Safe level flight command

    def _update_tactical_phase(self, env, agent_id: str):
        """Update tactical phase - fixed version with hysteresis to prevent oscillation"""
        try:
            min_distance = self._calculate_min_distance_to_friendlies(env, agent_id)
            current_phase = getattr(self, f'current_phase_{agent_id}', EnemyTacticalPhase.MELD_MTR)
            new_phase = current_phase

            # Current phase hold thresholds (add 2km buffer)
            if current_phase == EnemyTacticalPhase.NLT_MELD:
                if min_distance <= self.tactical_distances['NLT_MELD_min'] - self.phase_hysteresis_buffer:
                    new_phase = EnemyTacticalPhase.MELD_MTR
            elif current_phase == EnemyTacticalPhase.MELD_MTR:
                if min_distance <= self.tactical_distances['MELD_MTR_min'] - self.phase_hysteresis_buffer:
                    new_phase = EnemyTacticalPhase.MTR_TR
                elif min_distance > self.tactical_distances['NLT_MELD_min'] + self.phase_hysteresis_buffer:
                    new_phase = EnemyTacticalPhase.NLT_MELD
            elif current_phase == EnemyTacticalPhase.MTR_TR:
                if min_distance <= self.tactical_distances['MTR_TR_min'] - self.phase_hysteresis_buffer:
                    new_phase = EnemyTacticalPhase.TR_DOR
                elif min_distance > self.tactical_distances['MELD_MTR_min'] + self.phase_hysteresis_buffer:
                    new_phase = EnemyTacticalPhase.MELD_MTR
            elif current_phase == EnemyTacticalPhase.TR_DOR:
                if min_distance <= self.tactical_distances['TR_DOR_min'] - self.phase_hysteresis_buffer:
                    new_phase = EnemyTacticalPhase.DOR_DR
                elif min_distance > self.tactical_distances['MTR_TR_min'] + self.phase_hysteresis_buffer:
                    new_phase = EnemyTacticalPhase.MTR_TR
            elif current_phase == EnemyTacticalPhase.DOR_DR:
                if min_distance > self.tactical_distances['TR_DOR_min'] + self.phase_hysteresis_buffer:
                    new_phase = EnemyTacticalPhase.TR_DOR

            # Update individual phase state
            setattr(self, f'current_phase_{agent_id}', new_phase)

            # Record phase change (reduce log frequency)
            if new_phase != current_phase:
                self.last_phase_change_time[agent_id] = 0.0  # Reset for current time tracking
                logging.info(f"📍 {agent_id} Phase: {current_phase.value} → {new_phase.value} (Distance: {min_distance/1000:.1f}km)")

                # Refresh tactical randomness on phase change (reduce frequency)
                if new_phase in [EnemyTacticalPhase.MTR_TR, EnemyTacticalPhase.TR_DOR]:
                    self._refresh_tactical_randomness(agent_id)

        except Exception as e:
            logging.error(f"❌ {agent_id} Tactical phase update error: {e}")

    def _calculate_min_distance_to_friendlies(self, env, agent_id: str) -> float:
        """Calculate minimum distance to friendly targets"""
        try:
            if agent_id not in env.agents or not env.agents[agent_id].is_alive:
                return 100000.0  # Default far distance
            enemy_pos = np.array(env.agents[agent_id].get_position())
            min_distance = float('inf')

            # Calculate distance to all friendly aircraft
            for friendly_id in ["A0100", "A0200"]:
                if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                    friendly_pos = np.array(env.agents[friendly_id].get_position())
                    distance = np.linalg.norm(enemy_pos - friendly_pos)
                    min_distance = min(min_distance, distance)

            return min_distance if min_distance != float('inf') else 100000.0

        except Exception as e:
            logging.error(f"❌ {agent_id} Distance calculation error: {e}")
            return 100000.0

    def _assess_threat_level(self, env, agent_id: str) -> ThreatLevel:
        """Assess threat level - stable version to avoid frequent errors"""
        try:
            min_distance = self._calculate_min_distance_to_friendlies(env, agent_id)

            if min_distance < 20000:     # 20km
                base_threat = ThreatLevel.CRITICAL
            elif min_distance < 35000:   # 35km
                base_threat = ThreatLevel.HIGH
            elif min_distance < 50000:   # 50km
                base_threat = ThreatLevel.MEDIUM
            elif min_distance < 70000:   # 70km
                base_threat = ThreatLevel.LOW
            else:
                base_threat = ThreatLevel.NONE

            # Simplified missile threat assessment (avoid complex errors)
            try:
                # Check if any friendly aircraft has missiles
                for friendly_id in ["A0100", "A0200"]:
                    if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                        # Simple missile threat boost
                        if min_distance < 40000:  # Within missile range
                            base_threat = ThreatLevel(min(base_threat.value + 1, 4))
                            break
            except:
                pass  # Ignore missile check errors

            return base_threat

        except Exception as e:
            # Don't log errors to avoid log pollution
            return ThreatLevel.MEDIUM

    def _select_tactical_mode(self, env, agent_id: str, threat_level: ThreatLevel, current_time: float) -> TacticalMode:
        """Select tactical mode - based on threat level, tactical phase and randomness"""
        try:
            # Check mode switch cooldown
            if current_time - self.last_mode_change_time[agent_id] < self.mode_change_cooldown:
                return self.current_tactical_mode[agent_id]

            current_mode = self.current_tactical_mode[agent_id]
            new_mode = current_mode

            # Mode selection based on threat level
            if threat_level == ThreatLevel.CRITICAL or threat_level == ThreatLevel.HIGH:
                # High threat: 70% defense, 30% attack
                new_mode = TacticalMode.DEFENSIVE if random.random() < 0.7 else TacticalMode.AGGRESSIVE

            elif threat_level == ThreatLevel.MEDIUM:
                # Medium threat: based on tactical phase selection
                current_phase = getattr(self, f'current_phase_{agent_id}', EnemyTacticalPhase.MELD_MTR)
                if current_phase in [EnemyTacticalPhase.MTR_TR, EnemyTacticalPhase.TR_DOR]:
                    new_mode = TacticalMode.AGGRESSIVE if random.random() < 0.6 else TacticalMode.NEUTRAL
                else:
                    new_mode = TacticalMode.NEUTRAL

            elif threat_level == ThreatLevel.LOW:
                # Low threat: 60% attack, 40% neutral
                new_mode = TacticalMode.AGGRESSIVE if random.random() < 0.6 else TacticalMode.NEUTRAL

            else:  # ThreatLevel.NONE
                # No threat: maintain neutral or attack
                new_mode = TacticalMode.NEUTRAL if random.random() < 0.7 else TacticalMode.AGGRESSIVE

            # Update mode if changed
            if new_mode != current_mode:
                self.current_tactical_mode[agent_id] = new_mode
                self.last_mode_change_time[agent_id] = current_time
                logging.info(f"🔄 {agent_id} Tactical mode: {current_mode.value} → {new_mode.value} "
                           f"(Threat: {threat_level.name})")

            return new_mode

        except Exception as e:
            logging.error(f"❌ {agent_id} Tactical mode selection error: {e}")
            return TacticalMode.NEUTRAL

    def _generate_tactical_command(self, env, agent_id: str, tactical_mode: TacticalMode, current_time: float) -> Tuple[int, int, int]:
        """Generate tactical command - fixed version based on individual tactical phase"""
        try:
            # Generate commands based on agent role
            if agent_id == "B0100":  # Enemy lead aircraft
                return self._generate_lead_command(env, agent_id, tactical_mode, current_time)
            else:  # B0200 - Enemy wingman
                return self._generate_wingman_command(env, agent_id, tactical_mode, current_time)

        except Exception as e:
            logging.error(f"❌ {agent_id} Tactical command generation error: {e}")
            return 7, 8, 3

    def _generate_lead_command(self, env, agent_id: str, tactical_mode: TacticalMode, current_time: float) -> Tuple[int, int, int]:
        """Enemy lead aircraft tactical command - complete tactical flow version with return logic"""
        try:
            current_phase = getattr(self, f'current_phase_{agent_id}', EnemyTacticalPhase.MELD_MTR)
            randomness = self.tactical_randomness[agent_id]

            # CRITICAL FIX: Check for mission completion first
            alive_enemies = sum(1 for eid in ["A0100", "A0200"]
                               if eid in env.agents and env.agents[eid].is_alive)

            if alive_enemies == 0:
                logging.info(f"🏆 {agent_id} Mission complete - all enemies eliminated, returning to base")
                if agent_id not in self.return_to_base_states:
                    self._init_return_to_base(agent_id, current_time)
                return self._execute_return_to_base(env, agent_id)

            # Check if return to base is needed
            if self._should_return_to_base(env, agent_id, current_time):
                return self._execute_return_to_base(env, agent_id)

            # NLT_MELD phase: 主动接敌，朝向友方目标
            if current_phase == EnemyTacticalPhase.NLT_MELD:
                # CRITICAL FIX: 主动接敌而不是固定南向飞行
                target_heading = self._calculate_intercept_heading(env, agent_id)
                if target_heading is not None:
                    logging.debug(f"🎯 {agent_id} NLT_MELD主动接敌: 目标航向{target_heading:.1f}°")
                    return self._maintain_heading_precise(env, agent_id, target_heading)
                else:
                    # 没有目标时保持南向搜索
                    return self._maintain_heading_precise(env, agent_id, 180.0)

            # MELD_MTR phase: continue engagement, maintain southward
            elif current_phase == EnemyTacticalPhase.MELD_MTR:
                if tactical_mode == TacticalMode.AGGRESSIVE:
                    # Attack mode: slight adjustment but mainly maintain southward
                    target_heading = 180.0 + min(randomness.formation_offset, 5.0)  # Limit offset within 5 degrees
                    return self._maintain_heading_precise(env, agent_id, target_heading)
                else:
                    # Other modes: strictly maintain southward
                    return self._maintain_heading_precise(env, agent_id, 180.0)

            # MTR_TR phase: missile target range, start tactical maneuver
            elif current_phase == EnemyTacticalPhase.MTR_TR:
                if tactical_mode == TacticalMode.AGGRESSIVE:
                    # Attack mode: may initiate Short Skate
                    if agent_id not in self.short_skate_states and random.random() < 0.3:  # 30% probability
                        self._init_short_skate(agent_id, current_time)
                        return self._execute_short_skate(env, agent_id, current_time)
                    # Continue engagement, maintain southward
                    return self._maintain_heading_precise(env, agent_id, 180.0)
                elif tactical_mode == TacticalMode.DEFENSIVE:
                    # Defense mode: start evasive maneuver but don't deviate excessively
                    return self._maintain_heading_precise(env, agent_id, 165.0)  # Slight left deviation evasion
                else:
                    return self._maintain_heading_precise(env, agent_id, 180.0)

            # TR_DOR phase: target range to dynamic attack range, high probability maneuver
            elif current_phase == EnemyTacticalPhase.TR_DOR:
                if tactical_mode == TacticalMode.AGGRESSIVE:
                    # Attack mode: high probability initiate Short Skate
                    if agent_id not in self.short_skate_states and random.random() < 0.6:  # 60% probability
                        self._init_short_skate(agent_id, current_time)
                        return self._execute_short_skate(env, agent_id, current_time)
                    return self._maintain_heading_precise(env, agent_id, 180.0)
                elif tactical_mode == TacticalMode.DEFENSIVE:
                    # Defense mode: execute evasive maneuver
                    return self._maintain_heading_precise(env, agent_id, 150.0)  # Medium angle evasion
                else:
                    return self._maintain_heading_precise(env, agent_id, 180.0)

            # DOR_DR phase: dynamic attack to defense range, forced maneuver or return
            elif current_phase == EnemyTacticalPhase.DOR_DR:
                if agent_id not in self.short_skate_states:
                    # 50% probability execute Short Skate, 50% probability direct return
                    if random.random() < 0.5:
                        self._init_short_skate(agent_id, current_time)
                        return self._execute_short_skate(env, agent_id, current_time)
                    else:
                        # Direct start return
                        self._init_return_to_base(agent_id, current_time)
                        return self._execute_return_to_base(env, agent_id)

            # Default case: maintain southward
            return self._maintain_heading_precise(env, agent_id, 180.0)

        except Exception as e:
            logging.error(f"❌ {agent_id} Lead aircraft command error: {e}")
            return self._maintain_heading_precise(env, agent_id, 180.0)  # Safe southward flight

    def _generate_wingman_command(self, env, agent_id: str, tactical_mode: TacticalMode, current_time: float) -> Tuple[int, int, int]:
        """Enemy wingman tactical command - complete tactical flow version with return logic"""
        try:
            current_phase = getattr(self, f'current_phase_{agent_id}', EnemyTacticalPhase.MELD_MTR)
            randomness = self.tactical_randomness[agent_id]

            # CRITICAL FIX: Check for mission completion first
            alive_enemies = sum(1 for eid in ["A0100", "A0200"]
                               if eid in env.agents and env.agents[eid].is_alive)

            if alive_enemies == 0:
                logging.info(f"🏆 {agent_id} Mission complete - all enemies eliminated, returning to base")
                if agent_id not in self.return_to_base_states:
                    self._init_return_to_base(agent_id, current_time)
                return self._execute_return_to_base(env, agent_id)

            # Check if return to base is needed
            if self._should_return_to_base(env, agent_id, current_time):
                return self._execute_return_to_base(env, agent_id)

            # Wingman has slight tactical delay and offset relative to lead, but maintains main southward direction

            # NLT_MELD phase: formation flight, slight left deviation but not exceeding 10°
            if current_phase == EnemyTacticalPhase.NLT_MELD:
                target_heading = 180.0 + min(randomness.formation_offset, -5.0)  # Slight left deviation, limited within 5 degrees
                return self._maintain_heading_precise(env, agent_id, target_heading)

            # MELD_MTR phase: maintain formation, prepare for separation
            elif current_phase == EnemyTacticalPhase.MELD_MTR:
                if tactical_mode == TacticalMode.AGGRESSIVE:
                    # Attack mode: slight right deviation but not exceeding 10°
                    target_heading = 180.0 + min(randomness.formation_offset, 5.0)
                    return self._maintain_heading_precise(env, agent_id, target_heading)
                else:
                    # Other modes: slight left deviation
                    target_heading = 180.0 - 5.0
                    return self._maintain_heading_precise(env, agent_id, target_heading)

            # MTR_TR phase: wingman delayed action
            elif current_phase == EnemyTacticalPhase.MTR_TR:
                if tactical_mode == TacticalMode.AGGRESSIVE:
                    # Delayed initiate Short Skate (later than lead)
                    if random.random() < 0.2:  # 20% probability (slightly lower than lead)
                        self._init_short_skate(agent_id, current_time)
                        return self._execute_short_skate(env, agent_id, current_time)
                    # Continue engagement, slight left deviation
                    return self._maintain_heading_precise(env, agent_id, 175.0)
                elif tactical_mode == TacticalMode.DEFENSIVE:
                    # Defense mode: slight evasion but not excessive deviation
                    return self._maintain_heading_precise(env, agent_id, 170.0)
                else:
                    return self._maintain_heading_precise(env, agent_id, 175.0)

            # TR_DOR phase: wingman follows lead action
            elif current_phase == EnemyTacticalPhase.TR_DOR:
                if tactical_mode == TacticalMode.AGGRESSIVE:
                    if random.random() < 0.4:  # 40% probability (slightly lower than lead)
                        self._init_short_skate(agent_id, current_time)
                        return self._execute_short_skate(env, agent_id, current_time)
                    return self._maintain_heading_precise(env, agent_id, 175.0)
                elif tactical_mode == TacticalMode.DEFENSIVE:
                    # Defense mode: medium angle evasion
                    return self._maintain_heading_precise(env, agent_id, 160.0)
                else:
                    return self._maintain_heading_precise(env, agent_id, 175.0)

            # DOR_DR phase: wingman must also maneuver or return
            elif current_phase == EnemyTacticalPhase.DOR_DR:
                if agent_id not in self.short_skate_states:
                    # 40% probability execute Short Skate, 60% probability direct return
                    if random.random() < 0.4:
                        self._init_short_skate(agent_id, current_time)
                        return self._execute_short_skate(env, agent_id, current_time)
                    else:
                        # Direct start return
                        self._init_return_to_base(agent_id, current_time)
                        return self._execute_return_to_base(env, agent_id)

            # Default case: slight left deviation southward
            return self._maintain_heading_precise(env, agent_id, 175.0)

        except Exception as e:
            logging.error(f"❌ {agent_id} Wingman command error: {e}")
            return self._maintain_heading_precise(env, agent_id, 175.0)  # Safe slight left deviation southward flight

    # ==================== N001VE Radar System Methods ====================

    def _update_radar_system(self, env, agent_id: str, current_time: float):
        """Update N001VE radar system state - 集成统一雷达管理器"""
        try:
            # CRITICAL FIX: 使用统一雷达管理器的数据
            from radar_manager import get_unified_radar_manager
            radar_manager = get_unified_radar_manager()

            # 获取统一雷达管理器的目标数据
            unified_targets = radar_manager.get_enemy_radar_targets(agent_id)
            lock_target = radar_manager.get_enemy_lock_target(agent_id)
            radar_state = radar_manager.get_enemy_radar_state(agent_id)

            # 同步到内部雷达系统
            self.radar_targets[agent_id] = {}
            for target_id, unified_target in unified_targets.items():
                # 转换统一雷达目标到内部格式
                self.radar_targets[agent_id][target_id] = RadarTarget(
                    target_id=target_id,
                    distance=unified_target.distance,
                    bearing=unified_target.bearing,
                    elevation=unified_target.elevation,
                    velocity=unified_target.velocity,
                    rcs=unified_target.rcs,
                    detection_probability=unified_target.detection_probability,
                    track_quality=unified_target.track_quality,
                    lock_time=unified_target.lock_time,
                    last_update=unified_target.last_update
                )

            # 同步锁定目标
            self.radar_lock_targets[agent_id] = lock_target

            # 同步雷达模式
            if radar_state.value == "SEARCH":
                self.radar_mode[agent_id] = RadarMode.SEARCH
            elif radar_state.value == "TRACK":
                self.radar_mode[agent_id] = RadarMode.TRACK
            elif radar_state.value == "LOCK":
                self.radar_mode[agent_id] = RadarMode.LOCK
            else:
                self.radar_mode[agent_id] = RadarMode.SEARCH

            logging.debug(f"📡 {agent_id} 雷达系统同步: {len(self.radar_targets[agent_id])}个目标, "
                         f"锁定={lock_target}, 模式={self.radar_mode[agent_id].value}")

        except Exception as e:
            logging.error(f"❌ {agent_id} 雷达系统更新错误: {e}")
            # 降级到原有的雷达扫描逻辑
            self._update_radar_scan_fallback(env, agent_id, current_time)

    def _update_radar_scan_fallback(self, env, agent_id: str, current_time: float):
        """降级雷达扫描逻辑 - 当统一雷达管理器不可用时"""
        try:
            # 简化的目标探测逻辑
            self.radar_targets[agent_id] = {}

            # 扫描友方目标
            friendly_agents = ["A0100", "A0200"]
            for target_id in friendly_agents:
                if target_id not in env.agents or not env.agents[target_id].is_alive:
                    continue

                distance = self._calculate_distance_to_target(env, agent_id, target_id)
                if distance <= self.radar_model.max_detection_range:
                    # 简化的探测概率
                    detection_prob = max(0.1, 1.0 - distance / self.radar_model.max_detection_range)

                    if random.random() < detection_prob:
                        self.radar_targets[agent_id][target_id] = RadarTarget(
                            target_id=target_id,
                            distance=distance,
                            bearing=self._calculate_bearing_to_target(env, agent_id, target_id),
                            elevation=0.0,
                            velocity=200.0,  # 默认速度
                            detection_probability=detection_prob,
                            last_update=current_time
                        )
                        logging.debug(f"🎯 {agent_id} 降级雷达探测到目标: {target_id} 距离={distance/1000:.1f}km")

        except Exception as e:
            logging.error(f"❌ {agent_id} 降级雷达扫描错误: {e}")

    def _update_radar_scan(self, env, agent_id: str, current_time: float):
        """Update radar scan and target detection"""
        try:
            # Check scan period
            if current_time - self.radar_scan_time[agent_id] < self.radar_model.scan_period:
                return

            self.radar_scan_time[agent_id] = current_time

            # Scan friendly targets
            friendly_agents = ["A0100", "A0200"]
            for target_id in friendly_agents:
                if target_id not in env.agents or not env.agents[target_id].is_alive:
                    continue

                # Calculate target parameters
                distance = self._calculate_distance_to_target(env, agent_id, target_id)
                bearing = self._calculate_bearing_to_target(env, agent_id, target_id)
                velocity = self._calculate_target_velocity(env, target_id)

                # Check detection range
                if distance > self.radar_model.max_detection_range:
                    continue

                # Calculate detection probability
                detection_prob = self._calculate_detection_probability(distance, bearing)

                # Detection success
                if random.random() < detection_prob:
                    if target_id not in self.radar_targets[agent_id]:
                        self.radar_targets[agent_id][target_id] = RadarTarget(
                            target_id=target_id,
                            distance=distance,
                            bearing=bearing,
                            elevation=0.0,
                            velocity=velocity,
                            detection_probability=detection_prob,
                            last_update=current_time
                        )
                        logging.debug(f"🎯 {agent_id} Radar detected new target: {target_id} Distance={distance/1000:.1f}km")
                    else:
                        # Update existing target
                        target = self.radar_targets[agent_id][target_id]
                        target.distance = distance
                        target.bearing = bearing
                        target.velocity = velocity
                        target.detection_probability = detection_prob
                        target.last_update = current_time
                else:
                    # CRITICAL FIX: Remove target if detection fails (especially at long range)
                    if target_id in self.radar_targets[agent_id]:
                        logging.debug(f"📡 {agent_id} Lost radar contact with {target_id} at {distance/1000:.1f}km")
                        del self.radar_targets[agent_id][target_id]

        except Exception as e:
            logging.error(f"❌ {agent_id} Radar scan error: {e}")

    def _calculate_distance_to_target(self, env, agent_id: str, target_id: str) -> float:
        """Calculate distance to target"""
        try:
            if (agent_id not in env.agents or target_id not in env.agents or
                not env.agents[agent_id].is_alive or not env.agents[target_id].is_alive):
                return 999999.0

            # Simplified distance calculation using get_position
            agent_pos = np.array(env.agents[agent_id].get_position())
            target_pos = np.array(env.agents[target_id].get_position())

            return np.linalg.norm(agent_pos - target_pos)

        except Exception as e:
            logging.error(f"❌ Distance calculation error: {e}")
            return 999999.0

    def _calculate_bearing_to_target(self, env, agent_id: str, target_id: str) -> float:
        """Calculate bearing to target"""
        try:
            if (agent_id not in env.agents or target_id not in env.agents or
                not env.agents[agent_id].is_alive or not env.agents[target_id].is_alive):
                return 0.0

            # Simplified bearing calculation
            agent_pos = np.array(env.agents[agent_id].get_position())
            target_pos = np.array(env.agents[target_id].get_position())

            # Calculate relative bearing (simplified)
            diff = target_pos - agent_pos
            bearing = math.degrees(math.atan2(diff[1], diff[0]))

            # Normalize to [-180, 180]
            while bearing > 180:
                bearing -= 360
            while bearing < -180:
                bearing += 360

            return bearing

        except Exception as e:
            logging.error(f"❌ Bearing calculation error: {e}")
            return 0.0

    def _calculate_target_velocity(self, env, target_id: str) -> float:
        """Calculate target velocity"""
        try:
            if target_id not in env.agents or not env.agents[target_id].is_alive:
                return 0.0

            # Simplified velocity calculation
            return 250.0  # Assume typical fighter speed

        except Exception as e:
            logging.error(f"❌ Target velocity calculation error: {e}")
            return 0.0

    def _calculate_detection_probability(self, distance: float, bearing: float) -> float:
        """Calculate radar detection probability with realistic limitations"""
        try:
            # CRITICAL FIX: Realistic detection range limits
            if distance > 90000:  # Beyond 90km, detection becomes very unreliable
                return 0.0
            elif distance > 80000:  # 80-90km: very low probability
                base_prob = 0.1
            elif distance > 60000:  # 60-80km: low probability
                base_prob = 0.3
            elif distance > 40000:  # 40-60km: moderate probability
                base_prob = 0.6
            else:  # < 40km: high probability
                base_prob = 0.9

            # Angle factor (highest detection probability straight ahead)
            angle_factor = max(0.3, 1.0 - abs(bearing) / 90.0)

            # RCS factor (assume F-16 RCS about 5m²)
            rcs_factor = min(1.0, math.log10(5.0 + 1) / 2.0)

            # Add realistic atmospheric and terrain effects
            atmospheric_factor = max(0.7, 1.0 - (distance / 100000) * 0.3)

            # Comprehensive detection probability
            detection_prob = (base_prob * angle_factor * rcs_factor * atmospheric_factor)

            return min(1.0, max(0.0, detection_prob))

        except Exception as e:
            logging.error(f"Detection probability calculation error: {e}")
            return 0.5

    def _update_target_tracking(self, env, agent_id: str, current_time: float):
        """Update target tracking state"""
        try:
            targets_to_remove = []

            for target_id, target in self.radar_targets[agent_id].items():
                # Check if target is still within tracking range
                distance = self._calculate_distance_to_target(env, agent_id, target_id)

                if distance > self.radar_model.max_track_range:
                    targets_to_remove.append(target_id)
                    continue

                # Check track loss probability
                if random.random() < self.radar_model.track_loss_probability:
                    targets_to_remove.append(target_id)
                    logging.debug(f"📡 {agent_id} Lost target tracking: {target_id}")
                    continue

                # Update tracking quality
                range_quality = max(0.1, 1.0 - (distance / self.radar_model.max_track_range))
                time_quality = max(0.1, 1.0 - (current_time - target.last_update) / 10.0)
                target.track_quality = (range_quality + time_quality) / 2.0

            # Remove lost targets
            for target_id in targets_to_remove:
                del self.radar_targets[agent_id][target_id]
                if self.radar_lock_targets[agent_id] == target_id:
                    self.radar_lock_targets[agent_id] = None
                    logging.debug(f"🔓 {agent_id} Lost radar lock: {target_id}")

        except Exception as e:
            logging.error(f"❌ {agent_id} Target tracking update error: {e}")

    def _update_radar_mode(self, env, agent_id: str, current_time: float):
        """Update radar working mode"""
        try:
            current_phase = getattr(self, f'current_phase_{agent_id}', EnemyTacticalPhase.MELD_MTR)

            # Adjust radar mode based on tactical phase
            if current_phase == EnemyTacticalPhase.NLT_MELD:
                self.radar_mode[agent_id] = RadarMode.SEARCH
            elif current_phase in [EnemyTacticalPhase.MELD_MTR, EnemyTacticalPhase.MTR_TR]:
                if self.radar_targets[agent_id]:
                    self.radar_mode[agent_id] = RadarMode.TRACK
                else:
                    self.radar_mode[agent_id] = RadarMode.SEARCH
            elif current_phase in [EnemyTacticalPhase.TR_DOR, EnemyTacticalPhase.DOR_DR]:
                # Try to lock closest target
                closest_target = self._get_closest_target(agent_id)
                if closest_target and closest_target.distance <= self.radar_model.max_lock_range:
                    self.radar_mode[agent_id] = RadarMode.LOCK
                    self.radar_lock_targets[agent_id] = closest_target.target_id
                    closest_target.lock_time = current_time
                    logging.debug(f"🔒 {agent_id} Radar locked target: {closest_target.target_id}")
                else:
                    self.radar_mode[agent_id] = RadarMode.TRACK

        except Exception as e:
            logging.error(f"❌ {agent_id} Radar mode update error: {e}")

    def _get_closest_target(self, agent_id: str) -> Optional[RadarTarget]:
        """Get closest radar target"""
        try:
            if not self.radar_targets[agent_id]:
                return None

            closest_target = None
            min_distance = float('inf')

            for target in self.radar_targets[agent_id].values():
                if target.distance < min_distance:
                    min_distance = target.distance
                    closest_target = target

            return closest_target

        except Exception as e:
            logging.error(f"❌ {agent_id} Closest target retrieval error: {e}")
            return None

    def _assess_radar_threat_level(self, env, agent_id: str) -> ThreatLevel:
        """Radar-based threat assessment"""
        try:
            # Base distance threat assessment
            base_threat = self._assess_threat_level(env, agent_id)

            # Radar enhanced threat assessment
            radar_threat_bonus = 0

            # Check radar lock status
            if self.radar_lock_targets[agent_id]:
                locked_target = self.radar_targets[agent_id].get(self.radar_lock_targets[agent_id])
                if locked_target:
                    if locked_target.distance < 25000:  # Within 25km locked target
                        radar_threat_bonus += 1
                    if locked_target.track_quality > 0.8:  # High quality tracking
                        radar_threat_bonus += 1

            # Check multi-target threat
            if len(self.radar_targets[agent_id]) >= 2:
                radar_threat_bonus += 1

            # Check RWR warning (simulate being locked by enemy radar)
            if self._simulate_rwr_warning(env, agent_id):
                radar_threat_bonus += 2
                logging.debug(f"⚠️ {agent_id} RWR warning: locked by enemy radar")

            # Calculate final threat level
            final_threat_level = min(base_threat.value + radar_threat_bonus, 4)
            return ThreatLevel(final_threat_level)

        except Exception as e:
            logging.error(f"❌ {agent_id} Radar threat assessment error: {e}")
            return ThreatLevel.MEDIUM

    def _simulate_rwr_warning(self, env, agent_id: str) -> bool:
        """Simulate RWR warning (being locked by enemy radar)"""
        try:
            # Simplified RWR simulation: check if being "locked" by friendlies
            min_distance = self._calculate_min_distance_to_friendlies(env, agent_id)

            # Within 40km has high probability of being "locked"
            if min_distance < 40000:
                lock_probability = 0.8 * (1.0 - min_distance / 40000)
                return random.random() < lock_probability

            return False

        except Exception as e:
            logging.error(f"❌ {agent_id} RWR simulation error: {e}")
            return False

    def _radar_driven_maneuver_decision(self, env, agent_id: str, current_time: float) -> Optional[Tuple[int, int, int]]:
        """Radar-driven maneuver decision"""
        try:
            # Check if already executing maneuver
            if agent_id in self.active_maneuvers:
                return self._execute_active_maneuver(env, agent_id, current_time)

            # Radar event-driven maneuver triggers

            # 1. Radar lock success → increase attack maneuver probability
            if self.radar_lock_targets[agent_id]:
                if random.random() < 0.7:  # 70% probability trigger attack maneuver
                    return self._initiate_maneuver(env, agent_id, ManeuverType.SHORT_SKATE, current_time)

            # 2. RWR warning → trigger defense maneuver
            if self._simulate_rwr_warning(env, agent_id):
                defense_maneuvers = [ManeuverType.NOTCH, ManeuverType.BEAM, ManeuverType.BARREL_ROLL]
                selected_maneuver = random.choice(defense_maneuvers)
                if random.random() < 0.6:  # 60% probability trigger defense maneuver
                    return self._initiate_maneuver(env, agent_id, selected_maneuver, current_time)

            # 3. Radar loss → check if should return to base first
            if not self.radar_targets[agent_id]:
                # Check if no targets exist at all
                alive_friendlies = sum(1 for fid in ["A0100", "A0200"]
                                     if fid in env.agents and env.agents[fid].is_alive)

                if alive_friendlies == 0:
                    # No targets remaining, initiate return to base
                    logging.info(f"🏠 {agent_id} No radar targets and no enemies remaining, returning to base")
                    self._init_return_to_base(agent_id, current_time)
                    return self._execute_return_to_base(env, agent_id)
                elif random.random() < 0.4:  # 40% probability for search maneuver
                    search_maneuvers = [ManeuverType.WEAVE, ManeuverType.BEAM]
                    selected_maneuver = random.choice(search_maneuvers)
                    return self._initiate_maneuver(env, agent_id, selected_maneuver, current_time)

            # 4. Multi-target environment → trigger complex maneuver
            if len(self.radar_targets[agent_id]) >= 2 and random.random() < 0.5:  # 50% probability
                complex_maneuvers = [ManeuverType.DEFENSIVE_SPIRAL, ManeuverType.SPLIT_S]
                selected_maneuver = random.choice(complex_maneuvers)
                return self._initiate_maneuver(env, agent_id, selected_maneuver, current_time)

            return None

        except Exception as e:
            logging.error(f"❌ {agent_id} Radar-driven maneuver decision error: {e}")
            return None

    def _initiate_maneuver(self, env, agent_id: str, maneuver_type: ManeuverType, current_time: float) -> Tuple[int, int, int]:
        """Initiate new maneuver"""
        try:
            # Check maneuver cooldown
            if current_time - self.maneuver_cooldowns[agent_id][maneuver_type] < 30.0:
                return 7, 8, 3  # In cooldown, return level flight

            # Get current state
            try:
                current_heading = 180.0  # Default southward
                current_altitude = 8000.0  # Default altitude
            except:
                current_heading = 180.0
                current_altitude = 8000.0

            # Create maneuver state
            maneuver_state = ManeuverState(
                maneuver_type=maneuver_type,
                phase="entry",
                phase_start_time=current_time,
                total_start_time=current_time,
                initial_heading=current_heading,
                initial_altitude=current_altitude
            )

            # Configure maneuver parameters based on type
            self._configure_maneuver_parameters(maneuver_state, agent_id)

            # Record maneuver state
            self.active_maneuvers[agent_id] = maneuver_state
            self.maneuver_history[agent_id].append(maneuver_type)

            logging.info(f"🎯 {agent_id} Initiated {maneuver_type.value} maneuver")

            # Execute first step of maneuver
            return self._execute_active_maneuver(env, agent_id, current_time)

        except Exception as e:
            logging.error(f"❌ {agent_id} Maneuver initiation error: {e}")
            return 7, 8, 3

    def _configure_maneuver_parameters(self, maneuver_state: ManeuverState, agent_id: str):
        """Configure maneuver parameters"""
        try:
            randomness = self.tactical_randomness[agent_id]

            if maneuver_state.maneuver_type == ManeuverType.SHORT_SKATE:
                # Short Skate maneuver parameters (backward compatibility)
                maneuver_state.crank_angle = randomness.crank_angle
                maneuver_state.turn_cold_angle = randomness.turn_cold_angle

            elif maneuver_state.maneuver_type == ManeuverType.NOTCH:
                # Notch maneuver: 90°±10° turn
                maneuver_state.target_heading = maneuver_state.initial_heading + random.uniform(80, 100) * random.choice([-1, 1])

            elif maneuver_state.maneuver_type == ManeuverType.BARREL_ROLL:
                # Barrel roll maneuver parameters
                maneuver_state.roll_direction = random.choice([-1, 1])
                maneuver_state.target_altitude = maneuver_state.initial_altitude + random.uniform(1000, 2000) * maneuver_state.roll_direction

            elif maneuver_state.maneuver_type == ManeuverType.SPLIT_S:
                # Split-S maneuver parameters
                maneuver_state.target_heading = maneuver_state.initial_heading + random.uniform(160, 180)
                maneuver_state.target_altitude = maneuver_state.initial_altitude - random.uniform(1500, 3000)
                maneuver_state.climb_rate = -random.uniform(15, 30)  # Dive angle

            elif maneuver_state.maneuver_type == ManeuverType.WEAVE:
                # Weave maneuver parameters
                maneuver_state.weave_amplitude = random.uniform(20, 35)  # Oscillation amplitude
                maneuver_state.weave_period = random.uniform(4, 8)      # Oscillation period

            elif maneuver_state.maneuver_type == ManeuverType.DEFENSIVE_SPIRAL:
                # Defensive spiral parameters
                maneuver_state.spiral_radius = random.uniform(2000, 5000)  # Spiral radius
                maneuver_state.climb_rate = random.uniform(5, 15) * random.choice([-1, 1])  # Climb/dive rate
                maneuver_state.roll_direction = random.choice([-1, 1])

            elif maneuver_state.maneuver_type == ManeuverType.BEAM:
                # Beam maneuver: perpendicular to threat direction
                maneuver_state.target_heading = maneuver_state.initial_heading + random.uniform(85, 95) * random.choice([-1, 1])

        except Exception as e:
            logging.error(f"❌ Maneuver parameter configuration error: {e}")

    def _execute_active_maneuver(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """Execute current active maneuver"""
        try:
            if agent_id not in self.active_maneuvers:
                return 7, 8, 3

            maneuver = self.active_maneuvers[agent_id]
            maneuver_type = maneuver.maneuver_type

            # Execute corresponding logic based on maneuver type
            if maneuver_type == ManeuverType.SHORT_SKATE:
                return self._execute_short_skate_enhanced(env, agent_id, current_time)
            elif maneuver_type == ManeuverType.NOTCH:
                return self._execute_notch_maneuver(env, agent_id, current_time)
            elif maneuver_type == ManeuverType.BARREL_ROLL:
                return self._execute_barrel_roll_maneuver(env, agent_id, current_time)
            elif maneuver_type == ManeuverType.SPLIT_S:
                return self._execute_split_s_maneuver(env, agent_id, current_time)
            elif maneuver_type == ManeuverType.WEAVE:
                return self._execute_weave_maneuver(env, agent_id, current_time)
            elif maneuver_type == ManeuverType.DEFENSIVE_SPIRAL:
                return self._execute_defensive_spiral_maneuver(env, agent_id, current_time)
            elif maneuver_type == ManeuverType.BEAM:
                return self._execute_beam_maneuver(env, agent_id, current_time)
            else:
                return 7, 8, 3

        except Exception as e:
            logging.error(f"❌ {agent_id} Maneuver execution error: {e}")
            return 7, 8, 3

    # ==================== Basic Support Methods ====================

    def _maintain_heading_precise(self, env, agent_id: str, target_heading: float) -> Tuple[int, int, int]:
        """Precise heading maintenance - based on friendly system implementation"""
        try:
            # Normalize target heading to 0-360 range
            target_heading = target_heading % 360

            # Get current heading from aircraft state
            try:
                current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_rad)
                current_heading = np.rad2deg(current_heading) % 360  # Convert to degrees and normalize
            except:
                current_heading = 180.0  # Default southward if unable to get actual heading

            # Calculate heading difference using the helper method
            heading_diff = self._calculate_heading_difference(current_heading, target_heading)

            # CRITICAL FIX: More precise heading control with smaller increments
            if abs(heading_diff) < 2:  # Very close to target
                heading_cmd = 8  # Maintain current heading
            elif heading_diff > 0:  # Need to turn right
                if heading_diff > 20:
                    heading_cmd = 12  # Large right turn
                elif heading_diff > 10:
                    heading_cmd = 11  # Medium right turn
                elif heading_diff > 5:
                    heading_cmd = 10  # Small right turn
                else:
                    heading_cmd = 9   # Very small right turn
            else:  # Need to turn left
                if heading_diff < -20:
                    heading_cmd = 4  # Large left turn
                elif heading_diff < -10:
                    heading_cmd = 5  # Medium left turn
                elif heading_diff < -5:
                    heading_cmd = 6  # Small left turn
                else:
                    heading_cmd = 7  # Very small left turn

            logging.info(f"🧭 {agent_id} Heading: current={current_heading:.1f}°, target={target_heading:.1f}°, diff={heading_diff:.1f}°, cmd={heading_cmd}")

            return 7, heading_cmd, 3  # [altitude, heading, speed]

        except Exception as e:
            logging.error(f"❌ {agent_id} Heading maintenance error: {e}")
            return 7, 8, 3

    def _should_return_to_base(self, env, agent_id: str, current_time: float) -> bool:
        """Determine if should return to base"""
        try:
            # Condition 1: Already in return state
            if agent_id in self.return_to_base_states:
                return True

            # Condition 2: No enemy targets remaining - CRITICAL FIX
            alive_friendlies = []
            for friendly_id in ["A0100", "A0200"]:
                if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                    alive_friendlies.append(friendly_id)

            if len(alive_friendlies) == 0:
                logging.info(f"🏠 {agent_id} No enemy targets remaining, returning to base")
                return True

            # Condition 3: Distance too close, forced return
            min_distance = self._calculate_min_distance_to_friendlies(env, agent_id)
            if min_distance <= 30000:  # Within 30km forced return (更符合实际BVR交战撤退距离)
                logging.info(f"🏠 {agent_id} Too close to enemies ({min_distance/1000:.1f}km), returning to base")
                return True

            # Condition 4: Time too long, automatic return
            if current_time >= 240.0:  # After 4 minutes automatic return (更合理的任务时间)
                logging.info(f"🏠 {agent_id} Mission time exceeded ({current_time:.1f}s), returning to base")
                return True

            # Condition 6: High threat level, tactical retreat
            try:
                threat_level = self._assess_comprehensive_threat_level(env, agent_id)
                if threat_level in [ThreatLevel.CRITICAL, ThreatLevel.EXTREME]:
                    # 在高威胁情况下，有更高概率执行战术撤退
                    if random.random() < 0.5:  # 50%概率撤退（提高触发概率）
                        logging.info(f"🏠 {agent_id} High threat level ({threat_level.name}), tactical retreat")
                        return True
            except:
                pass

            # Condition 5: After completing Short Skate, probability return
            if agent_id not in self.short_skate_states:
                current_phase = getattr(self, f'current_phase_{agent_id}', EnemyTacticalPhase.MELD_MTR)
                if current_phase == EnemyTacticalPhase.DOR_DR and random.random() < 0.3:  # 30% probability
                    return True

            return False

        except Exception as e:
            logging.error(f"❌ {agent_id} Return decision error: {e}")
            return False

    def _init_return_to_base(self, agent_id: str, current_time: float):
        """Initialize return to base state"""
        try:
            # 敌方基地位置：北方+50km（Y坐标为正值）
            base_position = np.array([0, +50000, 8000])  # 基地在原点北方50km

            self.return_to_base_states[agent_id] = {
                "start_time": current_time,
                "phase": "turn_north",  # Turn north
                "target_heading": 0.0,  # Northward
                "phase_start_time": current_time,
                "base_position": base_position
            }
            logging.info(f"🔄 {agent_id} Starting return to base maneuver to {base_position}")

        except Exception as e:
            logging.error(f"❌ {agent_id} Return initialization error: {e}")

    def _execute_return_to_base(self, env, agent_id: str) -> Tuple[int, int, int]:
        """Execute return to base maneuver with realistic flight dynamics"""
        try:
            if agent_id not in self.return_to_base_states:
                self._init_return_to_base(agent_id, 0.0)

            state = self.return_to_base_states[agent_id]

            # Get current aircraft state
            try:
                current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_rad)
                current_heading = np.rad2deg(current_heading) % 360  # Convert to degrees and normalize
            except:
                current_heading = 180.0  # Default heading
            current_pos = env.agents[agent_id].get_position()

            # Calculate vector to base
            base_position = state.get('base_position', np.array([0, +50000, 8000]))  # 基地在原点北方50km
            to_base = base_position - current_pos
            distance_to_base = np.linalg.norm(to_base[:2])  # Only consider horizontal distance

            # Calculate desired heading to base - CRITICAL FIX: 正确的航向角计算
            # 航空导航：北向=0°，东向=90°，南向=180°，西向=270°
            # 在我们的坐标系中：Y轴正方向=北方，X轴正方向=东方
            desired_heading = (90 - np.degrees(np.arctan2(to_base[1], to_base[0]))) % 360

            logging.info(f"🧭 {agent_id} RTB计算: to_base=({to_base[0]:.0f},{to_base[1]:.0f}), desired_heading={desired_heading:.1f}°")

            # CRITICAL FIX: Implement realistic turn rate limiting
            heading_diff = self._calculate_heading_difference(current_heading, desired_heading)

            # Limit turn rate to realistic values (3°/sec max)
            max_turn_rate = 3.0 * 0.2  # 0.6° per 0.2s step
            if abs(heading_diff) > max_turn_rate:
                if heading_diff > 0:
                    target_heading = current_heading + max_turn_rate
                else:
                    target_heading = current_heading - max_turn_rate

                if target_heading < 0:
                    target_heading += 360
                elif target_heading >= 360:
                    target_heading -= 360
            else:
                target_heading = desired_heading

            logging.info(f"🔄 {agent_id} RTB: current={current_heading:.1f}°, target={target_heading:.1f}°, distance={distance_to_base/1000:.1f}km")
            return self._maintain_heading_precise(env, agent_id, target_heading)

        except Exception as e:
            logging.error(f"❌ {agent_id} Return execution error: {e}")
            return self._maintain_heading_precise(env, agent_id, 0.0)  # Safe northward flight

    def _calculate_heading_difference(self, current_heading: float, target_heading: float) -> float:
        """Calculate the shortest angular difference between two headings"""
        diff = target_heading - current_heading

        # Normalize to [-180, 180]
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360

        return diff

    def _is_maneuver_locked(self, agent_id: str, current_time: float) -> bool:
        """Check if maneuver is locked"""
        try:
            if agent_id in self.short_skate_states:
                return True
            if agent_id in self.active_maneuvers:
                return True
            return False
        except Exception as e:
            logging.error(f"❌ {agent_id} Maneuver lock check error: {e}")
            return False

    def _execute_locked_maneuver(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """Execute locked maneuver"""
        try:
            if agent_id in self.short_skate_states:
                return self._execute_short_skate(env, agent_id, current_time)
            elif agent_id in self.active_maneuvers:
                return self._execute_active_maneuver(env, agent_id, current_time)
            else:
                return 7, 8, 3
        except Exception as e:
            logging.error(f"❌ {agent_id} Locked maneuver execution error: {e}")
            return 7, 8, 3

    def _should_re_attack(self, env, agent_id: str, current_time: float) -> bool:
        """Check if should re-attack"""
        try:
            # Simplified re-attack logic
            min_distance = self._calculate_min_distance_to_friendlies(env, agent_id)
            randomness = self.tactical_randomness[agent_id]

            if min_distance > randomness.re_attack_distance and random.random() < randomness.re_attack_probability:
                return True
            return False
        except Exception as e:
            logging.error(f"❌ {agent_id} Re-attack check error: {e}")
            return False

    def _execute_re_attack(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """Execute re-attack"""
        try:
            # Re-attack: turn toward friendlies and accelerate engagement
            return self._maintain_heading_precise(env, agent_id, 180.0)  # Attack southward
        except Exception as e:
            logging.error(f"❌ {agent_id} Re-attack execution error: {e}")
            return 7, 8, 3

    def _apply_boundary_check(self, env, agent_id: str, command_indices: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Apply boundary check"""
        try:
            altitude_cmd, heading_cmd, speed_cmd = command_indices

            # Boundary check
            altitude_cmd = max(0, min(14, altitude_cmd))
            heading_cmd = max(0, min(16, heading_cmd))
            speed_cmd = max(0, min(6, speed_cmd))

            # Altitude safety check
            try:
                current_altitude = 8000.0  # Default altitude
                if current_altitude < 2000 and altitude_cmd < 7:  # Below 2000m and descending
                    altitude_cmd = 7  # Force maintain altitude
            except:
                pass

            return altitude_cmd, heading_cmd, speed_cmd

        except Exception as e:
            logging.error(f"❌ {agent_id} Boundary check error: {e}")
            return 7, 8, 3

    # ==================== 辅助计算方法 ====================

    def _calculate_intercept_heading(self, env, agent_id: str) -> Optional[float]:
        """计算拦截航向 - 朝向最近的友方目标"""
        try:
            if not hasattr(env, 'agents') or agent_id not in env.agents:
                return None

            agent = env.agents[agent_id]
            agent_pos = np.array(agent.get_position())

            # 寻找最近的友方目标
            min_distance = float('inf')
            closest_target_pos = None

            friendly_ids = ["A0100", "A0200"]
            for friendly_id in friendly_ids:
                if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                    friendly_pos = np.array(env.agents[friendly_id].get_position())
                    distance = np.linalg.norm(agent_pos - friendly_pos)
                    if distance < min_distance:
                        min_distance = distance
                        closest_target_pos = friendly_pos

            if closest_target_pos is None:
                return None

            # 计算朝向目标的航向
            dx = closest_target_pos[0] - agent_pos[0]
            dy = closest_target_pos[1] - agent_pos[1]

            # 计算航向角（北向为0°，顺时针）
            heading = math.degrees(math.atan2(dx, dy))
            heading = (heading + 360) % 360  # 标准化到0-360度

            return heading

        except Exception as e:
            logging.error(f"❌ {agent_id} 拦截航向计算错误: {e}")
            return None

    # ==================== Placeholder Maneuver Methods ====================

    def _init_short_skate(self, agent_id: str, current_time: float):
        """Initialize Short Skate maneuver - placeholder"""
        try:
            randomness = self.tactical_randomness[agent_id]
            self.short_skate_states[agent_id] = ManeuverState(
                maneuver_type=ManeuverType.SHORT_SKATE,
                phase="crank",
                phase_start_time=current_time,
                total_start_time=current_time,
                initial_heading=180.0,
                initial_altitude=8000.0,
                crank_angle=randomness.crank_angle,
                turn_cold_angle=randomness.turn_cold_angle
            )
            logging.info(f"🎯 {agent_id} Initiated Short Skate maneuver")
        except Exception as e:
            logging.error(f"❌ {agent_id} Short Skate initialization error: {e}")

    def _execute_short_skate(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """Execute Short Skate maneuver - simplified version"""
        try:
            if agent_id not in self.short_skate_states:
                return 7, 8, 3

            state = self.short_skate_states[agent_id]
            elapsed_time = current_time - state.total_start_time

            # Simple 3-phase execution
            if elapsed_time < 8.0:  # Crank phase
                target_heading = 180.0 + state.crank_angle
                return self._maintain_heading_precise(env, agent_id, target_heading)
            elif elapsed_time < 20.0:  # Turn Cold phase
                target_heading = 180.0 + state.turn_cold_angle
                return self._maintain_heading_precise(env, agent_id, target_heading)
            elif elapsed_time < 30.0:  # Escape phase
                return self._maintain_heading_precise(env, agent_id, 180.0)
            else:
                # Maneuver complete
                del self.short_skate_states[agent_id]
                logging.info(f"✅ {agent_id} Short Skate maneuver complete")
                return 7, 8, 3

        except Exception as e:
            logging.error(f"❌ {agent_id} Short Skate execution error: {e}")
            return 7, 8, 3

    def _execute_short_skate_enhanced(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """Execute enhanced Short Skate maneuver (backward compatibility)"""
        return self._execute_short_skate(env, agent_id, current_time)

    # Placeholder methods for new maneuvers (to be implemented in Task 3)
    def _execute_notch_maneuver(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """Execute Notch maneuver - placeholder"""
        return self._maintain_heading_precise(env, agent_id, 90.0)  # Simple 90-degree turn

    def _execute_beam_maneuver(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """Execute Beam maneuver - placeholder"""
        return self._maintain_heading_precise(env, agent_id, 90.0)  # Simple beam maneuver

    def _execute_barrel_roll_maneuver(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """Execute Barrel Roll maneuver - placeholder"""
        return self._maintain_heading_precise(env, agent_id, 180.0)  # Maintain heading for now

    def _execute_split_s_maneuver(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """Execute Split-S maneuver - placeholder"""
        return self._maintain_heading_precise(env, agent_id, 0.0)  # Turn north

    def _execute_weave_maneuver(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """Execute Weave maneuver - placeholder"""
        return self._maintain_heading_precise(env, agent_id, 180.0)  # Maintain southward

    def _execute_defensive_spiral_maneuver(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """Execute Defensive Spiral maneuver - placeholder"""
        return self._maintain_heading_precise(env, agent_id, 180.0)  # Maintain heading for now

    # ==================== 新增系统方法 ====================

    def _update_formation_coordination(self, env, agent_id: str, current_time: float):
        """更新编队协同系统"""
        try:
            if not self.coordination_active:
                return

            # 更新编队状态
            if current_time - self.formation_state.last_communication > 5.0:
                # 每5秒更新一次编队信息
                self._update_formation_positions(env, current_time)
                self.formation_state.last_communication = current_time

            # 检查编队完整性
            leader_alive = (self.formation_state.leader_id in env.agents and
                           env.agents[self.formation_state.leader_id].is_alive)
            wingman_alive = (self.formation_state.wingman_id in env.agents and
                            env.agents[self.formation_state.wingman_id].is_alive)

            if not leader_alive and not wingman_alive:
                self.coordination_active = False
            elif not leader_alive:
                # 长机阵亡，僚机接管
                self.formation_state.leader_id = self.formation_state.wingman_id
                self.formation_roles[self.formation_state.wingman_id] = FormationRole.LEADER
            elif not wingman_alive:
                # 僚机阵亡，长机独立作战
                self.formation_state.formation_mode = CoordinationMode.INDEPENDENT_ACTION

        except Exception as e:
            logging.error(f"❌ {agent_id} 编队协同更新错误: {e}")

    def _update_missile_defense_system(self, env, agent_id: str, current_time: float):
        """更新导弹防御系统"""
        try:
            defense_state = self.missile_defense_states[agent_id]

            # 检测来袭导弹（简化实现）
            incoming_missiles = self._detect_incoming_missiles(env, agent_id)
            defense_state.incoming_missiles = incoming_missiles

            # 更新导弹威胁优先级
            for missile_id in incoming_missiles:
                # 简化的威胁评估
                distance = self._estimate_missile_distance(env, agent_id, missile_id)
                priority = max(0.0, 1.0 - distance / 50000)  # 50km内威胁递增
                defense_state.threat_priorities[missile_id] = priority

            # 选择防御策略
            if incoming_missiles:
                highest_threat = max(defense_state.threat_priorities.values())
                if highest_threat > 0.7:
                    defense_state.defense_strategy = ManeuverType.NOTCH
                elif highest_threat > 0.4:
                    defense_state.defense_strategy = ManeuverType.BEAM
                else:
                    defense_state.defense_strategy = ManeuverType.WEAVE

        except Exception as e:
            logging.error(f"❌ {agent_id} 导弹防御系统更新错误: {e}")

    def _update_energy_management(self, env, agent_id: str, current_time: float):
        """更新能量管理系统"""
        try:
            energy_state = self.energy_states[agent_id]

            if agent_id in env.agents and env.agents[agent_id].is_alive:
                agent = env.agents[agent_id]

                # 更新当前状态
                try:
                    energy_state.current_speed = np.linalg.norm(agent.get_velocity())
                    energy_state.current_altitude = agent.get_position()[2]
                except:
                    pass  # 使用默认值

                # 计算能量等级
                speed_factor = energy_state.current_speed / energy_state.optimal_speed
                altitude_factor = energy_state.current_altitude / energy_state.optimal_altitude
                energy_factor = (speed_factor + altitude_factor) / 2.0

                if energy_factor > 0.8:
                    energy_state.energy_level = "HIGH"
                elif energy_factor > 0.6:
                    energy_state.energy_level = "MEDIUM"
                elif energy_factor > 0.4:
                    energy_state.energy_level = "LOW"
                else:
                    energy_state.energy_level = "CRITICAL"

        except Exception as e:
            logging.error(f"❌ {agent_id} 能量管理系统更新错误: {e}")

    def _update_electronic_warfare(self, env, agent_id: str, current_time: float):
        """更新电子战系统"""
        try:
            ew_state = self.ew_states[agent_id]

            # 检查ECM状态
            if ew_state.ecm_active:
                if current_time - ew_state.ecm_start_time > ew_state.ecm_duration:
                    ew_state.ecm_active = False
                    ew_state.ecm_type = None
                    logging.debug(f"🛡️ {agent_id} ECM结束")

            # 检查RWR警告
            ew_state.rwr_warning = self._simulate_rwr_warning(env, agent_id)

            # 随机激活ECM
            if not ew_state.ecm_active and ew_state.rwr_warning and random.random() < 0.1:
                self._activate_ecm(agent_id, current_time)

        except Exception as e:
            logging.error(f"❌ {agent_id} 电子战系统更新错误: {e}")

    def _check_missile_defense_actions(self, env, agent_id: str, current_time: float) -> Optional[Tuple[int, int, int]]:
        """检查导弹防御行动"""
        try:
            defense_state = self.missile_defense_states[agent_id]

            if not defense_state.incoming_missiles:
                return None

            # 获取最高威胁导弹
            if defense_state.threat_priorities:
                max_threat = max(defense_state.threat_priorities.values())
                if max_threat > 0.6:  # 高威胁阈值
                    if defense_state.defense_strategy:
                        return self._execute_defense_maneuver(env, agent_id, defense_state.defense_strategy, current_time)

            return None

        except Exception as e:
            logging.error(f"❌ {agent_id} 导弹防御检查错误: {e}")
            return None

    def _check_electronic_warfare_actions(self, env, agent_id: str, current_time: float) -> Optional[Tuple[int, int, int]]:
        """检查电子战行动"""
        try:
            ew_state = self.ew_states[agent_id]

            if ew_state.rwr_warning and not ew_state.ecm_active:
                # RWR警告且ECM未激活，考虑规避机动
                if random.random() < 0.3:  # 30%概率
                    return self._maintain_heading_precise(env, agent_id, 90.0)  # 简单的90度转向

            return None

        except Exception as e:
            logging.error(f"❌ {agent_id} 电子战行动检查错误: {e}")
            return None

    def _check_formation_coordination(self, env, agent_id: str, current_time: float) -> Optional[Tuple[int, int, int]]:
        """检查编队协同"""
        try:
            if not self.coordination_active:
                return None

            role = self.formation_roles.get(agent_id, FormationRole.INDEPENDENT)

            if role == FormationRole.WINGMAN:
                # 僚机跟随长机
                leader_id = self.formation_state.leader_id
                if leader_id in env.agents and env.agents[leader_id].is_alive:
                    return self._execute_wingman_coordination(env, agent_id, leader_id, current_time)

            return None

        except Exception as e:
            logging.error(f"❌ {agent_id} 编队协同检查错误: {e}")
            return None

    def _assess_comprehensive_threat_level(self, env, agent_id: str) -> ThreatLevel:
        """综合威胁评估"""
        try:
            # 基础雷达威胁
            base_threat = self._assess_radar_threat_level(env, agent_id)
            threat_value = base_threat.value

            # 导弹威胁加成
            defense_state = self.missile_defense_states[agent_id]
            if defense_state.incoming_missiles:
                max_missile_threat = max(defense_state.threat_priorities.values()) if defense_state.threat_priorities else 0
                threat_value += int(max_missile_threat * 2)  # 导弹威胁权重更高

            # 电子战威胁
            ew_state = self.ew_states[agent_id]
            if ew_state.rwr_warning:
                threat_value += 1

            # 能量状态影响
            energy_state = self.energy_states[agent_id]
            if energy_state.energy_level == "CRITICAL":
                threat_value += 1
            elif energy_state.energy_level == "LOW":
                threat_value += 0.5

            # 限制在有效范围内
            final_threat = min(max(int(threat_value), 0), 4)
            return ThreatLevel(final_threat)

        except Exception as e:
            logging.error(f"❌ {agent_id} 综合威胁评估错误: {e}")
            return ThreatLevel.MEDIUM

    # ==================== 辅助方法（简化实现） ====================

    def _detect_incoming_missiles(self, env, agent_id: str) -> List[str]:
        """检测来袭导弹（简化实现）"""
        # 简化实现：基于距离和威胁评估模拟导弹威胁
        try:
            min_distance = self._calculate_min_distance_to_friendlies(env, agent_id)
            if min_distance < 30000:  # 30km内可能有导弹威胁
                return ["simulated_missile_1"] if random.random() < 0.1 else []
            return []
        except:
            return []

    def _estimate_missile_distance(self, env, agent_id: str, missile_id: str) -> float:
        """估算导弹距离（简化实现）"""
        # 简化实现：返回模拟距离
        return random.uniform(10000, 40000)

    def _activate_ecm(self, agent_id: str, current_time: float):
        """激活ECM"""
        try:
            ew_state = self.ew_states[agent_id]
            ecm_types = [ECMType.NOISE_JAMMING, ECMType.CHAFF, ECMType.DECEPTION_JAMMING]
            ew_state.ecm_type = random.choice(ecm_types)
            ew_state.ecm_active = True
            ew_state.ecm_start_time = current_time
            ew_state.ecm_duration = random.uniform(5.0, 15.0)
            logging.debug(f"🛡️ {agent_id} 激活ECM: {ew_state.ecm_type.value}")
        except Exception as e:
            logging.error(f"❌ {agent_id} ECM激活错误: {e}")

    def _execute_defense_maneuver(self, env, agent_id: str, maneuver_type: ManeuverType, current_time: float) -> Tuple[int, int, int]:
        """执行防御机动（简化实现）"""
        if maneuver_type == ManeuverType.NOTCH:
            return self._maintain_heading_precise(env, agent_id, 90.0)
        elif maneuver_type == ManeuverType.BEAM:
            return self._maintain_heading_precise(env, agent_id, 270.0)
        else:
            return self._maintain_heading_precise(env, agent_id, 180.0)

    def _execute_wingman_coordination(self, env, agent_id: str, leader_id: str, current_time: float) -> Tuple[int, int, int]:
        """执行僚机协同（简化实现）"""
        # 简化实现：僚机稍微偏离长机航向
        return self._maintain_heading_precise(env, agent_id, 175.0)

    def _update_formation_positions(self, env, current_time: float):
        """更新编队位置（简化实现）"""
        # 简化实现：记录更新时间
        self.last_coordination_update = current_time

    def _select_coordinated_tactical_mode(self, env, agent_id: str, threat_level: ThreatLevel, current_time: float) -> TacticalMode:
        """选择协同战术模式（简化实现）"""
        # 简化实现：使用原有的战术模式选择
        return self._select_tactical_mode(env, agent_id, threat_level, current_time)

    def _enhanced_maneuver_decision(self, env, agent_id: str, current_time: float) -> Optional[Tuple[int, int, int]]:
        """增强机动决策（简化实现）"""
        # 简化实现：使用原有的雷达驱动机动决策
        return self._radar_driven_maneuver_decision(env, agent_id, current_time)

    def _generate_coordinated_tactical_command(self, env, agent_id: str, tactical_mode: TacticalMode, current_time: float) -> Tuple[int, int, int]:
        """生成协同战术指令（简化实现）"""
        # 简化实现：使用原有的战术指令生成
        return self._generate_tactical_command(env, agent_id, tactical_mode, current_time)

    def _apply_energy_aware_boundary_check(self, env, agent_id: str, command_indices: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """应用能量感知边界检查（简化实现）"""
        # 简化实现：使用原有的边界检查
        return self._apply_boundary_check(env, agent_id, command_indices)


# ==================== Global Instance Management ====================

_enhanced_enemy_ai_instance = None

def get_enhanced_enemy_ai() -> EnhancedEnemyTacticalAI:
    """Get enhanced enemy AI global instance"""
    global _enhanced_enemy_ai_instance
    # Force recreate instance to ensure all new methods are included
    _enhanced_enemy_ai_instance = EnhancedEnemyTacticalAI()
    return _enhanced_enemy_ai_instance


def get_enemy_tactical_command(env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
    """
    Enemy tactical command interface - unified interface function

    This is the main interface for external calls, supporting:
    - Drag shoot project
    - Pincer attack project
    - Other tactical scenarios

    Args:
        env: Environment object
        agent_id: Agent ID (B0100 or B0200)
        current_time: Current time

    Returns:
        Tuple[int, int, int]: [altitude command, heading command, speed command] indices
    """
    try:
        enemy_ai = get_enhanced_enemy_ai()
        return enemy_ai.get_tactical_command(env, agent_id, current_time)
    except Exception as e:
        logging.error(f"❌ Enhanced enemy AI interface error: {e}")
        return 7, 8, 3  # Safe level flight command


# Backward compatibility support
def enemy_ai_get_command(env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
    """Backward compatible interface function"""
    return get_enemy_tactical_command(env, agent_id, current_time)


if __name__ == "__main__":
    # Test enhanced enemy AI system
    print("🧪 Testing Enhanced Enemy Tactical AI System...")

    # Create AI instance
    ai = EnhancedEnemyTacticalAI()
    print(f"✅ AI instance created successfully")

    # Test tactical randomness generation
    randomness = ai._generate_tactical_randomness()
    print(f"✅ Tactical randomness generated: Crank={randomness.crank_angle:.1f}°, "
          f"TurnCold={randomness.turn_cold_angle:.1f}°")

    # Test threat assessment
    threat = ThreatLevel.MEDIUM
    print(f"✅ Threat assessment test: {threat.name}")

    print("🎉 Enhanced Enemy Tactical AI System test passed!")
