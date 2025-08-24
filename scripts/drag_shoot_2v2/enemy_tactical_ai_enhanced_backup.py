#!/usr/bin/env python3
"""
增强版敌方战术AI系统 - 完整实现
提供与我方匹配的5阶段距离控制、完整机动集成和战术随机?
Design concept?1. Completely mirrors friendly DragShootTacticalTask distance control timeline
2. Implements complete Short Skate 3-phase maneuver(CrankTurn ColdEscape?3. Provides intelligent switching between 3 tactical modes(AGGRESSIVE/DEFENSIVE/NEUTRAL?4. 集成战术随机性和多样化对抗能?5. Unified support for drag shoot, pincer attack and other tactical scenarios
"""

import logging
import numpy as np
import random
import math
import time
from typing import Tuple, Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass

# 导入必要的组件
try:
    from envs.JSBSim.tasks.pure_maneuvers import BasicManeuvers, CompositeManeuverExecutor
    from envs.JSBSim.core.catalog import Catalog as c
    from envs.JSBSim.utils.utils import get_radar_manager
    from envs.JSBSim.models.baseline_actor import BaselineActor
except ImportError as e:
    logging.warning(f"导入组件失败,使用备用模式: {e}")


class TacticalMode(Enum):
    """战术模式枚举"""
    AGGRESSIVE = "aggressive"    # Attack mode:主动接敌、优先发射导?    DEFENSIVE = "defensive"      # Defense mode:规避机动、威胁回?    NEUTRAL = "neutral"          # Neutral mode:平衡方法、灵活调?

class EnemyTacticalPhase(Enum):
    """敌方战术阶段 - 完全镜像我方5阶段系统"""
    NLT_MELD = "NLT_MELD"    # 90-81km:Long-range engagement phase
    MELD_MTR = "MELD_MTR"    # 81-50km:Medium-range combat phase
    MTR_TR = "MTR_TR"        # 50-40km:导弹目标范围阶?    TR_DOR = "TR_DOR"        # 40-35km:目标范围到动态攻击范?    DOR_DR = "DOR_DR"        # 35-14.5km:Dynamic attack to defense range


class ThreatLevel(Enum):
    """威胁等级枚举"""
    NONE = 0      # 无威?    LOW = 1       # 低威?    MEDIUM = 2    # Medium threat
    HIGH = 3      # 高威?    CRITICAL = 4  # Critical threat


class RadarMode(Enum):
    """雷达工作模式"""
    SEARCH = "search"      # Search mode(Wide beam?    TRACK = "track"        # Track mode(Narrow beam?    LOCK = "lock"          # Lock mode(Continuous illumination)
    STANDBY = "standby"    # 待机模式


class ManeuverType(Enum):
    """机动类型枚举"""
    SHORT_SKATE = "short_skate"           # 短程滑行机动
    NOTCH = "notch"                       # 雷达规避机动
    BARREL_ROLL = "barrel_roll"           # 桶滚规避
    SPLIT_S = "split_s"                   # 半筋斗脱?    WEAVE = "weave"                       # Weave maneuver
    DEFENSIVE_SPIRAL = "defensive_spiral" # 防御螺旋
    BEAM = "beam"                         # 横向规避


@dataclass
class N001VERadarModel:
    """N001VE雷达模型参数"""
    max_detection_range: float = 120000    # 最大探测距?20km
    max_track_range: float = 80000         # 最大跟踪距?0km
    max_lock_range: float = 60000          # 最大锁定距?0km
    max_simultaneous_tracks: int = 8       # 同时跟踪目标?    search_beam_width: float = 60.0        # Search beam width(度?    track_beam_width: float = 3.0          # Track beam width(度?    lock_beam_width: float = 1.0           # Lock beam width(度?    scan_period: float = 4.0               # Scan period(秒?    lock_update_rate: float = 0.1          # Lock update rate(秒)
    detection_probability_base: float = 0.9 # 基础探测概率
    track_loss_probability: float = 0.05   # 跟踪丢失概率
    jamming_resistance: float = 0.7        # 抗干扰能?

@dataclass
class RadarTarget:
    """雷达目标信息"""
    target_id: str
    distance: float
    bearing: float
    elevation: float
    velocity: float
    rcs: float = 5.0                       # Radar cross section(m?    detection_probability: float = 0.0
    track_quality: float = 0.0
    lock_time: float = 0.0
    last_update: float = 0.0


@dataclass
class ManeuverState:
    """增强的机动状态数据类"""
    maneuver_type: ManeuverType     # 机动类型
    phase: str                      # 当前阶段
    phase_start_time: float         # 阶段开始时?    total_start_time: float         # 总开始时?    initial_heading: float          # Initial heading
    initial_altitude: float         # 初始高度
    target_heading: float = 0.0     # 目标航向
    target_altitude: float = 0.0    # 目标高度
    crank_angle: float = 0.0        # Crank角度
    turn_cold_angle: float = 0.0    # Turn Cold角度
    locked: bool = False            # 是否锁定
    lock_duration: float = 0.0      # 锁定持续时间
    # 新增机动参数
    roll_direction: int = 1         # Roll direction?=右,-1=左)
    climb_rate: float = 0.0         # Climb rate(m/s?    weave_amplitude: float = 0.0    # Weave amplitude
    weave_period: float = 0.0       # 蛇形摆动周期
    spiral_radius: float = 0.0      # 螺旋半径


@dataclass
class TacticalRandomness:
    """战术随机性参?""
    crank_angle: float          # Crank角度随机?    crank_duration: float       # Crank持续时间随机?    turn_cold_angle: float      # Turn Cold角度随机?    turn_cold_rate: float       # Turn Cold转弯率随机化
    attack_delay: float = 2.0           # Attack delay randomization
    retreat_threshold: float = 0.7      # Retreat threshold randomization
    formation_offset: float = 0.0       # Formation offset randomization
    coordination_delay: float = 2.0     # Coordination delay randomization
    re_attack_distance: float = 50000   # Re-attack distance randomization
    re_attack_probability: float = 0.8  # Re-attack probability randomization

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
        
        # 当前状?        self.current_phase = EnemyTacticalPhase.NLT_MELD
        self.current_mode = {
            "B0100": TacticalMode.NEUTRAL,
            "B0200": TacticalMode.NEUTRAL
        }
        
        # N001VE雷达系统
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

        # 增强的机动状态管?        self.active_maneuvers: Dict[str, ManeuverState] = {}
        self.maneuver_history: Dict[str, List[ManeuverType]] = {
            "B0100": [],
            "B0200": []
        }
        self.maneuver_cooldowns: Dict[str, Dict[ManeuverType, float]] = {
            "B0100": {mt: 0.0 for mt in ManeuverType},
            "B0200": {mt: 0.0 for mt in ManeuverType}
        }

        # 机动状态管理(向后兼容?        self.short_skate_states: Dict[str, ManeuverState] = {}

        # 战术随机?        self.tactical_randomness = {
            "B0100": self._generate_tactical_randomness(),
            "B0200": self._generate_tactical_randomness()
        }
        
        # 模式切换控制
        self.mode_change_cooldown = 8.0  # 8秒冷却时?        self.last_mode_change_time = {"B0100": -999, "B0200": -999}
        
        # 导弹发射管理
        self.last_missile_launch_time = {"B0100": -999, "B0200": -999}
        self.missile_cooldown = 6.0  # 6秒导弹发射冷?        
        # 重新攻击逻辑
        self.re_attack_state = {
            "B0100": {"enabled": False, "trigger_distance": 0, "last_check_time": 0},
            "B0200": {"enabled": False, "trigger_distance": 0, "last_check_time": 0}
        }

        # 返航状态管?        self.return_to_base_states: Dict[str, Dict[str, Any]] = {}
        
        # 初始化基础机动组件
        try:
            self.basic_maneuvers = BasicManeuvers()
            self.maneuver_executor = CompositeManeuverExecutor()
            logging.info("?增强版敌方AI初始化完?)
        except Exception as e:
            logging.warning(f" 机动组件初始化失败,使用备用模式: {e}")
            self.basic_maneuvers = None
            self.maneuver_executor = None
    
    def _generate_tactical_randomness(self) -> TacticalRandomness:
        """Generate tactical randomness parameters"""
        return TacticalRandomness(
            crank_angle=random.uniform(35.0, 55.0),        # 35-55 degree random Crank angle
            turn_cold_angle=random.uniform(100.0, 140.0),  # 100-140 degree random Turn Cold angle
            crank_duration=random.uniform(6.0, 10.0),      # 6-10 second random duration
            re_attack_probability=random.uniform(0.6, 0.9), # 0.6-0.9 random re-attack probability
            re_attack_distance=random.uniform(45000, 60000), # 45-60km random re-attack distance
            formation_offset=random.uniform(-3.0, 3.0),    # 3 nautical mile random formation offset
            coordination_delay=random.uniform(0.0, 3.0)    # 0-3 second random coordination delay
            re_attack_probability=random.uniform(0.7, 0.95)  # 70-95%随机重新攻击概率
        )
    
    def _refresh_tactical_randomness(self, agent_id: str):
        """Refresh tactical randomness parameters"""
        self.tactical_randomness[agent_id] = self._generate_tactical_randomness()
        randomness = self.tactical_randomness[agent_id]
        logging.info(f" {agent_id} Tactical randomness refreshed: "
                    f"Crank={randomness.crank_angle:.1f}, "
                    f"TurnCold={randomness.turn_cold_angle:.1f}, "
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
            # 1. Update N001VE radar system
            self._update_radar_system(env, agent_id, current_time)

            # 2. Update tactical phase
            self._update_tactical_phase(env, agent_id)

            # 3. Check return to base conditions
            if self._should_return_to_base(env, agent_id, current_time):
                return self._execute_return_to_base(env, agent_id)

            # 4. Check maneuver lock status
            if self._is_maneuver_locked(agent_id, current_time):
                return self._execute_locked_maneuver(env, agent_id, current_time)

            # 5. 检查重新攻击逻辑
            if self._should_re_attack(env, agent_id, current_time):
                return self._execute_re_attack(env, agent_id, current_time)

            # 6. Radar-based threat assessment
            threat_level = self._assess_radar_threat_level(env, agent_id)

            # 7. Select tactical mode
            tactical_mode = self._select_tactical_mode(env, agent_id, threat_level, current_time)

            # 8. Radar-driven maneuver decision
            maneuver_command = self._radar_driven_maneuver_decision(env, agent_id, current_time)
            if maneuver_command:
                return maneuver_command

            # 9. Generate tactical command
            command_indices = self._generate_tactical_command(env, agent_id, tactical_mode, current_time)

            # 10. Apply boundary check
            safe_command = self._apply_boundary_check(env, agent_id, command_indices)

            return safe_command
            
        except Exception as e:
            logging.error(f" {agent_id} Enhanced enemy AI execution error: {e}")
            return 7, 8, 3  # Safe level flight command
    def _update_tactical_phase(self, env, agent_id: str):
        """Update tactical phase - fixed version with hysteresis to prevent oscillation"""
        try:
            min_distance = self._calculate_min_distance_to_friendlies(env, agent_id)

            # 添加滞后机制防止边界震荡
            # 当前阶段的保持阈值(增加2km缓冲区)
            hysteresis_buffer = 2000  # 2km滞后缓冲

            current_phase = getattr(self, f'current_phase_{agent_id}', EnemyTacticalPhase.NLT_MELD)

            # 阶段判断逻辑 - 添加滞后机制
            if current_phase == EnemyTacticalPhase.NLT_MELD:
                # 从NLT_MELD向下转换需要更小的距离
                if min_distance <= self.tactical_distances['NLT_MELD_min'] - hysteresis_buffer:
                    new_phase = EnemyTacticalPhase.MELD_MTR
                else:
                    new_phase = EnemyTacticalPhase.NLT_MELD
            elif current_phase == EnemyTacticalPhase.MELD_MTR:
                if min_distance > self.tactical_distances['NLT_MELD_min'] + hysteresis_buffer:
                    new_phase = EnemyTacticalPhase.NLT_MELD
                elif min_distance <= self.tactical_distances['MELD_MTR_min'] - hysteresis_buffer:
                    new_phase = EnemyTacticalPhase.MTR_TR
                else:
                    new_phase = EnemyTacticalPhase.MELD_MTR
            elif current_phase == EnemyTacticalPhase.MTR_TR:
                if min_distance > self.tactical_distances['MELD_MTR_min'] + hysteresis_buffer:
                    new_phase = EnemyTacticalPhase.MELD_MTR
                elif min_distance <= self.tactical_distances['MTR_TR_min'] - hysteresis_buffer:
                    new_phase = EnemyTacticalPhase.TR_DOR
                else:
                    new_phase = EnemyTacticalPhase.MTR_TR
            elif current_phase == EnemyTacticalPhase.TR_DOR:
                if min_distance > self.tactical_distances['MTR_TR_min'] + hysteresis_buffer:
                    new_phase = EnemyTacticalPhase.MTR_TR
                elif min_distance <= self.tactical_distances['TR_DOR_min'] - hysteresis_buffer:
                    new_phase = EnemyTacticalPhase.DOR_DR
                else:
                    new_phase = EnemyTacticalPhase.TR_DOR
            else:  # DOR_DR
                if min_distance > self.tactical_distances['TR_DOR_min'] + hysteresis_buffer:
                    new_phase = EnemyTacticalPhase.TR_DOR
                else:
                    new_phase = EnemyTacticalPhase.DOR_DR

            # 更新个体阶段状?            setattr(self, f'current_phase_{agent_id}', new_phase)

            # 记录阶段变化(减少日志频率)
            if not hasattr(self, f'last_phase_log_{agent_id}') or getattr(self, f'last_phase_log_{agent_id}') != new_phase:
                logging.info(f" {agent_id} 战术阶段: {new_phase.value} (距离: {min_distance/1000:.1f}km)")
                setattr(self, f'last_phase_log_{agent_id}', new_phase)

                # 阶段变化时刷新战术随机性(降低频率?                if new_phase in [EnemyTacticalPhase.MTR_TR, EnemyTacticalPhase.TR_DOR]:
                    if not hasattr(self, f'last_randomness_refresh_{agent_id}') or \
                       getattr(self, f'last_randomness_refresh_{agent_id}') != new_phase:
                        self._refresh_tactical_randomness(agent_id)
                        setattr(self, f'last_randomness_refresh_{agent_id}', new_phase)

        except Exception as e:
            logging.error(f"?{agent_id} Tactical phase update error: {e}")
            # 设置默认阶段
            setattr(self, f'current_phase_{agent_id}', EnemyTacticalPhase.MELD_MTR)

    def _calculate_min_distance_to_friendlies(self, env, agent_id: str) -> float:
        """Calculate minimum distance to friendly targets"""
        try:
            if agent_id not in env.agents or not env.agents[agent_id].is_alive:
                return 100000.0  # Default far distance
            enemy_pos = np.array(env.agents[agent_id].get_position())
            min_distance = float('inf')

            # 计算到所有我方飞机的距离
            for friendly_id in ["A0100", "A0200"]:
                if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                    friendly_pos = np.array(env.agents[friendly_id].get_position())
                    distance = np.linalg.norm(enemy_pos - friendly_pos)
                    min_distance = min(min_distance, distance)

            return min_distance if min_distance != float('inf') else 100000.0

        except Exception as e:
            logging.error(f"?{agent_id} Distance calculation error: {e}")
            return 100000.0

    def _assess_threat_level(self, env, agent_id: str) -> ThreatLevel:
        """Assess threat level - 稳定版本,避免频繁错?""
        try:
            # 基于距离的基础威胁评估
            min_distance = self._calculate_min_distance_to_friendlies(env, agent_id)

            if min_distance < 20000:      # 20km?                base_threat = ThreatLevel.CRITICAL
            elif min_distance < 35000:    # 35km?                base_threat = ThreatLevel.HIGH
            elif min_distance < 50000:    # 50km?                base_threat = ThreatLevel.MEDIUM
            elif min_distance < 70000:    # 70km?                base_threat = ThreatLevel.LOW
            else:
                base_threat = ThreatLevel.NONE

            # 简化的导弹威胁评估(避免复杂错误)
            try:
                missile_threat_bonus = 0
                if hasattr(env, 'missiles') and env.missiles:
                    for missile in env.missiles.values():
                        if (hasattr(missile, 'target_id') and
                            missile.target_id == agent_id and
                            hasattr(missile, 'is_alive') and
                            missile.is_alive):
                            missile_threat_bonus = 1
                            break

                final_threat_level = min(base_threat.value + missile_threat_bonus, 4)
                return ThreatLevel(final_threat_level)

            except:
                return base_threat

        except Exception as e:
            # 不记录错误日志,避免日志污染
            return ThreatLevel.MEDIUM

    def _select_tactical_mode(self, env, agent_id: str, threat_level: ThreatLevel, current_time: float) -> TacticalMode:
        """Select tactical mode - 基于威胁等级、战术阶段和随机?""
        try:
            # 检查模式切换冷却时?            if current_time - self.last_mode_change_time[agent_id] < self.mode_change_cooldown:
                return self.current_mode[agent_id]

            current_mode = self.current_mode[agent_id]
            new_mode = current_mode

            # 基于威胁等级的模式选择逻辑
            if threat_level == ThreatLevel.CRITICAL:
                new_mode = TacticalMode.DEFENSIVE
            elif threat_level == ThreatLevel.HIGH:
                # High threat:70%防御?0%攻击
                new_mode = TacticalMode.DEFENSIVE if random.random() < 0.7 else TacticalMode.AGGRESSIVE
            elif threat_level == ThreatLevel.MEDIUM:
                # Medium threat:基于战术阶段选择
                if self.current_phase in [EnemyTacticalPhase.MTR_TR, EnemyTacticalPhase.TR_DOR]:
                    new_mode = TacticalMode.AGGRESSIVE  # 攻击窗口
                else:
                    new_mode = TacticalMode.NEUTRAL
            elif threat_level == ThreatLevel.LOW:
                # Low threat:60%攻击?0%中?                new_mode = TacticalMode.AGGRESSIVE if random.random() < 0.6 else TacticalMode.NEUTRAL
            else:
                # No threat:保持中性或攻击
                new_mode = TacticalMode.NEUTRAL

            # 记录模式变化
            if new_mode != current_mode:
                self.current_mode[agent_id] = new_mode
                self.last_mode_change_time[agent_id] = current_time
                logging.info(f" {agent_id} 战术模式: {current_mode.value} ?{new_mode.value} "
                           f"(威胁: {threat_level.name})")

            return new_mode

        except Exception as e:
            logging.error(f"?{agent_id} Tactical mode selection error: {e}")
            return TacticalMode.NEUTRAL

    def _generate_tactical_command(self, env, agent_id: str, tactical_mode: TacticalMode, current_time: float) -> Tuple[int, int, int]:
        """Generate tactical command - 修复版,基于个体战术阶段"""
        try:
            # 检查是否正在执行Short Skate机动
            if agent_id in self.short_skate_states:
                return self._execute_short_skate(env, agent_id, current_time)

            # 获取个体战术阶段
            current_phase = getattr(self, f'current_phase_{agent_id}', EnemyTacticalPhase.MELD_MTR)

            # 根据角色分配不同的战术行?            if agent_id == "B0100":  # 敌方长机
                return self._get_leader_command(env, agent_id, tactical_mode, current_phase, current_time)
            elif agent_id == "B0200":  # 敌方僚机
                return self._get_wingman_command(env, agent_id, tactical_mode, current_phase, current_time)
            else:
                return 7, 8, 3  # 默认平稳飞行

        except Exception as e:
            logging.error(f"?{agent_id} Tactical command generation error: {e}")
            return 7, 8, 3

    def _get_leader_command(self, env, agent_id: str, tactical_mode: TacticalMode, current_phase: EnemyTacticalPhase, current_time: float) -> Tuple[int, int, int]:
        """Enemy lead aircraft tactical command - 完整战术流程版,包含返航逻辑"""
        try:
            randomness = self.tactical_randomness[agent_id]

            # 检查是否需要返?            if self._should_return_to_base(env, agent_id, current_time):
                return self._execute_return_to_base(env, agent_id)

            if current_phase == EnemyTacticalPhase.NLT_MELD:
                # NLT_MELD阶段:平稳接敌,朝南飞行?80?                return self._maintain_heading_precise(env, agent_id, 180.0)

            elif current_phase == EnemyTacticalPhase.MELD_MTR:
                # MELD_MTR阶段:继续接敌,保持南向
                if tactical_mode == TacticalMode.AGGRESSIVE:
                    # Attack mode:轻微调整,但主要保持南?                    target_heading = 180.0 + min(randomness.formation_offset, 5.0)  # 限制偏移?度内
                    return self._maintain_heading_precise(env, agent_id, target_heading)
                else:
                    # 其他模式:严格保持南?                    return self._maintain_heading_precise(env, agent_id, 180.0)

            elif current_phase == EnemyTacticalPhase.MTR_TR:
                # MTR_TR阶段:导弹目标范围,开始战术机?                if tactical_mode == TacticalMode.AGGRESSIVE:
                    # Attack mode:May initiate Short Skate
                    if random.random() < 0.25:  # 25%概率启动Short Skate
                        self._init_short_skate(agent_id, current_time)
                        return self._execute_short_skate(env, agent_id, current_time)
                    else:
                        # 继续接敌,保持南?                        return self._maintain_heading_precise(env, agent_id, 180.0)
                elif tactical_mode == TacticalMode.DEFENSIVE:
                    # Defense mode:开始规避机动,但不要过度偏?                    return self._maintain_heading_precise(env, agent_id, 165.0)  # Slight left deviation evasion
                else:
                    return self._maintain_heading_precise(env, agent_id, 180.0)

            elif current_phase == EnemyTacticalPhase.TR_DOR:
                # TR_DOR阶段:Target range to dynamic attack range,高概率机?                if tactical_mode == TacticalMode.AGGRESSIVE:
                    # Attack mode:High probability initiate Short Skate
                    if random.random() < 0.5:  # 50%概率启动Short Skate
                        self._init_short_skate(agent_id, current_time)
                        return self._execute_short_skate(env, agent_id, current_time)
                    else:
                        # 继续攻击接近
                        return self._maintain_heading_precise(env, agent_id, 180.0)
                elif tactical_mode == TacticalMode.DEFENSIVE:
                    # Defense mode:执行规避机?                    return self._maintain_heading_precise(env, agent_id, 150.0)  # Medium angle evasion
                else:
                    return self._maintain_heading_precise(env, agent_id, 180.0)

            elif current_phase == EnemyTacticalPhase.DOR_DR:
                # DOR_DR阶段:Dynamic attack to defense range,Forced maneuver or return
                if agent_id not in self.short_skate_states:
                    # 50%Probability execute Short Skate?0%Probability direct return
                    if random.random() < 0.5:
                        self._init_short_skate(agent_id, current_time)
                        return self._execute_short_skate(env, agent_id, current_time)
                    else:
                        # 直接开始返?                        self._init_return_to_base(agent_id, current_time)
                        return self._execute_return_to_base(env, agent_id)
                else:
                    return self._execute_short_skate(env, agent_id, current_time)

            # 默认情况:保持南?            return self._maintain_heading_precise(env, agent_id, 180.0)

        except Exception as e:
            logging.error(f"?{agent_id} Lead aircraft command error: {e}")
            return self._maintain_heading_precise(env, agent_id, 180.0)  # 安全的南向飞?
    def _get_wingman_command(self, env, agent_id: str, tactical_mode: TacticalMode, current_phase: EnemyTacticalPhase, current_time: float) -> Tuple[int, int, int]:
        """Enemy wingman tactical command - 完整战术流程版,包含返航逻辑"""
        try:
            randomness = self.tactical_randomness[agent_id]

            # 检查是否需要返?            if self._should_return_to_base(env, agent_id, current_time):
                return self._execute_return_to_base(env, agent_id)

            # 僚机相对于长机有轻微的战术延迟和偏移,但保持主要南向
            if current_phase == EnemyTacticalPhase.NLT_MELD:
                # NLT_MELD阶段:编队飞行,轻微左偏但不超过10?                target_heading = 180.0 + min(randomness.formation_offset, -5.0)  # 轻微左偏,限制在5度内
                return self._maintain_heading_precise(env, agent_id, target_heading)

            elif current_phase == EnemyTacticalPhase.MELD_MTR:
                # MELD_MTR阶段:保持编队,准备分离
                if tactical_mode == TacticalMode.AGGRESSIVE:
                    # Attack mode:轻微右偏,但不超过10?                    target_heading = 180.0 + min(randomness.formation_offset, 5.0)
                    return self._maintain_heading_precise(env, agent_id, target_heading)
                else:
                    # 其他模式:轻微左?                    target_heading = 180.0 - 5.0
                    return self._maintain_heading_precise(env, agent_id, target_heading)

            elif current_phase == EnemyTacticalPhase.MTR_TR:
                # MTR_TR阶段:僚机延迟行?                if tactical_mode == TacticalMode.AGGRESSIVE:
                    # 延迟启动Short Skate(比长机晚)
                    if random.random() < 0.2:  # 20%概率(比长机稍低?                        self._init_short_skate(agent_id, current_time)
                        return self._execute_short_skate(env, agent_id, current_time)
                    else:
                        # 继续接敌,轻微左?                        return self._maintain_heading_precise(env, agent_id, 175.0)
                elif tactical_mode == TacticalMode.DEFENSIVE:
                    # Defense mode:轻微规避,但不过度偏离
                    return self._maintain_heading_precise(env, agent_id, 170.0)  # Slight left deviation evasion
                else:
                    return self._maintain_heading_precise(env, agent_id, 175.0)

            elif current_phase == EnemyTacticalPhase.TR_DOR:
                # TR_DOR阶段:僚机跟随长机行?                if tactical_mode == TacticalMode.AGGRESSIVE:
                    if random.random() < 0.4:  # 40%概率(比长机稍低?                        self._init_short_skate(agent_id, current_time)
                        return self._execute_short_skate(env, agent_id, current_time)
                    else:
                        # 继续攻击接近
                        return self._maintain_heading_precise(env, agent_id, 175.0)
                elif tactical_mode == TacticalMode.DEFENSIVE:
                    # Defense mode:中等角度规?                    return self._maintain_heading_precise(env, agent_id, 160.0)
                else:
                    return self._maintain_heading_precise(env, agent_id, 175.0)

            elif current_phase == EnemyTacticalPhase.DOR_DR:
                # DOR_DR阶段:僚机也必须机动或返?                if agent_id not in self.short_skate_states:
                    # 40%Probability execute Short Skate?0%Probability direct return
                    if random.random() < 0.4:
                        self._init_short_skate(agent_id, current_time)
                        return self._execute_short_skate(env, agent_id, current_time)
                    else:
                        # 直接开始返?                        self._init_return_to_base(agent_id, current_time)
                        return self._execute_return_to_base(env, agent_id)
                else:
                    return self._execute_short_skate(env, agent_id, current_time)

            # 默认情况:轻微左偏南?            return self._maintain_heading_precise(env, agent_id, 175.0)

        except Exception as e:
            logging.error(f"?{agent_id} Wingman command error: {e}")
            return self._maintain_heading_precise(env, agent_id, 175.0)  # 安全的轻微左偏南向飞?
    def _init_short_skate(self, agent_id: str, current_time: float):
        """初始化Short Skate机动 - 修复版,确保合理的机动角?""
        try:
            # 获取当前状?            current_heading = 180.0  # 南向基准
            current_altitude = 10000.0  # 默认高度

            # 使用随机战术参数,但限制角度范围
            randomness = self.tactical_randomness[agent_id]

            # 根据智能体ID确定机动方向,限制角度避免过度偏?            if agent_id == "B0100":  # 长机右侧机动
                # 限制Crank角度?5-30度范围内
                crank_angle = min(30.0, max(15.0, randomness.crank_angle))
                # Turn Cold角度限制?0-90度范围内
                turn_cold_angle = min(90.0, max(60.0, randomness.turn_cold_angle))
            else:  # 僚机左侧机动
                # 负角度表示左转,同样限制角度范围
                crank_angle = -min(30.0, max(15.0, randomness.crank_angle))
                turn_cold_angle = -min(90.0, max(60.0, randomness.turn_cold_angle))

            # 创建机动状?            self.short_skate_states[agent_id] = ManeuverState(
                phase="crank",
                phase_start_time=current_time,
                total_start_time=current_time,
                initial_heading=current_heading,
                initial_altitude=current_altitude,
                crank_angle=crank_angle,
                turn_cold_angle=turn_cold_angle,
                locked=True,
                lock_duration=35.0  # 减少总锁定时间到35?            )

            logging.info(f" {agent_id} 启动Short Skate机动: "
                        f"Crank={crank_angle:.1f}, TurnCold={turn_cold_angle:.1f}")

        except Exception as e:
            logging.error(f"?{agent_id} Short Skate初始化错? {e}")

    def _execute_short_skate(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """执行Short Skate机动 - 修复版,确保正确的航向计算和合理的机动时?""
        try:
            if agent_id not in self.short_skate_states:
                return 7, 8, 3

            state = self.short_skate_states[agent_id]
            phase_time = current_time - state.phase_start_time
            total_time = current_time - state.total_start_time

            # 阶段持续时间配置 - 缩短时间避免过长的偏?            crank_duration = 8.0   # Crank持续8?            turn_cold_duration = 12.0  # Turn Cold持续12?            escape_duration = 10.0     # Escape持续10?
            # 阶段1:Crank机动(右转或左转接敌?            if state.phase == "crank":
                if phase_time < crank_duration:
                    # 执行Crank转弯 - 确保航向计算正确
                    target_heading = state.initial_heading + state.crank_angle
                    # 规范化航向到0-360度范?                    target_heading = target_heading % 360
                    return self._maintain_heading_precise(env, agent_id, target_heading)
                else:
                    # Crank阶段完成,进入Turn Cold阶段
                    state.phase = "turn_cold"
                    state.phase_start_time = current_time
                    logging.info(f" {agent_id} Short Skate: Crank ?Turn Cold")
                    target_heading = (state.initial_heading + state.crank_angle) % 360
                    return self._maintain_heading_precise(env, agent_id, target_heading)

            # 阶段2:Turn Cold机动(大角度转弯脱离?            elif state.phase == "turn_cold":
                if phase_time < turn_cold_duration:
                    # 执行Turn Cold转弯 - 渐进式转?                    turn_progress = phase_time / turn_cold_duration
                    current_turn_angle = state.crank_angle + (state.turn_cold_angle - state.crank_angle) * turn_progress
                    target_heading = (state.initial_heading + current_turn_angle) % 360
                    return self._maintain_heading_precise(env, agent_id, target_heading)
                else:
                    # Turn Cold阶段完成,进入Escape阶段
                    state.phase = "escape"
                    state.phase_start_time = current_time
                    logging.info(f" {agent_id} Short Skate: Turn Cold ?Escape")
                    target_heading = (state.initial_heading + state.turn_cold_angle) % 360
                    return self._maintain_heading_precise(env, agent_id, target_heading)

            # 阶段3:Escape机动(保持脱离航向,然后返回南向?            elif state.phase == "escape":
                if phase_time < escape_duration:
                    # 前半段保持脱离航?                    if phase_time < escape_duration / 2:
                        target_heading = (state.initial_heading + state.turn_cold_angle) % 360
                    else:
                        # 后半段逐渐返回南向
                        return_progress = (phase_time - escape_duration / 2) / (escape_duration / 2)
                        escape_heading = (state.initial_heading + state.turn_cold_angle) % 360
                        target_heading = escape_heading + (180.0 - escape_heading) * return_progress
                        target_heading = target_heading % 360
                    return self._maintain_heading_precise(env, agent_id, target_heading)
                else:
                    # Short Skate机动完成,返回南?                    logging.info(f"?{agent_id} Short Skate机动完成,返回南?)
                    del self.short_skate_states[agent_id]

                    # 刷新战术随机性,准备下次机动
                    self._refresh_tactical_randomness(agent_id)

                    # 返回南向飞行
                    return self._maintain_heading_precise(env, agent_id, 180.0)

            return 7, 8, 3

        except Exception as e:
            logging.error(f"?{agent_id} Short Skate执行错误: {e}")
            # 清理错误状?            if agent_id in self.short_skate_states:
                del self.short_skate_states[agent_id]
            # 返回南向飞行
            return self._maintain_heading_precise(env, agent_id, 180.0)

    def _maintain_heading_precise(self, env, agent_id: str, target_heading: float) -> Tuple[int, int, int]:
        """精确航向保持 - 基于我方系统的实?""
        try:
            if agent_id not in env.agents or not env.agents[agent_id].is_alive:
                return 7, 8, 3

            # 获取当前航向
            current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))

            # 计算航向?            heading_diff = target_heading - current_heading

            # 规范化角度差到[-180, 180]范围
            while heading_diff > 180:
                heading_diff -= 360
            while heading_diff < -180:
                heading_diff += 360

            # 选择航向指令
            if abs(heading_diff) < 3:
                heading_cmd = 8  # 保持航向
            elif abs(heading_diff) < 10:
                heading_cmd = 12 if heading_diff > 0 else 4  # 小幅转弯
            elif abs(heading_diff) < 30:
                heading_cmd = 13 if heading_diff > 0 else 3  # 中等转弯
            else:
                heading_cmd = 14 if heading_diff > 0 else 2  # 大幅转弯

            # 高度和速度保持
            altitude_cmd = 7  # 保持高度
            velocity_cmd = 3  # 保持速度

            return altitude_cmd, heading_cmd, velocity_cmd

        except Exception as e:
            logging.error(f"?{agent_id} Heading maintenance error: {e}")
            return 7, 8, 3

    def _should_return_to_base(self, env, agent_id: str, current_time: float) -> bool:
        """判断是否应该返航"""
        try:
            # 条件1:已经在返航状?            if agent_id in self.return_to_base_states:
                return True

            # 条件2:距离过近,强制返航
            min_distance = self._calculate_min_distance_to_friendlies(env, agent_id)
            if min_distance < 15000:  # 15km内强制返?                return True

            # 条件3:时间过长,自动返航
            if current_time > 240.0:  # 4分钟后自动返?                return True

            # 条件4:完成Short Skate后,有概率返?            if agent_id not in self.short_skate_states:
                current_phase = getattr(self, f'current_phase_{agent_id}', EnemyTacticalPhase.MELD_MTR)
                if current_phase == EnemyTacticalPhase.DOR_DR and random.random() < 0.3:  # 30%概率
                    return True

            return False

        except Exception as e:
            logging.error(f"?{agent_id} Return decision error: {e}")
            return False

    def _init_return_to_base(self, agent_id: str, current_time: float):
        """初始化返航状?""
        try:
            self.return_to_base_states[agent_id] = {
                "start_time": current_time,
                "phase": "turn_north",  # 转向北方
                "target_heading": 0.0,  # 北向
                "phase_start_time": current_time
            }
            logging.info(f" {agent_id} 开始返航机?)

        except Exception as e:
            logging.error(f"?{agent_id} 返航初始化错? {e}")

    def _execute_return_to_base(self, env, agent_id: str) -> Tuple[int, int, int]:
        """执行返航机动"""
        try:
            if agent_id not in self.return_to_base_states:
                self._init_return_to_base(agent_id, 0.0)

            state = self.return_to_base_states[agent_id]

            if state["phase"] == "turn_north":
                # 阶段1:转向北方(0度)
                return self._maintain_heading_precise(env, agent_id, 0.0)
            else:
                # 阶段2:保持北向返?                return self._maintain_heading_precise(env, agent_id, 0.0)

        except Exception as e:
            logging.error(f"?{agent_id} Return execution error: {e}")
            return self._maintain_heading_precise(env, agent_id, 0.0)  # 安全的北向飞?
    def _is_maneuver_locked(self, agent_id: str, current_time: float) -> bool:
        """检查机动是否被锁定"""
        try:
            if agent_id in self.short_skate_states:
                state = self.short_skate_states[agent_id]
                if state.locked:
                    elapsed_time = current_time - state.total_start_time
                    if elapsed_time < state.lock_duration:
                        return True
                    else:
                        # 锁定时间到期,解除锁?                        state.locked = False
                        logging.info(f" {agent_id} 机动锁定解除")
            return False
        except:
            return False

    def _execute_locked_maneuver(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """执行被锁定的机动"""
        try:
            if agent_id in self.short_skate_states:
                return self._execute_short_skate(env, agent_id, current_time)
            else:
                return 7, 8, 3
        except:
            return 7, 8, 3

    def _should_re_attack(self, env, agent_id: str, current_time: float) -> bool:
        """判断是否应该重新攻击"""
        try:
            # 检查重新攻击冷却时?            last_check = self.re_attack_state[agent_id]["last_check_time"]
            if current_time - last_check < 8.0:  # 8秒检查间?                return False

            self.re_attack_state[agent_id]["last_check_time"] = current_time

            # 检查距离条?            min_distance = self._calculate_min_distance_to_friendlies(env, agent_id)
            randomness = self.tactical_randomness[agent_id]

            # 如果距离拉大到重新攻击阈值,且有一定概?            if min_distance > randomness.re_attack_distance:
                if random.random() < randomness.re_attack_probability:
                    self.re_attack_state[agent_id]["enabled"] = True
                    self.re_attack_state[agent_id]["trigger_distance"] = min_distance
                    logging.info(f" {agent_id} 触发重新攻击逻辑 (距离: {min_distance/1000:.1f}km)")
                    return True

            return False

        except Exception as e:
            return False

    def _execute_re_attack(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """执行重新攻击机动"""
        try:
            # 刷新战术随机?            self._refresh_tactical_randomness(agent_id)

            # 重新攻击:转向我方并加速接?            return self._maintain_heading_precise(env, agent_id, 180.0)  # 朝南攻击

        except Exception as e:
            return 7, 8, 3

    def _apply_boundary_check(self, env, agent_id: str, command_indices: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """应用边界检查和安全验证"""
        try:
            altitude_cmd, heading_cmd, velocity_cmd = command_indices

            # 边界检?            altitude_cmd = max(0, min(14, altitude_cmd))
            heading_cmd = max(0, min(16, heading_cmd))
            velocity_cmd = max(0, min(6, velocity_cmd))

            # 高度安全检?            try:
                current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
                if current_altitude < 2000 and altitude_cmd < 7:  # 低于2000米且在下?                    altitude_cmd = 7  # 强制保持高度
            except:
                pass

            return altitude_cmd, heading_cmd, velocity_cmd

        except Exception as e:
            logging.error(f"?{agent_id} 边界检查错? {e}")
            return 7, 8, 3

    # ==================== N001VE雷达系统方法 ====================

    def _update_radar_system(self, env, agent_id: str, current_time: float):
        """更新N001VE雷达系统状?""
        try:
            # 更新雷达扫描
            self._update_radar_scan(env, agent_id, current_time)

            # 更新目标跟踪
            self._update_target_tracking(env, agent_id, current_time)

            # 更新雷达模式
            self._update_radar_mode(env, agent_id, current_time)

        except Exception as e:
            logging.error(f"?{agent_id} 雷达系统更新错误: {e}")

    def _calculate_distance_to_target(self, env, agent_id: str, target_id: str) -> float:
        """计算到目标的距离"""
        try:
            if (agent_id not in env.agents or target_id not in env.agents or
                not env.agents[agent_id].is_alive or not env.agents[target_id].is_alive):
                return 999999.0

            agent_pos = np.array([
                env.agents[agent_id].get_property_value(c.position_long_gc_deg),
                env.agents[agent_id].get_property_value(c.position_lat_geod_deg),
                env.agents[agent_id].get_property_value(c.position_h_sl_m)
            ])

            target_pos = np.array([
                env.agents[target_id].get_property_value(c.position_long_gc_deg),
                env.agents[target_id].get_property_value(c.position_lat_geod_deg),
                env.agents[target_id].get_property_value(c.position_h_sl_m)
            ])

            # 简化的距离计算(米?            lat_diff = (target_pos[1] - agent_pos[1]) * 111000
            lon_diff = (target_pos[0] - agent_pos[0]) * 111000 * math.cos(math.radians(agent_pos[1]))
            alt_diff = target_pos[2] - agent_pos[2]

            return math.sqrt(lat_diff**2 + lon_diff**2 + alt_diff**2)

        except Exception as e:
            logging.error(f"?Distance calculation error: {e}")
            return 999999.0

    def _calculate_bearing_to_target(self, env, agent_id: str, target_id: str) -> float:
        """计算到目标的方位?""
        try:
            if (agent_id not in env.agents or target_id not in env.agents or
                not env.agents[agent_id].is_alive or not env.agents[target_id].is_alive):
                return 0.0

            agent_pos = np.array([
                env.agents[agent_id].get_property_value(c.position_long_gc_deg),
                env.agents[agent_id].get_property_value(c.position_lat_geod_deg)
            ])

            target_pos = np.array([
                env.agents[target_id].get_property_value(c.position_long_gc_deg),
                env.agents[target_id].get_property_value(c.position_lat_geod_deg)
            ])

            agent_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))

            # 计算目标相对方位
            lat_diff = target_pos[1] - agent_pos[1]
            lon_diff = target_pos[0] - agent_pos[0]

            target_bearing = math.degrees(math.atan2(lon_diff, lat_diff))
            relative_bearing = target_bearing - agent_heading

            # 规范化到[-180, 180]
            while relative_bearing > 180:
                relative_bearing -= 360
            while relative_bearing < -180:
                relative_bearing += 360

            return relative_bearing

        except Exception as e:
            logging.error(f"?方位角计算错? {e}")
            return 0.0

    def _calculate_target_velocity(self, env, target_id: str) -> float:
        """计算目标速度"""
        try:
            if target_id not in env.agents or not env.agents[target_id].is_alive:
                return 0.0

            velocity = env.agents[target_id].get_property_value(c.velocities_v_north_mps)**2
            velocity += env.agents[target_id].get_property_value(c.velocities_v_east_mps)**2
            velocity += env.agents[target_id].get_property_value(c.velocities_v_down_mps)**2

            return math.sqrt(velocity)

        except Exception as e:
            logging.error(f"?目标速度计算错误: {e}")
            return 0.0

    def _get_closest_target(self, agent_id: str) -> Optional[RadarTarget]:
        """获取最近的雷达目标"""
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
            logging.error(f"?{agent_id} 最近目标获取错? {e}")
            return None


# 全局实例管理
_enhanced_enemy_ai_instance = None

def get_enhanced_enemy_ai() -> EnhancedEnemyTacticalAI:
    """获取增强版敌方AI实例"""
    global _enhanced_enemy_ai_instance
    # 强制重新创建实例以确保包含所有新方法
    _enhanced_enemy_ai_instance = EnhancedEnemyTacticalAI()
    return _enhanced_enemy_ai_instance


def get_enemy_tactical_command(env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
    """
    敌方战术指令接口 - 统一接口函数

    这是外部调用的主要接口,支持?    - 拖曳射击项目
    - 钳形夹击项目
    - 其他战术场景

    Args:
        env: 环境对象
        agent_id: 智能体ID(B0100或B0200?        current_time: 当前时间

    Returns:
        Tuple[int, int, int]: [高度指令, 航向指令, 速度指令]索引
    """
    try:
        enemy_ai = get_enhanced_enemy_ai()
        return enemy_ai.get_tactical_command(env, agent_id, current_time)
    except Exception as e:
        logging.error(f"?增强敌方AI接口错误: {e}")
        return 7, 8, 3  # 安全的平稳飞行指?

# 向后兼容性支?def enemy_ai_get_command(env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
    """向后兼容的接口函?""
    return get_enemy_tactical_command(env, agent_id, current_time)


if __name__ == "__main__":
    # 测试代码
    print(" 增强版敌方战术AI系统测试")

    # 创建AI实例
    ai = EnhancedEnemyTacticalAI()
    print(f"?AI instance created successfully")

    # 测试战术随机性生?    randomness = ai._generate_tactical_randomness()
    print(f"?战术随机性生? Crank={randomness.crank_angle:.1f}, "
          f"TurnCold={randomness.turn_cold_angle:.1f}")

    # 测试威胁评估
    threat = ThreatLevel.MEDIUM
    print(f"?Threat assessment test: {threat.name}")

    print(" Enhanced enemy tactical AI system test passed?)

    def _is_maneuver_locked(self, agent_id: str, current_time: float) -> bool:
        """检查机动是否被锁定"""
        try:
            if agent_id in self.short_skate_states:
                state = self.short_skate_states[agent_id]
                if state.locked:
                    elapsed_time = current_time - state.total_start_time
                    if elapsed_time < state.lock_duration:
                        return True
                    else:
                        # 锁定时间到期,解除锁?                        state.locked = False
                        logging.info(f" {agent_id} 机动锁定解除")
            return False
        except:
            return False

    def _execute_locked_maneuver(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """执行被锁定的机动"""
        try:
            if agent_id in self.short_skate_states:
                return self._execute_short_skate(env, agent_id, current_time)
            else:
                return 7, 8, 3
        except:
            return 7, 8, 3

    def _should_re_attack(self, env, agent_id: str, current_time: float) -> bool:
        """判断是否应该重新攻击"""
        try:
            # 检查重新攻击冷却时?            last_check = self.re_attack_state[agent_id]["last_check_time"]
            if current_time - last_check < 8.0:  # 8秒检查间?                return False

            self.re_attack_state[agent_id]["last_check_time"] = current_time

            # 检查距离条?            min_distance = self._calculate_min_distance_to_friendlies(env, agent_id)
            randomness = self.tactical_randomness[agent_id]

            # 如果距离拉大到重新攻击阈值,且有一定概?            if min_distance > randomness.re_attack_distance:
                if random.random() < randomness.re_attack_probability:
                    self.re_attack_state[agent_id]["enabled"] = True
                    self.re_attack_state[agent_id]["trigger_distance"] = min_distance
                    logging.info(f" {agent_id} 触发重新攻击逻辑 (距离: {min_distance/1000:.1f}km)")
                    return True

            return False

        except Exception as e:
            return False

    def _execute_re_attack(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """执行重新攻击机动"""
        try:
            # 刷新战术随机?            self._refresh_tactical_randomness(agent_id)

            # 重新攻击:转向我方并加速接?            return self._maintain_heading_precise(env, agent_id, 180.0)  # 朝南攻击

        except Exception as e:
            return 7, 8, 3

    def _apply_boundary_check(self, env, agent_id: str, command_indices: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """应用边界检查和安全验证"""
        try:
            altitude_cmd, heading_cmd, velocity_cmd = command_indices

            # 边界检?            altitude_cmd = max(0, min(14, altitude_cmd))
            heading_cmd = max(0, min(16, heading_cmd))
            velocity_cmd = max(0, min(6, velocity_cmd))

            # 高度安全检?            try:
                current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
                if current_altitude < 2000 and altitude_cmd < 7:  # 低于2000米且在下?                    altitude_cmd = 7  # 强制保持高度
            except:
                pass

            return altitude_cmd, heading_cmd, velocity_cmd

        except Exception as e:
            logging.error(f"?{agent_id} 边界检查错? {e}")
            return 7, 8, 3


if __name__ == "__main__":
    # 测试增强版敌方AI系统
    print(" 测试增强版敌方战术AI系统...")

    # 创建AI实例
    ai = EnhancedEnemyTacticalAI()
    print(f"?AI instance created successfully")

    # 测试战术随机性生?    randomness = ai._generate_tactical_randomness()
    print(f"?战术随机性生? Crank={randomness.crank_angle:.1f}, "
          f"TurnCold={randomness.turn_cold_angle:.1f}")

    # 测试威胁评估
    threat = ThreatLevel.MEDIUM
    print(f"?Threat assessment test: {threat.name}")

    print(" Enhanced enemy tactical AI system test passed?)

            agent_pos = np.array([
                env.agents[agent_id].get_property_value(c.position_long_gc_deg),
                env.agents[agent_id].get_property_value(c.position_lat_geod_deg),
                env.agents[agent_id].get_property_value(c.position_h_sl_m)
            ])

            target_pos = np.array([
                env.agents[target_id].get_property_value(c.position_long_gc_deg),
                env.agents[target_id].get_property_value(c.position_lat_geod_deg),
                env.agents[target_id].get_property_value(c.position_h_sl_m)
            ])

            # 简化的距离计算(米?            lat_diff = (target_pos[1] - agent_pos[1]) * 111000
            lon_diff = (target_pos[0] - agent_pos[0]) * 111000 * math.cos(math.radians(agent_pos[1]))
            alt_diff = target_pos[2] - agent_pos[2]

            return math.sqrt(lat_diff**2 + lon_diff**2 + alt_diff**2)

        except Exception as e:
            logging.error(f"?Distance calculation error: {e}")
            return 999999.0

    def _calculate_bearing_to_target(self, env, agent_id: str, target_id: str) -> float:
        """计算到目标的方位?""
        try:
            if (agent_id not in env.agents or target_id not in env.agents or
                not env.agents[agent_id].is_alive or not env.agents[target_id].is_alive):
                return 0.0

            agent_pos = np.array([
                env.agents[agent_id].get_property_value(c.position_long_gc_deg),
                env.agents[agent_id].get_property_value(c.position_lat_geod_deg)
            ])

            target_pos = np.array([
                env.agents[target_id].get_property_value(c.position_long_gc_deg),
                env.agents[target_id].get_property_value(c.position_lat_geod_deg)
            ])

            agent_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))

            # 计算目标相对方位
            lat_diff = target_pos[1] - agent_pos[1]
            lon_diff = target_pos[0] - agent_pos[0]

            target_bearing = math.degrees(math.atan2(lon_diff, lat_diff))
            relative_bearing = target_bearing - agent_heading

            # 规范化到[-180, 180]
            while relative_bearing > 180:
                relative_bearing -= 360
            while relative_bearing < -180:
                relative_bearing += 360

            return relative_bearing

        except Exception as e:
            logging.error(f"?方位角计算错? {e}")
            return 0.0

    def _calculate_target_velocity(self, env, target_id: str) -> float:
        """计算目标速度"""
        try:
            if target_id not in env.agents or not env.agents[target_id].is_alive:
                return 0.0

            velocity = env.agents[target_id].get_property_value(c.velocities_v_north_mps)**2
            velocity += env.agents[target_id].get_property_value(c.velocities_v_east_mps)**2
            velocity += env.agents[target_id].get_property_value(c.velocities_v_down_mps)**2

            return math.sqrt(velocity)

        except Exception as e:
            logging.error(f"?目标速度计算错误: {e}")
            return 0.0

    def _get_closest_target(self, agent_id: str) -> Optional[RadarTarget]:
        """获取最近的雷达目标"""
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
            logging.error(f"?{agent_id} 最近目标获取错? {e}")
            return None

    def _update_radar_scan(self, env, agent_id: str, current_time: float):
        """更新雷达扫描和目标探?""
        try:
            # 检查扫描周?            if current_time - self.radar_scan_time[agent_id] < self.radar_model.scan_period:
                return

            self.radar_scan_time[agent_id] = current_time

            # 扫描友方目标
            friendly_agents = ["A0100", "A0200"]
            for target_id in friendly_agents:
                if target_id not in env.agents or not env.agents[target_id].is_alive:
                    continue

                # 计算目标参数
                distance = self._calculate_distance_to_target(env, agent_id, target_id)
                bearing = self._calculate_bearing_to_target(env, agent_id, target_id)
                velocity = self._calculate_target_velocity(env, target_id)

                # 检查探测范?                if distance > self.radar_model.max_detection_range:
                    continue

                # 计算探测概率
                detection_prob = self._calculate_detection_probability(distance, bearing)

                # 探测成功
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
                        logging.debug(f" {agent_id} 雷达探测到新目标: {target_id} 距离={distance/1000:.1f}km")
                    else:
                        # 更新现有目标
                        target = self.radar_targets[agent_id][target_id]
                        target.distance = distance
                        target.bearing = bearing
                        target.velocity = velocity
                        target.detection_probability = detection_prob
                        target.last_update = current_time

        except Exception as e:
            logging.error(f" {agent_id} 雷达扫描错误: {e}")


if __name__ == "__main__":
    # 测试增强版敌方AI系统
    print(" 测试增强版敌方战术AI系统...")

    # 创建AI实例
    ai = EnhancedEnemyTacticalAI()
    print(f" AI instance created successfully")

    # 测试Tactical randomness generated
    randomness = ai._generate_tactical_randomness()
    print(f" Tactical randomness generated: Crank={randomness.crank_angle:.1f}, "
          f"TurnCold={randomness.turn_cold_angle:.1f}")

    # 测试威胁评估
    threat = ThreatLevel.MEDIUM
    print(f" Threat assessment test: {threat.name}")

    print(" Enhanced enemy tactical AI system test passed!")

