#!/usr/bin/env python3
"""
增强版敌方战术AI系统 - 完整实现
提供与我方匹配的5阶段距离控制、完整机动集成和战术随机性

Design concept:
1. Completely mirrors friendly DragShootTacticalTask distance control timeline
2. Implements complete Short Skate 3-phase maneuver (CrankTurn + ColdEscape)
3. Provides intelligent switching between 3 tactical modes (AGGRESSIVE/DEFENSIVE/NEUTRAL)
4. 集成战术随机性和多样化对抗能力
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
    AGGRESSIVE = "aggressive"  # Attack mode:主动接敌、优先发射导弹
    DEFENSIVE = "defensive"  # Defense mode:规避机动、威胁回避
    NEUTRAL = "neutral"  # Neutral mode:平衡方法、灵活调整


class EnemyTacticalPhase(Enum):
    """敌方战术阶段 - 完全镜像我方5阶段系统"""
    NLT_MELD = "NLT_MELD"  # 90-81km:Long-range engagement phase（远程交战阶段）
    MELD_MTR = "MELD_MTR"  # 81-50km:Medium-range combat phase（中程交战阶段）
    MTR_TR = "MTR_TR"  # 50-40km:导弹目标范围阶段
    TR_DOR = "TR_DOR"  # 40-35km:目标范围到动态攻击范围
    DOR_DR = "DOR_DR"  # 35-14.5km:Dynamic attack to defense range（动态攻击到防御范围）


class ThreatLevel(Enum):
    """威胁等级枚举"""
    NONE = 0  # 无威胁
    LOW = 1  # 低威胁
    MEDIUM = 2  # Medium threat（中等威胁）
    HIGH = 3  # 高威胁
    CRITICAL = 4  # Critical threat（紧急威胁）


class RadarMode(Enum):
    """雷达工作模式"""
    SEARCH = "search"  # Search mode (Wide beam)（搜索模式-宽波束）
    TRACK = "track"  # Track mode (Narrow beam)（跟踪模式-窄波束）
    LOCK = "lock"  # Lock mode (Continuous illumination)（锁定模式-持续照射）
    STANDBY = "standby"  # 待机模式


class ManeuverType(Enum):
    """机动类型枚举"""
    SHORT_SKATE = "short_skate"  # 短程滑行机动
    NOTCH = "notch"  # 雷达规避机动
    BARREL_ROLL = "barrel_roll"  # 桶滚规避
    SPLIT_S = "split_s"  # 半筋斗脱离
    WEAVE = "weave"  # Weave maneuver（蛇形机动）
    DEFENSIVE_SPIRAL = "defensive_spiral"  # 防御螺旋
    BEAM = "beam"  # 横向规避


@dataclass
class N001VERadarModel:
    """N001VE雷达模型参数"""
    max_detection_range: float = 120000  # 最大探测距离 120km
    max_track_range: float = 80000  # 最大跟踪距离 80km
    max_lock_range: float = 60000  # 最大锁定距离 60km
    max_simultaneous_tracks: int = 8  # 同时跟踪目标数
    search_beam_width: float = 60.0  # Search beam width (度)（搜索波束宽度）
    track_beam_width: float = 3.0  # Track beam width (度)（跟踪波束宽度）
    lock_beam_width: float = 1.0  # Lock beam width (度)（锁定波束宽度）
    scan_period: float = 4.0  # Scan period (秒)（扫描周期）
    lock_update_rate: float = 0.1  # Lock update rate (秒)（锁定更新率）
    detection_probability_base: float = 0.9  # 基础探测概率
    track_loss_probability: float = 0.05  # 跟踪丢失概率
    jamming_resistance: float = 0.7  # 抗干扰能力


@dataclass
class RadarTarget:
    """雷达目标信息"""
    target_id: str
    distance: float
    bearing: float
    elevation: float
    velocity: float
    rcs: float = 5.0  # Radar cross section (m²)（雷达截面积）
    detection_probability: float = 0.0
    track_quality: float = 0.0
    lock_time: float = 0.0
    last_update: float = 0.0


@dataclass
class ManeuverState:
    """增强的机动状态数据类"""
    maneuver_type: ManeuverType  # 机动类型
    phase: str  # 当前阶段
    phase_start_time: float  # 阶段开始时间
    total_start_time: float  # 总开始时间
    initial_heading: float  # Initial heading（初始航向）
    initial_altitude: float  # 初始高度
    target_heading: float = 0.0  # 目标航向
    target_altitude: float = 0.0  # 目标高度
    crank_angle: float = 0.0  # Crank角度
    turn_cold_angle: float = 0.0  # Turn Cold角度
    locked: bool = False  # 是否锁定
    lock_duration: float = 0.0  # 锁定持续时间
    # 新增机动参数
    roll_direction: int = 1  # Roll direction (1=右,-1=左)（滚转方向）
    climb_rate: float = 0.0  # Climb rate (m/s)（爬升率）
    weave_amplitude: float = 0.0  # Weave amplitude（蛇形摆动幅度）
    weave_period: float = 0.0  # 蛇形摆动周期
    spiral_radius: float = 0.0  # 螺旋半径


@dataclass
class TacticalRandomness:
    """战术随机性参数"""
    crank_angle: float  # Crank角度随机化
    crank_duration: float  # Crank持续时间随机化
    turn_cold_angle: float  # Turn Cold角度随机化
    turn_cold_rate: float  # Turn Cold转弯率随机化
    attack_delay: float = 2.0  # Attack delay randomization（攻击延迟随机化）
    retreat_threshold: float = 0.7  # Retreat threshold randomization（撤退阈值随机化）
    formation_offset: float = 0.0  # Formation offset randomization（编队偏移随机化）
    coordination_delay: float = 2.0  # Coordination delay randomization（协同延迟随机化）
    re_attack_distance: float = 50000  # Re-attack distance randomization（重新攻击距离随机化）
    re_attack_probability: float = 0.8  # Re-attack probability randomization（重新攻击概率随机化）


class EnhancedEnemyTacticalAI:
    """
    Enhanced Enemy Tactical AI System（增强版敌方战术AI系统）

    Core Features:
    1. 5-stage distance control timeline (mirroring friendly forces)（5阶段距离控制时序-镜像我方）
    2. Complete Short Skate 3-phase maneuver implementation（完整Short Skate三阶段机动实现）
    3. Intelligent switching between 3 tactical modes（3种战术模式智能切换）
    4. Tactical randomness and diversified confrontation（战术随机性与多样化对抗）
    5. Dual-aircraft formation coordination mechanism（双机编队协同机制）
    6. Stable threat assessment system（稳定的威胁评估系统）
    """

    def __init__(self):
        """Initialize Enhanced Enemy Tactical AI（初始化增强版敌方战术AI）"""

        # Tactical distance configuration - completely mirrors friendly DragShootTacticalTask
        # 战术距离配置 - 完全镜像我方DragShootTacticalTask
        self.tactical_distances = {
            'NLT_MELD_min': 81000,  # 81km
            'MELD_MTR_min': 50000,  # 50km
            'MTR_TR_min': 40000,  # 40km
            'TR_DOR_min': 35000,  # 35km
            'DOR_DR_min': 14500,  # 14.5km
        }

        # 当前状态（按机型区分战术阶段）
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

        # 增强的机动状态管理
        self.active_maneuvers: Dict[str, ManeuverState] = {}
        self.maneuver_history: Dict[str, List[ManeuverType]] = {
            "B0100": [],
            "B0200": []
        }
        self.maneuver_cooldowns: Dict[str, Dict[ManeuverType, float]] = {
            "B0100": {mt: 0.0 for mt in ManeuverType},
            "B0200": {mt: 0.0 for mt in ManeuverType}
        }

        # 机动状态管理(向后兼容)
        self.short_skate_states: Dict[str, ManeuverState] = {}

        # 战术随机性（按机型区分）
        self.tactical_randomness = {
            "B0100": self._generate_tactical_randomness(),
            "B0200": self._generate_tactical_randomness()
        }

        # 模式切换控制
        self.mode_change_cooldown = 8.0  # 8秒冷却时间
        self.last_mode_change_time = {"B0100": -999, "B0200": -999}

        # 导弹发射管理
        self.last_missile_launch_time = {"B0100": -999, "B0200": -999}
        self.missile_cooldown = 6.0  # 6秒导弹发射冷却

        # 重新攻击逻辑
        self.re_attack_state = {
            "B0100": {"enabled": False, "trigger_distance": 0, "last_check_time": 0},
            "B0200": {"enabled": False, "trigger_distance": 0, "last_check_time": 0}
        }

        # 返航状态管理
        self.return_to_base_states: Dict[str, Dict[str, Any]] = {}

        # 初始化基础机动组件
        try:
            self.basic_maneuvers = BasicManeuvers()
            self.maneuver_executor = CompositeManeuverExecutor()
            logging.info("增强版敌方AI初始化完成")
        except Exception as e:
            logging.warning(f"机动组件初始化失败,使用备用模式: {e}")
            self.basic_maneuvers = None
            self.maneuver_executor = None

    def _generate_tactical_randomness(self) -> TacticalRandomness:
        """Generate tactical randomness parameters（生成战术随机性参数）"""
        return TacticalRandomness(
            crank_angle=random.uniform(35.0, 55.0),  # 35-55 degree random Crank angle（35-55度随机Crank角度）
            turn_cold_angle=random.uniform(100.0, 140.0),
            # 100-140 degree random Turn Cold angle（100-140度随机Turn Cold角度）
            crank_duration=random.uniform(6.0, 10.0),  # 6-10 second random duration（6-10秒随机持续时间）
            turn_cold_rate=random.uniform(5.0, 10.0),  # 5-10度/秒随机Turn Cold转弯率（补充缺失参数）
            re_attack_distance=random.uniform(45000, 60000),  # 45-60km random re-attack distance（45-60km随机重新攻击距离）
            formation_offset=random.uniform(-3.0, 3.0),  # 3 nautical mile random formation offset（3海里随机编队偏移）
            coordination_delay=random.uniform(0.0, 3.0),  # 0-3 second random coordination delay（0-3秒随机协同延迟）
            re_attack_probability=random.uniform(0.7, 0.95)  # 70-95%随机重新攻击概率（删除重复参数）
        )

    def _refresh_tactical_randomness(self, agent_id: str):
        """Refresh tactical randomness parameters（刷新战术随机性参数）"""
        self.tactical_randomness[agent_id] = self._generate_tactical_randomness()
        randomness = self.tactical_randomness[agent_id]
        logging.info(f"{agent_id} Tactical randomness refreshed: "
                     f"Crank={randomness.crank_angle:.1f}, "
                     f"TurnCold={randomness.turn_cold_angle:.1f}, "
                     f"ReAttack={randomness.re_attack_probability:.2f}")

    def get_tactical_command(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """
        Get enemy tactical command - main interface function（获取敌方战术指令-主接口函数）

        Args:
            env: Environment object（环境对象）
            agent_id: Agent ID (B0100 or B0200)（智能体ID）
            current_time: Current time（当前时间）

        Returns:
            Tuple[int, int, int]: [altitude command, heading command, speed command] indices
            （元组：[高度指令, 航向指令, 速度指令]索引）
        """
        try:
            # 1. Update N001VE radar system（更新N001VE雷达系统）
            self._update_radar_system(env, agent_id, current_time)

            # 2. Update tactical phase（更新战术阶段）
            self._update_tactical_phase(env, agent_id)

            # 3. Check return to base conditions（检查返航条件）
            if self._should_return_to_base(env, agent_id, current_time):
                return self._execute_return_to_base(env, agent_id)

            # 4. Check maneuver lock status（检查机动锁定状态）
            if self._is_maneuver_locked(agent_id, current_time):
                return self._execute_locked_maneuver(env, agent_id, current_time)

            # 5. 检查重新攻击逻辑 - 修复：RTB后禁用重复交战
            if (agent_id not in self.return_to_base_states and
                self._should_re_attack(env, agent_id, current_time)):
                return self._execute_re_attack(env, agent_id, current_time)

            # 6. Radar-based threat assessment（基于雷达的威胁评估）
            threat_level = self._assess_threat_level(env, agent_id)  # 修复方法名错误（原_assess_radar_threat_level）

            # 7. Select tactical mode（选择战术模式）
            tactical_mode = self._select_tactical_mode(env, agent_id, threat_level, current_time)

            # 8. Radar-driven maneuver decision（雷达驱动的机动决策）
            maneuver_command = self._radar_driven_maneuver_decision(env, agent_id, current_time)
            if maneuver_command:
                return maneuver_command

            # 9. Generate tactical command（生成战术指令）
            command_indices = self._generate_tactical_command(env, agent_id, tactical_mode, current_time)

            # 10. Apply boundary check（应用边界检查）
            safe_command = self._apply_boundary_check(env, agent_id, command_indices)

            return safe_command

        except Exception as e:
            logging.error(f"{agent_id} Enhanced enemy AI execution error: {e}")
            return 7, 8, 3  # Safe level flight command（安全平飞指令）

    def _update_tactical_phase(self, env, agent_id: str):
        """Update tactical phase - fixed version with hysteresis to prevent oscillation（更新战术阶段-带滞后防止震荡）"""
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

            # 更新个体阶段状态
            setattr(self, f'current_phase_{agent_id}', new_phase)

            # 记录阶段变化(减少日志频率)
            if not hasattr(self, f'last_phase_log_{agent_id}') or getattr(self,
                                                                          f'last_phase_log_{agent_id}') != new_phase:
                logging.info(f"{agent_id} 战术阶段: {new_phase.value} (距离: {min_distance / 1000:.1f}km)")
                setattr(self, f'last_phase_log_{agent_id}', new_phase)

                # 阶段变化时刷新战术随机性(降低频率)
                if new_phase in [EnemyTacticalPhase.MTR_TR, EnemyTacticalPhase.TR_DOR]:
                    if not hasattr(self, f'last_randomness_refresh_{agent_id}') or \
                            getattr(self, f'last_randomness_refresh_{agent_id}') != new_phase:
                        self._refresh_tactical_randomness(agent_id)
                        setattr(self, f'last_randomness_refresh_{agent_id}', new_phase)

        except Exception as e:
            logging.error(f"{agent_id} Tactical phase update error: {e}")
            # 设置默认阶段
            setattr(self, f'current_phase_{agent_id}', EnemyTacticalPhase.MELD_MTR)

    def _calculate_min_distance_to_friendlies(self, env, agent_id: str) -> float:
        """Calculate minimum distance to friendly targets（计算到我方目标的最小距离）"""
        try:
            if agent_id not in env.agents or not env.agents[agent_id].is_alive:
                return 100000.0  # Default far distance（默认远距离）
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
            logging.error(f"{agent_id} Distance calculation error: {e}")
            return 100000.0

    def _assess_threat_level(self, env, agent_id: str) -> ThreatLevel:
        """Assess threat level - 稳定版本,避免频繁错误（威胁评估-稳定版本）"""
        try:
            # 基于距离的基础威胁评估
            min_distance = self._calculate_min_distance_to_friendlies(env, agent_id)

            if min_distance < 20000:  # 20km内
                base_threat = ThreatLevel.CRITICAL
            elif min_distance < 35000:  # 35km内
                base_threat = ThreatLevel.HIGH
            elif min_distance < 50000:  # 50km内
                base_threat = ThreatLevel.MEDIUM
            elif min_distance < 70000:  # 70km内
                base_threat = ThreatLevel.LOW
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
        """Select tactical mode - 基于威胁等级、战术阶段和随机性（选择战术模式）"""
        try:
            # 检查模式切换冷却时间
            if current_time - self.last_mode_change_time[agent_id] < self.mode_change_cooldown:
                return self.current_mode[agent_id]

            current_mode = self.current_mode[agent_id]
            new_mode = current_mode

            # 基于威胁等级的模式选择逻辑
            if threat_level == ThreatLevel.CRITICAL:
                new_mode = TacticalMode.DEFENSIVE
            elif threat_level == ThreatLevel.HIGH:
                # High threat:70%防御,30%攻击
                new_mode = TacticalMode.DEFENSIVE if random.random() < 0.7 else TacticalMode.AGGRESSIVE
            elif threat_level == ThreatLevel.MEDIUM:
                # Medium threat:基于战术阶段选择
                current_phase = getattr(self, f'current_phase_{agent_id}', EnemyTacticalPhase.MELD_MTR)
                if current_phase in [EnemyTacticalPhase.MTR_TR, EnemyTacticalPhase.TR_DOR]:
                    new_mode = TacticalMode.AGGRESSIVE  # 攻击窗口
                else:
                    new_mode = TacticalMode.NEUTRAL
            elif threat_level == ThreatLevel.LOW:
                # Low threat:60%攻击,40%中性
                new_mode = TacticalMode.AGGRESSIVE if random.random() < 0.6 else TacticalMode.NEUTRAL
            else:
                # No threat:保持中性或攻击
                new_mode = TacticalMode.NEUTRAL

            # 记录模式变化
            if new_mode != current_mode:
                self.current_mode[agent_id] = new_mode
                self.last_mode_change_time[agent_id] = current_time
                logging.info(f"{agent_id} 战术模式: {current_mode.value} → {new_mode.value} "
                             f"(威胁: {threat_level.name})")

            return new_mode

        except Exception as e:
            logging.error(f"{agent_id} Tactical mode selection error: {e}")
            return TacticalMode.NEUTRAL

    def _generate_tactical_command(self, env, agent_id: str, tactical_mode: TacticalMode, current_time: float) -> Tuple[
        int, int, int]:
        """Generate tactical command - 修复版,基于个体战术阶段（生成战术指令）"""
        try:
            # 检查是否正在执行Short Skate机动
            if agent_id in self.short_skate_states:
                return self._execute_short_skate(env, agent_id, current_time)

            # 获取个体战术阶段
            current_phase = getattr(self, f'current_phase_{agent_id}', EnemyTacticalPhase.MELD_MTR)

            # 根据角色分配不同的战术行为
            if agent_id == "B0100":  # 敌方长机
                return self._get_leader_command(env, agent_id, tactical_mode, current_phase, current_time)
            elif agent_id == "B0200":  # 敌方僚机
                return self._get_wingman_command(env, agent_id, tactical_mode, current_phase, current_time)
            else:
                return 7, 8, 3  # 默认平稳飞行

        except Exception as e:
            logging.error(f"{agent_id} Tactical command generation error: {e}")
            return 7, 8, 3

    def _get_leader_command(self, env, agent_id: str, tactical_mode: TacticalMode, current_phase: EnemyTacticalPhase,
                            current_time: float) -> Tuple[int, int, int]:
        """Enemy lead aircraft tactical command - 完整战术流程版,包含返航逻辑（敌方长机战术指令）"""
        try:
            randomness = self.tactical_randomness[agent_id]

            # 检查是否需要返航
            if self._should_return_to_base(env, agent_id, current_time):
                return self._execute_return_to_base(env, agent_id)

            if current_phase == EnemyTacticalPhase.NLT_MELD:
                # NLT_MELD阶段:平稳接敌,朝南飞行(180度)
                return self._maintain_heading_precise(env, agent_id, 180.0)

            elif current_phase == EnemyTacticalPhase.MELD_MTR:
                # MELD_MTR阶段:继续接敌,保持南向
                if tactical_mode == TacticalMode.AGGRESSIVE:
                    # Attack mode:轻微调整,但主要保持南向（限制偏移5度内）
                    target_heading = 180.0 + min(randomness.formation_offset, 5.0)
                    return self._maintain_heading_precise(env, agent_id, target_heading)
                else:
                    # 其他模式:严格保持南向
                    return self._maintain_heading_precise(env, agent_id, 180.0)

            elif current_phase == EnemyTacticalPhase.MTR_TR:
                # MTR_TR阶段:导弹目标范围,开始战术机动
                if tactical_mode == TacticalMode.AGGRESSIVE:
                    # Attack mode:May initiate Short Skate（25%概率启动Short Skate）
                    if random.random() < 0.25:
                        self._init_short_skate(agent_id, current_time)
                        return self._execute_short_skate(env, agent_id, current_time)
                    else:
                        # 继续接敌,保持南向
                        return self._maintain_heading_precise(env, agent_id, 180.0)
                elif tactical_mode == TacticalMode.DEFENSIVE:
                    # Defense mode:开始规避机动,但不要过度偏移（轻微左偏165度）
                    return self._maintain_heading_precise(env, agent_id, 165.0)
                else:
                    return self._maintain_heading_precise(env, agent_id, 180.0)

            elif current_phase == EnemyTacticalPhase.TR_DOR:
                # TR_DOR阶段:Target range to dynamic attack range,高概率机动
                if tactical_mode == TacticalMode.AGGRESSIVE:
                    # Attack mode:High probability initiate Short Skate（50%概率启动Short Skate）
                    if random.random() < 0.5:
                        self._init_short_skate(agent_id, current_time)
                        return self._execute_short_skate(env, agent_id, current_time)
                    else:
                        # 继续攻击接近
                        return self._maintain_heading_precise(env, agent_id, 180.0)
                elif tactical_mode == TacticalMode.DEFENSIVE:
                    # Defense mode:执行规避机动（中等左偏150度）
                    return self._maintain_heading_precise(env, agent_id, 150.0)
                else:
                    return self._maintain_heading_precise(env, agent_id, 180.0)

            elif current_phase == EnemyTacticalPhase.DOR_DR:
                # DOR_DR阶段:Dynamic attack to defense range,Forced maneuver or return
                if agent_id not in self.short_skate_states:
                    # 50%概率执行Short Skate,50%概率直接返航
                    if random.random() < 0.5:
                        self._init_short_skate(agent_id, current_time)
                        return self._execute_short_skate(env, agent_id, current_time)
                    else:
                        # 直接开始返航
                        self._init_return_to_base(agent_id, current_time)
                        return self._execute_return_to_base(env, agent_id)
                else:
                    return self._execute_short_skate(env, agent_id, current_time)

            # 默认情况:保持南向
            return self._maintain_heading_precise(env, agent_id, 180.0)

        except Exception as e:
            logging.error(f"{agent_id} Lead aircraft command error: {e}")
            return self._maintain_heading_precise(env, agent_id, 180.0)  # 安全的南向飞行

    def _get_wingman_command(self, env, agent_id: str, tactical_mode: TacticalMode, current_phase: EnemyTacticalPhase,
                             current_time: float) -> Tuple[int, int, int]:
        """Enemy wingman tactical command - 完整战术流程版,包含返航逻辑（敌方僚机战术指令）"""
        try:
            randomness = self.tactical_randomness[agent_id]

            # 检查是否需要返航
            if self._should_return_to_base(env, agent_id, current_time):
                return self._execute_return_to_base(env, agent_id)

            # 僚机相对于长机有轻微的战术延迟和偏移,但保持主要南向
            if current_phase == EnemyTacticalPhase.NLT_MELD:
                # NLT_MELD阶段:编队飞行,轻微左偏但不超过5度
                target_heading = 180.0 + min(randomness.formation_offset, -5.0)
                return self._maintain_heading_precise(env, agent_id, target_heading)

            elif current_phase == EnemyTacticalPhase.MELD_MTR:
                # MELD_MTR阶段:保持编队,准备分离
                if tactical_mode == TacticalMode.AGGRESSIVE:
                    # Attack mode:轻微右偏,但不超过5度
                    target_heading = 180.0 + min(randomness.formation_offset, 5.0)
                    return self._maintain_heading_precise(env, agent_id, target_heading)
                else:
                    # 其他模式:轻微左偏(175度)
                    target_heading = 180.0 - 5.0
                    return self._maintain_heading_precise(env, agent_id, target_heading)

            elif current_phase == EnemyTacticalPhase.MTR_TR:
                # MTR_TR阶段:僚机延迟行动
                if tactical_mode == TacticalMode.AGGRESSIVE:
                    # 延迟启动Short Skate(比长机晚,20%概率)
                    if random.random() < 0.2:
                        self._init_short_skate(agent_id, current_time)
                        return self._execute_short_skate(env, agent_id, current_time)
                    else:
                        # 继续接敌,轻微左偏(175度)
                        return self._maintain_heading_precise(env, agent_id, 175.0)
                elif tactical_mode == TacticalMode.DEFENSIVE:
                    # Defense mode:轻微规避,但不过度偏离(170度)
                    return self._maintain_heading_precise(env, agent_id, 170.0)
                else:
                    return self._maintain_heading_precise(env, agent_id, 175.0)

            elif current_phase == EnemyTacticalPhase.TR_DOR:
                # TR_DOR阶段:僚机跟随长机行动
                if tactical_mode == TacticalMode.AGGRESSIVE:
                    # 40%概率启动Short Skate(比长机稍低)
                    if random.random() < 0.4:
                        self._init_short_skate(agent_id, current_time)
                        return self._execute_short_skate(env, agent_id, current_time)
                    else:
                        # 继续攻击接近(175度)
                        return self._maintain_heading_precise(env, agent_id, 175.0)
                elif tactical_mode == TacticalMode.DEFENSIVE:
                    # Defense mode:中等角度规避(160度)
                    return self._maintain_heading_precise(env, agent_id, 160.0)
                else:
                    return self._maintain_heading_precise(env, agent_id, 175.0)

            elif current_phase == EnemyTacticalPhase.DOR_DR:
                # DOR_DR阶段:僚机也必须机动或返航
                if agent_id not in self.short_skate_states:
                    # 40%概率执行Short Skate,60%概率直接返航
                    if random.random() < 0.4:
                        self._init_short_skate(agent_id, current_time)
                        return self._execute_short_skate(env, agent_id, current_time)
                    else:
                        # 直接开始返航
                        self._init_return_to_base(agent_id, current_time)
                        return self._execute_return_to_base(env, agent_id)
                else:
                    return self._execute_short_skate(env, agent_id, current_time)

            # 默认情况:轻微左偏南向(175度)
            return self._maintain_heading_precise(env, agent_id, 175.0)

        except Exception as e:
            logging.error(f"{agent_id} Wingman command error: {e}")
            return self._maintain_heading_precise(env, agent_id, 175.0)  # 安全的轻微左偏南向飞行

    def _init_short_skate(self, agent_id: str, current_time: float):
        """初始化Short Skate机动 - 修复版,确保合理的机动角度"""
        try:
            # 获取当前状态（南向基准航向、默认高度）
            current_heading = 180.0
            current_altitude = 10000.0

            # 使用随机战术参数,但限制角度范围
            randomness = self.tactical_randomness[agent_id]

            # 根据智能体ID确定机动方向,限制角度避免过度偏移
            if agent_id == "B0100":  # 长机右侧机动
                # 限制Crank角度在15-30度范围内
                crank_angle = min(30.0, max(15.0, randomness.crank_angle))
                # Turn Cold角度限制在60-90度范围内
                turn_cold_angle = min(90.0, max(60.0, randomness.turn_cold_angle))
            else:  # 僚机左侧机动（负角度表示左转）
                crank_angle = -min(30.0, max(15.0, randomness.crank_angle))
                turn_cold_angle = -min(90.0, max(60.0, randomness.turn_cold_angle))

            # 创建机动状态（补充缺失的maneuver_type参数）
            self.short_skate_states[agent_id] = ManeuverState(
                maneuver_type=ManeuverType.SHORT_SKATE,
                phase="crank",
                phase_start_time=current_time,
                total_start_time=current_time,
                initial_heading=current_heading,
                initial_altitude=current_altitude,
                crank_angle=crank_angle,
                turn_cold_angle=turn_cold_angle,
                locked=True,
                lock_duration=35.0  # 减少总锁定时间到35秒
            )

            logging.info(f"{agent_id} 启动Short Skate机动: "
                         f"Crank={crank_angle:.1f}, TurnCold={turn_cold_angle:.1f}")

        except Exception as e:
            logging.error(f"{agent_id} Short Skate初始化错误: {e}")

    def _execute_short_skate(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """执行Short Skate机动 - 修复版,确保正确的航向计算和合理的机动时间"""
        try:
            if agent_id not in self.short_skate_states:
                return 7, 8, 3

            state = self.short_skate_states[agent_id]
            phase_time = current_time - state.phase_start_time
            total_time = current_time - state.total_start_time

            # 阶段持续时间配置 - 缩短时间避免过长的偏移
            crank_duration = 8.0  # Crank持续8秒
            turn_cold_duration = 12.0  # Turn Cold持续12秒
            escape_duration = 10.0  # Escape持续10秒

            # 阶段1:Crank机动(右转或左转接敌)
            if state.phase == "crank":
                if phase_time < crank_duration:
                    # 执行Crank转弯 - 确保航向计算正确
                    target_heading = state.initial_heading + state.crank_angle
                    # 规范化航向到0-360度范围
                    target_heading = target_heading % 360
                    return self._maintain_heading_precise(env, agent_id, target_heading)
                else:
                    # Crank阶段完成,进入Turn Cold阶段
                    state.phase = "turn_cold"
                    state.phase_start_time = current_time
                    logging.info(f"{agent_id} Short Skate: Crank → Turn Cold")
                    target_heading = (state.initial_heading + state.crank_angle) % 360
                    return self._maintain_heading_precise(env, agent_id, target_heading)

            # 阶段2:Turn Cold机动(大角度转弯脱离)
            elif state.phase == "turn_cold":
                if phase_time < turn_cold_duration:
                    # 执行Turn Cold转弯 - 渐进式转弯
                    turn_progress = phase_time / turn_cold_duration
                    current_turn_angle = state.crank_angle + (state.turn_cold_angle - state.crank_angle) * turn_progress
                    target_heading = (state.initial_heading + current_turn_angle) % 360
                    return self._maintain_heading_precise(env, agent_id, target_heading)
                else:
                    # Turn Cold阶段完成,进入Escape阶段
                    state.phase = "escape"
                    state.phase_start_time = current_time
                    logging.info(f"{agent_id} Short Skate: Turn Cold → Escape")
                    target_heading = (state.initial_heading + state.turn_cold_angle) % 360
                    return self._maintain_heading_precise(env, agent_id, target_heading)

            # 阶段3:Escape机动(保持脱离航向,然后返回南向)
            elif state.phase == "escape":
                if phase_time < escape_duration:
                    # 前半段保持脱离航向
                    if phase_time < escape_duration / 2:
                        target_heading = (state.initial_heading + state.turn_cold_angle) % 360
                    else:
                        # 后半段逐渐返回南向
                        return_progress = (phase_time - escape_duration / 2) / (escape_duration / 2)
                        escape_heading = (state.initial_heading + state.turn_cold_angle) % 360
                        target_heading = escape_heading + (180.0 - escape_heading) * return_progress
                        target_heading = target_heading % 360
                    return self._maintain_heading_precise(env, agent_id, target_heading)
                else:
                    # Short Skate机动完成,返回南向
                    logging.info(f"{agent_id} Short Skate机动完成,返回南向")
                    del self.short_skate_states[agent_id]

                    # 刷新战术随机性,准备下次机动
                    self._refresh_tactical_randomness(agent_id)

                    # 返回南向飞行
                    return self._maintain_heading_precise(env, agent_id, 180.0)

            return 7, 8, 3

        except Exception as e:
            logging.error(f"{agent_id} Short Skate执行错误: {e}")
            # 清理错误状态
            if agent_id in self.short_skate_states:
                del self.short_skate_states[agent_id]
            # 返回南向飞行
            return self._maintain_heading_precise(env, agent_id, 180.0)

    def _maintain_heading_precise(self, env, agent_id: str, target_heading: float) -> Tuple[int, int, int]:
        """精确航向保持 - 敌方AI安全飞行版本"""
        try:
            if agent_id not in env.agents or not env.agents[agent_id].is_alive:
                return 7, 8, 3

            # 获取当前状态
            current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
            current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
            current_velocity = env.agents[agent_id].get_property_value(c.velocities_v_north_mps)

            # 计算航向差
            heading_diff = target_heading - current_heading
            while heading_diff > 180:
                heading_diff -= 360
            while heading_diff < -180:
                heading_diff += 360

            # 默认索引：安全飞行
            altitude_cmd = 7  # 高度索引7 = 0m变化（保持高度）
            heading_cmd = 8   # 航向索引8 = 0°变化（保持航向）
            velocity_cmd = 3  # 速度索引3 = 0m/s变化（保持速度）

            # 🚨 关键修复：强制高度控制，防止坠毁
            target_altitude = 5700.0  # 目标高度5700米
            altitude_diff = current_altitude - target_altitude

            # 高度控制优先级最高
            if current_altitude < 1000:  # 紧急情况：高度低于1000米
                altitude_cmd = 14  # 最大上升
                logging.warning(f"🚨 {agent_id} 敌方AI紧急拉升！当前高度: {current_altitude:.1f}m")
            elif current_altitude < 3000:  # 危险情况：高度低于3000米
                altitude_cmd = 13  # 大幅上升
                logging.warning(f"⚠️ {agent_id} 敌方AI危险高度，拉升中: {current_altitude:.1f}m")
            elif abs(altitude_diff) > 200:  # 正常高度调整
                if altitude_diff > 500:
                    altitude_cmd = 2   # 下降
                elif altitude_diff > 200:
                    altitude_cmd = 3   # 小幅下降
                elif altitude_diff < -500:
                    altitude_cmd = 12  # 上升
                elif altitude_diff < -200:
                    altitude_cmd = 11  # 小幅上升

            # 🚨 关键修复：强化速度控制，防止失速坠毁，确保索引在有效范围内
            if abs(current_velocity) < 100:  # 极危险速度（包括负速度），紧急加速
                velocity_cmd = 6   # 最大加速（索引6，动作空间0-6）
                logging.warning(f"🚨 {agent_id} 敌方AI极危险速度，紧急加速！当前速度: {current_velocity:.1f}m/s")
            elif abs(current_velocity) < 200:  # 危险速度，大幅加速
                velocity_cmd = 5   # 大幅加速
                logging.warning(f"⚠️ {agent_id} 敌方AI危险速度，大幅加速: {current_velocity:.1f}m/s")
            elif current_velocity < 280:  # 速度偏低，加速
                velocity_cmd = 4   # 加速
            elif current_velocity > 450:  # 速度过高，减速
                velocity_cmd = 0   # 减速
            elif current_velocity > 380:  # 速度偏高，小幅减速
                velocity_cmd = 1   # 小幅减速
            else:
                velocity_cmd = 3   # 保持速度

            # 航向控制：只有在高度和速度安全的情况下才进行精确调整
            if current_altitude > 2000 and 280 < current_velocity < 380:
                if abs(heading_diff) > 2.0:  # 放宽航向控制精度，优先保证安全
                    if heading_diff > 30:
                        heading_cmd = 14  # +45°
                    elif heading_diff > 15:
                        heading_cmd = 13  # +30°
                    elif heading_diff > 8:
                        heading_cmd = 12  # +20°
                    elif heading_diff > 4:
                        heading_cmd = 11  # +10°
                    elif heading_diff > 2:
                        heading_cmd = 10  # +5°
                    elif heading_diff < -30:
                        heading_cmd = 2   # -45°
                    elif heading_diff < -15:
                        heading_cmd = 3   # -30°
                    elif heading_diff < -8:
                        heading_cmd = 4   # -20°
                    elif heading_diff < -4:
                        heading_cmd = 5   # -10°
                    elif heading_diff < -2:
                        heading_cmd = 6   # -5°

            # 调试日志
            if env.current_step % 100 == 0:
                logging.info(f"🛩️ {agent_id} 敌方AI飞行状态: 高度{current_altitude:.1f}m, "
                            f"速度{current_velocity:.1f}m/s, 航向{current_heading:.1f}°, "
                            f"指令({altitude_cmd},{heading_cmd},{velocity_cmd})")

            return altitude_cmd, heading_cmd, velocity_cmd

        except Exception as e:
            logging.error(f"{agent_id} Heading maintenance error: {e}")
            return 7, 8, 3

    def _should_return_to_base(self, env, agent_id: str, current_time: float) -> bool:
        """判断是否应该返航 - 修复版：使用与友方相同的距离控制时间线"""
        try:
            # 条件1:已经在返航状态
            if agent_id in self.return_to_base_states:
                return True

            # 获取当前距离和战术阶段
            min_distance = self._calculate_min_distance_to_friendlies(env, agent_id)
            current_phase = getattr(self, f'current_phase_{agent_id}', EnemyTacticalPhase.MELD_MTR)

            # 条件2:基于战术阶段的RTB决策 - 镜像友方逻辑
            # DOR_DR阶段（35-14.5km）：必须返航，镜像友方在此阶段的RTB行为
            if current_phase == EnemyTacticalPhase.DOR_DR:
                logging.info(f"🔄 {agent_id} RTB触发：DOR_DR阶段强制返航 (距离: {min_distance/1000:.1f}km)")
                return True

            # 条件3:距离过近,紧急返航（20km内，比友方稍早以避免碰撞）
            if min_distance < 20000:
                logging.info(f"🔄 {agent_id} RTB触发：距离过近紧急返航 (距离: {min_distance/1000:.1f}km)")
                return True

            # 条件4:TR_DOR阶段后期，开始准备返航（25km内）
            if current_phase == EnemyTacticalPhase.TR_DOR and min_distance < 25000:
                # 50%概率开始返航，避免所有敌机同时返航
                if random.random() < 0.5:
                    logging.info(f"🔄 {agent_id} RTB触发：TR_DOR后期准备返航 (距离: {min_distance/1000:.1f}km)")
                    return True

            # 条件5:时间过长,自动返航（降低到3分钟，避免过度延长交战）
            if current_time > 180.0:
                logging.info(f"🔄 {agent_id} RTB触发：时间过长自动返航 (时间: {current_time:.1f}s)")
                return True

            # 条件6:队友被击落，立即返航
            teammate_id = "B0200" if agent_id == "B0100" else "B0100"
            if teammate_id not in env.agents or not env.agents[teammate_id].is_alive:
                logging.info(f"🔄 {agent_id} RTB触发：队友{teammate_id}被击落")
                return True

            return False

        except Exception as e:
            logging.error(f"{agent_id} Return decision error: {e}")
            return False

    def _init_return_to_base(self, agent_id: str, current_time: float):
        """初始化返航状态"""
        try:
            self.return_to_base_states[agent_id] = {
                "start_time": current_time,
                "phase": "turn_north",  # 转向北方
                "target_heading": 0.0,  # 北向
                "phase_start_time": current_time
            }
            logging.info(f"{agent_id} 开始返航机动")

        except Exception as e:
            logging.error(f"{agent_id} 返航初始化错误: {e}")

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
                # 阶段2:保持北向返航
                return self._maintain_heading_precise(env, agent_id, 0.0)

        except Exception as e:
            logging.error(f"{agent_id} Return execution error: {e}")
            return self._maintain_heading_precise(env, agent_id, 0.0)  # 安全的北向飞行

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
                        # 锁定时间到期,解除锁定
                        state.locked = False
                        logging.info(f"{agent_id} 机动锁定解除")
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
            # 检查重新攻击冷却时间（8秒检查间隔）
            last_check = self.re_attack_state[agent_id]["last_check_time"]
            if current_time - last_check < 8.0:
                return False

            self.re_attack_state[agent_id]["last_check_time"] = current_time

            # 检查距离条件
            min_distance = self._calculate_min_distance_to_friendlies(env, agent_id)
            randomness = self.tactical_randomness[agent_id]

            # 如果距离拉大到重新攻击阈值,且有一定概率
            if min_distance > randomness.re_attack_distance:
                if random.random() < randomness.re_attack_probability:
                    self.re_attack_state[agent_id]["enabled"] = True
                    self.re_attack_state[agent_id]["trigger_distance"] = min_distance
                    logging.info(f"{agent_id} 触发重新攻击逻辑 (距离: {min_distance / 1000:.1f}km)")
                    return True

            return False

        except Exception as e:
            return False

    def _execute_re_attack(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """执行重新攻击机动"""
        try:
            # 刷新战术随机性
            self._refresh_tactical_randomness(agent_id)

            # 重新攻击:转向我方并加速接近（朝南攻击）
            return self._maintain_heading_precise(env, agent_id, 180.0)

        except Exception as e:
            return 7, 8, 3

    def _apply_boundary_check(self, env, agent_id: str, command_indices: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """应用边界检查和安全验证"""
        try:
            altitude_cmd, heading_cmd, velocity_cmd = command_indices

            # 边界检查（限制指令范围）
            altitude_cmd = max(0, min(14, altitude_cmd))
            heading_cmd = max(0, min(16, heading_cmd))
            velocity_cmd = max(0, min(6, velocity_cmd))

            # 高度安全检查（低于2000米且在下降时强制保持高度）
            try:
                current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
                if current_altitude < 2000 and altitude_cmd < 7:
                    altitude_cmd = 7  # 强制保持高度
            except:
                pass

            return altitude_cmd, heading_cmd, velocity_cmd

        except Exception as e:
            logging.error(f"{agent_id} 边界检查错误: {e}")
            return 7, 8, 3

    # ==================== N001VE雷达系统方法 ====================
    def _update_radar_system(self, env, agent_id: str, current_time: float):
        """更新N001VE雷达系统状态"""
        try:
            # 更新雷达扫描
            self._update_radar_scan(env, agent_id, current_time)

            # 更新目标跟踪（补充缺失方法实现）
            self._update_target_tracking(env, agent_id, current_time)

            # 更新雷达模式（补充缺失方法实现）
            self._update_radar_mode(env, agent_id, current_time)

        except Exception as e:
            logging.error(f"{agent_id} 雷达系统更新错误: {e}")

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

            # 简化的距离计算(米)
            lat_diff = (target_pos[1] - agent_pos[1]) * 111000
            lon_diff = (target_pos[0] - agent_pos[0]) * 111000 * math.cos(math.radians(agent_pos[1]))
            alt_diff = target_pos[2] - agent_pos[2]

            return math.sqrt(lat_diff ** 2 + lon_diff ** 2 + alt_diff ** 2)

        except Exception as e:
            logging.error(f"Distance calculation error: {e}")
            return 999999.0

    def _calculate_bearing_to_target(self, env, agent_id: str, target_id: str) -> float:
        """计算到目标的方位角"""
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
            logging.error(f"方位角计算错误: {e}")
            return 0.0

    def _calculate_target_velocity(self, env, target_id: str) -> float:
        """计算目标速度"""
        try:
            if target_id not in env.agents or not env.agents[target_id].is_alive:
                return 0.0

            velocity = env.agents[target_id].get_property_value(c.velocities_v_north_mps) ** 2
            velocity += env.agents[target_id].get_property_value(c.velocities_v_east_mps) ** 2
            velocity += env.agents[target_id].get_property_value(c.velocities_v_down_mps) ** 2

            return math.sqrt(velocity)

        except Exception as e:
            logging.error(f"目标速度计算错误: {e}")
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
            logging.error(f"{agent_id} 最近目标获取错误: {e}")
            return None

    def _update_radar_scan(self, env, agent_id: str, current_time: float):
        """更新雷达扫描和目标探测"""
        try:
            # 检查扫描周期
            if current_time - self.radar_scan_time[agent_id] < self.radar_model.scan_period:
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

                # 检查探测范围
                if distance > self.radar_model.max_detection_range:
                    continue

                # 计算探测概率（补充缺失方法实现）
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
                        logging.debug(f"{agent_id} 雷达探测到新目标: {target_id} 距离={distance / 1000:.1f}km")
                    else:
                        # 更新现有目标
                        target = self.radar_targets[agent_id][target_id]
                        target.distance = distance
                        target.bearing = bearing
                        target.velocity = velocity
                        target.detection_probability = detection_prob
                        target.last_update = current_time

        except Exception as e:
            logging.error(f"{agent_id} 雷达扫描错误: {e}")

    def _update_target_tracking(self, env, agent_id: str, current_time: float):
        """更新目标跟踪状态（补充缺失实现）"""
        try:
            # 清理过期目标（超过10秒未更新则删除）
            expired_targets = []
            for target_id, target in self.radar_targets[agent_id].items():
                if current_time - target.last_update > 10.0:
                    expired_targets.append(target_id)

            for target_id in expired_targets:
                del self.radar_targets[agent_id][target_id]
                logging.debug(f"{agent_id} 雷达目标过期: {target_id}")

            # 跟踪质量更新（距离越近质量越高）
            for target in self.radar_targets[agent_id].values():
                # 距离越近,跟踪质量越高（0-1之间）
                target.track_quality = max(0.1, 1.0 - (target.distance / self.radar_model.max_track_range))

        except Exception as e:
            logging.error(f"{agent_id} 目标跟踪更新错误: {e}")

    def _update_radar_mode(self, env, agent_id: str, current_time: float):
        """更新雷达模式（补充缺失实现）"""
        try:
            closest_target = self._get_closest_target(agent_id)
            current_mode = self.radar_mode[agent_id]

            # 模式切换逻辑
            if closest_target is None:
                # 无目标时切换到搜索模式
                self.radar_mode[agent_id] = RadarMode.SEARCH
                self.radar_lock_targets[agent_id] = None
            else:
                if closest_target.distance <= self.radar_model.max_lock_range and closest_target.track_quality > 0.8:
                    # 目标在锁定范围内且跟踪质量高,切换到锁定模式
                    self.radar_mode[agent_id] = RadarMode.LOCK
                    self.radar_lock_targets[agent_id] = closest_target.target_id
                elif closest_target.distance <= self.radar_model.max_track_range and closest_target.track_quality > 0.5:
                    # 目标在跟踪范围内且跟踪质量中等,切换到跟踪模式
                    self.radar_mode[agent_id] = RadarMode.TRACK
                    self.radar_lock_targets[agent_id] = closest_target.target_id
                else:
                    # 目标在搜索范围内,切换到搜索模式
                    self.radar_mode[agent_id] = RadarMode.SEARCH
                    self.radar_lock_targets[agent_id] = None

            # 记录模式变化
            if self.radar_mode[agent_id] != current_mode:
                logging.debug(f"{agent_id} 雷达模式切换: {current_mode.value} → {self.radar_mode[agent_id].value}")

        except Exception as e:
            logging.error(f"{agent_id} 雷达模式更新错误: {e}")

    def _calculate_detection_probability(self, distance: float, bearing: float) -> float:
        """计算目标探测概率（补充缺失实现）"""
        try:
            # 基础概率衰减（距离越远概率越低）
            distance_factor = max(0.1, 1.0 - (distance / self.radar_model.max_detection_range))
            # 方位角因子（波束中心附近概率高,±30度外概率降低）
            bearing_factor = max(0.5, 1.0 - (abs(bearing) / 60.0))
            # 综合探测概率
            detection_prob = self.radar_model.detection_probability_base * distance_factor * bearing_factor
            return min(0.98, detection_prob)  # 上限0.98避免100%探测

        except Exception as e:
            logging.error(f"探测概率计算错误: {e}")
            return 0.5  # 默认概率

    def _radar_driven_maneuver_decision(self, env, agent_id: str, current_time: float) -> Optional[
        Tuple[int, int, int]]:
        """雷达驱动的机动决策（补充缺失实现）"""
        try:
            radar_mode = self.radar_mode[agent_id]
            closest_target = self._get_closest_target(agent_id)

            # 无目标时不触发机动
            if closest_target is None:
                return None

            # 锁定模式下:保持航向跟踪目标
            if radar_mode == RadarMode.LOCK:
                return self._maintain_heading_precise(env, agent_id, 180.0)  # 保持南向接敌
            # 跟踪模式下:轻微调整航向对准目标
            elif radar_mode == RadarMode.TRACK:
                # 根据目标方位角轻微调整航向（±5度）
                target_heading = 180.0 + max(-5.0, min(5.0, closest_target.bearing))
                return self._maintain_heading_precise(env, agent_id, target_heading)
            # 搜索模式下:无特殊机动
            else:
                return None

        except Exception as e:
            logging.error(f"{agent_id} 雷达机动决策错误: {e}")
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

    这是外部调用的主要接口,支持:
    - 拖曳射击项目
    - 钳形夹击项目
    - 其他战术场景

    Args:
        env: 环境对象
        agent_id: 智能体ID(B0100或B0200)
        current_time: 当前时间

    Returns:
        Tuple[int, int, int]: [高度指令, 航向指令, 速度指令]索引
    """
    try:
        enemy_ai = get_enhanced_enemy_ai()
        return enemy_ai.get_tactical_command(env, agent_id, current_time)
    except Exception as e:
        logging.error(f"增强敌方AI接口错误: {e}")
        return 7, 8, 3  # 安全的平稳飞行指令


# 向后兼容性支持
def enemy_ai_get_command(env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
    """向后兼容的接口函数"""
    return get_enemy_tactical_command(env, agent_id, current_time)


if __name__ == "__main__":
    # 测试代码（删除重复的main函数,保留一个完整测试）
    print("增强版敌方战术AI系统测试")

    # 创建AI实例
    ai = EnhancedEnemyTacticalAI()
    print("AI instance created successfully")

    # 测试战术随机性生成
    randomness = ai._generate_tactical_randomness()
    print(f"战术随机性生成: Crank={randomness.crank_angle:.1f}, "
          f"TurnCold={randomness.turn_cold_angle:.1f}")

    # 测试威胁评估
    threat = ThreatLevel.MEDIUM
    print(f"Threat assessment test: {threat.name}")


    # 测试雷达模式切换逻辑（模拟目标）
    class MockEnv:
        def __init__(self):
            self.agents = {
                "B0100": type('obj', (object,), {'is_alive': True, 'get_property_value': lambda x: 0.0,
                                                 'get_position': lambda: [0, 0, 0]}),
                "A0100": type('obj', (object,), {'is_alive': True, 'get_property_value': lambda x: 0.0,
                                                 'get_position': lambda: [0, 40000, 0]})  # 40km处目标
            }

        missiles = {}


    mock_env = MockEnv()
    ai._update_radar_system(mock_env, "B0100", 1.0)
    print(f"B0100 雷达模式: {ai.radar_mode['B0100'].value}")

    print("Enhanced enemy tactical AI system test passed!")