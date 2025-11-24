# 导入导弹与雷达检查（优先用旧工程 r27er_missile，如不可用再回退本项目实现）
import os, sys
R27ERMissileSimulator = None
try:
    legacy_path = os.path.join(os.path.dirname(__file__), '..', 'tacticalTemplateProject')
    if legacy_path not in sys.path:
        sys.path.insert(0, legacy_path)
    try:
        from r27er_missile import R27ERMissileSimulator as _LegacyR27ER
        R27ERMissileSimulator = _LegacyR27ER
    except Exception:
        pass
    if R27ERMissileSimulator is None:
        from simulation.r27er_missile import R27ERMissileSimulator as _LocalR27ER
        R27ERMissileSimulator = _LocalR27ER
except Exception:
    pass
try:
    from simulation.radar_manager import check_missile_launch_conditions
except Exception:
    check_missile_launch_conditions = None
#!/usr/bin/env python3
"""
统一敌方战术AI系统 - 适用于所有战术项目的通用敌方AI架构
解决现有系统的逻辑问题，建立基于动作组合的随机化战术选择系统

设计特点：
1. 保持固定的核心框架（三模式、五阶段、雷达系统）
2. 实现动作层面的随机化和参数化
3. 建立合理的返航机制
4. 确保战术多样性和不可预测性

作者：统一战术AI系统
版本：v2.0
"""

import logging
import numpy as np
import random
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# 尝试导入JSBSim目录，如果失败则使用模拟版本
try:
    from envs.JSBSim.core.catalog import Catalog as c
except ImportError:
    # 模拟Catalog类用于测试
    class MockCatalog:
        attitude_psi_rad = "attitude/psi-rad"
        attitude_pitch_rad = "attitude/pitch-rad"
        attitude_phi_rad = "attitude/phi-rad"
        position_h_sl_m = "position/h-sl-m"
        position_lat_geod_deg = "position/lat-geod-deg"
        position_long_gc_deg = "position/long-gc-deg"

    c = MockCatalog()


class TacticalMode(Enum):
    """战术模式枚举 - 核心框架保持不变"""
    AGGRESSIVE = "aggressive"  # 攻击模式：主动接敌、优先发射导弹
    DEFENSIVE = "defensive"   # 防御模式：规避机动、威胁回避
    NEUTRAL = "neutral"       # 中性模式：平衡方法、灵活调整


class EnemyTacticalPhase(Enum):
    """敌方战术阶段 - 五阶段框架保持不变"""
    NLT_MELD = "NLT_MELD"    # 90-81km：远程交战阶段
    MELD_MTR = "MELD_MTR"    # 81-50km：中程交战阶段
    MTR_TR = "MTR_TR"        # 50-40km：导弹目标范围阶段
    TR_DOR = "TR_DOR"        # 40-35km：目标范围到动态攻击范围
    DOR_DR = "DOR_DR"        # 35-14.5km：动态攻击到防御范围


class ThreatLevel(Enum):
    """威胁等级枚举 - 扩展威胁评估"""
    NONE = 0      # 无威胁
    LOW = 1       # 低威胁
    MEDIUM = 2    # 中等威胁
    HIGH = 3      # 高威胁
    CRITICAL = 4  # 紧急威胁
    SEVERE = 5    # 严重威胁（新增）


class RadarMode(Enum):
    """雷达工作模式 - 保持现有系统"""
    SEARCH = "search"    # 搜索模式（宽波束）
    TRACK = "track"      # 跟踪模式（窄波束）
    LOCK = "lock"        # 锁定模式（持续照射）
    STANDBY = "standby"  # 待机模式


class ActionType(Enum):
    """基础动作类型枚举"""
    # 基础机动动作
    MAINTAIN_HEADING = "maintain_heading"      # 保持航向
    TURN_LEFT = "turn_left"                   # 左转
    TURN_RIGHT = "turn_right"                 # 右转
    CLIMB = "climb"                           # 爬升
    DESCEND = "descend"                       # 下降
    ACCELERATE = "accelerate"                 # 加速
    DECELERATE = "decelerate"                 # 减速
    
    # 战术机动动作
    CRANK_LEFT = "crank_left"                 # 左侧Crank
    CRANK_RIGHT = "crank_right"               # 右侧Crank
    NOTCH_MANEUVER = "notch_maneuver"         # Notch机动
    BEAM_MANEUVER = "beam_maneuver"           # Beam机动

    # 导弹规避专用机动
    DIVE_ESCAPE = "dive_escape"               # 俯冲脱离
    CHAFF_FLARE_MANEUVER = "chaff_flare_maneuver"  # 干扰弹配合机动
    SPIRAL_DIVE = "spiral_dive"               # 螺旋俯冲

    # 组合动作
    SHORT_SKATE = "short_skate"               # Short Skate机动
    DEFENSIVE_SPLIT = "defensive_split"        # 防御分离
    AGGRESSIVE_APPROACH = "aggressive_approach" # 攻击接近
    RETURN_TO_BASE = "return_to_base"         # 返航


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


class UnifiedEnemyTacticalAI:
    """统一敌方战术AI系统"""
    
    def __init__(self):
        """初始化统一敌方战术AI系统"""
        # 核心状态跟踪
        self.tactical_mode = {}          # 每个智能体的战术模式
        self.current_phase = {}          # 每个智能体的战术阶段
        self.radar_mode = {}             # 每个智能体的雷达模式
        
        # 动作执行状态
        self.current_action = {}         # 当前执行的动作
        self.action_start_time = {}      # 动作开始时间
        self.action_parameters = {}      # 动作参数
        
        # 态势感知数据
        self.situation_data = {}         # 态势数据缓存
        self.threat_assessment = {}      # 威胁评估缓存

        # 导弹发射管理
        self.last_missile_launch_time = {}  # 上次导弹发射时间
        self.enemy_missile_cooldown = 10.0  # 敌方10秒冷却时间
        self._last_enemy_missile_log = {}
        self._last_enemy_heading_log = {}
        
        # 敌方阶段状态管理
        self._enemy_phases = {}  # 添加缺少的属性

        # 随机化参数
        self.mode_switch_cooldown = {}   # 模式切换冷却时间
        self.last_mode_switch = {}       # 上次模式切换时间
        
        # 🛩️ 飞机型号相关参数
        self.aircraft_parameters = {
            "f16": {
                "min_altitude": 2500,      # F-16最低高度
                "max_altitude": 15000,     # F-16最高高度
                "min_speed": 120,          # F-16最低速度
                "max_speed": 400,          # F-16最高速度
                "turn_rate": 8.0,          # F-16转弯率
                "climb_rate": 50.0         # F-16爬升率
            },
            "su27sk": {
                "min_altitude": 3000,      # SU-27最低高度（更高）
                "max_altitude": 18000,     # SU-27最高高度
                "min_speed": 140,          # SU-27最低速度（更高）
                "max_speed": 500,          # SU-27最高速度
                "turn_rate": 6.5,          # SU-27转弯率（较低）
                "climb_rate": 60.0         # SU-27爬升率
            }
        }
        
        # ✈️ 添加工作的SU-27控制函数索引数组（从pure_maneuver_task移植）
        self.norm_delta_altitude = np.array([-1.5, -1.0, -0.75, -0.5, -0.25, -0.1, 0.0, 0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5])
        self.norm_delta_heading = np.array([-1.0, -0.75, -0.5, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0])
        self.norm_delta_velocity = np.array([-1.0, -0.75, -0.5, -0.25, -0.125, 0.0, 0.125, 0.25, 0.5, 0.75, 1.0])
        
        # 为SU-27优化的控制数值（和pure_maneuver_task保持一致）
        self._inner_rnn_states = {}  # RNN状态缓存
        
        # 动作权重配置
        self._init_action_weights()
        
        logging.info("🎯 统一敌方战术AI系统初始化完成")
    
    def _detect_aircraft_model(self, env, agent_id: str) -> str:
        """检测飞机型号"""
        try:
            if hasattr(env, 'agents') and agent_id in env.agents:
                agent = env.agents[agent_id]
                # 尝试从agent获取模型信息
                if hasattr(agent, 'model_name'):
                    return agent.model_name
                elif hasattr(agent, 'config') and hasattr(agent.config, 'model'):
                    return agent.config.model
            
            # 从环境配置中获取
            if hasattr(env, 'config') and hasattr(env.config, 'aircraft_configs'):
                if agent_id in env.config.aircraft_configs:
                    model_type = env.config.aircraft_configs[agent_id].get('model', 'f16')
                    return model_type
            
            return "f16"  # 默认F-16
        except Exception as e:
            logging.warning(f"检测飞机型号失败: {e}，使用默认F-16")
            return "f16"
    
    def _get_aircraft_parameters(self, env, agent_id: str) -> Dict:
        """获取飞机特定参数"""
        model_type = self._detect_aircraft_model(env, agent_id)
        return self.aircraft_parameters.get(model_type, self.aircraft_parameters["f16"])

    def handle_missile_launch(self, env, agent_id: str, current_time: float):
        """处理敌方导弹发射逻辑"""
        if not agent_id.startswith('B'):  # 只处理敌方
            return

        if not env.agents[agent_id].is_alive:
            return

        # 检查导弹数量
        if env.agents[agent_id].num_missiles <= 0:
            return

        # 检查冷却时间
        last_launch = self.last_missile_launch_time.get(agent_id, -999)
        if current_time - last_launch < self.enemy_missile_cooldown:
            return

        # 寻找目标
        target = self._find_best_target(env, agent_id)
        if target is None:
            return

        # 计算距离
        current_pos = env.agents[agent_id].get_position()
        target_pos = target.get_position()
        distance = np.linalg.norm(np.array(current_pos) - np.array(target_pos))

        # 敌方导弹发射判断
        should_launch = self._enemy_should_launch_missile(env, agent_id, target, distance, current_time)

        if should_launch:
            self._launch_missile(env, agent_id, target, current_time)

    def _find_best_target(self, env, agent_id: str):
        """寻找最佳攻击目标"""
        best_target = None
        min_distance = float('inf')

        for target_id, target_agent in env.agents.items():
            if target_id.startswith('A') and target_agent.is_alive:  # 友方目标
                current_pos = env.agents[agent_id].get_position()
                target_pos = target_agent.get_position()
                distance = np.linalg.norm(np.array(current_pos) - np.array(target_pos))

                if distance < min_distance:
                    min_distance = distance
                    best_target = target_agent

        return best_target

    def _enemy_should_launch_missile(self, env, agent_id: str, target, distance: float, current_time: float) -> bool:
        """敌方智能导弹发射判断 - 修复问题7：添加朝向检查"""
        last = self._last_enemy_missile_log.get(agent_id, -999)
        if current_time - last >= 5.0:
            logging.debug(f"[T={current_time:.1f}s][敌方导弹] {agent_id} 检查发射条件: 距离={distance/1000:.1f}km")
            self._last_enemy_missile_log[agent_id] = current_time
        
        # 基本条件检查
        if env.agents[agent_id].num_missiles <= 0:
            logging.info(f"[T={current_time:.1f}s][敌方导弹] {agent_id} 导弹已用尽")
            return False

        # 冷却时间检查
        last_launch = self.last_missile_launch_time.get(agent_id, -999)
        if current_time - last_launch < self.enemy_missile_cooldown:
            return False  # 冷却中不打印日志

        # 距离条件：20-80km范围内发射
        if distance < 20000 or distance > 80000:
            return False  # 距离不满足不打印日志

        # 🔧 修复问题7：添加朝向检查，确保朝向目标才发射
        current_pos = env.agents[agent_id].get_position()
        target_pos = target.get_position()
        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_rad)
        
        # 计算目标方位角
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        target_bearing = np.arctan2(dy, dx)
        
        # 计算朝向偏差（航向与目标方位的夹角）
        heading_error = abs(((target_bearing - current_heading + np.pi) % (2*np.pi)) - np.pi)
        heading_error_deg = np.rad2deg(heading_error)
        
        # 🔧 修复：只有朝向目标±45度范围内才允许发射
        if heading_error_deg > 45.0:
            last = self._last_enemy_heading_log.get(agent_id, -999)
            if current_time - last >= 5.0:
                logging.debug(f"[T={current_time:.1f}s][敌方导弹] {agent_id} ❌ 朝向偏离{heading_error_deg:.1f}° > 45°，不发射")
                self._last_enemy_heading_log[agent_id] = current_time
            return False

        # 威胁评估：在高威胁情况下更积极发射
        threat = self.threat_assessment.get(agent_id)
        if threat and threat.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            logging.info(f"[T={current_time:.1f}s][敌方导弹] {agent_id} ✅ 高威胁+朝向正确({heading_error_deg:.1f}°)，满足发射条件！")
            return True

        # 正常发射条件：30-70km最佳发射窗口
        if 30000 <= distance <= 70000:
            last = self._last_enemy_heading_log.get(agent_id, -999)
            if current_time - last >= 5.0:
                logging.info(f"[T={current_time:.1f}s][敌方导弹] {agent_id} ✅ 距离+朝向正确({heading_error_deg:.1f}°)，满足发射条件！")
                self._last_enemy_heading_log[agent_id] = current_time
            # 追加 3.7.2 雷达/锁定/Notch/探测概率检查
            if check_missile_launch_conditions is not None:
                try:
                    chk = check_missile_launch_conditions(env, agent_id, target.uid)
                    if not chk.get('can_launch', False):
                        return False
                except Exception:
                    pass
            return True

        return False

    def _launch_missile(self, env, agent_id: str, target, current_time: float):
        """发射导弹"""
        try:
            from envs.JSBSim.core.simulatior import MissileSimulator

            aircraft = env.agents[agent_id]

            # 创建导弹ID - 使用递增计数器避免ID复用，格式与我方一致
            if not hasattr(self, '_enemy_missile_counter'):
                self._enemy_missile_counter = {}
            if agent_id not in self._enemy_missile_counter:
                self._enemy_missile_counter[agent_id] = 0
            self._enemy_missile_counter[agent_id] += 1
            missile_uid = f"{agent_id}{self._enemy_missile_counter[agent_id]:02d}"

            # 创建导弹模拟器（优先使用R-27ER仿真器）
            if R27ERMissileSimulator is not None:
                missile = R27ERMissileSimulator.create(
                    parent=aircraft,
                    target=target,
                    uid=missile_uid
                )
            else:
                missile = MissileSimulator.create(
                    parent=aircraft,
                    target=target,
                    uid=missile_uid
                )

            # 添加到环境
            env.add_temp_simulator(missile)
            # 记录到环境导弹表，供威胁评估/RWR使用
            if not hasattr(env, 'missiles') or env.missiles is None:
                env.missiles = {}
            env.missiles[missile_uid] = missile

            # 更新发射时间
            self.last_missile_launch_time[agent_id] = current_time

            # 减少导弹数量
            aircraft.num_missiles -= 1

            logging.info(f"🚀 {agent_id} 发射导弹 {missile_uid} 攻击目标")

        except Exception as e:
            logging.error(f"导弹发射失败 {agent_id}: {e}")

    def _init_action_weights(self):
        """初始化动作权重配置"""
        # 基于战术模式和阶段的动作权重矩阵
        # 🔧 已优化：大幅降低激进机动频率，增加温和机动和返航频率
        self.action_weights = {
            TacticalMode.AGGRESSIVE: {
                EnemyTacticalPhase.NLT_MELD: {
                    ActionType.MAINTAIN_HEADING: 0.4,
                    ActionType.AGGRESSIVE_APPROACH: 0.3,
                    ActionType.CRANK_LEFT: 0.15,
                    ActionType.CRANK_RIGHT: 0.15
                },
                EnemyTacticalPhase.MELD_MTR: {
                    ActionType.AGGRESSIVE_APPROACH: 0.4,
                    ActionType.CRANK_LEFT: 0.2,
                    ActionType.CRANK_RIGHT: 0.2,
                    ActionType.MAINTAIN_HEADING: 0.2
                },
                EnemyTacticalPhase.MTR_TR: {
                    ActionType.SHORT_SKATE: 0.25,
                    ActionType.CRANK_LEFT: 0.25,
                    ActionType.CRANK_RIGHT: 0.25,
                    ActionType.AGGRESSIVE_APPROACH: 0.25
                },
                EnemyTacticalPhase.TR_DOR: {
                    ActionType.SHORT_SKATE: 0.5,
                    ActionType.CRANK_LEFT: 0.2,
                    ActionType.CRANK_RIGHT: 0.2,
                    ActionType.AGGRESSIVE_APPROACH: 0.1
                },
                EnemyTacticalPhase.DOR_DR: {
                    ActionType.SHORT_SKATE: 0.4,
                    ActionType.RETURN_TO_BASE: 0.6
                }
            },
            TacticalMode.DEFENSIVE: {
                EnemyTacticalPhase.NLT_MELD: {
                    ActionType.MAINTAIN_HEADING: 0.5,
                    ActionType.BEAM_MANEUVER: 0.3,
                    ActionType.DEFENSIVE_SPLIT: 0.2
                },
                EnemyTacticalPhase.MELD_MTR: {
                    ActionType.BEAM_MANEUVER: 0.4,
                    ActionType.NOTCH_MANEUVER: 0.3,
                    ActionType.DEFENSIVE_SPLIT: 0.3
                },
                EnemyTacticalPhase.MTR_TR: {
                    ActionType.NOTCH_MANEUVER: 0.08,  # 进一步降低频率：0.15 → 0.08
                    ActionType.BEAM_MANEUVER: 0.08,   # 进一步降低频率：0.15 → 0.08
                    ActionType.DIVE_ESCAPE: 0.05,     # 极大降低频率：0.08 → 0.05
                    ActionType.CHAFF_FLARE_MANEUVER: 0.04,  # 极大降低频率：0.07 → 0.04
                    ActionType.TURN_LEFT: 0.375,      # 大幅增加温和机动：0.275 → 0.375
                    ActionType.TURN_RIGHT: 0.375      # 大幅增加温和机动：0.275 → 0.375
                },
                EnemyTacticalPhase.TR_DOR: {
                    ActionType.NOTCH_MANEUVER: 0.1,   # 进一步降低频率：0.2 → 0.1
                    ActionType.DIVE_ESCAPE: 0.05,     # 极大降低频率：0.1 → 0.05
                    ActionType.SPIRAL_DIVE: 0.03,     # 极大降低频率：0.08 → 0.03
                    ActionType.DEFENSIVE_SPLIT: 0.32, # 增加温和机动：0.27 → 0.32
                    ActionType.RETURN_TO_BASE: 0.5    # 大幅增加返航频率：0.35 → 0.5
                },
                EnemyTacticalPhase.DOR_DR: {
                    ActionType.RETURN_TO_BASE: 0.8,   # 极大增加返航频率：0.65 → 0.8
                    ActionType.SPIRAL_DIVE: 0.05,     # 极大降低频率：0.1 → 0.05
                    ActionType.DIVE_ESCAPE: 0.1,      # 进一步降低频率：0.15 → 0.1
                    ActionType.NOTCH_MANEUVER: 0.05   # 极大降低频率：0.1 → 0.05
                }
            },
            TacticalMode.NEUTRAL: {
                EnemyTacticalPhase.NLT_MELD: {
                    ActionType.MAINTAIN_HEADING: 0.6,
                    ActionType.TURN_LEFT: 0.2,
                    ActionType.TURN_RIGHT: 0.2
                },
                EnemyTacticalPhase.MELD_MTR: {
                    ActionType.MAINTAIN_HEADING: 0.4,
                    ActionType.BEAM_MANEUVER: 0.3,
                    ActionType.CRANK_LEFT: 0.15,
                    ActionType.CRANK_RIGHT: 0.15
                },
                EnemyTacticalPhase.MTR_TR: {
                    ActionType.MAINTAIN_HEADING: 0.3,
                    ActionType.CRANK_LEFT: 0.2,
                    ActionType.CRANK_RIGHT: 0.2,
                    ActionType.SHORT_SKATE: 0.15,
                    ActionType.BEAM_MANEUVER: 0.15
                },
                EnemyTacticalPhase.TR_DOR: {
                    ActionType.SHORT_SKATE: 0.3,
                    ActionType.BEAM_MANEUVER: 0.3,
                    ActionType.RETURN_TO_BASE: 0.4
                },
                EnemyTacticalPhase.DOR_DR: {
                    ActionType.RETURN_TO_BASE: 0.8,
                    ActionType.SHORT_SKATE: 0.2
                }
            }
        }

    def get_enemy_command(self, env, agent_id: str, current_time: float, task=None) -> Tuple[int, int, int]:
        """获取敌方战术指令 - 统一入口点"""
        try:
            # 1. 态势感知
            situation = self._analyze_situation(env, agent_id, current_time)

            # 2. 检查是否处于返航状态
            if agent_id in self._enemy_phases and self._enemy_phases[agent_id] == "RETURNING":
                return self._execute_return_to_base_unified(env, agent_id, current_time, task)

            # 3. 威胁评估
            threat = self._assess_threat(situation, agent_id, current_time)

            # 4. 更新战术阶段
            self._update_tactical_phase(env, agent_id, situation)

            # 5. 战术模式选择
            tactical_mode = self._select_tactical_mode(agent_id, threat, current_time)

            # 6. 机动决策
            action_type = self._select_action(agent_id, tactical_mode, current_time)

            # 7. 动作执行
            alt_cmd, hdg_cmd, vel_cmd = self._execute_action(env, agent_id, action_type, current_time, task)
            
            # ✅ 全局高度安全检查（最终防线）- 传递task参数
            alt_cmd, hdg_cmd, vel_cmd = self._apply_global_safety_check(
                env, agent_id, alt_cmd, hdg_cmd, vel_cmd, task
            )
            
            return alt_cmd, hdg_cmd, vel_cmd

        except Exception as e:
            logging.error(f"敌方{agent_id}战术指令生成失败: {e}")
            return 7, 8, 3  # 默认平稳飞行

    def get_enemy_command_indices(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """获取敌方战术指令索引 - 兼容接口"""
        return self.get_enemy_command(env, agent_id, current_time)

    def get_enemy_action(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """获取敌方行动指令 - 钳形夹击项目兼容接口"""
        return self.get_enemy_command(env, agent_id, current_time)

    def get_detailed_action_info(self, agent_id: str) -> Dict[str, Any]:
        """获取详细的行动信息用于注释"""
        try:
            # 获取当前状态信息
            current_phase = self.current_phase.get(agent_id, EnemyTacticalPhase.NLT_MELD)
            tactical_mode = self.tactical_mode.get(agent_id, TacticalMode.NEUTRAL)
            last_action = self.current_action.get(agent_id, ActionType.MAINTAIN_HEADING)

            # 获取威胁等级
            threat_level = self._get_current_threat_level(agent_id)

            # 根据当前阶段和行动类型生成详细描述
            phase_name = current_phase.name.lower()
            mode_name = tactical_mode.name.lower()
            action_name = last_action.name.lower().replace('_', ' ')

            # 返回字典格式，符合enhanced_action_annotator的期望
            return {
                'action_annotation': f"{phase_name}_{mode_name}_{action_name}",
                'tactical_mode': tactical_mode,
                'current_phase': current_phase,
                'current_action': last_action,
                'threat_level': threat_level,
                'phase_name': phase_name,
                'mode_name': mode_name,
                'action_name': action_name
            }

        except Exception as e:
            logging.error(f"获取{agent_id}详细行动信息失败: {e}")
            return {
                'action_annotation': 'error',
                'tactical_mode': TacticalMode.NEUTRAL,
                'current_phase': EnemyTacticalPhase.NLT_MELD,
                'current_action': ActionType.MAINTAIN_HEADING,
                'threat_level': 'NONE',
                'phase_name': 'error',
                'mode_name': 'neutral',
                'action_name': 'maintain heading'
            }

    def _get_current_threat_level(self, agent_id: str) -> str:
        """获取当前威胁等级"""
        try:
            # 从威胁评估数据中获取威胁等级
            if hasattr(self, 'threat_assessment') and agent_id in self.threat_assessment:
                threat_data = self.threat_assessment[agent_id]
                if isinstance(threat_data, dict) and 'level' in threat_data:
                    return threat_data['level']

            # 从态势数据中推断威胁等级
            situation = self.situation_data.get(agent_id)
            if situation and hasattr(situation, 'threat_level'):
                return situation.threat_level

            # 根据战术模式推断威胁等级
            tactical_mode = self.tactical_mode.get(agent_id, TacticalMode.NEUTRAL)
            if tactical_mode == TacticalMode.AGGRESSIVE:
                return 'MEDIUM'
            elif tactical_mode == TacticalMode.DEFENSIVE:
                return 'HIGH'
            else:
                return 'LOW'

        except Exception as e:
            logging.debug(f"威胁等级获取失败 {agent_id}: {e}")
            return 'NONE'

    def _analyze_situation(self, env, agent_id: str, current_time: float) -> SituationData:
        """态势感知模块 - 收集敌我双方动态信息"""
        try:
            aircraft = env.agents[agent_id]

            # 获取当前状态
            current_pos = aircraft.get_position()
            current_heading = np.rad2deg(aircraft.get_property_value(c.attitude_psi_rad))
            current_altitude = aircraft.get_property_value(c.position_h_sl_m)
            current_velocity = np.linalg.norm(aircraft.get_velocity())

            # 计算与友方的最近距离和方位
            min_distance = float('inf')
            closest_bearing = None  # 初始化为None，表示没有找到敌机

            for friendly_id, friendly_aircraft in env.agents.items():
                if friendly_id.startswith('A') and friendly_aircraft.is_alive:
                    friendly_pos = friendly_aircraft.get_position()
                    distance = np.linalg.norm(np.array(current_pos) - np.array(friendly_pos))

                    if distance < min_distance:
                        min_distance = distance
                        # 计算方位角
                        dx = friendly_pos[0] - current_pos[0]
                        dy = friendly_pos[1] - current_pos[1]
                        closest_bearing = np.rad2deg(np.arctan2(dy, dx))

            # 返航条件检查
            should_return_home = self._should_return_to_base(env, agent_id, current_time, min_distance)
            if should_return_home:
                # 强制切换到返航阶段
                if agent_id not in self._enemy_phases:
                    self._enemy_phases[agent_id] = "RETURNING"
                elif self._enemy_phases[agent_id] != "RETURNING":
                    self._enemy_phases[agent_id] = "RETURNING"
                    logging.debug(f"{agent_id} 切换到返航阶段")
                # 不直接返回，继续创建SituationData但标记为返航状态
                returning_to_base = True
            else:
                returning_to_base = False

            # 如果没有找到敌机，使用默认方位
            if closest_bearing is None:
                closest_bearing = 180.0  # 默认南向

            # 检查导弹威胁
            missile_threats = []
            if hasattr(env, '_tempsims'):
                for missile_id, missile_sim in env._tempsims.items():
                    if missile_id.startswith('A'):  # 友方导弹
                        missile_pos = missile_sim.get_position()
                        missile_distance = np.linalg.norm(np.array(current_pos) - np.array(missile_pos))
                        if missile_distance < 50000:  # 50km内的导弹威胁
                            missile_threats.append({
                                'id': missile_id,
                                'distance': missile_distance,
                                'position': missile_pos
                            })

            # 检查雷达锁定状态（简化实现）
            radar_locked = False
            lock_duration = 0.0
            if hasattr(self, 'radar_lock_time') and agent_id in self.radar_lock_time:
                lock_duration = current_time - self.radar_lock_time[agent_id]
                radar_locked = lock_duration > 2.0  # 锁定超过2秒

            # 检查队友状态
            teammate_alive = False
            teammate_distance = float('inf')
            teammate_id = "B0200" if agent_id == "B0100" else "B0100"

            if teammate_id in env.agents and env.agents[teammate_id].is_alive:
                teammate_alive = True
                teammate_pos = env.agents[teammate_id].get_position()
                teammate_distance = np.linalg.norm(np.array(current_pos) - np.array(teammate_pos))

            situation = SituationData(
                min_enemy_distance=min_distance,
                closest_enemy_bearing=closest_bearing,
                missile_threats=missile_threats,
                radar_locked=radar_locked,
                lock_duration=lock_duration,
                teammate_alive=teammate_alive,
                teammate_distance=teammate_distance,
                current_altitude=current_altitude,
                current_velocity=current_velocity,
                current_heading=current_heading
            )

            self.situation_data[agent_id] = situation
            return situation

        except Exception as e:
            logging.error(f"态势感知失败 {agent_id}: {e}")
            # 返回默认态势数据
            return SituationData(
                min_enemy_distance=100000.0,
                closest_enemy_bearing=0.0,
                missile_threats=[],
                radar_locked=False,
                lock_duration=0.0,
                teammate_alive=True,
                teammate_distance=2000.0,
                current_altitude=10000.0,
                current_velocity=250.0,
                current_heading=180.0
            )

    def _assess_threat(self, situation: SituationData, agent_id: str, current_time: float) -> ThreatAssessment:
        """威胁评估模块 - 基于态势分析结果评估威胁等级"""
        try:
            threat_score = 0.0

            # 距离威胁评估 (权重: 0.4)
            distance_threat = False
            if situation.min_enemy_distance < 20000:  # 20km
                threat_score += 40.0
                distance_threat = True
            elif situation.min_enemy_distance < 35000:  # 35km
                threat_score += 25.0
                distance_threat = True
            elif situation.min_enemy_distance < 50000:  # 50km
                threat_score += 15.0
            elif situation.min_enemy_distance < 80000:  # 80km
                threat_score += 5.0

            # 雷达锁定威胁评估 (权重: 0.3)
            lock_threat = False
            if situation.radar_locked:
                lock_threat = True
                if situation.lock_duration > 10.0:
                    threat_score += 30.0
                elif situation.lock_duration > 5.0:
                    threat_score += 20.0
                else:
                    threat_score += 10.0

            # 导弹威胁评估 (权重: 0.4)
            missile_threat_count = len(situation.missile_threats)
            if missile_threat_count > 0:
                threat_score += missile_threat_count * 15.0
                # 检查最近导弹距离
                closest_missile_distance = min([m['distance'] for m in situation.missile_threats])
                if closest_missile_distance < 10000:  # 10km
                    threat_score += 25.0
                elif closest_missile_distance < 20000:  # 20km
                    threat_score += 15.0

            # 队友损失威胁修正 (权重: 0.1)
            if not situation.teammate_alive:
                threat_score += 10.0

            # 威胁等级映射
            if threat_score >= 80:
                threat_level = ThreatLevel.SEVERE
            elif threat_score >= 60:
                threat_level = ThreatLevel.CRITICAL
            elif threat_score >= 40:
                threat_level = ThreatLevel.HIGH
            elif threat_score >= 20:
                threat_level = ThreatLevel.MEDIUM
            elif threat_score >= 5:
                threat_level = ThreatLevel.LOW
            else:
                threat_level = ThreatLevel.NONE

            # 确定主要威胁目标
            primary_threat_id = None
            if situation.missile_threats:
                # 优先考虑最近的导弹威胁
                closest_missile = min(situation.missile_threats, key=lambda x: x['distance'])
                primary_threat_id = closest_missile['id']

            assessment = ThreatAssessment(
                threat_level=threat_level,
                threat_score=threat_score,
                primary_threat_id=primary_threat_id,
                missile_threat_count=missile_threat_count,
                lock_threat=lock_threat,
                distance_threat=distance_threat
            )

            self.threat_assessment[agent_id] = assessment
            return assessment

        except Exception as e:
            logging.error(f"威胁评估失败 {agent_id}: {e}")
            return ThreatAssessment(
                threat_level=ThreatLevel.LOW,
                threat_score=10.0,
                primary_threat_id=None,
                missile_threat_count=0,
                lock_threat=False,
                distance_threat=False
            )

    def _update_tactical_phase(self, env, agent_id: str, situation: SituationData):
        """更新战术阶段 - 基于距离的阶段转换 + 修复问题6：改进返航逻辑"""
        try:
            distance = situation.min_enemy_distance

            # 🔧 修复问题6：记录最小距离，判断是否应该返航
            if not hasattr(self, '_min_distance_reached'):
                self._min_distance_reached = {}
            
            if agent_id not in self._min_distance_reached:
                self._min_distance_reached[agent_id] = float('inf')
            
            # 更新最小距离
            if distance < self._min_distance_reached[agent_id]:
                self._min_distance_reached[agent_id] = distance

            # 🔧 修复问题6：如果已经接近过(<40km)，现在距离又拉大(>65km)，说明应该返航了
            should_return = (self._min_distance_reached[agent_id] < 40000 and distance > 65000)

            # 基于距离的阶段判断
            if should_return:
                # 强制进入返航阶段
                new_phase = EnemyTacticalPhase.DOR_DR
            elif distance > 81000:
                new_phase = EnemyTacticalPhase.NLT_MELD
            elif distance > 50000:
                new_phase = EnemyTacticalPhase.MELD_MTR
            elif distance > 40000:
                new_phase = EnemyTacticalPhase.MTR_TR
            elif distance > 35000:
                new_phase = EnemyTacticalPhase.TR_DOR
            else:
                new_phase = EnemyTacticalPhase.DOR_DR

            # 更新阶段
            old_phase = self.current_phase.get(agent_id, EnemyTacticalPhase.MELD_MTR)
            if old_phase != new_phase:
                self.current_phase[agent_id] = new_phase
                # 只在关键阶段转换时打印日志
                if new_phase == EnemyTacticalPhase.DOR_DR and should_return:
                    logging.info(f"[敌方{agent_id}] 接敌后拉开距离({distance/1000:.1f}km)，进入返航阶段")

        except Exception as e:
            logging.error(f"战术阶段更新失败 {agent_id}: {e}")
            self.current_phase[agent_id] = EnemyTacticalPhase.MELD_MTR

    def _select_tactical_mode(self, agent_id: str, threat: ThreatAssessment, current_time: float) -> TacticalMode:
        """战术模式选择模块 - 基于威胁等级和随机化因子"""
        try:
            # 检查模式切换冷却时间
            if agent_id in self.last_mode_switch:
                time_since_switch = current_time - self.last_mode_switch[agent_id]
                if time_since_switch < 5.0:  # 5秒冷却时间
                    return self.tactical_mode.get(agent_id, TacticalMode.NEUTRAL)

            # 基于威胁等级的模式选择权重
            mode_weights = {}

            if threat.threat_level == ThreatLevel.SEVERE:
                mode_weights = {TacticalMode.DEFENSIVE: 0.9, TacticalMode.NEUTRAL: 0.1}
            elif threat.threat_level == ThreatLevel.CRITICAL:
                mode_weights = {TacticalMode.DEFENSIVE: 0.8, TacticalMode.NEUTRAL: 0.15, TacticalMode.AGGRESSIVE: 0.05}
            elif threat.threat_level == ThreatLevel.HIGH:
                mode_weights = {TacticalMode.DEFENSIVE: 0.6, TacticalMode.NEUTRAL: 0.25, TacticalMode.AGGRESSIVE: 0.15}
            elif threat.threat_level == ThreatLevel.MEDIUM:
                # 基于战术阶段调整权重
                current_phase = self.current_phase.get(agent_id, EnemyTacticalPhase.MELD_MTR)
                if current_phase in [EnemyTacticalPhase.MTR_TR, EnemyTacticalPhase.TR_DOR]:
                    mode_weights = {TacticalMode.AGGRESSIVE: 0.5, TacticalMode.NEUTRAL: 0.3, TacticalMode.DEFENSIVE: 0.2}
                else:
                    mode_weights = {TacticalMode.NEUTRAL: 0.5, TacticalMode.AGGRESSIVE: 0.3, TacticalMode.DEFENSIVE: 0.2}
            elif threat.threat_level == ThreatLevel.LOW:
                mode_weights = {TacticalMode.AGGRESSIVE: 0.5, TacticalMode.NEUTRAL: 0.4, TacticalMode.DEFENSIVE: 0.1}
            else:  # NONE
                mode_weights = {TacticalMode.NEUTRAL: 0.6, TacticalMode.AGGRESSIVE: 0.4}

            # 长机-僚机角色调整
            if agent_id == "B0100":  # 长机更激进
                if TacticalMode.AGGRESSIVE in mode_weights:
                    mode_weights[TacticalMode.AGGRESSIVE] *= 1.2
                if TacticalMode.DEFENSIVE in mode_weights:
                    mode_weights[TacticalMode.DEFENSIVE] *= 0.8
            elif agent_id == "B0200":  # 僚机更保守
                if TacticalMode.DEFENSIVE in mode_weights:
                    mode_weights[TacticalMode.DEFENSIVE] *= 1.2
                if TacticalMode.AGGRESSIVE in mode_weights:
                    mode_weights[TacticalMode.AGGRESSIVE] *= 0.8

            # 归一化权重
            total_weight = sum(mode_weights.values())
            mode_weights = {mode: weight/total_weight for mode, weight in mode_weights.items()}

            # 随机选择模式
            rand_val = random.random()
            cumulative_weight = 0.0
            selected_mode = TacticalMode.NEUTRAL

            for mode, weight in mode_weights.items():
                cumulative_weight += weight
                if rand_val <= cumulative_weight:
                    selected_mode = mode
                    break

            # 更新模式
            old_mode = self.tactical_mode.get(agent_id, TacticalMode.NEUTRAL)
            if old_mode != selected_mode:
                self.tactical_mode[agent_id] = selected_mode
                self.last_mode_switch[agent_id] = current_time
                # logging.info(f"敌方{agent_id}战术模式切换: {old_mode.value} → {selected_mode.value} (威胁: {threat.threat_level.name})")

            return selected_mode

        except Exception as e:
            logging.error(f"战术模式选择失败 {agent_id}: {e}")
            return TacticalMode.NEUTRAL

    def _select_action(self, agent_id: str, tactical_mode: TacticalMode, current_time: float) -> ActionType:
        """机动决策模块 - 基于权重矩阵的随机化动作选择"""
        try:
            current_phase = self.current_phase.get(agent_id, EnemyTacticalPhase.MELD_MTR)

            # 检查当前动作是否需要继续执行
            if agent_id in self.current_action and agent_id in self.action_start_time:
                action_duration = current_time - self.action_start_time[agent_id]
                current_action_type = self.current_action[agent_id]

                # 获取动作参数中的持续时间
                if agent_id in self.action_parameters:
                    required_duration = self.action_parameters[agent_id].duration
                    if action_duration < required_duration:
                        # 继续执行当前动作
                        return current_action_type

            # 获取当前模式和阶段的动作权重
            if tactical_mode not in self.action_weights:
                tactical_mode = TacticalMode.NEUTRAL

            if current_phase not in self.action_weights[tactical_mode]:
                current_phase = EnemyTacticalPhase.MELD_MTR

            action_weights = self.action_weights[tactical_mode][current_phase].copy()

            # 长机-僚机差异化调整
            if agent_id == "B0100":  # 长机
                # 长机更倾向于主动动作
                if ActionType.SHORT_SKATE in action_weights:
                    action_weights[ActionType.SHORT_SKATE] *= 1.3
                if ActionType.AGGRESSIVE_APPROACH in action_weights:
                    action_weights[ActionType.AGGRESSIVE_APPROACH] *= 1.2
            elif agent_id == "B0200":  # 僚机
                # 僚机更倾向于支援和防御动作
                if ActionType.DEFENSIVE_SPLIT in action_weights:
                    action_weights[ActionType.DEFENSIVE_SPLIT] *= 1.3
                if ActionType.RETURN_TO_BASE in action_weights:
                    action_weights[ActionType.RETURN_TO_BASE] *= 1.1

            # 归一化权重
            total_weight = sum(action_weights.values())
            if total_weight == 0:
                return ActionType.MAINTAIN_HEADING

            action_weights = {action: weight/total_weight for action, weight in action_weights.items()}

            # 随机选择动作
            rand_val = random.random()
            cumulative_weight = 0.0
            selected_action = ActionType.MAINTAIN_HEADING

            for action, weight in action_weights.items():
                cumulative_weight += weight
                if rand_val <= cumulative_weight:
                    selected_action = action
                    break

            # 记录新动作
            self.current_action[agent_id] = selected_action
            self.action_start_time[agent_id] = current_time

            logging.debug(f"敌方{agent_id}选择动作: {selected_action.value} (模式: {tactical_mode.value}, 阶段: {current_phase.value})")

            return selected_action

        except Exception as e:
            logging.error(f"机动决策失败 {agent_id}: {e}")
            return ActionType.MAINTAIN_HEADING

    def _execute_action(self, env, agent_id: str, action_type: ActionType, current_time: float, task=None) -> Tuple[int, int, int]:
        """动作执行模块 - 将动作类型转换为具体的飞行指令 - 🛡️ 多层安全保护机制"""
        try:
            # 🛡️ 设置环境引用供参数生成使用
            self._current_env = env

            # 🛡️ 高度安全检查 - 防止飞机坠毁
            current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)

            # 🛡️ 低高度禁用俯冲动作，高空允许俯冲攻击
            dangerous_actions = [ActionType.DIVE_ESCAPE, ActionType.SPIRAL_DIVE, ActionType.CHAFF_FLARE_MANEUVER, ActionType.DESCEND]
            if action_type in dangerous_actions:
                # 只在低高度(<4000m)时禁用俯冲，高空允许俯冲攻击
                if current_altitude < 4000.0:
                    logging.warning(f"🛡️ {agent_id} 低高度({current_altitude:.0f}m)禁用俯冲，改为水平飞行")
                    return self._execute_maintain_heading(env, agent_id)
                # 高空允许俯冲，但限制俯冲角度
                else:
                    logging.debug(f"✅ {agent_id} 高空({current_altitude:.0f}m)允许俯冲攻击")

            # 🛡️ 低高度强制爬升 - 降低阈值到1200米，避免过于频繁触发
            if current_altitude < 1200.0:
                # logging.error(f"🚨 {agent_id} 高度{current_altitude:.0f}m过低，强制爬升！")
                return self._execute_altitude_change(env, agent_id, 500.0)  # 爬升500米

            # 生成动作参数（如果还没有）- 🛡️ 现在包含智能高度感知
            if agent_id not in self.action_parameters or self.current_action.get(agent_id) != action_type:
                self.action_parameters[agent_id] = self._generate_action_parameters(action_type, agent_id)

            params = self.action_parameters[agent_id]

            # 根据动作类型执行相应的机动
            if action_type == ActionType.MAINTAIN_HEADING:
                return self._execute_maintain_heading(env, agent_id)

            elif action_type == ActionType.TURN_LEFT:
                turn_angle = params.turn_angle if params.turn_angle is not None else 30.0
                turn_rate = params.turn_rate if params.turn_rate is not None else 5.0
                return self._execute_turn(env, agent_id, -turn_angle, turn_rate)

            elif action_type == ActionType.TURN_RIGHT:
                turn_angle = params.turn_angle if params.turn_angle is not None else 30.0
                turn_rate = params.turn_rate if params.turn_rate is not None else 5.0
                return self._execute_turn(env, agent_id, turn_angle, turn_rate)

            elif action_type == ActionType.CRANK_LEFT:
                turn_angle = params.turn_angle if params.turn_angle is not None else 35.0
                return self._execute_crank(env, agent_id, -turn_angle)

            elif action_type == ActionType.CRANK_RIGHT:
                turn_angle = params.turn_angle if params.turn_angle is not None else 35.0
                return self._execute_crank(env, agent_id, turn_angle)

            elif action_type == ActionType.NOTCH_MANEUVER:
                return self._execute_notch_maneuver(env, agent_id)

            elif action_type == ActionType.BEAM_MANEUVER:
                return self._execute_beam_maneuver(env, agent_id)

            elif action_type == ActionType.DIVE_ESCAPE:
                return self._execute_dive_escape(env, agent_id)

            elif action_type == ActionType.CHAFF_FLARE_MANEUVER:
                return self._execute_chaff_flare_maneuver(env, agent_id)

            elif action_type == ActionType.SPIRAL_DIVE:
                return self._execute_spiral_dive(env, agent_id)

            elif action_type == ActionType.SHORT_SKATE:
                return self._execute_short_skate_unified(env, agent_id, current_time)

            elif action_type == ActionType.AGGRESSIVE_APPROACH:
                return self._execute_aggressive_approach(env, agent_id)

            elif action_type == ActionType.DEFENSIVE_SPLIT:
                return self._execute_defensive_split(env, agent_id)

            elif action_type == ActionType.RETURN_TO_BASE:
                return self._execute_return_to_base_unified(env, agent_id, current_time, task)

            elif action_type == ActionType.CLIMB:
                altitude_change = params.altitude_change if params.altitude_change is not None else 500.0
                return self._execute_altitude_change(env, agent_id, altitude_change)

            elif action_type == ActionType.DESCEND:
                # 🛡️ 完全禁用下降动作，改为水平飞行
                current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
                logging.warning(f"🛡️ {agent_id} 下降动作已禁用（高度{current_altitude:.0f}m），改为水平飞行")
                return self._execute_maintain_heading(env, agent_id)

            else:
                # 默认保持航向
                return self._execute_maintain_heading(env, agent_id)

        except Exception as e:
            logging.error(f"动作执行失败 {agent_id} - {action_type.value}: {e}")
            return 7, 8, 3  # 默认平稳飞行
    
    def _apply_global_safety_check(self, env, agent_id: str, alt_cmd: int, hdg_cmd: int, vel_cmd: int, task=None) -> Tuple[int, int, int]:
        """
        全局高度安全检查（最终防线）
        确保所有指令都不会导致撞地 - 支持不同飞机型号
        """
        try:
            current_alt = env.agents[agent_id].get_property_value(c.position_h_sl_m)
            
            # 🛩️ 获取飞机特定的安全参数 - 优先使用task提供的参数
            if task and hasattr(task, 'get_aircraft_flight_params'):
                aircraft_params = task.get_aircraft_flight_params(agent_id)
                min_altitude = aircraft_params['min_safe_altitude']
                min_speed = aircraft_params['min_speed']
                max_speed = aircraft_params['max_speed']
            else:
                # 回退到内置参数
                aircraft_params = self._get_aircraft_parameters(env, agent_id)
                min_altitude = aircraft_params["min_altitude"]
                min_speed = aircraft_params["min_speed"]  
                max_speed = aircraft_params["max_speed"]
            
            # ✅ 三级保护：紧急拉升、强制水平、禁止下降
            emergency_threshold = min_altitude * 0.5   # 紧急高度（如SU-27: 1750m）
            warning_threshold = min_altitude * 1.0     # 警告高度（如SU-27: 3500m）  
            caution_threshold = min_altitude * 1.67    # 注意高度（如SU-27: 5833m）
            
            if current_alt < emergency_threshold:
                # 紧急拉升：强制最大爬升+减速
                alt_cmd = 0  # 最大爬升
                vel_cmd = min(vel_cmd, 3)  # 限制速度
                if current_alt < min_altitude * 0.17:  # 极度危险（如SU-27: 595m）
                    logging.error(f"🚨 [{agent_id}] 极度危险！高度{current_alt:.0f}m < {min_altitude*0.17:.0f}m，紧急拉升！")
                else:
                    logging.warning(f"🛡️ [{agent_id}] 紧急拉升: 高度{current_alt:.0f}m < {emergency_threshold:.0f}m")
            
            elif current_alt < warning_threshold:
                # 强制水平/爬升：禁止下降
                from envs.JSBSim.core import catalog as c_local
                alt_change = self.norm_delta_altitude[alt_cmd] * 1000
                
                if alt_change < 0:
                    # 禁止下降，改为水平飞行
                    alt_cmd = 7  # 保持高度
                    if env.current_step % 120 == 0:
                        logging.info(f"🛡️ [{agent_id}] 高度保护: 高度{current_alt:.0f}m < {warning_threshold:.0f}m，禁止下降")
            
            elif current_alt < caution_threshold:
                # 限制下降：只允许小幅下降
                from envs.JSBSim.core import catalog as c_local
                alt_change = self.norm_delta_altitude[alt_cmd] * 1000
                
                if alt_change < -300:
                    # 限制下降幅度到300m
                    alt_cmd = 5  # -100m
                    if env.current_step % 120 == 0:
                        logging.info(f"🛡️ [{agent_id}] 高度保护: 高度{current_alt:.0f}m < {caution_threshold:.0f}m，限制下降")
            
            # 速度安全检查
            current_velocity = env.agents[agent_id].get_property_value(c.velocities_u_mps)
            if current_velocity < min_speed:
                vel_cmd = min(6, vel_cmd + 1)  # 加速
                if env.current_step % 120 == 0:
                    logging.info(f"🛡️ [{agent_id}] 速度保护: 速度{current_velocity:.0f}m/s < {min_speed:.0f}m/s")
            elif current_velocity > max_speed:
                vel_cmd = max(0, vel_cmd - 1)  # 减速
                if env.current_step % 120 == 0:
                    logging.info(f"🛡️ [{agent_id}] 速度限制: 速度{current_velocity:.0f}m/s > {max_speed:.0f}m/s")
            
            return alt_cmd, hdg_cmd, vel_cmd
            
        except Exception as e:
            logging.error(f"全局安全检查失败 {agent_id}: {e}")
            return 7, 8, 3  # 默认平稳飞行
            
    # ✈️ 从pure_maneuver_task移植的工作控制函数
    def _convert_altitude_to_index(self, altitude_offset):
        """高度偏移转换为索引 - SU-27优化版本"""
        altitude_values = np.array([-1500.0, -1000.0, -750.0, -500.0, -250.0, -100.0, 0.0, 0.0, 100.0, 250.0, 500.0, 750.0, 1000.0, 1500.0])
        if altitude_offset >= 1200.0:  # 大幅爬升
            return 13
        elif altitude_offset >= 800.0:  # 中等爬升
            return 12
        elif altitude_offset >= 400.0:  # 轻微爬升
            return 11
        elif altitude_offset >= 150.0:  # 微调爬升
            return 10
        elif altitude_offset >= 50.0:  # 小幅爬升
            return 9
        elif altitude_offset >= -50.0:  # 平飞
            return 7
        elif altitude_offset >= -150.0:  # 小幅下降
            return 5
        elif altitude_offset >= -400.0:  # 轻微下降
            return 4
        elif altitude_offset >= -800.0:  # 中等下降
            return 2
        elif altitude_offset <= -1200.0:  # 大幅下降
            return 1
        else:
            distances = np.abs(altitude_values - altitude_offset)
            return np.argmin(distances)

    def _convert_heading_to_index(self, heading_offset):
        """航向偏移转换为索引 - SU-27优化版本"""
        heading_values = np.array([-180.0, -135.0, -90.0, -45.0, -22.5, -11.25, 0.0, 11.25, 22.5, 45.0, 90.0, 135.0, 180.0])
        if heading_offset >= 160.0:  # 大转弯右
            return 12
        elif heading_offset >= 110.0:  # 中转弯右
            return 11
        elif heading_offset >= 70.0:  # 小转弯右
            return 10
        elif heading_offset >= 35.0:  # 微调右
            return 9
        elif heading_offset >= 15.0:  # 轻微右
            return 8
        elif heading_offset >= -15.0:  # 直飞
            return 6
        elif heading_offset >= -35.0:  # 轻微左
            return 5
        elif heading_offset >= -70.0:  # 微调左
            return 4
        elif heading_offset >= -110.0:  # 小转弯左
            return 3
        elif heading_offset >= -160.0:  # 中转弯左
            return 2
        elif heading_offset <= -160.0:  # 大转弯左
            return 1
        else:
            distances = np.abs(heading_values - heading_offset)
            return np.argmin(distances)

    def _convert_velocity_to_index(self, velocity_offset):
        """速度偏移转换为索引 - SU-27优化版本"""
        velocity_values = np.array([-100.0, -75.0, -50.0, -25.0, -12.5, 0.0, 12.5, 25.0, 50.0, 75.0, 100.0])
        if velocity_offset >= 80.0:  # 大幅加速
            return 10
        elif velocity_offset >= 60.0:  # 中等加速
            return 9
        elif velocity_offset >= 30.0:  # 轻微加速
            return 8
        elif velocity_offset >= 15.0:  # 微调加速
            return 7
        elif velocity_offset >= 5.0:  # 小幅加速
            return 6
        elif velocity_offset >= -5.0:  # 保持速度
            return 5
        elif velocity_offset >= -15.0:  # 小幅减速
            return 4
        elif velocity_offset >= -30.0:  # 轻微减速
            return 3
        elif velocity_offset >= -60.0:  # 中等减速
            return 2
        elif velocity_offset <= -80.0:  # 大幅减速
            return 1
        else:
            distances = np.abs(velocity_values - velocity_offset)
            return np.argmin(distances)
            
    def _direct_control_mapping(self, env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id, basic_maneuver_name="default"):
        """直接控制映射 - 从pure_maneuver_task移植的SU-27版本"""
        try:
            from envs.JSBSim.core import catalog as c
            current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
            current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_rad)
            current_velocity = env.agents[agent_id].get_property_value(c.velocities_u_mps)
            
            # 安全索引访问
            altitude_cmd_id = min(altitude_cmd_id, len(self.norm_delta_altitude) - 1)
            heading_cmd_id = min(heading_cmd_id, len(self.norm_delta_heading) - 1)
            velocity_cmd_id = min(velocity_cmd_id, len(self.norm_delta_velocity) - 1)
            
            target_altitude_change = self.norm_delta_altitude[altitude_cmd_id] * 1000
            target_heading_change = self.norm_delta_heading[heading_cmd_id] * 180
            target_velocity_change = self.norm_delta_velocity[velocity_cmd_id] * 100
            
            aileron = 0.0
            elevator = 0.0
            rudder = 0.0
            throttle = 0.7
            
            # 高度控制
            if target_altitude_change > 300:
                elevator = 0.3
                throttle = 0.9
            elif target_altitude_change > 100:
                elevator = 0.15
                throttle = 0.8
            elif target_altitude_change < -300:
                elevator = -0.2
                throttle = 0.5
            elif target_altitude_change < -100:
                elevator = -0.1
                throttle = 0.6
            
            # 航向控制
            if target_heading_change > 20:
                aileron = 0.3
                rudder = 0.15
            elif target_heading_change > 5:
                aileron = 0.15
                rudder = 0.08
            elif target_heading_change < -20:
                aileron = -0.3
                rudder = -0.15
            elif target_heading_change < -5:
                aileron = -0.15
                rudder = -0.08
            
            # 速度控制
            if target_velocity_change > 30:
                throttle = min(1.0, throttle + 0.2)
            elif target_velocity_change < -30:
                throttle = max(0.3, throttle - 0.2)
            
            # 关键：低高度保护（和pure_maneuver_task保持一致）
            if current_altitude < 1000:
                elevator = max(elevator, 0.1)
                throttle = max(throttle, 0.8)
                
            return np.array([aileron, elevator, rudder, throttle])
            
        except Exception as e:
            logging.error(f"直接控制映射错误: {e}")
            return np.array([0.0, 0.0, 0.0, 0.7])

    def _calculate_safe_altitude_change(self, current_altitude: Optional[float], action_name: str) -> float:
        """🛡️ 智能高度感知俯冲策略 - 根据当前高度计算安全的高度变化"""
        if current_altitude is None:
            # 无法获取高度信息，采用保守策略
            logging.warning(f"🛡️ 无法获取高度信息，{action_name}采用保守策略（禁用俯冲）")
            return random.uniform(100.0, 300.0)  # 强制爬升

        if current_altitude > 3000.0:
            # 高度 > 3000m：允许适度俯冲（最大200-300m深度）
            max_dive = min(300.0, (current_altitude - 2500.0) * 0.5)  # 确保不低于2500m
            altitude_change = random.uniform(-max_dive, 200.0)  # 可俯冲或爬升
            logging.info(f"🛡️ {action_name}: 高度{current_altitude:.0f}m > 3000m，允许俯冲{max_dive:.0f}m")
            return altitude_change
        elif current_altitude > 2000.0:
            # 高度 2000-3000m：允许小幅俯冲（最大100-150m深度）
            max_dive = min(150.0, (current_altitude - 1800.0) * 0.3)  # 确保不低于1800m
            altitude_change = random.uniform(-max_dive, 150.0)  # 小幅俯冲或爬升
            logging.info(f"🛡️ {action_name}: 高度{current_altitude:.0f}m在2000-3000m，允许小幅俯冲{max_dive:.0f}m")
            return altitude_change
        else:
            # 高度 < 2000m：完全禁止俯冲，强制爬升
            altitude_change = random.uniform(200.0, 500.0)  # 强制爬升
            logging.warning(f"🛡️ {action_name}: 高度{current_altitude:.0f}m < 2000m，禁用俯冲，强制爬升{altitude_change:.0f}m")
            return altitude_change

    def _generate_action_parameters(self, action_type: ActionType, agent_id: str) -> ActionParameters:
        """生成随机化的动作参数 - 🛡️ 集成智能高度感知安全机制"""
        try:
            # 🛡️ 获取当前高度进行智能参数生成
            current_altitude = None
            if hasattr(self, '_current_env') and self._current_env and agent_id in self._current_env.agents:
                try:
                    current_altitude = self._current_env.agents[agent_id].get_property_value(c.position_h_sl_m)
                except:
                    current_altitude = None
            if action_type == ActionType.MAINTAIN_HEADING:
                return ActionParameters(duration=random.uniform(5.0, 15.0))

            elif action_type in [ActionType.TURN_LEFT, ActionType.TURN_RIGHT]:
                return ActionParameters(
                    duration=random.uniform(8.0, 20.0),
                    turn_angle=random.uniform(15.0, 45.0),
                    turn_rate=random.uniform(3.0, 8.0)
                )

            elif action_type in [ActionType.CRANK_LEFT, ActionType.CRANK_RIGHT]:
                return ActionParameters(
                    duration=random.uniform(10.0, 25.0),
                    turn_angle=random.uniform(25.0, 45.0),
                    turn_rate=random.uniform(4.0, 7.0)
                )

            elif action_type == ActionType.NOTCH_MANEUVER:
                return ActionParameters(
                    duration=random.uniform(12.0, 20.0),
                    turn_angle=random.choice([75.0, 80.0, 85.0]),  # 减少转弯角度：75-85°（原来85-95°）
                    turn_rate=random.uniform(4.0, 7.0)  # 降低转弯率：4-7°/s（原来6-10°/s）
                )

            elif action_type == ActionType.BEAM_MANEUVER:
                return ActionParameters(
                    duration=random.uniform(15.0, 30.0),
                    turn_angle=random.choice([70.0, 75.0, 80.0, -70.0, -75.0, -80.0]),  # 减少转弯角度：±70-80°（原来±85-95°）
                    turn_rate=random.uniform(3.0, 5.0)  # 降低转弯率：3-5°/s（原来4-6°/s）
                )

            elif action_type == ActionType.DIVE_ESCAPE:
                # 🛡️ 智能高度感知俯冲策略
                altitude_change = self._calculate_safe_altitude_change(current_altitude, "dive_escape")
                return ActionParameters(
                    duration=random.uniform(8.0, 15.0),
                    altitude_change=altitude_change,  # 🛡️ 智能高度感知俯冲
                    turn_angle=random.uniform(-30.0, 30.0),  # 减少转弯角度：±30°
                    turn_rate=random.uniform(4.0, 6.0)  # 降低转弯率：4-6°/s
                )

            elif action_type == ActionType.CHAFF_FLARE_MANEUVER:
                # 🛡️ 智能高度感知策略
                altitude_change = self._calculate_safe_altitude_change(current_altitude, "chaff_flare")
                return ActionParameters(
                    duration=random.uniform(10.0, 20.0),
                    turn_angle=random.choice([45.0, -45.0, 60.0, -60.0]),  # 进一步减少转弯角度：±45-60°
                    turn_rate=random.uniform(3.0, 6.0),  # 进一步降低转弯率：3-6°/s
                    altitude_change=altitude_change  # 🛡️ 智能高度感知
                )

            # elif action_type == ActionType.SPIRAL_DIVE:
            #     # ⚠️ 螺旋俯冲机动已被注释 - 过于激进，容易导致坠机
            #     # 原参数：转弯角度±360-720°，转弯率10-15°/s，俯冲800-1500m
            #     return ActionParameters(
            #         duration=random.uniform(12.0, 25.0),
            #         turn_angle=random.choice([180.0, -180.0]),  # 减少到半圈转弯
            #         turn_rate=random.uniform(6.0, 8.0),  # 大幅降低转弯率
            #         altitude_change=random.uniform(-600.0, -300.0)  # 减少俯冲深度
            #     )
            # elif action_type == ActionType.SPIRAL_DIVE:
            #     # ⚠️ 螺旋俯冲机动已完全注释 - 过于危险，容易导致坠机
            #     # 即使优化后仍有坠机风险，建议完全禁用
            #     return ActionParameters(
            #         duration=random.uniform(12.0, 20.0),
            #         turn_angle=random.choice([90.0, -90.0]),  # 进一步减少转弯角度
            #         turn_rate=random.uniform(3.0, 5.0),  # 进一步降低转弯率
            #         altitude_change=random.uniform(-200.0, -100.0)  # 最小俯冲深度
            #     )
            elif action_type == ActionType.SPIRAL_DIVE:
                # 🚫 螺旋俯冲机动已禁用 - 改为温和的转弯机动
                return ActionParameters(
                    duration=random.uniform(10.0, 15.0),
                    turn_angle=random.choice([60.0, -60.0]),  # 温和转弯：±60°
                    turn_rate=random.uniform(3.0, 5.0),  # 低转弯率：3-5°/s
                    altitude_change=0.0  # 不改变高度，避免俯冲风险
                )

            elif action_type == ActionType.SHORT_SKATE:
                return ActionParameters(
                    duration=random.uniform(35.0, 50.0),  # 总持续时间
                    turn_angle=random.uniform(35.0, 45.0),  # Crank角度
                    turn_rate=random.uniform(5.0, 8.0)
                )

            elif action_type == ActionType.AGGRESSIVE_APPROACH:
                # 🛡️ 智能高度感知策略
                altitude_change = self._calculate_safe_altitude_change(current_altitude, "aggressive_approach")
                # 攻击接近通常需要爬升，确保为正值
                if altitude_change < 0:
                    altitude_change = random.uniform(200.0, 500.0)
                return ActionParameters(
                    duration=random.uniform(20.0, 40.0),
                    velocity_change=random.uniform(20.0, 50.0),  # 加速
                    altitude_change=altitude_change  # 🛡️ 智能高度感知爬升
                )

            elif action_type == ActionType.DEFENSIVE_SPLIT:
                # 🛡️ 智能高度感知策略
                altitude_change = self._calculate_safe_altitude_change(current_altitude, "defensive_split")
                # 防御分离通常需要爬升，确保为正值
                if altitude_change < 0:
                    altitude_change = random.uniform(200.0, 400.0)
                return ActionParameters(
                    duration=random.uniform(15.0, 25.0),
                    turn_angle=random.uniform(35.0, 60.0),  # 大幅减少转弯角度：35-60°
                    altitude_change=altitude_change  # 🛡️ 智能高度感知
                )

            elif action_type == ActionType.RETURN_TO_BASE:
                return ActionParameters(
                    duration=float('inf'),  # 持续到任务结束
                    target_heading=0.0  # 北向返航
                )

            elif action_type == ActionType.CLIMB:
                return ActionParameters(
                    duration=random.uniform(10.0, 20.0),
                    altitude_change=random.uniform(200.0, 500.0)  # 爬升200-500m
                )

            elif action_type == ActionType.DESCEND:
                # 🛡️ DESCEND动作参数生成已禁用，改为爬升
                return ActionParameters(
                    duration=random.uniform(10.0, 20.0),
                    altitude_change=random.uniform(200.0, 500.0)  # 改为爬升200-500m
                )

            else:
                return ActionParameters(duration=10.0)

        except Exception as e:
            logging.error(f"动作参数生成失败 {action_type.value}: {e}")
            return ActionParameters(duration=10.0)

    def _execute_maintain_heading(self, env, agent_id: str) -> Tuple[int, int, int]:
        """执行保持航向动作"""
        # 根据敌方角色确定基本航向
        if agent_id == "B0100":
            target_heading = 180.0  # 长机南向
        elif agent_id == "B0200":
            target_heading = 175.0  # 僚机略微左偏
        else:
            target_heading = 180.0

        return self._maintain_heading_precise(env, agent_id, target_heading)

    def _execute_turn(self, env, agent_id: str, turn_angle: float, turn_rate: float) -> Tuple[int, int, int]:
        """执行转弯动作"""
        try:
            # 确保参数不为None
            if turn_angle is None:
                turn_angle = 0.0
            if turn_rate is None:
                turn_rate = 5.0

            current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
            target_heading = (current_heading + turn_angle) % 360.0

            # 限制转弯率
            heading_diff = ((target_heading - current_heading + 540) % 360) - 180
            max_turn = turn_rate * 0.2  # 假设0.2秒间隔
            actual_turn = np.clip(heading_diff, -max_turn, max_turn)
            final_heading = (current_heading + actual_turn) % 360.0

            return self._maintain_heading_precise(env, agent_id, final_heading)
        except Exception as e:
            logging.error(f"动作执行失败 {agent_id} - turn: {e}")
            # 返回默认平稳飞行指令
            return 7, 8, 3

    def _execute_crank(self, env, agent_id: str, crank_angle: float) -> Tuple[int, int, int]:
        """执行Crank机动"""
        try:
            # 获取最近敌机方位
            situation = self.situation_data.get(agent_id)
            if situation and situation.closest_enemy_bearing is not None:
                target_bearing = situation.closest_enemy_bearing
                # Crank机动：相对于目标方位偏转
                crank_heading = (target_bearing + crank_angle) % 360.0
            else:
                # 默认相对于当前航向Crank
                current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
                crank_heading = (current_heading + crank_angle) % 360.0

            return self._maintain_heading_precise(env, agent_id, crank_heading)
        except Exception as e:
            logging.error(f"动作执行失败 {agent_id} - crank: {e}")
            # 返回默认平稳飞行指令
            return 7, 8, 3

    def _execute_notch_maneuver(self, env, agent_id: str) -> Tuple[int, int, int]:
        """执行Notch机动 - 安全的侧向规避"""
        current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        current_velocity = np.linalg.norm(env.agents[agent_id].get_velocity())

        # 飞行安全检查
        MINIMUM_SAFE_ALTITUDE = 1000.0
        MINIMUM_SAFE_VELOCITY = 150.0

        if current_altitude < MINIMUM_SAFE_ALTITUDE:
            logging.warning(f"⚠️ {agent_id} Notch机动时高度过低，执行紧急爬升")
            return 7, 0, 3  # 直飞+爬升+保持速度

        if current_velocity < MINIMUM_SAFE_VELOCITY:
            logging.warning(f"⚠️ {agent_id} Notch机动时速度过低，执行加速")
            return 7, 8, 1  # 直飞+保持高度+加速

        situation = self.situation_data.get(agent_id)
        if situation and situation.missile_threats:
            # 相对于最近导弹威胁进行安全规避
            closest_missile = min(situation.missile_threats, key=lambda x: x['distance'])
            missile_pos = closest_missile['position']
            current_pos = env.agents[agent_id].get_position()

            # 计算导弹方位
            dx = missile_pos[0] - current_pos[0]
            dy = missile_pos[1] - current_pos[1]
            missile_bearing = np.rad2deg(np.arctan2(dy, dx))

            # 安全的规避角度（限制在45度以内）
            notch_angle = 45.0 if random.random() > 0.5 else -45.0  # 减小转弯角度
            notch_heading = (missile_bearing + notch_angle) % 360.0
        else:
            # 默认相对于当前航向安全转弯
            current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
            notch_angle = 45.0 if random.random() > 0.5 else -45.0  # 限制转弯角度
            notch_heading = (current_heading + notch_angle) % 360.0

        # logging.info(f"🔄 {agent_id} 安全Notch机动（高度{current_altitude:.0f}m）: {notch_angle:.1f}°")  # 注释掉，减少日志
        return self._maintain_heading_precise(env, agent_id, notch_heading)

    def _execute_beam_maneuver(self, env, agent_id: str) -> Tuple[int, int, int]:
        """执行Beam机动 - 侧向飞行规避锁定"""
        try:
            situation = self.situation_data.get(agent_id)
            if situation and situation.closest_enemy_bearing is not None:
                # 相对于最近敌机进行侧向机动
                enemy_bearing = situation.closest_enemy_bearing
                beam_angle = random.choice([85.0, 90.0, 95.0, -85.0, -90.0, -95.0])
                beam_heading = (enemy_bearing + beam_angle) % 360.0
            else:
                # 默认侧向机动
                current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
                beam_angle = random.choice([90.0, -90.0])
                beam_heading = (current_heading + beam_angle) % 360.0

            return self._maintain_heading_precise(env, agent_id, beam_heading)
        except Exception as e:
            logging.error(f"动作执行失败 {agent_id} - beam_maneuver: {e}")
            # 返回默认平稳飞行指令
            return 7, 8, 3

    def _execute_dive_escape(self, env, agent_id: str) -> Tuple[int, int, int]:
        """执行俯冲脱离机动 - 快速俯冲规避导弹"""
        try:
            current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
            current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)

            # 🛡️ 安全检查 - 如果高度过低，改为水平机动
            # 获取飞机特定参数
            aircraft_params = self._get_aircraft_parameters(env, agent_id)
            safe_altitude_threshold = aircraft_params["min_altitude"]  # 使用飞机特定的最低高度
            if current_altitude < safe_altitude_threshold:
                # 只在状态改变时输出日志
                if not hasattr(self, '_dive_escape_blocked') or agent_id not in self._dive_escape_blocked:
                    if not hasattr(self, '_dive_escape_blocked'):
                        self._dive_escape_blocked = {}
                    self._dive_escape_blocked[agent_id] = True
                    logging.warning(f"🛡️ {agent_id} 高度{current_altitude:.0f}m过低，俯冲脱离改为水平转弯")
                # 改为水平大角度转弯
                turn_angle = random.choice([90.0, -90.0])  # 大角度转弯
                target_heading = (current_heading + turn_angle) % 360.0
                return self._maintain_heading_with_altitude_speed(env, agent_id, target_heading, 1, 5)  # 水平转弯+爬升+加速
            else:
                # 高度足够，清除阻止标记
                if hasattr(self, '_dive_escape_blocked') and agent_id in self._dive_escape_blocked:
                    del self._dive_escape_blocked[agent_id]

            # 随机选择俯冲方向（可选择性转弯）
            turn_angle = random.uniform(-30.0, 30.0)  # 减少转弯角度
            target_heading = (current_heading + turn_angle) % 360.0

            # 🛡️ 俯冲高度：下降200-300米，但不低于安全高度
            dive_altitude = random.uniform(200.0, 300.0)  # 进一步减少俯冲深度
            target_altitude = max(current_altitude - dive_altitude, 4000.0)  # 确保不低于4000米

            # 如果计算出的目标高度等于安全高度，说明俯冲受限，改为水平机动
            if target_altitude >= current_altitude - 100:  # 实际俯冲小于100米
                logging.info(f"🛡️ {agent_id} 俯冲受限，改为水平机动")
                return self._maintain_heading_with_altitude_speed(env, agent_id, target_heading, 0, 5)  # 水平转弯+加速

            #logging.info(f"敌方{agent_id}执行俯冲脱离: 转弯{turn_angle:.1f}°, 俯冲{dive_altitude:.0f}m")
            return self._maintain_heading_with_altitude_speed(env, agent_id, target_heading, -1, 5)  # 温和俯冲+加速
        except Exception as e:
            logging.error(f"动作执行失败 {agent_id} - dive_escape: {e}")
            return 7, 8, 3

    def _execute_chaff_flare_maneuver(self, env, agent_id: str) -> Tuple[int, int, int]:
        """执行干扰弹配合机动 - 大角度转弯配合电子对抗"""
        try:
            current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
            current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)

            # 中等角度转弯（60-75度）- 已优化参数
            turn_angle = random.choice([60.0, -60.0, 75.0, -75.0])  # 使用优化后的参数
            target_heading = (current_heading + turn_angle) % 360.0

            # 🛡️ 完全安全的高度变化 - 完全禁用俯冲
            altitude_change = random.choice([7, 8, 9])  # 保持高度或温和爬升（修复：原错误用0,1都是俯冲！）
            logging.info(f"🛡️ {agent_id} 高度{current_altitude:.0f}m，干扰弹机动使用安全高度变化")

            speed_change = 5  # 加速脱离

            logging.info(f"敌方{agent_id}执行干扰弹机动: 转弯{turn_angle:.1f}°, 高度变化={altitude_change}")
            return self._maintain_heading_with_altitude_speed(env, agent_id, target_heading, altitude_change, speed_change)
        except Exception as e:
            logging.error(f"动作执行失败 {agent_id} - chaff_flare_maneuver: {e}")
            return 7, 8, 3

    def _execute_spiral_dive(self, env, agent_id: str) -> Tuple[int, int, int]:
        """执行螺旋俯冲机动 - 🛡️ 已改为安全的螺旋转弯机动"""
        try:
            current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
            current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)

            # 🛡️ 螺旋俯冲已禁用，改为安全的螺旋转弯
            #logging.info(f"🛡️ {agent_id} 螺旋俯冲已改为安全螺旋转弯（高度{current_altitude:.0f}m）")

            # 温和的螺旋转弯（180度或360度）
            spiral_angle = random.choice([180.0, -180.0, 360.0, -360.0])  # 减少转弯角度
            target_heading = (current_heading + spiral_angle) % 360.0

            # 🛡️ 完全禁用俯冲，改为水平或爬升
            safe_altitude_threshold = 5000.0
            if current_altitude < safe_altitude_threshold:
                altitude_change = 9  # 温和爬升150m（修复：原错误用1会导致严重俯冲1000m！）
                logging.info(f"🛡️ {agent_id} 高度较低，螺旋转弯配合爬升")
            else:
                altitude_change = random.choice([7, 8, 9])  # 保持高度或温和爬升（修复：原错误用0,1都是俯冲！）

            # logging.info(f"敌方{agent_id}执行安全螺旋转弯: 螺旋{spiral_angle:.1f}°, 高度变化={altitude_change}")
            return self._maintain_heading_with_altitude_speed(env, agent_id, target_heading, altitude_change, 4)  # 螺旋转弯+中等加速
        except Exception as e:
            logging.error(f"动作执行失败 {agent_id} - spiral_dive: {e}")
            return 7, 8, 3

    def _execute_short_skate_unified(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """执行统一的Short Skate机动 - 三阶段复合机动"""
        try:
            # 初始化Short Skate状态（如果需要）
            if not hasattr(self, 'short_skate_states'):
                self.short_skate_states = {}

            if agent_id not in self.short_skate_states:
                self._init_short_skate_unified(agent_id, current_time)

            state = self.short_skate_states[agent_id]
            phase_time = current_time - state['phase_start_time']

            # 阶段持续时间（随机化）
            crank_duration = random.uniform(6.0, 12.0)
            turn_cold_duration = random.uniform(15.0, 25.0)

            if state['phase'] == 'crank':
                if phase_time < crank_duration:
                    # 阶段1：Crank机动
                    crank_heading = (state['initial_heading'] + state['crank_angle']) % 360.0
                    return self._maintain_heading_precise(env, agent_id, crank_heading)
                else:
                    # 转入Turn Cold阶段
                    state['phase'] = 'turn_cold'
                    state['phase_start_time'] = current_time
                    logging.debug(f"敌方{agent_id} Short Skate: Crank → Turn Cold")

            if state['phase'] == 'turn_cold':
                if phase_time < turn_cold_duration:
                    # 阶段2：Turn Cold机动
                    turn_cold_heading = (state['initial_heading'] + state['turn_cold_angle']) % 360.0
                    return self._maintain_heading_precise(env, agent_id, turn_cold_heading)
                else:
                    # 转入Escape阶段
                    state['phase'] = 'escape'
                    state['phase_start_time'] = current_time
                    logging.debug(f"敌方{agent_id} Short Skate: Turn Cold → Escape")

            if state['phase'] == 'escape':
                # 阶段3：Escape - 加速逃离
                escape_heading = (state['initial_heading'] + state['turn_cold_angle']) % 360.0
                # 加速指令
                return self._maintain_heading_with_speed(env, agent_id, escape_heading, 5)  # 加速

            # 默认情况
            return self._maintain_heading_precise(env, agent_id, 180.0)

        except Exception as e:
            logging.error(f"Short Skate执行失败 {agent_id}: {e}")
            return 7, 8, 3

    def _init_short_skate_unified(self, agent_id: str, current_time: float):
        """初始化统一的Short Skate状态"""
        current_heading = 180.0  # 默认南向

        # 随机化Crank和Turn Cold角度
        if agent_id == "B0100":  # 长机
            crank_angle = random.uniform(-45.0, -25.0)  # 左侧Crank
            turn_cold_angle = random.uniform(-120.0, -80.0)  # 左侧Turn Cold
        elif agent_id == "B0200":  # 僚机
            crank_angle = random.uniform(25.0, 45.0)   # 右侧Crank
            turn_cold_angle = random.uniform(80.0, 120.0)   # 右侧Turn Cold
        else:
            # 随机选择方向
            side = random.choice([-1, 1])
            crank_angle = side * random.uniform(25.0, 45.0)
            turn_cold_angle = side * random.uniform(80.0, 120.0)

        self.short_skate_states[agent_id] = {
            'phase': 'crank',
            'phase_start_time': current_time,
            'initial_heading': current_heading,
            'crank_angle': crank_angle,
            'turn_cold_angle': turn_cold_angle
        }

        # logging.info(f"敌方{agent_id}启动Short Skate: Crank={crank_angle:.1f}°, Turn Cold={turn_cold_angle:.1f}°")

    def _execute_aggressive_approach(self, env, agent_id: str) -> Tuple[int, int, int]:
        """执行攻击接近机动"""
        try:
            situation = self.situation_data.get(agent_id)
            if situation and situation.closest_enemy_bearing is not None:
                # 朝向最近敌机
                target_heading = situation.closest_enemy_bearing
            else:
                # 默认南向接敌
                target_heading = 180.0

            # 攻击接近：保持航向，爬升，加速
            return self._maintain_heading_with_altitude_speed(env, agent_id, target_heading, 1, 5)  # 爬升+加速
        except Exception as e:
            logging.error(f"动作执行失败 {agent_id} - aggressive_approach: {e}")
            # 返回默认平稳飞行指令
            return 7, 8, 3

    def _execute_defensive_split(self, env, agent_id: str) -> Tuple[int, int, int]:
        """执行防御分离机动 - 🛡️ 添加飞行安全保护机制"""
        current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        current_velocity = np.linalg.norm(env.agents[agent_id].get_velocity())
        current_step = getattr(env, 'current_step', 0)

        # 飞行安全检查
        MINIMUM_SAFE_ALTITUDE = 1000.0  # 最低安全高度1000m
        MINIMUM_SAFE_VELOCITY = 150.0   # 最低安全速度150m/s

        if current_altitude < MINIMUM_SAFE_ALTITUDE:
            logging.warning(f"⚠️ {agent_id} 高度过低({current_altitude:.0f}m)，执行紧急爬升")
            return 7, 0, 3  # 直飞+爬升+保持速度

        if current_velocity < MINIMUM_SAFE_VELOCITY:
            logging.warning(f"⚠️ {agent_id} 速度过低({current_velocity:.0f}m/s)，执行加速")
            return 7, 8, 1  # 直飞+保持高度+加速

        # 初始化防御分离状态管理
        if not hasattr(self, '_defensive_split_states'):
            self._defensive_split_states = {}

        if agent_id not in self._defensive_split_states:
            self._defensive_split_states[agent_id] = {
                'start_step': current_step,
                'split_angle': None,
                'duration_limit': 150,  # 30秒限制（150步 * 0.2秒/步）
                'completed': False
            }

        state = self._defensive_split_states[agent_id]

        # 检查是否已经完成防御分离机动
        if state['completed'] or (current_step - state['start_step']) > state['duration_limit']:
            # 防御分离完成，切换到正常机动
            if agent_id in self._defensive_split_states:
                del self._defensive_split_states[agent_id]
            # 返回正常的直飞指令
            return 7, 8, 3  # 直飞+保持高度+保持速度

        # 确定分离角度（限制在安全范围内）
        if state['split_angle'] is None:
            if agent_id == "B0100":  # 长机左分离 - 限制角度
                state['split_angle'] = random.uniform(-35.0, -20.0)  # 减小角度范围
            elif agent_id == "B0200":  # 僚机右分离 - 限制角度
                state['split_angle'] = random.uniform(20.0, 35.0)   # 减小角度范围
            else:
                state['split_angle'] = random.choice([-30.0, 30.0])  # 限制最大角度

        split_angle = state['split_angle']

        # 安全的高度指令 - 根据当前高度决定
        if current_altitude > 12000:  # 高空时可以保持或轻微爬升
            altitude_cmd = random.choice([7, 8, 9])  # 保持高度或轻微/温和爬升
        else:  # 中低空时优先爬升
            altitude_cmd = 9  # 温和爬升150m（修复：原错误用0会导致极度俯冲1500m！）

        # 减少日志频率，避免刷屏
        if not hasattr(self, '_last_defensive_log_step'):
            self._last_defensive_log_step = {}
        if agent_id not in self._last_defensive_log_step:
            self._last_defensive_log_step[agent_id] = 0

        if current_step - self._last_defensive_log_step[agent_id] >= 50:  # 每10秒打印一次
            remaining_time = (state['duration_limit'] - (current_step - state['start_step'])) * 0.2
            altitude_action = "爬升" if altitude_cmd == 0 else ("保持" if altitude_cmd == 8 else "轻微下降")
            # logging.info(f"🛡️ {agent_id} 安全防御分离（高度{current_altitude:.0f}m，速度{current_velocity:.0f}m/s）：转弯{split_angle:.1f}°，{altitude_action}，剩余{remaining_time:.1f}s")  # 注释掉
            self._last_defensive_log_step[agent_id] = current_step

        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        split_heading = (current_heading + split_angle) % 360.0

        return self._maintain_heading_with_altitude(env, agent_id, split_heading, altitude_cmd)

    def _execute_return_to_base_unified(self, env, agent_id: str, current_time: float, task=None) -> Tuple[int, int, int]:
        """执行统一的返航机动 - 修复问题6：平稳返航，朝0度北向飞行"""
        try:
            # 初始化返航状态
            if not hasattr(self, 'return_states'):
                self.return_states = {}

            if agent_id not in self.return_states:
                self._init_return_to_base_unified(agent_id, current_time)
                logging.info(f"[T={current_time:.1f}s][敌方{agent_id}] 开始返航，目标航向0° (北向)")

            state = self.return_states[agent_id]

            # 🔧 修复问题6：简化返航逻辑，直接朝北方平稳飞行
            # 目标航向0度（北向），保持高度，保持速度
            return self._maintain_heading_with_altitude_speed(env, agent_id, 0.0, 7, 3)

        except Exception as e:
            logging.error(f"返航执行失败 {agent_id}: {e}")
            return self._maintain_heading_precise(env, agent_id, 0.0)

    def _init_return_to_base_unified(self, agent_id: str, current_time: float):
        """初始化统一的返航状态"""
        # 随机决定是否使用Short Skate作为返航机动
        use_short_skate = random.random() < 0.4  # 40%概率使用Short Skate返航

        if not hasattr(self, 'return_states'):
            self.return_states = {}

        self.return_states[agent_id] = {
            'phase': 'tactical_return',
            'start_time': current_time,
            'use_short_skate': use_short_skate
        }

        # if use_short_skate:
        #     logging.info(f"敌方{agent_id}开始战术返航 (包含Short Skate)")
        # else:
        #     logging.info(f"敌方{agent_id}开始直接返航")

    def _execute_altitude_change(self, env, agent_id: str, altitude_change: float) -> Tuple[int, int, int]:
        """执行高度变化 - 🛡️ 智能高度感知安全机制"""
        current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        
        # 初始化动作状态跟踪
        if not hasattr(self, 'altitude_action_state'):
            self.altitude_action_state = {}
        
        # 为该智能体创建状态
        if agent_id not in self.altitude_action_state:
            self.altitude_action_state[agent_id] = {'last_action': None, 'count': 0}

        if altitude_change > 0:
            # 爬升指令
            altitude_cmd = 9  # 温和爬升150m（修复：原错误用0会导致极度俯冲1500m！）
            action_key = 'climb'
            # 只在状态改变或每10次输出一次日志
            if self.altitude_action_state[agent_id]['last_action'] != action_key:
                self.altitude_action_state[agent_id]['last_action'] = action_key
                self.altitude_action_state[agent_id]['count'] = 0
                logging.info(f"🛡️ {agent_id} 执行爬升{altitude_change:.0f}m（当前高度{current_altitude:.0f}m）")
        else:
            # 俯冲指令 - 🛡️ 智能安全检查
            target_altitude = current_altitude + altitude_change  # altitude_change为负值

            if current_altitude < 2000.0:
                # 高度过低，完全禁用俯冲
                altitude_cmd = 10  # 小幅爬升300m（修复：原错误用0会导致极度俯冲1500m！）
                action_key = 'climb_emergency'
                if self.altitude_action_state[agent_id]['last_action'] != action_key:
                    self.altitude_action_state[agent_id]['last_action'] = action_key
                    self.altitude_action_state[agent_id]['count'] = 0
                    logging.warning(f"🛡️ {agent_id} 高度{current_altitude:.0f}m过低，俯冲{altitude_change:.0f}m已禁用，改为爬升")
            elif target_altitude < 1800.0:
                # 俯冲会导致过低，限制俯冲深度
                safe_altitude_change = current_altitude - 1800.0  # 最低到1800m
                altitude_cmd = 6  # 轻微俯冲50m（修复：原错误用-1实际是极度爬升1500m）
                action_key = 'dive_limited'
                if self.altitude_action_state[agent_id]['last_action'] != action_key:
                    self.altitude_action_state[agent_id]['last_action'] = action_key
                    self.altitude_action_state[agent_id]['count'] = 0
                    logging.warning(f"🛡️ {agent_id} 俯冲受限：原计划{altitude_change:.0f}m，限制为{safe_altitude_change:.0f}m")
            else:
                # 安全俯冲
                altitude_cmd = 6  # 轻微俯冲50m（修复：原错误用-1实际是极度爬升1500m）
                action_key = 'dive_safe'
                if self.altitude_action_state[agent_id]['last_action'] != action_key:
                    self.altitude_action_state[agent_id]['last_action'] = action_key
                    self.altitude_action_state[agent_id]['count'] = 0
                    logging.info(f"🛡️ {agent_id} 执行安全俯冲{altitude_change:.0f}m（当前高度{current_altitude:.0f}m → {target_altitude:.0f}m）")

        return altitude_cmd, 8, 3  # 保持航向和速度

    def _should_return_to_base(self, env, agent_id: str, current_time: float, closest_enemy_distance: float) -> bool:
        """检查是否应该返航 - 优化版本，减少长时间纠缠"""
        try:
            aircraft = env.agents[agent_id]
            
            # 返航条件1：导弹用尽且距离敌机较远且无敌方导弹威胁
            missiles_remaining = getattr(aircraft, 'num_missiles', 0)
            
            # 检查是否存在敌方导弹威胁（二次进攻时）
            enemy_missile_threat = False
            if hasattr(env, '_tempsims'):
                for missile_id, missile_sim in env._tempsims.items():
                    if missile_id.startswith('A'):  # 友方导弹威胁
                        missile_pos = missile_sim.get_position()
                        aircraft_pos = aircraft.get_position()
                        missile_distance = np.linalg.norm(np.array(aircraft_pos) - np.array(missile_pos))
                        if missile_distance < 80000:  # 80km内的导弹威胁
                            enemy_missile_threat = True
                            break
            
            # 只有在无导弹威胁时才考虑因弹药耗尽返航
            if missiles_remaining == 0 and closest_enemy_distance > 80000 and not enemy_missile_threat:
                # 减少日志频率
                if not hasattr(self, '_last_rtb_log_time'):
                    self._last_rtb_log_time = {}
                if agent_id not in self._last_rtb_log_time or (current_time - self._last_rtb_log_time[agent_id]) > 30.0:
                    logging.info(f"🚀 {agent_id} 导弹用尽且距离较远({closest_enemy_distance/1000:.1f}km)且无威胁，返航")
                    self._last_rtb_log_time[agent_id] = current_time
                return True
            
            # 返航条件2：仿真时间超过8分钟（480秒）- 延长任务时间支持二次进攻
            if current_time > 480.0:
                if not hasattr(self, '_last_rtb_log_time'):
                    self._last_rtb_log_time = {}
                if agent_id not in self._last_rtb_log_time or (current_time - self._last_rtb_log_time[agent_id]) > 30.0:
                    logging.info(f"⏰ {agent_id} 任务时间结束({current_time:.1f}s)，返航")
                    self._last_rtb_log_time[agent_id] = current_time
                return True
            
            # 返航条件3：高度过低且无法爬升
            current_altitude = aircraft.get_property_value(c.position_h_sl_m)
            if current_altitude < 2000 and closest_enemy_distance > 40000:  # 降低距离阈值
                return True
                
            # 返航条件4：距离敌机超过200km且无明确威胁且仿真时间超过6分钟
            if closest_enemy_distance > 200000 and current_time > 360.0:
                if not hasattr(self, '_last_rtb_log_time'):
                    self._last_rtb_log_time = {}
                if agent_id not in self._last_rtb_log_time or (current_time - self._last_rtb_log_time[agent_id]) > 30.0:
                    logging.info(f"📏 {agent_id} 距离过远({closest_enemy_distance/1000:.1f}km)且任务时间长，返航")
                    self._last_rtb_log_time[agent_id] = current_time
                return True
            
            # 返航条件5：任务目标达成（敌机数量减少或威胁消除）
            if self._is_mission_complete(env, agent_id):
                return True
                
            return False
            
        except Exception as e:
            logging.debug(f"返航条件检查失败 {agent_id}: {e}")
            return False

    def _is_mission_complete(self, env, agent_id: str) -> bool:
        """判断任务是否完成 - 增强版支持二次进攻"""
        try:
            # 检查友方敌机的存活状态
            friendly_agents = ['A0100', 'A0200']
            active_friendlies = 0
            
            for friendly_id in friendly_agents:
                if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                    active_friendlies += 1
            
            # 如果己方被全部击毁，敌方任务完成
            if active_friendlies == 0:
                return True
                
            # 检查己方（敌方）的状态
            enemy_agents = ['B0100', 'B0200']
            active_enemies = 0
            
            for enemy_id in enemy_agents:
                if enemy_id in env.agents and env.agents[enemy_id].is_alive:
                    active_enemies += 1
            
            # 检查是否在敌方导弹威胁下 - 二次进攻期间不应撤退
            current_time = getattr(env, 'current_step', 0) * getattr(env, 'time_interval', 0.2)
            under_missile_threat = False
            
            if hasattr(env, '_tempsims'):
                for missile_id, missile_sim in env._tempsims.items():
                    if missile_id.startswith('A'):  # 友方导弹威胁
                        try:
                            missile_pos = missile_sim.get_position()
                            aircraft = env.agents[agent_id]
                            aircraft_pos = aircraft.get_position()
                            missile_distance = np.linalg.norm(np.array(aircraft_pos) - np.array(missile_pos))
                            if missile_distance < 100000:  # 100km内的导弹威胁
                                under_missile_threat = True
                                break
                        except:
                            continue
            
            # 在导弹威胁下或仿真时间较短时不撤退（支持二次进攻）
            if under_missile_threat or current_time < 300.0:  # 5分钟内不考虑撤退
                return False
            
            # 只有在极端劣势且无威胁时才撤退
            if active_enemies == 1 and active_friendlies >= 2:
                return True
                
            return False
            
        except Exception as e:
            logging.debug(f"任务完成判断失败 {agent_id}: {e}")
            return False

    # ==================== 辅助函数 ====================

    def _maintain_heading_precise(self, env, agent_id: str, target_heading: float) -> Tuple[int, int, int]:
        """精确保持航向 - 基于现有系统的实现"""
        try:
            current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))

            # 计算航向差
            heading_diff = ((target_heading - current_heading + 540) % 360) - 180

            # 转换为航向指令索引
            if abs(heading_diff) < 2.0:
                heading_cmd_id = 8  # 保持航向
            elif heading_diff > 0:
                # 需要右转
                if abs(heading_diff) > 20:
                    heading_cmd_id = 12  # 大角度右转
                elif abs(heading_diff) > 10:
                    heading_cmd_id = 11  # 中角度右转
                else:
                    heading_cmd_id = 10  # 小角度右转
            else:
                # 需要左转
                if abs(heading_diff) > 20:
                    heading_cmd_id = 4   # 大角度左转
                elif abs(heading_diff) > 10:
                    heading_cmd_id = 5   # 中角度左转
                else:
                    heading_cmd_id = 6   # 小角度左转

            return 7, heading_cmd_id, 3  # 保持高度和速度

        except Exception as e:
            logging.error(f"航向保持失败 {agent_id}: {e}")
            return 7, 8, 3

    def _maintain_heading_with_speed(self, env, agent_id: str, target_heading: float, speed_cmd: int) -> Tuple[int, int, int]:
        """保持航向并调整速度"""
        altitude_cmd, heading_cmd, _ = self._maintain_heading_precise(env, agent_id, target_heading)
        return altitude_cmd, heading_cmd, speed_cmd

    def _maintain_heading_with_altitude(self, env, agent_id: str, target_heading: float, altitude_cmd: int) -> Tuple[int, int, int]:
        """保持航向并调整高度"""
        _, heading_cmd, speed_cmd = self._maintain_heading_precise(env, agent_id, target_heading)
        return altitude_cmd, heading_cmd, speed_cmd

    def _maintain_heading_with_altitude_speed(self, env, agent_id: str, target_heading: float,
                                            altitude_cmd: int, speed_cmd: int) -> Tuple[int, int, int]:
        """保持航向并调整高度和速度"""
        _, heading_cmd, _ = self._maintain_heading_precise(env, agent_id, target_heading)
        return altitude_cmd, heading_cmd, speed_cmd

    # ==================== 系统接口函数 ====================

    def reset_agent(self, agent_id: str):
        """重置智能体状态 - 用于新任务开始"""
        if agent_id in self.tactical_mode:
            del self.tactical_mode[agent_id]
        if agent_id in self.current_phase:
            del self.current_phase[agent_id]
        if agent_id in self.radar_mode:
            del self.radar_mode[agent_id]
        if agent_id in self.current_action:
            del self.current_action[agent_id]
        if agent_id in self.action_start_time:
            del self.action_start_time[agent_id]
        if agent_id in self.action_parameters:
            del self.action_parameters[agent_id]
        if agent_id in self.situation_data:
            del self.situation_data[agent_id]
        if agent_id in self.threat_assessment:
            del self.threat_assessment[agent_id]
        if agent_id in self.mode_switch_cooldown:
            del self.mode_switch_cooldown[agent_id]
        if agent_id in self.last_mode_switch:
            del self.last_mode_switch[agent_id]

        # 清理动作状态
        if hasattr(self, 'short_skate_states') and agent_id in self.short_skate_states:
            del self.short_skate_states[agent_id]
        if hasattr(self, 'return_states') and agent_id in self.return_states:
            del self.return_states[agent_id]

        # logging.info(f"敌方{agent_id}状态已重置")

    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """获取智能体当前状态信息 - 用于调试和监控"""
        return {
            'tactical_mode': self.tactical_mode.get(agent_id, TacticalMode.NEUTRAL).value,
            'current_phase': self.current_phase.get(agent_id, EnemyTacticalPhase.MELD_MTR).value,
            'current_action': self.current_action.get(agent_id, ActionType.MAINTAIN_HEADING).value,
            'action_start_time': self.action_start_time.get(agent_id, 0.0),
            'threat_level': self.threat_assessment.get(agent_id, ThreatAssessment(
                ThreatLevel.NONE, 0.0, None, 0, False, False)).threat_level.name,
            'has_short_skate': hasattr(self, 'short_skate_states') and agent_id in self.short_skate_states,
            'has_return_state': hasattr(self, 'return_states') and agent_id in self.return_states
        }

    def force_action(self, agent_id: str, action_type: ActionType, current_time: float):
        """强制执行特定动作 - 用于测试和调试"""
        self.current_action[agent_id] = action_type
        self.action_start_time[agent_id] = current_time
        self.action_parameters[agent_id] = self._generate_action_parameters(action_type, agent_id)
        logging.info(f"强制敌方{agent_id}执行动作: {action_type.value}")

    def reset_for_new_episode(self):
        """重置AI系统状态，准备新的仿真回合"""
        try:
            # 清空所有状态记录
            self.tactical_mode.clear()
            self.current_phase.clear()
            self.radar_mode.clear()
            self.current_action.clear()
            self.action_start_time.clear()
            self.action_parameters.clear()
            self.last_mode_switch.clear()
            self.threat_assessment.clear()
            self.situation_data.clear()

            # 清空特殊状态
            if hasattr(self, 'short_skate_states'):
                self.short_skate_states.clear()
            if hasattr(self, 'return_states'):
                self.return_states.clear()

            logging.info("🎯 统一敌方战术AI系统已重置，准备新回合")
        except Exception as e:
            logging.error(f"AI系统重置失败: {e}")

    def get_action_annotation(self, agent_id: str) -> str:
        """获取当前动作的Action_Intent注释信息 - 简化版本"""
        try:
            if agent_id not in self.current_action:
                return "search"  # 默认返回search而不是unknown

            action_type = self.current_action[agent_id]

            # 调试信息：检查action_type的类型
            if not isinstance(action_type, ActionType):
                logging.error(f"错误的action_type类型 {agent_id}: {type(action_type)} = {action_type}")
                return "search"

            tactical_mode = self.tactical_mode.get(agent_id, TacticalMode.NEUTRAL)
            situation = self.situation_data.get(agent_id)

            # 生成动作意图
            return self._generate_action_intent(action_type, tactical_mode, situation)

        except Exception as e:
            logging.error(f"动作注释获取失败 {agent_id}: {e}")
            return "search"  # 默认返回search而不是unknown

    def get_action_annotation_for_csv(self, agent_id: str) -> Dict[str, str]:
        """获取CSV格式的动作注释信息 - 简化版本，只返回Action_Intent"""
        try:
            action_intent = self.get_action_annotation(agent_id)
            return {
                'Action_Intent': action_intent
            }
        except Exception as e:
            logging.error(f"CSV动作注释获取失败 {agent_id}: {e}")
            return {
                'Action_Intent': 'search'  # 默认返回search而不是unknown
            }

    def get_action_type_for_csv(self, agent_id: str) -> Dict[str, str]:
        """获取CSV格式的具体战术动作类型信息"""
        try:
            if agent_id not in self.current_action:
                return {'action_type': ''}

            action_type = self.current_action[agent_id]

            # 确保action_type是ActionType枚举类型
            if not isinstance(action_type, ActionType):
                logging.error(f"错误的action_type类型 {agent_id}: {type(action_type)} = {action_type}")
                return {'action_type': ''}

            # 将ActionType枚举值转换为具体的战术动作名称
            action_type_name = self._convert_action_type_to_name(action_type)

            return {
                'action_type': action_type_name
            }
        except Exception as e:
            logging.error(f"CSV动作类型获取失败 {agent_id}: {e}")
            return {
                'action_type': ''
            }

    def _convert_action_type_to_name(self, action_type: ActionType) -> str:
        """将ActionType枚举转换为具体的战术动作名称"""
        try:
            # 映射ActionType到具体的战术动作名称
            action_mapping = {
                ActionType.MAINTAIN_HEADING: 'neutral_flight',
                ActionType.TURN_LEFT: 'defensive_turn',
                ActionType.TURN_RIGHT: 'defensive_turn',
                ActionType.CLIMB: 'climb_escape',
                ActionType.DESCEND: 'dive_escape',
                ActionType.ACCELERATE: 'aggressive_approach',
                ActionType.DECELERATE: 'defensive_positioning',
                ActionType.CRANK_LEFT: 'crank_left',
                ActionType.CRANK_RIGHT: 'crank_right',
                ActionType.NOTCH_MANEUVER: 'notch',
                ActionType.BEAM_MANEUVER: 'evasive_maneuver',
                ActionType.DIVE_ESCAPE: 'dive_escape',
                ActionType.CHAFF_FLARE_MANEUVER: 'evasive_maneuver',
                ActionType.SPIRAL_DIVE: 'evasive_maneuver',
                ActionType.SHORT_SKATE: 'aggressive_approach',
                ActionType.DEFENSIVE_SPLIT: 'split',
                ActionType.AGGRESSIVE_APPROACH: 'aggressive_approach',
                ActionType.RETURN_TO_BASE: 'escape'
            }

            return action_mapping.get(action_type, 'neutral_flight')
        except Exception as e:
            logging.error(f"动作类型转换失败: {e}")
            return 'neutral_flight'

    def _generate_action_intent(self, action_type: ActionType, tactical_mode: TacticalMode, situation) -> str:
        """生成动作意图"""
        try:
            # 基于动作类型和战术模式生成意图
            if action_type == ActionType.MAINTAIN_HEADING:
                if tactical_mode == TacticalMode.NEUTRAL:
                    return "search"
                elif tactical_mode == TacticalMode.AGGRESSIVE:
                    return "lock_on"
                else:
                    return "defensive_positioning"

            elif action_type in [ActionType.TURN_LEFT, ActionType.TURN_RIGHT]:
                if tactical_mode == TacticalMode.AGGRESSIVE:
                    return "attack_maneuver"
                elif tactical_mode == TacticalMode.DEFENSIVE:
                    return "evasive_maneuver"
                else:
                    return "formation_maintain"

            elif action_type in [ActionType.CRANK_LEFT, ActionType.CRANK_RIGHT]:
                return "attack_maneuver"

            elif action_type in [ActionType.NOTCH_MANEUVER, ActionType.BEAM_MANEUVER,
                               ActionType.DIVE_ESCAPE, ActionType.CHAFF_FLARE_MANEUVER,
                               ActionType.SPIRAL_DIVE]:
                return "evasive_maneuver"

            elif action_type == ActionType.SHORT_SKATE:
                return "attack_maneuver"

            elif action_type == ActionType.AGGRESSIVE_APPROACH:
                return "attack_maneuver"

            elif action_type == ActionType.DEFENSIVE_SPLIT:
                return "defensive_positioning"

            elif action_type == ActionType.RETURN_TO_BASE:
                return "escape"

            elif action_type in [ActionType.CLIMB, ActionType.DESCEND]:
                if tactical_mode == TacticalMode.DEFENSIVE:
                    return "evasive_maneuver"
                else:
                    return "formation_maintain"

            else:
                return "search"  # 默认返回search而不是unknown

        except Exception as e:
            logging.error(f"动作意图生成失败: {e}")
            return "search"  # 默认返回search而不是unknown

    def _get_short_skate_phase_annotation(self, agent_id: str) -> str:
        """获取Short Skate阶段注释"""
        if hasattr(self, 'short_skate_states') and agent_id in self.short_skate_states:
            phase = self.short_skate_states[agent_id]['phase']
            if phase == 'crank':
                return "SHORT_SKATE_CRANK"
            elif phase == 'turn_cold':
                return "SHORT_SKATE_TURN_COLD"
            elif phase == 'escape':
                return "SHORT_SKATE_ESCAPE"
        return "SHORT_SKATE"

    # ✈️ 从pure_maneuver_task移植的工作控制函数
    def _convert_altitude_to_index(self, altitude_offset):
        """高度偏移转换为索引 - SU-27优化版本"""
        altitude_values = np.array([-1500.0, -1000.0, -750.0, -500.0, -250.0, -100.0, 0.0, 0.0, 100.0, 250.0, 500.0, 750.0, 1000.0, 1500.0])
        if altitude_offset >= 1200.0:  # 大幅爬升
            return 13
        elif altitude_offset >= 800.0:  # 中等爬升
            return 12
        elif altitude_offset >= 400.0:  # 轻微爬升
            return 11
        elif altitude_offset >= 150.0:  # 微调爬升
            return 10
        elif altitude_offset >= 50.0:  # 小幅爬升
            return 9
        elif altitude_offset >= -50.0:  # 平飞
            return 7
        elif altitude_offset >= -150.0:  # 小幅下降
            return 5
        elif altitude_offset >= -400.0:  # 轻微下降
            return 4
        elif altitude_offset >= -800.0:  # 中等下降
            return 2
        elif altitude_offset <= -1200.0:  # 大幅下降
            return 1
        else:
            distances = np.abs(altitude_values - altitude_offset)
            return np.argmin(distances)

    def _convert_heading_to_index(self, heading_offset):
        """航向偏移转换为索引 - SU-27优化版本"""
        heading_values = np.array([-180.0, -135.0, -90.0, -45.0, -22.5, -11.25, 0.0, 11.25, 22.5, 45.0, 90.0, 135.0, 180.0])
        if heading_offset >= 160.0:  # 大转弯右
            return 12
        elif heading_offset >= 110.0:  # 中转弯右
            return 11
        elif heading_offset >= 70.0:  # 小转弯右
            return 10
        elif heading_offset >= 35.0:  # 微调右
            return 9
        elif heading_offset >= 15.0:  # 轻微右
            return 8
        elif heading_offset >= -15.0:  # 直飞
            return 6
        elif heading_offset >= -35.0:  # 轻微左
            return 5
        elif heading_offset >= -70.0:  # 微调左
            return 4
        elif heading_offset >= -110.0:  # 小转弯左
            return 3
        elif heading_offset >= -160.0:  # 中转弯左
            return 2
        elif heading_offset <= -160.0:  # 大转弯左
            return 1
        else:
            distances = np.abs(heading_values - heading_offset)
            return np.argmin(distances)

    def _convert_velocity_to_index(self, velocity_offset):
        """速度偏移转换为索引 - SU-27优化版本"""
        velocity_values = np.array([-100.0, -75.0, -50.0, -25.0, -12.5, 0.0, 12.5, 25.0, 50.0, 75.0, 100.0])
        if velocity_offset >= 80.0:  # 大幅加速
            return 10
        elif velocity_offset >= 60.0:  # 中等加速
            return 9
        elif velocity_offset >= 30.0:  # 轻微加速
            return 8
        elif velocity_offset >= 15.0:  # 微调加速
            return 7
        elif velocity_offset >= 5.0:  # 小幅加速
            return 6
        elif velocity_offset >= -5.0:  # 保持速度
            return 5
        elif velocity_offset >= -15.0:  # 小幅减速
            return 4
        elif velocity_offset >= -30.0:  # 轻微减速
            return 3
        elif velocity_offset >= -60.0:  # 中等减速
            return 2
        elif velocity_offset <= -80.0:  # 大幅减速
            return 1
        else:
            distances = np.abs(velocity_values - velocity_offset)
            return np.argmin(distances)
            
    def _direct_control_mapping(self, env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id, basic_maneuver_name="default"):
        """直接控制映射 - 从pure_maneuver_task移植的SU-27版本"""
        try:
            from envs.JSBSim.core import catalog as c
            current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
            current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_rad)
            current_velocity = env.agents[agent_id].get_property_value(c.velocities_u_mps)
            
            # 安全索引访问
            altitude_cmd_id = min(altitude_cmd_id, len(self.norm_delta_altitude) - 1)
            heading_cmd_id = min(heading_cmd_id, len(self.norm_delta_heading) - 1)
            velocity_cmd_id = min(velocity_cmd_id, len(self.norm_delta_velocity) - 1)
            
            target_altitude_change = self.norm_delta_altitude[altitude_cmd_id] * 1000
            target_heading_change = self.norm_delta_heading[heading_cmd_id] * 180
            target_velocity_change = self.norm_delta_velocity[velocity_cmd_id] * 100
            
            aileron = 0.0
            elevator = 0.0
            rudder = 0.0
            throttle = 0.7
            
            # 高度控制（SU-27特定参数）
            if target_altitude_change > 300:
                elevator = 0.3
                throttle = 0.9
            elif target_altitude_change > 100:
                elevator = 0.15
                throttle = 0.8
            elif target_altitude_change < -300:
                elevator = -0.2
                throttle = 0.5
            elif target_altitude_change < -100:
                elevator = -0.1
                throttle = 0.6
            
            # 航向控制（SU-27特定参数）
            if target_heading_change > 20:
                aileron = 0.3
                rudder = 0.15
            elif target_heading_change > 5:
                aileron = 0.15
                rudder = 0.08
            elif target_heading_change < -20:
                aileron = -0.3
                rudder = -0.15
            elif target_heading_change < -5:
                aileron = -0.15
                rudder = -0.08
            
            # 速度控制
            if target_velocity_change > 30:
                throttle = min(1.0, throttle + 0.2)
            elif target_velocity_change < -30:
                throttle = max(0.3, throttle - 0.2)
            
            # 关键：低高度保护（和pure_maneuver_task保持一致）
            if current_altitude < 1000:
                elevator = max(elevator, 0.1)
                throttle = max(throttle, 0.8)
                logging.debug(f"🛡️ [{agent_id}] 低高度保护激活: {current_altitude:.0f}m")
                
            return np.array([aileron, elevator, rudder, throttle])
            
        except Exception as e:
            logging.error(f"直接控制映射错误: {e}")
            return np.array([0.0, 0.0, 0.0, 0.7])


# ==================== 工厂函数 ====================

def create_unified_enemy_ai() -> UnifiedEnemyTacticalAI:
    """创建统一敌方战术AI实例"""
    return UnifiedEnemyTacticalAI()


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 创建AI实例
    enemy_ai = create_unified_enemy_ai()

    # 模拟使用
    print("🎯 统一敌方战术AI系统测试")
    print(f"支持的动作类型: {len(ActionType)} 种")
    print(f"支持的战术模式: {len(TacticalMode)} 种")
    print(f"支持的战术阶段: {len(EnemyTacticalPhase)} 种")
    print(f"支持的威胁等级: {len(ThreatLevel)} 种")

    # 测试动作权重配置
    for mode in TacticalMode:
        for phase in EnemyTacticalPhase:
            if mode in enemy_ai.action_weights and phase in enemy_ai.action_weights[mode]:
                actions = list(enemy_ai.action_weights[mode][phase].keys())
                print(f"{mode.value} + {phase.value}: {len(actions)} 种动作")

    print("✅ 统一敌方战术AI系统初始化完成")
