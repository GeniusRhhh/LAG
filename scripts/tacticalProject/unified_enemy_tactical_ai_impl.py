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
from typing import Dict, List, Tuple, Optional, Any

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

try:
    from . import enemy_ai_rtb_helpers as _erh
    from . import enemy_ai_maneuver_helpers as _emh
    from . import enemy_ai_refactor_helpers as _rfh
    from .enemy_ai_types import (
        ActionParameters,
        ActionType,
        EnemyTacticalPhase,
        RadarMode,
        SituationData,
        TacticalMode,
        ThreatAssessment,
        ThreatLevel,
    )
except ImportError:
    import enemy_ai_rtb_helpers as _erh
    import enemy_ai_maneuver_helpers as _emh
    import enemy_ai_refactor_helpers as _rfh
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
        self.action_parameter_type = {}  # 动作参数对应的动作类型
        
        # 态势感知数据
        self.situation_data = {}         # 态势数据缓存
        self.threat_assessment = {}      # 威胁评估缓存

        # 导弹发射管理
        self.last_missile_launch_time = {}  # 上次导弹发射时间
        self.enemy_missile_cooldown = 10.0  # 敌方10秒冷却时间
        self._last_enemy_missile_log = {}
        self._last_enemy_heading_log = {}
        self._enemy_launch_block_log = {}
        
        # 敌方阶段状态管理
        self._enemy_phases = {}  # 添加缺少的属性
        # ✅ 敌方“二次进攻”状态（不与我方完全一样，但具备：撤离→再进攻→再撤离）
        self._enemy_second_attack = {}
        self._enemy_second_attack_end_time = {}
        self._enemy_return_start_time = {}
        self._enemy_second_attack_done = set()
        
        # 🔥 简化模式：记录初始位置，用于计算飞行距离
        self._initial_positions = {}  # 记录每个飞机的初始位置
        # ✅ 敌方更强对抗：默认允许更久的交战（仍可用环境变量调回保守）
        self._flight_distance_threshold = float(os.getenv("ENEMY_FLIGHT_DISTANCE_THRESHOLD", "120000"))  # 米，默认120km
        # ✅ 敌方前出硬限制：不允许越过“前出线”，否则强制撤离/返航
        # 坐标轴约定：本项目常见初始化为A在x≈-30km，B在x≈+30km，敌方前出意味着x向0甚至负方向移动
        self._forward_limit_x = float(os.getenv("ENEMY_FORWARD_LIMIT_X", "-5000"))  # 米，默认x<=-5km才撤离（更敢前出）
        # 默认稍放宽：允许敌方更接近一些再撤离（避免过早返航导致我方出现“追击画面”）
        self._min_separation_to_friendly = float(os.getenv("ENEMY_MIN_SEPARATION", "25000"))  # 米，默认25km（更敢贴近）
        # 多样性控制：0=更确定，1=更多样（建议0.2~0.4）
        self._behavior_diversity = float(np.clip(float(os.getenv("ENEMY_BEHAVIOR_DIVERSITY", "0.30")), 0.0, 1.0))
        # 远距阶段动作多样化开关
        self._long_range_mix_enabled = str(os.getenv("ENEMY_LONG_RANGE_MIX", "1")).strip().lower() not in ("0", "false", "off")
        # 敌方开局风格与短时行为记忆：用于提升“初始与全程”机动多样性
        self._opening_style = {}
        self._last_action_history = {}
        # 导弹威胁短时规避窗口：威胁出现时优先进入规避动作而非继续旧动作
        self._missile_evasion_until = {}

        # 随机化参数
        self.mode_switch_cooldown = {}   # 模式切换冷却时间
        self.last_mode_switch = {}       # 上次模式切换时间
        
        # 🛩️ 飞机型号相关参数 - 🔥 修复版本：提高安全阈值
        self.aircraft_parameters = {
            "f16": {
                "min_altitude": 2000,      # F-16最低安全高度（降低基准值）
                "max_altitude": 15000,     # F-16最高高度
                "min_speed": 120,          # F-16最低速度
                "max_speed": 400,          # F-16最高速度
                "turn_rate": 8.0,          # F-16转弯率
                "climb_rate": 50.0         # F-16爬升率
            },
            "su27sk": {
                "min_altitude": 2500,      # SU-27最低安全高度（降低基准值）
                "max_altitude": 18000,     # SU-27最高高度
                "min_speed": 140,          # SU-27最低速度（更高）
                "max_speed": 500,          # SU-27最高速度
                "turn_rate": 6.5,          # SU-27转弯率（较低）
                "climb_rate": 60.0         # SU-27爬升率
            }
        }
        
        # ✈️ 修正：与tactical_task.py保持一致的控制函数索引数组
        self.norm_delta_altitude = np.array([
            -1500, -1000, -750, -500, -300, -150, -50, 0, 50, 150, 300, 500, 750, 1000, 1500
        ]) / 1000.0
        
        self.norm_delta_heading = np.array([
            -np.pi, -2*np.pi/3, -np.pi/2, -5*np.pi/12, -np.pi/3, -np.pi/4, -np.pi/6, -np.pi/12,
            0, np.pi/12, np.pi/6, np.pi/4, np.pi/3, 5*np.pi/12, np.pi/2, 2*np.pi/3, np.pi
        ])
        
        self.norm_delta_velocity = np.array([-150, -100, -50, 0, 50, 100, 150]) / 100.0
        
        # 为SU-27优化的控制数值（和pure_maneuver_task保持一致）
        self._inner_rnn_states = {}  # RNN状态缓存
        
        # 动作权重配置
        self._init_action_weights()
        
        # 🔥 动态威胁响应系统
        self.threat_response_multipliers = {}
        self._init_threat_response_system()
        
        # 🎯 意图识别映射表 - 17动作类型到4意图标签
        self.action_to_intent_mapping = {
            ActionType.MAINTAIN_HEADING: "RECONNAISSANCE",
            ActionType.TURN_LEFT: "DEFENSE",
            ActionType.TURN_RIGHT: "DEFENSE",
            ActionType.CLIMB: "DEFENSE",
            ActionType.DESCEND: "DEFENSE",
            ActionType.ACCELERATE: "ATTACK",
            ActionType.DECELERATE: "DEFENSE",
            ActionType.CRANK_LEFT: "ATTACK",
            ActionType.CRANK_RIGHT: "ATTACK",
            ActionType.NOTCH_MANEUVER: "DEFENSE",
            ActionType.BEAM_MANEUVER: "DEFENSE",
            ActionType.DIVE_ESCAPE: "DEFENSE",
            ActionType.CHAFF_FLARE_MANEUVER: "DEFENSE",
            ActionType.SPIRAL_DIVE: "DEFENSE",
            ActionType.SHORT_SKATE: "DEFENSE",
            ActionType.DEFENSIVE_SPLIT: "DEFENSE",
            ActionType.AGGRESSIVE_APPROACH: "ATTACK",
            ActionType.RETURN_TO_BASE: "RETREAT"
        }
        
        logging.info("[敌方战术AI] 统一敌方战术AI系统初始化完成")

    def _enemy_rtb_enabled(self) -> bool:
        # 默认启用返航（可通过环境变量 ENEMY_RTB_ENABLED=0 临时关闭）
        return str(os.getenv("ENEMY_RTB_ENABLED", "1")).strip().lower() not in ("0", "false", "off")

    def _get_enemy_teammate_id(self, agent_id: str) -> Optional[str]:
        team_map = {
            'B0100': 'B0200',
            'B0200': 'B0100',
            'B0300': 'B0400',
            'B0400': 'B0300',
        }
        return team_map.get(agent_id)

    def _get_preferred_target_id(self, agent_id: str) -> Optional[str]:
        target_map = {
            'B0100': 'A0100',
            'B0200': 'A0200',
            'B0300': 'A0300',
            'B0400': 'A0400',
        }
        return target_map.get(agent_id)

    def _get_opening_style(self, agent_id: str) -> str:
        return _rfh._get_opening_style(self, agent_id)

    def _opening_phase_action_weights(
        self,
        agent_id: str,
        situation: Optional[SituationData],
        current_phase: EnemyTacticalPhase,
        current_time: float,
    ) -> Optional[Dict[ActionType, float]]:
        return _rfh._opening_phase_action_weights(self, agent_id, situation, current_phase, current_time)

    def _should_interrupt_current_action_for_missile(self, agent_id: str) -> bool:
        return _rfh._should_interrupt_current_action_for_missile(self, agent_id)

    def _apply_missile_evasion_override(
        self,
        agent_id: str,
        action_weights: Dict[ActionType, float],
        current_time: float,
    ) -> Dict[ActionType, float]:
        return _rfh._apply_missile_evasion_override(self, agent_id, action_weights, current_time)

    def _apply_anti_repeat_diversity(
        self,
        agent_id: str,
        action_weights: Dict[ActionType, float],
    ) -> Dict[ActionType, float]:
        return _rfh._apply_anti_repeat_diversity(self, agent_id, action_weights)

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
    
    def _init_threat_response_system(self):
        """初始化动态威胁响应系统"""
        # 威胁等级对规避动作的权重乘数
        self.threat_response_multipliers = {
            ThreatLevel.SEVERE: {
                ActionType.NOTCH_MANEUVER: 4.0,      # 严重威胁时Notch权重x4
                ActionType.BEAM_MANEUVER: 3.5,       # Beam权重x3.5
                ActionType.DIVE_ESCAPE: 3.0,         # 俯冲脱离x3
                ActionType.CHAFF_FLARE_MANEUVER: 2.8, # 干扰弹x2.8
                ActionType.SPIRAL_DIVE: 2.5,         # 螺旋俯冲x2.5
                ActionType.DEFENSIVE_SPLIT: 2.0      # 防御分离x2
            },
            ThreatLevel.CRITICAL: {
                ActionType.NOTCH_MANEUVER: 3.0,      # 严重威胁时权重x3
                ActionType.BEAM_MANEUVER: 2.5,       # Beam权重x2.5
                ActionType.DIVE_ESCAPE: 2.2,         # 俯冲脱离x2.2
                ActionType.CHAFF_FLARE_MANEUVER: 2.0, # 干扰弹x2
                ActionType.DEFENSIVE_SPLIT: 1.8      # 防御分离x1.8
            },
            ThreatLevel.HIGH: {
                ActionType.NOTCH_MANEUVER: 2.0,      # 高威胁时权重x2
                ActionType.BEAM_MANEUVER: 1.8,       # Beam权重x1.8
                ActionType.DIVE_ESCAPE: 1.5,         # 俯冲脱离x1.5
                ActionType.CHAFF_FLARE_MANEUVER: 1.3, # 干扰弹x1.3
            },
            ThreatLevel.MEDIUM: {
                ActionType.NOTCH_MANEUVER: 1.3,      # 中等威胁时权重x1.3
                ActionType.BEAM_MANEUVER: 1.2,       # Beam权重x1.2
            }
        }
        
        # 导弹接近距离对规避动作的紧急权重乘数
        self.missile_distance_multipliers = {
            5000: 5.0,   # 5km内：x5权重
            10000: 3.5,  # 10km内：x3.5权重
            15000: 2.5,  # 15km内：x2.5权重
            20000: 1.8,  # 20km内：x1.8权重
            30000: 1.3   # 30km内：x1.3权重
        }
        
        logging.info("[敌方威胁响应] 动态威胁响应系统初始化完成")

    def handle_missile_launch(self, env, agent_id: str, current_time: float):
        """处理敌方导弹发射逻辑"""
        if not agent_id.startswith('B'):  # 只处理敌方
            return

        if not env.agents[agent_id].is_alive:
            return

        # 检查导弹数量
        missiles_left = int(
            getattr(env.agents[agent_id], 'num_left_missiles', getattr(env.agents[agent_id], 'num_missiles', 0)) or 0
        )
        if missiles_left <= 0:
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

        # 🔥 优化：敌方导弹发射判断 - 添加合理分配和冷却控制
        should_launch = self._enemy_should_launch_missile(env, agent_id, target, distance, current_time)
        if (not should_launch) and distance <= 70000.0:
            fallback = self._find_launchable_fallback_target(env, agent_id, target, current_time)
            if fallback is not None:
                target, distance = fallback
                should_launch = True

        if should_launch:
            # 🔥 修复：避免一次性全部发射，添加发射间隔控制
            # 初始化发射间隔跟踪
            if not hasattr(self, '_enemy_missile_launch_intervals'):
                self._enemy_missile_launch_intervals = {}
            
            # 检查是否在发射间隔内（避免一次性全部发射）
            last_launch_time = self.last_missile_launch_time.get(agent_id, -999)
            min_launch_interval = 3.0  # 最小发射间隔3秒，避免一次性全部发射
            
            if current_time - last_launch_time < min_launch_interval:
                # 在发射间隔内，不发射
                return
            
            # 检查是否已经有太多导弹在飞行（避免过度发射）
            active_missiles_count = 0
            if hasattr(env, 'missiles') and env.missiles:
                for missile_id, missile in env.missiles.items():
                    if missile_id.startswith(agent_id) and hasattr(missile, 'is_alive') and missile.is_alive:
                        active_missiles_count += 1
            
            # 最多同时有2枚导弹在飞行
            if active_missiles_count >= 2:
                # 🔥 调试：敌方AI的日志不打印
                # if env.current_step % 300 == 0:
                #     logging.debug(f"[T={current_time:.1f}s][敌方导弹] {agent_id} 已有{active_missiles_count}枚导弹在飞行，暂不发射")
                return
            
            self._launch_missile(env, agent_id, target, current_time)

    def _log_enemy_launch_block(
        self,
        agent_id: str,
        reason: str,
        current_time: float,
        target_id: Optional[str] = None,
        distance: Optional[float] = None,
        heading_error_deg: Optional[float] = None,
        extra: str = "",
    ) -> None:
        key = (str(agent_id), str(reason), str(target_id or ""))
        last_time = float(self._enemy_launch_block_log.get(key, -999.0))
        if current_time - last_time < 8.0:
            return
        self._enemy_launch_block_log[key] = float(current_time)

        parts = [f"[ENEMY_LAUNCH_BLOCK] {agent_id} reason={reason}"]
        if target_id:
            parts.append(f"target={target_id}")
        if distance is not None and np.isfinite(distance):
            parts.append(f"dist={distance/1000.0:.1f}km")
        if heading_error_deg is not None and np.isfinite(heading_error_deg):
            parts.append(f"heading_err={heading_error_deg:.1f}deg")
        if extra:
            parts.append(str(extra))
        logging.info(" ".join(parts))

    def _find_launchable_fallback_target(self, env, agent_id: str, primary_target, current_time: float):
        primary_uid = getattr(primary_target, 'uid', None)
        current_pos = np.array(env.agents[agent_id].get_position(), dtype=np.float64)
        candidates = []
        for target_id, target_agent in env.agents.items():
            if not str(target_id).startswith('A') or not getattr(target_agent, 'is_alive', False):
                continue
            if getattr(target_agent, 'uid', None) == primary_uid:
                continue
            try:
                target_pos = np.array(target_agent.get_position(), dtype=np.float64)
                distance = float(np.linalg.norm(current_pos - target_pos))
            except Exception:
                continue
            if distance > 80000.0:
                continue
            candidates.append((distance, target_agent))

        candidates.sort(key=lambda item: item[0])
        for distance, target_agent in candidates[:2]:
            if self._enemy_should_launch_missile(env, agent_id, target_agent, distance, current_time):
                logging.info(
                    "[ENEMY_LAUNCH_RETARGET] %s primary=%s fallback=%s dist=%.1fkm",
                    agent_id,
                    primary_uid,
                    getattr(target_agent, 'uid', None),
                    distance / 1000.0,
                )
                return target_agent, distance
        return None

    def _find_best_target(self, env, agent_id: str):
        """寻找最佳攻击目标 - 🔥 优化：实现合理目标分配，避免都打同一架飞机"""
        current_pos = env.agents[agent_id].get_position()
        nearest_target = None
        nearest_distance = float('inf')
        for target_id, target_agent in env.agents.items():
            if target_id.startswith('A') and target_agent.is_alive:
                target_pos = target_agent.get_position()
                distance = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_target = target_agent

        if nearest_target is not None and nearest_distance <= 55000:
            return nearest_target

        # 🔥 修复：实现目标分配逻辑
        # B0100优先打A0100，B0200优先打A0200，但如果目标已死亡或距离过远，则选择另一个
        preferred_target_id = self._get_preferred_target_id(agent_id)
        
        # 检查首选目标是否可用
        if preferred_target_id and preferred_target_id in env.agents:
            preferred_target = env.agents[preferred_target_id]
            if preferred_target.is_alive:
                target_pos = preferred_target.get_position()
                distance = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
                # 如果首选目标在合理范围内（20-100km），优先选择
                if 20000 <= distance <= 100000 and (nearest_target is None or (distance - nearest_distance) <= 12000):
                    return preferred_target
        
        # 如果首选目标不可用或距离不合理，选择最近的目标
        best_target = None
        min_distance = float('inf')

        for target_id, target_agent in env.agents.items():
            if target_id.startswith('A') and target_agent.is_alive:  # 友方目标
                target_pos = target_agent.get_position()
                distance = np.linalg.norm(np.array(current_pos) - np.array(target_pos))

                if distance < min_distance:
                    min_distance = distance
                    best_target = target_agent

        return best_target

    def _enemy_should_launch_missile(self, env, agent_id: str, target, distance: float, current_time: float) -> bool:
        """敌方智能导弹发射判断 - 修复问题7：添加朝向检查"""
        def _missiles_left(aircraft) -> int:
            try:
                if hasattr(aircraft, 'num_left_missiles'):
                    return max(0, int(getattr(aircraft, 'num_left_missiles')))
            except Exception:
                pass
            return max(0, int(getattr(aircraft, 'num_missiles', 0)))

        last = self._last_enemy_missile_log.get(agent_id, -999)
        target_uid = getattr(target, 'uid', None)
        # 🔥 调试：敌方AI的日志不打印
        # if current_time - last >= 5.0:
        #     logging.debug(f"[T={current_time:.1f}s][敌方导弹] {agent_id} 检查发射条件: 距离={distance/1000:.1f}km")
        #     self._last_enemy_missile_log[agent_id] = current_time
        
        # 基本条件检查
        missiles_remaining = _missiles_left(env.agents[agent_id])
        if missiles_remaining <= 0:
            # 🔥 调试：敌方AI的日志不打印
            # logging.info(f"[T={current_time:.1f}s][敌方导弹] {agent_id} 导弹已用尽")
            return False

        # 冷却时间检查
        last_launch = self.last_missile_launch_time.get(agent_id, -999)
        if current_time - last_launch < self.enemy_missile_cooldown:
            return False  # 冷却中不打印日志

        # 距离条件：敌方同样限制在受控 BVR 窗口，但残局允许更近的收官一发。
        min_launch_range = float(os.getenv("ENEMY_LAUNCH_MIN_M", "40000"))
        max_launch_range = float(os.getenv("ENEMY_LAUNCH_MAX_M", "76000"))

        # 修复：放宽朝向检查，从±45度放宽到±60度
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
        
        # 修复：放宽朝向检查，从±45度放宽到±60度，增加发射机会
        if heading_error_deg > 75.0:
            if distance <= 65000.0:
                self._log_enemy_launch_block(
                    agent_id,
                    "heading_window",
                    current_time,
                    target_id=str(target_uid or ""),
                    distance=distance,
                    heading_error_deg=heading_error_deg,
                )
            return False

        enemy_alive = [
            aid for aid, aircraft in env.agents.items()
            if str(aid).startswith('B') and getattr(aircraft, 'is_alive', False)
        ]
        alive_targets = [
            aid for aid, aircraft in env.agents.items()
            if str(aid).startswith('A') and getattr(aircraft, 'is_alive', False)
        ]
        team_missiles_remaining = sum(
            _missiles_left(other_aircraft)
            for other_id, other_aircraft in env.agents.items()
            if str(other_id).startswith('B') and getattr(other_aircraft, 'is_alive', False)
        )
        friendly_team_missiles_remaining = sum(
            _missiles_left(other_aircraft)
            for other_id, other_aircraft in env.agents.items()
            if str(other_id).startswith('A') and getattr(other_aircraft, 'is_alive', False)
        )
        numerical_advantage = int(len(enemy_alive) - len(alive_targets))
        late_target_pool = len(alive_targets) <= 2
        strong_finish_window = bool(
            late_target_pool
            and numerical_advantage >= 1
        )
        missile_finish_window = bool(
            team_missiles_remaining >= (friendly_team_missiles_remaining + 2)
            and (
                friendly_team_missiles_remaining <= 1
                or late_target_pool
            )
        )
        finish_attack_window = bool(
            friendly_team_missiles_remaining <= 0
            or strong_finish_window
            or missile_finish_window
        )
        owner_task = getattr(env, "task", None)
        enemy_adapter = getattr(owner_task, "enemy_adapter", None) if owner_task is not None else None
        north_phase_status = {}
        if enemy_adapter is not None and hasattr(enemy_adapter, "evaluate_north_phase_status"):
            target_zone = "UNKNOWN"
            target_y_km = float("nan")
            try:
                if owner_task is not None and hasattr(owner_task, "_classify_target_risk_zone"):
                    target_zone = str(owner_task._classify_target_risk_zone(target, env=env) or "UNKNOWN")
            except Exception:
                target_zone = "UNKNOWN"
            try:
                if owner_task is not None and hasattr(owner_task, "_get_aircraft_battlefield_depth_km"):
                    target_y_km = float(owner_task._get_aircraft_battlefield_depth_km(env, target))
            except Exception:
                target_y_km = float("nan")
            try:
                north_phase_status = enemy_adapter.evaluate_north_phase_status(
                    agent_id,
                    target_y_km=target_y_km,
                    target_zone=target_zone,
                    distance_km=distance / 1000.0,
                ) or {}
            except Exception:
                north_phase_status = {}
        north_phase_hard_escape = bool(north_phase_status.get("hard_escape_active", False))
        north_phase_block = bool(
            str(north_phase_status.get("phase", "") or "").upper() in {"TURN_NORTH", "REGROUP_NORTH"}
            and (
                north_phase_hard_escape
                or bool(north_phase_status.get("disengaged", False))
                or not bool(north_phase_status.get("pressing_override", False))
            )
        )
        endgame_release_range = float(os.getenv("ENEMY_ENDGAME_RELEASE_M", "54000"))
        endgame_commit_window = bool(
            late_target_pool
            and distance <= max(min_launch_range + 8000.0, endgame_release_range - (4000.0 if missiles_remaining <= 1 else 0.0))
            and heading_error_deg <= 45.0
            and (
                missiles_remaining <= 1
                or team_missiles_remaining <= (len(alive_targets) + 1)
            )
            and (
                strong_finish_window
                or friendly_team_missiles_remaining <= 1
                or team_missiles_remaining <= (friendly_team_missiles_remaining + 1)
            )
            and not north_phase_block
        )
        target_uid = getattr(target, 'uid', None)
        north_phase_name = str(north_phase_status.get("phase", "") or "").upper()
        north_phase_press_shot = bool(
            north_phase_name in {"TURN_NORTH", "REGROUP_NORTH"}
            and not north_phase_hard_escape
            and 42000.0 <= distance <= 60000.0
            and heading_error_deg <= 32.0
            and (
                bool(north_phase_status.get("pressing_override", False))
                or target_zone in {"MEDIUM", "HIGH"}
                or missiles_remaining <= 1
                or team_missiles_remaining >= friendly_team_missiles_remaining
            )
        )
        north_phase_finish_override = bool(
            north_phase_name in {"TURN_NORTH", "REGROUP_NORTH"}
            and not north_phase_hard_escape
            and 35000.0 <= distance < min_launch_range
            and distance <= min(46000.0, max(min_launch_range, 42000.0))
            and heading_error_deg <= 18.0
            and (
                finish_attack_window
                or target_zone in {"MEDIUM", "HIGH"}
                or missiles_remaining <= 1
                or team_missiles_remaining <= (friendly_team_missiles_remaining + 2)
            )
        )
        close_finish_shot = bool(
            (
                35000.0 <= distance < min_launch_range
                and heading_error_deg <= 50.0
                and (
                    finish_attack_window
                    or missiles_remaining <= 1
                    or len(alive_targets) <= 1
                )
            )
            or north_phase_finish_override
        ) and not north_phase_hard_escape
        if distance < min_launch_range or distance > max_launch_range:
            if close_finish_shot:
                logging.info(
                    "[ENEMY_LAUNCH_OVERRIDE] %s reason=close_finish_shot target=%s dist=%.1fkm heading_err=%.1fdeg left=%d",
                    agent_id,
                    str(target_uid or ""),
                    distance / 1000.0,
                    heading_error_deg,
                    missiles_remaining,
                )
                return True
            if distance < min_launch_range and distance >= max(25000.0, min_launch_range - 18000.0):
                self._log_enemy_launch_block(
                    agent_id,
                    "under_min_range",
                    current_time,
                    target_id=str(target_uid or ""),
                    distance=distance,
                    heading_error_deg=heading_error_deg,
                    extra=f"window={min_launch_range/1000.0:.0f}-{max_launch_range/1000.0:.0f}km left={missiles_remaining}",
                )
            elif distance > max_launch_range and distance <= (max_launch_range + 12000.0):
                self._log_enemy_launch_block(
                    agent_id,
                    "over_max_range",
                    current_time,
                    target_id=str(target_uid or ""),
                    distance=distance,
                    heading_error_deg=heading_error_deg,
                    extra=f"window={min_launch_range/1000.0:.0f}-{max_launch_range/1000.0:.0f}km",
                )
            return False
        if north_phase_block and distance > 32000.0 and not north_phase_press_shot:
            if distance <= 65000.0:
                self._log_enemy_launch_block(
                    agent_id,
                    "north_phase_hold",
                    current_time,
                    target_id=str(target_uid or ""),
                    distance=distance,
                    heading_error_deg=heading_error_deg,
                    extra=str(north_phase_status.get("reason", "north_phase_hold")),
                )
            return False
        if north_phase_press_shot:
            logging.info(
                "[ENEMY_LAUNCH_OVERRIDE] %s reason=north_phase_press_shot target=%s dist=%.1fkm heading_err=%.1fdeg left=%d",
                agent_id,
                str(target_uid or ""),
                distance / 1000.0,
                heading_error_deg,
                missiles_remaining,
            )
        active_on_target = 0
        if hasattr(env, 'missiles') and env.missiles:
            for missile in env.missiles.values():
                if not getattr(missile, 'is_alive', False):
                    continue
                missile_uid = str(getattr(missile, 'uid', ''))
                if not missile_uid.startswith('B'):
                    continue
                missile_target_id = getattr(missile, 'target_id', None)
                if missile_target_id is None and hasattr(missile, 'target_aircraft'):
                    missile_target_id = getattr(getattr(missile, 'target_aircraft', None), 'uid', None)
                if missile_target_id == target_uid:
                    active_on_target += 1

        close_second_shot_ready = bool(
            distance <= 42000.0
            and heading_error_deg <= 45.0
        )
        second_shot_override = bool(
            finish_attack_window
            or endgame_commit_window
            or close_finish_shot
            or len(alive_targets) <= 1
        )
        if len(alive_targets) > 1 and active_on_target >= 2 and distance > 18000.0:
            if distance <= 65000.0:
                self._log_enemy_launch_block(
                    agent_id,
                    "target_saturation_2",
                    current_time,
                    target_id=str(target_uid or ""),
                    distance=distance,
                )
            return False
        if len(alive_targets) > 1 and active_on_target >= 1 and distance > 70000.0:
            return False
        if (
            len(alive_targets) > 1
            and active_on_target >= 1
            and distance > 42000.0
            and not (close_second_shot_ready or second_shot_override)
        ):
            if distance <= 65000.0:
                self._log_enemy_launch_block(
                    agent_id,
                    "shoot_look_shoot_hold",
                    current_time,
                    target_id=str(target_uid or ""),
                    distance=distance,
                    heading_error_deg=heading_error_deg,
                    extra=f"active_on_target={active_on_target}",
                )
            return False
        if (
            len(alive_targets) > 1
            and missiles_remaining <= 1
            and distance > 35000.0
            and not (finish_attack_window or endgame_commit_window)
        ):
            if distance <= 65000.0:
                self._log_enemy_launch_block(
                    agent_id,
                    "missile_reserve",
                    current_time,
                    target_id=str(target_uid or ""),
                    distance=distance,
                    extra=f"left={missiles_remaining}",
                )
            return False
        if (
            len(alive_targets) > 1
            and team_missiles_remaining <= (len(alive_targets) + 1)
            and distance > 50000.0
            and not (finish_attack_window or endgame_commit_window)
        ):
            if distance <= 65000.0:
                self._log_enemy_launch_block(
                    agent_id,
                    "team_reserve",
                    current_time,
                    target_id=str(target_uid or ""),
                    distance=distance,
                    extra=f"team_left={team_missiles_remaining}",
                )
            return False

        # 威胁评估：在高威胁情况下更积极发射
        threat = self.threat_assessment.get(agent_id)
        if threat and threat.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            # 调试：敌方AI的日志不打印
            # logging.info(f"[T={current_time:.1f}s][敌方导弹] {agent_id} ✅ 高威胁+朝向正确({heading_error_deg:.1f}°)，满足发射条件！")
            return True

        # 正常发射条件：默认最佳窗口 42-68km，支持多次拉开-再进入 BVR 攻击。
        opt_min = float(os.getenv("ENEMY_LAUNCH_OPT_MIN_M", "42000"))
        opt_max = float(os.getenv("ENEMY_LAUNCH_OPT_MAX_M", "72000"))
        close_window_ready = bool(
            min_launch_range <= distance < opt_min
            and heading_error_deg <= 55.0
        )
        if opt_min <= distance <= opt_max or close_window_ready:
            # 🔥 调试：敌方AI的日志不打印
            # last = self._last_enemy_heading_log.get(agent_id, -999)
            # if current_time - last >= 5.0:
            #     logging.info(f"[T={current_time:.1f}s][敌方导弹] {agent_id} ✅ 距离+朝向正确({heading_error_deg:.1f}°)，满足发射条件！")
            #     self._last_enemy_heading_log[agent_id] = current_time
            # 追加 3.7.2 雷达/锁定/Notch/探测概率检查
            if check_missile_launch_conditions is not None:
                try:
                    chk = check_missile_launch_conditions(env, agent_id, target.uid)
                    if not chk.get('can_launch', False):
                        soft_enemy_shot = bool(
                            distance <= 62000.0
                            and heading_error_deg <= 35.0
                        )
                        close_enemy_shot = bool(
                            distance <= 52000.0
                            and heading_error_deg <= 55.0
                        )
                        finish_soft_enemy_shot = bool(
                            finish_attack_window
                            and distance <= 68000.0
                            and heading_error_deg <= 40.0
                            and not north_phase_block
                        )
                        endgame_soft_enemy_shot = bool(
                            endgame_commit_window
                            and distance <= max(min_launch_range + 12000.0, endgame_release_range)
                            and heading_error_deg <= 50.0
                        )
                        if not (soft_enemy_shot or close_enemy_shot or finish_soft_enemy_shot or endgame_soft_enemy_shot):
                            self._log_enemy_launch_block(
                                agent_id,
                                "radar_gate",
                                current_time,
                                target_id=str(target_uid or ""),
                                distance=distance,
                                heading_error_deg=heading_error_deg,
                            )
                            return False
                except Exception:
                    pass
            if endgame_commit_window and current_time - last >= 4.0:
                logging.info(
                    "[ENEMY_LAUNCH_OVERRIDE] %s reason=endgame_commit target=%s dist=%.1fkm heading_err=%.1fdeg left=%d team_left=%d enemy=%dv%d",
                    agent_id,
                    str(target_uid or ""),
                    distance / 1000.0,
                    heading_error_deg,
                    missiles_remaining,
                    team_missiles_remaining,
                    len(enemy_alive),
                    len(alive_targets),
                )
                self._last_enemy_missile_log[agent_id] = current_time
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

            # ✅ 敌方(B*)：导弹对调为 AIM-120（使用引擎内置导弹工厂）
            # 如果后续需要更真实的AIM-120动力学，可再接入 dedicated 模拟器。
            missile = MissileSimulator.create(parent=aircraft, target=target, uid=missile_uid)
            try:
                frozen_target_id = getattr(target, "real_id", None) or getattr(target, "uid", None)
                if frozen_target_id:
                    missile.launch_target_real_id = frozen_target_id
                    missile.launch_target_id = frozen_target_id
                    missile.launch_target_uid = frozen_target_id
                    missile.frozen_target_real_id = frozen_target_id
                    missile.frozen_target_id = frozen_target_id
                    missile.frozen_target_uid = frozen_target_id
            except Exception:
                pass

            # 添加到环境
            env.add_temp_simulator(missile)
            # 记录到环境导弹表，供威胁评估/RWR使用
            if not hasattr(env, 'missiles') or env.missiles is None:
                env.missiles = {}
            env.missiles[missile_uid] = missile
            try:
                owner_task = getattr(env, "task", None)
                target_id = getattr(target, "real_id", None) or getattr(target, "uid", None)
                missile_adapter = getattr(owner_task, "missile_adapter", None) if owner_task is not None else None
                if missile_adapter is not None and target_id and hasattr(missile_adapter, "sync_external_launch_event"):
                    try:
                        shooter_pos = np.asarray(aircraft.get_position(), dtype=float)
                        target_pos = np.asarray(target.get_position(), dtype=float)
                        distance_km = float(np.linalg.norm(target_pos - shooter_pos) / 1000.0)
                    except Exception:
                        distance_km = 0.0
                    missile_adapter.sync_external_launch_event(
                        shooter_id=str(agent_id),
                        target_id=str(target_id),
                        missile_id=str(missile_uid),
                        current_time=float(current_time),
                        distance_km=float(distance_km),
                        missile_model=type(missile).__name__,
                        guidance_mode="active",
                    )
                    if hasattr(missile_adapter, "register_external_missile"):
                        missile_adapter.register_external_missile(
                            missile_id=str(missile_uid),
                            missile_obj=missile,
                            shooter_id=str(agent_id),
                            target_id=str(target_id),
                            env=env,
                        )
            except Exception:
                pass

            # 更新发射时间
            self.last_missile_launch_time[agent_id] = current_time

            # 减少导弹数量
            missiles_left_after = max(0, int(getattr(aircraft, 'num_missiles', 0)) - 1)
            aircraft.num_missiles = missiles_left_after
            if hasattr(aircraft, 'num_left_missiles'):
                aircraft.num_left_missiles = missiles_left_after

            # ✅ 文件复盘：敌方真实“导弹发射事件”（不污染控制台）
            try:
                from utils.trace_logger import trace_event
                tgt_uid = getattr(target, 'uid', None) or getattr(target, 'name', None) or None
                dist_km = None
                try:
                    pa = np.array(aircraft.get_position(), dtype=np.float64)
                    pb = np.array(target.get_position(), dtype=np.float64)
                    dist_km = float(np.linalg.norm(pa - pb) / 1000.0)
                except Exception:
                    dist_km = None
                missile_model = getattr(missile, 'model', None) or type(missile).__name__
                trace_event(
                    事件="导弹发射",
                    env=env,
                    模块="unified_enemy_tactical_ai",
                    类型="ACTION",
                    状态="OK",
                    我机=str(agent_id),
                    敌机=str(tgt_uid) if tgt_uid is not None else None,
                    说明=(
                        f"[敌方导弹发射] [{agent_id}] [{missile_uid}]({missile_model}) -> [{tgt_uid}] 距离{dist_km:.1f}km"
                        if (tgt_uid is not None and dist_km is not None) else
                        f"[敌方导弹发射] [{agent_id}] [{missile_uid}]({missile_model}) -> [{tgt_uid}]"
                    ),
                    数据={
                        "missile_id": str(missile_uid),
                        "missile_model": str(missile_model),
                        "distance_km": float(dist_km) if dist_km is not None else None,
                        "missiles_left": int(getattr(aircraft, 'num_missiles', 0)) if hasattr(aircraft, 'num_missiles') else None,
                    },
                )
            except Exception:
                pass

        except Exception as e:
            logging.error(f"导弹发射失败 {agent_id}: {e}")

    def _get_dynamic_velocity_cmd(self, env, agent_id: str) -> int:
        """
        动态速度管理：根据当前速度选择合适的速度指令
        
        Args:
            env: 环境
            agent_id: 飞机ID
        
        Returns:
            velocity_cmd_id: 速度指令索引
                - 3: 保持当前速度 (delta=0)
                - 4: 轻微加速 (delta=+50m/s)
                - 5: 中等加速 (delta=+100m/s)
        """
        current_velocity = np.linalg.norm(env.agents[agent_id].get_velocity())
        
        # 阈值设计：保持速度在220-280 m/s范围内（降低阈值，减缓接近速度）
        VELOCITY_LOW_THRESHOLD = 220.0   # 低速阈值（从240降低到220）
        VELOCITY_HIGH_THRESHOLD = 280.0  # 高速阈值
        VELOCITY_CRITICAL_LOW = 200.0    # 危险低速阈值（从220降低到200）
        
        if current_velocity < VELOCITY_CRITICAL_LOW:
            # 危险低速：紧急加速
            return 5  # 中等加速 (+100m/s)
        elif current_velocity < VELOCITY_LOW_THRESHOLD:
            # 低速：轻微加速
            return 4  # 轻微加速 (+50m/s)
        elif current_velocity > VELOCITY_HIGH_THRESHOLD:
            # 高速：保持当前速度（让速度自然衰减）
            return 3  # 保持速度 (delta=0)
        else:
            # 正常范围（220-280 m/s）：保持当前速度
            return 3  # 保持速度 (delta=0)

    def _init_action_weights(self):
        """初始化动作权重配置"""
        # 基于战术模式和阶段的动作权重矩阵
        # ✅ 加强敌方机动AI：动作更多样（攻击/防御/规避导弹），但仍避免“短时间大转弯”导致转圈
        self.action_weights = {
            TacticalMode.AGGRESSIVE: {
                EnemyTacticalPhase.NLT_MELD: {
                    ActionType.AGGRESSIVE_APPROACH: 0.5,  # 🔥 任务1：大幅增加主动接敌
                    ActionType.MAINTAIN_HEADING: 0.2,     # 🔥 任务1：减少平飞
                    ActionType.CRANK_LEFT: 0.15,
                    ActionType.CRANK_RIGHT: 0.15
                },
                EnemyTacticalPhase.MELD_MTR: {
                    ActionType.AGGRESSIVE_APPROACH: 0.6,  # 🔥 任务1：进一步增加接敌
                    ActionType.CRANK_LEFT: 0.15,
                    ActionType.CRANK_RIGHT: 0.15,
                    ActionType.MAINTAIN_HEADING: 0.1      # 🔥 任务1：减少平飞
                },
                EnemyTacticalPhase.MTR_TR: {
                    ActionType.AGGRESSIVE_APPROACH: 0.28,
                    ActionType.CRANK_LEFT: 0.18,
                    ActionType.CRANK_RIGHT: 0.18,
                    ActionType.SHORT_SKATE: 0.16,
                    ActionType.BEAM_MANEUVER: 0.10,
                    ActionType.NOTCH_MANEUVER: 0.10
                },
                EnemyTacticalPhase.TR_DOR: {
                    ActionType.SHORT_SKATE: 0.32,
                    ActionType.CRANK_LEFT: 0.16,
                    ActionType.CRANK_RIGHT: 0.16,
                    ActionType.BEAM_MANEUVER: 0.14,
                    ActionType.NOTCH_MANEUVER: 0.10,
                    ActionType.AGGRESSIVE_APPROACH: 0.12
                },
                EnemyTacticalPhase.DOR_DR: {
                    ActionType.SHORT_SKATE: 0.24,
                    ActionType.BEAM_MANEUVER: 0.18,
                    ActionType.NOTCH_MANEUVER: 0.12,
                    ActionType.DIVE_ESCAPE: 0.08,
                    ActionType.AGGRESSIVE_APPROACH: 0.08,
                    ActionType.RETURN_TO_BASE: 0.30
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
                    ActionType.NOTCH_MANEUVER: 0.20,  # 🔧 增强规避：0.08 → 0.20
                    ActionType.BEAM_MANEUVER: 0.15,   # 🔧 增强规避：0.08 → 0.15
                    ActionType.DIVE_ESCAPE: 0.10,     # 🔧 增强规避：0.05 → 0.10
                    ActionType.CHAFF_FLARE_MANEUVER: 0.08,  # 🔧 增强规避：0.04 → 0.08
                    ActionType.TURN_LEFT: 0.235,      # 🔧 平衡调整：0.375 → 0.235
                    ActionType.TURN_RIGHT: 0.235      # 🔧 平衡调整：0.375 → 0.235
                },
                EnemyTacticalPhase.TR_DOR: {
                    ActionType.NOTCH_MANEUVER: 0.15,     # 🔥 任务1：适度规避
                    ActionType.DIVE_ESCAPE: 0.1,         # 🔥 任务1：减少逃避动作
                    ActionType.SPIRAL_DIVE: 0.02,        # 保持低频：螺旋俯冲危险
                    ActionType.DEFENSIVE_SPLIT: 0.2,     # 🔥 任务1：减少分离动作
                    ActionType.AGGRESSIVE_APPROACH: 0.23, # 🔥 任务1：新增主动接敌
                    ActionType.RETURN_TO_BASE: 0.3       # 🔥 任务1：大幅降低返航频率
                },
                EnemyTacticalPhase.DOR_DR: {
                    ActionType.RETURN_TO_BASE: 0.5,      # 🔥 任务1：降低返航频率
                    ActionType.AGGRESSIVE_APPROACH: 0.2, # 🔥 任务1：新增重新接敌
                    ActionType.SHORT_SKATE: 0.15,        # 🔥 任务1：增加机动选择
                    ActionType.SPIRAL_DIVE: 0.05,        # 保持低频
                    ActionType.DIVE_ESCAPE: 0.05,        # 减少逃避
                    ActionType.NOTCH_MANEUVER: 0.05      # 保持低频
                }
            },
            TacticalMode.NEUTRAL: {
                EnemyTacticalPhase.NLT_MELD: {
                    ActionType.AGGRESSIVE_APPROACH: 0.3,  # 🔥 任务1：新增主动接敌
                    ActionType.MAINTAIN_HEADING: 0.3,     # 🔥 任务1：减少平飞
                    ActionType.TURN_LEFT: 0.2,
                    ActionType.TURN_RIGHT: 0.2
                },
                EnemyTacticalPhase.MELD_MTR: {
                    ActionType.AGGRESSIVE_APPROACH: 0.4,  # 🔥 任务1：增加主动接敌
                    ActionType.MAINTAIN_HEADING: 0.2,     # 🔥 任务1：减少平飞
                    ActionType.BEAM_MANEUVER: 0.2,
                    ActionType.CRANK_LEFT: 0.1,
                    ActionType.CRANK_RIGHT: 0.1
                },
                EnemyTacticalPhase.MTR_TR: {
                    ActionType.MAINTAIN_HEADING: 0.3,
                    ActionType.CRANK_LEFT: 0.2,
                    ActionType.CRANK_RIGHT: 0.2,
                    ActionType.SHORT_SKATE: 0.15,
                    ActionType.BEAM_MANEUVER: 0.15
                },
                EnemyTacticalPhase.TR_DOR: {
                    ActionType.SHORT_SKATE: 0.35,           # 🔥 任务1：增加机动性
                    ActionType.BEAM_MANEUVER: 0.25,         # 🔥 任务1：适度规避
                    ActionType.AGGRESSIVE_APPROACH: 0.2,    # 🔥 任务1：新增主动接敌
                    ActionType.RETURN_TO_BASE: 0.2          # 🔥 任务1：大幅降低返航
                },
                EnemyTacticalPhase.DOR_DR: {
                    ActionType.RETURN_TO_BASE: 0.4,         # 🔥 任务1：降低返航频率
                    ActionType.AGGRESSIVE_APPROACH: 0.3,    # 🔥 任务1：重新接敌
                    ActionType.SHORT_SKATE: 0.3             # 🔥 任务1：增加机动选择
                }
            }
        }

    def get_enemy_command(self, env, agent_id: str, current_time: float, task=None) -> Tuple[int, int, int]:
        """获取敌方战术指令 - 统一入口点"""
        try:
            root_trace = os.getenv('CAP_ROOTCAUSE_TRACE', '').strip().lower() in ('1', 'true', 'yes', 'on')
            # 1. 态势感知
            situation = self._analyze_situation(env, agent_id, current_time)
            if not self._enemy_rtb_enabled() and self._enemy_phases.get(agent_id) in ("RETURNING", "SECOND_ATTACK"):
                self._enemy_phases[agent_id] = "ENGAGING"
            # 2. 检查是否处于返航状态
            if agent_id in self._enemy_phases and self._enemy_phases[agent_id] == "RETURNING":
                raw_cmd = self._execute_return_to_base_unified(env, agent_id, current_time, task)
                cmd = self._apply_global_safety_check(env, agent_id, raw_cmd[0], raw_cmd[1], raw_cmd[2], task)
                cmd = self._normalize_command_to_cap_space(cmd)
                if root_trace and hasattr(env, 'agents') and agent_id in env.agents and env.agents[agent_id].is_alive:
                    try:
                        alt_now = float(env.agents[agent_id].get_property_value(c.position_h_sl_m))
                        if alt_now < 5000.0 or (hasattr(env, 'current_step') and env.current_step % 30 == 0):
                            logging.warning(
                                f"🧩 [根因链-绕过安全检查] {agent_id} phase=RETURNING "
                                f"direct_cmd=(alt={raw_cmd[0]},hdg={raw_cmd[1]},vel={raw_cmd[2]}) "
                                f"final_cmd=(alt={cmd[0]},hdg={cmd[1]},vel={cmd[2]}) "
                                f"alt={alt_now:.0f}m step={getattr(env, 'current_step', '?')}"
                            )
                    except Exception:
                        pass
                return cmd
            # ✅ 二次进攻阶段：持续到结束时间（然后自动回到RETURNING）
            if self._enemy_rtb_enabled() and agent_id in self._enemy_phases and self._enemy_phases[agent_id] == "SECOND_ATTACK":
                end_t = self._enemy_second_attack_end_time.get(agent_id, current_time + 1.0)
                if current_time >= end_t:
                    self._enemy_phases[agent_id] = "RETURNING"
                    self._enemy_return_start_time[agent_id] = current_time
                    logging.warning(f"[敌方二次进攻结束] {agent_id} 转入返航")
                    raw_cmd = self._execute_return_to_base_unified(env, agent_id, current_time, task)
                    cmd = self._apply_global_safety_check(env, agent_id, raw_cmd[0], raw_cmd[1], raw_cmd[2], task)
                    cmd = self._normalize_command_to_cap_space(cmd)
                    if root_trace and hasattr(env, 'agents') and agent_id in env.agents and env.agents[agent_id].is_alive:
                        try:
                            alt_now = float(env.agents[agent_id].get_property_value(c.position_h_sl_m))
                            logging.warning(
                                f"🧩 [根因链-二次进攻结束绕过] {agent_id} direct_cmd=(alt={raw_cmd[0]},hdg={raw_cmd[1]},vel={raw_cmd[2]}) "
                                f"final_cmd=(alt={cmd[0]},hdg={cmd[1]},vel={cmd[2]}) "
                                f"alt={alt_now:.0f}m step={getattr(env, 'current_step', '?')}"
                            )
                        except Exception:
                            pass
                    return cmd
                
                # 二次进攻持续期间继续执行常规决策流，导弹发射由统一入口处理。

            # 🔥 关键修复：敌方导弹发射逻辑应在所有非返航阶段运行，
            # 不能仅限于 SECOND_ATTACK，否则常见 START/ENGAGE 阶段会“零发射”。
            if self._enemy_phases.get(agent_id) != "RETURNING":
                self.handle_missile_launch(env, agent_id, current_time)
            # 3. 威胁评估
            threat = self._assess_threat(situation, agent_id, current_time)
            # 4. 更新战术阶段
            self._update_tactical_phase(env, agent_id, situation)
            # 5. 战术模式选择
            tactical_mode = self._select_tactical_mode(agent_id, threat, current_time)
            # 6. 机动决策
            action_type = self._select_action(agent_id, tactical_mode, current_time, env)
            # 7. 动作执行
            alt_cmd, hdg_cmd, vel_cmd = self._execute_action(env, agent_id, action_type, current_time, task)
            alt_cmd, hdg_cmd, vel_cmd = self._normalize_command_to_cap_space((alt_cmd, hdg_cmd, vel_cmd))
            pre_safety_cmd = (int(alt_cmd), int(hdg_cmd), int(vel_cmd))
            # ✅ 全局高度安全检查（最终防线）- 传递task参数
            alt_cmd, hdg_cmd, vel_cmd = self._apply_global_safety_check(
                env, agent_id, alt_cmd, hdg_cmd, vel_cmd, task
            )
            alt_cmd, hdg_cmd, vel_cmd = self._normalize_command_to_cap_space((alt_cmd, hdg_cmd, vel_cmd))
            if root_trace and hasattr(env, 'agents') and agent_id in env.agents and env.agents[agent_id].is_alive:
                try:
                    alt_now = float(env.agents[agent_id].get_property_value(c.position_h_sl_m))
                    if alt_now < 5000.0 or pre_safety_cmd != (int(alt_cmd), int(hdg_cmd), int(vel_cmd)):
                        logging.warning(
                            f"🧩 [根因链-高层决策] {agent_id} action={action_type.value} "
                            f"phase={self.current_phase.get(agent_id, EnemyTacticalPhase.NLT_MELD).value} "
                            f"mode={self.tactical_mode.get(agent_id, TacticalMode.NEUTRAL).value} "
                            f"cmd_pre={pre_safety_cmd} cmd_post=({int(alt_cmd)},{int(hdg_cmd)},{int(vel_cmd)}) "
                            f"alt={alt_now:.0f}m step={getattr(env, 'current_step', '?')}"
                        )
                except Exception:
                    pass
            return alt_cmd, hdg_cmd, vel_cmd

        except Exception as e:
            logging.error(f"敌方{agent_id}战术指令生成失败: {e}")
            return 7, 8, 4  # 默认平稳飞行

    def _normalize_command_to_cap_space(self, cmd: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """将敌方AI输出统一整理到CAP动作空间。"""
        alt_cmd, hdg_cmd, vel_cmd = (int(cmd[0]), int(cmd[1]), int(cmd[2]))
        alt_cmd = int(np.clip(alt_cmd, 0, len(self.norm_delta_altitude) - 1))
        hdg_cmd = int(np.clip(hdg_cmd, 0, len(self.norm_delta_heading) - 1))
        vel_cmd = int(np.clip(vel_cmd, 0, len(self.norm_delta_velocity) - 1))
        return alt_cmd, hdg_cmd, vel_cmd

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
    
    def get_current_intent(self, agent_id: str) -> str:
        """获取当前意图标签 - 基于当前执行的动作"""
        current_action = self.current_action.get(agent_id, ActionType.MAINTAIN_HEADING)
        return self.action_to_intent_mapping.get(current_action, "RECONNAISSANCE")
    
    def get_current_action_type(self, agent_id: str) -> str:
        """获取当前动作类型名称"""
        current_action = self.current_action.get(agent_id, ActionType.MAINTAIN_HEADING)
        return current_action.value

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
        return _rfh._analyze_situation(self, env, agent_id, current_time)

    def _assess_threat(self, situation: SituationData, agent_id: str, current_time: float) -> ThreatAssessment:
        return _rfh._assess_threat(self, situation, agent_id, current_time)

    def _update_tactical_phase(self, env, agent_id: str, situation: SituationData):
        return _rfh._update_tactical_phase(self, env, agent_id, situation)

    def _select_tactical_mode(self, agent_id: str, threat: ThreatAssessment, current_time: float) -> TacticalMode:
        return _rfh._select_tactical_mode(self, agent_id, threat, current_time)

    def _select_action(self, agent_id: str, tactical_mode: TacticalMode, current_time: float, env=None) -> ActionType:
        return _rfh._select_action(self, agent_id, tactical_mode, current_time, env)

    def _apply_threat_response_weights(self, agent_id: str, base_weights: Dict[ActionType, float], env=None) -> Dict[ActionType, float]:
        return _rfh._apply_threat_response_weights(self, agent_id, base_weights, env)

    def _execute_action(self, env, agent_id: str, action_type: ActionType, current_time: float, task=None) -> Tuple[int, int, int]:
        return _rfh._execute_action(self, env, agent_id, action_type, current_time, task)

    def _apply_global_safety_check(self, env, agent_id: str, alt_cmd: int, hdg_cmd: int, vel_cmd: int, task=None) -> Tuple[int, int, int]:
        return _rfh._apply_global_safety_check(self, env, agent_id, alt_cmd, hdg_cmd, vel_cmd, task)

    def _calculate_safe_altitude_change(self, current_altitude: Optional[float], action_name: str) -> float:
        return _emh._calculate_safe_altitude_change(self, current_altitude, action_name)

    def _generate_action_parameters(self, action_type: ActionType, agent_id: str) -> ActionParameters:
        return _emh._generate_action_parameters(self, action_type, agent_id)

    def _execute_maintain_heading(self, env, agent_id: str) -> Tuple[int, int, int]:
        return _emh._execute_maintain_heading(self, env, agent_id)

    def _execute_turn(self, env, agent_id: str, turn_angle: float, turn_rate: float) -> Tuple[int, int, int]:
        return _emh._execute_turn(self, env, agent_id, turn_angle, turn_rate)

    def _execute_crank(self, env, agent_id: str, crank_angle: float) -> Tuple[int, int, int]:
        return _emh._execute_crank(self, env, agent_id, crank_angle)

    def _execute_notch_maneuver(self, env, agent_id: str) -> Tuple[int, int, int]:
        return _emh._execute_notch_maneuver(self, env, agent_id)

    def _execute_beam_maneuver(self, env, agent_id: str) -> Tuple[int, int, int]:
        return _emh._execute_beam_maneuver(self, env, agent_id)

    def _execute_dive_escape(self, env, agent_id: str) -> Tuple[int, int, int]:
        return _emh._execute_dive_escape(self, env, agent_id)

    def _execute_chaff_flare_maneuver(self, env, agent_id: str) -> Tuple[int, int, int]:
        return _emh._execute_chaff_flare_maneuver(self, env, agent_id)

    def _execute_spiral_dive(self, env, agent_id: str) -> Tuple[int, int, int]:
        return _emh._execute_spiral_dive(self, env, agent_id)

    def _execute_short_skate_unified(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        return _emh._execute_short_skate_unified(self, env, agent_id, current_time)

    def _init_short_skate_unified(self, agent_id: str, current_time: float):
        return _emh._init_short_skate_unified(self, agent_id, current_time)

    def _execute_aggressive_approach(self, env, agent_id: str) -> Tuple[int, int, int]:
        return _emh._execute_aggressive_approach(self, env, agent_id)

    def _execute_defensive_split(self, env, agent_id: str) -> Tuple[int, int, int]:
        return _erh._execute_defensive_split(self, env, agent_id)

    def _execute_return_to_base_unified(self, env, agent_id: str, current_time: float, task=None) -> Tuple[int, int, int]:
        return _erh._execute_return_to_base_unified(self, env, agent_id, current_time, task)

    def _init_return_to_base_unified(self, agent_id: str, current_time: float):
        return _erh._init_return_to_base_unified(self, agent_id, current_time)

    def _execute_altitude_change(self, env, agent_id: str, altitude_change: float) -> Tuple[int, int, int]:
        return _erh._execute_altitude_change(self, env, agent_id, altitude_change)

    def _should_return_to_base(self, env, agent_id: str, current_time: float, closest_enemy_distance: float) -> bool:
        return _erh._should_return_to_base(self, env, agent_id, current_time, closest_enemy_distance)

    def _is_mission_complete(self, env, agent_id: str) -> bool:
        return _erh._is_mission_complete(self, env, agent_id)

    # ==================== 辅助函数 ====================

    def _maintain_heading_precise(self, env, agent_id: str, target_heading: float) -> Tuple[int, int, int]:
        return _emh._maintain_heading_precise(self, env, agent_id, target_heading)

    def _maintain_heading_with_speed(self, env, agent_id: str, target_heading: float, speed_cmd: int) -> Tuple[int, int, int]:
        return _emh._maintain_heading_with_speed(self, env, agent_id, target_heading, speed_cmd)

    def _maintain_heading_with_altitude(self, env, agent_id: str, target_heading: float, altitude_cmd: int) -> Tuple[int, int, int]:
        return _emh._maintain_heading_with_altitude(self, env, agent_id, target_heading, altitude_cmd)

    def _maintain_heading_with_altitude_speed(self, env, agent_id: str, target_heading: float,
                                            altitude_cmd: int, speed_cmd: int) -> Tuple[int, int, int]:
        return _emh._maintain_heading_with_altitude_speed(
            self,
            env,
            agent_id,
            target_heading,
            altitude_cmd,
            speed_cmd,
        )

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
        if agent_id in self.action_parameter_type:
            del self.action_parameter_type[agent_id]
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
        if agent_id in self._opening_style:
            del self._opening_style[agent_id]
        if agent_id in self._last_action_history:
            del self._last_action_history[agent_id]
        if agent_id in self._missile_evasion_until:
            del self._missile_evasion_until[agent_id]

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
        self.action_parameter_type[agent_id] = action_type
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
            self.action_parameter_type.clear()
            self.last_mode_switch.clear()
            self.threat_assessment.clear()
            self.situation_data.clear()

            # 清空特殊状态
            if hasattr(self, 'short_skate_states'):
                self.short_skate_states.clear()
            if hasattr(self, 'return_states'):
                self.return_states.clear()
            self._opening_style.clear()
            self._last_action_history.clear()
            self._missile_evasion_until.clear()

            logging.info("[敌方战术AI] 已重置，准备新回合")
        except Exception as e:
            logging.error(f"AI系统重置失败: {e}")

    def get_action_annotation(self, agent_id: str) -> str:
        return _rfh.get_action_annotation(self, agent_id)

    def get_action_annotation_for_csv(self, agent_id: str) -> Dict[str, str]:
        return _rfh.get_action_annotation_for_csv(self, agent_id)

    def get_action_type_for_csv(self, agent_id: str) -> Dict[str, str]:
        return _rfh.get_action_type_for_csv(self, agent_id)

    def _convert_action_type_to_name(self, action_type: ActionType) -> str:
        return _rfh._convert_action_type_to_name(self, action_type)

    def _generate_action_intent(self, action_type: ActionType, tactical_mode: TacticalMode, situation) -> str:
        return _rfh._generate_action_intent(self, action_type, tactical_mode, situation)

    def _get_short_skate_phase_annotation(self, agent_id: str) -> str:
        return _rfh._get_short_skate_phase_annotation(self, agent_id)

    def _convert_altitude_to_index(self, altitude_offset):
        return _rfh._convert_altitude_to_index(self, altitude_offset)

    def _convert_heading_to_index(self, heading_offset):
        return _rfh._convert_heading_to_index(self, heading_offset)

    def _convert_velocity_to_index(self, velocity_offset):
        return _rfh._convert_velocity_to_index(self, velocity_offset)

    def _direct_control_mapping(self, env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id, basic_maneuver_name="default"):
        return _rfh._direct_control_mapping(self, env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id, basic_maneuver_name)

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


