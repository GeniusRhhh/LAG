#!/usr/bin/env python3
"""
敌方战术AI系统
实现基于威胁感知的智能机动逻辑，区别于我方的拖曳射击战术
"""

import numpy as np
import logging
from enum import Enum
from typing import Dict, Tuple, Optional, Any
from envs.JSBSim.core.catalog import JsbsimCatalog as c

class ThreatLevel(Enum):
    """威胁等级"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class EnemyManeuverType(Enum):
    """敌方机动类型"""
    CAP_PATROL = "cap_patrol"           # CAP巡逻
    AGGRESSIVE_APPROACH = "aggressive"   # 攻击性接敌
    DEFENSIVE_TURN = "defensive_turn"    # 防御转弯
    EVASIVE_MANEUVER = "evasive"        # 规避机动
    ATTACK_POSITIONING = "attack_pos"    # 攻击定位
    RETREAT = "retreat"                  # 撤退
    NOTCH_MANEUVER = "notch"            # Notch机动（90度规避）
    SPLIT_S = "split_s"                 # Split-S机动
    BARREL_ROLL = "barrel_roll"         # 桶滚机动

class EnemyTacticalAI:
    """敌方战术AI系统"""
    
    def __init__(self):
        # 机动状态跟踪
        self.maneuver_states = {}  # agent_id -> maneuver_state
        self.threat_history = {}   # agent_id -> threat_history
        self.last_maneuver_time = {}  # agent_id -> last_maneuver_time
        
        # 机动参数
        self.maneuver_params = {
            EnemyManeuverType.CAP_PATROL: {
                "duration": 30.0,
                "turn_rate": 15.0,  # 度/秒
                "altitude_change": 0,
                "speed_change": 0
            },
            EnemyManeuverType.AGGRESSIVE_APPROACH: {
                "duration": 20.0,
                "turn_rate": 25.0,
                "altitude_change": 500,  # 爬升500m
                "speed_change": 50      # 加速50m/s
            },
            EnemyManeuverType.DEFENSIVE_TURN: {
                "duration": 15.0,
                "turn_rate": 35.0,
                "altitude_change": -200,  # 俯冲200m
                "speed_change": 30
            },
            EnemyManeuverType.EVASIVE_MANEUVER: {
                "duration": 12.0,
                "turn_rate": 45.0,
                "altitude_change": -300,
                "speed_change": 40
            },
            EnemyManeuverType.ATTACK_POSITIONING: {
                "duration": 10.0,
                "turn_rate": 20.0,
                "altitude_change": 200,
                "speed_change": 25
            },
            EnemyManeuverType.NOTCH_MANEUVER: {
                "duration": 8.0,
                "turn_rate": 60.0,  # 快速90度转弯
                "altitude_change": 0,
                "speed_change": 20
            },
            EnemyManeuverType.SPLIT_S: {
                "duration": 6.0,
                "turn_rate": 40.0,
                "altitude_change": -500,  # 快速俯冲
                "speed_change": 60
            }
        }
        
        # 威胁评估参数
        self.threat_ranges = {
            "missile_critical": 15000,   # 15km内导弹为严重威胁
            "missile_high": 30000,       # 30km内导弹为高威胁
            "missile_medium": 50000,     # 50km内导弹为中等威胁
            "radar_lock": 45000,         # 45km内雷达锁定
            "enemy_close": 35000,        # 35km内敌机接近
            "enemy_medium": 60000        # 60km内敌机中等威胁
        }
    
    def evaluate_threat_level(self, env, agent_id: str) -> ThreatLevel:
        """评估威胁等级 - 简化版确保稳定性"""
        if agent_id not in env.agents or not env.agents[agent_id].is_alive:
            return ThreatLevel.NONE

        try:
            agent = env.agents[agent_id]
            max_threat = ThreatLevel.NONE

            # 1. 导弹威胁评估
            try:
                # 检查环境中是否有针对该智能体的导弹
                missile_threats = []
                if hasattr(env, 'missiles'):
                    for missile_id, missile in env.missiles.items():
                        if missile.is_alive and hasattr(missile, 'target_agent_id') and missile.target_agent_id == agent_id:
                            missile_threats.append(missile)

                # 评估最近的导弹威胁
                if missile_threats:
                    min_missile_distance = float('inf')
                    max_missile_velocity = 0

                    for missile in missile_threats:
                        try:
                            missile_distance = np.linalg.norm(
                                np.array(missile.get_position()) - np.array(agent.get_position())
                            )
                            missile_velocity = np.linalg.norm(missile.get_velocity())

                            min_missile_distance = min(min_missile_distance, missile_distance)
                            max_missile_velocity = max(max_missile_velocity, missile_velocity)
                        except:
                            continue

                    # 基于距离和速度评估导弹威胁
                    if min_missile_distance < self.threat_ranges["missile_critical"] and max_missile_velocity > 500:
                        if max_threat.value < ThreatLevel.CRITICAL.value:
                            max_threat = ThreatLevel.CRITICAL
                    elif min_missile_distance < self.threat_ranges["missile_high"] and max_missile_velocity > 400:
                        if max_threat.value < ThreatLevel.HIGH.value:
                            max_threat = ThreatLevel.HIGH
                    elif min_missile_distance < self.threat_ranges["missile_medium"]:
                        if max_threat.value < ThreatLevel.MEDIUM.value:
                            max_threat = ThreatLevel.MEDIUM
            except Exception as e:
                logging.debug(f"导弹威胁评估错误: {e}")

            # 2. 雷达锁定威胁评估（简化版）
            try:
                # 简化的雷达威胁评估 - 基于距离推断
                for friendly_id in ["A0100", "A0200"]:
                    if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                        distance = np.linalg.norm(agent.get_position() - env.agents[friendly_id].get_position())
                        if distance < self.threat_ranges["radar_lock"]:
                            if max_threat.value < ThreatLevel.MEDIUM.value:
                                max_threat = ThreatLevel.MEDIUM
            except Exception as e:
                logging.debug(f"Radar threat evaluation error: {e}")

            # 3. 敌机距离威胁评估（增强版）
            min_enemy_distance = float('inf')
            enemy_approach_rate = 0.0
            enemy_heading_threat = False

            for friendly_id in ["A0100", "A0200"]:
                if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                    friendly_agent = env.agents[friendly_id]
                    distance = np.linalg.norm(agent.get_position() - friendly_agent.get_position())
                    min_enemy_distance = min(min_enemy_distance, distance)

                    # 计算敌机接近速度
                    try:
                        friendly_velocity = friendly_agent.get_velocity()
                        agent_velocity = agent.get_velocity()
                        relative_velocity = np.array(friendly_velocity) - np.array(agent_velocity)
                        position_diff = np.array(friendly_agent.get_position()) - np.array(agent.get_position())

                        # 计算径向接近速度
                        if np.linalg.norm(position_diff) > 0:
                            approach_rate = np.dot(relative_velocity, position_diff) / np.linalg.norm(position_diff)
                            enemy_approach_rate = max(enemy_approach_rate, approach_rate)

                        # 检查敌机是否朝向我方
                        try:
                            friendly_heading = np.rad2deg(friendly_agent.get_property_value(c.attitude_psi_rad))
                            bearing_to_us = np.rad2deg(np.arctan2(position_diff[1], position_diff[0]))
                            heading_diff = abs(friendly_heading - bearing_to_us)
                            if heading_diff > 180:
                                heading_diff = 360 - heading_diff
                            if heading_diff < 45:  # 敌机朝向我方
                                enemy_heading_threat = True
                        except:
                            pass
                    except:
                        pass

            # 基于距离、接近速度和航向威胁评估
            if min_enemy_distance < self.threat_ranges["enemy_close"]:
                if enemy_approach_rate > 100 or enemy_heading_threat:  # 快速接近或直接威胁
                    if max_threat.value < ThreatLevel.HIGH.value:
                        max_threat = ThreatLevel.HIGH
                else:
                    if max_threat.value < ThreatLevel.MEDIUM.value:
                        max_threat = ThreatLevel.MEDIUM
            elif min_enemy_distance < self.threat_ranges["enemy_medium"]:
                if enemy_approach_rate > 50 or enemy_heading_threat:
                    if max_threat.value < ThreatLevel.MEDIUM.value:
                        max_threat = ThreatLevel.MEDIUM
                else:
                    if max_threat.value < ThreatLevel.LOW.value:
                        max_threat = ThreatLevel.LOW

            # 4. 战术态势威胁评估（新增）
            tactical_threat = self._evaluate_tactical_situation(env, agent_id)
            if tactical_threat.value > max_threat.value:
                max_threat = tactical_threat

            # 5. 能量状态威胁评估（新增）
            energy_threat = self._evaluate_energy_disadvantage(env, agent_id)
            if energy_threat.value > max_threat.value:
                max_threat = energy_threat

            return max_threat

        except Exception as e:
            logging.error(f"威胁评估错误 {agent_id}: {e}")
            # 返回基于距离的简单威胁评估
            try:
                agent = env.agents[agent_id]
                min_distance = float('inf')
                for friendly_id in ["A0100", "A0200"]:
                    if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                        distance = np.linalg.norm(agent.get_position() - env.agents[friendly_id].get_position())
                        min_distance = min(min_distance, distance)

                if min_distance < 20000:
                    return ThreatLevel.HIGH
                elif min_distance < 40000:
                    return ThreatLevel.MEDIUM
                elif min_distance < 60000:
                    return ThreatLevel.LOW
                else:
                    return ThreatLevel.NONE
            except:
                return ThreatLevel.LOW  # 默认低威胁
    
    def select_maneuver(self, env, agent_id: str, threat_level: ThreatLevel,
                       current_time: float) -> EnemyManeuverType:
        """选择机动类型"""
        agent = env.agents[agent_id]

        # 战场边界检查 - 防止飞机离开战斗区域
        position = agent.get_position()
        x, y = position[0], position[1]

        # 定义战场边界（以公里为单位）
        battlefield_limit = 100000  # 100公里边界

        # 如果接近边界，强制返回战斗位置
        if abs(x) > battlefield_limit or abs(y) > battlefield_limit:
            logging.warning(f"{agent_id} 接近战场边界，强制返回战斗位置")
            return EnemyManeuverType.AGGRESSIVE_APPROACH  # 强制接敌

        # 检查是否在执行机动中
        if agent_id in self.maneuver_states:
            maneuver_state = self.maneuver_states[agent_id]
            if current_time - maneuver_state.get("start_time", 0) < maneuver_state.get("duration", 0):
                return maneuver_state["type"]  # 继续当前机动
        
        # 获取战术态势信息
        try:
            tactical_info = self._analyze_tactical_situation(env, agent_id)
        except Exception as e:
            logging.debug(f"战术态势分析错误: {e}")
            # 使用默认态势信息
            tactical_info = {
                "min_enemy_distance": 50000,
                "outnumbered": False,
                "energy_advantage": False
            }

        # 基于威胁等级和战术态势选择机动
        if threat_level == ThreatLevel.CRITICAL:
            # 严重威胁：智能选择最佳规避机动
            try:
                # 检查导弹威胁
                missile_threats = []
                if hasattr(env, 'missiles'):
                    for missile_id, missile in env.missiles.items():
                        if missile.is_alive and hasattr(missile, 'target_agent_id') and missile.target_agent_id == agent_id:
                            missile_threats.append(missile)

                if missile_threats:
                    min_missile_distance = float('inf')
                    max_missile_velocity = 0

                    for missile in missile_threats:
                        try:
                            missile_distance = np.linalg.norm(
                                np.array(missile.get_position()) - np.array(agent.get_position())
                            )
                            missile_velocity = np.linalg.norm(missile.get_velocity())

                            min_missile_distance = min(min_missile_distance, missile_distance)
                            max_missile_velocity = max(max_missile_velocity, missile_velocity)
                        except:
                            continue

                    # 根据导弹类型和距离选择最佳规避
                    if min_missile_distance < 15000 and max_missile_velocity > 600:
                        return EnemyManeuverType.SPLIT_S  # 近距离高速导弹：急俯冲
                    elif min_missile_distance < 25000:
                        return EnemyManeuverType.BARREL_ROLL  # 中距离：桶滚机动
                    else:
                        return EnemyManeuverType.NOTCH_MANEUVER  # 远距离：Notch机动
            except:
                pass

            # 无导弹威胁但威胁严重：可能是近距离敌机威胁
            if tactical_info["min_enemy_distance"] < 20000:
                return EnemyManeuverType.EVASIVE_MANEUVER  # 高机动规避
            else:
                return EnemyManeuverType.DEFENSIVE_TURN  # 防御转弯

        elif threat_level == ThreatLevel.HIGH:
            # 高威胁：根据战术态势选择防御或反击
            if tactical_info["outnumbered"]:
                # 数量劣势：优先防御
                return EnemyManeuverType.DEFENSIVE_TURN
            elif tactical_info["energy_advantage"]:
                # 能量优势：可以考虑反击
                if self._has_launch_opportunity(env, agent_id):
                    return EnemyManeuverType.ATTACK_POSITIONING
                else:
                    return EnemyManeuverType.AGGRESSIVE_APPROACH
            else:
                return EnemyManeuverType.DEFENSIVE_TURN

        elif threat_level == ThreatLevel.MEDIUM:
            # 中等威胁：平衡攻防，根据机会选择
            if self._has_launch_opportunity(env, agent_id):
                return EnemyManeuverType.ATTACK_POSITIONING
            elif tactical_info["energy_advantage"] and not tactical_info["outnumbered"]:
                return EnemyManeuverType.AGGRESSIVE_APPROACH
            else:
                return EnemyManeuverType.DEFENSIVE_TURN

        elif threat_level == ThreatLevel.LOW:
            # 低威胁：主动攻击
            if tactical_info["energy_advantage"]:
                return EnemyManeuverType.AGGRESSIVE_APPROACH
            elif self._has_launch_opportunity(env, agent_id):
                return EnemyManeuverType.ATTACK_POSITIONING
            else:
                return EnemyManeuverType.AGGRESSIVE_APPROACH

        else:
            # 无威胁：根据战术态势选择巡逻或接敌
            if tactical_info["min_enemy_distance"] < 60000:  # 敌机在探测范围内
                return EnemyManeuverType.AGGRESSIVE_APPROACH  # 主动接敌
            else:
                return EnemyManeuverType.CAP_PATROL  # CAP巡逻
    
    def _has_launch_opportunity(self, env, agent_id: str) -> bool:
        """检查是否有导弹发射机会 - 增强版评估"""
        agent = env.agents[agent_id]

        # 检查导弹数量
        if agent.num_missiles <= 0:
            return False

        # 检查目标距离、角度和相对运动
        for friendly_id in ["A0100", "A0200"]:
            if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                target = env.agents[friendly_id]
                distance = np.linalg.norm(agent.get_position() - target.get_position())

                # 动态距离范围：根据目标速度和航向调整
                min_range = 20000  # 最小发射距离
                max_range = 60000  # 最大发射距离

                try:
                    # 获取目标相对运动信息
                    target_velocity = np.array(target.get_velocity())
                    agent_velocity = np.array(agent.get_velocity())
                    relative_velocity = target_velocity - agent_velocity

                    # 计算目标接近/远离速度
                    position_diff = np.array(target.get_position()) - np.array(agent.get_position())
                    if np.linalg.norm(position_diff) > 0:
                        closing_rate = -np.dot(relative_velocity, position_diff) / np.linalg.norm(position_diff)

                        # 如果目标正在接近，可以在更远距离发射
                        if closing_rate > 50:  # 目标接近速度 > 50m/s
                            max_range = 70000
                        elif closing_rate < -50:  # 目标远离速度 > 50m/s
                            max_range = 45000
                except:
                    pass

                if min_range <= distance <= max_range:
                    # 检查攻击角度
                    try:
                        relative_pos = target.get_position() - agent.get_position()
                        agent_heading = agent.get_property_value(c.attitude_psi_rad)
                        target_bearing = np.arctan2(relative_pos[1], relative_pos[0])
                        angle_diff = abs(agent_heading - target_bearing)
                        if angle_diff > np.pi:
                            angle_diff = 2 * np.pi - angle_diff

                        # 动态角度阈值：近距离要求更精确
                        angle_threshold = np.pi/3  # 60度
                        if distance < 30000:
                            angle_threshold = np.pi/4  # 45度
                        elif distance < 40000:
                            angle_threshold = np.pi/3  # 60度
                        else:
                            angle_threshold = np.pi/2.5  # 72度

                        if angle_diff < angle_threshold:
                            # 额外检查：目标不应该处于高机动状态
                            try:
                                target_g_force = abs(target.get_property_value(c.accelerations_n_pilot_z_norm))
                                if target_g_force < 3.0:  # 目标不在高G机动中
                                    return True
                            except:
                                return True  # 无法获取G力信息时默认可以发射
                    except:
                        pass

        return False
    
    def execute_maneuver(self, env, agent_id: str, maneuver_type: EnemyManeuverType, 
                        current_time: float) -> Tuple[int, int, int]:
        """执行机动并返回指令索引"""
        # 初始化或更新机动状态
        if (agent_id not in self.maneuver_states or 
            self.maneuver_states[agent_id]["type"] != maneuver_type):
            
            self.maneuver_states[agent_id] = {
                "type": maneuver_type,
                "start_time": current_time,
                "duration": self.maneuver_params[maneuver_type]["duration"],
                "phase": 0  # 机动阶段
            }
        
        maneuver_state = self.maneuver_states[agent_id]
        elapsed_time = current_time - maneuver_state["start_time"]
        progress = elapsed_time / maneuver_state["duration"]
        
        # 根据机动类型生成指令
        return self._generate_maneuver_commands(env, agent_id, maneuver_type, progress)
    
    def _generate_maneuver_commands(self, env, agent_id: str, maneuver_type: EnemyManeuverType,
                                   progress: float) -> Tuple[int, int, int]:
        """生成具体的机动指令 - 增强版战术机动"""
        agent = env.agents[agent_id]
        try:
            current_heading = np.rad2deg(agent.get_property_value(c.attitude_psi_rad))
            current_altitude = agent.get_property_value(c.position_h_sl_m)
            current_velocity = agent.get_property_value(c.velocities_u_mps)
        except:
            # 如果获取属性失败，使用默认值
            current_heading = 180.0  # 敌方默认朝南
            current_altitude = 6000.0
            current_velocity = 250.0

        # 获取敌机位置和我方位置，用于智能机动决策
        enemy_pos = agent.get_position()
        friendly_positions = []
        for friendly_id in ["A0100", "A0200"]:
            if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                friendly_positions.append(env.agents[friendly_id].get_position())

        # 计算到最近我方飞机的距离和方位
        min_distance = float('inf')
        target_bearing = 0.0
        if friendly_positions:
            for friendly_pos in friendly_positions:
                distance = np.linalg.norm(np.array(friendly_pos) - np.array(enemy_pos))
                if distance < min_distance:
                    min_distance = distance
                    # 计算方位角
                    dx = friendly_pos[0] - enemy_pos[0]
                    dy = friendly_pos[1] - enemy_pos[1]
                    target_bearing = np.rad2deg(np.arctan2(dy, dx))

        if maneuver_type == EnemyManeuverType.CAP_PATROL:
            # CAP巡逻：智能巡逻模式，保持战斗准备
            if progress < 0.4:
                return 7, 8, 3  # 直飞保持警戒
            elif progress < 0.7:
                # 根据距离调整巡逻模式
                if min_distance > 50000:  # 远距离：保持巡逻
                    return 7, 6, 3  # 轻微左转
                else:  # 中近距离：提高警戒
                    return 8, 6, 4  # 轻微爬升，左转，加速
            else:
                return 7, 10, 3  # 轻微右转

        elif maneuver_type == EnemyManeuverType.AGGRESSIVE_APPROACH:
            # 攻击性接敌：智能接敌，根据距离和威胁调整
            if min_distance > 40000:  # 远距离接敌
                if progress < 0.5:
                    return 9, self._calculate_intercept_heading(current_heading, target_bearing), 5  # 爬升，转向目标，大幅加速
                else:
                    return 8, self._calculate_intercept_heading(current_heading, target_bearing), 4  # 轻微爬升，转向目标，加速
            else:  # 中近距离接敌
                if progress < 0.3:
                    return 8, self._calculate_intercept_heading(current_heading, target_bearing), 5  # 轻微爬升，转向目标，大幅加速
                elif progress < 0.7:
                    return 7, self._calculate_intercept_heading(current_heading, target_bearing), 4  # 保持高度，转向目标，加速
                else:
                    return 7, 8, 3  # 保持当前状态，准备攻击

        elif maneuver_type == EnemyManeuverType.DEFENSIVE_TURN:
            # 防御转弯：智能防御，根据威胁方向选择最佳规避方向
            threat_direction = self._assess_threat_direction(env, agent_id)
            if progress < 0.4:
                if threat_direction == "left":
                    return 5, 12, 5  # 俯冲，大幅右转，大幅加速
                else:
                    return 5, 4, 5  # 俯冲，大幅左转，大幅加速
            elif progress < 0.8:
                # 继续规避并准备反击
                if threat_direction == "left":
                    return 6, 10, 4  # 轻微爬升，右转，加速
                else:
                    return 6, 6, 4  # 轻微爬升，左转，加速
            else:
                return 7, 8, 3  # 稳定飞行，评估态势

        elif maneuver_type == EnemyManeuverType.EVASIVE_MANEUVER:
            # 规避机动：高机动性S型机动，增加不可预测性
            if progress < 0.2:
                return 3, 2, 6  # 大幅俯冲，急左转，最大加速
            elif progress < 0.4:
                return 9, 8, 5  # 大幅爬升，直飞，大幅加速
            elif progress < 0.6:
                return 3, 14, 6  # 大幅俯冲，急右转，最大加速
            elif progress < 0.8:
                return 9, 8, 5  # 大幅爬升，直飞，大幅加速
            else:
                return 7, 8, 3  # 保持高度，直飞，正常速度

        elif maneuver_type == EnemyManeuverType.NOTCH_MANEUVER:
            # Notch机动：90度转弯规避雷达，智能选择规避方向
            if progress < 0.5:
                # 前半段：选择最佳90度规避方向
                if target_bearing > current_heading:
                    return 7, 4, 5  # 保持高度，90度左转，大幅加速
                else:
                    return 7, 12, 5  # 保持高度，90度右转，大幅加速
            else:
                # 后半段：重新定向敌机，防止离开战场
                return 7, self._calculate_return_heading(current_heading, target_bearing), 4

        elif maneuver_type == EnemyManeuverType.SPLIT_S:
            # Split-S机动：快速俯冲转弯，增强机动性
            if progress < 0.25:
                return 1, 8, 6  # 急俯冲，直飞，最大加速
            elif progress < 0.5:
                return 2, 4, 6  # 俯冲，左转，最大加速
            elif progress < 0.75:
                return 3, 12, 5  # 俯冲，右转，大幅加速
            else:
                return 8, 8, 4  # 拉起，直飞，加速

        elif maneuver_type == EnemyManeuverType.ATTACK_POSITIONING:
            # 攻击定位：智能定位到最佳攻击位置
            optimal_heading = self._calculate_attack_heading(current_heading, target_bearing, min_distance)
            if progress < 0.3:
                return 8, optimal_heading, 4  # 轻微爬升，转向最佳位置，加速
            elif progress < 0.7:
                return 7, optimal_heading, 3  # 保持高度，微调位置，正常速度
            else:
                return 7, 8, 3  # 保持当前状态，准备攻击

        elif maneuver_type == EnemyManeuverType.BARREL_ROLL:
            # 桶滚机动：复杂的三维机动
            if progress < 0.25:
                return 8, 6, 4  # 爬升，左转，加速
            elif progress < 0.5:
                return 5, 10, 4  # 俯冲，右转，加速
            elif progress < 0.75:
                return 8, 6, 4  # 爬升，左转，加速
            else:
                return 7, 8, 3  # 恢复水平飞行

        # 默认：智能平稳飞行，保持对敌方的警戒
        if min_distance < 30000:  # 近距离保持警戒
            return 7, self._calculate_intercept_heading(current_heading, target_bearing), 4
        else:  # 远距离正常巡逻
            return 7, 8, 3
    
    def _calculate_intercept_heading(self, current_heading: float, target_bearing: float) -> int:
        """计算拦截航向的指令索引"""
        # 计算需要转向的角度
        heading_diff = target_bearing - current_heading

        # 规范化角度差到[-180, 180]
        while heading_diff > 180:
            heading_diff -= 360
        while heading_diff < -180:
            heading_diff += 360

        # 根据角度差选择合适的航向指令索引
        # 航向指令索引：0-16，对应-180°到+180°
        if abs(heading_diff) < 5:
            return 8  # 直飞
        elif heading_diff > 0:  # 需要右转
            if heading_diff > 90:
                return 16  # 大幅右转
            elif heading_diff > 45:
                return 14  # 中等右转
            elif heading_diff > 15:
                return 12  # 小幅右转
            else:
                return 10  # 轻微右转
        else:  # 需要左转
            if heading_diff < -90:
                return 0  # 大幅左转
            elif heading_diff < -45:
                return 2  # 中等左转
            elif heading_diff < -15:
                return 4  # 小幅左转
            else:
                return 6  # 轻微左转

    def _calculate_return_heading(self, current_heading: float, target_bearing: float) -> int:
        """计算返回战场的航向指令索引"""
        # 计算返回战场中心的方向
        return_bearing = target_bearing + 180  # 朝向战场中心
        if return_bearing >= 360:
            return_bearing -= 360

        return self._calculate_intercept_heading(current_heading, return_bearing)

    def _calculate_attack_heading(self, current_heading: float, target_bearing: float, distance: float) -> int:
        """计算最佳攻击航向的指令索引"""
        # 根据距离调整攻击角度
        if distance > 40000:  # 远距离：直接接敌
            return self._calculate_intercept_heading(current_heading, target_bearing)
        elif distance > 20000:  # 中距离：侧向接敌
            attack_bearing = target_bearing + 30  # 30度侧向接敌
            if attack_bearing >= 360:
                attack_bearing -= 360
            return self._calculate_intercept_heading(current_heading, attack_bearing)
        else:  # 近距离：机动接敌
            attack_bearing = target_bearing + 45  # 45度机动接敌
            if attack_bearing >= 360:
                attack_bearing -= 360
            return self._calculate_intercept_heading(current_heading, attack_bearing)

    def _assess_threat_direction(self, env, agent_id: str) -> str:
        """评估威胁方向"""
        agent = env.agents[agent_id]
        agent_pos = np.array(agent.get_position())

        # 检查导弹威胁方向
        for missile_id, missile in env.missiles.items():
            if missile.is_alive and missile.target_agent_id == agent_id:
                missile_pos = np.array(missile.get_position())
                relative_pos = missile_pos - agent_pos

                # 简化的威胁方向判断
                if relative_pos[0] > 0:  # 导弹在右侧
                    return "right"
                else:  # 导弹在左侧
                    return "left"

        # 检查敌机威胁方向
        for friendly_id in ["A0100", "A0200"]:
            if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                friendly_pos = np.array(env.agents[friendly_id].get_position())
                relative_pos = friendly_pos - agent_pos

                if relative_pos[0] > 0:  # 敌机在右侧
                    return "right"
                else:  # 敌机在左侧
                    return "left"

        return "center"  # 默认中央威胁

    def _evaluate_tactical_situation(self, env, agent_id: str) -> ThreatLevel:
        """评估战术态势威胁"""
        agent = env.agents[agent_id]

        # 检查是否处于数量劣势
        friendly_count = 0
        enemy_count = 0

        for aid in env.agents:
            if env.agents[aid].is_alive:
                if aid.startswith("B"):  # 敌方（我方）
                    friendly_count += 1
                elif aid.startswith("A"):  # 我方（敌方）
                    enemy_count += 1

        # 数量劣势威胁
        if friendly_count < enemy_count:
            if friendly_count == 1 and enemy_count == 2:
                return ThreatLevel.HIGH  # 1v2严重劣势
            else:
                return ThreatLevel.MEDIUM  # 一般数量劣势

        # 检查是否被包围
        agent_pos = np.array(agent.get_position())
        enemy_positions = []
        for friendly_id in ["A0100", "A0200"]:
            if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                enemy_positions.append(np.array(env.agents[friendly_id].get_position()))

        if len(enemy_positions) >= 2:
            # 计算是否被包围（敌机在不同方向）
            bearings = []
            for enemy_pos in enemy_positions:
                diff = enemy_pos - agent_pos
                bearing = np.rad2deg(np.arctan2(diff[1], diff[0]))
                bearings.append(bearing)

            if len(bearings) >= 2:
                bearing_diff = abs(bearings[0] - bearings[1])
                if bearing_diff > 180:
                    bearing_diff = 360 - bearing_diff
                if bearing_diff > 90:  # 敌机分布在不同象限
                    return ThreatLevel.MEDIUM

        return ThreatLevel.NONE

    def _evaluate_energy_disadvantage(self, env, agent_id: str) -> ThreatLevel:
        """评估能量劣势威胁"""
        agent = env.agents[agent_id]

        try:
            # 获取当前能量状态
            current_altitude = agent.get_property_value(c.position_h_sl_m)
            current_velocity = agent.get_property_value(c.velocities_u_mps)

            # 计算能量（简化：动能+势能）
            kinetic_energy = 0.5 * current_velocity ** 2
            potential_energy = 9.81 * current_altitude
            total_energy = kinetic_energy + potential_energy

            # 与敌机比较能量状态
            enemy_energies = []
            for friendly_id in ["A0100", "A0200"]:
                if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                    try:
                        enemy_alt = env.agents[friendly_id].get_property_value(c.position_h_sl_m)
                        enemy_vel = env.agents[friendly_id].get_property_value(c.velocities_u_mps)
                        enemy_ke = 0.5 * enemy_vel ** 2
                        enemy_pe = 9.81 * enemy_alt
                        enemy_total = enemy_ke + enemy_pe
                        enemy_energies.append(enemy_total)
                    except:
                        pass

            if enemy_energies:
                max_enemy_energy = max(enemy_energies)
                energy_ratio = total_energy / max_enemy_energy

                if energy_ratio < 0.7:  # 能量严重劣势
                    return ThreatLevel.MEDIUM
                elif energy_ratio < 0.85:  # 能量轻微劣势
                    return ThreatLevel.LOW

        except:
            pass

        return ThreatLevel.NONE

    def _analyze_tactical_situation(self, env, agent_id: str) -> Dict[str, Any]:
        """分析当前战术态势"""
        agent = env.agents[agent_id]
        agent_pos = np.array(agent.get_position())

        # 初始化态势信息
        tactical_info = {
            "min_enemy_distance": float('inf'),
            "enemy_count": 0,
            "friendly_count": 0,
            "outnumbered": False,
            "energy_advantage": False,
            "surrounded": False,
            "missile_threat_count": 0
        }

        # 统计敌我双方数量
        for aid in env.agents:
            if env.agents[aid].is_alive:
                if aid.startswith("B"):  # 我方（敌方视角）
                    tactical_info["friendly_count"] += 1
                elif aid.startswith("A"):  # 敌方（敌方视角）
                    tactical_info["enemy_count"] += 1

        tactical_info["outnumbered"] = tactical_info["friendly_count"] < tactical_info["enemy_count"]

        # 分析敌机威胁
        enemy_positions = []
        enemy_energies = []

        for friendly_id in ["A0100", "A0200"]:
            if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                enemy_agent = env.agents[friendly_id]
                enemy_pos = np.array(enemy_agent.get_position())
                distance = np.linalg.norm(agent_pos - enemy_pos)
                tactical_info["min_enemy_distance"] = min(tactical_info["min_enemy_distance"], distance)
                enemy_positions.append(enemy_pos)

                # 计算敌机能量
                try:
                    enemy_alt = enemy_agent.get_property_value(c.position_h_sl_m)
                    enemy_vel = enemy_agent.get_property_value(c.velocities_u_mps)
                    enemy_energy = 0.5 * enemy_vel ** 2 + 9.81 * enemy_alt
                    enemy_energies.append(enemy_energy)
                except:
                    pass

        # 计算自身能量
        try:
            my_alt = agent.get_property_value(c.position_h_sl_m)
            my_vel = agent.get_property_value(c.velocities_u_mps)
            my_energy = 0.5 * my_vel ** 2 + 9.81 * my_alt

            if enemy_energies:
                max_enemy_energy = max(enemy_energies)
                tactical_info["energy_advantage"] = my_energy > max_enemy_energy * 1.1  # 10%优势
        except:
            pass

        # 检查是否被包围
        if len(enemy_positions) >= 2:
            bearings = []
            for enemy_pos in enemy_positions:
                diff = enemy_pos - agent_pos
                bearing = np.rad2deg(np.arctan2(diff[1], diff[0]))
                bearings.append(bearing)

            if len(bearings) >= 2:
                max_bearing_diff = 0
                for i in range(len(bearings)):
                    for j in range(i+1, len(bearings)):
                        diff = abs(bearings[i] - bearings[j])
                        if diff > 180:
                            diff = 360 - diff
                        max_bearing_diff = max(max_bearing_diff, diff)

                tactical_info["surrounded"] = max_bearing_diff > 120  # 敌机分布超过120度

        # 统计导弹威胁
        for missile_id, missile in env.missiles.items():
            if missile.is_alive and missile.target_agent_id == agent_id:
                tactical_info["missile_threat_count"] += 1

        return tactical_info

# 全局AI实例
enemy_ai = EnemyTacticalAI()

def get_enemy_tactical_command(env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
    """获取敌方战术指令 - 基于真实BVR作战原则的智能AI系统"""
    try:
        # 战斗状态日志
        if current_time % 8.0 < 0.2:  # 每8秒记录一次
            logging.info(f"🎯 {agent_id} BVR战术AI被调用 (时间: {current_time:.1f}s)")

        agent = env.agents[agent_id]

        # 1. BVR战场态势感知
        bvr_situation = _analyze_bvr_situation(env, agent_id, current_time)

        # 2. 长机-僚机角色确定
        formation_role = _determine_formation_role(agent_id, bvr_situation)

        # 3. BVR交战阶段判断
        engagement_phase = _determine_bvr_phase(bvr_situation, formation_role)

        # 4. 脱离接触决策
        should_disengage = _evaluate_disengagement_criteria(bvr_situation, engagement_phase, current_time)

        # 5. 战术行为选择
        if should_disengage:
            tactical_behavior = _select_disengagement_behavior(bvr_situation, formation_role)
        else:
            tactical_behavior = _select_bvr_behavior(engagement_phase, bvr_situation, formation_role)

        # 6. 生成BVR战术指令
        commands = _generate_bvr_commands(tactical_behavior, bvr_situation, formation_role, agent_id)

        # 7. 记录详细战术信息
        if current_time % 6.0 < 0.2:  # 每6秒记录一次
            primary_target = bvr_situation.get('primary_target')
            target_distance = primary_target['distance']/1000 if primary_target else 0
            missile_threat = bvr_situation['immediate_missile_threat']
            logging.info(f"🎯 {agent_id}: 角色={formation_role}, 阶段={engagement_phase}, 行为={tactical_behavior}, 距离={target_distance:.1f}km, 导弹威胁={missile_threat}, 脱离={should_disengage}, 指令={commands}")

        return commands


    except Exception as e:
        logging.error(f"❌ Enemy AI error for {agent_id}: {e}")
        # 返回安全的默认指令
        return 7, 8, 3  # 默认平稳飞行


def _analyze_battlefield_situation(env, agent_id: str, current_time: float) -> Dict[str, Any]:
    """全面战场态势感知 - 强对抗性AI的核心感知系统"""
    agent = env.agents[agent_id]
    agent_pos = np.array(agent.get_position())

    try:
        # 获取飞机状态
        my_altitude = agent.get_property_value(c.position_h_sl_m)
        my_velocity = agent.get_velocity() if hasattr(agent, 'get_velocity') else np.array([0, 0, 0])
        my_speed = np.linalg.norm(my_velocity)
        my_heading = agent.get_property_value(c.attitude_heading_true_rad) * 180 / np.pi
        missile_count = agent.num_missiles if hasattr(agent, 'num_missiles') else 0
    except:
        my_altitude = 10000
        my_speed = 300
        my_heading = 0
        missile_count = 0

    # 能量状态评估（高度+速度）
    energy_state = _calculate_energy_state(my_altitude, my_speed)

    # 友机状态分析
    friendly_aircraft = []
    for friendly_id in ["B0100", "B0200"]:
        if friendly_id != agent_id and friendly_id in env.agents and env.agents[friendly_id].is_alive:
            friendly = env.agents[friendly_id]
            friendly_pos = np.array(friendly.get_position())
            friendly_distance = np.linalg.norm(agent_pos - friendly_pos)
            try:
                friendly_altitude = friendly.get_property_value(c.position_h_sl_m)
                friendly_speed = np.linalg.norm(friendly.get_velocity()) if hasattr(friendly, 'get_velocity') else 300
                friendly_missiles = friendly.num_missiles if hasattr(friendly, 'num_missiles') else 0
            except:
                friendly_altitude = 10000
                friendly_speed = 300
                friendly_missiles = 0

            friendly_aircraft.append({
                'id': friendly_id,
                'agent': friendly,
                'distance': friendly_distance,
                'altitude': friendly_altitude,
                'speed': friendly_speed,
                'missiles': friendly_missiles,
                'position': friendly_pos
            })

    # 敌机目标分析
    enemy_targets = []
    for enemy_id in ["A0100", "A0200"]:
        if enemy_id in env.agents and env.agents[enemy_id].is_alive:
            enemy = env.agents[enemy_id]
            enemy_pos = np.array(enemy.get_position())
            distance = np.linalg.norm(agent_pos - enemy_pos)

            try:
                enemy_altitude = enemy.get_property_value(c.position_h_sl_m)
                enemy_velocity = enemy.get_velocity() if hasattr(enemy, 'get_velocity') else np.array([0, 0, 0])
                enemy_speed = np.linalg.norm(enemy_velocity)
                enemy_heading = enemy.get_property_value(c.attitude_heading_true_rad) * 180 / np.pi
                enemy_missiles = enemy.num_missiles if hasattr(enemy, 'num_missiles') else 0
            except:
                enemy_altitude = 7000
                enemy_velocity = np.array([0, 0, 0])  # 修复：添加默认速度向量
                enemy_speed = 250
                enemy_heading = 180
                enemy_missiles = 0

            # 计算相对位置优势
            altitude_advantage = my_altitude - enemy_altitude
            speed_advantage = my_speed - enemy_speed

            # 计算角度关系
            relative_bearing = _calculate_relative_bearing(agent_pos, enemy_pos, my_heading)
            aspect_angle = _calculate_aspect_angle(agent_pos, enemy_pos, enemy_velocity)

            enemy_targets.append({
                'id': enemy_id,
                'agent': enemy,
                'distance': distance,
                'altitude': enemy_altitude,
                'speed': enemy_speed,
                'heading': enemy_heading,
                'missiles': enemy_missiles,
                'position': enemy_pos,
                'velocity': enemy_velocity,
                'altitude_advantage': altitude_advantage,
                'speed_advantage': speed_advantage,
                'relative_bearing': relative_bearing,
                'aspect_angle': aspect_angle,
                'threat_level': _calculate_enemy_threat_level(distance, enemy_missiles, altitude_advantage)
            })

    # 导弹威胁分析
    missile_threats = _analyze_missile_threats(env, agent_id, agent_pos)

    return {
        'agent_id': agent_id,
        'current_time': current_time,
        'my_position': agent_pos,
        'my_altitude': my_altitude,
        'my_speed': my_speed,
        'my_heading': my_heading,
        'missile_count': missile_count,
        'energy_state': energy_state,
        'friendly_aircraft': friendly_aircraft,
        'enemy_targets': enemy_targets,
        'missile_threats': missile_threats,
        'battlefield_control': _assess_battlefield_control(enemy_targets, friendly_aircraft)
    }


def _analyze_comprehensive_tactical_situation(env, agent_id: str, current_time: float) -> Dict[str, Any]:
    """全面分析战术态势 - 包括威胁评估、机会分析和协调需求"""
    agent = env.agents[agent_id]
    agent_pos = np.array(agent.get_position())

    # 1. 目标分析和优先级排序
    targets = []
    for friendly_id in ["A0100", "A0200"]:
        if friendly_id in env.agents and env.agents[friendly_id].is_alive:
            target = env.agents[friendly_id]
            target_pos = np.array(target.get_position())
            distance = np.linalg.norm(agent_pos - target_pos)

            # 计算目标威胁度和脆弱性
            threat_score = _calculate_target_threat_score(env, target, distance)
            vulnerability = _calculate_target_vulnerability(env, target, distance)

            targets.append({
                'agent': target,
                'distance': distance,
                'threat_score': threat_score,
                'vulnerability': vulnerability,
                'priority': threat_score + vulnerability
            })

    # 按优先级排序目标
    targets.sort(key=lambda x: x['priority'], reverse=True)
    primary_target = targets[0] if targets else None

    # 2. 导弹威胁分析 - 关键改进
    missile_threats = []
    immediate_threat = False
    threat_urgency = 0
    closest_missile_distance = float('inf')

    if hasattr(env, 'missiles'):
        for missile_id, missile in env.missiles.items():
            if missile.is_alive and hasattr(missile, 'target_agent_id') and missile.target_agent_id == agent_id:
                # 计算导弹距离和威胁紧急度
                missile_pos = np.array(missile.get_position()) if hasattr(missile, 'get_position') else None
                if missile_pos is not None:
                    missile_distance = np.linalg.norm(agent_pos - missile_pos)
                    closest_missile_distance = min(closest_missile_distance, missile_distance)

                    # 威胁紧急度评估
                    if missile_distance < 5000:  # 5km内极度危险
                        threat_urgency = max(threat_urgency, 5)
                        immediate_threat = True
                    elif missile_distance < 10000:  # 10km内高度危险
                        threat_urgency = max(threat_urgency, 4)
                        immediate_threat = True
                    elif missile_distance < 20000:  # 20km内中度威胁
                        threat_urgency = max(threat_urgency, 3)

                missile_threats.append({
                    'missile': missile,
                    'distance': missile_distance if missile_pos is not None else float('inf'),
                    'urgency': threat_urgency
                })

    # 3. 友军协调分析
    friendly_aircraft = []
    for friendly_id in ["B0100", "B0200"]:
        if friendly_id != agent_id and friendly_id in env.agents and env.agents[friendly_id].is_alive:
            friendly = env.agents[friendly_id]
            friendly_pos = np.array(friendly.get_position())
            friendly_distance = np.linalg.norm(agent_pos - friendly_pos)
            friendly_aircraft.append({
                'agent': friendly,
                'distance': friendly_distance
            })

    # 4. 战术环境评估
    try:
        my_altitude = agent.get_property_value(c.position_h_sl_m)
        my_speed = np.linalg.norm(agent.get_velocity()) if hasattr(agent, 'get_velocity') else 0
        missile_count = agent.num_missiles if hasattr(agent, 'num_missiles') else 0
    except:
        my_altitude = 0
        my_speed = 0
        missile_count = 0

    # 5. 战术优势评估
    altitude_advantage = False
    speed_advantage = False
    if primary_target:
        try:
            target_altitude = primary_target['agent'].get_property_value(c.position_h_sl_m)
            target_speed = np.linalg.norm(primary_target['agent'].get_velocity()) if hasattr(primary_target['agent'], 'get_velocity') else 0
            altitude_advantage = my_altitude > target_altitude + 1000  # 1km优势
            speed_advantage = my_speed > target_speed + 50  # 50m/s优势
        except:
            pass

    return {
        'primary_target': primary_target,
        'all_targets': targets,
        'target_distance': primary_target['distance'] if primary_target else float('inf'),
        'missile_threats': missile_threats,
        'immediate_missile_threat': immediate_threat,
        'threat_urgency': threat_urgency,
        'closest_missile_distance': closest_missile_distance,
        'friendly_aircraft': friendly_aircraft,
        'missile_count': missile_count,
        'my_altitude': my_altitude,
        'my_speed': my_speed,
        'altitude_advantage': altitude_advantage,
        'speed_advantage': speed_advantage,
        'can_attack': missile_count > 0,
        'engagement_phase': _determine_engagement_phase(current_time, primary_target['distance'] if primary_target else float('inf'))
    }


def _calculate_target_vulnerability(env, target, distance: float) -> int:
    """计算目标脆弱性分数"""
    vulnerability = 0

    # 距离因子 - 距离越近威胁越大
    if distance < 25000:  # 25km内
        vulnerability += 4
    elif distance < 40000:  # 40km内
        vulnerability += 3
    elif distance < 60000:  # 60km内
        vulnerability += 2
    else:
        vulnerability += 1

    # 目标导弹数量 - 导弹越少越脆弱
    try:
        target_missiles = target.num_missiles if hasattr(target, 'num_missiles') else 0
        if target_missiles == 0:
            vulnerability += 3
        elif target_missiles == 1:
            vulnerability += 2
    except:
        pass

    # 目标是否正在被攻击
    if hasattr(env, 'missiles'):
        for missile_id, missile in env.missiles.items():
            if missile.is_alive and hasattr(missile, 'target_agent_id') and missile.target_agent_id == target.agent_id:
                vulnerability += 2
                break

    return vulnerability


def _calculate_energy_state(altitude: float, speed: float) -> str:
    """计算能量状态"""
    # 能量 = 动能 + 势能 (简化计算)
    kinetic_energy = 0.5 * speed * speed / 1000  # 简化
    potential_energy = altitude / 1000  # 简化
    total_energy = kinetic_energy + potential_energy

    if total_energy > 25:
        return "高能量"
    elif total_energy > 15:
        return "中等能量"
    else:
        return "低能量"


def _calculate_relative_bearing(my_pos, target_pos, my_heading):
    """计算相对方位角"""
    dx = target_pos[0] - my_pos[0]
    dy = target_pos[1] - my_pos[1]
    target_bearing = np.arctan2(dy, dx) * 180 / np.pi
    relative_bearing = target_bearing - my_heading

    # 标准化到-180到180度
    while relative_bearing > 180:
        relative_bearing -= 360
    while relative_bearing < -180:
        relative_bearing += 360

    return relative_bearing


def _calculate_aspect_angle(my_pos, target_pos, target_velocity):
    """计算目标纵横比角度"""
    if np.linalg.norm(target_velocity) < 1:
        return 0

    # 从我到目标的向量
    to_target = target_pos - my_pos
    to_target_norm = to_target / np.linalg.norm(to_target)

    # 目标速度向量标准化
    target_vel_norm = target_velocity / np.linalg.norm(target_velocity)

    # 计算角度
    dot_product = np.dot(to_target_norm, target_vel_norm)
    angle = np.arccos(np.clip(dot_product, -1, 1)) * 180 / np.pi

    return angle


def _calculate_enemy_threat_level(distance: float, enemy_missiles: int, altitude_advantage: float) -> int:
    """计算敌机威胁等级"""
    threat_level = 0

    # 距离威胁
    if distance < 20000:
        threat_level += 4
    elif distance < 40000:
        threat_level += 3
    elif distance < 60000:
        threat_level += 2
    else:
        threat_level += 1

    # 导弹威胁
    threat_level += min(enemy_missiles, 3)

    # 高度劣势增加威胁
    if altitude_advantage < -1000:
        threat_level += 2
    elif altitude_advantage < 0:
        threat_level += 1

    return threat_level


def _analyze_missile_threats(env, agent_id: str, agent_pos):
    """分析导弹威胁"""
    missile_threats = []

    if hasattr(env, 'missiles'):
        for missile_id, missile in env.missiles.items():
            if missile.is_alive and hasattr(missile, 'target_agent_id') and missile.target_agent_id == agent_id:
                try:
                    missile_pos = np.array(missile.get_position()) if hasattr(missile, 'get_position') else None
                    if missile_pos is not None:
                        missile_distance = np.linalg.norm(agent_pos - missile_pos)
                        missile_velocity = missile.get_velocity() if hasattr(missile, 'get_velocity') else np.array([0, 0, 0])
                        missile_speed = np.linalg.norm(missile_velocity)

                        # 计算威胁紧急度
                        if missile_distance < 3000:
                            urgency = 5  # 极度紧急
                        elif missile_distance < 8000:
                            urgency = 4  # 高度紧急
                        elif missile_distance < 15000:
                            urgency = 3  # 中度紧急
                        elif missile_distance < 25000:
                            urgency = 2  # 低度紧急
                        else:
                            urgency = 1  # 远程威胁

                        missile_threats.append({
                            'missile_id': missile_id,
                            'missile': missile,
                            'distance': missile_distance,
                            'speed': missile_speed,
                            'urgency': urgency,
                            'time_to_impact': missile_distance / max(missile_speed, 1)
                        })
                except:
                    pass

    # 按威胁紧急度排序
    missile_threats.sort(key=lambda x: x['urgency'], reverse=True)
    return missile_threats


def _assess_battlefield_control(enemy_targets, friendly_aircraft):
    """评估战场控制状况"""
    total_enemies = len(enemy_targets)
    total_friendlies = len(friendly_aircraft) + 1  # +1 for self

    if total_enemies == 0:
        return "完全控制"
    elif total_friendlies > total_enemies:
        return "优势控制"
    elif total_friendlies == total_enemies:
        return "均势"
    else:
        return "劣势"


def _calculate_target_threat_score(env, target, distance: float) -> int:
    """计算目标威胁分数"""
    threat_score = 0

    # 距离威胁 - 距离越近威胁越大
    if distance < 20000:  # 20km内
        threat_score += 4
    elif distance < 40000:  # 40km内
        threat_score += 3
    elif distance < 60000:  # 60km内
        threat_score += 2
    else:
        threat_score += 1

    # 目标导弹威胁
    try:
        target_missiles = target.num_missiles if hasattr(target, 'num_missiles') else 0
        if target_missiles > 2:
            threat_score += 3
        elif target_missiles > 0:
            threat_score += 2
    except:
        pass

    # 目标是否正在攻击我方
    if hasattr(env, 'missiles'):
        for missile_id, missile in env.missiles.items():
            if missile.is_alive and hasattr(missile, 'launcher_agent_id') and missile.launcher_agent_id == target.agent_id:
                threat_score += 2
                break

    return threat_score


def _determine_engagement_phase(current_time: float, target_distance: float) -> str:
    """确定交战阶段"""
    if current_time < 30:
        return "初始接敌"
    elif current_time < 60:
        return "中距离交战"
    elif current_time < 120:
        return "近距离激战"
    else:
        return "持续作战"


def _determine_tactical_role(agent_id: str, situation: Dict[str, Any]) -> str:
    """确定战术角色 - 实现角色分化"""
    # 基于飞机ID和战术情况分配角色
    if agent_id == "B0100":
        # B0100作为主攻击者
        if situation['immediate_missile_threat']:
            return "主攻击者-规避"
        elif situation['can_attack'] and situation['target_distance'] < 40000:
            return "主攻击者-进攻"
        else:
            return "主攻击者-接敌"

    elif agent_id == "B0200":
        # B0200作为支援者/侧翼攻击者
        if situation['immediate_missile_threat']:
            return "侧翼攻击者-规避"
        elif len(situation['friendly_aircraft']) > 0:
            return "侧翼攻击者-协调"
        else:
            return "侧翼攻击者-独立"

    else:
        return "通用战斗者"


def _select_intelligent_behavior(situation: Dict[str, Any], role: str, agent) -> str:
    """选择智能战斗行为 - 基于角色和威胁的复杂决策"""
    distance = situation['target_distance']
    missile_threat = situation['immediate_missile_threat']
    threat_urgency = situation['threat_urgency']
    can_attack = situation['can_attack']
    engagement_phase = situation['engagement_phase']
    altitude_advantage = situation['altitude_advantage']

    # 紧急威胁处理 - 最高优先级
    if missile_threat and threat_urgency >= 4:
        if role.startswith("主攻击者"):
            return "紧急规避-反击"  # 主攻击者在规避时准备反击
        else:
            return "紧急规避-支援"  # 侧翼攻击者规避并准备支援

    # 基于角色的战术决策
    if role == "主攻击者-进攻":
        if distance < 15000 and can_attack:
            return "主攻击者-近距离突击"
        elif distance < 30000 and can_attack:
            return "主攻击者-中距离强攻"
        elif distance < 50000:
            return "主攻击者-快速接敌"
        else:
            return "主攻击者-搜索接敌"

    elif role == "主攻击者-接敌":
        if distance < 40000:
            return "主攻击者-战术接敌"
        else:
            return "主攻击者-高速接敌"

    elif role == "侧翼攻击者-协调":
        if distance < 20000 and can_attack:
            return "侧翼攻击者-协调攻击"
        elif distance < 40000:
            return "侧翼攻击者-侧翼机动"
        else:
            return "侧翼攻击者-包抄接敌"

    elif role == "侧翼攻击者-独立":
        if distance < 25000 and can_attack:
            return "侧翼攻击者-独立攻击"
        elif altitude_advantage:
            return "侧翼攻击者-高度攻击"
        else:
            return "侧翼攻击者-机动攻击"

    elif "规避" in role:
        if threat_urgency >= 4:
            return "高机动规避"
        elif threat_urgency >= 3:
            return "战术规避"
        else:
            return "预防性机动"

    # 默认基于距离的行为
    if distance < 20000:
        return "近距离战斗"
    elif distance < 40000:
        return "中距离交战"
    else:
        return "远距离接敌"


def _generate_intelligent_commands(behavior: str, situation: Dict[str, Any], role: str, agent) -> Tuple[int, int, int]:
    """生成智能战斗指令 - 基于行为、角色和威胁的复杂指令生成"""
    # 动作空间: [高度(0-14), 航向(0-16), 速度(0-6)]

    # 紧急规避指令 - 最高优先级
    if "紧急规避" in behavior:
        if situation['threat_urgency'] >= 5:
            # 极度紧急：急剧机动
            return (2, 1, 6)    # 急降+大左转+最大速度
        else:
            # 高度紧急：大幅机动
            return (4, 3, 5)    # 下降+左转+高速

    # 主攻击者指令
    elif behavior == "主攻击者-近距离突击":
        return (11, 13, 6)  # 爬升+大右转+最大速度

    elif behavior == "主攻击者-中距离强攻":
        return (9, 11, 5)   # 爬升+右转+高速

    elif behavior == "主攻击者-快速接敌":
        return (8, 9, 6)    # 轻微爬升+轻微右转+最大速度

    elif behavior == "主攻击者-战术接敌":
        return (7, 10, 4)   # 保持高度+右转+加速

    elif behavior == "主攻击者-高速接敌":
        return (7, 8, 6)    # 保持高度+直飞+最大速度

    elif behavior == "主攻击者-搜索接敌":
        return (8, 8, 4)    # 轻微爬升+直飞+加速

    # 侧翼攻击者指令 - 与主攻击者差异化
    elif behavior == "侧翼攻击者-协调攻击":
        return (10, 6, 5)   # 爬升+左转+高速 (与主攻击者形成夹击)

    elif behavior == "侧翼攻击者-侧翼机动":
        return (6, 5, 4)    # 下降+左转+加速 (低空侧翼)

    elif behavior == "侧翼攻击者-包抄接敌":
        return (9, 4, 5)    # 爬升+大左转+高速 (大范围包抄)

    elif behavior == "侧翼攻击者-独立攻击":
        return (8, 12, 5)   # 轻微爬升+右转+高速

    elif behavior == "侧翼攻击者-高度攻击":
        return (12, 7, 4)   # 大幅爬升+轻微左转+加速

    elif behavior == "侧翼攻击者-机动攻击":
        return (7, 6, 5)    # 保持高度+左转+高速

    # 规避机动指令
    elif behavior == "高机动规避":
        return (3, 2, 6)    # 下降+大左转+最大速度

    elif behavior == "战术规避":
        return (5, 4, 5)    # 轻微下降+左转+高速

    elif behavior == "预防性机动":
        return (6, 6, 4)    # 轻微下降+左转+加速

    # 通用战斗指令
    elif behavior == "近距离战斗":
        if role.startswith("主攻击者"):
            return (10, 11, 5)  # 主攻击者：爬升+右转+高速
        else:
            return (8, 5, 5)    # 侧翼攻击者：轻微爬升+左转+高速

    elif behavior == "中距离交战":
        if role.startswith("主攻击者"):
            return (8, 9, 4)    # 主攻击者：轻微爬升+轻微右转+加速
        else:
            return (7, 7, 4)    # 侧翼攻击者：保持高度+轻微左转+加速

    elif behavior == "远距离接敌":
        if role.startswith("主攻击者"):
            return (7, 8, 5)    # 主攻击者：保持高度+直飞+高速
        else:
            return (8, 6, 4)    # 侧翼攻击者：轻微爬升+左转+加速

    else:
        # 默认指令：基于角色的差异化
        if role.startswith("主攻击者"):
            return (8, 9, 4)    # 主攻击者默认
        else:
            return (7, 7, 4)    # 侧翼攻击者默认


def _evaluate_threats_and_opportunities(env, agent_id: str, battlefield_awareness: Dict[str, Any]) -> Dict[str, Any]:
    """威胁评估和机会分析 - 强对抗性AI的决策核心"""

    # 1. 导弹威胁评估
    missile_threats = battlefield_awareness['missile_threats']
    immediate_missile_threat = len([m for m in missile_threats if m['urgency'] >= 4]) > 0
    missile_threat_level = max([m['urgency'] for m in missile_threats]) if missile_threats else 0
    closest_missile = missile_threats[0] if missile_threats else None

    # 2. 敌机威胁和机会评估
    enemy_targets = battlefield_awareness['enemy_targets']

    # 选择主要目标（威胁最大或机会最好）
    primary_target = None
    best_opportunity = None
    highest_threat = None

    for target in enemy_targets:
        # 威胁评估
        if highest_threat is None or target['threat_level'] > highest_threat['threat_level']:
            highest_threat = target

        # 机会评估
        opportunity_score = _calculate_opportunity_score(target, battlefield_awareness)
        target['opportunity_score'] = opportunity_score

        if best_opportunity is None or opportunity_score > best_opportunity['opportunity_score']:
            best_opportunity = target

    # 根据战术情况选择主要目标
    if immediate_missile_threat:
        # 有导弹威胁时，优先考虑最大威胁
        primary_target = highest_threat
    else:
        # 无导弹威胁时，优先考虑最佳机会
        primary_target = best_opportunity

    # 3. 战术优势评估
    tactical_advantages = _assess_tactical_advantages(battlefield_awareness, primary_target)

    # 4. 协调机会评估
    coordination_opportunities = _assess_coordination_opportunities(battlefield_awareness, primary_target)

    return {
        'immediate_missile_threat': immediate_missile_threat,
        'missile_threat_level': missile_threat_level,
        'closest_missile': closest_missile,
        'primary_target': primary_target,
        'all_targets': enemy_targets,
        'highest_threat': highest_threat,
        'best_opportunity': best_opportunity,
        'tactical_advantages': tactical_advantages,
        'coordination_opportunities': coordination_opportunities,
        'engagement_range': _determine_engagement_range(primary_target['distance'] if primary_target else float('inf'))
    }


def _calculate_opportunity_score(target: Dict[str, Any], battlefield_awareness: Dict[str, Any]) -> float:
    """计算攻击机会分数"""
    score = 0.0

    # 距离因子（中距离最佳）
    distance = target['distance']
    if 15000 <= distance <= 35000:
        score += 5.0
    elif 10000 <= distance <= 50000:
        score += 3.0
    elif distance < 10000:
        score += 1.0  # 太近危险
    else:
        score += 0.5  # 太远效果差

    # 高度优势
    if target['altitude_advantage'] > 2000:
        score += 3.0
    elif target['altitude_advantage'] > 500:
        score += 1.5
    elif target['altitude_advantage'] < -1000:
        score -= 2.0

    # 速度优势
    if target['speed_advantage'] > 100:
        score += 2.0
    elif target['speed_advantage'] > 50:
        score += 1.0
    elif target['speed_advantage'] < -50:
        score -= 1.0

    # 纵横比角度（侧面攻击最佳）
    aspect = target['aspect_angle']
    if 60 <= aspect <= 120:
        score += 2.0  # 侧面攻击
    elif aspect < 30:
        score -= 1.0  # 正面对头
    elif aspect > 150:
        score += 1.0  # 尾追

    # 目标导弹数量（导弹少的目标更容易攻击）
    if target['missiles'] == 0:
        score += 3.0
    elif target['missiles'] == 1:
        score += 1.0
    else:
        score -= 1.0

    # 我方导弹数量
    my_missiles = battlefield_awareness['missile_count']
    if my_missiles > 0:
        score += 2.0
    else:
        score -= 3.0  # 没有导弹大幅降低机会

    return max(score, 0.0)


def _assess_tactical_advantages(battlefield_awareness: Dict[str, Any], primary_target) -> Dict[str, bool]:
    """评估战术优势"""
    advantages = {
        'altitude_advantage': False,
        'speed_advantage': False,
        'energy_advantage': False,
        'position_advantage': False,
        'numerical_advantage': False
    }

    if primary_target:
        # 高度优势
        advantages['altitude_advantage'] = primary_target['altitude_advantage'] > 1000

        # 速度优势
        advantages['speed_advantage'] = primary_target['speed_advantage'] > 50

        # 能量优势
        my_energy = battlefield_awareness['energy_state']
        advantages['energy_advantage'] = my_energy in ["高能量", "中等能量"]

        # 位置优势（侧面或后方）
        aspect = primary_target['aspect_angle']
        advantages['position_advantage'] = aspect > 90

        # 数量优势
        total_enemies = len(battlefield_awareness['enemy_targets'])
        total_friendlies = len(battlefield_awareness['friendly_aircraft']) + 1
        advantages['numerical_advantage'] = total_friendlies >= total_enemies

    return advantages


def _assess_coordination_opportunities(battlefield_awareness: Dict[str, Any], primary_target) -> Dict[str, Any]:
    """评估协调机会"""
    opportunities = {
        'can_coordinate': False,
        'pincer_attack': False,
        'high_low_split': False,
        'distraction_support': False
    }

    friendly_aircraft = battlefield_awareness['friendly_aircraft']

    if friendly_aircraft and primary_target:
        opportunities['can_coordinate'] = True

        # 检查钳形攻击机会
        my_pos = battlefield_awareness['my_position']
        target_pos = primary_target['position']

        for friendly in friendly_aircraft:
            friendly_pos = friendly['position']

            # 计算角度关系
            my_to_target = target_pos - my_pos
            friendly_to_target = target_pos - friendly_pos

            angle = np.arccos(np.clip(np.dot(my_to_target, friendly_to_target) /
                                    (np.linalg.norm(my_to_target) * np.linalg.norm(friendly_to_target)), -1, 1))
            angle_deg = angle * 180 / np.pi

            if 90 <= angle_deg <= 180:
                opportunities['pincer_attack'] = True

            # 检查高低分离机会
            altitude_diff = abs(battlefield_awareness['my_altitude'] - friendly['altitude'])
            if altitude_diff > 2000:
                opportunities['high_low_split'] = True

    return opportunities


def _determine_engagement_range(distance: float) -> str:
    """确定交战距离范围"""
    if distance < 15000:
        return "近距离"
    elif distance < 35000:
        return "中距离"
    elif distance < 60000:
        return "远距离"
    else:
        return "超远距离"


def _analyze_bvr_situation(env, agent_id: str, current_time: float) -> Dict[str, Any]:
    """BVR战场态势感知 - 基于真实BVR作战原则"""
    agent = env.agents[agent_id]
    agent_pos = np.array(agent.get_position())

    try:
        # 获取自身状态
        my_altitude = agent.get_property_value(c.position_h_sl_m)
        my_velocity = agent.get_velocity() if hasattr(agent, 'get_velocity') else np.array([0, 0, 0])
        my_speed = np.linalg.norm(my_velocity)
        my_heading = agent.get_property_value(c.attitude_heading_true_rad) * 180 / np.pi
        missile_count = agent.num_missiles if hasattr(agent, 'num_missiles') else 0
        fuel_remaining = getattr(agent, 'fuel_remaining', 1.0)  # 假设燃料剩余比例
    except:
        my_altitude = 10000
        my_velocity = np.array([0, 0, 0])  # 修复：添加默认速度向量
        my_speed = 300
        my_heading = 180
        missile_count = 0
        fuel_remaining = 1.0

    # 友机状态分析（编队协调）
    friendly_aircraft = []
    for friendly_id in ["B0100", "B0200"]:
        if friendly_id != agent_id and friendly_id in env.agents and env.agents[friendly_id].is_alive:
            friendly = env.agents[friendly_id]
            friendly_pos = np.array(friendly.get_position())
            friendly_distance = np.linalg.norm(agent_pos - friendly_pos)
            try:
                friendly_altitude = friendly.get_property_value(c.position_h_sl_m)
                friendly_speed = np.linalg.norm(friendly.get_velocity()) if hasattr(friendly, 'get_velocity') else 300
                friendly_missiles = friendly.num_missiles if hasattr(friendly, 'num_missiles') else 0
                friendly_heading = friendly.get_property_value(c.attitude_heading_true_rad) * 180 / np.pi
            except:
                friendly_altitude = 10000
                friendly_speed = 300
                friendly_missiles = 0
                friendly_heading = 180

            friendly_aircraft.append({
                'id': friendly_id,
                'distance': friendly_distance,
                'altitude': friendly_altitude,
                'speed': friendly_speed,
                'heading': friendly_heading,
                'missiles': friendly_missiles,
                'position': friendly_pos
            })

    # 敌机目标分析（BVR威胁评估）
    enemy_targets = []
    for enemy_id in ["A0100", "A0200"]:
        if enemy_id in env.agents and env.agents[enemy_id].is_alive:
            enemy = env.agents[enemy_id]
            enemy_pos = np.array(enemy.get_position())
            distance = np.linalg.norm(agent_pos - enemy_pos)

            try:
                enemy_altitude = enemy.get_property_value(c.position_h_sl_m)
                enemy_velocity = enemy.get_velocity() if hasattr(enemy, 'get_velocity') else np.array([0, 0, 0])
                enemy_speed = np.linalg.norm(enemy_velocity)
                enemy_heading = enemy.get_property_value(c.attitude_heading_true_rad) * 180 / np.pi
                enemy_missiles = enemy.num_missiles if hasattr(enemy, 'num_missiles') else 0
            except:
                enemy_altitude = 7000
                enemy_velocity = np.array([0, 0, 0])
                enemy_speed = 250
                enemy_heading = 0
                enemy_missiles = 0

            # BVR关键参数计算
            altitude_advantage = my_altitude - enemy_altitude
            speed_advantage = my_speed - enemy_speed
            relative_bearing = _calculate_relative_bearing(agent_pos, enemy_pos, my_heading)
            aspect_angle = _calculate_aspect_angle(agent_pos, enemy_pos, enemy_velocity)
            closure_rate = _calculate_closure_rate(my_velocity, enemy_velocity, agent_pos, enemy_pos)

            # BVR威胁等级评估
            bvr_threat_level = _calculate_bvr_threat_level(distance, enemy_missiles, closure_rate, aspect_angle)

            enemy_targets.append({
                'id': enemy_id,
                'distance': distance,
                'altitude': enemy_altitude,
                'speed': enemy_speed,
                'heading': enemy_heading,
                'missiles': enemy_missiles,
                'position': enemy_pos,
                'velocity': enemy_velocity,
                'altitude_advantage': altitude_advantage,
                'speed_advantage': speed_advantage,
                'relative_bearing': relative_bearing,
                'aspect_angle': aspect_angle,
                'closure_rate': closure_rate,
                'bvr_threat_level': bvr_threat_level
            })

    # 导弹威胁分析（BVR关键）
    missile_threats = _analyze_bvr_missile_threats(env, agent_id, agent_pos)
    immediate_missile_threat = len([m for m in missile_threats if m['urgency'] >= 4]) > 0

    # 选择主要目标（BVR优先级）
    primary_target = None
    if enemy_targets:
        # BVR目标优先级：距离适中、威胁高、易攻击
        enemy_targets.sort(key=lambda x: (x['bvr_threat_level'], -x['distance']), reverse=True)
        primary_target = enemy_targets[0]

    # BVR战场控制评估
    battlefield_control = _assess_bvr_battlefield_control(enemy_targets, friendly_aircraft, missile_threats)

    return {
        'agent_id': agent_id,
        'current_time': current_time,
        'my_position': agent_pos,
        'my_altitude': my_altitude,
        'my_speed': my_speed,
        'my_heading': my_heading,
        'missile_count': missile_count,
        'fuel_remaining': fuel_remaining,
        'friendly_aircraft': friendly_aircraft,
        'enemy_targets': enemy_targets,
        'primary_target': primary_target,
        'missile_threats': missile_threats,
        'immediate_missile_threat': immediate_missile_threat,
        'battlefield_control': battlefield_control,
        'formation_integrity': len(friendly_aircraft) > 0  # 编队完整性
    }


def _calculate_closure_rate(my_velocity, enemy_velocity, my_pos, enemy_pos):
    """计算接近速率"""
    relative_velocity = my_velocity - enemy_velocity
    range_vector = enemy_pos - my_pos
    range_distance = np.linalg.norm(range_vector)

    if range_distance < 1:
        return 0

    range_unit = range_vector / range_distance
    closure_rate = -np.dot(relative_velocity, range_unit)  # 负号表示接近
    return closure_rate


def _calculate_bvr_threat_level(distance: float, enemy_missiles: int, closure_rate: float, aspect_angle: float) -> int:
    """计算BVR威胁等级"""
    threat_level = 0

    # 距离威胁（BVR关键）
    if distance < 25000:  # 25km内高威胁
        threat_level += 5
    elif distance < 40000:  # 40km内中威胁
        threat_level += 3
    elif distance < 60000:  # 60km内低威胁
        threat_level += 1

    # 导弹威胁
    threat_level += min(enemy_missiles * 2, 6)

    # 接近速率威胁
    if closure_rate > 200:  # 高速接近
        threat_level += 3
    elif closure_rate > 100:
        threat_level += 1

    # 纵横比威胁（正面接近最危险）
    if aspect_angle < 30:  # 正面对头
        threat_level += 2
    elif aspect_angle > 150:  # 尾追
        threat_level -= 1

    return max(threat_level, 0)


def _analyze_bvr_missile_threats(env, agent_id: str, agent_pos):
    """分析BVR导弹威胁"""
    missile_threats = []

    if hasattr(env, 'missiles'):
        for missile_id, missile in env.missiles.items():
            if missile.is_alive and hasattr(missile, 'target_agent_id') and missile.target_agent_id == agent_id:
                try:
                    missile_pos = np.array(missile.get_position()) if hasattr(missile, 'get_position') else None
                    if missile_pos is not None:
                        missile_distance = np.linalg.norm(agent_pos - missile_pos)
                        missile_velocity = missile.get_velocity() if hasattr(missile, 'get_velocity') else np.array([0, 0, 0])
                        missile_speed = np.linalg.norm(missile_velocity)

                        # BVR导弹威胁紧急度
                        if missile_distance < 5000:
                            urgency = 5  # 极度紧急
                        elif missile_distance < 12000:
                            urgency = 4  # 高度紧急
                        elif missile_distance < 20000:
                            urgency = 3  # 中度紧急
                        elif missile_distance < 35000:
                            urgency = 2  # 低度紧急
                        else:
                            urgency = 1  # 远程威胁

                        missile_threats.append({
                            'missile_id': missile_id,
                            'distance': missile_distance,
                            'speed': missile_speed,
                            'urgency': urgency,
                            'time_to_impact': missile_distance / max(missile_speed, 1)
                        })
                except:
                    pass

    # 按威胁紧急度排序
    missile_threats.sort(key=lambda x: x['urgency'], reverse=True)
    return missile_threats


def _assess_bvr_battlefield_control(enemy_targets, friendly_aircraft, missile_threats):
    """评估BVR战场控制状况"""
    total_enemies = len(enemy_targets)
    total_friendlies = len(friendly_aircraft) + 1  # +1 for self
    active_missile_threats = len([m for m in missile_threats if m['urgency'] >= 3])

    if total_enemies == 0:
        return "完全控制"
    elif active_missile_threats >= 2:
        return "严重威胁"
    elif total_friendlies > total_enemies and active_missile_threats == 0:
        return "优势控制"
    elif total_friendlies == total_enemies:
        return "均势"
    else:
        return "劣势"


def _determine_formation_role(agent_id: str, bvr_situation: Dict[str, Any]) -> str:
    """确定编队角色 - 长机僚机分工"""
    friendly_aircraft = bvr_situation['friendly_aircraft']

    if agent_id == "B0100":
        # B0100作为长机
        if bvr_situation['immediate_missile_threat']:
            return "长机-防御"
        elif bvr_situation['primary_target'] and bvr_situation['primary_target']['distance'] < 50000:
            return "长机-攻击"
        else:
            return "长机-搜索"

    elif agent_id == "B0200":
        # B0200作为僚机
        if bvr_situation['immediate_missile_threat']:
            return "僚机-支援防御"
        elif len(friendly_aircraft) > 0:
            # 有长机存在，执行协调任务
            leader_distance = friendly_aircraft[0]['distance'] if friendly_aircraft else float('inf')
            if leader_distance > 10000:  # 编队分散
                return "僚机-重新编队"
            else:
                return "僚机-协调攻击"
        else:
            # 长机已被击落，独立作战
            return "僚机-独立作战"

    return "独立作战"


def _determine_bvr_phase(bvr_situation: Dict[str, Any], formation_role: str) -> str:
    """确定BVR交战阶段"""
    primary_target = bvr_situation['primary_target']

    if not primary_target:
        return "搜索阶段"

    distance = primary_target['distance']
    missile_count = bvr_situation['missile_count']
    immediate_threat = bvr_situation['immediate_missile_threat']

    # BVR阶段划分（基于真实作战距离）
    if immediate_threat:
        return "防御阶段"
    elif distance > 80000:  # 80km以上
        return "远程BVR"
    elif distance > 50000:  # 50-80km
        return "中程BVR"
    elif distance > 25000:  # 25-50km
        return "近程BVR"
    elif distance > 15000:  # 15-25km
        return "BVR-WVR过渡"
    else:  # 15km以下
        return "近距格斗"


def _evaluate_disengagement_criteria(bvr_situation: Dict[str, Any], engagement_phase: str, current_time: float) -> bool:
    """评估脱离接触标准 - BVR作战核心决策"""

    # 1. 强制脱离条件
    fuel_remaining = bvr_situation['fuel_remaining']
    missile_count = bvr_situation['missile_count']
    immediate_threat = bvr_situation['immediate_missile_threat']
    battlefield_control = bvr_situation['battlefield_control']

    # 燃料不足强制脱离
    if fuel_remaining < 0.3:  # 燃料少于30%
        return True

    # 导弹耗尽且面临威胁
    if missile_count == 0 and immediate_threat:
        return True

    # 严重威胁环境
    if battlefield_control == "严重威胁":
        missile_threats = bvr_situation['missile_threats']
        high_urgency_threats = len([m for m in missile_threats if m['urgency'] >= 4])
        if high_urgency_threats >= 2:  # 多枚导弹威胁
            return True

    # 2. 战术脱离条件
    primary_target = bvr_situation['primary_target']
    if primary_target:
        distance = primary_target['distance']
        closure_rate = primary_target['closure_rate']

        # 距离过近且无优势
        if distance < 20000 and closure_rate > 150:  # 20km内高速接近
            altitude_advantage = primary_target['altitude_advantage']
            speed_advantage = primary_target['speed_advantage']
            if altitude_advantage < -1000 and speed_advantage < -50:  # 高度速度双劣势
                return True

    # 3. 编队脱离条件
    friendly_aircraft = bvr_situation['friendly_aircraft']
    if len(friendly_aircraft) == 0:  # 失去编队支援
        if engagement_phase in ["近程BVR", "BVR-WVR过渡", "近距格斗"]:
            return True

    # 4. 时间脱离条件
    if current_time > 240:  # 4分钟后考虑脱离
        if missile_count <= 1 and not immediate_threat:
            return True

    return False


def _select_disengagement_behavior(bvr_situation: Dict[str, Any], formation_role: str) -> str:
    """选择脱离接触行为"""
    immediate_threat = bvr_situation['immediate_missile_threat']
    battlefield_control = bvr_situation['battlefield_control']

    if immediate_threat:
        if formation_role.startswith("长机"):
            return "长机紧急脱离"
        else:
            return "僚机紧急脱离"

    elif battlefield_control == "严重威胁":
        if formation_role.startswith("长机"):
            return "长机战术脱离"
        else:
            return "僚机掩护脱离"

    else:
        if formation_role.startswith("长机"):
            return "长机有序脱离"
        else:
            return "僚机跟随脱离"


def _select_bvr_behavior(engagement_phase: str, bvr_situation: Dict[str, Any], formation_role: str) -> str:
    """选择BVR战术行为"""
    primary_target = bvr_situation['primary_target']
    missile_count = bvr_situation['missile_count']
    immediate_threat = bvr_situation['immediate_missile_threat']

    # 威胁响应优先
    if immediate_threat:
        if formation_role.startswith("长机"):
            return "长机导弹规避"
        else:
            return "僚机支援规避"

    # 基于阶段的行为选择
    if engagement_phase == "远程BVR":
        if formation_role.startswith("长机"):
            return "长机远程搜索"
        else:
            return "僚机编队保持"

    elif engagement_phase == "中程BVR":
        if missile_count > 0 and primary_target:
            if formation_role.startswith("长机"):
                return "长机中程攻击"
            else:
                return "僚机协调攻击"
        else:
            if formation_role.startswith("长机"):
                return "长机中程机动"
            else:
                return "僚机支援机动"

    elif engagement_phase == "近程BVR":
        if missile_count > 0 and primary_target:
            if formation_role.startswith("长机"):
                return "长机近程突击"
            else:
                return "僚机侧翼攻击"
        else:
            if formation_role.startswith("长机"):
                return "长机近程防御"
            else:
                return "僚机近程支援"

    elif engagement_phase == "BVR-WVR过渡":
        if formation_role.startswith("长机"):
            return "长机过渡机动"
        else:
            return "僚机过渡支援"

    elif engagement_phase == "近距格斗":
        if formation_role.startswith("长机"):
            return "长机格斗机动"
        else:
            return "僚机格斗支援"

    else:  # 搜索阶段
        if formation_role.startswith("长机"):
            return "长机搜索接敌"
        else:
            return "僚机搜索支援"


def _generate_bvr_commands(tactical_behavior: str, bvr_situation: Dict[str, Any],
                          formation_role: str, agent_id: str) -> Tuple[int, int, int]:
    """生成BVR战术指令 - 基于真实BVR作战原则"""

    # 动作空间: [高度(0-14), 航向(0-16), 速度(0-6)]
    # 7=保持当前, <7下降/左转/减速, >7爬升/右转/加速

    primary_target = bvr_situation['primary_target']
    my_altitude = bvr_situation['my_altitude']
    my_heading = bvr_situation['my_heading']

    # 1. 脱离接触指令
    if "脱离" in tactical_behavior:
        if "紧急" in tactical_behavior:
            # 紧急脱离：最大机动
            return (2, 1, 6)  # 急降+最大左转+最大速度
        elif "战术" in tactical_behavior:
            # 战术脱离：有序撤退
            return (4, 3, 5)  # 下降+左转+高速
        else:
            # 有序脱离：保持编队
            return (6, 5, 4)  # 轻微下降+左转+加速

    # 2. 导弹规避指令
    elif "规避" in tactical_behavior:
        missile_threats = bvr_situation['missile_threats']
        if missile_threats and missile_threats[0]['urgency'] >= 4:
            # 高威胁导弹规避
            if formation_role.startswith("长机"):
                return (3, 2, 6)  # 长机：下降+大左转+最大速度
            else:
                return (2, 4, 6)  # 僚机：急降+左转+最大速度
        else:
            # 预防性规避
            return (5, 4, 5)  # 轻微下降+左转+高速

    # 3. 远程BVR指令
    elif "远程" in tactical_behavior:
        if formation_role.startswith("长机"):
            # 长机远程搜索：保持高度优势
            return (8, 8, 4)  # 轻微爬升+直飞+加速
        else:
            # 僚机编队保持：与长机协调
            return (7, 7, 4)  # 保持高度+轻微左转+加速

    # 4. 中程BVR指令
    elif "中程" in tactical_behavior:
        if "攻击" in tactical_behavior:
            if formation_role.startswith("长机"):
                # 长机中程攻击：主动接敌
                return (9, 10, 5)  # 爬升+右转+高速
            else:
                # 僚机协调攻击：侧翼支援
                return (8, 6, 5)   # 轻微爬升+左转+高速
        else:
            # 中程机动：保持机动性
            return (7, 9, 4)   # 保持高度+轻微右转+加速

    # 5. 近程BVR指令
    elif "近程" in tactical_behavior:
        if "突击" in tactical_behavior:
            # 长机近程突击：最大攻击性
            return (11, 12, 6)  # 爬升+右转+最大速度
        elif "侧翼" in tactical_behavior:
            # 僚机侧翼攻击：包抄机动
            return (9, 5, 5)    # 爬升+左转+高速
        elif "防御" in tactical_behavior:
            # 近程防御：垂直机动
            return (6, 4, 5)    # 轻微下降+左转+高速
        else:
            # 近程支援：灵活机动
            return (8, 7, 5)    # 轻微爬升+轻微左转+高速

    # 6. 过渡阶段指令
    elif "过渡" in tactical_behavior:
        if formation_role.startswith("长机"):
            # 长机过渡：准备近距格斗
            return (10, 11, 5)  # 爬升+右转+高速
        else:
            # 僚机过渡：支援准备
            return (8, 6, 5)    # 轻微爬升+左转+高速

    # 7. 格斗指令
    elif "格斗" in tactical_behavior:
        if formation_role.startswith("长机"):
            # 长机格斗：主动攻击
            return (12, 13, 6)  # 大幅爬升+大右转+最大速度
        else:
            # 僚机格斗：支援攻击
            return (10, 5, 6)   # 爬升+左转+最大速度

    # 8. 搜索指令
    elif "搜索" in tactical_behavior:
        if formation_role.startswith("长机"):
            # 长机搜索：主动搜索
            return (8, 9, 4)    # 轻微爬升+轻微右转+加速
        else:
            # 僚机搜索：编队搜索
            return (7, 7, 4)    # 保持高度+轻微左转+加速

    # 9. 编队相关指令
    elif "编队" in tactical_behavior:
        friendly_aircraft = bvr_situation['friendly_aircraft']
        if friendly_aircraft:
            # 重新编队：向长机靠拢
            leader_pos = friendly_aircraft[0]['position']
            my_pos = bvr_situation['my_position']

            # 计算相对位置
            dx = leader_pos[0] - my_pos[0]
            dy = leader_pos[1] - my_pos[1]

            if abs(dx) > abs(dy):
                if dx > 0:
                    return (7, 10, 4)  # 向东靠拢
                else:
                    return (7, 6, 4)   # 向西靠拢
            else:
                if dy > 0:
                    return (7, 8, 4)   # 向北靠拢
                else:
                    return (7, 12, 4)  # 向南靠拢
        else:
            return (7, 8, 4)  # 默认直飞

    # 10. 独立作战指令
    elif "独立" in tactical_behavior:
        if primary_target and primary_target['distance'] < 40000:
            # 独立攻击
            return (9, 10, 5)  # 爬升+右转+高速
        else:
            # 独立搜索
            return (8, 8, 4)   # 轻微爬升+直飞+加速

    # 默认指令：基于角色的标准机动
    else:
        if formation_role.startswith("长机"):
            return (8, 9, 4)    # 长机默认：轻微爬升+轻微右转+加速
        else:
            return (7, 7, 4)    # 僚机默认：保持高度+轻微左转+加速


def _determine_tactical_mode(agent_id: str, threat_assessment: Dict[str, Any],
                           battlefield_awareness: Dict[str, Any], current_time: float) -> str:
    """确定战术模式 - 进攻/防守/中立的动态切换"""

    # 1. 紧急防守模式 - 最高优先级
    if threat_assessment['immediate_missile_threat']:
        missile_threat_level = threat_assessment['missile_threat_level']
        if missile_threat_level >= 5:
            return "紧急防守"
        elif missile_threat_level >= 4:
            return "积极防守"
        else:
            return "预防防守"

    # 2. 基于战场态势的模式选择
    primary_target = threat_assessment['primary_target']
    tactical_advantages = threat_assessment['tactical_advantages']
    coordination_opportunities = threat_assessment['coordination_opportunities']

    if not primary_target:
        return "搜索模式"

    # 3. 进攻模式条件评估
    offensive_score = 0

    # 有导弹且目标在有效范围内
    if battlefield_awareness['missile_count'] > 0 and primary_target['distance'] < 50000:
        offensive_score += 3

    # 战术优势
    if tactical_advantages['altitude_advantage']:
        offensive_score += 2
    if tactical_advantages['speed_advantage']:
        offensive_score += 1
    if tactical_advantages['energy_advantage']:
        offensive_score += 2
    if tactical_advantages['position_advantage']:
        offensive_score += 2

    # 协调机会
    if coordination_opportunities['pincer_attack']:
        offensive_score += 3
    if coordination_opportunities['high_low_split']:
        offensive_score += 2

    # 目标脆弱性
    if primary_target['missiles'] == 0:
        offensive_score += 3
    elif primary_target['missiles'] == 1:
        offensive_score += 1

    # 4. 防守模式条件评估
    defensive_score = 0

    # 目标威胁高
    if primary_target['threat_level'] >= 6:
        defensive_score += 3

    # 我方劣势
    if not tactical_advantages['altitude_advantage']:
        defensive_score += 1
    if not tactical_advantages['energy_advantage']:
        defensive_score += 2
    if battlefield_awareness['missile_count'] == 0:
        defensive_score += 4

    # 数量劣势
    if not tactical_advantages['numerical_advantage']:
        defensive_score += 2

    # 5. 角色特定的模式倾向
    if agent_id == "B0100":  # 主攻击者，更倾向于进攻
        offensive_score += 1
    elif agent_id == "B0200":  # 侧翼攻击者，更灵活
        if coordination_opportunities['can_coordinate']:
            offensive_score += 1

    # 6. 时间因素
    if current_time < 60:  # 早期更积极
        offensive_score += 1
    elif current_time > 180:  # 后期更谨慎
        defensive_score += 1

    # 7. 模式决策
    if offensive_score >= 6:
        return "主动进攻"
    elif offensive_score >= 4:
        return "机会进攻"
    elif defensive_score >= 5:
        return "战术防守"
    elif defensive_score >= 3:
        return "谨慎防守"
    else:
        return "中立机动"


def _select_tactical_behavior(tactical_mode: str, threat_assessment: Dict[str, Any],
                            battlefield_awareness: Dict[str, Any], agent_id: str) -> str:
    """选择具体战术行为"""

    primary_target = threat_assessment['primary_target']
    engagement_range = threat_assessment['engagement_range']
    coordination_opportunities = threat_assessment['coordination_opportunities']

    # 1. 防守模式行为
    if tactical_mode == "紧急防守":
        return "紧急规避机动"
    elif tactical_mode == "积极防守":
        return "攻击性规避"
    elif tactical_mode == "预防防守":
        return "预防性机动"
    elif tactical_mode == "战术防守":
        if engagement_range == "近距离":
            return "近距离防守"
        else:
            return "远程防守"
    elif tactical_mode == "谨慎防守":
        return "保守机动"

    # 2. 进攻模式行为
    elif tactical_mode == "主动进攻":
        if engagement_range == "近距离":
            return "近距离突击"
        elif engagement_range == "中距离":
            if coordination_opportunities['pincer_attack']:
                return "协调钳击"
            else:
                return "中距离强攻"
        else:
            return "高速接敌"

    elif tactical_mode == "机会进攻":
        if coordination_opportunities['high_low_split']:
            return "高低分离攻击"
        elif primary_target and primary_target['aspect_angle'] > 120:
            return "尾追攻击"
        else:
            return "机会攻击"

    # 3. 中立模式行为
    elif tactical_mode == "中立机动":
        if agent_id == "B0100":
            return "主攻击者机动"
        else:
            return "侧翼机动"

    elif tactical_mode == "搜索模式":
        return "搜索接敌"

    # 默认行为
    return "标准机动"


def _generate_combat_commands(tactical_behavior: str, threat_assessment: Dict[str, Any],
                            battlefield_awareness: Dict[str, Any], agent_id: str) -> Tuple[int, int, int]:
    """生成精确战斗指令 - 强对抗性AI的核心输出"""

    # 动作空间: [高度(0-14), 航向(0-16), 速度(0-6)]
    # 7=保持当前, <7下降/左转/减速, >7爬升/右转/加速

    primary_target = threat_assessment['primary_target']
    my_energy = battlefield_awareness['energy_state']

    # 1. 紧急防守指令
    if tactical_behavior == "紧急规避机动":
        # 极度紧急：急剧机动 + 最大速度
        return (1, 0, 6)  # 急降+最大左转+最大速度

    elif tactical_behavior == "攻击性规避":
        # 规避但保持攻击能力
        return (3, 2, 6)  # 下降+大左转+最大速度

    elif tactical_behavior == "预防性机动":
        # 预防性规避
        return (5, 4, 5)  # 轻微下降+左转+高速

    elif tactical_behavior == "近距离防守":
        # 近距离防守：垂直机动
        return (2, 3, 6)  # 急降+左转+最大速度

    elif tactical_behavior == "远程防守":
        # 远程防守：保持距离
        return (4, 5, 4)  # 下降+左转+加速

    elif tactical_behavior == "保守机动":
        # 保守机动：小幅调整
        return (6, 6, 4)  # 轻微下降+轻微左转+加速

    # 2. 进攻指令
    elif tactical_behavior == "近距离突击":
        # 近距离突击：最大攻击性
        if my_energy == "高能量":
            return (13, 14, 6)  # 大幅爬升+大右转+最大速度
        else:
            return (11, 12, 6)  # 爬升+右转+最大速度

    elif tactical_behavior == "中距离强攻":
        # 中距离强攻：平衡攻击
        return (9, 11, 5)  # 爬升+右转+高速

    elif tactical_behavior == "协调钳击":
        # 协调钳击：根据角色分工
        if agent_id == "B0100":
            return (10, 13, 5)  # 主攻：爬升+大右转+高速
        else:
            return (8, 4, 5)   # 侧翼：轻微爬升+左转+高速

    elif tactical_behavior == "高低分离攻击":
        # 高低分离：根据当前高度调整
        if battlefield_awareness['my_altitude'] > 12000:
            return (6, 10, 5)  # 高空：下降+右转+高速
        else:
            return (12, 10, 5) # 低空：爬升+右转+高速

    elif tactical_behavior == "尾追攻击":
        # 尾追攻击：保持追击
        return (8, 9, 6)   # 轻微爬升+轻微右转+最大速度

    elif tactical_behavior == "机会攻击":
        # 机会攻击：快速定位
        return (9, 10, 5)  # 爬升+右转+高速

    elif tactical_behavior == "高速接敌":
        # 高速接敌：最大速度接近
        return (7, 8, 6)   # 保持高度+直飞+最大速度

    # 3. 中立机动指令
    elif tactical_behavior == "主攻击者机动":
        # 主攻击者：保持攻击态势
        if primary_target and primary_target['distance'] < 40000:
            return (8, 9, 5)  # 轻微爬升+轻微右转+高速
        else:
            return (7, 8, 5)  # 保持高度+直飞+高速

    elif tactical_behavior == "侧翼机动":
        # 侧翼机动：寻找侧翼位置
        return (7, 6, 4)   # 保持高度+左转+加速

    elif tactical_behavior == "搜索接敌":
        # 搜索接敌：保持搜索态势
        return (8, 8, 4)   # 轻微爬升+直飞+加速

    # 4. 默认指令
    else:
        # 标准机动：基于角色的默认行为
        if agent_id == "B0100":
            return (8, 9, 4)  # 主攻击者：轻微爬升+轻微右转+加速
        else:
            return (7, 7, 4)  # 侧翼攻击者：保持高度+轻微左转+加速
