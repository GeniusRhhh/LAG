"""
统一雷达管理系统 - 支持多项目复用
包含敌我双方的完整雷达系统实现，支持拖曳射击、钳形夹击等多种战术场景

设计理念：
1. 友方雷达：保持简单的状态转换机制，不影响现有战术逻辑
2. 敌方雷达：实现真实的N001VE雷达物理特性，提供挑战性对抗
3. 统一接口：支持多项目复用，便于未来扩展
4. 真实建模：包含完整的电子战、地形遮蔽、大气干扰等效果
"""

import logging
import numpy as np
import math
import random
import time
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

# 导入JSBSim catalog
try:
    from envs.JSBSim.core.catalog import Catalog as c
except ImportError:
    # 如果导入失败，创建一个基础的catalog类
    class c:
        attitude_psi_rad = "attitude/psi-rad"
        position_h_sl_m = "position/h-sl-m"


class RadarStatus(Enum):
    """雷达状态枚举"""
    SEARCH = "SEARCH"      # 搜索模式
    TRACK = "TRACK"        # 跟踪模式
    LOCK = "LOCK"          # 锁定模式
    STANDBY = "STANDBY"    # 待机模式
    JAMMING = "JAMMING"    # 被干扰状态
    MAINTENANCE = "MAINTENANCE"  # 维护状态


class ECMType(Enum):
    """电子对抗措施类型"""
    NOISE_JAMMING = "NOISE_JAMMING"        # 噪声干扰
    DECEPTION_JAMMING = "DECEPTION_JAMMING" # 欺骗干扰
    CHAFF = "CHAFF"                        # 箔条干扰
    FLARE = "FLARE"                        # 红外诱饵
    FREQUENCY_AGILITY = "FREQUENCY_AGILITY" # 频率捷变
    SIDELOBE_BLANKING = "SIDELOBE_BLANKING" # 旁瓣消隐


@dataclass
class RadarTarget:
    """雷达目标信息"""
    target_id: str
    distance: float
    bearing: float
    elevation: float
    velocity: float
    rcs: float = 5.0                       # 雷达截面积 (m²)
    detection_probability: float = 0.0
    track_quality: float = 0.0
    lock_time: float = 0.0
    last_update: float = 0.0
    doppler_shift: float = 0.0             # 多普勒频移
    snr: float = 0.0                       # 信噪比
    multipath_factor: float = 1.0          # 多径效应因子
    atmospheric_loss: float = 0.0          # 大气损耗


@dataclass
class N001VERadarModel:
    """N001VE雷达模型参数 - 真实物理特性"""
    # 基本性能参数
    max_detection_range: float = 120000    # 最大探测距离 120km
    max_track_range: float = 80000         # 最大跟踪距离 80km
    max_lock_range: float = 60000          # 最大锁定距离 60km
    max_simultaneous_tracks: int = 8       # 同时跟踪目标数

    # 波束参数
    search_beam_width: float = 60.0        # 搜索波束宽度 (度)
    track_beam_width: float = 3.0          # 跟踪波束宽度 (度)
    lock_beam_width: float = 1.0           # 锁定波束宽度 (度)

    # 时间参数
    scan_period: float = 1.0               # 扫描周期 (秒) - 提高扫描频率
    lock_update_rate: float = 0.1          # 锁定更新率 (秒)
    track_update_rate: float = 0.5         # 跟踪更新率 (秒)

    # 性能参数
    detection_probability_base: float = 0.9 # 基础探测概率
    track_loss_probability: float = 0.01   # 跟踪丢失概率（降低）
    lock_loss_probability: float = 0.005   # 锁定丢失概率（降低）

    # 电子战参数
    jamming_resistance: float = 0.7        # 抗干扰能力
    eccm_capability: float = 0.8           # 电子反对抗能力
    frequency_agility: bool = True         # 频率捷变能力
    sidelobe_suppression: float = 0.9      # 旁瓣抑制能力

    # 环境适应参数
    weather_degradation: float = 0.1       # 天气影响因子
    terrain_masking_threshold: float = 500.0 # 地形遮蔽阈值 (m)
    atmospheric_absorption: float = 0.001   # 大气吸收系数


class UnifiedRadarManager:
    """统一雷达管理系统 - 支持多项目复用"""

    def __init__(self):
        """初始化统一雷达管理系统"""
        # 友方雷达状态 - 保持简单的基础逻辑
        self.friendly_radar_states = {
            "A0100": RadarStatus.SEARCH,
            "A0200": RadarStatus.SEARCH
        }

        # 敌方雷达状态 - 真实N001VE雷达系统
        self.enemy_radar_states = {
            "B0100": RadarStatus.SEARCH,
            "B0200": RadarStatus.SEARCH
        }

        # N001VE雷达模型实例
        self.n001ve_radar = N001VERadarModel()

        # 敌方雷达目标跟踪
        self.enemy_radar_targets = {
            "B0100": {},
            "B0200": {}
        }

        # 敌方雷达扫描时间记录
        self.enemy_scan_times = {
            "B0100": 0.0,
            "B0200": 0.0
        }

        # 敌方雷达锁定目标
        self.enemy_lock_targets = {
            "B0100": None,
            "B0200": None
        }

        # 电子战状态
        self.ecm_states = {
            "B0100": {"active": False, "type": None, "start_time": 0.0, "duration": 0.0},
            "B0200": {"active": False, "type": None, "start_time": 0.0, "duration": 0.0}
        }

        # 环境参数
        self.environmental_conditions = {
            "weather_factor": 1.0,      # 天气因子
            "terrain_height": 0.0,      # 地形高度
            "atmospheric_density": 1.0,  # 大气密度
            "temperature": 15.0,        # 温度 (°C)
            "humidity": 50.0           # 湿度 (%)
        }

        logging.info("📡 统一雷达管理系统初始化完成")

    # ==================== 友方雷达系统方法 ====================

    def update_friendly_radar_states(self, env, current_time: float):
        """更新友方雷达状态 - 保持简单的基础逻辑"""
        for agent_id in self.friendly_radar_states.keys():
            if agent_id in env.agents and env.agents[agent_id].is_alive:
                self._update_single_friendly_radar(env, agent_id, current_time)

    def _update_single_friendly_radar(self, env, agent_id: str, current_time: float):
        """更新单个友方雷达状态 - 保持原有的基础逻辑"""
        try:
            # 找到最近的敌机
            target = self._find_closest_enemy_target(env, agent_id)
            if not target:
                self.friendly_radar_states[agent_id] = RadarStatus.SEARCH
                return

            distance = self._calculate_distance(env.agents[agent_id], target)

            # 根据距离确定雷达状态 - 保持原有的基础逻辑
            if distance > 90000:  # 90km
                self.friendly_radar_states[agent_id] = RadarStatus.SEARCH
            elif distance > 81000:  # 81km
                self.friendly_radar_states[agent_id] = RadarStatus.SEARCH
            elif distance > 45000:  # 45km
                self.friendly_radar_states[agent_id] = RadarStatus.TRACK
            else:  # < 45km
                self.friendly_radar_states[agent_id] = RadarStatus.LOCK

        except Exception as e:
            logging.error(f"❌ {agent_id} 友方雷达状态更新错误: {e}")
            self.friendly_radar_states[agent_id] = RadarStatus.SEARCH

    def get_friendly_radar_state(self, agent_id: str) -> RadarStatus:
        """获取友方雷达状态"""
        return self.friendly_radar_states.get(agent_id, RadarStatus.SEARCH)

    def _find_closest_enemy_target(self, env, agent_id: str):
        """找到最近的敌方目标"""
        try:
            agent = env.agents[agent_id]
            min_distance = float('inf')
            closest_target = None

            # 搜索敌方目标
            enemy_ids = ["B0100", "B0200"] if agent_id.startswith("A") else ["A0100", "A0200"]

            for enemy_id in enemy_ids:
                if enemy_id in env.agents and env.agents[enemy_id].is_alive:
                    distance = self._calculate_distance(agent, env.agents[enemy_id])
                    if distance < min_distance:
                        min_distance = distance
                        closest_target = env.agents[enemy_id]

            return closest_target

        except Exception as e:
            logging.error(f"❌ {agent_id} 寻找最近目标错误: {e}")
            return None

    def _calculate_distance(self, agent1, agent2) -> float:
        """计算两个智能体之间的距离"""
        try:
            pos1 = np.array(agent1.get_position())
            pos2 = np.array(agent2.get_position())
            return np.linalg.norm(pos1 - pos2)
        except Exception as e:
            logging.error(f"❌ 距离计算错误: {e}")
            return float('inf')
    
    def update_friendly_radar_states(self, env, current_time: float):
        """更新我方雷达状态 - 基础的状态转换器"""
        for agent_id in self.friendly_radar_states.keys():
            if agent_id in env.agents and env.agents[agent_id].is_alive:
                self._update_single_friendly_radar(env, agent_id, current_time)
    
    # ==================== 敌方雷达系统方法 ====================

    def update_enemy_radar_states(self, env, current_time: float):
        """更新敌方雷达状态 - 真实N001VE雷达系统"""
        for agent_id in self.enemy_radar_states.keys():
            if agent_id in env.agents and env.agents[agent_id].is_alive:
                self._update_single_enemy_radar(env, agent_id, current_time)
    
    def _update_single_enemy_radar(self, env, agent_id: str, current_time: float):
        """更新单个敌方雷达状态 - 真实N001VE雷达物理特性"""
        try:
            # 检查任务完成状态
            alive_enemies = sum(1 for eid in ["A0100", "A0200"]
                               if eid in env.agents and env.agents[eid].is_alive)

            if alive_enemies == 0:
                self.enemy_radar_states[agent_id] = RadarStatus.STANDBY
                return

            # 更新雷达扫描
            self._update_enemy_radar_scan(env, agent_id, current_time)

            # 更新目标跟踪
            self._update_enemy_target_tracking(env, agent_id, current_time)

            # 更新雷达工作模式
            self._update_enemy_radar_mode(env, agent_id, current_time)

            # 处理电子战效果
            self._process_electronic_warfare(env, agent_id, current_time)

        except Exception as e:
            logging.error(f"❌ {agent_id} 敌方雷达状态更新错误: {e}")
            self.enemy_radar_states[agent_id] = RadarStatus.SEARCH

    def _update_enemy_radar_scan(self, env, agent_id: str, current_time: float):
        """更新敌方雷达扫描 - 真实物理建模"""
        try:
            # 检查扫描周期
            if current_time - self.enemy_scan_times[agent_id] < self.n001ve_radar.scan_period:
                return

            self.enemy_scan_times[agent_id] = current_time

            # 扫描友方目标
            friendly_agents = ["A0100", "A0200"]
            for target_id in friendly_agents:
                if target_id not in env.agents or not env.agents[target_id].is_alive:
                    continue

                # 计算目标参数
                distance = self._calculate_distance(env.agents[agent_id], env.agents[target_id])
                bearing = self._calculate_bearing(env, agent_id, target_id)
                velocity = self._calculate_target_velocity(env, target_id)
                elevation = self._calculate_elevation(env, agent_id, target_id)

                # 检查探测距离
                if distance > self.n001ve_radar.max_detection_range:
                    # 移除超出距离的目标
                    if target_id in self.enemy_radar_targets[agent_id]:
                        del self.enemy_radar_targets[agent_id][target_id]
                    continue

                # 计算真实探测概率
                detection_prob = self._calculate_realistic_detection_probability(
                    distance, bearing, elevation, velocity, current_time)

                # 探测成功
                if random.random() < detection_prob:
                    if target_id not in self.enemy_radar_targets[agent_id]:
                        # 创建新目标
                        self.enemy_radar_targets[agent_id][target_id] = RadarTarget(
                            target_id=target_id,
                            distance=distance,
                            bearing=bearing,
                            elevation=elevation,
                            velocity=velocity,
                            detection_probability=detection_prob,
                            last_update=current_time,
                            doppler_shift=self._calculate_doppler_shift(velocity, bearing),
                            snr=self._calculate_snr(distance, bearing),
                            multipath_factor=self._calculate_multipath_factor(distance, elevation),
                            atmospheric_loss=self._calculate_atmospheric_loss(distance)
                        )
                        logging.debug(f"🎯 {agent_id} N001VE雷达探测到新目标: {target_id} 距离={distance/1000:.1f}km")
                    else:
                        # 更新现有目标
                        target = self.enemy_radar_targets[agent_id][target_id]
                        target.distance = distance
                        target.bearing = bearing
                        target.elevation = elevation
                        target.velocity = velocity
                        target.detection_probability = detection_prob
                        target.last_update = current_time
                        target.doppler_shift = self._calculate_doppler_shift(velocity, bearing)
                        target.snr = self._calculate_snr(distance, bearing)
                        target.multipath_factor = self._calculate_multipath_factor(distance, elevation)
                        target.atmospheric_loss = self._calculate_atmospheric_loss(distance)
                else:
                    # 探测失败，移除目标
                    if target_id in self.enemy_radar_targets[agent_id]:
                        logging.debug(f"📡 {agent_id} N001VE雷达失去目标: {target_id} 距离={distance/1000:.1f}km")
                        del self.enemy_radar_targets[agent_id][target_id]

        except Exception as e:
            logging.error(f"❌ {agent_id} 敌方雷达扫描错误: {e}")
    
    def _calculate_bearing(self, env, agent_id: str, target_id: str) -> float:
        """计算方位角"""
        try:
            agent_pos = np.array(env.agents[agent_id].get_position())
            target_pos = np.array(env.agents[target_id].get_position())

            dx = target_pos[0] - agent_pos[0]
            dy = target_pos[1] - agent_pos[1]

            bearing = math.degrees(math.atan2(dx, dy))
            return (bearing + 360) % 360  # 标准化到0-360度
        except Exception as e:
            logging.error(f"❌ 方位角计算错误: {e}")
            return 0.0

    def _calculate_elevation(self, env, agent_id: str, target_id: str) -> float:
        """计算仰角"""
        try:
            agent_pos = np.array(env.agents[agent_id].get_position())
            target_pos = np.array(env.agents[target_id].get_position())

            horizontal_distance = np.linalg.norm(agent_pos[:2] - target_pos[:2])
            height_diff = target_pos[2] - agent_pos[2]

            elevation = math.degrees(math.atan2(height_diff, horizontal_distance))
            return elevation
        except Exception as e:
            logging.error(f"❌ 仰角计算错误: {e}")
            return 0.0

    def _calculate_target_velocity(self, env, target_id: str) -> float:
        """计算目标速度"""
        try:
            target = env.agents[target_id]
            velocity = np.linalg.norm(target.get_velocity())
            return velocity
        except Exception as e:
            logging.error(f"❌ 目标速度计算错误: {e}")
            return 0.0

    def _calculate_realistic_detection_probability(self, distance: float, bearing: float,
                                                 elevation: float, velocity: float, current_time: float) -> float:
        """计算真实的N001VE雷达探测概率 - 基于真实物理模型"""
        try:
            # 基础距离衰减 - 基于雷达方程但调整为实用值
            if distance > self.n001ve_radar.max_detection_range:
                return 0.0
            elif distance > 100000:  # 100-120km：中等概率
                base_prob = 0.5
            elif distance > 80000:  # 80-100km：高概率
                base_prob = 0.8
            elif distance > 60000:  # 60-80km：极高概率
                base_prob = 0.9
            elif distance > 40000:  # 40-60km：高概率
                base_prob = 0.9
            else:  # < 40km：极高概率
                base_prob = 0.95

            # 雷达截面积因子（F-16约5m²）
            rcs_factor = min(1.0, math.log10(5.0 + 1) / 2.0)

            # 角度因子 - 基于波束宽度（提高最小值）
            angle_factor = max(0.7, 1.0 - abs(bearing) / (self.n001ve_radar.search_beam_width / 2))

            # 仰角因子 - 低仰角性能更好（提高最小值）
            elevation_factor = max(0.8, 1.0 - abs(elevation) / 45.0)

            # 多普勒因子 - 高速目标更容易探测
            doppler_factor = min(1.2, 1.0 + velocity / 500.0)

            # 大气衰减因子（减少衰减影响）
            atmospheric_factor = max(0.8, 1.0 - (distance / self.n001ve_radar.max_detection_range) *
                                   self.n001ve_radar.atmospheric_absorption * 500)

            # 天气影响
            weather_factor = self.environmental_conditions["weather_factor"]

            # 综合探测概率
            total_prob = (base_prob * rcs_factor * angle_factor * elevation_factor *
                         doppler_factor * atmospheric_factor * weather_factor)

            return min(1.0, max(0.0, total_prob))

        except Exception as e:
            logging.error(f"❌ 雷达探测概率计算错误: {e}")
            return 0.5
    
    def _calculate_doppler_shift(self, velocity: float, bearing: float) -> float:
        """计算多普勒频移"""
        try:
            # 雷达频率 (X波段，约10GHz)
            radar_frequency = 10e9  # Hz
            c = 3e8  # 光速 m/s

            # 径向速度分量
            radial_velocity = velocity * math.cos(math.radians(bearing))

            # 多普勒频移 = 2 * f0 * vr / c
            doppler_shift = 2 * radar_frequency * radial_velocity / c

            return doppler_shift
        except Exception as e:
            logging.error(f"❌ 多普勒频移计算错误: {e}")
            return 0.0

    def _calculate_snr(self, distance: float, bearing: float) -> float:
        """计算信噪比"""
        try:
            # 基础SNR (dB) - N001VE雷达参数
            base_snr = 40.0

            # 距离衰减 (R^4 law)
            distance_loss = -40 * math.log10(distance / 10000)

            # 角度损失
            angle_loss = -3 * (abs(bearing) / (self.n001ve_radar.search_beam_width / 2))

            # 大气损失
            atmospheric_loss = -0.1 * (distance / 1000)  # 0.1 dB/km

            total_snr = base_snr + distance_loss + angle_loss + atmospheric_loss
            return total_snr
        except Exception as e:
            logging.error(f"❌ SNR计算错误: {e}")
            return 0.0
    
    def _calculate_multipath_factor(self, distance: float, elevation: float) -> float:
        """计算多径效应因子"""
        try:
            # 低仰角时多径效应更明显
            if abs(elevation) < 5.0:  # 5度以下
                multipath_loss = 0.3 * (5.0 - abs(elevation)) / 5.0
                return max(0.7, 1.0 - multipath_loss)
            else:
                return 1.0
        except Exception as e:
            logging.error(f"❌ 多径效应计算错误: {e}")
            return 1.0

    def _calculate_atmospheric_loss(self, distance: float) -> float:
        """计算大气损耗"""
        try:
            # 大气损耗 (dB) = 距离(km) * 损耗系数
            loss_coefficient = 0.1  # dB/km
            atmospheric_loss = (distance / 1000) * loss_coefficient
            return atmospheric_loss
        except Exception as e:
            logging.error(f"❌ 大气损耗计算错误: {e}")
            return 0.0
    
    def _update_enemy_target_tracking(self, env, agent_id: str, current_time: float):
        """更新敌方目标跟踪"""
        try:
            # 清理过期目标
            expired_targets = []
            for target_id, target in self.enemy_radar_targets[agent_id].items():
                if current_time - target.last_update > 10.0:  # 10秒未更新
                    expired_targets.append(target_id)

            for target_id in expired_targets:
                del self.enemy_radar_targets[agent_id][target_id]
                logging.debug(f"📡 {agent_id} 清理过期目标: {target_id}")

            # 更新跟踪质量
            for target_id, target in self.enemy_radar_targets[agent_id].items():
                # 基于距离和时间更新跟踪质量
                time_factor = min(1.0, (current_time - target.last_update) / 5.0)
                distance_factor = max(0.1, 1.0 - target.distance / self.n001ve_radar.max_track_range)
                target.track_quality = distance_factor * (1.0 - time_factor)

                # 检查跟踪丢失
                if (target.track_quality < 0.2 or
                    target.distance > self.n001ve_radar.max_track_range or
                    random.random() < self.n001ve_radar.track_loss_probability):
                    expired_targets.append(target_id)

            # 移除跟踪丢失的目标
            for target_id in expired_targets:
                if target_id in self.enemy_radar_targets[agent_id]:
                    del self.enemy_radar_targets[agent_id][target_id]
                    logging.debug(f"📡 {agent_id} 跟踪丢失目标: {target_id}")

        except Exception as e:
            logging.error(f"❌ {agent_id} 目标跟踪更新错误: {e}")
    
    def _update_enemy_radar_mode(self, env, agent_id: str, current_time: float):
        """更新敌方雷达工作模式"""
        try:
            targets = self.enemy_radar_targets[agent_id]

            if not targets:
                self.enemy_radar_states[agent_id] = RadarStatus.SEARCH
                self.enemy_lock_targets[agent_id] = None
                return

            # 找到最高优先级目标
            best_target = None
            best_priority = -1

            for target_id, target in targets.items():
                # 计算目标优先级（距离越近，优先级越高）
                priority = (1.0 / max(target.distance, 1000)) * target.track_quality
                if priority > best_priority:
                    best_priority = priority
                    best_target = target

            if not best_target:
                self.enemy_radar_states[agent_id] = RadarStatus.SEARCH
                self.enemy_lock_targets[agent_id] = None
                return

            # 根据距离和跟踪质量确定雷达模式
            if (best_target.distance <= self.n001ve_radar.max_lock_range and
                best_target.track_quality > 0.6):
                self.enemy_radar_states[agent_id] = RadarStatus.LOCK
                self.enemy_lock_targets[agent_id] = best_target.target_id
                best_target.lock_time = current_time
            elif (best_target.distance <= self.n001ve_radar.max_track_range and
                  best_target.track_quality > 0.3):
                self.enemy_radar_states[agent_id] = RadarStatus.TRACK
                if self.enemy_lock_targets[agent_id] == best_target.target_id:
                    self.enemy_lock_targets[agent_id] = None
            else:
                self.enemy_radar_states[agent_id] = RadarStatus.SEARCH
                self.enemy_lock_targets[agent_id] = None

        except Exception as e:
            logging.error(f"❌ {agent_id} 雷达模式更新错误: {e}")
            self.enemy_radar_states[agent_id] = RadarStatus.SEARCH
    
    def _process_electronic_warfare(self, env, agent_id: str, current_time: float):
        """处理电子战效果"""
        try:
            ecm_state = self.ecm_states[agent_id]

            # 检查ECM状态
            if ecm_state["active"]:
                # 检查ECM持续时间
                if current_time - ecm_state["start_time"] > ecm_state["duration"]:
                    ecm_state["active"] = False
                    ecm_state["type"] = None
                    logging.debug(f"🛡️ {agent_id} ECM结束")
                else:
                    # ECM激活期间，降低探测概率
                    for target_id in list(self.enemy_radar_targets[agent_id].keys()):
                        target = self.enemy_radar_targets[agent_id][target_id]
                        # 根据ECM类型调整探测概率
                        if ecm_state["type"] == ECMType.NOISE_JAMMING:
                            target.detection_probability *= 0.3
                        elif ecm_state["type"] == ECMType.DECEPTION_JAMMING:
                            target.detection_probability *= 0.5
                        elif ecm_state["type"] == ECMType.CHAFF:
                            target.detection_probability *= 0.2
            else:
                # 随机激活ECM
                if random.random() < 0.01:  # 1%概率每次更新
                    self._activate_ecm(agent_id, current_time)

        except Exception as e:
            logging.error(f"❌ {agent_id} 电子战处理错误: {e}")

    def _activate_ecm(self, agent_id: str, current_time: float):
        """激活电子对抗措施"""
        try:
            ecm_types = [ECMType.NOISE_JAMMING, ECMType.DECEPTION_JAMMING, ECMType.CHAFF]
            ecm_type = random.choice(ecm_types)
            duration = random.uniform(5.0, 15.0)  # 5-15秒

            self.ecm_states[agent_id] = {
                "active": True,
                "type": ecm_type,
                "start_time": current_time,
                "duration": duration
            }

            logging.debug(f"🛡️ {agent_id} 激活ECM: {ecm_type.value} 持续{duration:.1f}秒")

        except Exception as e:
            logging.error(f"❌ {agent_id} ECM激活错误: {e}")
    
    # ==================== 统一接口方法 ====================

    def get_friendly_radar_state(self, agent_id: str) -> RadarStatus:
        """获取友方雷达状态"""
        return self.friendly_radar_states.get(agent_id, RadarStatus.SEARCH)

    def get_enemy_radar_state(self, agent_id: str) -> RadarStatus:
        """获取敌方雷达状态"""
        return self.enemy_radar_states.get(agent_id, RadarStatus.SEARCH)

    def get_enemy_radar_targets(self, agent_id: str) -> Dict[str, RadarTarget]:
        """获取敌方雷达目标"""
        return self.enemy_radar_targets.get(agent_id, {})

    def get_enemy_lock_target(self, agent_id: str) -> Optional[str]:
        """获取敌方锁定目标"""
        return self.enemy_lock_targets.get(agent_id, None)

    def is_enemy_ecm_active(self, agent_id: str) -> bool:
        """检查敌方ECM是否激活"""
        return self.ecm_states.get(agent_id, {}).get("active", False)

    def get_enemy_ecm_type(self, agent_id: str) -> Optional[ECMType]:
        """获取敌方ECM类型"""
        return self.ecm_states.get(agent_id, {}).get("type", None)
    
    def update_environmental_conditions(self, weather_factor: float = 1.0,
                                       terrain_height: float = 0.0,
                                       atmospheric_density: float = 1.0,
                                       temperature: float = 15.0,
                                       humidity: float = 50.0):
        """更新环境条件"""
        self.environmental_conditions.update({
            "weather_factor": weather_factor,
            "terrain_height": terrain_height,
            "atmospheric_density": atmospheric_density,
            "temperature": temperature,
            "humidity": humidity
        })
        logging.debug(f"📡 环境条件已更新: 天气因子={weather_factor}, 地形高度={terrain_height}m")

    def get_radar_performance_summary(self) -> Dict[str, Any]:
        """获取雷达性能摘要"""
        summary = {
            "friendly_radars": {
                agent_id: {
                    "status": status.value,
                    "type": "AN/APG-68(V)9"
                }
                for agent_id, status in self.friendly_radar_states.items()
            },
            "enemy_radars": {
                agent_id: {
                    "status": status.value,
                    "type": "N001VE",
                    "targets_tracked": len(self.enemy_radar_targets.get(agent_id, {})),
                    "lock_target": self.enemy_lock_targets.get(agent_id, None),
                    "ecm_active": self.ecm_states.get(agent_id, {}).get("active", False)
                }
                for agent_id, status in self.enemy_radar_states.items()
            },
            "n001ve_specs": {
                "max_detection_range_km": self.n001ve_radar.max_detection_range / 1000,
                "max_track_range_km": self.n001ve_radar.max_track_range / 1000,
                "max_lock_range_km": self.n001ve_radar.max_lock_range / 1000,
                "max_simultaneous_tracks": self.n001ve_radar.max_simultaneous_tracks,
                "jamming_resistance": self.n001ve_radar.jamming_resistance
            }
        }
        return summary

    def record_radar_data(self, env, current_time: float) -> List[Dict[str, Any]]:
        """记录雷达数据到CSV格式 - 兼容性方法"""
        radar_data = []

        try:
            # 记录友方雷达数据 - 基础状态
            for agent_id, radar_state in self.friendly_radar_states.items():
                if hasattr(env, 'agents') and agent_id in env.agents and env.agents[agent_id].is_alive:
                    # 找到目标
                    target_id = self._find_closest_target_id(env, agent_id)
                    target_distance = self._get_target_distance(env, agent_id, target_id) if target_id else 0.0

                    radar_data.append({
                        'Time_s': current_time,
                        'Agent_ID': agent_id,
                        'Radar_Type': 'AN/APG-68(V)9',
                        'Status': radar_state.value,
                        'Target_ID': target_id or 'None',
                        'Target_Distance_km': target_distance / 1000.0,
                        'Side': 'Friendly'
                    })

            # 记录敌方雷达数据 - 详细数据
            for agent_id, radar_state in self.enemy_radar_states.items():
                if hasattr(env, 'agents') and agent_id in env.agents and env.agents[agent_id].is_alive:
                    # 获取雷达目标信息
                    targets = self.enemy_radar_targets.get(agent_id, {})
                    lock_target = self.enemy_lock_targets.get(agent_id, None)

                    if targets:
                        # 记录每个目标的雷达数据
                        for target_id, target in targets.items():
                            radar_data.append({
                                'Time_s': current_time,
                                'Agent_ID': agent_id,
                                'Radar_Type': 'N001VE',
                                'Status': radar_state.value,
                                'Target_ID': target_id,
                                'Target_Distance_km': target.distance / 1000.0,
                                'SNR_dB': target.snr,
                                'Doppler_Shift_m_s': target.doppler_shift,
                                'Lock_Quality': target.track_quality,
                                'Detection_Probability': target.detection_probability,
                                'Beam_Angle_deg': np.rad2deg(target.bearing),
                                'Side': 'Enemy',
                                'Lock_Target': target_id == lock_target
                            })
                    else:
                        # 无目标时记录搜索状态
                        radar_data.append({
                            'Time_s': current_time,
                            'Agent_ID': agent_id,
                            'Radar_Type': 'N001VE',
                            'Status': 'SEARCH',
                            'Target_ID': 'None',
                            'Target_Distance_km': 0.0,
                            'SNR_dB': 0.0,
                            'Doppler_Shift_m_s': 0.0,
                            'Lock_Quality': 0.0,
                            'Detection_Probability': 0.0,
                            'Beam_Angle_deg': 0.0,
                            'Side': 'Enemy',
                            'Lock_Target': False
                        })

        except Exception as e:
            logging.error(f"❌ 雷达数据记录错误: {e}")

        return radar_data

    def _find_closest_target_id(self, env, agent_id: str) -> Optional[str]:
        """找到最近的目标ID"""
        try:
            if not hasattr(env, 'agents') or agent_id not in env.agents:
                return None

            agent = env.agents[agent_id]
            min_distance = float('inf')
            closest_target_id = None

            # 搜索敌方目标
            enemy_ids = ["B0100", "B0200"] if agent_id.startswith("A") else ["A0100", "A0200"]

            for enemy_id in enemy_ids:
                if enemy_id in env.agents and env.agents[enemy_id].is_alive:
                    distance = self._calculate_distance(agent, env.agents[enemy_id])
                    if distance < min_distance:
                        min_distance = distance
                        closest_target_id = enemy_id

            return closest_target_id

        except Exception as e:
            logging.error(f"❌ {agent_id} 寻找最近目标错误: {e}")
            return None

    def _get_target_distance(self, env, agent_id: str, target_id: str) -> float:
        """获取目标距离"""
        try:
            if (not target_id or not hasattr(env, 'agents') or
                agent_id not in env.agents or target_id not in env.agents):
                return 0.0

            return self._calculate_distance(env.agents[agent_id], env.agents[target_id])

        except Exception as e:
            logging.error(f"❌ 获取目标距离错误: {e}")
            return 0.0


# ==================== 全局实例管理 ====================

# 全局统一雷达管理器实例
_unified_radar_manager = None

def get_unified_radar_manager() -> UnifiedRadarManager:
    """获取全局统一雷达管理器实例"""
    global _unified_radar_manager
    if _unified_radar_manager is None:
        _unified_radar_manager = UnifiedRadarManager()
        logging.info("📡 全局统一雷达管理器实例已创建")
    return _unified_radar_manager

def update_all_radars(env, current_time: float):
    """更新所有雷达状态 - 统一接口"""
    radar_manager = get_unified_radar_manager()
    radar_manager.update_friendly_radar_states(env, current_time)
    radar_manager.update_enemy_radar_states(env, current_time)

def get_friendly_radar_state(agent_id: str) -> RadarStatus:
    """获取友方雷达状态 - 统一接口"""
    radar_manager = get_unified_radar_manager()
    return radar_manager.get_friendly_radar_state(agent_id)

def get_enemy_radar_state(agent_id: str) -> RadarStatus:
    """获取敌方雷达状态 - 统一接口"""
    radar_manager = get_unified_radar_manager()
    return radar_manager.get_enemy_radar_state(agent_id)

def get_enemy_radar_targets(agent_id: str) -> Dict[str, RadarTarget]:
    """获取敌方雷达目标 - 统一接口"""
    radar_manager = get_unified_radar_manager()
    return radar_manager.get_enemy_radar_targets(agent_id)

def get_enemy_lock_target(agent_id: str) -> Optional[str]:
    """获取敌方锁定目标 - 统一接口"""
    radar_manager = get_unified_radar_manager()
    return radar_manager.get_enemy_lock_target(agent_id)

def is_enemy_ecm_active(agent_id: str) -> bool:
    """检查敌方ECM是否激活 - 统一接口"""
    radar_manager = get_unified_radar_manager()
    return radar_manager.is_enemy_ecm_active(agent_id)

def get_radar_performance_summary() -> Dict[str, Any]:
    """获取雷达性能摘要 - 统一接口"""
    radar_manager = get_unified_radar_manager()
    return radar_manager.get_radar_performance_summary()

# ==================== 向后兼容接口 ====================

# 为了保持向后兼容性，保留旧的接口名称
RadarManager = UnifiedRadarManager
get_radar_manager = get_unified_radar_manager

def get_friendly_radar_states() -> Dict[str, str]:
    """获取我方雷达状态"""
    radar_manager = get_radar_manager()
    return radar_manager.get_friendly_radar_states()

def get_enemy_radar_states() -> Dict[str, str]:
    """获取敌方雷达状态"""
    radar_manager = get_radar_manager()
    return radar_manager.get_enemy_radar_states()

def get_enemy_radar_data() -> Dict[str, Dict[str, Any]]:
    """获取敌方雷达数据"""
    radar_manager = get_radar_manager()
    return radar_manager.get_enemy_radar_data() 