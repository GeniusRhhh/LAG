import logging
import numpy as np
from typing import Dict, Any


class RadarModel:
    """增强版雷达模型，支持精细建模以配合14种战术模板。"""

    def __init__(self, max_range: float = 120000, h_beamwidth: float = 60, v_beamwidth: float = 30):
        """初始化雷达模型。

        Args:
            max_range: 最大探测距离（米）(default: 120km).
            h_beamwidth: 水平波束宽度（度）(default: 60).
            v_beamwidth: 垂直波束宽度（度）(default: 30).
        """
        self.max_range = max_range
        self.h_beamwidth = np.deg2rad(h_beamwidth)
        self.v_beamwidth = np.deg2rad(v_beamwidth)

        # 雷达性能参数
        self.snr_threshold = 8.0  # 信噪比阈值 (dB)
        self.doppler_threshold = 5.0  # 多普勒速度阈值 (m/s)
        self.lock_time = 0
        self.lock_duration = 30  # 锁定持续时间 (steps)

        # 雷达工作模式
        self.radar_modes = {
            "search": {"power": 1.0, "range_multiplier": 1.0, "angle_accuracy": 2.0},
            "track": {"power": 0.8, "range_multiplier": 0.9, "angle_accuracy": 0.5},
            "illuminate": {"power": 1.2, "range_multiplier": 1.1, "angle_accuracy": 0.3},
            "jam_resist": {"power": 0.6, "range_multiplier": 0.7, "angle_accuracy": 3.0}
        }
        self.current_mode = "search"

        # 地面杂波模型
        self.ground_clutter = {
            "altitude_threshold": 1000,  # 地面杂波影响高度 (m)
            "clutter_power": -20,  # 杂波功率 (dB)
            "terrain_factor": 1.0  # 地形因子
        }

        # ECM对抗能力
        self.ecm_resistance = {
            "jam_threshold": 15.0,  # 干扰阈值 (dB)
            "burnthrough_factor": 0.3,  # 烧穿因子
            "agility": 0.8  # 频率捷变能力
        }

        # logging.info(f"Enhanced RadarModel initialized: max_range={max_range}m, "
        #              f"beamwidth={h_beamwidth}°×{v_beamwidth}°, SNR_threshold={self.snr_threshold}dB")

    def get_radar_state(self, state: Dict[str, Any], env, agent_id: str) -> Dict[str, Any]:
        """获取雷达状态。

        Args:
            state: 状态字典。
            env: 环境实例。
            agent_id: 智能体标识符。

        Returns:
            Dict: 雷达状态 (radar_lock, has_warning, snr, doppler_shift等).
        """
        if not hasattr(env, 'agents') or agent_id not in env.agents:
            return self._default_radar_state()

        agent = env.agents[agent_id]
        if not agent.is_alive:
            return self._default_radar_state()

        # 获取基础参数
        enemy_distance = state.get("enemy_distance", np.inf)
        enemy_angle_off = np.deg2rad(state.get("enemy_angle_off", 0))
        missile_distance = state.get("missile_distance", np.inf)
        enemy_velocity = state.get("enemy_velocity", 340.0)
        current_altitude = state.get("current_altitude", 5000)

        # 雷达锁定判断
        basic_conditions = (
                enemy_distance <= 100000 and  # 100km内可探测
                enemy_distance >= 3000 and  # 最小距离
                abs(enemy_angle_off) < np.radians(120) and  # 放宽到120度
                current_altitude > 1000  # 基本高度要求
        )

        # 多普勒判断,计算径向速度分量
        try:
            doppler_shift = self.calculate_doppler_shift(env, agent_id)
            # 大幅放宽多普勒阈值
            doppler_detectable = abs(doppler_shift) > 5  # 从5.0放宽到10.0
        except:
            doppler_detectable = True  # 如果计算失败，默认可检测

        # SNR计算
        try:
            snr = self.calculate_snr(enemy_distance, enemy_angle_off, enemy_velocity, current_altitude)
            ground_clutter_effect = self.calculate_ground_clutter_effect(current_altitude, enemy_distance)
            ecm_effect = self.calculate_ecm_effect(env, agent_id)
            effective_snr = snr - ground_clutter_effect - ecm_effect
            snr_acceptable = effective_snr > 3.0  # 从8.0降低到3.0
        except:
            snr_acceptable = True  # 如果计算失败，默认可接受

        # 雷达锁定条件
        radar_lock = (
                basic_conditions and
                (snr_acceptable or enemy_distance < 55000) and  # 近距离时忽略SNR
                (doppler_detectable or enemy_distance < 45000)  # 近距离时忽略多普勒
        )

        # 导弹威胁警告
        has_warning = self.detect_missile_threat(env, agent_id, missile_distance)

        # 更新锁定时间
        if radar_lock:
            self.lock_time += 1
        else:
            self.lock_time = max(0, self.lock_time - 1)  # 衰减

        lock_stable = self.lock_time >= 3  # 稳定锁定需要5步

        radar_state = {
            "radar_lock": radar_lock and lock_stable,
            "has_warning": has_warning,
            "snr": effective_snr,
            "doppler_shift": doppler_shift,
            "ground_clutter": ground_clutter_effect,
            "ecm_effect": ecm_effect,
            "lock_quality": min(self.lock_time / self.lock_duration, 1.0),
            "beam_angle": enemy_angle_off,
            "radar_mode": self.current_mode,
            "detection_range": self._get_effective_range(),
            "angle_accuracy": self.radar_modes[self.current_mode]["angle_accuracy"]
        }

        # 战术模板相关的雷达状态
        radar_state.update(self._get_tactical_radar_state(state, radar_state))

        if env.current_step % 500 == 0:  # 减少日志频率
            logging.info(f"Agent {agent_id} radar: lock={radar_lock}, SNR={effective_snr:.1f}dB, "
                          f"doppler={doppler_shift:.1f}m/s, clutter={ground_clutter_effect:.1f}dB, "
                          f"ecm={ecm_effect:.1f}dB, distance={enemy_distance:.0f}m")

        return radar_state

    def calculate_snr(self, distance: float, angle_off: float, velocity: float, altitude: float) -> float:
        """计算信噪比。"""
        if distance <= 0:
            return -np.inf

        # 基础雷达方程：SNR = P_t * G^2 * λ^2 * σ / ((4π)^3 * R^4 * k * T * B * F)
        base_snr = 45.0  # 基础SNR (dB)
        # 距离衰减 (R^4 law)
        distance_attenuation = -40 * np.log10(distance / 10000)
        # 角度衰减（天线方向图）
        angle_attenuation = -12 * (angle_off / (self.h_beamwidth / 2)) ** 2
        # 速度影响（多普勒增强）
        velocity_enhancement = 3 * np.log10(max(velocity / 340, 0.1))
        # 高度影响（大气折射）
        altitude_factor = -2 * np.log10(max(altitude / 10000, 0.1))
        # 雷达模式影响
        mode_factor = 10 * np.log10(self.radar_modes[self.current_mode]["power"])
        total_snr = (base_snr + distance_attenuation + angle_attenuation +
                     velocity_enhancement + altitude_factor + mode_factor)
        return total_snr

    def calculate_doppler_shift(self, env, agent_id: str) -> float:
        """计算多普勒频移。"""
        if not hasattr(env, 'agents') or agent_id not in env.agents:
            return 0.0

        agent = env.agents[agent_id]
        enemies = agent.enemies

        if not enemies or not enemies[0].is_alive:
            return 0.0

        # 计算径向速度
        ego_vel = agent.get_velocity()
        enemy_vel = enemies[0].get_velocity()
        relative_pos = enemies[0].get_position() - agent.get_position()

        if np.linalg.norm(relative_pos) == 0:
            return 0.0

        # 径向速度分量
        relative_vel = enemy_vel - ego_vel
        unit_los = relative_pos / np.linalg.norm(relative_pos)
        radial_velocity = np.dot(relative_vel, unit_los)

        # 多普勒频移 (假设X波段雷达，频率约10GHz)
        c = 3e8  # 光速
        frequency = 10e9  # 10GHz
        doppler_shift = 2 * frequency * radial_velocity / c

        return radial_velocity  # 返回径向速度而非频移

    def calculate_ground_clutter_effect(self, altitude: float, enemy_distance: float) -> float:
        """计算地面杂波影响。"""
        if altitude > self.ground_clutter["altitude_threshold"]:
            return 0.0

        # 地面杂波功率随高度和距离变化
        clutter_power = self.ground_clutter["clutter_power"]
        altitude_factor = (self.ground_clutter["altitude_threshold"] - altitude) / self.ground_clutter[
            "altitude_threshold"]
        distance_factor = max(0, 1 - enemy_distance / 50000)  # 50km内有杂波影响
        clutter_effect = abs(clutter_power) * altitude_factor * distance_factor * self.ground_clutter["terrain_factor"]
        return clutter_effect

    def calculate_ecm_effect(self, env, agent_id: str) -> float:
        """计算电子对抗影响。"""
        if not hasattr(env, 'agents') or agent_id not in env.agents:
            return 0.0

        agent = env.agents[agent_id]
        enemies = agent.enemies

        if not enemies:
            return 0.0

        # 简化的ECM模型：基于距离和敌机能力
        ecm_effect = 0.0
        for enemy in enemies:
            if enemy.is_alive:
                distance = np.linalg.norm(enemy.get_position() - agent.get_position())
                # 假设ECM功率随距离衰减
                if distance < 30000:  # 30km内有ECM影响
                    jam_power = 20 * (1 - distance / 30000)  # 最大20dB干扰
                    if jam_power > self.ecm_resistance["jam_threshold"]:
                        ecm_effect += (jam_power - self.ecm_resistance["jam_threshold"]) * (
                                    1 - self.ecm_resistance["agility"])

        return min(ecm_effect, 25.0)  # 最大25dB ECM影响

    def detect_missile_threat(self, env, agent_id: str, missile_distance: float) -> bool:
        """检测导弹威胁。"""
        if not hasattr(env, 'agents') or agent_id not in env.agents:
            return False

        agent = env.agents[agent_id]
        missile_sim = agent.check_missile_warning()

        if missile_sim is None:
            return False

        distance = np.linalg.norm(missile_sim.get_position() - agent.get_position())
        velocity = np.linalg.norm(missile_sim.get_velocity())

        # 威胁判断：基于距离、速度和接近率
        threat_range = 60000  # 60km威胁距离
        threat_velocity = 500  # 500m/s威胁速度

        is_threat = (distance < threat_range and velocity > threat_velocity)

        if is_threat:
            logging.debug(f"Agent {agent_id} missile threat detected: distance={distance:.1f}m, "
                          f"velocity={velocity:.1f}m/s")

        return is_threat

    def _get_effective_range(self) -> float:
        """获取有效探测距离。"""
        return self.max_range * self.radar_modes[self.current_mode]["range_multiplier"]

    def _get_tactical_radar_state(self, state: Dict[str, Any], radar_state: Dict[str, Any]) -> Dict[str, Any]:
        """获取战术相关的雷达状态。"""
        enemy_distance = state.get("enemy_distance", np.inf)

        # Beam机动相关
        beam_optimal = (
                abs(radar_state["doppler_shift"]) < self.doppler_threshold and
                abs(radar_state["beam_angle"]) > np.deg2rad(85)
        )

        # Notch机动相关
        notch_effective = (
                radar_state["ground_clutter"] > 5.0 and
                abs(radar_state["beam_angle"]) > np.deg2rad(80)
        )

        # Crank机动相关
        crank_optimal = (
                radar_state["radar_lock"] and
                30 <= np.rad2deg(abs(radar_state["beam_angle"])) <= 60
        )

    def _default_radar_state(self) -> Dict[str, Any]:
        """返回默认雷达状态
        Returns:
            Dict: 默认状态
        """
        return {
            "radar_lock": False,
            "has_warning": False,
            "snr": 0.0,
            "doppler_shift": 0.0,
            "ground_clutter": 0.0,
            "ecm_effect": 0.0,
            "lock_quality": 0.0,
            "beam_angle": 0.0,
            "radar_mode": self.current_mode,
            "detection_range": self._get_effective_range(),
            "angle_accuracy": self.radar_modes[self.current_mode]["angle_accuracy"],
            "beam_optimal": False,
            "notch_effective": False,
            "crank_optimal": False
        }