"""Helper functions extracted from radar_manager.py."""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from .radar_types import APG68RadarModel, ECMType, RadarStatus
except ImportError:
    from radar_types import APG68RadarModel, ECMType, RadarStatus


def calculate_distance(manager, agent1, agent2) -> float:
    try:
        pos1 = np.array(agent1.get_position())
        pos2 = np.array(agent2.get_position())
        return np.linalg.norm(pos1 - pos2)
    except Exception as exc:
        logging.error(f"❌ 距离计算错误: {exc}")
        return float("inf")


def calculate_bearing(manager, agent, target) -> float:
    try:
        agent_pos = np.array(agent.get_position())
        target_pos = np.array(target.get_position())
        dx = target_pos[0] - agent_pos[0]
        dy = target_pos[1] - agent_pos[1]
        bearing = math.degrees(math.atan2(dx, dy))
        return (bearing + 360) % 360
    except Exception as exc:
        logging.error(f"❌ 方位角计算错误: {exc}")
        return 0.0


def calculate_elevation(manager, agent, target) -> float:
    try:
        agent_pos = np.array(agent.get_position())
        target_pos = np.array(target.get_position())
        horizontal_distance = np.linalg.norm(agent_pos[:2] - target_pos[:2])
        height_diff = target_pos[2] - agent_pos[2]
        return math.degrees(math.atan2(height_diff, horizontal_distance))
    except Exception as exc:
        logging.error(f"❌ 仰角计算错误: {exc}")
        return 0.0


def calculate_dynamic_rcs(manager, agent, target, target_id: str) -> float:
    try:
        baseline_rcs = (
            manager.aircraft_rcs_baseline["F16"]
            if target_id.startswith("A")
            else manager.aircraft_rcs_baseline["Su27"]
        )
        aspect_angle = manager._calculate_aspect_angle(agent, target)
        aspect_rad = math.radians(abs(aspect_angle))
        if aspect_rad <= math.pi / 2:
            horizontal_factor = 1.0 + 1.5 * math.sin(aspect_rad)
        else:
            horizontal_factor = 2.5 - 1.3 * math.sin(aspect_rad)

        try:
            target_pitch = target.get_rpy()[1]
            pitch_factor = 1.0 + 0.3 * abs(math.sin(target_pitch))
        except Exception:
            pitch_factor = 1.0

        try:
            velocity_mag = np.linalg.norm(target.get_velocity())
            config_factor = 1.4 if velocity_mag < 150 else 1.2
        except Exception:
            config_factor = 1.2

        dynamic_rcs = baseline_rcs * horizontal_factor * pitch_factor * config_factor
        logging.debug(
            f"📡 动态RCS: {target_id} 基准={baseline_rcs:.1f}m² "
            f"视角={aspect_angle:.0f}° 系数={horizontal_factor:.2f} "
            f"最终={dynamic_rcs:.1f}m²"
        )
        return dynamic_rcs
    except Exception as exc:
        logging.error(f"❌ 动态RCS计算错误: {exc}")
        key = "Su27" if target_id.startswith("B") else "F16"
        return manager.aircraft_rcs_baseline.get(key, 5.0)


def calculate_aspect_angle(manager, agent, target) -> float:
    try:
        agent_pos = np.array(agent.get_position())
        target_pos = np.array(target.get_position())
        to_radar_vec = agent_pos - target_pos
        to_radar_vec_2d = to_radar_vec[:2]
        try:
            target_heading = target.get_rpy()[2]
            target_heading_vec = np.array([math.cos(target_heading), math.sin(target_heading)])
        except Exception:
            target_vel = np.array(target.get_velocity()[:2])
            if np.linalg.norm(target_vel) > 1.0:
                target_heading_vec = target_vel / np.linalg.norm(target_vel)
            else:
                return 90.0

        to_radar_norm = to_radar_vec_2d / (np.linalg.norm(to_radar_vec_2d) + 1e-6)
        cos_angle = np.dot(target_heading_vec, to_radar_norm)
        return math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))
    except Exception as exc:
        logging.error(f"❌ 视角计算错误: {exc}")
        return 90.0


def calculate_radial_velocity(manager, agent, target) -> float:
    try:
        agent_pos = np.array(agent.get_position())
        target_pos = np.array(target.get_position())
        target_vel = np.array(target.get_velocity())
        los_vec = agent_pos - target_pos
        los_distance = np.linalg.norm(los_vec)
        if los_distance < 1.0:
            return 0.0
        los_unit = los_vec / los_distance
        return float(np.dot(target_vel, los_unit))
    except Exception as exc:
        logging.error(f"❌ 径向速度计算错误: {exc}")
        return 0.0


def check_notch_condition(manager, radial_velocity: float, radar_model) -> bool:
    del manager
    notch_threshold = 50.0 if isinstance(radar_model, APG68RadarModel) else 80.0
    return abs(radial_velocity) < notch_threshold


def calculate_ground_clutter_factor(
    manager,
    elevation: float,
    altitude: float,
    distance: float,
    radar_model,
) -> float:
    del manager, distance
    try:
        if altitude > 100 or elevation >= 0:
            return 1.0
        altitude_factor = (100 - altitude) / 100.0
        angle_factor = abs(elevation) / 45.0
        base_clutter = 0.3 + 0.5 * altitude_factor * angle_factor
        clutter_suppression = 0.7 if isinstance(radar_model, APG68RadarModel) else 0.4
        clutter_factor = 1.0 - base_clutter * (1.0 - clutter_suppression)
        return max(0.1, clutter_factor)
    except Exception as exc:
        logging.error(f"❌ 地面杂波计算错误: {exc}")
        return 1.0


def check_terrain_masking(
    manager,
    target_altitude: float,
    distance: float,
    elevation: float,
    radar_model,
) -> bool:
    try:
        terrain_height = manager.environmental_conditions.get("terrain_height", 0.0)
        masking_threshold = radar_model.terrain_masking_threshold
        target_relative_height = target_altitude - terrain_height
        if elevation >= 0:
            return False
        dynamic_threshold = masking_threshold * (1.0 + distance / 50000.0)
        return target_relative_height < dynamic_threshold
    except Exception as exc:
        logging.error(f"❌ 地形遮蔽检查错误: {exc}")
        return False


def calculate_target_velocity(manager, env, target_id: str) -> float:
    del manager
    try:
        target = env.agents[target_id]
        return float(np.linalg.norm(target.get_velocity()))
    except Exception as exc:
        logging.error(f"❌ 目标速度计算错误: {exc}")
        return 0.0


def calculate_doppler_shift(manager, velocity: float, bearing: float) -> float:
    del manager
    try:
        radar_frequency = 10e9
        light_speed = 3e8
        radial_velocity = velocity * math.cos(math.radians(bearing))
        return 2 * radar_frequency * radial_velocity / light_speed
    except Exception as exc:
        logging.error(f"❌ 多普勒频移计算错误: {exc}")
        return 0.0


def calculate_snr(manager, distance: float, bearing: float, radar_model=None) -> float:
    try:
        radar_model = radar_model or manager.n001ve_radar
        base_snr = 42.0 if isinstance(radar_model, APG68RadarModel) else 40.0
        distance_loss = -40 * math.log10(distance / 10000)
        beam_width = radar_model.search_beam_width
        angle_loss = -3 * (abs(bearing) / (beam_width / 2))
        atmospheric_loss = -0.1 * (distance / 1000)
        return base_snr + distance_loss + angle_loss + atmospheric_loss
    except Exception as exc:
        logging.error(f"❌ SNR计算错误: {exc}")
        return 0.0


def calculate_multipath_factor(manager, distance: float, elevation: float) -> float:
    del manager, distance
    try:
        if abs(elevation) < 5.0:
            multipath_loss = 0.3 * (5.0 - abs(elevation)) / 5.0
            return max(0.7, 1.0 - multipath_loss)
        return 1.0
    except Exception as exc:
        logging.error(f"❌ 多径效应计算错误: {exc}")
        return 1.0


def calculate_atmospheric_loss(manager, distance: float) -> float:
    del manager
    try:
        return (distance / 1000) * 0.1
    except Exception as exc:
        logging.error(f"❌ 大气损耗计算错误: {exc}")
        return 0.0


def is_being_jammed(manager, agent_id: str) -> bool:
    try:
        return any(manager.ecm_states.get(enemy_id, {}).get("active", False) for enemy_id in _opponent_ids(agent_id))
    except Exception as exc:
        logging.error(f"❌ 查询 {agent_id} 受干扰状态错误: {exc}")
        return False


def get_jamming_sources(manager, agent_id: str) -> Dict[str, ECMType]:
    try:
        return {
            enemy_id: manager.ecm_states.get(enemy_id, {}).get("type")
            for enemy_id in _opponent_ids(agent_id)
            if manager.ecm_states.get(enemy_id, {}).get("active", False)
        }
    except Exception as exc:
        logging.error(f"❌ 获取 {agent_id} 干扰源错误: {exc}")
        return {}


def update_environmental_conditions(
    manager,
    weather_factor: float = 1.0,
    terrain_height: float = 0.0,
    atmospheric_density: float = 1.0,
    temperature: float = 15.0,
    humidity: float = 50.0,
):
    manager.environmental_conditions.update(
        {
            "weather_factor": weather_factor,
            "terrain_height": terrain_height,
            "atmospheric_density": atmospheric_density,
            "temperature": temperature,
            "humidity": humidity,
        }
    )
    logging.debug(f"📡 环境条件已更新: 天气因子={weather_factor}, 地形高度={terrain_height}m")


def get_radar_performance_summary(manager) -> Dict[str, Any]:
    return {
        "friendly_radars": {
            agent_id: {
                "status": status.value,
                "type": "AN/APG-68(V)9",
                "targets_tracked": len(manager.friendly_radar_targets.get(agent_id, {})),
                "lock_target": manager.friendly_lock_targets.get(agent_id, None),
                "ecm_active": manager.ecm_states.get(agent_id, {}).get("active", False),
            }
            for agent_id, status in manager.friendly_radar_states.items()
        },
        "enemy_radars": {
            agent_id: {
                "status": status.value,
                "type": "N001VE",
                "targets_tracked": len(manager.enemy_radar_targets.get(agent_id, {})),
                "lock_target": manager.enemy_lock_targets.get(agent_id, None),
                "ecm_active": manager.ecm_states.get(agent_id, {}).get("active", False),
            }
            for agent_id, status in manager.enemy_radar_states.items()
        },
        "apg68_specs": {
            "max_detection_range_km": manager.apg68_radar.max_detection_range / 1000,
            "max_track_range_km": manager.apg68_radar.max_track_range / 1000,
            "max_lock_range_km": manager.apg68_radar.max_lock_range / 1000,
            "max_simultaneous_tracks": manager.apg68_radar.max_simultaneous_tracks,
            "max_simultaneous_engagement": manager.apg68_radar.max_simultaneous_engagement,
            "scan_period_s": manager.apg68_radar.scan_period,
            "jamming_resistance": manager.apg68_radar.jamming_resistance,
            "has_look_down_shoot_down": manager.apg68_radar.has_look_down_shoot_down,
        },
        "n001ve_specs": {
            "max_detection_range_km": manager.n001ve_radar.max_detection_range / 1000,
            "max_track_range_km": manager.n001ve_radar.max_track_range / 1000,
            "max_lock_range_km": manager.n001ve_radar.max_lock_range / 1000,
            "max_simultaneous_tracks": manager.n001ve_radar.max_simultaneous_tracks,
            "max_simultaneous_engagement": manager.n001ve_radar.max_simultaneous_engagement,
            "scan_period_s": manager.n001ve_radar.scan_period,
            "jamming_resistance": manager.n001ve_radar.jamming_resistance,
        },
    }


def record_radar_data(manager, env, current_time: float) -> List[Dict[str, Any]]:
    radar_data: List[Dict[str, Any]] = []
    try:
        _append_side_records(
            manager=manager,
            env=env,
            current_time=current_time,
            radar_data=radar_data,
            states=manager.friendly_radar_states,
            targets_by_agent=manager.friendly_radar_targets,
            lock_targets=manager.friendly_lock_targets,
            radar_type="AN/APG-68(V)9",
            side="Friendly",
        )
        _append_side_records(
            manager=manager,
            env=env,
            current_time=current_time,
            radar_data=radar_data,
            states=manager.enemy_radar_states,
            targets_by_agent=manager.enemy_radar_targets,
            lock_targets=manager.enemy_lock_targets,
            radar_type="N001VE",
            side="Enemy",
        )
    except Exception as exc:
        logging.error(f"❌ 雷达数据记录错误: {exc}")
    return radar_data


def find_closest_target_id(manager, env, agent_id: str) -> Optional[str]:
    try:
        if not hasattr(env, "agents") or agent_id not in env.agents:
            return None
        agent = env.agents[agent_id]
        min_distance = float("inf")
        closest_target_id = None
        enemy_ids = ["B0100", "B0200"] if agent_id.startswith("A") else ["A0100", "A0200"]
        for enemy_id in enemy_ids:
            if enemy_id in env.agents and env.agents[enemy_id].is_alive:
                distance = manager._calculate_distance(agent, env.agents[enemy_id])
                if distance < min_distance:
                    min_distance = distance
                    closest_target_id = enemy_id
        return closest_target_id
    except Exception as exc:
        logging.error(f"❌ {agent_id} 寻找最近目标错误: {exc}")
        return None


def get_target_distance(manager, env, agent_id: str, target_id: str) -> float:
    try:
        if (
            not target_id
            or not hasattr(env, "agents")
            or agent_id not in env.agents
            or target_id not in env.agents
        ):
            return 0.0
        return manager._calculate_distance(env.agents[agent_id], env.agents[target_id])
    except Exception as exc:
        logging.error(f"❌ 获取目标距离错误: {exc}")
        return 0.0


def _opponent_ids(agent_id: str) -> List[str]:
    return (
        ["B0100", "B0200", "B0300", "B0400"]
        if agent_id.startswith("A")
        else ["A0100", "A0200", "A0300", "A0400"]
    )


def _jamming_snapshot(manager, agent_id: str) -> Tuple[bool, str]:
    is_jammed = manager.is_being_jammed(agent_id)
    jamming_sources = manager.get_jamming_sources(agent_id)
    if not is_jammed or not jamming_sources:
        return is_jammed, ""
    jamming_list = [f"{src}:{ecm_type.value}" for src, ecm_type in jamming_sources.items()]
    return is_jammed, ",".join(jamming_list)


def _append_side_records(
    manager,
    env,
    current_time: float,
    radar_data: List[Dict[str, Any]],
    states: Dict[str, RadarStatus],
    targets_by_agent: Dict[str, Dict[str, Any]],
    lock_targets: Dict[str, Optional[str]],
    radar_type: str,
    side: str,
):
    if not hasattr(env, "agents"):
        return
    for agent_id, radar_state in states.items():
        if agent_id not in env.agents or not env.agents[agent_id].is_alive:
            continue
        targets = targets_by_agent.get(agent_id, {})
        lock_target = lock_targets.get(agent_id, None)
        is_jammed, jamming_info = _jamming_snapshot(manager, agent_id)
        if targets:
            for target_id, target in targets.items():
                radar_data.append(
                    {
                        "Time_s": current_time,
                        "Agent_ID": agent_id,
                        "Radar_Type": radar_type,
                        "Status": radar_state.value,
                        "Target_ID": target_id,
                        "Target_Distance_km": target.distance / 1000.0,
                        "SNR_dB": target.snr,
                        "Doppler_Shift_m_s": target.doppler_shift,
                        "Lock_Quality": target.track_quality,
                        "Detection_Probability": target.detection_probability,
                        "Beam_Angle_deg": target.bearing,
                        "Side": side,
                        "Lock_Target": target_id == lock_target,
                        "Being_Jammed": is_jammed,
                        "Jamming_Sources": jamming_info,
                    }
                )
            continue

        radar_data.append(
            {
                "Time_s": current_time,
                "Agent_ID": agent_id,
                "Radar_Type": radar_type,
                "Status": "SEARCH",
                "Target_ID": "None",
                "Target_Distance_km": 0.0,
                "SNR_dB": 0.0,
                "Doppler_Shift_m_s": 0.0,
                "Lock_Quality": 0.0,
                "Detection_Probability": 0.0,
                "Beam_Angle_deg": 0.0,
                "Side": side,
                "Lock_Target": False,
                "Being_Jammed": is_jammed,
                "Jamming_Sources": jamming_info,
            }
        )
