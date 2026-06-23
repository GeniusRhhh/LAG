"""Extracted maneuver and command-shaping helpers for UnifiedEnemyTacticalAI."""

from __future__ import annotations

import logging
import random
from typing import Optional, Tuple

import numpy as np

try:
    from envs.JSBSim.core.catalog import Catalog as c
except ImportError:
    class MockCatalog:
        attitude_psi_rad = "attitude/psi-rad"
        attitude_phi_rad = "attitude/phi-rad"
        attitude_theta_rad = "attitude/theta-rad"
        position_h_sl_m = "position/h-sl-m"
        velocities_vc_mps = "velocities/vc-mps"
        velocities_v_down_fps = "velocities/v-down-fps"

    c = MockCatalog()

try:
    from .enemy_ai_types import ActionParameters, ActionType
except ImportError:
    from enemy_ai_types import ActionParameters, ActionType


def _calculate_safe_altitude_change(self, current_altitude: Optional[float], action_name: str) -> float:
    """Return a conservative altitude delta for action parameter generation."""
    if current_altitude is None:
        defaults = {
            "dive_escape": -200.0,
            "chaff_flare": -100.0,
            "aggressive_approach": 300.0,
            "defensive_split": 250.0,
        }
        return defaults.get(action_name, 0.0)

    altitude_m = float(current_altitude)

    if action_name in ("dive_escape", "chaff_flare"):
        if altitude_m >= 8000.0:
            return -300.0
        if altitude_m >= 5000.0:
            return -150.0
        if altitude_m >= 3500.0:
            return -50.0
        return 150.0

    if action_name == "aggressive_approach":
        if altitude_m >= 12000.0:
            return 200.0
        if altitude_m >= 6000.0:
            return 300.0
        return 500.0

    if action_name == "defensive_split":
        if altitude_m >= 12000.0:
            return 200.0
        if altitude_m >= 6000.0:
            return 300.0
        return 400.0

    return 0.0


def _generate_action_parameters(self, action_type: ActionType, agent_id: str) -> ActionParameters:
    """Generate randomized action parameters with altitude-aware safeguards."""
    try:
        current_altitude = None
        if hasattr(self, '_current_env') and self._current_env and agent_id in self._current_env.agents:
            try:
                current_altitude = self._current_env.agents[agent_id].get_property_value(c.position_h_sl_m)
            except Exception:
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
                turn_angle=random.choice([75.0, 80.0, 85.0]),
                turn_rate=random.uniform(4.0, 7.0)
            )

        elif action_type == ActionType.BEAM_MANEUVER:
            return ActionParameters(
                duration=random.uniform(15.0, 30.0),
                turn_angle=random.choice([70.0, 75.0, 80.0, -70.0, -75.0, -80.0]),
                turn_rate=random.uniform(3.0, 5.0)
            )

        elif action_type == ActionType.DIVE_ESCAPE:
            altitude_change = self._calculate_safe_altitude_change(current_altitude, "dive_escape")
            return ActionParameters(
                duration=random.uniform(8.0, 15.0),
                altitude_change=altitude_change,
                turn_angle=random.uniform(-30.0, 30.0),
                turn_rate=random.uniform(4.0, 6.0)
            )

        elif action_type == ActionType.CHAFF_FLARE_MANEUVER:
            altitude_change = self._calculate_safe_altitude_change(current_altitude, "chaff_flare")
            return ActionParameters(
                duration=random.uniform(10.0, 20.0),
                turn_angle=random.choice([45.0, -45.0, 60.0, -60.0]),
                turn_rate=random.uniform(3.0, 6.0),
                altitude_change=altitude_change
            )

        elif action_type == ActionType.SPIRAL_DIVE:
            return ActionParameters(
                duration=random.uniform(10.0, 15.0),
                turn_angle=random.choice([60.0, -60.0]),
                turn_rate=random.uniform(3.0, 5.0),
                altitude_change=0.0
            )

        elif action_type == ActionType.SHORT_SKATE:
            return ActionParameters(
                duration=random.uniform(35.0, 50.0),
                turn_angle=random.uniform(35.0, 45.0),
                turn_rate=random.uniform(5.0, 8.0)
            )

        elif action_type == ActionType.AGGRESSIVE_APPROACH:
            altitude_change = self._calculate_safe_altitude_change(current_altitude, "aggressive_approach")
            if altitude_change < 0:
                altitude_change = random.uniform(200.0, 500.0)
            return ActionParameters(
                duration=random.uniform(20.0, 40.0),
                velocity_change=random.uniform(20.0, 50.0),
                altitude_change=altitude_change
            )

        elif action_type == ActionType.DEFENSIVE_SPLIT:
            altitude_change = self._calculate_safe_altitude_change(current_altitude, "defensive_split")
            if altitude_change < 0:
                altitude_change = random.uniform(200.0, 400.0)
            return ActionParameters(
                duration=random.uniform(15.0, 25.0),
                turn_angle=random.uniform(35.0, 60.0),
                altitude_change=altitude_change
            )

        elif action_type == ActionType.RETURN_TO_BASE:
            return ActionParameters(
                duration=float('inf'),
                target_heading=0.0
            )

        elif action_type == ActionType.CLIMB:
            return ActionParameters(
                duration=random.uniform(10.0, 20.0),
                altitude_change=random.uniform(200.0, 500.0)
            )

        elif action_type == ActionType.DESCEND:
            return ActionParameters(
                duration=random.uniform(10.0, 20.0),
                altitude_change=random.uniform(200.0, 500.0)
            )

        else:
            return ActionParameters(duration=10.0)

    except Exception as e:
        logging.error(f"动作参数生成失败 {action_type.value}: {e}")
        return ActionParameters(duration=10.0)


def _execute_maintain_heading(self, env, agent_id: str) -> Tuple[int, int, int]:
    """Execute maintain-heading action."""
    target_heading = None
    params = self.action_parameters.get(agent_id)
    if params is not None:
        target_heading = params.target_heading

    if target_heading is None:
        try:
            target_heading = float(
                np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad)) % 360.0
            )
        except Exception:
            target_heading = 180.0

    return self._maintain_heading_precise(env, agent_id, float(target_heading))


def _execute_turn(self, env, agent_id: str, turn_angle: float, turn_rate: float) -> Tuple[int, int, int]:
    """Execute turn action."""
    try:
        if turn_angle is None:
            turn_angle = 0.0
        if turn_rate is None:
            turn_rate = 5.0

        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        target_heading = (current_heading + turn_angle) % 360.0

        heading_diff = ((target_heading - current_heading + 540) % 360) - 180
        max_turn = turn_rate * 0.2
        actual_turn = np.clip(heading_diff, -max_turn, max_turn)
        final_heading = (current_heading + actual_turn) % 360.0

        return self._maintain_heading_precise(env, agent_id, final_heading)
    except Exception as e:
        logging.error(f"动作执行失败 {agent_id} - turn: {e}")
        return 7, 8, 4


def _execute_crank(self, env, agent_id: str, crank_angle: float) -> Tuple[int, int, int]:
    """Execute crank maneuver."""
    try:
        situation = self.situation_data.get(agent_id)
        if situation and situation.closest_enemy_bearing is not None:
            target_bearing = situation.closest_enemy_bearing
            crank_heading = (target_bearing + crank_angle) % 360.0
        else:
            current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
            crank_heading = (current_heading + crank_angle) % 360.0

        return self._maintain_heading_precise(env, agent_id, crank_heading)
    except Exception as e:
        logging.error(f"动作执行失败 {agent_id} - crank: {e}")
        return 7, 8, 4


def _execute_notch_maneuver(self, env, agent_id: str) -> Tuple[int, int, int]:
    """Execute notch maneuver."""
    current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
    current_velocity = np.linalg.norm(env.agents[agent_id].get_velocity())

    MINIMUM_SAFE_ALTITUDE = 1000.0
    MINIMUM_SAFE_VELOCITY = 150.0

    if current_altitude < MINIMUM_SAFE_ALTITUDE:
        logging.warning(f"⚠️ {agent_id} Notch机动时高度过低，执行紧急爬升")
        return 7, 0, 3

    if current_velocity < MINIMUM_SAFE_VELOCITY:
        logging.debug(f"⚠️ {agent_id} Notch机动时速度过低，执行加速")
        return 7, 8, 1

    situation = self.situation_data.get(agent_id)
    if situation and situation.missile_threats:
        closest_missile = min(situation.missile_threats, key=lambda x: x['distance'])
        missile_pos = closest_missile['position']
        current_pos = env.agents[agent_id].get_position()

        dx = missile_pos[0] - current_pos[0]
        dy = missile_pos[1] - current_pos[1]
        missile_bearing = np.rad2deg(np.arctan2(dy, dx))

        notch_angle = 45.0 if random.random() > 0.5 else -45.0
        notch_heading = (missile_bearing + notch_angle) % 360.0
    else:
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        notch_angle = 45.0 if random.random() > 0.5 else -45.0
        notch_heading = (current_heading + notch_angle) % 360.0

    return self._maintain_heading_precise(env, agent_id, notch_heading)


def _execute_beam_maneuver(self, env, agent_id: str) -> Tuple[int, int, int]:
    """Execute beam maneuver."""
    try:
        situation = self.situation_data.get(agent_id)
        if situation and situation.closest_enemy_bearing is not None:
            enemy_bearing = situation.closest_enemy_bearing
            beam_angle = random.choice([85.0, 90.0, 95.0, -85.0, -90.0, -95.0])
            beam_heading = (enemy_bearing + beam_angle) % 360.0
        else:
            current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
            beam_angle = random.choice([90.0, -90.0])
            beam_heading = (current_heading + beam_angle) % 360.0

        return self._maintain_heading_precise(env, agent_id, beam_heading)
    except Exception as e:
        logging.error(f"动作执行失败 {agent_id} - beam_maneuver: {e}")
        return 7, 8, 4


def _execute_dive_escape(self, env, agent_id: str) -> Tuple[int, int, int]:
    """Execute dive-escape maneuver with additional safety guards."""
    try:
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)

        aircraft_params = self._get_aircraft_parameters(env, agent_id)
        min_safe_altitude = aircraft_params["min_altitude"] * 1.8

        if current_altitude < min_safe_altitude:
            if not hasattr(self, '_dive_escape_blocked') or agent_id not in self._dive_escape_blocked:
                if not hasattr(self, '_dive_escape_blocked'):
                    self._dive_escape_blocked = {}
                self._dive_escape_blocked[agent_id] = True
                logging.warning(f"🛡️ {agent_id} 高度{current_altitude:.0f}m过低，俯冲脱离改为Beam机动")
            turn_angle = random.choice([90.0, -90.0])
            target_heading = (current_heading + turn_angle) % 360.0
            return self._maintain_heading_with_altitude_speed(env, agent_id, target_heading, 9, 6)
        else:
            if hasattr(self, '_dive_escape_blocked') and agent_id in self._dive_escape_blocked:
                del self._dive_escape_blocked[agent_id]

        turn_angle = random.uniform(-20.0, 20.0)
        target_heading = (current_heading + turn_angle) % 360.0

        max_dive_depth = min(200.0, current_altitude - min_safe_altitude)
        if max_dive_depth <= 50.0:
            logging.info(f"🛡️ {agent_id} 俯冲空间不足（{max_dive_depth:.0f}m），改为水平机动")
            return self._maintain_heading_with_altitude_speed(env, agent_id, target_heading, 7, 6)

        dive_altitude = random.uniform(50.0, min(150.0, max_dive_depth))
        target_altitude = current_altitude - dive_altitude
        target_altitude = max(target_altitude, min_safe_altitude + 500.0)

        actual_dive = current_altitude - target_altitude
        if actual_dive < 30.0:
            logging.debug(f"🛡️ {agent_id} 俯冲幅度过小（{actual_dive:.0f}m），改为水平机动")
            return self._maintain_heading_with_altitude_speed(env, agent_id, target_heading, 7, 6)

        logging.debug(f"{agent_id}执行安全俯冲脱离: 转弯{turn_angle:.1f}°, 俯冲{actual_dive:.0f}m")
        return self._maintain_heading_with_altitude_speed(env, agent_id, target_heading, 5, 6)
    except Exception as e:
        logging.error(f"动作执行失败 {agent_id} - dive_escape: {e}")
        return 7, 8, 4


def _execute_chaff_flare_maneuver(self, env, agent_id: str) -> Tuple[int, int, int]:
    """Execute chaff-flare integrated maneuver."""
    try:
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        current_velocity = env.agents[agent_id].get_property_value(c.velocities_vc_mps)

        perf = self._get_aircraft_parameters(env, agent_id)
        min_speed = perf.get('min_speed', 120.0)
        if current_velocity < min_speed * 1.15:
            return self._maintain_heading_with_altitude_speed(
                env,
                agent_id,
                current_heading,
                8,
                6,
            )

        if not hasattr(self, '_chaff_flare_last_step'):
            self._chaff_flare_last_step = {}
        last_step = self._chaff_flare_last_step.get(agent_id, -9999)
        if env.current_step - last_step < 8:
            return self._maintain_heading_with_altitude_speed(
                env,
                agent_id,
                current_heading,
                8,
                5,
            )
        self._chaff_flare_last_step[agent_id] = env.current_step

        turn_angle = random.choice([60.0, -60.0, 75.0, -75.0])
        target_heading = (current_heading + turn_angle) % 360.0

        altitude_change = random.choice([7, 8, 9])
        logging.info(f"🛡️ {agent_id} 高度{current_altitude:.0f}m，干扰弹机动使用安全高度变化")

        speed_change = 6

        logging.info(f"敌方{agent_id}执行干扰弹机动: 转弯{turn_angle:.1f}°, 高度变化={altitude_change}")
        return self._maintain_heading_with_altitude_speed(env, agent_id, target_heading, altitude_change, speed_change)
    except Exception as e:
        logging.error(f"动作执行失败 {agent_id} - chaff_flare_maneuver: {e}")
        return 7, 8, 4


def _execute_spiral_dive(self, env, agent_id: str) -> Tuple[int, int, int]:
    """Execute safe replacement for legacy spiral-dive maneuver."""
    try:
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)

        spiral_angle = random.choice([180.0, -180.0, 360.0, -360.0])
        target_heading = (current_heading + spiral_angle) % 360.0

        safe_altitude_threshold = 5000.0
        if current_altitude < safe_altitude_threshold:
            altitude_change = 9
            logging.info(f"🛡️ {agent_id} 高度较低，螺旋转弯配合爬升")
        else:
            altitude_change = random.choice([7, 8, 9])

        return self._maintain_heading_with_altitude_speed(env, agent_id, target_heading, altitude_change, 4)
    except Exception as e:
        logging.error(f"动作执行失败 {agent_id} - spiral_dive: {e}")
        return 7, 8, 4


def _execute_short_skate_unified(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
    """Execute three-stage short-skate maneuver."""
    try:
        if not hasattr(self, 'short_skate_states'):
            self.short_skate_states = {}

        if agent_id not in self.short_skate_states:
            self._init_short_skate_unified(agent_id, current_time)

        state = self.short_skate_states[agent_id]
        phase_time = current_time - state['phase_start_time']

        crank_duration = random.uniform(6.0, 12.0)
        turn_cold_duration = random.uniform(15.0, 25.0)

        if state['phase'] == 'crank':
            if phase_time < crank_duration:
                crank_heading = (state['initial_heading'] + state['crank_angle']) % 360.0
                return self._maintain_heading_precise(env, agent_id, crank_heading)
            else:
                state['phase'] = 'turn_cold'
                state['phase_start_time'] = current_time
                logging.debug(f"敌方{agent_id} Short Skate: Crank → Turn Cold")

        if state['phase'] == 'turn_cold':
            if phase_time < turn_cold_duration:
                turn_cold_heading = (state['initial_heading'] + state['turn_cold_angle']) % 360.0
                return self._maintain_heading_precise(env, agent_id, turn_cold_heading)
            else:
                state['phase'] = 'escape'
                state['phase_start_time'] = current_time
                logging.debug(f"敌方{agent_id} Short Skate: Turn Cold → Escape")

        if state['phase'] == 'escape':
            escape_heading = (state['initial_heading'] + state['turn_cold_angle']) % 360.0
            return self._maintain_heading_with_speed(env, agent_id, escape_heading, 5)

        return self._maintain_heading_precise(env, agent_id, 180.0)

    except Exception as e:
        logging.error(f"Short Skate执行失败 {agent_id}: {e}")
        return 7, 8, 4


def _init_short_skate_unified(self, agent_id: str, current_time: float):
    """Initialize state for short-skate maneuver."""
    current_heading = 180.0

    if agent_id == "B0100":
        crank_angle = random.uniform(-45.0, -25.0)
        turn_cold_angle = random.uniform(-120.0, -80.0)
    elif agent_id == "B0200":
        crank_angle = random.uniform(25.0, 45.0)
        turn_cold_angle = random.uniform(80.0, 120.0)
    else:
        side = random.choice([-1, 1])
        crank_angle = side * random.uniform(25.0, 45.0)
        turn_cold_angle = side * random.uniform(80.0, 120.0)

    self.short_skate_states[agent_id] = {
        'phase': 'crank',
        'phase_start_time': current_time,
        'initial_heading': current_heading,
        'crank_angle': crank_angle,
        'turn_cold_angle': turn_cold_angle,
    }


def _execute_aggressive_approach(self, env, agent_id: str) -> Tuple[int, int, int]:
    """Execute aggressive approach maneuver."""
    try:
        situation = self.situation_data.get(agent_id)
        if situation and situation.closest_enemy_bearing is not None:
            target_heading = situation.closest_enemy_bearing
        else:
            target_heading = 180.0

        style = self._get_opening_style(agent_id)
        if style == "offset_left":
            target_heading = (target_heading - random.uniform(12.0, 30.0)) % 360.0
        elif style == "offset_right":
            target_heading = (target_heading + random.uniform(12.0, 30.0)) % 360.0
        elif style == "bracket":
            left_member = agent_id in ("B0100", "B0300")
            bracket_offset = random.uniform(16.0, 34.0)
            target_heading = (target_heading - bracket_offset) % 360.0 if left_member else (target_heading + bracket_offset) % 360.0

        heading_jitter = random.uniform(-8.0, 8.0) * self._behavior_diversity
        altitude_cmd = 8 if random.random() < (0.45 * self._behavior_diversity) else 9
        speed_cmd = random.choice([4, 5, 5])
        return self._maintain_heading_with_altitude_speed(
            env,
            agent_id,
            (target_heading + heading_jitter) % 360.0,
            altitude_cmd,
            speed_cmd,
        )
    except Exception as e:
        logging.error(f"动作执行失败 {agent_id} - aggressive_approach: {e}")
        return 7, 8, 4


def _maintain_heading_precise(self, env, agent_id: str, target_heading: float) -> Tuple[int, int, int]:
    """Maintain heading precisely with turn-induced altitude compensation."""
    try:
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))

        heading_diff = ((target_heading - current_heading + 540) % 360) - 180
        if abs(heading_diff) < 2.0:
            heading_cmd_id = 8
        elif heading_diff > 0:
            heading_cmd_id = 12 if abs(heading_diff) > 20 else (11 if abs(heading_diff) > 10 else 10)
        else:
            heading_cmd_id = 4 if abs(heading_diff) > 20 else (5 if abs(heading_diff) > 10 else 6)

        alt_cmd = 7
        roll_rad = abs(env.agents[agent_id].get_property_value(c.attitude_phi_rad))
        if heading_cmd_id != 8 and roll_rad > np.deg2rad(15):
            if roll_rad > np.deg2rad(45):
                alt_cmd = 9
            elif roll_rad > np.deg2rad(30):
                alt_cmd = 8

        vsi = env.agents[agent_id].get_property_value(c.velocities_v_down_fps)
        if vsi > 30:
            alt_cmd = max(alt_cmd, 9)

        return alt_cmd, heading_cmd_id, 3

    except Exception as e:
        logging.error(f"航向保持失败 {agent_id}: {e}")
        return 7, 8, 3


def _maintain_heading_with_speed(self, env, agent_id: str, target_heading: float, speed_cmd: int) -> Tuple[int, int, int]:
    """Maintain heading and adjust speed."""
    altitude_cmd, heading_cmd, _ = self._maintain_heading_precise(env, agent_id, target_heading)
    return altitude_cmd, heading_cmd, speed_cmd


def _maintain_heading_with_altitude(self, env, agent_id: str, target_heading: float, altitude_cmd: int) -> Tuple[int, int, int]:
    """Maintain heading and adjust altitude."""
    _, heading_cmd, speed_cmd = self._maintain_heading_precise(env, agent_id, target_heading)
    return altitude_cmd, heading_cmd, speed_cmd


def _maintain_heading_with_altitude_speed(
    self,
    env,
    agent_id: str,
    target_heading: float,
    altitude_cmd: int,
    speed_cmd: int,
) -> Tuple[int, int, int]:
    """Maintain heading and adjust altitude plus speed."""
    _, heading_cmd, _ = self._maintain_heading_precise(env, agent_id, target_heading)
    return altitude_cmd, heading_cmd, speed_cmd
