"""Extracted helper methods for UnifiedEnemyTacticalAI.

This module keeps logic identical while reducing class file size.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from envs.JSBSim.core.catalog import Catalog as c
except ImportError:
    class MockCatalog:
        attitude_psi_rad = "attitude/psi-rad"
        attitude_pitch_rad = "attitude/pitch-rad"
        attitude_phi_rad = "attitude/phi-rad"
        attitude_theta_rad = "attitude/theta-rad"
        position_h_sl_m = "position/h-sl-m"
        position_h_sl_ft = "position/h-sl-ft"
        position_lat_geod_deg = "position/lat-geod-deg"
        position_long_gc_deg = "position/long-gc-deg"
        velocities_u_mps = "velocities/u-mps"
        velocities_v_mps = "velocities/v-mps"
        velocities_w_mps = "velocities/w-mps"
        velocities_vc_mps = "velocities/vc-mps"
        velocities_v_down_mps = "velocities/v-down-mps"
        attitude_heading_true_rad = "attitude/heading-true-rad"
        aero_alpha_deg = "aero/alpha-deg"
        fcs_throttle_cmd_norm = "fcs/throttle-cmd-norm"
        delta_altitude = "delta_altitude"
        delta_heading = "delta_heading"
        delta_velocities_u = "delta_velocities_u"
        target_heading_deg = "target_heading_deg"
        target_altitude_ft = "target_altitude_ft"
        target_velocities_u_mps = "target_velocities_u_mps"

    c = MockCatalog()

try:
    from .enemy_ai_types import (
        ActionType,
        EnemyTacticalPhase,
        SituationData,
        TacticalMode,
        ThreatAssessment,
        ThreatLevel,
    )
except ImportError:
    from enemy_ai_types import (
        ActionType,
        EnemyTacticalPhase,
        SituationData,
        TacticalMode,
        ThreatAssessment,
        ThreatLevel,
    )

def _get_opening_style(self, agent_id: str) -> str:
    """为每架敌机锁定一次开局风格，避免每回合初始机动同质化。"""
    if agent_id not in self._opening_style:
        style_weights = {
            "press": 0.38,       # 主动压迫接敌
            "offset_left": 0.21, # 左偏接敌
            "offset_right": 0.21,# 右偏接敌
            "bracket": 0.20,     # 交错夹击
        }
        pick = random.random() * sum(style_weights.values())
        cdf = 0.0
        selected_style = "press"
        for style, weight in style_weights.items():
            cdf += weight
            if pick <= cdf:
                selected_style = style
                break
        self._opening_style[agent_id] = selected_style
    return self._opening_style[agent_id]

def _opening_phase_action_weights(
    self,
    agent_id: str,
    situation: Optional[SituationData],
    current_phase: EnemyTacticalPhase,
    current_time: float,
) -> Optional[Dict[ActionType, float]]:
    """远距/开局阶段使用风格化权重，提升首段轨迹与编队几何多样性。"""
    if not self._long_range_mix_enabled or situation is None or situation.missile_threats:
        return None

    opening_window_s = float(os.getenv("ENEMY_OPENING_WINDOW_S", "120"))
    if current_time > opening_window_s:
        return None

    if current_phase == EnemyTacticalPhase.NLT_MELD and situation.min_enemy_distance > 100000:
        base = {
            ActionType.AGGRESSIVE_APPROACH: 0.52,
            ActionType.MAINTAIN_HEADING: 0.12,
            ActionType.CRANK_LEFT: 0.14,
            ActionType.CRANK_RIGHT: 0.14,
            ActionType.TURN_LEFT: 0.04,
            ActionType.TURN_RIGHT: 0.04,
        }
    elif current_phase == EnemyTacticalPhase.MELD_MTR and situation.min_enemy_distance > 70000:
        base = {
            ActionType.AGGRESSIVE_APPROACH: 0.44,
            ActionType.MAINTAIN_HEADING: 0.11,
            ActionType.CRANK_LEFT: 0.18,
            ActionType.CRANK_RIGHT: 0.18,
            ActionType.BEAM_MANEUVER: 0.06,
            ActionType.SHORT_SKATE: 0.03,
        }
    else:
        return None

    style = self._get_opening_style(agent_id)
    if style == "press":
        base[ActionType.AGGRESSIVE_APPROACH] *= 1.22
        base[ActionType.SHORT_SKATE] = base.get(ActionType.SHORT_SKATE, 0.02) * 1.20
    elif style == "offset_left":
        base[ActionType.CRANK_LEFT] *= 1.65
        base[ActionType.TURN_LEFT] = base.get(ActionType.TURN_LEFT, 0.03) * 1.35
        base[ActionType.CRANK_RIGHT] *= 0.70
    elif style == "offset_right":
        base[ActionType.CRANK_RIGHT] *= 1.65
        base[ActionType.TURN_RIGHT] = base.get(ActionType.TURN_RIGHT, 0.03) * 1.35
        base[ActionType.CRANK_LEFT] *= 0.70
    elif style == "bracket":
        # 同风格下通过单双号实现左右异构，减少“一字型”观感。
        side_is_left = int(agent_id[2]) % 2 == 1 if len(agent_id) >= 3 and agent_id[2].isdigit() else (agent_id in ("B0100", "B0300"))
        if side_is_left:
            base[ActionType.CRANK_LEFT] *= 1.80
            base[ActionType.CRANK_RIGHT] *= 0.55
        else:
            base[ActionType.CRANK_RIGHT] *= 1.80
            base[ActionType.CRANK_LEFT] *= 0.55
        base[ActionType.MAINTAIN_HEADING] *= 0.75

    if self._behavior_diversity > 0.0:
        jitter_scale = 0.18 * self._behavior_diversity
        for action in list(base.keys()):
            base[action] = max(1e-6, base[action] * (1.0 + random.uniform(-jitter_scale, jitter_scale)))

    return base

def _should_interrupt_current_action_for_missile(self, agent_id: str) -> bool:
    """导弹威胁出现时允许打断长动作，避免“明明被锁还继续按原动作飞”。"""
    threat = self.threat_assessment.get(agent_id)
    situation = self.situation_data.get(agent_id)
    if threat is None or situation is None or not situation.missile_threats:
        return False

    try:
        closest_missile_distance = min(m['distance'] for m in situation.missile_threats)
    except Exception:
        closest_missile_distance = float("inf")

    interrupt_range_m = float(os.getenv("ENEMY_MISSILE_INTERRUPT_RANGE_M", "32000"))
    if closest_missile_distance <= interrupt_range_m:
        return True

    return threat.threat_level in (ThreatLevel.CRITICAL, ThreatLevel.SEVERE)

def _apply_missile_evasion_override(
    self,
    agent_id: str,
    action_weights: Dict[ActionType, float],
    current_time: float,
) -> Dict[ActionType, float]:
    """导弹威胁下建立短时规避窗口，显式提升规避动作优先级。"""
    situation = self.situation_data.get(agent_id)
    threat = self.threat_assessment.get(agent_id)
    if situation is None or threat is None:
        return action_weights

    if situation.missile_threats:
        evasion_window_s = float(os.getenv("ENEMY_MISSILE_EVASION_WINDOW_S", "12"))
        self._missile_evasion_until[agent_id] = max(
            self._missile_evasion_until.get(agent_id, 0.0),
            current_time + evasion_window_s,
        )

    if current_time > self._missile_evasion_until.get(agent_id, 0.0):
        return action_weights

    adjusted = action_weights.copy()
    evasion_boost = {
        ActionType.NOTCH_MANEUVER: 3.2,
        ActionType.BEAM_MANEUVER: 2.8,
        ActionType.DIVE_ESCAPE: 2.3,
        ActionType.CHAFF_FLARE_MANEUVER: 2.0,
        ActionType.DEFENSIVE_SPLIT: 1.7,
        ActionType.SHORT_SKATE: 1.4,
    }
    for action, multiplier in evasion_boost.items():
        if action in adjusted:
            adjusted[action] *= multiplier

    for action in (ActionType.AGGRESSIVE_APPROACH, ActionType.MAINTAIN_HEADING):
        if action in adjusted:
            adjusted[action] *= 0.35

    if threat.threat_level in (ThreatLevel.CRITICAL, ThreatLevel.SEVERE):
        if ActionType.RETURN_TO_BASE in adjusted and self._enemy_rtb_enabled():
            adjusted[ActionType.RETURN_TO_BASE] *= 0.55

    return adjusted

def _apply_anti_repeat_diversity(
    self,
    agent_id: str,
    action_weights: Dict[ActionType, float],
) -> Dict[ActionType, float]:
    """抑制连续重复同一动作，增加全过程动作序列变化。"""
    history = self._last_action_history.get(agent_id, [])
    if len(history) < 2:
        return action_weights

    last_action = history[-1]
    repeat_count = 1
    for idx in range(len(history) - 2, -1, -1):
        if history[idx] == last_action:
            repeat_count += 1
        else:
            break

    if repeat_count < 2:
        return action_weights

    adjusted = action_weights.copy()
    if last_action in adjusted:
        penalty = 0.62 if repeat_count == 2 else 0.45
        adjusted[last_action] *= penalty
    return adjusted

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

        # 用户要求：移除高度物理保护机制，改为纯诊断日志输出
        if current_altitude < 1000:
            try:
                roll_deg = np.degrees(env.agents[agent_id].get_property_value(c.attitude_phi_rad))
                pitch_deg = np.degrees(env.agents[agent_id].get_property_value(c.attitude_pitch_rad))
                is_abnormal = abs(pitch_deg) > 60 or abs(roll_deg) > 120
                if is_abnormal or getattr(env, 'current_step', 0) % 120 == 0:
                    tas = np.linalg.norm(env.agents[agent_id].get_velocity())
                    v_up = env.agents[agent_id].get_velocity()[2]
                    aoa = env.agents[agent_id].get_property_value(c.aero_alpha_deg) if hasattr(c, 'aero_alpha_deg') else 0
                    logging.warning(
                        f"🩺 [诊断-敌机极低空-V5] {agent_id} Alt={current_altitude:.0f}m Roll={roll_deg:.1f}° Pitch={pitch_deg:.1f}° "
                        f"TAS={tas:.1f} Vup={v_up:.1f} AoA={aoa:.1f} cmd_in(ail={aileron:+.2f},ele={elevator:+.2f},rud={rudder:+.2f},thr={throttle:.2f})"
                    )
            except Exception:
                pass

        # 🩺 低速诊断V5：追溯敌机失速坠毁根因（速度<100m/s时每帧输出）
        try:
            ac = env.agents[agent_id]
            vel = np.linalg.norm(ac.get_velocity())
            if vel < 100.0:
                pitch_deg = np.degrees(ac.get_property_value(c.attitude_pitch_rad))
                roll_deg = np.degrees(ac.get_property_value(c.attitude_phi_rad))
                aoa = ac.get_property_value(c.aero_alpha_deg) if hasattr(c, 'aero_alpha_deg') else 0
                cur_thr = ac.get_property_value(c.fcs_throttle_cmd_norm)
                logging.warning(
                    f"🩺 [诊断-敌机低速-V5] {agent_id} TAS={vel:.1f}m/s Alt={current_altitude:.0f}m "
                    f"Pitch={pitch_deg:.1f}° Roll={roll_deg:.1f}° AoA={aoa:.1f}° "
                    f"Thr={cur_thr:.2f} cmd_out(ail={aileron:.2f},ele={elevator:.2f},rud={rudder:.2f},thr={throttle:.2f}) "
                    f"step={getattr(env, 'current_step', '?')}"
                )
        except Exception:
            pass

        return np.array([aileron, elevator, rudder, throttle])

    except Exception as e:
        logging.error(f"直接控制映射错误: {e}")
        return np.array([0.0, 0.0, 0.0, 0.7])

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
        should_return_home = self._enemy_rtb_enabled() and self._should_return_to_base(env, agent_id, current_time, min_distance)
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

        # 导弹威胁检测：去重、按目标过滤，并使用合理威胁距离阈值。
        missile_threats = []
        missile_threat_range_m = float(os.getenv("ENEMY_MISSILE_THREAT_RANGE_M", "60000"))
        seen_missile_ids = set()

        # 检查多种可能的导弹存储位置
        missile_sources = []
        if hasattr(env, '_tempsims') and env._tempsims:
            missile_sources.append(('_tempsims', env._tempsims))
        if hasattr(env, 'missiles') and env.missiles:
            missile_sources.append(('missiles', env.missiles))
        if hasattr(env, 'active_missiles') and env.active_missiles:
            missile_sources.append(('active_missiles', env.active_missiles))

        for source_name, missile_dict in missile_sources:
            try:
                for missile_id, missile_sim in missile_dict.items():
                    if missile_id in seen_missile_ids:
                        continue

                    # 只统计仍然存活的导弹，避免历史残留条目抬高威胁。
                    try:
                        sim_alive = getattr(missile_sim, 'is_alive', True)
                        sim_alive = bool(sim_alive() if callable(sim_alive) else sim_alive)
                    except Exception:
                        sim_alive = True
                    if not sim_alive:
                        continue

                    # 仅统计确实以当前敌机为目标的导弹。
                    target_ok = False
                    target_id = getattr(missile_sim, 'target_id', None)
                    if isinstance(target_id, str) and target_id == agent_id:
                        target_ok = True
                    if not target_ok and hasattr(missile_sim, 'target_aircraft'):
                        target_aircraft = getattr(missile_sim, 'target_aircraft', None)
                        target_uid = getattr(target_aircraft, 'uid', None) if target_aircraft is not None else None
                        if target_uid == agent_id:
                            target_ok = True
                    if not target_ok:
                        continue

                    # 🔧 改进导弹ID识别规则：检查多种可能的敌方导弹命名规则
                    is_enemy_missile = (
                        missile_id.startswith('A') or  # A开头的导弹（原有规则）
                        'blue' in missile_id.lower() or 
                        'friendly' in missile_id.lower() or 
                        'ally' in missile_id.lower() or
                        missile_id.startswith('F')  # F开头的导弹（可能的命名）
                    )

                    if is_enemy_missile and hasattr(missile_sim, 'get_position'):
                        try:
                            missile_pos = missile_sim.get_position()
                            missile_distance = np.linalg.norm(np.array(current_pos) - np.array(missile_pos))

                            if missile_distance < missile_threat_range_m:
                                seen_missile_ids.add(missile_id)
                                missile_threats.append({
                                    'id': missile_id,
                                    'distance': missile_distance,
                                    'position': missile_pos,
                                    'time_detected': current_time,
                                    'source': source_name  # 🔧 记录来源
                                })
                        except Exception as e:
                            logging.debug(f"导弹{missile_id}位置获取失败: {e}")
                            continue
            except Exception as e:
                logging.debug(f"导弹源{source_name}检查失败: {e}")
                continue

        # 🔧 如果检测到导弹威胁，记录详细信息
        # 🔥 调试：敌方AI的日志不打印
        # if missile_threats and not hasattr(self, '_last_missile_threat_log'):
        #     self._last_missile_threat_log = {}
        # if missile_threats:
        #     if agent_id not in self._last_missile_threat_log or (current_time - self._last_missile_threat_log[agent_id]) > 10.0:
        #         closest_missile = min(missile_threats, key=lambda x: x['distance'])
        #         logging.warning(f"🚨 {agent_id} 检测到{len(missile_threats)}个导弹威胁！最近{closest_missile['distance']/1000:.1f}km")
        #         self._last_missile_threat_log[agent_id] = current_time

        # 检查雷达锁定状态（简化实现）
        radar_locked = False
        lock_duration = 0.0
        if hasattr(self, 'radar_lock_time') and agent_id in self.radar_lock_time:
            lock_duration = current_time - self.radar_lock_time[agent_id]
            radar_locked = lock_duration > 2.0  # 锁定超过2秒

        # 检查队友状态
        teammate_alive = False
        teammate_distance = float('inf')
        teammate_id = self._get_enemy_teammate_id(agent_id)

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
            # 检查最近导弹距离 - 🔥 修复问题1：扩大威胁距离阈值，提升生存能力
            closest_missile_distance = min([m['distance'] for m in situation.missile_threats])
            if closest_missile_distance < 15000:  # 15km
                threat_score += 35.0  # 增加威胁分数
            elif closest_missile_distance < 30000:  # 30km
                threat_score += 25.0  # 增加威胁分数
            elif closest_missile_distance < 50000:  # 50km
                threat_score += 15.0  # 新增中距离威胁

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
    """更新战术阶段 - 修复 Bug 4: 敌方35km折返与80km重新接敌循环"""
    try:
        if str(os.getenv("ENEMY_DISABLE_WAVE_MODE", "1")).strip().lower() == "0":
            distance = situation.min_enemy_distance
            if distance > 81000:
                new_phase = EnemyTacticalPhase.NLT_MELD
            elif distance > 50000:
                new_phase = EnemyTacticalPhase.MELD_MTR
            elif distance > 40000:
                new_phase = EnemyTacticalPhase.MTR_TR
            elif distance > 35000:
                new_phase = EnemyTacticalPhase.TR_DOR
            else:
                new_phase = EnemyTacticalPhase.DOR_DR
            self.current_phase[agent_id] = new_phase
            return

        distance = situation.min_enemy_distance

        if not hasattr(self, '_is_retreating'):
            self._is_retreating = {}

        # 初始化折返状态
        if agent_id not in self._is_retreating:
            self._is_retreating[agent_id] = False

        # 状态转换逻辑
        if not self._is_retreating[agent_id] and distance <= 35000:
            # 距离过近，触发折返
            self._is_retreating[agent_id] = True
            logging.info(f"🔄 [敌方{agent_id}] 距离达到{distance/1000:.1f}km <= 35km，进入折返规避状态")
        elif self._is_retreating[agent_id] and distance >= 80000:
            # 折返拉开足够距离，重新进入攻击态势
            self._is_retreating[agent_id] = False
            logging.info(f"⚔️ [敌方{agent_id}] 折返拉开距离达到{distance/1000:.1f}km >= 80km，重新前出攻击")

        # 基于状态和距离更新战术阶段
        if self._is_retreating[agent_id]:
            # 正在折返中，保持在 DOR_DR 阶段（返航机动）
            new_phase = EnemyTacticalPhase.DOR_DR
        else:
            # 正在进攻中，根据距离正常推进阶段
            if distance > 81000:
                new_phase = EnemyTacticalPhase.NLT_MELD
            elif distance > 50000:
                new_phase = EnemyTacticalPhase.MELD_MTR
            elif distance > 40000:
                new_phase = EnemyTacticalPhase.MTR_TR
            elif distance > 35000:
                new_phase = EnemyTacticalPhase.TR_DOR
            else:
                new_phase = EnemyTacticalPhase.DOR_DR

        # 更新阶段记录
        old_phase = self.current_phase.get(agent_id, EnemyTacticalPhase.NLT_MELD)
        self.current_phase[agent_id] = new_phase
        if old_phase != new_phase:
            if new_phase == EnemyTacticalPhase.DOR_DR and getattr(self, '_tempsims', None):
                 logging.info(f"🚀 [敌方{agent_id}] 进入DOR_DR规避(状态: {'折返' if self._is_retreating[agent_id] else '正常'})")

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

        # 🔥 修复任务1：调整战术模式权重，降低防御模式触发概率，增加交战持续性
        if threat.threat_level == ThreatLevel.SEVERE:
            mode_weights = {TacticalMode.DEFENSIVE: 0.7, TacticalMode.NEUTRAL: 0.2, TacticalMode.AGGRESSIVE: 0.1}
        elif threat.threat_level == ThreatLevel.CRITICAL:
            mode_weights = {TacticalMode.DEFENSIVE: 0.5, TacticalMode.NEUTRAL: 0.3, TacticalMode.AGGRESSIVE: 0.2}
        elif threat.threat_level == ThreatLevel.HIGH:
            mode_weights = {TacticalMode.DEFENSIVE: 0.4, TacticalMode.NEUTRAL: 0.35, TacticalMode.AGGRESSIVE: 0.25}
        elif threat.threat_level == ThreatLevel.MEDIUM:
            # 基于战术阶段调整权重 - 增加攻击性
            current_phase = self.current_phase.get(agent_id, EnemyTacticalPhase.NLT_MELD)
            if current_phase in [EnemyTacticalPhase.MTR_TR, EnemyTacticalPhase.TR_DOR]:
                mode_weights = {TacticalMode.AGGRESSIVE: 0.6, TacticalMode.NEUTRAL: 0.3, TacticalMode.DEFENSIVE: 0.1}
            else:
                mode_weights = {TacticalMode.NEUTRAL: 0.4, TacticalMode.AGGRESSIVE: 0.4, TacticalMode.DEFENSIVE: 0.2}
        elif threat.threat_level == ThreatLevel.LOW:
            mode_weights = {TacticalMode.AGGRESSIVE: 0.6, TacticalMode.NEUTRAL: 0.3, TacticalMode.DEFENSIVE: 0.1}
        else:  # NONE
            mode_weights = {TacticalMode.NEUTRAL: 0.5, TacticalMode.AGGRESSIVE: 0.5}

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

def _select_action(self, agent_id: str, tactical_mode: TacticalMode, current_time: float, env=None) -> ActionType:
    """机动决策模块 - 基于权重矩阵的随机化动作选择"""
    try:
        current_phase = self.current_phase.get(agent_id, EnemyTacticalPhase.NLT_MELD)
        situation = self.situation_data.get(agent_id)

        # ==================== 敌方“前出→交战→撤离/返航”硬约束 ====================
        # 目的：限制敌方贴脸，避免因为动作空间过大导致“每次都很靠近我方”
        wave_mode_active = str(os.getenv("ENEMY_DISABLE_WAVE_MODE", "1")).strip().lower() == "0"
        if (not wave_mode_active) and env is not None and hasattr(env, "agents") and agent_id in env.agents and env.agents[agent_id].is_alive:
            pos = env.agents[agent_id].get_position()
            if agent_id not in self._initial_positions:
                self._initial_positions[agent_id] = pos

            # 1) 前出线限制：敌方x过小（过于靠近中心/我方）则强制返航
            if self._enemy_rtb_enabled() and pos[0] <= self._forward_limit_x:
                self._enemy_phases[agent_id] = "RETURNING"
                if hasattr(env, "current_step") and env.current_step % 120 == 0:
                    logging.warning(f"🏠 [敌方前出限制] {agent_id} x={pos[0]/1000:.1f}km ≤ {self._forward_limit_x/1000:.1f}km，强制返航")
                return ActionType.RETURN_TO_BASE

            # 2) 最小安全距离：与最近我方距离过近则强制撤离（不等到40km再处理）
            nearest = float("inf")
            has_friendly_alive = False
            for fid in ("A0100", "A0200", "A0300", "A0400"):
                if fid in env.agents and env.agents[fid].is_alive:
                    has_friendly_alive = True
                    fpos = env.agents[fid].get_position()
                    d = np.linalg.norm(np.array(pos) - np.array(fpos))
                    nearest = min(nearest, d)
            if self._enemy_rtb_enabled() and has_friendly_alive and nearest < self._min_separation_to_friendly:
                self._enemy_phases[agent_id] = "RETURNING"
                if hasattr(env, "current_step") and env.current_step % 120 == 0:
                    logging.warning(f"🏠 [敌方安全距离] {agent_id} 距最近我方{nearest/1000:.1f}km < {self._min_separation_to_friendly/1000:.1f}km，强制撤离/返航")
                return ActionType.RETURN_TO_BASE

            # 3) 飞行距离阈值：飞行过久/过远后返航（原有逻辑加强）
            init = self._initial_positions.get(agent_id)
            if init is not None:
                flown = np.linalg.norm(np.array(pos) - np.array(init))
                if self._enemy_rtb_enabled() and flown >= self._flight_distance_threshold:
                    self._enemy_phases[agent_id] = "RETURNING"
                    if hasattr(env, "current_step") and env.current_step % 120 == 0:
                        logging.warning(f"🏠 [敌方航程阈值] {agent_id} 累计航程{flown/1000:.1f}km ≥ {self._flight_distance_threshold/1000:.1f}km，强制返航")
                    return ActionType.RETURN_TO_BASE

        # 远距前出阶段采用“受控随机”而非完全确定性，避免每次仿真动作轨迹趋同。
        # 检查当前动作是否需要继续执行
        if agent_id in self.current_action and agent_id in self.action_start_time:
            action_duration = current_time - self.action_start_time[agent_id]
            current_action_type = self.current_action[agent_id]

            # 获取动作参数中的持续时间
            if agent_id in self.action_parameters:
                required_duration = self.action_parameters[agent_id].duration
                if action_duration < required_duration and not self._should_interrupt_current_action_for_missile(agent_id):
                    # 继续执行当前动作
                    return current_action_type

        # 获取当前模式和阶段的动作权重
        if tactical_mode not in self.action_weights:
            tactical_mode = TacticalMode.NEUTRAL

        if current_phase not in self.action_weights[tactical_mode]:
            current_phase = EnemyTacticalPhase.NLT_MELD

        action_weights = self.action_weights[tactical_mode][current_phase].copy()

        # 开局阶段风格化权重：优先替换远距初段权重，减少“开局总是一字型”。
        opening_weights = self._opening_phase_action_weights(agent_id, situation, current_phase, current_time)
        if opening_weights is not None:
            action_weights = opening_weights

        # 🔥 应用动态威胁响应权重调整
        action_weights = self._apply_threat_response_weights(agent_id, action_weights, env)
        action_weights = self._apply_missile_evasion_override(agent_id, action_weights, current_time)

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
            if self._enemy_rtb_enabled() and ActionType.RETURN_TO_BASE in action_weights:
                action_weights[ActionType.RETURN_TO_BASE] *= 1.1

        # 增加轻量随机扰动，避免权重矩阵在同态势下长期收敛到固定动作
        if self._behavior_diversity > 0.0:
            jitter_scale = 0.10 * self._behavior_diversity
            for action in list(action_weights.keys()):
                jitter = 1.0 + random.uniform(-jitter_scale, jitter_scale)
                action_weights[action] = max(1e-6, action_weights[action] * jitter)

        # 全程去重：抑制动作连续重复，增强轨迹与决策序列变化。
        action_weights = self._apply_anti_repeat_diversity(agent_id, action_weights)

        if not self._enemy_rtb_enabled():
            action_weights.pop(ActionType.RETURN_TO_BASE, None)

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
        history = self._last_action_history.get(agent_id, [])
        history.append(selected_action)
        if len(history) > 6:
            history = history[-6:]
        self._last_action_history[agent_id] = history

        logging.debug(f"敌方{agent_id}选择动作: {selected_action.value} (模式: {tactical_mode.value}, 阶段: {current_phase.value})")

        return selected_action

    except Exception as e:
        logging.error(f"机动决策失败 {agent_id}: {e}")
        return ActionType.MAINTAIN_HEADING

def _apply_threat_response_weights(self, agent_id: str, base_weights: Dict[ActionType, float], env=None) -> Dict[ActionType, float]:
    """应用动态威胁响应权重调整"""
    try:
        # 获取当前威胁评估
        threat_assessment = self.threat_assessment.get(agent_id)
        if not threat_assessment:
            return base_weights

        adjusted_weights = base_weights.copy()

        # 1. 根据威胁等级调整规避动作权重
        threat_level = threat_assessment.threat_level
        if threat_level in self.threat_response_multipliers:
            multipliers = self.threat_response_multipliers[threat_level]
            for action_type, multiplier in multipliers.items():
                if action_type in adjusted_weights:
                    old_weight = adjusted_weights[action_type]
                    adjusted_weights[action_type] *= multiplier
                    if env and hasattr(env, 'current_step') and env.current_step % 300 == 0:  # 每15秒记录一次
                        logging.debug(f"🔥 {agent_id} 威胁{threat_level.name}调整{action_type.value}: {old_weight:.3f}→{adjusted_weights[action_type]:.3f}")

        # 2. 根据导弹接近距离进一步调整
        if threat_assessment.missile_threat_count > 0:
            situation = self.situation_data.get(agent_id)
            if situation and situation.missile_threats:
                closest_missile_distance = min([m['distance'] for m in situation.missile_threats])

                # 查找适用的距离乘数
                distance_multiplier = 1.0
                for distance_threshold, multiplier in sorted(self.missile_distance_multipliers.items()):
                    if closest_missile_distance <= distance_threshold:
                        distance_multiplier = multiplier
                        break

                if distance_multiplier > 1.0:
                    # 对所有规避动作应用距离紧急乘数
                    evasion_actions = [
                        ActionType.NOTCH_MANEUVER, ActionType.BEAM_MANEUVER, 
                        ActionType.DIVE_ESCAPE, ActionType.CHAFF_FLARE_MANEUVER,
                        ActionType.SPIRAL_DIVE, ActionType.DEFENSIVE_SPLIT
                    ]
                    for action_type in evasion_actions:
                        if action_type in adjusted_weights:
                            adjusted_weights[action_type] *= distance_multiplier

                    if env and hasattr(env, 'current_step') and env.current_step % 60 == 0:  # 每3秒记录一次
                        logging.warning(f"🚨 {agent_id} 导弹接近{closest_missile_distance/1000:.1f}km，规避权重x{distance_multiplier}")

        # 3. 多导弹威胁时的额外调整
        if threat_assessment.missile_threat_count >= 2:
            multi_threat_multiplier = 1.0 + (threat_assessment.missile_threat_count - 1) * 0.5
            evasion_actions = [ActionType.NOTCH_MANEUVER, ActionType.BEAM_MANEUVER, ActionType.SPIRAL_DIVE]
            for action_type in evasion_actions:
                if action_type in adjusted_weights:
                    adjusted_weights[action_type] *= multi_threat_multiplier

            if env and hasattr(env, 'current_step') and env.current_step % 180 == 0:  # 每9秒记录一次
                logging.warning(f"🎯 {agent_id} 面临{threat_assessment.missile_threat_count}个导弹威胁，规避权重x{multi_threat_multiplier:.1f}")

        return adjusted_weights

    except Exception as e:
        logging.error(f"威胁响应权重调整失败 {agent_id}: {e}")
        return base_weights

def _execute_action(self, env, agent_id: str, action_type: ActionType, current_time: float, task=None) -> Tuple[int, int, int]:
    """动作执行模块 - 将动作类型转换为具体的飞行指令 - 🛡️ 多层安全保护机制"""
    try:
        # 🛡️ 设置环境引用供参数生成使用
        self._current_env = env

        # 🛡️ 第一道防线：高度安全检查 - 防止飞机坠毁
        current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)

        # 🛡️ 获取飞机特定安全高度
        aircraft_params = self._get_aircraft_parameters(env, agent_id)
        critical_altitude = aircraft_params["min_altitude"] * 0.8  # 危险高度（如F-16: 2000m）
        safe_altitude = aircraft_params["min_altitude"] * 1.5     # 安全高度（如F-16: 3750m）

        # 🛡️ 严格的低高度俯冲动作检查
        dangerous_actions = [ActionType.DIVE_ESCAPE, ActionType.SPIRAL_DIVE, ActionType.CHAFF_FLARE_MANEUVER, ActionType.DESCEND]
        if action_type in dangerous_actions:
            # 🛡️ 低于安全高度时完全禁用俯冲动作
            if current_altitude < safe_altitude:  # 使用飞机特定的安全高度
                if action_type == ActionType.DIVE_ESCAPE:
                    logging.warning(f"🛡️ {agent_id} 高度{current_altitude:.0f}m < {safe_altitude:.0f}m，俯冲脱离改为Beam机动")
                    return self._execute_beam_maneuver(env, agent_id)  # 改为Beam机动
                elif action_type == ActionType.SPIRAL_DIVE:
                    logging.warning(f"🛡️ {agent_id} 高度{current_altitude:.0f}m < {safe_altitude:.0f}m，螺旋俯冲改为水平转弯")
                    return self._execute_turn(env, agent_id, 90.0, 5.0)  # 改为水平转弯
                else:
                    logging.warning(f"🛡️ {agent_id} 高度{current_altitude:.0f}m过低，禁用{action_type.value}，改为水平飞行")
                    return self._execute_maintain_heading(env, agent_id)

        # 🛡️ 第二道防线：严重低高度强制拉升
        if current_altitude < critical_altitude:
            # 紧急情况：强制拉升，不执行原始动作
            logging.error(f"🚨 {agent_id} 高度{current_altitude:.0f}m < {critical_altitude:.0f}m，紧急拉升！")
            return self._execute_altitude_change(env, agent_id, 1000.0)  # 紧急拉升1000米

        # 生成动作参数（如果还没有，或上一套参数属于别的动作）
        if (
            agent_id not in self.action_parameters
            or self.action_parameter_type.get(agent_id) != action_type
        ):
            self.action_parameters[agent_id] = self._generate_action_parameters(action_type, agent_id)
            self.action_parameter_type[agent_id] = action_type

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
        return 7, 8, 4  # 默认平稳飞行

def _apply_global_safety_check(self, env, agent_id: str, alt_cmd: int, hdg_cmd: int, vel_cmd: int, task=None) -> Tuple[int, int, int]:
    """
    全局高度安全检查（最终防线）- 🔥 修复版本
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

        # 🔥 修复4：大幅提高保护阈值，确保飞机安全
        emergency_threshold = max(float(min_altitude) + 400.0, 2400.0)
        warning_threshold = max(float(min_altitude) + 1000.0, 3200.0)
        caution_threshold = max(float(min_altitude) + 2200.0, 4800.0)

        # 🔥 修复：记录原始指令用于调试
        original_alt_cmd = alt_cmd

        if current_alt < emergency_threshold:
            # 🚨 紧急拉升：强制温和拉升 + 保持平推
            # 🔥 修复5：彻底告别底层网络失速螺旋！
            # 当飞机由于战术动作进入低空又低速的状态，强制要求网络输出极限爬升 (alt=14, pitch极大)
            # 在这种未曾学习过的边界奇点，网络因梯度散度爆出了每秒震荡的副翼狂打(aileron=±1)和满反舵(rudder=-0.45)！
            # 现在缓解命令烈度，从 14 下调至 10，让网络保持在已收敛的飞行包线内平稳拉起。
            alt_cmd = 10  # 爬升（+200m）
            vel_cmd = 7   # 继续当前速度区间轻微加速 (+30m/s)
            hdg_cmd = 8   # 保持航向 (0 deg turn)，尝试改平

            # 🔥 修复4：滚转角保护 - 检查滚转角和俯仰角
            current_roll = env.agents[agent_id].get_property_value(c.attitude_phi_rad)
            current_pitch = env.agents[agent_id].get_property_value(c.attitude_theta_rad)

            # 如果滚转角过大，需要改平
            if abs(current_roll) > np.deg2rad(45):
                # 大坡度时，优先改平而不是爬升
                # 如果右滚（roll>0），需要左滚指令（heading_cmd向左）
                # 但这里我们保持航向指令，让底层控制器改平
                hdg_cmd = 8  # 保持航向，触发改平
                # 如果滚转角过大，降低爬升指令，优先改平
                if abs(current_roll) > np.deg2rad(60):
                    alt_cmd = 9  # 降低爬升幅度，优先改平

            # 如果俯仰角过大（机头过度上仰），可能导致失速
            if current_pitch > np.deg2rad(30):
                # 机头过度上仰，降低爬升指令，避免失速
                alt_cmd = 9  # 降低爬升幅度
                vel_cmd = 10  # 保持最大加速

            if env.current_step % 30 == 0:  # 每6秒输出一次
                logging.error(f"🚨 [{agent_id}] 紧急拉升: 高度{current_alt:.0f}m 滚转{np.rad2deg(current_roll):.1f}° 俯仰{np.rad2deg(current_pitch):.1f}° 指令={alt_cmd}")
            return alt_cmd, hdg_cmd, vel_cmd  # 🔥 立即返回，不被后续代码修改

        elif current_alt < warning_threshold:
            # 🛡️ 强制水平/爬升：完全禁止下降
            alt_change = self.norm_delta_altitude[alt_cmd] * 1000

            if alt_change < 0:
                # 禁止下降，强制爬升
                alt_cmd = 9  # 小幅爬升（+150m）
                # 🔧 降低日志频率：每60步输出一次（约12秒）
                log_key = f"_alt_protect_log_{agent_id}"
                current_step = getattr(env, 'current_step', 0)
                if not hasattr(self, log_key) or current_step - getattr(self, log_key, 0) > 60:
                    logging.warning(f"🛡️ [{agent_id}] 高度保护: 高度{current_alt:.0f}m < {warning_threshold:.0f}m，禁止下降")
                    setattr(self, log_key, current_step)

        elif current_alt < caution_threshold:
            # ⚠️ 限制下降：只允许小幅下降
            alt_change = self.norm_delta_altitude[alt_cmd] * 1000

            if alt_change < -300:
                # 限制下降幅度到100m
                alt_cmd = 5  # -100m
                # 🔥 调试：敌方AI的日志不打印
                # if env.current_step % 60 == 0:  # 🔥 增加日志频率便于调试
                #     logging.info(f"🛡️ [{agent_id}] 限制下降: 高度{current_alt:.0f}m < {caution_threshold:.0f}m，原始指令={original_alt_cmd}(变化{alt_change:.0f}m)→修改为={alt_cmd}")

        # 🔥 根因修复：失速/低速判据使用真空速（速度矢量模），
        # 避免高空使用校准空速(vc)造成“误判低速→持续触发恢复链”。
        current_velocity = float(np.linalg.norm(env.agents[agent_id].get_velocity()))
        current_vc = float(env.agents[agent_id].get_property_value(c.velocities_vc_mps))
        current_pitch = env.agents[agent_id].get_property_value(c.attitude_theta_rad)

        # 🔥 关键失速恢复阈值
        stall_speed = min_speed * 0.8  # 失速临界速度（最低速度的80%）
        recovery_speed = min_speed * 1.2  # 安全恢复速度

        # 🔥 修复4：检查是否处于失速状态（速度低且机头上仰）
        is_stalling = current_velocity < stall_speed or (current_velocity < min_speed and current_pitch > np.deg2rad(20))

        if is_stalling:
            # 🚨 紧急失速恢复：优先保向+加速，禁止继续下发下降链（避免低速-下沉正反馈）
            max_vel_cmd = len(self.norm_delta_velocity) - 1  # CAP有效速度索引上限(0..6)
            vel_cmd = max_vel_cmd
            # 仅在机头严重上仰且高度充足时允许极小幅减仰，否则一律保持/爬升
            if current_pitch > np.deg2rad(30) and current_alt > caution_threshold + 600.0:
                alt_cmd = 6   # -50m，轻度减仰，避免过度拉杆
            elif current_alt < warning_threshold:
                alt_cmd = 9   # 低空强制温和爬升
            else:
                alt_cmd = max(alt_cmd, 7)  # 至少保持高度
            hdg_cmd = 8   # 保持航向
            if env.current_step % 30 == 0:  # 每6秒输出一次
                logging.error(
                    f"🚨 [{agent_id}] 紧急失速恢复: Vtrue={current_velocity:.0f}m/s Vc={current_vc:.0f}m/s "
                    f"俯仰{np.rad2deg(current_pitch):.1f}° 指令={alt_cmd},{vel_cmd}"
                )
            return alt_cmd, hdg_cmd, vel_cmd  # 立即返回，优先恢复能量与姿态

        elif current_velocity < min_speed:
            # 🔥 优化：更积极的速度补偿，防止速度过低（如72m/s）
            # 初始化速度补偿状态跟踪
            if not hasattr(self, '_speed_compensation_states'):
                self._speed_compensation_states = {}
            if agent_id not in self._speed_compensation_states:
                self._speed_compensation_states[agent_id] = {
                    'last_compensation_time': -999.0,
                    'compensation_cooldown': 3.0,  # 🔥 缩短冷却时间到3秒，更积极补偿
                    'compensation_count': 0
                }

            state = self._speed_compensation_states[agent_id]
            current_time = env.current_step * env.time_interval
            time_since_last = current_time - state['last_compensation_time']

            # 紧急低速阈值按机型最小速度动态计算，避免固定200m/s导致长期误报。
            emergency_speed_threshold = max(stall_speed + 5.0, min_speed * 0.90)

            # 仅在接近失速边缘时触发“紧急速度补偿”分支，不等待冷却。
            if current_velocity < emergency_speed_threshold:
                # CAP速度动作索引仅0..6，这里直接使用有效档位，避免后续裁剪失真。
                max_vel_cmd = len(self.norm_delta_velocity) - 1
                speed_ratio = current_velocity / max(min_speed, 1e-6)
                if speed_ratio < 0.75:
                    vel_cmd = max_vel_cmd  # 最强加速
                elif speed_ratio < 0.85:
                    vel_cmd = min(max_vel_cmd, 5)  # 中等加速
                else:
                    vel_cmd = min(max_vel_cmd, 4)  # 轻微加速

                # 根因修复：低速补偿优先通过加速恢复能量，避免长期把低速翻译成持续下降链。
                # 失速恢复已在上方分支单独处理，这里不再主动压低高度换速度。
                alt_cmd = max(alt_cmd, 7)

                if time_since_last >= state['compensation_cooldown']:
                    state['last_compensation_time'] = current_time
                    state['compensation_count'] += 1
                    if env.current_step % 30 == 0:
                        logging.warning(
                            f"⚡ [{agent_id}] 低速补偿触发: Vtrue={current_velocity:.1f}m/s Vc={current_vc:.1f}m/s "
                            f"(min={min_speed:.1f}) -> cmd=({alt_cmd},{hdg_cmd},{vel_cmd})"
                        )

            elif current_velocity < recovery_speed:
                # 非紧急但偏低速：保持温和加速，并避免继续压低机头。
                max_vel_cmd = len(self.norm_delta_velocity) - 1
                vel_cmd = max(vel_cmd, max_vel_cmd)
                if current_pitch > np.deg2rad(10):
                    alt_cmd = max(alt_cmd, 7)

        # 高速保护：避免过冲导致后续大幅机动失稳。
        if current_velocity > max_speed * 1.30:
            vel_cmd = min(vel_cmd, 4)

        return alt_cmd, hdg_cmd, vel_cmd

    except Exception as e:
        logging.error(f"❌ 全局安全检查异常 {agent_id}: {e}")
        return alt_cmd, hdg_cmd, vel_cmd
