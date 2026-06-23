"""Extracted return-to-base and defensive maneuver helpers for UnifiedEnemyTacticalAI."""

from __future__ import annotations

import logging
import os
import random
from typing import Tuple

import numpy as np

try:
    from envs.JSBSim.core.catalog import Catalog as c
except ImportError:
    class MockCatalog:
        attitude_psi_rad = "attitude/psi-rad"
        position_h_sl_m = "position/h-sl-m"
        velocities_vc_mps = "velocities/vc-mps"

    c = MockCatalog()

try:
    from .enemy_ai_types import ActionType
except ImportError:
    from enemy_ai_types import ActionType


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
        # 🔥 精简输出：注释掉速度过低警告
        # logging.warning(f"⚠️ {agent_id} 速度过低({current_velocity:.0f}m/s)，执行加速")
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
        return 7, 8, 4  # 直飞+保持高度+轻微加速

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
        if not self._enemy_rtb_enabled():
            self.current_action[agent_id] = ActionType.AGGRESSIVE_APPROACH
            return self._execute_aggressive_approach(env, agent_id)
        # 🔧 修复：确保current_action被正确设置为RETURN_TO_BASE
        self.current_action[agent_id] = ActionType.RETURN_TO_BASE

        # 记录返航开始时间
        if agent_id not in self._enemy_return_start_time:
            self._enemy_return_start_time[agent_id] = current_time

        # ✅ 敌方二次进攻：返航一段时间后，有概率重新前出一次（更强、更像真实对抗）
        # 条件：还活着、有导弹、我方仍有存活目标、且未执行过二次进攻
        if agent_id.startswith("B") and agent_id in env.agents and env.agents[agent_id].is_alive:
            missiles_remaining = getattr(env.agents[agent_id], "num_missiles", 0)
            alive_friends = [aid for aid in ("A0100", "A0200", "A0300", "A0400") if aid in env.agents and env.agents[aid].is_alive]
            if (agent_id not in self._enemy_second_attack_done) and missiles_remaining > 0 and alive_friends:
                # 低能量保护：返航低速/低空时禁止触发二次进攻，避免在恢复窗口被重新拉回交战导致失速链。
                ac = env.agents[agent_id]
                current_vc = float(ac.get_property_value(c.velocities_vc_mps))
                current_alt = float(ac.get_property_value(c.position_h_sl_m))
                params = self._get_aircraft_parameters(env, agent_id)
                min_speed_sa = float(params.get("min_speed", 120.0))
                min_alt_sa = float(params.get("min_altitude", 2500.0))
                enough_energy_for_second_attack = (
                    current_vc >= (min_speed_sa * 1.20)
                    and current_alt >= (min_alt_sa + 1200.0)
                )

                # 返航超过20秒后，按概率触发二次进攻
                if (current_time - self._enemy_return_start_time.get(agent_id, current_time)) >= 20.0 and enough_energy_for_second_attack:
                    p = float(os.getenv("ENEMY_SECOND_ATTACK_PROB", "0.65"))
                    if random.random() < p:
                        self._enemy_second_attack[agent_id] = True
                        self._enemy_second_attack_done.add(agent_id)
                        # 二次进攻持续时间（例如60秒），之后再返航
                        self._enemy_second_attack_end_time[agent_id] = current_time + float(os.getenv("ENEMY_SECOND_ATTACK_DURATION", "60"))
                        self._enemy_phases[agent_id] = "SECOND_ATTACK"
                        logging.warning(f"🔁 [敌方二次进攻] {agent_id} 返航后重新前出(持续{self._enemy_second_attack_end_time[agent_id]-current_time:.0f}s)")
                        # 🔧 修复：更新current_action为AGGRESSIVE_APPROACH
                        self.current_action[agent_id] = ActionType.AGGRESSIVE_APPROACH
                        # 立即转入主动接敌（不再维持返航）
                        return self._execute_aggressive_approach(env, agent_id)

        # 初始化返航状态
        if not hasattr(self, 'return_states'):
            self.return_states = {}

        if agent_id not in self.return_states:
            self._init_return_to_base_unified(agent_id, current_time)
            logging.info(f"[T={current_time:.1f}s][敌方{agent_id}] 开始返航，目标航向0° (北向)")

        state = self.return_states[agent_id]

        if state.get('phase') == 'break_turn_180':
            current_heading = float(np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad)) % 360.0)
            if state.get('break_target_heading') is None:
                state['break_target_heading'] = 0.0
                state['break_end_time'] = float(current_time + state.get('break_duration_s', 8.0))
                logging.info(f"[T={current_time:.1f}s][敌方{agent_id}] 返航阶段1：执行原地快速回转，目标航向{state['break_target_heading']:.1f}°")

            target_heading = float(state['break_target_heading'])
            heading_diff = ((target_heading - current_heading + 180.0) % 360.0) - 180.0
            if abs(heading_diff) <= 15.0 or current_time >= float(state.get('break_end_time', current_time)):
                state['phase'] = 'tactical_return'
                logging.info(f"[T={current_time:.1f}s][敌方{agent_id}] 返航阶段2：偏转完成，转入朝0°返航")
            else:
                # 参考我方快速掉头逻辑：返航开始时使用最大转向率，尽量原地180°回转。
                if heading_diff > 0:
                    hdg_cmd = 16 if abs(heading_diff) > 40.0 else (12 if abs(heading_diff) > 20.0 else 10)
                else:
                    hdg_cmd = 0 if abs(heading_diff) > 40.0 else (4 if abs(heading_diff) > 20.0 else 6)
                current_alt = float(env.agents[agent_id].get_property_value(c.position_h_sl_m))
                alt_cmd = 6 if current_alt > 3500.0 else 7
                vel_cmd = max(5, self._get_dynamic_velocity_cmd(env, agent_id))
                return alt_cmd, hdg_cmd, vel_cmd

        # 🔧 修复问题6：简化返航逻辑，直接朝北方平稳飞行
        # 🔥 修复问题8：使用动态速度管理，防止返航时高度持续下降
        # 目标航向0度（北向），保持高度，动态调整速度防止速度衰减
        vel_cmd = self._get_dynamic_velocity_cmd(env, agent_id)  # 动态速度管理
        return self._maintain_heading_with_altitude_speed(env, agent_id, 0.0, 7, vel_cmd)

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
        'phase': 'break_turn_180',
        'start_time': current_time,
        'use_short_skate': use_short_skate,
        'break_target_heading': None,
        'break_duration_s': float(os.getenv("ENEMY_RTB_BREAK_DURATION", "4.0")),
        'break_end_time': None,
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
        # 🔥 优化：爬升时速度补偿
        current_velocity = env.agents[agent_id].get_property_value(c.velocities_vc_mps)
        # 爬升指令
        altitude_cmd = 9  # 温和爬升150m（修复：原错误用0会导致极度俯冲1500m！）
        # 🔥 关键修复：爬升时动态速度补偿
        vel_cmd = self._get_dynamic_velocity_cmd(env, agent_id)  # 使用动态速度管理
        if altitude_change > 500:  # 大幅爬升（>500m）
            vel_cmd = min(vel_cmd + 2, 6)  # 额外加速补偿
        elif altitude_change > 200:  # 中等爬升（200-500m）
            vel_cmd = min(vel_cmd + 1, 6)  # 轻微额外加速
        # 如果速度已经很低，更积极补偿
        if current_velocity < 200:
            vel_cmd = max(vel_cmd, 6)  # 至少中等加速
        elif current_velocity < 250:
            vel_cmd = max(vel_cmd, 5)  # 至少轻微加速

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
            # 🔥 俯冲时使用动态速度管理
            vel_cmd = self._get_dynamic_velocity_cmd(env, agent_id)

    # 🔥 修复：使用计算好的vel_cmd（爬升时已设置速度补偿，俯冲时动态速度）
    # 确保vel_cmd已初始化（如果前面没有设置，使用默认值）
    if 'vel_cmd' not in locals():
        vel_cmd = self._get_dynamic_velocity_cmd(env, agent_id)  # 🔥 使用动态速度管理

    return altitude_cmd, 8, vel_cmd  # 保持航向，使用动态速度补偿


def _should_return_to_base(self, env, agent_id: str, current_time: float, closest_enemy_distance: float) -> bool:
    """检查是否应该返航 - 优化版本，减少长时间纠缠"""
    try:
        if not self._enemy_rtb_enabled():
            return False
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
                    if missile_distance < 120000:  # 🔥 修复问题2：扩大导弹威胁检测范围到120km
                        enemy_missile_threat = True
                        break

        # BVR循环：导弹打空后拉开返航，再争取触发一次二次进攻
        if missiles_remaining == 0 and closest_enemy_distance > 120000 and not enemy_missile_threat:
            # 减少日志频率
            if not hasattr(self, '_last_rtb_log_time'):
                self._last_rtb_log_time = {}
            if agent_id not in self._last_rtb_log_time or (current_time - self._last_rtb_log_time[agent_id]) > 30.0:
                logging.info(f"🚀 {agent_id} 导弹用尽且距离较远({closest_enemy_distance/1000:.1f}km)且无威胁，返航")
                self._last_rtb_log_time[agent_id] = current_time
            return True

        # 返航条件2：仿真时间超15分钟
        if current_time > 900.0:
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

        # 返航条件4：距离敌机超过220km且无明确威胁且仿真时间超过8分钟
        if closest_enemy_distance > 220000 and current_time > 480.0:
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
        friendly_agents = ['A0100', 'A0200', 'A0300', 'A0400']
        active_friendlies = 0

        for friendly_id in friendly_agents:
            if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                active_friendlies += 1

        # 如果己方被全部击毁，敌方任务完成
        if active_friendlies == 0:
            return True

        # 检查己方（敌方）的状态
        enemy_agents = ['B0100', 'B0200', 'B0300', 'B0400']
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
                    except Exception:
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
