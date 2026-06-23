"""Extracted tactic execution methods for TacticalExecutor.

This module keeps logic identical while reducing class file size.
"""

import logging
import numpy as np
from tactical_types import TacticalPhase
from envs.JSBSim.core.catalog import Catalog as c
from core.target_assignment import get_enemy_team, get_target_with_fallback
from tactical_utils import TacticalUtils


def _allow_legacy_launch_request(self, env, agent_id: str, current_phase, current_time: float) -> bool:
    aircraft = env.agents.get(agent_id)
    if aircraft is None or not getattr(aircraft, 'is_alive', False):
        return False

    missiles_left = int(getattr(aircraft, 'num_left_missiles', getattr(aircraft, 'num_missiles', 0)) or 0)
    if missiles_left <= 0:
        return False

    if float(self.task.last_missile_launch_time.get(agent_id, -999.0)) >= 0.0:
        return False

    phase_name = getattr(current_phase, 'value', str(current_phase))
    if agent_id.startswith('A'):
        try:
            preserve_state = self.task._evaluate_red_defensive_posture(
                env,
                agent_id,
                current_phase_name=phase_name,
                reason="legacy_launch_request",
            )
            if bool(preserve_state.get('preserve', False)):
                return False
        except Exception:
            pass

    launch_req_map = getattr(getattr(self.task, 'state_manager', None), 'missile_launched', {})
    if bool(launch_req_map.get(agent_id, False)):
        return False

    if not hasattr(self.task, '_legacy_launch_request_gate'):
        self.task._legacy_launch_request_gate = {}
    gate_map = self.task._legacy_launch_request_gate
    gate = gate_map.get(agent_id, {})
    if gate.get('phase') == phase_name and (float(current_time) - float(gate.get('time', -999.0))) < 8.0:
        return False

    gate_map[agent_id] = {
        'phase': phase_name,
        'time': float(current_time),
    }
    return True

def execute_drag_shoot(self, env, agent_id: str) -> tuple:
    """
    战术1: 拖曳射击 (DRAG_SHOOT)
    核心思想: 长机诱敌，僚机射击
    """
    # 🔥 关键修复：导弹规避优先
    evasion = self._check_and_evade_missile(env, agent_id)
    if evasion is not None:
        return evasion

    is_lead = self._is_formation_lead(env, agent_id)
    current_time = env.current_step * env.time_interval

    # 二次进攻特殊处理
    is_second_attack = self._is_second_attack(agent_id)
    if is_second_attack:
        selected_tactic = getattr(self.task, '_get_agent_tactic', lambda _aid: getattr(self.task, 'selected_tactic', 'DRAG_SHOOT'))(agent_id)

        # 🔥 添加详细调试日志
        if env.current_step % 60 == 0:
            logging.info(f"🔍 [二次进攻执行] {agent_id} is_second_attack={is_second_attack}, selected_tactic={selected_tactic}")

        if selected_tactic == 'DRAG_SHOOT':
            if env.current_step % 60 == 0:
                logging.info(f"🎯 [二次进攻] {agent_id} 执行DRAG_SHOOT二次进攻战术")
            return self._execute_second_attack_drag_shoot(env, agent_id, is_lead, current_time)
        elif selected_tactic == 'PINCER_ATTACK':
            if env.current_step % 60 == 0:
                logging.info(f"🎯 [二次进攻] {agent_id} 执行PINCER_ATTACK二次进攻战术")
            return self._execute_second_attack_pincer(env, agent_id, is_lead, current_time)
        elif selected_tactic == 'HIGH_LOW_ATTACK':
            if env.current_step % 60 == 0:
                logging.info(f"🎯 [二次进攻] {agent_id} 执行HIGH_LOW_ATTACK二次进攻战术")
            return self._execute_second_attack_high_low(env, agent_id, is_lead, current_time)
        else:
            # 默认使用拖曳射击
            if env.current_step % 60 == 0:
                logging.info(f"🎯 [二次进攻] {agent_id} 默认执行DRAG_SHOOT二次进攻战术")
            return self._execute_second_attack_drag_shoot(env, agent_id, is_lead, current_time)
    else:
        # 🔥 添加调试日志 - 非二次进攻状态
        if env.current_step % 60 == 0 and agent_id.startswith('A'):
            logging.info(f"[常规战术] {agent_id} is_second_attack={is_second_attack}, 执行常规DRAG_SHOOT")

    # 调试：确认方法被调用
    current_phase = self.task.agent_phases.get(agent_id, self.task.current_phase)
    if env.current_step % 60 == 0 and agent_id.startswith('A'):
        logging.info(f"[DRAG_SHOOT] {agent_id} execute_drag_shoot被调用，当前阶段: {current_phase.value}")

    # 检查Short Skate状态
    if agent_id in self.task.short_skate_states:
        return self.task._execute_short_skate_precise(env, agent_id, current_time)

    # 长机动作序列
    if is_lead:
        # 调试：打印长机全局阶段
        if env.current_step % 30 == 0:
            logging.info(f"[拖曳射击-长机] {agent_id} 全局阶段: {current_phase.value}")

        base_heading = self._get_base_heading(env, agent_id)

        if current_phase.value in ['BEYOND_NLT', 'NLT_MELD', 'MELD_MTR']:
            # 🔥 修复：使用动态航向而非硬编码 0度
            if env.current_step % 60 == 0:
                logging.info(f"[DRAG_SHOOT] {agent_id} 追踪敌机方位: {base_heading:.1f}°")
            return self.task._maintain_heading_precise(env, agent_id, base_heading)
        elif current_phase.value == 'MTR_LR':
            # 🔥 修复：拖曳射击长机直飞诱敌，使用基于敌机的方位
            if env.current_step % 60 == 0:
                logging.info(f"[DRAG_SHOOT-长机MTR_LR] {agent_id} 保持直飞诱敌, 航向: {base_heading:.1f}°")
            return self.task._maintain_heading_precise(env, agent_id, base_heading)
        elif current_phase.value == 'LR_TR':
            last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
            if last_launch < 0 and _allow_legacy_launch_request(self, env, agent_id, current_phase, current_time):
                logging.info(f"[DRAG_SHOOT-长机] {agent_id} 设置导弹发射标记")
                self.task.missile_launched[agent_id] = True

            # 🎯 LR阶段：发射后轻微偏转中制导（R-27ER半主动需要持续照射）
            last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
            if last_launch > 0 and (current_time - last_launch) <= 8.0:
                # 发射后8秒内：轻微偏转（±10°）支持中制导
                target_id = get_target_with_fallback(agent_id, env)
                if target_id and target_id in env._jsbsims:
                    target_aircraft = env._jsbsims[target_id]
                    from tactical_utils import TacticalUtils
                    target_bearing = TacticalUtils.calculate_bearing(env.agents[agent_id], target_aircraft)
                    # 长机左偏10°，僚机右偏10°（形成轻微分离，但保持照射）
                    offset = -10.0 if is_lead else 10.0
                    guidance_heading = (target_bearing + offset) % 360.0
                    if env.current_step % 60 == 0:
                        logging.info(f"[LR中制导] {agent_id} 发射后{current_time - last_launch:.1f}s，轻微偏转{offset:.1f}°支持中制导")
                    return self.task._maintain_heading_precise(env, agent_id, guidance_heading)

            # 🎯 战术优先：DRAG_SHOOT长机在LR_TR阶段保持直飞，不执行Crank
            if env.current_step % 60 == 0:
                logging.info(f"[DRAG_SHOOT-长机LR_TR] {agent_id} 战术特定：保持直飞诱敌")
            return self.task._maintain_heading_precise(env, agent_id, base_heading)
        elif current_phase.value == 'TR_DOR':
            # TR_DOR：执行一次Short Skate（防止重复触发死循环）
            skate_key = f'{agent_id}_skate_done'
            if not getattr(self.task, '_skate_completed', {}).get(skate_key, False):
                return self.task._execute_short_skate_precise(env, agent_id, current_time)
            # Short Skate已完成，保持返航
            self._promote_drag_shoot_post_skate(env, agent_id, current_time, is_lead=True)
            return self.task._maintain_heading_precise(env, agent_id, 180.0)
        elif current_phase.value == 'DOR_DR':
            # DOR_DR：继续返航脱离
            if env.current_step % 60 == 0:
                logging.info(f"[DRAG_SHOOT-长机DOR_DR] {agent_id} 继续返航脱离")
            return self.task._maintain_heading_precise(env, agent_id, 180.0)
        elif current_phase.value in ['DR_MAR', 'BEYOND_MAR']:
            # DR_MAR/BEYOND_MAR：返航
            if env.current_step % 60 == 0:
                logging.info(f"[DRAG_SHOOT-长机] {agent_id} 阶段{current_phase.value}，执行返航")
            return self.task._maintain_heading_precise(env, agent_id, 180.0)
        else:
            return self.task._maintain_heading_precise(env, agent_id, base_heading)

    # 僚机动作序列（独立阶段判断）
    else:
        base_heading = self._get_base_heading(env, agent_id)
        target_id = get_target_with_fallback(agent_id, env)
        target_aircraft = env._jsbsims.get(target_id)
        if target_aircraft and target_aircraft.is_alive:
            distance = self.task._calculate_distance_between(env.agents[agent_id], target_aircraft)
            wingman_phase = self.task._get_wingman_phase_by_distance(distance)
        else:
            wingman_phase = self.task.agent_phases.get(agent_id, self.task.current_phase)

        if wingman_phase.value in ['BEYOND_NLT', 'NLT_MELD']:
            # 🔥 修复：使用动态航向，展开角35度
            target_heading = (base_heading + 35.0) % 360.0
            if env.current_step % 60 == 0:
                logging.info(f"[DRAG_SHOOT-僚机] {agent_id} NLT_MELD 展开方位: {target_heading:.1f}°")
            return self.task._maintain_heading_precise(env, agent_id, target_heading)
        elif wingman_phase.value == 'MELD_MTR':
            # 🔥 修复：追踪敌机方向
            return self.task._maintain_heading_precise(env, agent_id, base_heading)
        elif wingman_phase.value == 'MTR_LR':
            hdg = self._heading_to_attack_wp_or_default(env, agent_id, default_heading=base_heading)
            return self.task._maintain_heading_precise(env, agent_id, hdg)
        elif wingman_phase.value == 'LR_TR':
            # 规范实现：LR阶段僚机通常滞后，尚未进入最大发射距离，不请求发射
            if env.current_step % 60 == 0:
                logging.info(f"[DRAG_SHOOT-僚机LR_TR] {agent_id} 滞后保持，不请求发射")
            return self.task._maintain_heading_precise(env, agent_id, base_heading)
        elif wingman_phase.value == 'TR_DOR':
            # 僚机发射逻辑（仅在长机已发射后）
            last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
            teammate_id = self._get_teammate_id(agent_id)
            lead_launch = self.task.last_missile_launch_time.get(teammate_id, -999) if teammate_id else -999
            if last_launch < 0 and lead_launch > 0:
                try:
                    if self.task.missile_manager.should_launch_missile(
                        env, agent_id, TacticalPhase.TR_DOR, target_id,
                        is_second_attack=is_second_attack,
                    ):
                        self.task.missile_launched[agent_id] = True
                except Exception:
                    pass

            # Short Skate返航（防止重复触发死循环）
            skate_key = f'{agent_id}_skate_done'
            if not getattr(self.task, '_skate_completed', {}).get(skate_key, False):
                return self.task._execute_short_skate_precise(env, agent_id, current_time)
            self._promote_drag_shoot_post_skate(env, agent_id, current_time, is_lead=False)
            return self.task._maintain_heading_precise(env, agent_id, 180.0)
        elif wingman_phase.value in ['DOR_DR', 'DR_MAR', 'BEYOND_MAR']:
            # 后续阶段：返航
            if env.current_step % 60 == 0:
                logging.info(f"[DRAG_SHOOT-僚机] {agent_id} 阶段{wingman_phase.value}，执行返航")
            return self.task._maintain_heading_precise(env, agent_id, 180.0)
        else:
            return self.task._maintain_heading_precise(env, agent_id, base_heading)

def execute_pincer_attack(self, env, agent_id: str) -> tuple:
    """
    战术2: 钳形攻势 (Pincer Attack)
    # 核心思想: 双机从两侧包夹敌机
    """
    # 🔥 关键修复：导弹规避优先
    evasion = self._check_and_evade_missile(env, agent_id)
    if evasion is not None:
        return evasion

    is_lead = self._is_formation_lead(env, agent_id)
    current_time = env.current_step * env.time_interval

    # 收拢阶段方向控制：为钳形攻势提供短时间“向内收拢”窗口
    # 需求：收拢阶段长机应往右、僚机往左，避免默认最短转向导致长机往左绕圈。
    if not hasattr(self, '_pincer_converge_start'):
        self._pincer_converge_start = {}

    # 二次进攻特殊处理
    if self._is_second_attack(agent_id):
        return self._execute_second_attack_pincer(env, agent_id, is_lead, current_time)

    skate_direction = 'right' if is_lead else 'left'

    teammate_id = self._get_teammate_id(agent_id)

    if is_lead:
        lead_phase = self.task.agent_phases.get(agent_id, TacticalPhase.BEYOND_NLT)

        if agent_id in self.task.short_skate_states and lead_phase.value not in ['TR_DOR']:
            self.task.short_skate_states.pop(agent_id, None)

        # 非收拢阶段：清理收拢计时
        if lead_phase.value not in ['DOR_DR', 'DR_MAR']:
            self._pincer_converge_start.pop(agent_id, None)

        if env.current_step % 300 == 0:
            logging.info(f"  🎯 [钳形攻势-长机] {agent_id} 阶段:{lead_phase.value}")

        lead_pos = env.agents[agent_id].get_position()
        wing_pos = env.agents[teammate_id].get_position() if teammate_id in env.agents and env.agents[teammate_id].is_alive else lead_pos
        lateral_separation = abs(wing_pos[1] - lead_pos[1])
        target_lateral_separation = 10000

        lead_y = lead_pos[1]
        approaching_boundary = abs(lead_y) > 50000

        if lead_phase.value in ['BEYOND_NLT', 'NLT_MELD']:
            base_heading = self._get_base_heading(env, agent_id)
            # 🔥 钳形展开：长机左转35° (实际为-35°)
            target_heading = (base_heading - 35.0) % 360.0

            # 🔥 调试：输出当前状态
            current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
            if env.current_step % 25 == 0:
                logging.warning(f"🔥 [钳形展开-长机NLT] {agent_id} 当前航向={current_heading:.1f}°, 目标航向={target_heading:.1f}°")

            return self.task._maintain_heading_precise(env, agent_id, target_heading)
        elif lead_phase.value == 'MELD_MTR':
            # 🔥 修复：MELD阶段前期继续钳形展开，后期回正
            # 获取目标距离
            target_id = get_target_with_fallback(agent_id, env)
            if target_id and target_id in env._jsbsims:
                target_aircraft = env._jsbsims[target_id]
                distance = self.task._calculate_distance_between(env.agents[agent_id], target_aircraft)

                base_heading = self._get_base_heading(env, agent_id)
                # 🔥 关键修复：在MELD前期（>150km）继续展开，后期（<150km）回正
                if distance > 150000:
                    # 继续钳形展开
                    target_heading = (base_heading - 35.0) % 360.0
                    current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
                    if env.current_step % 25 == 0:
                        logging.warning(f"🔥 [钳形展开-长机MELD前期] {agent_id} 距离={distance/1000:.1f}km, 当前航向={current_heading:.1f}°, 目标航向={target_heading:.1f}°")
                    return self.task._maintain_heading_precise(env, agent_id, target_heading)
                else:
                    # 回正
                    if env.current_step % 60 == 0:
                        logging.info(f"[PINCER_ATTACK-长机MELD后期] {agent_id} 距离={distance/1000:.1f}km，回正追踪目标")
                    return self.task._maintain_heading_precise(env, agent_id, base_heading)
            else:
                # 无目标，回正追踪目标
                return self.task._maintain_heading_precise(env, agent_id, self._get_base_heading(env, agent_id))
        elif lead_phase.value == 'MTR_LR':
            # 🎯 修复：钳形攻势MTR阶段：长机回调方向
            if env.current_step % 60 == 0:
                logging.info(f"[PINCER_ATTACK-长机MTR_LR] {agent_id} 回正追踪目标")
            return self.task._maintain_heading_precise(env, agent_id, self._get_base_heading(env, agent_id))
        elif lead_phase.value == 'LR_TR':
            last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
            if last_launch < 0:
                self.task.missile_launched[agent_id] = True

            return self.task._maintain_heading_precise(env, agent_id, self._get_base_heading(env, agent_id))
        elif lead_phase.value == 'TR_DOR':
            last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
            skate_key = f'{agent_id}_skate_done'
            if not getattr(self.task, '_skate_completed', {}).get(skate_key, False):
                return self.task._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
            return self.task._maintain_heading_precise(env, agent_id, self._get_rtb_heading(env, agent_id))
        elif lead_phase.value == 'DOR_DR':
            if agent_id not in self._pincer_converge_start:
                self._pincer_converge_start[agent_id] = current_time
            elapsed = current_time - self._pincer_converge_start[agent_id]
            if elapsed < 12.0:
                return self.task._execute_tactical_crank(env, agent_id, direction='right', climb=False)
            return self.task._maintain_heading_precise(env, agent_id, self._get_rtb_heading(env, agent_id))
        elif lead_phase.value == 'DR_MAR':
            if agent_id not in self._pincer_converge_start:
                self._pincer_converge_start[agent_id] = current_time
            elapsed = current_time - self._pincer_converge_start[agent_id]
            if elapsed < 12.0:
                return self.task._execute_tactical_crank(env, agent_id, direction='right', climb=False)
            return self.task._maintain_heading_precise(env, agent_id, self._get_rtb_heading(env, agent_id))
        elif lead_phase.value == 'BEYOND_MAR':
            return self.task._maintain_heading_precise(env, agent_id, self._get_rtb_heading(env, agent_id))
        else:
            # 默认情况：直飞
            return 7, 8, 3

    # 僚机逻辑
    else:
        target_id = get_target_with_fallback(agent_id, env)
        target_aircraft = env._jsbsims.get(target_id)
        if target_aircraft and target_aircraft.is_alive:
            distance = self.task._calculate_distance_between(env.agents[agent_id], target_aircraft)
            wingman_phase = self.task._get_wingman_phase_by_distance(distance)
        else:
            wingman_phase = self.task.agent_phases.get(agent_id, self.task.current_phase)

        if agent_id in self.task.short_skate_states and wingman_phase.value not in ['TR_DOR']:
            self.task.short_skate_states.pop(agent_id, None)

        # 非收拢阶段：清理收拢计时
        if wingman_phase.value not in ['DOR_DR', 'DR_MAR']:
            self._pincer_converge_start.pop(agent_id, None)

        lead_pos = env.agents[teammate_id].get_position() if teammate_id in env.agents and env.agents[teammate_id].is_alive else env.agents[agent_id].get_position()
        wing_pos = env.agents[agent_id].get_position()
        lateral_separation = abs(wing_pos[1] - lead_pos[1])
        target_lateral_separation = 10000

        wing_y = wing_pos[1]
        approaching_boundary = abs(wing_y) > 50000

        if wingman_phase.value in ['BEYOND_NLT', 'NLT_MELD']:
            base_heading = self._get_base_heading(env, agent_id)
            # 🔥 钳形展开：僚机右转35°
            target_heading = (base_heading + 35.0) % 360.0

            # 🔥 调试：输出当前状态
            current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
            if env.current_step % 25 == 0:
                logging.warning(f"🔥 [钳形展开-僚机NLT] {agent_id} 当前航向={current_heading:.1f}°, 目标航向={target_heading:.1f}°")

            return self.task._maintain_heading_precise(env, agent_id, target_heading)
        elif wingman_phase.value == 'MELD_MTR':
            # 🔥 修复：MELD阶段前期继续钳形展开，后期回正
            # 获取目标距离
            target_id = get_target_with_fallback(agent_id, env)
            if target_id and target_id in env._jsbsims:
                target_aircraft = env._jsbsims[target_id]
                distance = self.task._calculate_distance_between(env.agents[agent_id], target_aircraft)

                base_heading = self._get_base_heading(env, agent_id)
                # 🔥 关键修复：在MELD前期（>150km）继续展开，后期（<150km）回正
                if distance > 150000:
                    # 继续钳形展开
                    target_heading = (base_heading + 35.0) % 360.0
                    current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
                    if env.current_step % 25 == 0:
                        logging.warning(f"🔥 [钳形展开-僚机MELD前期] {agent_id} 距离={distance/1000:.1f}km, 当前航向={current_heading:.1f}°, 目标航向={target_heading:.1f}°")
                    return self.task._maintain_heading_precise(env, agent_id, target_heading)
                else:
                    # 回正
                    if env.current_step % 60 == 0:
                        logging.info(f"[PINCER_ATTACK-僚机MELD后期] {agent_id} 距离={distance/1000:.1f}km，回正追踪目标")
                    return self.task._maintain_heading_precise(env, agent_id, base_heading)
            else:
                # 无目标，回正追踪目标
                return self.task._maintain_heading_precise(env, agent_id, self._get_base_heading(env, agent_id))
        elif wingman_phase.value == 'MTR_LR':
            return self.task._maintain_heading_precise(env, agent_id, self._get_base_heading(env, agent_id))
        elif wingman_phase.value == 'LR_TR':
            # 规范实现：LR阶段僚机滞后，未进入最大发射距离，保持姿态不请求发射
            return self.task._maintain_heading_precise(env, agent_id, self._get_base_heading(env, agent_id))
        elif wingman_phase.value == 'TR_DOR':
            skate_key = f'{agent_id}_skate_done'
            if not getattr(self.task, '_skate_completed', {}).get(skate_key, False):
                return self.task._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
            return self.task._maintain_heading_precise(env, agent_id, self._get_rtb_heading(env, agent_id))
        elif wingman_phase.value == 'DOR_DR':
            if agent_id not in self._pincer_converge_start:
                self._pincer_converge_start[agent_id] = current_time
            elapsed = current_time - self._pincer_converge_start[agent_id]
            if elapsed < 12.0:
                return self.task._execute_tactical_crank(env, agent_id, direction='left', climb=False)
            return self.task._maintain_heading_precise(env, agent_id, self._get_rtb_heading(env, agent_id))
        elif wingman_phase.value == 'DR_MAR':
            if agent_id not in self._pincer_converge_start:
                self._pincer_converge_start[agent_id] = current_time
            elapsed = current_time - self._pincer_converge_start[agent_id]
            if elapsed < 12.0:
                return self.task._execute_tactical_crank(env, agent_id, direction='left', climb=False)
            return self.task._maintain_heading_precise(env, agent_id, self._get_rtb_heading(env, agent_id))
        elif wingman_phase.value == 'BEYOND_MAR':
            return self.task._maintain_heading_precise(env, agent_id, self._get_rtb_heading(env, agent_id))
        else:
            return self.task._maintain_heading_precise(env, agent_id, self._get_rtb_heading(env, agent_id))

def execute_high_low_attack(self, env, agent_id: str) -> tuple:
    """
    战术3: 上下夹击 (High-Low Attack)
    # 核心思想: 僚机高空，长机低飞
    """
    # 🔥 关键修复：导弹规避优先
    evasion = self._check_and_evade_missile(env, agent_id)
    if evasion is not None:
        return evasion

    is_lead = self._is_formation_lead(env, agent_id)
    current_time = env.current_step * env.time_interval
    current_alt = env.agents[agent_id].get_position()[2]

    if agent_id in self.task.short_skate_states:
        skate_direction = 'left' if is_lead else 'right'
        return self.task._execute_short_skate_precise(env, agent_id, current_time, skate_direction)

    # 长机动作序列（低空）
    if is_lead:
        current_phase = self._get_agent_phase(agent_id)
        # 🔥 修复：长机在上下夹击中应该平飞，不应该主动下降
        # 长机保持当前高度，只有僚机上升高度
        # 如果vertical_split_targets中有设置，且目标高度低于当前高度，则忽略（不允许下降）
        target_altitude = self.task.vertical_split_targets.get(agent_id, None)

        base_heading = self._get_base_heading(env, agent_id)
        if current_phase.value in ['BEYOND_NLT', 'NLT_MELD', 'MELD_MTR', 'MTR_LR']:
            # 🔥 关键修复：长机不允许下降，只允许平飞
            if target_altitude is not None:
                alt_diff = target_altitude - current_alt
                # 如果目标高度低于当前高度，忽略（不允许下降）
                if alt_diff < 0:
                    # 长机不允许下降，保持平飞
                    if env.current_step % 200 == 0:
                        logging.info(f"🎯 [HIGH_LOW_ATTACK-长机] {agent_id} 目标高度{target_altitude:.0f}m低于当前{current_alt:.0f}m，保持平飞（不允许下降）")
                    return self.task._maintain_heading_precise(env, agent_id, base_heading)
                # 如果目标高度高于当前高度（理论上不应该发生），允许轻微爬升
                elif alt_diff > 200:
                    if alt_diff > 1000:
                        alt_cmd_value = 1000
                    elif alt_diff > 500:
                        alt_cmd_value = 500
                    elif alt_diff > 200:
                        alt_cmd_value = 300
                    else:
                        alt_cmd_value = alt_diff

                    alt_cmd = self.task._convert_altitude_to_index(alt_cmd_value)
                    vel_cmd = 4  # 加速补偿升力损失
                    return alt_cmd, 8, vel_cmd

            # 默认：长机平飞，保持当前高度
            return self.task._maintain_heading_precise(env, agent_id, base_heading)
        elif current_phase.value == 'LR_TR':
            # LR阶段?8km）：发射主动雷达弹，中制导开?
            last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
            if last_launch < 0:
                self.task.missile_launched[agent_id] = True

            # 🎯 战术优先：HIGH_LOW_ATTACK长机在LR_TR阶段保持低空直飞，不执行Crank
            if self.should_log('high_low_attack_info', current_time):
                logging.info(f"🎯 [HIGH_LOW_ATTACK-长机LR_TR] {agent_id} 战术特定：低空直飞")
            return self.task._maintain_heading_precise(env, agent_id, base_heading)
        elif current_phase.value == 'TR_DOR':
            # TR阶段?5km）：中制导结束，准备规避
            skate_key = f'{agent_id}_skate_done'
            if not getattr(self.task, '_skate_completed', {}).get(skate_key, False):
                return self.task._execute_short_skate_precise(env, agent_id, current_time, 'left')
            return self.task._maintain_heading_precise(env, agent_id, 180.0)
        elif current_phase.value in ['DOR_DR', 'DR_MAR', 'BEYOND_MAR']:
            if env.current_step % 60 == 0:
                logging.info(f"🏠 [HIGH_LOW_ATTACK-长机] {agent_id} 阶段{current_phase.value}，执行返航")
            return self.task._maintain_heading_precise(env, agent_id, 180.0)
        else:
            return self.task._maintain_heading_precise(env, agent_id, base_heading)

    # 僚机动作序列（高空）
    else:
        target_id = get_target_with_fallback(agent_id, env)
        target_aircraft = env._jsbsims.get(target_id)
        if target_aircraft and target_aircraft.is_alive:
            distance = self.task._calculate_distance_between(env.agents[agent_id], target_aircraft)
            wingman_phase = self.task._get_wingman_phase_by_distance(distance)
        else:
            wingman_phase = self._get_agent_phase(agent_id)

        target_altitude = self.task.vertical_split_targets.get(agent_id, 9144.0)

        # 🎯 高低夹击：僚机沿编队攻击轴线爬升，不再动态追逐敌机方位
        target_heading = self._get_locked_high_low_heading(env, agent_id)

        current_heading = float(env.agents[agent_id].get_property_value(c.attitude_psi_deg))
        if env.current_step % 200 == 0:
            logging.info(f"[HIGH_LOW_ATTACK-僚机] {agent_id} 战术特定：仅做高度变化，锁定航向{target_heading:.1f}° (当前{current_heading:.1f}°)")

        # 🔥 修复：确保在NLT_MELD和MELD_MTR阶段都能爬升高度
        if wingman_phase.value in ['BEYOND_NLT', 'NLT_MELD']:
            # 🔥 修复：NLT_MELD阶段也应该开始爬升，而不是只保持航向
            alt_diff = target_altitude - current_alt
            if abs(alt_diff) > 200:
                if alt_diff > 1000:
                    alt_cmd_value = 1000
                elif alt_diff > 500:
                    alt_cmd_value = 500
                elif alt_diff > 200:
                    alt_cmd_value = 300
                else:
                    alt_cmd_value = alt_diff

                action = self._build_altitude_heading_action(env, agent_id, alt_cmd_value, target_heading)
                if env.current_step % 100 == 0 and agent_id == "A0200":
                    logging.info(f"🔍 [DEBUG-A0200-NLT_MELD] 高度差{alt_diff:.0f}m, 目标高度{target_altitude:.0f}m, 返回动作: {action}")
                return action
            else:
                action = self.task._maintain_heading_precise(env, agent_id, target_heading)
                if env.current_step % 100 == 0 and agent_id == "A0200":
                    logging.info(f"🔍 [DEBUG-A0200-NLT_MELD] 目标航向={target_heading:.1f}°, 返回动作: {action}")
                return action
        elif wingman_phase.value == 'MELD_MTR':
            alt_diff = target_altitude - current_alt
            if abs(alt_diff) > 200:
                if alt_diff > 1000:
                    alt_cmd_value = 1000
                elif alt_diff > 500:
                    alt_cmd_value = 500
                elif alt_diff > 200:
                    alt_cmd_value = 300
                elif alt_diff < -1000:
                    alt_cmd_value = -1000
                elif alt_diff < -500:
                    alt_cmd_value = -500
                elif alt_diff < -200:
                    alt_cmd_value = -300
                else:
                    alt_cmd_value = alt_diff

                action = self._build_altitude_heading_action(env, agent_id, alt_cmd_value, target_heading)
                if env.current_step % 100 == 0 and agent_id == "A0200":
                    logging.info(f"🔍 [DEBUG-A0200-MELD_MTR] 高度{alt_diff:.0f}m, 目标航向={target_heading:.1f}°, 返回动作: {action}")
                return action
            else:
                action = self.task._maintain_heading_precise(env, agent_id, target_heading)
                if env.current_step % 100 == 0 and agent_id == "A0200":
                    logging.info(f"🔍 [DEBUG-A0200-MELD_MTR] 高度接近目标，目标航{target_heading:.1f}°, 返回动作: {action}")
                return action
        elif wingman_phase.value == 'MTR_LR':
            action = self.task._maintain_heading_precise(env, agent_id, target_heading)
            if env.current_step % 100 == 0 and agent_id == "A0200":
                logging.info(f"🔍 [DEBUG-A0200-MTR_LR] 目标航向={target_heading:.1f}°, 返回动作: {action}")
            return action
        elif wingman_phase.value == 'LR_TR':
            # 规范实现：僚机此阶段保持高空直飞，不请求发射
            if self.should_log('high_low_attack_info', current_time):
                logging.info(f"[HIGH_LOW_ATTACK-僚机LR_TR] {agent_id} 高空直飞(不发射)，固定航向{target_heading:.1f}°")
            return self.task._maintain_heading_precise(env, agent_id, target_heading)
        elif wingman_phase.value == 'TR_DOR':
            last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
            # TR-DOR末端：僚机俯冲并请求发射
            if last_launch < 0:
                try:
                    target_id = get_target_with_fallback(agent_id, env)
                    if target_id:
                        target_aircraft = env._jsbsims.get(target_id)
                        if target_aircraft and target_aircraft.is_alive:
                            distance = self.task._calculate_distance_between(env.agents[agent_id], target_aircraft)
                            if distance <= 72000:
                                self.task.missile_launched[agent_id] = True
                                if env.current_step % 60 == 0:
                                    logging.info(f"[HIGH_LOW_ATTACK-僚机TR_DOR] {agent_id} 俯冲并准备发射(距{distance/1000:.1f}km)")
                except Exception:
                    pass

            # TR_DOR：强制Short Skate返航（不依赖发射/等待门槛）
            skate_key = f'{agent_id}_skate_done'
            if not getattr(self.task, '_skate_completed', {}).get(skate_key, False):
                return self.task._execute_short_skate_precise(env, agent_id, current_time, 'right')
            return self.task._maintain_heading_precise(env, agent_id, 180.0)
        elif wingman_phase.value in ['DOR_DR', 'DR_MAR', 'BEYOND_MAR']:
            if env.current_step % 60 == 0:
                logging.info(f"[HIGH_LOW_ATTACK-僚机] {agent_id} 阶段{wingman_phase.value}，执行返航")
            return self.task._maintain_heading_precise(env, agent_id, 180.0)
        else:
            return self.task._maintain_heading_precise(env, agent_id, target_heading)

def execute_front_back(self, env, agent_id: str) -> tuple:
    """
    战术4: 前后攻击 (Front-Back Attack)
    # 核心思想: 僚机藏在长机后方，形成一字型纵队
    """
    # 🔥 关键修复：导弹规避优先
    evasion = self._check_and_evade_missile(env, agent_id)
    if evasion is not None:
        return evasion

    is_lead = self._is_formation_lead(env, agent_id)
    current_time = env.current_step * env.time_interval
    is_second_attack = self._is_second_attack(agent_id)

    if agent_id in self.task.short_skate_states:
        skate_direction = 'left' if is_lead else 'right'
        return self.task._execute_short_skate_precise(env, agent_id, current_time, skate_direction)

    # 长机动作序列（前机）
    if is_lead:
        current_phase = self._get_agent_phase(agent_id)
        base_heading = self._get_base_heading(env, agent_id)
        if agent_id in ('A0300', 'A0400') and env.current_step % 20 == 0:
            try:
                target_id_dbg = get_target_with_fallback(agent_id, env)
                target_dbg = env._jsbsims.get(target_id_dbg) if target_id_dbg else None
                dist_dbg = self.task._calculate_distance_between(env.agents[agent_id], target_dbg) if (target_dbg and target_dbg.is_alive) else -1
                launch_req_map = getattr(getattr(self.task, 'state_manager', None), 'missile_launched', {})
                launch_req = bool(launch_req_map.get(agent_id, False))
                logging.warning(
                    "[右编队-FRONT_BACK-长机执行] %s phase=%s tactic=%s target=%s dist=%.1fkm base_hdg=%.1f° launch_req=%s",
                    agent_id,
                    current_phase.value if current_phase else 'None',
                    self.task._get_agent_tactic(agent_id) if hasattr(self.task, '_get_agent_tactic') else 'None',
                    target_id_dbg,
                    dist_dbg / 1000.0 if dist_dbg >= 0 else -1.0,
                    base_heading,
                    launch_req,
                )
            except Exception:
                pass
        if current_phase.value in ['BEYOND_NLT', 'NLT_MELD', 'MELD_MTR']:
            if current_time < 4.0:
                return 7, 8, 4
            else:
                # 🔥 修复：追踪敌机方向
                if env.current_step % 60 == 0:
                    logging.info(f"[FRONT_BACK-长机] {agent_id} 追踪目标方位: {base_heading:.1f}°")
                return self.task._maintain_heading_precise(env, agent_id, base_heading)
        elif current_phase.value == 'MTR_LR':
            # 🔥 修复：前后攻击长机直飞诱敌，使用base_heading
            if env.current_step % 60 == 0:
                logging.info(f"[FRONT_BACK-长机MTR_LR] {agent_id} 保持直飞: {base_heading:.1f}°")
            return self.task._maintain_heading_precise(env, agent_id, base_heading)
        elif current_phase.value == 'LR_TR':
            last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
            if last_launch < 0:
                self.task.missile_launched[agent_id] = True

            # 规范实现：发射窗口期保持平稳直飞
            return self.task._maintain_heading_precise(env, agent_id, base_heading)
        elif current_phase.value == 'TR_DOR':
            last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
            # TR_DOR：强制Short Skate返航（不依赖发射/等待门槛）
            skate_key = f'{agent_id}_skate_done'
            if not getattr(self.task, '_skate_completed', {}).get(skate_key, False):
                return self.task._execute_short_skate_precise(env, agent_id, current_time, 'left')
            return self.task._maintain_heading_precise(env, agent_id, 180.0)
        elif current_phase.value == 'DOR_DR':
            # 🔥 修复：DOR_DR阶段继续接近敌机，不要返航
            if env.current_step % 100 == 0:
                logging.info(f"🎯 [FRONT_BACK-DOR_DR] {agent_id} 继续接近敌机，目标航向{base_heading:.1f}°")
            return self.task._maintain_heading_precise(env, agent_id, base_heading)
        elif current_phase.value == 'DR_MAR':
            # DR_MAR阶段：等待DR决策窗口
            return self.task._maintain_heading_precise(env, agent_id, base_heading)
        else:
            # 其他阶段（如BEYOND_MAR）：返航
            return self.task._maintain_heading_precise(env, agent_id, (base_heading + 180.0) % 360.0)

    # 僚机动作序列
    else:
        target_id = get_target_with_fallback(agent_id, env)
        target_aircraft = env._jsbsims.get(target_id)
        if target_aircraft and target_aircraft.is_alive:
            distance = self.task._calculate_distance_between(env.agents[agent_id], target_aircraft)
            wingman_phase = self.task._get_wingman_phase_by_distance(distance)
            phase_source = 'distance_mapping'
        else:
            wingman_phase = self._get_agent_phase(agent_id)
            phase_source = 'agent_phase_fallback'

        if agent_id in ('A0300', 'A0400') and env.current_step % 20 == 0:
            try:
                leader_id_dbg = self._get_teammate_id(agent_id)
                launch_req_map = getattr(getattr(self.task, 'state_manager', None), 'missile_launched', {})
                launch_req = bool(launch_req_map.get(agent_id, False))
                logging.warning(
                    "🎬 [右编队-FRONT_BACK-僚机执行] %s phase=%s source=%s leader=%s target=%s dist=%.1fkm tactic=%s launch_req=%s",
                    agent_id,
                    wingman_phase.value if wingman_phase else 'None',
                    phase_source,
                    leader_id_dbg,
                    target_id,
                    distance / 1000.0 if 'distance' in locals() else -1.0,
                    self.task._get_agent_tactic(agent_id) if hasattr(self.task, '_get_agent_tactic') else 'None',
                    launch_req,
                )
            except Exception:
                pass

        if wingman_phase.value in ['BEYOND_NLT', 'NLT_MELD', 'MELD_MTR']:
            # 🔥 修复：在二次进攻模式下，僚机也追踪敌机方向
            if is_second_attack:
                enemy_bearing = self.task._get_enemy_bearing(env, agent_id)
                if enemy_bearing is not None:
                    if env.current_step % 60 == 0:
                        logging.info(f"[FRONT_BACK-僚机二次进攻] {agent_id} 追踪敌机方位: {enemy_bearing:.1f}°")
                    return self.task._maintain_heading_precise(env, agent_id, enemy_bearing)

            leader_id = self._get_teammate_id(agent_id)
            return self._build_front_back_follow_action(env, agent_id, leader_id, current_time)
        elif wingman_phase.value == 'MTR_LR':
            if hasattr(self.task, 'wingman_crank_state') and self.task.wingman_crank_state.get("completed", False):
                recent_completion = current_time - self.task.wingman_crank_state.get("completed_time", 0) < 60.0
                if recent_completion:
                    logging.info(f"🔄 [{agent_id}]刚完成一字型，继续使用establish逻辑保持队形")
                    return self.task._establish_rear_formation(env, agent_id, current_time)
            base_heading = self._get_base_heading(env, agent_id)
            hdg = self._heading_to_attack_wp_or_default(env, agent_id, default_heading=base_heading)
            return self.task._maintain_heading_precise(env, agent_id, hdg)
        elif wingman_phase.value == 'LR_TR':
            # 规范实现：僚机此阶段保持一字队形/后方队形，不请求发射
            leader_id = self._get_teammate_id(agent_id)
            leader = env._jsbsims.get(leader_id)
            if leader and leader.is_alive:
                lx, ly, lz = leader.get_position()
                import math
                l_hdg_rad = leader.get_property_value(c.attitude_psi_rad)
                back_dist, offset_dist = 10000.0, 2000.0
                tgt_x = lx - back_dist * math.sin(l_hdg_rad) + offset_dist * math.cos(l_hdg_rad)
                tgt_y = ly - back_dist * math.cos(l_hdg_rad) - offset_dist * math.sin(l_hdg_rad)
                return self.task._maintain_heading_precise(env, agent_id, self.task._heading_to_point(env, agent_id, tgt_x, tgt_y))
            return self.task._maintain_heading_precise(env, agent_id, self._get_base_heading(env, agent_id))
        elif wingman_phase.value == 'TR_DOR':
            last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
            leader_id = self._get_teammate_id(agent_id)
            lead_launch = self.task.last_missile_launch_time.get(leader_id, -999) if leader_id else -999

            # TR-DOR后段（接近DOR阈值）请求发射
            if last_launch < 0 and lead_launch > 0:
                try:
                    target_id = get_target_with_fallback(agent_id, env)
                    if target_id:
                        target_aircraft = env._jsbsims.get(target_id)
                        if target_aircraft and target_aircraft.is_alive:
                            distance = self.task._calculate_distance_between(env.agents[agent_id], target_aircraft)
                            if distance <= 72000:
                                self.task.missile_launched[agent_id] = True
                                if env.current_step % 60 == 0:
                                    logging.info(f"🚀 [FRONT_BACK-僚机TR_DOR] {agent_id} 接近DOR阈值({distance/1000:.1f}km)，准备发射")
                except Exception:
                    pass

            # TR_DOR：强制Short Skate返航（不依赖发射/等待门槛）
            skate_key = f'{agent_id}_skate_done'
            if not getattr(self.task, '_skate_completed', {}).get(skate_key, False):
                return self.task._execute_short_skate_precise(env, agent_id, current_time, 'right')
            return self.task._maintain_heading_precise(env, agent_id, 180.0)
        elif wingman_phase.value == 'DOR_DR':
            # 🔥 修复：DOR_DR阶段继续接近敌机，不要返航
            # 需要继续接近到65km才能进入DR_MAR阶段，触发DR决策窗口
            if is_second_attack:
                enemy_bearing = self.task._get_enemy_bearing(env, agent_id)
                if enemy_bearing is not None:
                    if env.current_step % 100 == 0:
                        logging.info(f"🎯 [FRONT_BACK-DOR_DR-二次进攻] {agent_id} 继续接近敌机，航向{enemy_bearing:.1f}°")
                    return self.task._maintain_heading_precise(env, agent_id, enemy_bearing)
            # 常规攻击：保持后方队形，继续接近
            if env.current_step % 100 == 0:
                logging.info(f"🎯 [FRONT_BACK-DOR_DR] {agent_id} 继续接近敌机，保持后方队形")
            leader_id = self._get_teammate_id(agent_id)
            leader = env._jsbsims.get(leader_id)
            if leader and leader.is_alive:
                lx, ly, lz = leader.get_position()
                import math
                l_hdg_rad = leader.get_property_value(c.attitude_psi_rad)
                back_dist, offset_dist = 10000.0, 2000.0
                tgt_x = lx - back_dist * math.sin(l_hdg_rad) + offset_dist * math.cos(l_hdg_rad)
                tgt_y = ly - back_dist * math.cos(l_hdg_rad) - offset_dist * math.sin(l_hdg_rad)
                return self.task._maintain_heading_precise(env, agent_id, self.task._heading_to_point(env, agent_id, tgt_x, tgt_y))
            return self.task._maintain_heading_precise(env, agent_id, self._get_base_heading(env, agent_id))
        elif wingman_phase.value == 'DR_MAR':
            # DR_MAR阶段：执行Beam机动，等待DR决策窗口
            # 这个阶段由节点决策系统控制，不需要特殊处理
            leader_id = self._get_teammate_id(agent_id)
            leader = env._jsbsims.get(leader_id)
            if leader and leader.is_alive:
                lx, ly, lz = leader.get_position()
                import math
                l_hdg_rad = leader.get_property_value(c.attitude_psi_rad)
                tgt_x = lx - 10000.0 * math.sin(l_hdg_rad) + 2000.0 * math.cos(l_hdg_rad)
                tgt_y = ly - 10000.0 * math.cos(l_hdg_rad) - 2000.0 * math.sin(l_hdg_rad)
                return self.task._maintain_heading_precise(env, agent_id, self.task._heading_to_point(env, agent_id, tgt_x, tgt_y))
            return self.task._maintain_heading_precise(env, agent_id, self._get_base_heading(env, agent_id))
        else:
            if hasattr(self.task, 'wingman_crank_state') and self.task.wingman_crank_state.get("completed", False):
                recent_completion = current_time - self.task.wingman_crank_state.get("completed_time", 0) < 60.0
                if recent_completion:
                    logging.info(f"🔄 [{agent_id}]刚完成一字型，继续保持队形而非返航")
                    leader_id = self._get_teammate_id(agent_id)
                    leader = env._jsbsims.get(leader_id)
                    if leader and leader.is_alive:
                        lx, ly, lz = leader.get_position()
                        import math
                        l_hdg_rad = leader.get_property_value(c.attitude_psi_rad)
                        tgt_x = lx - 10000.0 * math.sin(l_hdg_rad) + 2000.0 * math.cos(l_hdg_rad)
                        tgt_y = ly - 10000.0 * math.cos(l_hdg_rad) - 2000.0 * math.sin(l_hdg_rad)
                        return self.task._maintain_heading_precise(env, agent_id, self.task._heading_to_point(env, agent_id, tgt_x, tgt_y))
                    return self.task._maintain_heading_precise(env, agent_id, self._get_base_heading(env, agent_id))
            return self.task._execute_short_skate_precise(env, agent_id, current_time, 'right')

def execute_side_by_side(self, env, agent_id: str) -> tuple:
    """
    战术5: 并排射击 (Side-by-Side / SIDE_BY_SIDE_SHOOTING)
    # 核心思想: 双机保持队形，同时发射导弹
    """
    # 🔥 关键修复：导弹规避优先
    evasion = self._check_and_evade_missile(env, agent_id)
    if evasion is not None:
        return evasion

    is_lead = self._is_formation_lead(env, agent_id)
    current_time = env.current_step * env.time_interval
    is_second_attack = self._is_second_attack(agent_id)

    # 二次进攻特殊处理
    if is_second_attack:
        return self._execute_second_attack_side_by_side(env, agent_id, is_lead, current_time)

    skate_direction = 'left' if is_lead else 'right'

    if agent_id in self.task.short_skate_states:
        return self.task._execute_short_skate_precise(env, agent_id, current_time, skate_direction)

    base_heading = self._get_base_heading(env, agent_id)

    if is_lead:
        current_phase = self._get_agent_phase(agent_id)
        if current_phase.value in ['BEYOND_NLT', 'NLT_MELD', 'MELD_MTR', 'MTR_LR']:
            return self.task._maintain_heading_precise(env, agent_id, base_heading)
        elif current_phase.value == 'LR_TR':
            return self.task._maintain_heading_precise(env, agent_id, base_heading)
        elif current_phase.value == 'TR_DOR':
            last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
            if last_launch < 0:
                try:
                    target_id = get_target_with_fallback(agent_id, env)
                    if target_id:
                        target_aircraft = env._jsbsims.get(target_id)
                        if target_aircraft and target_aircraft.is_alive:
                            distance = self.task._calculate_distance_between(env.agents[agent_id], target_aircraft)
                            # TR末端（接近DOR阈值）才请求发射：约<=72km
                            if distance <= 72000:
                                self.task.missile_launched[agent_id] = True
                                if env.current_step % 60 == 0:
                                    logging.info(f"🚀 [SIDE_BY_SIDE-长机TR_DOR] {agent_id} 接近DOR阈值({distance/1000:.1f}km)，准备发射")
                except Exception:
                    pass
            # TR_DOR：强制Short Skate返航（不依赖发射/等待门槛）
            return self.task._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
        else:
            return self.task._execute_short_skate_precise(env, agent_id, current_time, skate_direction)

    else:
        target_id = get_target_with_fallback(agent_id, env)
        target_aircraft = env._jsbsims.get(target_id)
        if target_aircraft and target_aircraft.is_alive:
            distance = self.task._calculate_distance_between(env.agents[agent_id], target_aircraft)
            wingman_phase = self.task._get_wingman_phase_by_distance(distance)
        else:
            wingman_phase = self._get_agent_phase(agent_id)

        if wingman_phase.value in ['BEYOND_NLT', 'NLT_MELD', 'MELD_MTR']:
            return self.task._maintain_heading_precise(env, agent_id, base_heading)
        elif wingman_phase.value == 'MTR_LR':
            hdg = self._heading_to_attack_wp_or_default(env, agent_id, default_heading=base_heading)
            return self.task._maintain_heading_precise(env, agent_id, hdg)
        elif wingman_phase.value == 'LR_TR':
            return self.task._maintain_heading_precise(env, agent_id, base_heading)
        elif wingman_phase.value == 'TR_DOR':
            last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
            if last_launch < 0:
                try:
                    target_id = get_target_with_fallback(agent_id, env)
                    if target_id:
                        target_aircraft = env._jsbsims.get(target_id)
                        if target_aircraft and target_aircraft.is_alive:
                            distance = self.task._calculate_distance_between(env.agents[agent_id], target_aircraft)
                            if distance <= 72000:
                                self.task.missile_launched[agent_id] = True
                                if env.current_step % 60 == 0:
                                    logging.info(f"🚀 [SIDE_BY_SIDE-僚机TR_DOR] {agent_id} 接近DOR阈值({distance/1000:.1f}km)，准备发射")
                except Exception:
                    pass
            # TR_DOR：强制Short Skate返航（不依赖发射/等待门槛）
            return self.task._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
        else:
            return self.task._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
# ==================== 统一二次进攻协同系统 ====================

def _execute_second_attack_drag_shoot(self, env, agent_id: str, is_lead: bool, current_time: float) -> tuple:
    """二次进攻 - 拖曳射击编队
    🔥 核心修复：规避完成后检查返航，规避期间不检查返航
    """
    # 🔥 调试：记录函数被调用
    if env.current_step % 50 == 0:
        logging.warning(f"🔍 [DEBUG] _execute_second_attack_drag_shoot被调用: {agent_id}")

    my_aircraft = env.agents[agent_id]

    # 🔥 获取目标位置和方位
    target_id = get_target_with_fallback(agent_id, env)

    # ✅ 距离节点判定：必须使用“最近敌机距离”（1v2 时尤其关键）
    min_distance, nearest_enemy_id = self._get_min_distance_to_alive_enemies(env, agent_id)

    if target_id and target_id in env._jsbsims:
        target_aircraft = env._jsbsims[target_id]
    else:
        target_aircraft = None
        if nearest_enemy_id and nearest_enemy_id in getattr(env, "_jsbsims", {}):
            target_aircraft = env._jsbsims.get(nearest_enemy_id)

    if target_aircraft is not None:
        # 计算到目标的方位角
        from tactical_utils import TacticalUtils
        target_bearing = TacticalUtils.calculate_bearing(my_aircraft, target_aircraft)
        # 距离用于节点/MAR判定：使用最近敌机距离
        distance = min_distance if np.isfinite(min_distance) else self.task._calculate_distance_between(my_aircraft, target_aircraft)

        # 🔥 关键修复：无论是否进入导弹规避，都要持续记录“最小距离”
        # 否则会出现：规避中曾<40km，但规避分支提前return导致最小距离未更新 → 永不触发MAR返航
        if not hasattr(self, 'second_attack_min_distance'):
            self.second_attack_min_distance = {}

        prev_min = self.second_attack_min_distance.get(agent_id)
        if prev_min is None:
            self.second_attack_min_distance[agent_id] = distance
        else:
            self.second_attack_min_distance[agent_id] = min(prev_min, distance)

        check_distance = self.second_attack_min_distance[agent_id]

        # 🔥 调试：记录距离
        if env.current_step % 50 == 0:
            logging.warning(f"🔍 [DEBUG] {agent_id} 最近敌机={nearest_enemy_id}, 距离={distance/1000:.1f}km, <=40km={distance<=40000}")

        # ✅ 规避结束后强制返航：规避期间曾<40km，则直接开始Short Skate→最终180返航
        if self.force_rtb_after_evasion.get(agent_id, False):
            if env.current_step % 50 == 0:
                logging.warning(f"🏠 [规避后强制返航] {agent_id} 执行Short Skate→返航(180°)")
            return self._exit_second_attack_to_rtb(env, agent_id, "规避后强制返航")

        # 🔥 修复问题：检查是否正在规避导弹
        evasion = self._check_and_evade_missile(env, agent_id)

        # 🔥 关键修复：如果之前记录过规避距离<40km，无论是否还在规避，都必须返航
        if hasattr(self, 'evasion_distances') and agent_id in self.evasion_distances:
            prev_distance = self.evasion_distances[agent_id]
            if prev_distance < 40000:
                # 之前规避时距离<40km，现在必须返航（即使还在规避或距离已拉大）
                if env.current_step % 50 == 0:
                    logging.warning(f"🏠 [规避期间MAR返航] {agent_id} 规避开始时距离{prev_distance/1000:.1f}km<40km，当前距离{distance/1000:.1f}km，强制返航")
                self.evasion_distances.pop(agent_id, None)
                return self._exit_second_attack_to_rtb(env, agent_id, "规避期间MAR返航")

        # 🔥 如果正在规避，记录“最小距离”（不是只记录开始距离）
        if evasion is not None:
            # 若规避期间已触达MAR阈值，则在规避结束后强制进入返航序列
            if check_distance <= 40000:
                self.force_rtb_after_evasion[agent_id] = True
                if env.current_step % 50 == 0:
                    logging.warning(
                        f"🏠 [规避中触达MAR] {agent_id} 最小距离{check_distance/1000:.1f}km<=40km，规避结束后强制返航"
                    )
            if not hasattr(self, 'evasion_distances'):
                self.evasion_distances = {}
            prev = self.evasion_distances.get(agent_id, distance)
            self.evasion_distances[agent_id] = min(prev, distance)
            if env.current_step % 50 == 0 and prev == distance:
                logging.warning(f"🔍 [规避开始] {agent_id} 记录距离{distance/1000:.1f}km")
            return evasion

        # 🔥 规避完成，清除记录
        if hasattr(self, 'evasion_distances') and agent_id in self.evasion_distances:
            prev_distance = self.evasion_distances[agent_id]
            if env.current_step % 50 == 0:
                logging.warning(f"🔍 [规避完成] {agent_id} 规避开始时距离{prev_distance/1000:.1f}km，当前距离{distance/1000:.1f}km")
            self.evasion_distances.pop(agent_id, None)

        # 🔥 统一的二次进攻距离节点逻辑（所有战术统一）
        # DR=65km, MTR'=55km, LR'=53km, TR'=50km, MAR=40km

        if env.current_step % 50 == 0 and abs(check_distance - distance) > 1000:
            logging.warning(f"🔍 [节点距离-拖曳] {agent_id} 使用最小距离{check_distance/1000:.1f}km（当前{distance/1000:.1f}km）防止节点回退")

        # ✅ 二次进攻优先发射：如果仍有导弹，优先发射而不是立即返航
        # 检查全队返航标志
        team_retreat = getattr(self, 'team_retreat_triggered', False)

        # ✅ 二次进攻压制返航：若仍有弹且目标有效，则允许短时破锁继续进攻（由 TacticalTask 统一闸门控制）
        is_second_attack = False
        still_have_missiles = False
        try:
            is_second_attack = bool(getattr(self.task, 'is_agent_second_attack', lambda _aid: False)(agent_id))
        except Exception:
            is_second_attack = False
        try:
            still_have_missiles = bool(getattr(my_aircraft, 'num_missiles', 0) > 0)
        except Exception:
            still_have_missiles = False

        # ✅ 二次进攻时：只要仍有导弹，就优先发射，延迟返航
        # 在45km就开始设置发射请求，给更多时间发射
        if is_second_attack and still_have_missiles:
            if check_distance >= 45000:
                # 提前设置发射请求（45km就开始）
                try:
                    if hasattr(self.task, 'state_manager') and hasattr(self.task.state_manager, 'missile_launched'):
                        self.task.state_manager.missile_launched[agent_id] = True
                    else:
                        self.task.missile_launched[agent_id] = True
                except Exception:
                    try:
                        self.task.missile_launched[agent_id] = True
                    except Exception:
                        pass

            # 二次进攻时：即使进入MAR，也优先发射，延迟返航到35km
            if check_distance <= 40000:
                # 设置发射请求，给10秒窗口发射
                try:
                    if hasattr(self.task, 'state_manager') and hasattr(self.task.state_manager, 'missile_launched'):
                        self.task.state_manager.missile_launched[agent_id] = True
                    else:
                        self.task.missile_launched[agent_id] = True
                except Exception:
                    try:
                        self.task.missile_launched[agent_id] = True
                    except Exception:
                        pass

                # 二次进攻时：延迟返航到35km，给更多时间发射
                if check_distance > 35000:
                    if env.current_step % 50 == 0:
                        logging.warning(
                            f"🎯 [二次进攻-延迟返航] {agent_id} 距离{check_distance/1000:.1f}km，仍有导弹，优先发射延迟返航到35km"
                        )
                    # 继续对准目标，尝试发射
                    return self.task._maintain_heading_precise(env, agent_id, target_bearing, duration=2.0)

        block_rtb = False
        try:
            if is_second_attack and still_have_missiles and hasattr(self.task, '_should_block_rtb_lock_for_second_attack'):
                block_rtb = bool(self.task._should_block_rtb_lock_for_second_attack(env, agent_id, current_time, reason="executor:mar_rtb"))
        except Exception:
            block_rtb = False

        # 🔥 优先级1：距离过近（<=40km MAR阈值），立即返航（但二次进攻时延迟到35km）
        if (check_distance <= 40000 or team_retreat) and (not block_rtb):
            # 二次进攻时：如果仍有导弹且距离>35km，继续尝试发射
            if is_second_attack and still_have_missiles and check_distance > 35000:
                if env.current_step % 50 == 0:
                    logging.warning(
                        f"🎯 [二次进攻-继续发射] {agent_id} 距离{check_distance/1000:.1f}km，仍有导弹，继续尝试发射"
                    )
                # 设置发射请求
                try:
                    if hasattr(self.task, 'state_manager') and hasattr(self.task.state_manager, 'missile_launched'):
                        self.task.state_manager.missile_launched[agent_id] = True
                    else:
                        self.task.missile_launched[agent_id] = True
                except Exception:
                    try:
                        self.task.missile_launched[agent_id] = True
                    except Exception:
                        pass
                return self.task._maintain_heading_precise(env, agent_id, target_bearing, duration=2.0)
            # 触发全队返航
            if not team_retreat:
                self.team_retreat_triggered = True
                logging.warning(f"🏠 [MAR返航] {agent_id} 触发全队返航！")

            if env.current_step % 100 == 0:
                logging.warning(f"🏠 [MAR返航] {agent_id} 最小距离{check_distance/1000:.1f}km<40km (或队友触发)，执行返航（当前{distance/1000:.1f}km）")

            # 严格返航航向：A方固定180°（B方固定0°）
            target_heading = self._get_rtb_heading(env, agent_id)

            # 执行short skate然后返航
            if agent_id not in self.task.short_skate_states:
                return self._exit_second_attack_to_rtb(env, agent_id, "MAR返航")
            else:
                if env.current_step % 50 == 0:
                    logging.warning(f"🏠 [MAR返航] {agent_id} Short Skate完成，执行返航(航向{target_heading:.1f}°)")
                return self._exit_second_attack_to_rtb(env, agent_id, "MAR返航")

        # ✅ 被闸门拦截：输出一次解释，避免“怎么一直返航但又说二次进攻”的困惑
        if (check_distance <= 40000 or team_retreat) and block_rtb:
            if env.current_step % 100 == 0:
                logging.warning(
                    f"🧯 [二次进攻-压制返航] {agent_id} 阻止MAR/全队返航抢占: min={check_distance/1000:.1f}km cur={distance/1000:.1f}km missiles={int(getattr(my_aircraft, 'num_missiles', 0))}"
                )

            # ✅ 关键需求：已到MAR，返航前争取最后一次发射机会
            # 做法：短窗口(6s)内保持对准目标 + 置位发射请求；发射审批由 TacticalTask/_handle_missile_launches+missile_manager 完成。
            try:
                if not hasattr(self, '_mar_last_chance_until'):
                    self._mar_last_chance_until = {}
                until = float(self._mar_last_chance_until.get(agent_id, 0.0))
                if current_time > until:
                    self._mar_last_chance_until[agent_id] = float(current_time + 6.0)
                    until = float(current_time + 6.0)

                if current_time <= until:
                    # 置位发射请求（写入 state_manager）
                    try:
                        if hasattr(self.task, 'state_manager') and hasattr(self.task.state_manager, 'missile_launched'):
                            self.task.state_manager.missile_launched[agent_id] = True
                        else:
                            self.task.missile_launched[agent_id] = True
                    except Exception:
                        pass

                    if env.current_step % 50 == 0:
                        logging.warning(
                            f"🎯 [MAR最后一射] {agent_id} 对准+请求发射窗口: until={until:.1f}s bearing={target_bearing:.1f}° dist={distance/1000:.1f}km"
                        )
                    return self.task._maintain_heading_precise(env, agent_id, target_bearing, duration=2.0)
            except Exception:
                pass

        # 🔥 优先级2：距离拉大（>80km），敌机撤退，我方返航
        if distance > 80000 and (not (is_second_attack and still_have_missiles)):
            if env.current_step % 100 == 0:
                logging.warning(f"🏠 [敌机撤退] {agent_id} 距离{distance/1000:.1f}km>80km，执行返航")
            return self._exit_second_attack_to_rtb(env, agent_id, "敌机撤退")

        if distance > 80000 and (is_second_attack and still_have_missiles):
            if env.current_step % 100 == 0:
                logging.warning(f"🎯 [二次进攻继续] {agent_id} 距离{distance/1000:.1f}km>80km，但仍有弹：不执行敌机撤退返航，继续占位/逼近")

        # 🔥 优先级3：MTR'节点（55-80km）- 前往占位点
        if check_distance >= 55000:
            if is_lead:
                attack_heading = 350.0
            else:
                attack_heading = 10.0
            if env.current_step % 100 == 0:
                logging.info(f"🎯 [MTR'节点] {agent_id} 前往占位点，航向{attack_heading:.1f}° (最小距离{check_distance/1000:.1f}km，当前{distance/1000:.1f}km)")
            return self.task._maintain_heading_precise(env, agent_id, attack_heading)

        # ✅ 二次进攻：更早设置发射请求，确保尽量发射所有导弹
        # 在MTR'节点（55km）就开始设置发射请求，而不是等到LR'节点（53km）
        if check_distance >= 55000:
            # ✅ 统一写入 TacticalStateManager（避免写到旧兼容字段导致请求不生效）
            try:
                if hasattr(self.task, 'state_manager') and hasattr(self.task.state_manager, 'missile_launched'):
                    self.task.state_manager.missile_launched[agent_id] = True
                else:
                    self.task.missile_launched[agent_id] = True
            except Exception:
                try:
                    self.task.missile_launched[agent_id] = True
                except Exception:
                    pass

        # 🔥 优先级4：LR'节点（53-55km）- 发射导弹，评估威胁度
        if check_distance >= 53000:
            rwr_threat = my_aircraft.get_rwr_threat_level() if hasattr(my_aircraft, 'get_rwr_threat_level') else 0
            if rwr_threat >= 3:
                if env.current_step % 100 == 0:
                    logging.warning(f"🏠 [LR'撤离] {agent_id} 威胁度高(RWR={rwr_threat})，执行撤离")
                return self._exit_second_attack_to_rtb(env, agent_id, "LR撤离")
            else:
                if is_lead:
                    attack_heading = 350.0
                else:
                    attack_heading = 10.0
                if env.current_step % 100 == 0:
                    logging.info(f"🎯 [LR'节点] {agent_id} 发射导弹，航向{attack_heading:.1f}° (最小距离{check_distance/1000:.1f}km，当前{distance/1000:.1f}km)")
                # ✅ 发射后中制导：短时间轻微偏置（R-27ER需持续支持，偏置≤10°）
                last_launch = getattr(self.task.state_manager, "last_missile_launch_time", {}).get(agent_id, -999)
                if (last_launch > 0) and ((current_time - last_launch) <= 8.0) and (target_aircraft is not None):
                    offset = -10.0 if is_lead else 10.0
                    support_heading = (target_bearing + offset) % 360.0
                    return self.task._maintain_heading_precise(env, agent_id, support_heading)
                return self.task._maintain_heading_precise(env, agent_id, attack_heading)

        # 🔥 优先级5：TR'节点（50-53km）- 中制导
        if check_distance >= 50000:
            if is_lead:
                attack_heading = 350.0
            else:
                attack_heading = 10.0
            if env.current_step % 100 == 0:
                logging.info(f"🎯 [TR'节点] {agent_id} 中制导，航向{attack_heading:.1f}° (最小距离{check_distance/1000:.1f}km，当前{distance/1000:.1f}km)")
            last_launch = getattr(self.task.state_manager, "last_missile_launch_time", {}).get(agent_id, -999)
            if (last_launch > 0) and ((current_time - last_launch) <= 8.0) and (target_aircraft is not None):
                offset = -10.0 if is_lead else 10.0
                support_heading = (target_bearing + offset) % 360.0
                return self.task._maintain_heading_precise(env, agent_id, support_heading)
            return self.task._maintain_heading_precise(env, agent_id, attack_heading)

        # 🔥 优先级6：接近MAR（40-50km）- 继续进攻
        # check_distance在40-50km之间
        if is_lead:
            attack_heading = 350.0
        else:
            attack_heading = 10.0
        # ✅ 统一写入 TacticalStateManager（避免写到旧兼容字段导致请求不生效）
        try:
            if hasattr(self.task, 'state_manager') and hasattr(self.task.state_manager, 'missile_launched'):
                self.task.state_manager.missile_launched[agent_id] = True
            else:
                self.task.missile_launched[agent_id] = True
        except Exception:
            try:
                self.task.missile_launched[agent_id] = True
            except Exception:
                pass
        if env.current_step % 100 == 0:
            logging.info(f"🎯 [接近MAR] {agent_id} 继续进攻，航向{attack_heading:.1f}° (最小距离{check_distance/1000:.1f}km，当前{distance/1000:.1f}km)")
        return self.task._maintain_heading_precise(env, agent_id, attack_heading)
    else:
        # 没有目标时，直接返航
        if env.current_step % 100 == 0:
            logging.warning(f"⚠️ [{agent_id}] 二次进攻无目标，执行返航")
        return self._exit_second_attack_to_rtb(env, agent_id, "二次进攻无目标")

def _execute_second_attack_pincer(self, env, agent_id: str, is_lead: bool, current_time: float) -> tuple:
    """二次进攻 - 钳形攻势编队
    🔥 核心修复：规避完成后检查返航，规避期间不检查返航
    """
    my_aircraft = env.agents[agent_id]

    # 🔥 获取目标位置和方位
    target_id = get_target_with_fallback(agent_id, env)

    if target_id and target_id in env._jsbsims:
        target_aircraft = env._jsbsims[target_id]

        # 计算到目标的方位角和距离
        from tactical_utils import TacticalUtils
        target_bearing = TacticalUtils.calculate_bearing(my_aircraft, target_aircraft)
        distance = self.task._calculate_distance_between(my_aircraft, target_aircraft)

        # 🔥 关键修复：无论是否进入导弹规避，都要持续记录“最小距离”
        if not hasattr(self, 'second_attack_min_distance'):
            self.second_attack_min_distance = {}
        prev_min = self.second_attack_min_distance.get(agent_id)
        self.second_attack_min_distance[agent_id] = distance if prev_min is None else min(prev_min, distance)
        check_distance = self.second_attack_min_distance[agent_id]

        # ✅ 规避结束后强制返航：规避期间曾触达MAR则直接开始Short Skate→最终180返航
        if self.force_rtb_after_evasion.get(agent_id, False):
            if env.current_step % 50 == 0:
                logging.warning(f"🏠 [规避后强制返航] {agent_id} 执行Short Skate→返航(180°)")
            return self._exit_second_attack_to_rtb(env, agent_id, "规避后强制返航")

        # 🔥 修复问题：检查是否正在规避导弹
        evasion = self._check_and_evade_missile(env, agent_id)

        # 🔥 关键修复：如果之前记录过规避距离<40km，无论是否还在规避，都必须返航
        if hasattr(self, 'evasion_distances') and agent_id in self.evasion_distances:
            prev_distance = self.evasion_distances[agent_id]
            if prev_distance < 40000:
                if env.current_step % 50 == 0:
                    logging.warning(f"🏠 [规避期间MAR返航-钳形] {agent_id} 规避开始时距离{prev_distance/1000:.1f}km<40km，当前距离{distance/1000:.1f}km，强制返航")
                self.evasion_distances.pop(agent_id, None)
                return self._exit_second_attack_to_rtb(env, agent_id, "规避期间MAR返航")

        # 🔥 如果正在规避，记录距离
        if evasion is not None:
            if check_distance <= 40000:
                self.force_rtb_after_evasion[agent_id] = True
            if not hasattr(self, 'evasion_distances'):
                self.evasion_distances = {}
            prev = self.evasion_distances.get(agent_id, distance)
            self.evasion_distances[agent_id] = min(prev, distance)
            return evasion

        # 🔥 规避完成，清除记录
        if hasattr(self, 'evasion_distances') and agent_id in self.evasion_distances:
            self.evasion_distances.pop(agent_id, None)

        # 🔥 统一的二次进攻距离节点逻辑（所有战术统一）
        # DR=65km, MTR'=55km, LR'=53km, TR'=50km, MAR=40km

        # check_distance 已在规避判断前计算（包含规避期间的最小距离）

        if env.current_step % 50 == 0 and abs(check_distance - distance) > 1000:
            logging.warning(f"🔍 [节点距离-钳形] {agent_id} 使用最小距离{check_distance/1000:.1f}km（当前{distance/1000:.1f}km）防止节点回退")

        # ✅ 二次进攻优先发射：如果仍有导弹，优先发射而不是立即返航
        is_second_attack = False
        still_have_missiles = False
        try:
            is_second_attack = bool(getattr(self.task, 'is_agent_second_attack', lambda _aid: False)(agent_id))
        except Exception:
            is_second_attack = False
        try:
            still_have_missiles = bool(getattr(my_aircraft, 'num_missiles', 0) > 0)
        except Exception:
            still_have_missiles = False

        # ✅ 二次进攻时：只要仍有导弹，就优先发射，延迟返航到35km
        if is_second_attack and still_have_missiles and check_distance <= 40000:
            # 设置发射请求
            try:
                if hasattr(self.task, 'state_manager') and hasattr(self.task.state_manager, 'missile_launched'):
                    self.task.state_manager.missile_launched[agent_id] = True
                else:
                    self.task.missile_launched[agent_id] = True
            except Exception:
                try:
                    self.task.missile_launched[agent_id] = True
                except Exception:
                    pass

            # 延迟返航到35km，给更多时间发射
            if check_distance > 35000:
                if env.current_step % 50 == 0:
                    logging.warning(
                        f"🎯 [二次进攻-延迟返航] {agent_id} 距离{check_distance/1000:.1f}km，仍有导弹，优先发射延迟返航到35km"
                    )
                # 继续对准目标，尝试发射
                return self.task._maintain_heading_precise(env, agent_id, target_bearing, duration=2.0)

        # 🔥 优先级1：距离过近（<=40km MAR阈值），立即返航（但二次进攻时延迟到35km）
        if check_distance <= 40000:
            # 二次进攻时：如果仍有导弹且距离>35km，继续尝试发射
            if is_second_attack and still_have_missiles and check_distance > 35000:
                if env.current_step % 50 == 0:
                    logging.warning(
                        f"🎯 [二次进攻-继续发射] {agent_id} 距离{check_distance/1000:.1f}km，仍有导弹，继续尝试发射"
                    )
                # 设置发射请求
                try:
                    if hasattr(self.task, 'state_manager') and hasattr(self.task.state_manager, 'missile_launched'):
                        self.task.state_manager.missile_launched[agent_id] = True
                    else:
                        self.task.missile_launched[agent_id] = True
                except Exception:
                    try:
                        self.task.missile_launched[agent_id] = True
                    except Exception:
                        pass
                return self.task._maintain_heading_precise(env, agent_id, target_bearing, duration=2.0)

            if env.current_step % 100 == 0:
                logging.warning(f"🏠 [MAR返航] {agent_id} 最小距离{check_distance/1000:.1f}km<40km，执行返航（当前{distance/1000:.1f}km）")
            if agent_id not in self.task.short_skate_states:
                return self._exit_second_attack_to_rtb(env, agent_id, "MAR返航")
            else:
                if env.current_step % 50 == 0:
                    logging.warning(f"🏠 [MAR返航] {agent_id} Short Skate完成，执行180°返航")
                return self._exit_second_attack_to_rtb(env, agent_id, "MAR返航")

        # 🔥 优先级2：距离拉大（>80km），敌机撤退，我方返航
        if distance > 80000:
            if env.current_step % 100 == 0:
                logging.warning(f"🏠 [敌机撤退] {agent_id} 距离{distance/1000:.1f}km>80km，执行返航")
            return self._exit_second_attack_to_rtb(env, agent_id, "敌机撤退")

        # ✅ 二次进攻：更早设置发射请求，确保尽量发射所有导弹
        # 在MTR'节点（55km）就开始设置发射请求
        if check_distance >= 55000:
            # ✅ 统一写入 TacticalStateManager
            try:
                if hasattr(self.task, 'state_manager') and hasattr(self.task.state_manager, 'missile_launched'):
                    self.task.state_manager.missile_launched[agent_id] = True
                else:
                    self.task.missile_launched[agent_id] = True
            except Exception:
                try:
                    self.task.missile_launched[agent_id] = True
                except Exception:
                    pass

        # 🔥 优先级3：MTR'节点（55-80km）- 前往占位点（钳形编队）
        if check_distance >= 55000:
            if is_lead:
                attack_heading = 340.0  # 长机偏左20°
            else:
                attack_heading = 20.0   # 僚机偏右20°
            if env.current_step % 100 == 0:
                logging.info(f"🎯 [MTR'节点] {agent_id} 钳形编队，航向{attack_heading:.1f}° (最小距离{check_distance/1000:.1f}km，当前{distance/1000:.1f}km)")
            return self.task._maintain_heading_precise(env, agent_id, attack_heading)

        # 🔥 优先级4：LR'节点（53-55km）- 发射导弹，评估威胁度
        if check_distance >= 53000:
            rwr_threat = my_aircraft.get_rwr_threat_level() if hasattr(my_aircraft, 'get_rwr_threat_level') else 0
            if rwr_threat >= 3:
                if env.current_step % 100 == 0:
                    logging.warning(f"🏠 [LR'撤离] {agent_id} 威胁度高(RWR={rwr_threat})，执行撤离")
                return self._exit_second_attack_to_rtb(env, agent_id, "LR撤离")
            else:
                if is_lead:
                    attack_heading = 340.0
                else:
                    attack_heading = 20.0
                if env.current_step % 100 == 0:
                    logging.info(f"🎯 [LR'节点] {agent_id} 钳形编队发射导弹，航向{attack_heading:.1f}° (最小距离{check_distance/1000:.1f}km，当前{distance/1000:.1f}km)")
                return self.task._maintain_heading_precise(env, agent_id, attack_heading)

        # 🔥 优先级5：TR'节点（50-53km）- 中制导
        if check_distance >= 50000:
            if is_lead:
                attack_heading = 340.0
            else:
                attack_heading = 20.0
            if env.current_step % 100 == 0:
                logging.info(f"🎯 [TR'节点] {agent_id} 钳形编队中制导，航向{attack_heading:.1f}° (最小距离{check_distance/1000:.1f}km，当前{distance/1000:.1f}km)")
            return self.task._maintain_heading_precise(env, agent_id, attack_heading)

        # 🔥 优先级6：接近MAR（40-50km）- 继续进攻
        if is_lead:
            attack_heading = 340.0
        else:
            attack_heading = 20.0
        self.task.missile_launched[agent_id] = True
        if env.current_step % 100 == 0:
            logging.info(f"🎯 [接近MAR] {agent_id} 钳形编队继续进攻，航向{attack_heading:.1f}° (最小距离{check_distance/1000:.1f}km，当前{distance/1000:.1f}km)")
        return self.task._maintain_heading_precise(env, agent_id, attack_heading)
    else:
        if env.current_step % 100 == 0:
            logging.warning(f"⚠️ [{agent_id}] 二次进攻无目标，执行返航")
        return self._exit_second_attack_to_rtb(env, agent_id, "二次进攻无目标")

def _execute_second_attack_high_low(self, env, agent_id: str, is_lead: bool, current_time: float) -> tuple:
    """二次进攻 - 高低协同编队
    🔥 核心修复：规避完成后检查返航，规避期间不检查返航
    """
    my_aircraft = env.agents[agent_id]

    # 🔥 获取目标位置和方位
    target_id = get_target_with_fallback(agent_id, env)

    if target_id and target_id in env._jsbsims:
        target_aircraft = env._jsbsims[target_id]

        # 计算到目标的方位角和距离
        from tactical_utils import TacticalUtils
        target_bearing = TacticalUtils.calculate_bearing(my_aircraft, target_aircraft)
        distance = self.task._calculate_distance_between(my_aircraft, target_aircraft)

        # 🔥 关键修复：无论是否进入导弹规避，都要持续记录“最小距离”
        if not hasattr(self, 'second_attack_min_distance'):
            self.second_attack_min_distance = {}
        prev_min = self.second_attack_min_distance.get(agent_id)
        self.second_attack_min_distance[agent_id] = distance if prev_min is None else min(prev_min, distance)
        check_distance = self.second_attack_min_distance[agent_id]

        # ✅ 规避结束后强制返航：规避期间曾触达MAR则直接开始Short Skate→最终180返航
        if self.force_rtb_after_evasion.get(agent_id, False):
            if env.current_step % 50 == 0:
                logging.warning(f"🏠 [规避后强制返航] {agent_id} 执行Short Skate→返航(180°)")
            return self._exit_second_attack_to_rtb(env, agent_id, "规避后强制返航")

        # 🔥 修复问题：检查是否正在规避导弹
        evasion = self._check_and_evade_missile(env, agent_id)

        # 🔥 关键修复：如果之前记录过规避距离<40km，无论是否还在规避，都必须返航
        if hasattr(self, 'evasion_distances') and agent_id in self.evasion_distances:
            prev_distance = self.evasion_distances[agent_id]
            if prev_distance < 40000:
                if env.current_step % 50 == 0:
                    logging.warning(f"🏠 [规避期间MAR返航-高低] {agent_id} 规避开始时距离{prev_distance/1000:.1f}km<40km，当前距离{distance/1000:.1f}km，强制返航")
                self.evasion_distances.pop(agent_id, None)
                return self._exit_second_attack_to_rtb(env, agent_id, "规避期间MAR返航")

        # 🔥 如果正在规避，记录距离
        if evasion is not None:
            if check_distance <= 40000:
                self.force_rtb_after_evasion[agent_id] = True
            if not hasattr(self, 'evasion_distances'):
                self.evasion_distances = {}
            prev = self.evasion_distances.get(agent_id, distance)
            self.evasion_distances[agent_id] = min(prev, distance)
            return evasion

        # 🔥 规避完成，清除记录
        if hasattr(self, 'evasion_distances') and agent_id in self.evasion_distances:
            self.evasion_distances.pop(agent_id, None)

        # 🔥 统一的二次进攻距离节点逻辑（所有战术统一）
        # DR=65km, MTR'=55km, LR'=53km, TR'=50km, MAR=40km

        # check_distance 已在规避判断前计算（包含规避期间的最小距离）

        if env.current_step % 50 == 0 and abs(check_distance - distance) > 1000:
            logging.warning(f"🔍 [节点距离-高低] {agent_id} 使用最小距离{check_distance/1000:.1f}km（当前{distance/1000:.1f}km）防止节点回退")

        # ✅ 二次进攻优先发射：如果仍有导弹，优先发射而不是立即返航
        is_second_attack = False
        still_have_missiles = False
        try:
            is_second_attack = bool(getattr(self.task, 'is_agent_second_attack', lambda _aid: False)(agent_id))
        except Exception:
            is_second_attack = False
        try:
            still_have_missiles = bool(getattr(my_aircraft, 'num_missiles', 0) > 0)
        except Exception:
            still_have_missiles = False

        # ✅ 二次进攻时：只要仍有导弹，就优先发射，延迟返航到35km
        if is_second_attack and still_have_missiles and check_distance <= 40000:
            # 设置发射请求
            try:
                if hasattr(self.task, 'state_manager') and hasattr(self.task.state_manager, 'missile_launched'):
                    self.task.state_manager.missile_launched[agent_id] = True
                else:
                    self.task.missile_launched[agent_id] = True
            except Exception:
                try:
                    self.task.missile_launched[agent_id] = True
                except Exception:
                    pass

            # 延迟返航到35km，给更多时间发射
            if check_distance > 35000:
                if env.current_step % 50 == 0:
                    logging.warning(
                        f"🎯 [二次进攻-延迟返航] {agent_id} 距离{check_distance/1000:.1f}km，仍有导弹，优先发射延迟返航到35km"
                    )
                # 继续对准目标，尝试发射
                return self.task._maintain_heading_precise(env, agent_id, target_bearing, duration=2.0)

        # 🔥 优先级1：距离过近（<=40km MAR阈值），立即返航（但二次进攻时延迟到35km）
        if check_distance <= 40000:
            # 二次进攻时：如果仍有导弹且距离>35km，继续尝试发射
            if is_second_attack and still_have_missiles and check_distance > 35000:
                if env.current_step % 50 == 0:
                    logging.warning(
                        f"🎯 [二次进攻-继续发射] {agent_id} 距离{check_distance/1000:.1f}km，仍有导弹，继续尝试发射"
                    )
                # 设置发射请求
                try:
                    if hasattr(self.task, 'state_manager') and hasattr(self.task.state_manager, 'missile_launched'):
                        self.task.state_manager.missile_launched[agent_id] = True
                    else:
                        self.task.missile_launched[agent_id] = True
                except Exception:
                    try:
                        self.task.missile_launched[agent_id] = True
                    except Exception:
                        pass
                return self.task._maintain_heading_precise(env, agent_id, target_bearing, duration=2.0)

            if env.current_step % 100 == 0:
                logging.warning(f"🏠 [MAR返航] {agent_id} 最小距离{check_distance/1000:.1f}km<40km，执行返航（当前{distance/1000:.1f}km）")
            if agent_id not in self.task.short_skate_states:
                return self._exit_second_attack_to_rtb(env, agent_id, "MAR返航")
            else:
                if env.current_step % 50 == 0:
                    logging.warning(f"🏠 [MAR返航] {agent_id} Short Skate完成，执行180°返航")
                return self._exit_second_attack_to_rtb(env, agent_id, "MAR返航")

        # 🔥 优先级2：距离拉大（>80km），敌机撤退，我方返航
        if distance > 80000:
            if env.current_step % 100 == 0:
                logging.warning(f"🏠 [敌机撤退] {agent_id} 距离{distance/1000:.1f}km>80km，执行返航")
            return self._exit_second_attack_to_rtb(env, agent_id, "敌机撤退")

        # ✅ 二次进攻：更早设置发射请求，确保尽量发射所有导弹
        # 在MTR'节点（55km）就开始设置发射请求
        if check_distance >= 55000:
            try:
                if hasattr(self.task, 'state_manager') and hasattr(self.task.state_manager, 'missile_launched'):
                    self.task.state_manager.missile_launched[agent_id] = True
                else:
                    self.task.missile_launched[agent_id] = True
            except Exception:
                try:
                    self.task.missile_launched[agent_id] = True
                except Exception:
                    pass

        # 🔥 优先级3：MTR'节点（55-80km）- 前往占位点（高低编队）
        if check_distance >= 55000:
            attack_heading = 0.0
            role = "高空" if is_lead else "低空"
            if env.current_step % 100 == 0:
                logging.info(f"🎯 [MTR'节点] {agent_id} 高低协同({role})，航向{attack_heading:.1f}° (最小距离{check_distance/1000:.1f}km，当前{distance/1000:.1f}km)")
            return self.task._maintain_heading_precise(env, agent_id, attack_heading)

        # 🔥 优先级4：LR'节点（53-55km）- 发射导弹，评估威胁度
        if check_distance >= 53000:
            rwr_threat = my_aircraft.get_rwr_threat_level() if hasattr(my_aircraft, 'get_rwr_threat_level') else 0
            if rwr_threat >= 3:
                if env.current_step % 100 == 0:
                    logging.warning(f"🏠 [LR'撤离] {agent_id} 威胁度高(RWR={rwr_threat})，执行撤离")
                return self._exit_second_attack_to_rtb(env, agent_id, "LR撤离")
            else:
                attack_heading = 0.0
                role = "高空" if is_lead else "低空"
                if env.current_step % 100 == 0:
                    logging.info(f"🎯 [LR'节点] {agent_id} 高低协同({role})发射导弹，航向{attack_heading:.1f}° (最小距离{check_distance/1000:.1f}km，当前{distance/1000:.1f}km)")
                return self.task._maintain_heading_precise(env, agent_id, attack_heading)

        # 🔥 优先级5：TR'节点（50-53km）- 中制导
        if check_distance >= 50000:
            attack_heading = 0.0
            role = "高空" if is_lead else "低空"
            if env.current_step % 100 == 0:
                logging.info(f"🎯 [TR'节点] {agent_id} 高低协同({role})中制导，航向{attack_heading:.1f}° (最小距离{check_distance/1000:.1f}km，当前{distance/1000:.1f}km)")
            return self.task._maintain_heading_precise(env, agent_id, attack_heading)

        # 🔥 优先级6：接近MAR（40-50km）- 继续进攻
        attack_heading = 0.0
        self.task.missile_launched[agent_id] = True
        role = "高空" if is_lead else "低空"
        if env.current_step % 100 == 0:
            logging.info(f"🎯 [接近MAR] {agent_id} 高低协同({role})继续进攻，航向{attack_heading:.1f}° (最小距离{check_distance/1000:.1f}km，当前{distance/1000:.1f}km)")
        return self.task._maintain_heading_precise(env, agent_id, attack_heading)
    else:
        if env.current_step % 100 == 0:
            logging.warning(f"⚠️ [{agent_id}] 二次进攻无目标，执行返航")
        return self._exit_second_attack_to_rtb(env, agent_id, "二次进攻无目标")

def _execute_second_attack_side_by_side(self, env, agent_id: str, is_lead: bool, current_time: float) -> tuple:
    """二次进攻 - 并排射击编队
    🔥 核心修复：使用统一的节点逻辑和返航逻辑
    """
    my_aircraft = env.agents[agent_id]

    # 🔥 获取目标位置和方位
    target_id = get_target_with_fallback(agent_id, env)

    if target_id and target_id in env._jsbsims:
        target_aircraft = env._jsbsims[target_id]

        # 计算到目标的方位角和距离
        from tactical_utils import TacticalUtils
        target_bearing = TacticalUtils.calculate_bearing(my_aircraft, target_aircraft)
        distance = self.task._calculate_distance_between(my_aircraft, target_aircraft)

        # ===== 1) 先更新并排射击二次进攻“最小敌机距离”(用于队伍级MAR判定) =====
        prev_min = self.sbs_second_attack_min_distance.get(agent_id, distance)
        self.sbs_second_attack_min_distance[agent_id] = min(prev_min, distance)

        # 队伍级最小距离（只统计存活我机）
        team_ids = self.task._get_alive_formation_agents(
            env,
            agent_id,
            preserve_nominal_pair=True,
        ) if hasattr(self.task, '_get_alive_formation_agents') else [
            aid for aid in self._get_formation_agents(agent_id)
            if aid in env.agents and env._jsbsims[aid].is_alive
        ]
        team_min = min([self.sbs_second_attack_min_distance.get(aid, float('inf')) for aid in team_ids] or [float('inf')])

        # 队伍级MAR触发：只要任一架进入<=40km，双机都应返航（并排射击协同脱离）
        if (not self.sbs_team_rtb) and team_min <= 40000:
            self.sbs_team_rtb = True
            self.sbs_team_rtb_reason = f"team_min_enemy_distance={team_min/1000:.1f}km<40km"
            logging.warning(f"🏠 [并排-队伍MAR触发] {self.sbs_team_rtb_reason}，双机统一返航")

        # ===== 2) 导弹规避：必须执行完；期间记录“规避期间最小敌机距离” =====
        # 🔥 修复问题：检查是否正在规避导弹
        evasion = self._check_and_evade_missile(env, agent_id)

        if evasion is not None:
            # 标记规避中，并记录规避期间最小“敌机距离”
            self.sbs_evasion_active[agent_id] = True
            prev_e_min = self.sbs_evasion_min_distance.get(agent_id, distance)
            self.sbs_evasion_min_distance[agent_id] = min(prev_e_min, distance)

            # 若规避期间(敌机距离)已进入MAR，也触发队伍级返航（但必须等规避结束后执行返航动作）
            if (not self.sbs_team_rtb) and self.sbs_evasion_min_distance[agent_id] < 40000:
                self.sbs_team_rtb = True
                self.sbs_team_rtb_reason = f"evasion_min_enemy_distance({agent_id})={self.sbs_evasion_min_distance[agent_id]/1000:.1f}km<40km"
                logging.warning(f"🏠 [并排-规避触发MAR] {self.sbs_team_rtb_reason}，规避结束后双机统一返航")
            return evasion

        # 规避结束：清理规避中标记（不在规避时禁止返航，现在允许进入返航流程）
        if self.sbs_evasion_active.pop(agent_id, None):
            if env.current_step % 50 == 0:
                e_min = self.sbs_evasion_min_distance.get(agent_id, None)
                if e_min is not None:
                    logging.warning(f"✅ [并排-规避完成] {agent_id} 规避期间最小敌机距离={e_min/1000:.1f}km")
            self.sbs_evasion_min_distance.pop(agent_id, None)

        # ===== 3) 队伍级返航执行（不打断规避；规避结束立即 short-skate→180）=====
        if self.sbs_team_rtb:
            # short-skate 一次后保持180
            if agent_id not in self.task.short_skate_states:
                if env.current_step % 50 == 0:
                    logging.warning(f"🏠 [并排-统一返航] {agent_id} Short Skate开始，原因: {self.sbs_team_rtb_reason}")
                return self._exit_second_attack_to_rtb(env, agent_id, self.sbs_team_rtb_reason or "并排统一返航")
            if env.current_step % 50 == 0:
                logging.warning(f"🏠 [并排-统一返航] {agent_id} Short Skate完成，保持180°返航")
            return self._exit_second_attack_to_rtb(env, agent_id, self.sbs_team_rtb_reason or "并排统一返航")

        # 🔥 统一的二次进攻距离节点逻辑（所有战术统一）
        # DR=65km, MTR'=55km, LR'=53km, TR'=50km, MAR=40km

        # 🔥 关键：并排二次进攻使用自己的最小距离（避免受其它战术共享状态影响）
        check_distance = self.sbs_second_attack_min_distance.get(agent_id, distance)

        if env.current_step % 50 == 0 and abs(check_distance - distance) > 1000:
            logging.warning(f"🔍 [节点距离-并排] {agent_id} 使用最小距离{check_distance/1000:.1f}km（当前{distance/1000:.1f}km）防止节点回退")

        # ✅ 二次进攻优先发射：如果仍有导弹，优先发射而不是立即返航
        is_second_attack = False
        still_have_missiles = False
        try:
            is_second_attack = bool(getattr(self.task, 'is_agent_second_attack', lambda _aid: False)(agent_id))
        except Exception:
            is_second_attack = False
        try:
            still_have_missiles = bool(getattr(my_aircraft, 'num_missiles', 0) > 0)
        except Exception:
            still_have_missiles = False

        # ✅ 二次进攻时：只要仍有导弹，就优先发射，延迟返航到35km
        if is_second_attack and still_have_missiles and check_distance <= 40000:
            # 设置发射请求
            try:
                if hasattr(self.task, 'state_manager') and hasattr(self.task.state_manager, 'missile_launched'):
                    self.task.state_manager.missile_launched[agent_id] = True
                else:
                    self.task.missile_launched[agent_id] = True
            except Exception:
                try:
                    self.task.missile_launched[agent_id] = True
                except Exception:
                    pass

            # 延迟返航到35km，给更多时间发射
            if check_distance > 35000:
                if env.current_step % 50 == 0:
                    logging.warning(
                        f"🎯 [二次进攻-延迟返航] {agent_id} 距离{check_distance/1000:.1f}km，仍有导弹，优先发射延迟返航到35km"
                    )
                # 继续对准目标，尝试发射
                return self.task._maintain_heading_precise(env, agent_id, target_bearing, duration=2.0)

        # 🔥 优先级1：距离过近（<=40km MAR阈值），立即返航（但二次进攻时延迟到35km）
        if check_distance <= 40000:
            # 二次进攻时：如果仍有导弹且距离>35km，继续尝试发射
            if is_second_attack and still_have_missiles and check_distance > 35000:
                if env.current_step % 50 == 0:
                    logging.warning(
                        f"🎯 [二次进攻-继续发射] {agent_id} 距离{check_distance/1000:.1f}km，仍有导弹，继续尝试发射"
                    )
                # 设置发射请求
                try:
                    if hasattr(self.task, 'state_manager') and hasattr(self.task.state_manager, 'missile_launched'):
                        self.task.state_manager.missile_launched[agent_id] = True
                    else:
                        self.task.missile_launched[agent_id] = True
                except Exception:
                    try:
                        self.task.missile_launched[agent_id] = True
                    except Exception:
                        pass
                return self.task._maintain_heading_precise(env, agent_id, target_bearing, duration=2.0)

            if env.current_step % 100 == 0:
                logging.warning(f"🏠 [MAR返航] {agent_id} 最小距离{check_distance/1000:.1f}km<40km，执行返航（当前{distance/1000:.1f}km）")
            if agent_id not in self.task.short_skate_states:
                return self._exit_second_attack_to_rtb(env, agent_id, "MAR返航")
            else:
                if env.current_step % 50 == 0:
                    logging.warning(f"🏠 [MAR返航] {agent_id} Short Skate完成，执行180°返航")
                return self._exit_second_attack_to_rtb(env, agent_id, "MAR返航")

        # 🔥 优先级2：距离拉大（>80km），敌机撤退，我方返航
        if distance > 80000:
            if env.current_step % 100 == 0:
                logging.warning(f"🏠 [敌机撤退] {agent_id} 距离{distance/1000:.1f}km>80km，执行返航")
            return self._exit_second_attack_to_rtb(env, agent_id, "敌机撤退")

        # ✅ 二次进攻：更早设置发射请求，确保尽量发射所有导弹
        # 在MTR'节点（55km）就开始设置发射请求
        if check_distance >= 55000:
            try:
                if hasattr(self.task, 'state_manager') and hasattr(self.task.state_manager, 'missile_launched'):
                    self.task.state_manager.missile_launched[agent_id] = True
                else:
                    self.task.missile_launched[agent_id] = True
            except Exception:
                try:
                    self.task.missile_launched[agent_id] = True
                except Exception:
                    pass

        # 🔥 优先级3：MTR'节点（55-80km）- 前往占位点（并排编队）
        if check_distance >= 55000:
            if is_lead:
                attack_heading = (target_bearing - 5) % 360  # 长机偏左5°
            else:
                attack_heading = (target_bearing + 5) % 360  # 僚机偏右5°
            if env.current_step % 100 == 0:
                logging.info(f"🎯 [MTR'节点] {agent_id} 并排编队，航向{attack_heading:.1f}° (最小距离{check_distance/1000:.1f}km，当前{distance/1000:.1f}km)")
            return self.task._maintain_heading_precise(env, agent_id, attack_heading)

        # 🔥 优先级4：LR'节点（53-55km）- 发射导弹，评估威胁度
        if check_distance >= 53000:
            rwr_threat = my_aircraft.get_rwr_threat_level() if hasattr(my_aircraft, 'get_rwr_threat_level') else 0
            if rwr_threat >= 3:
                if env.current_step % 100 == 0:
                    logging.warning(f"🏠 [LR'撤离] {agent_id} 威胁度高(RWR={rwr_threat})，执行撤离")
                return self._exit_second_attack_to_rtb(env, agent_id, "LR撤离")
            else:
                if is_lead:
                    attack_heading = (target_bearing - 5) % 360
                else:
                    attack_heading = (target_bearing + 5) % 360
                if env.current_step % 100 == 0:
                    logging.info(f"🎯 [LR'节点] {agent_id} 并排编队发射导弹，航向{attack_heading:.1f}° (最小距离{check_distance/1000:.1f}km，当前{distance/1000:.1f}km)")
                return self.task._maintain_heading_precise(env, agent_id, attack_heading)

        # 🔥 优先级5：TR'节点（50-53km）- 中制导
        if check_distance >= 50000:
            if is_lead:
                attack_heading = (target_bearing - 5) % 360
            else:
                attack_heading = (target_bearing + 5) % 360
            if env.current_step % 100 == 0:
                logging.info(f"🎯 [TR'节点] {agent_id} 并排编队中制导，航向{attack_heading:.1f}° (最小距离{check_distance/1000:.1f}km，当前{distance/1000:.1f}km)")
            return self.task._maintain_heading_precise(env, agent_id, attack_heading)

        # 🔥 优先级6：接近MAR（40-50km）- 继续进攻
        if is_lead:
            attack_heading = (target_bearing - 5) % 360
        else:
            attack_heading = (target_bearing + 5) % 360
        self.task.missile_launched[agent_id] = True
        if env.current_step % 100 == 0:
            logging.info(f"🎯 [接近MAR] {agent_id} 并排编队继续进攻，航向{attack_heading:.1f}° (最小距离{check_distance/1000:.1f}km，当前{distance/1000:.1f}km)")
        return self.task._maintain_heading_precise(env, agent_id, attack_heading)
    else:
        if env.current_step % 100 == 0:
            logging.warning(f"⚠️ [{agent_id}] 二次进攻无目标，执行返航")
        return self._exit_second_attack_to_rtb(env, agent_id, "二次进攻无目标")

def _execute_second_attack_front_back(self, env, agent_id: str, is_lead: bool, current_time: float) -> tuple:
    """二次进攻 - 前后攻击编队
    🔥 核心修复：使用统一的节点逻辑和返航逻辑
    """
    my_aircraft = env.agents[agent_id]

    # 🔥 获取目标位置和方位
    target_id = get_target_with_fallback(agent_id, env)

    if target_id and target_id in env._jsbsims:
        target_aircraft = env._jsbsims[target_id]

        # 计算到目标的方位角和距离
        from tactical_utils import TacticalUtils
        target_bearing = TacticalUtils.calculate_bearing(my_aircraft, target_aircraft)
        distance = self.task._calculate_distance_between(my_aircraft, target_aircraft)

        # 🔥 关键修复：无论是否进入导弹规避，都要持续记录“最小距离”
        if not hasattr(self, 'second_attack_min_distance'):
            self.second_attack_min_distance = {}
        prev_min = self.second_attack_min_distance.get(agent_id)
        self.second_attack_min_distance[agent_id] = distance if prev_min is None else min(prev_min, distance)
        check_distance = self.second_attack_min_distance[agent_id]

        # ✅ 规避结束后强制返航：规避期间曾触达MAR则直接开始Short Skate→最终180返航
        if self.force_rtb_after_evasion.get(agent_id, False):
            if env.current_step % 50 == 0:
                logging.warning(f"🏠 [规避后强制返航] {agent_id} 执行Short Skate→返航(180°)")
            return self._exit_second_attack_to_rtb(env, agent_id, "规避后强制返航")

        # 🔥 修复问题：检查是否正在规避导弹
        evasion = self._check_and_evade_missile(env, agent_id)

        # 🔥 关键修复：如果之前记录过规避距离<40km，无论是否还在规避，都必须返航
        if hasattr(self, 'evasion_distances') and agent_id in self.evasion_distances:
            prev_distance = self.evasion_distances[agent_id]
            if prev_distance < 40000:
                if env.current_step % 50 == 0:
                    logging.warning(f"🏠 [规避期间MAR返航-前后] {agent_id} 规避开始时距离{prev_distance/1000:.1f}km<40km，当前距离{distance/1000:.1f}km，强制返航")
                self.evasion_distances.pop(agent_id, None)
                return self._exit_second_attack_to_rtb(env, agent_id, "规避期间MAR返航")

        # 🔥 如果正在规避，记录距离
        if evasion is not None:
            if check_distance <= 40000:
                self.force_rtb_after_evasion[agent_id] = True
            if not hasattr(self, 'evasion_distances'):
                self.evasion_distances = {}
            prev = self.evasion_distances.get(agent_id, distance)
            self.evasion_distances[agent_id] = min(prev, distance)
            return evasion

        # 🔥 规避完成，清除记录
        if hasattr(self, 'evasion_distances') and agent_id in self.evasion_distances:
            self.evasion_distances.pop(agent_id, None)

        # 🔥 统一的二次进攻距离节点逻辑（所有战术统一）
        # DR=65km, MTR'=55km, LR'=53km, TR'=50km, MAR=40km

        # check_distance 已在规避判断前计算（包含规避期间的最小距离）

        if env.current_step % 50 == 0 and abs(check_distance - distance) > 1000:
            logging.warning(f"🔍 [节点距离-前后] {agent_id} 使用最小距离{check_distance/1000:.1f}km（当前{distance/1000:.1f}km）防止节点回退")

        # ✅ 二次进攻优先发射：如果仍有导弹，优先发射而不是立即返航
        is_second_attack = False
        still_have_missiles = False
        try:
            is_second_attack = bool(getattr(self.task, 'is_agent_second_attack', lambda _aid: False)(agent_id))
        except Exception:
            is_second_attack = False
        try:
            still_have_missiles = bool(getattr(my_aircraft, 'num_missiles', 0) > 0)
        except Exception:
            still_have_missiles = False

        # ✅ 二次进攻时：只要仍有导弹，就优先发射，延迟返航到35km
        if is_second_attack and still_have_missiles and check_distance <= 40000:
            # 设置发射请求
            try:
                if hasattr(self.task, 'state_manager') and hasattr(self.task.state_manager, 'missile_launched'):
                    self.task.state_manager.missile_launched[agent_id] = True
                else:
                    self.task.missile_launched[agent_id] = True
            except Exception:
                try:
                    self.task.missile_launched[agent_id] = True
                except Exception:
                    pass

            # 延迟返航到35km，给更多时间发射
            if check_distance > 35000:
                if env.current_step % 50 == 0:
                    logging.warning(
                        f"🎯 [二次进攻-延迟返航] {agent_id} 距离{check_distance/1000:.1f}km，仍有导弹，优先发射延迟返航到35km"
                    )
                # 继续对准目标，尝试发射
                return self.task._maintain_heading_precise(env, agent_id, target_bearing, duration=2.0)

        # 🔥 优先级1：距离过近（<=40km MAR阈值），立即返航（但二次进攻时延迟到35km）
        if check_distance <= 40000:
            # 二次进攻时：如果仍有导弹且距离>35km，继续尝试发射
            if is_second_attack and still_have_missiles and check_distance > 35000:
                if env.current_step % 50 == 0:
                    logging.warning(
                        f"🎯 [二次进攻-继续发射] {agent_id} 距离{check_distance/1000:.1f}km，仍有导弹，继续尝试发射"
                    )
                # 设置发射请求
                try:
                    if hasattr(self.task, 'state_manager') and hasattr(self.task.state_manager, 'missile_launched'):
                        self.task.state_manager.missile_launched[agent_id] = True
                    else:
                        self.task.missile_launched[agent_id] = True
                except Exception:
                    try:
                        self.task.missile_launched[agent_id] = True
                    except Exception:
                        pass
                return self.task._maintain_heading_precise(env, agent_id, target_bearing, duration=2.0)

            if env.current_step % 100 == 0:
                logging.warning(f"🏠 [MAR返航] {agent_id} 最小距离{check_distance/1000:.1f}km<40km，执行返航（当前{distance/1000:.1f}km）")
            if agent_id not in self.task.short_skate_states:
                return self._exit_second_attack_to_rtb(env, agent_id, "MAR返航")
            else:
                if env.current_step % 50 == 0:
                    logging.warning(f"🏠 [MAR返航] {agent_id} Short Skate完成，执行180°返航")
                return self._exit_second_attack_to_rtb(env, agent_id, "MAR返航")

        # 🔥 优先级2：距离拉大（>80km），敌机撤退，我方返航
        if distance > 80000:
            if env.current_step % 100 == 0:
                logging.warning(f"🏠 [敌机撤退] {agent_id} 距离{distance/1000:.1f}km>80km，执行返航")
            return self._exit_second_attack_to_rtb(env, agent_id, "敌机撤退")

        # ✅ 二次进攻：更早设置发射请求，确保尽量发射所有导弹
        # 在MTR'节点（55km）就开始设置发射请求
        if check_distance >= 55000:
            try:
                if hasattr(self.task, 'state_manager') and hasattr(self.task.state_manager, 'missile_launched'):
                    self.task.state_manager.missile_launched[agent_id] = True
                else:
                    self.task.missile_launched[agent_id] = True
            except Exception:
                try:
                    self.task.missile_launched[agent_id] = True
                except Exception:
                    pass

        # 🔥 优先级3：MTR'节点（55-80km）- 前往占位点（前后编队）
        if check_distance >= 55000:
            attack_heading = 0.0  # 前后编队都朝0度
            if env.current_step % 100 == 0:
                logging.info(f"🎯 [MTR'节点] {agent_id} 前后编队，航向{attack_heading:.1f}° (最小距离{check_distance/1000:.1f}km，当前{distance/1000:.1f}km)")
            return self.task._maintain_heading_precise(env, agent_id, attack_heading)

        # 🔥 优先级4：LR'节点（53-55km）- 发射导弹，评估威胁度
        if check_distance >= 53000:
            rwr_threat = my_aircraft.get_rwr_threat_level() if hasattr(my_aircraft, 'get_rwr_threat_level') else 0
            if rwr_threat >= 3:
                if env.current_step % 100 == 0:
                    logging.warning(f"🏠 [LR'撤离] {agent_id} 威胁度高(RWR={rwr_threat})，执行撤离")
                return self._exit_second_attack_to_rtb(env, agent_id, "LR撤离")
            else:
                attack_heading = 0.0
                if env.current_step % 100 == 0:
                    logging.info(f"🎯 [LR'节点] {agent_id} 前后编队发射导弹，航向{attack_heading:.1f}° (最小距离{check_distance/1000:.1f}km，当前{distance/1000:.1f}km)")
                return self.task._maintain_heading_precise(env, agent_id, attack_heading)

        # 🔥 优先级5：TR'节点（50-53km）- 中制导
        if check_distance >= 50000:
            attack_heading = 0.0
            if env.current_step % 100 == 0:
                logging.info(f"🎯 [TR'节点] {agent_id} 前后编队中制导，航向{attack_heading:.1f}° (最小距离{check_distance/1000:.1f}km，当前{distance/1000:.1f}km)")
            return self.task._maintain_heading_precise(env, agent_id, attack_heading)

        # 🔥 优先级6：接近MAR（40-50km）- 继续进攻
        attack_heading = 0.0
        self.task.missile_launched[agent_id] = True
        if env.current_step % 100 == 0:
            logging.info(f"🎯 [接近MAR] {agent_id} 前后编队继续进攻，航向{attack_heading:.1f}° (最小距离{check_distance/1000:.1f}km，当前{distance/1000:.1f}km)")
        return self.task._maintain_heading_precise(env, agent_id, attack_heading)
    else:
        if env.current_step % 100 == 0:
            logging.warning(f"⚠️ [{agent_id}] 二次进攻无目标，执行返航")
        return self._exit_second_attack_to_rtb(env, agent_id, "二次进攻无目标")
