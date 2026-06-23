"""Support helpers extracted from TacticalExecutor."""

import logging
import numpy as np

from envs.JSBSim.core.catalog import Catalog as c
from core.target_assignment import get_enemy_team, get_target_with_fallback
from tactical_types import TacticalPhase
from tactical_utils import TacticalUtils


class TacticalExecutorSupportMixin:
    def _get_min_distance_to_alive_enemies(self, env, agent_id: str) -> tuple:
        my = env.agents.get(agent_id)
        if my is None or (hasattr(my, "is_alive") and (not my.is_alive)):
            return float("inf"), None

        candidates = []
        for eid in get_enemy_team(agent_id):
            try:
                e = None
                if hasattr(env, "_jsbsims"):
                    e = env._jsbsims.get(eid)
                if e is None and hasattr(env, "agents"):
                    e = env.agents.get(eid)
                if e is None:
                    continue
                if hasattr(e, "is_alive") and (not e.is_alive):
                    continue
                d = self.task._calculate_distance_between(my, e)
                candidates.append((d, eid))
            except Exception:
                continue

        if not candidates:
            return float("inf"), None
        return min(candidates, key=lambda x: x[0])

    def _get_agent_phase(self, agent_id: str):
        return self.task.agent_phases.get(agent_id, self.task.current_phase)

    def should_log(self, log_key: str, current_time: float) -> bool:
        if log_key not in self.log_intervals:
            return True

        last_time = self.last_log_times.get(log_key, 0)
        interval = self.log_intervals[log_key]

        if current_time - last_time >= interval:
            self.last_log_times[log_key] = current_time
            return True
        return False

    def _is_second_attack(self, agent_id: str) -> bool:
        if hasattr(self.task, 'is_agent_second_attack'):
            try:
                return bool(self.task.is_agent_second_attack(agent_id))
            except Exception:
                pass
        if hasattr(self.task, 'agent_second_attack'):
            try:
                return bool(getattr(self.task, 'agent_second_attack', {}).get(agent_id, False))
            except Exception:
                pass
        return bool(getattr(self.task, 'is_second_attack', False))

    def _get_formation_agents(self, agent_id: str):
        if hasattr(self.task, '_get_formation_agents'):
            try:
                return list(self.task._get_formation_agents(agent_id))
            except Exception:
                pass
        if agent_id in ('A0100', 'A0200'):
            return ['A0100', 'A0200']
        if agent_id in ('A0300', 'A0400'):
            return ['A0300', 'A0400']
        if agent_id in ('B0100', 'B0200'):
            return ['B0100', 'B0200']
        if agent_id in ('B0300', 'B0400'):
            return ['B0300', 'B0400']
        return [agent_id]

    def _get_teammate_id(self, agent_id: str):
        for teammate_id in self._get_formation_agents(agent_id):
            if teammate_id != agent_id:
                return teammate_id
        return None

    def _get_formation_role_by_position(self, env, agent_id: str) -> str:
        if hasattr(self.task, '_get_formation_role_by_position'):
            try:
                return self.task._get_formation_role_by_position(env, agent_id)
            except Exception:
                pass

        formation = self._get_formation_agents(agent_id)
        alive_positions = []
        for aid in formation:
            if aid in env.agents and getattr(env.agents[aid], 'is_alive', False):
                try:
                    pos = env.agents[aid].get_position()
                    alive_positions.append((aid, float(pos[1])))
                except Exception:
                    continue

        if len(alive_positions) >= 2:
            alive_positions.sort(key=lambda it: it[1])
            lead_id = alive_positions[0][0]
            return 'lead' if agent_id == lead_id else 'wingman'

        default_leads = {'A0100', 'A0300', 'B0100', 'B0300'}
        return 'lead' if agent_id in default_leads else 'wingman'

    def _is_formation_lead(self, env, agent_id: str) -> bool:
        return self._get_formation_role_by_position(env, agent_id) == 'lead'

    def _promote_drag_shoot_post_skate(self, env, agent_id: str, current_time: float, is_lead: bool) -> None:
        try:
            current_phase = self.task.agent_phases.get(agent_id, self.task.current_phase)
            if current_phase != TacticalPhase.TR_DOR:
                return

            promote_locks = getattr(self.task, '_drag_shoot_post_skate_promote_until', None)
            if promote_locks is None:
                promote_locks = {}
                setattr(self.task, '_drag_shoot_post_skate_promote_until', promote_locks)
            if current_time < float(promote_locks.get(agent_id, -1.0)):
                return

            skate_key = f'{agent_id}_skate_done'
            skate_done = bool(getattr(self.task, '_skate_completed', {}).get(skate_key, False))
            has_active_skate = agent_id in getattr(self.task, 'short_skate_states', {})
            phase_duration = 0.0
            if hasattr(self.task, 'state_manager') and hasattr(self.task.state_manager, 'get_phase_duration'):
                phase_duration = float(self.task.state_manager.get_phase_duration(agent_id, current_time))

            if not skate_done and (has_active_skate or phase_duration < 18.0):
                return

            target_distance = float('inf')
            target_id = get_target_with_fallback(agent_id, env)
            my_aircraft = getattr(env, 'agents', {}).get(agent_id)
            if target_id and my_aircraft is not None:
                target_aircraft = None
                if hasattr(env, '_jsbsims'):
                    target_aircraft = env._jsbsims.get(target_id)
                if target_aircraft is None and hasattr(env, 'agents'):
                    target_aircraft = env.agents.get(target_id)
                if target_aircraft is not None and getattr(target_aircraft, 'is_alive', True):
                    target_distance = float(self.task._calculate_distance_between(my_aircraft, target_aircraft))
            if not np.isfinite(target_distance):
                target_distance, _ = self._get_min_distance_to_alive_enemies(env, agent_id)

            dor_boundary = float(getattr(self.task, 'tactical_distances', {}).get('DOR', float('inf')))
            missiles_left = 0
            if my_aircraft is not None:
                missiles_left = int(
                    getattr(my_aircraft, 'num_left_missiles', getattr(my_aircraft, 'num_missiles', 0))
                )
            guidance_commit = False
            try:
                if hasattr(self.task, '_formation_has_active_guidance_commit'):
                    guidance_commit = bool(self.task._formation_has_active_guidance_commit(env, agent_id))
            except Exception:
                guidance_commit = False
            promote_margin = 0.0
            promote_reason = "dor_boundary"
            if guidance_commit and phase_duration >= 20.0:
                promote_margin = max(promote_margin, 6000.0)
                promote_reason = "guidance_commit"
            if missiles_left <= 2 and phase_duration >= 24.0:
                promote_margin = max(promote_margin, 8000.0)
                promote_reason = "low_inventory"
            if phase_duration >= 32.0:
                promote_margin = max(promote_margin, 8000.0)
                promote_reason = "phase_timeout_32"
            if phase_duration >= 40.0:
                promote_margin = max(promote_margin, 12000.0)
                promote_reason = "phase_timeout_40"
            promote_distance = dor_boundary + promote_margin
            if (not np.isfinite(target_distance)) or target_distance > promote_distance:
                if env.current_step % 60 == 0:
                    logging.info(
                        "[DRAG_SHOOT-阶段推进抑制] %s 保持TR_DOR (distance=%.1fkm, DOR=%.1fkm, promote=%.1fkm, reason=%s)",
                        agent_id,
                        target_distance / 1000.0,
                        dor_boundary / 1000.0,
                        promote_distance / 1000.0,
                        promote_reason,
                    )
                return
            if (not np.isfinite(target_distance)) or target_distance > promote_distance:
                if env.current_step % 60 == 0:
                    logging.info(
                        f"[DRAG_SHOOT-阶段推进抑制] {agent_id} 保持TR_DOR "
                        f"(distance={target_distance / 1000.0:.1f}km, DOR={dor_boundary / 1000.0:.1f}km)"
                    )
                return

            next_phase = TacticalPhase.DOR_DR
            if hasattr(self.task, 'state_manager') and hasattr(self.task.state_manager, 'set_agent_phase'):
                self.task.state_manager.set_agent_phase(agent_id, next_phase, current_time)
                if is_lead and getattr(self.task.state_manager, 'current_phase', None) == TacticalPhase.TR_DOR:
                    self.task.state_manager.current_phase = next_phase
            if hasattr(self.task, 'phase_lock'):
                self.task.phase_lock[agent_id] = next_phase
            promote_locks[agent_id] = current_time + 12.0

            if env.current_step % 30 == 0:
                logging.info(
                    f"[DRAG_SHOOT-阶段推进] {agent_id} TR_DOR -> {next_phase.value} "
                    f"(phase_time={phase_duration:.1f}s)"
                )
        except Exception:
            pass

    def _get_formation_label(self, agent_id: str) -> str:
        if hasattr(self.task, '_get_formation_label'):
            try:
                return str(self.task._get_formation_label(agent_id))
            except Exception:
                pass
        return '/'.join(self._get_formation_agents(agent_id))

    def _get_base_heading(self, env, agent_id: str) -> float:
        bearing = TacticalUtils.get_enemy_bearing(env, agent_id)
        if bearing is not None:
            return float(bearing)
        return float(env.agents[agent_id].get_property_value(c.attitude_psi_deg))

    def _get_horizontal_course_heading(self, aircraft) -> float:
        try:
            velocity = aircraft.get_velocity()
            vx = float(velocity[0])
            vy = float(velocity[1])
            if np.hypot(vx, vy) > 1.0:
                return float((np.rad2deg(np.arctan2(vx, vy)) + 360.0) % 360.0)
        except Exception:
            pass
        try:
            return float(aircraft.get_property_value(c.attitude_psi_deg))
        except Exception:
            return 0.0

    def _smooth_heading(self, key: str, target_heading: float, alpha: float = 0.3) -> float:
        target_heading = float(target_heading) % 360.0
        previous = self._heading_smoother.get(key)
        if previous is None:
            smoothed = target_heading
        else:
            delta = self.task._normalize_angle_diff(target_heading - previous)
            smoothed = (previous + alpha * delta) % 360.0
        self._heading_smoother[key] = smoothed
        return smoothed

    def _get_formation_axis_heading(self, env, agent_id: str, alpha: float = 0.25) -> float:
        formation_label = self._get_formation_label(agent_id)
        leader_id = agent_id if self._is_formation_lead(env, agent_id) else self._get_teammate_id(agent_id)
        leader_pool = getattr(env, '_jsbsims', {})
        leader = leader_pool.get(leader_id) if leader_id else None
        if leader and getattr(leader, 'is_alive', False):
            raw_heading = self._get_horizontal_course_heading(leader)
        else:
            raw_heading = float(env.agents[agent_id].get_property_value(c.attitude_psi_deg))
        return self._smooth_heading(f'formation_axis:{formation_label}', raw_heading, alpha=alpha)

    def on_formation_tactic_changed(self, formation_label: str, old_tactic: str, new_tactic: str):
        keep_lock = {'HIGH_LOW_ATTACK', 'TACTICAL_EVASION'}
        if (
            (old_tactic == 'HIGH_LOW_ATTACK' and new_tactic not in keep_lock)
            or (new_tactic == 'HIGH_LOW_ATTACK' and old_tactic not in keep_lock)
        ):
            self.high_low_heading_locks.pop(formation_label, None)
        self.pre_nlt_shape_states.pop(formation_label, None)

    def _get_locked_high_low_heading(self, env, agent_id: str) -> float:
        formation_label = self._get_formation_label(agent_id)
        lock_state = self.high_low_heading_locks.get(formation_label)
        if lock_state is not None:
            return float(lock_state['heading'])

        formation_agents = self._get_formation_agents(agent_id)
        leader_id = next(
            (
                aid for aid in formation_agents
                if aid in env.agents
                and getattr(env.agents[aid], 'is_alive', False)
                and self._is_formation_lead(env, aid)
            ),
            agent_id
        )

        heading_source = env.agents.get(leader_id) or env.agents.get(agent_id)
        if heading_source is not None:
            locked_heading = float(heading_source.get_property_value(c.attitude_psi_deg))
        else:
            locked_heading = 0.0

        self.high_low_heading_locks[formation_label] = {
            'heading': locked_heading,
            'leader': leader_id,
        }
        logging.info(f"[HIGH_LOW_ATTACK-航向锁定] {formation_label} 锁定航向{locked_heading:.1f}°")
        return locked_heading

    def _exit_second_attack_to_rtb(self, env, agent_id: str, reason: str) -> tuple:
        formation_agents = list(self.task._get_formation_agents(agent_id, env)) if hasattr(self.task, '_get_formation_agents') else [agent_id]
        alive_enemy_exists = self.task._has_alive_enemy_anywhere(env, agent_id)

        if agent_id.startswith('A') and alive_enemy_exists:
            override_plan = self.task._build_red_reengage_override_plan(
                env,
                agent_id,
                current_phase_name="second_attack_rtb",
                reason=reason or "second_attack_rtb",
            ) if hasattr(self.task, '_build_red_reengage_override_plan') else {}
            preserve_defense = bool(override_plan.get("preserve_defense", False))
            alive_formation_agents = list(override_plan.get("formation_agents", []))
            replacement_tactic = str(override_plan.get("replacement_tactic", 'ADAPTIVE_ATTACK'))

            for aid in formation_agents:
                self.second_attack_coordination.pop(aid, None)
                self.force_rtb_after_evasion.pop(aid, None)
                self.evasion_min_distances.pop(aid, None)
                self.sbs_second_attack_min_distance.pop(aid, None)
                self.sbs_evasion_min_distance.pop(aid, None)
                self.sbs_evasion_active.pop(aid, None)
                if hasattr(self, 'second_attack_min_distance'):
                    self.second_attack_min_distance.pop(aid, None)
                if hasattr(self, 'evasion_distances'):
                    self.evasion_distances.pop(aid, None)
                if hasattr(self.task, 'returning_agents'):
                    self.task.returning_agents.discard(aid)
                if hasattr(self.task, 'set_agent_second_attack'):
                    self.task.set_agent_second_attack(aid, not preserve_defense)

            if preserve_defense and hasattr(self.task, '_execute_defensive_preserve_action'):
                logging.info(f"🛡️ [二次进攻RTB守区保留] {agent_id} 原因={reason}")
                return self.task._execute_defensive_preserve_action(
                    env,
                    agent_id,
                    reason=f"block_rtb_preserve:{reason}",
                )

            for aid in alive_formation_agents:
                if hasattr(self.task, '_set_agent_tactic'):
                    self.task._set_agent_tactic(aid, replacement_tactic)

            if replacement_tactic == 'FORMATION_RESET' and alive_formation_agents:
                formation_manager = getattr(self.task, '_get_formation_reset_manager', lambda _aid: None)(agent_id)
                if formation_manager is not None and hasattr(formation_manager, 'start_formation_reset') and (not formation_manager.is_reset_active()):
                    try:
                        formation_manager.start_formation_reset(env, env.current_step * env.time_interval, alive_formation_agents)
                    except Exception:
                        pass
                if hasattr(self.task, '_execute_formation_reset_procedure'):
                    logging.warning(f"🛑 [二次进攻RTB拦截] {agent_id} 原因={reason} -> FORMATION_RESET")
                    return self.task._execute_formation_reset_procedure(env, agent_id)

            if replacement_tactic == 'DEFENSIVE_GUARD' and hasattr(self.task, '_execute_defensive_preserve_action'):
                logging.warning(f"🛑 [二次进攻RTB拦截] {agent_id} 原因={reason} -> DEFENSIVE_GUARD")
                return self.task._execute_defensive_preserve_action(
                    env,
                    agent_id,
                    reason=f"block_rtb_guard:{reason}",
                )

            if hasattr(self.task, '_execute_adaptive_attack'):
                logging.warning(f"🛑 [二次进攻RTB拦截] {agent_id} 原因={reason} -> {replacement_tactic}")
                return self.task._execute_adaptive_attack(
                    env,
                    agent_id,
                    env.current_step * env.time_interval,
                    reason=f"block_rtb:{reason}",
                )

        if hasattr(self.task, '_transition_formation_to_rtb'):
            self.task._transition_formation_to_rtb(env, agent_id, reason)
        else:
            if hasattr(self.task, '_set_agent_tactic'):
                self.task._set_agent_tactic(agent_id, 'TACTICAL_TURN')
            if hasattr(self.task, 'returning_agents'):
                for aid in formation_agents:
                    self.task.returning_agents.add(aid)
            if hasattr(self.task, 'set_agent_second_attack'):
                for aid in formation_agents:
                    self.task.set_agent_second_attack(aid, False)

        for aid in formation_agents:
            self.second_attack_coordination.pop(aid, None)
            self.force_rtb_after_evasion.pop(aid, None)
            self.evasion_min_distances.pop(aid, None)
            self.sbs_second_attack_min_distance.pop(aid, None)
            self.sbs_evasion_min_distance.pop(aid, None)
            self.sbs_evasion_active.pop(aid, None)
            if hasattr(self, 'second_attack_min_distance'):
                self.second_attack_min_distance.pop(aid, None)
            if hasattr(self, 'evasion_distances'):
                self.evasion_distances.pop(aid, None)

        return self.execute_tactical_turn(env, agent_id)

    def _build_altitude_heading_action(self, env, agent_id: str, alt_cmd_value: float,
                                       target_heading: float, climb_vel_cmd: int = 4,
                                       descend_vel_cmd: int = 3) -> tuple:
        alt_cmd = self.task._convert_altitude_to_index(alt_cmd_value)
        hdg_cmd = self.task._get_heading_cmd(env, agent_id, target_heading) if hasattr(self.task, '_get_heading_cmd') else 8
        vel_cmd = climb_vel_cmd if alt_cmd_value > 0 else descend_vel_cmd
        return alt_cmd, hdg_cmd, vel_cmd

    def _get_vertical_speed_mps(self, env, agent_id: str) -> float:
        try:
            return float(-env.agents[agent_id].get_property_value(c.velocities_v_down_fps) * 0.3048)
        except Exception:
            return 0.0

    def _project_offsets_along_heading(self, reference_heading: float, origin_pos: tuple, target_pos: tuple) -> tuple:
        heading_rad = np.deg2rad(reference_heading)
        forward = np.array([np.cos(heading_rad), np.sin(heading_rad)])
        right = np.array([-np.sin(heading_rad), np.cos(heading_rad)])
        rel = np.array([
            float(target_pos[0]) - float(origin_pos[0]),
            float(target_pos[1]) - float(origin_pos[1]),
        ])
        longitudinal = float(np.dot(rel, forward))
        lateral = float(np.dot(rel, right))
        return longitudinal, lateral

    def _get_rtb_heading(self, env, agent_id: str) -> float:
        return 180.0 if agent_id.startswith('A') else 0.0

    def _build_rtb_command_with_alt_guard(self, env, agent_id: str, target_heading: float = None) -> tuple:
        if target_heading is None:
            target_heading = self._get_rtb_heading(env, agent_id)

        if hasattr(self.task, 'build_return_command_with_alt_guard'):
            return self.task.build_return_command_with_alt_guard(env, agent_id, target_heading)

        current_time = env.current_step * env.time_interval
        current_heading = float(env.agents[agent_id].get_property_value(c.attitude_psi_deg))
        current_alt = float(env.agents[agent_id].get_position()[2])
        current_speed = float(np.linalg.norm(env.agents[agent_id].get_velocity()))
        vertical_speed = self._get_vertical_speed_mps(env, agent_id)
        heading_diff = self.task._normalize_angle_diff(float(target_heading) - current_heading)

        if current_alt < 1800.0:
            heading_diff = float(np.clip(heading_diff, -12.0, 12.0))
        elif current_alt < 3000.0:
            heading_diff = float(np.clip(heading_diff, -20.0, 20.0))
        elif current_alt < 4500.0:
            heading_diff = float(np.clip(heading_diff, -30.0, 30.0))

        adjusted_heading = (current_heading + heading_diff) % 360.0
        alt_cmd = 7
        if current_alt < 1800.0:
            alt_cmd = 5
        elif current_alt > 6000.0:
            alt_cmd = 6 if vertical_speed > 2.0 else 7

        vel_cmd = 4
        if current_speed < 240.0:
            vel_cmd = 5
        elif current_speed > 360.0:
            vel_cmd = 3

        hdg_cmd = self.task._get_heading_cmd(env, agent_id, adjusted_heading) if hasattr(self.task, '_get_heading_cmd') else 8

        if env.current_step % 60 == 0:
            logging.info(
                f"🛡️ [RTB-保护] {agent_id} alt={current_alt:.0f}m speed={current_speed:.0f}m/s "
                f"v_speed={vertical_speed:.1f}m/s heading={adjusted_heading:.1f}°"
            )

        return alt_cmd, hdg_cmd, vel_cmd

    def _calculate_shortest_turn_to_north(self, current_heading: float) -> dict:
        """
        计算从当前航向到 0° 的最短转向路径。

        Args:
            current_heading: 当前航向（0-360 度）

        Returns:
            dict: {
                'turn_direction': 'left'/'right'/'none',
                'turn_angle': 转向角度（-180~180 度）, 
                'target_heading': 0.0
            }
        """
        current_heading = current_heading % 360.0
        target_heading = 0.0
        heading_diff = target_heading - current_heading

        if heading_diff > 180:
            heading_diff -= 360
        elif heading_diff < -180:
            heading_diff += 360

        if abs(heading_diff) <= 5.0:
            return {
                'turn_direction': 'none',
                'turn_angle': 0.0,
                'target_heading': 0.0
            }
        elif heading_diff > 0:
            return {
                'turn_direction': 'right',
                'turn_angle': abs(heading_diff),
                'target_heading': 0.0
            }
        else:
            return {
                'turn_direction': 'left',
                'turn_angle': abs(heading_diff),
                'target_heading': 0.0
            }

    def execute_tactical_turn(self, env, agent_id: str) -> tuple:
        """TACTICAL_TURN锛氫弗鏍艰繑鑸紙RTB锛夈€?

        鐢ㄦ埛绾︽潫锛氫笉鍋氱紪闃熼噸鏁?浜屾杩涙敾/閲嶆柊浜ゆ垬锛涗粎鍏佽鈥滄湁瀵煎脊鍏堣閬库€濄€?
        """
        current_time = env.current_step * env.time_interval

        if agent_id.startswith('A'):
            alive_enemy_exists = (
                self.task._has_alive_enemy_anywhere(env, agent_id)
                if hasattr(self.task, '_has_alive_enemy_anywhere')
                else False
            )
            guard_release_until = float(
                getattr(self.task, '_red_guard_release_until', {}).get(agent_id, -999.0)
            )
            if (
                alive_enemy_exists
                and current_time < guard_release_until
                and hasattr(self.task, '_execute_defensive_preserve_action')
            ):
                if hasattr(self.task, 'returning_agents'):
                    self.task.returning_agents.discard(agent_id)
                self.tactical_turn_states.pop(agent_id, None)
                if getattr(env, 'current_step', 0) % 60 == 0:
                    logging.info(
                        "[TACTICAL_TURN冷却拦截] %s 刚解除返航，保持守区防御 %.1fs",
                        agent_id,
                        max(0.0, guard_release_until - current_time),
                    )
                return self.task._execute_defensive_preserve_action(
                    env,
                    agent_id,
                    reason='tactical_turn_guard_release_hold',
                )

        if hasattr(self.task, 'returning_agents'):
            self.task.returning_agents.add(agent_id)

        if hasattr(self.task, '_defense_prev_tactic'):
            self.task._defense_prev_tactic = None
        if hasattr(self.task, 'set_agent_second_attack'):
            try:
                self.task.set_agent_second_attack(agent_id, False)
            except Exception:
                pass

        turn_state = self.tactical_turn_states.get(agent_id)
        if turn_state is None:
            self.tactical_turn_states[agent_id] = {
                'entered_at': current_time,
                'short_skate_done': False,
                'mode': 'return',
            }
            if hasattr(self.task, 'short_skate_states'):
                self.task.short_skate_states.pop(agent_id, None)
            turn_state = self.tactical_turn_states[agent_id]

        current_alt = float(env.agents[agent_id].get_position()[2])
        current_speed = float(np.linalg.norm(env.agents[agent_id].get_velocity()))
        vertical_speed = self._get_vertical_speed_mps(env, agent_id)

        recovering = bool(turn_state.get('mode') == 'recover')
        recover_required = (
            current_alt < 3000.0
            or current_speed < 215.0
            or vertical_speed < -10.0
            or (current_alt < 4500.0 and vertical_speed < -4.0)
        )
        recover_hold = recovering and (
            current_alt < 3600.0
            or current_speed < 235.0
            or vertical_speed < -2.0
        )
        if recover_required or recover_hold:
            if turn_state.get('mode') != 'recover' and self.should_log('tactical_turn_protection', current_time):
                logging.warning(
                    f"[TACTICAL_TURN] {agent_id} alt={current_alt:.0f}m speed={current_speed:.0f}m/s v_speed={vertical_speed:.1f}m/s, switching to recovery mode"
                )
            turn_state['mode'] = 'recover'
            turn_state['short_skate_done'] = True
            return self._build_rtb_command_with_alt_guard(env, agent_id)
        if turn_state.get('mode') == 'recover':
            turn_state['mode'] = 'return'
            if self.should_log('tactical_turn_protection', current_time):
                logging.info(f"[TACTICAL_TURN] {agent_id} recovery complete, resuming RTB")


        if current_alt < 5000.0 or current_speed < 235.0 or vertical_speed < -4.0:
            turn_state['short_skate_done'] = True
            return self._build_rtb_command_with_alt_guard(env, agent_id)

        if not turn_state.get('short_skate_done', False):
            if agent_id not in getattr(self.task, 'short_skate_states', {}):
                return self.task.maneuver_lib.execute_short_skate_precise(env, agent_id, current_time, direction='rtb')


    def _calculate_shortest_turn_to_north(self, current_heading: float) -> dict:
        current_heading = current_heading % 360.0
        heading_diff = -current_heading

        if heading_diff > 180.0:
            heading_diff -= 360.0
        elif heading_diff < -180.0:
            heading_diff += 360.0

        if abs(heading_diff) <= 5.0:
            turn_direction = 'none'
            turn_angle = 0.0
        elif heading_diff > 0.0:
            turn_direction = 'right'
            turn_angle = abs(heading_diff)
        else:
            turn_direction = 'left'
            turn_angle = abs(heading_diff)

        return {
            'turn_direction': turn_direction,
            'turn_angle': turn_angle,
            'target_heading': 0.0,
        }

    def execute_tactical_turn(self, env, agent_id: str) -> tuple:
        current_time = env.current_step * env.time_interval

        if agent_id.startswith('A'):
            alive_enemy_exists = self.task._has_alive_enemy_anywhere(env, agent_id)
            local_alive_enemy_exists = (
                self.task._has_alive_enemy_in_current_env(env, agent_id)
                if hasattr(self.task, '_has_alive_enemy_in_current_env')
                else alive_enemy_exists
            )
            guard_release_until = float(
                getattr(self.task, '_red_guard_release_until', {}).get(agent_id, -999.0)
            )
            if (
                alive_enemy_exists
                and local_alive_enemy_exists
                and current_time < guard_release_until
                and hasattr(self.task, '_execute_defensive_preserve_action')
            ):
                if hasattr(self.task, 'returning_agents'):
                    self.task.returning_agents.discard(agent_id)
                self.tactical_turn_states.pop(agent_id, None)
                if getattr(env, 'current_step', 0) % 60 == 0:
                    logging.info(
                        "[TACTICAL_TURN冷却拦截] %s 刚解除返航，保持守区防御 %.1fs",
                        agent_id,
                        max(0.0, guard_release_until - current_time),
                    )
                return self.task._execute_defensive_preserve_action(
                    env,
                    agent_id,
                    reason='tactical_turn_guard_release_hold',
                )
            if alive_enemy_exists and local_alive_enemy_exists and hasattr(self.task, '_execute_adaptive_attack'):
                override_plan = self.task._build_red_reengage_override_plan(
                    env,
                    agent_id,
                    current_phase_name="block_tactical_turn",
                    reason="block_tactical_turn",
                ) if hasattr(self.task, '_build_red_reengage_override_plan') else {}
                preserve_defense = bool(override_plan.get("preserve_defense", False))
                replacement_tactic = str(override_plan.get("replacement_tactic", 'ADAPTIVE_ATTACK'))
                formation_agents = list(override_plan.get("formation_agents", []))
                if not formation_agents:
                    formation_agents = [agent_id]
                for aid in formation_agents:
                    if hasattr(self.task, 'returning_agents'):
                        self.task.returning_agents.discard(aid)
                    if hasattr(self.task, 'set_agent_second_attack'):
                        try:
                            self.task.set_agent_second_attack(aid, not preserve_defense)
                        except Exception:
                            pass
                    if hasattr(self.task, '_set_agent_tactic'):
                        self.task._set_agent_tactic(aid, 'DEFENSIVE_GUARD' if preserve_defense else replacement_tactic)
                    self.tactical_turn_states.pop(aid, None)
                if preserve_defense:
                    logging.info(f"🛡️ [TACTICAL_TURN守区保留] {agent_id} 敌机仍存活，但保持守区止追")
                    return self.task._execute_defensive_preserve_action(
                        env,
                        agent_id,
                        reason='block_tactical_turn_preserve',
                    )
                logging.warning(f"🛑 [TACTICAL_TURN阻断] {agent_id} 敌机仍存活，禁止返航回转，改为{replacement_tactic}")
                if replacement_tactic == 'DEFENSIVE_GUARD':
                    return self.task._execute_defensive_preserve_action(
                        env,
                        agent_id,
                        reason='block_tactical_turn_guard',
                    )
                if replacement_tactic == 'FORMATION_RESET' and hasattr(self.task, '_execute_formation_reset_procedure'):
                    return self.task._execute_formation_reset_procedure(env, agent_id)
                return self.task._execute_adaptive_attack(
                    env,
                    agent_id,
                    current_time,
                    reason='block_tactical_turn',
                )

        if hasattr(self.task, 'returning_agents'):
            self.task.returning_agents.add(agent_id)

        if hasattr(self.task, '_defense_prev_tactic'):
            self.task._defense_prev_tactic = None
        if hasattr(self.task, 'set_agent_second_attack'):
            try:
                self.task.set_agent_second_attack(agent_id, False)
            except Exception:
                pass

        turn_state = self.tactical_turn_states.get(agent_id)
        if turn_state is None:
            self.tactical_turn_states[agent_id] = {
                'entered_at': current_time,
                'short_skate_done': False,
                'mode': 'return',
            }
            if hasattr(self.task, 'short_skate_states'):
                self.task.short_skate_states.pop(agent_id, None)
            turn_state = self.tactical_turn_states[agent_id]

        current_alt = float(env.agents[agent_id].get_position()[2])
        current_speed = float(np.linalg.norm(env.agents[agent_id].get_velocity()))
        vertical_speed = self._get_vertical_speed_mps(env, agent_id)

        recovering = bool(turn_state.get('mode') == 'recover')
        recover_required = (
            current_alt < 3000.0
            or current_speed < 215.0
            or vertical_speed < -10.0
            or (current_alt < 4500.0 and vertical_speed < -4.0)
        )
        recover_hold = recovering and (
            current_alt < 3600.0
            or current_speed < 235.0
            or vertical_speed < -2.0
        )
        if recover_required or recover_hold:
            if turn_state.get('mode') != 'recover' and self.should_log('tactical_turn_protection', current_time):
                logging.warning(
                    f"[TACTICAL_TURN] {agent_id} alt={current_alt:.0f}m speed={current_speed:.0f}m/s "
                    f"v_speed={vertical_speed:.1f}m/s, switching to recovery mode"
                )
            turn_state['mode'] = 'recover'
            turn_state['short_skate_done'] = True
            return self._build_rtb_command_with_alt_guard(env, agent_id)
        if turn_state.get('mode') == 'recover':
            turn_state['mode'] = 'return'
            if self.should_log('tactical_turn_protection', current_time):
                logging.info(
                    f"[TACTICAL_TURN] {agent_id} recovery complete, resuming RTB"
                )

        if current_alt < 5000.0 or current_speed < 235.0 or vertical_speed < -4.0:
            turn_state['short_skate_done'] = True
            return self._build_rtb_command_with_alt_guard(env, agent_id)

        if not turn_state.get('short_skate_done', False):
            if agent_id not in getattr(self.task, 'short_skate_states', {}):
                return self.task.maneuver_lib.execute_short_skate_precise(env, agent_id, current_time, direction='rtb')

            short_skate_state = self.task.short_skate_states.get(agent_id, {})
            if short_skate_state.get('phase') == 'escape':
                phase_start = float(short_skate_state.get('phase_start_time', current_time))
                if current_time - phase_start >= 15.0:
                    turn_state['short_skate_done'] = True

            if not turn_state.get('short_skate_done', False):
                return self.task.maneuver_lib.execute_short_skate_precise(env, agent_id, current_time, direction='rtb')

        return self._build_rtb_command_with_alt_guard(env, agent_id)

    def _heading_to_attack_wp_or_default(self, env, agent_id: str, default_heading: float) -> float:
        try:
            wp = self.task.attack_waypoint.get(agent_id)
            if not wp:
                return default_heading
            cur = env.agents[agent_id].get_position()
            vec = np.array(wp) - np.array(cur)
            if np.linalg.norm(vec[:2]) < 1.0:
                return default_heading
            desired = float(np.rad2deg(np.arctan2(vec[1], vec[0])) % 360.0)
            current_heading = float(env.agents[agent_id].get_property_value(c.attitude_psi_deg))
            delta = self.task._normalize_angle_diff(desired - current_heading)
            delta = float(np.clip(delta, -10.0, 10.0))
            target = (current_heading + delta) % 360.0
            return target
        except Exception:
            return default_heading

