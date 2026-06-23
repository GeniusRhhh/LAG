from __future__ import annotations

import logging

import numpy as np

from envs.JSBSim.core.catalog import Catalog as c

from core.target_assignment import get_enemy_team
from tactical_types import TacticalPhase, get_target_with_fallback
from tactical_utils import TacticalUtils


class TacticalTaskActionMixin:
    def _execute_red_reengage_override_action(
        self,
        env,
        agent_id,
        current_time,
        *,
        current_phase_name="",
        reason="",
        allow_template_reset=True,
    ):
        override_plan = {}
        if hasattr(self, '_build_red_reengage_override_plan'):
            try:
                override_plan = self._build_red_reengage_override_plan(
                    env,
                    agent_id,
                    current_phase_name=current_phase_name,
                    reason=reason or current_phase_name or "reengage_override",
                    allow_template_reset=allow_template_reset,
                )
            except Exception:
                override_plan = {}
        formation_agents = override_plan.get("formation_agents", []) or [agent_id]
        replacement_tactic = str(override_plan.get("replacement_tactic", 'ADAPTIVE_ATTACK'))
        formation_manager = None
        if hasattr(self, '_get_formation_reset_manager'):
            try:
                formation_manager = self._get_formation_reset_manager(agent_id)
            except Exception:
                formation_manager = None
        if formation_manager is not None and replacement_tactic != 'FORMATION_RESET':
            try:
                formation_manager.active = False
                formation_manager.phase = 'inactive'
                formation_manager.active_agents = []
                formation_manager.zero_heading_hold_start = None
            except Exception:
                pass

        if replacement_tactic == 'DEFENSIVE_GUARD':
            for aid in formation_agents:
                if hasattr(self, 'set_agent_second_attack'):
                    self.set_agent_second_attack(aid, False)
                try:
                    self.executor.tactical_turn_states.pop(aid, None)
                except Exception:
                    pass
                if hasattr(self, 'formation_reset_agents'):
                    self.formation_reset_agents.discard(aid)
                self._set_agent_tactic(aid, 'DEFENSIVE_GUARD')
            if hasattr(self, '_deescalate_red_formation_to_defense'):
                try:
                    self._deescalate_red_formation_to_defense(
                        env,
                        agent_id,
                        reason=reason or current_phase_name or "reengage_override_guard",
                        clear_returning=False,
                    )
                except Exception:
                    pass
            return self._execute_defensive_preserve_action(
                env,
                agent_id,
                reason=reason or current_phase_name or "reengage_override_guard",
            )

        for aid in formation_agents:
            self.returning_agents.discard(aid)
            try:
                self.executor.tactical_turn_states.pop(aid, None)
            except Exception:
                pass
            self._set_agent_tactic(aid, replacement_tactic)
            if hasattr(self, 'formation_reset_agents') and replacement_tactic != 'FORMATION_RESET':
                self.formation_reset_agents.discard(aid)

        if replacement_tactic == 'FORMATION_RESET':
            return self._execute_formation_reset_procedure(env, agent_id)

        return self._execute_adaptive_attack(
            env,
            agent_id,
            current_time,
            reason=reason or current_phase_name or "reengage_override",
        )

    def get_action(self, env, agent_id):
        """
        主决策入口 - 重构版本
        委托给各个模块处理，保持简洁
        """
        current_time = env.current_step * env.time_interval
        self.env = env
        effective_tactic = self._get_agent_tactic(agent_id)
        formation_label = self._get_formation_label(agent_id, env)

        # 🔥 调试：只打印我方（A开头）的函数调用追踪（降低频率，每300步=60秒）
        if agent_id.startswith('A') and env.current_step % 300 == 0:
            current_phase = self.agent_phases.get(agent_id, None)
            phase_str = current_phase.value if current_phase else 'None'
            pos = env.agents[agent_id].get_position()
            heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
            logging.warning(
                f"🔍 [函数调用-{formation_label}-{agent_id}] 进入get_action, 阶段={phase_str}, 战术={effective_tactic}, "
                f"位置=({pos[0]/1000:.1f}, {pos[1]/1000:.1f}, {pos[2]:.1f}m), 航向={heading:.1f}°, 时间={current_time:.1f}s"
            )
            if hasattr(self, '_log_key_event'):
                self._log_key_event(
                    env,
                    agent_id,
                    "进入函数",
                    current_time=current_time,
                    当前阶段=phase_str,
                    当前战术=effective_tactic,
                    进入函数="get_action",
                    执行机动="待分派",
                    当前状态=f"formation={formation_label}",
                    目标点=f"pos=({pos[0]/1000:.1f},{pos[1]/1000:.1f},{pos[2]:.1f}m), heading={heading:.1f}deg",
                    当前指令="pending",
                    退出条件="返回动作或默认动作",
                    是否满足退出="否",
                )

        # ✅ 记录到trace_logger：进入get_action（只在节点切换或战术变化时记录，降低频率）
        if agent_id.startswith('A'):
            try:
                current_phase = self.agent_phases.get(agent_id, None)
                phase_str = current_phase.value if current_phase else 'None'
                should_log = False

                if not hasattr(self, '_last_logged_phase'):
                    self._last_logged_phase = {}
                if agent_id not in self._last_logged_phase or self._last_logged_phase[agent_id] != phase_str:
                    should_log = True
                    self._last_logged_phase[agent_id] = phase_str

                if not hasattr(self, '_last_logged_tactic'):
                    self._last_logged_tactic = {}
                if agent_id not in self._last_logged_tactic or self._last_logged_tactic[agent_id] != self.selected_tactic:
                    should_log = True
                    self._last_logged_tactic[agent_id] = self.selected_tactic

                if env.current_step % 300 == 0:
                    should_log = True

                if should_log:
                    from utils.trace_logger import trace_event
                    pos = env.agents[agent_id].get_position()
                    heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
                    trace_event(
                        事件="进入get_action",
                        env=env,
                        模块="tactical_task",
                        类型="CALL",
                        状态="ENTER",
                        我机=agent_id,
                        阶段=phase_str,
                        战术=str(self.selected_tactic),
                        说明=f"位置=({pos[0]/1000:.1f}, {pos[1]/1000:.1f}, {pos[2]:.1f}m), 航向={heading:.1f}°",
                    )
            except Exception:
                pass

        if not env.agents[agent_id].is_alive:
            return self._get_default_action(env, agent_id)

        # ✅ 算法1意图识别“预热/持续喂窗口”
        if agent_id.startswith('A') and self.situation_algorithm_switcher is not None:
            try:
                if getattr(self, "_intent_prewarm_last_step", None) != env.current_step:
                    self._intent_prewarm_last_step = env.current_step
                    if not hasattr(self, "_enemy_intent_cache"):
                        self._enemy_intent_cache = {}
                    my_ref = env.agents.get("A0100") if ("A0100" in env.agents and env.agents["A0100"].is_alive) else env.agents[agent_id]
                    enemy_ids = sorted(
                        eid
                        for eid, enemy in env.agents.items()
                        if str(eid).startswith("B") and getattr(enemy, "is_alive", False)
                    )
                    for eid in enemy_ids:
                        if eid in env.agents and env.agents[eid].is_alive:
                            if eid in self._enemy_intent_cache:
                                cached_intent, cached_step = self._enemy_intent_cache[eid]
                                if cached_step == env.current_step:
                                    continue
                            intent = self.situation_algorithm_switcher.recognize_intent(env.agents[eid], my_ref, env)
                            self._enemy_intent_cache[eid] = (intent, env.current_step)
            except Exception:
                pass

        self._update_phase(env, agent_id, current_time)

        if agent_id.startswith('A') and env.current_step % 20 == 0:
            try:
                from utils.trace_logger import trace_throttle
                snapshot_data = {}
                for aid in ['A0100', 'A0200', 'B0100', 'B0200']:
                    if aid in env.agents and env.agents[aid].is_alive:
                        ac = env.agents[aid]
                        pos = ac.get_position()
                        vel = ac.get_velocity()
                        try:
                            heading_rad = ac.get_property_value('attitude/psi-rad')
                            heading_deg = ac.get_property_value('attitude/psi-deg')
                            heading = None
                            if heading_rad is not None:
                                heading = float(np.rad2deg(float(heading_rad)))
                            elif heading_deg is not None:
                                heading = float(heading_deg)
                            if heading is None or not np.isfinite(heading):
                                heading = 0.0
                            heading = (heading + 360.0) % 360.0
                        except Exception:
                            heading = 0.0

                        missiles_remaining = None
                        try:
                            if hasattr(self, 'missile_manager'):
                                missiles_remaining = self.missile_manager.get_remaining_missiles(aid)
                        except Exception:
                            pass

                        if missiles_remaining is None:
                            try:
                                if hasattr(self, 'state_manager') and hasattr(self.state_manager, 'missiles_remaining'):
                                    missiles_remaining = self.state_manager.missiles_remaining.get(aid, None)
                            except Exception:
                                pass

                        if missiles_remaining is None:
                            try:
                                for attr in ['num_left_missiles', 'num_missiles', 'missiles_remaining', 'missile_remaining', 'missile_count', 'missiles']:
                                    if hasattr(ac, attr):
                                        try:
                                            val = getattr(ac, attr)
                                            if val is not None:
                                                missiles_remaining = int(val)
                                                break
                                        except Exception:
                                            continue
                            except Exception:
                                pass

                        if missiles_remaining is None:
                            missiles_remaining = 0
                        snapshot_data[aid] = {
                            '位置_m': [round(pos[0], 1), round(pos[1], 1), round(pos[2], 1)],
                            '速度_m/s': round(np.linalg.norm(vel), 1),
                            '航向_deg': round(heading, 1),
                            '剩余导弹': int(missiles_remaining),
                        }
                if hasattr(self, 'executor') and hasattr(self.executor, '_last_action'):
                    for aid in ['A0100', 'A0200']:
                        if aid in self.executor._last_action:
                            action = self.executor._last_action[aid]
                            if aid in snapshot_data:
                                snapshot_data[aid]['动作指令'] = (
                                    f"alt={action[0]},hdg={action[1]},vel={action[2]}"
                                    if isinstance(action, (list, tuple)) and len(action) >= 3
                                    else str(action)
                                )

                trace_throttle(
                    key="situation_snapshot:all",
                    min_steps=20,
                    标题="态势快照-周期",
                    env=env,
                    模块="tactical_task",
                    类型="STATE",
                    状态="PERIODIC",
                    我机=agent_id,
                    说明=f"周期态势快照：记录所有飞机位置/航向/速度/剩弹/动作指令",
                    数据=snapshot_data,
                )
            except Exception as e:
                logging.debug(f"态势快照记录失败: {e}")

        self.missile_manager.update_missile_status(env, current_time)
        self._handle_missile_launches(env, current_time)

        if agent_id.startswith('A'):
            try:
                global_enemy_alive = int(self._get_global_alive_enemy_count(env, agent_id))
                alive_enemy_exists = global_enemy_alive > 0 if global_enemy_alive >= 0 else self._has_alive_enemy_anywhere(env, agent_id)
            except Exception:
                global_enemy_alive = -1
                alive_enemy_exists = True
            if not alive_enemy_exists:
                self.returning_agents.discard(agent_id)
                try:
                    self.executor.tactical_turn_states.pop(agent_id, None)
                except Exception:
                    pass
                try:
                    self._deescalate_red_formation_to_defense(env, agent_id, reason="mission_guard_no_enemy")
                except Exception:
                    pass
                patrol_command = self._build_red_defensive_patrol_command(
                    env,
                    agent_id,
                    target=None,
                    distance=None,
                    reason="mission_guard_no_enemy",
                )
                if patrol_command is not None:
                    if env.current_step % 120 == 0:
                        logging.info(f"🛡️ [NO_ENEMY_GUARD] {agent_id} 无存活敌机，继续守区巡逻直到任务结束")
                    return patrol_command
            try:
                from simulation.radar_manager import get_unified_radar_manager
                rm = get_unified_radar_manager()
                rwr_lvl = rm.get_rwr_threat_level(agent_id) if rm else 0
                lock_tgt = getattr(rm, 'friendly_lock_targets', {}).get(agent_id) if rm else None
                ecm_on = bool(getattr(rm, 'ecm_states', {}).get(agent_id, False)) if rm else False
                incoming_missiles = any(
                    getattr(missile, 'is_alive', False)
                    for missile in getattr(env.agents[agent_id], 'under_missiles', [])
                )

                if rwr_lvl >= 5:
                    if self.selected_tactic not in ('TACTICAL_EVASION', 'FORMATION_RESET'):
                        self._defense_prev_tactic = self.selected_tactic
                        self._set_agent_tactic(agent_id, 'TACTICAL_EVASION')
                        if env.current_step % 30 == 0:
                            logging.warning(
                                f"🛡️ [一级打断] {agent_id} RWR={rwr_lvl} → 强制TACTICAL_EVASION (ECM={'ON' if ecm_on else 'OFF'})"
                            )
                elif rwr_lvl >= 4:
                    if getattr(self.config, "friendly_intent", "CONSERVATIVE_CLEAR") == "AGGRESSIVE_CLEAR":
                        self.executor.evasion_aggressive_mode = getattr(self.executor, "evasion_aggressive_mode", {})
                        self.executor.evasion_aggressive_mode[agent_id] = True
                        if env.current_step % 60 == 0:
                            logging.warning(f"🛡️ [二级打断-激进] {agent_id} RWR={rwr_lvl} → 继续战术但增强规避参数")
                    else:
                        _cur_tac = self._get_agent_tactic(agent_id)
                        if _cur_tac not in ('TACTICAL_EVASION', 'FORMATION_RESET'):
                            self._defense_prev_tactic = _cur_tac
                            self._set_agent_tactic(agent_id, 'TACTICAL_EVASION')
                            if env.current_step % 60 == 0:
                                logging.warning(f"🛡️ [二级打断] {agent_id} RWR={rwr_lvl} → TACTICAL_TURN")
                elif rwr_lvl >= 2:
                    if lock_tgt is None:
                        self.executor.evasion_aggressive_mode = getattr(self.executor, "evasion_aggressive_mode", {})
                        self.executor.evasion_aggressive_mode[agent_id] = False
                        if env.current_step % 120 == 0:
                            logging.info(f"🛡️ [三级调整] {agent_id} RWR={rwr_lvl} 且雷达失锁 → 降低机动激进度")

                    if rwr_lvl == 2 and (lock_tgt is not None or incoming_missiles):
                        if not hasattr(self, '_rwr_crank_state'):
                            self._rwr_crank_state = {}
                        if agent_id not in self._rwr_crank_state:
                            import random
                            self._rwr_crank_state[agent_id] = {'direction': random.choice(['left', 'right']), 'last_change': current_time}
                        elif current_time - self._rwr_crank_state[agent_id]['last_change'] > 30.0:
                            import random
                            self._rwr_crank_state[agent_id] = {'direction': random.choice(['left', 'right']), 'last_change': current_time}

                        direction = self._rwr_crank_state[agent_id]['direction']
                        if env.current_step % 180 == 0:
                            logging.info(f"🛡️ [三级调整-动作] {agent_id} RWR=2 → 执行轻微{direction}Crank(±10°)降低锁定概率")
                        if direction == 'left':
                            return self._execute_tactical_crank(env, agent_id, direction='left', climb=False, angle=10.0)
                        return self._execute_tactical_crank(env, agent_id, direction='right', climb=False, angle=10.0)
                    elif rwr_lvl == 3 and (lock_tgt is not None or incoming_missiles):
                        if env.current_step % 180 == 0:
                            logging.info(f"🛡️ [三级调整-动作] {agent_id} RWR=3 → 执行中等Crank(±15°)")
                        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
                        target_bearing = self._get_enemy_bearing(env, agent_id)
                        if target_bearing is not None:
                            heading_diff = self._normalize_angle_diff(target_bearing - current_heading)
                            direction = 'right' if heading_diff > 0 else 'left'
                            return self._execute_tactical_crank(env, agent_id, direction=direction, climb=False, angle=15.0)
            except Exception:
                pass

        if agent_id.startswith('B'):
            return self._get_enemy_action(env, agent_id)

        if agent_id.startswith('A') and agent_id in self.returning_agents:
            alive_enemy_exists = self._has_alive_enemy_anywhere(env, agent_id)
            if alive_enemy_exists:
                guard_release_until = float(getattr(self, '_red_guard_release_until', {}).get(agent_id, -999.0))
                guard_release_active = float(current_time) < guard_release_until
                current_tactic = str(self._get_agent_tactic(agent_id) or '')
                preserve_defense = bool(guard_release_active or current_tactic == 'DEFENSIVE_GUARD')
                try:
                    preserve_defense = bool(
                        preserve_defense
                        or
                        self._evaluate_red_defensive_posture(
                            env,
                            agent_id,
                            current_phase_name="rtb_block",
                            reason="rtb_block",
                        ).get("preserve", False)
                    )
                except Exception:
                    preserve_defense = bool(guard_release_active or current_tactic == 'DEFENSIVE_GUARD')
                if preserve_defense and env.current_step % 120 == 0:
                    logging.info(f"🛡️ [RTB守区保留] {agent_id} 保持防御态势，不强制改回攻击")
                if preserve_defense:
                    return self._execute_defensive_preserve_action(
                        env,
                        agent_id,
                        reason="rtb_block_preserve",
                    )
                else:
                    try:
                        current_phase_obj = self.state_manager.get_agent_phase(agent_id)
                        current_phase_name = getattr(current_phase_obj, 'value', '')
                    except Exception:
                        current_phase_name = ''
                    if env.current_step % 120 == 0:
                        logging.warning(
                            f"🛑 [RTB动作拦截] {agent_id} 敌机仍存活，禁止返航，改由统一仲裁处理"
                        )
                    return self._execute_red_reengage_override_action(
                        env,
                        agent_id,
                        current_time,
                        current_phase_name=current_phase_name,
                        reason=f"rtb_block:{current_phase_name or 'unknown'}",
                        allow_template_reset=True,
                    )
                    override_plan = self._build_red_reengage_override_plan(
                        env,
                        agent_id,
                        current_phase_name=current_phase_name,
                        reason="rtb_block",
                    )
                    formation_agents = override_plan.get("formation_agents", []) or [agent_id]
                    replacement_tactic = override_plan.get("replacement_tactic", 'ADAPTIVE_ATTACK')
                    for aid in formation_agents:
                        self.returning_agents.discard(aid)
                        try:
                            self.executor.tactical_turn_states.pop(aid, None)
                        except Exception:
                            pass
                        self._set_agent_tactic(aid, replacement_tactic)
                    self.returning_agents.discard(agent_id)
                    try:
                        self.executor.tactical_turn_states.pop(agent_id, None)
                    except Exception:
                        pass
                    if env.current_step % 120 == 0:
                        logging.warning(
                        f"🛑 [RTB动作拦截] {agent_id} 敌机仍存活，禁止返航，改为{replacement_tactic}"
                        )
                    if replacement_tactic == 'DEFENSIVE_GUARD':
                        return self._execute_defensive_preserve_action(
                            env,
                            agent_id,
                            reason=f"rtb_block:{current_phase_name or 'unknown'}",
                        )
                    if replacement_tactic == 'FORMATION_RESET':
                        return self._execute_formation_reset_procedure(env, agent_id)
                    return self._execute_adaptive_attack(
                        env,
                        agent_id,
                        current_time,
                        reason=f"rtb_block:{current_phase_name or 'unknown'}",
                )

        if agent_id in self.returning_agents:
            if agent_id.startswith('A'):
                alive_enemy_exists = self._has_alive_enemy_anywhere(env, agent_id)
                if alive_enemy_exists:
                    guard_release_until = float(getattr(self, '_red_guard_release_until', {}).get(agent_id, -999.0))
                    guard_release_active = float(current_time) < guard_release_until
                    current_tactic = str(self._get_agent_tactic(agent_id) or '')
                    if guard_release_active or current_tactic == 'DEFENSIVE_GUARD':
                        return self._execute_defensive_preserve_action(
                            env,
                            agent_id,
                            reason="rtb_hard_block_guard_hold",
                        )
                    return self._execute_red_reengage_override_action(
                        env,
                        agent_id,
                        current_time,
                        current_phase_name="rtb_hard_block",
                        reason="rtb_hard_block",
                        allow_template_reset=True,
                    )
                    override_plan = self._build_red_reengage_override_plan(
                        env,
                        agent_id,
                        current_phase_name="rtb_hard_block",
                        reason="rtb_hard_block",
                    )
                    replacement_tactic = override_plan.get("replacement_tactic", 'ADAPTIVE_ATTACK')
                    if replacement_tactic == 'DEFENSIVE_GUARD':
                        return self._execute_defensive_preserve_action(
                            env,
                            agent_id,
                            reason="rtb_hard_block_preserve",
                        )
                    self.returning_agents.discard(agent_id)
                    try:
                        self.executor.tactical_turn_states.pop(agent_id, None)
                    except Exception:
                        pass
                    self._set_agent_tactic(agent_id, replacement_tactic)
                    if replacement_tactic == 'FORMATION_RESET':
                        return self._execute_formation_reset_procedure(env, agent_id)
                    return self._execute_adaptive_attack(
                        env,
                        agent_id,
                        current_time,
                        reason="rtb_hard_block",
                    )
            current_tactic = self._get_agent_tactic(agent_id)
            if current_tactic == 'TACTICAL_TURN':
                return self.executor.execute_tactical_turn(env, agent_id)

            current_alt = float(env.agents[agent_id].get_position()[2])
            current_speed = float(np.linalg.norm(env.agents[agent_id].get_velocity()))
            if current_alt < 4500.0 or current_speed < 230.0:
                if env.current_step % 120 == 0:
                    logging.warning(
                        f"🛡️ [{agent_id}] 返航能量保护触发: 高度={current_alt:.0f}m 速度={current_speed:.0f}m/s，切换TACTICAL_TURN恢复"
                    )
                self._set_agent_tactic(agent_id, 'TACTICAL_TURN')
                return self.executor.execute_tactical_turn(env, agent_id)

            current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
            target_heading = 180.0
            heading_diff = self._normalize_angle_diff(target_heading - current_heading)
            heading_cmd = self._convert_heading_to_index(np.deg2rad(heading_diff))

            if env.current_step % 120 == 0:
                logging.info(
                    f"🏠 [{agent_id}] 执行返航: 当前航向={current_heading:.1f}°, 目标返航航向={target_heading:.1f}°"
                )

            return 7, heading_cmd, 4

        if self._get_agent_tactic(agent_id) is None:
            self._select_tactic_at_phase(env, agent_id)
        selected_tactic = self._get_agent_tactic(agent_id)
        current_phase_obj = None
        try:
            current_phase_obj = self.state_manager.get_agent_phase(agent_id)
        except Exception:
            current_phase_obj = None
        current_phase_name = getattr(current_phase_obj, 'value', '')
        alive_enemy_exists = self._has_alive_enemy_anywhere(env, agent_id)
        nearest_enemy_distance = float('inf')
        if alive_enemy_exists:
            try:
                enemy_id, enemy = self._get_nearest_alive_enemy_in_current_env(env, agent_id)
                if enemy is None and hasattr(self, '_get_nearest_alive_enemy_anywhere'):
                    enemy_id, enemy = self._get_nearest_alive_enemy_anywhere(env, agent_id)
                aircraft = env.agents.get(agent_id)
                if aircraft is not None and enemy is not None and getattr(enemy, 'is_alive', False):
                    nearest_enemy_distance = float(self._calculate_distance_between(aircraft, enemy))
            except Exception:
                nearest_enemy_distance = float('inf')

        if (
            agent_id.startswith('A')
            and selected_tactic is None
            and alive_enemy_exists
            and nearest_enemy_distance <= 160000.0
        ):
            if env.current_step % 60 == 0:
                logging.warning(
                    f"🧭 [NULL_TACTIC_FALLBACK] {agent_id} phase={current_phase_name or 'None'} "
                    f"dist={nearest_enemy_distance/1000:.1f}km -> ADAPTIVE_ATTACK"
                    )
            return self._execute_red_reengage_override_action(
                env,
                agent_id,
                current_time,
                current_phase_name=current_phase_name or "NULL_TACTIC",
                reason=f"null_tactic_range:{nearest_enemy_distance/1000:.1f}km",
                allow_template_reset=True,
            )

        if (
            agent_id.startswith('A')
            and selected_tactic is None
            and current_phase_name in ('TR_DOR', 'DOR_DR', 'DR_MAR', 'BEYOND_MAR')
            and alive_enemy_exists
        ):
            return self._execute_red_reengage_override_action(
                env,
                agent_id,
                current_time,
                current_phase_name=current_phase_name,
                reason=f"late_phase_null_tactic:{current_phase_name}",
                allow_template_reset=True,
            )

        if (
            agent_id.startswith('A')
            and selected_tactic in ('DRAG_SHOOT', 'HIGH_LOW_ATTACK', 'TACTICAL_TURN')
            and current_phase_name in ('DOR_DR', 'DR_MAR', 'BEYOND_MAR')
            and alive_enemy_exists
        ):
            force_direct = True
            if hasattr(self, 'is_agent_second_attack') and self.is_agent_second_attack(agent_id):
                try:
                    force_direct = bool(self._should_force_direct_second_attack(env, agent_id))
                except Exception:
                    force_direct = True
            preserve_defense = False
            try:
                preserve_defense = bool(
                    self._evaluate_red_defensive_posture(
                        env,
                        agent_id,
                        current_phase_name=current_phase_name,
                        reason=f"phase_override:{selected_tactic}",
                    ).get("preserve", False)
                )
            except Exception:
                preserve_defense = False
            if force_direct:
                if preserve_defense:
                    return self._execute_defensive_preserve_action(
                        env,
                        agent_id,
                        reason=f"phase_override_preserve:{selected_tactic}:{current_phase_name}",
                    )
                return self._execute_red_reengage_override_action(
                    env,
                    agent_id,
                    current_time,
                    current_phase_name=current_phase_name,
                    reason=f"phase_override:{selected_tactic}:{current_phase_name}",
                    allow_template_reset=False,
                )

        if (
            agent_id.startswith('A')
            and hasattr(self, '_should_bypass_template_tactic')
            and self._should_bypass_template_tactic(env, agent_id, selected_tactic)
        ):
            return self._execute_red_reengage_override_action(
                env,
                agent_id,
                current_time,
                current_phase_name=current_phase_name or str(selected_tactic or "BYPASS"),
                reason=f"bypass:{selected_tactic}",
                allow_template_reset=False,
            )

        is_second_attack = self.is_agent_second_attack(agent_id) if hasattr(self, 'is_agent_second_attack') else False
        still_have_missiles = False
        if is_second_attack:
            my_aircraft = env.agents.get(agent_id)
            if my_aircraft:
                still_have_missiles = getattr(my_aircraft, 'num_missiles', 0) > 0

        if agent_id.startswith('A') and env.current_step % 120 == 0:
            try:
                from utils.trace_logger import trace_event
                trace_event(
                    事件="调用_check_and_evade_missile",
                    env=env,
                    模块="tactical_task",
                    类型="CALL",
                    状态="ENTER",
                    我机=agent_id,
                    战术=str(selected_tactic),
                )
            except Exception:
                pass

        missile_evasion = self.executor._check_and_evade_missile(env, agent_id)

        if missile_evasion is not None and is_second_attack and still_have_missiles:
            try:
                target_id = None
                for eid in get_enemy_team(agent_id):
                    if eid in env.agents and env.agents[eid].is_alive:
                        target_id = eid
                        break
                if target_id:
                    distance = TacticalUtils.calculate_distance_between(env.agents[agent_id], env.agents[target_id])
                    if 8000 <= distance <= 120000:
                        has_launch_request = False
                        if hasattr(self, 'state_manager') and hasattr(self.state_manager, 'missile_launched'):
                            has_launch_request = self.state_manager.missile_launched.get(agent_id, False)
                        elif hasattr(self, 'missile_launched'):
                            has_launch_request = self.missile_launched.get(agent_id, False)

                        if has_launch_request:
                            if env.current_step % 50 == 0:
                                logging.warning(f"🎯 [二次进攻-优先发射] {agent_id} 距离{distance/1000:.1f}km，优先发射而非规避")
                            missile_evasion = None
            except Exception as e:
                logging.debug(f"⚠️ [二次进攻发射检查异常] {agent_id}: {e}")

        if missile_evasion is not None:
            if selected_tactic == 'TACTICAL_EVASION':
                if agent_id.startswith('A'):
                    logging.warning(
                        f"🚨 [get_action-{agent_id}] 检测到导弹威胁，放弃TACTICAL_EVASION战术，执行导弹规避，时间={current_time:.1f}s"
                    )
                if hasattr(self.executor, 'evasion_states') and agent_id in self.executor.evasion_states:
                    self.executor.evasion_states.pop(agent_id, None)
                state_manager = getattr(self, 'state_manager', None)
                beam_state = getattr(state_manager, 'beam_maneuver_state', None)
                if isinstance(beam_state, dict):
                    beam_state.pop(agent_id, None)
                elif hasattr(self, 'beam_maneuver_state') and agent_id in self.beam_maneuver_state:
                    self.beam_maneuver_state.pop(agent_id, None)
                if hasattr(self, '_defense_prev_tactic') and self._defense_prev_tactic:
                    self._set_agent_tactic(agent_id, self._defense_prev_tactic)
                    self._defense_prev_tactic = None
                else:
                    self._set_agent_tactic(agent_id, None)
            if agent_id.startswith('A') and env.current_step % 200 == 0:
                logging.warning(
                    f"📋 [get_action-{agent_id}] 中断战术={selected_tactic}, 返回导弹规避动作={missile_evasion}, 时间={current_time:.1f}s"
                )
            if agent_id.startswith('A') and hasattr(self, '_log_key_event'):
                phase_obj = self.agent_phases.get(agent_id, None)
                phase_str = phase_obj.value if phase_obj else None
                self._log_key_event(
                    env,
                    agent_id,
                    "导弹规避接管",
                    current_time=current_time,
                    当前阶段=phase_str,
                    当前战术=selected_tactic,
                    进入函数="get_action",
                    执行机动="MISSILE_EVADE",
                    当前状态=f"selected_tactic={selected_tactic}",
                    当前指令=str(missile_evasion),
                    退出条件="missile_evasion is None",
                    是否满足退出="否",
                )
                self._log_key_event(
                    env,
                    agent_id,
                    "最终动作",
                    current_time=current_time,
                    当前阶段=phase_str,
                    当前战术=selected_tactic,
                    进入函数="get_action",
                    执行机动="MISSILE_EVADE_FINAL",
                    当前状态=f"owner=missile_evasion selected_tactic={selected_tactic}",
                    当前指令=str(missile_evasion),
                    退出条件="caller consumes action",
                    是否满足退出="是",
                )
            return missile_evasion

        if (
            agent_id.startswith('A')
            and selected_tactic in (
                'DRAG_SHOOT',
                'PINCER_ATTACK',
                'HIGH_LOW_ATTACK',
                'FRONT_BACK',
                'SIDE_BY_SIDE',
                'TACTICAL_TURN',
                'INTELLIGENT_REENGAGEMENT',
                'UNIFIED_SECOND_ATTACK',
            )
            and alive_enemy_exists
        ):
            preserve_state = {"preserve": False}
            try:
                preserve_state = self._evaluate_red_defensive_posture(
                    env,
                    agent_id,
                    current_phase_name=current_phase_name or "template_dispatch",
                    reason=f"template_dispatch:{selected_tactic}",
                )
            except Exception:
                preserve_state = {"preserve": False}
            if preserve_state.get("preserve", False):
                preserve_reason = preserve_state.get(
                    "reason",
                    f"template_dispatch:{selected_tactic}",
                )
                self._deescalate_red_formation_to_defense(
                    env,
                    agent_id,
                    reason=preserve_reason,
                )
                if env.current_step % 80 == 0:
                    logging.info(
                        "🛡️ [TACTIC_PRESERVE_REDIRECT] %s tactic=%s phase=%s pair=%s reason=%s",
                        agent_id,
                        selected_tactic,
                        current_phase_name or "UNKNOWN",
                        self._get_formation_label(agent_id, env),
                        preserve_reason,
                    )
                return self._execute_defensive_preserve_action(
                    env,
                    agent_id,
                    reason=f"template_preserve_redirect:{selected_tactic}:{current_phase_name or 'unknown'}",
                )

        if selected_tactic == 'FORMATION_RESET':
            if (
                agent_id.startswith('A')
                and alive_enemy_exists
                and (
                    current_phase_name in ('DOR_DR', 'DR_MAR', 'BEYOND_MAR')
                    or (
                        hasattr(self, '_should_bypass_template_tactic')
                        and self._should_bypass_template_tactic(env, agent_id, 'FORMATION_RESET')
                    )
                )
            ):
                preserve_defense = False
                try:
                    preserve_defense = bool(
                        self._evaluate_red_defensive_posture(
                            env,
                            agent_id,
                            current_phase_name=current_phase_name,
                            reason="formation_reset_block",
                        ).get("preserve", False)
                    )
                except Exception:
                    preserve_defense = False
                if preserve_defense:
                    self._deescalate_red_formation_to_defense(env, agent_id, reason="formation_reset_block")
                    if env.current_step % 120 == 0:
                        logging.info(
                            f"🛡️ [FORMATION_RESET守区保留] {agent_id} phase={current_phase_name or 'UNKNOWN'} 保持守区止追"
                        )
                    return self._execute_defensive_preserve_action(
                        env,
                        agent_id,
                        reason=f"formation_reset_preserve:{current_phase_name or 'unknown'}",
                    )
                if env.current_step % 120 == 0:
                    logging.warning(
                        f"🛑 [FORMATION_RESET拦截] {agent_id} phase={current_phase_name or 'UNKNOWN'} 敌机仍存活，交由统一仲裁处理"
                    )
                return self._execute_red_reengage_override_action(
                    env,
                    agent_id,
                    current_time,
                    current_phase_name=current_phase_name or "FORMATION_RESET",
                    reason=f"formation_reset_block:{current_phase_name or 'unknown'}",
                    allow_template_reset=False,
                )
                override_plan = self._build_red_reengage_override_plan(
                    env,
                    agent_id,
                    current_phase_name=current_phase_name or "FORMATION_RESET",
                    reason="formation_reset_block",
                    allow_template_reset=False,
                )
                formation_agents = override_plan.get("formation_agents", [])
                replacement_tactic = override_plan.get("replacement_tactic", 'ADAPTIVE_ATTACK')
                for aid in formation_agents:
                    self._set_agent_tactic(aid, replacement_tactic)
                    if hasattr(self, 'formation_reset_agents'):
                        self.formation_reset_agents.discard(aid)
                formation_manager = getattr(self, '_get_formation_reset_manager', lambda _aid: None)(agent_id)
                if formation_manager is not None:
                    try:
                        formation_manager.active = False
                        formation_manager.phase = 'inactive'
                        formation_manager.active_agents = []
                        formation_manager.zero_heading_hold_start = None
                    except Exception:
                        pass
                if env.current_step % 120 == 0:
                    logging.warning(
                        f"🛑 [FORMATION_RESET拦截] {agent_id} phase={current_phase_name or 'UNKNOWN'} 敌机仍存活，改为{replacement_tactic}"
                    )
                if replacement_tactic == 'DEFENSIVE_GUARD':
                    return self._execute_defensive_preserve_action(
                        env,
                        agent_id,
                        reason=f"formation_reset_block:{current_phase_name or 'unknown'}",
                    )
                return self._execute_adaptive_attack(
                    env,
                    agent_id,
                    current_time,
                    reason=f"formation_reset_block:{current_phase_name or 'unknown'}",
                )
            if agent_id.startswith('A'):
                try:
                    current_phase = self.agent_phases.get(agent_id, None)
                    phase_str = current_phase.value if current_phase else 'None'
                    if not hasattr(self, '_last_formation_reset_phase'):
                        self._last_formation_reset_phase = {}
                    if agent_id not in self._last_formation_reset_phase or self._last_formation_reset_phase[agent_id] != phase_str:
                        from utils.trace_logger import trace_event
                        trace_event(
                            事件="调用_execute_formation_reset_procedure",
                            env=env,
                            模块="tactical_task",
                            类型="CALL",
                            状态="ENTER",
                            我机=agent_id,
                            战术="FORMATION_RESET",
                            阶段=phase_str,
                        )
                        self._last_formation_reset_phase[agent_id] = phase_str
                except Exception:
                    pass

            return self._execute_formation_reset_procedure(env, agent_id)

        if False and selected_tactic == 'TACTICAL_EVASION':
            if agent_id.startswith('A') and env.current_step % 60 == 0:
                logging.warning(f"🔍 [函数调用-{agent_id}] 调用execute_tactical_evasion, 战术=TACTICAL_EVASION, 时间={current_time:.1f}s")

            if agent_id.startswith('A'):
                try:
                    if not hasattr(self, '_last_tactical_evasion_tactic'):
                        self._last_tactical_evasion_tactic = {}
                    if agent_id not in self._last_tactical_evasion_tactic or self._last_tactical_evasion_tactic[agent_id] != self.selected_tactic:
                        from utils.trace_logger import trace_event
                        trace_event(
                            事件="调用execute_tactical_evasion",
                            env=env,
                            模块="tactical_task",
                            类型="CALL",
                            状态="ENTER",
                            我机=agent_id,
                            战术="TACTICAL_EVASION",
                        )
                        self._last_tactical_evasion_tactic[agent_id] = selected_tactic
                except Exception:
                    pass

            evasion_action = self.executor.execute_tactical_evasion(env, agent_id)
            if agent_id.startswith('A') and env.current_step % 60 == 0:
                logging.warning(f"📋 [get_action-{agent_id}] TACTICAL_EVASION返回动作={evasion_action}, 时间={current_time:.1f}s")
            if agent_id.startswith('A') and hasattr(self, '_log_key_event'):
                phase_obj = self.agent_phases.get(agent_id, None)
                phase_str = phase_obj.value if phase_obj else None
                self._log_key_event(
                    env,
                    agent_id,
                    "战术规避返回",
                    current_time=current_time,
                    当前阶段=phase_str,
                    当前战术="TACTICAL_EVASION",
                    进入函数="get_action",
                    执行机动="TACTICAL_EVASION",
                    当前状态="executor.execute_tactical_evasion returned",
                    当前指令=str(evasion_action),
                    退出条件="ttl超时或威胁消失",
                    是否满足退出="否",
                )
            if hasattr(self.executor, 'evasion_states') and agent_id in self.executor.evasion_states:
                state = self.executor.evasion_states[agent_id]
                if current_time - state.get('start', 0) > state.get('ttl', 10.0):
                    if hasattr(self, '_defense_prev_tactic') and self._defense_prev_tactic:
                        self._set_agent_tactic(agent_id, self._defense_prev_tactic)
                        self._defense_prev_tactic = None
                        logging.info(f"🛡️ [{agent_id}] 防御完成，恢复原战术: {self._get_agent_tactic(agent_id)}")
                    state_manager = getattr(self, 'state_manager', None)
                    beam_state = getattr(state_manager, 'beam_maneuver_state', None)
                    if isinstance(beam_state, dict):
                        beam_state.pop(agent_id, None)
                    elif hasattr(self, 'beam_maneuver_state') and agent_id in self.beam_maneuver_state:
                        self.beam_maneuver_state.pop(agent_id, None)
                    self.executor.evasion_states.pop(agent_id, None)
            return evasion_action

        if selected_tactic == 'DEFENSIVE_GUARD':
            preserve_reason = "defensive_guard"
            release_for_engage = False
            if agent_id.startswith('A'):
                try:
                    preserve_state = self._evaluate_red_defensive_posture(
                        env,
                        agent_id,
                        current_phase_name=current_phase_name or "DEFENSIVE_GUARD",
                        reason="defensive_guard",
                    )
                    preserve_reason = str(preserve_state.get("reason") or preserve_reason)
                    release_for_engage = bool(preserve_state.get("release_for_engage", False))
                except Exception:
                    preserve_reason = "defensive_guard"
                    release_for_engage = False
                if release_for_engage:
                    return self._execute_adaptive_attack(
                        env,
                        agent_id,
                        current_time,
                        reason=f"guard_release:{preserve_reason}",
                    )
                plan = self._plan_red_defensive_guard_action(
                    env,
                    agent_id,
                    target=None,
                    distance=None,
                    reason=preserve_reason,
                    source="get_action",
                )
                final_command = tuple(plan.get("command") or (7, 8, 3))
                if hasattr(self, '_log_key_event'):
                    phase_obj = self.agent_phases.get(agent_id, None)
                    phase_str = phase_obj.value if phase_obj else None
                    self._log_key_event(
                        env,
                        agent_id,
                        "最终动作",
                        current_time=current_time,
                        当前阶段=phase_str,
                        当前战术="DEFENSIVE_GUARD",
                        进入函数="get_action",
                        执行机动=str(plan.get("maneuver") or "DEFENSIVE_DEFAULT_FINAL"),
                        当前状态=str(plan.get("state") or f"owner=defensive_guard reason={preserve_reason}"),
                        当前指令=str(final_command),
                        退出条件="caller consumes action",
                        是否满足退出="是",
                    )
                return final_command
                self._deescalate_red_formation_to_defense(
                    env,
                    agent_id,
                    reason=preserve_reason,
                )

            if agent_id.startswith('A'):
                aircraft = self._get_aircraft_from_any_collection(env, agent_id)
                if aircraft is not None and getattr(aircraft, 'is_alive', False):
                    nearest_enemy_id, nearest_enemy = self._get_nearest_alive_enemy_anywhere(env, agent_id)
                    nearest_distance = None
                    if nearest_enemy is not None and getattr(nearest_enemy, 'is_alive', False):
                        try:
                            nearest_distance = float(self._calculate_distance_between(aircraft, nearest_enemy))
                        except Exception:
                            nearest_distance = None
                    entry_floor_m = float(self._get_friendly_bvr_entry_floor_m(agent_id))
                    if nearest_distance is not None and nearest_distance <= entry_floor_m and nearest_enemy is not None:
                        standoff_command = self._build_bvr_standoff_command(
                            env,
                            agent_id,
                            nearest_enemy,
                            nearest_distance,
                            reason=f"defensive_guard_hard_bvr_floor:{nearest_distance/1000.0:.1f}km|{preserve_reason}",
                        )
                        if standoff_command is not None:
                            if hasattr(self, '_log_key_event'):
                                phase_obj = self.agent_phases.get(agent_id, None)
                                phase_str = phase_obj.value if phase_obj else None
                                self._log_key_event(
                                    env,
                                    agent_id,
                                    "BVR底线接管",
                                    current_time=current_time,
                                    当前阶段=phase_str,
                                    当前战术="DEFENSIVE_GUARD",
                                    进入函数="get_action",
                                    执行机动="BVR_STANDOFF",
                                    当前状态=f"owner=defensive_guard dist={nearest_distance/1000.0:.1f}km",
                                    目标点=str(nearest_enemy_id),
                                    当前指令=str(standoff_command),
                                    退出条件="distance > 35km / guidance_commit",
                                    是否满足退出="否",
                                )
                            if env.current_step % 40 == 0:
                                logging.warning(
                                    "[BVR底线约束] %s dist=%.1fkm reason=defensive_guard_hard_floor nearest=%s",
                                    agent_id,
                                    nearest_distance / 1000.0,
                                    nearest_enemy_id or "-",
                                )
                            return standoff_command

            single_ship_recovery = bool(
                agent_id.startswith('A')
                and self._get_formation_alive_count(env, agent_id) <= 1
                and any(
                    tag in str(preserve_reason or "")
                    for tag in (
                        "solo_winchester",
                        "winchester_guard",
                        "winchester_low_zone",
                        "solo_low_zone",
                        "enemy_regroup_phase_hold",
                        "enemy_regroup_north",
                        "enemy_regroup_no_chase",
                        "latched_defense",
                    )
                )
            )
            multi_ship_recovery = bool(
                agent_id.startswith('A')
                and self._get_formation_alive_count(env, agent_id) >= 2
                and any(
                    tag in str(preserve_reason or "")
                    for tag in (
                        "winchester_guard",
                        "pair_winchester",
                        "pair_low_inventory",
                        "winchester_relief_handoff",
                        "enemy_regroup_phase_hold",
                        "enemy_regroup_north",
                        "enemy_regroup_no_chase",
                        "latched_defense",
                        "low_zone_hold_line",
                    )
                )
            )
            if multi_ship_recovery:
                corridor_command = self._build_multi_ship_defensive_hold_command(
                    env,
                    agent_id,
                    target=None,
                    distance=None,
                    reason=preserve_reason,
                )
                if corridor_command is not None:
                    if agent_id.startswith('A') and hasattr(self, '_log_key_event'):
                        phase_obj = self.agent_phases.get(agent_id, None)
                        phase_str = phase_obj.value if phase_obj else None
                        self._log_key_event(
                            env,
                            agent_id,
                            "最终动作",
                            current_time=current_time,
                            当前阶段=phase_str,
                            当前战术="DEFENSIVE_GUARD",
                            进入函数="get_action",
                            执行机动="DEFENSIVE_CORRIDOR_HOLD_FINAL",
                            当前状态=f"owner=defensive_guard reason={preserve_reason}",
                            当前指令=str(corridor_command),
                            退出条件="caller consumes action",
                            是否满足退出="是",
                        )
                    return corridor_command
            if single_ship_recovery:
                hold_command = self._build_solo_defensive_hold_command(
                    env,
                    agent_id,
                    target=None,
                    distance=None,
                    reason=preserve_reason,
                )
                if hold_command is not None:
                    if agent_id.startswith('A') and hasattr(self, '_log_key_event'):
                        phase_obj = self.agent_phases.get(agent_id, None)
                        phase_str = phase_obj.value if phase_obj else None
                        self._log_key_event(
                            env,
                            agent_id,
                            "最终动作",
                            current_time=current_time,
                            当前阶段=phase_str,
                            当前战术="DEFENSIVE_GUARD",
                            进入函数="get_action",
                            执行机动="DEFENSIVE_HOLD_FINAL",
                            当前状态=f"owner=defensive_guard reason={preserve_reason}",
                            当前指令=str(hold_command),
                            退出条件="caller consumes action",
                            是否满足退出="是",
                        )
                    return hold_command

            patrol_command = self._build_red_defensive_patrol_command(
                env,
                agent_id,
                target=None,
                distance=None,
                reason=preserve_reason,
            )
            if patrol_command is not None:
                if agent_id.startswith('A') and hasattr(self, '_log_key_event'):
                    phase_obj = self.agent_phases.get(agent_id, None)
                    phase_str = phase_obj.value if phase_obj else None
                    self._log_key_event(
                        env,
                        agent_id,
                        "最终动作",
                        current_time=current_time,
                        当前阶段=phase_str,
                        当前战术="DEFENSIVE_GUARD",
                        进入函数="get_action",
                        执行机动="DEFENSIVE_PATROL_FINAL",
                        当前状态=f"owner=defensive_guard reason={preserve_reason}",
                        当前指令=str(patrol_command),
                        退出条件="caller consumes action",
                        是否满足退出="是",
                    )
                return patrol_command

            hold_command = self._build_solo_defensive_hold_command(
                env,
                agent_id,
                target=None,
                distance=None,
                reason=preserve_reason,
            )
            if hold_command is not None:
                if agent_id.startswith('A') and hasattr(self, '_log_key_event'):
                    phase_obj = self.agent_phases.get(agent_id, None)
                    phase_str = phase_obj.value if phase_obj else None
                    self._log_key_event(
                        env,
                        agent_id,
                        "最终动作",
                        current_time=current_time,
                        当前阶段=phase_str,
                        当前战术="DEFENSIVE_GUARD",
                        进入函数="get_action",
                        执行机动="DEFENSIVE_HOLD_FINAL",
                        当前状态=f"owner=defensive_guard reason={preserve_reason}",
                        当前指令=str(hold_command),
                        退出条件="caller consumes action",
                        是否满足退出="是",
                    )
                return hold_command

            if agent_id.startswith('A'):
                return self.executor.execute_tactical_turn(env, agent_id)
            return 7, 8, 3

        if self._check_intelligent_reengagement_active(agent_id):
            return self._execute_intelligent_reengagement(env, agent_id, current_time)

        current_phase = self.agent_phases.get(agent_id, None)
        if current_phase == TacticalPhase.BEYOND_NLT:
            if agent_id.startswith('A') and selected_tactic == 'ADAPTIVE_ATTACK':
                preserve_defense = False
                try:
                    preserve_defense = bool(
                        self._evaluate_red_defensive_posture(
                            env,
                            agent_id,
                            current_phase_name="BEYOND_NLT",
                            reason="BEYOND_NLT",
                        ).get("preserve", False)
                    )
                except Exception:
                    preserve_defense = False
                if preserve_defense:
                    return self._execute_defensive_preserve_action(
                        env,
                        agent_id,
                        reason="beyond_nlt_preserve",
                    )
            if env.current_step % 60 == 0:
                logging.info(f"[BEYOND_NLT] [{agent_id}] 远距抵近飞行，等待进入 NLT 节点展开 {selected_tactic}")
            return self.executor.execute_pre_nlt_formation_shaping(env, agent_id, selected_tactic)

        if selected_tactic == 'DRAG_SHOOT':
            if env.current_step % 60 == 0 and agent_id == 'A0100':
                logging.info(f"[战术执行] [{agent_id}] 执行 DRAG_SHOOT")
            return self.executor.execute_drag_shoot(env, agent_id)
        elif selected_tactic == 'PINCER_ATTACK':
            if env.current_step % 25 == 0 and agent_id == 'A0100':
                current_phase = self.state_manager.get_agent_phase(agent_id)
                logging.warning(
                    f"[钳形攻势调试] {agent_id} 当前阶段={current_phase.value if current_phase else 'None'}, "
                    "战术=PINCER_ATTACK"
                )
            return self.executor.execute_pincer_attack(env, agent_id)
        elif selected_tactic == 'HIGH_LOW_ATTACK':
            return self.executor.execute_high_low_attack(env, agent_id)
        elif selected_tactic == 'FRONT_BACK':
            if agent_id.startswith('A'):
                aircraft = self._get_aircraft_from_any_collection(env, agent_id)
                if aircraft is not None and getattr(aircraft, 'is_alive', False):
                    nearest_enemy_id, nearest_enemy = self._get_nearest_alive_enemy_anywhere(env, agent_id)
                    nearest_distance = None
                    if nearest_enemy is not None and getattr(nearest_enemy, 'is_alive', False):
                        try:
                            nearest_distance = float(self._calculate_distance_between(aircraft, nearest_enemy))
                        except Exception:
                            nearest_distance = None
                    entry_floor_m = float(self._get_friendly_bvr_entry_floor_m(agent_id))
                    if nearest_distance is not None and nearest_distance <= entry_floor_m and nearest_enemy is not None:
                        standoff_command = self._build_bvr_hard_escape_command(
                            env,
                            agent_id,
                            nearest_enemy,
                            nearest_distance,
                            reason=f"front_back_entry_bvr_floor:{nearest_distance/1000.0:.1f}km",
                        )
                        if standoff_command is not None:
                            if hasattr(self, '_log_key_event'):
                                phase_obj = self.agent_phases.get(agent_id, None)
                                phase_str = phase_obj.value if phase_obj else None
                                self._log_key_event(
                                    env,
                                    agent_id,
                                    "BVR底线接管",
                                    current_time=current_time,
                                    当前阶段=phase_str,
                                    当前战术="FRONT_BACK",
                                    进入函数="get_action",
                                    执行机动="BVR_STANDOFF",
                                    当前状态=f"owner=front_back dist={nearest_distance/1000.0:.1f}km",
                                    目标点=str(nearest_enemy_id),
                                    当前指令=str(standoff_command),
                                    退出条件="distance > 35km / guidance_commit",
                                    是否满足退出="否",
                                )
                            return standoff_command
            return self.executor.execute_front_back(env, agent_id)
        elif selected_tactic == 'SIDE_BY_SIDE':
            if agent_id.startswith('A'):
                aircraft = self._get_aircraft_from_any_collection(env, agent_id)
                if aircraft is not None and getattr(aircraft, 'is_alive', False):
                    nearest_enemy_id, nearest_enemy = self._get_nearest_alive_enemy_anywhere(env, agent_id)
                    nearest_distance = None
                    if nearest_enemy is not None and getattr(nearest_enemy, 'is_alive', False):
                        try:
                            nearest_distance = float(self._calculate_distance_between(aircraft, nearest_enemy))
                        except Exception:
                            nearest_distance = None
                    entry_floor_m = float(self._get_friendly_bvr_entry_floor_m(agent_id))
                    if nearest_distance is not None and nearest_distance <= entry_floor_m and nearest_enemy is not None:
                        standoff_command = self._build_bvr_hard_escape_command(
                            env,
                            agent_id,
                            nearest_enemy,
                            nearest_distance,
                            reason=f"side_by_side_entry_bvr_floor:{nearest_distance/1000.0:.1f}km",
                        )
                        if standoff_command is not None:
                            if hasattr(self, '_log_key_event'):
                                phase_obj = self.agent_phases.get(agent_id, None)
                                phase_str = phase_obj.value if phase_obj else None
                                self._log_key_event(
                                    env,
                                    agent_id,
                                    "BVR底线接管",
                                    current_time=current_time,
                                    当前阶段=phase_str,
                                    当前战术="SIDE_BY_SIDE",
                                    进入函数="get_action",
                                    执行机动="BVR_STANDOFF",
                                    当前状态=f"owner=side_by_side dist={nearest_distance/1000.0:.1f}km",
                                    目标点=str(nearest_enemy_id),
                                    当前指令=str(standoff_command),
                                    退出条件="distance > 35km / guidance_commit",
                                    是否满足退出="否",
                                )
                            return standoff_command
            return self.executor.execute_side_by_side(env, agent_id)
        elif selected_tactic == 'TACTICAL_EVASION':
            return self._execute_defensive_preserve_action(
                env,
                agent_id,
                reason="tactical_evasion_fallback",
            )
        elif selected_tactic == 'TACTICAL_TURN':
            return self.executor.execute_tactical_turn(env, agent_id)
        elif selected_tactic == 'INTELLIGENT_REENGAGEMENT':
            return self._execute_intelligent_reengagement(env, agent_id, current_time)
        elif selected_tactic == 'UNIFIED_SECOND_ATTACK':
            return self._execute_unified_second_attack(env, agent_id)
        elif selected_tactic == 'ADAPTIVE_ATTACK':
            return self._execute_adaptive_attack(env, agent_id, current_time, reason="selected")
        else:
            action = 7, 8, 3
            return action
