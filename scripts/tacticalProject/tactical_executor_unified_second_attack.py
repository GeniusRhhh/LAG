"""Unified second-attack routing helpers for TacticalExecutor."""

from __future__ import annotations

import logging

from envs.JSBSim.core.catalog import Catalog as c
from core.target_assignment import get_target_with_fallback
from tactical_types import TacticalPhase
from tactical_utils import TacticalUtils


def execute_unified_second_attack(self, env, agent_id: str) -> tuple:
    current_time = env.current_step * env.time_interval

    if agent_id not in self.second_attack_coordination:
        existing_tactic = None
        if hasattr(self.task, '_selected_second_attack_tactic'):
            for aid, tactic in self.task._selected_second_attack_tactic.items():
                if aid.startswith('A'):
                    existing_tactic = tactic
                    break
        if existing_tactic is None:
            for aid, coord in self.second_attack_coordination.items():
                if aid.startswith('A') and coord.get('tactic'):
                    existing_tactic = coord['tactic']
                    break
        if existing_tactic is None:
            existing_tactic = self.task._select_second_attack_tactic() if hasattr(self.task, '_select_second_attack_tactic') else 'DRAG_SHOOT'

        our_agents = self.task._get_alive_formation_agents(
            env,
            agent_id,
            preserve_nominal_pair=True,
        ) if hasattr(self.task, '_get_alive_formation_agents') else [
            aid for aid in self._get_formation_agents(agent_id)
            if aid in env.agents and env._jsbsims[aid].is_alive
        ]
        for aid in our_agents:
            if aid not in self.second_attack_coordination:
                self.second_attack_coordination[aid] = {
                    'tactic': existing_tactic,
                    'ready': True,
                    'tactical_info': {'mtr_prime_completed': True},
                }
        logging.info(f"🔄 [二次进攻-{self._get_formation_label(agent_id)}] 统一战术选择: {existing_tactic}，应用于 {our_agents}")

    coordination = self.second_attack_coordination[agent_id]
    tactical_info = coordination.get('tactical_info', {})
    selected_tactic = coordination.get('tactic', 'DRAG_SHOOT')
    our_agents = self.task._get_alive_formation_agents(
        env,
        agent_id,
        preserve_nominal_pair=True,
    ) if hasattr(self.task, '_get_alive_formation_agents') else [
        aid for aid in self._get_formation_agents(agent_id)
        if aid in env.agents and env._jsbsims[aid].is_alive
    ]

    if len(our_agents) < 2 or selected_tactic == 'ADAPTIVE_ATTACK':
        return self.task._execute_adaptive_attack(env, agent_id, current_time, reason="single_ship_second_attack")

    if not tactical_info.get('mtr_prime_completed', False):
        mtr_command = self.mtr_prime_node.execute_mtr_prime(env, agent_id, tactical_info)
        coordination['tactical_info'] = tactical_info
        if tactical_info.get('mtr_prime_completed', False):
            selected_tactic = tactical_info.get('second_attack_tactic', 'DRAG_SHOOT')
            logging.info(f"✅ [{agent_id}] MTR'节点完成，开始执行{selected_tactic}战术")
        return mtr_command

    target_id = get_target_with_fallback(agent_id, env)
    if (not target_id) or (target_id not in env._jsbsims) or (not env._jsbsims[target_id].is_alive):
        if agent_id.startswith('A') and any(str(other_id).startswith('B') and getattr(other_aircraft, 'is_alive', False) for other_id, other_aircraft in env.agents.items()):
            return self.task._execute_adaptive_attack(env, agent_id, current_time, reason="second_attack_no_target")
        if env.current_step % 300 == 0:
            logging.info(f"🔔 [{agent_id}] 二次进攻无有效目标，执行返航(180°)")
        return self._exit_second_attack_to_rtb(env, agent_id, "二次进攻无有效目标")

    is_lead = self._is_formation_lead(env, agent_id)
    teammate_id = self._get_teammate_id(agent_id)
    if teammate_id in env.agents and env.agents[teammate_id].is_alive:
        my_aircraft = env.agents[agent_id]
        teammate_aircraft = env.agents[teammate_id]
        formation_distance = TacticalUtils.calculate_horizontal_distance(my_aircraft, teammate_aircraft)
        target_distance = float("inf")
        try:
            target_aircraft = env.agents.get(target_id)
            if target_aircraft is not None and getattr(target_aircraft, "is_alive", False):
                target_distance = TacticalUtils.calculate_distance_between(my_aircraft, target_aircraft)
        except Exception:
            target_distance = float("inf")
        force_direct = False
        if hasattr(self.task, "_should_force_direct_second_attack"):
            try:
                force_direct = bool(self.task._should_force_direct_second_attack(env, agent_id))
            except Exception:
                force_direct = False
        close_commit_window = bool(
            str(agent_id).startswith('A')
            and formation_distance <= 18000.0
            and target_distance <= 65000.0
        )
        if target_distance >= 58000.0 and not force_direct and (not close_commit_window) and hasattr(self.task, '_build_dynamic_regroup_command'):
            try:
                regroup_command = self.task._build_dynamic_regroup_command(env, agent_id)
            except Exception:
                regroup_command = None
            if regroup_command is not None:
                return regroup_command
        if close_commit_window and env.current_step % 80 == 0:
            logging.info(
                "🧭 [SECOND_ATTACK_ARBITRATION] %s keep_attack pair_spacing=%.1fkm target_dist=%.1fkm",
                agent_id,
                formation_distance / 1000.0,
                target_distance / 1000.0,
            )
        if formation_distance > 18000:
            current_phase_name = ""
            try:
                current_phase = self.task.state_manager.get_agent_phase(agent_id)
                current_phase_name = getattr(current_phase, "value", "")
            except Exception:
                current_phase_name = ""
            if force_direct:
                if env.current_step % 80 == 0:
                    logging.warning(
                        f"🛑 [SECOND_ATTACK回整降级] {agent_id} pair_spacing={formation_distance/1000:.1f}km "
                        f"target_dist={target_distance/1000:.1f}km phase={current_phase_name or 'UNKNOWN'} -> ADAPTIVE_ATTACK"
                    )
                return self.task._execute_adaptive_attack(
                    env,
                    agent_id,
                    current_time,
                    reason=f"second_attack_regroup_bypass:{current_phase_name or 'unknown'}",
                )
            if not hasattr(self, '_formation_converge_state'):
                self._formation_converge_state = {}
            if agent_id not in self._formation_converge_state:
                self._formation_converge_state[agent_id] = {'start_time': current_time, 'converged': False}
            converge_state = self._formation_converge_state[agent_id]
            converge_duration = current_time - converge_state['start_time']
            if converge_duration < 10.0 and not converge_state['converged']:
                current_heading = my_aircraft.get_property_value(c.attitude_psi_deg)
                converge_heading = (current_heading + 15) % 360 if is_lead else (current_heading - 15) % 360
                if env.current_step % 100 == 0:
                    logging.info(f"🔧 [{agent_id}] 编队靠拢: 间距{formation_distance/1000:.1f}km, 目标航向{converge_heading:.1f}°")
                return self.task._maintain_heading_precise(env, agent_id, converge_heading)
            if converge_duration >= 10.0:
                converge_state['converged'] = True

    if selected_tactic == 'DRAG_SHOOT':
        return self._execute_unified_drag_shoot(env, agent_id, is_lead)
    if selected_tactic == 'PINCER_ATTACK':
        return self._execute_unified_pincer(env, agent_id, is_lead)
    if selected_tactic == 'HIGH_LOW_ATTACK':
        return self._execute_unified_high_low(env, agent_id, is_lead)
    if selected_tactic == 'SIDE_BY_SIDE':
        return self._execute_unified_side_by_side(env, agent_id, is_lead)
    if selected_tactic == 'FRONT_BACK':
        return self._execute_unified_front_back(env, agent_id, is_lead)
    return self._execute_unified_drag_shoot(env, agent_id, is_lead)


def _execute_unified_drag_shoot(self, env, agent_id: str, is_lead: bool) -> tuple:
    current_time = env.current_step * env.time_interval
    return self._execute_second_attack_drag_shoot(env, agent_id, is_lead, current_time)


def _execute_unified_pincer(self, env, agent_id: str, is_lead: bool) -> tuple:
    current_time = env.current_step * env.time_interval
    return self._execute_second_attack_pincer(env, agent_id, is_lead, current_time)


def _execute_unified_high_low(self, env, agent_id: str, is_lead: bool) -> tuple:
    current_time = env.current_step * env.time_interval
    return self._execute_second_attack_high_low(env, agent_id, is_lead, current_time)


def _execute_unified_side_by_side(self, env, agent_id: str, is_lead: bool) -> tuple:
    current_time = env.current_step * env.time_interval
    return self._execute_second_attack_side_by_side(env, agent_id, is_lead, current_time)


def _execute_unified_front_back(self, env, agent_id: str, is_lead: bool) -> tuple:
    current_time = env.current_step * env.time_interval
    return self._execute_second_attack_front_back(env, agent_id, is_lead, current_time)
