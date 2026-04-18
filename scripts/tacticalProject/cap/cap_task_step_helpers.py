"""CAPTask lifecycle helpers for the main step loop."""

import logging
import os
from math import inf

import numpy as np

try:
    from .cap_state_machine import CAPState
    from .control_ranges import DEFAULT_RANGES
    from .patrol_state_machine import PatrolState
    from .formation_manager import PatrolPhase
except ImportError:
    from cap.cap_state_machine import CAPState
    from cap.control_ranges import DEFAULT_RANGES
    from cap.patrol_state_machine import PatrolState
    from cap.formation_manager import PatrolPhase

log = logging.getLogger(__name__)
_CAP_DEBUG_PRINT = os.environ.get('CAP_DEBUG_PRINT') == '1'


def run_cap_step(self, env):
    self.step_count += 1
    current_time = self.step_count * 0.2
    alive_enemy_exists = any(
        str(aid).startswith('B') and getattr(aircraft, 'is_alive', False)
        for aid, aircraft in getattr(env, 'agents', {}).items()
    )

    if not alive_enemy_exists:
        self._awacs_updated_this_step = False
        if hasattr(self, "_clear_enemy_contact_state"):
            try:
                self._clear_enemy_contact_state(env, clear_picture=True, reason="run_step_no_enemy_early")
            except Exception:
                pass

        previous_state = self.cap_state_machine.state
        self.cap_state_machine._prev_state = previous_state
        self.cap_state_machine.state = CAPState.PATROL
        self.cap_state_machine._state_enter_time = current_time
        self.cap_state_machine._state_hold_until = current_time
        cap_state = CAPState.PATROL
        changed = previous_state != CAPState.PATROL
        if changed:
            log.info(f"[CAP] [CAP状态] {previous_state.value} -> {cap_state.value} (no_enemy)")

        if hasattr(self, "_set_agent_tactic"):
            for aid, aircraft in getattr(env, "agents", {}).items():
                if str(aid).startswith("A") and getattr(aircraft, "is_alive", False):
                    try:
                        self._set_agent_tactic(aid, "DEFENSIVE_GUARD")
                    except Exception:
                        pass

        ctx = self._calculate_cap_context(env, current_time)
        ctx.min_threat_distance = inf
        ctx.has_awacs_info = False
        ctx.has_hostile_in_high_zone = False
        ctx.has_hostile_in_medium_zone = False
        ctx.hostile_alive_count = 0

        self._update_missiles(env, current_time)
        self._update_mission_evaluation(env, current_time)
        if hasattr(self, "_tactic_assignments_by_agent"):
            self._tactic_assignments_by_agent = {}

        if not hasattr(self, '_first_radar_detection_time'):
            self._first_radar_detection_time = {}
        if not hasattr(self, '_radar_range_entry_times'):
            self._radar_range_entry_times = {}
        if not hasattr(self, '_locked_compression'):
            self._locked_compression = None

        if self.step_count % 300 == 1 or changed:
            self._log_situation(env, ctx, cap_state, current_time)

        obs = {aid: self.get_obs(env, aid) for aid in env.agents}
        all_obs = np.stack([obs[aid] for aid in sorted(env.agents)], axis=0)
        share = {aid: all_obs.flatten() for aid in env.agents}
        rewards = {aid: np.array([0.1 if env.agents[aid].is_alive else 0.0]) for aid in env.agents}
        dones = {aid: [not env.agents[aid].is_alive] for aid in env.agents}

        infos = {aid: {
            "step": self.step_count,
            "cap_state": cap_state.value,
            "detection_mode": self.coop_detection.mode.value,
        } for aid in env.agents}

        return obs, share, rewards, dones, infos

    in_awacs_phase = self.cap_state_machine.state in (CAPState.PATROL, CAPState.INTERCEPT)

    if in_awacs_phase:
        self._update_awacs_data(env, current_time)
    else:
        self._awacs_updated_this_step = False

    self._update_picture(env, current_time)

    ctx = self._calculate_cap_context(env, current_time)
    cap_state, changed = self.cap_state_machine.update(ctx, current_time)
    if changed:
        log.info(f"[CAP] [CAP状态] {self.cap_state_machine._prev_state.value} -> {cap_state.value}")

    if not hasattr(self, '_state_log_interval_steps'):
        self._state_log_interval_steps = max(1, int(float(os.getenv('CAP_STATE_LOG_INTERVAL_S', '20')) / 0.2))
    if self.step_count % self._state_log_interval_steps == 0:
        log.info(
            f"[状态机] 当前状态={cap_state.value} | "
            f"最近威胁距离={ctx.min_threat_distance:.1f}km | "
            f"预警信息={'有' if ctx.has_awacs_info else '无'} | "
            f"时间={current_time:.1f}s"
        )

    if alive_enemy_exists and cap_state in (CAPState.PATROL, CAPState.INTERCEPT):
        self._update_cooperative_detection(env, current_time)
    elif alive_enemy_exists:
        self._update_radar_tracking_only(env, current_time)
    # 用本步最新的协同探测/雷达结果刷新一次 picture，避免战术层只能看到上一步航迹
    self._update_picture(env, current_time)

    if alive_enemy_exists:
        self._update_engagement(env, current_time)

        if cap_state == CAPState.INTERCEPT and ctx.min_threat_distance <= DEFAULT_RANGES.NLT:
            self._update_intent_recognition(env, current_time)

        if cap_state == CAPState.ENGAGE and DEFAULT_RANGES.LR < ctx.min_threat_distance <= DEFAULT_RANGES.MTR:
            self._update_cooperative_tracking(env, current_time)

    self._update_missiles(env, current_time)

    self._update_mission_evaluation(env, current_time)
    if alive_enemy_exists:
        self._update_tactic_selection(env, current_time)
    elif hasattr(self, "_tactic_assignments_by_agent"):
        self._tactic_assignments_by_agent = {}

    if not hasattr(self, '_first_radar_detection_time'):
        self._first_radar_detection_time = {}
    if not hasattr(self, '_radar_range_entry_times'):
        self._radar_range_entry_times = {}
    if not hasattr(self, '_locked_compression'):
        self._locked_compression = None

    for aid in env.agents:
        if not aid.startswith('A') or not env.agents[aid].is_alive:
            continue

        if aid not in self._radar_range_entry_times:
            my_pos = self._get_battlefield_pos(env, aid)
            min_dist = float('inf')
            min_eid = None

            all_distances = {}
            for eid in env.agents:
                if eid.startswith('B') and env.agents[eid].is_alive:
                    enemy_pos = self._get_battlefield_pos(env, eid)
                    dist = np.sqrt((enemy_pos[0] - my_pos[0]) ** 2 + (enemy_pos[1] - my_pos[1]) ** 2)
                    all_distances[eid] = dist
                    if dist < min_dist:
                        min_dist = dist
                        min_eid = eid

            if min_dist <= 200.0:
                self._radar_range_entry_times[aid] = current_time
                current_hdg = self._get_heading(env, aid)
                if _CAP_DEBUG_PRINT:
                    log.info(
                        f"\n[雷达激活详细] {aid} @ {current_time:.1f}s | 位置:({my_pos[0]:.1f}, {my_pos[1]:.1f}) | 航向:{current_hdg:.1f}°"
                    )
                    log.info(f"  最近敌机: {min_eid} @ {min_dist:.1f}km")

                for eid in sorted(all_distances.keys()):
                    enemy_pos = self._get_battlefield_pos(env, eid)
                    dist = all_distances[eid]
                    dx = enemy_pos[0] - my_pos[0]
                    dy = enemy_pos[1] - my_pos[1]
                    abs_bearing = np.degrees(np.arctan2(dx, dy)) % 360
                    rel_bearing = (abs_bearing - current_hdg + 180) % 360 - 180
                    in_range = "✅" if dist <= 200.0 else "❌超距"
                    in_scan = "✅" if abs(rel_bearing) <= 5.0 else f"❌偏离{rel_bearing:+.1f}°"
                    if _CAP_DEBUG_PRINT:
                        log.info(
                            f"  {eid}: 距离{dist:>6.1f}km {in_range} | 方位{rel_bearing:>+6.1f}° {in_scan} | 位置({enemy_pos[0]:.1f}, {enemy_pos[1]:.1f})"
                        )

    for aid, tracks in self.cap_radar._radar_tracks.items():
        for tid in tracks:
            if tid not in self._first_radar_detection_time:
                self._first_radar_detection_time[tid] = {}
            if aid not in self._first_radar_detection_time[tid]:
                self._first_radar_detection_time[tid][aid] = current_time
                if _CAP_DEBUG_PRINT:
                    log.info(f"[首次探测] {aid} -> {tid} @ {current_time:.1f}s")

    if self.step_count % 300 == 1 or changed:
        self._log_situation(env, ctx, cap_state, current_time)

    positions = {}
    for aid in env.agents:
        if aid.startswith('A') and env.agents[aid].is_alive:
            positions[aid] = self._get_battlefield_pos(env, aid)

    if alive_enemy_exists and self.cap_state_machine.is_patrol and self.formation.check_cycle_switch(positions):
        self.formation.execute_cycle_switch()
        log.info(f"[CAP] 冷热交替 #{self.formation.cycle_count}")
        for aid, pm in self.patrol_machines.items():
            phase = self.formation.get_current_phase(aid)
            pm.state = PatrolState.HOT_NORTH if phase == PatrolPhase.HOT else PatrolState.COLD_SOUTH

    obs = {aid: self.get_obs(env, aid) for aid in env.agents}
    all_obs = np.stack([obs[aid] for aid in sorted(env.agents)], axis=0)
    share = {aid: all_obs.flatten() for aid in env.agents}
    rewards = {aid: np.array([0.1 if env.agents[aid].is_alive else 0.0]) for aid in env.agents}
    dones = {aid: [not env.agents[aid].is_alive] for aid in env.agents}

    infos = {aid: {
        "step": self.step_count,
        "cap_state": cap_state.value,
        "detection_mode": self.coop_detection.mode.value,
    } for aid in env.agents}

    return obs, share, rewards, dones, infos
