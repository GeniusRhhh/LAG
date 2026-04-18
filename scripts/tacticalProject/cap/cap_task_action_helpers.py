"""CAP action helpers extracted from CAPTask."""

from __future__ import annotations

import logging
import os
from typing import Tuple

import numpy as np

from envs.JSBSim.core.catalog import Catalog as c

from .cap_state_machine import CAPState

log = logging.getLogger(__name__)


def get_action(self, env, agent_id: str) -> Tuple[int, int, int]:
    ac = env.agents.get(agent_id)
    if not ac or not ac.is_alive:
        return 7, 8, 3
    if agent_id.startswith('B'):
        if os.environ.get('CAP_ENEMY_SIMPLE', '0').strip().lower() in ('1', 'true', 'yes', 'on'):
            return _get_enemy_action_simple(self, env, agent_id)
        return _get_enemy_action(self, env, agent_id)

    verbose_diag = os.environ.get('CAP_VERBOSE_DIAG', '').strip().lower() in ('1', 'true', 'yes', 'on')
    if verbose_diag and self.step_count % 60 == 0 and agent_id == 'A0100':
        cap_state = self.cap_state_machine.state
        threats = self._get_picture_threats(purpose="decision")
        log.info("=" * 80)
        log.info(f"[诊断1-CAP状态] step={self.step_count} agent={agent_id}")
        log.info(f"  当前状态: {cap_state.value}")
        log.info(f"  威胁数量: {len(threats)}")
        if threats:
            try:
                x, y = self._get_battlefield_pos(env, agent_id)
                nearest = self._get_picture_nearest_threat(np.array([x, y, 8.0]), purpose="decision")
                if nearest:
                    distance = np.sqrt((nearest.x - x) ** 2 + (nearest.y - y) ** 2)
                    log.info(f"  最近威胁: {nearest.track_id} 距离={distance:.1f}km")
                    log.info(f"  ENGAGE阈值: 200km (当前{'<' if distance < 200 else '>='}200km)")
            except Exception as e:
                log.warning(f"  计算威胁距离失败: {e}")
        log.info("=" * 80)

    cap_state = self.cap_state_machine.state
    if cap_state == CAPState.PATROL:
        legacy = getattr(self, '_get_patrol_action_legacy', None)
        if legacy is not None:
            return legacy(env, agent_id)
        return _get_patrol_action(self, env, agent_id)
    if cap_state == CAPState.INTERCEPT:
        legacy = getattr(self, '_get_intercept_action_legacy', None)
        if legacy is not None:
            return legacy(env, agent_id)
        return _get_intercept_action(self, env, agent_id)
    if cap_state == CAPState.ENGAGE:
        return self._get_engage_action(env, agent_id)
    if cap_state == CAPState.EVADE:
        return self._get_engage_action(env, agent_id)
    if cap_state == CAPState.RTB:
        return self._get_rtb_action(env, agent_id)
    legacy = getattr(self, '_get_patrol_action_legacy', None)
    if legacy is not None:
        return legacy(env, agent_id)
    return _get_patrol_action(self, env, agent_id)


def _get_enemy_action_simple(self, env, agent_id: str) -> Tuple[int, int, int]:
    ac = env.agents.get(agent_id)
    if not ac or not ac.is_alive:
        return 7, 8, 3
    current_hdg = self._get_heading(env, agent_id)
    target_hdg = 180.0
    if hasattr(self, 'enemy_target_headings') and agent_id in self.enemy_target_headings:
        target_hdg = self.enemy_target_headings[agent_id]
    alt_cmd = 7
    spd_cmd = 3
    is_verification_scenario = hasattr(self, 'enemy_scenario') and self.enemy_scenario is not None
    altitude_debug = os.environ.get('CAP_ALTITUDE_DEBUG', '').strip() in ('1', 'true', 'True', 'YES', 'yes')

    if altitude_debug and self.step_count % 50 == 0 and agent_id == 'B0100':
        log.info(
            f"[V6诊断 _get_enemy_action] step={self.step_count} "
            f"is_verification={is_verification_scenario} "
            f"has_altitudes={hasattr(self, 'enemy_target_altitudes_ft')} "
            f"altitudes_dict={getattr(self, 'enemy_target_altitudes_ft', {})}"
        )

    if is_verification_scenario and hasattr(self, 'enemy_target_altitudes_ft'):
        if agent_id in self.enemy_target_altitudes_ft:
            target_alt_ft = self.enemy_target_altitudes_ft[agent_id]
            current_alt_ft = ac.get_property_value(c.position_h_sl_ft)
            alt_diff_ft = target_alt_ft - current_alt_ft
            alt_diff_m = alt_diff_ft * 0.3048
            target_delta_alt_km = float(alt_diff_m) / 1000.0
            if abs(target_delta_alt_km) < 0.01:
                alt_cmd = 7
            else:
                target_delta_alt_km = float(np.clip(target_delta_alt_km, self.norm_alt.min(), self.norm_alt.max()))
                alt_cmd = int(np.argmin(np.abs(self.norm_alt - target_delta_alt_km)))
            if target_delta_alt_km > 0.05:
                try:
                    tas_mps = float(ac.get_property_value(c.velocities_vc_mps))
                except Exception:
                    tas_mps = None
                if tas_mps is None:
                    spd_cmd = max(spd_cmd, 5)
                elif tas_mps < 200.0:
                    spd_cmd = 6
                elif tas_mps < 230.0:
                    spd_cmd = 5
                else:
                    spd_cmd = max(spd_cmd, 4)

    if is_verification_scenario:
        try:
            tas_mps = float(ac.get_property_value(c.velocities_vc_mps))
        except Exception:
            tas_mps = None
        if tas_mps is not None and tas_mps < 210.0:
            spd_cmd = max(int(spd_cmd), 6)
        elif tas_mps is not None and tas_mps < 235.0:
            spd_cmd = max(int(spd_cmd), 5)
            if altitude_debug and self.step_count % 50 == 0:
                log.info(
                    f"[能量管理V6] {agent_id} step={self.step_count} "
                    f"current_alt={current_alt_ft:.0f}ft target_alt={target_alt_ft:.0f}ft "
                    f"diff={alt_diff_ft:.0f}ft ({alt_diff_m:.1f}m) "
                    f"delta_alt={target_delta_alt_km:.3f}km "
                    f"alt_cmd={alt_cmd} (norm_alt={self.norm_alt[alt_cmd]:.3f}km) "
                    f"spd_cmd={spd_cmd} (norm_vel={self.norm_vel[spd_cmd]:.3f})"
                )
    else:
        is_left_formation = agent_id in ['B0100', 'B0200']
        step = self.step_count
        if step < 100:
            pass
        elif 100 <= step < 300:
            alt_cmd = 10 if is_left_formation else 4
        elif 300 <= step < 500:
            alt_cmd = 4 if is_left_formation else 10
        elif 500 <= step < 600:
            target_hdg = 120 if is_left_formation else 240
        elif 600 <= step < 700:
            target_hdg = 180
        else:
            spd_cmd = 5 if is_left_formation else 1

    heading_diff = target_hdg - current_hdg
    if heading_diff > 180:
        heading_diff -= 360
    elif heading_diff < -180:
        heading_diff += 360
    if abs(heading_diff) < 5.0:
        hdg_cmd = 8
    else:
        hdg_cmd = int(np.argmin(np.abs(self.norm_hdg - np.radians(heading_diff))))
    return alt_cmd, hdg_cmd, spd_cmd


def _get_patrol_action(self, env, agent_id: str) -> Tuple[int, int, int]:
    legacy = getattr(self, '_get_patrol_action_legacy', None)
    if legacy is not None:
        return legacy(env, agent_id)
    ac = env.agents.get(agent_id)
    if ac is None or not ac.is_alive:
        return 7, 8, 3

    pos = ac.get_position()
    x_km = float(pos[1]) / 1000.0
    y_km = float(pos[0]) / 1000.0
    heading_deg = float(np.degrees(ac.get_property_values([c.attitude_heading_true_rad])[0])) % 360.0

    pm = self.patrol_machines.get(agent_id)
    if pm is None:
        return 7, 8, 3

    old_state = pm.state
    patrol_state, target_heading = pm.update(x_km, y_km)
    target_heading = float(target_heading) % 360.0
    if patrol_state != old_state:
        log.info("🔄 [%s] 状态: %s -> %s @ (%.1f, %.1f)", agent_id, old_state.value, patrol_state.value, x_km, y_km)

    if agent_id not in self.aircraft_states:
        self.aircraft_states[agent_id] = {
            'target_heading': target_heading,
            'force_left_turn': False,
            'patrol_state': str(patrol_state),
        }

    st = self.aircraft_states[agent_id]
    prev_target_heading = float(st.get('target_heading', target_heading)) % 360.0
    prev_patrol_state = st.get('patrol_state', 'UNKNOWN')
    st['patrol_state'] = str(patrol_state)
    st['target_heading'] = target_heading

    if target_heading != prev_target_heading or st.get('patrol_state') != prev_patrol_state:
        delta = abs(target_heading - prev_target_heading)
        delta = min(delta, 360.0 - delta)
        st['force_left_turn'] = (abs(delta - 180.0) < 1e-6)

    shortest_diff = target_heading - heading_deg
    if shortest_diff > 180:
        shortest_diff -= 360
    elif shortest_diff < -180:
        shortest_diff += 360

    heading_diff = shortest_diff
    if st.get('force_left_turn', False):
        if abs(shortest_diff) < 5.0:
            st['force_left_turn'] = False
        else:
            left_amount = (heading_deg - target_heading) % 360.0
            forced_left = -left_amount
            if forced_left < -180.0:
                forced_left = -180.0
            if forced_left > 0.0:
                forced_left = -abs(forced_left)
            heading_diff = forced_left

    if agent_id in ['A0200', 'A0400'] and abs(heading_diff) >= 175.0:
        hdg_cmd = 0
    else:
        hdg_cmd = int(np.argmin(np.abs(self.norm_hdg - np.radians(heading_diff))))
        if abs(shortest_diff) < 5.0:
            hdg_cmd = 8
    return 7, hdg_cmd, 3


def _get_intercept_action(self, env, agent_id: str) -> Tuple[int, int, int]:
    legacy = getattr(self, '_get_intercept_action_legacy', None)
    if legacy is not None:
        return legacy(env, agent_id)
    ac = env.agents.get(agent_id)
    if not ac or not ac.is_alive:
        return 7, 8, 3

    x, y = self._get_battlefield_pos(env, agent_id)
    current_hdg = self._get_heading(env, agent_id)
    current_time = self.step_count * 0.2

    if getattr(self, 'experiment_mode', 'proposed') == 'baseline':
        awacs_tracks = self.awacs.get_tracks() if hasattr(self, 'awacs') and self.awacs is not None else {}
        pts = []
        for t in awacs_tracks.values():
            pos = getattr(t, 'position', None)
            if pos is not None and len(pos) >= 2:
                pts.append((float(pos[0]), float(pos[1])))
        if not pts:
            for tr in self._get_picture_threats(current_time=current_time, purpose="search"):
                if hasattr(tr, 'x') and hasattr(tr, 'y'):
                    pts.append((float(tr.x), float(tr.y)))
        if not pts:
            return _get_patrol_action(self, env, agent_id)

        ex = float(np.mean([p[0] for p in pts]))
        ey = float(np.mean([p[1] for p in pts]))
        target_hdg = float(np.degrees(np.arctan2(ex - x, ey - y)) % 360)
        heading_diff = target_hdg - current_hdg
        if heading_diff > 180:
            heading_diff -= 360
        elif heading_diff < -180:
            heading_diff += 360
        if abs(heading_diff) < 5.0:
            hdg_cmd = 8
        else:
            hdg_cmd = int(np.argmin(np.abs(self.norm_hdg - np.radians(heading_diff))))
        return 7, hdg_cmd, 3

    if hasattr(self, "_get_alive_awacs_tracks"):
        awacs_tracks = self._get_alive_awacs_tracks(env)
    else:
        awacs_tracks = self.awacs.get_tracks() if hasattr(self, 'awacs') and self.awacs is not None else {}
    alive_enemy_ids = set()
    if hasattr(self, "_get_alive_enemy_ids"):
        try:
            alive_enemy_ids = set(self._get_alive_enemy_ids(env))
        except Exception:
            alive_enemy_ids = set()
    has_awacs_info = bool(awacs_tracks)
    if not has_awacs_info:
        grace_s = float(getattr(self, '_awacs_grace_seconds', 15.0))
        last_t = getattr(self, '_last_awacs_info_time', None)
        if alive_enemy_ids and last_t is not None and (current_time - float(last_t)) <= grace_s:
            has_awacs_info = True

    source_mode = getattr(self, 'intercept_target_source', 'auto')
    use_awacs_tracks = (source_mode in ('auto', 'awacs')) and has_awacs_info
    if source_mode == 'picture':
        use_awacs_tracks = False

    target_positions_raw = {}
    target_measure_times = {}
    if use_awacs_tracks:
        for tid, t in awacs_tracks.items():
            pos = getattr(t, 'position', None)
            if pos is None or len(pos) < 2:
                continue
            target_positions_raw[tid] = (float(pos[0]), float(pos[1]))
            ts = getattr(t, 'timestamp', None)
            target_measure_times[tid] = float(ts) if ts is not None else current_time
    else:
        radar_update_times = {}
        try:
            for _, tracks in self.cap_radar._radar_tracks.items():
                for tid, trk in tracks.items():
                    last_upd = float(getattr(trk, 'last_update', 0.0))
                    if last_upd <= 0:
                        continue
                    radar_update_times[tid] = max(radar_update_times.get(tid, 0.0), last_upd)
        except Exception:
            radar_update_times = {}

        for track in self._get_picture_threats(current_time=current_time, purpose="search"):
            if hasattr(track, 'x') and hasattr(track, 'y'):
                target_positions_raw[track.track_id] = (float(track.x), float(track.y))
                ts_pic = getattr(track, 'timestamp', None)
                if ts_pic is not None:
                    target_measure_times[track.track_id] = float(ts_pic)
                else:
                    target_measure_times[track.track_id] = radar_update_times.get(track.track_id, current_time)

    if not target_positions_raw:
        return _get_patrol_action(self, env, agent_id)

    target_lost = self.coop_detection.any_target_lost(current_time)
    target_positions_estimated = {}
    for tid, pos in target_positions_raw.items():
        meas_time = float(target_measure_times.get(tid, current_time))
        last_meas = self.coop_detection.get_last_track_time(tid)
        if last_meas is None or meas_time > last_meas + 1e-6:
            est_pos, _ = self.coop_detection.update_target_state_imm(target_id=tid, position=pos, current_time=meas_time)
        else:
            est_pos, _ = self.coop_detection.predict_target_state(tid, current_time)
        target_positions_estimated[tid] = est_pos

    sigma_man, r_targets = self.coop_detection.propagate_sigma_man(
        current_time=current_time,
        gamma=0.99,
        target_ids=list(target_positions_estimated.keys()),
    )

    sigma_source_km = 2.5 if use_awacs_tracks else 0.5
    sigma_enemy, R_target = self.coop_detection.compute_distribution_range(
        target_positions=target_positions_estimated,
        current_time=current_time,
        sigma_awacs=sigma_source_km,
        k_sigma=3.0,
    )
    try:
        raw_cap = os.environ.get('CAP_R_TARGET_MAX_KM', '').strip()
        if raw_cap:
            cap_km = float(raw_cap)
            if cap_km > 0:
                R_target = min(float(R_target), cap_km)
    except Exception:
        pass

    enemy_center, _ = self.coop_detection.estimate_enemy_center(target_positions_estimated)
    if self.cap_state_machine.state == CAPState.INTERCEPT and agent_id == 'A0100' and self.step_count % 150 == 0:
        log.info("=" * 80)
        log.info(f"[算法0] 协同探测7层架构 | 时间={current_time:.1f}s 步数={self.step_count}")
        log.info(f"[层1-信息] 预警信息={'有' if has_awacs_info else '无'} | 目标数={len(target_positions_raw)} | 目标丢失={'是' if target_lost else '否'}")
        log.info(f"[层2-估计] IMM-EKF已更新 {len(target_positions_estimated)} 个目标状态")
        for tid, pos in list(target_positions_estimated.items())[:3]:
            r = r_targets.get(tid, 0)
            log.info(f"  目标{tid}: 位置=({pos[0]:.1f},{pos[1]:.1f})km 置信半径={r:.2f}km")
        log.info(f"[层3-预测] sigma_man={sigma_man:.2f}km (机动不确定性)")
        log.info(f"[层4-区域] sigma_enemy={sigma_enemy:.2f}km | R_target={R_target:.2f}km")
        log.info(f"  敌方中心=({enemy_center[0]:.1f},{enemy_center[1]:.1f})km")

    fighter_positions = {}
    fighter_speeds = {}
    current_headings = {}
    for aid in ['A0100', 'A0200', 'A0300', 'A0400']:
        if aid in env.agents and env.agents[aid].is_alive:
            fighter_positions[aid] = self._get_battlefield_pos(env, aid)
            current_headings[aid] = self._get_heading(env, aid)
            try:
                fighter_speeds[aid] = float(env.agents[aid].get_property_value(c.velocities_vc_mps))
            except Exception:
                fighter_speeds[aid] = 250.0

    my_current_speed = fighter_speeds.get(agent_id, 250.0)
    guidance = self.formation_guidance.compute_guidance(
        enemy_center=enemy_center,
        confidence_radius=R_target,
        fighter_positions=fighter_positions,
    )

    my_guidance = guidance.get(agent_id)
    spd_cmd = 3
    target_hdg = current_hdg

    if my_guidance:
        target_hdg_raw = my_guidance.target_heading
        target_positions_for_speed = {fid: g.target_point for fid, g in guidance.items()}
        current_phase = self.cap_state_machine.state.value if hasattr(self.cap_state_machine, 'state') else 'INTERCEPT'
        self.velocity_coordination.compute_coordinated_speeds(
            fighter_positions=fighter_positions,
            fighter_speeds=fighter_speeds,
            target_positions=target_positions_for_speed,
            mission_phase=current_phase,
        )
        cmd = self.velocity_coordination.get_command(agent_id)
        target_speed_raw = my_current_speed
        if cmd:
            target_speed_raw = cmd.target_speed
            spd_cmd = self._speed_error_to_spd_cmd(target_speed_raw - my_current_speed)

        target_headings = {fid: g.target_heading for fid, g in guidance.items()}
        target_speeds = {}
        for fid in fighter_speeds:
            cmd_fid = self.velocity_coordination.get_command(fid)
            target_speeds[fid] = cmd_fid.target_speed if cmd_fid else fighter_speeds[fid]

        event_awacs_update = bool(getattr(self, '_awacs_updated_this_step', False)) and has_awacs_info
        event_maneuver = sigma_man > 5.0
        event_awacs_lost = not has_awacs_info
        event_target_lost_flag = target_lost
        new_headings, new_speeds, action_taken = self.coop_detection.dynamic_path_adjustment(
            current_headings=current_headings,
            current_speeds=fighter_speeds,
            target_headings=target_headings,
            target_speeds=target_speeds,
            event_awacs_update=event_awacs_update,
            event_maneuver_detected=event_maneuver,
            event_awacs_lost=event_awacs_lost,
            event_target_lost=event_target_lost_flag,
            alpha=0.7,
        )

        target_hdg = new_headings.get(agent_id, target_hdg_raw)
        target_speed_smoothed = new_speeds.get(agent_id, target_speed_raw)
        spd_cmd = self._speed_error_to_spd_cmd(target_speed_smoothed - my_current_speed)

        if _CAP_DEBUG_PRINT and self.cap_state_machine.state == CAPState.INTERCEPT and agent_id == 'A0100' and self.step_count % 150 == 0:
            log.info(f"[层5-规划] 编队引导（使用R_target={R_target:.2f}km）")
            for fid, g in guidance.items():
                log.info(f"  {fid}: 目标点=({g.target_point[0]:.1f},{g.target_point[1]:.1f}) "
                         f"航向={g.target_heading:.1f}° 距离={g.distance_to_target:.1f}km")
            log.info(f"[层6-协调V2] 速度协调 | 阶段={current_phase}")
            for fid in fighter_speeds:
                cmd_fid = self.velocity_coordination.get_command(fid)
                if cmd_fid:
                    log.info(f"  {fid}: 当前={fighter_speeds[fid]:.0f}m/s → "
                             f"目标={cmd_fid.target_speed:.0f}m/s (ETA={cmd_fid.eta:.1f}s)")
            constraints = self.velocity_coordination._get_phase_constraints()
            log.info(f"  阶段约束: v_min={constraints['v_min']:.0f} v_nominal={constraints['v_nominal']:.0f} v_max={constraints['v_max']:.0f} m/s")
            log.info(f"[层7-执行] 动态路径调整 | 动作={action_taken}")
            log.info(f"  事件: AWACS更新={'是' if event_awacs_update else '否'} | "
                     f"机动检测={'是' if event_maneuver else '否'} | "
                     f"AWACS丢失={'是' if event_awacs_lost else '否'} | "
                     f"目标丢失={'是' if event_target_lost_flag else '否'}")
            log.info(f"  {agent_id}: 当前航向={current_hdg:.1f}° → 原始目标={target_hdg_raw:.1f}° → 平滑后={target_hdg:.1f}° | "
                     f"速度 {my_current_speed:.0f}→{target_speed_smoothed:.0f}m/s")
            log.info(
                f"  {agent_id}: speed_error={speed_diff_smoothed:+.1f}m/s -> spd_cmd={spd_cmd} "
                f"(norm_vel={self.norm_vel[spd_cmd]:.3f})"
            )
            try:
                if os.environ.get('CAP_DEBUG_WINGMEN', '').strip().lower() in ('1', 'true', 'yes', 'y', 'on'):
                    for wid in ('A0200', 'A0400'):
                        if wid in current_headings and wid in target_headings:
                            hdg_cur = float(current_headings[wid])
                            hdg_raw = float(target_headings[wid])
                            hdg_new = float(new_headings.get(wid, hdg_raw))
                            log.info(
                                f"  [僚机引导] {wid}: cur={hdg_cur:.1f}° raw={hdg_raw:.1f}° new={hdg_new:.1f}° "
                                f"Δraw={((hdg_raw-current_hdg+180)%360-180):+.1f}° "
                                f"Δnew={((hdg_new-current_hdg+180)%360-180):+.1f}°"
                            )
            except Exception:
                pass
    else:
        dx = enemy_center[0] - x
        dy = enemy_center[1] - y
        target_hdg = np.degrees(np.arctan2(dx, dy)) % 360

    diff = target_hdg - current_hdg
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360

    diff_rad = np.radians(diff)
    if abs(diff) < 5.0:
        hdg_cmd = 8
    else:
        hdg_cmd = int(np.argmin(np.abs(self.norm_hdg - diff_rad)))

    if _CAP_DEBUG_PRINT and self.cap_state_machine.state == CAPState.INTERCEPT and agent_id == 'A0100' and self.step_count % 150 == 0:
        log.info(f"[航向命令V3] 当前:{current_hdg:.1f}° → 目标:{target_hdg:.1f}° | "
                 f"差值:{diff:.1f}° ({diff_rad:.3f}rad) | "
                 f"命令:hdg_cmd={hdg_cmd} (norm_hdg[{hdg_cmd}]={self.norm_hdg[hdg_cmd]:.3f}rad={np.degrees(self.norm_hdg[hdg_cmd]):.1f}°)")

    return 7, hdg_cmd, spd_cmd


def _get_enemy_action(self, env, agent_id: str) -> Tuple[int, int, int]:
    legacy = getattr(self, '_get_enemy_action_legacy', None)
    if legacy is not None:
        return legacy(env, agent_id)
    ac = env.agents.get(agent_id)
    if not ac or not ac.is_alive:
        return 7, 8, 3

    current_time = self.step_count * 0.2
    my_pos = self._get_battlefield_pos(env, agent_id)
    min_dist = float('inf')
    for aid in env.agents:
        if aid.startswith('A') and env.agents[aid].is_alive:
            pos = self._get_battlefield_pos(env, aid)
            min_dist = min(min_dist, np.sqrt((my_pos[0] - pos[0]) ** 2 + (my_pos[1] - pos[1]) ** 2))

    # 敌方桥接我方 CAPTask 逻辑已整体注释保留，默认不再进入该分支：
    # use_enemy_bridge = os.getenv("CAP_ENEMY_FRIENDLY_BRIDGE_ENABLED", "1").strip().lower() in (
    #     "1", "true", "yes", "on"
    # )
    # if use_enemy_bridge and hasattr(self, 'tactical_bridge') and self.tactical_bridge:
    #     pair_id = 'enemy_group_1' if agent_id in ('B0100', 'B0200') else 'enemy_group_2'
    #     bridge_takeover_distance_km = float(os.getenv('CAP_ENEMY_BRIDGE_TAKEOVER_DISTANCE_KM', '180'))
    #     bridge_takeover_time_s = float(os.getenv('CAP_ENEMY_BRIDGE_TAKEOVER_TIME_S', '90'))
    #     if (
    #         pair_id in self._enemy_bridge_takeover_pairs
    #         or (current_time >= bridge_takeover_time_s and min_dist <= bridge_takeover_distance_km)
    #     ):
    #         self._enemy_bridge_takeover_pairs.add(pair_id)
    #         try:
    #             bridge_action = self.tactical_bridge.get_enemy_action(env, agent_id, self)
    #             if bridge_action and len(bridge_action) >= 3:
    #                 return bridge_action
    #         except Exception as exc:
    #             log.warning(f"[EnemyBridge] fallback to EnemyAIAdapter for {agent_id}: {exc}")

    return self.enemy_adapter.get_enemy_action(env, agent_id, current_time, min_dist, task=self)
