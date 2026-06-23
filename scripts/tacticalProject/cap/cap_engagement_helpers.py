"""Extracted CAP cooperative-detection and engagement helpers."""

import logging
import os

import numpy as np
from envs.JSBSim.core.catalog import Catalog as c

try:
    from .cap_state_machine import CAPState
    from .control_ranges import DEFAULT_RANGES
    from .tactics import DetectionMode
except ImportError:
    from cap_state_machine import CAPState
    from control_ranges import DEFAULT_RANGES
    from tactics import DetectionMode

log = logging.getLogger(__name__)
_CAP_DEBUG_PRINT = os.environ.get('CAP_DEBUG_PRINT') == '1'


def _cap_print(*args, **kwargs):
    if _CAP_DEBUG_PRINT:
        print(*args, **kwargs)


def _should_emit_structured_log(self, time_attr: str, sig_attr: str, current_time: float, interval_s: float, signature):
    last_time = float(getattr(self, time_attr, -9999.0))
    last_sig = getattr(self, sig_attr, None)
    if (current_time - last_time) >= float(interval_s) or signature != last_sig:
        setattr(self, time_attr, float(current_time))
        setattr(self, sig_attr, signature)
        return True
    return False


def _format_short_agent_list(agent_ids):
    return "/".join(sorted(agent_ids)) if agent_ids else "-"


def _format_picture_summary(threats, ref_xy):
    if not threats:
        return "-"
    ref_x, ref_y = ref_xy
    items = []
    for threat in threats[:4]:
        try:
            distance = float(np.hypot(float(threat.x) - ref_x, float(threat.y) - ref_y))
        except Exception:
            distance = -1.0
        try:
            bearing = float(np.degrees(np.arctan2(float(threat.x) - ref_x, float(threat.y) - ref_y)) % 360.0)
        except Exception:
            bearing = float("nan")
        if np.isfinite(bearing):
            items.append(f"{threat.track_id}:{distance:.0f}km/{bearing:.0f}°")
        else:
            items.append(f"{threat.track_id}:{distance:.0f}km/-")
    return "; ".join(items) if items else "-"


def _mode_name(mode) -> str:
    if mode is None:
        return "UNKNOWN"
    return str(getattr(mode, "name", getattr(mode, "value", mode)))


def _prioritize_scan_tracks(tracks, current_time: float, max_targets: int = 10):
    """按新鲜度与置信度筛选扫描候选，落实态势级最多10目标约束。"""
    def _key(track):
        try:
            confidence = float(getattr(track, "confidence", 1.0) or 1.0)
        except Exception:
            confidence = 1.0
        try:
            timestamp = float(getattr(track, "timestamp", -1.0))
        except Exception:
            timestamp = -1.0
        age = max(0.0, float(current_time) - timestamp) if timestamp >= 0.0 else 9999.0
        return (confidence, -age, timestamp)

    ordered = sorted(list(tracks or []), key=_key, reverse=True)
    return ordered[: max(0, int(max_targets))]


def _record_first_awacs_detection(self, target_id: str, current_time: float) -> None:
    if not hasattr(self, "_target_first_awacs_time"):
        return
    tid = str(target_id or "")
    if not tid.startswith("B") or tid in self._target_first_awacs_time:
        return
    self._target_first_awacs_time[tid] = float(current_time)
    log.info("[DETECT_EVENT] t=%.0fs target=%s source=AWACS", current_time, tid)


def _record_first_fcr_detection(self, target_id: str, tracker_id: str, current_time: float, track_quality: float = 0.0) -> None:
    if not hasattr(self, "_target_first_fcr_time"):
        return
    tid = str(target_id or "")
    if not tid.startswith("B") or tid in self._target_first_fcr_time:
        return
    self._target_first_fcr_time[tid] = float(current_time)
    log.info(
        "[DETECT_EVENT] t=%.0fs target=%s source=FCR tracker=%s q=%.2f",
        current_time,
        tid,
        str(tracker_id or "-"),
        float(track_quality),
    )


def _sync_first_awacs_detection_from_live_tracks(self, env, current_time: float) -> dict:
    if hasattr(self, "_get_alive_awacs_tracks"):
        awacs_tracks = self._get_alive_awacs_tracks(env)
    else:
        awacs_tracks = self.awacs.get_tracks() if hasattr(self, "awacs") and self.awacs is not None else {}
    for tid in sorted(awacs_tracks.keys()):
        _record_first_awacs_detection(self, tid, current_time)
    return awacs_tracks


def _sync_first_fcr_detection_from_radar_tracks(self, radar_tracks, current_time: float) -> None:
    for aid, tracks in dict(radar_tracks or {}).items():
        for tid, trk in dict(tracks or {}).items():
            last_upd = float(getattr(trk, "last_update", 0.0))
            if last_upd > 0:
                self.coop_detection.update_track_time(tid, last_upd)
            try:
                track_q = float(self.cap_radar.get_track_quality(aid, tid))
            except Exception:
                track_q = 0.0
            _record_first_fcr_detection(self, tid, str(aid or "-"), current_time, track_q)


def _get_defensive_engagement_suppressed_agents(self, current_step: int) -> set[str]:
    store = getattr(self, "_defensive_engagement_suppressed_until", None)
    if not isinstance(store, dict):
        return set()

    now_step = int(current_step)
    suppressed = set()
    for aid, until_step in list(store.items()):
        try:
            until_step = int(until_step)
        except Exception:
            until_step = -1
        if until_step > now_step:
            suppressed.add(str(aid))
        else:
            store.pop(aid, None)
    return suppressed


def _clear_agent_engagement_state(self, agent_id: str) -> None:
    aid = str(agent_id or "")
    if not aid.startswith("A"):
        return

    radar = getattr(self, "cap_radar", None)
    if radar is not None:
        try:
            radar.lock_target(aid, None)
        except Exception:
            pass

    impacted_targets = set()
    coop_engagement = getattr(self, "coop_engagement", None)
    if coop_engagement is not None:
        assignments = getattr(coop_engagement, "assignments", None)
        if isinstance(assignments, dict):
            assignment = assignments.pop(aid, None)
            target_id = getattr(assignment, "target_id", None)
            if target_id:
                impacted_targets.add(str(target_id))

        tracking_status = getattr(coop_engagement, "tracking_status", None)
        if isinstance(tracking_status, dict):
            for tid, status in list(tracking_status.items()):
                if aid in (
                    str(getattr(status, "primary_tracker", "") or ""),
                    str(getattr(status, "secondary_tracker", "") or ""),
                ):
                    impacted_targets.add(str(tid))
                    tracking_status.pop(tid, None)

        last_assignment_time = getattr(coop_engagement, "_last_assignment_time", None)
        if isinstance(last_assignment_time, dict):
            for tid in impacted_targets:
                last_assignment_time.pop(tid, None)

    track_event_last = getattr(self, "_track_event_last", None)
    if isinstance(track_event_last, dict):
        track_event_last.pop(aid, None)

    trackers_store = getattr(self, "_stable_tracking_trackers", None)
    elapsed_store = getattr(self, "_stable_tracking_elapsed", None)
    last_ok_store = getattr(self, "_stable_tracking_last_ok", None)
    ready_store = getattr(self, "_stable_tracking_ready_announced", None)
    if isinstance(trackers_store, dict):
        for tid, trackers in list(trackers_store.items()):
            if aid not in trackers:
                continue
            remaining = [tracker_id for tracker_id in trackers if tracker_id != aid]
            if len(remaining) >= 2:
                trackers_store[tid] = remaining
                continue
            trackers_store.pop(tid, None)
            if isinstance(elapsed_store, dict):
                elapsed_store[tid] = 0.0
            if isinstance(last_ok_store, dict):
                last_ok_store.pop(tid, None)
            if isinstance(ready_store, set):
                ready_store.discard(tid)
    if impacted_targets:
        if isinstance(elapsed_store, dict):
            for tid in impacted_targets:
                elapsed_store[tid] = 0.0
        if isinstance(last_ok_store, dict):
            for tid in impacted_targets:
                last_ok_store.pop(tid, None)
        if isinstance(ready_store, set):
            for tid in impacted_targets:
                ready_store.discard(tid)


def _collect_detection_targets(self, awacs_tracks):
    positions = {}
    samples = []
    if awacs_tracks:
        for tid in sorted(awacs_tracks.keys()):
            track = awacs_tracks.get(tid)
            pos = getattr(track, "position", None)
            if pos is None or len(pos) < 2:
                continue
            xy = (float(pos[0]), float(pos[1]))
            positions[tid] = xy
            samples.append((tid, xy[0], xy[1]))
        return "AWACS", positions, samples

    for threat in list(self._get_picture_threats(purpose="search")):
        if hasattr(threat, "x") and hasattr(threat, "y"):
            xy = (float(threat.x), float(threat.y))
            positions[threat.track_id] = xy
            samples.append((threat.track_id, xy[0], xy[1]))
    return "PICTURE", positions, samples


def _compute_detection_min_distance(agent_positions, target_positions) -> float:
    min_distance = float("inf")
    for ax, ay in agent_positions.values():
        for tx, ty in target_positions.values():
            min_distance = min(min_distance, float(np.hypot(tx - ax, ty - ay)))
    return min_distance


def _estimate_detection_region(self, target_positions, has_awacs: bool, current_time: float):
    if not target_positions:
        return None

    positions = list(target_positions.values())
    center_x = float(np.mean([p[0] for p in positions]))
    center_y = float(np.mean([p[1] for p in positions]))

    if len(positions) <= 1:
        sigma_spatial = 5.0
    else:
        sigma_spatial = float(
            np.sqrt(sum((p[0] - center_x) ** 2 + (p[1] - center_y) ** 2 for p in positions) / len(positions))
        )

    sigma_source = 2.5 if has_awacs else 0.5
    sigma_man = float(getattr(self.coop_detection, "confidence_radius", 10.0))
    try:
        sigma_man_prop, _ = self.coop_detection.propagate_sigma_man(
            current_time=current_time,
            gamma=0.99,
            target_ids=list(target_positions.keys()),
        )
        sigma_man = float(sigma_man_prop)
    except Exception:
        pass

    sigma_enemy = max(sigma_spatial, sigma_source) + max(0.0, sigma_man - sigma_source)
    r_target = 3.0 * sigma_enemy
    return center_x, center_y, sigma_man, sigma_enemy, r_target


def _predict_search_bearing(self, hot_agents, agent_positions, current_time: float):
    pos_list = [agent_positions[aid] for aid in hot_agents if aid in agent_positions]
    if not pos_list:
        return None

    center_x = float(np.mean([p[0] for p in pos_list]))
    center_y = float(np.mean([p[1] for p in pos_list]))
    predicted_bearings = []
    for info in getattr(self.coop_detection, "_last_known_targets", {}).values():
        if not info:
            continue
        pos = info.get("pos", (0.0, 100.0))
        vel = info.get("vel", (0.0, -0.3))
        dt = max(0.0, current_time - float(info.get("time", current_time)))
        pred_x = float(pos[0]) + float(vel[0]) * dt
        pred_y = float(pos[1]) + float(vel[1]) * dt
        bearing = float(np.degrees(np.arctan2(pred_x - center_x, pred_y - center_y)))
        bearing = (bearing + 180.0) % 360.0 - 180.0
        predicted_bearings.append(bearing)

    if not predicted_bearings:
        return None
    return float(np.mean(predicted_bearings))


def _emit_coop_detection_story(
    self,
    current_time: float,
    cap_state,
    has_awacs: bool,
    awacs_tracks,
    target_lost: bool,
    hot_agents,
    cold_agents,
    agent_positions,
    assignments,
    radar_tracks,
    mode_name: str,
    prev_mode_name: str,
):
    source_name, target_positions, target_samples = _collect_detection_targets(self, awacs_tracks)
    if not target_positions:
        return

    region = _estimate_detection_region(self, target_positions, has_awacs, current_time)
    if region is None:
        return
    center_x, center_y, sigma_man, sigma_enemy, r_target = region
    min_distance = _compute_detection_min_distance(agent_positions, target_positions)
    radar_track_count = sum(len(tracks) for tracks in radar_tracks.values())

    summary_signature = (
        mode_name,
        prev_mode_name,
        bool(has_awacs),
        bool(target_lost),
        tuple(sorted(target_positions.keys())),
        tuple(sorted(assignments.keys())),
        int(min_distance // 10.0) if min_distance != float("inf") else -1,
    )
    if not _should_emit_structured_log(
        self,
        "_last_coop_detect_story_time",
        "_last_coop_detect_story_signature",
        current_time,
        30.0,
        summary_signature,
    ):
        return

    scan_items = []
    for aid in sorted(assignments.keys())[:4]:
        asgn = assignments[aid]
        scan_items.append(
            f"{aid}:{float(asgn.azimuth_center):+.0f}°/±{float(asgn.azimuth_range):.0f}°x{int(asgn.elevation_bars)}"
        )
    events = []
    if bool(getattr(self, "_awacs_updated_this_step", False)) and has_awacs:
        events.append("AWACS更新")
    if target_lost:
        events.append("目标丢失")
    if prev_mode_name != mode_name:
        events.append(f"模式切换{prev_mode_name}->{mode_name}")

    log.info(
        "[%ds] 距离:%dkm 模式:DetectionMode.%s 阶段:%s 锁定:%d",
        int(round(current_time)),
        int(round(min_distance)) if min_distance != float("inf") else -1,
        mode_name,
        getattr(cap_state, "value", str(cap_state)),
        radar_track_count,
    )
    log.info("[算法0] 协同探测7层架构 | 时间=%.1fs 步数=%d", current_time, self.step_count)
    log.info(
        "[层1-信息] 预警信息=%s | 目标数=%d | 目标丢失=%s | 航迹源=%s",
        "有" if has_awacs else "无",
        len(target_positions),
        "是" if target_lost else "否",
        source_name,
    )
    log.info("[层2-估计] 目标状态汇总 %d 个目标", len(target_positions))
    for tid, x_pos, y_pos in target_samples[:3]:
        log.info("  目标%s: 位置=(%.1f, %.1f)km", tid, x_pos, y_pos)
    log.info("[层3-预测] sigma_man=%.2fkm (机动不确定性)", sigma_man)
    log.info("[层4-区域] sigma_enemy=%.2fkm | R_target=%.2fkm", sigma_enemy, r_target)
    log.info("  敌方中心=(%.1f, %.1f)km", center_x, center_y)
    log.info(
        "[层5-规划] 扫描分配=%s",
        "; ".join(scan_items) if scan_items else "-",
    )
    log.info(
        "[层6-协调] 热段=%s | 冷段=%s",
        _format_short_agent_list(hot_agents),
        _format_short_agent_list(cold_agents),
    )
    log.info(
        "[层7-执行] 模式=%s | 事件=%s",
        mode_name,
        "/".join(events) if events else "稳定跟踪",
    )

def _update_cooperative_detection(self, env, current_time: float):
    """更新协同探测和雷达
    
    改进：所有我方飞机都进行雷达扫描，不仅仅是热段飞机
    - 热段飞机：协同探测分配（推磨/定向/全域）
    - 冷段飞机：默认前向扫描（±60°）
    """
    # ===== Baseline：固定扫描（不使用协同扫描分配/SEARCH恢复等） =====
    if getattr(self, 'experiment_mode', 'proposed') == 'baseline':
        all_friendlies = [aid for aid in env.agents if aid.startswith('A') and env.agents[aid].is_alive]
        if not self.cap_radar._radar_modes:
            self.cap_radar.init_agents(all_friendlies)

        if hasattr(self, "_get_alive_awacs_tracks"):
            awacs_tracks = self._get_alive_awacs_tracks(env)
        else:
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

        if pts:
            ex = float(np.mean([p[0] for p in pts]))
            ey = float(np.mean([p[1] for p in pts]))
        else:
            ex, ey = 100.0, 250.0

        # 仅用于验证脚本统计模式驻留；baseline不参与SWEEP/DIRECTED/SEARCH优化
        try:
            self.coop_detection._mode = DetectionMode.DIRECTED if awacs_tracks else DetectionMode.SWEEP
        except Exception:
            pass

        for aid in all_friendlies:
            ax, ay = self._get_battlefield_pos(env, aid)
            try:
                my_heading = float(np.degrees(env.agents[aid].get_property_value(c.attitude_heading_true_rad)) % 360)
            except Exception:
                my_heading = 0.0
            bearing_true = float(np.degrees(np.arctan2(ex - ax, ey - ay)) % 360)
            rel = bearing_true - my_heading
            if rel > 180:
                rel -= 360
            elif rel <= -180:
                rel += 360

            # 固定MEDIUM扫描：±30°，2bars
            self.cap_radar.set_scan_assignment(aid, rel, 30.0, 2)

        self.cap_radar.update(env, current_time)
        return

    # 初始化雷达（首次）
    all_friendlies = [aid for aid in env.agents if aid.startswith('A') and env.agents[aid].is_alive]
    if not self.cap_radar._radar_modes:
        self.cap_radar.init_agents(all_friendlies)
    
    # 获取热段飞机和位置（ENGAGE/INTERCEPT阶段所有飞机参与）
    patrol_states = {aid: str(pm.state) for aid, pm in self.patrol_machines.items()}
    cap_state = str(self.cap_state_machine.state) if hasattr(self, 'cap_state_machine') else None
    if getattr(self, "_detection_focus_mode", False) and bool(getattr(self, "_focus_force_all_hot", False)):
        hot_agents = list(all_friendlies)
    else:
        hot_agents = self.coop_detection.get_hot_agents(patrol_states, cap_state=cap_state)
    
    # 🔥 Debug: 打印hot_agents筛选结果（每60秒一次）
    if current_time % 60 < 0.3:
        _cap_print(f"[热段筛选] cap_state={cap_state} | hot_agents={hot_agents} (共{len(hot_agents)}架)")
        _cap_print(f"  patrol_states: {patrol_states}")
    
    # 获取所有飞机位置
    agent_positions = {}
    agent_headings = {}
    for aid in all_friendlies:
        if aid in env.agents and env.agents[aid].is_alive:
            pos = self._get_battlefield_pos(env, aid)
            agent_positions[aid] = pos
            try:
                agent_headings[aid] = float(np.degrees(env.agents[aid].get_property_value(c.attitude_heading_true_rad)) % 360)
            except Exception:
                agent_headings[aid] = 0.0
    
    # 检查是否有目标丢失（覆盖“掉出当前输入列表”的目标）
    target_lost = self.coop_detection.any_target_lost(current_time)
    prev_mode_name = _mode_name(getattr(self.coop_detection, "mode", None))
    
    # 获取预警机信息
    awacs_tracks = _sync_first_awacs_detection_from_live_tracks(self, env, current_time)
    has_awacs = bool(awacs_tracks)
    predictive_search_active = (
        (not has_awacs)
        and (
            bool(target_lost)
            or bool(self.coop_detection._has_predictive_context(current_time))
        )
    )
    detection_agents = list(all_friendlies) if predictive_search_active else list(hot_agents)

    # 由AWACS航迹计算目标平均方位（相对编队中心/热段中心），不使用固定(x=100,y=0)
    awacs_bearing = None
    if has_awacs:
        # 参考中心：优先热段飞机中心，否则用全部飞机中心
        ref_agents = detection_agents if detection_agents else list(agent_positions.keys())
        if ref_agents:
            xs = [agent_positions[a][0] for a in ref_agents if a in agent_positions]
            ys = [agent_positions[a][1] for a in ref_agents if a in agent_positions]
            cx = float(np.mean(xs)) if xs else 100.0
            cy = float(np.mean(ys)) if ys else 0.0
        else:
            cx, cy = 100.0, 0.0

        track_positions = []
        for t in awacs_tracks.values():
            pos = getattr(t, 'position', None)
            if pos is None or len(pos) < 2:
                continue
            track_positions.append((float(pos[0]), float(pos[1])))

        if track_positions:
            txm = float(np.mean([p[0] for p in track_positions]))
            tym = float(np.mean([p[1] for p in track_positions]))
            awacs_bearing = float((np.degrees(np.arctan2(txm - cx, tym - cy)) + 360) % 360)
    
    # 🔥 修复：为每架飞机计算目标方位（不使用固定编队中心）
    # 使用最近威胁的平均方位作为代表（简化处理）
    search_threats = _prioritize_scan_tracks(
        self._get_picture_threats(current_time=current_time, purpose="search"),
        current_time=current_time,
        max_targets=10,
    )
    all_target_bearings = []
    target_bearing_infos = []
    if detection_agents and agent_positions:
        # 对每个目标，计算其相对各热段飞机的平均方位
        for threat in search_threats:
            bearings_for_this_threat = []
            for aid in detection_agents:
                if aid in agent_positions:
                    ax, ay = agent_positions[aid]
                    # 从飞机位置计算目标方位（真方位，真北=0°）
                    bearing = np.degrees(np.arctan2(threat.x - ax, threat.y - ay))
                    bearings_for_this_threat.append(bearing)
            if bearings_for_this_threat:
                # 使用平均方位（处理角度环绕）
                avg_bearing = np.degrees(np.arctan2(
                    np.mean([np.sin(np.radians(b)) for b in bearings_for_this_threat]),
                    np.mean([np.cos(np.radians(b)) for b in bearings_for_this_threat])
                ))
                avg_bearing = float(avg_bearing)
                all_target_bearings.append(avg_bearing)
                target_bearing_infos.append({
                    "target_id": str(getattr(threat, "track_id", f"slot_{len(target_bearing_infos)}")),
                    "bearing": avg_bearing,
                    "confidence": float(getattr(threat, "confidence", 1.0) or 1.0),
                    "priority": float(getattr(threat, "confidence", 1.0) or 1.0),
                })
    
    # 🔥 Debug: 打印目标方位信息（每120秒一次）
    if current_time % 120 < 0.3 and all_target_bearings:
        threat_info = []
        for i, threat in enumerate(search_threats):
            if i < len(all_target_bearings):
                bearing = all_target_bearings[i]
                threat_info.append(f"{threat.track_id}@{bearing:.1f}°(x={threat.x:.1f})")
        _cap_print(f"[目标方位] 共{len(all_target_bearings)}个: {', '.join(threat_info)}")
    
    # 更新热段飞机的扫描分配（传递所有目标方位实现自适应覆盖）
    filtered_positions = {k: v for k, v in agent_positions.items() if k in detection_agents}
    
    # 🔥 Debug: 打印传递给update的参数（每60秒一次）
    if current_time % 60 < 0.3:
        _cap_print(f"[协同探测调用] hot_agents={hot_agents} (共{len(hot_agents)}架)")
        _cap_print(f"  agent_positions (全部): {list(agent_positions.keys())}")
        _cap_print(f"  filtered_positions: {list(filtered_positions.keys())} (共{len(filtered_positions)}个)")
        _cap_print(f"  all_target_bearings: {len(all_target_bearings)}个")
    
    assignments = self.coop_detection.update(
        detection_agents, has_awacs, target_lost, awacs_bearing, current_time,
        agent_positions=filtered_positions,
        all_target_bearings=all_target_bearings,
        agent_headings={k: v for k, v in agent_headings.items() if k in detection_agents},
        target_bearing_infos=target_bearing_infos,
    )
    
    # 将扫描分配传递给CAP雷达
    scan_summary = {}
    for aid, asgn in assignments.items():
        # [Fix] 坐标系转换：协同探测输出真方位 -> 雷达需要相对方位
        try:
            ac = env.agents[aid]
            my_heading = np.degrees(ac.get_property_value(c.attitude_heading_true_rad)) % 360
        except:
            my_heading = 0.0
            
        # 🔥 修复：正确的相对方位计算
        # 真方位 - 飞机航向 = 相对方位（机头为0°）
        # 例如：目标真方位15.6°，飞机航向346°，相对方位 = 15.6 - 346 = -330.4° → 归一化为 29.6°
        relative_center = asgn.azimuth_center - my_heading
        # 归一化到 (-180, +180]
        if relative_center > 180:
            relative_center -= 360
        elif relative_center <= -180:
            relative_center += 360
        
        # [Debug] 输出雷达指向校准信息 (仅A0100且低频)
        if _CAP_DEBUG_PRINT and aid == 'A0100' and self.step_count % 100 == 1:
            log.info(f"[雷达校准] A0100 航向:{my_heading:.0f}° | 目标真方位:{asgn.azimuth_center:.0f}° -> 相对方位:{relative_center:+.0f}°")
        
        # 🔥 诊断：打印扇区分配传递详情（每300次=60秒）
        if _CAP_DEBUG_PRINT and self.step_count % 300 == 1 and self.cap_state_machine.state in (CAPState.PATROL, CAPState.INTERCEPT):
            _cap_print(f"[扇区传递] {aid} | 航向:{my_heading:.1f}° 真方位中心:{asgn.azimuth_center:.1f}° -> 相对方位中心:{relative_center:+.1f}° 范围:±{asgn.azimuth_range:.1f}°")
        scan_summary[aid] = f"{aid}:{relative_center:+.0f}/{float(asgn.azimuth_range):.0f}x{int(asgn.elevation_bars)}"
        self.cap_radar.set_scan_assignment(aid, relative_center, asgn.azimuth_range, asgn.elevation_bars)
    
    # 协同探测专用验证下，不关闭冷段机雷达，否则 proposed 天然少于 baseline 的并行搜索资源。
    cold_agents = [aid for aid in all_friendlies if aid not in detection_agents]
    if getattr(self, "_detection_focus_mode", False):
        for aid in cold_agents:
            if aid in agent_positions:
                ax, ay = agent_positions[aid]
                fallback_true = awacs_bearing
                if fallback_true is None and all_target_bearings:
                    try:
                        fallback_true = float(
                            np.degrees(
                                np.arctan2(
                                    np.mean([np.sin(np.radians(b)) for b in all_target_bearings]),
                                    np.mean([np.cos(np.radians(b)) for b in all_target_bearings]),
                                )
                            )
                        )
                    except Exception:
                        fallback_true = None
                if fallback_true is None:
                    fallback_true = 0.0
                try:
                    my_heading = float(np.degrees(env.agents[aid].get_property_value(c.attitude_heading_true_rad)) % 360)
                except Exception:
                    my_heading = 0.0
                relative_center = float(fallback_true) - my_heading
                if relative_center > 180:
                    relative_center -= 360
                elif relative_center <= -180:
                    relative_center += 360
                self.cap_radar.set_scan_assignment(aid, relative_center, 30.0, 2)
    else:
        for aid in cold_agents:
            self.cap_radar.set_radar_off(aid)  # 关闭雷达
    
    # 更新CAP雷达
    radar_tracks = self.cap_radar.update(env, current_time)
    
    # 日志（每240秒，减少频率）
    if self.step_count % 300 == 1:
        # 简单打印
        pass
    
    # 更新目标跟踪时间（用于丢失检测/σ_man传播）
    # 仅在雷达航迹本步真实更新时推进时间戳（使用RadarTrack.last_update）
    _sync_first_fcr_detection_from_radar_tracks(self, radar_tracks, current_time)

    mode_name = _mode_name(getattr(self.coop_detection, "mode", None))
    if prev_mode_name != mode_name:
        log.info(
            "[%ds] 探测模式变化: DetectionMode.%s -> DetectionMode.%s",
            int(round(current_time)),
            prev_mode_name,
            mode_name,
        )

    if mode_name == "SEARCH" and target_lost and _should_emit_structured_log(
        self,
        "_last_detect_search_log_time",
        "_last_detect_search_log_signature",
        current_time,
        15.0,
        ("SEARCH", True),
    ):
        log.info("[协同] 目标丢失，立即进入SEARCH模式")
        predicted_bearing = _predict_search_bearing(self, detection_agents, agent_positions, current_time)
        if predicted_bearing is not None:
            log.info("[协同] 智能搜索: 预测方位=%.1f° 覆盖范围=±60°", predicted_bearing)
    elif prev_mode_name == "SEARCH" and mode_name == "DIRECTED":
        log.info("[协同] 目标重新获取，恢复DIRECTED模式")

    # 日志（每60秒一次）
    if self.step_count % 300 == 1:
        total_tracks = sum(len(self.cap_radar.get_tracks(aid)) for aid in self.cap_radar._radar_tracks)
        log.info(f"[协同] [协同探测] 模式:{mode_name} 雷达航迹:{total_tracks}")

    detect_signature = (
        mode_name,
        tuple(sorted(detection_agents)),
        tuple(sorted(cold_agents)),
        bool(has_awacs),
        bool(target_lost),
        tuple(sorted(assignments.keys())),
    )
    if _should_emit_structured_log(
        self,
        "_last_coop_detect_log_time",
        "_last_coop_detect_signature",
        current_time,
        30.0,
        detect_signature,
    ):
        ref_agents = detection_agents or all_friendlies
        if ref_agents:
            ref_x = float(np.mean([agent_positions[aid][0] for aid in ref_agents if aid in agent_positions]))
            ref_y = float(np.mean([agent_positions[aid][1] for aid in ref_agents if aid in agent_positions]))
        else:
            ref_x, ref_y = 100.0, 0.0
        picture_summary = _format_picture_summary(
            list(self._get_picture_threats(current_time=current_time, purpose="search")),
            (ref_x, ref_y),
        )
        scan_text = "; ".join(scan_summary.get(aid, f"{aid}:OFF") for aid in sorted(all_friendlies)) or "-"
        track_text_items = []
        for aid in sorted(all_friendlies):
            tids = sorted(list(radar_tracks.get(aid, {}).keys()))[:2]
            track_text_items.append(f"{aid}->{'/'.join(tids) if tids else '-'}")
        track_text = "; ".join(track_text_items) if track_text_items else "-"
        awacs_bearing_text = "-" if awacs_bearing is None else f"{float(awacs_bearing):.0f}°"
        log.info(
            "[COOP_DETECT] t=%.0fs mode=%s awacs=%s lost=%s bearing=%s hot=%s cold=%s picture=%s scans=%s tracks=%s",
            current_time,
            mode_name,
            len(awacs_tracks),
            "Y" if target_lost else "N",
            awacs_bearing_text,
            _format_short_agent_list(detection_agents),
            _format_short_agent_list(cold_agents),
            picture_summary,
            scan_text,
            track_text,
        )
        _emit_coop_detection_story(
            self,
            current_time,
            self.cap_state_machine.state,
            has_awacs,
            awacs_tracks,
            target_lost,
            detection_agents,
            cold_agents,
            agent_positions,
            assignments,
            radar_tracks,
            mode_name,
            prev_mode_name,
        )

def _update_radar_tracking_only(self, env, current_time: float):
    """交战后雷达跟踪更新：不再执行协同探测7层分配。"""
    all_friendlies = [aid for aid in env.agents if aid.startswith('A') and env.agents[aid].is_alive]
    if not all_friendlies:
        return
    _sync_first_awacs_detection_from_live_tracks(self, env, current_time)
    alive_enemy_ids = set()
    if hasattr(self, "_clear_enemy_contact_state"):
        try:
            alive_enemy_ids = set(self._clear_enemy_contact_state(env, reason="tracking_only_sync"))
        except Exception:
            alive_enemy_ids = set()
    else:
        alive_enemy_ids = {
            str(aid) for aid, aircraft in getattr(env, "agents", {}).items()
            if str(aid).startswith("B") and getattr(aircraft, "is_alive", False)
        }
    if not alive_enemy_ids:
        for aid in all_friendlies:
            self.cap_radar.lock_target(aid, None)
        return

    if not self.cap_radar._radar_modes:
        self.cap_radar.init_agents(all_friendlies)

    for aid in all_friendlies:
        ax, ay = self._get_battlefield_pos(env, aid)
        try:
            my_heading = float(np.degrees(env.agents[aid].get_property_value(c.attitude_heading_true_rad)) % 360)
        except Exception:
            my_heading = 0.0

        target_id = None
        assignment = self.coop_engagement.get_assignment(aid) if hasattr(self, 'coop_engagement') else None
        if assignment:
            target_id = assignment.target_id

        target = None
        if target_id:
            for threat in self._get_picture_threats(purpose="decision"):
                if threat.track_id == target_id:
                    target = threat
                    break
        if target is None:
            target = self._get_picture_nearest_threat(np.array([ax, ay, 8.0]), purpose="decision")

        if target is None:
            self.cap_radar.set_scan_assignment(aid, 0.0, 60.0, 2)
            continue

        bearing_true = float(np.degrees(np.arctan2(target.x - ax, target.y - ay)) % 360)
        rel = bearing_true - my_heading
        if rel > 180:
            rel -= 360
        elif rel <= -180:
            rel += 360

        self.cap_radar.set_scan_assignment(aid, rel, 70.0, 4)

    radar_tracks = self.cap_radar.update(env, current_time)
    _sync_first_fcr_detection_from_radar_tracks(self, radar_tracks, current_time)
    try:
        track_qualities = self.cap_radar.get_all_track_qualities()
        self._update_stable_tracking_window(track_qualities, current_time)
    except Exception:
        pass

def _update_engagement(self, env, current_time: float):
    """更新协同跟踪、目标分配和接力制导"""
    # Baseline：不做协同交战/锁定分配
    if getattr(self, 'experiment_mode', 'proposed') == 'baseline':
        for aid in [a for a in env.agents if a.startswith('A')]:
            self.cap_radar.lock_target(aid, None)
        return

    alive_enemy_ids = set()
    if hasattr(self, "_clear_enemy_contact_state"):
        try:
            alive_enemy_ids = set(self._clear_enemy_contact_state(env, reason="engagement_sync"))
        except Exception:
            alive_enemy_ids = set()
    else:
        alive_enemy_ids = {
            str(aid) for aid, aircraft in getattr(env, "agents", {}).items()
            if str(aid).startswith("B") and getattr(aircraft, "is_alive", False)
        }
    if not alive_enemy_ids:
        for aid in [a for a in env.agents if a.startswith('A')]:
            self.cap_radar.lock_target(aid, None)
        return

    # 只在交战和拦截阶段进行分配计算
    if self.cap_state_machine.state not in (CAPState.ENGAGE, CAPState.INTERCEPT):
        for aid in [a for a in env.agents if a.startswith('A')]:
            self.cap_radar.lock_target(aid, None)
        return
         
    # 获取可用射手和位置
    current_step = int(getattr(env, "current_step", self.step_count))
    suppressed_agents = _get_defensive_engagement_suppressed_agents(self, current_step)
    all_friendlies = [aid for aid in env.agents if aid.startswith('A') and env.agents[aid].is_alive]
    for aid in sorted(suppressed_agents):
        if aid in all_friendlies:
            _clear_agent_engagement_state(self, aid)

    friendlies = [aid for aid in all_friendlies if aid not in suppressed_agents]
    if not friendlies:
        for aid in all_friendlies:
            self.cap_radar.lock_target(aid, None)
        return
    my_positions = {}
    for aid in friendlies:
        ac = env.agents[aid]
        lon = ac.get_property_value(c.position_long_gc_deg)
        lat = ac.get_property_value(c.position_lat_geod_deg)
        alt_ft = ac.get_property_value(c.position_h_sl_ft)
        x, y = self.coord_sys.geodetic_to_battlefield(lon, lat)
        z = alt_ft * 0.3048 / 1000.0
        my_positions[aid] = np.array([x, y, z])
    
    # 获取跟踪质量
    track_qualities = self.cap_radar.get_all_track_qualities()
    
    # 计算距发射距离（用于判断是否启动协同跟踪）
    ctx = self._build_state_context(env, current_time)
    current_ranges = getattr(self, "ranges", DEFAULT_RANGES)
    distance_to_launch = ctx.min_threat_distance - float(getattr(current_ranges, "LR", DEFAULT_RANGES.LR))
    
    # 更新分配（支持协同跟踪）
    actionable_threats = self._get_picture_threats(current_time=current_time, purpose="decision")
    assignments = self.coop_engagement.update(
        friendlies,
        self.picture,
        my_positions,
        track_qualities=track_qualities,
        current_time=current_time,
        distance_to_launch=distance_to_launch,
        threats=actionable_threats,
    )
    pruned_assignments = {}
    dropped_targets = set()
    guard_log_state = getattr(self, "_pair_assign_guard_log", None)
    if not isinstance(guard_log_state, dict):
        guard_log_state = {}
        setattr(self, "_pair_assign_guard_log", guard_log_state)
    for aid, asgn in list(assignments.items()):
        target_id = getattr(asgn, "target_id", None)
        target_status = self._get_picture_target_actionability(
            target_id,
            current_time=current_time,
            purpose="decision",
        )
        pair_ok = self._is_agent_target_pair_compatible(
            aid,
            target_id,
            env=env,
            current_time=current_time,
        )
        if bool(target_status.get("actionable", True)) and pair_ok:
            pruned_assignments[aid] = asgn
            continue

        dropped_targets.add(str(target_id or ""))
        _clear_agent_engagement_state(self, aid)
        guard_signature = (
            aid,
            str(target_id or ""),
            bool(target_status.get("actionable", True)),
            tuple(target_status.get("reasons", [])),
            bool(pair_ok),
        )
        last_signature, last_time = guard_log_state.get(aid, (None, -1e9))
        if guard_signature != last_signature or (float(current_time) - float(last_time)) >= 15.0:
            log.info(
                "[PAIR_ASSIGN_GUARD] t=%.0fs %s drop=%s actionable=%s reasons=%s pair_ok=%s",
                current_time,
                aid,
                target_id or "-",
                "Y" if bool(target_status.get("actionable", True)) else "N",
                ",".join(target_status.get("reasons", [])) or "-",
                "Y" if pair_ok else "N",
            )
            guard_log_state[aid] = (guard_signature, float(current_time))

    if len(pruned_assignments) != len(assignments):
        assignments = pruned_assignments
        self.coop_engagement.assignments = dict(pruned_assignments)
        valid_target_ids = {str(getattr(asgn, "target_id", "") or "") for asgn in pruned_assignments.values()}

        tracking_status = getattr(self.coop_engagement, "tracking_status", None)
        if isinstance(tracking_status, dict):
            self.coop_engagement.tracking_status = {
                tid: status
                for tid, status in tracking_status.items()
                if str(tid) in valid_target_ids
            }

        last_assignment_time = getattr(self.coop_engagement, "_last_assignment_time", None)
        if isinstance(last_assignment_time, dict):
            for tid in dropped_targets:
                last_assignment_time.pop(str(tid), None)
    assigned_ids = set(assignments.keys())
    for aid in all_friendlies:
        if aid not in assigned_ids:
            self.cap_radar.lock_target(aid, None)
    
    # 执行锁定
    for aid, asgn in assignments.items():
        if self.cap_state_machine.state == CAPState.ENGAGE:
            # 🔥 修复：只锁定有雷达航迹的目标（符合物理规律）
            # 避免在Tacview中显示不准确的LockedTarget红线
            my_tracks = self.cap_radar.get_tracks(aid)
            if asgn.target_id in my_tracks:
                self.cap_radar.lock_target(aid, asgn.target_id)
            else:
                self.cap_radar.lock_target(aid, None)
        else:
            self.cap_radar.lock_target(aid, None)
    
    # 行 3387: 删除目标分配打印
    # 日志（每500步或有协同跟踪时）
    coop_targets = self.coop_engagement.get_cooperative_tracking_targets()
    if coop_targets and self.step_count % 500 == 1:
        log.info(f"[协同跟踪目标] 目标: {coop_targets}")

    # 更新“发射前稳定双机跟踪”窗口
    self._update_stable_tracking_window(track_qualities, current_time)
    self._cooperative_tracking_active = bool(coop_targets)

    assignment_summary = []
    assignment_signature = []
    for aid in sorted(assignments.keys()):
        target_id = getattr(assignments[aid], "target_id", None)
        q = float(track_qualities.get(aid, {}).get(target_id, 0.0)) if target_id else 0.0
        locked = self.cap_radar.get_lock_target(aid)
        lock_ok = bool(locked == target_id and target_id)
        assignment_summary.append(
            f"{aid}->{target_id or '-'}(q={q:.2f},lock={'Y' if lock_ok else 'N'})"
        )
        assignment_signature.append((aid, target_id, lock_ok))
        prev_event = self._track_event_last.get(aid)
        current_event = (target_id or "-", lock_ok)
        if prev_event != current_event:
            self._track_event_last[aid] = current_event
            prev_target = prev_event[0] if prev_event else "-"
            prev_lock = bool(prev_event[1]) if prev_event else False
            log.info(
                "[跟踪事件] t=%.0fs %s target=%s->%s lock=%s->%s q=%.2f",
                current_time,
                aid,
                prev_target,
                target_id or "-",
                "Y" if prev_lock else "N",
                "Y" if lock_ok else "N",
                q,
            )

    stable_summary = []
    stable_signature = []
    for tid in sorted(self._stable_tracking_trackers.keys()):
        trackers = list(self._stable_tracking_trackers.get(tid, []))
        hold = float(self._stable_tracking_elapsed.get(tid, 0.0))
        stable_summary.append(f"{tid}:{'/'.join(trackers)} {hold:.1f}/{self._stable_tracking_window_s:.0f}s")
        stable_signature.append((tid, tuple(trackers), int(hold // 5.0)))

    track_signature = (
        tuple(assignment_signature),
        tuple(sorted(coop_targets)),
        tuple(sorted(self._stable_tracking_ready_announced)),
        tuple(stable_signature),
    )
    if _should_emit_structured_log(
        self,
        "_last_coop_track_log_time",
        "_last_coop_track_signature",
        current_time,
        45.0,
        track_signature,
    ):
        log.info(
            "[协同跟踪] t=%.0fs coop=%s assignments=%s stable=%s ready=%s",
            current_time,
            "/".join(sorted(coop_targets)) if coop_targets else "-",
            "; ".join(assignment_summary) if assignment_summary else "-",
            "; ".join(stable_summary[:4]) if stable_summary else "-",
            "/".join(sorted(self._stable_tracking_ready_announced)) if self._stable_tracking_ready_announced else "-",
        )
