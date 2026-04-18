"""CAP tracking, intent, and missile update helpers."""

import logging
import numpy as np


log = logging.getLogger(__name__)


def _update_intent_recognition(self, env, current_time: float):
    """更新意图识别（在 NLT 节点 180km 触发）。"""
    tracks = []
    for t in self._get_picture_threats(current_time=current_time, purpose="decision"):
        center_x, center_y = 100.0, 50.0
        distance = np.sqrt((t.x - center_x) ** 2 + (t.y - center_y) ** 2)
        tracks.append(
            {
                "track_id": t.track_id,
                "position": [t.x, t.y, t.altitude],
                "velocity": [0, 0, 0],
                "heading": t.heading,
                "distance": distance,
            }
        )

    if not tracks:
        return

    my_aircraft = next(
        (
            env.agents[aid]
            for aid in ("A0100", "A0200", "A0300", "A0400")
            if aid in env.agents and env.agents[aid].is_alive
        ),
        None,
    )
    self.intent_adapter.analyze_intent(tracks, env=env, my_aircraft=my_aircraft)

    highest = self.intent_adapter.get_highest_threat()
    if highest and highest.threat_level.value in ("high", "critical"):
        current_step = int(getattr(env, "current_step", 0) or 0)
        dt = float(getattr(env, "time_interval", 0.2) or 0.2)
        min_change_steps = max(1, int(round(12.0 / max(dt, 1e-6))))
        ready_now = self.intent_adapter._is_model_ready(highest.model_window)
        signature = (
            highest.track_id,
            highest.intent.value,
            highest.threat_level.value,
            highest.pred_label or "",
            ready_now,
        )
        prev_signature = getattr(self, "_last_high_threat_signature", None)
        prev_step = int(getattr(self, "_last_high_threat_step", -10**9))
        ready_transition = bool(prev_signature) and bool(prev_signature[-1] is False and ready_now is True)
        if prev_signature != signature and (ready_transition or (current_step - prev_step) >= min_change_steps):
            self._last_high_threat_signature = signature
            self._last_high_threat_step = current_step
            conf_text = f"{highest.confidence:.3f}" if highest.confidence > 0.0 else "0.000"
            window_text = highest.model_window or "-"
            label_text = highest.pred_label or highest.intent.value
            log.info(
                f"[意图高威胁] target={highest.track_id} intent={highest.intent.value} "
                f"label={label_text} threat={highest.threat_level.value} conf={conf_text} "
                f"window={window_text} dist={highest.distance_km:.1f}km"
            )


def _update_cooperative_tracking(self, env, current_time: float):
    """更新协同跟踪（在 MTR 节点 120km 触发）。"""
    center_pos = np.array([100.0, 50.0, 8.0])
    nearest = self._get_picture_nearest_threat(center_pos, current_time=current_time, purpose="decision")
    if not nearest:
        self._cooperative_tracking_active = False
        return

    target_id = nearest.track_id

    available_trackers = []
    for aid in env.agents:
        if aid.startswith("A") and env.agents[aid].is_alive:
            tracks = self.cap_radar.get_tracks(aid)
            if target_id in tracks:
                quality = self.cap_radar.get_track_quality(aid, target_id)
                available_trackers.append((aid, quality))

    available_trackers.sort(key=lambda x: x[1], reverse=True)
    selected = [t[0] for t in available_trackers[:2]]

    if len(selected) >= 2:
        if not self._cooperative_tracking_active:
            log.info(f"[协同跟踪启动] 目标:{target_id} 跟踪机:{selected}")
        self._cooperative_tracking_active = True
        self._tracking_targets[target_id] = selected
        for aid in selected:
            self.cap_radar.lock_target(aid, target_id)
    else:
        self._cooperative_tracking_active = False


def _update_missiles(self, env, current_time: float):
    """更新导弹状态。"""
    target_positions = {}
    for t in self._get_picture_threats(current_time=current_time, purpose="decision"):
        target_positions[t.track_id] = (t.x, t.y)

    self.missile_adapter.update(current_time, target_positions, env)
    self._update_relay_guidance(env, current_time)

    active_guided = 0
    active_missiles = self.missile_adapter.get_active_missiles()
    for m in active_missiles:
        gid = getattr(m, "guide_agent_id", None)
        if gid and str(gid).startswith("A"):
            active_guided += 1

    try:
        env_temp = getattr(env, "_tempsims", {}) or {}
        for m in env_temp.values():
            if not getattr(m, "is_alive", False):
                continue
            gid = getattr(m, "guide_agent_id", None)
            if gid and str(gid).startswith("A"):
                active_guided += 1
    except Exception:
        pass

    self._guidance_verify["active_guided_missile_samples"] += int(active_guided)
    self._guidance_verify["active_guided_missile_peak"] = max(
        int(self._guidance_verify.get("active_guided_missile_peak", 0)),
        int(active_guided),
    )

    if self.step_count % 50 == 0 and active_guided > 0:
        details = []
        for m in active_missiles:
            gid = getattr(m, "guide_agent_id", None)
            tid = getattr(m, "target_id", None)
            if gid and str(gid).startswith("A"):
                details.append(f"{m.missile_id}:{gid}->{tid}")
        if details:
            log.info(f"[导引链] active={active_guided} links={'; '.join(details[:4])}")


def _start_cooperative_tracking(self, env, target_id: str):
    """启动协同跟踪（在 MTR 节点调用）。"""
    trackers = []
    for aid in env.agents:
        if aid.startswith("A") and env.agents[aid].is_alive:
            trackers.append(aid)
            if len(trackers) >= 2:
                break

    if len(trackers) >= 2:
        self._tracking_targets[target_id] = trackers
        for aid in trackers:
            self.cap_radar.lock_target(aid, target_id)
        log.info(f"[协同跟踪启动] 目标:{target_id} | 跟踪机:{trackers}")
        return True
    return False
