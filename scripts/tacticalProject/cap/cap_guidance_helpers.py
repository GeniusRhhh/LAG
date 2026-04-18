"""Extracted guidance/relay helpers for CAPTask."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

from envs.JSBSim.core.catalog import Catalog as c

try:
    from .cap_state_machine import CAPState
except ImportError:
    from cap_state_machine import CAPState


log = logging.getLogger(__name__)


def _update_stable_tracking_window(self, track_qualities: Dict[str, Dict[str, float]], current_time: float):
    """Track continuous high-quality dual-ship tracking time per target."""
    if not track_qualities and not hasattr(self, "cap_radar"):
        return

    def _clear_stable_target(target_id: str, reason: str) -> None:
        self._stable_tracking_elapsed[target_id] = 0.0
        self._stable_tracking_trackers.pop(target_id, None)
        self._stable_tracking_last_ok.pop(target_id, None)
        if target_id in self._stable_tracking_ready_announced:
            self._stable_tracking_ready_announced.discard(target_id)
            log.info("[协同跟踪重置] target=%s reason=%s", target_id, reason)

    target_trackers: Dict[str, list[str]] = {}
    suppressed_friendlies = set()
    suppress_store = getattr(self, "_defensive_engagement_suppressed_until", None)
    if isinstance(suppress_store, dict):
        current_step = int(getattr(self, "step_count", round(float(current_time) / 0.2)))
        for aid, until_step in list(suppress_store.items()):
            try:
                until_step = int(until_step)
            except Exception:
                until_step = -1
            if until_step > current_step:
                suppressed_friendlies.add(str(aid))

    all_friendlies = [aid for aid in getattr(track_qualities, "keys", lambda: [])() if str(aid).startswith("A")]
    if hasattr(self, "cap_radar"):
        for aid in getattr(getattr(self, "cap_radar", None), "_radar_tracks", {}).keys():
            if str(aid).startswith("A") and aid not in all_friendlies:
                all_friendlies.append(aid)

    for aid in all_friendlies:
        if not str(aid).startswith("A") or str(aid) in suppressed_friendlies:
            continue
        qualities = track_qualities.get(aid, {}) if isinstance(track_qualities, dict) else {}
        radar_tracks = self.cap_radar.get_tracks(aid) if hasattr(self, "cap_radar") else {}
        lock_tid = self.cap_radar.get_lock_target(aid) if hasattr(self, "cap_radar") else None

        candidate_tids = set(qualities.keys()) | set(radar_tracks.keys())
        if lock_tid:
            candidate_tids.add(lock_tid)

        for tid in candidate_tids:
            if hasattr(self, "_get_picture_target_actionability"):
                target_status = self._get_picture_target_actionability(
                    tid,
                    current_time=current_time,
                    purpose="decision",
                )
                if not bool(target_status.get("actionable", True)):
                    continue
            q = float(qualities.get(tid, 0.0))
            tracked_by_fcr = tid in radar_tracks
            locked = lock_tid == tid
            if tracked_by_fcr and (locked or q >= self._stable_tracking_min_quality):
                target_trackers.setdefault(tid, []).append(aid)

    self._stable_tracking_trackers = {
        tid: sorted(list(trackers))
        for tid, trackers in target_trackers.items()
        if len(trackers) >= 2
    }

    tracked_target_ids = (
        set(target_trackers.keys())
        | set(getattr(self, "_stable_tracking_elapsed", {}).keys())
        | set(getattr(self, "_stable_tracking_ready_announced", set()))
    )
    for tid in list(tracked_target_ids):
        if not hasattr(self, "_get_picture_target_actionability"):
            continue
        target_status = self._get_picture_target_actionability(
            tid,
            current_time=current_time,
            purpose="decision",
        )
        if not bool(target_status.get("actionable", True)):
            reason_text = ",".join(target_status.get("reasons", [])) or "target_filtered"
            target_trackers.pop(tid, None)
            self._stable_tracking_trackers.pop(tid, None)
            _clear_stable_target(tid, f"target_filtered:{reason_text}")

    dt = 0.2
    for tid in list(self._stable_tracking_elapsed.keys()):
        if tid in target_trackers and len(target_trackers.get(tid, [])) >= 2:
            continue
        last_ok = float(self._stable_tracking_last_ok.get(tid, -1e9))
        if float(current_time) - last_ok > float(self._stable_tracking_grace_s):
            _clear_stable_target(tid, "grace_timeout")

    for tid, trackers in target_trackers.items():
        if len(trackers) >= 2:
            self._stable_tracking_elapsed[tid] = self._stable_tracking_elapsed.get(tid, 0.0) + dt
            self._stable_tracking_last_ok[tid] = current_time
            if (
                self._stable_tracking_elapsed.get(tid, 0.0) >= self._stable_tracking_window_s
                and tid not in self._stable_tracking_ready_announced
            ):
                self._stable_tracking_ready_announced.add(tid)
                if hasattr(self, "_target_first_ready_time"):
                    self._target_first_ready_time.setdefault(tid, float(current_time))
                ready_trackers = self._stable_tracking_trackers.get(tid, sorted(list(trackers)))
                log.info(
                    f"[协同跟踪就绪] target={tid} stable={self._stable_tracking_elapsed.get(tid, 0.0):.1f}s "
                    f"trackers={ready_trackers}"
                )
        else:
            _clear_stable_target(tid, "tracker_drop")


def get_guidance_verification_snapshot(self) -> Dict[str, object]:
    gv = dict(self._guidance_verify)

    pass_count = int(gv.get("prelaunch_gate_pass_count", 0))
    block_count = int(gv.get("prelaunch_gate_block_count", 0))
    gate_total = pass_count + block_count
    gv["prelaunch_gate_total_requests"] = gate_total
    gv["prelaunch_gate_pass_rate"] = (float(pass_count) / float(gate_total)) if gate_total > 0 else 0.0

    relay_attempt = int(gv.get("relay_attempt_count", 0))
    relay_success = int(gv.get("relay_success_count", 0))
    gv["relay_success_rate"] = (float(relay_success) / float(relay_attempt)) if relay_attempt > 0 else 0.0

    gv["stable_tracking_window_s"] = float(self._stable_tracking_window_s)
    gv["stable_tracking_min_quality"] = float(self._stable_tracking_min_quality)
    gv["stable_tracking_ready_targets"] = sorted(list(self._stable_tracking_ready_announced))
    gv["stable_tracking_trackers"] = {k: list(v) for k, v in self._stable_tracking_trackers.items()}
    gv["active_relay"] = bool(self._relay_guidance_active)
    return gv


def record_external_launch_event(
    self,
    shooter_id: str,
    target_id: str,
    missile_id: str,
    current_time: float,
    source: str = "external",
) -> None:
    if not target_id:
        return

    missile_adapter = getattr(self, "missile_adapter", None)
    if missile_adapter is not None and hasattr(missile_adapter, "sync_external_launch_event"):
        distance_km = 0.0
        env = getattr(self, "env", None)

        def _resolve_actor(actor_id: str):
            for collection_name in ("agents", "_jsbsims"):
                collection = getattr(env, collection_name, None)
                if isinstance(collection, dict) and actor_id in collection:
                    return collection.get(actor_id)
            return None

        try:
            shooter = _resolve_actor(str(shooter_id))
            target = _resolve_actor(str(target_id))
            if shooter is not None and target is not None:
                shooter_pos = np.asarray(shooter.get_position(), dtype=float)
                target_pos = np.asarray(target.get_position(), dtype=float)
                distance_km = float(np.linalg.norm(target_pos - shooter_pos)) / 1000.0
        except Exception:
            distance_km = 0.0

        try:
            synced = missile_adapter.sync_external_launch_event(
                shooter_id=str(shooter_id),
                target_id=str(target_id),
                missile_id=str(missile_id),
                current_time=float(current_time),
                distance_km=float(distance_km),
            )
            missile_obj = None
            if env is not None and hasattr(env, "missiles") and isinstance(env.missiles, dict):
                missile_obj = env.missiles.get(str(missile_id))
            if missile_obj is not None and hasattr(missile_adapter, "register_external_missile"):
                missile_adapter.register_external_missile(
                    missile_id=str(missile_id),
                    missile_obj=missile_obj,
                    shooter_id=str(shooter_id),
                    target_id=str(target_id),
                    env=env,
                )
            if synced:
                log.info(
                    "[MISSILE_SYNC_EXT] shooter=%s target=%s missile=%s source=%s dist=%.1fkm",
                    shooter_id,
                    target_id,
                    missile_id,
                    source,
                    distance_km,
                )
        except Exception:
            pass

    self._guidance_verify["prelaunch_gate_pass_count"] += 1
    pass_by_target = self._guidance_verify["prelaunch_gate_pass_by_target"]
    pass_by_target[target_id] = int(pass_by_target.get(target_id, 0)) + 1
    first_pass = self._guidance_verify["prelaunch_gate_first_pass_time_by_target"]
    if target_id not in first_pass:
        first_pass[target_id] = float(current_time)
    log.info(f"[发射门禁通过] shooter={shooter_id} target={target_id} missile={missile_id} source={source}")


def record_external_gate_block(
    self,
    shooter_id: str,
    target_id: str,
    current_time: float,
    reason: str = "external",
) -> None:
    if not target_id:
        return
    key = (shooter_id, target_id, reason)
    last_t = float(self._external_gate_block_last_time.get(key, -1e9))
    if float(current_time) - last_t < 2.0:
        return
    self._external_gate_block_last_time[key] = float(current_time)

    self._guidance_verify["prelaunch_gate_block_count"] += 1
    block_by_target = self._guidance_verify["prelaunch_gate_block_by_target"]
    block_by_target[target_id] = int(block_by_target.get(target_id, 0)) + 1
    log.info(f"[发射门禁阻断] shooter={shooter_id} target={target_id} reason={reason}")

    missile_manager = getattr(self, "missile_manager", None)
    if missile_manager is None or not hasattr(missile_manager, "last_gate_decision"):
        return
    gate_decision = missile_manager.last_gate_decision.get(str(shooter_id), {})
    if not isinstance(gate_decision, dict):
        return
    if str(gate_decision.get("reason")) != str(reason):
        return
    decision_target = str(gate_decision.get("target_id", "") or "")
    if decision_target and decision_target != str(target_id):
        return

    details = gate_decision.get("details", {})
    if not isinstance(details, dict):
        return
    summary_parts = []
    phase = gate_decision.get("phase")
    if phase:
        summary_parts.append(f"phase={phase}")
    distance_km = details.get("distance_km", details.get("璺濈_km"))
    if distance_km is not None:
        try:
            summary_parts.append(f"dist={float(distance_km):.1f}km")
        except Exception:
            pass
    heading_error_deg = details.get("heading_error_deg", details.get("鑸悜璇樊_deg"))
    if heading_error_deg is not None:
        try:
            summary_parts.append(f"hdg_err={float(heading_error_deg):.1f}deg")
        except Exception:
            pass
    zone = details.get("target_risk_zone")
    if zone:
        summary_parts.append(f"zone={zone}")
    active_on_target = details.get("active_on_target")
    effective_active_on_target = details.get("effective_active_on_target")
    if active_on_target is not None:
        try:
            active_desc = int(effective_active_on_target) if effective_active_on_target is not None else int(active_on_target)
            summary_parts.append(f"active={active_desc}")
        except Exception:
            pass
    radar_reason = details.get("radar_check_reason")
    if radar_reason:
        summary_parts.append(f"radar={radar_reason}")
    coop_stable_seconds = details.get("coop_stable_seconds")
    coop_tracker_count = details.get("coop_tracker_count")
    if coop_stable_seconds is not None:
        try:
            summary_parts.append(f"coop={float(coop_stable_seconds):.1f}s/{int(coop_tracker_count or 0)}trk")
        except Exception:
            pass

    if not summary_parts:
        return
    if not hasattr(self, "_external_gate_block_detail_state"):
        self._external_gate_block_detail_state = {}
    detail_key = (str(shooter_id), str(target_id), str(reason))
    detail_summary = " ".join(summary_parts)
    last_summary, last_detail_t = self._external_gate_block_detail_state.get(detail_key, ("", -1e9))
    if detail_summary == last_summary and (float(current_time) - float(last_detail_t)) < 6.0:
        return
    self._external_gate_block_detail_state[detail_key] = (detail_summary, float(current_time))
    log.info(
        "[PRELAUNCH_GATE_DETAIL] shooter=%s target=%s reason=%s %s",
        shooter_id,
        target_id,
        reason,
        detail_summary,
    )


def _has_stable_dual_tracking(self, target_id: str) -> bool:
    return self._stable_tracking_elapsed.get(target_id, 0.0) >= self._stable_tracking_window_s


def check_external_prelaunch_gate(self, target_id: str, current_time: float) -> Tuple[bool, Dict[str, object]]:
    target = str(target_id or "").strip()
    if not target:
        return False, {
            "target_id": target,
            "stable_seconds": 0.0,
            "required_seconds": float(self._stable_tracking_window_s),
            "trackers": [],
            "min_quality": float(self._stable_tracking_min_quality),
            "current_time": float(current_time),
            "reason": "invalid_target",
        }

    stable_seconds = float(self._stable_tracking_elapsed.get(target, 0.0))
    trackers = list(self._stable_tracking_trackers.get(target, []))
    last_ok = float(self._stable_tracking_last_ok.get(target, -1e9))
    in_grace = (float(current_time) - last_ok) <= float(self._stable_tracking_grace_s)
    passed = self._has_stable_dual_tracking(target) and (len(trackers) >= 2 or in_grace)

    return passed, {
        "target_id": target,
        "stable_seconds": stable_seconds,
        "required_seconds": float(self._stable_tracking_window_s),
        "trackers": trackers,
        "min_quality": float(self._stable_tracking_min_quality),
        "grace_seconds": float(self._stable_tracking_grace_s),
        "current_time": float(current_time),
        "reason": "ok" if passed else "stable_dual_track_not_ready",
    }


def _is_agent_in_defensive_maneuver(self, env, agent_id: str) -> bool:
    aircraft = env.agents.get(agent_id)
    if aircraft is None or not aircraft.is_alive:
        return True

    try:
        if getattr(aircraft, "check_missile_warning", lambda: None)() is not None:
            return True
    except Exception:
        pass

    try:
        roll_deg = abs(np.degrees(float(aircraft.get_property_value(c.attitude_phi_rad))))
        if roll_deg >= 55.0:
            return True
    except Exception:
        pass

    return False


def _update_relay_guidance(self, env, current_time: float):
    active_missiles = self.missile_adapter.get_active_missiles()
    if not active_missiles:
        return

    for missile in active_missiles:
        missile_state = str(getattr(getattr(missile, "state", None), "value", ""))
        terminal_guidance = missile_state == "terminal"

        missile_id = missile.missile_id
        guide_id = missile.guide_agent_id
        target_id = missile.target_id
        if not missile_id or not guide_id or not target_id:
            continue
        if not str(guide_id).startswith("A"):
            continue

        last_relay_t = self._relay_last_time.get(missile_id, -999.0)
        relay_cooldown_s = 4.0 if terminal_guidance else 6.0
        if current_time - last_relay_t < relay_cooldown_s:
            continue

        guide_aircraft = env.agents.get(guide_id)
        if guide_aircraft is None:
            continue

        guide_alive = bool(guide_aircraft.is_alive)
        if not guide_alive:
            relay_reason = "guide_destroyed"
        else:
            guide_pos = np.array(self._get_battlefield_pos(env, guide_id), dtype=float)
            if target_id in env.agents:
                target_pos = np.array(self._get_battlefield_pos(env, target_id), dtype=float)
                guide_distance = float(np.linalg.norm(guide_pos - target_pos))
            else:
                guide_distance = float("inf")

            q = float(self.cap_radar.get_track_quality(guide_id, target_id))
            missile_distance = float(getattr(missile, "distance_to_target", 999.0))
            missile_tof = float(getattr(missile, "time_of_flight", 0.0))
            needs_relay = self.missile_adapter.needs_relay(missile_id, guide_alive, guide_distance)
            defensive = self._is_agent_in_defensive_maneuver(env, guide_id)
            defensive = bool(
                defensive and (
                    terminal_guidance
                    or missile_distance <= 55.0
                    or missile_tof >= 20.0
                )
            )
            low_quality = bool(
                q < (0.55 if terminal_guidance else 0.45)
                and (
                    terminal_guidance
                    or missile_distance <= 45.0
                    or missile_tof >= 25.0
                )
            )

            if not (needs_relay or defensive or low_quality):
                continue

            if terminal_guidance and (needs_relay or defensive or low_quality):
                relay_reason = "terminal_illumination"
            elif needs_relay:
                relay_reason = "range_or_survivability"
            elif defensive:
                relay_reason = "guide_maneuvering"
            else:
                relay_reason = "low_track_quality"

        relay_id = self._select_relay_candidate(
            env,
            exclude_agent=guide_id,
            target_id=target_id,
            terminal_priority=terminal_guidance,
        )
        self._guidance_verify["relay_attempt_count"] += 1
        if not relay_id:
            self._guidance_verify["relay_fail_count"] += 1
            if not hasattr(self, "_relay_skip_last_time"):
                self._relay_skip_last_time = {}
            last_log_t = float(self._relay_skip_last_time.get(missile_id, -9999.0))
            if current_time - last_log_t >= 10.0:
                self._relay_skip_last_time[missile_id] = current_time
                log.info(f"⚠️ [RELAY_SKIP] missile={missile_id} guide={guide_id} target={target_id} reason={relay_reason} no_candidate")
            continue

        relay_q = float(self.cap_radar.get_track_quality(relay_id, target_id))
        q_margin = 0.08 if terminal_guidance else 0.12
        if relay_reason not in ("guide_destroyed", "range_or_survivability", "terminal_illumination") and relay_q < (q + q_margin):
            self._guidance_verify["relay_fail_count"] += 1
            continue

        if self.missile_adapter.apply_relay_guidance(missile_id, relay_id, env=env):
            self._relay_last_time[missile_id] = current_time
            self._relay_guidance_active = True
            self._guidance_verify["relay_success_count"] += 1
            reason_map = self._guidance_verify["relay_success_by_reason"]
            reason_map[relay_reason] = int(reason_map.get(relay_reason, 0)) + 1
            event_key = (guide_id, relay_id, target_id, relay_reason)
            last_event_key = self._relay_event_last_key.get(missile_id)
            last_event_t = float(self._relay_event_last_time.get(missile_id, -9999.0))
            if event_key != last_event_key or (current_time - last_event_t) >= 4.0:
                self._relay_event_last_key[missile_id] = event_key
                self._relay_event_last_time[missile_id] = float(current_time)
                log.info(
                    "🔄 [接力制导] missile=%s target=%s %s->%s reason=%s guide_q=%.2f relay_q=%.2f",
                    missile_id,
                    target_id,
                    guide_id,
                    relay_id,
                    relay_reason,
                    q if guide_alive else 0.0,
                    relay_q,
                )
        else:
            self._guidance_verify["relay_fail_count"] += 1
            if (current_time - float(self._relay_event_last_time.get(f"fail:{missile_id}", -9999.0))) >= 6.0:
                self._relay_event_last_time[f"fail:{missile_id}"] = float(current_time)
                log.info(
                    "❌ [RELAY_FAIL] missile=%s from=%s to=%s target=%s reason=%s",
                    missile_id,
                    guide_id,
                    relay_id,
                    target_id,
                    relay_reason,
                )


def _check_relay_guidance(self, env, agent_id: str, distance: float):
    del agent_id, distance
    self._update_relay_guidance(env, self.step_count * 0.2)


def _select_relay_candidate(
    self,
    env,
    exclude_agent: str,
    target_id: str,
    terminal_priority: bool = False,
) -> Optional[str]:
    best_agent = None
    best_score = -1.0
    fallback_agent = None
    fallback_score = -1.0
    terminal_best_agent = None
    terminal_best_score = -1.0
    nearest_supported_agent = None
    nearest_supported_distance = float("inf")
    defensive_fallback_agent = None
    defensive_fallback_score = -1.0
    target_pos = None
    if target_id in env.agents:
        try:
            target_pos = np.array(self._get_battlefield_pos(env, target_id), dtype=float)
        except Exception:
            target_pos = None
    for aid in env.agents:
        if not str(aid).startswith("A") or aid == exclude_agent:
            continue
        if not env.agents[aid].is_alive:
            continue

        q = self.cap_radar.get_track_quality(aid, target_id)
        defensive = self._is_agent_in_defensive_maneuver(env, aid)
        aid_distance = float("inf")
        if target_pos is not None:
            try:
                aid_pos = np.array(self._get_battlefield_pos(env, aid), dtype=float)
                aid_distance = float(np.linalg.norm(aid_pos - target_pos))
            except Exception:
                aid_distance = float("inf")
        if defensive:
            if q >= 0.15 and q > defensive_fallback_score:
                defensive_fallback_score = q
                defensive_fallback_agent = aid
            continue
        score = q * 100.0 - min(aid_distance, 120.0) * 0.35
        if terminal_priority and q >= 0.45 and score > terminal_best_score:
            terminal_best_score = score
            terminal_best_agent = aid
            continue
        if q >= 0.40 and score > best_score:
            best_score = score
            best_agent = aid
            continue
        if q >= 0.22 and score > fallback_score:
            fallback_score = score
            fallback_agent = aid
        if q >= 0.15 and aid_distance < nearest_supported_distance:
            nearest_supported_distance = aid_distance
            nearest_supported_agent = aid
    if terminal_priority:
        return terminal_best_agent or best_agent or fallback_agent or nearest_supported_agent or defensive_fallback_agent
    return best_agent or fallback_agent or nearest_supported_agent or defensive_fallback_agent
