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


def _increment_counter(counter_map: Dict[str, object], key: str, delta: int = 1) -> None:
    counter_map[key] = int(counter_map.get(key, 0) or 0) + int(delta)


def _normalize_gate_reason(reason: Optional[str]) -> str:
    text = str(reason or "").strip().lower()
    if not text:
        return "other"
    if "stable" in text and "ready" in text:
        return "stable_not_ready"
    if "friendly" in text or "safe" in text or "close_shot" in text:
        return "friendly_safe"
    if "long_shot" in text or "longshot" in text:
        return "long_shot"
    if "launch_exec_fail" in text or "launch_execute_fail" in text or "missile_execute_fail" in text:
        return "launch_exec_fail"
    return "other"


def _apply_gate_reason_bucket(self, normalized_reason: str) -> None:
    if normalized_reason == "stable_not_ready":
        _increment_counter(self._guidance_verify, "prelaunch_gate_block_stable_not_ready_count")
    elif normalized_reason == "friendly_safe":
        _increment_counter(self._guidance_verify, "prelaunch_gate_block_friendly_safe_count")
    elif normalized_reason == "long_shot":
        _increment_counter(self._guidance_verify, "prelaunch_gate_block_long_shot_count")
    elif normalized_reason == "launch_exec_fail":
        _increment_counter(self._guidance_verify, "prelaunch_gate_block_launch_exec_fail_count")
    else:
        _increment_counter(self._guidance_verify, "prelaunch_gate_block_other_count")


def _record_prelaunch_gate_event(
    self,
    shooter_id: str,
    target_id: str,
    current_time: float,
    *,
    event_type: str,
    stable_ready: bool,
    reason: str = "",
) -> None:
    if not target_id:
        return

    shooter_id = str(shooter_id or "")
    target_id = str(target_id or "")
    unique_key = (shooter_id, target_id)
    unique_window_s = float(getattr(self, "_stable_tracking_unique_window_s", self._stable_tracking_window_s))

    if event_type == "pass":
        _increment_counter(self._guidance_verify, "prelaunch_gate_pass_count")
        last_unique_pass_t = float(getattr(self, "_external_gate_unique_pass_last_time", {}).get(unique_key, -1e9))
        if float(current_time) - last_unique_pass_t >= unique_window_s:
            _increment_counter(self._guidance_verify, "prelaunch_gate_unique_pass_count")
            self._external_gate_unique_pass_last_time[unique_key] = float(current_time)

        if stable_ready:
            _increment_counter(self._guidance_verify, "prelaunch_gate_request_after_ready_count")
            _increment_counter(self._guidance_verify, "prelaunch_gate_ready_pass_count")
            last_ready_req_t = float(getattr(self, "_external_gate_ready_unique_request_last_time", {}).get(unique_key, -1e9))
            if float(current_time) - last_ready_req_t >= unique_window_s:
                _increment_counter(self._guidance_verify, "prelaunch_gate_request_after_ready_unique_count")
                self._external_gate_ready_unique_request_last_time[unique_key] = float(current_time)
            last_ready_pass_t = float(getattr(self, "_external_gate_ready_unique_pass_last_time", {}).get(unique_key, -1e9))
            if float(current_time) - last_ready_pass_t >= unique_window_s:
                _increment_counter(self._guidance_verify, "prelaunch_gate_ready_unique_pass_count")
                self._external_gate_ready_unique_pass_last_time[unique_key] = float(current_time)
    elif event_type == "block":
        _increment_counter(self._guidance_verify, "prelaunch_gate_block_count")
        last_unique_block_t = float(getattr(self, "_external_gate_unique_block_last_time", {}).get(unique_key, -1e9))
        if float(current_time) - last_unique_block_t >= unique_window_s:
            _increment_counter(self._guidance_verify, "prelaunch_gate_unique_block_count")
            self._external_gate_unique_block_last_time[unique_key] = float(current_time)

        normalized_reason = _normalize_gate_reason(reason)
        _apply_gate_reason_bucket(self, normalized_reason)
        if stable_ready:
            _increment_counter(self._guidance_verify, "prelaunch_gate_request_after_ready_count")
            _increment_counter(self._guidance_verify, "prelaunch_gate_ready_block_count")
            last_ready_req_t = float(getattr(self, "_external_gate_ready_unique_request_last_time", {}).get(unique_key, -1e9))
            if float(current_time) - last_ready_req_t >= unique_window_s:
                _increment_counter(self._guidance_verify, "prelaunch_gate_request_after_ready_unique_count")
                self._external_gate_ready_unique_request_last_time[unique_key] = float(current_time)
            last_ready_block_t = float(getattr(self, "_external_gate_ready_unique_block_last_time", {}).get(unique_key, -1e9))
            if float(current_time) - last_ready_block_t >= unique_window_s:
                _increment_counter(self._guidance_verify, "prelaunch_gate_ready_unique_block_count")
                self._external_gate_ready_unique_block_last_time[unique_key] = float(current_time)


def _infer_prelaunch_phase_name(self, distance_km: Optional[float], is_second_attack: bool = False) -> str:
    if distance_km is None or not np.isfinite(float(distance_km)):
        return ""

    ranges = getattr(self, "ranges", None)
    if is_second_attack and ranges is not None:
        lr_prime = float(getattr(ranges, "LR_PRIME", 53.0))
        tr_prime = float(getattr(ranges, "TR_PRIME", 50.0))
        mar = float(getattr(ranges, "MAR", 40.0))
        if distance_km >= lr_prime:
            return "LR_PRIME"
        if distance_km >= tr_prime:
            return "TR_PRIME"
        if distance_km >= mar:
            return "DR_MAR"
        return "BELOW_MAR"

    if ranges is not None:
        lr = float(getattr(ranges, "LR", 78.0))
        tr = float(getattr(ranges, "TR", 75.0))
        dor = float(getattr(ranges, "DOR", 70.0))
        dr = float(getattr(ranges, "DR", 65.0))
        mar = float(getattr(ranges, "MAR", 40.0))
        if distance_km >= lr:
            return "MTR_LR"
        if distance_km >= tr:
            return "LR_TR"
        if distance_km >= dor:
            return "TR_DOR"
        if distance_km >= dr:
            return "DOR_DR"
        if distance_km >= mar:
            return "DR_MAR"
        return "BELOW_MAR"

    return ""


def _get_prelaunch_gate_profile(
    self,
    target_id: str,
    current_time: float,
    shooter_id: Optional[str] = None,
    distance_km: Optional[float] = None,
    phase_name: str = "",
    is_second_attack: bool = False,
    target_zone: Optional[str] = None,
) -> Dict[str, object]:
    base_required = float(getattr(self, "_stable_tracking_window_s", 5.0))
    min_quality = float(getattr(self, "_stable_tracking_min_quality", 0.45))
    zone = str(target_zone or "UNKNOWN").upper()
    phase = str(phase_name or "").strip().upper()
    if not phase:
        phase = _infer_prelaunch_phase_name(self, distance_km, is_second_attack=is_second_attack)
    return {
        "target_id": str(target_id or ""),
        "required_seconds": base_required,
        "min_quality": min_quality,
        "phase_name": phase,
        "target_zone": zone,
        "distance_km": None if distance_km is None or not np.isfinite(float(distance_km)) else float(distance_km),
        "is_second_attack": bool(is_second_attack),
    }


def _update_stable_tracking_window(self, track_qualities: Dict[str, Dict[str, float]], current_time: float):
    """Track continuous high-quality stable tracking time per target."""
    if not track_qualities and not hasattr(self, "cap_radar"):
        return
    grace_s = float(getattr(self, "_stable_tracking_grace_s", 2.5))
    holdover_s = max(0.0, min(
        grace_s,
        float(getattr(self, "_stable_tracking_holdover_s", 0.6)),
    ))

    def _clear_stable_target(target_id: str, reason: str) -> None:
        self._stable_tracking_elapsed[target_id] = 0.0
        self._stable_tracking_trackers.pop(target_id, None)
        self._stable_tracking_last_ok.pop(target_id, None)
        if target_id in self._stable_tracking_ready_announced:
            self._stable_tracking_ready_announced.discard(target_id)
            log.info("[协同跟踪重置] target=%s reason=%s", target_id, reason)

    target_trackers: Dict[str, list[str]] = {}

    all_friendlies = [aid for aid in getattr(track_qualities, "keys", lambda: [])() if str(aid).startswith("A")]
    if hasattr(self, "cap_radar"):
        for aid in getattr(getattr(self, "cap_radar", None), "_radar_tracks", {}).keys():
            if str(aid).startswith("A") and aid not in all_friendlies:
                all_friendlies.append(aid)

    for aid in all_friendlies:
        if not str(aid).startswith("A"):
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
            if hasattr(self, "_is_agent_target_pair_compatible") and not self._is_agent_target_pair_compatible(
                aid,
                tid,
                current_time=current_time,
            ):
                continue
            q = float(qualities.get(tid, 0.0))
            tracked_by_fcr = tid in radar_tracks
            locked = lock_tid == tid
            if tracked_by_fcr and (locked or q >= self._stable_tracking_min_quality):
                target_trackers.setdefault(tid, []).append(aid)

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

    previous_trackers = {
        tid: list(trackers)
        for tid, trackers in getattr(self, "_stable_tracking_trackers", {}).items()
    }
    updated_trackers: Dict[str, list[str]] = {}
    for tid, trackers in target_trackers.items():
        if len(trackers) >= 1:
            updated_trackers[tid] = sorted(list(trackers))
            continue
        last_ok = float(self._stable_tracking_last_ok.get(tid, -1e9))
        if tid in previous_trackers and (float(current_time) - last_ok) <= grace_s:
            updated_trackers[tid] = list(previous_trackers[tid])
    for tid, trackers in previous_trackers.items():
        if tid in updated_trackers:
            continue
        last_ok = float(self._stable_tracking_last_ok.get(tid, -1e9))
        if (float(current_time) - last_ok) <= grace_s:
            updated_trackers[tid] = list(trackers)
    self._stable_tracking_trackers = updated_trackers

    dt = 0.2
    for tid in list(self._stable_tracking_elapsed.keys()):
        if tid in target_trackers and len(target_trackers.get(tid, [])) >= 1:
            continue
        last_ok = float(self._stable_tracking_last_ok.get(tid, -1e9))
        if float(current_time) - last_ok > grace_s:
            _clear_stable_target(tid, "grace_timeout")

    for tid, trackers in target_trackers.items():
        if len(trackers) >= 1:
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
        elif tid in self._stable_tracking_elapsed:
            last_ok = float(self._stable_tracking_last_ok.get(tid, -1e9))
            if (float(current_time) - last_ok) <= holdover_s:
                self._stable_tracking_elapsed[tid] = min(
                    float(self._stable_tracking_window_s),
                    self._stable_tracking_elapsed.get(tid, 0.0) + dt,
                )
                if (
                    self._stable_tracking_elapsed.get(tid, 0.0) >= self._stable_tracking_window_s
                    and tid not in self._stable_tracking_ready_announced
                ):
                    self._stable_tracking_ready_announced.add(tid)
                    if hasattr(self, "_target_first_ready_time"):
                        self._target_first_ready_time.setdefault(tid, float(current_time))
                    ready_trackers = self._stable_tracking_trackers.get(tid, [])
                    log.info(
                        f"[协同跟踪就绪] target={tid} stable={self._stable_tracking_elapsed.get(tid, 0.0):.1f}s "
                        f"trackers={ready_trackers}"
                    )


def get_guidance_verification_snapshot(self) -> Dict[str, object]:
    gv = dict(self._guidance_verify)

    pass_count = int(gv.get("prelaunch_gate_pass_count", 0))
    block_count = int(gv.get("prelaunch_gate_block_count", 0))
    raw_gate_total = pass_count + block_count
    gv["prelaunch_gate_raw_total_requests"] = raw_gate_total
    gv["prelaunch_gate_raw_pass_rate"] = (
        (float(pass_count) / float(raw_gate_total)) if raw_gate_total > 0 else 0.0
    )
    unique_pass_count = int(gv.get("prelaunch_gate_unique_pass_count", 0))
    unique_block_count = int(gv.get("prelaunch_gate_unique_block_count", 0))
    unique_total = unique_pass_count + unique_block_count
    gv["prelaunch_gate_unique_total_requests"] = unique_total
    gv["prelaunch_gate_unique_pass_rate"] = (
        float(unique_pass_count) / float(unique_total)
        if unique_total > 0 else 0.0
    )
    gv["prelaunch_gate_total_requests"] = unique_total if unique_total > 0 else raw_gate_total
    gv["prelaunch_gate_pass_rate"] = (
        gv["prelaunch_gate_unique_pass_rate"] if unique_total > 0 else gv["prelaunch_gate_raw_pass_rate"]
    )

    ready_pass_count = int(gv.get("prelaunch_gate_ready_pass_count", 0))
    ready_block_count = int(gv.get("prelaunch_gate_ready_block_count", 0))
    ready_raw_total = ready_pass_count + ready_block_count
    gv["prelaunch_gate_ready_total_requests"] = ready_raw_total
    gv["prelaunch_gate_ready_pass_rate_raw"] = (
        float(ready_pass_count) / float(ready_raw_total) if ready_raw_total > 0 else 0.0
    )

    ready_unique_pass_count = int(gv.get("prelaunch_gate_ready_unique_pass_count", ready_pass_count))
    ready_unique_block_count = int(gv.get("prelaunch_gate_ready_unique_block_count", ready_block_count))
    ready_unique_total = ready_unique_pass_count + ready_unique_block_count
    gv["prelaunch_gate_ready_unique_total_requests"] = ready_unique_total
    gv["prelaunch_gate_ready_unique_pass_rate"] = (
        float(ready_unique_pass_count) / float(ready_unique_total)
        if ready_unique_total > 0 else 0.0
    )
    gv["prelaunch_gate_ready_pass_rate"] = (
        gv["prelaunch_gate_ready_unique_pass_rate"]
        if ready_unique_total > 0 else gv["prelaunch_gate_ready_pass_rate_raw"]
    )
    gv["prelaunch_gate_request_after_ready_total"] = int(gv.get("prelaunch_gate_request_after_ready_count", ready_raw_total))
    gv["prelaunch_gate_request_after_ready_unique_total"] = int(
        gv.get("prelaunch_gate_request_after_ready_unique_count", ready_unique_total)
    )

    relay_attempt = int(gv.get("relay_attempt_count", 0))
    relay_success = int(gv.get("relay_success_count", 0))
    relay_unique_attempt = int(gv.get("relay_unique_attempt_count", 0))
    relay_unique_success = int(gv.get("relay_unique_success_count", 0))
    gv["relay_raw_success_rate"] = (float(relay_success) / float(relay_attempt)) if relay_attempt > 0 else 0.0
    gv["relay_unique_success_rate"] = (
        float(relay_unique_success) / float(relay_unique_attempt)
        if relay_unique_attempt > 0 else 0.0
    )
    gv["relay_success_rate"] = (
        gv["relay_unique_success_rate"] if relay_unique_attempt > 0 else gv["relay_raw_success_rate"]
    )

    gv["launch_count"] = int(getattr(getattr(self, "missile_adapter", None), "get_summary_snapshot", lambda: {})().get("launch_count", 0) or 0) if getattr(self, "missile_adapter", None) is not None else 0
    gv["gate_pass_per_launch_rate"] = (
        float(unique_pass_count) / float(gv["launch_count"])
        if int(gv["launch_count"]) > 0 else 0.0
    )
    gv["gate_ready_pass_per_launch_rate"] = (
        float(ready_unique_pass_count) / float(gv["launch_count"])
        if int(gv["launch_count"]) > 0 else 0.0
    )
    gv["relay_success_per_launch_rate"] = (
        float(relay_unique_success) / float(gv["launch_count"])
        if int(gv["launch_count"]) > 0 else 0.0
    )

    gv["stable_tracking_window_s"] = float(self._stable_tracking_window_s)
    gv["stable_tracking_unique_window_s"] = float(
        getattr(self, "_stable_tracking_unique_window_s", self._stable_tracking_window_s)
    )
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

    _record_prelaunch_gate_event(
        self,
        shooter_id=str(shooter_id),
        target_id=str(target_id),
        current_time=float(current_time),
        event_type="pass",
        stable_ready=True,
        reason="external_launch",
    )
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

    stable_ready = False
    try:
        stable_ready = bool(self._has_stable_tracking_ready(str(target_id)))
    except Exception:
        stable_ready = False
    _record_prelaunch_gate_event(
        self,
        shooter_id=str(shooter_id),
        target_id=str(target_id),
        current_time=float(current_time),
        event_type="block",
        stable_ready=stable_ready,
        reason=str(reason or "external"),
    )
    block_by_target = self._guidance_verify["prelaunch_gate_block_by_target"]
    block_by_target[target_id] = int(block_by_target.get(target_id, 0)) + 1
    block_by_reason = self._guidance_verify.get("prelaunch_gate_block_by_reason", {})
    if isinstance(block_by_reason, dict):
        block_by_reason[str(reason)] = int(block_by_reason.get(str(reason), 0)) + 1
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


def _has_stable_tracking_ready(self, target_id: str, required_seconds: Optional[float] = None) -> bool:
    threshold = float(self._stable_tracking_window_s)
    if required_seconds is not None:
        try:
            required = float(required_seconds)
            if np.isfinite(required) and required > 0.0:
                threshold = required
        except Exception:
            pass
    return self._stable_tracking_elapsed.get(target_id, 0.0) >= threshold


def _has_stable_dual_tracking(self, target_id: str, required_seconds: Optional[float] = None) -> bool:
    return _has_stable_tracking_ready(self, target_id, required_seconds=required_seconds)


def check_external_prelaunch_gate(
    self,
    target_id: str,
    current_time: float,
    shooter_id: Optional[str] = None,
    distance_km: Optional[float] = None,
    phase_name: str = "",
    is_second_attack: bool = False,
    target_zone: Optional[str] = None,
) -> Tuple[bool, Dict[str, object]]:
    target = str(target_id or "").strip()
    profile = _get_prelaunch_gate_profile(
        self,
        target,
        current_time,
        shooter_id=shooter_id,
        distance_km=distance_km,
        phase_name=phase_name,
        is_second_attack=is_second_attack,
        target_zone=target_zone,
    )
    if not target:
        return False, {
            "target_id": target,
            "stable_seconds": 0.0,
            "required_seconds": float(profile.get("required_seconds", self._stable_tracking_window_s)),
            "trackers": [],
            "min_quality": float(profile.get("min_quality", self._stable_tracking_min_quality)),
            "current_time": float(current_time),
            "reason": "invalid_target",
            "phase_name": str(profile.get("phase_name", "")),
            "target_zone": str(profile.get("target_zone", "UNKNOWN")),
            "distance_km": profile.get("distance_km"),
            "is_second_attack": bool(profile.get("is_second_attack", False)),
        }

    stable_seconds = float(self._stable_tracking_elapsed.get(target, 0.0))
    trackers = list(self._stable_tracking_trackers.get(target, []))
    last_ok = float(self._stable_tracking_last_ok.get(target, -1e9))
    in_grace = (float(current_time) - last_ok) <= float(self._stable_tracking_grace_s)
    required_seconds = float(profile.get("required_seconds", self._stable_tracking_window_s))
    min_quality = float(profile.get("min_quality", self._stable_tracking_min_quality))
    passed = self._has_stable_tracking_ready(target, required_seconds=required_seconds) and (len(trackers) >= 1 or in_grace)

    return passed, {
        "target_id": target,
        "stable_seconds": stable_seconds,
        "required_seconds": required_seconds,
        "trackers": trackers,
        "min_quality": min_quality,
        "grace_seconds": float(self._stable_tracking_grace_s),
        "current_time": float(current_time),
        "reason": "ok" if passed else "stable_track_not_ready",
        "phase_name": str(profile.get("phase_name", "")),
        "target_zone": str(profile.get("target_zone", "UNKNOWN")),
        "distance_km": profile.get("distance_km"),
        "is_second_attack": bool(profile.get("is_second_attack", False)),
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

    mar_km = float(getattr(getattr(self, "ranges", None), "MAR", 35.0))
    terminal_hold_km = max(12.0, min(15.0, mar_km - 20.0))
    terminal_guard_km = max(18.0, terminal_hold_km + 4.0)

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

        relay_switch_count = int(getattr(self, "_relay_switch_count", {}).get(missile_id, 0))
        previous_guide = getattr(self, "_relay_previous_guide", {}).get(missile_id)
        previous_guide_time = float(getattr(self, "_relay_previous_guide_time", {}).get(missile_id, -9999.0))
        terminal_lock_until = float(getattr(self, "_relay_terminal_lock_until", {}).get(missile_id, -9999.0))
        last_relay_t = self._relay_last_time.get(missile_id, -999.0)
        relay_cooldown_s = (5.0 if terminal_guidance else 4.5) + (
            min(relay_switch_count, 3) * (0.75 if terminal_guidance else 0.35)
        )
        if current_time - last_relay_t < relay_cooldown_s:
            continue

        guide_aircraft = env.agents.get(guide_id)
        if guide_aircraft is None:
            continue

        guide_alive = bool(guide_aircraft.is_alive)
        q = 0.0
        guide_distance = float("inf")
        defensive = False
        missile_distance = float(getattr(missile, "distance_to_target", 999.0))
        missile_tof = float(getattr(missile, "time_of_flight", 0.0))
        relay_trackers = set(self._stable_tracking_trackers.get(target_id, []))
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
            needs_relay = self.missile_adapter.needs_relay(missile_id, guide_alive, guide_distance)
            defensive = self._is_agent_in_defensive_maneuver(env, guide_id)
            defensive = bool(
                defensive and (
                    terminal_guidance
                    or missile_distance <= 65.0
                    or missile_tof >= 16.0
                )
            )
            guide_supported = guide_id in relay_trackers
            terminal_hold_active = terminal_guidance and missile_distance <= terminal_hold_km
            terminal_preserve_range_km = max(
                mar_km + (4.0 if terminal_hold_active else 6.0),
                39.0 if terminal_hold_active else 41.0,
            )
            terminal_hold_q_floor = 0.44 if terminal_hold_active else 0.50
            if (
                terminal_guidance
                and missile_distance <= terminal_guard_km
                and not defensive
                and guide_distance >= terminal_preserve_range_km
                and (guide_supported or q >= terminal_hold_q_floor)
            ):
                continue
            terminal_low_q = 0.46 if terminal_hold_active else (0.54 if missile_distance <= 28.0 else 0.60)
            low_quality = bool(
                q < (terminal_low_q if terminal_guidance else 0.50)
                and (
                    terminal_guidance
                    or missile_distance <= 58.0
                    or missile_tof >= 18.0
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

            if (
                terminal_guidance
                and current_time < terminal_lock_until
                and relay_reason != "guide_destroyed"
            ):
                catastrophic_q = q < 0.18
                catastrophic_geometry = guide_distance <= max(mar_km + 1.0, 37.0)
                settle_maneuver_break = defensive and missile_distance > terminal_hold_km
                if not (catastrophic_q or catastrophic_geometry or settle_maneuver_break):
                    continue

        relay_id = self._select_relay_candidate(
            env,
            exclude_agent=guide_id,
            target_id=target_id,
            terminal_priority=terminal_guidance,
        )
        self._guidance_verify["relay_attempt_count"] += 1
        unique_attempt_key = (str(missile_id), str(target_id), str(relay_id or "-"))
        last_unique_attempt_t = float(getattr(self, "_relay_unique_attempt_last_time", {}).get(unique_attempt_key, -1e9))
        unique_window_s = float(getattr(self, "_stable_tracking_unique_window_s", self._stable_tracking_window_s))
        if current_time - last_unique_attempt_t >= unique_window_s:
            self._guidance_verify["relay_unique_attempt_count"] += 1
            self._relay_unique_attempt_last_time[unique_attempt_key] = float(current_time)
        if not relay_id:
            self._guidance_verify["relay_fail_count"] += 1
            if not hasattr(self, "_relay_skip_last_time"):
                self._relay_skip_last_time = {}
            last_log_t = float(self._relay_skip_last_time.get(missile_id, -9999.0))
            if current_time - last_log_t >= 10.0:
                self._relay_skip_last_time[missile_id] = current_time
                log.info(f"⚠️ [RELAY_SKIP] missile={missile_id} guide={guide_id} target={target_id} reason={relay_reason} no_candidate")
            continue

        if (
            terminal_guidance
            and previous_guide
            and relay_id == previous_guide
            and (current_time - previous_guide_time) < 12.0
            and relay_reason != "guide_destroyed"
        ):
            self._guidance_verify["relay_fail_count"] += 1
            continue

        relay_q = float(self.cap_radar.get_track_quality(relay_id, target_id))
        relay_supported = relay_id in relay_trackers
        if terminal_guidance:
            terminal_hold_active = missile_distance <= terminal_hold_km
            q_margin = 0.08 if terminal_hold_active else 0.05
            min_relay_q = max(0.38 if terminal_hold_active else 0.30, q + q_margin)
            if relay_supported:
                q_margin = 0.05 if terminal_hold_active else 0.03
                min_relay_q = max(0.34 if terminal_hold_active else 0.26, q + q_margin)
        else:
            q_margin = 0.05
            min_relay_q = max(0.24, q + q_margin)
            if relay_supported:
                q_margin = 0.03
                min_relay_q = max(0.20, q + q_margin)

        if relay_reason == "guide_destroyed":
            min_relay_q = min(min_relay_q, 0.12 if terminal_guidance else 0.10)
        elif relay_reason == "guide_maneuvering":
            min_relay_q = min(min_relay_q, 0.20 if terminal_guidance else 0.16)
        elif relay_reason == "range_or_survivability":
            min_relay_q = min(min_relay_q, 0.24 if terminal_guidance else 0.20)
        if terminal_guidance and relay_switch_count >= 2 and relay_reason != "guide_destroyed":
            extra_margin = 0.07 + 0.02 * min(relay_switch_count - 2, 2)
            min_relay_q = max(min_relay_q, q + extra_margin)

        if relay_q < min_relay_q:
            self._guidance_verify["relay_fail_count"] += 1
            continue

        if self.missile_adapter.apply_relay_guidance(missile_id, relay_id, env=env):
            self._relay_last_time[missile_id] = current_time
            self._relay_guidance_active = True
            self._relay_switch_count[missile_id] = relay_switch_count + 1
            self._relay_previous_guide[missile_id] = guide_id
            self._relay_previous_guide_time[missile_id] = float(current_time)
            if terminal_guidance:
                self._relay_terminal_lock_until[missile_id] = float(current_time) + 6.5 + min(relay_switch_count, 2)
            self._guidance_verify["relay_success_count"] += 1
            unique_success_key = (str(missile_id), str(target_id), str(relay_id))
            last_unique_success_t = float(getattr(self, "_relay_unique_success_last_time", {}).get(unique_success_key, -1e9))
            unique_window_s = float(getattr(self, "_stable_tracking_unique_window_s", self._stable_tracking_window_s))
            if current_time - last_unique_success_t >= unique_window_s:
                self._guidance_verify["relay_unique_success_count"] += 1
                self._relay_unique_success_last_time[unique_success_key] = float(current_time)
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
    stable_trackers = set(self._stable_tracking_trackers.get(target_id, []))
    guide_formation_label = None
    if hasattr(self, "_get_formation_label"):
        try:
            guide_formation_label = self._get_formation_label(exclude_agent, env)
        except Exception:
            guide_formation_label = None
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
        same_pair = False
        if guide_formation_label and hasattr(self, "_get_formation_label"):
            try:
                same_pair = self._get_formation_label(aid, env) == guide_formation_label
            except Exception:
                same_pair = False
        lock_bonus = 0.0
        try:
            if self.cap_radar.get_lock_target(aid) == target_id:
                lock_bonus = 10.0
        except Exception:
            lock_bonus = 0.0
        guidance_bonus = 0.0
        try:
            if self.missile_adapter.has_active_guidance_commit(aid, target_id=target_id):
                guidance_bonus = 6.0 if (same_pair or aid in stable_trackers) else 3.0
        except Exception:
            guidance_bonus = 0.0
        same_pair_bonus = 16.0 if same_pair else 0.0
        tracker_bonus = 14.0 if aid in stable_trackers else 0.0
        score = q * 100.0 - min(aid_distance, 120.0) * 0.30 + same_pair_bonus + tracker_bonus + lock_bonus + guidance_bonus
        if defensive:
            if q >= 0.15 and score > defensive_fallback_score:
                defensive_fallback_score = score
                defensive_fallback_agent = aid
            continue
        terminal_q_threshold = 0.30 if (same_pair or aid in stable_trackers) else 0.34
        best_q_threshold = 0.26 if (same_pair or aid in stable_trackers) else 0.30
        fallback_q_threshold = 0.12 if (same_pair or aid in stable_trackers) else 0.15
        nearest_q_threshold = 0.10 if (same_pair or aid in stable_trackers) else 0.12
        if terminal_priority and q >= terminal_q_threshold and score > terminal_best_score:
            terminal_best_score = score
            terminal_best_agent = aid
            continue
        if q >= best_q_threshold and score > best_score:
            best_score = score
            best_agent = aid
            continue
        if q >= fallback_q_threshold and score > fallback_score:
            fallback_score = score
            fallback_agent = aid
        if q >= nearest_q_threshold and aid_distance < nearest_supported_distance:
            nearest_supported_distance = aid_distance
            nearest_supported_agent = aid
    if terminal_priority:
        return terminal_best_agent or best_agent or fallback_agent or nearest_supported_agent or defensive_fallback_agent
    return best_agent or fallback_agent or nearest_supported_agent or defensive_fallback_agent
