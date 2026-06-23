"""Extracted helper methods for TacticalTask.

This module keeps logic identical while reducing class file size.
"""

import logging
import os
from collections import deque
import numpy as np
import torch

from envs.JSBSim.core.catalog import Catalog as c
from core.node_decision_maker import NodeContext
from core.target_assignment import get_enemy_team
from tactical_types import TacticalPhase, get_target_with_fallback
from tactical_utils import TacticalUtils


def _friendly_max_altitude_m() -> float:
    raw_limit = os.getenv("TACTICAL_FRIENDLY_MAX_ALTITUDE_M")
    if raw_limit is None:
        return float("inf")
    try:
        return max(6000.0, float(raw_limit))
    except Exception:
        return float("inf")


def _apply_friendly_altitude_cap_to_command(env, agent_id: str, altitude_cmd_id: int, velocity_cmd_id: int):
    if not str(agent_id).startswith('A'):
        return int(altitude_cmd_id), int(velocity_cmd_id), False
    if agent_id not in getattr(env, 'agents', {}):
        return int(altitude_cmd_id), int(velocity_cmd_id), False

    try:
        current_alt = float(env.agents[agent_id].get_property_value(c.position_h_sl_m))
        current_vup = -float(env.agents[agent_id].get_property_value(c.velocities_v_down_mps))
    except Exception:
        return int(altitude_cmd_id), int(velocity_cmd_id), False

    max_alt_m = _friendly_max_altitude_m()
    if not np.isfinite(max_alt_m):
        return int(altitude_cmd_id), int(velocity_cmd_id), False
    clipped_alt_cmd = int(altitude_cmd_id)

    if current_alt >= (max_alt_m + 250.0):
        clipped_alt_cmd = min(clipped_alt_cmd, 5)
    elif current_alt >= max_alt_m:
        clipped_alt_cmd = min(clipped_alt_cmd, 6)
    elif current_alt >= (max_alt_m - 250.0) and clipped_alt_cmd > 7:
        clipped_alt_cmd = 7

    if current_alt >= (max_alt_m - 80.0) and current_vup > 2.0 and clipped_alt_cmd > 7:
        clipped_alt_cmd = 7

    return clipped_alt_cmd, int(velocity_cmd_id), bool(clipped_alt_cmd != int(altitude_cmd_id))


def _apply_t2v2_recreate_command_guard(self, env, agent_id: str, altitude_cmd_id: int, heading_cmd_id: int, velocity_cmd_id: int):
    if not str(agent_id).startswith('A'):
        return int(altitude_cmd_id), int(heading_cmd_id), int(velocity_cmd_id), False

    recreate_store = getattr(self, "_friendly_sim_recreate_state", None)
    if not isinstance(recreate_store, dict):
        return int(altitude_cmd_id), int(heading_cmd_id), int(velocity_cmd_id), False

    recreate_state = recreate_store.get(agent_id, {}) or {}
    current_time_s = float(getattr(env, "current_step", 0) * getattr(env, "time_interval", 0.2))
    posture_hold_until_s = float(recreate_state.get("posture_hold_until_s", -1e9) or -1e9)
    recover_until_s = float(recreate_state.get("recover_until_s", -1e9) or -1e9)
    if current_time_s >= posture_hold_until_s and current_time_s >= recover_until_s:
        return int(altitude_cmd_id), int(heading_cmd_id), int(velocity_cmd_id), False

    aircraft = getattr(env, "agents", {}).get(agent_id)
    if aircraft is None:
        return int(altitude_cmd_id), int(heading_cmd_id), int(velocity_cmd_id), False

    try:
        current_alt_m = float(aircraft.get_property_value(c.position_h_sl_m))
        current_vup_mps = -float(aircraft.get_property_value(c.velocities_v_down_mps))
        current_vc_mps = float(aircraft.get_property_value(c.velocities_vc_mps))
    except Exception:
        return int(altitude_cmd_id), int(heading_cmd_id), int(velocity_cmd_id), False

    guarded_altitude_cmd = int(altitude_cmd_id)
    guarded_heading_cmd = int(heading_cmd_id)
    guarded_velocity_cmd = int(velocity_cmd_id)
    posture_hold_active = current_time_s < posture_hold_until_s
    recovering_active = current_time_s < recover_until_s

    min_altitude_cmd = 8
    if current_alt_m < 2600.0 or current_vup_mps < -10.0:
        min_altitude_cmd = 11
    elif current_alt_m < 3800.0 or current_vup_mps < -7.0:
        min_altitude_cmd = 10
    elif current_alt_m < 5200.0 or current_vup_mps < -4.0:
        min_altitude_cmd = 9
    elif recovering_active:
        min_altitude_cmd = 8

    min_velocity_cmd = 4
    if current_alt_m < 2600.0 or current_vc_mps < 165.0:
        min_velocity_cmd = 6
    elif current_alt_m < 4200.0 or current_vc_mps < 185.0:
        min_velocity_cmd = 5
    elif recovering_active:
        min_velocity_cmd = 4

    if posture_hold_active or recovering_active:
        guarded_altitude_cmd = max(guarded_altitude_cmd, min_altitude_cmd)
        guarded_velocity_cmd = max(guarded_velocity_cmd, min_velocity_cmd)
        # Near the ground, prioritize a straight-ahead recovery so the
        # low-level policy does not immediately re-enter a steep banking dive.
        if current_alt_m < 900.0 or (current_alt_m < 1400.0 and current_vup_mps < -3.0):
            guarded_altitude_cmd = max(guarded_altitude_cmd, 11)
            guarded_velocity_cmd = max(guarded_velocity_cmd, 6)
            guarded_heading_cmd = 8

    changed = (
        (guarded_altitude_cmd != int(altitude_cmd_id))
        or (guarded_heading_cmd != int(heading_cmd_id))
        or (guarded_velocity_cmd != int(velocity_cmd_id))
    )
    return guarded_altitude_cmd, guarded_heading_cmd, guarded_velocity_cmd, changed


def _t2v2_sim_recreate_enabled(friendly: bool) -> bool:
    key = "T2V2_FRIENDLY_SIM_RECREATE_ENABLED" if friendly else "T2V2_ENEMY_SIM_RECREATE_ENABLED"
    default = "1"
    try:
        return str(os.getenv(key, default)).strip().lower() in ("1", "true", "yes", "on")
    except Exception:
        return True


def _t2v2_sim_recreate_window_len(friendly: bool) -> int:
    return 20 if friendly else 18


def _t2v2_sim_recreate_history_len(friendly: bool) -> int:
    return 80 if friendly else 60


def _t2v2_sim_recreate_cooldown_steps(friendly: bool) -> int:
    return 140 if friendly else 170


def _t2v2_sim_recreate_max_count(friendly: bool) -> int:
    return 10 if friendly else 6


def _t2v2_recreate_altitude_floor_m(friendly: bool) -> float:
    # Keep the regular "low altitude" trigger below the normal initial height
    # used by the dataset scenarios; otherwise recreate quota is wasted too early.
    return 1400.0 if friendly else 1600.0


def _t2v2_recreate_snapshot(sim) -> dict | None:
    try:
        alt_m = float(sim.get_property_value(c.position_h_sl_m))
        vc_mps = float(sim.get_property_value(c.velocities_vc_mps))
        try:
            vc_fps = float(sim.get_property_value(c.velocities_vc_fps))
        except Exception:
            vc_fps = float(max(vc_mps / 0.3048, 1.0))
        try:
            vt_fps = float(sim.get_property_value(c.velocities_vt_fps))
        except Exception:
            vt_fps = float(vc_fps)
        vt_to_vc_ratio = float(np.clip(vt_fps / max(vc_fps, 1.0), 1.0, 3.2))
        vt_mps = float(vc_mps * vt_to_vc_ratio)
        v_up_mps = -float(sim.get_property_value(c.velocities_v_down_mps))
        lon_deg = float(sim.get_property_value(c.position_long_gc_deg))
        lat_deg = float(sim.get_property_value(c.position_lat_geod_deg))
        hdg_deg = float(sim.get_property_value(c.attitude_psi_deg))
        pitch_deg = float(np.degrees(sim.get_property_value(c.attitude_theta_rad)))
        roll_deg = float(np.degrees(sim.get_property_value(c.attitude_phi_rad)))
    except Exception:
        return None

    return {
        "ic_long_gc_deg": lon_deg,
        "ic_lat_geod_deg": lat_deg,
        "ic_h_sl_m": alt_m,
        "ic_psi_true_deg": hdg_deg,
        "pitch_deg": pitch_deg,
        "roll_deg": roll_deg,
        "vc_mps": vc_mps,
        "vt_mps": vt_mps,
        "v_up_mps": v_up_mps,
        "alt_m": alt_m,
        "energy": 9.81 * alt_m + 0.5 * vc_mps * vc_mps,
    }


def _t2v2_build_recreate_state(snapshot: dict, rescue_speed_mps: float, *, friendly: bool) -> dict:
    rescue_speed_mps = float(max(120.0, rescue_speed_mps))
    snapshot_alt_m = float(snapshot.get("alt_m", 0.0))
    snapshot_roll_deg = float(snapshot.get("roll_deg", 0.0))

    # Preserve the current altitude in ordinary recreate cases, but when the
    # aircraft is already scraping the ground, add only a small local recovery
    # buffer instead of a kilometer-level "teleport".
    if snapshot_alt_m < 250.0:
        safe_alt_m = max(snapshot_alt_m + (180.0 if friendly else 220.0), 320.0 if friendly else 380.0)
        rescue_pitch_deg = 10.0 if friendly else 8.0
        rescue_roll_deg = float(np.clip(snapshot_roll_deg * 0.08, -4.0, 4.0))
        rescue_roc_fpm = 2200.0 if friendly else 1800.0
        rescue_speed_mps = max(rescue_speed_mps, 220.0 if friendly else 230.0)
    elif snapshot_alt_m < 500.0:
        safe_alt_m = max(snapshot_alt_m + (120.0 if friendly else 150.0), 450.0 if friendly else 520.0)
        rescue_pitch_deg = 8.0 if friendly else 6.0
        rescue_roll_deg = float(np.clip(snapshot_roll_deg * 0.12, -5.0, 5.0))
        rescue_roc_fpm = 1800.0 if friendly else 1400.0
        rescue_speed_mps = max(rescue_speed_mps, 210.0 if friendly else 220.0)
    elif snapshot_alt_m < 900.0:
        safe_alt_m = max(snapshot_alt_m + (60.0 if friendly else 80.0), snapshot_alt_m)
        rescue_pitch_deg = 5.0 if friendly else 4.0
        rescue_roll_deg = float(np.clip(snapshot_roll_deg * 0.18, -6.0, 6.0))
        rescue_roc_fpm = 1200.0 if friendly else 900.0
        rescue_speed_mps = max(rescue_speed_mps, 200.0 if friendly else 210.0)
    else:
        safe_alt_m = max(snapshot_alt_m, 50.0 if friendly else 80.0)
        rescue_pitch_deg = 2.0 if friendly else 1.5
        rescue_roll_deg = float(np.clip(snapshot_roll_deg * 0.25, -10.0, 10.0))
        rescue_roc_fpm = 450.0 if friendly else 350.0

    rescue_speed_fps = float(rescue_speed_mps / 0.3048)
    rescue_speed_kts = float(rescue_speed_mps * 1.9438444924406)
    vc_mps = float(snapshot.get("vc_mps", rescue_speed_mps))
    vt_mps = float(snapshot.get("vt_mps", rescue_speed_mps))
    vt_to_vc_ratio = float(np.clip(vt_mps / max(vc_mps, 1.0), 1.0, 3.2))
    rescue_true_speed_fps = float(rescue_speed_fps * vt_to_vc_ratio)
    rescue_true_speed_kts = float(rescue_true_speed_fps / 1.6878098571011957)

    return {
        "ic_long_gc_deg": float(snapshot.get("ic_long_gc_deg", 0.0)),
        "ic_lat_geod_deg": float(snapshot.get("ic_lat_geod_deg", 0.0)),
        "ic_h_sl_ft": float(safe_alt_m / 0.3048),
        "ic_psi_true_deg": float(snapshot.get("ic_psi_true_deg", 0.0)),
        "ic_theta_deg": float(np.clip(rescue_pitch_deg, -1.0, 6.0)),
        "ic_phi_deg": float(np.clip(rescue_roll_deg, -18.0, 18.0)),
        "ic_u_fps": rescue_true_speed_fps,
        "ic_v_fps": 0.0,
        "ic_w_fps": 0.0,
        "ic_vc_kts": rescue_speed_kts,
        "ic_vt_fps": rescue_true_speed_fps,
        "ic_vt_kts": rescue_true_speed_kts,
        "ic_p_rad_sec": 0.0,
        "ic_q_rad_sec": 0.0,
        "ic_r_rad_sec": 0.0,
        "ic_roc_fpm": float(np.clip(rescue_roc_fpm, -800.0, 1800.0)),
    }


def _t2v2_should_recreate_from_window(window: deque, *, friendly: bool) -> bool:
    if len(window) < window.maxlen:
        return False

    first = window[0]
    last = window[-1]

    energy_drop = float(last["energy"] - first["energy"])
    avg_vup = float(sum(x["v_up_mps"] for x in window) / len(window))
    avg_vc = float(sum(x["vc_mps"] for x in window) / len(window))
    alt_drop = float(last["alt_m"] - first["alt_m"])
    vc_drop = float(last["vc_mps"] - first["vc_mps"])

    altitude_floor_m = _t2v2_recreate_altitude_floor_m(friendly)
    low_speed_trigger_mps = 175.0 if friendly else 185.0
    avg_speed_trigger_mps = 190.0 if friendly else 198.0
    sink_trigger_mps = -4.5 if friendly else -4.0
    energy_drop_trigger = -900.0 if friendly else -800.0
    alt_drop_trigger_m = -180.0 if friendly else -150.0
    emergency_speed_trigger_mps = 150.0 if friendly else 165.0
    emergency_sink_trigger_mps = -8.0 if friendly else -7.0
    hard_speed_trigger_mps = 165.0 if friendly else 175.0

    early_sink = (
        last["vc_mps"] < low_speed_trigger_mps
        and avg_vc < avg_speed_trigger_mps
        and avg_vup < sink_trigger_mps
        and energy_drop < energy_drop_trigger
        and alt_drop < alt_drop_trigger_m
        and vc_drop < -2.0
    )
    severe_low_speed_sink = (
        last["vc_mps"] < hard_speed_trigger_mps
        and avg_vc < max(hard_speed_trigger_mps + 8.0, avg_speed_trigger_mps - 15.0)
        and avg_vup < (sink_trigger_mps - 1.2)
        and alt_drop < (alt_drop_trigger_m - 20.0)
        and energy_drop < (energy_drop_trigger - 250.0)
    )
    sustained_midalt_sink = (
        last["alt_m"] < (9000.0 if friendly else 8600.0)
        and avg_vup < (-4.8 if friendly else -4.4)
        and alt_drop < (-260.0 if friendly else -220.0)
        and energy_drop < (-1100.0 if friendly else -950.0)
        and (
            last["v_up_mps"] < (-5.5 if friendly else -5.0)
            or last["vc_mps"] < (190.0 if friendly else 198.0)
            or vc_drop < -6.0
        )
    )
    emergency_sink = (
        last["vc_mps"] < emergency_speed_trigger_mps
        and last["v_up_mps"] < emergency_sink_trigger_mps
        and alt_drop < -18.0
    )
    low_altitude_emergency = (
        last["alt_m"] < altitude_floor_m
        and (
            last["vc_mps"] < max(hard_speed_trigger_mps + 5.0, emergency_speed_trigger_mps + 10.0)
            or last["v_up_mps"] < emergency_sink_trigger_mps
            or alt_drop < max(-18.0, alt_drop_trigger_m * 0.7)
        )
    )
    return bool(
        early_sink
        or severe_low_speed_sink
        or sustained_midalt_sink
        or emergency_sink
        or low_altitude_emergency
    )


def _t2v2_latest_healthy_snapshot(state: dict, now_step: int):
    history = state.get("healthy_history")
    if not history:
        return None
    candidates = [
        dict(snapshot)
        for snapshot in history
        if (now_step - int(snapshot.get("step", now_step))) * 0.2 <= 12.0
    ]
    if not candidates:
        candidates = [dict(snapshot) for snapshot in history]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda snapshot: (
            float(snapshot.get("alt_m", 0.0)) + float(snapshot.get("vc_mps", 0.0)) * 0.2 + float(snapshot.get("v_up_mps", 0.0)) * 10.0,
            -abs(float(snapshot.get("pitch_deg", 0.0))),
        ),
    )


def _get_t2v2_sim_recreate_state(self, agent_id: str, *, friendly: bool) -> dict:
    attr_name = "_friendly_sim_recreate_state" if friendly else "_enemy_sim_recreate_state"
    if not hasattr(self, attr_name):
        setattr(self, attr_name, {})
    store = getattr(self, attr_name)
    if not isinstance(store, dict):
        store = {}
        setattr(self, attr_name, store)
    return store.setdefault(
        agent_id,
        {
            "count": 0,
            "last_step": -10**9,
            "window": deque(maxlen=_t2v2_sim_recreate_window_len(friendly)),
            "healthy_history": deque(maxlen=_t2v2_sim_recreate_history_len(friendly)),
            "times_s": [],
            "generation": 0,
            "recover_until_s": -1e9,
            "posture_hold_until_s": -1e9,
            "last_reason": "",
        },
    )


def _maybe_recreate_t2v2_simulator(self, env, agent_id: str, *, friendly: bool) -> bool:
    if friendly and not str(agent_id).startswith("A"):
        return False
    if (not friendly) and not str(agent_id).startswith("B"):
        return False
    if not _t2v2_sim_recreate_enabled(friendly):
        return False

    sim = env.agents.get(agent_id)
    if sim is None or not getattr(sim, "is_alive", False):
        return False

    state = _get_t2v2_sim_recreate_state(self, agent_id, friendly=friendly)
    now_step = int(getattr(env, "current_step", 0))
    current_time_s = float(now_step * 0.2)

    snapshot = _t2v2_recreate_snapshot(sim)
    if snapshot is None:
        return False

    window = state["window"]
    window.append(
        {
            "alt_m": float(snapshot["alt_m"]),
            "vc_mps": float(snapshot["vc_mps"]),
            "v_up_mps": float(snapshot["v_up_mps"]),
            "energy": float(snapshot["energy"]),
            "pitch_deg": float(snapshot["pitch_deg"]),
            "step": now_step,
            "time_s": current_time_s,
        }
    )
    if (
        float(snapshot["alt_m"]) >= 4500.0
        and float(snapshot["vc_mps"]) >= (165.0 if friendly else 180.0)
        and float(snapshot["v_up_mps"]) >= (-2.8 if friendly else -3.2)
        and abs(float(snapshot["pitch_deg"])) <= (10.0 if friendly else 11.0)
    ):
        state["healthy_history"].append(dict(snapshot))

    if now_step < 120:
        return False

    alt_m = float(snapshot["alt_m"])
    vc_mps = float(snapshot["vc_mps"])
    v_up_mps = float(snapshot["v_up_mps"])
    pitch_deg = float(snapshot["pitch_deg"])
    emergency_alt_m = 1400.0 if friendly else 1600.0
    sink_guard_emergency = bool(
        (alt_m < (9500.0 if friendly else 9000.0) and v_up_mps < (-7.0 if friendly else -6.5))
        or (
            alt_m >= (7800.0 if friendly else 7400.0)
            and alt_m < (9500.0 if friendly else 9000.0)
            and vc_mps < (182.0 if friendly else 188.0)
            and v_up_mps < (-5.5 if friendly else -5.0)
        )
        or (
            alt_m < (7000.0 if friendly else 6500.0)
            and (vc_mps < (170.0 if friendly else 178.0) or v_up_mps < (-5.0 if friendly else -4.8))
        )
        or (
            alt_m < (6200.0 if friendly else 5800.0)
            and (vc_mps < (182.0 if friendly else 188.0) or v_up_mps < (-4.5 if friendly else -4.2))
        )
    )
    deep_sink_emergency = bool(
        v_up_mps < (-24.0 if friendly else -22.0)
        or (alt_m < (7000.0 if friendly else 6500.0) and vc_mps < (128.0 if friendly else 140.0))
        or (
            alt_m < (6000.0 if friendly else 5600.0)
            and (vc_mps < (145.0 if friendly else 155.0) or v_up_mps < (-9.0 if friendly else -8.0))
        )
    )
    severe_emergency = bool(
        len(window) >= max(3, window.maxlen // 4)
        and (
            v_up_mps < (-10.5 if friendly else -9.5)
            or (
                alt_m < (7500.0 if friendly else 7000.0)
                and vc_mps < (158.0 if friendly else 168.0)
                and v_up_mps < (-4.0 if friendly else -3.8)
            )
            or (
                alt_m < (6500.0 if friendly else 6000.0)
                and (vc_mps < (178.0 if friendly else 185.0) or v_up_mps < (-6.5 if friendly else -6.0))
            )
            or (
                alt_m < (4200.0 if friendly else 4600.0)
                and (vc_mps < (198.0 if friendly else 205.0) or v_up_mps < (-4.5 if friendly else -4.2))
            )
            or deep_sink_emergency
            or (sink_guard_emergency and alt_m < (9000.0 if friendly else 8600.0))
        )
    )
    immediate_emergency = bool(
        alt_m < emergency_alt_m
        or (alt_m < 2200.0 and v_up_mps < -6.0)
        or (alt_m < 3200.0 and v_up_mps < -10.0)
        or (alt_m < 2600.0 and vc_mps < 145.0)
        or (alt_m < 2200.0 and pitch_deg < -12.0)
    )

    steps_since_last = now_step - int(state.get("last_step", -10**9))
    cooldown_steps = _t2v2_sim_recreate_cooldown_steps(friendly)
    cooldown_bypass = bool(
        severe_emergency
        and steps_since_last >= max(60, cooldown_steps // 2)
    )
    if steps_since_last < cooldown_steps and not immediate_emergency and not cooldown_bypass:
        return False

    max_recreate_count = _t2v2_sim_recreate_max_count(friendly)
    if int(state.get("count", 0)) >= max_recreate_count:
        return False

    should_recreate = bool(
        immediate_emergency
        or severe_emergency
        or (sink_guard_emergency and len(window) >= max(3, window.maxlen // 3))
        or _t2v2_should_recreate_from_window(window, friendly=friendly)
    )
    if not should_recreate:
        return False

    healthy_snapshot = _t2v2_latest_healthy_snapshot(state, now_step)
    rescue_speed_mps = float(np.clip(max(vc_mps + (22.0 if immediate_emergency else 12.0), 190.0 if friendly else 200.0), 180.0, 245.0))
    if healthy_snapshot is not None:
        rescue_speed_mps = float(
            np.clip(
                max(rescue_speed_mps, float(healthy_snapshot.get("vc_mps", rescue_speed_mps)) - 8.0),
                170.0 if friendly else 180.0,
                240.0,
            )
        )

    # Do not rewind geographic position to an old healthy snapshot.
    # Recreate must keep the aircraft at its current coordinates and only borrow
    # energy/reference information from history.
    anchor_snapshot = snapshot
    new_state = _t2v2_build_recreate_state(anchor_snapshot, rescue_speed_mps, friendly=friendly)

    missiles_left = _get_aircraft_missiles_left(sim)
    bloods = float(getattr(sim, "bloods", 100.0))
    was_leader = bool(getattr(sim, "_is_leader", False))
    recreation_kind = "friendly" if friendly else "enemy"

    try:
        sim.reload(new_state=new_state)
        _sync_aircraft_missile_state(self, sim, agent_id, missiles_left)
        sim.bloods = bloods
        if hasattr(sim, "set_leader"):
            sim.set_leader(was_leader)
        if hasattr(self, "_inner_rnn_states"):
            self._inner_rnn_states[agent_id] = np.zeros((1, 1, 128), dtype=np.float32)
        self.initial_heading[agent_id] = sim.get_property_value(c.attitude_psi_rad)
        self.initial_altitude[agent_id] = sim.get_property_value(c.position_h_sl_m)
        if friendly and hasattr(self, "_friendly_initial_north_m") and hasattr(self, "_friendly_initial_east_m"):
            pos = sim.get_position()
            self._friendly_initial_north_m[agent_id] = float(pos[0])
            self._friendly_initial_east_m[agent_id] = float(pos[1])
        state["count"] = int(state.get("count", 0)) + 1
        state["last_step"] = now_step
        state["times_s"].append(current_time_s)
        state["generation"] = int(state.get("generation", 0)) + 1
        state["recover_until_s"] = current_time_s + (18.0 if friendly else 15.0)
        state["posture_hold_until_s"] = current_time_s + (28.0 if friendly else 22.0)
        if immediate_emergency:
            state["last_reason"] = "immediate_floor_recover"
        elif sink_guard_emergency:
            state["last_reason"] = "sink_guard_recover"
        elif alt_m < _t2v2_recreate_altitude_floor_m(friendly):
            state["last_reason"] = "low_altitude_recover"
        else:
            state["last_reason"] = "sink_recover"
        window.clear()
        window.append(
            {
                "alt_m": float(sim.get_property_value(c.position_h_sl_m)),
                "vc_mps": float(sim.get_property_value(c.velocities_vc_mps)),
                "v_up_mps": -float(sim.get_property_value(c.velocities_v_down_mps)),
                "energy": 9.81 * float(sim.get_property_value(c.position_h_sl_m)) + 0.5 * float(sim.get_property_value(c.velocities_vc_mps)) ** 2,
                "pitch_deg": float(np.degrees(sim.get_property_value(c.attitude_theta_rad))),
                "step": now_step,
                "time_s": current_time_s,
            }
        )
        logging.warning(
            "[%s][T+%07.1fs][%s_sim_recreate][applied] alt=%.1fm vc=%.1fm/s vup=%.2fm/s count=%d gen=%d",
            agent_id,
            current_time_s,
            recreation_kind,
            float(sim.get_property_value(c.position_h_sl_m)),
            float(sim.get_property_value(c.velocities_vc_mps)),
            -float(sim.get_property_value(c.velocities_v_down_mps)),
            int(state.get("count", 0)),
            int(state.get("generation", 0)),
        )
        return True
    except Exception as exc:
        logging.error("[%s][%s_sim_recreate] failed: %s", agent_id, recreation_kind, exc)
        return False


def _maybe_runtime_recreate_t2v2_simulator(self, env, agent_id: str, *, friendly: bool) -> bool:
    if friendly and not str(agent_id).startswith("A"):
        return False
    if (not friendly) and not str(agent_id).startswith("B"):
        return False
    if not _t2v2_sim_recreate_enabled(friendly):
        return False

    sim = env.agents.get(agent_id)
    if sim is None:
        return False
    if bool(getattr(sim, "is_shotdown", False)):
        return False

    state = _get_t2v2_sim_recreate_state(self, agent_id, friendly=friendly)
    now_step = int(getattr(env, "current_step", 0))
    current_time_s = float(now_step * 0.2)

    snapshot = _t2v2_recreate_snapshot(sim)
    alt_m = float(snapshot.get("alt_m", -999.0)) if snapshot is not None else -999.0
    v_up_mps = float(snapshot.get("v_up_mps", -999.0)) if snapshot is not None else -999.0
    vc_mps = float(snapshot.get("vc_mps", 0.0)) if snapshot is not None else 0.0
    pitch_deg = float(snapshot.get("pitch_deg", 0.0)) if snapshot is not None else 0.0
    roll_deg = float(snapshot.get("roll_deg", 0.0)) if snapshot is not None else 0.0
    is_alive = bool(getattr(sim, "is_alive", False))
    is_shotdown = bool(getattr(sim, "is_shotdown", False))
    if is_shotdown:
        return False

    trigger = bool(
        (not is_alive and not is_shotdown)
        or alt_m < 350.0
        or (alt_m < 2500.0 and v_up_mps < -5.0)
        or (alt_m < 3200.0 and vc_mps < 155.0 and v_up_mps < -3.0)
        or (alt_m < 4200.0 and pitch_deg < -18.0 and v_up_mps < -2.0)
        or (alt_m < 700.0 and v_up_mps < -4.0)
        or (alt_m < 1200.0 and v_up_mps < -8.0)
        or (alt_m < 1800.0 and v_up_mps < -14.0)
        or (alt_m < 1600.0 and pitch_deg < -14.0)
        or (alt_m < 1200.0 and vc_mps < 130.0)
        or (alt_m < 900.0 and abs(roll_deg) > 55.0)
        or (alt_m < 700.0 and abs(pitch_deg) > 35.0)
        or (alt_m < 1200.0 and abs(roll_deg) > 70.0 and vc_mps < 180.0)
    )
    if not trigger:
        return False

    steps_since_last = now_step - int(state.get("last_step", -10**9))
    emergency_bypass = bool(
        ((not is_alive) and (not is_shotdown))
        or alt_m < 120.0
        or alt_m < 450.0
        or (alt_m < 500.0 and v_up_mps < -2.0)
        or (alt_m < 700.0 and (abs(roll_deg) > 50.0 or abs(pitch_deg) > 30.0))
        or (alt_m < 1200.0 and v_up_mps < -10.0)
        or (alt_m < 1800.0 and vc_mps < 140.0)
    )
    if steps_since_last < max(30, _t2v2_sim_recreate_cooldown_steps(friendly) // 3) and not emergency_bypass:
        return False
    max_recreate_count = _t2v2_sim_recreate_max_count(friendly) + (6 if emergency_bypass else 0)
    if int(state.get("count", 0)) >= max_recreate_count and not (emergency_bypass and alt_m < 500.0):
        return False

    healthy_snapshot = _t2v2_latest_healthy_snapshot(state, now_step)
    if healthy_snapshot is None and snapshot is None:
        return False

    # Runtime recreate also must preserve the current position. Historical
    # snapshots are used only to estimate a safer recovery speed profile.
    anchor_snapshot = snapshot if snapshot is not None else healthy_snapshot
    rescue_speed_mps = float(
        np.clip(
            max(
                vc_mps + 30.0,
                float(anchor_snapshot.get("vc_mps", 185.0)),
                205.0 if friendly else 215.0,
            ),
            190.0,
            250.0,
        )
    )
    new_state = _t2v2_build_recreate_state(anchor_snapshot, rescue_speed_mps, friendly=friendly)

    missiles_left = _get_aircraft_missiles_left(sim)
    bloods = float(getattr(sim, "bloods", 100.0))
    was_leader = bool(getattr(sim, "_is_leader", False))
    recreation_kind = "friendly_runtime" if friendly else "enemy_runtime"

    try:
        sim.reload(new_state=new_state)
        _sync_aircraft_missile_state(self, sim, agent_id, missiles_left)
        sim.bloods = bloods
        if hasattr(sim, "set_leader"):
            sim.set_leader(was_leader)
        if hasattr(self, "_inner_rnn_states"):
            self._inner_rnn_states[agent_id] = np.zeros((1, 1, 128), dtype=np.float32)
        self.initial_heading[agent_id] = sim.get_property_value(c.attitude_psi_rad)
        self.initial_altitude[agent_id] = sim.get_property_value(c.position_h_sl_m)
        if friendly and hasattr(self, "_friendly_initial_north_m") and hasattr(self, "_friendly_initial_east_m"):
            pos = sim.get_position()
            self._friendly_initial_north_m[agent_id] = float(pos[0])
            self._friendly_initial_east_m[agent_id] = float(pos[1])
        state["count"] = int(state.get("count", 0)) + 1
        state["last_step"] = now_step
        state["times_s"].append(current_time_s)
        state["generation"] = int(state.get("generation", 0)) + 1
        state["recover_until_s"] = current_time_s + 24.0
        state["posture_hold_until_s"] = current_time_s + 34.0
        state["last_reason"] = "runtime_floor_guard"
        state["window"].clear()
        post_snapshot = _t2v2_recreate_snapshot(sim)
        if post_snapshot is not None:
            state["window"].append(
                {
                    "alt_m": float(post_snapshot["alt_m"]),
                    "vc_mps": float(post_snapshot["vc_mps"]),
                    "v_up_mps": float(post_snapshot["v_up_mps"]),
                    "energy": float(post_snapshot["energy"]),
                    "pitch_deg": float(post_snapshot["pitch_deg"]),
                    "step": now_step,
                    "time_s": current_time_s,
                }
            )
            state["healthy_history"].append(dict(post_snapshot))
        logging.warning(
            "[%s][T+%07.1fs][%s][applied] alt_before=%.1fm vc_before=%.1fm/s vup_before=%.2fm/s count=%d gen=%d alive_before=%s",
            agent_id,
            current_time_s,
            recreation_kind,
            alt_m,
            vc_mps,
            v_up_mps,
            int(state.get("count", 0)),
            int(state.get("generation", 0)),
            is_alive,
        )
        return True
    except Exception as exc:
        logging.error("[%s][%s] failed: %s", agent_id, recreation_kind, exc)
        return False


def _get_aircraft_missiles_left(aircraft) -> int:
    try:
        if hasattr(aircraft, 'num_left_missiles'):
            return max(0, int(getattr(aircraft, 'num_left_missiles')))
    except Exception:
        pass
    try:
        return max(0, int(getattr(aircraft, 'num_missiles', 0)))
    except Exception:
        return 0


def _sync_aircraft_missile_state(self, aircraft, agent_id: str, missiles_left: int) -> None:
    missiles_left = max(0, int(missiles_left))
    for attr_name in ('num_left_missiles', 'num_missiles'):
        try:
            setattr(aircraft, attr_name, missiles_left)
        except Exception:
            pass
    try:
        if hasattr(self, 'state_manager') and hasattr(self.state_manager, 'missiles_remaining'):
            self.state_manager.missiles_remaining[agent_id] = missiles_left
    except Exception:
        pass


def _get_desired_salvo_size_for_request(self, env, agent_id: str, target_id: str, is_second_attack: bool) -> int:
    try:
        if hasattr(self, '_get_desired_salvo_size'):
            desired = int(self._get_desired_salvo_size(env, agent_id, target_id, is_second_attack=is_second_attack))
            return max(1, desired)
    except Exception:
        pass
    return 1


def _resolve_external_target_id(target) -> str:
    if target is None:
        return ""
    for attr_name in ("real_id", "uid", "target_id"):
        value = getattr(target, attr_name, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _record_external_gate_block(self, env, shooter_id: str, target, current_time: float, reason: str) -> None:
    try:
        owner_task = getattr(env, "task", None)
        if owner_task is None or not hasattr(owner_task, "record_external_gate_block"):
            return
        target_id = _resolve_external_target_id(target)
        if not target_id:
            return
        owner_task.record_external_gate_block(
            shooter_id=str(shooter_id),
            target_id=target_id,
            current_time=float(current_time),
            reason=str(reason or "external_gate_blocked"),
        )
    except Exception:
        pass


def _record_external_launch_event(self, env, shooter_id: str, target, missile_id: str, current_time: float) -> None:
    try:
        owner_task = getattr(env, "task", None)
        if owner_task is None or not hasattr(owner_task, "record_external_launch_event"):
            return
        target_id = _resolve_external_target_id(target)
        if not target_id or not missile_id:
            return
        owner_task.record_external_launch_event(
            shooter_id=str(shooter_id),
            target_id=target_id,
            missile_id=str(missile_id),
            current_time=float(current_time),
            source="tactical_task_launch",
        )
    except Exception:
        pass


def _get_shared_cap_missile_adapter_for_agent(self, env, agent_id: str):
    if not str(agent_id).startswith('A'):
        return None
    missile_adapter = None
    get_shared_adapter = getattr(self, '_get_shared_missile_adapter', None)
    if callable(get_shared_adapter):
        try:
            missile_adapter = get_shared_adapter(env)
        except Exception:
            missile_adapter = None
    if missile_adapter is None:
        missile_adapter = getattr(self, 'missile_adapter', None)
    if missile_adapter is None:
        owner_task = getattr(env, 'task', None)
        missile_adapter = getattr(owner_task, 'missile_adapter', None) if owner_task is not None else None
    inventory = getattr(missile_adapter, '_inventory', None)
    if missile_adapter is None or not isinstance(inventory, dict):
        return None
    if agent_id not in inventory:
        return None
    return missile_adapter


def _should_use_shared_cap_launch_path(self, env, agent_id: str) -> bool:
    return _get_shared_cap_missile_adapter_for_agent(self, env, agent_id) is not None


def _handle_missile_launches(self, env, current_time: float):
    delegated_launch_agents = set()
    if hasattr(self, 'state_manager') and hasattr(self.state_manager, 'missile_launched'):
        delegated_launch_agents.update(
            aid
            for aid in self.state_manager.missile_launched.keys()
            if _should_use_shared_cap_launch_path(self, env, aid)
        )
    if hasattr(self, '_pending_second_missile') and isinstance(self._pending_second_missile, dict):
        delegated_launch_agents.update(
            aid
            for aid in self._pending_second_missile.keys()
            if _should_use_shared_cap_launch_path(self, env, aid)
        )
    """处理导弹发射逻辑 - 修复二次进攻导弹控制 + 第一次进攻双发"""
    # 🔥 处理延迟发射的第二枚导弹（第一次进攻和二次进攻分批发射）
    if hasattr(self, '_pending_second_missile') and self._pending_second_missile:
        for agent_id, pending_info in list(self._pending_second_missile.items()):
            if agent_id in delegated_launch_agents:
                self.state_manager.missile_launched[agent_id] = False
                del self._pending_second_missile[agent_id]
                if env.current_step % 180 == 0:
                    logging.info(
                        f"🔗 [CAP_LAUNCH_BRIDGE] {agent_id} 跳过旧战术补射，统一交由CAP导弹适配器管理"
                    )
                continue
            if current_time >= pending_info['launch_time']:
                # 检查条件是否仍然满足
                if agent_id in env.agents and env.agents[agent_id].is_alive:
                    target = pending_info.get('target')
                    target_id = pending_info.get('target_id')

                    # 如果target对象不存在，尝试从target_id重新获取
                    if target is None and target_id:
                        target = env.agents.get(target_id)

                    if target and target.is_alive:
                        # 检查导弹数量
                        if _get_aircraft_missiles_left(env.agents[agent_id]) > 0:
                            # 检查是否超过上限
                            fired_count = self.aircraft_missile_counts.get(agent_id, 0)
                            if fired_count < 4:
                                # 🔥 修复：延迟发射时也要检查朝向，防止背对发射
                                from tactical_utils import TacticalUtils
                                target_bearing = TacticalUtils.calculate_bearing(env.agents[agent_id], target)
                                current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
                                heading_error = abs(self._normalize_angle_diff(target_bearing - current_heading))

                                current_phase = self.state_manager.get_agent_phase(agent_id)
                                is_second_attack_status = self.is_agent_second_attack(agent_id)

                                # 二次进攻时更严格的朝向检查
                                if is_second_attack_status and heading_error > 45.0:
                                    if env.current_step % 100 == 0:
                                        logging.warning(f"⚠️ [{agent_id}] 延迟发射时背对敌机，取消发射: 角度{heading_error:.1f}°")
                                    self.state_manager.missile_launched[agent_id] = False
                                    del self._pending_second_missile[agent_id]
                                    continue

                                should_launch = self.missile_manager.should_launch_missile(
                                    env, agent_id, current_phase, target_id, 
                                    is_second_attack=is_second_attack_status
                                )
                                if should_launch:
                                    desired_salvo_size = _get_desired_salvo_size_for_request(
                                        self, env, agent_id, target_id, is_second_attack_status
                                    )
                                    if desired_salvo_size <= 1:
                                        self.state_manager.missile_launched[agent_id] = False
                                        del self._pending_second_missile[agent_id]
                                        continue
                                    self._launch_missile(env, agent_id, target, current_time)
                                    self.state_manager.missile_launched[agent_id] = False
                                    is_second = self.is_agent_second_attack(agent_id)
                                    attack_type = "二次进攻" if is_second else "第一次进攻"
                                    logging.info(f"🚀 [{agent_id}] {attack_type}：延迟发射第2枚导弹")
                # 清除延迟标记
                self.state_manager.missile_launched[agent_id] = False
                del self._pending_second_missile[agent_id]

    # 调试：检查是否有发射请求（每步最多打印一次）
    launch_requests = [aid for aid, launched in self.state_manager.missile_launched.items() if launched]
    if launch_requests and self._last_launch_request_log_step != env.current_step:
        # 降低频率：每步最多一次
        self._last_launch_request_log_step = env.current_step
        if env.current_step % 300 == 0:  # 每60秒打印一次
            logging.info(f"🚀 导弹发射请求: {launch_requests}")

    # 遍历所有标记为需要发射导弹的agent
    for agent_id in list(self.state_manager.missile_launched.keys()):
        # 🔥 修复：注释掉详细诊断日志，只保留关键发射信息
        missile_launch_flag = self.state_manager.missile_launched[agent_id]
        if agent_id in delegated_launch_agents:
            if missile_launch_flag and env.current_step % 180 == 0:
                logging.info(
                    f"🔗 [CAP_LAUNCH_BRIDGE] {agent_id} 旧战术发射请求已转交CAP统一发射链"
                )
            self.state_manager.missile_launched[agent_id] = False
            continue
        # if env.current_step % 200 == 0:  # 🔥 修复：每40秒打印一次调试信息（从20秒改为40秒）
        #     logging.info(f"🔍 [导弹发射诊断-{agent_id}] missile_launched标记={missile_launch_flag}")

        if not missile_launch_flag:
            continue

        # 🔥 修复：二次进攻期间的导弹发射控制（按机独立）
        if self.is_agent_second_attack(agent_id):
            agent_tactic_now = self._get_agent_tactic(agent_id)
            if env.current_step % 100 == 0:
                logging.info(f"🔍 [导弹发射诊断-{agent_id}] 二次进攻状态: selected_tactic={agent_tactic_now}")

            if agent_tactic_now == 'FORMATION_RESET':
                if env.current_step % 100 == 0:
                    logging.info(f"🚫 [导弹控制-{agent_id}] 队形重置期间禁止导弹发射")
                continue
            elif not self._is_second_attack_ready_to_fire(agent_id):
                if env.current_step % 100 == 0:
                    logging.info(f"🚫 [导弹控制-{agent_id}] 二次进攻未准备好，禁止导弹发射")
                continue
            else:
                if env.current_step % 100 == 0:
                    logging.info(f"✅ [导弹控制-{agent_id}] 二次进攻准备就绪，继续检查导弹发射请求")

        # 检查agent是否还活着
        if agent_id not in env.agents or not env.agents[agent_id].is_alive:
            logging.info(f"🔍 [导弹发射诊断-{agent_id}] 飞机不存在或已死亡，取消发射")
            self.state_manager.missile_launched[agent_id] = False
            continue

        # 检查冷却时间
        last_launch = self.state_manager.last_missile_launch_time.get(agent_id, -999)
        cooldown_remaining = self.state_manager.missile_cooldown - (current_time - last_launch)
        if current_time - last_launch < self.state_manager.missile_cooldown:
            # 注释掉详细日志
            # if env.current_step % 200 == 0:
            #     logging.info(f"🔍 [导弹发射诊断-{agent_id}] 冷却时间未到，剩余{cooldown_remaining:.1f}秒")
            continue

        # 检查导弹数量
        missiles_remaining = _get_aircraft_missiles_left(env.agents[agent_id])
        if missiles_remaining <= 0:
            logging.warning(f"⚠️ {agent_id} 导弹已用尽")
            self.state_manager.missile_launched[agent_id] = False
            continue

        # 🔥 修复：注释掉详细诊断日志
        # if env.current_step % 200 == 0:
        #     logging.info(f"🔍 [导弹发射诊断-{agent_id}] 基本条件检查通过: 剩余导弹={missiles_remaining}, 冷却时间={current_time - last_launch:.1f}s")

        # 寻找目标
        from core.target_assignment import get_target_with_fallback
        target_id = get_target_with_fallback(agent_id, env)
        target = env.agents.get(target_id) if target_id in env.agents else None

        # 🔥 注释掉详细调试日志
        # if env.current_step % 100 == 0:
        #     logging.info(f"🔍 [导弹发射诊断-{agent_id}] 目标检查: target_id={target_id}, target_alive={target.is_alive if target else False}")

        if target is None or not target.is_alive:
            if env.current_step % 60 == 0:
                logging.warning(f"⚠️ {agent_id} 未找到有效目标或目标已死亡 (目标ID:{target_id})，取消发射请求")
            self.state_manager.missile_launched[agent_id] = False
            if hasattr(self, '_pending_second_missile') and agent_id in self._pending_second_missile:
                del self._pending_second_missile[agent_id]
            continue

        # ✅ 大幅放宽：节点/模板会频繁提出发射请求，最终发射由 missile_manager + 雷达系统决定。
        # 这里不再做“过早的航向拦截”，否则会造成“请求发射很多但真正发射很少”。
        # 仍保留一个极端背对的兜底（>150°）避免荒谬发射。
        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)

        # 统一使用 TacticalUtils.calculate_bearing 计算敌机方位（导航方位：北为0°，顺时针为正）
        from tactical_utils import TacticalUtils
        target_bearing = TacticalUtils.calculate_bearing(env.agents[agent_id], target)
        # 计算航向与目标方位的夹角（统一坐标系）
        heading_error = abs(self._normalize_angle_diff(target_bearing - current_heading))

        # 🔥 修复5：严格限制航向偏差，防止背对发射
        # 只有在正面或雷达锁定照射时才允许发射导弹
        current_phase = self.state_manager.get_agent_phase(agent_id)
        is_second_attack_status = self.is_agent_second_attack(agent_id)

        # 🔥 修复：二次进攻时更严格的朝向检查，不允许背对发射
        if is_second_attack_status:
            if heading_error > 45.0:  # 二次进攻必须正对敌机（45度以内）
                if env.current_step % 100 == 0:
                    logging.warning(f"⚠️ [{agent_id}] 二次进攻背对敌机禁止发射: 角度{heading_error:.1f}° > 45°")
                continue
        elif heading_error > 150.0:  # 其他情况：严重背对（>150度）禁止发射
            continue

        # 在发射前执行完整条件检查
        current_phase = self.state_manager.get_agent_phase(agent_id)
        is_second_attack_status = self.is_agent_second_attack(agent_id)
        should_launch = self.missile_manager.should_launch_missile(env, agent_id, current_phase, target_id, 
                                                                   is_second_attack=is_second_attack_status)

        # 🔥 注释掉详细调试日志
        # if env.current_step % 100 == 0:
        #     logging.info(f"🔍 [导弹发射诊断-{agent_id}] 当前阶段={current_phase.value if current_phase else 'None'}")
        #     logging.info(f"🔍 [导弹发射诊断-{agent_id}] should_launch_missile返回={should_launch}")

        if not should_launch:
            gate_reason = "launch_gate_blocked"
            if hasattr(self.missile_manager, "last_gate_decision"):
                gate_decision = self.missile_manager.last_gate_decision.get(agent_id, {})
                gate_reason = gate_decision.get("reason", gate_reason)
            shooter_id = getattr(env.agents[agent_id], "real_id", agent_id)
            _record_external_gate_block(self, env, shooter_id, target, current_time, gate_reason)
            # 🔥 修复：打印不发射的详细原因
            if env.current_step % 100 == 0:
                # 检查各项条件
                reasons = []
                if missiles_remaining <= 0:
                    reasons.append("导弹已用尽")
                if current_time - last_launch < self.state_manager.missile_cooldown:
                    reasons.append(f"冷却时间未到(剩余{self.state_manager.missile_cooldown - (current_time - last_launch):.1f}秒)")
                if target is None or not target.is_alive:
                    reasons.append("目标不存在或已死亡")
                if heading_error > 45.0 and is_second_attack_status:
                    reasons.append(f"二次进攻背对(角度{heading_error:.1f}°)")
                elif heading_error > 150.0:
                    reasons.append(f"严重背对(角度{heading_error:.1f}°)")
                if not reasons:
                    reasons.append("missile_manager.should_launch_missile()返回False")
                # 🔥 修复：计算距离，避免NameError
                if target is not None:
                    distance = self._calculate_distance_between(env.agents[agent_id], target)
                    logging.warning(f"⛔ [{agent_id}] 导弹发射被阻止: {', '.join(reasons)}, 阶段={current_phase.value if current_phase else 'None'}, 距离={distance/1000:.1f}km")
                else:
                    logging.warning(f"⛔ [{agent_id}] 导弹发射被阻止: {', '.join(reasons)}, 阶段={current_phase.value if current_phase else 'None'}")
            continue

        # 🔥 只保留关键发射信息
        # logging.info(f"🚀 [导弹发射诊断-{agent_id}] 所有条件满足，即将发射导弹！")

        # 确保计数器已初始化
        if not hasattr(self, 'aircraft_missile_counts'):
            self.aircraft_missile_counts = {}
        if agent_id not in self.aircraft_missile_counts:
            self.aircraft_missile_counts[agent_id] = 0

        is_second_attack = self.is_agent_second_attack(agent_id)
        fired_count = self.aircraft_missile_counts.get(agent_id, 0)
        desired_salvo_size = _get_desired_salvo_size_for_request(
            self, env, agent_id, target_id, is_second_attack
        )
        allow_followup = desired_salvo_size >= 2 and missiles_remaining >= 2
        has_pending_followup = hasattr(self, '_pending_second_missile') and agent_id in self._pending_second_missile

        if has_pending_followup:
            pending_info = self._pending_second_missile.get(agent_id, {})
            if current_time < float(pending_info.get('launch_time', current_time)):
                continue

        self._launch_missile(env, agent_id, target, current_time)

        if allow_followup and not has_pending_followup:
            import random
            delay = random.uniform(1.5, 3.0)
            if not hasattr(self, '_pending_second_missile'):
                self._pending_second_missile = {}
            self._pending_second_missile[agent_id] = {
                'target': target,
                'launch_time': current_time + delay,
                'target_id': target_id
            }
            attack_type = "二次进攻" if is_second_attack else "第一次进攻"
            logging.info(f"🚀 [{agent_id}] {attack_type}：条件满足，{delay:.1f}秒后允许补射第2枚")
        else:
            self.state_manager.missile_launched[agent_id] = False

def _launch_missile(self, env, agent_id: str, target, current_time: float):
    """发射导弹 - 使用正确的计数器避免ID重复"""
    try:
        if _should_use_shared_cap_launch_path(self, env, agent_id):
            try:
                self.state_manager.missile_launched[agent_id] = False
            except Exception:
                pass
            try:
                if hasattr(self, '_pending_second_missile'):
                    self._pending_second_missile.pop(agent_id, None)
            except Exception:
                pass
            if env is not None and getattr(env, 'current_step', 0) % 180 == 0:
                logging.info(
                    f"🔗 [CAP_LAUNCH_BRIDGE] {agent_id} 拦截旧战术造弹入口，避免重复生成导弹"
                )
            return None
        from envs.JSBSim.core.simulatior import MissileSimulator
        # ✅ 我方导弹切换为 R-27ER（按用户要求与敌方对调）
        try:
            from simulation.r27er_missile import R27ERMissileSimulator
        except Exception:
            R27ERMissileSimulator = None

        aircraft = env.agents[agent_id]

        # **修复导弹ID生成 - 使用独立计数器**
        # 初始化计数器（与missile_manager共享）
        if not hasattr(self, 'aircraft_missile_counts'):
            self.aircraft_missile_counts = {}
        if agent_id not in self.aircraft_missile_counts:
            self.aircraft_missile_counts[agent_id] = 0

        # **同步missile_manager的计数器**
        if hasattr(self.missile_manager, 'aircraft_missile_counts'):
            # 使用missile_manager的计数器（如果存在）
            if agent_id not in self.missile_manager.aircraft_missile_counts:
                self.missile_manager.aircraft_missile_counts[agent_id] = 0
            current_count = self.missile_manager.aircraft_missile_counts[agent_id]
            self.aircraft_missile_counts[agent_id] = current_count

        # **严格4枚导弹限制检查**
        MAX_MISSILES_PER_AIRCRAFT = 4
        if self.aircraft_missile_counts[agent_id] >= MAX_MISSILES_PER_AIRCRAFT:
            logging.warning(f"⛔ [{agent_id}] 已发射{self.aircraft_missile_counts[agent_id]}枚导弹，达到上限{MAX_MISSILES_PER_AIRCRAFT}枚，取消发射")
            return

        # 🔥 修复：第一次进攻和二次进攻都分批发射（每次只发射1枚）
        # 注意：分批发射逻辑在_handle_missile_launches中处理，这里只发射1枚
        launch_count = 1

        # 发射导弹
        for i in range(launch_count):
            # 检查是否超过上限
            if self.aircraft_missile_counts[agent_id] >= MAX_MISSILES_PER_AIRCRAFT:
                break

            # 递增计数并生成唯一ID
            self.aircraft_missile_counts[agent_id] += 1
            missile_sequence = self.aircraft_missile_counts[agent_id]

            # **同步更新missile_manager的计数器**
            if hasattr(self.missile_manager, 'aircraft_missile_counts'):
                self.missile_manager.aircraft_missile_counts[agent_id] = self.aircraft_missile_counts[agent_id]

            # 尝试获取真实ID以防止同为Virtual A0100生成重复Missile ID
            real_id = getattr(aircraft, 'real_id', agent_id)
            base_id = real_id[0] + real_id[2:]  # A0100 → A100, A0300 → A300
            missile_uid = f"{base_id}{missile_sequence:0>2}"  # A100 → A10001, A300 → A30001

            # 我方(A*)：优先使用R-27ER仿真器；若不可用则回退引擎导弹工厂
            if agent_id.startswith('A') and R27ERMissileSimulator is not None:
                missile = R27ERMissileSimulator.create(parent=aircraft, target=target, uid=missile_uid)
            else:
                missile = MissileSimulator.create(parent=aircraft, target=target, uid=missile_uid)

            # 🔥 修复：检查导弹初始状态，确保不是立即完成状态
            if hasattr(missile, 'is_done') and missile.is_done:
                logging.error(f"⚠️ [{agent_id}] 导弹{missile_uid}创建时即为完成状态，跳过添加")
                # 恢复导弹计数
                self.aircraft_missile_counts[agent_id] -= 1
                if hasattr(self.missile_manager, 'aircraft_missile_counts'):
                    self.missile_manager.aircraft_missile_counts[agent_id] = self.aircraft_missile_counts[agent_id]
                max_missiles = int(getattr(getattr(self, 'missile_manager', None), 'MAX_MISSILES_PER_AIRCRAFT', 4))
                remaining_missiles = max(0, max_missiles - self.aircraft_missile_counts[agent_id])
                _sync_aircraft_missile_state(self, aircraft, agent_id, remaining_missiles)
                continue

            # 添加到环境
            env.add_temp_simulator(missile)
            # ✅ 设置环境引用，以便导弹可以记录日志
            if hasattr(missile, '_env'):
                missile._env = env
            # 记录到环境导弹表，供威胁评估/RWR使用
            if not hasattr(env, 'missiles') or env.missiles is None:
                env.missiles = {}
            env.missiles[missile_uid] = missile

            # 记录导弹
            if not hasattr(self, 'missiles'):
                self.missiles = {}
            self.missiles[missile_uid] = missile

            # 更新发射时间和计数
            self.state_manager.last_missile_launch_time[agent_id] = current_time
            self.state_manager.missiles_fired[agent_id] = self.aircraft_missile_counts[agent_id]
            max_missiles = int(getattr(getattr(self, 'missile_manager', None), 'MAX_MISSILES_PER_AIRCRAFT', 4))
            remaining_missiles = max(0, max_missiles - self.aircraft_missile_counts[agent_id])
            _sync_aircraft_missile_state(self, aircraft, agent_id, remaining_missiles)
            target_log_id = _resolve_external_target_id(target) or getattr(target, 'uid', None) or getattr(target, 'target_id', None) or ""
            target_env_key = ""
            for collection_name in ('agents', '_jsbsims'):
                collection = getattr(env, collection_name, None)
                if not isinstance(collection, dict):
                    continue
                for candidate_id, candidate in collection.items():
                    if candidate is target:
                        target_env_key = str(candidate_id)
                        break
                if target_env_key:
                    break
            target_display_id = target_env_key or target_log_id or getattr(target, 'uid', None) or getattr(target, 'target_id', None) or ""
            _record_external_launch_event(self, env, getattr(aircraft, 'real_id', agent_id), target, missile_uid, current_time)
            if hasattr(self, 'missile_manager') and hasattr(self.missile_manager, 'record_external_missile_launch'):
                try:
                    self.missile_manager.record_external_missile_launch(
                        env=env,
                        launcher_id=getattr(aircraft, 'real_id', agent_id),
                        missile_id=missile_uid,
                        target_id=target_display_id,
                        target_aircraft=target,
                        current_time=current_time,
                    )
                except Exception:
                    pass

            # 计算距离和角度用于日志
            my_pos = aircraft.get_position()
            target_pos = target.get_position()
            distance = np.linalg.norm(np.array(target_pos) - np.array(my_pos))

            # 计算发射角度（theta和phi）
            los = np.array(target_pos) - np.array(my_pos)
            los_norm = np.linalg.norm(los)
            if los_norm > 0:
                los_unit = los / los_norm
                theta = np.arcsin(los_unit[2])  # 俯仰角（度）
                phi = np.arctan2(los_unit[1], los_unit[0])  # 偏航角（度）
                theta_deg = np.rad2deg(theta)
                phi_deg = np.rad2deg(phi)
            else:
                theta_deg = 0.0
                phi_deg = 0.0

            # 获取发射速度
            vel = aircraft.get_velocity()
            velocity_mag = np.linalg.norm(vel)
            altitude = my_pos[2]

            # **显示导弹计数信息 + 导弹型号验证**
            missile_model = getattr(missile, "model", type(missile).__name__)
            shooter_log_id = real_id
            try:
                logging.info(
                    f"[MISSILE] {shooter_log_id} launch {missile_uid}({missile_model}) -> {target_display_id or target.uid} "
                    f"range={distance/1000:.1f}km count={missile_sequence}/{MAX_MISSILES_PER_AIRCRAFT}"
                )
                logging.info(
                    f"[MISSILE] {missile_model} {missile_uid} kinematics: v={velocity_mag:.1f}m/s alt={altitude:.0f}m"
                )
                logging.info(
                    f"[MISSILE] {missile_model} {missile_uid} target={target_display_id or target.uid} theta={theta_deg:.1f}deg phi={phi_deg:.1f}deg"
                )
            except Exception:
                pass

            # ✅ 记录到trace_logger：导弹发射原因和条件
            try:
                from utils.trace_logger import trace_event
                from tactical_utils import TacticalUtils

                # 获取发射原因（从last_gate_decision）
                launch_reason = "unknown"
                launch_details = {}
                allow_reasons_list = []
                if hasattr(self.missile_manager, 'last_gate_decision') and agent_id in self.missile_manager.last_gate_decision:
                    gate_decision = self.missile_manager.last_gate_decision[agent_id]
                    launch_reason = gate_decision.get('reason', 'unknown')
                    launch_details = gate_decision.get('details', {})
                    # 提取允许发射理由列表（更易读）
                    allow_reasons_list = launch_details.get('允许发射理由', [])
                    if allow_reasons_list is None:
                        allow_reasons_list = []

                # 计算航向误差
                target_bearing = TacticalUtils.calculate_bearing(aircraft, target)
                current_heading = aircraft.get_property_value(c.attitude_psi_deg)
                heading_error = abs(TacticalUtils.normalize_angle_diff(target_bearing - current_heading))

                # 获取当前阶段
                current_phase = self.agent_phases.get(agent_id, None)
                phase_str = current_phase.value if current_phase else 'None'

                # 检查是否是二次进攻
                is_second_attack = self.is_agent_second_attack(agent_id) if hasattr(self, 'is_agent_second_attack') else False

                # 构建发射原因说明（更易读）
                reason_desc = {
                    "allowed_all_gates": "所有门限通过（距离、朝向、雷达、冷却、配额）",
                    "allowed_second_attack_radar_relaxed": "二次进攻雷达放宽（距离和朝向合适）",
                    "allowed_soft_channel_radar_failed": "强软发射通道（雷达未通过但距离和朝向合适）",
                    "unknown": "未知原因"
                }
                reason_description = reason_desc.get(launch_reason, launch_reason)

                # 构建满足条件的说明
                conditions_met = []
                if allow_reasons_list:
                    conditions_met = allow_reasons_list
                elif launch_details:
                    # 如果没有理由列表，从details中提取关键信息
                    if launch_details.get('距离_km'):
                        conditions_met.append(f"距离={launch_details.get('距离_km'):.1f}km")
                    if launch_details.get('航向误差_deg') is not None:
                        conditions_met.append(f"航向误差={launch_details.get('航向误差_deg'):.1f}°")
                    if launch_details.get('radar_check_passed'):
                        conditions_met.append("雷达门限通过")
                    elif launch_details.get('radar_check_reason'):
                        conditions_met.append(f"雷达状态: {launch_details.get('radar_check_reason')}")

                trace_event(
                    事件="导弹发射",
                    env=env,
                    模块="tactical_task",
                    类型="ACTION",
                    状态="OK",
                    我机=agent_id,
                    敌机=target_log_id or target.uid,
                    阶段=phase_str,
                    战术=str(self.selected_tactic),
                    说明=f"{missile_model} {missile_uid} launched: v={velocity_mag:.1f}m/s, alt={altitude:.0f}m, theta={theta_deg:.1f}°, phi={phi_deg:.1f}° | 发射原因: {reason_description} | 满足条件: {', '.join(conditions_met) if conditions_met else '未知'}",
                    数据={
                        "导弹型号": missile_model,
                        "导弹ID": missile_uid,
                        "目标": target_log_id or target.uid,
                        "距离_km": round(distance / 1000.0, 1),
                        "速度_m/s": round(velocity_mag, 1),
                        "高度_m": round(altitude, 0),
                        "俯仰角_deg": round(theta_deg, 1),
                        "偏航角_deg": round(phi_deg, 1),
                        "发射序号": int(missile_sequence),
                        "总导弹数": int(MAX_MISSILES_PER_AIRCRAFT),
                        "发射原因": str(launch_reason),
                        "发射原因说明": reason_description,
                        "满足条件": conditions_met,
                        "航向误差_deg": round(heading_error, 1),
                        "二次进攻": bool(is_second_attack),
                        **launch_details,
                    },
                )
            except Exception as e:
                logging.debug(f"⚠️ [导弹发射记录异常] {agent_id}: {e}")

    except Exception as e:
        logging.error(f"[{agent_id}] 导弹发射失败: {e}")

def normalize_action(self, env, agent_id, action):
    """
    战术模板核心方法 - 动作归一化
    这是环境调用的主入口，将战术指令转换为底层控制
    """
    if agent_id not in env.agents:
        return np.array([0.0, 0.0, 0.0, 0.7])

    runtime_recovered_before_action = False
    if not env.agents[agent_id].is_alive:
        if agent_id.startswith("A"):
            runtime_recovered_before_action = _maybe_runtime_recreate_t2v2_simulator(
                self,
                env,
                agent_id,
                friendly=True,
            )
        elif agent_id.startswith("B"):
            runtime_recovered_before_action = _maybe_runtime_recreate_t2v2_simulator(
                self,
                env,
                agent_id,
                friendly=False,
            )
        if (not runtime_recovered_before_action) or agent_id not in env.agents or not env.agents[agent_id].is_alive:
            return np.array([0.0, 0.0, 0.0, 0.7])

    if agent_id.startswith("A") and not runtime_recovered_before_action:
        recreated = _maybe_recreate_t2v2_simulator(self, env, agent_id, friendly=True)
        if not recreated:
            recreated = _maybe_runtime_recreate_t2v2_simulator(self, env, agent_id, friendly=True)
        if recreated and (agent_id not in env.agents or not env.agents[agent_id].is_alive):
            return np.array([0.0, 0.0, 0.0, 0.7])

    if agent_id.startswith("B") and not runtime_recovered_before_action:
        recreated = _maybe_recreate_t2v2_simulator(self, env, agent_id, friendly=False)
        if not recreated:
            recreated = _maybe_runtime_recreate_t2v2_simulator(self, env, agent_id, friendly=False)
        if recreated and (agent_id not in env.agents or not env.agents[agent_id].is_alive):
            return np.array([0.0, 0.0, 0.0, 0.7])

    current_time = env.current_step * env.time_interval
    alive_enemy_exists = self._has_alive_enemy_anywhere(env, agent_id)

    # 初始化agent状态
    if agent_id not in self.initial_heading:
        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_rad)
        self.initial_heading[agent_id] = np.rad2deg(current_heading)
        current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        self.initial_altitude[agent_id] = current_altitude
        logging.info(f"{agent_id} 初始状态: 航向{self.initial_heading[agent_id]:.1f}°, 高度{self.initial_altitude[agent_id]:.1f}m")

    # 获取战术指令索引（调用get_action）
    action_result = self.get_action(env, agent_id)
    if action_result is None:
        logging.error(f"🚨 {agent_id} get_action返回None，使用默认动作(7,8,3)")
        altitude_cmd_id, heading_cmd_id, velocity_cmd_id = 7, 8, 3
    elif not isinstance(action_result, tuple) or len(action_result) != 3:
        logging.error(f"🚨 {agent_id} get_action返回格式错误: {action_result}，使用默认动作(7,8,3)")
        altitude_cmd_id, heading_cmd_id, velocity_cmd_id = 7, 8, 3
    else:
        altitude_cmd_id, heading_cmd_id, velocity_cmd_id = action_result

        if agent_id.startswith('A'):
            altitude_cmd_id, heading_cmd_id, velocity_cmd_id, _ = _apply_t2v2_recreate_command_guard(
                self,
                env,
                agent_id,
                altitude_cmd_id,
                heading_cmd_id,
                velocity_cmd_id,
            )
        
        if False:
            try:
                current_altitude = float(env.agents[agent_id].get_property_value(c.position_h_sl_m))
            except Exception:
                current_altitude = float('nan')
            max_altitude = _friendly_max_altitude_m()
            logging.info(
                "[ALT_CAP] %s 当前高度=%.1fm，限制高度指令为 alt=%d 以保持不超过%.0fm",
                agent_id,
                current_altitude,
                altitude_cmd_id,
                max_altitude,
            )
        # 🔥 调试：只打印我方（A开头）的日志，每60步（约3秒）打印一次
        if agent_id.startswith('A') and env.current_step % 60 == 0:
            logging.warning(f"✅ [动作追踪-{agent_id}] get_action返回: {action_result}, 战术={self.selected_tactic}")

        # ✅ 记录到trace_logger：get_action返回（只在动作变化或节点切换时记录）
        if agent_id.startswith('A'):
            try:
                # 检查动作是否变化
                if not hasattr(self, '_last_logged_action'):
                    self._last_logged_action = {}
                last_action = self._last_logged_action.get(agent_id)
                current_action = tuple(action_result) if isinstance(action_result, (list, tuple)) else action_result

                should_log = False
                if last_action != current_action:
                    should_log = True
                    self._last_logged_action[agent_id] = current_action

                # 检查阶段是否变化
                current_phase = self.agent_phases.get(agent_id, None)
                phase_str = current_phase.value if current_phase else 'None'
                if not hasattr(self, '_last_action_phase'):
                    self._last_action_phase = {}
                if agent_id not in self._last_action_phase or self._last_action_phase[agent_id] != phase_str:
                    should_log = True
                    self._last_action_phase[agent_id] = phase_str

                if should_log:
                    from utils.trace_logger import trace_event
                    action_str = f"alt={action_result[0]},hdg={action_result[1]},vel={action_result[2]}" if isinstance(action_result, (list, tuple)) and len(action_result) >= 3 else str(action_result)
                    trace_event(
                        事件="get_action返回",
                        env=env,
                        模块="tactical_task",
                        类型="ACTION",
                        状态="RETURN",
                        我机=agent_id,
                        阶段=phase_str,
                        战术=str(self.selected_tactic),
                        说明=f"动作={action_str}",
                    )
            except Exception:
                pass

    # 使用baseline模型生成底层控制
    # 🔥 调试：只打印我方（A开头）的日志，每60步（约3秒）打印一次
    if agent_id.startswith('A') and env.current_step % 60 == 0:
        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
        logging.warning(f"🔍 [底层控制-{agent_id}] 输入动作: alt={altitude_cmd_id}, hdg={heading_cmd_id}, vel={velocity_cmd_id}, 当前航向={current_heading:.1f}°")

    # ✅ 记录到trace_logger：底层控制输入（只在输入变化时记录）
    if agent_id.startswith('A'):
        try:
            # 检查输入是否变化
            if not hasattr(self, '_last_logged_control_input'):
                self._last_logged_control_input = {}
            last_input = self._last_logged_control_input.get(agent_id)
            current_input = (altitude_cmd_id, heading_cmd_id, velocity_cmd_id)

            if last_input != current_input:
                from utils.trace_logger import trace_event
                trace_event(
                    事件="底层控制-输入动作",
                    env=env,
                    模块="tactical_task",
                    类型="ACTION",
                    状态="INPUT",
                    我机=agent_id,
                    说明=f"alt={altitude_cmd_id},hdg={heading_cmd_id},vel={velocity_cmd_id},当前航向={current_heading:.1f}°",
                )
                self._last_logged_control_input[agent_id] = current_input
        except Exception:
            pass

    result = self._use_lowlevel_policy(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)
    if agent_id.startswith('A') and env.current_step % 60 == 0:
        logging.warning(f"🔍 [底层控制-{agent_id}] 输出控制: {result}")

    # ✅ 记录到trace_logger：底层控制输出（每8秒约40步记录一次，降低频率）
    if agent_id.startswith('A'):
        try:
            # 初始化记录字典
            if not hasattr(self, '_last_logged_control_output'):
                self._last_logged_control_output = {}
            if not hasattr(self, '_last_control_output_step'):
                self._last_control_output_step = {}

            last_output = self._last_logged_control_output.get(agent_id)
            last_step = self._last_control_output_step.get(agent_id, -999)
            current_step = env.current_step

            # 每40步（约8秒）记录一次，或者首次记录
            should_log = False
            if last_output is None:
                should_log = True
            elif current_step - last_step >= 40:
                should_log = True

            if should_log:
                from utils.trace_logger import trace_event
                result_str = f"[{result[0]:.2f}, {result[1]:.2f}, {result[2]:.2f}, {result[3]:.2f}]" if isinstance(result, (list, tuple, np.ndarray)) and len(result) >= 4 else str(result)
                trace_event(
                    事件="底层控制-输出控制",
                    env=env,
                    模块="tactical_task",
                    类型="ACTION",
                    状态="RETURN",
                    我机=agent_id,
                    说明=f"控制={result_str}",
                )
                if isinstance(result, (list, tuple, np.ndarray)) and len(result) >= 4:
                    self._last_logged_control_output[agent_id] = result[:4]
                self._last_control_output_step[agent_id] = current_step
        except Exception:
            pass
    return result

def _use_lowlevel_policy(self, env, agent_id: str, altitude_cmd_id: int, heading_cmd_id: int, velocity_cmd_id: int) -> np.ndarray:
    """
    使用低层模型生成底层控制

    - 默认我方使用 F16 BaselineActor，敌方使用 SU27 PPOActor
    - 如果指定模型不可用，则自动回退到 F16 BaselineActor
    """
    # 根据阵营选择目标模型类型
    is_enemy = agent_id.startswith('B')
    preferred_type = getattr(self, 'enemy_lowlevel_type' if is_enemy else 'friend_lowlevel_type', 'F16').upper()

    use_su27 = (preferred_type == 'SU27' and getattr(self, 'su27_baseline_actor', None) is not None)
    use_f16 = (not use_su27 and getattr(self, 'baseline_model', None) is not None)

    if not use_su27 and not use_f16:
        # 没有任何低层模型可用，退回默认控制
        return np.array([0.0, 0.0, 0.0, 0.7])

    try:
        # 1. 获取原始观测
        raw_obs = self.get_obs(env, agent_id)

        # 2. 构造12维输入
        input_obs = np.zeros(12)
        altitude_cmd_id = min(altitude_cmd_id, len(self.norm_delta_altitude) - 1)
        heading_cmd_id = min(heading_cmd_id, len(self.norm_delta_heading) - 1)
        velocity_cmd_id = min(velocity_cmd_id, len(self.norm_delta_velocity) - 1)

        # 3. 初始化RNN状态
        if agent_id not in self._inner_rnn_states:
            self._inner_rnn_states[agent_id] = np.zeros((1, 1, 128), dtype=np.float32)

        # 4. 模型推理
        if use_su27:
            # SU27 BaselineActor 路径（与 verify_su27_tactical_compliance_v2.py 保持一致）
            aircraft = env.agents[agent_id]

            # 获取属性 (使用Body Frame速度 u,v,w)
            h_sl_m = aircraft.get_property_value(c.position_h_sl_m)
            roll_rad = aircraft.get_property_value(c.attitude_roll_rad)
            pitch_rad = aircraft.get_property_value(c.attitude_pitch_rad)
            u_mps = aircraft.get_property_value(c.velocities_u_mps)
            v_mps = aircraft.get_property_value(c.velocities_v_mps)
            w_mps = aircraft.get_property_value(c.velocities_w_mps)
            vc_mps = aircraft.get_property_value(c.velocities_vc_mps)

            # 构造SU27特定的raw_obs (9维) - 严格匹配验证脚本v2
            su27_raw_obs = np.zeros(9)
            su27_raw_obs[0] = h_sl_m / 5000.0
            su27_raw_obs[1] = np.sin(roll_rad)
            su27_raw_obs[2] = np.cos(roll_rad)
            su27_raw_obs[3] = np.sin(pitch_rad)
            su27_raw_obs[4] = np.cos(pitch_rad)
            su27_raw_obs[5] = u_mps / 340.0
            su27_raw_obs[6] = v_mps / 340.0
            su27_raw_obs[7] = w_mps / 340.0
            su27_raw_obs[8] = vc_mps / 340.0

            input_obs[0] = self.norm_delta_altitude[altitude_cmd_id]
            input_obs[1] = self.norm_delta_heading[heading_cmd_id]
            input_obs[2] = self.norm_delta_velocity[velocity_cmd_id]
            input_obs[3:12] = su27_raw_obs
            input_obs = np.nan_to_num(input_obs, nan=0.0)

            obs_tensor = torch.FloatTensor(input_obs).unsqueeze(0)
            rnn_states_tensor = torch.FloatTensor(self._inner_rnn_states[agent_id])

            with torch.no_grad():
                actions, rnn_states_out = self.su27_baseline_actor(
                    obs_tensor,
                    rnn_states_tensor
                )

            action_output = actions.detach().cpu().numpy()[0]
            self._inner_rnn_states[agent_id] = rnn_states_out.detach().cpu().numpy()
        else:
            # F16 BaselineActor 路径（原始实现）
            input_obs[0] = self.norm_delta_altitude[altitude_cmd_id]
            input_obs[1] = self.norm_delta_heading[heading_cmd_id]
            input_obs[2] = self.norm_delta_velocity[velocity_cmd_id]
            input_obs[3:12] = raw_obs[:9]
            input_obs = np.nan_to_num(input_obs, nan=0.0)

            obs_expanded = np.expand_dims(input_obs, axis=0)
            _action, _rnn_states = self.baseline_model(
                torch.FloatTensor(obs_expanded),
                torch.FloatTensor(self._inner_rnn_states[agent_id])
            )
            action_output = _action.detach().cpu().numpy().squeeze(0)
            self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()

        # 5. 归一化输出（两种模型共享同一映射）
        norm_act = np.zeros(4)
        norm_act[0] = action_output[0] / 20 - 1.
        norm_act[1] = action_output[1] / 20 - 1.
        norm_act[2] = action_output[2] / 20 - 1.
        norm_act[3] = action_output[3] / 58 + 0.4

        # 6. 安全检查：低高度保护（增强版 - 防止倒飞坠毁）
        current_alt = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        roll_rad = env.agents[agent_id].get_property_value(c.attitude_roll_rad)

        if current_alt < 3000:
            if current_alt < 1000:
                # 极低高度：全油门
                norm_act[3] = 1.0

                # 检查滚转角，防止倒飞拉杆导致坠毁
                if abs(roll_rad) > np.deg2rad(60):
                    # 滚转角过大，优先改平
                    # 如果 roll > 0 (右倾), 需要左滚 (aileron < 0)
                    norm_act[0] = -np.sign(roll_rad) * 1.0 
                    # 此时不要过度拉杆，防止螺旋，保持适度拉杆维持机头
                    norm_act[1] = 0.0 
                    if env.current_step % 30 == 0:
                        logging.error(f"🛡️ [{agent_id}] 极低高度紧急改平: Alt={current_alt:.0f}m, Roll={np.rad2deg(roll_rad):.1f}°")
                else:
                    # 滚转角正常，全力拉升
                    norm_act[0] = 0.0 # 保持水平
                    norm_act[1] = -1.0 # 全力拉升
                    if env.current_step % 30 == 0:
                        logging.error(f"🛡️ [{agent_id}] 极低高度紧急拉升: Alt={current_alt:.0f}m")

            elif current_alt < 1500:
                # 低高度：强制较大爬升和全油门
                norm_act[1] = min(norm_act[1], -0.7)
                norm_act[3] = 1.0
                if env.current_step % 60 == 0:
                    logging.warning(f"🛡️ [{agent_id}] 紧急拉升: 高度{current_alt:.0f}m < 1500m")
            else:
                # 中低高度：优先爬升并提高油门
                norm_act[1] = min(norm_act[1], -0.3)
                norm_act[3] = max(norm_act[3], 0.9)
                if env.current_step % 120 == 0:
                    logging.info(f"🛡️ [{agent_id}] 高度保护: 高度{current_alt:.0f}m < 3000m")

        return norm_act

    except Exception as e:
        logging.error(f"[{agent_id}] Baseline模型执行错误: {e}")
        return np.array([0.0, 0.0, 0.0, 0.7])

def _get_dynamic_velocity_cmd(self, env, agent_id: str) -> int:
    """
    动态速度管理：根据当前速度选择合适的速度指令

    Args:
        env: 环境
        agent_id: 飞机ID

    Returns:
        velocity_cmd_id: 速度指令索引
            - 3: 保持当前速度 (delta=0)
            - 4: 轻微加速 (delta=+50m/s)
            - 5: 中等加速 (delta=+100m/s)
    """
    current_velocity = np.linalg.norm(env.agents[agent_id].get_velocity())

    # 阈值设计：保持速度在220-280 m/s范围内（降低阈值，减缓接近速度）
    VELOCITY_LOW_THRESHOLD = 220.0   # 低速阈值（从240降低到220）
    VELOCITY_HIGH_THRESHOLD = 280.0  # 高速阈值
    VELOCITY_CRITICAL_LOW = 200.0    # 危险低速阈值（从220降低到200）

    if current_velocity < VELOCITY_CRITICAL_LOW:
        # 危险低速：紧急加速
        return 5  # 中等加速 (+100m/s)
    elif current_velocity < VELOCITY_LOW_THRESHOLD:
        # 低速：轻微加速
        return 4  # 轻微加速 (+50m/s)
    elif current_velocity > VELOCITY_HIGH_THRESHOLD:
        # 高速：保持当前速度（让速度自然衰减）
        return 3  # 保持速度 (delta=0)
    else:
        # 正常范围（220-280 m/s）：保持当前速度
        return 3  # 保持速度 (delta=0)

def _format_phase_name(self, agent_id: str, phase) -> str:
    """格式化阶段名称为NLT1/MELD1等格式"""
    if phase is None:
        return 'N/A'

    # 判断是长机还是僚机
    is_lead = agent_id in ('A0100', 'A0300', 'B0100', 'B0300')
    role_suffix = '1' if is_lead else '2'

    # 判断是否二次进攻（按机）
    is_second = self.is_agent_second_attack(agent_id)

    # 阶段映射
    phase_map = {
        'NLT_MELD': 'NLT',
        'MELD_MTR': 'MELD',
        'MTR_LR': 'MTR',
        'LR_TR': 'LR',
        'TR_DOR': 'TR',
        'DOR_DR': 'DOR',
        'DR_MAR': 'DR',
        'BEYOND_MAR': 'MAR'
    }

    phase_str = phase.value if hasattr(phase, 'value') else str(phase)
    base_name = phase_map.get(phase_str, phase_str)

    # 如果是二次进攻的MTR/LR/TR阶段，添加'标记
    if is_second and base_name in ['MTR', 'LR', 'TR']:
        return f"{base_name}{role_suffix}'"
    else:
        return f"{base_name}{role_suffix}"

def _update_phase(self, env, agent_id: str, current_time: float):
    """更新飞机所处的战术阶段（原版逻辑：僚机滞后）"""
    # ✅ 距离判定必须是“最近存活敌机距离”（1v2 等场景不能只盯一个目标）
    enemy_prefix = "B" if agent_id.startswith("A") else "A"
    candidates = []
    enemy_ids = []
    try:
        enemy_ids = list(get_enemy_team(agent_id))
    except Exception:
        enemy_ids = []
    if not enemy_ids:
        enemy_ids = [eid for eid in getattr(env, 'agents', {}).keys() if str(eid).startswith(enemy_prefix)]

    for eid in enemy_ids:
        try:
            e = env._jsbsims.get(eid) if hasattr(env, "_jsbsims") else None
            if e is None:
                e = env.agents.get(eid) if hasattr(env, "agents") else None
            if e is None:
                continue
            if hasattr(e, "is_alive") and (not e.is_alive):
                continue
            d = TacticalUtils.calculate_distance_between(env.agents[agent_id], e)
            candidates.append((d, eid))
        except Exception:
            continue

    if not candidates:
        return

    distance, nearest_enemy_id = min(candidates, key=lambda x: x[0])
    adjusted_distance = distance  # 初始化为实际距离

    # 🔥 修复问题4&5: 检测返航状态，防止阶段回退导致的异常行为
    if self._is_in_rtb_mode(env, agent_id, distance, current_time):
        # 返航状态：不更新阶段，保持当前状态
        return

    # 🔥 修复问题2: 移除僚机由于 assigned tactic 加上的 4000/8000 距离补偿
    # 让实际几何距离成为唯一跨入战术阶段的标准，防止跨阶段反应迟钝
    is_lead = self._is_formation_lead(env, agent_id)

    # 🔥 修复问题4: 正确的阶段切换逻辑 - 从大到小判断距离（包含 MELD_MTR）
    if distance >= self.tactical_distances['NLT']:
        new_phase = TacticalPhase.BEYOND_NLT
    elif distance >= self.tactical_distances['MELD']:
        new_phase = TacticalPhase.NLT_MELD
    elif distance >= self.tactical_distances['MTR']:
        new_phase = TacticalPhase.MELD_MTR
    elif adjusted_distance >= self.tactical_distances['LR']:
        new_phase = TacticalPhase.MTR_LR
    elif adjusted_distance >= self.tactical_distances['TR']:
        new_phase = TacticalPhase.LR_TR
    elif adjusted_distance >= self.tactical_distances['DOR']:
        new_phase = TacticalPhase.TR_DOR
    elif adjusted_distance >= self.tactical_distances['DR']:
        new_phase = TacticalPhase.DOR_DR
    elif adjusted_distance >= self.tactical_distances['MAR']:
        new_phase = TacticalPhase.DR_MAR
    else:
        new_phase = TacticalPhase.BEYOND_MAR

    old_phase = self.state_manager.get_agent_phase(agent_id)
    boundary_hysteresis_m = 3500.0 if agent_id.startswith('A') else 2500.0
    phase_hysteresis_specs = [
        (TacticalPhase.LR_TR, TacticalPhase.TR_DOR, float(self.tactical_distances['TR'])),
        (TacticalPhase.TR_DOR, TacticalPhase.DOR_DR, float(self.tactical_distances['DOR'])),
        (TacticalPhase.DOR_DR, TacticalPhase.DR_MAR, float(self.tactical_distances['DR'])),
    ]
    for upper_phase, lower_phase, boundary_m in phase_hysteresis_specs:
        if old_phase == lower_phase and boundary_m <= adjusted_distance < (boundary_m + boundary_hysteresis_m):
            new_phase = lower_phase
            break
        if old_phase == upper_phase and (boundary_m - boundary_hysteresis_m) < adjusted_distance < boundary_m:
            new_phase = upper_phase
            break

    # 🔥 新增：DR窗口期间锁定阶段，防止被MAR打断
    if old_phase == TacticalPhase.DR_MAR:
        # 检查DR窗口是否还在进行
        if hasattr(self.node_decision_maker, 'agent_dr_windows'):
            if agent_id in self.node_decision_maker.agent_dr_windows:
                window_start = self.node_decision_maker.agent_dr_windows[agent_id]
                elapsed = current_time - window_start
                if elapsed < 10.0:
                    # DR窗口期间，锁定在DR_MAR阶段，不允许切换
                    # 🔥 调试：只打印我方（A开头）的日志
                    if agent_id.startswith('A') and env.current_step % 60 == 0:
                        logging.info(f"🔒 [DR窗口锁定] {agent_id} DR窗口进行中({elapsed:.1f}s/10s)，锁定阶段")
                    return  # 不切换阶段
                # ✅ 通用修复：窗口刚结束时，必须先执行一次“窗口结束决策”
                # 否则会先离开DR_MAR进入BEYOND_MAR触发MAR强制防御，导致编队重整/二次进攻永远无法触发
                # 该逻辑不改变各战术机动细节，仅保证DR窗口结束的“评估/重攻/撤退”至少执行一次
                try:
                    if hasattr(self.node_decision_maker, 'agent_dr_decisions') and agent_id in self.node_decision_maker.agent_dr_decisions:
                        dr_state = self.node_decision_maker.agent_dr_decisions.get(agent_id, {})
                        if not dr_state.get('window_logged', False):
                            if env.current_step % 60 == 0:
                                logging.info(f"🧩 [DR窗口结束-强制决策] {agent_id} elapsed={elapsed:.1f}s，先执行DR窗口结束评估，避免被MAR强制防御抢占")
                            # 在离开DR阶段前，强制跑一次DR节点：
                            # - 打印“时间窗口结束，评估是否重新进攻”
                            # - 若同意重攻：_handle_dr_reengage_unified -> FORMATION_RESET/UNIFIED_SECOND_ATTACK
                            # - 若撤退：_handle_dr_retreat_unified
                            self._node_decision_at_node(env, agent_id, 'DR', current_time)
                            # 让新战术（FORMATION_RESET / TACTICAL_TURN 等）先接管至少一帧，避免立刻进入MAR强制防御
                            return
                except Exception:
                    pass

    # 🔥 强化：全局阶段锁定机制 - 严禁任何阶段回退
    # 定义阶段优先级（数值越大越后期）
    phase_priority = {
        TacticalPhase.BEYOND_NLT: 0,
        TacticalPhase.NLT_MELD: 1,
        TacticalPhase.MELD_MTR: 2,
        TacticalPhase.MTR_LR: 3,
        TacticalPhase.LR_TR: 4,
        TacticalPhase.TR_DOR: 5,
        TacticalPhase.DOR_DR: 6,
        TacticalPhase.DR_MAR: 7,
        TacticalPhase.BEYOND_MAR: 8
    }

    old_priority = phase_priority.get(old_phase, 0)
    new_priority = phase_priority.get(new_phase, 0)
    alive_enemy_exists = self._has_alive_enemy_anywhere(env, agent_id)
    if agent_id.startswith('A') and alive_enemy_exists and agent_id in getattr(self, 'returning_agents', set()):
        guard_release_until = float(getattr(self, '_red_guard_release_until', {}).get(agent_id, -999.0))
        guard_release_active = float(current_time) < guard_release_until
        current_tactic_name = str(self._get_agent_tactic(agent_id) or '')
        preserve_defense = bool(guard_release_active or current_tactic_name == 'DEFENSIVE_GUARD')
        if not preserve_defense and hasattr(self, '_evaluate_red_defensive_posture'):
            try:
                preserve_defense = bool(
                    self._evaluate_red_defensive_posture(
                        env,
                        agent_id,
                        current_phase_name="phase_update_guard_hold",
                        reason="phase_update_guard_hold",
                    ).get("preserve", False)
                )
            except Exception:
                preserve_defense = bool(guard_release_active or current_tactic_name == 'DEFENSIVE_GUARD')
        if preserve_defense:
            if hasattr(self, '_deescalate_red_formation_to_defense'):
                self._deescalate_red_formation_to_defense(
                    env,
                    agent_id,
                    reason="phase_update_guard_hold",
                    clear_returning=False,
                )
            if env.current_step % 50 == 0:
                logging.info("[阶段守区保留] %s 敌机仍存活，保留DEFENSIVE_GUARD，不清除RTB冷却链", agent_id)
            return
        formation_agents = self._get_alive_formation_agents(
            env,
            agent_id,
            preserve_nominal_pair=True,
        )
        for aid in formation_agents:
            self.returning_agents.discard(aid)
            try:
                self.executor.tactical_turn_states.pop(aid, None)
            except Exception:
                pass
        if env.current_step % 50 == 0:
            logging.warning(f"🛑 [{agent_id}] 敌机仍存活，清除残留RTB/TACTICAL_TURN状态，恢复循环进攻")
    allow_backward_phase = agent_id.startswith('A') and alive_enemy_exists

    # 🔒 严禁阶段回退 + 强制顺序推进（不允许跳阶段）
    if new_priority < old_priority:
        if not allow_backward_phase:
            if env.current_step % 50 == 0:
                logging.warning(f"🔒 [{agent_id}] 全局阶段锁定: 严禁从{old_phase.value}回退到{new_phase.value}，距离={distance/1000:.1f}km")
            return  # 保持当前阶段，不更新
        if env.current_step % 50 == 0:
            logging.info(
                f"🌀 [{agent_id}] 敌机仍存活，允许按距离回退阶段继续再攻击: "
                f"{old_phase.value} -> {new_phase.value}, 距离={distance/1000:.1f}km"
            )

    # 🔒 强制顺序推进：不允许跨越阶段（如 NLT_MELD 必须先到 MELD_MTR 才能到 MTR_LR）
    if new_priority > old_priority + 1:
        # 只前进一步
        sequential_phases = [
            TacticalPhase.BEYOND_NLT, TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR,
            TacticalPhase.MTR_LR, TacticalPhase.LR_TR, TacticalPhase.TR_DOR,
            TacticalPhase.DOR_DR, TacticalPhase.DR_MAR, TacticalPhase.BEYOND_MAR
        ]
        new_phase = sequential_phases[old_priority + 1]
        new_priority = old_priority + 1

    # 记录阶段锁定（用于调试）
    if agent_id not in self.phase_lock or phase_priority.get(new_phase, 0) > phase_priority.get(self.phase_lock.get(agent_id), 0):
        self.phase_lock[agent_id] = new_phase

    self.state_manager.set_agent_phase(agent_id, new_phase, current_time)

    # ✅ 阶段转移
    if is_lead:
        old_global_phase = self.state_manager.current_phase
        if new_phase != old_global_phase:
            self.state_manager.current_phase = new_phase

    # 阶段切换时执行决策
    if old_phase != new_phase:
        # 离开DR阶段时，重置该机的DR窗口状态（确保下次进入DR时重新计时/决策）
        if old_phase == TacticalPhase.DR_MAR and new_phase != TacticalPhase.DR_MAR:
            try:
                if hasattr(self, 'node_decision_maker'):
                    if hasattr(self.node_decision_maker, 'agent_dr_windows'):
                        self.node_decision_maker.agent_dr_windows.pop(agent_id, None)
                    if hasattr(self.node_decision_maker, 'agent_dr_decisions'):
                        self.node_decision_maker.agent_dr_decisions.pop(agent_id, None)
            except Exception:
                pass

        if new_phase == TacticalPhase.LR_TR:
            self._decide_at_lr(env, agent_id)
            # 基于节点方法的参数/发射评估
            self._node_decision_at_node(env, agent_id, 'LR', current_time)
            if self.is_agent_second_attack(agent_id):
                self._node_decision_at_node(env, agent_id, 'LR2', current_time)
        elif new_phase == TacticalPhase.MTR_LR:
            # 首轮MTR：占位点/轻微队形调整
            self._node_decision_at_node(env, agent_id, 'MTR', current_time)
            if self.is_agent_second_attack(agent_id):
                self._node_decision_at_node(env, agent_id, 'MTR2', current_time)
        elif new_phase == TacticalPhase.MELD_MTR:
            # MELD节点：允许战术切换与编队调整（二级控制距离）
            self._node_decision_at_node(env, agent_id, 'MELD', current_time)
        elif new_phase == TacticalPhase.NLT_MELD:
            # NLT节点：策略与战术初始选择（一级控制距离）
            self._node_decision_at_node(env, agent_id, 'NLT', current_time)
        elif new_phase == TacticalPhase.DOR_DR:
            self._node_decision_at_node(env, agent_id, 'DOR', current_time)
        elif new_phase == TacticalPhase.DR_MAR:
            self._node_decision_at_node(env, agent_id, 'DR', current_time)
        elif new_phase == TacticalPhase.BEYOND_MAR:
            self._node_decision_at_node(env, agent_id, 'MAR', current_time)
        elif new_phase == TacticalPhase.TR_DOR:
            self._node_decision_at_node(env, agent_id, 'TR', current_time)
            if self.is_agent_second_attack(agent_id):
                self._node_decision_at_node(env, agent_id, 'TR2', current_time)

def _select_tactic_at_phase(self, env, agent_id: str):
    """在关键节点选择战术"""
    current_phase = self.state_manager.get_agent_phase(agent_id)
    current_time = env.current_step * env.time_interval

    # 🔥 修复问题4&5: 返航状态下不重新选择战术
    alive_enemy_exists = self._has_alive_enemy_anywhere(env, agent_id)
    if agent_id in self.returning_agents and agent_id.startswith('A') and alive_enemy_exists:
        guard_release_until = float(getattr(self, '_red_guard_release_until', {}).get(agent_id, -999.0))
        guard_release_active = float(current_time) < guard_release_until
        current_tactic_name = str(self._get_agent_tactic(agent_id) or '')
        preserve_defense = bool(guard_release_active or current_tactic_name == 'DEFENSIVE_GUARD')
        if not preserve_defense and hasattr(self, '_evaluate_red_defensive_posture'):
            try:
                preserve_defense = bool(
                    self._evaluate_red_defensive_posture(
                        env,
                        agent_id,
                        current_phase_name="select_tactic_guard_hold",
                        reason="select_tactic_guard_hold",
                    ).get("preserve", False)
                )
            except Exception:
                preserve_defense = bool(guard_release_active or current_tactic_name == 'DEFENSIVE_GUARD')
        if preserve_defense:
            if hasattr(self, '_deescalate_red_formation_to_defense'):
                self._deescalate_red_formation_to_defense(
                    env,
                    agent_id,
                    reason="select_tactic_guard_hold",
                    clear_returning=False,
                )
            if env.current_step % 120 == 0:
                logging.info("[战术选择守区保留] %s 敌机仍存活，保留守区，不清除返航冷却", agent_id)
            return
        formation_agents = self._get_alive_formation_agents(
            env,
            agent_id,
            preserve_nominal_pair=True,
        )
        for aid in formation_agents:
            self.returning_agents.discard(aid)
        if env.current_step % 120 == 0:
            logging.warning("[战术选择RTB拦截] %s 敌机仍存活，禁止保留返航状态", agent_id)
    if agent_id in self.returning_agents:
        if env.current_step % 120 == 0:
            logging.info(f"🏠 [{agent_id}] 返航中，跳过战术选择")
        return  # 返航时不选择新战术

    # 🔥 二次进攻期间锁定战术（按机）
    if self.is_agent_second_attack(agent_id):
        _agent_tac_now = self._get_agent_tactic(agent_id)
        if agent_id.startswith('A') and alive_enemy_exists and _agent_tac_now == 'TACTICAL_TURN':
            override_plan = self._build_red_reengage_override_plan(
                env,
                agent_id,
                current_phase_name=str(getattr(current_phase, 'value', '') or ''),
                reason="second_attack_turn_block",
            )
            _agent_tac_now = str(override_plan.get("replacement_tactic", 'ADAPTIVE_ATTACK'))
            if _agent_tac_now != 'DEFENSIVE_GUARD':
                self.returning_agents.discard(agent_id)
            self._set_agent_tactic(agent_id, _agent_tac_now)
        if _agent_tac_now in ['FORMATION_RESET', 'UNIFIED_SECOND_ATTACK', 'TACTICAL_TURN', 'DEFENSIVE_GUARD']:
            if env.current_step % 120 == 0:
                logging.info(f"🔒 [战术锁定-{agent_id}] 二次进攻进行中，保持战术: {_agent_tac_now}")
            return  # 不重新选择战术

    # 战术选择条件：NLT节点 或 MELD节点 或 MTR_LR节点且还未选择战术
    should_select_tactic = False
    phase_name = ""

    if current_phase == TacticalPhase.NLT_MELD and not self.decision_made['NLT']:
        should_select_tactic = True
        phase_name = "NLT节点"

    elif current_phase == TacticalPhase.MELD_MTR and not self.decision_made['MELD']:
        should_select_tactic = True
        phase_name = "MELD节点"

    elif current_phase == TacticalPhase.MTR_LR and self._get_agent_tactic(agent_id) is None:
        should_select_tactic = True
        phase_name = "MTR_LR节点（强制选择）"

    if should_select_tactic:
        formation_agents = self._get_alive_formation_agents(
            env,
            agent_id,
            preserve_nominal_pair=True,
        )
        enemy_agents = [eid for eid in get_enemy_team(agent_id) if eid in env.agents]
        lead_id = None
        wingman_id = None
        if formation_agents:
            for aid in formation_agents:
                if self._is_formation_lead(env, aid):
                    lead_id = aid
                    break
            if lead_id is None:
                lead_id = formation_agents[0]
            wingman_candidates = [aid for aid in formation_agents if aid != lead_id]
            wingman_id = wingman_candidates[0] if wingman_candidates else lead_id
        lead_target = enemy_agents[0] if enemy_agents else None
        wingman_target = enemy_agents[1] if len(enemy_agents) > 1 else lead_target

        # 🎯 优先检查force_tactic参数（来自run_simulation.py）
        if self.force_tactic and self.force_tactic in ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'FRONT_BACK', 'SIDE_BY_SIDE']:
            logging.info(f"🎯 强制选择战术: {self.force_tactic}")
            self.selected_tactic = self.force_tactic
            self._set_agent_tactic(agent_id, self.force_tactic)
            # 设置默认角色
            if self.force_tactic == 'PINCER_ATTACK':
                self.tactic_roles = {'lead': lead_id, 'wingman': wingman_id, 'lead_target': lead_target, 'wingman_target': wingman_target, 'tactic': 'PINCER_ATTACK'}
            elif self.force_tactic == 'HIGH_LOW_ATTACK':
                self.tactic_roles = {'high': lead_id, 'low': wingman_id, 'high_target': lead_target, 'low_target': wingman_target, 'tactic': 'HIGH_LOW_ATTACK'}
            elif self.force_tactic == 'FRONT_BACK':
                self.tactic_roles = {'leader': lead_id, 'wingman': wingman_id, 'leader_target': lead_target, 'wingman_target': wingman_target, 'tactic': 'FRONT_BACK'}
            elif self.force_tactic == 'TACTICAL_TURN':
                self.tactic_roles = {'lead': lead_id, 'wingman': wingman_id, 'lead_target': lead_target, 'wingman_target': wingman_target, 'tactic': 'TACTICAL_TURN'}
            else:  # DRAG_SHOOT
                self.tactic_roles = {'lead': lead_id, 'wingman': wingman_id, 'lead_target': lead_target, 'wingman_target': wingman_target, 'tactic': 'DRAG_SHOOT'}
        # 检查是否有环境变量强制指定战术（保留向后兼容性）
        else:
            import os
            force_tactic = os.environ.get('FORCE_TACTIC')
            if force_tactic and force_tactic in ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'FRONT_BACK', 'SIDE_BY_SIDE']:
                logging.info(f"🧪 环境变量强制选择战术: {force_tactic}")
                self.selected_tactic = force_tactic
                self._set_agent_tactic(agent_id, force_tactic)
                # 设置默认角色
                if force_tactic == 'PINCER_ATTACK':
                    self.tactic_roles = {
                        'lead': lead_id,
                        'wingman': wingman_id,
                        'lead_target': lead_target,
                        'wingman_target': wingman_target,
                        'tactic': 'PINCER_ATTACK'
                    }
                elif force_tactic == 'HIGH_LOW_ATTACK':
                    self.tactic_roles = {
                        'high': lead_id,
                        'low': wingman_id,
                        'high_target': lead_target,
                        'low_target': wingman_target,
                        'tactic': 'HIGH_LOW_ATTACK'
                    }
                elif force_tactic == 'SIDE_BY_SIDE':
                    self.tactic_roles = {
                        'left': lead_id,
                        'right': wingman_id,
                        'left_target': lead_target,
                        'right_target': wingman_target,
                        'tactic': 'SIDE_BY_SIDE'
                    }
                elif force_tactic == 'FRONT_BACK':
                    self.tactic_roles = {
                        'leader': lead_id,
                        'wingman': wingman_id,
                        'leader_target': lead_target,
                        'wingman_target': wingman_target,
                        'tactic': 'FRONT_BACK'
                    }
                else:  # DRAG_SHOOT
                    self.tactic_roles = {
                        'lead': lead_id,
                        'wingman': wingman_id,
                        'lead_target': lead_target,
                        'wingman_target': wingman_target,
                        'tactic': 'DRAG_SHOOT'
                    }
            else:
                # 正常战术选择流程
                logging.info("=" * 80)
                logging.info(f"🎯 [{phase_name}] 开始战术决策（完整智能系统）")
                # 准备飞机列表
                my_aircraft_list = [env.agents[aid] for aid in formation_agents]
                enemy_aircraft_list = [env.agents[eid] for eid in enemy_agents]

                # 调用完整战术系统选择战术（修正参数顺序：control_distance, my_list, enemy_list, env）
                if current_phase == TacticalPhase.NLT_MELD:
                    control_distance = 'NLT'
                elif current_phase == TacticalPhase.MELD_MTR:
                    control_distance = 'MELD'
                else:
                    control_distance = 'MTR'
                tactic_result = self.complete_tactical_system.select_tactic(
                    control_distance,
                    my_aircraft_list,
                    enemy_aircraft_list,
                    env
                )

                # 处理返回结果
                if isinstance(tactic_result, dict):
                    self.selected_tactic = tactic_result['tactic']
                    self.tactic_roles = tactic_result['roles']
                else:
                    # 假设返回的是(tactic, roles)元组
                    self.selected_tactic = tactic_result[0]
                    self.tactic_roles = tactic_result[1]

                self._set_agent_tactic(agent_id, self.selected_tactic)

                logging.info(f"✅ 智能选择战术: {self.selected_tactic}")
                logging.info(f"✅ 角色分配: {self.tactic_roles}")

        # 更新决策标记
        if current_phase == TacticalPhase.NLT_MELD:
            self.decision_made['NLT'] = True
        elif current_phase == TacticalPhase.MELD_MTR:
            self.decision_made['MELD'] = True

        # 🔥 日志：显示真实飞机ID（如果环境是MockEnv，则反映Group 2的真实ID）
        real_map = getattr(env, 'real_id_map', {})
        display_roles = {}
        if self.tactic_roles:
            for k, v in self.tactic_roles.items():
                display_roles[k] = real_map.get(v, v) if isinstance(v, str) and v.startswith(('A', 'B')) else v
        logging.info(f"   ✅ 最终选定战术: {self.selected_tactic}")
        logging.info(f"   ✅ 最终角色分配: {display_roles if display_roles else self.tactic_roles}")
        logging.info("=" * 80)

def _decide_at_lr(self, env, agent_id: str):
    """LR节点决策：根据态势决定是Crank还是平飞"""
    try:
        # 🎯 HIGH_LOW_ATTACK战术特殊处理：僚机不执行Crank，保持高空直飞
        if (self.selected_tactic == 'HIGH_LOW_ATTACK' and 
            self.tactic_roles.get(agent_id) == 'low' and 
            agent_id == 'A0200'):
            self.lr_maneuver[agent_id] = 'straight'
            logging.info(f"🎯 [HIGH_LOW_ATTACK-僚机LR] {agent_id} 战术特定：跳过Crank决策，保持直飞")
            return

        # 获取敌机方位
        target_id = get_target_with_fallback(agent_id, env)
        if target_id is None:
            self.lr_maneuver[agent_id] = 'straight'
            return

        target_aircraft = env._jsbsims.get(target_id)
        if not target_aircraft or not target_aircraft.is_alive:
            self.lr_maneuver[agent_id] = 'straight'
            return

        distance = TacticalUtils.calculate_distance_between(env.agents[agent_id], target_aircraft)
        missiles_left = 0
        try:
            missiles_left = int(self._get_aircraft_missiles_left(env.agents[agent_id]))
        except Exception:
            missiles_left = 0
        preserve_defense = False
        try:
            preserve_defense = bool(
                self._evaluate_red_defensive_posture(
                    env,
                    agent_id,
                    target=target_aircraft,
                    distance=distance,
                    current_phase_name='LR',
                    reason='lr_align',
                ).get('preserve', False)
            )
        except Exception:
            preserve_defense = False
        if (
            preserve_defense
            or missiles_left <= 0
            or (distance is not None and np.isfinite(distance) and (distance < 55000.0 or distance > 115000.0))
        ):
            self.lr_maneuver[agent_id] = 'straight'
            logging.info(
                f"🎯 [LR决策] {agent_id} 守区/无弹/距离不适合(dist={distance/1000.0:.1f}km, msl={missiles_left}) → 保持直飞"
            )
            if hasattr(self, '_log_key_event'):
                phase_obj = self.state_manager.get_agent_phase(agent_id) if hasattr(self, 'state_manager') else None
                phase_str = phase_obj.value if hasattr(phase_obj, 'value') else phase_obj
                self._log_key_event(
                    env,
                    agent_id,
                    "LR决策",
                    当前阶段=phase_str,
                    当前战术=getattr(self, 'selected_tactic', None),
                    进入函数="_decide_at_lr",
                    执行机动="STRAIGHT",
                    当前状态=f"preserve={preserve_defense}, missiles_left={missiles_left}, distance_km={distance/1000.0:.1f}",
                    当前指令="straight",
                    退出条件="进入可发射/可转向窗口",
                    是否满足退出="否",
                )
            return

        # 获取当前航向和敌机方位
        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)

        # 计算敌机方位角（使用原版逻辑）
        my_pos = np.array(env.agents[agent_id].get_position())
        enemy_pos = np.array(target_aircraft.get_position())
        delta_x = enemy_pos[0] - my_pos[0]
        delta_y = enemy_pos[1] - my_pos[1]
        bearing_rad = np.arctan2(delta_y, delta_x)
        bearing_deg = np.rad2deg(bearing_rad)
        enemy_bearing = (90 - bearing_deg) % 360.0  # 从数学坐标系转为航空坐标系

        # 计算航向差（带符号，原版公式）
        heading_diff = ((enemy_bearing - current_heading + 180) % 360) - 180
        heading_diff_abs = abs(heading_diff)

        # 🔥 LR决策说明：LR节点是导弹发射节点，需要调整航向对准敌机
        # Crank的作用：在保持雷达照射的同时，侧向机动以优化发射角度
        # 
        # 修复转圈问题：大角度航向差时，不执行Crank，改为直接转向敌机
        # - 小角度（<30°）：已经对准，不需要Crank，保持当前航向
        # - 中等角度（30-60°）：执行温和Crank调整
        # - 大角度（>60°）：直接转向敌机，不用Crank（Crank会转圈）

        if heading_diff_abs < 35.0:
            # 已经基本对准敌机，保持当前航向即可
            self.lr_maneuver[agent_id] = 'crank'
            if not hasattr(self, "_lr_realign_state"):
                self._lr_realign_state = {}
            self._lr_realign_state[agent_id] = {
                'mode': 'limited_realign',
                'heading_error': float(heading_diff),
                'target_bearing': float(enemy_bearing),
                'time': float(getattr(env, "current_step", 0)) * float(getattr(env, "time_interval", 0.2)),
            }
            logging.info(f"🎯 [LR决策] {agent_id} 航向差{heading_diff_abs:.1f}°，已对准，保持航向")
            if hasattr(self, '_log_key_event'):
                phase_obj = self.state_manager.get_agent_phase(agent_id) if hasattr(self, 'state_manager') else None
                phase_str = phase_obj.value if hasattr(phase_obj, 'value') else phase_obj
                self._log_key_event(
                    env,
                    agent_id,
                    "LR决策",
                    当前阶段=phase_str,
                    当前战术=getattr(self, 'selected_tactic', None),
                    进入函数="_decide_at_lr",
                    执行机动="STRAIGHT",
                    当前状态=f"heading_diff={heading_diff_abs:.1f}deg",
                    当前指令="straight",
                    退出条件="heading_diff扩大或进入其他节点",
                    是否满足退出="否",
                )
        elif heading_diff_abs < 48.0:
            # 中等角度差，执行温和Crank调整
            self.lr_maneuver[agent_id] = 'crank'
            logging.info(f"🎯 [LR决策] {agent_id} 航向差{heading_diff_abs:.1f}°，执行温和Crank调整")
            if hasattr(self, '_log_key_event'):
                phase_obj = self.state_manager.get_agent_phase(agent_id) if hasattr(self, 'state_manager') else None
                phase_str = phase_obj.value if hasattr(phase_obj, 'value') else phase_obj
                self._log_key_event(
                    env,
                    agent_id,
                    "LR决策",
                    当前阶段=phase_str,
                    当前战术=getattr(self, 'selected_tactic', None),
                    进入函数="_decide_at_lr",
                    执行机动="CRANK",
                    当前状态=f"heading_diff={heading_diff_abs:.1f}deg",
                    当前指令="crank",
                    退出条件="heading_diff收敛或进入下一节点",
                    是否满足退出="否",
                )
        else:
            # 🔥 大角度差（>60°），禁止直接转向敌机（会导致坠毁）
            # 改为保持直飞，让底层模型自然调整航向
            self.lr_maneuver[agent_id] = 'straight'
            logging.warning(f"⚠️ [LR决策] {agent_id} 航向差{heading_diff_abs:.1f}°过大，禁止转向敌机（防止坠毁），保持直飞")
            if hasattr(self, '_log_key_event'):
                phase_obj = self.state_manager.get_agent_phase(agent_id) if hasattr(self, 'state_manager') else None
                phase_str = phase_obj.value if hasattr(phase_obj, 'value') else phase_obj
                self._log_key_event(
                    env,
                    agent_id,
                    "LR决策",
                    当前阶段=phase_str,
                    当前战术=getattr(self, 'selected_tactic', None),
                    进入函数="_decide_at_lr",
                    执行机动="STRAIGHT",
                    当前状态=f"heading_diff={heading_diff_abs:.1f}deg over_limit",
                    当前指令="straight",
                    退出条件="heading_diff回到可控范围",
                    是否满足退出="否",
                    level=logging.WARNING,
                )

    except Exception as e:
        logging.error(f"LR决策错误: {e}")
        self.lr_maneuver[agent_id] = 'straight'

def _node_decision_at_node(self, env, agent_id: str, node: str, current_time: float):
    try:
        target_id = get_target_with_fallback(agent_id, env)
        if not target_id:
            return
        target_aircraft = env._jsbsims.get(target_id)
        if not target_aircraft or not target_aircraft.is_alive:
            return
        my_aircraft = env.agents.get(agent_id)
        distance = TacticalUtils.calculate_distance_between(my_aircraft, target_aircraft)
        # ✅ 使用“当前阶段”进行态势评估（以前固定用NLT权重，会导致各节点态势不真实）
        from core.situation_evaluator import TacticalPhase as SEPhase
        try:
            cur_phase = self.state_manager.get_agent_phase(agent_id)
            se_phase = SEPhase(cur_phase.value) if hasattr(cur_phase, "value") else SEPhase.NLT_MELD
        except Exception:
            se_phase = SEPhase.NLT_MELD
        situation_score = self.situation_evaluator.evaluate_situation(my_aircraft, target_aircraft, se_phase)

        # ✅ 威胁评估：传感器威胁 tl_sensor 与 几何威胁 T_geo≈(1-F) 融合（更贴近文档6.3）
        tl_sensor = self.threat_evaluator.calculate_total_threat(my_aircraft, target_aircraft, env)
        try:
            t_geo = float(np.clip(1.0 - float(getattr(situation_score, "total", 0.5)), 0.0, 1.0))
        except Exception:
            t_geo = 0.5
        try:
            alpha = float(os.getenv("TACTICAL_THREAT_ALPHA", "0.5"))
        except Exception:
            alpha = 0.5
        threat_level = float(np.clip(alpha * t_geo + (1.0 - alpha) * float(tl_sensor), 0.0, 1.0))
        # ✅ 让“我方意图”在工程上真正产生差异：同一几何态势下对态势阈值做轻微偏置
        # 激进更乐观（更可能判定ADVANTAGE），防御更保守（更可能判定DISADVANTAGE）
        try:
            bias_map = {'AGGRESSIVE_CLEAR': 0.05, 'CONSERVATIVE_CLEAR': 0.0, 'DEFENSIVE': -0.05}
            eff_total = float(np.clip(float(situation_score.total) + bias_map.get(self.my_intent_str, 0.0), 0.0, 1.0))
        except Exception:
            eff_total = situation_score.total

        if eff_total >= 0.6:
            situation = 'ADVANTAGE'
        elif eff_total >= 0.4:
            situation = 'NEUTRAL'
        else:
            situation = 'DISADVANTAGE'
        my_aircraft_list = [env.agents.get('A0100'), env.agents.get('A0200')]
        enemy_aircraft_list = [env.agents.get('B0100'), env.agents.get('B0200')]
        enemies_alive = sum(1 for eid in ['B0100', 'B0200'] if eid in env.agents and env.agents[eid].is_alive)
        missiles_remaining = getattr(my_aircraft, 'num_missiles', 0)

        # 🔄 使用算法切换器进行敌方意图识别（默认算法1，缺权重则回退规则）
        enemy_intent = 'ATTACK'  # 默认值
        if self.situation_algorithm_switcher is not None:
            try:
                # ✅ 优先使用预热缓存（主要面向我方A队的决策）
                if agent_id.startswith('A') and hasattr(self, "_enemy_intent_cache"):
                    cached = self._enemy_intent_cache.get(target_id)
                    if cached:
                        cached_intent, cached_step = cached
                        if cached_intent and (env.current_step - cached_step) <= 50:
                            enemy_intent = cached_intent
                            raise StopIteration
                # 🔥 修复：recognize_intent的参数顺序是 (enemy_aircraft, my_aircraft, env)
                # 但实际调用时，target_id是单个飞机ID，需要转换为飞机对象
                target_aircraft = env.agents.get(target_id) if target_id in env.agents else None
                if target_aircraft and my_aircraft_list:
                    # 取第一个我方飞机作为代表
                    my_aircraft_obj = my_aircraft_list[0] if isinstance(my_aircraft_list, list) and my_aircraft_list else None
                    if my_aircraft_obj:
                        enemy_intent = self.situation_algorithm_switcher.recognize_intent(
                            target_aircraft, my_aircraft_obj, env
                        )
                logging.debug(f"[敌方意图识别] {target_id} -> {enemy_intent}")
            except StopIteration:
                pass
            except Exception as e:
                logging.warning(f"意图识别失败: {e}, 使用默认值ATTACK")
                enemy_intent = 'ATTACK'
        commit_snapshot = {}
        if agent_id.startswith('A') and hasattr(self, "_get_red_enemy_commit_snapshot"):
            try:
                commit_snapshot = self._get_red_enemy_commit_snapshot(
                    env,
                    agent_id,
                    target=target_aircraft,
                    distance=distance,
                ) or {}
            except Exception:
                commit_snapshot = {}

        context = NodeContext(
            env=env,
            agent_id=agent_id,
            current_time=current_time,
            distance=distance,
            threat_level=threat_level,
            situation=situation,
            enemy_intent=enemy_intent,  # 🔄 使用识别的敌方意图
            our_intent=self.my_intent_str,
            my_aircraft=my_aircraft_list,
            enemy_aircraft=enemy_aircraft_list,
            is_second_attack=self.is_agent_second_attack(agent_id),
            custom_data={
                'current_tactic': self.selected_tactic,
                'enemies_alive': enemies_alive,
                'fuel_remaining': 80,
                'missiles_remaining': missiles_remaining,
                'enemy_group_phase': str(commit_snapshot.get('enemy_group_phase', 'UNKNOWN')),
                'enemy_group_pressure_level': int(commit_snapshot.get('enemy_group_pressure_level', 0) or 0),
                'enemy_wave_retreating': bool(commit_snapshot.get('enemy_wave_retreating', False)),
                'target_zone': str(commit_snapshot.get('target_zone', 'UNKNOWN')),
            }
        )
        decision = self.node_decision_maker.make_decision(node, context)

        # 🔥 修复：处理HIGH_LOW_ATTACK的vertical_split_targets初始化
        if isinstance(decision, dict) and decision.get('formation'):
            formation = decision['formation']
            if formation.get('type') == 'altitude_adjust' and 'target_alt_diff' in formation:
                # HIGH_LOW_ATTACK：设置垂直分离目标高度
                alt_diff = formation['target_alt_diff']  # 3000m
                current_alt = my_aircraft.get_position()[2]
                is_lead = self._is_formation_lead(env, agent_id)

                if is_lead:
                    # 🔥 修复：长机在上下夹击中应该平飞，不应该设置下降目标
                    # 长机保持当前高度，不设置目标高度（或设置为当前高度）
                    target_alt = current_alt  # 长机保持当前高度，不下降
                    # 不设置vertical_split_targets，让长机保持平飞
                    # self.state_manager.vertical_split_targets[agent_id] = target_alt  # 注释掉，不设置目标高度
                    if env.current_step % 200 == 0:
                        logging.info(f"🎯 [HIGH_LOW_ATTACK-长机高度设置] {agent_id} 保持当前高度={current_alt:.0f}m（平飞，不下降）")
                else:
                    # 僚机：高空（当前高度 + 3000m，但最高不超过12000m）
                    target_alt = min(current_alt + alt_diff, 12000)
                    self.state_manager.vertical_split_targets[agent_id] = target_alt
                    if env.current_step % 200 == 0:
                        logging.info(f"🎯 [HIGH_LOW_ATTACK-僚机高度设置] {agent_id} 目标高度={target_alt:.0f}m (当前{current_alt:.0f}m, 差值{alt_diff:.0f}m)")

        # ✅ 把"节点层的发射建议"真正接到发射请求标记上
        # 之前多数发射请求来自模板内部 self.task.missile_launched[aid]=True，导致有些战术/阶段发射过少。
        try:
            if isinstance(decision, dict) and decision.get('launch_missile', False):
                # 只对我方触发请求，敌方由自己的AI发射逻辑处理
                if agent_id.startswith('A'):
                    self.state_manager.missile_launched[agent_id] = True
                    if env.current_step % 100 == 0:
                        logging.info(f"🚀 [节点发射请求-{agent_id}] node={node} dist={distance/1000:.1f}km 请求发射")
        except Exception:
            pass

        # ==================== 论文/答辩用：决策链路快照（可开关） ====================
        # 目的：把“态势评估/威胁评估/意图识别/模板选择/机动/参数”串联成一条可读解释。
        # 开关：设置环境变量 TACTICAL_DECISION_TRACE=1
        if os.getenv("TACTICAL_DECISION_TRACE", "").strip() in ("1", "true", "True", "YES", "yes"):
            key = (agent_id, node)
            last = self._decision_trace_last_step.get(key, -10**9)
            # 每个节点每架机最多每 50 step 打一次，避免刷屏
            if env.current_step - last >= 50:
                self._decision_trace_last_step[key] = env.current_step
                algo = None
                try:
                    if self.situation_algorithm_switcher is not None and hasattr(self.situation_algorithm_switcher, "get_current_algorithm"):
                        algo = self.situation_algorithm_switcher.get_current_algorithm()
                except Exception:
                    algo = None
                logging.info(
                    "🧾 [决策快照] "
                    f"node={node} agent={agent_id} "
                    f"I_self={self.my_intent_str} "
                    f"I_enemy={enemy_intent}(algo={algo or 'unknown'}) "
                    f"S={situation}(F≈{getattr(situation_score,'total',0.0):.2f}) "
                    f"tl={threat_level:.2f} dist={distance/1000:.1f}km "
                    f"selected_tactic={self.selected_tactic} "
                    f"decision.tactic={getattr(decision,'get',lambda _k,_d=None:None)('tactic',None)} "
                    f"decision.maneuver={getattr(decision,'get',lambda _k,_d=None:None)('maneuver',None)}"
                )

        # 🔥 修复2: 统一的DR决策处理 - 避免重复触发
        if node == 'DR' and decision.get('reengage', False):
            self._handle_dr_reengage_unified(env, agent_id, current_time, enemies_alive, threat_level)
            return decision

        # 🔥 修复2: 统一的DR撤退处理 - 移除重复代码
        if node == 'DR' and decision.get('retreat', False):
            self._handle_dr_retreat_unified(env, agent_id, current_time)
            return decision

        if decision.get('tactic'):
            new_tactic = decision['tactic']
            alive_enemy_exists = self._has_alive_enemy_anywhere(env, agent_id)
            if agent_id.startswith('A') and new_tactic == 'TACTICAL_EVASION' and self.is_agent_second_attack(agent_id):
                formation_under_missile = False
                try:
                    formation_under_missile = bool(self._formation_has_incoming_missile(env, agent_id))
                except Exception:
                    formation_under_missile = False
                rwr_level = 0
                try:
                    from simulation.radar_manager import get_unified_radar_manager
                    rm = get_unified_radar_manager()
                    rwr_level = int(rm.get_rwr_threat_level(agent_id)) if rm else 0
                except Exception:
                    rwr_level = 0
                if not formation_under_missile and rwr_level < 4:
                    logging.info(
                        f"🧭 [防御战术门禁] {agent_id} 二次进攻无实弹威胁(RWR={rwr_level})，保持ADAPTIVE_ATTACK而非{new_tactic}"
                    )
                    override_plan = self._build_red_reengage_override_plan(
                        env,
                        agent_id,
                        current_phase_name="second_attack_evasion_gate",
                        reason="second_attack_evasion_gate",
                        allow_template_reset=False,
                    )
                    replacement_tactic = str(override_plan.get("replacement_tactic", 'ADAPTIVE_ATTACK'))
                    logging.info(
                        "[防御战术门禁-Override] %s 二次进攻无实弹威胁(RWR=%s)，改为%s",
                        agent_id,
                        rwr_level,
                        replacement_tactic,
                    )
                    new_tactic = replacement_tactic
            if agent_id.startswith('A') and alive_enemy_exists and new_tactic == 'TACTICAL_TURN':
                override_plan = self._build_red_reengage_override_plan(
                    env,
                    agent_id,
                    current_phase_name="decision_tactical_turn",
                    reason="decision_tactical_turn",
                )
                preserve_defense = bool(override_plan.get("preserve_defense", False))
                replacement_tactic = str(override_plan.get("replacement_tactic", 'ADAPTIVE_ATTACK'))
                posture_state = override_plan.get("posture", {}) or {}
                if preserve_defense:
                    logging.info(
                        "🛡️ [防御返航保留] %s decision.tactic=TACTICAL_TURN reason=%s",
                        agent_id,
                        posture_state.get("reason", "decision_tactical_turn"),
                    )
                    new_tactic = 'DEFENSIVE_GUARD'
                else:
                    logging.warning(
                        "🛑 [防御返航拦截] %s decision.tactic=TACTICAL_TURN 但敌机仍存活，改为%s",
                        agent_id,
                        replacement_tactic,
                    )
                    new_tactic = replacement_tactic

            # 🎯 检查是否有强制战术设置，防御战术除外
            if self.force_tactic and new_tactic not in ['TACTICAL_EVASION', 'TACTICAL_TURN']:
                logging.info(f"🎯 节点决策尝试切换到{new_tactic}，但强制战术{self.force_tactic}优先级更高，忽略切换")
                return  # 忽略非防御战术的切换

            # 若切换到防御类战术，记录之前的战术用于TTL后恢复
            if new_tactic in ['TACTICAL_EVASION', 'TACTICAL_TURN']:
                # 🔥 关键修复：如果正在执行FORMATION_RESET或TACTICAL_TURN（撤退），不允许防御战术干扰
                _cur_tac_for_agent = self._get_agent_tactic(agent_id)
                if _cur_tac_for_agent in ('FORMATION_RESET', 'TACTICAL_TURN'):
                    logging.info(f"🛡️ [防御战术] {agent_id} 正在执行{_cur_tac_for_agent}，忽略防御战术切换到{new_tactic}")
                    return  # 忽略防御战术切换

                if getattr(self, '_defense_prev_tactic', None) is None:
                    # 如果有强制战术，备份强制战术而不是当前战术
                    self._defense_prev_tactic = self.force_tactic if self.force_tactic else _cur_tac_for_agent
                self._set_agent_tactic(agent_id, new_tactic)
                logging.info(f"🛡️ 切换到防御战术{new_tactic}，备份战术: {self._defense_prev_tactic}")
            else:
                self._set_agent_tactic(agent_id, new_tactic)
                # 切回普通战术后清理备份
                if hasattr(self, '_defense_prev_tactic'):
                    self._defense_prev_tactic = None
        if decision.get('is_second_attack'):
            self.set_agent_second_attack(agent_id, True)
        if decision.get('retreat'):
            alive_enemy_exists = self._has_alive_enemy_anywhere(env, agent_id)
            if agent_id.startswith('A') and alive_enemy_exists:
                override_plan = self._build_red_reengage_override_plan(
                    env,
                    agent_id,
                    current_phase_name="decision_retreat",
                    reason="decision_retreat",
                )
                posture_state = override_plan.get("posture", {}) or {}
                preserve_defense = bool(override_plan.get("preserve_defense", False))
                replacement_tactic = str(override_plan.get("replacement_tactic", 'ADAPTIVE_ATTACK'))
                formation_agents = override_plan.get("formation_agents", []) or [agent_id]
                if preserve_defense:
                    if hasattr(self, '_deescalate_red_formation_to_defense'):
                        self._deescalate_red_formation_to_defense(
                            env,
                            agent_id,
                            reason=posture_state.get("reason", "decision_retreat"),
                        )
                    for aid in formation_agents:
                        self.set_agent_second_attack(aid, False)
                        self._set_agent_tactic(aid, 'DEFENSIVE_GUARD')
                    logging.info(
                        "🛡️ [撤退守区保留] %s decision.retreat=True reason=%s",
                        agent_id,
                        posture_state.get("reason", "decision_retreat"),
                    )
                    return decision
                for aid in self._get_formation_agents(agent_id, env):
                    self.returning_agents.discard(aid)
                    self.set_agent_second_attack(aid, True)
                for aid in formation_agents:
                    self._set_agent_tactic(aid, replacement_tactic)
                logging.warning(
                    "🛑 [撤退返航拦截] %s decision.retreat=True preserve=0 phase=%s grp=%s zone=%s dist=%.1fkm reason=%s -> %s",
                    agent_id,
                    posture_state.get("enemy_group_phase", "UNKNOWN") if posture_state else "UNKNOWN",
                    posture_state.get("enemy_group_id", "") if posture_state else "",
                    posture_state.get("target_zone", "UNKNOWN") if posture_state else "UNKNOWN",
                    float(posture_state.get("distance_km", float('nan'))) if posture_state else float('nan'),
                    posture_state.get("reason", "decision_retreat") if posture_state else "decision_retreat",
                    replacement_tactic,
                )
            else:
                self._set_agent_tactic(agent_id, 'TACTICAL_TURN')
        if decision.get('attack_waypoint') is not None:
            self.attack_waypoint[agent_id] = decision['attack_waypoint']

        return decision
    except Exception as e:
        logging.error(f"节点决策错误[{node}]: {e}")
        return None
