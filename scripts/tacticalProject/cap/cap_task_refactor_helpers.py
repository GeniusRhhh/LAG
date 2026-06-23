"""Extracted helper methods for CAPTask.

This module keeps logic identical while reducing class file size.
"""

import logging
import os
from collections import deque

import numpy as np
import torch

from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.core.simulatior import AircraftSimulator
from envs.JSBSim.utils.utils import get_root_dir

try:
    from ..action_codec import ALT_HOLD, VEL_ACCEL
except ImportError:
    from action_codec import ALT_HOLD, VEL_ACCEL

log = logging.getLogger(__name__)

_CAP_DEBUG_PRINT = os.environ.get('CAP_DEBUG_PRINT') == '1'
_CAP_CONTROL_DEBUG = os.environ.get('CAP_CONTROL_DEBUG', '0').strip().lower() in ('1', 'true', 'yes', 'on')
_B0100_DEEP_TRACE = os.environ.get('CAP_B0100_DEEP_TRACE', '1').strip().lower() in ('1', 'true', 'yes', 'on')
_ALL_B_DEEP_TRACE = os.environ.get('CAP_ALL_B_DEEP_TRACE', '0').strip().lower() in ('1', 'true', 'yes', 'on')


def _deep_trace_enabled(agent_id: str) -> bool:
    if str(agent_id).startswith('B') and _ALL_B_DEEP_TRACE:
        return True
    return agent_id == 'B0100' and _B0100_DEEP_TRACE


def _sim_recreate_window_len(agent_id: str) -> int:
    env_key = "CAP_ENEMY_SIM_RECREATE_WINDOW" if str(agent_id).startswith("B") else "CAP_FRIENDLY_SIM_RECREATE_WINDOW"
    try:
        return max(10, int(os.getenv(env_key, "15")))
    except Exception:
        return 15


def _sim_recreate_history_len(agent_id: str) -> int:
    env_key = "CAP_ENEMY_SIM_RECREATE_HISTORY" if str(agent_id).startswith("B") else "CAP_FRIENDLY_SIM_RECREATE_HISTORY"
    try:
        return max(20, int(os.getenv(env_key, "60")))
    except Exception:
        return 60


def _sim_recreate_cooldown_steps(agent_id: str) -> int:
    if str(agent_id).startswith("B"):
        env_key = "CAP_ENEMY_SIM_RECREATE_COOLDOWN_STEPS"
        default = "300"
    else:
        env_key = "CAP_FRIENDLY_SIM_RECREATE_COOLDOWN_STEPS"
        default = "260"
    try:
        return max(0, int(os.getenv(env_key, default)))
    except Exception:
        return int(default)


def _sim_recreate_reference_age_s(agent_id: str) -> float:
    if str(agent_id).startswith("B"):
        env_key = "CAP_ENEMY_SIM_RECREATE_REFERENCE_AGE_S"
        default = "4.0"
    else:
        env_key = "CAP_FRIENDLY_SIM_RECREATE_REFERENCE_AGE_S"
        default = "4.0"
    try:
        return max(0.4, float(os.getenv(env_key, default)))
    except Exception:
        return float(default)


def _sim_recreate_posture_hold_s(agent_id: str) -> float:
    if str(agent_id).startswith("B"):
        env_key = "CAP_ENEMY_SIM_RECREATE_POSTURE_HOLD_S"
        default = "16.0"
    else:
        env_key = "CAP_FRIENDLY_SIM_RECREATE_POSTURE_HOLD_S"
        default = "32.0"
    try:
        return max(4.0, float(os.getenv(env_key, default)))
    except Exception:
        return float(default)


def _task_max_altitude_m(*, friendly: bool) -> float:
    env_key = "CAP_FRIENDLY_MAX_ALTITUDE_M" if friendly else "CAP_ENEMY_MAX_ALTITUDE_M"
    try:
        return max(6000.0, float(os.getenv(env_key, "13000")))
    except Exception:
        return 13000.0


def _task_max_altitude_ft(*, friendly: bool) -> float:
    return float(_task_max_altitude_m(friendly=friendly) / 0.3048)


def _estimate_recreate_speed(window: deque, current_vc_mps: float, friendly: bool) -> float:
    samples = [float(item.get("vc", current_vc_mps)) for item in list(window)]
    if not samples:
        baseline = float(current_vc_mps)
    else:
        early_window = samples[: max(3, len(samples) // 2)]
        baseline = float(np.percentile(early_window, 70))

    blended = 0.62 * float(current_vc_mps) + 0.38 * baseline
    if friendly:
        floor_speed = 175.0
        ceil_speed = 235.0
    else:
        floor_speed = 160.0
        ceil_speed = 235.0

    min_allowed = max(floor_speed, min(float(current_vc_mps) - 10.0, baseline - 6.0))
    max_allowed = min(ceil_speed, max(float(current_vc_mps) + 12.0, floor_speed + 4.0), baseline + 8.0)
    if max_allowed < min_allowed:
        return float(np.clip(0.5 * (float(current_vc_mps) + baseline), floor_speed, ceil_speed))
    return float(np.clip(blended, min_allowed, max_allowed))


def _clamp_recreate_speed_gain(
    agent_id: str,
    current_vc_mps: float,
    rescue_speed_mps: float,
    *,
    alt_m: float,
    severe_emergency: bool,
    recent_recreate_count: int,
) -> float:
    friendly = str(agent_id).startswith("A")
    floor_speed = 175.0 if friendly else 160.0
    if rescue_speed_mps <= current_vc_mps:
        return float(max(floor_speed, rescue_speed_mps))

    max_gain = 12.0
    if severe_emergency and current_vc_mps < 150.0:
        max_gain += 6.0
    elif severe_emergency and current_vc_mps < 170.0:
        max_gain += 3.0
    if alt_m > 11000.0:
        max_gain = min(max_gain, 8.0)
    if recent_recreate_count >= 2:
        max_gain = min(max_gain, 5.0)
    if recent_recreate_count >= 3:
        max_gain = min(max_gain, 4.0)

    return float(max(floor_speed, min(rescue_speed_mps, current_vc_mps + max_gain)))


def _estimate_recreate_pitch_and_roc(
    agent_id: str,
    *,
    alt_m: float,
    current_vc_mps: float,
    current_vup_mps: float,
    severe_emergency: bool,
    reference_snapshot=None,
) -> tuple[float, float]:
    friendly = str(agent_id).startswith("A")
    altitude_cap_m = _task_max_altitude_m(friendly=friendly)
    high_alt_soft_recover = bool(friendly and alt_m >= 10000.0)
    altitude_near_cap = bool(friendly and alt_m >= (altitude_cap_m - 250.0))
    altitude_over_cap = bool(friendly and alt_m >= altitude_cap_m)
    sink_guard = bool(
        friendly
        and (
            (alt_m < 9500.0 and current_vup_mps < -7.0)
            or (alt_m >= 7000.0 and alt_m < 9500.0 and current_vc_mps < 178.0 and current_vup_mps < -5.5)
            or (alt_m < 6500.0 and (current_vc_mps < 168.0 or current_vup_mps < -5.0))
        )
    )
    deep_sink_guard = bool(
        friendly
        and (
            current_vup_mps < -24.0
            or (alt_m < 7000.0 and current_vc_mps < 128.0)
            or (alt_m < 6000.0 and (current_vc_mps < 145.0 or current_vup_mps < -9.0))
        )
    )
    ref_pitch_deg = 0.0
    ref_vup_mps = 0.0
    if isinstance(reference_snapshot, dict):
        try:
            ref_pitch_deg = float(reference_snapshot.get("pitch_deg", 0.0) or 0.0)
        except Exception:
            ref_pitch_deg = 0.0
        try:
            ref_vup_mps = float(reference_snapshot.get("vup", 0.0) or 0.0)
        except Exception:
            ref_vup_mps = 0.0

    target_pitch_deg = max(1.2, min(4.0 if friendly else 3.6, max(0.0, ref_pitch_deg)))
    if current_vup_mps < -2.0:
        target_pitch_deg += min(1.4, max(0.0, -current_vup_mps - 2.0) * 0.18)
    if alt_m < (5200.0 if friendly else 4800.0):
        target_pitch_deg += 0.5
    if severe_emergency:
        target_pitch_deg += 0.4
    if sink_guard:
        target_pitch_deg = max(
            target_pitch_deg,
            min(5.0, 2.6 + max(0.0, -current_vup_mps - 7.0) * 0.10),
        )
    if deep_sink_guard:
        target_pitch_deg = max(target_pitch_deg, 3.4 if current_vc_mps >= 135.0 else 2.8)
    if current_vc_mps < (165.0 if friendly else 150.0):
        target_pitch_deg = min(target_pitch_deg, 2.4)
    if friendly and sink_guard and current_vc_mps < 150.0:
        target_pitch_deg = max(target_pitch_deg, 2.2)
    if high_alt_soft_recover:
        target_pitch_deg = min(target_pitch_deg, 1.6)
    if altitude_near_cap:
        target_pitch_deg = min(target_pitch_deg, 1.4 if not altitude_over_cap else 0.9)
    target_pitch_deg = float(np.clip(target_pitch_deg, 1.0, 5.0 if friendly else 3.8))

    target_vup_mps = max(0.8, ref_vup_mps)
    if current_vup_mps < -1.0:
        target_vup_mps = max(target_vup_mps, min(3.2 if friendly else 2.8, max(0.8, -current_vup_mps * 0.32)))
    if severe_emergency:
        target_vup_mps = max(target_vup_mps, 1.6 if friendly else 1.4)
    if sink_guard:
        target_vup_mps = max(
            target_vup_mps,
            min(4.2, 2.1 + max(0.0, -current_vup_mps - 7.0) * 0.08),
        )
    if deep_sink_guard:
        target_vup_mps = max(target_vup_mps, 2.8 if current_vc_mps >= 135.0 else 2.3)
    if alt_m > 9800.0 and not sink_guard:
        target_vup_mps = min(target_vup_mps, 1.8)
    if current_vc_mps < (165.0 if friendly else 150.0) and not sink_guard:
        target_vup_mps = min(target_vup_mps, 1.9 if friendly else 1.7)
    if high_alt_soft_recover:
        target_vup_mps = min(target_vup_mps, 1.2)
    if altitude_near_cap:
        target_vup_mps = min(target_vup_mps, 0.9 if not altitude_over_cap else 0.6)
    target_vup_mps = float(np.clip(target_vup_mps, 0.6, 4.2 if friendly else 3.0))
    return target_pitch_deg, float(target_vup_mps * 196.85039370078738)


def _normalize_missile_state_value(state) -> str:
    try:
        value = getattr(state, "value", state)
    except Exception:
        value = state
    return str(value).strip().lower()


def _get_effective_missiles_left(self, env, agent_id: str, sim: AircraftSimulator) -> int:
    candidates = []
    for attr_name in ("num_left_missiles", "num_missiles"):
        try:
            if hasattr(sim, attr_name):
                candidates.append(max(0, int(getattr(sim, attr_name))))
        except Exception:
            pass

    state_manager = getattr(self, "state_manager", None)
    if state_manager is not None and hasattr(state_manager, "missiles_remaining"):
        try:
            if agent_id in state_manager.missiles_remaining:
                remaining = state_manager.missiles_remaining.get(agent_id)
                candidates.append(max(0, int(remaining)))
        except Exception:
            pass

    task_aircraft_counts = getattr(self, "aircraft_missile_counts", None)
    if isinstance(task_aircraft_counts, dict):
        try:
            if agent_id in task_aircraft_counts:
                max_missiles = int(getattr(getattr(self, "missile_manager", None), "MAX_MISSILES_PER_AIRCRAFT", 4))
                fired = max(0, int(task_aircraft_counts.get(agent_id, 0)))
                candidates.append(max(0, max_missiles - fired))
        except Exception:
            pass

    missile_manager = getattr(self, "missile_manager", None)
    if missile_manager is not None and isinstance(getattr(missile_manager, "aircraft_missile_counts", None), dict):
        try:
            if agent_id in missile_manager.aircraft_missile_counts:
                max_missiles = int(getattr(missile_manager, "MAX_MISSILES_PER_AIRCRAFT", 4))
                fired = max(0, int(missile_manager.aircraft_missile_counts.get(agent_id, 0)))
                candidates.append(max(0, max_missiles - fired))
        except Exception:
            pass

    missile_adapter = getattr(self, "missile_adapter", None)
    if missile_adapter is not None and isinstance(getattr(missile_adapter, "_inventory", None), dict):
        try:
            if agent_id in missile_adapter._inventory:
                inventory = missile_adapter.get_inventory(agent_id)
                candidates.append(max(0, int(inventory)))
        except Exception:
            pass

    if not candidates:
        return 0
    return int(min(candidates))


def _restore_effective_missiles_left(self, env, agent_id: str, sim: AircraftSimulator, missiles_left: int) -> None:
    missiles_left = max(0, int(missiles_left))
    for attr_name in ("num_left_missiles", "num_missiles"):
        try:
            setattr(sim, attr_name, missiles_left)
        except Exception:
            pass

    state_manager = getattr(self, "state_manager", None)
    if state_manager is not None and hasattr(state_manager, "missiles_remaining"):
        try:
            if agent_id in state_manager.missiles_remaining:
                state_manager.missiles_remaining[agent_id] = missiles_left
        except Exception:
            pass

    max_missiles = int(getattr(getattr(self, "missile_manager", None), "MAX_MISSILES_PER_AIRCRAFT", 4))
    task_aircraft_counts = getattr(self, "aircraft_missile_counts", None)
    if isinstance(task_aircraft_counts, dict):
        try:
            if agent_id in task_aircraft_counts:
                task_aircraft_counts[agent_id] = max(0, max_missiles - missiles_left)
        except Exception:
            pass

    missile_manager = getattr(self, "missile_manager", None)
    if missile_manager is not None and isinstance(getattr(missile_manager, "aircraft_missile_counts", None), dict):
        try:
            if agent_id in missile_manager.aircraft_missile_counts:
                missile_manager.aircraft_missile_counts[agent_id] = max(0, max_missiles - missiles_left)
        except Exception:
            pass

    missile_adapter = getattr(self, "missile_adapter", None)
    if missile_adapter is not None and isinstance(getattr(missile_adapter, "_inventory", None), dict):
        try:
            if agent_id in missile_adapter._inventory:
                missile_adapter._inventory[agent_id] = missiles_left
        except Exception:
            pass


def _has_active_cap_missile_refs(self, env, agent_id: str, sim: AircraftSimulator) -> bool:
    if _enemy_has_active_missile_refs(sim):
        return True

    missile_adapter = getattr(self, "missile_adapter", None)
    if missile_adapter is None:
        return False

    active_states = {"launched", "guiding", "terminal"}
    try:
        for status in getattr(missile_adapter, "_missiles", {}).values():
            state_name = _normalize_missile_state_value(getattr(status, "state", None))
            if state_name not in active_states:
                continue
            guide_id = str(getattr(status, "guide_agent_id", "") or "")
            target_id = str(getattr(status, "target_id", "") or "")
            if guide_id == str(agent_id) or target_id == str(agent_id):
                return True
    except Exception:
        return False
    return False


def _get_enemy_sim_recreate_state(self, agent_id: str) -> dict:
    if not hasattr(self, "_enemy_sim_recreate_state"):
        self._enemy_sim_recreate_state = {}
    return self._enemy_sim_recreate_state.setdefault(
        agent_id,
        {
            "count": 0,
            "last_step": -10**9,
            "window": deque(maxlen=_sim_recreate_window_len(agent_id)),
            "healthy_history": deque(maxlen=_sim_recreate_history_len(agent_id)),
            "times_s": [],
            "generation": 0,
            "recover_until_s": -1e9,
            "posture_hold_until_s": -1e9,
            "formation_rebind_until_s": -1e9,
            "last_reason": "",
        },
    )


def _get_friendly_sim_recreate_state(self, agent_id: str) -> dict:
    if not hasattr(self, "_friendly_sim_recreate_state"):
        self._friendly_sim_recreate_state = {}
    return self._friendly_sim_recreate_state.setdefault(
        agent_id,
        {
            "count": 0,
            "last_step": -10**9,
            "window": deque(maxlen=_sim_recreate_window_len(agent_id)),
            "healthy_history": deque(maxlen=_sim_recreate_history_len(agent_id)),
            "times_s": [],
            "generation": 0,
            "recover_until_s": -1e9,
            "posture_hold_until_s": -1e9,
            "formation_rebind_until_s": -1e9,
            "last_reason": "",
        },
    )


def _build_enemy_recreate_state(
    sim: AircraftSimulator,
    rescue_speed_mps: float,
    *,
    friendly: bool = False,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    roc_fpm: float = 0.0,
) -> dict:
    lon_deg = float(sim.get_property_value(c.position_long_gc_deg))
    lat_deg = float(sim.get_property_value(c.position_lat_geod_deg))
    alt_ft = min(
        float(sim.get_property_value(c.position_h_sl_ft)),
        _task_max_altitude_ft(friendly=friendly),
    )
    hdg_deg = float(sim.get_property_value(c.attitude_psi_deg))
    rescue_speed_fps = float(rescue_speed_mps / 0.3048)
    rescue_speed_kts = float(rescue_speed_mps * 1.9438444924406)
    try:
        current_vc_fps = float(sim.get_property_value(c.velocities_vc_fps))
        current_vt_fps = float(sim.get_property_value(c.velocities_vt_fps))
        vt_to_vc_ratio = float(np.clip(current_vt_fps / max(current_vc_fps, 1.0), 1.0, 3.2))
    except Exception:
        vt_to_vc_ratio = 1.0
    rescue_true_speed_fps = float(rescue_speed_fps * vt_to_vc_ratio)
    rescue_true_speed_kts = float(rescue_true_speed_fps / 1.6878098571011957)
    return {
        "ic_long_gc_deg": lon_deg,
        "ic_lat_geod_deg": lat_deg,
        "ic_h_sl_ft": alt_ft,
        "ic_psi_true_deg": hdg_deg,
        "ic_theta_deg": float(np.clip(pitch_deg, -1.0, 6.0)),
        "ic_phi_deg": float(np.clip(roll_deg, -18.0, 18.0)),
        "ic_u_fps": rescue_true_speed_fps,
        "ic_v_fps": 0.0,
        "ic_w_fps": 0.0,
        "ic_vc_kts": rescue_speed_kts,
        "ic_vt_fps": rescue_true_speed_fps,
        "ic_vt_kts": rescue_true_speed_kts,
        "ic_p_rad_sec": 0.0,
        "ic_q_rad_sec": 0.0,
        "ic_r_rad_sec": 0.0,
        "ic_roc_fpm": float(np.clip(roc_fpm, -800.0, 1800.0)),
    }


def _snapshot_sim_state(sim: AircraftSimulator):
    try:
        lon_deg = float(sim.get_property_value(c.position_long_gc_deg))
        lat_deg = float(sim.get_property_value(c.position_lat_geod_deg))
        alt_ft = float(sim.get_property_value(c.position_h_sl_ft))
        hdg_deg = float(sim.get_property_value(c.attitude_psi_deg))
        roll_deg = float(np.degrees(sim.get_property_value(c.attitude_roll_rad)))
        vc_mps = float(sim.get_property_value(c.velocities_vc_mps))
        vc_fps = float(sim.get_property_value(c.velocities_vc_fps))
        vt_fps = float(sim.get_property_value(c.velocities_vt_fps))
    except Exception:
        return None

    vt_to_vc_ratio = float(np.clip(vt_fps / max(vc_fps, 1.0), 1.0, 3.2))
    return {
        "ic_long_gc_deg": lon_deg,
        "ic_lat_geod_deg": lat_deg,
        "ic_h_sl_ft": alt_ft,
        "ic_psi_true_deg": hdg_deg,
        "roll_deg": roll_deg,
        "vc_mps": vc_mps,
        "vt_to_vc_ratio": vt_to_vc_ratio,
    }


def _build_recreate_state_from_snapshot(
    snapshot: dict,
    rescue_speed_mps: float,
    *,
    friendly: bool = False,
    reference_snapshot=None,
    pitch_deg: float = 0.0,
    roll_deg=None,
    roc_fpm: float = 0.0,
) -> dict:
    rescue_speed_fps = float(rescue_speed_mps / 0.3048)
    rescue_speed_kts = float(rescue_speed_mps * 1.9438444924406)
    reference = reference_snapshot if reference_snapshot is not None else snapshot
    vt_to_vc_ratio = float(np.clip(reference.get("vt_to_vc_ratio", snapshot.get("vt_to_vc_ratio", 1.0)), 1.0, 3.2))
    rescue_true_speed_fps = float(rescue_speed_fps * vt_to_vc_ratio)
    rescue_true_speed_kts = float(rescue_true_speed_fps / 1.6878098571011957)
    snapshot_roll_deg = float(snapshot.get("roll_deg", 0.0) or 0.0)
    if roll_deg is None:
        roll_deg = float(np.clip(snapshot_roll_deg * 0.35, -12.0, 12.0))
    capped_alt_ft = min(float(snapshot["ic_h_sl_ft"]), _task_max_altitude_ft(friendly=friendly))
    return {
        "ic_long_gc_deg": float(snapshot["ic_long_gc_deg"]),
        "ic_lat_geod_deg": float(snapshot["ic_lat_geod_deg"]),
        "ic_h_sl_ft": capped_alt_ft,
        "ic_psi_true_deg": float(snapshot["ic_psi_true_deg"]),
        "ic_theta_deg": float(np.clip(pitch_deg, -1.0, 6.0)),
        "ic_phi_deg": float(np.clip(roll_deg, -18.0, 18.0)),
        "ic_u_fps": rescue_true_speed_fps,
        "ic_v_fps": 0.0,
        "ic_w_fps": 0.0,
        "ic_vc_kts": rescue_speed_kts,
        "ic_vt_fps": rescue_true_speed_fps,
        "ic_vt_kts": rescue_true_speed_kts,
        "ic_p_rad_sec": 0.0,
        "ic_q_rad_sec": 0.0,
        "ic_r_rad_sec": 0.0,
        "ic_roc_fpm": float(np.clip(roc_fpm, -800.0, 1800.0)),
    }


def _is_healthy_recreate_sample(sample: dict, *, friendly: bool) -> bool:
    min_alt = 4200.0 if friendly else 4500.0
    min_vc = 205.0 if friendly else 195.0
    min_vup = -2.5 if friendly else -3.0
    max_pitch = 8.0 if friendly else 10.0
    max_alt = _task_max_altitude_m(friendly=friendly)
    return bool(
        float(sample.get("alt", 0.0)) >= min_alt
        and float(sample.get("alt", 0.0)) <= max_alt
        and float(sample.get("vc", 0.0)) >= min_vc
        and float(sample.get("vup", -999.0)) >= min_vup
        and abs(float(sample.get("pitch_deg", 0.0))) <= max_pitch
    )


def _remember_healthy_recreate_snapshot(state: dict, sim: AircraftSimulator, sample: dict, *, friendly: bool) -> None:
    history = state.get("healthy_history")
    if history is None:
        return
    snapshot = _snapshot_sim_state(sim)
    if snapshot is None:
        return
    snapshot["healthy"] = _is_healthy_recreate_sample(sample, friendly=friendly)
    snapshot.update(
        {
            "step": int(sample.get("step", 0)),
            "time_s": float(sample.get("time_s", 0.0)),
            "alt": float(sample.get("alt", 0.0)),
            "vc": float(sample.get("vc", 0.0)),
            "vup": float(sample.get("vup", 0.0)),
            "energy": float(sample.get("energy", 0.0)),
            "pitch_deg": float(sample.get("pitch_deg", 0.0)),
        }
    )
    history.append(snapshot)


def _recreate_snapshot_score(snapshot: dict, *, friendly: bool) -> float:
    alt_m = float(snapshot.get("alt", 0.0))
    vc_mps = float(snapshot.get("vc", 0.0))
    v_up_mps = float(snapshot.get("vup", -999.0))
    pitch_deg = abs(float(snapshot.get("pitch_deg", 0.0)))
    energy = float(snapshot.get("energy", 0.0))
    altitude_cap_m = _task_max_altitude_m(friendly=friendly)
    healthy_bonus = 1000.0 if bool(snapshot.get("healthy", False)) else 0.0
    altitude_bonus = 80.0 if alt_m >= (6500.0 if friendly else 6000.0) else 0.0
    altitude_cap_penalty = max(0.0, alt_m - altitude_cap_m) * 0.8
    return float(
        healthy_bonus
        + altitude_bonus
        + min(alt_m, 14000.0) * 0.08
        + min(vc_mps, 280.0) * 4.5
        + max(v_up_mps, -2.0) * 18.0
        - pitch_deg * 16.0
        - altitude_cap_penalty
        + energy * 0.0002
    )


def _latest_healthy_recreate_snapshot(state: dict, now_step: int, agent_id: str):
    history = state.get("healthy_history")
    if not history:
        return None
    now_s = float(now_step * 0.2)
    max_age_s = _sim_recreate_reference_age_s(agent_id)
    recent_recreate_count = sum(
        1
        for ts in state.get("times_s", [])
        if (now_s - float(ts)) <= 90.0
    )
    rewind_age_s = 0.0
    if recent_recreate_count >= 2:
        rewind_age_s = min(12.0, 2.5 * recent_recreate_count)
        max_age_s = max(max_age_s, rewind_age_s + 8.0)
    recent_history = [
        snapshot
        for snapshot in history
        if max(0.0, (now_step - int(snapshot.get("step", now_step))) * 0.2) <= max_age_s
    ]
    if not recent_history:
        recent_history = list(history)
    friendly = str(agent_id).startswith("A")

    def _age_s(snapshot: dict) -> float:
        return max(0.0, (now_step - int(snapshot.get("step", now_step))) * 0.2)

    healthy_candidates = [dict(snapshot) for snapshot in recent_history if bool(snapshot.get("healthy", False))]
    if healthy_candidates:
        rewind_candidates = [snapshot for snapshot in healthy_candidates if _age_s(snapshot) >= rewind_age_s]
        candidate_pool = rewind_candidates or healthy_candidates
        return max(
            candidate_pool,
            key=lambda snapshot: (
                _recreate_snapshot_score(snapshot, friendly=friendly),
                -_age_s(snapshot),
            ),
        )

    rewind_candidates = [dict(snapshot) for snapshot in recent_history if _age_s(snapshot) >= rewind_age_s]
    candidate_pool = rewind_candidates or [dict(snapshot) for snapshot in recent_history]
    return max(
        candidate_pool,
        key=lambda snapshot: (
            _recreate_snapshot_score(snapshot, friendly=friendly),
            -_age_s(snapshot),
        ),
    )


def _should_recreate_from_window(
    window: deque,
    *,
    altitude_floor_m: float,
    low_speed_trigger_mps: float,
    avg_speed_trigger_mps: float,
    sink_trigger_mps: float,
    energy_drop_trigger: float,
    alt_drop_trigger_m: float,
    emergency_speed_trigger_mps: float,
    emergency_sink_trigger_mps: float,
    hard_speed_trigger_mps: float,
) -> bool:
    if len(window) < window.maxlen:
        return False
    first = window[0]
    last = window[-1]

    energy_drop = float(last["energy"] - first["energy"])
    avg_vup = float(sum(x["vup"] for x in window) / len(window))
    avg_vc = float(sum(x["vc"] for x in window) / len(window))
    alt_drop = float(last["alt"] - first["alt"])
    vc_drop = float(last["vc"] - first["vc"])

    early_sink = (
        last["vc"] < low_speed_trigger_mps
        and avg_vc < avg_speed_trigger_mps
        and avg_vup < sink_trigger_mps
        and energy_drop < energy_drop_trigger
        and alt_drop < alt_drop_trigger_m
        and vc_drop < -2.0
    )
    severe_low_speed_sink = (
        last["vc"] < hard_speed_trigger_mps
        and avg_vc < max(hard_speed_trigger_mps + 10.0, avg_speed_trigger_mps - 18.0)
        and avg_vup < (sink_trigger_mps - 1.5)
        and alt_drop < (alt_drop_trigger_m - 10.0)
        and energy_drop < (energy_drop_trigger - 400.0)
    )
    emergency_sink = (
        last["vc"] < emergency_speed_trigger_mps
        and last["vup"] < emergency_sink_trigger_mps
        and alt_drop < -18.0
    )
    low_altitude_emergency = (
        last["alt"] < altitude_floor_m
        and (
            last["vc"] < max(hard_speed_trigger_mps + 5.0, emergency_speed_trigger_mps + 10.0)
            or last["vup"] < emergency_sink_trigger_mps
            or alt_drop < max(-18.0, alt_drop_trigger_m * 0.7)
        )
    )
    return bool(early_sink or severe_low_speed_sink or emergency_sink or low_altitude_emergency)


def _should_recreate_enemy_from_window(window: deque) -> bool:
    return _should_recreate_from_window(
        window,
        altitude_floor_m=3000.0,
        low_speed_trigger_mps=195.0,
        avg_speed_trigger_mps=210.0,
        sink_trigger_mps=-4.5,
        energy_drop_trigger=-900.0,
        alt_drop_trigger_m=-35.0,
        emergency_speed_trigger_mps=170.0,
        emergency_sink_trigger_mps=-7.5,
        hard_speed_trigger_mps=180.0,
    )


def _should_recreate_friendly_from_window(window: deque) -> bool:
    return _should_recreate_from_window(
        window,
        altitude_floor_m=2800.0,
        low_speed_trigger_mps=178.0,
        avg_speed_trigger_mps=192.0,
        sink_trigger_mps=-5.2,
        energy_drop_trigger=-1200.0,
        alt_drop_trigger_m=-42.0,
        emergency_speed_trigger_mps=152.0,
        emergency_sink_trigger_mps=-8.5,
        hard_speed_trigger_mps=166.0,
    )


def _friendly_guard_recreate_stable_enough(
    *,
    in_guard_patrol: bool,
    under_active: bool,
    missiles_left: int,
    alt_m: float,
    vc_mps: float,
    v_up_mps: float,
    pitch_deg: float,
    roll_deg: float,
    recent_recreate_count: int,
    steps_since_last: int,
    cooldown_steps: int,
    hard_emergency: bool,
    severe_emergency: bool,
    sink_guard_emergency: bool,
    deep_sink_emergency: bool,
) -> bool:
    if not in_guard_patrol or under_active:
        return False
    if hard_emergency or deep_sink_emergency:
        return False
    if alt_m < 4800.0 or vc_mps < 145.0 or v_up_mps < -18.0:
        return False
    if abs(pitch_deg) > 18.0 or abs(roll_deg) > 85.0:
        return False
    if sink_guard_emergency and alt_m < 6200.0:
        return False

    low_inventory_guard = missiles_left <= 1
    if not low_inventory_guard and recent_recreate_count < 2:
        return False

    long_guard_cooldown = max(600, int(cooldown_steps * 2.2))
    if recent_recreate_count >= 2:
        return True
    if low_inventory_guard and recent_recreate_count >= 1 and steps_since_last < long_guard_cooldown:
        return True
    if severe_emergency and not sink_guard_emergency:
        return False
    return False


def _enemy_returning_state(self, agent_id: str) -> bool:
    adapter = getattr(self, "enemy_adapter", None)
    combat_ai = getattr(adapter, "_combat_ai", None) if adapter is not None else None
    phases = getattr(combat_ai, "_enemy_phases", None) if combat_ai is not None else None
    if isinstance(phases, dict):
        return str(phases.get(agent_id, "")).upper() == "RETURNING"
    return False


def _rebuild_enemy_relationships(env) -> None:
    try:
        if hasattr(env, "_setup_formation_relationships"):
            env._setup_formation_relationships()
    except Exception:
        pass


def _enemy_has_active_missile_refs(sim: AircraftSimulator) -> bool:
    launch_active = any(getattr(missile, "is_alive", False) for missile in getattr(sim, "launch_missiles", []))
    under_active = any(getattr(missile, "is_alive", False) for missile in getattr(sim, "under_missiles", []))
    return bool(launch_active or under_active)


def _maybe_recreate_enemy_simulator(self, env, agent_id: str) -> bool:
    if not str(agent_id).startswith("B"):
        return False
    if not bool(getattr(self, "enemy_sim_recreate_enabled", False)):
        return False
    if bool(getattr(self, "enemy_rule_lowlevel_enabled", False)):
        return False

    sim = env.agents.get(agent_id)
    if sim is None or not getattr(sim, "is_alive", False):
        return False

    state = _get_enemy_sim_recreate_state(self, agent_id)
    now_step = int(self.step_count)
    current_time_s = float(now_step * 0.2)

    try:
        alt_m = float(sim.get_property_value(c.position_h_sl_m))
        vc_mps = float(sim.get_property_value(c.velocities_vc_mps))
        v_up_mps = -float(sim.get_property_value(c.velocities_v_down_mps))
        pitch_deg = float(np.degrees(sim.get_property_value(c.attitude_theta_rad)))
        roll_deg = float(np.degrees(sim.get_property_value(c.attitude_roll_rad)))
        energy = 9.81 * alt_m + 0.5 * vc_mps * vc_mps
    except Exception:
        return False

    window = state["window"]
    sample = {
        "alt": alt_m,
        "vc": vc_mps,
        "vup": v_up_mps,
        "energy": energy,
        "pitch_deg": pitch_deg,
        "step": now_step,
        "time_s": float(now_step * 0.2),
    }
    window.append(sample)
    _remember_healthy_recreate_snapshot(state, sim, sample, friendly=False)
    if now_step < 240:
        return False
    should_recreate = _should_recreate_enemy_from_window(window)
    if (not should_recreate) and _enemy_returning_state(self, agent_id) and len(window) >= max(8, window.maxlen // 2):
        first = window[0]
        last = window[-1]
        energy_drop = float(last["energy"] - first["energy"])
        avg_vup = float(sum(x["vup"] for x in window) / len(window))
        avg_vc = float(sum(x["vc"] for x in window) / len(window))
        alt_drop = float(last["alt"] - first["alt"])
        should_recreate = (
            last["alt"] > 6000.0
            and last["vc"] < 165.0
            and avg_vc < 180.0
            and avg_vup < -5.0
            and alt_drop < -40.0
            and energy_drop < -1200.0
        )
    severe_emergency = (
        len(window) >= max(6, window.maxlen // 2)
        and (
            vc_mps < 138.0
            or v_up_mps < -12.0
            or (alt_m < 3800.0 and v_up_mps < -8.0)
            or (alt_m < 2400.0 and vc_mps < 165.0)
        )
    )
    if not should_recreate and severe_emergency:
        should_recreate = True
    cooldown_steps = _sim_recreate_cooldown_steps(agent_id)
    steps_since_last = now_step - int(state.get("last_step", -10**9))
    recover_until_s = float(state.get("recover_until_s", -1e9))
    if current_time_s < recover_until_s and not severe_emergency:
        if vc_mps >= 145.0 and v_up_mps >= -9.0:
            return False
    cooldown_bypass = (
        severe_emergency
        and steps_since_last >= max(120, int(cooldown_steps * 0.65))
        and (alt_m < 4500.0 or vc_mps < 150.0 or v_up_mps < -11.0)
    )
    if steps_since_last < cooldown_steps and not cooldown_bypass:
        return False
    max_recreate_count = int(getattr(self, "enemy_sim_recreate_max_count", 8))
    recent_recreate_times = [
        float(ts)
        for ts in state.get("times_s", [])
        if (current_time_s - float(ts)) <= 60.0
    ]
    if (
        len(recent_recreate_times) >= 2
        and steps_since_last < max(420, int(cooldown_steps * 1.5))
        and not severe_emergency
    ):
        return False
    count_cap_bypass = (
        severe_emergency
        and (
            vc_mps < 132.0
            or v_up_mps < -14.0
            or (alt_m < 3200.0 and vc_mps < 165.0)
        )
        and steps_since_last >= max(180, int(cooldown_steps * 0.95))
        and len(recent_recreate_times) < 2
        and (alt_m < 4500.0 or vc_mps < 150.0)
    )
    if int(state.get("count", 0)) >= max_recreate_count and not count_cap_bypass:
        return False
    hard_emergency = bool(
        vc_mps < 132.0
        or v_up_mps < -14.0
        or (alt_m < 4500.0 and vc_mps < 175.0)
    )
    high_energy_hold = bool(
        not severe_emergency
        and not hard_emergency
        and (
            (alt_m >= 10500.0 and vc_mps >= 160.0 and v_up_mps >= -8.5 and abs(pitch_deg) <= 10.0 and abs(roll_deg) <= 50.0)
            or (alt_m >= 8500.0 and vc_mps >= 172.0 and v_up_mps >= -6.5 and abs(pitch_deg) <= 8.0 and abs(roll_deg) <= 35.0)
        )
    )
    if high_energy_hold:
        return False
    if _has_active_cap_missile_refs(self, env, agent_id, sim) and not hard_emergency:
        return False
    if not should_recreate:
        return False

    healthy_snapshot = _latest_healthy_recreate_snapshot(state, now_step, agent_id)
    rescue_speed_mps = _estimate_recreate_speed(window, vc_mps, friendly=False)
    recreate_source = "current"
    snapshot_age_s = 0.0
    anchor_mode = "current"
    if healthy_snapshot is not None:
        snapshot_vc = float(healthy_snapshot.get("vc", rescue_speed_mps))
        rescue_speed_mps = float(np.clip(max(rescue_speed_mps, snapshot_vc - 8.0), 160.0, 235.0))
    rescue_speed_mps = _clamp_recreate_speed_gain(
        agent_id,
        vc_mps,
        rescue_speed_mps,
        alt_m=alt_m,
        severe_emergency=severe_emergency,
        recent_recreate_count=len(recent_recreate_times),
    )
    rescue_pitch_deg, rescue_roc_fpm = _estimate_recreate_pitch_and_roc(
        agent_id,
        alt_m=alt_m,
        current_vc_mps=vc_mps,
        current_vup_mps=v_up_mps,
        severe_emergency=severe_emergency,
        reference_snapshot=healthy_snapshot if healthy_snapshot is not None else sample,
    )
    if healthy_snapshot is not None:
        current_snapshot = _snapshot_sim_state(sim)
        anchor_snapshot = current_snapshot if current_snapshot is not None else healthy_snapshot
        new_state = _build_recreate_state_from_snapshot(
            anchor_snapshot,
            rescue_speed_mps,
            reference_snapshot=healthy_snapshot,
            pitch_deg=rescue_pitch_deg,
            roc_fpm=rescue_roc_fpm,
        )
        snapshot_age_s = max(0.0, (now_step - int(healthy_snapshot.get("step", now_step))) * 0.2)
        if current_snapshot is None:
            anchor_mode = "snapshot"
        if bool(healthy_snapshot.get("healthy", False)):
            recreate_source = "healthy_reference" if current_snapshot is not None else ("healthy_rewind" if snapshot_age_s >= 1.0 else "healthy_history")
        else:
            recreate_source = "recent_reference" if current_snapshot is not None else ("recent_rewind" if snapshot_age_s >= 1.0 else "recent_history")
    else:
        new_state = _build_enemy_recreate_state(
            sim,
            rescue_speed_mps,
            pitch_deg=rescue_pitch_deg,
            roc_fpm=rescue_roc_fpm,
        )
    missiles_left = _get_effective_missiles_left(self, env, agent_id, sim)
    bloods = float(getattr(sim, "bloods", 100.0))
    was_leader = bool(getattr(sim, "is_leader", False)() if hasattr(sim, "is_leader") else getattr(sim, "_is_leader", False))

    log.warning(
        "[%s][T+%07.1fs][enemy_sim_recreate][trigger] alt=%.1fm vc=%.1fm/s v_up=%.2fm/s pitch=%.2fdeg count=%d rescue_speed=%.1fm/s source=%s age=%.1fs anchor=%s severe=%s cap_bypass=%s",
        agent_id,
        current_time_s,
        alt_m,
        vc_mps,
        v_up_mps,
        pitch_deg,
        int(state.get("count", 0)),
        rescue_speed_mps,
        recreate_source,
        snapshot_age_s,
        anchor_mode,
        severe_emergency,
        count_cap_bypass,
    )

    try:
        sim.reload(new_state=new_state)
        _restore_effective_missiles_left(self, env, agent_id, sim, missiles_left)
        sim.bloods = bloods
        if hasattr(sim, "set_leader"):
            sim.set_leader(was_leader)
        _rebuild_enemy_relationships(env)

        if hasattr(self, "_inner_rnn_states"):
            self._inner_rnn_states[agent_id] = np.zeros((1, 1, 128), dtype=np.float32)
        if hasattr(self, "_enemy_rnn_reset_state") and agent_id in self._enemy_rnn_reset_state:
            self._enemy_rnn_reset_state[agent_id]["abnormal_output_hits"] = 0
            self._enemy_rnn_reset_state[agent_id]["was_in_hard_recovery"] = False
            self._enemy_rnn_reset_state[agent_id]["model_rejoin_until_step"] = -1

        adapter = getattr(self, "enemy_adapter", None)
        combat_ai = getattr(adapter, "_combat_ai", None) if adapter is not None else None
        if combat_ai is not None and hasattr(combat_ai, "reset_agent"):
            try:
                combat_ai.reset_agent(agent_id)
            except Exception:
                pass
    except Exception as exc:
        log.error("[%s][enemy_sim_recreate] failed: %s", agent_id, exc)
        return False

    state["count"] = int(state.get("count", 0)) + 1
    state["last_step"] = now_step
    state["times_s"].append(current_time_s)
    posture_hold_s = _sim_recreate_posture_hold_s(agent_id) + min(12.0, 3.0 * len(recent_recreate_times))
    state["recover_until_s"] = max(float(state.get("recover_until_s", -1e9)), current_time_s + 22.0)
    state["posture_hold_until_s"] = max(float(state.get("posture_hold_until_s", -1e9)), current_time_s + posture_hold_s)
    state["last_reason"] = "severe_emergency" if severe_emergency else "energy_recover"
    window.clear()

    try:
        alt_after = float(sim.get_property_value(c.position_h_sl_m))
        vc_after = float(sim.get_property_value(c.velocities_vc_mps))
        pitch_after = float(np.degrees(sim.get_property_value(c.attitude_theta_rad)))
        v_up_after = -float(sim.get_property_value(c.velocities_v_down_mps))
        energy_after = 9.81 * alt_after + 0.5 * vc_after * vc_after
        post_sample = {
            "alt": alt_after,
            "vc": vc_after,
            "vup": v_up_after,
            "energy": energy_after,
            "pitch_deg": pitch_after,
            "step": now_step,
            "time_s": float(now_step * 0.2),
        }
        window.append(post_sample)
        _remember_healthy_recreate_snapshot(state, sim, post_sample, friendly=False)
        log.warning(
            "[%s][T+%07.1fs][enemy_sim_recreate][applied] alt=%.1fm vc=%.1fm/s pitch=%.2fdeg v_up=%.2fm/s missiles_left=%d",
            agent_id,
            current_time_s,
            alt_after,
            vc_after,
            pitch_after,
            v_up_after,
            missiles_left,
        )
    except Exception:
        pass

    return True


def _maybe_recreate_friendly_simulator(self, env, agent_id: str) -> bool:
    if not str(agent_id).startswith("A"):
        return False
    if not bool(getattr(self, "friendly_sim_recreate_enabled", False)):
        return False

    sim = env.agents.get(agent_id)
    if sim is None or not getattr(sim, "is_alive", False):
        return False

    state = _get_friendly_sim_recreate_state(self, agent_id)
    now_step = int(self.step_count)
    current_time_s = float(now_step * 0.2)

    try:
        alt_m = float(sim.get_property_value(c.position_h_sl_m))
        vc_mps = float(sim.get_property_value(c.velocities_vc_mps))
        v_up_mps = -float(sim.get_property_value(c.velocities_v_down_mps))
        pitch_deg = float(np.degrees(sim.get_property_value(c.attitude_theta_rad)))
        roll_deg = float(np.degrees(sim.get_property_value(c.attitude_roll_rad)))
        energy = 9.81 * alt_m + 0.5 * vc_mps * vc_mps
    except Exception:
        return False

    window = state["window"]
    sample = {
        "alt": alt_m,
        "vc": vc_mps,
        "vup": v_up_mps,
        "energy": energy,
        "pitch_deg": pitch_deg,
        "step": now_step,
        "time_s": float(now_step * 0.2),
    }
    window.append(sample)
    _remember_healthy_recreate_snapshot(state, sim, sample, friendly=True)
    if now_step < 240:
        return False
    current_tactic = ""
    try:
        current_tactic = str(self._get_agent_tactic(agent_id) or "")
    except Exception:
        current_tactic = ""
    in_guard_patrol = current_tactic.upper() == "DEFENSIVE_GUARD"
    missiles_left = _get_effective_missiles_left(self, env, agent_id, sim)
    under_active = any(getattr(missile, "is_alive", False) for missile in getattr(sim, "under_missiles", []))
    should_recreate = _should_recreate_friendly_from_window(window)
    altitude_cap_m = _task_max_altitude_m(friendly=True)
    sink_guard_emergency = bool(
        (alt_m < 9500.0 and v_up_mps < -7.0)
        or (alt_m >= 7800.0 and alt_m < 9500.0 and vc_mps < 182.0 and v_up_mps < -5.5)
        or (alt_m < 7000.0 and (vc_mps < 170.0 or v_up_mps < -5.0))
        or (alt_m < 6200.0 and (vc_mps < 182.0 or v_up_mps < -4.5))
    )
    deep_sink_emergency = bool(
        v_up_mps < -24.0
        or (alt_m < 7000.0 and vc_mps < 128.0)
        or (alt_m < 6000.0 and (vc_mps < 145.0 or v_up_mps < -9.0))
    )
    severe_emergency = (
        len(window) >= max(3, window.maxlen // 4)
        and (
            v_up_mps < -10.5
            or (alt_m < 7500.0 and vc_mps < 158.0 and v_up_mps < -4.0)
            or (alt_m < 6500.0 and (vc_mps < 178.0 or v_up_mps < -6.5))
            or (alt_m < 4200.0 and (vc_mps < 198.0 or v_up_mps < -4.5))
            or deep_sink_emergency
            or (sink_guard_emergency and alt_m < 9000.0)
        )
    )
    if not should_recreate and (severe_emergency or (sink_guard_emergency and len(window) >= max(3, window.maxlen // 3))):
        should_recreate = True
    cooldown_steps = _sim_recreate_cooldown_steps(agent_id)
    steps_since_last = now_step - int(state.get("last_step", -10**9))
    hard_emergency = bool(
        vc_mps < 138.0
        or v_up_mps < -13.0
        or (alt_m < 6000.0 and vc_mps < 176.0)
        or (alt_m < 4500.0 and vc_mps < 188.0)
        or deep_sink_emergency
    )
    recover_until_s = float(state.get("recover_until_s", -1e9))
    if current_time_s < recover_until_s and not severe_emergency and not sink_guard_emergency:
        if vc_mps >= 132.0 and v_up_mps >= -8.0:
            return False
    cooldown_bypass = (
        severe_emergency
        and hard_emergency
        and steps_since_last >= max(120, cooldown_steps // 2)
    )
    if steps_since_last < cooldown_steps and not cooldown_bypass:
        return False
    max_recreate_count = int(getattr(self, "friendly_sim_recreate_max_count", 8))
    recent_recreate_times = [
        float(ts)
        for ts in state.get("times_s", [])
        if (current_time_s - float(ts)) <= 60.0
    ]
    high_alt_visual_guard = bool(
        alt_m >= 10000.0
        and not under_active
        and len(recent_recreate_times) >= 1
        and steps_since_last < max(520, int(cooldown_steps * 2.0))
        and vc_mps >= 120.0
        and v_up_mps > -24.0
    )
    if high_alt_visual_guard:
        return False
    if (
        len(recent_recreate_times) >= 2
        and steps_since_last < max(340, int(cooldown_steps * 1.35))
        and not severe_emergency
        and not sink_guard_emergency
    ):
        return False
    count_cap_bypass = (
        (
            severe_emergency
            and hard_emergency
            and steps_since_last >= max(180, int(cooldown_steps * 0.90))
            and len(recent_recreate_times) < 1
            and alt_m < 5500.0
        )
        or (
            deep_sink_emergency
            and steps_since_last >= max(110, int(cooldown_steps * 0.45))
            and len(recent_recreate_times) < 2
        )
    )
    if int(state.get("count", 0)) >= max_recreate_count and not count_cap_bypass:
        return False
    if _friendly_guard_recreate_stable_enough(
        in_guard_patrol=in_guard_patrol,
        under_active=under_active,
        missiles_left=missiles_left,
        alt_m=alt_m,
        vc_mps=vc_mps,
        v_up_mps=v_up_mps,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        recent_recreate_count=len(recent_recreate_times),
        steps_since_last=steps_since_last,
        cooldown_steps=cooldown_steps,
        hard_emergency=hard_emergency,
        severe_emergency=severe_emergency,
        sink_guard_emergency=sink_guard_emergency,
        deep_sink_emergency=deep_sink_emergency,
    ):
        return False
    if (
        alt_m >= (altitude_cap_m - 300.0)
        and not under_active
        and not deep_sink_emergency
        and vc_mps >= 145.0
        and v_up_mps >= -12.0
        and abs(pitch_deg) <= 10.5
    ):
        return False
    if (
        in_guard_patrol
        and not under_active
        and not hard_emergency
        and not severe_emergency
        and not sink_guard_emergency
        and len(recent_recreate_times) >= 1
    ):
        high_hold_ok = alt_m >= 12000.0 and vc_mps >= 128.0 and v_up_mps >= -9.5 and abs(pitch_deg) <= 8.5
        medium_hold_ok = alt_m >= 9000.0 and vc_mps >= 120.0 and v_up_mps >= -7.5 and abs(roll_deg) <= 35.0
        if high_hold_ok or medium_hold_ok:
            return False
    if (
        not under_active
        and not hard_emergency
        and not severe_emergency
        and not sink_guard_emergency
        and (
            (alt_m >= 11000.0 and vc_mps >= 138.0 and v_up_mps >= -9.0 and abs(pitch_deg) <= 9.5 and abs(roll_deg) <= 45.0)
            or (alt_m >= 9000.0 and vc_mps >= 148.0 and v_up_mps >= -7.5 and abs(pitch_deg) <= 7.5 and abs(roll_deg) <= 32.0)
        )
    ):
        return False
    if _has_active_cap_missile_refs(self, env, agent_id, sim) and not (hard_emergency or sink_guard_emergency):
        return False
    if not should_recreate:
        return False

    healthy_snapshot = _latest_healthy_recreate_snapshot(state, now_step, agent_id)
    rescue_speed_mps = _estimate_recreate_speed(window, vc_mps, friendly=True)
    recreate_source = "current"
    snapshot_age_s = 0.0
    anchor_mode = "current"
    if healthy_snapshot is not None:
        snapshot_vc = float(healthy_snapshot.get("vc", rescue_speed_mps))
        rescue_speed_mps = float(np.clip(max(rescue_speed_mps, snapshot_vc - 10.0), 165.0, 228.0))
    rescue_speed_mps = _clamp_recreate_speed_gain(
        agent_id,
        vc_mps,
        rescue_speed_mps,
        alt_m=alt_m,
        severe_emergency=severe_emergency,
        recent_recreate_count=len(recent_recreate_times),
    )
    rescue_pitch_deg, rescue_roc_fpm = _estimate_recreate_pitch_and_roc(
        agent_id,
        alt_m=alt_m,
        current_vc_mps=vc_mps,
        current_vup_mps=v_up_mps,
        severe_emergency=severe_emergency,
        reference_snapshot=healthy_snapshot if healthy_snapshot is not None else sample,
    )
    rescue_roll_deg = float(np.clip(roll_deg * (0.18 if severe_emergency else 0.40), -10.0, 10.0))
    if healthy_snapshot is not None:
        current_snapshot = _snapshot_sim_state(sim)
        snapshot_age_s = max(0.0, (now_step - int(healthy_snapshot.get("step", now_step))) * 0.2)
        healthy_alt_m = float(healthy_snapshot.get("alt", alt_m))
        altitude_gap_m = max(0.0, healthy_alt_m - alt_m)
        use_reference_anchor = bool(
            deep_sink_emergency
            and alt_m < 6200.0
            and snapshot_age_s <= 2.5
            and altitude_gap_m >= 220.0
            and len(recent_recreate_times) < 2
        )
        if in_guard_patrol and not under_active:
            use_reference_anchor = bool(
                use_reference_anchor
                and missiles_left <= 0
                and alt_m < 4800.0
                and v_up_mps < -18.0
            )
        anchor_snapshot = healthy_snapshot if use_reference_anchor else (current_snapshot if current_snapshot is not None else healthy_snapshot)
        new_state = _build_recreate_state_from_snapshot(
            anchor_snapshot,
            rescue_speed_mps,
            friendly=True,
            reference_snapshot=healthy_snapshot,
            pitch_deg=rescue_pitch_deg,
            roll_deg=rescue_roll_deg,
            roc_fpm=rescue_roc_fpm,
        )
        if use_reference_anchor:
            anchor_mode = "reference_rewind"
        elif current_snapshot is None:
            anchor_mode = "snapshot"
        if bool(healthy_snapshot.get("healthy", False)):
            recreate_source = (
                "healthy_reference"
                if (current_snapshot is not None and not use_reference_anchor)
                else ("healthy_rewind" if snapshot_age_s >= 1.0 else "healthy_history")
            )
        else:
            recreate_source = (
                "recent_reference"
                if (current_snapshot is not None and not use_reference_anchor)
                else ("recent_rewind" if snapshot_age_s >= 1.0 else "recent_history")
            )
    else:
        new_state = _build_enemy_recreate_state(
            sim,
            rescue_speed_mps,
            friendly=True,
            pitch_deg=rescue_pitch_deg,
            roll_deg=rescue_roll_deg,
            roc_fpm=rescue_roc_fpm,
        )
    bloods = float(getattr(sim, "bloods", 100.0))
    was_leader = bool(getattr(sim, "is_leader", False)() if hasattr(sim, "is_leader") else getattr(sim, "_is_leader", False))

    log.warning(
        "[%s][T+%07.1fs][friendly_sim_recreate][trigger] alt=%.1fm vc=%.1fm/s v_up=%.2fm/s pitch=%.2fdeg count=%d generation=%d rescue_speed=%.1fm/s source=%s age=%.1fs anchor=%s severe=%s cap_bypass=%s",
        agent_id,
        float(now_step * 0.2),
        alt_m,
        vc_mps,
        v_up_mps,
        pitch_deg,
        int(state.get("count", 0)),
        int(state.get("generation", 0)) + 1,
        rescue_speed_mps,
        recreate_source,
        snapshot_age_s,
        anchor_mode,
        severe_emergency,
        count_cap_bypass,
    )

    try:
        sim.reload(new_state=new_state)
        _restore_effective_missiles_left(self, env, agent_id, sim, missiles_left)
        sim.bloods = bloods
        if hasattr(sim, "set_leader"):
            sim.set_leader(was_leader)
        _rebuild_enemy_relationships(env)

        if hasattr(self, "_inner_rnn_states"):
            self._inner_rnn_states[agent_id] = np.zeros((1, 1, 128), dtype=np.float32)
        for attr_name in (
            "_friendly_recovery_state",
            "_native_residual_target_state",
            "_safety_diag_last_print_step",
            "_safety_diag_last_level",
            "aircraft_states",
        ):
            store = getattr(self, attr_name, None)
            if isinstance(store, dict):
                store.pop(agent_id, None)
    except Exception as exc:
        log.error("[%s][friendly_sim_recreate] failed: %s", agent_id, exc)
        return False

    state["count"] = int(state.get("count", 0)) + 1
    state["last_step"] = now_step
    state["times_s"].append(current_time_s)
    state["generation"] = int(state.get("generation", 0)) + 1
    posture_hold_s = _sim_recreate_posture_hold_s(agent_id) + min(16.0, 4.0 * len(recent_recreate_times))
    if in_guard_patrol and not under_active:
        posture_hold_s += 6.0
    state["recover_until_s"] = max(float(state.get("recover_until_s", -1e9)), current_time_s + 20.0)
    state["posture_hold_until_s"] = max(float(state.get("posture_hold_until_s", -1e9)), current_time_s + posture_hold_s)
    state["formation_rebind_until_s"] = max(float(state.get("formation_rebind_until_s", -1e9)), current_time_s + posture_hold_s)
    state["last_reason"] = "guard_hold" if in_guard_patrol and not under_active else ("sink_guard" if sink_guard_emergency else ("severe_emergency" if severe_emergency else "energy_recover"))
    window.clear()

    try:
        alt_after = float(sim.get_property_value(c.position_h_sl_m))
        vc_after = float(sim.get_property_value(c.velocities_vc_mps))
        pitch_after = float(np.degrees(sim.get_property_value(c.attitude_theta_rad)))
        v_up_after = -float(sim.get_property_value(c.velocities_v_down_mps))
        energy_after = 9.81 * alt_after + 0.5 * vc_after * vc_after
        post_sample = {
            "alt": alt_after,
            "vc": vc_after,
            "vup": v_up_after,
            "energy": energy_after,
            "pitch_deg": pitch_after,
            "step": now_step,
            "time_s": float(now_step * 0.2),
        }
        window.append(post_sample)
        _remember_healthy_recreate_snapshot(state, sim, post_sample, friendly=True)
        log.warning(
            "[%s][T+%07.1fs][friendly_sim_recreate][applied] alt=%.1fm vc=%.1fm/s pitch=%.2fdeg v_up=%.2fm/s missiles_left=%d generation=%d posture_hold_until=%.1fs",
            agent_id,
            current_time_s,
            alt_after,
            vc_after,
            pitch_after,
            v_up_after,
            missiles_left,
            int(state.get("generation", 0)),
            float(state.get("posture_hold_until_s", -1e9)),
        )
    except Exception:
        pass

    return True


def _apply_enemy_envelope_protection(
    self,
    env,
    agent_id: str,
    norm_act: np.ndarray,
    alt_cmd: int,
    hdg_cmd: int,
    spd_cmd: int,
) -> np.ndarray:
    profile = _update_enemy_energy_profile(self, env, agent_id)
    if profile is None or profile.get('stage', 'TACTICAL_NORMAL') == 'TACTICAL_NORMAL':
        return norm_act

    try:
        safe_act = _synthesize_enemy_recovery_action(self, profile, base_act=norm_act, hard=False)
        stage = str(profile.get('stage', 'TACTICAL_NORMAL'))
        kind = str(profile.get('kind', 'NEUTRAL_LOW_ENERGY'))
        now_step = int(self.step_count)
        if now_step - int(profile.get('last_log_step', -9999)) >= 300:
            profile['last_log_step'] = now_step
            log.warning(
                "[ENVELOPE][%s][step=%d] stage=%s kind=%s cmd=(%d,%d,%d) alt=%.1f vc=%.1f v_up=%.1f pitch=%.1f roll=%.1f act_in=%s act_out=%s",
                agent_id,
                now_step,
                stage,
                kind,
                int(alt_cmd),
                int(hdg_cmd),
                int(spd_cmd),
                float(profile.get('alt_m', 0.0)),
                float(profile.get('vc_mps', 0.0)),
                float(profile.get('v_up_mps', 0.0)),
                float(profile.get('pitch_deg', 0.0)),
                float(profile.get('roll_deg', 0.0)),
                _fmt_full_array(norm_act),
                _fmt_full_array(safe_act),
            )
        return safe_act
    except Exception:
        return norm_act


def _get_enemy_rnn_reset_state(self, agent_id: str) -> dict:
    if not hasattr(self, '_enemy_rnn_reset_state'):
        self._enemy_rnn_reset_state = {}
    return self._enemy_rnn_reset_state.setdefault(
        agent_id,
        {
            'last_reset_step': -9999,
            'last_reset_reason': '',
            'was_in_hard_recovery': False,
            'model_rejoin_until_step': -1,
            'last_kind': 'NEUTRAL_LOW_ENERGY',
            'kind_transition_reset_step': -9999,
            'abnormal_output_hits': 0,
            'last_abnormal_step': -9999,
        },
    )


def _maybe_reset_enemy_lowlevel_rnn_state(self, agent_id: str, now_step: int, reason: str, cooldown_seconds: float = 2.5) -> bool:
    state = _get_enemy_rnn_reset_state(self, agent_id)
    try:
        dt = float(getattr(self, 'time_interval', 0.2) or 0.2)
    except Exception:
        dt = 0.2
    cooldown_steps = max(1, int(round(cooldown_seconds / max(dt, 1e-3))))
    if now_step - int(state.get('last_reset_step', -9999)) < cooldown_steps:
        return False
    try:
        self._reset_lowlevel_rnn_state(agent_id)
        state['last_reset_step'] = now_step
        state['last_reset_reason'] = reason
        state['abnormal_output_hits'] = 0
        return True
    except Exception:
        return False


def _update_enemy_energy_profile(self, env, agent_id: str):
    """Track smoothed energy / attitude state and classify recovery type."""
    if not agent_id.startswith('B'):
        return None

    if not hasattr(self, '_enemy_energy_state'):
        self._enemy_energy_state = {}

    state = self._enemy_energy_state.setdefault(
        agent_id,
        {
            'initialized': False,
            'stage': 'TACTICAL_NORMAL',
            'kind': 'NEUTRAL_LOW_ENERGY',
            'prev_kind': 'NEUTRAL_LOW_ENERGY',
            'kind_steps': 0,
            'climb_drain_steps': 0,
            'climb_recover_good_steps': 0,
            'l1_hits': 0,
            'l2_hits': 0,
            'lock_until_step': -1,
            'release_until_step': -1,
            'good_recovery_steps': 0,
            'last_log_step': -9999,
            'last_step': -1,
            'last_vc': None,
            'last_vc_step': -1,
            'vc_hist': [],
            'vc_ema': 0.0,
            'v_up_ema': 0.0,
            'pitch_ema': 0.0,
            'aoa_ema': 0.0,
            'roll_ema': 0.0,
            'nz_ema': 0.0,
            'dv_dt': 0.0,
            'dv_dt_1s': 0.0,
            'es_dot': 0.0,
            'reason': '',
        },
    )

    try:
        ac = env.agents[agent_id]
        now_step = int(self.step_count)
        dt = float(getattr(env, 'time_interval', 0.2) or 0.2)

        alt_m = float(ac.get_property_value(c.position_h_sl_m))
        vc_mps = float(ac.get_property_value(c.velocities_vc_mps))
        v_up_mps = -float(ac.get_property_value(c.velocities_v_down_mps))
        pitch_deg = float(np.degrees(ac.get_property_value(c.attitude_theta_rad)))
        roll_deg = float(np.degrees(ac.get_property_value(c.attitude_phi_rad)))
        try:
            aoa_deg = float(ac.get_property_value(c.aero_alpha_deg))
        except Exception:
            aoa_deg = 0.0
        try:
            nz_ema_src = float(ac.get_property_value(c.accelerations_n_pilot_z_norm)) if hasattr(c, 'accelerations_n_pilot_z_norm') else 0.0
        except Exception:
            nz_ema_src = 0.0

        alpha_fast = 0.32
        alpha_slow = 0.22
        if not state['initialized']:
            state['vc_ema'] = vc_mps
            state['v_up_ema'] = v_up_mps
            state['pitch_ema'] = pitch_deg
            state['aoa_ema'] = aoa_deg
            state['roll_ema'] = roll_deg
            state['nz_ema'] = nz_ema_src
            state['initialized'] = True
        else:
            state['vc_ema'] = (1.0 - alpha_fast) * float(state['vc_ema']) + alpha_fast * vc_mps
            state['v_up_ema'] = (1.0 - alpha_fast) * float(state['v_up_ema']) + alpha_fast * v_up_mps
            state['pitch_ema'] = (1.0 - alpha_slow) * float(state['pitch_ema']) + alpha_slow * pitch_deg
            state['aoa_ema'] = (1.0 - alpha_slow) * float(state['aoa_ema']) + alpha_slow * aoa_deg
            state['roll_ema'] = (1.0 - alpha_slow) * float(state['roll_ema']) + alpha_slow * roll_deg
            state['nz_ema'] = (1.0 - alpha_slow) * float(state['nz_ema']) + alpha_slow * nz_ema_src

        prev_vc = state.get('last_vc')
        prev_step = int(state.get('last_vc_step', -1))
        if prev_vc is None or prev_step < 0:
            state['dv_dt'] = 0.0
        else:
            step_gap = max(1, now_step - prev_step)
            state['dv_dt'] = (vc_mps - float(prev_vc)) / (step_gap * dt)
        state['last_vc'] = vc_mps
        state['last_vc_step'] = now_step

        vc_hist = list(state.get('vc_hist', []))
        vc_hist.append((now_step, vc_mps))
        max_hist_steps = int(max(10, round(2.0 / max(dt, 1e-3))))
        while len(vc_hist) > max_hist_steps:
            vc_hist.pop(0)
        state['vc_hist'] = vc_hist

        target_gap_steps = int(max(1, round(1.0 / max(dt, 1e-3))))
        vc_old = vc_hist[0][1]
        vc_old_step = vc_hist[0][0]
        for hist_step, hist_vc in reversed(vc_hist):
            if now_step - int(hist_step) >= target_gap_steps:
                vc_old = float(hist_vc)
                vc_old_step = int(hist_step)
                break
        gap_steps = max(1, now_step - vc_old_step)
        state['dv_dt_1s'] = (vc_mps - vc_old) / (gap_steps * dt)

        # Approximate specific-energy rate; enough for classification and hysteresis.
        state['es_dot'] = 9.81 * float(state['v_up_ema']) + float(state['vc_ema']) * float(state['dv_dt_1s'])

        vc_ema = float(state['vc_ema'])
        v_up_ema = float(state['v_up_ema'])
        pitch_ema = float(state['pitch_ema'])
        aoa_ema = float(state['aoa_ema'])
        roll_ema = float(state['roll_ema'])
        nz_ema = float(state['nz_ema'])
        dv_dt = float(state['dv_dt'])
        dv_dt_1s = float(state['dv_dt_1s'])
        es_dot = float(state['es_dot'])

        # Type classification drives the recovery direction.
        climb_drain_raw = (
            v_up_ema > 3.0
            and 0.5 <= pitch_ema <= 8.0
            and abs(roll_ema) < 15.0
            and dv_dt_1s < -1.0
        )
        if climb_drain_raw:
            state['climb_drain_steps'] = int(state.get('climb_drain_steps', 0)) + 1
        else:
            state['climb_drain_steps'] = max(0, int(state.get('climb_drain_steps', 0)) - 1)

        if v_up_ema < 1.5 and dv_dt_1s >= -0.2:
            state['climb_recover_good_steps'] = int(state.get('climb_recover_good_steps', 0)) + 1
        else:
            state['climb_recover_good_steps'] = 0

        climb_lock = int(state.get('climb_drain_steps', 0)) >= 8
        climb_hold = str(state.get('prev_kind', 'NEUTRAL_LOW_ENERGY')) == 'CLIMB_DRAIN' and int(state.get('climb_recover_good_steps', 0)) < 10

        if (pitch_ema > 10.0 and vc_ema < 205.0 and v_up_ema > -5.0) or (pitch_deg > 14.0 and vc_mps < 190.0 and v_up_mps > -8.0):
            kind = 'NOSE_HIGH'
        elif (abs(roll_ema) > 38.0 and vc_ema < 205.0 and (dv_dt < -2.0 or es_dot < -120.0)) or (abs(roll_deg) > 45.0 and vc_mps < 200.0):
            kind = 'TURN_DRAIN'
        elif (v_up_ema < -8.0 and pitch_ema < 7.0) or (v_up_mps < -12.0 and pitch_deg < 8.0):
            kind = 'SINK'
        elif climb_lock or climb_hold:
            kind = 'CLIMB_DRAIN'
        else:
            kind = 'NEUTRAL_LOW_ENERGY'

        prev_kind = str(state.get('prev_kind', 'NEUTRAL_LOW_ENERGY'))
        if kind == prev_kind:
            state['kind_steps'] = int(state.get('kind_steps', 0)) + 1
        else:
            state['kind_steps'] = 1
        state['prev_kind'] = kind

        rnn_state = _get_enemy_rnn_reset_state(self, agent_id)
        last_kind = str(rnn_state.get('last_kind', prev_kind))
        dangerous_kind_jump = {
            ('NOSE_HIGH', 'SINK'),
            ('SINK', 'NOSE_HIGH'),
            ('CLIMB_DRAIN', 'SINK'),
            ('SINK', 'CLIMB_DRAIN'),
        }
        kind_jump = (last_kind, kind)
        if kind != last_kind:
            current_stage_name = str(state.get('stage', 'TACTICAL_NORMAL'))
            if (
                kind_jump in dangerous_kind_jump
                and current_stage_name in ('ENERGY_L2', 'ENERGY_L3', 'SAFETY_HARD_RECOVERY')
                and now_step - int(rnn_state.get('kind_transition_reset_step', -9999)) >= max(1, int(round(3.0 / max(dt, 1e-3))))
            ):
                if _maybe_reset_enemy_lowlevel_rnn_state(self, agent_id, now_step, f'kind_jump:{last_kind}->{kind}', cooldown_seconds=3.0):
                    rnn_state['kind_transition_reset_step'] = now_step
                    rnn_state['model_rejoin_until_step'] = max(int(rnn_state.get('model_rejoin_until_step', -1)), now_step + 10)
            rnn_state['last_kind'] = kind

        severe_risk = (
            vc_ema < 160.0
            or (vc_ema < 175.0 and dv_dt_1s < -3.0)
            or (pitch_ema < -20.0 and v_up_ema < -20.0)
            or (pitch_ema > 16.0 and vc_ema < 180.0)
            or (abs(roll_ema) > 55.0 and vc_ema < 185.0)
            or (alt_m < 2200.0 and v_up_ema < -18.0)
        )
        hard_risk = (
            pitch_ema < -30.0
            or v_up_ema < -40.0
            or (pitch_ema < -12.0 and vc_ema < 185.0 and dv_dt_1s < -4.0)
            or (pitch_ema > 18.0 and vc_ema < 170.0)
            or (alt_m < 1600.0 and v_up_ema < -25.0)
        )

        l1_raw = (
            (vc_ema < 205.0 and dv_dt_1s < -1.0)
            or (abs(roll_ema) > 25.0 and vc_ema < 215.0 and dv_dt_1s < -1.0)
            or (v_up_ema < -4.0 and pitch_ema < 1.0 and dv_dt_1s < -0.5)
            or (kind == 'TURN_DRAIN' and vc_ema < 215.0 and dv_dt_1s < -1.0)
            or (kind == 'CLIMB_DRAIN' and vc_ema < 210.0 and dv_dt_1s < -1.0)
        )
        l2_raw = (
            (vc_ema < 185.0 and dv_dt_1s < -1.5)
            or (kind in ('SINK', 'CLIMB_DRAIN') and vc_ema < 195.0 and int(state.get('kind_steps', 0)) >= 6)
        )

        state['l1_hits'] = int(state.get('l1_hits', 0)) + 1 if l1_raw else 0
        state['l2_hits'] = int(state.get('l2_hits', 0)) + 1 if l2_raw else 0

        # Recovery stage with hysteresis and warm release.
        stage_order = ['TACTICAL_NORMAL', 'ENERGY_L1', 'ENERGY_L2', 'ENERGY_L3', 'SAFETY_HARD_RECOVERY']
        requested_stage = 'TACTICAL_NORMAL'
        if hard_risk:
            requested_stage = 'SAFETY_HARD_RECOVERY'
        elif severe_risk:
            requested_stage = 'ENERGY_L3'
        elif int(state.get('l2_hits', 0)) >= 6:
            requested_stage = 'ENERGY_L2'
        elif int(state.get('l1_hits', 0)) >= 8:
            requested_stage = 'ENERGY_L1'

        current_stage = str(state.get('stage', 'TACTICAL_NORMAL'))
        current_idx = stage_order.index(current_stage) if current_stage in stage_order else 0
        requested_idx = stage_order.index(requested_stage)

        if requested_idx > current_idx:
            current_stage = requested_stage
            state['stage'] = current_stage
            mild_stage_entry = (
                vc_ema > 220.0
                and abs(pitch_ema) < 8.0
                and abs(roll_ema) < 25.0
                and abs(v_up_ema) < 10.0
                and aoa_ema < 14.0
                and str(kind) == 'NEUTRAL_LOW_ENERGY'
            )
            default_lock = 120 if current_stage == 'SAFETY_HARD_RECOVERY' else 60 if current_stage == 'ENERGY_L3' else 40 if current_stage == 'ENERGY_L2' else 24
            default_release = 70 if current_stage == 'SAFETY_HARD_RECOVERY' else 45 if current_stage == 'ENERGY_L3' else 30
            if mild_stage_entry and current_stage == 'SAFETY_HARD_RECOVERY':
                default_lock = min(default_lock, 20)
                default_release = min(default_release, 12)
            elif mild_stage_entry and current_stage == 'ENERGY_L3':
                default_lock = min(default_lock, 12)
                default_release = min(default_release, 8)
            state['lock_until_step'] = max(int(state.get('lock_until_step', -1)), now_step + default_lock)
            state['release_until_step'] = max(int(state.get('release_until_step', -1)), now_step + default_release)
            state['good_recovery_steps'] = 0
            state['reason'] = requested_stage.lower()
        elif current_stage != 'TACTICAL_NORMAL':
            good_recovery = (
                vc_ema > 205.0
                and abs(pitch_ema) < 6.0
                and abs(roll_ema) < 22.0
                and abs(v_up_ema) < 8.0
                and aoa_ema < 12.0
            )
            if good_recovery:
                state['good_recovery_steps'] = int(state.get('good_recovery_steps', 0)) + 1
            else:
                state['good_recovery_steps'] = 0

            lock_until_step = int(state.get('lock_until_step', -1))
            release_until_step = int(state.get('release_until_step', -1))
            fast_release = (
                current_stage in ('ENERGY_L3', 'SAFETY_HARD_RECOVERY')
                and vc_ema > 225.0
                and abs(pitch_ema) < 8.0
                and abs(roll_ema) < 25.0
                and abs(v_up_ema) < 8.0
                and aoa_ema < 14.0
                and not hard_risk
                and not severe_risk
            )
            if fast_release and int(state.get('good_recovery_steps', 0)) >= 6:
                state['stage'] = 'ENERGY_L1'
                state['lock_until_step'] = now_step + 6
                state['release_until_step'] = now_step + 4
                state['reason'] = ''
            if now_step >= lock_until_step and now_step >= release_until_step and int(state.get('good_recovery_steps', 0)) >= 15:
                state['stage'] = 'TACTICAL_NORMAL'
                state['kind'] = kind
                state['reason'] = ''
            elif requested_idx < current_idx and good_recovery and current_idx > 0:
                # Warm release: demote one stage at a time.
                state['stage'] = stage_order[current_idx - 1]

        state['kind'] = kind
        state['alt_m'] = alt_m
        state['vc_mps'] = vc_mps
        state['v_up_mps'] = v_up_mps
        state['pitch_deg'] = pitch_deg
        state['roll_deg'] = roll_deg
        state['aoa_deg'] = aoa_deg
        state['nz_ema_src'] = nz_ema_src

        return state
    except Exception:
        return state


def _synthesize_enemy_recovery_action(self, profile: dict, base_act=None, hard: bool = False) -> np.ndarray:
    """Generate final enemy control action from the recovery profile."""
    if profile is None:
        return np.array(base_act if base_act is not None else [0.0, 0.0, 0.0, 0.7], dtype=np.float32)

    act = np.array(base_act if base_act is not None else [0.0, 0.0, 0.0, 0.7], dtype=np.float32, copy=True)

    stage = str(profile.get('stage', 'TACTICAL_NORMAL'))
    kind = str(profile.get('kind', 'NEUTRAL_LOW_ENERGY'))
    vc_ema = float(profile.get('vc_ema', 0.0))
    v_up_ema = float(profile.get('v_up_ema', 0.0))
    pitch_ema = float(profile.get('pitch_ema', 0.0))
    roll_ema = float(profile.get('roll_ema', 0.0))
    aoa_ema = float(profile.get('aoa_ema', 0.0))
    dv_dt = float(profile.get('dv_dt', 0.0))
    dv_dt_1s = float(profile.get('dv_dt_1s', dv_dt))
    alt_m = float(profile.get('alt_m', 0.0))
    base_rudder = float(act[2])

    if stage == 'TACTICAL_NORMAL':
        return np.clip(act, [-1, -1, -1, 0], [1, 1, 1, 1]).astype(np.float32)

    # The elevator sign is project-specific: negative = pull-up, positive = unload / nose-down.
    # Hard/soft recovery uses the profile type, not a single fixed sign.
    if stage == 'ENERGY_L1':
        throttle_target = 0.94
        roll_limit = 0.22
        rudder_limit = 0.22
        pitch_band = 0.05
    elif stage == 'ENERGY_L2':
        throttle_target = 0.98
        roll_limit = 0.16
        rudder_limit = 0.16
        pitch_band = 0.10
    elif stage == 'ENERGY_L3':
        throttle_target = 1.00
        roll_limit = 0.10
        rudder_limit = 0.08
        pitch_band = 0.16
    else:
        throttle_target = 1.00
        roll_limit = 0.06
        rudder_limit = 0.06
        pitch_band = 0.24

    if alt_m > 11000.0 and dv_dt_1s < -1.0 and stage in ('ENERGY_L1', 'ENERGY_L2'):
        throttle_target = max(throttle_target, 0.98)

    act[3] = max(float(act[3]), throttle_target)
    act[0] = float(np.clip(-roll_ema / (120.0 if stage == 'ENERGY_L1' else 90.0 if stage == 'ENERGY_L2' else 70.0), -roll_limit, roll_limit))
    act[2] = 0.0

    if kind == 'NOSE_HIGH':
        # Unload first: positive elevator means nose-down in this project.
        if hard or stage == 'ENERGY_L3':
            ele_target = 0.35 if vc_ema < 170.0 else 0.25
        elif stage == 'ENERGY_L2':
            ele_target = 0.18
        else:
            ele_target = 0.08
        if pitch_ema > 18.0 and vc_ema < 180.0:
            ele_target = max(ele_target, 0.45)
        if v_up_ema > 0.0:
            ele_target = max(ele_target, 0.22)
        act[1] = max(float(act[1]), ele_target)
    elif kind == 'SINK':
        # Stop the sink, but keep the pull-up bounded.
        if hard or stage == 'ENERGY_L3':
            ele_target = -0.72 if pitch_ema < 4.0 else -0.58
        elif stage == 'ENERGY_L2':
            ele_target = -0.42
        else:
            ele_target = -0.22
        if vc_ema < 155.0:
            ele_target = min(ele_target, -0.78)
        if dv_dt < -5.0 and v_up_ema < -15.0:
            ele_target = min(ele_target, -0.84)
        act[1] = min(float(act[1]), ele_target)
    elif kind == 'TURN_DRAIN':
        # Reduce bank/load first. Elevator remains near neutral unless pitch is clearly low.
        if pitch_ema > 6.0:
            act[1] = max(float(act[1]), 0.10)
        elif pitch_ema < -4.0:
            act[1] = min(float(act[1]), -0.14)
        else:
            act[1] = float(np.clip(act[1], -pitch_band, pitch_band))
        act[0] = float(np.clip(-roll_ema / 100.0, -roll_limit, roll_limit))
        act[2] = float(np.clip(base_rudder, -0.05, 0.05))
    elif kind == 'CLIMB_DRAIN':
        # Convert sustained climb drain into level-and-accelerate intent.
        if hard or stage == 'ENERGY_L3':
            ele_target = 0.20
            act[3] = max(float(act[3]), 1.00)
        elif stage == 'ENERGY_L2':
            ele_target = 0.14
            act[3] = max(float(act[3]), 0.98)
        else:
            ele_target = 0.08
            act[3] = max(float(act[3]), 0.96)
        act[1] = max(float(act[1]), ele_target)
        act[0] = float(np.clip(act[0], -0.08, 0.08))
        act[2] = 0.0
    else:
        # NEUTRAL_LOW_ENERGY: reduce load and add thrust, avoid fixed shallow-climb template.
        if v_up_ema > 2.0 and dv_dt_1s < 0.0:
            if hard or stage == 'ENERGY_L3':
                act[1] = max(float(act[1]), 0.20)
                act[3] = max(float(act[3]), 1.00)
            elif stage == 'ENERGY_L2':
                act[1] = max(float(act[1]), 0.14)
                act[3] = max(float(act[3]), 0.98)
            else:
                act[1] = max(float(act[1]), 0.08)
                act[3] = max(float(act[3]), 0.96)
        else:
            act[1] = float(np.clip(act[1], -0.02, 0.08))
            act[3] = max(float(act[3]), throttle_target)
        act[0] = float(np.clip(act[0], -0.12, 0.12))
        act[2] = 0.0

    # Do not let the recovery law create excessive pitch changes at low speed.
    if vc_ema < 140.0:
        if kind == 'SINK':
            act[1] = float(np.clip(act[1], -0.72, 0.25))
        elif kind in ('NOSE_HIGH', 'CLIMB_DRAIN'):
            act[1] = float(np.clip(act[1], -0.25, 0.25))
        else:
            act[1] = float(np.clip(act[1], -0.45, 0.30))
        act[0] = float(np.clip(act[0], -0.18, 0.18))
    elif vc_ema < 180.0:
        if kind == 'SINK':
            act[1] = float(np.clip(act[1], -0.55, 0.35))
        else:
            act[1] = float(np.clip(act[1], -0.35, 0.35))

    if hard:
        act[3] = 1.0
        act[0] = float(np.clip(act[0], -0.12, 0.12))
        act[2] = 0.0
        if kind in ('NOSE_HIGH', 'CLIMB_DRAIN'):
            act[1] = float(np.clip(act[1], -0.25, 0.25))
        elif kind == 'SINK':
            act[1] = float(np.clip(act[1], -0.72, 0.35))

    return np.clip(act, [-1, -1, -1, 0], [1, 1, 1, 1]).astype(np.float32)


def _get_enemy_hard_recovery_action(self, env, agent_id: str):
    """Hard takeover controller for enemy aircraft in dangerous attitudes.

    When triggered, this controller bypasses low-level model output and returns
    directly synthesized control actions to break sustained dive chains.
    """
    profile = _update_enemy_energy_profile(self, env, agent_id)
    if profile is None:
        return None, None, None

    stage = str(profile.get('stage', 'TACTICAL_NORMAL'))
    if stage not in ('ENERGY_L3', 'SAFETY_HARD_RECOVERY'):
        return None, None, profile

    vc_now = float(profile.get('vc_mps', 0.0))
    v_up_now = float(profile.get('v_up_mps', 0.0))
    pitch_now = float(profile.get('pitch_deg', 0.0))
    roll_now = abs(float(profile.get('roll_deg', 0.0)))
    kind_now = str(profile.get('kind', 'NEUTRAL_LOW_ENERGY'))
    mild_energy_state = (
        kind_now == 'NEUTRAL_LOW_ENERGY'
        and vc_now >= 230.0
        and abs(v_up_now) <= 20.0
        and -4.0 <= pitch_now <= 8.0
        and roll_now <= 15.0
    )
    if mild_energy_state:
        profile['stage'] = 'ENERGY_L1'
        profile['lock_until_step'] = min(int(profile.get('lock_until_step', -1)), int(self.step_count) + 4)
        profile['release_until_step'] = min(int(profile.get('release_until_step', -1)), int(self.step_count) + 2)
        profile['reason'] = ''
        return None, None, profile

    try:
        act = _synthesize_enemy_recovery_action(self, profile, base_act=None, hard=True)
        now_step = int(self.step_count)
        if now_step - int(profile.get('last_log_step', -9999)) >= 160:
            profile['last_log_step'] = now_step
            log.warning(
                "[HARD_RECOVERY][%s][step=%d] stage=%s kind=%s lock_until=%d alt=%.1f vc=%.1f v_up=%.1f pitch=%.1f roll=%.1f act=%s",
                agent_id,
                now_step,
                stage,
                str(profile.get('kind', 'NEUTRAL_LOW_ENERGY')),
                int(profile.get('lock_until_step', -1)),
                float(profile.get('alt_m', 0.0)),
                float(profile.get('vc_mps', 0.0)),
                float(profile.get('v_up_mps', 0.0)),
                float(profile.get('pitch_deg', 0.0)),
                float(profile.get('roll_deg', 0.0)),
                _fmt_full_array(act),
            )

        return act, stage.lower(), profile
    except Exception:
        return None, None, profile


def _fmt_full_array(values) -> str:
    return np.array2string(
        np.asarray(values),
        precision=6,
        separator=",",
        threshold=np.inf,
        max_line_width=100000,
    )


def _maybe_apply_enemy_safe_teacher(self, env, agent_id: str, alt_cmd: int, hdg_cmd: int, spd_cmd: int):
    teacher = getattr(self, "enemy_safe_teacher", None)
    if not agent_id.startswith('B') or teacher is None or not getattr(self, "enemy_safe_teacher_enabled", False):
        return alt_cmd, hdg_cmd, spd_cmd, None
    if agent_id in getattr(self, "_enemy_safe_teacher_skip_agents", set()):
        return alt_cmd, hdg_cmd, spd_cmd, None

    snap = teacher.snapshot(env, agent_id)
    if snap is None:
        return alt_cmd, hdg_cmd, spd_cmd, None

    model_cmd = (int(alt_cmd), int(hdg_cmd), int(spd_cmd))
    decision = teacher.build_teacher_command(snap, model_cmd)
    stats = getattr(self, "_enemy_safe_teacher_stats", None)
    if stats is not None:
        agent_stats = stats.setdefault(
            agent_id,
            {"total": 0, "safe_hold": 0, "energy_recover": 0, "last_reason": ""},
        )
        agent_stats["total"] += 1
        agent_stats["last_reason"] = str(decision.reason)
        if decision.mode.value == "safe_hold":
            agent_stats["safe_hold"] += 1
        elif decision.mode.value == "energy_recover":
            agent_stats["energy_recover"] += 1

    if tuple(decision.command) != model_cmd:
        log.warning(
            "[ENEMY_SAFE_TEACHER][%s][step=%d] mode=%s reason=%s cmd_raw=%s cmd_safe=%s alt=%.1f vc=%.1f v_up=%.1f pitch=%.1f roll=%.1f",
            agent_id,
            int(getattr(self, "step_count", 0)),
            decision.mode.value,
            str(decision.reason),
            model_cmd,
            tuple(int(x) for x in decision.command),
            float(snap.altitude_m),
            float(snap.vc_mps),
            float(snap.v_up_mps),
            float(snap.pitch_deg),
            float(snap.roll_deg),
        )

    if decision.action_override_raw is not None:
        override_map = getattr(self, "_enemy_lowlevel_action_overrides", None)
        if isinstance(override_map, dict):
            override_map[agent_id] = np.asarray(decision.action_override_raw, dtype=np.int64)

    safe_alt, safe_hdg, safe_spd = (int(x) for x in decision.command)
    return safe_alt, safe_hdg, safe_spd, decision


def _remap_for_legacy_cmd(delta_alt_km: float, delta_hdg_rad: float, delta_vel_norm: float):
    if delta_alt_km > 0.05:
        alt_legacy = 0.1
    elif delta_alt_km < -0.05:
        alt_legacy = -0.1
    else:
        alt_legacy = 0.0

    hdg_deg = float(np.degrees(delta_hdg_rad))
    if hdg_deg <= -22.5:
        hdg_legacy = -np.pi / 6
    elif hdg_deg <= -7.5:
        hdg_legacy = -np.pi / 12
    elif hdg_deg < 7.5:
        hdg_legacy = 0.0
    elif hdg_deg < 22.5:
        hdg_legacy = np.pi / 12
    else:
        hdg_legacy = np.pi / 6

    if delta_vel_norm > 0.025:
        vel_legacy = 0.05
    elif delta_vel_norm < -0.025:
        vel_legacy = -0.05
    else:
        vel_legacy = 0.0

    return float(alt_legacy), float(hdg_legacy), float(vel_legacy)


def _build_lowlevel_input_for_agent(self, env, agent_id: str, alt: int, hdg: int, vel: int, model_key: str):
    raw = self.get_obs(env, agent_id)
    inp = np.zeros(12, dtype=np.float32)
    inp[0] = self.norm_alt[min(int(alt), len(self.norm_alt) - 1)]
    inp[1] = self.norm_hdg[min(int(hdg), len(self.norm_hdg) - 1)]
    inp[2] = self.norm_vel[min(int(vel), len(self.norm_vel) - 1)]

    if model_key == 'su27_legacy':
        remap_alt, remap_hdg, remap_vel = _remap_for_legacy_cmd(inp[0], inp[1], inp[2])
        inp[0], inp[1], inp[2] = remap_alt, remap_hdg, remap_vel

        ac = env.agents[agent_id]
        inp[3] = ac.get_property_value(c.position_h_sl_m) / 5000.0
        inp[4] = np.sin(ac.get_property_value(c.attitude_roll_rad))
        inp[5] = np.cos(ac.get_property_value(c.attitude_roll_rad))
        inp[6] = np.sin(ac.get_property_value(c.attitude_pitch_rad))
        inp[7] = np.cos(ac.get_property_value(c.attitude_pitch_rad))
        inp[8] = ac.get_property_value(c.velocities_u_mps) / 340.0
        inp[9] = ac.get_property_value(c.velocities_v_mps) / 340.0
        inp[10] = ac.get_property_value(c.velocities_w_mps) / 340.0
        inp[11] = ac.get_property_value(c.velocities_vc_mps) / 340.0
    else:
        if model_key == 'f16_legacy':
            remap_alt, remap_hdg, remap_vel = _remap_for_legacy_cmd(inp[0], inp[1], inp[2])
            inp[0], inp[1], inp[2] = remap_alt, remap_hdg, remap_vel
        if len(raw) >= 9:
            inp[3:12] = raw[:9]

    return raw, inp


def _raw_bins_to_norm_act(raw_bins):
    out = np.asarray(raw_bins, dtype=np.float32).reshape(-1)
    return np.array([out[0] / 20 - 1, out[1] / 20 - 1, out[2] / 20 - 1, out[3] / 58 + 0.4], dtype=np.float32)


def _norm_act_to_raw_bins(norm_act):
    act = np.asarray(norm_act, dtype=np.float32).reshape(-1)
    return np.array(
        [
            int(np.clip(np.round((float(act[0]) + 1.0) * 20.0), 0, 40)),
            int(np.clip(np.round((float(act[1]) + 1.0) * 20.0), 0, 40)),
            int(np.clip(np.round((float(act[2]) + 1.0) * 20.0), 0, 40)),
            int(np.clip(np.round((float(act[3]) - 0.4) * 58.0), 0, 29)),
        ],
        dtype=np.int64,
    )


def normalize_action(self, env, agent_id: str, action: np.ndarray) -> np.ndarray:
    """将高层动作(alt_cmd, hdg_cmd, spd_cmd, shoot)转换为控制面命令

    这是关键的动作转换方法。multiplecombat_env.step()会调用此方法。
    如果没有此方法，父类MultipleCombatTask.normalize_action会被调用，
    它会把(alt=7, hdg=8, spd=3)当作原始控制索引，导致飞机立即俯冲坠毁。
    """
    # 🔥 DEBUG: 详细追踪敌方飞机动作生成过程（每50步打印一次）
    is_enemy = agent_id.startswith('B')
    if is_enemy and _CAP_CONTROL_DEBUG and self.step_count % 50000 == 0:
        try:
            current_alt_m = float(env.agents[agent_id].get_property_value(c.position_h_sl_m))
            current_alt_ft = float(env.agents[agent_id].get_property_value(c.position_h_sl_ft))
            current_pitch = np.degrees(float(env.agents[agent_id].get_property_value(c.attitude_pitch_rad)))
            log.info(f"[DEBUG normalize_action V1] {agent_id} step={self.step_count} "
                    f"action={action[:3]} "
                    f"alt_m={current_alt_m:.1f} alt_ft={current_alt_ft:.0f} pitch={current_pitch:.1f}°")
        except Exception as e:
            log.warning(f"[DEBUG normalize_action V1] {agent_id} 获取状态失败: {e}")

    # 检查飞机状态
    if agent_id not in env.agents or not env.agents[agent_id].is_alive:
        return np.array([0.0, 0.0, 0.0, 0.7])

    if agent_id.startswith("A"):
        recreated = _maybe_recreate_friendly_simulator(self, env, agent_id)
        if recreated and (agent_id not in env.agents or not env.agents[agent_id].is_alive):
            return np.array([0.0, 0.0, 0.0, 0.7])

    if agent_id.startswith("B"):
        recreated = _maybe_recreate_enemy_simulator(self, env, agent_id)
        if recreated and (agent_id not in env.agents or not env.agents[agent_id].is_alive):
            return np.array([0.0, 0.0, 0.0, 0.7])

    # 获取当前飞机观测
    try:
        raw_obs = self.get_obs(env, agent_id)
    except Exception:
        raw_obs = np.zeros(self.obs_length if hasattr(self, 'obs_length') else 27)

    # 解析动作索引 (确保在有效范围内)
    alt_cmd = int(np.clip(action[0], 0, len(self.norm_alt) - 1))
    hdg_cmd = int(np.clip(action[1], 0, len(self.norm_hdg) - 1))
    spd_cmd = int(np.clip(action[2], 0, len(self.norm_vel) - 1))
    teacher_decision = None
    if False:
        alt_cmd, hdg_cmd, spd_cmd = self._rewrite_friendly_recovery_command(
            env,
            agent_id,
            alt_cmd,
            hdg_cmd,
            spd_cmd,
        )
    alt_cmd, hdg_cmd, spd_cmd, teacher_decision = _maybe_apply_enemy_safe_teacher(
        self,
        env,
        agent_id,
        alt_cmd,
        hdg_cmd,
        spd_cmd,
    )
    root_trace = os.environ.get('CAP_ROOTCAUSE_TRACE', '').strip().lower() in ('1', 'true', 'yes', 'on')

    if _deep_trace_enabled(agent_id):
        try:
            ac = env.agents[agent_id]
            current_alt_m = float(ac.get_property_value(c.position_h_sl_m))
            current_vc_mps = float(ac.get_property_value(c.velocities_vc_mps))
            current_hdg_deg = float(np.degrees(ac.get_property_value(c.attitude_heading_true_rad))) % 360.0
            current_pos = ac.get_position()
            g_load = float(ac.get_property_value(c.accelerations_n_pilot_z_norm)) if hasattr(c, 'accelerations_n_pilot_z_norm') else 0.0
            log.warning(
                "[%s][T+%07.1fs][normalize_action][action_decode][state] raw_action=%s clipped=(%d,%d,%d) raw_obs=%s alt=%.3fm vc=%.3fmps hdg=%.3fdeg pos=%s g=%.3f",
                agent_id,
                float(self.step_count * 0.2),
                _fmt_full_array(action),
                alt_cmd,
                hdg_cmd,
                spd_cmd,
                _fmt_full_array(raw_obs),
                current_alt_m,
                current_vc_mps,
                current_hdg_deg,
                _fmt_full_array(current_pos),
                g_load,
            )
        except Exception as exc:
            log.warning("[%s][T+%07.1fs][normalize_action][action_decode][state] failed=%s", agent_id, float(self.step_count * 0.2), exc)

    if is_enemy and root_trace:
        try:
            current_alt_m = float(env.agents[agent_id].get_property_value(c.position_h_sl_m))
            raw_alt = int(action[0])
            raw_hdg = int(action[1])
            raw_spd = int(action[2])
            clipped = (raw_alt != alt_cmd) or (raw_hdg != hdg_cmd) or (raw_spd != spd_cmd)
            if clipped or current_alt_m < 5000.0:
                log.warning(
                    f"🧩 [根因链-动作裁剪] {agent_id} raw=({raw_alt},{raw_hdg},{raw_spd}) "
                    f"clipped=({alt_cmd},{hdg_cmd},{spd_cmd}) "
                    f"delta=(alt={self.norm_alt[alt_cmd]:.3f}km,hdg={self.norm_hdg[hdg_cmd]:.3f}rad,vel={self.norm_vel[spd_cmd]:.3f}) "
                    f"alt={current_alt_m:.0f}m step={self.step_count}"
                )
            if teacher_decision is not None and tuple(int(x) for x in teacher_decision.command) != (raw_alt, raw_hdg, raw_spd):
                log.warning(
                    f"🧩 [根因链-安全老师] {agent_id} mode={teacher_decision.mode.value} reason={teacher_decision.reason} "
                    f"raw=({raw_alt},{raw_hdg},{raw_spd}) safe=({alt_cmd},{hdg_cmd},{spd_cmd}) "
                    f"alt={current_alt_m:.0f}m step={self.step_count}"
                )
        except Exception:
            pass
    elif root_trace and self._is_friendly_rootcause_return(agent_id):
        try:
            if self.step_count % 10 == 0:
                current_alt_m = float(env.agents[agent_id].get_property_value(c.position_h_sl_m))
                phase_name = self._get_bridge_phase_name(agent_id) or 'UNKNOWN'
                log.warning(
                    f"🧩 [友机根因记录-动作裁剪] {agent_id} phase={phase_name} "
                    f"cmd=({alt_cmd},{hdg_cmd},{spd_cmd}) "
                    f"delta=(alt={self.norm_alt[alt_cmd]:.3f}km,hdg={self.norm_hdg[hdg_cmd]:.3f}rad,vel={self.norm_vel[spd_cmd]:.3f}) "
                    f"alt={current_alt_m:.0f}m step={self.step_count}"
                )
        except Exception:
            pass

    # 🔥 诊断：追踪我方飞机的航向命令执行（每30秒，仅A0100）
    if _CAP_DEBUG_PRINT and agent_id == 'A0100' and self.step_count % 150 == 0:
        try:
            current_hdg_deg = np.degrees(env.agents[agent_id].get_property_value(c.attitude_heading_true_rad)) % 360
            log.info(f"[normalize_action] {agent_id} step={self.step_count}")
            log.info(f"  输入动作: alt_cmd={action[0]:.0f} hdg_cmd={action[1]:.0f} spd_cmd={action[2]:.0f}")
            log.info(f"  解析后: alt_cmd={alt_cmd} hdg_cmd={hdg_cmd} spd_cmd={spd_cmd}")
            log.info(f"  当前航向: {current_hdg_deg:.1f}°")
            log.info(f"  航向变化量: {self.norm_hdg[hdg_cmd]:.3f}rad = {np.degrees(self.norm_hdg[hdg_cmd]):.1f}°")
        except:
            pass

    if is_enemy and _CAP_CONTROL_DEBUG and self.step_count % 50000 == 0:
        log.info(f"[DEBUG normalize_action] {agent_id} 解析动作: alt_cmd={alt_cmd} hdg_cmd={hdg_cmd} spd_cmd={spd_cmd} "
                f"norm_alt[{alt_cmd}]={self.norm_alt[alt_cmd]:.3f} "
                f"norm_hdg[{hdg_cmd}]={self.norm_hdg[hdg_cmd]:.3f} "
                f"norm_vel[{spd_cmd}]={self.norm_vel[spd_cmd]:.3f}")

    # 构建低级策略输入 [delta_alt, delta_hdg, delta_vel, obs...]
    input_obs = np.zeros(12, dtype=np.float32)
    input_obs[0] = self.norm_alt[alt_cmd]
    input_obs[1] = self.norm_hdg[hdg_cmd]
    input_obs[2] = self.norm_vel[spd_cmd]
    input_obs[3:12] = raw_obs[:9] if len(raw_obs) >= 9 else np.zeros(9)

    if _deep_trace_enabled(agent_id):
        log.warning(
            "[%s][T+%07.1fs][normalize_action][model_input_build][input_obs] input_obs=%s alt_idx=%d hdg_idx=%d vel_idx=%d",
            agent_id,
            float(self.step_count * 0.2),
            _fmt_full_array(input_obs),
            alt_cmd,
            hdg_cmd,
            spd_cmd,
        )

    # NaN检查
    input_obs = np.nan_to_num(input_obs, nan=0.0, posinf=1.0, neginf=-1.0)
    input_obs = np.expand_dims(input_obs, axis=0)

    # 初始化RNN状态
    if agent_id not in self._inner_rnn_states:
        self._inner_rnn_states[agent_id] = np.zeros((1, 1, 128), dtype=np.float32)

    rnn_state = _get_enemy_rnn_reset_state(self, agent_id) if agent_id.startswith('B') else None
    prev_used_hard_recovery = bool(rnn_state.get('was_in_hard_recovery', False)) if rnn_state is not None else False


    enemy_rule_lowlevel_active = bool(
        agent_id.startswith('B') and getattr(self, 'enemy_rule_lowlevel_enabled', False)
    )

    # 1. 基础控制生成 (F16/SU27模型) 或危险姿态硬接管
    # ------------------------------------------------------------------
    # 敌方在危险姿态时直接旁路模型，防止RNN持续输出深压头链路。
    if enemy_rule_lowlevel_active:
        hard_act, hard_reason, hard_meta = None, None, None
    else:
        hard_act, hard_reason, hard_meta = _get_enemy_hard_recovery_action(self, env, agent_id)
    used_hard_recovery = hard_act is not None

    if rnn_state is not None:
        now_step = int(self.step_count)
        if used_hard_recovery and not prev_used_hard_recovery:
            _maybe_reset_enemy_lowlevel_rnn_state(self, agent_id, now_step, 'hard_recovery_entry', cooldown_seconds=0.0)
            rnn_state['was_in_hard_recovery'] = True
        elif prev_used_hard_recovery and not used_hard_recovery:
            _maybe_reset_enemy_lowlevel_rnn_state(self, agent_id, now_step, 'hard_recovery_exit', cooldown_seconds=0.0)
            rnn_state['was_in_hard_recovery'] = False
            rnn_state['model_rejoin_until_step'] = max(int(rnn_state.get('model_rejoin_until_step', -1)), now_step + 10)
        elif used_hard_recovery:
            rnn_state['was_in_hard_recovery'] = True

    if used_hard_recovery:
        norm_act = np.array(hard_act, dtype=np.float32, copy=True)
    else:
        # 敌我双方都统一使用 _lowlevel_control 生成基础控制量
        # 内部会自动选择模型：敌方(B)->F16, 我方(A)->SU27
        base = self._lowlevel_control(env, agent_id, alt_cmd, hdg_cmd, spd_cmd)
        norm_act = base.copy() if isinstance(base, np.ndarray) else np.array(base, dtype=np.float32)

    enemy_lowlevel_override_active = False
    if is_enemy:
        adapter = getattr(self, "enemy_adapter", None)
        if adapter is not None and hasattr(adapter, "consume_lowlevel_override_active"):
            try:
                enemy_lowlevel_override_active = bool(adapter.consume_lowlevel_override_active(agent_id))
            except Exception:
                enemy_lowlevel_override_active = False

    if _deep_trace_enabled(agent_id):
        try:
            ac = env.agents[agent_id]
            current_alt_m = float(ac.get_property_value(c.position_h_sl_m))
            current_vc_mps = float(ac.get_property_value(c.velocities_vc_mps))
            current_hdg_deg = float(np.degrees(ac.get_property_value(c.attitude_heading_true_rad))) % 360.0
            current_pos = ac.get_position()
            g_load = float(ac.get_property_value(c.accelerations_n_pilot_z_norm)) if hasattr(c, 'accelerations_n_pilot_z_norm') else 0.0
            log.warning(
                "[%s][T+%07.1fs][normalize_action][lowlevel_return][norm_act] norm_act=%s alt=%.3fm vc=%.3fmps hdg=%.3fdeg pos=%s g=%.3f",
                agent_id,
                float(self.step_count * 0.2),
                _fmt_full_array(norm_act),
                current_alt_m,
                current_vc_mps,
                current_hdg_deg,
                _fmt_full_array(current_pos),
                g_load,
            )
        except Exception as exc:
            log.warning("[%s][T+%07.1fs][normalize_action][lowlevel_return][norm_act] failed=%s", agent_id, float(self.step_count * 0.2), exc)

    if is_enemy and root_trace:
        try:
            current_alt_m = float(env.agents[agent_id].get_property_value(c.position_h_sl_m))
            if current_alt_m < 5000.0:
                log.warning(
                    f"🧩 [根因链-舵面输出] {agent_id} cmd_idx=({alt_cmd},{hdg_cmd},{spd_cmd}) "
                    f"act=(ail={float(norm_act[0]):+.3f},ele={float(norm_act[1]):+.3f},rud={float(norm_act[2]):+.3f},thr={float(norm_act[3]):.3f}) "
                    f"alt={current_alt_m:.0f}m step={self.step_count}"
                )
        except Exception:
            pass
    elif root_trace and self._is_friendly_rootcause_return(agent_id):
        try:
            if self.step_count % 10 == 0:
                current_alt_m = float(env.agents[agent_id].get_property_value(c.position_h_sl_m))
                phase_name = self._get_bridge_phase_name(agent_id) or 'UNKNOWN'
                log.warning(
                    f"🧩 [友机根因记录-舵面输出] {agent_id} phase={phase_name} "
                    f"cmd_idx=({alt_cmd},{hdg_cmd},{spd_cmd}) "
                    f"act=(ail={float(norm_act[0]):+.3f},ele={float(norm_act[1]):+.3f},rud={float(norm_act[2]):+.3f},thr={float(norm_act[3]):.3f}) "
                    f"alt={current_alt_m:.0f}m step={self.step_count}"
                )
        except Exception:
            pass

    is_verification_scenario = hasattr(self, 'enemy_scenario') and self.enemy_scenario is not None

    # 低层速度响应补偿：当高层明确要求加速时，提高油门下限。
    # 目的：让“速度协调/拦截阶段加速”在JSBSim中可观测，避免模型输出偏保守导致速度不升反降。
    if False:
        try:
            if spd_cmd >= 5:
                floor = 0.95 if agent_id.startswith('A') else 0.98
                norm_act[3] = max(float(norm_act[3]), floor)
            elif spd_cmd == 4:
                floor = 0.90 if agent_id.startswith('A') else 0.94
                norm_act[3] = max(float(norm_act[3]), floor)
        except Exception:
            pass

    if False:
        norm_act = self._apply_friendly_safety_override(env, agent_id, norm_act)

    if rnn_state is not None and not used_hard_recovery:
        now_step = int(self.step_count)
        rejoin_until_step = int(rnn_state.get('model_rejoin_until_step', -1))
        if now_step < rejoin_until_step:
            norm_act = np.array(norm_act, dtype=np.float32, copy=True)
            norm_act[0] = float(np.clip(norm_act[0], -0.10, 0.10))
            norm_act[1] = float(np.clip(norm_act[1], -0.20, 0.12))
            norm_act[2] = 0.0
            norm_act[3] = max(float(norm_act[3]), 0.96)

    # 2. 敌方模型后置包线保护（最终执行前硬约束）
    # ------------------------------------------------------------------
    if not used_hard_recovery and not enemy_rule_lowlevel_active and not enemy_lowlevel_override_active:
        norm_act = _apply_enemy_envelope_protection(
            self,
            env,
            agent_id,
            norm_act,
            alt_cmd,
            hdg_cmd,
            spd_cmd,
        )

    # 3. 最终执行动作日志（避免只看到模型原始输出）
    # ------------------------------------------------------------------
    if _deep_trace_enabled(agent_id):
        try:
            ac = env.agents[agent_id]
            current_alt_m = float(ac.get_property_value(c.position_h_sl_m))
            current_vc_mps = float(ac.get_property_value(c.velocities_vc_mps))
            current_hdg_deg = float(np.degrees(ac.get_property_value(c.attitude_heading_true_rad))) % 360.0
            pitch_deg = float(np.degrees(ac.get_property_value(c.attitude_theta_rad)))
            v_up_mps = -float(ac.get_property_value(c.velocities_v_down_mps))
            if used_hard_recovery:
                src = f"hard_recovery:{hard_reason}"
            elif enemy_lowlevel_override_active:
                src = "enemy_rtb_break_override"
            elif enemy_rule_lowlevel_active:
                src = "enemy_rule_lowlevel"
            elif teacher_decision is not None and teacher_decision.mode.value != "model":
                src = f"safe_teacher:{teacher_decision.mode.value}"
            else:
                src = "model_or_envelope"
            log.warning(
                "[%s][T+%07.1fs][normalize_action][final_execute][act] source=%s cmd_idx=(%d,%d,%d) act=%s alt=%.3fm vc=%.3fmps hdg=%.3fdeg pitch=%.3fdeg v_up=%.3fmps",
                agent_id,
                float(self.step_count * 0.2),
                src,
                alt_cmd,
                hdg_cmd,
                spd_cmd,
                _fmt_full_array(norm_act),
                current_alt_m,
                current_vc_mps,
                current_hdg_deg,
                pitch_deg,
                v_up_mps,
            )
        except Exception:
            pass

    # 4. 敌方安全保护与坠毁诊断 (黑匣子跟踪)
    # ------------------------------------------------------------------
    if agent_id.startswith('B'):
        try:
            current_alt = env.agents[agent_id].get_property_value(c.position_h_sl_m)

            # 收集完整飞行状态数据用于黑匣子
            tas_mps = float(env.agents[agent_id].get_property_value(c.velocities_vc_mps)) if abs(float(env.agents[agent_id].get_property_value(c.velocities_vc_mps))) < 10000 else 0.0
            v_up_mps = -float(env.agents[agent_id].get_property_value(c.velocities_v_down_mps)) if abs(float(env.agents[agent_id].get_property_value(c.velocities_v_down_mps))) < 10000 else 0.0
            pitch_rad = float(env.agents[agent_id].get_property_value(c.attitude_theta_rad))
            roll_rad = float(env.agents[agent_id].get_property_value(c.attitude_phi_rad))
            pitch_deg = np.degrees(pitch_rad)
            roll_deg = np.degrees(roll_rad)
            alpha_deg = float(env.agents[agent_id].get_property_value(c.aero_alpha_deg))

            # === 滚动防坠毁黑匣子记录 (保留最近40秒 / 200步) ===
            if not hasattr(self, '_crash_blackbox'):
                self._crash_blackbox = {}
            if agent_id not in self._crash_blackbox:
                self._crash_blackbox[agent_id] = []

            # 记录格式: [step, alt, roll, pitch, tas, v_up, alpha, ail, ele, rud, thr]
            record = [
                self.step_count, current_alt, roll_deg, pitch_deg, tas_mps, v_up_mps, alpha_deg,
                norm_act[0], norm_act[1], norm_act[2], norm_act[3]
            ]
            self._crash_blackbox[agent_id].append(record)

            if len(self._crash_blackbox[agent_id]) > 200:
                self._crash_blackbox[agent_id].pop(0)

            # 当高度降至严重坠毁界限且尚未在此次触发过写入时：强制保存过去40秒数据用于坠毁复盘！
            if current_alt < 100:
                last_crash_flush = getattr(self, '_last_crash_flush_step', {}).get(agent_id, 0)
                if self.step_count - last_crash_flush > 500:  # 避免频繁写盘
                    self._last_crash_flush_step = getattr(self, '_last_crash_flush_step', {})
                    self._last_crash_flush_step[agent_id] = self.step_count

                    crash_log_dir = os.path.abspath(os.path.join(get_root_dir(), "scripts", "tacticalProject", "logs"))
                    os.makedirs(crash_log_dir, exist_ok=True)
                    import datetime
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    trace_file = os.path.join(crash_log_dir, f"crash_trace_{agent_id}_{timestamp}_step{self.step_count}.csv")

                    try:
                        with open(trace_file, 'w', encoding='utf-8') as f:
                            f.write("step,alt_m,roll_deg,pitch_deg,tas_mps,v_up_mps,alpha_deg,cmd_aileron,cmd_elevator,cmd_rudder,cmd_throttle\n")
                            for r in self._crash_blackbox[agent_id]:
                                f.write(",".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r) + "\n")
                        log.error(f"🚨🚨 [{agent_id}] 发生严重坠毁(高度<{current_alt:.0f}m)！过去40秒的遥测与动作已导出至: {trace_file}")
                    except Exception as trace_ex:
                        log.error(f"Failed to write crash trace: {trace_ex}")

        except Exception as e:
            log.error(f"⚠️ [{agent_id}] 安全记录执行失败: {e}")

    return norm_act
    # 🔥 DEBUG: 追踪最终输出（每50步打印一次）
    if _CAP_CONTROL_DEBUG and agent_id.startswith('B') and self.step_count % 50000 == 0:
        try:
            current_alt_msl = env.agents[agent_id].get_property_value(c.position_h_sl_m)
            log.warning(f"[DEBUG normalize_action V1最终] {agent_id} step={self.step_count} "
                       f"alt_m={current_alt_msl:.1f} "
                       f"控制命令: {norm_act} "
                       f"⚠️ elevator={norm_act[1]:.3f}")
        except:
            pass

    return norm_act

def _lowlevel_control(self, env, agent_id: str, alt: int, hdg: int, vel: int) -> np.ndarray:
    """Bottom-layer control shared by friendly and enemy aircraft."""
    is_enemy = agent_id.startswith('B')
    root_trace = os.environ.get('CAP_ROOTCAUSE_TRACE', '').strip().lower() in ('1', 'true', 'yes', 'on')

    model_key, model = self._select_lowlevel_model_for_agent(agent_id)
    if model is None:
        return np.array([0.0, 0.0, 0.0, 0.7])

    try:
        raw = self.get_obs(env, agent_id)
        if agent_id not in self._inner_rnn_states:
            self._inner_rnn_states[agent_id] = np.zeros((1, 1, 128), dtype=np.float32)

        rnn_before = self._inner_rnn_states[agent_id].copy()
        input_model_key = model_key
        if is_enemy and getattr(self, "enemy_rule_lowlevel_enabled", False):
            input_model_key = 'f16_legacy'
        raw, inp = _build_lowlevel_input_for_agent(self, env, agent_id, alt, hdg, vel, input_model_key)

        override_raw = None
        if is_enemy:
            adapter = getattr(self, "enemy_adapter", None)
            if adapter is not None and hasattr(adapter, "get_lowlevel_override_raw"):
                try:
                    override_raw = adapter.get_lowlevel_override_raw(env, agent_id)
                except Exception:
                    override_raw = None
        override_map = getattr(self, "_enemy_lowlevel_action_overrides", None)
        if override_raw is None and is_enemy and isinstance(override_map, dict):
            override_raw = override_map.pop(agent_id, None)

        if override_raw is not None:
            out = np.asarray(override_raw, dtype=np.float32).reshape(4)
            norm = _raw_bins_to_norm_act(out)
            active_model_key = "f16_online_override"
        elif is_enemy and getattr(self, "enemy_rule_lowlevel_enabled", False):
            energy_mode = None
            adapter = getattr(self, "enemy_adapter", None)
            if adapter is not None and hasattr(adapter, "get_energy_mode"):
                try:
                    energy_mode = adapter.get_energy_mode(agent_id)
                except Exception:
                    energy_mode = None
            norm = np.asarray(
                self.enemy_rule_lowlevel.compute_action(
                    env,
                    agent_id,
                    alt,
                    hdg,
                    vel,
                    self.norm_alt,
                    self.norm_hdg,
                    self.norm_vel,
                    energy_mode=energy_mode,
                ),
                dtype=np.float32,
            )
            out = _norm_act_to_raw_bins(norm)
            active_model_key = "enemy_rule_pid"
        else:
            obs_t = torch.FloatTensor(np.nan_to_num(inp, nan=0.0)).unsqueeze(0)
            rnn_t = torch.FloatTensor(self._inner_rnn_states[agent_id])

            with torch.no_grad():
                act, rnn_out = model(obs_t, rnn_t)

            out = act.detach().cpu().numpy()[0] if model_key == 'su27_legacy' else act.detach().cpu().numpy().squeeze(0)
            self._inner_rnn_states[agent_id] = rnn_out.detach().cpu().numpy()
            norm = _raw_bins_to_norm_act(out)
            active_model_key = model_key

        if agent_id.startswith('B') and override_raw is None and active_model_key not in ('enemy_rule_pid',):
            ac = env.agents[agent_id]
            reset_state = _get_enemy_rnn_reset_state(self, agent_id)
            now_step = int(self.step_count)
            pitch_deg_now = float(np.degrees(ac.get_property_value(c.attitude_theta_rad)))
            v_up_now = -float(ac.get_property_value(c.velocities_v_down_mps))
            vc_now = float(ac.get_property_value(c.velocities_vc_mps))
            suspicious = (
                (pitch_deg_now > 8.0 and float(norm[1]) < -0.18)
                or (v_up_now < -10.0 and float(norm[1]) < -0.12)
                or (vc_now < 220.0 and float(norm[3]) < 0.74 and v_up_now < 2.0)
            )
            if suspicious:
                reset_state['abnormal_output_hits'] = int(reset_state.get('abnormal_output_hits', 0)) + 1
                reset_state['last_abnormal_step'] = now_step
            else:
                reset_state['abnormal_output_hits'] = 0

            if int(reset_state.get('abnormal_output_hits', 0)) >= 8:
                if _maybe_reset_enemy_lowlevel_rnn_state(self, agent_id, now_step, 'abnormal_lowlevel_output', cooldown_seconds=2.5):
                    reset_state['model_rejoin_until_step'] = max(int(reset_state.get('model_rejoin_until_step', -1)), now_step + 10)
                    reset_state['abnormal_output_hits'] = 0

        if _deep_trace_enabled(agent_id):
            try:
                ac = env.agents[agent_id]
                current_alt_m = float(ac.get_property_value(c.position_h_sl_m))
                current_vc_mps = float(ac.get_property_value(c.velocities_vc_mps))
                current_hdg_deg = float(np.degrees(ac.get_property_value(c.attitude_heading_true_rad))) % 360.0
                current_pos = ac.get_position()
                g_load = float(ac.get_property_value(c.accelerations_n_pilot_z_norm)) if hasattr(c, 'accelerations_n_pilot_z_norm') else 0.0
                log.warning(
                    "[%s][T+%07.1fs][_lowlevel_control][model_forward][state] model=%s in_alt=%d in_hdg=%d in_vel=%d raw_obs=%s inp=%s rnn_before=%s out_raw=%s rnn_after=%s norm=%s alt=%.3fm vc=%.3fmps hdg=%.3fdeg pos=%s g=%.3f",
                    agent_id,
                    float(self.step_count * 0.2),
                    active_model_key,
                    alt,
                    hdg,
                    vel,
                    _fmt_full_array(raw),
                    _fmt_full_array(inp),
                    _fmt_full_array(rnn_before.reshape(-1)),
                    _fmt_full_array(out),
                    (
                        _fmt_full_array(self._inner_rnn_states[agent_id].reshape(-1))
                        if override_raw is None and active_model_key != "enemy_rule_pid"
                        else "override_no_rnn_update"
                    ),
                    _fmt_full_array(norm),
                    current_alt_m,
                    current_vc_mps,
                    current_hdg_deg,
                    _fmt_full_array(current_pos),
                    g_load,
                )
            except Exception as exc:
                log.warning("[%s][T+%07.1fs][_lowlevel_control][model_forward][state] failed=%s", agent_id, float(self.step_count * 0.2), exc)

        if is_enemy and root_trace:
            try:
                current_alt_m = float(env.agents[agent_id].get_property_value(c.position_h_sl_m))
                if current_alt_m < 5000.0 and (self.step_count % 5 == 0):
                    log.warning(
                        f"🧩 [根因链-低层网络] {agent_id} model={active_model_key} in=(alt={alt},hdg={hdg},vel={vel}) "
                        f"out_raw=({float(out[0]):.2f},{float(out[1]):.2f},{float(out[2]):.2f},{float(out[3]):.2f}) "
                        f"out_norm=(ail={float(norm[0]):+.3f},ele={float(norm[1]):+.3f},rud={float(norm[2]):+.3f},thr={float(norm[3]):.3f}) "
                        f"alt={current_alt_m:.0f}m step={self.step_count}"
                    )
            except Exception:
                pass
        elif root_trace and self._is_friendly_rootcause_return(agent_id):
            try:
                if self.step_count % 10 == 0:
                    current_alt_m = float(env.agents[agent_id].get_property_value(c.position_h_sl_m))
                    phase_name = self._get_bridge_phase_name(agent_id) or 'UNKNOWN'
                    log.warning(
                        f"🧩 [友机根因记录-低层网络] {agent_id} phase={phase_name} model={model_key} "
                        f"in=(alt={alt},hdg={hdg},vel={vel}) "
                        f"out_raw=({float(out[0]):.2f},{float(out[1]):.2f},{float(out[2]):.2f},{float(out[3]):.2f}) "
                        f"out_norm=(ail={float(norm[0]):+.3f},ele={float(norm[1]):+.3f},rud={float(norm[2]):+.3f},thr={float(norm[3]):.3f}) "
                        f"alt={current_alt_m:.0f}m step={self.step_count}"
                    )
            except Exception:
                pass

        if is_enemy and _CAP_CONTROL_DEBUG and self.step_count % 50000 == 0:
            try:
                current_alt_m = float(env.agents[agent_id].get_property_value(c.position_h_sl_m))
                current_pitch = np.degrees(float(env.agents[agent_id].get_property_value(c.attitude_pitch_rad)))
                delta_alt_val = self.norm_alt[min(alt, len(self.norm_alt) - 1)] if hasattr(self, 'norm_alt') else 0
                log.warning(
                    f"[DEBUG _lowlevel_control] {agent_id} step={self.step_count} "
                    f"??: alt={alt}(delta={delta_alt_val:.3f}) hdg={hdg} vel={vel} "
                    f"????: alt_m={current_alt_m:.1f} pitch={current_pitch:.1f}? "
                    f"????: out={out} ????: norm={norm} elevator={norm[1]:.3f}"
                )
            except Exception as exc:
                log.warning(f"[DEBUG _lowlevel_control] {agent_id} ??????: {exc}")

        return np.clip(norm, [-1, -1, -1, 0], [1, 1, 1, 1]).astype(np.float32)
    except Exception as exc:
        log.error(f"?????? {agent_id}: {exc}")
        return np.array([0.0, 0.0, 0.0, 0.7])
