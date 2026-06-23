from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np

try:
    from envs.JSBSim.core.catalog import Catalog as c
except ImportError:
    class _MockCatalog:
        attitude_psi_rad = "attitude/psi-rad"
        attitude_pitch_rad = "attitude/pitch-rad"
        attitude_phi_rad = "attitude/phi-rad"
        position_h_sl_m = "position/h-sl-m"
        velocities_vc_mps = "velocities/vc-mps"
        velocities_v_down_mps = "velocities/v-down-mps"

    c = _MockCatalog()


class TeacherMode(str, Enum):
    MODEL = "model"
    SAFE_HOLD = "safe_hold"
    ENERGY_RECOVER = "energy_recover"


@dataclass(frozen=True)
class EnemyEnergySnapshot:
    altitude_m: float
    vc_mps: float
    v_up_mps: float
    pitch_deg: float
    roll_deg: float
    heading_deg: float
    g_load: float


@dataclass(frozen=True)
class EnemySafeTeacherDecision:
    mode: TeacherMode
    command: Tuple[int, int, int]
    reason: str
    action_override_raw: Optional[Tuple[int, int, int, int]] = None


class EnemySafeTeacher:
    """
    Safety supervisor for the enemy low-level controller.

    This module is intentionally not wired into the runtime yet. It provides the
    decision interface that later CAP low-level training and teacher-data
    collection can reuse without rebuilding the safety logic from scratch.
    """

    def __init__(
        self,
        hold_speed_floor_mps: float = 240.0,
        recover_speed_floor_mps: float = 225.0,
        high_altitude_floor_m: float = 7000.0,
        severe_sink_rate_mps: float = -15.0,
        hold_bank_limit_deg: float = 35.0,
    ) -> None:
        self.hold_speed_floor_mps = float(hold_speed_floor_mps)
        self.recover_speed_floor_mps = float(recover_speed_floor_mps)
        self.high_altitude_floor_m = float(high_altitude_floor_m)
        self.severe_sink_rate_mps = float(severe_sink_rate_mps)
        self.hold_bank_limit_deg = float(hold_bank_limit_deg)

    def snapshot(self, env, agent_id: str) -> Optional[EnemyEnergySnapshot]:
        aircraft = getattr(env, "agents", {}).get(agent_id)
        if aircraft is None or not getattr(aircraft, "is_alive", False):
            return None

        altitude_m = float(aircraft.get_property_value(c.position_h_sl_m))
        vc_mps = float(aircraft.get_property_value(c.velocities_vc_mps))
        v_up_mps = -float(aircraft.get_property_value(c.velocities_v_down_mps))
        pitch_deg = float(np.rad2deg(aircraft.get_property_value(c.attitude_pitch_rad)))
        roll_deg = abs(float(np.rad2deg(aircraft.get_property_value(c.attitude_phi_rad))))
        heading_deg = float(np.rad2deg(aircraft.get_property_value(c.attitude_psi_rad)) % 360.0)
        try:
            g_load = float(aircraft.get_property_value(c.accelerations_n_pilot_z_norm))
        except Exception:
            g_load = 0.0

        return EnemyEnergySnapshot(
            altitude_m=altitude_m,
            vc_mps=vc_mps,
            v_up_mps=v_up_mps,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            heading_deg=heading_deg,
            g_load=g_load,
        )

    def classify_mode(self, snap: EnemyEnergySnapshot) -> TeacherMode:
        high_altitude = snap.altitude_m >= self.high_altitude_floor_m
        climb_drain = (
            snap.v_up_mps >= 10.0
            and snap.pitch_deg >= 4.0
            and snap.vc_mps <= self.hold_speed_floor_mps + 5.0
        )
        aggressive_climb_drain = (
            snap.v_up_mps >= 20.0
            and snap.pitch_deg >= 7.0
        )
        heavy_load = abs(snap.g_load) >= 2.2
        extreme_load = abs(snap.g_load) >= 4.0
        hard_bank = snap.roll_deg >= max(self.hold_bank_limit_deg, 65.0)
        extreme_bank = snap.roll_deg >= 110.0
        deep_sink = snap.v_up_mps <= self.severe_sink_rate_mps

        if (
            high_altitude
            and (
                snap.vc_mps < self.recover_speed_floor_mps
                or deep_sink
                or extreme_bank
                or extreme_load
                or aggressive_climb_drain
            )
        ):
            return TeacherMode.ENERGY_RECOVER

        if (
            high_altitude
            and (
                snap.vc_mps < self.hold_speed_floor_mps
                or hard_bank
                or heavy_load
                or climb_drain
                or snap.pitch_deg < -6.0
                or snap.v_up_mps <= -8.0
            )
        ):
            return TeacherMode.SAFE_HOLD

        return TeacherMode.MODEL

    def _norm_act_to_raw_bins(
        self,
        aileron: float,
        elevator: float,
        rudder: float,
        throttle: float,
    ) -> Tuple[int, int, int, int]:
        ail_raw = int(round(np.clip(float(aileron), -1.0, 1.0) * 20.0 + 20.0))
        ele_raw = int(round(np.clip(float(elevator), -1.0, 1.0) * 20.0 + 20.0))
        rud_raw = int(round(np.clip(float(rudder), -1.0, 1.0) * 20.0 + 20.0))
        thr_norm = float(np.clip(float(throttle), 0.4, 0.9))
        thr_raw = int(round((thr_norm - 0.4) * 58.0))
        return (
            int(np.clip(ail_raw, 0, 40)),
            int(np.clip(ele_raw, 0, 40)),
            int(np.clip(rud_raw, 0, 40)),
            int(np.clip(thr_raw, 0, 29)),
        )

    def _build_safe_hold_override(self, snap: EnemyEnergySnapshot) -> Tuple[int, int, int, int]:
        bank_limit = 0.45 if snap.roll_deg >= 100.0 else 0.32
        aileron = float(np.clip(-snap.roll_deg / 45.0, -bank_limit, bank_limit))
        if snap.v_up_mps < -12.0 and snap.pitch_deg < 6.0:
            elevator = -0.52
        elif snap.v_up_mps > 20.0 or snap.pitch_deg > 8.0 or abs(snap.g_load) > 2.5:
            elevator = 0.32
        elif snap.v_up_mps > 8.0 or snap.pitch_deg > 4.0:
            elevator = 0.22
        else:
            elevator = 0.12
        return self._norm_act_to_raw_bins(aileron, elevator, 0.0, 0.9)

    def _build_energy_recover_override(self, snap: EnemyEnergySnapshot) -> Tuple[int, int, int, int]:
        bank_limit = 0.50 if snap.roll_deg >= 120.0 else 0.36
        aileron = float(np.clip(-snap.roll_deg / 35.0, -bank_limit, bank_limit))
        if snap.v_up_mps < -20.0 and snap.pitch_deg < 8.0:
            elevator = -0.78 if snap.vc_mps < 230.0 else -0.62
        elif snap.v_up_mps > 25.0 or snap.pitch_deg > 9.0:
            elevator = 0.38
        elif snap.vc_mps < self.recover_speed_floor_mps:
            elevator = 0.26
        else:
            elevator = 0.18
        return self._norm_act_to_raw_bins(aileron, elevator, 0.0, 0.9)

    def build_teacher_command(
        self,
        snap: EnemyEnergySnapshot,
        model_cmd: Tuple[int, int, int],
    ) -> EnemySafeTeacherDecision:
        mode = self.classify_mode(snap)

        if mode == TeacherMode.MODEL:
            return EnemySafeTeacherDecision(
                mode=mode,
                command=tuple(int(x) for x in model_cmd),
                reason="model_ok",
                action_override_raw=None,
            )

        if mode == TeacherMode.SAFE_HOLD:
            return EnemySafeTeacherDecision(
                mode=mode,
                command=(7, 8, 6),
                reason="hold_unload_and_accelerate",
                action_override_raw=self._build_safe_hold_override(snap),
            )

        return EnemySafeTeacherDecision(
            mode=mode,
            command=(6, 8, 6),
            reason="recover_shallow_descent_and_accelerate",
            action_override_raw=self._build_energy_recover_override(snap),
        )
