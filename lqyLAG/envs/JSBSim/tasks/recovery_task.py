import math
import json
import logging
import os

import numpy as np

from .heading_task import HeadingTask
from ..core.catalog import Catalog as c
from ..reward_functions import RecoveryReward
from ..termination_conditions import RecoverySuccess, ExtremeState, LowAltitude, Overload, Timeout


class RecoveryTask(HeadingTask):
    """Single-aircraft recovery task for F16 low-energy control distillation / evaluation.

    Design goals:
    - keep exactly the same 12-dim observation and [41,41,41,30] action interface as
      the legacy baseline controller
    - sample a mixture of normal and low-energy initial conditions
    - expose helper functions so training scripts can choose between
      legacy-baseline teacher and rule-based recovery teacher
    """

    PROFILE_WEIGHTS = {
        "cruise": 0.20,
        "turn": 0.15,
        "climb": 0.10,
        "recover_high": 0.20,
        "recover_mid": 0.20,
        "ground_escape": 0.15,
    }
    RECOVERY_PROFILE_NAMES = {"recover_high", "recover_mid", "ground_escape"}
    HARDCASE_PROFILE_PREFIXES = ("cap_hardcase",)

    def __init__(self, config):
        super().__init__(config)
        self.reward_functions = [RecoveryReward(self.config)]
        self.termination_conditions = [
            RecoverySuccess(self.config),
            LowAltitude(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            Timeout(self.config),
        ]
        self._pending_profiles = {}
        self._active_profiles = {}
        self._stable_steps = {}
        self._external_init_pool = []
        self._external_init_prob = 0.0
        self._teacher_latch = {}
        self._teacher_release_stable_steps = int(getattr(self.config, "RecoveryTeacher_release_stable_steps", 45))
        self._teacher_min_hold_steps = int(getattr(self.config, "RecoveryTeacher_min_hold_steps", 30))
        self._teacher_early_vc_mps = float(getattr(self.config, "RecoveryTeacher_early_vc_mps", 150.0))
        self._teacher_early_sink_mps = float(getattr(self.config, "RecoveryTeacher_early_sink_mps", 3.0))
        self._teacher_early_bank_deg = float(getattr(self.config, "RecoveryTeacher_early_bank_deg", 35.0))
        self._teacher_early_aoa_deg = float(getattr(self.config, "RecoveryTeacher_early_aoa_deg", 10.0))
        self._load_external_init_pool_from_env()

    def set_external_init_pool(self, cases, probability: float = 0.35):
        self._external_init_pool = [dict(case) for case in (cases or [])]
        self._external_init_prob = float(np.clip(probability, 0.0, 1.0))

    def reset(self, env):
        self._stable_steps = {agent_id: 0 for agent_id in env.agents.keys()}
        if self._pending_profiles:
            self._active_profiles = dict(self._pending_profiles)
            self._pending_profiles = {}
        else:
            self._active_profiles = {agent_id: {"name": "cruise"} for agent_id in env.agents.keys()}
        self._teacher_latch = {
            agent_id: {
                "active": self._is_recovery_profile(self._active_profiles.get(agent_id, {}).get("name", "cruise")),
                "active_steps": 0,
                "min_hold_steps": int(
                    self._active_profiles.get(agent_id, {}).get("teacher_min_hold_steps", self._teacher_min_hold_steps)
                ),
                "release_stable_steps": int(
                    self._active_profiles.get(agent_id, {}).get(
                        "teacher_release_stable_steps",
                        self._teacher_release_stable_steps,
                    )
                ),
            }
            for agent_id in env.agents.keys()
        }
        super().reset(env)
        for agent_id in env.agents.keys():
            self._update_targets_for_agent(env, agent_id)

    def step(self, env):
        for agent_id in env.agents.keys():
            self._update_targets_for_agent(env, agent_id)
            metrics = self.get_recovery_metrics(env, agent_id)
            stable = (
                metrics["current_vc"] >= 135.0
                and metrics["aoa_deg"] <= 9.0
                and metrics["v_up_mps"] >= -2.0
                and abs(metrics["roll_deg"]) <= 20.0
            )
            self._stable_steps[agent_id] = self._stable_steps.get(agent_id, 0) + 1 if stable else 0

    def sample_init_state(self, env, agent_id, base_state):
        rng = env.np_random
        if self._external_init_pool and float(rng.uniform()) < self._external_init_prob:
            case = dict(self._external_init_pool[int(rng.integers(0, len(self._external_init_pool)))])
            init_heading = float(case.get("ic_psi_true_deg", rng.uniform(0.0, 180.0)))
            target_heading_offset_deg = float(case.get("target_heading_offset_deg", 0.0))
            nominal_alt_ft = float(
                case.get(
                    "target_altitude_ft",
                    case.get("ic_h_sl_ft", base_state.get("ic_h_sl_ft", 20000.0)) + 1200.0,
                )
            )
            nominal_speed_mps = float(
                case.get(
                    "target_velocities_u_mps",
                    max(float(case.get("ic_u_fps", 600.0)) * 0.3048 + 35.0, 165.0),
                )
            )
            target_heading_deg = float(case.get("target_heading_deg", init_heading + target_heading_offset_deg))

            base_state = dict(base_state)
            base_state.update(
                {
                    "ic_psi_true_deg": init_heading,
                    "ic_h_sl_ft": float(case.get("ic_h_sl_ft", base_state.get("ic_h_sl_ft", 20000.0))),
                    "ic_u_fps": float(case.get("ic_u_fps", base_state.get("ic_u_fps", 650.0))),
                    "ic_theta_deg": float(case.get("ic_theta_deg", base_state.get("ic_theta_deg", 0.0))),
                    "ic_phi_deg": float(case.get("ic_phi_deg", base_state.get("ic_phi_deg", 0.0))),
                    "ic_alpha_deg": float(case.get("ic_alpha_deg", base_state.get("ic_alpha_deg", 0.0))),
                    "ic_q_rad_sec": float(case.get("ic_q_rad_sec", 0.0)),
                    "ic_p_rad_sec": float(case.get("ic_p_rad_sec", 0.0)),
                    "ic_r_rad_sec": float(case.get("ic_r_rad_sec", 0.0)),
                    "ic_roc_fpm": float(case.get("ic_roc_fpm", 0.0)),
                    "target_heading_deg": float((target_heading_deg + 360.0) % 360.0),
                    "target_altitude_ft": nominal_alt_ft,
                    "target_velocities_u_mps": nominal_speed_mps,
                }
            )

            self._pending_profiles[agent_id] = {
                "name": case.get("profile_name", "cap_hardcase"),
                "initial_heading_deg": init_heading,
                "target_heading_offset_deg": target_heading_offset_deg,
                "nominal_alt_ft": nominal_alt_ft,
                "nominal_speed_mps": nominal_speed_mps,
                "teacher_min_hold_steps": int(case.get("teacher_min_hold_steps", self._teacher_min_hold_steps)),
                "teacher_release_stable_steps": int(
                    case.get("teacher_release_stable_steps", self._teacher_release_stable_steps)
                ),
                "source_trace": case.get("source_trace", ""),
                "source_step": int(case.get("source_step", -1)),
                "window_tag": case.get("window_tag", ""),
            }
            return base_state

        profile_name = rng.choice(
            list(self.PROFILE_WEIGHTS.keys()),
            p=np.array(list(self.PROFILE_WEIGHTS.values()), dtype=np.float64),
        )
        init_heading = float(rng.uniform(0.0, 180.0))

        if profile_name == "cruise":
            alt_ft = float(rng.uniform(18000.0, 32000.0))
            u_fps = float(rng.uniform(550.0, 900.0))
            theta_deg = float(rng.uniform(-3.0, 6.0))
            phi_deg = float(rng.uniform(-5.0, 5.0))
            alpha_deg = float(rng.uniform(-1.0, 5.0))
            roc_fpm = float(rng.uniform(-500.0, 500.0))
            target_heading_offset_deg = float(rng.uniform(-10.0, 10.0))
            nominal_alt_ft = alt_ft
            nominal_speed_mps = float(u_fps * 0.3048)
        elif profile_name == "turn":
            alt_ft = float(rng.uniform(16000.0, 28000.0))
            u_fps = float(rng.uniform(500.0, 850.0))
            theta_deg = float(rng.uniform(-2.0, 5.0))
            phi_deg = float(rng.uniform(-10.0, 10.0))
            alpha_deg = float(rng.uniform(-1.0, 6.0))
            roc_fpm = float(rng.uniform(-400.0, 400.0))
            target_heading_offset_deg = float(rng.choice([-45.0, -30.0, 30.0, 45.0]))
            nominal_alt_ft = alt_ft + float(rng.uniform(-500.0, 500.0))
            nominal_speed_mps = float(rng.uniform(180.0, 250.0))
        elif profile_name == "climb":
            alt_ft = float(rng.uniform(10000.0, 22000.0))
            u_fps = float(rng.uniform(400.0, 750.0))
            theta_deg = float(rng.uniform(0.0, 8.0))
            phi_deg = float(rng.uniform(-8.0, 8.0))
            alpha_deg = float(rng.uniform(0.0, 7.0))
            roc_fpm = float(rng.uniform(-300.0, 600.0))
            target_heading_offset_deg = float(rng.uniform(-10.0, 10.0))
            nominal_alt_ft = alt_ft + float(rng.uniform(800.0, 2500.0))
            nominal_speed_mps = float(rng.uniform(170.0, 240.0))
        elif profile_name == "recover_high":
            alt_ft = float(rng.uniform(18000.0, 32000.0))
            u_fps = float(rng.uniform(180.0, 420.0))
            theta_deg = float(rng.uniform(6.0, 18.0))
            phi_deg = float(rng.uniform(-8.0, 8.0))
            alpha_deg = float(rng.uniform(6.0, 14.0))
            roc_fpm = float(rng.uniform(-4000.0, 500.0))
            target_heading_offset_deg = 0.0
            nominal_alt_ft = alt_ft
            nominal_speed_mps = 185.0
        elif profile_name == "recover_mid":
            alt_ft = float(rng.uniform(5000.0, 14000.0))
            u_fps = float(rng.uniform(170.0, 360.0))
            theta_deg = float(rng.uniform(4.0, 16.0))
            phi_deg = float(rng.uniform(-10.0, 10.0))
            alpha_deg = float(rng.uniform(4.0, 12.0))
            roc_fpm = float(rng.uniform(-6000.0, 0.0))
            target_heading_offset_deg = 0.0
            nominal_alt_ft = alt_ft
            nominal_speed_mps = 165.0
        else:
            profile_name = "ground_escape"
            alt_ft = float(rng.uniform(1200.0, 3500.0))
            u_fps = float(rng.uniform(180.0, 320.0))
            theta_deg = float(rng.uniform(-20.0, 8.0))
            phi_deg = float(rng.uniform(-8.0, 8.0))
            alpha_deg = float(rng.uniform(0.0, 8.0))
            roc_fpm = float(rng.uniform(-7000.0, -500.0))
            target_heading_offset_deg = 0.0
            nominal_alt_ft = alt_ft + 1000.0
            nominal_speed_mps = 150.0

        base_state = dict(base_state)
        base_state.update(
            {
                "ic_psi_true_deg": init_heading,
                "ic_h_sl_ft": alt_ft,
                "ic_u_fps": u_fps,
                "ic_theta_deg": theta_deg,
                "ic_phi_deg": phi_deg,
                "ic_alpha_deg": alpha_deg,
                "ic_q_rad_sec": 0.0,
                "ic_p_rad_sec": 0.0,
                "ic_r_rad_sec": 0.0,
                "ic_roc_fpm": roc_fpm,
                "target_heading_deg": init_heading + target_heading_offset_deg,
                "target_altitude_ft": nominal_alt_ft,
                "target_velocities_u_mps": nominal_speed_mps,
            }
        )

        self._pending_profiles[agent_id] = {
            "name": profile_name,
            "initial_heading_deg": init_heading,
            "target_heading_offset_deg": target_heading_offset_deg,
            "nominal_alt_ft": nominal_alt_ft,
            "nominal_speed_mps": nominal_speed_mps,
            "teacher_min_hold_steps": self._teacher_min_hold_steps,
            "teacher_release_stable_steps": self._teacher_release_stable_steps,
        }
        return base_state

    def get_profile_name(self, agent_id):
        return self._active_profiles.get(agent_id, {}).get("name", "cruise")

    def _load_external_init_pool_from_env(self):
        path = os.getenv("RECOVERY_HARDCASE_JSON", "").strip()
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                cases = payload.get("cases", [payload])
            elif isinstance(payload, list):
                cases = payload
            else:
                raise ValueError(f"Unsupported hardcase payload type: {type(payload)!r}")
            probability = float(os.getenv("RECOVERY_HARDCASE_PROB", "0.0"))
            self.set_external_init_pool(cases, probability)
            logging.info(
                "[recovery-task] loaded hardcases from env: path=%s cases=%d prob=%.2f",
                path,
                len(self._external_init_pool),
                self._external_init_prob,
            )
        except Exception as exc:
            logging.warning("[recovery-task] failed to load hardcases from %s: %s", path, exc)

    def _is_recovery_profile(self, profile_name: str) -> bool:
        return (
            profile_name in self.RECOVERY_PROFILE_NAMES
            or any(profile_name.startswith(prefix) for prefix in self.HARDCASE_PROFILE_PREFIXES)
        )

    def _is_precovery_state(self, metrics) -> bool:
        return (
            metrics["current_vc"] < self._teacher_early_vc_mps
            or (
                metrics["current_vc"] < (self._teacher_early_vc_mps + 15.0)
                and metrics["v_up_mps"] < -self._teacher_early_sink_mps
            )
            or metrics["v_up_mps"] < -(self._teacher_early_sink_mps + 6.0)
            or metrics["aoa_deg"] > self._teacher_early_aoa_deg
            or (
                abs(metrics["roll_deg"]) > self._teacher_early_bank_deg
                and metrics["current_vc"] < (self._teacher_early_vc_mps + 20.0)
            )
        )

    def _get_energy_rebuild_targets(self, metrics, current_alt_ft: float, nominal_alt_ft: float, nominal_speed_mps: float):
        if metrics["alt_m"] < 1200.0:
            target_alt_ft = current_alt_ft + 600.0
        elif (
            metrics["current_vc"] < (self._teacher_early_vc_mps - 10.0)
            or metrics["v_up_mps"] < -(self._teacher_early_sink_mps + 4.0)
            or metrics["aoa_deg"] > (self._teacher_early_aoa_deg + 2.0)
        ):
            if metrics["alt_m"] > 2500.0:
                target_alt_ft = current_alt_ft - 200.0
            elif metrics["alt_m"] > 1600.0:
                target_alt_ft = current_alt_ft
            else:
                target_alt_ft = current_alt_ft + 200.0
        else:
            target_alt_ft = min(max(current_alt_ft, nominal_alt_ft), current_alt_ft + 300.0)

        target_speed_mps = max(metrics["current_vc"] + 35.0, nominal_speed_mps, 185.0)
        return float(target_alt_ft), float(target_speed_mps)

    def get_recovery_metrics(self, env, agent_id):
        aircraft = env.agents[agent_id]
        current_alt = float(aircraft.get_property_value(c.position_h_sl_m))
        current_heading_deg = float(math.degrees(aircraft.get_property_value(c.attitude_heading_true_rad)))
        current_roll_deg = float(math.degrees(aircraft.get_property_value(c.attitude_roll_rad)))
        current_pitch_deg = float(math.degrees(aircraft.get_property_value(c.attitude_pitch_rad)))
        current_vc = float(aircraft.get_property_value(c.velocities_vc_mps))
        current_tas = float(np.linalg.norm(aircraft.get_velocity()))
        v_up_mps = float(aircraft.get_velocity()[-1])
        aoa_deg = float(aircraft.get_property_value(c.aero_alpha_deg))

        ground_escape = current_alt < 350.0 or (current_alt < 700.0 and (v_up_mps < -25.0 or current_pitch_deg < -25.0))
        hard_stall = current_vc < 70.0 or (current_vc < 85.0 and abs(aoa_deg) > 16.0)
        deep_stall = (
            hard_stall
            or current_vc < 85.0
            or (current_vc < 100.0 and (abs(aoa_deg) > 14.0 or v_up_mps < -8.0))
        )
        low_energy = deep_stall or current_vc < 120.0 or (current_vc < 140.0 and v_up_mps < -4.0)

        return {
            "alt_m": current_alt,
            "heading_deg": current_heading_deg,
            "roll_deg": current_roll_deg,
            "pitch_deg": current_pitch_deg,
            "current_vc": current_vc,
            "current_tas": current_tas,
            "v_up_mps": v_up_mps,
            "aoa_deg": aoa_deg,
            "ground_escape": ground_escape,
            "hard_stall": hard_stall,
            "deep_stall": deep_stall,
            "low_energy": low_energy,
        }

    def should_use_recovery_teacher(self, env, agent_id):
        profile_name = self.get_profile_name(agent_id)
        profile = self._active_profiles.get(agent_id, {})
        metrics = self.get_recovery_metrics(env, agent_id)
        latch = self._teacher_latch.setdefault(
            agent_id,
            {
                "active": False,
                "active_steps": 0,
                "min_hold_steps": self._teacher_min_hold_steps,
                "release_stable_steps": self._teacher_release_stable_steps,
            },
        )
        hard_recovery_profile = self._is_recovery_profile(profile_name)
        trigger_recovery = (
            metrics["ground_escape"]
            or metrics["low_energy"]
            or (hard_recovery_profile and self._is_precovery_state(metrics))
        )

        if trigger_recovery:
            latch["active"] = True

        if not latch["active"]:
            latch["active_steps"] = 0
            return False

        latch["active_steps"] += 1
        min_hold_steps = max(int(profile.get("teacher_min_hold_steps", self._teacher_min_hold_steps)), 1)
        release_stable_steps = max(
            int(profile.get("teacher_release_stable_steps", self._teacher_release_stable_steps)),
            1,
        )
        latch["min_hold_steps"] = min_hold_steps
        latch["release_stable_steps"] = release_stable_steps

        stable_steps = self._stable_steps.get(agent_id, 0)
        if stable_steps >= release_stable_steps and latch["active_steps"] >= min_hold_steps:
            latch["active"] = False
            latch["active_steps"] = 0
            return False

        return True

    def build_teacher_observation(self, env, agent_id):
        return self.get_obs(env, agent_id).copy()

    def get_rule_teacher_action(self, env, agent_id):
        metrics = self.get_recovery_metrics(env, agent_id)
        aircraft = env.agents[agent_id]
        delta_alt_m = float(aircraft.get_property_value(c.delta_altitude))
        delta_heading_rad = math.radians(float(aircraft.get_property_value(c.delta_heading)))
        delta_velocity_mps = float(aircraft.get_property_value(c.delta_velocities_u))

        if metrics["low_energy"]:
            aileron = float(np.clip(-metrics["roll_deg"] / 90.0, -0.20, 0.20))
            rudder = 0.0
        else:
            aileron = float(np.clip(delta_heading_rad / math.radians(45.0), -0.35, 0.35))
            rudder = float(np.clip(aileron * 0.35, -0.15, 0.15))

        throttle = 0.9

        if metrics["ground_escape"]:
            elevator = -0.90
        elif metrics["hard_stall"]:
            elevator = 0.20 if metrics["alt_m"] > 900.0 else -0.75
        elif metrics["deep_stall"]:
            elevator = 0.14 if metrics["alt_m"] > 2200.0 else -0.60
        elif self._is_precovery_state(metrics):
            if metrics["alt_m"] > 2200.0:
                elevator = 0.12
            elif metrics["alt_m"] > 1400.0:
                elevator = -0.05
            else:
                elevator = -0.45
        else:
            if delta_alt_m > 300.0:
                elevator = -0.22
            elif delta_alt_m > 80.0:
                elevator = -0.12
            elif delta_alt_m < -300.0:
                elevator = 0.16
            elif delta_alt_m < -80.0:
                elevator = 0.08
            else:
                elevator = 0.0

        if delta_velocity_mps > 30.0:
            throttle = 0.9
        elif delta_velocity_mps > 10.0:
            throttle = 0.82
        elif delta_velocity_mps < -25.0:
            throttle = 0.55
        elif delta_velocity_mps < -10.0:
            throttle = 0.65

        if metrics["low_energy"]:
            throttle = 0.9

        return self._norm_action_to_indices(
            np.array([aileron, elevator, rudder, throttle], dtype=np.float32)
        )

    def _update_targets_for_agent(self, env, agent_id):
        aircraft = env.agents[agent_id]
        profile = self._active_profiles.get(agent_id, {"name": "cruise"})
        metrics = self.get_recovery_metrics(env, agent_id)
        teacher_active = self._teacher_latch.get(agent_id, {}).get("active", False)
        recovery_profile = self._is_recovery_profile(profile.get("name", "cruise"))

        current_alt_ft = float(metrics["alt_m"] / 0.3048)
        current_heading_deg = float((metrics["heading_deg"] + 360.0) % 360.0)
        nominal_alt_ft = float(profile.get("nominal_alt_ft", current_alt_ft))
        nominal_speed_mps = float(profile.get("nominal_speed_mps", metrics["current_vc"]))

        if metrics["ground_escape"]:
            target_heading_deg = current_heading_deg
            target_alt_ft = current_alt_ft + 1200.0
            target_speed_mps = max(metrics["current_vc"], 170.0)
        elif metrics["hard_stall"]:
            target_heading_deg = current_heading_deg
            target_alt_ft = current_alt_ft - 500.0 if metrics["alt_m"] > 700.0 else current_alt_ft + 900.0
            target_speed_mps = max(metrics["current_vc"] + 45.0, 165.0)
        elif metrics["deep_stall"]:
            target_heading_deg = current_heading_deg
            target_alt_ft = current_alt_ft - 350.0 if metrics["alt_m"] > 2200.0 else current_alt_ft + 600.0
            target_speed_mps = max(metrics["current_vc"] + 35.0, 155.0)
        elif teacher_active or (recovery_profile and self._is_precovery_state(metrics)):
            target_heading_deg = current_heading_deg
            target_alt_ft, target_speed_mps = self._get_energy_rebuild_targets(
                metrics,
                current_alt_ft,
                nominal_alt_ft,
                nominal_speed_mps,
            )
        else:
            target_heading_deg = current_heading_deg + float(profile.get("target_heading_offset_deg", 0.0))
            target_alt_ft = nominal_alt_ft
            target_speed_mps = nominal_speed_mps

        aircraft.set_property_value(c.target_heading_deg, float((target_heading_deg + 360.0) % 360.0))
        aircraft.set_property_value(c.target_altitude_ft, float(target_alt_ft))
        aircraft.set_property_value(c.target_velocities_u_mps, float(target_speed_mps))

    @staticmethod
    def _norm_action_to_indices(norm_act):
        norm_act = np.asarray(norm_act, dtype=np.float32)
        action = np.zeros(4, dtype=np.int64)
        action[0] = int(np.clip(np.rint((norm_act[0] + 1.0) * 20.0), 0, 40))
        action[1] = int(np.clip(np.rint((norm_act[1] + 1.0) * 20.0), 0, 40))
        action[2] = int(np.clip(np.rint((norm_act[2] + 1.0) * 20.0), 0, 40))
        action[3] = int(np.clip(np.rint((norm_act[3] - 0.4) * 58.0), 0, 29))
        return action
