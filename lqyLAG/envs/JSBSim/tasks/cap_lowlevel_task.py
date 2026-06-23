import copy
import json
import logging
import math
import os
from typing import Dict, List, Tuple

import numpy as np
from gymnasium import spaces

from .task_base import BaseTask
from ..core.catalog import Catalog as c
from ..reward_functions import AltitudeReward, CapTrackingReward, EnergyManagementReward, RecoveryReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout


log = logging.getLogger(__name__)

FT_PER_M = 3.280839895013123
FPS_PER_MPS = 3.280839895013123
FPM_PER_MPS = 196.8503937007874
TRUTHY_VALUES = {"1", "true", "yes", "on"}


class CapLowLevelTask(BaseTask):
    """
    CAP-native low-level control task.

    This task matches the deployment contract used by tacticalProject:
    - command bins are exactly the 15/17/7 bins used online
    - target properties are rebuilt from the current command segment instead of
      the short-horizon heading-task scheduler
    - observation semantics can be switched between:
      * direct_command: obs[:3] are command values themselves
      * residual: obs[:3] are live tracking residuals to the segment target
    """

    ALT_COMMAND_M = np.array(
        [-1500, -1000, -750, -500, -300, -150, -50, 0, 50, 150, 300, 500, 750, 1000, 1500],
        dtype=np.float32,
    )
    HDG_COMMAND_RAD = np.array(
        [
            -math.pi,
            -2 * math.pi / 3,
            -math.pi / 2,
            -5 * math.pi / 12,
            -math.pi / 3,
            -math.pi / 4,
            -math.pi / 6,
            -math.pi / 12,
            0.0,
            math.pi / 12,
            math.pi / 6,
            math.pi / 4,
            math.pi / 3,
            5 * math.pi / 12,
            math.pi / 2,
            2 * math.pi / 3,
            math.pi,
        ],
        dtype=np.float32,
    )
    VEL_COMMAND_MPS = np.array([-150, -100, -50, 0, 50, 100, 150], dtype=np.float32)

    FOUNDATION_PROFILE_WEIGHTS = {
        "cruise": 0.22,
        "turn": 0.20,
        "vertical": 0.18,
        "aggressive": 0.16,
        "recover_high": 0.10,
        "recover_mid": 0.08,
        "ground_escape": 0.06,
    }
    HARDCASE_PROFILE_WEIGHTS = {
        "cruise": 0.12,
        "turn": 0.14,
        "vertical": 0.14,
        "aggressive": 0.16,
        "recover_high": 0.18,
        "recover_mid": 0.16,
        "ground_escape": 0.10,
    }
    RECOVERY_PROFILES = {"recover_high", "recover_mid", "ground_escape"}
    HARDCODED_RECOVERY_HDG_INDEX = 8

    def __init__(self, config):
        self.config = config
        self.profile_mode = str(getattr(config, "cap_profile_mode", "foundation")).strip().lower() or "foundation"
        self.command_hold_min_steps = int(getattr(config, "command_hold_min_steps", 12))
        self.command_hold_max_steps = int(getattr(config, "command_hold_max_steps", 72))
        self.min_target_speed_mps = float(getattr(config, "min_target_speed_mps", 120.0))
        self.max_target_speed_mps = float(getattr(config, "max_target_speed_mps", 330.0))
        self.min_target_altitude_ft = float(getattr(config, "min_target_altitude_ft", 800.0))
        self.max_target_altitude_ft = float(getattr(config, "max_target_altitude_ft", 45000.0))
        self.low_energy_vc_mps = float(getattr(config, "low_energy_vc_mps", 135.0))
        self.severe_low_energy_vc_mps = float(getattr(config, "severe_low_energy_vc_mps", 105.0))
        self.high_sink_rate_mps = float(getattr(config, "high_sink_rate_mps", 6.0))
        self.severe_sink_rate_mps = float(getattr(config, "severe_sink_rate_mps", 12.0))
        self.min_recovery_altitude_m = float(getattr(config, "min_recovery_altitude_m", 900.0))
        self.energy_guard_altitude_m = float(getattr(config, "energy_guard_altitude_m", 8500.0))
        self.energy_guard_vc_mps = float(getattr(config, "energy_guard_vc_mps", 225.0))
        self.energy_guard_hard_altitude_m = float(getattr(config, "energy_guard_hard_altitude_m", 10500.0))
        self.energy_guard_hard_vc_mps = float(getattr(config, "energy_guard_hard_vc_mps", 210.0))
        self.timeout_reward_bonus = float(getattr(config, "timeout_reward_bonus", 140.0))
        self.low_altitude_penalty = float(getattr(config, "low_altitude_penalty", 220.0))
        self.extreme_state_penalty = float(getattr(config, "extreme_state_penalty", 220.0))
        self.overload_penalty = float(getattr(config, "overload_penalty", 180.0))
        self.other_failure_penalty = float(getattr(config, "other_failure_penalty", 160.0))
        self.foundation_profile_weights = self._load_profile_weights(self.FOUNDATION_PROFILE_WEIGHTS, "foundation")
        self.hardcase_profile_weights = self._load_profile_weights(self.HARDCASE_PROFILE_WEIGHTS, "hardcase")
        self.obs_semantics = str(getattr(config, "cap_obs_semantics", "direct_command")).strip().lower() or "direct_command"
        self.command_sampling_mode = (
            str(os.getenv("CAP_LOWLEVEL_COMMAND_SAMPLING", getattr(config, "cap_command_sampling_mode", "profile")))
            .strip()
            .lower()
            or "profile"
        )
        self.command_trace = str(os.getenv("CAP_LOWLEVEL_TRACE", "")).strip().lower() in TRUTHY_VALUES

        self._external_init_pool: List[dict] = []
        self._external_init_prob = 0.0
        self._external_curriculum_pool: List[dict] = []
        self._external_curriculum_prob = 0.0
        self._pending_profiles: Dict[str, dict] = {}
        self._active_profiles: Dict[str, dict] = {}
        self._segment_states: Dict[str, dict] = {}

        super().__init__(config)

        self.reward_functions = [
            CapTrackingReward(self.config),
            AltitudeReward(self.config),
            EnergyManagementReward(self.config),
            RecoveryReward(self.config),
        ]
        self.termination_conditions = [
            LowAltitude(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            Timeout(self.config),
        ]
        self._load_external_init_pool_from_env()
        self._load_external_curriculum_pool_from_env()

    def get_reward(self, env, agent_id, info={}):
        reward = 0.0
        apply_recovery = self._should_apply_recovery_reward(env, agent_id)
        for reward_function in self.reward_functions:
            if isinstance(reward_function, RecoveryReward) and not apply_recovery:
                continue
            reward += reward_function.get_reward(self, env, agent_id)
        reward += self._get_outcome_reward(env, agent_id, info)
        return reward, info

    @property
    def num_agents(self):
        return 1

    def load_variables(self):
        self.state_var = [
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.velocities_u_mps,
            c.velocities_v_mps,
            c.velocities_w_mps,
            c.velocities_vc_mps,
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,
            c.fcs_elevator_cmd_norm,
            c.fcs_rudder_cmd_norm,
            c.fcs_throttle_cmd_norm,
        ]
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(12,))

    def load_action_space(self):
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])

    def reset(self, env):
        if self._pending_profiles:
            self._active_profiles = copy.deepcopy(self._pending_profiles)
            self._pending_profiles = {}
        else:
            self._active_profiles = {agent_id: {"name": "cruise"} for agent_id in env.agents.keys()}

        self._segment_states = {}
        super().reset(env)

        for agent_id in env.agents.keys():
            self._activate_new_segment(env, agent_id, initial=True)

    def step(self, env):
        for agent_id in env.agents.keys():
            segment = self._segment_states.get(agent_id)
            if segment is None:
                self._activate_new_segment(env, agent_id, initial=True)
                continue

            segment["elapsed_steps"] += 1
            if segment["elapsed_steps"] >= segment["hold_steps"]:
                self._activate_new_segment(env, agent_id, initial=False)

    def get_obs(self, env, agent_id):
        obs = np.zeros(12, dtype=np.float32)
        aircraft = env.agents[agent_id]
        segment = self._segment_states.get(agent_id)
        if segment is None:
            alt_idx, hdg_idx, vel_idx = 7, 8, 3
        else:
            alt_idx = int(segment["alt_idx"])
            hdg_idx = int(segment["hdg_idx"])
            vel_idx = int(segment["vel_idx"])

        if self.obs_semantics == "residual":
            obs[0] = float(aircraft.get_property_value(c.delta_altitude)) / 1000.0
            obs[1] = math.radians(float(aircraft.get_property_value(c.delta_heading)))
            obs[2] = float(aircraft.get_property_value(c.delta_velocities_u)) / 340.0
        else:
            obs[0] = self.ALT_COMMAND_M[alt_idx] / 1000.0
            obs[1] = self.HDG_COMMAND_RAD[hdg_idx]
            obs[2] = self.VEL_COMMAND_MPS[vel_idx] / 100.0

        altitude_m = float(aircraft.get_property_value(c.position_h_sl_m))
        roll_rad = float(aircraft.get_property_value(c.attitude_roll_rad))
        pitch_rad = float(aircraft.get_property_value(c.attitude_pitch_rad))
        u_mps = float(aircraft.get_property_value(c.velocities_u_mps))
        v_mps = float(aircraft.get_property_value(c.velocities_v_mps))
        w_mps = float(aircraft.get_property_value(c.velocities_w_mps))
        vc_mps = float(aircraft.get_property_value(c.velocities_vc_mps))

        obs[3] = altitude_m / 5000.0
        obs[4] = math.sin(roll_rad)
        obs[5] = math.cos(roll_rad)
        obs[6] = math.sin(pitch_rad)
        obs[7] = math.cos(pitch_rad)
        obs[8] = u_mps / 340.0
        obs[9] = v_mps / 340.0
        obs[10] = w_mps / 340.0
        obs[11] = vc_mps / 340.0
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def normalize_action(self, env, agent_id, action):
        norm_act = np.zeros(4, dtype=np.float32)
        norm_act[0] = action[0] * 2.0 / (self.action_space.nvec[0] - 1.0) - 1.0
        norm_act[1] = action[1] * 2.0 / (self.action_space.nvec[1] - 1.0) - 1.0
        norm_act[2] = action[2] * 2.0 / (self.action_space.nvec[2] - 1.0) - 1.0
        norm_act[3] = action[3] * 0.5 / (self.action_space.nvec[3] - 1.0) + 0.4
        return norm_act

    def set_external_init_pool(self, cases, probability: float = 0.35):
        self._external_init_pool = [dict(case) for case in (cases or [])]
        self._external_init_prob = float(np.clip(probability, 0.0, 1.0))

    def set_external_curriculum_pool(self, cases, probability: float = 1.0):
        self._external_curriculum_pool = [copy.deepcopy(case) for case in (cases or [])]
        self._external_curriculum_prob = float(np.clip(probability, 0.0, 1.0))

    def sample_init_state(self, env, agent_id, base_state):
        rng = env.np_random
        if self._external_curriculum_pool and float(rng.uniform()) < self._external_curriculum_prob:
            case = copy.deepcopy(self._external_curriculum_pool[int(rng.integers(0, len(self._external_curriculum_pool)))])
            sequence_segments = [dict(segment) for segment in case.get("sequence_segments", [])]
            first_segment = dict(case.get("first_segment", sequence_segments[0] if sequence_segments else {}))
            profile = {
                "name": str(case.get("profile_name", "cap_tactical")),
                "source": "tactical_curriculum",
                "first_segment": first_segment,
                "sequence_segments": sequence_segments,
                "sequence_cursor": 0,
                "source_log": str(case.get("source_log", "")),
                "source_agent": str(case.get("source_agent", "")),
                "source_case_id": str(case.get("case_id", "")),
            }
            self._pending_profiles[agent_id] = profile
            return self._build_case_init_state(base_state, case)

        if self._external_init_pool and float(rng.uniform()) < self._external_init_prob:
            case = dict(self._external_init_pool[int(rng.integers(0, len(self._external_init_pool)))])
            profile = {
                "name": str(case.get("profile_name", "cap_hardcase")),
                "source": "hardcase",
                "first_segment": dict(case.get("first_segment", {})),
                "source_trace": str(case.get("source_trace", "")),
                "source_step": int(case.get("source_step", -1)),
            }
            self._pending_profiles[agent_id] = profile
            return self._build_case_init_state(base_state, case)

        profile_name = self._sample_profile_name(rng)
        self._pending_profiles[agent_id] = {"name": profile_name, "source": "curriculum"}
        return self._build_profile_init_state(rng, base_state, profile_name)

    def get_profile_name(self, agent_id: str) -> str:
        return self._active_profiles.get(agent_id, {}).get("name", "cruise")

    def get_flight_metrics(self, env, agent_id: str) -> dict:
        aircraft = env.agents[agent_id]
        heading_deg = float(math.degrees(aircraft.get_property_value(c.attitude_heading_true_rad))) % 360.0
        roll_deg = float(math.degrees(aircraft.get_property_value(c.attitude_roll_rad)))
        pitch_deg = float(math.degrees(aircraft.get_property_value(c.attitude_pitch_rad)))
        altitude_m = float(aircraft.get_property_value(c.position_h_sl_m))
        vc_mps = float(aircraft.get_property_value(c.velocities_vc_mps))
        tas_mps = float(np.linalg.norm(aircraft.get_velocity()))
        v_up_mps = -float(aircraft.get_property_value(c.velocities_v_down_mps))
        aoa_deg = float(aircraft.get_property_value(c.aero_alpha_deg))
        return {
            "heading_deg": heading_deg,
            "roll_deg": roll_deg,
            "pitch_deg": pitch_deg,
            "altitude_m": altitude_m,
            "vc_mps": vc_mps,
            "tas_mps": tas_mps,
            "v_up_mps": v_up_mps,
            "aoa_deg": aoa_deg,
        }

    def _load_external_init_pool_from_env(self):
        path = os.getenv("CAP_LOWLEVEL_HARDCASE_JSON", "").strip()
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

            probability = float(os.getenv("CAP_LOWLEVEL_HARDCASE_PROB", "0.0"))
            self.set_external_init_pool(cases, probability)
            log.info(
                "[cap-lowlevel] loaded hardcases: path=%s cases=%d prob=%.2f",
                path,
                len(self._external_init_pool),
                self._external_init_prob,
            )
        except Exception as exc:
            log.warning("[cap-lowlevel] failed to load hardcases from %s: %s", path, exc)

    def _load_external_curriculum_pool_from_env(self):
        path = os.getenv("CAP_LOWLEVEL_CURRICULUM_JSON", "").strip()
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
                raise ValueError(f"Unsupported curriculum payload type: {type(payload)!r}")

            probability = float(os.getenv("CAP_LOWLEVEL_CURRICULUM_PROB", "1.0"))
            self.set_external_curriculum_pool(cases, probability)
            log.info(
                "[cap-lowlevel] loaded tactical curriculum: path=%s cases=%d prob=%.2f",
                path,
                len(self._external_curriculum_pool),
                self._external_curriculum_prob,
            )
        except Exception as exc:
            log.warning("[cap-lowlevel] failed to load tactical curriculum from %s: %s", path, exc)

    def _sample_profile_name(self, rng) -> str:
        mode = os.getenv("CAP_LOWLEVEL_PROFILE_MODE", self.profile_mode).strip().lower() or self.profile_mode
        weights = self.hardcase_profile_weights if mode == "hardcase" else self.foundation_profile_weights
        names = list(weights.keys())
        probs = np.asarray(list(weights.values()), dtype=np.float64)
        probs = probs / probs.sum()
        return str(rng.choice(names, p=probs))

    def _load_profile_weights(self, default_weights: Dict[str, float], prefix: str) -> Dict[str, float]:
        weights = {}
        for key, default_value in default_weights.items():
            config_key = f"{prefix}_profile_weight_{key}"
            value = float(getattr(self.config, config_key, default_value))
            if value > 0.0:
                weights[key] = value
        if not weights:
            weights = dict(default_weights)
        return weights

    def _get_outcome_reward(self, env, agent_id: str, info: dict) -> float:
        if not isinstance(info, dict):
            return 0.0
        reason = str(info.get(f"{agent_id}_termination_reason", "")).strip().lower()
        if not reason:
            return 0.0
        if reason == "timeout":
            return float(self.timeout_reward_bonus)
        if reason == "low_altitude":
            return -float(self.low_altitude_penalty)
        if reason == "extreme_state":
            return -float(self.extreme_state_penalty)
        if reason == "overload":
            return -float(self.overload_penalty)
        return -float(self.other_failure_penalty)

    def _build_case_init_state(self, base_state: dict, case: dict) -> dict:
        updated = dict(base_state)
        updated.update(
            {
                "ic_h_sl_ft": float(case.get("ic_h_sl_ft", updated.get("ic_h_sl_ft", 20000.0))),
                "ic_u_fps": float(case.get("ic_u_fps", updated.get("ic_u_fps", 650.0))),
                "ic_theta_deg": float(case.get("ic_theta_deg", updated.get("ic_theta_deg", 0.0))),
                "ic_phi_deg": float(case.get("ic_phi_deg", updated.get("ic_phi_deg", 0.0))),
                "ic_alpha_deg": float(case.get("ic_alpha_deg", updated.get("ic_alpha_deg", 0.0))),
                "ic_q_rad_sec": float(case.get("ic_q_rad_sec", 0.0)),
                "ic_p_rad_sec": float(case.get("ic_p_rad_sec", 0.0)),
                "ic_r_rad_sec": float(case.get("ic_r_rad_sec", 0.0)),
                "ic_roc_fpm": float(case.get("ic_roc_fpm", 0.0)),
                "ic_psi_true_deg": float(case.get("ic_psi_true_deg", updated.get("ic_psi_true_deg", 0.0))),
            }
        )
        return updated

    def _build_profile_init_state(self, rng, base_state: dict, profile_name: str) -> dict:
        updated = dict(base_state)
        heading_deg = float(rng.uniform(0.0, 360.0))

        if profile_name == "cruise":
            alt_ft = float(rng.uniform(18000.0, 32000.0))
            u_fps = float(rng.uniform(550.0, 900.0))
            theta_deg = float(rng.uniform(-3.0, 6.0))
            phi_deg = float(rng.uniform(-8.0, 8.0))
            alpha_deg = float(rng.uniform(-1.0, 5.0))
            roc_fpm = float(rng.uniform(-800.0, 800.0))
        elif profile_name == "turn":
            alt_ft = float(rng.uniform(15000.0, 28000.0))
            u_fps = float(rng.uniform(520.0, 880.0))
            theta_deg = float(rng.uniform(-2.0, 6.0))
            phi_deg = float(rng.uniform(-18.0, 18.0))
            alpha_deg = float(rng.uniform(0.0, 7.0))
            roc_fpm = float(rng.uniform(-1200.0, 1200.0))
        elif profile_name == "vertical":
            alt_ft = float(rng.uniform(9000.0, 26000.0))
            u_fps = float(rng.uniform(430.0, 780.0))
            theta_deg = float(rng.uniform(-8.0, 12.0))
            phi_deg = float(rng.uniform(-12.0, 12.0))
            alpha_deg = float(rng.uniform(2.0, 10.0))
            roc_fpm = float(rng.uniform(-5000.0, 3500.0))
        elif profile_name == "aggressive":
            alt_ft = float(rng.uniform(12000.0, 26000.0))
            u_fps = float(rng.uniform(420.0, 760.0))
            theta_deg = float(rng.uniform(-10.0, 14.0))
            phi_deg = float(rng.uniform(-30.0, 30.0))
            alpha_deg = float(rng.uniform(2.0, 12.0))
            roc_fpm = float(rng.uniform(-4500.0, 4500.0))
        elif profile_name == "recover_high":
            alt_ft = float(rng.uniform(18000.0, 32000.0))
            u_fps = float(rng.uniform(190.0, 420.0))
            theta_deg = float(rng.uniform(4.0, 16.0))
            phi_deg = float(rng.uniform(-15.0, 15.0))
            alpha_deg = float(rng.uniform(5.0, 14.0))
            roc_fpm = float(rng.uniform(-6000.0, 500.0))
        elif profile_name == "recover_mid":
            alt_ft = float(rng.uniform(3500.0, 14000.0))
            u_fps = float(rng.uniform(180.0, 360.0))
            theta_deg = float(rng.uniform(0.0, 14.0))
            phi_deg = float(rng.uniform(-15.0, 15.0))
            alpha_deg = float(rng.uniform(4.0, 12.0))
            roc_fpm = float(rng.uniform(-7000.0, 500.0))
        else:
            profile_name = "ground_escape"
            alt_ft = float(rng.uniform(1200.0, 3500.0))
            u_fps = float(rng.uniform(180.0, 320.0))
            theta_deg = float(rng.uniform(-18.0, 8.0))
            phi_deg = float(rng.uniform(-10.0, 10.0))
            alpha_deg = float(rng.uniform(2.0, 10.0))
            roc_fpm = float(rng.uniform(-7000.0, -500.0))

        updated.update(
            {
                "ic_psi_true_deg": heading_deg,
                "ic_h_sl_ft": alt_ft,
                "ic_u_fps": u_fps,
                "ic_theta_deg": theta_deg,
                "ic_phi_deg": phi_deg,
                "ic_alpha_deg": alpha_deg,
                "ic_q_rad_sec": 0.0,
                "ic_p_rad_sec": 0.0,
                "ic_r_rad_sec": 0.0,
                "ic_roc_fpm": roc_fpm,
            }
        )
        return updated

    def _activate_new_segment(self, env, agent_id: str, initial: bool):
        metrics = self.get_flight_metrics(env, agent_id)
        profile = self._active_profiles.get(agent_id, {"name": "cruise"})
        scripted_segment = self._pop_scripted_segment(profile, initial)
        if scripted_segment is None:
            alt_idx, hdg_idx, vel_idx = self._sample_command_indices(env, agent_id, profile, metrics, initial)
            hold_steps = self._estimate_hold_steps(env, alt_idx, hdg_idx, vel_idx, profile.get("name", "cruise"))
            segment_tag = ""
        else:
            alt_idx = int(scripted_segment.get("alt_cmd_idx", 7))
            hdg_idx = int(scripted_segment.get("hdg_cmd_idx", 8))
            vel_idx = int(scripted_segment.get("vel_cmd_idx", 3))
            hold_steps = int(
                np.clip(
                    int(scripted_segment.get("hold_steps", self.command_hold_min_steps)),
                    self.command_hold_min_steps,
                    self.command_hold_max_steps,
                )
            )
            segment_tag = str(scripted_segment.get("tag", ""))
        self._apply_segment_targets(env, agent_id, alt_idx, hdg_idx, vel_idx)
        self._segment_states[agent_id] = {
            "alt_idx": int(alt_idx),
            "hdg_idx": int(hdg_idx),
            "vel_idx": int(vel_idx),
            "hold_steps": int(hold_steps),
            "elapsed_steps": 0,
            "profile_name": str(profile.get("name", "cruise")),
            "segment_tag": segment_tag,
        }

        if self.command_trace:
            log.warning(
                "[cap-lowlevel-cmd] agent=%s profile=%s cmd=(%d,%d,%d) values=(%+.0fm,%+.0fdeg,%+.0fmps) hold=%d",
                agent_id,
                profile.get("name", "cruise"),
                alt_idx,
                hdg_idx,
                vel_idx,
                self.ALT_COMMAND_M[alt_idx],
                math.degrees(self.HDG_COMMAND_RAD[hdg_idx]),
                self.VEL_COMMAND_MPS[vel_idx],
                hold_steps,
            )

    def _pop_scripted_segment(self, profile: dict, initial: bool):
        sequence_segments = profile.get("sequence_segments")
        if not isinstance(sequence_segments, list) or not sequence_segments:
            return None

        cursor = int(profile.get("sequence_cursor", 0))
        if initial:
            cursor = 0
        if cursor >= len(sequence_segments):
            return None
        profile["sequence_cursor"] = cursor + 1
        return dict(sequence_segments[cursor])

    def _sample_command_indices(
        self,
        env,
        agent_id: str,
        profile: dict,
        metrics: dict,
        initial: bool,
    ) -> Tuple[int, int, int]:
        rng = env.np_random
        first_segment = profile.get("first_segment")
        if initial and isinstance(first_segment, dict) and first_segment:
            return (
                int(first_segment.get("alt_cmd_idx", 9)),
                int(first_segment.get("hdg_cmd_idx", self.HARDCODED_RECOVERY_HDG_INDEX)),
                int(first_segment.get("vel_cmd_idx", 6)),
            )

        altitude_m = metrics["altitude_m"]
        vc_mps = metrics["vc_mps"]
        sink_mps = metrics["v_up_mps"]

        severe_low_energy = vc_mps <= self.severe_low_energy_vc_mps or sink_mps <= -self.severe_sink_rate_mps
        low_energy = severe_low_energy or vc_mps <= self.low_energy_vc_mps or sink_mps <= -self.high_sink_rate_mps

        if altitude_m < 350.0:
            return 12, self.HARDCODED_RECOVERY_HDG_INDEX, 6
        if altitude_m < self.min_recovery_altitude_m and low_energy:
            return 11, self.HARDCODED_RECOVERY_HDG_INDEX, 6
        if low_energy or profile.get("name", "") in self.RECOVERY_PROFILES or str(profile.get("name", "")).startswith("cap_hardcase"):
            return self._sample_recovery_command(rng, metrics)
        if altitude_m >= self.energy_guard_hard_altitude_m and vc_mps <= self.energy_guard_hard_vc_mps:
            return 6, 8, 5
        if altitude_m >= self.energy_guard_altitude_m and vc_mps <= self.energy_guard_vc_mps:
            return self._sample_energy_guard_command(rng, metrics)
        if self.command_sampling_mode == "balanced":
            return self._sample_balanced_command(rng, profile.get("name", "cruise"), metrics)
        if self.command_sampling_mode == "energy_safe_balanced":
            return self._sample_balanced_command(rng, profile.get("name", "cruise"), metrics, energy_safe=True)

        profile_name = profile.get("name", "cruise")
        if profile_name == "cruise":
            return self._sample_from_candidates(rng, [6, 7, 8, 9], [7, 8, 9], [2, 3, 4])
        if profile_name == "turn":
            return self._sample_from_candidates(rng, [6, 7, 8, 9], [5, 6, 10, 11, 12], [3, 4, 5])
        if profile_name == "vertical":
            return self._sample_from_candidates(rng, [3, 4, 5, 9, 10, 11, 12], [7, 8, 9], [2, 3, 4, 5])
        if profile_name == "aggressive":
            return self._sample_from_candidates(rng, [2, 3, 5, 7, 9, 11, 12, 13], [3, 4, 5, 11, 12, 13, 14], [2, 3, 4, 5, 6])
        return self._sample_from_candidates(rng, [6, 7, 8, 9], [7, 8, 9], [3, 4, 5])

    def _sample_recovery_command(self, rng, metrics: dict) -> Tuple[int, int, int]:
        altitude_m = metrics["altitude_m"]
        vc_mps = metrics["vc_mps"]
        sink_mps = metrics["v_up_mps"]

        if altitude_m < 800.0:
            alt_candidates = [10, 11, 12, 13, 14]
        elif altitude_m < 1500.0:
            alt_candidates = [9, 10, 11, 12, 13]
        elif vc_mps < self.severe_low_energy_vc_mps or sink_mps < -self.severe_sink_rate_mps:
            alt_candidates = [7, 8, 9, 10]
        else:
            alt_candidates = [6, 7, 8, 9]

        hdg_candidates = [7, 8, 9]
        vel_candidates = [4, 5, 6] if vc_mps > 125.0 else [5, 6]
        return self._sample_from_candidates(rng, alt_candidates, hdg_candidates, vel_candidates)

    def _sample_energy_guard_command(self, rng, metrics: dict) -> Tuple[int, int, int]:
        altitude_m = float(metrics["altitude_m"])
        vc_mps = float(metrics["vc_mps"])
        sink_mps = float(metrics["v_up_mps"])

        if altitude_m >= self.energy_guard_hard_altitude_m or vc_mps <= self.energy_guard_hard_vc_mps:
            alt_candidates = [5, 6, 7]
            vel_candidates = [5, 6]
        elif sink_mps < -2.5:
            alt_candidates = [7, 8]
            vel_candidates = [4, 5, 6]
        else:
            alt_candidates = [6, 7]
            vel_candidates = [4, 5]
        hdg_candidates = [7, 8, 9]
        return self._sample_from_candidates(rng, alt_candidates, hdg_candidates, vel_candidates)

    def _sample_balanced_command(self, rng, profile_name: str, metrics: dict | None = None, energy_safe: bool = False) -> Tuple[int, int, int]:
        profile_name = str(profile_name or "cruise")

        if profile_name in {"turn", "aggressive", "cap_tactical"}:
            hdg_mode_probs = np.asarray([0.12, 0.44, 0.44], dtype=np.float64)
            alt_mode_probs = np.asarray([0.28, 0.36, 0.36], dtype=np.float64)
            vel_mode_probs = np.asarray([0.18, 0.41, 0.41], dtype=np.float64)
        elif profile_name == "vertical":
            hdg_mode_probs = np.asarray([0.25, 0.375, 0.375], dtype=np.float64)
            alt_mode_probs = np.asarray([0.12, 0.44, 0.44], dtype=np.float64)
            vel_mode_probs = np.asarray([0.18, 0.41, 0.41], dtype=np.float64)
        else:
            hdg_mode_probs = np.asarray([0.22, 0.39, 0.39], dtype=np.float64)
            alt_mode_probs = np.asarray([0.18, 0.41, 0.41], dtype=np.float64)
            vel_mode_probs = np.asarray([0.14, 0.43, 0.43], dtype=np.float64)

        if metrics is not None and energy_safe:
            altitude_m = float(metrics["altitude_m"])
            vc_mps = float(metrics["vc_mps"])
            sink_mps = float(metrics["v_up_mps"])
            if altitude_m >= self.energy_guard_altitude_m and vc_mps <= self.energy_guard_vc_mps + 10.0:
                alt_mode_probs = np.asarray([0.45, 0.10, 0.45], dtype=np.float64)
                hdg_mode_probs = np.asarray([0.45, 0.275, 0.275], dtype=np.float64)
                vel_mode_probs = np.asarray([0.18, 0.72, 0.10], dtype=np.float64)
            if altitude_m >= self.energy_guard_hard_altitude_m and vc_mps <= self.energy_guard_hard_vc_mps + 10.0:
                alt_mode_probs = np.asarray([0.50, 0.04, 0.46], dtype=np.float64)
                hdg_mode_probs = np.asarray([0.60, 0.20, 0.20], dtype=np.float64)
                vel_mode_probs = np.asarray([0.08, 0.84, 0.08], dtype=np.float64)
            if sink_mps < -4.0:
                alt_mode_probs = np.asarray([0.55, 0.20, 0.25], dtype=np.float64)
                vel_mode_probs = np.asarray([0.10, 0.72, 0.18], dtype=np.float64)

        alt_mode = str(rng.choice(np.asarray(["hold", "climb", "descend"]), p=alt_mode_probs / alt_mode_probs.sum()))
        hdg_mode = str(rng.choice(np.asarray(["hold", "left", "right"]), p=hdg_mode_probs / hdg_mode_probs.sum()))
        vel_mode = str(rng.choice(np.asarray(["hold", "accel", "decel"]), p=vel_mode_probs / vel_mode_probs.sum()))

        if alt_mode == "hold":
            alt_candidates = [7]
        elif alt_mode == "climb":
            alt_candidates = [8, 9, 10, 11, 12, 13, 14]
        else:
            alt_candidates = [0, 1, 2, 3, 4, 5, 6]

        if hdg_mode == "hold":
            hdg_candidates = [8]
        elif hdg_mode == "left":
            hdg_candidates = [7, 6, 5, 4, 3, 2, 1, 0]
        else:
            hdg_candidates = [9, 10, 11, 12, 13, 14, 15, 16]

        if vel_mode == "hold":
            vel_candidates = [3]
        elif vel_mode == "accel":
            vel_candidates = [4, 5, 6]
        else:
            vel_candidates = [0, 1, 2]

        return (
            int(rng.choice(np.asarray(alt_candidates, dtype=np.int64))),
            int(rng.choice(np.asarray(hdg_candidates, dtype=np.int64))),
            int(rng.choice(np.asarray(vel_candidates, dtype=np.int64))),
        )

    @staticmethod
    def _sample_from_candidates(rng, alt_candidates, hdg_candidates, vel_candidates) -> Tuple[int, int, int]:
        return (
            int(rng.choice(np.asarray(alt_candidates, dtype=np.int64))),
            int(rng.choice(np.asarray(hdg_candidates, dtype=np.int64))),
            int(rng.choice(np.asarray(vel_candidates, dtype=np.int64))),
        )

    def _should_apply_recovery_reward(self, env, agent_id: str) -> bool:
        profile_name = self.get_profile_name(agent_id)
        metrics = self.get_flight_metrics(env, agent_id)
        vc_mps = float(metrics["vc_mps"])
        sink_mps = float(metrics["v_up_mps"])
        altitude_m = float(metrics["altitude_m"])
        low_energy = (
            vc_mps <= self.low_energy_vc_mps
            or sink_mps <= -self.high_sink_rate_mps
            or altitude_m < self.min_recovery_altitude_m
        )
        return low_energy or profile_name in self.RECOVERY_PROFILES or str(profile_name).startswith("cap_hardcase")

    def _estimate_hold_steps(self, env, alt_idx: int, hdg_idx: int, vel_idx: int, profile_name: str) -> int:
        alt_delta_m = abs(float(self.ALT_COMMAND_M[alt_idx]))
        hdg_delta_deg = abs(float(math.degrees(self.HDG_COMMAND_RAD[hdg_idx])))
        vel_delta_mps = abs(float(self.VEL_COMMAND_MPS[vel_idx]))

        duration_s = 2.0 + hdg_delta_deg / 18.0 + alt_delta_m / 260.0 + vel_delta_mps / 32.0
        if profile_name in {"aggressive", "recover_high", "recover_mid", "ground_escape"}:
            duration_s += 1.0

        steps = int(round(duration_s / env.time_interval))
        return int(np.clip(steps, self.command_hold_min_steps, self.command_hold_max_steps))

    def _apply_segment_targets(self, env, agent_id: str, alt_idx: int, hdg_idx: int, vel_idx: int) -> None:
        aircraft = env.agents[agent_id]
        metrics = self.get_flight_metrics(env, agent_id)
        current_heading_deg = metrics["heading_deg"]
        current_altitude_ft = metrics["altitude_m"] * FT_PER_M
        current_vc_mps = metrics["vc_mps"]

        target_heading_deg = (current_heading_deg + math.degrees(self.HDG_COMMAND_RAD[hdg_idx])) % 360.0
        target_altitude_ft = current_altitude_ft + float(self.ALT_COMMAND_M[alt_idx] * FT_PER_M)
        target_speed_mps = current_vc_mps + float(self.VEL_COMMAND_MPS[vel_idx])

        target_altitude_ft = float(np.clip(target_altitude_ft, self.min_target_altitude_ft, self.max_target_altitude_ft))
        target_speed_mps = float(np.clip(target_speed_mps, self.min_target_speed_mps, self.max_target_speed_mps))

        aircraft.set_property_value(c.target_heading_deg, target_heading_deg)
        aircraft.set_property_value(c.target_altitude_ft, target_altitude_ft)
        aircraft.set_property_value(c.target_velocities_u_mps, target_speed_mps)
