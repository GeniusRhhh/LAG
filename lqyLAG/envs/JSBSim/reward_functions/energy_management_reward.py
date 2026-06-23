import math
from collections import defaultdict

import numpy as np

from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c


class EnergyManagementReward(BaseRewardFunction):
    """Always-on reward for proactive energy preservation."""

    def __init__(self, config):
        super().__init__(config)
        self.base_safe_vc_mps = float(getattr(config, f"{self.__class__.__name__}_base_safe_vc_mps", 190.0))
        self.high_altitude_m = float(getattr(config, f"{self.__class__.__name__}_high_altitude_m", 8000.0))
        self.very_high_altitude_m = float(getattr(config, f"{self.__class__.__name__}_very_high_altitude_m", 10500.0))
        self.high_alt_speed_margin_mps = float(
            getattr(config, f"{self.__class__.__name__}_high_alt_speed_margin_mps", 20.0)
        )
        self.very_high_alt_speed_margin_mps = float(
            getattr(config, f"{self.__class__.__name__}_very_high_alt_speed_margin_mps", 35.0)
        )
        self.speed_margin_band_mps = float(getattr(config, f"{self.__class__.__name__}_speed_margin_band_mps", 35.0))
        self.max_sink_mps = float(getattr(config, f"{self.__class__.__name__}_max_sink_mps", 10.0))
        self.safe_aoa_deg = float(getattr(config, f"{self.__class__.__name__}_safe_aoa_deg", 10.0))
        self.bad_aoa_deg = float(getattr(config, f"{self.__class__.__name__}_bad_aoa_deg", 18.0))
        self.climb_drain_pitch_deg = float(getattr(config, f"{self.__class__.__name__}_climb_drain_pitch_deg", 4.0))
        self.climb_drain_vup_mps = float(getattr(config, f"{self.__class__.__name__}_climb_drain_vup_mps", 2.0))
        self.climb_drain_penalty = float(getattr(config, f"{self.__class__.__name__}_climb_drain_penalty", 0.35))
        self.esdot_scale = float(getattr(config, f"{self.__class__.__name__}_esdot_scale", 800.0))
        self.prev_vc = defaultdict(lambda: None)
        self.reward_item_names = [
            self.__class__.__name__,
            self.__class__.__name__ + "_speed_margin",
            self.__class__.__name__ + "_esdot",
            self.__class__.__name__ + "_aoa",
            self.__class__.__name__ + "_sink",
            self.__class__.__name__ + "_climb_drain",
        ]

    def reset(self, task, env):
        super().reset(task, env)
        self.prev_vc.clear()

    def _safe_speed_floor(self, altitude_m: float) -> float:
        floor = self.base_safe_vc_mps
        if altitude_m >= self.high_altitude_m:
            floor += self.high_alt_speed_margin_mps
        if altitude_m >= self.very_high_altitude_m:
            floor += self.very_high_alt_speed_margin_mps
        return floor

    def get_reward(self, task, env, agent_id):
        aircraft = env.agents[agent_id]
        altitude_m = float(aircraft.get_property_value(c.position_h_sl_m))
        vc_mps = float(aircraft.get_property_value(c.velocities_vc_mps))
        pitch_deg = float(math.degrees(aircraft.get_property_value(c.attitude_pitch_rad)))
        aoa_deg = abs(float(aircraft.get_property_value(c.aero_alpha_deg)))
        v_up_mps = -float(aircraft.get_property_value(c.velocities_v_down_mps))

        safe_floor = self._safe_speed_floor(altitude_m)
        speed_margin_r = np.clip((vc_mps - safe_floor) / max(self.speed_margin_band_mps, 1.0), -1.0, 1.0)

        prev_vc = self.prev_vc[agent_id]
        if prev_vc is None:
            dv_dt = 0.0
        else:
            dt = max(float(getattr(env, "time_interval", 0.2)), 1e-3)
            dv_dt = (vc_mps - float(prev_vc)) / dt
        self.prev_vc[agent_id] = vc_mps

        es_dot = 9.81 * v_up_mps + vc_mps * dv_dt
        esdot_r = np.clip(es_dot / max(self.esdot_scale, 1.0), -1.0, 1.0)

        aoa_r = 1.0 - np.clip((aoa_deg - self.safe_aoa_deg) / max(self.bad_aoa_deg - self.safe_aoa_deg, 1.0), 0.0, 1.0)
        sink_r = 1.0 if v_up_mps >= 0.0 else 1.0 - np.clip(abs(v_up_mps) / max(self.max_sink_mps, 1.0), 0.0, 1.0)

        climb_drain = (
            altitude_m >= self.high_altitude_m
            and pitch_deg >= self.climb_drain_pitch_deg
            and v_up_mps >= self.climb_drain_vup_mps
            and vc_mps < safe_floor + 10.0
            and dv_dt < -0.5
        )
        climb_drain_term = -self.climb_drain_penalty if climb_drain else 0.0

        reward = (
            0.40 * speed_margin_r
            + 0.25 * esdot_r
            + 0.15 * aoa_r
            + 0.10 * sink_r
            + climb_drain_term
        )
        return self._process(reward, agent_id, (speed_margin_r, esdot_r, aoa_r, sink_r, climb_drain_term))
