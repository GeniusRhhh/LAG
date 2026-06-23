import math
from collections import defaultdict

import numpy as np

from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c


class CapTrackingReward(BaseRewardFunction):
    """Progress-oriented reward for CAP low-level command tracking.

    The old heading-task reward assumes tiny residuals and penalizes bank
    aggressively. CAP command segments regularly ask for sustained heading,
    altitude, and speed changes, so this reward focuses on:
    - reducing residual magnitude over time
    - rewarding arrival inside a reasonable tolerance band
    - avoiding explicit punishment on bank while the aircraft is still turning
    """

    def __init__(self, config):
        super().__init__(config)
        self.heading_progress_scale_deg = float(
            getattr(config, f"{self.__class__.__name__}_heading_progress_scale_deg", 18.0)
        )
        self.alt_progress_scale_m = float(
            getattr(config, f"{self.__class__.__name__}_alt_progress_scale_m", 240.0)
        )
        self.speed_progress_scale_mps = float(
            getattr(config, f"{self.__class__.__name__}_speed_progress_scale_mps", 24.0)
        )
        self.heading_tolerance_deg = float(
            getattr(config, f"{self.__class__.__name__}_heading_tolerance_deg", 4.0)
        )
        self.alt_tolerance_m = float(
            getattr(config, f"{self.__class__.__name__}_alt_tolerance_m", 120.0)
        )
        self.speed_tolerance_mps = float(
            getattr(config, f"{self.__class__.__name__}_speed_tolerance_mps", 10.0)
        )
        self.completion_bonus = float(
            getattr(config, f"{self.__class__.__name__}_completion_bonus", 0.25)
        )
        self.prev_errors = defaultdict(lambda: None)
        self.reward_item_names = [
            self.__class__.__name__,
            self.__class__.__name__ + "_heading_progress",
            self.__class__.__name__ + "_alt_progress",
            self.__class__.__name__ + "_speed_progress",
            self.__class__.__name__ + "_steady",
        ]

    def reset(self, task, env):
        super().reset(task, env)
        self.prev_errors.clear()

    def _normalized_progress(self, previous: float, current: float, scale: float) -> float:
        if previous is None:
            return 0.0
        return float(np.clip((previous - current) / max(scale, 1e-6), -1.0, 1.0))

    def get_reward(self, task, env, agent_id):
        heading_error_deg = abs(float(env.agents[agent_id].get_property_value(c.delta_heading)))
        alt_error_m = abs(float(env.agents[agent_id].get_property_value(c.delta_altitude)))
        speed_error_mps = abs(float(env.agents[agent_id].get_property_value(c.delta_velocities_u)))

        previous = self.prev_errors[agent_id]
        heading_progress = self._normalized_progress(
            None if previous is None else previous[0],
            heading_error_deg,
            self.heading_progress_scale_deg,
        )
        alt_progress = self._normalized_progress(
            None if previous is None else previous[1],
            alt_error_m,
            self.alt_progress_scale_m,
        )
        speed_progress = self._normalized_progress(
            None if previous is None else previous[2],
            speed_error_mps,
            self.speed_progress_scale_mps,
        )

        heading_track = 1.0 - np.clip(heading_error_deg / max(self.heading_progress_scale_deg * 2.0, 1.0), 0.0, 1.0)
        alt_track = 1.0 - np.clip(alt_error_m / max(self.alt_progress_scale_m * 2.0, 1.0), 0.0, 1.0)
        speed_track = 1.0 - np.clip(speed_error_mps / max(self.speed_progress_scale_mps * 2.0, 1.0), 0.0, 1.0)

        steady = 0.0
        if (
            heading_error_deg <= self.heading_tolerance_deg
            and alt_error_m <= self.alt_tolerance_m
            and speed_error_mps <= self.speed_tolerance_mps
        ):
            steady = self.completion_bonus

        reward = (
            0.30 * heading_progress
            + 0.20 * alt_progress
            + 0.15 * speed_progress
            + 0.20 * heading_track
            + 0.10 * alt_track
            + 0.05 * speed_track
            + steady
        )

        self.prev_errors[agent_id] = (heading_error_deg, alt_error_m, speed_error_mps)
        return self._process(reward, agent_id, (heading_progress, alt_progress, speed_progress, steady))
