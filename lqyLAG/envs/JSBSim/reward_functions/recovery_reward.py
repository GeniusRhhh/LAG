import math
import numpy as np

from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c


class RecoveryReward(BaseRewardFunction):
    """Reward for low-energy recovery while preserving basic command tracking.

    This reward is intentionally state-centric:
    - recover calibrated airspeed first
    - reduce excessive AoA
    - suppress large sink rate
    - avoid steep bank / over-rotation
    - reward entering a stable, flyable envelope
    """

    def __init__(self, config):
        super().__init__(config)
        self.target_vc_mps = float(getattr(config, f"{self.__class__.__name__}_target_vc_mps", 145.0))
        self.min_vc_mps = float(getattr(config, f"{self.__class__.__name__}_min_vc_mps", 75.0))
        self.safe_aoa_deg = float(getattr(config, f"{self.__class__.__name__}_safe_aoa_deg", 10.0))
        self.bad_aoa_deg = float(getattr(config, f"{self.__class__.__name__}_bad_aoa_deg", 22.0))
        self.max_sink_mps = float(getattr(config, f"{self.__class__.__name__}_max_sink_mps", 18.0))
        self.safe_roll_deg = float(getattr(config, f"{self.__class__.__name__}_safe_roll_deg", 35.0))
        self.reward_item_names = [
            self.__class__.__name__,
            self.__class__.__name__ + "_vc",
            self.__class__.__name__ + "_aoa",
            self.__class__.__name__ + "_sink",
            self.__class__.__name__ + "_roll",
            self.__class__.__name__ + "_stable",
        ]

    def get_reward(self, task, env, agent_id):
        vc = float(env.agents[agent_id].get_property_value(c.velocities_vc_mps))
        aoa = abs(float(env.agents[agent_id].get_property_value(c.aero_alpha_deg)))
        roll_deg = abs(math.degrees(float(env.agents[agent_id].get_property_value(c.attitude_phi_rad))))
        v_up_mps = float(env.agents[agent_id].get_velocity()[-1])

        vc_r = np.clip((vc - self.min_vc_mps) / max(self.target_vc_mps - self.min_vc_mps, 1.0), -1.0, 1.0)
        aoa_r = 1.0 - np.clip((aoa - self.safe_aoa_deg) / max(self.bad_aoa_deg - self.safe_aoa_deg, 1.0), 0.0, 1.0)
        sink_r = 1.0 if v_up_mps >= 0.0 else 1.0 - np.clip(abs(v_up_mps) / max(self.max_sink_mps, 1.0), 0.0, 1.0)
        roll_r = 1.0 - np.clip(roll_deg / max(self.safe_roll_deg, 1.0), 0.0, 1.0)

        stable = (
            vc >= self.target_vc_mps
            and aoa <= self.safe_aoa_deg
            and v_up_mps >= -2.0
            and roll_deg <= 20.0
        )
        stable_bonus = 1.0 if stable else 0.0

        reward = (
            0.40 * vc_r
            + 0.20 * aoa_r
            + 0.20 * sink_r
            + 0.10 * roll_r
            + 0.10 * stable_bonus
        )
        return self._process(reward, agent_id, (vc_r, aoa_r, sink_r, roll_r, stable_bonus))
