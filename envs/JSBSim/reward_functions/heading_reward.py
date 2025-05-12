import math
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c
import numpy as np

class HeadingReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_heading', '_roll', '_speed']]

    def get_reward(self, task, env, agent_id):
        heading_error_scale = 30.0  # 减小尺度，提高敏感性
        heading_error = env.agents[agent_id].get_property_value(c.delta_heading)
        heading_r = math.exp(-(heading_error / heading_error_scale) ** 2)
        heading_r = 2.0 * heading_r - 1.0

        roll_error_scale = 0.7
        roll_error = env.agents[agent_id].get_property_value(c.attitude_roll_rad)
        roll_r = math.exp(-(roll_error / roll_error_scale) ** 2)
        roll_r = 2.0 * roll_r - 1.0

        speed_error_scale = 50.0
        speed_error = env.agents[agent_id].get_property_value(c.delta_velocities_u)
        speed_r = math.exp(-(speed_error / speed_error_scale) ** 2)
        speed_r = 2.0 * speed_r - 1.0

        reward = 0.6 * heading_r + 0.2 * roll_r + 0.2 * speed_r  # 提高航向权重
        if abs(heading_error) < 10.0:
            reward += 0.5  # 阶段性正向奖励
        reward = np.clip(reward, -2.0, 2.0)
        return self._process(reward, agent_id, (heading_r, roll_r, speed_r))