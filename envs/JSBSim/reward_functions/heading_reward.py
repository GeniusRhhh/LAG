# reward_functions/heading_reward.py
import math
import numpy as np
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c

class HeadingReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_heading', '_alt', '_roll', '_speed']]

    def get_reward(self, task, env, agent_id):
        delta_heading = env.agents[agent_id].get_property_value(c.delta_heading)
        delta_altitude = env.agents[agent_id].get_property_value(c.delta_altitude)
        roll_rad = env.agents[agent_id].get_property_value(c.attitude_roll_rad)
        delta_speed = env.agents[agent_id].get_property_value(c.delta_velocities_u)

        # 放宽高斯尺度
        heading_r = math.exp(-((delta_heading / 10.0) ** 2))  # 5.0 → 10.0
        alt_r = math.exp(-((delta_altitude / 15.24) ** 2))
        roll_r = math.exp(-((roll_rad / 0.35) ** 2))
        speed_r = math.exp(-((delta_speed / 24.0) ** 2))

        # 添加正奖励项
        heading_bonus = 0.5 * (1.0 - abs(delta_heading) / 180.0)  # 航向误差越小，奖励越高
        reward = (heading_r * alt_r * roll_r * speed_r) ** (1 / 4) + heading_bonus

        return self._process(reward, agent_id, (heading_r, alt_r, roll_r, speed_r))