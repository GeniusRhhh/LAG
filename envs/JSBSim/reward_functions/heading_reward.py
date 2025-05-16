import math
import numpy as np
import logging
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c

class HeadingReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_heading', '_alt', '_roll', '_pitch', '_speed', '_heading_bonus']]
        self.normalize_reward = getattr(config, 'normalize_reward', False)
        self.reward_buffer = []
        self.buffer_size = 1000

    def get_reward(self, task, env, agent_id):
        delta_heading = env.agents[agent_id].get_property_value(c.delta_heading)
        delta_altitude = env.agents[agent_id].get_property_value(c.delta_altitude)
        roll_rad = env.agents[agent_id].get_property_value(c.attitude_roll_rad)
        pitch_rad = env.agents[agent_id].get_property_value(c.attitude_pitch_rad)
        delta_speed = env.agents[agent_id].get_property_value(c.delta_velocities_u)

        heading_r = math.exp(-((delta_heading / 30.0) ** 2))  # 放宽至 30 度
        alt_r = math.exp(-((delta_altitude / 10.0) ** 2))
        roll_r = math.exp(-((roll_rad / 0.35) ** 2))
        pitch_r = math.exp(-((pitch_rad / 0.35) ** 2))
        speed_r = math.exp(-((delta_speed / 24.0) ** 2))
        heading_bonus = 0.5 * (1.0 - abs(delta_heading) / 180.0)

        # 增强姿态惩罚
        pitch_penalty = -0.5 if abs(pitch_rad) > 0.35 else 0.0
        roll_penalty = -0.5 if abs(roll_rad) > 0.35 else 0.0

        reward = (heading_r * alt_r * roll_r * pitch_r * speed_r) ** (1 / 5) + heading_bonus + pitch_penalty + roll_penalty

        if task.step_count % 500 == 0:
            logging.info(
                f"Agent {agent_id} HeadingReward: total={reward:.4f}, delta_heading={delta_heading:.2f}°, delta_altitude={delta_altitude:.2f}m"
            )
        return self._process(reward, agent_id, (heading_r, alt_r, roll_r, pitch_r, speed_r, heading_bonus))