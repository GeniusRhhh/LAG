# reward_functions/heading_reward.py
import math
import numpy as np
import logging
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c

class HeadingReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_heading', '_alt', '_roll', '_speed', '_heading_bonus']]
        self.normalize_reward = getattr(config, 'normalize_reward', False)
        self.reward_buffer = []
        self.buffer_size = 1000

    def get_reward(self, task, env, agent_id):
        delta_heading = env.agents[agent_id].get_property_value(c.delta_heading)
        delta_altitude = env.agents[agent_id].get_property_value(c.delta_altitude)
        roll_rad = env.agents[agent_id].get_property_value(c.attitude_roll_rad)
        delta_speed = env.agents[agent_id].get_property_value(c.delta_velocities_u)  # 替换为 delta_velocities_u

        # 保持原高斯尺度（不放宽，避免复杂化）
        heading_r = math.exp(-((delta_heading / 10.0) ** 2))
        alt_r = math.exp(-((delta_altitude / 15.24) ** 2))
        roll_r = math.exp(-((roll_rad / 0.35) ** 2))
        speed_r = math.exp(-((delta_speed / 24.0) ** 2))

        # 保持原权重
        heading_bonus = 0.5 * (1.0 - abs(delta_heading) / 180.0)

        # 计算总奖励
        reward = (heading_r * alt_r * roll_r * speed_r) ** (1 / 4) + heading_bonus

        # 奖励归一化（若启用）
        if self.normalize_reward:
            self.reward_buffer.append(reward)
            if len(self.reward_buffer) > self.buffer_size:
                self.reward_buffer.pop(0)
            reward_mean = np.mean(self.reward_buffer)
            reward_std = np.std(self.reward_buffer) + 1e-6
            reward = (reward - reward_mean) / reward_std

        # 简单日志
        if task.step_count % 500 == 0:
            logging.info(
                f"Agent {agent_id} HeadingReward: total={reward:.4f}, "
                f"delta_heading={delta_heading:.2f}°, delta_speed={delta_speed:.2f}m/s"
            )

        return self._process(reward, agent_id, (heading_r, alt_r, roll_r, speed_r, heading_bonus))