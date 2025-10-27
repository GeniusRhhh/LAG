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
        delta_heading = abs(env.agents[agent_id].get_property_value(c.delta_heading))
        delta_altitude = env.agents[agent_id].get_property_value(c.delta_altitude)
        roll_rad = abs(env.agents[agent_id].get_property_value(c.attitude_roll_rad))
        pitch_rad = abs(env.agents[agent_id].get_property_value(c.attitude_pitch_rad))
        delta_speed = abs(env.agents[agent_id].get_property_value(c.delta_velocities_u))

        # F-16风格：指数奖励，始终为正，收紧容忍度
        heading_r = math.exp(-((delta_heading / 15.0) ** 2))  # 从30度收紧到15度
        alt_r = math.exp(-((delta_altitude / 300.0) ** 2))    # 高度次要
        roll_r = math.exp(-((roll_rad / 0.25) ** 2))          # 从0.35收紧到0.25
        pitch_r = math.exp(-((pitch_rad / 0.25) ** 2))        # 从0.35收紧到0.25
        speed_r = math.exp(-((delta_speed / 15.0) ** 2))      # 从24收紧到15
        
        # 几何平均，保持奖励在[0,1]范围
        reward = (heading_r ** 2 * alt_r * roll_r * pitch_r * speed_r) ** (1/6)
        
        # 航向精确奖励（小幅增强）
        if delta_heading < 5.0:
            reward += 0.2 * (1.0 - delta_heading / 5.0)  # 最多+0.2

        if task.step_count % 500 == 0:
            logging.info(
                f"Agent {agent_id} HeadingReward: total={reward:.4f}, delta_heading={delta_heading:.2f}°, heading_r={heading_r:.4f}, alt_r={alt_r:.4f}, roll_r={roll_r:.4f}"
            )
        return self._process(reward, agent_id, (heading_r, alt_r, roll_r, pitch_r, speed_r, 0))