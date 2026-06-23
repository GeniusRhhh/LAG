import numpy as np
from .reward_function_base import BaseRewardFunction


class MissilePostureReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.previous_missile_v = {}

    def reset(self, task, env):
        self.previous_missile_v = {}
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        reward = 0
        missiles = env.agents[agent_id].check_missile_warning(multi=True)  # 获取所有威胁导弹
        if missiles:  # missiles 现在是列表
            for missile in missiles:
                missile_v = missile.get_velocity()
                aircraft_v = env.agents[agent_id].get_velocity()
                if missile.uid not in self.previous_missile_v:
                    self.previous_missile_v[missile.uid] = missile_v
                v_decrease = (np.linalg.norm(self.previous_missile_v[missile.uid]) - np.linalg.norm(missile_v)) / 340
                angle = np.dot(missile_v, aircraft_v) / (np.linalg.norm(missile_v) * np.linalg.norm(aircraft_v) + 1e-6)
                reward += -0.1 * angle + 0.2 * max(v_decrease, 0)  # 奖励逻辑保持不变
                self.previous_missile_v[missile.uid] = missile_v
        else:
            self.previous_missile_v = {}  # 清空记录
        self.reward_trajectory[agent_id].append([reward])
        return reward