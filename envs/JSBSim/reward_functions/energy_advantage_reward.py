import numpy as np
from .reward_function_base import BaseRewardFunction

class EnergyAdvantageReward(BaseRewardFunction):
    """
    EnergyAdvantageReward:
    奖励智能体在能量态势（高度+速度）上相对敌机占优。
    简化模型：
    E_ego = ego_z + k * ego_speed
    E_enm = enm_z + k * enm_speed
    若E_ego > E_enm则奖励，差距越大奖励越高，但设上限防止极端情况。

    配置参数：
    - EnergyAdvantageReward_k : 速度权重
    - EnergyAdvantageReward_max : 最大奖励
    """
    def __init__(self, config):
        super().__init__(config)
        self.k = getattr(self.config, f'{self.__class__.__name__}_k', 0.02)
        self.max_reward = getattr(self.config, f'{self.__class__.__name__}_max', 0.5)

    def get_reward(self, task, env, agent_id):
        ego_pos = env.agents[agent_id].get_position()
        enm_pos = env.agents[agent_id].enemies[0].get_position()
        ego_z = ego_pos[-1]/1000
        enm_z = enm_pos[-1]/1000

        ego_v = np.linalg.norm(env.agents[agent_id].get_velocity())/340
        enm_v = np.linalg.norm(env.agents[agent_id].enemies[0].get_velocity())/340

        E_ego = ego_z + self.k * ego_v
        E_enm = enm_z + self.k * enm_v

        diff = E_ego - E_enm
        # diff正表示能量优势，加分，负表示劣势给0或小惩罚
        reward = np.clip(diff, -0.2, self.max_reward)
        return self._process(reward, agent_id)
