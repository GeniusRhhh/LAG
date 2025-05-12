import numpy as np
from .reward_function_base import BaseRewardFunction

class RelativeAltitudeReward(BaseRewardFunction):
    """
    RelativeAltitudeReward（改进版）
    新增点：
    - 定义理想相对高度区间。例如希望比敌机高0.5km上下0.3km范围是理想区间
    - 超出区间则惩罚或减少奖励
    """
    def __init__(self, config):
        super().__init__(config)
        self.KH = getattr(self.config, f'{self.__class__.__name__}_KH', 1.0) # 基准奖励缩放
        self.ideal_diff = getattr(self.config, f'{self.__class__.__name__}_ideal_diff', 0.5) # 理想差值
        self.diff_tolerance = getattr(self.config, f'{self.__class__.__name__}_diff_tolerance', 0.3)

    def get_reward(self, task, env, agent_id):
        ego_z = env.agents[agent_id].get_position()[-1] / 1000
        enm_z = env.agents[agent_id].enemies[0].get_position()[-1] / 1000

        diff = ego_z - enm_z
        # 若在ideal_diff±diff_tolerance内则奖励较高，否则随偏差递减
        if abs(diff - self.ideal_diff) <= self.diff_tolerance:
            # 完全匹配ideal_diff则奖励最高0.5，否则线性递减
            rel = 0.5 - (abs(diff - self.ideal_diff)/self.diff_tolerance)*0.5
        else:
            # 超出理想区间则奖励减少甚至为负
            rel = -min(abs(diff - self.ideal_diff)-self.diff_tolerance,1.0)

        new_reward = self.KH * rel
        return self._process(new_reward, agent_id)
