import numpy as np
from .reward_function_base import BaseRewardFunction


class RelativeAltitudeReward(BaseRewardFunction):
    """
    RelativeAltitudeReward
    Punish if current fighter doesn't satisfy some constraints. Typically negative.
    - Punishment of relative altitude when larger than 1000  (range: [-1, 0])

    NOTE:
    - Only support one-to-one environments.
    """
    def __init__(self, config):
        super().__init__(config)
        self.KH = getattr(self.config, f'{self.__class__.__name__}_KH', 1.0)     # km

    def get_reward(self, task, env, agent_id):
        """
        根据智能体与敌机的相对高度计算奖励。

        Args:
            task: 当前任务实例
            env: 环境实例，提供智能体和敌机的状态信息
            agent_id: 智能体的唯一标识符

        Returns:
            (float): 奖励值
        """
        # 获取智能体的高度（单位：千米）
        ego_z = env.agents[agent_id].get_position()[-1] / 1000
        # 获取敌机的高度（单位：千米）
        enm_z = env.agents[agent_id].enemies[0].get_position()[-1] / 1000
        # 计算相对高度惩罚
        new_reward = min(self.KH - np.abs(ego_z - enm_z), 0)
        # 返回处理后的奖励值
        return self._process(new_reward, agent_id)
