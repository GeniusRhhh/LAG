import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict


class BaseRewardFunction(ABC):
    """
    Base RewardFunction class
    Reward-specific reset and get_reward methods are implemented in subclasses
    """

    def __init__(self, config):
        self.config = config
        # 奖励缩放因子，默认为 1.0
        self.reward_scale = getattr(self.config, f'{self.__class__.__name__}_scale', 1.0)
        # 是否启用潜在奖励，默认为 False
        self.is_potential = getattr(self.config, f'{self.__class__.__name__}_potential', False)
        # 保存上一时间步的奖励值，用于潜在奖励计算
        self.pre_rewards = defaultdict(float)
        # 保存奖励轨迹，每个智能体的奖励以列表形式存储
        self.reward_trajectory = defaultdict(list)
        # 奖励项名称列表，子类可以扩展此列表
        self.reward_item_names = [self.__class__.__name__]

    def reset(self, task, env):
        """重置奖励函数内部状态，在每个回合开始时调用。

        Args:
            task: 当前任务实例
            env: 环境实例
        """
        if self.is_potential:
            # 清空上一时间步奖励记录
            self.pre_rewards.clear()
            for agent_id in env.agents.keys():
                # 初始化每个智能体的潜在奖励值
                self.pre_rewards[agent_id] = self.get_reward(task, env, agent_id)
        # 清空奖励轨迹
        self.reward_trajectory.clear()

    @abstractmethod
    def get_reward(self, task, env, agent_id):
        """计算当前时间步的奖励。

        Args:
            task: 当前任务实例
            env: 环境实例
            agent_id: 智能体的唯一标识符

        Returns:
            (float): 奖励值
        """
        raise NotImplementedError

    def _process(self, new_reward, agent_id, render_items=()):
        """处理奖励值，包括潜在奖励计算和奖励轨迹记录。

        Args:
            new_reward (float): 当前时间步的奖励值
            agent_id (str): 智能体的唯一标识符
            render_items (tuple, optional): 额外的奖励项信息，默认为空

        Returns:
            (float): 处理后的奖励值
        """
        # 按缩放因子调整奖励值
        reward = new_reward * self.reward_scale
        if self.is_potential:
            # 计算潜在奖励
            reward, self.pre_rewards[agent_id] = reward - self.pre_rewards[agent_id], reward
        # 记录奖励轨迹，包括额外的奖励项
        self.reward_trajectory[agent_id].append([reward, *render_items])
        return reward

    def get_reward_trajectory(self):
        """获取当前回合所有智能体的奖励轨迹。

        Returns:
            (dict): 奖励轨迹的字典，键为奖励项名称，值为奖励轨迹的 numpy 数组。
        """
        return dict(zip(self.reward_item_names, np.array(self.reward_trajectory.values()).transpose(2, 0, 1)))
