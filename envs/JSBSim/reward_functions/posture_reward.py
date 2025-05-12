import numpy as np
from wandb import agent
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R


class PostureReward(BaseRewardFunction):
    """
    姿态奖励 (PostureReward)
    - 方向奖励（Orientation Reward）：鼓励智能体朝向敌方战机，并惩罚被敌方战机指向的情况。
    - 距离奖励（Range Reward）：鼓励智能体靠近敌方战机，同时避免距离过远或过近。

    注意：
    - 仅支持一对一环境（One-to-One Environments）。
    """

    def __init__(self, config):
        super().__init__(config)
        # 配置中获取方向奖励和距离奖励的版本
        self.orientation_version = getattr(self.config, f'{self.__class__.__name__}_orientation_version', 'v2')
        self.range_version = getattr(self.config, f'{self.__class__.__name__}_range_version', 'v3')
        self.target_dist = getattr(self.config, f'{self.__class__.__name__}_target_dist', 3.0)  # 目标距离，单位：km

        # 根据版本号选择奖励计算函数
        self.orientation_fn = self.get_orientation_function(self.orientation_version)
        self.range_fn = self.get_range_funtion(self.range_version)

        # 奖励项的名称列表
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_orn', '_range']]

    def get_reward(self, task, env, agent_id):
        """
        计算智能体的总奖励，奖励由方向奖励和距离奖励的乘积构成。

        Args:
            task: 任务实例
            env: 环境实例
            agent_id: 智能体的唯一标识符

        Returns:
            (float): 奖励值
        """
        new_reward = 0  # 初始化奖励为 0

        # 获取智能体的特征，包括位置和速度
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])

        # 遍历所有敌方战机
        for enm in env.agents[agent_id].enemies:
            # 获取敌方战机的特征
            enm_feature = np.hstack([enm.get_position(),
                                     enm.get_velocity()])

            # 计算智能体和敌方战机的 AO（角度）、TA（目标角度）和 R（距离）
            AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)

            # 根据 AO 和 TA 计算方向奖励
            orientation_reward = self.orientation_fn(AO, TA)

            # 根据距离 R 计算距离奖励
            range_reward = self.range_fn(R / 1000)  # 将距离从米转换为千米

            # 累加总奖励
            new_reward += orientation_reward * range_reward

        # 对奖励进行后处理，并返回最终奖励值
        return self._process(new_reward, agent_id, (orientation_reward, range_reward))

    def get_orientation_function(self, version):
        """
        获取方向奖励函数，根据指定的版本号返回不同的函数实现。

        Args:
            version: 方向奖励函数的版本号

        Returns:
            (function): 方向奖励计算函数
        """
        if version == 'v0':
            # 版本 v0 的方向奖励函数
            return lambda AO, TA: (1. - np.tanh(9 * (AO - np.pi / 9))) / 3. + 1 / 3. \
                + min((np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5
        elif version == 'v1':
            # 版本 v1 的方向奖励函数
            return lambda AO, TA: (1. - np.tanh(2 * (AO - np.pi / 2))) / 2. \
                * (np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi) + 0.5
        elif version == 'v2':
            # 版本 v2 的方向奖励函数（默认版本）
            return lambda AO, TA: 1 / (50 * AO / np.pi + 2) + 1 / 2 \
                + min((np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5
        else:
            # 如果版本号未知，抛出异常
            raise NotImplementedError(f"未知的方向奖励函数版本：{version}")

    def get_range_funtion(self, version):
        """
        获取距离奖励函数，根据指定的版本号返回不同的函数实现。

        Args:
            version: 距离奖励函数的版本号

        Returns:
            (function): 距离奖励计算函数
        """
        if version == 'v0':
            # 版本 v0 的距离奖励函数
            return lambda R: np.exp(-(R - self.target_dist) ** 2 * 0.004) / (1. + np.exp(-(R - self.target_dist + 2) * 2))
        elif version == 'v1':
            # 版本 v1 的距离奖励函数
            return lambda R: np.clip(1.2 * np.min([np.exp(-(R - self.target_dist) * 0.21), 1]) /
                                     (1. + np.exp(-(R - self.target_dist + 1) * 0.8)), 0.3, 1)
        elif version == 'v2':
            # 版本 v2 的距离奖励函数
            return lambda R: max(np.clip(1.2 * np.min([np.exp(-(R - self.target_dist) * 0.21), 1]) /
                                         (1. + np.exp(-(R - self.target_dist + 1) * 0.8)), 0.3, 1), np.sign(7 - R))
        elif version == 'v3':
            # 版本 v3 的距离奖励函数（默认版本）
            return lambda R: 1 * (R < 5) + (R >= 5) * np.clip(-0.032 * R**2 + 0.284 * R + 0.38, 0, 1) + np.clip(np.exp(-0.16 * R), 0, 0.2)
        else:
            # 如果版本号未知，抛出异常
            raise NotImplementedError(f"未知的距离奖励函数版本：{version}")
