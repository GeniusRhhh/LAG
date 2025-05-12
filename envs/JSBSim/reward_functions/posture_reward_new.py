import numpy as np
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R

class PostureReward(BaseRewardFunction):
    """
    PostureReward（改进版）
    新增点：
    - 可配置对AO、TA、R的权重单独调整
    - 增加相对角速度（AO变化率）作为稳定性参考，AO变化过大减少奖励
    - 当接近理想R和AO时给予更高奖励，加大梯度吸引策略向优姿态靠拢
    """
    def __init__(self, config):
        super().__init__(config)
        self.orientation_version = getattr(self.config, f'{self.__class__.__name__}_orientation_version', 'v2')
        self.range_version = getattr(self.config, f'{self.__class__.__name__}_range_version', 'v3')
        self.target_dist = getattr(self.config, f'{self.__class__.__name__}_target_dist', 3.0)
        self.AO_weight = getattr(self.config, f'{self.__class__.__name__}_AO_weight', 1.0)
        self.TA_weight = getattr(self.config, f'{self.__class__.__name__}_TA_weight', 1.0)
        self.R_weight = getattr(self.config, f'{self.__class__.__name__}_R_weight', 1.0)
        self.AO_change_penalty = getattr(self.config, f'{self.__class__.__name__}_AO_change_penalty', 0.1)

        self.orientation_fn = self.get_orientation_function(self.orientation_version)
        self.range_fn = self.get_range_funtion(self.range_version)

        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_orn', '_range','_AOchange']]
        self.prev_AO = {}

    def reset(self, task, env):
        self.prev_AO.clear()
        return super().reset(task, env)
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
    def get_reward(self, task, env, agent_id):
        new_reward = 0
        agent = env.agents[agent_id]

        ego_feature = np.hstack([agent.get_position(), agent.get_velocity()])
        for enm in agent.enemies:
            enm_feature = np.hstack([enm.get_position(), enm.get_velocity()])
            AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)

            orientation_reward = self.orientation_fn(AO, TA)
            range_reward = self.range_fn(R / 1000)

            # 根据AO变化率进行微调：若AO变化剧烈则减小奖励，鼓励平稳接近对手尾部
            prev_ao = self.prev_AO.get(agent_id, AO)
            AO_diff = abs(AO - prev_ao)
            AO_stability = np.exp(-self.AO_change_penalty * AO_diff)

            self.prev_AO[agent_id] = AO

            # 综合考虑权重和AO变化率
            combined = (orientation_reward**self.AO_weight) * (range_reward**self.R_weight) * AO_stability
            # TA用于微调，可对TA不理想时减分
            # 假设TA>某值时减少奖励
            if TA > np.pi/2: # TA过大意味着位置不理想
                combined *= (1 - self.TA_weight * (TA - np.pi/2)/np.pi)

            new_reward += combined

        return self._process(new_reward, agent_id, (orientation_reward, range_reward, AO_stability))

