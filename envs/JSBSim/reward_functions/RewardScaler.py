import numpy as np

class RewardScaler:
    """奖励缩放器，防止奖励爆炸"""

    def __init__(self, scale_factor=0.01):
        self.scale_factor = scale_factor
        self.running_mean = 0.0
        self.running_std = 1.0
        self.count = 0

    def update_stats(self, rewards):
        """更新运行统计"""
        self.count += 1
        delta = rewards.mean() - self.running_mean
        self.running_mean += delta / self.count
        self.running_std = np.sqrt(self.running_std ** 2 + delta ** 2 / self.count)

    def scale(self, rewards):
        """缩放奖励"""
        # 标准化
        if self.running_std > 0:
            normalized = (rewards - self.running_mean) / (self.running_std + 1e-8)
        else:
            normalized = rewards - self.running_mean

        # 限制范围
        scaled = np.clip(normalized * self.scale_factor, -10, 10)
        return scaled
