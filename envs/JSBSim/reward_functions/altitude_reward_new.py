import numpy as np
from .reward_function_base import BaseRewardFunction

class AltitudeReward(BaseRewardFunction):
    """
    AltitudeReward（改进版）
    功能：
    - 当高度低于安全高度时，根据垂直速度和高度差给予负奖励，鼓励爬升至安全区。
    - 当高度低于危险高度时，惩罚更大，防止过低飞行。
    - 增加对于理想高度的轻微正向奖励（可选），如果在理想高度附近（上下浮动一定范围），则给予小幅加分，以维持最佳高度态势。

    配置参数：
    - AltitudeReward_safe_altitude：安全高度（km）
    - AltitudeReward_danger_altitude：危险高度（km）
    - AltitudeReward_Kv：惩罚缩放系数
    - AltitudeReward_ideal_altitude：理想高度（km），可选
    - AltitudeReward_ideal_tolerance：理想高度范围容忍度（km）
    """
    def __init__(self, config):
        super().__init__(config)
        self.safe_altitude = getattr(self.config, f'{self.__class__.__name__}_safe_altitude', 4.0)         # km
        self.danger_altitude = getattr(self.config, f'{self.__class__.__name__}_danger_altitude', 3.5)     # km
        self.Kv = getattr(self.config, f'{self.__class__.__name__}_Kv', 0.2)     # mh

        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_Pv', '_PH']]

    def get_reward(self, task, env, agent_id):
        ego_z = env.agents[agent_id].get_position()[-1] / 1000  # unit: km
        ego_vz = env.agents[agent_id].get_velocity()[-1] / 340  # unit: mh
        target_altitude = 6.096  # 20000 ft ≈ 6.096 km
        altitude_error = abs(ego_z - target_altitude)
        alt_reward = -altitude_error * 0.5 + 1.0 * (ego_z > 5.0)  # 正向奖励
        Pv = 0.
        if ego_z <= self.safe_altitude:
            Pv = -np.clip(ego_vz / self.Kv * (self.safe_altitude - ego_z) / self.safe_altitude, 0., 1.)
        PH = 0.
        if ego_z <= self.danger_altitude:
            PH = np.clip(ego_z / self.danger_altitude, 0., 1.) - 1. - 1.
        new_reward = alt_reward + Pv + PH
        return self._process(new_reward, agent_id, (alt_reward, Pv, PH))
