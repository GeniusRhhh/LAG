import numpy as np
import logging
from .reward_function_base import BaseRewardFunction

class AltitudeReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.safe_altitude = getattr(self.config, f'{self.__class__.__name__}_safe_altitude', 6.0)
        self.danger_altitude = getattr(self.config, f'{self.__class__.__name__}_danger_altitude', 5.0)
        self.Kv = getattr(self.config, f'{self.__class__.__name__}_Kv', 0.2)
        self.target_altitude = getattr(self.config, 'target_altitude', 7.0)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_Pv', '_PH', '_Ptarget']]

    def get_reward(self, task, env, agent_id):
        ego_z = env.agents[agent_id].get_position()[-1] / 1000  # km
        ego_vz = env.agents[agent_id].get_velocity()[-1] / 340  # normalized vz
        Pv = -0.05 * np.clip(ego_vz / self.Kv * (self.safe_altitude - ego_z) / self.safe_altitude, 0.0, 1.0) if ego_z <= self.safe_altitude else 0.0
        PH = -0.2 * (1.0 - np.clip(ego_z / self.danger_altitude, 0.0, 1.0)) if ego_z <= self.danger_altitude else 0.0  # 降低惩罚力度
        Ptarget = 1.0 * np.exp(-((ego_z - self.target_altitude) ** 2) / 2.0)  # 增强目标高度奖励
        Pvz = -0.01 * abs(ego_vz) if abs(ego_vz) > 0.5 else 0.0  # 放宽速度惩罚阈值
        new_reward = Pv + PH + Ptarget + Pvz
        if task.step_count % 500 == 0:
            logging.info(f"Agent {agent_id} AltitudeReward: total={new_reward:.4f}, Pv={Pv:.4f}, PH={PH:.4f}, Ptarget={Ptarget:.4f}, Pvz={Pvz:.4f}, altitude={ego_z * 1000:.2f}m")
        return self._process(new_reward, agent_id, (Pv, PH, Ptarget, Pvz))