# reward_functions/altitude_reward.py
import numpy as np
import logging
from .reward_function_base import BaseRewardFunction

class AltitudeReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.safe_altitude = getattr(self.config, f'{self.__class__.__name__}_safe_altitude', 5.0)
        self.danger_altitude = getattr(self.config, f'{self.__class__.__name__}_danger_altitude', 3.5)
        self.Kv = getattr(self.config, f'{self.__class__.__name__}_Kv', 0.2)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_Pv', '_PH']]

    def get_reward(self, task, env, agent_id):
        ego_z = env.agents[agent_id].get_position()[-1] / 1000
        ego_vz = env.agents[agent_id].get_velocity()[-1] / 340
        Pv = 0.0
        if ego_z <= self.safe_altitude:
            Pv = -0.05 * np.clip(ego_vz / self.Kv * (self.safe_altitude - ego_z) / self.safe_altitude, 0.0, 1.0)  # -0.1 → -0.05
        PH = 0.0
        if ego_z <= self.danger_altitude:
            PH = np.clip(ego_z / self.danger_altitude, 0.0, 1.0) - 1.0
        new_reward = Pv + PH

        # 日志记录
        if task.step_count % 500 == 0:
            logging.info(
                f"Agent {agent_id} AltitudeReward (Step {task.step_count}): "
                f"total={new_reward:.4f}, Pv={Pv:.4f}, PH={PH:.4f}, "
                f"altitude={ego_z*1000:.2f}m"
            )

        return self._process(new_reward, agent_id, (Pv, PH))