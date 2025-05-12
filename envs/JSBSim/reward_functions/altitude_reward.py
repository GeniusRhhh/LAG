# import numpy as np
# from .reward_function_base import BaseRewardFunction
#
#
# class AltitudeReward(BaseRewardFunction):
#     """
#     AltitudeReward
#     Punish if current fighter doesn't satisfy some constraints. Typically negative.
#     - Punishment of velocity when lower than safe altitude   (range: [-1, 0])
#     - Punishment of altitude when lower than danger altitude (range: [-1, 0])
#     """
#     def __init__(self, config):
#         super().__init__(config)
#         self.safe_altitude = getattr(self.config, f'{self.__class__.__name__}_safe_altitude', 4.0)         # km
#         self.danger_altitude = getattr(self.config, f'{self.__class__.__name__}_danger_altitude', 3.5)     # km
#         self.Kv = getattr(self.config, f'{self.__class__.__name__}_Kv', 0.2)     # mh
#
#         self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_Pv', '_PH']]
#
#     def get_reward(self, task, env, agent_id):
#         """
#         Reward is the sum of all the punishments.
#
#         Args:
#             task: task instance
#             env: environment instance
#
#         Returns:
#             (float): reward
#         """
#         ego_z = env.agents[agent_id].get_position()[-1] / 1000    # unit: km
#         ego_vz = env.agents[agent_id].get_velocity()[-1] / 340    # unit: mh
#         Pv = 0.
#         if ego_z <= self.safe_altitude:
#             Pv = -np.clip(ego_vz / self.Kv * (self.safe_altitude - ego_z) / self.safe_altitude, 0., 1.)
#         PH = 0.
#         if ego_z <= self.danger_altitude:
#             PH = np.clip(ego_z / self.danger_altitude, 0., 1.) - 1. - 1.
#         new_reward = Pv + PH
#         return self._process(new_reward, agent_id, (Pv, PH))


import math
import numpy as np
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c

class AltitudeReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.safe_altitude = getattr(self.config, f'{self.__class__.__name__}_safe_altitude', 4.0)  # km
        self.danger_altitude = getattr(self.config, f'{self.__class__.__name__}_danger_altitude', 3.5)  # km
        self.Kv = getattr(self.config, f'{self.__class__.__name__}_Kv', 0.5)  # Velocity penalty factor
        self.ideal_tolerance = getattr(self.config, f'{self.__class__.__name__}_ideal_tolerance', 2.0)  # km
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_alt', '_Pv', '_PH', '_ideal']]

    def get_reward(self, task, env, agent_id):
        ego_z = env.agents[agent_id].get_position()[-1] / 1000  # km
        ego_vz = env.agents[agent_id].get_velocity()[-1] / 340  # mh
        target_altitude = env.agents[agent_id].get_property_value(c.target_altitude_ft) / 3280.84  # ft to km

        altitude_error = abs(ego_z - target_altitude)
        # 强化的正向奖励
        alt_reward = 5.0 - 0.5 * altitude_error  # 更高基准奖励
        alt_reward += 2.0 * math.exp(-(altitude_error / 0.3) ** 2)  # 更强的精确控制奖励

        ideal_reward = 1.0 * math.exp(-(altitude_error / self.ideal_tolerance) ** 2)

        Pv = 0.
        if ego_z <= self.safe_altitude:
            Pv = -np.clip(ego_vz / self.Kv * (self.safe_altitude - ego_z) / self.safe_altitude, 0., 0.1) * (
                        1 + abs(ego_vz))

        PH = 0.
        if ego_z <= self.danger_altitude:
            PH = -3.0  # 更强的危险高度惩罚

        new_reward = alt_reward + Pv + PH + ideal_reward
        new_reward = np.clip(new_reward, -5.0, 5.0)
        return self._process(new_reward, agent_id, (alt_reward, Pv, PH, ideal_reward))