import math
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c

class FlightStabilityReward(BaseRewardFunction):
    """
    FlightStabilityReward:
    根据滚转、俯仰变化率给予奖励，变化越小越接近1，变化越大则奖励减少。

    配置参数：
    - FlightStabilityReward_factor: 控制变化率敏感度
    """
    def __init__(self, config):
        super().__init__(config)
        self.factor = getattr(self.config, f'{self.__class__.__name__}_factor', 0.1)
        self.prev_attitude = {}

    def reset(self, task, env):
        self.prev_attitude.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]
        cur_roll = agent.get_property_value(c.attitude_roll_rad)
        cur_pitch = agent.get_property_value(c.attitude_pitch_rad)

        prev_roll, prev_pitch = self.prev_attitude.get(agent_id, (cur_roll, cur_pitch))
        roll_change = abs(cur_roll - prev_roll)
        pitch_change = abs(cur_pitch - prev_pitch)
        stability_r = math.exp(-self.factor*(roll_change+pitch_change))

        self.prev_attitude[agent_id] = (cur_roll, cur_pitch)
        return self._process(stability_r, agent_id)
