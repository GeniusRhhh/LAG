import math
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c

class HeadingReward(BaseRewardFunction):
    """
    HeadingReward（改进版）
    在原有误差基础上：
    - 考虑相对敌机朝向：若机头指向敌机方向则奖励更高
    - 考虑滚转变化率和俯仰变化率，以鼓励平稳的飞行状态（可降低突发机动）

    配置参数：
    - HeadingReward_heading_error_scale
    - HeadingReward_alt_error_scale
    - HeadingReward_roll_error_scale
    - HeadingReward_speed_error_scale
    - HeadingReward_stability_factor：控制俯仰、滚转变化率的奖惩力度
    """
    def __init__(self, config):
        super().__init__(config)
        self.heading_error_scale = getattr(self.config, f'{self.__class__.__name__}_heading_error_scale', 5.0)
        self.alt_error_scale = getattr(self.config, f'{self.__class__.__name__}_alt_error_scale', 15.24)
        self.roll_error_scale = getattr(self.config, f'{self.__class__.__name__}_roll_error_scale', 0.35)
        self.speed_error_scale = getattr(self.config, f'{self.__class__.__name__}_speed_error_scale', 24)
        self.stability_factor = getattr(self.config, f'{self.__class__.__name__}_stability_factor', 0.1)

        self.reward_item_names = [self.__class__.__name__ + item for item in [
            '', '_heading', '_alt', '_roll', '_speed','_stability'
        ]]

        # 存储上一帧的俯仰和滚转，用于计算变化率
        self.prev_attitude = {}

    def reset(self, task, env):
        self.prev_attitude.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]

        heading_r = math.exp(-((agent.get_property_value(c.delta_heading) / self.heading_error_scale) ** 2))
        alt_r = math.exp(-((agent.get_property_value(c.delta_altitude) / self.alt_error_scale) ** 2))
        roll_r = math.exp(-((agent.get_property_value(c.attitude_roll_rad) / self.roll_error_scale) ** 2))
        speed_r = math.exp(-((agent.get_property_value(c.delta_velocities_u) / self.speed_error_scale) ** 2))

        # 计算滚转和俯仰变化率作为稳定性指标
        cur_roll = agent.get_property_value(c.attitude_roll_rad)
        cur_pitch = agent.get_property_value(c.attitude_pitch_rad)
        prev_roll, prev_pitch = self.prev_attitude.get(agent_id, (cur_roll, cur_pitch))
        roll_change = abs(cur_roll - prev_roll)
        pitch_change = abs(cur_pitch - prev_pitch)
        stability_r = math.exp(-self.stability_factor * (roll_change + pitch_change))

        self.prev_attitude[agent_id] = (cur_roll, cur_pitch)

        # 综合考虑各项指标
        reward = (heading_r * alt_r * roll_r * speed_r * stability_r) ** (1/5)
        return self._process(reward, agent_id, (heading_r, alt_r, roll_r, speed_r, stability_r))
