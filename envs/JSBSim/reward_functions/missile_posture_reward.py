import numpy as np
from .reward_function_base import BaseRewardFunction


class MissilePostureReward(BaseRewardFunction):
    """
    MissilePostureReward
    Use the velocity attenuation
    """
    def __init__(self, config):
        super().__init__(config)
        self.previous_missile_v = None

    def reset(self, task, env):
        self.previous_missile_v = None  # 重置导弹速度为 None
        return super().reset(task, env)  # 调用父类的 reset 方法

    def get_reward(self, task, env, agent_id):
        reward = 0
        missile_sim = env.agents[agent_id].check_missile_warning()  # 检查当前是否有导弹威胁
        if missile_sim is not None:
            missile_v = missile_sim.get_velocity()  # 获取导弹的速度
            aircraft_v = env.agents[agent_id].get_velocity()  # 获取飞机的速度
            if self.previous_missile_v is None:
                self.previous_missile_v = missile_v  # 如果没有记录上一帧导弹速度，设置为当前速度
            v_decrease = (np.linalg.norm(self.previous_missile_v) - np.linalg.norm(missile_v)) / 340 * self.reward_scale
            angle = np.dot(missile_v, aircraft_v) / (
                        np.linalg.norm(missile_v) * np.linalg.norm(aircraft_v))  # 计算导弹与飞机速度方向的余弦角
            if angle < 0:  # 如果导弹方向背离飞机
                reward = angle / (max(v_decrease, 0) + 1)  # 奖励与角度和速度衰减相关
            else:  # 如果导弹方向指向飞机
                reward = angle * max(v_decrease, 0)  # 奖励为角度与速度衰减的乘积
        else:
            self.previous_missile_v = None  # 没有导弹威胁时，清空导弹速度记录
            reward = 0  # 奖励为 0
        self.reward_trajectory[agent_id].append([reward])  # 将奖励记录到智能体的奖励轨迹中
        return reward
