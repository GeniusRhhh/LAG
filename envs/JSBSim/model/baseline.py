from abc import ABC, abstractmethod
import sys
import os
import torch
import numpy as np
from typing import Literal
from .baseline_actor import BaselineActor
from ..utils.utils import get_root_dir


# 基线代理的抽象基类
class BaselineAgent(ABC):
    def __init__(self, agent_id) -> None:
        # 初始化基线代理，加载预训练的模型。
        self.model_path = get_root_dir() + '/model/baseline_model.pt'  # 模型路径
        self.actor = BaselineActor()  # 创建模型实例
        self.actor.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True))
        self.actor.eval()  # 将模型置于评估模式
        self.agent_id = agent_id  # 代理的ID
        self.reset()  # 重置代理状态

    def reset(self):
        # 重置代理的内部状态，主要是RNN状态。
        self.rnn_states = np.zeros((1, 1, 128))

    @abstractmethod
    def set_delta_value(self, observation):
        # 抽象方法，子类需要实现，用于计算基于观察的动作差值。
        raise NotImplementedError

    def get_observation(self, observation, delta_value):
        '''
        根据观察值和动作差值构建神经网络输入。

        参数:
        observation: 观测到的环境状态
        delta_value: 计算出的动作差值

        返回:
        构建的观察值数组，用于输入到神经网络
        Baseline  observation:
#  0. ego delta altitude      (unit: 1km)
#  1. ego delta heading       (unit rad)
#  2. ego delta velocities_u  (unit: mh)
#  3. ego_altitude            (unit: 5km)
#  4. ego_roll_sin
#  5. ego_roll_cos
#  6. ego_pitch_sin
#  7. ego_pitch_cos
#  8. ego_body_v_x            (unit: mh)
#  9. ego_body_v_y            (unit: mh)
#  10. ego_body_v_z           (unit: mh)
#  11. ego_vc                 (unit: mh)
        '''

        norm_obs = np.zeros(12)
        norm_obs[:3] = delta_value  # 前三个是动作差值
        norm_obs[3:12] = observation[:9]  # 后续是环境观察值
        norm_obs = np.expand_dims(norm_obs, axis=0)  # 增加批次维度
        return norm_obs

    def get_action(self, observation):
        '''
        根据观察值生成动作。

        参数:
        observation: 环境观察值

        返回:
        生成的动作
        '''
        delta_value = self.set_delta_value(observation[self.agent_id])
        obs = self.get_observation(observation[self.agent_id], delta_value)
        _action, self.rnn_states = self.actor(obs, self.rnn_states)
        action = _action.detach().cpu().numpy().squeeze()
        return action


# 追击代理，专门用于追踪目标
class PursueAgent(BaselineAgent):
    def __init__(self, agent_id) -> None:
        super().__init__(agent_id)

    def set_delta_value(self, observation):
        # 计算追踪目标时的高度、航向和速度差。
        delta_altitude = observation[10]
        delta_heading = observation[14] * observation[11]
        delta_velocity = observation[9]
        return np.array([delta_altitude, delta_heading, delta_velocity])


# 机动代理，能执行复杂的机动动作
class ManeuverAgent(BaselineAgent):
    def __init__(self, agent_id, maneuver: Literal['l', 'r', 'n']) -> None:
        super().__init__(agent_id)
        self.turn_interval = 7  # 转向间隔秒数
        self.env_time_interval = 0.2  # 环境时间间隔秒数
        self.dodge_missile = True  # 检测到导弹时开始转向
        # 设置不同转向策略的列表
        if maneuver == 'l':
            self.delta_heading_list = [0, 0, 0, 0]
        elif maneuver == 'r':
            self.delta_heading_list = [np.pi / 2, 0, 0, 0]
        elif maneuver == 'n':
            self.delta_heading_list = [np.pi / 2, np.pi / 2, 0, 0]
        self.target_altitude_list = [6096] * 4  # 目标高度列表
        self.target_velocity_list = [243] * 4  # 目标速度列表

    def reset(self):
        # 重置代理状态
        self.step = 0
        self.rnn_states = np.zeros((1, 1, 128))

    def set_delta_value(self, observation):
        # 计算需要的机动动作改变值
        step_list = np.arange(1, len(self.delta_heading_list) + 1) * self.turn_interval / self.env_time_interval
        if not self.dodge_missile or (len(observation) > 15 and observation[15] != 0):
            for i, interval in enumerate(step_list):
                if self.step <= interval:
                    break
            delta_heading = self.delta_heading_list[i]
            delta_altitude = (self.target_altitude_list[i] - observation[0] * 5000) / 1000
            delta_velocity = (self.target_velocity_list[i] - observation[5] * 340) / 340
            self.step += 1
        else:
            delta_heading = 0
            delta_altitude = (6096 - observation[0] * 5000) / 1000
            delta_velocity = (243 - observation[5] * 340) / 340
        return np.array([delta_altitude, delta_heading, delta_velocity])
