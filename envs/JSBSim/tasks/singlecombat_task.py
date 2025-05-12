# import logging
#
# import torch
# import numpy as np
# from gymnasium import spaces
# from typing import Literal
# from .task_base import BaseTask
# from ..core.simulatior import AircraftSimulator
# from ..core.catalog import Catalog as c
# from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, SafeReturn
# from ..reward_functions import AltitudeReward, PostureReward, EventDrivenReward,RelativeAltitudeReward
# from ..utils.utils import get_AO_TA_R, get2d_AO_TA_R, in_range_rad, LLA2NEU, get_root_dir
# from ..model.baseline_actor import BaselineActor
#
# class SingleCombatTask(BaseTask):
#     """
#     单一战斗任务类，继承自基础任务类 BaseTask。
#     此类专门用于处理与单一战斗相关的配置和动态行为。
#     """
#     def __init__(self, config):
#         super().__init__(config)
#         logging.info(f"Config loaded: {self.config}")  # 打印完整的 config
#
#         self.use_baseline = getattr(self.config, 'use_baseline', False)
#         logging.info(f"Using baseline: {self.use_baseline}")  # Debug log
#
#         self.use_artillery = getattr(self.config, 'use_artillery', False)
#         if self.use_baseline:
#             self.baseline_agent = self.load_agent(self.config.baseline_type)
#         self.agent_ids = []  # 保存实际 Agent ID
#
#         self.reward_functions = [
#             AltitudeReward(self.config),
#             PostureReward(self.config),
#             EventDrivenReward(self.config),
#             RelativeAltitudeReward(self.config)
#         ]
#
#         self.termination_conditions = [
#             LowAltitude(self.config),
#             ExtremeState(self.config),
#             Overload(self.config),
#             SafeReturn(self.config),
#             Timeout(self.config),
#         ]
#

    # @property
    # def num_agents(self) -> int:
    #     return 2 if not self.use_baseline else 1
#     def load_variables(self):
#         """
#         加载与任务相关的状态变量和动作变量。
#         """
#         self.state_var = [
#         c.position_long_gc_deg,             # 经度
#         c.position_lat_geod_deg,            # 纬度
#         c.position_h_sl_m,                  # 海拔高度
#         c.attitude_roll_rad,                # 滚转角
#         c.attitude_pitch_rad,               # 俯仰角
#         c.attitude_heading_true_rad,        # 真航向
#             c.velocities_v_north_mps,           # 北向速度
#             c.velocities_v_east_mps,            # 东向速度
#             c.velocities_v_down_mps,            # 下降速度
#         c.velocities_u_mps,                 # 机体x轴速度
#         c.velocities_v_mps,                 # 机体y轴速度
#         c.velocities_w_mps,                 # 机体z轴速度
#             c.velocities_vc_mps,                # 真空速
#             c.accelerations_n_pilot_x_norm,     # 北向加速度
#             c.accelerations_n_pilot_y_norm,     # 东向加速度
#             c.accelerations_n_pilot_z_norm,     # 下降加速度
#         ]
#         self.action_var = [
#             c.fcs_aileron_cmd_norm,             # 副翼控制
#             c.fcs_elevator_cmd_norm,            # 升降舵控制
#             c.fcs_rudder_cmd_norm,              # 方向舵控制
#             c.fcs_throttle_cmd_norm,            # 油门控制
#         ]
#         self.render_var = [
#             c.position_long_gc_deg,
#             c.position_lat_geod_deg,
#             c.position_h_sl_m,
#             c.attitude_roll_rad,
#             c.attitude_pitch_rad,
#             c.attitude_heading_true_rad,
#         ]
#
#     def load_observation_space(self):
#         """
#         定义观察空间，使用高低限制定义状态变量的范围。
#         """
#         self.observation_space = spaces.Box(low=-10, high=10., shape=(15,))
#
#     def load_action_space(self):
#         """
#         定义动作空间，这里使用多离散动作空间。
#         """
#         self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])
#
#
#     def get_obs(self, env, agent_id):
#         """
#         从环境中获取观察值，转换模拟状态为观察空间格式。
#
#         返回:
#         - (np.ndarray) 观察值数组
#         """
#         norm_obs = np.zeros(15)
#         ego_obs_list = np.array(env.agents[agent_id].get_property_values(self.state_var))
#         enm_obs_list = np.array(env.agents[agent_id].enemies[0].get_property_values(self.state_var))
#         # 提取特征：北向、东向、下降、北向速度、东向速度、下降速度
#         ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
#         enm_cur_ned = LLA2NEU(*enm_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
#         ego_feature = np.array([*ego_cur_ned, *(ego_obs_list[6:9])])
#         enm_feature = np.array([*enm_cur_ned, *(enm_obs_list[6:9])])
#         # 规范化ego信息
#         norm_obs[0] = ego_obs_list[2] / 5000            # 海拔高度（单位：5公里）
#         norm_obs[1] = np.sin(ego_obs_list[3])           # 滚转角正弦值
#         norm_obs[2] = np.cos(ego_obs_list[3])           # 滚转角余弦值
#         norm_obs[3] = np.sin(ego_obs_list[4])           # 俯仰角正弦值
#         norm_obs[4] = np.cos(ego_obs_list[4])           # 俯仰角余弦值
#         norm_obs[5] = ego_obs_list[9] / 340             # 机体x轴速度（单位：音速）
#         norm_obs[6] = ego_obs_list[10] / 340            # 机体y轴速度（单位：音速）
#         norm_obs[7] = ego_obs_list[11] / 340            # 机体z轴速度（单位：音速）
#         norm_obs[8] = ego_obs_list[12] / 340            # 真空速（单位：音速）
#         # 与敌机的相对信息
#         ego_AO, ego_TA, R, side_flag = get2d_AO_TA_R(ego_feature, enm_feature, return_side=True)
#         norm_obs[9] = (enm_obs_list[9] - ego_obs_list[9]) / 340
#         norm_obs[10] = (enm_obs_list[2] - ego_obs_list[2]) / 1000
#         norm_obs[11] = ego_AO
#         norm_obs[12] = ego_TA
#         norm_obs[13] = R / 10000
#         norm_obs[14] = side_flag
#         norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
#         return norm_obs
#
#     def normalize_action(self, env, agent_id, action):
#         """
#         将离散动作索引转换为连续值。
#         """
#         if self.use_baseline and agent_id in env.enm_ids:
#             action = self.baseline_agent.get_action(env.agents[agent_id])
#             return action
#         else:
#             norm_act = np.zeros(4)
#             norm_act[0] = action[0] / 20 - 1.
#             norm_act[1] = action[1] / 20 - 1.
#             norm_act[2] = action[2] / 20 - 1.
#             norm_act[3] = action[3] / 58 + 0.4
#             return norm_act
#
#     def reset(self, env):
#         """
#         任务特定的重置方法，包括奖励函数的重置。
#         """
#         self._agent_die_flag = {}
#         if self.use_baseline:
#             self.baseline_agent.reset()
#         self.agent_ids = list(env.agents.keys())  # 获取环境中的 Agent ID (如 ['agent_0100', 'agent_0200'])
#         return super().reset(env)
#
#     def step(self, env):
#         """
#         环境的单步执行方法，处理导弹避免和其他战斗动态。
#         """
#         def _orientation_fn(AO):
#             if AO >= 0 and AO <= 0.5236:  # [0, pi/6]
#                 return 1 - AO / 0.5236
#             elif AO >= -0.5236 and AO <= 0: # [-pi/6, 0]
#                 return 1 + AO / 0.5236
#             return 0
#         def _distance_fn(R):
#             if R <=1: # [0, 1km]
#                 return 1
#             elif R > 1 and R <= 3: # [1km, 3km]
#                 return (3 - R) / 2.
#             else:
#                 return 0
#         if self.use_artillery:
#             for agent_id in env.agents.keys():
#                 ego_feature = np.hstack([env.agents[agent_id].get_position(),
#                                         env.agents[agent_id].get_velocity()])
#                 for enm in env.agents[agent_id].enemies:
#                     if enm.is_alive:
#                         enm_feature = np.hstack([enm.get_position(),
#                                                 enm.get_velocity()])
#                         AO, _, R = get_AO_TA_R(ego_feature, enm_feature)
#                         enm.bloods -= _orientation_fn(AO) * _distance_fn(R/1000)
#                         # if agent_id == 'A0100' and enm.uid == 'B0100':
#                         #     print(f"AO: {AO * 180 / np.pi}, {_orientation_fn(AO)}, dis:{R/1000}, {_distance_fn(R/1000)}")
#
#     def get_reward(self, env, agent_id, info=...):
#         """
#         获取环境中指定代理的奖励。
#         """
#         if self._agent_die_flag.get(agent_id, False):
#             return 0.0, info
#         else:
#             self._agent_die_flag[agent_id] = not env.agents[agent_id].is_alive
#             return super().get_reward(env, agent_id, info=info)
#
#     def load_agent(self, name):
#         """
#         根据指定名称加载代理模型。
#         """
#         if name == 'pursue':
#             return PursueAgent()
#         elif name == 'maneuver':
#             return ManeuverAgent(maneuver='r')
#         elif name == 'dodge':
#             return DodgeMissileAgent()
#         elif name == 'straight':
#             return StraightFlyAgent()
#         else:
#             raise NotImplementedError
#
# class HierarchicalSingleCombatTask(SingleCombatTask):
#     """
#     分层单一战斗任务，扩展自单一战斗任务，增加了更复杂的动作和策略处理。
#     """
#     def __init__(self, config: str):
#         super().__init__(config)
#         self.lowlevel_policy = BaselineActor()
#         self.lowlevel_policy.load_state_dict(torch.load(get_root_dir() + '/model/baseline_model.pt', map_location=torch.device('cpu'), weights_only=True))
#         self.lowlevel_policy.eval()
#         self.norm_delta_altitude = np.array([0.1, 0, -0.1])
#         self.norm_delta_heading = np.array([-np.pi / 6, -np.pi / 12, 0, np.pi / 12, np.pi / 6])
#         self.norm_delta_velocity = np.array([0.05, 0, -0.05])
#
#     def load_action_space(self):
#         """
#         为分层任务加载动作空间，这里使用多离散动作空间。
#         """
#         self.action_space = spaces.MultiDiscrete([3, 5, 3])
#
#     def normalize_action(self, env, agent_id, action):
#         """
#         将高级动作转换为低级动作。
#         """
#         if self.use_baseline and agent_id in env.enm_ids:
#             action = self.baseline_agent.get_action(env.agents[agent_id])
#             return action
#         else:
#             # 生成低级输入观察值
#             raw_obs = self.get_obs(env, agent_id)
#             input_obs = np.zeros(12)
#             # (1) delta altitude/heading/velocity
#             input_obs[0] = self.norm_delta_altitude[action[0]]
#             input_obs[1] = self.norm_delta_heading[action[1]]
#             input_obs[2] = self.norm_delta_velocity[action[2]]
#             # (2) ego info
#             input_obs[3:12] = raw_obs[:9]
#             input_obs = np.expand_dims(input_obs, axis=0)#为了匹配网络输入格式，增加一个新的批处理维度。
#             # 输出低级动作
#             _action, _rnn_states = self.lowlevel_policy(input_obs, self._inner_rnn_states[agent_id])#使用低级策略网络 (lowlevel_policy) 处理输入观察值，并传入当前的RNN状态，获取动作和新的RNN状态。
#             action = _action.detach().cpu().numpy().squeeze(0)#将动作从张量转换为numpy数组，并移除批处理维度。
#             self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()#更新RNN状态。
#             # 规范化低级动作
#             norm_act = np.zeros(4)
#             norm_act[0] = action[0] / 20 - 1.
#             norm_act[1] = action[1] / 20 - 1.
#             norm_act[2] = action[2] / 20 - 1.
#             norm_act[3] = action[3] / 58 + 0.4
#             return norm_act
#
#     def reset(self, env):
#         """
#         任务特定的重置方法，包括奖励函数的重置以及内部RNN状态的初始化。
#         """
#         self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
#         return super().reset(env)
#
#
# class StraightFlyAgent:
#     """
#     直线飞行代理类，简单控制飞行器直线飞行。
#     """
#     def normalize_action(self, action):
#         """
#         规范化动作值，将离散动作转换为连续动作值。
#         """
#         norm_act = np.zeros(4)
#         norm_act[0] = action[0] / 20 - 1.   # 0~40 => -1~1
#         norm_act[1] = action[1] / 20 - 1.   # 0~40 => -1~1
#         norm_act[2] = action[2] / 20 - 1.   # 0~40 => -1~1
#         norm_act[3] = action[3] / 58 + 0.4  # 0~29 => 0.4~0.9
#         return norm_act
#
#     def get_action(self, sim: AircraftSimulator):
#         """
#         获取直线飞行的动作。
#         """
#         action = np.array([20, 18.6, 20, 0])
#         return self.normalize_action(action)
#
#     def reset(self):
#         """
#         重置代理状态。
#         """
#         pass
#
#
# class BaselineAgent:
#     """
#     基线代理类，定义了基础的飞行行为。
#     """
#     def __init__(self) -> None:
#         self.model_path = get_root_dir() + '/model/baseline_model.pt'
#         self.actor = BaselineActor()
#         self.actor.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True))
#         self.actor.eval()
#         self.state_var = [
#             c.delta_altitude,                   # 高度差
#             c.delta_heading,                    # 航向差
#             c.delta_velocities_u,               # 速度差
#             c.attitude_roll_rad,                # 滚转角
#             c.attitude_pitch_rad,               # 俯仰角
#             c.velocities_u_mps,                 # 机体x轴速度
#             c.velocities_v_mps,                 # 机体y轴速度
#             c.velocities_w_mps,                 # 机体z轴速度
#             c.velocities_vc_mps,                # 真空速
#             c.position_h_sl_m                   # 海拔高度
#         ]
#         self.reset()
#
#     def normalize_action(self, action):
#         """
#         规范化动作值，将离散动作转换为连续动作值。
#         """
#         norm_act = np.zeros(4)
#         norm_act[0] = action[0] / 20 - 1.   # 0~40 => -1~1
#         norm_act[1] = action[1] / 20 - 1.   # 0~40 => -1~1
#         norm_act[2] = action[2] / 20 - 1.   # 0~40 => -1~1
#         norm_act[3] = action[3] / 58 + 0.4  # 0~29 => 0.4~0.9
#         return norm_act
#
#     def reset(self):
#         """
#         重置代理状态，初始化RNN状态。
#         """
#         self.rnn_states = np.zeros((1, 1, 128))
#
#     def set_delta_value(self, sim: AircraftSimulator):
#         """
#         设置与目标敌机的相对位置和速度。
#         """
#         raise NotImplementedError
#
#     def get_observation(self, sim: AircraftSimulator, delta_value):
#         """
#         获取当前模拟状态的观察值。
#         """
#         obs = sim.get_property_values(self.state_var)
#         norm_obs = np.zeros(12)
#         norm_obs[0] = delta_value[0] / 1000          # 高度差（单位：公里）
#         norm_obs[1] = in_range_rad(delta_value[1])   # 航向差（单位：弧度）
#         norm_obs[2] = delta_value[2] / 340           # 速度差（单位：音速）
#         norm_obs[3] = obs[9] / 5000                  # 海拔高度（单位：5公里）
#         norm_obs[4] = np.sin(obs[3])                 # 滚转角正弦值
#         norm_obs[5] = np.cos(obs[3])                 # 滚转角余弦值
#         norm_obs[6] = np.sin(obs[4])                 # 俯仰角正弦值
#         norm_obs[7] = np.cos(obs[4])                 # 俯仰角余弦值
#         norm_obs[8] = obs[5] / 340                   # 机体x轴速度（单位：音速）
#         norm_obs[9] = obs[6] / 340                   # 机体y轴速度（单位：音速）
#         norm_obs[10] = obs[7] / 340                  # 机体z轴速度（单位：音速）
#         norm_obs[11] = obs[8] / 340                  # 真空速（单位：音速）
#         norm_obs = np.expand_dims(norm_obs, axis=0)  # 维度：(1,12)
#         return norm_obs
#
#     def get_action(self, sim: AircraftSimulator):
#         """
#         根据当前模拟状态和目标值获取代理动作。
#         """
#         delta_value = self.set_delta_value(sim)
#         observation = self.get_observation(sim, delta_value)
#         _action, self.rnn_states = self.actor(observation, self.rnn_states)
#         action = _action.detach().cpu().numpy().squeeze()
#         return self.normalize_action(action)
#
#
# class PursueAgent(BaselineAgent):
#     """
#     追击代理类，继承自基线代理，专注于追踪敌机。
#     """
#     def __init__(self) -> None:
#         super().__init__()
#
#     def set_delta_value(self, sim: AircraftSimulator):
#         """
#         计算与目标敌机的相对位置和速度差。
#         """
#         # NOTE: 仅适用于1对1场景
#         ego_x, ego_y, ego_z = sim.get_position()
#         ego_vx, ego_vy, ego_vz = sim.get_velocity()
#         enm_x, enm_y, enm_z = sim.enemies[0].get_position()
#         # 高度差
#         delta_altitude = enm_z - ego_z
#         # 航向差
#         ego_v = np.linalg.norm([ego_vx, ego_vy])
#         delta_x, delta_y = enm_x - ego_x, enm_y - ego_y
#         R = np.linalg.norm([delta_x, delta_y])
#         proj_dist = delta_x * ego_vx + delta_y * ego_vy
#         ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
#         side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
#         delta_heading = ego_AO * side_flag
#         # 速度差
#         delta_velocity = sim.enemies[0].get_property_value(c.velocities_u_mps) - \
#                          sim.get_property_value(c.velocities_u_mps)
#         return np.array([delta_altitude, delta_heading, delta_velocity])
#
#
# class ManeuverAgent(BaselineAgent):
#     """
#     机动代理类，继承自基线代理，可以进行特定的机动动作。
#     """
#     def __init__(self, maneuver: Literal['l', 'r', 'n']) -> None:
#         super().__init__()
#         self.turn_interval = 30
#         self.dodge_missile = True  # 如果设置为真，当检测到导弹时开始转向
#         if maneuver == 'l':
#             self.target_heading_list = [0]
#         elif maneuver == 'r':
#             self.target_heading_list = [np.pi/2, np.pi/2, np.pi/2, np.pi/2]
#         elif maneuver == 'n':
#             self.target_heading_list = [np.pi, np.pi, np.pi, np.pi]
#         self.target_altitude_list = [6096] * 4
#         self.target_velocity_list = [243] * 4
#
#     def reset(self):
#         """
#         重置代理状态，初始化步骤和RNN状态。
#         """
#         self.step = 0
#         self.rnn_states = np.zeros((1, 1, 128))
#         self.init_heading = None
#
#     def set_delta_value(self, sim: AircraftSimulator):
#         """
#         根据设定的目标值和当前状态计算动作所需的改变值。
#         """
#         step_list = np.arange(1, len(self.target_heading_list)+1) * self.turn_interval / 0.2
#         cur_heading = sim.get_property_value(c.attitude_heading_true_rad)
#         if self.init_heading is None:
#             self.init_heading = cur_heading
#         if not self.dodge_missile or len(sim.under_missiles) != 0:
#             for i, interval in enumerate(step_list):
#                 if self.step <= interval:
#                     break
#             delta_heading = self.init_heading + self.target_heading_list[i] - cur_heading
#             delta_altitude = self.target_altitude_list[i] - sim.get_property_value(c.position_h_sl_m)
#             delta_velocity = self.target_velocity_list[i] - sim.get_property_value(c.velocities_u_mps)
#             self.step += 1
#         else:
#             delta_heading = self.init_heading  - cur_heading
#             delta_altitude = 6096 - sim.get_property_value(c.position_h_sl_m)
#             delta_velocity = 243 - sim.get_property_value(c.velocities_u_mps)
#
#         return np.array([delta_altitude, delta_heading, delta_velocity])
#
#
# class DodgeMissileAgent:
#     """
#     躲避导弹代理类，使用特定的模型处理导弹躲避行为。
#     """
#     def __init__(self) -> None:
#         self.model_path = get_root_dir() + '/model/dodge_missile_model.pt'
#         self.actor = BaselineActor(input_dim=21, use_mlp_actlayer=True)
#         self.actor.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True))
#         self.state_var = [
#             c.position_long_gc_deg,             # 经度
#             c.position_lat_geod_deg,            # 纬度
#             c.position_h_sl_m,                  # 海拔高度
#             c.attitude_roll_rad,                # 滚转角
#             c.attitude_pitch_rad,               # 俯仰角
#             c.attitude_heading_true_rad,        # 真航向
#             c.velocities_v_north_mps,           # 北向速度
#             c.velocities_v_east_mps,            # 东向速度
#             c.velocities_v_down_mps,            # 下降速度
#             c.velocities_u_mps,                 # 机体x轴速度
#             c.velocities_v_mps,                 # 机体y轴速度
#             c.velocities_w_mps,                 # 机体z轴速度
#             c.velocities_vc_mps,                # 真空速
#         ]
#         self.reset()
#
#     def get_observation(self, sim: AircraftSimulator):
#         """
#         获取当前模拟状态的观察值。
#         """
#         norm_obs = np.zeros(21)
#         ego_obs_list = np.array(sim.get_property_values(self.state_var))
#         enm_obs_list = np.array(sim.enemies[0].get_property_values(self.state_var))
#         # 提取特征：北向、东向、下降、北向速度、东向速度、下降速度
#         ego_cur_ned = LLA2NEU(*ego_obs_list[:3], 120.0, 60.0, 0.0)
#         enm_cur_ned = LLA2NEU(*enm_obs_list[:3], 120.0, 60.0, 0.0)
#         ego_feature = np.array([*ego_cur_ned, *(ego_obs_list[6:9])])
#         enm_feature = np.array([*enm_cur_ned, *(enm_obs_list[6:9])])
#         # 规范化ego信息
#         norm_obs[0] = ego_obs_list[2] / 5000            # 海拔高度（单位：5公里）
#         norm_obs[1] = np.sin(ego_obs_list[3])           # 滚转角正弦值
#         norm_obs[2] = np.cos(ego_obs_list[3])           # 滚转角余弦值
#         norm_obs[3] = np.sin(ego_obs_list[4])           # 俯仰角正弦值
#         norm_obs[4] = np.cos(ego_obs_list[4])           # 俯仰角余弦值
#         norm_obs[5] = ego_obs_list[9] / 340             # 机体x轴速度（单位：音速）
#         norm_obs[6] = ego_obs_list[10] / 340            # 机体y轴速度（单位：音速）
#         norm_obs[7] = ego_obs_list[11] / 340            # 机体z轴速度（单位：音速）
#         norm_obs[8] = ego_obs_list[12] / 340            # 真空速（单位：音速）
#         # 与敌机的相对信息
#         ego_AO, ego_TA, R, side_flag = get2d_AO_TA_R(ego_feature, enm_feature, return_side=True)
#         norm_obs[9] = (enm_obs_list[9] - ego_obs_list[9]) / 340
#         norm_obs[10] = (enm_obs_list[2] - ego_obs_list[2]) / 1000
#         norm_obs[11] = ego_AO
#         norm_obs[12] = ego_TA
#         norm_obs[13] = R / 10000
#         norm_obs[14] = side_flag
#         # 与导弹的相对信息
#         if len(sim.under_missiles) != 0 and sim.under_missiles[0].is_alive:
#             missile_sim = sim.under_missiles[0]
#         else:
#             missile_sim = None
#         if missile_sim is not None:
#             missile_feature = np.concatenate((missile_sim.get_position(), missile_sim.get_velocity()))
#             ego_AO, ego_TA, R, side_flag = get2d_AO_TA_R(ego_feature, missile_feature, return_side=True)
#             norm_obs[15] = (np.linalg.norm(missile_sim.get_velocity()) - ego_obs_list[9]) / 340
#             norm_obs[16] = (missile_feature[2] - ego_obs_list[2]) / 1000
#             norm_obs[17] = ego_AO
#             norm_obs[18] = ego_TA
#             norm_obs[19] = R / 10000
#             norm_obs[20] = side_flag
#         norm_obs = np.expand_dims(norm_obs, axis=0)
#         return norm_obs
#
#     def get_action(self, sim: AircraftSimulator):
#         """
#         根据当前观察值获取导弹躲避动作。
#         """
#         obs = self.get_observation(sim)
#         _action, self.rnn_states = self.actor(obs, self.rnn_states)
#         action = _action.squeeze().detach().cpu().numpy().squeeze()
#         return action
#
#     def reset(self):
#         """
#         重置代理状态，初始化RNN状态。
#         """
#         self.rnn_states = np.zeros((1, 1, 128))


import logging

import torch
import numpy as np
from gymnasium import spaces
from typing import Literal
from .task_base import BaseTask
from ..core.simulatior import AircraftSimulator
from ..core.catalog import Catalog as c
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, SafeReturn
from ..reward_functions import AltitudeReward, PostureReward, EventDrivenReward,RelativeAltitudeReward
from ..utils.utils import get_AO_TA_R, get2d_AO_TA_R, in_range_rad, LLA2NEU, get_root_dir
from ..model.baseline_actor import BaselineActor

class SingleCombatTask(BaseTask):
    """
    单一战斗任务类，继承自基础任务类 BaseTask。
    此类专门用于处理与单一战斗相关的配置和动态行为。
    适配 SAC 算法，支持连续动作空间。
    """
    def __init__(self, config):
        super().__init__(config)
        logging.info(f"Config loaded: {self.config}")  # 打印完整的 config

        self.use_baseline = getattr(self.config, 'use_baseline', False)
        logging.info(f"Using baseline: {self.use_baseline}")  # Debug log

        self.use_artillery = getattr(self.config, 'use_artillery', False)
        if self.use_baseline:
            self.baseline_agent = self.load_agent(self.config.baseline_type)
        self.agent_ids = []  # 保存实际 Agent ID

        self.reward_functions = [
            AltitudeReward(self.config),
            PostureReward(self.config),
            EventDrivenReward(self.config),
            RelativeAltitudeReward(self.config)
        ]

        self.termination_conditions = [
            LowAltitude(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            SafeReturn(self.config),
            Timeout(self.config),
        ]

    @property
    def num_agents(self) -> int:
        return 2 if not self.use_baseline else 1

    def load_variables(self):
        """
        加载与任务相关的状态变量和动作变量。
        """
        self.state_var = [
            c.position_long_gc_deg,             # 经度
            c.position_lat_geod_deg,            # 纬度
            c.position_h_sl_m,                  # 海拔高度
            c.attitude_roll_rad,                # 滚转角
            c.attitude_pitch_rad,               # 俯仰角
            c.attitude_heading_true_rad,        # 真航向
            c.velocities_v_north_mps,           # 北向速度
            c.velocities_v_east_mps,            # 东向速度
            c.velocities_v_down_mps,            # 下降速度
            c.velocities_u_mps,                 # 机体x轴速度
            c.velocities_v_mps,                 # 机体y轴速度
            c.velocities_w_mps,                 # 机体z轴速度
            c.velocities_vc_mps,                # 真空速
            c.accelerations_n_pilot_x_norm,     # 北向加速度
            c.accelerations_n_pilot_y_norm,     # 东向加速度
            c.accelerations_n_pilot_z_norm,     # 下降加速度
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,             # 副翼控制
            c.fcs_elevator_cmd_norm,            # 升降舵控制
            c.fcs_rudder_cmd_norm,              # 方向舵控制
            c.fcs_throttle_cmd_norm,            # 油门控制
        ]
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    def load_observation_space(self):
        """
        定义观察空间，使用高低限制定义状态变量的范围。
        """
        self.observation_space = spaces.Box(low=-10, high=10, shape=(15,), dtype=np.float32)

    def load_action_space(self):
        """
        定义动作空间，使用连续动作空间适配 SAC。
        """
        low = np.array([-1, -1, -1, 0.4], dtype=np.float32)
        high = np.array([1, 1, 1, 0.9], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def normalize_action(self, env, agent_id, action):
        """
        规范化动作，确保动作在连续动作空间范围内。
        """
        if self.use_baseline and agent_id in env.enm_ids:
            action = self.baseline_agent.get_action(env.agents[agent_id])
            return action
        low = self.action_space.low
        high = self.action_space.high
        return np.clip(action, low, high)

    def get_obs(self, env, agent_id):
        """
        从环境中获取观察值，转换模拟状态为观察空间格式。
        返回:
        - (np.ndarray) 观察值数组
        """
        norm_obs = np.zeros(15, dtype=np.float32)
        ego_obs_list = np.array(env.agents[agent_id].get_property_values(self.state_var))
        enm_obs_list = np.array(env.agents[agent_id].enemies[0].get_property_values(self.state_var))
        # 提取特征：北向、东向、下降、北向速度、东向速度、下降速度
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_obs_list[6:9])])
        enm_feature = np.array([*enm_cur_ned, *(enm_obs_list[6:9])])
        # 规范化 ego 信息
        norm_obs[0] = ego_obs_list[2] / 5000            # 海拔高度（单位：5公里）
        norm_obs[1] = np.sin(ego_obs_list[3])           # 滚转角正弦值
        norm_obs[2] = np.cos(ego_obs_list[3])           # 滚转角余弦值
        norm_obs[3] = np.sin(ego_obs_list[4])           # 俯仰角正弦值
        norm_obs[4] = np.cos(ego_obs_list[4])           # 俯仰角余弦值
        norm_obs[5] = ego_obs_list[9] / 340             # 机体x轴速度（单位：音速）
        norm_obs[6] = ego_obs_list[10] / 340            # 机体y轴速度（单位：音速）
        norm_obs[7] = ego_obs_list[11] / 340            # 机体z轴速度（单位：音速）
        norm_obs[8] = ego_obs_list[12] / 340            # 真空速（单位：音速）
        # 与敌机的相对信息
        ego_AO, ego_TA, R, side_flag = get2d_AO_TA_R(ego_feature, enm_feature, return_side=True)
        norm_obs[9] = (enm_obs_list[9] - ego_obs_list[9]) / 340
        norm_obs[10] = (enm_obs_list[2] - ego_obs_list[2]) / 1000
        norm_obs[11] = ego_AO
        norm_obs[12] = ego_TA
        norm_obs[13] = R / 10000
        norm_obs[14] = side_flag
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def reset(self, env):
        """
        任务特定的重置方法，包括奖励函数的重置。
        """
        self._agent_die_flag = {}
        if self.use_baseline:
            self.baseline_agent.reset()
        self.agent_ids = list(env.agents.keys())  # 获取环境中的 Agent ID (如 ['agent_0100', 'agent_0200'])
        return super().reset(env)

    def step(self, env):
        """
        环境的单步执行方法，处理导弹避免和其他战斗动态。
        """
        logging.debug(f"step: entering step method, env={env.__dict__}")

        # 处理 use_artillery 逻辑
        def _orientation_fn(AO):
            if AO >= 0 and AO <= 0.5236:  # [0, pi/6]
                return 1 - AO / 0.5236
            elif AO >= -0.5236 and AO <= 0:  # [-pi/6, 0]
                return 1 + AO / 0.5236
            return 0

        def _distance_fn(R):
            if R <= 1:  # [0, 1km]
                return 1
            elif R > 1 and R <= 3:  # [1km, 3km]
                return (3 - R) / 2.
            else:
                return 0

        if self.use_artillery:
            for agent_id in env.agents.keys():
                ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                         env.agents[agent_id].get_velocity()])
                for enm in env.agents[agent_id].enemies:
                    if enm.is_alive:
                        enm_feature = np.hstack([enm.get_position(),
                                                 enm.get_velocity()])
                        AO, _, R = get_AO_TA_R(ego_feature, enm_feature)
                        enm.bloods -= _orientation_fn(AO) * _distance_fn(R / 1000)

        # 调用父类 step 方法
        result = super().step(env)
        if result is None:
            logging.error(f"step: super().step(env) returned None, env={env.__dict__}")
            raise ValueError("BaseTask.step returned None, expected (obs, rewards, dones, infos)")

        obs, rewards, dones, infos = result
        # logging.info(f"step: obs_shape={obs.shape}, rewards_shape={rewards.shape}, dones_shape={dones.shape}")
        return obs, rewards, dones, infos

    def get_reward(self, env, agent_id, info=None):
        """
        获取环境中指定代理的奖励。
        """
        if self._agent_die_flag.get(agent_id, False):
            return 0.0, info
        else:
            self._agent_die_flag[agent_id] = not env.agents[agent_id].is_alive
            return super().get_reward(env, agent_id, info=info)

    def load_agent(self, name):
        """
        根据指定名称加载代理模型。
        """
        if name == 'pursue':
            return PursueAgent()
        elif name == 'maneuver':
            return ManeuverAgent(maneuver='r')
        elif name == 'dodge':
            return DodgeMissileAgent()
        elif name == 'straight':
            return StraightFlyAgent()
        else:
            raise NotImplementedError

class HierarchicalSingleCombatTask(SingleCombatTask):
    """
    分层单一战斗任务，扩展自单一战斗任务，增加了更复杂的动作和策略处理。
    适配 SAC 算法，支持连续动作空间。
    """
    def __init__(self, config):
        super().__init__(config)
        self.lowlevel_policy = BaselineActor()
        self.lowlevel_policy.load_state_dict(
            torch.load(get_root_dir() + '/model/baseline_model.pt', map_location=torch.device('cpu'), weights_only=True)
        )
        self.lowlevel_policy.eval()
        self.norm_delta_altitude = np.array([0.1, 0, -0.1])
        self.norm_delta_heading = np.array([-np.pi / 6, -np.pi / 12, 0, np.pi / 12, np.pi / 6])
        self.norm_delta_velocity = np.array([0.05, 0, -0.05])

    def load_action_space(self):
        """
        为分层任务加载动作空间，保留多离散动作空间用于高层次策略。
        """
        self.action_space = spaces.MultiDiscrete([3, 5, 3])

    def normalize_action(self, env, agent_id, action):
        """
        将高级动作转换为低级连续动作，适配 SAC。
        """
        if self.use_baseline and agent_id in env.enm_ids:
            action = self.baseline_agent.get_action(env.agents[agent_id])
            return action
        else:
            # 生成低级输入观察值
            raw_obs = self.get_obs(env, agent_id)
            input_obs = np.zeros(12, dtype=np.float32)
            # (1) delta altitude/heading/velocity
            input_obs[0] = self.norm_delta_altitude[action[0]]
            input_obs[1] = self.norm_delta_heading[action[1]]
            input_obs[2] = self.norm_delta_velocity[action[2]]
            # (2) ego info
            input_obs[3:12] = raw_obs[:9]
            input_obs = np.expand_dims(input_obs, axis=0)
            # 输出低级动作
            _action, _rnn_states = self.lowlevel_policy(input_obs, self._inner_rnn_states[agent_id])
            action = _action.detach().cpu().numpy().squeeze(0)
            self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()
            # 规范化低级动作到 Box 范围
            low = np.array([-1, -1, -1, 0.4], dtype=np.float32)
            high = np.array([1, 1, 1, 0.9], dtype=np.float32)
            norm_act = np.clip(action, low, high)
            return norm_act

    def reset(self, env):
        """
        任务特定的重置方法，包括奖励函数的重置以及内部RNN状态的初始化。
        """
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        return super().reset(env)

class StraightFlyAgent:
    """
    直线飞行代理类，简单控制飞行器直线飞行。
    输出连续动作适配 SAC。
    """
    def __init__(self):
        self.reset()

    def get_action(self, sim: AircraftSimulator):
        """
        获取直线飞行的连续动作。
        """
        action = np.array([0.0, -0.07, 0.0, 0.4], dtype=np.float32)  # 副翼=0, 升降舵=-0.07, 方向舵=0, 油门=0.4
        low = np.array([-1, -1, -1, 0.4], dtype=np.float32)
        high = np.array([1, 1, 1, 0.9], dtype=np.float32)
        return np.clip(action, low, high)

    def reset(self):
        """
        重置代理状态。
        """
        pass

class BaselineAgent:
    """
    基线代理类，定义了基础的飞行行为。
    输出连续动作适配 SAC。
    """
    def __init__(self):
        self.model_path = get_root_dir() + '/model/baseline_model.pt'
        self.actor = BaselineActor()
        self.actor.load_state_dict(
            torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True)
        )
        self.actor.eval()
        self.state_var = [
            c.delta_altitude,                   # 高度差
            c.delta_heading,                    # 航向差
            c.delta_velocities_u,               # 速度差
            c.attitude_roll_rad,                # 滚转角
            c.attitude_pitch_rad,               # 俯仰角
            c.velocities_u_mps,                 # 机体x轴速度
            c.velocities_v_mps,                 # 机体y轴速度
            c.velocities_w_mps,                 # 机体z轴速度
            c.velocities_vc_mps,                # 真空速
            c.position_h_sl_m                   # 海拔高度
        ]
        self.reset()

    def reset(self):
        """
        重置代理状态，初始化RNN状态。
        """
        self.rnn_states = np.zeros((1, 1, 128))

    def set_delta_value(self, sim: AircraftSimulator):
        """
        设置与目标敌机的相对位置和速度。
        """
        raise NotImplementedError

    def get_observation(self, sim: AircraftSimulator, delta_value):
        """
        获取当前模拟状态的观察值。
        """
        obs = sim.get_property_values(self.state_var)
        norm_obs = np.zeros(12, dtype=np.float32)
        norm_obs[0] = delta_value[0] / 1000          # 高度差（单位：公里）
        norm_obs[1] = in_range_rad(delta_value[1])   # 航向差（单位：弧度）
        norm_obs[2] = delta_value[2] / 340           # 速度差（单位：音速）
        norm_obs[3] = obs[9] / 5000                  # 海拔高度（单位：5公里）
        norm_obs[4] = np.sin(obs[3])                 # 滚转角正弦值
        norm_obs[5] = np.cos(obs[3])                 # 滚转角余弦值
        norm_obs[6] = np.sin(obs[4])                 # 俯仰角正弦值
        norm_obs[7] = np.cos(obs[4])                 # 俯仰角余弦值
        norm_obs[8] = obs[5] / 340                   # 机体x轴速度（单位：音速）
        norm_obs[9] = obs[6] / 340                   # 机体y轴速度（单位：音速）
        norm_obs[10] = obs[7] / 340                  # 机体z轴速度（单位：音速）
        norm_obs[11] = obs[8] / 340                  # 真空速（单位：音速）
        norm_obs = np.expand_dims(norm_obs, axis=0)  # 维度：(1,12)
        return norm_obs

    def get_action(self, sim: AircraftSimulator):
        """
        根据当前模拟状态和目标值获取连续动作。
        """
        delta_value = self.set_delta_value(sim)
        observation = self.get_observation(sim, delta_value)
        _action, self.rnn_states = self.actor(observation, self.rnn_states)
        action = _action.detach().cpu().numpy().squeeze()
        low = np.array([-1, -1, -1, 0.4], dtype=np.float32)
        high = np.array([1, 1, 1, 0.9], dtype=np.float32)
        return np.clip(action, low, high)

class PursueAgent(BaselineAgent):
    """
    追击代理类，继承自基线代理，专注于追踪敌机。
    """
    def __init__(self):
        super().__init__()

    def set_delta_value(self, sim: AircraftSimulator):
        """
        计算与目标敌机的相对位置和速度差。
        """
        # NOTE: 仅适用于1对1场景
        ego_x, ego_y, ego_z = sim.get_position()
        ego_vx, ego_vy, ego_vz = sim.get_velocity()
        enm_x, enm_y, enm_z = sim.enemies[0].get_position()
        # 高度差
        delta_altitude = enm_z - ego_z
        # 航向差
        ego_v = np.linalg.norm([ego_vx, ego_vy])
        delta_x, delta_y = enm_x - ego_x, enm_y - ego_y
        R = np.linalg.norm([delta_x, delta_y])
        proj_dist = delta_x * ego_vx + delta_y * ego_vy
        ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
        side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
        delta_heading = ego_AO * side_flag
        # 速度差
        delta_velocity = sim.enemies[0].get_property_value(c.velocities_u_mps) - \
                         sim.get_property_value(c.velocities_u_mps)
        return np.array([delta_altitude, delta_heading, delta_velocity], dtype=np.float32)

class ManeuverAgent(BaselineAgent):
    """
    机动代理类，继承自基线代理，可以进行特定的机动动作。
    """
    def __init__(self, maneuver: Literal['l', 'r', 'n']):
        super().__init__()
        self.turn_interval = 30
        self.dodge_missile = True  # 如果设置为真，当检测到导弹时开始转向
        if maneuver == 'l':
            self.target_heading_list = [0]
        elif maneuver == 'r':
            self.target_heading_list = [np.pi/2, np.pi/2, np.pi/2, np.pi/2]
        elif maneuver == 'n':
            self.target_heading_list = [np.pi, np.pi, np.pi, np.pi]
        self.target_altitude_list = [6096] * 4
        self.target_velocity_list = [243] * 4

    def reset(self):
        """
        重置代理状态，初始化步骤和RNN状态。
        """
        self.step = 0
        self.rnn_states = np.zeros((1, 1, 128))
        self.init_heading = None

    def set_delta_value(self, sim: AircraftSimulator):
        """
        根据设定的目标值和当前状态计算动作所需的改变值。
        """
        step_list = np.arange(1, len(self.target_heading_list)+1) * self.turn_interval / 0.2
        cur_heading = sim.get_property_value(c.attitude_heading_true_rad)
        if self.init_heading is None:
            self.init_heading = cur_heading
        if not self.dodge_missile or len(sim.under_missiles) != 0:
            for i, interval in enumerate(step_list):
                if self.step <= interval:
                    break
            delta_heading = self.init_heading + self.target_heading_list[i] - cur_heading
            delta_altitude = self.target_altitude_list[i] - sim.get_property_value(c.position_h_sl_m)
            delta_velocity = self.target_velocity_list[i] - sim.get_property_value(c.velocities_u_mps)
            self.step += 1
        else:
            delta_heading = self.init_heading - cur_heading
            delta_altitude = 6096 - sim.get_property_value(c.position_h_sl_m)
            delta_velocity = 243 - sim.get_property_value(c.velocities_u_mps)
        return np.array([delta_altitude, delta_heading, delta_velocity], dtype=np.float32)

class DodgeMissileAgent:
    """
    躲避导弹代理类，使用特定的模型处理导弹躲避行为。
    输出连续动作适配 SAC。
    """
    def __init__(self):
        self.model_path = get_root_dir() + '/model/dodge_missile_model.pt'
        self.actor = BaselineActor(input_dim=21, use_mlp_actlayer=True)
        self.actor.load_state_dict(
            torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True)
        )
        self.actor.eval()
        self.state_var = [
            c.position_long_gc_deg,             # 经度
            c.position_lat_geod_deg,            # 纬度
            c.position_h_sl_m,                  # 海拔高度
            c.attitude_roll_rad,                # 滚转角
            c.attitude_pitch_rad,               # 俯仰角
            c.attitude_heading_true_rad,        # 真航向
            c.velocities_v_north_mps,           # 北向速度
            c.velocities_v_east_mps,            # 东向速度
            c.velocities_v_down_mps,            # 下降速度
            c.velocities_u_mps,                 # 机体x轴速度
            c.velocities_v_mps,                 # 机体y轴速度
            c.velocities_w_mps,                 # 机体z轴速度
            c.velocities_vc_mps,                # 真空速
        ]
        self.reset()

    def get_observation(self, sim: AircraftSimulator):
        """
        获取当前模拟状态的观察值。
        """
        norm_obs = np.zeros(21, dtype=np.float32)
        ego_obs_list = np.array(sim.get_property_values(self.state_var))
        enm_obs_list = np.array(sim.enemies[0].get_property_values(self.state_var))
        # 提取特征：北向、东向、下降、北向速度、东向速度、下降速度
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], 120.0, 60.0, 0.0)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], 120.0, 60.0, 0.0)
        ego_feature = np.array([*ego_cur_ned, *(ego_obs_list[6:9])])
        enm_feature = np.array([*enm_cur_ned, *(enm_obs_list[6:9])])
        # 规范化 ego 信息
        norm_obs[0] = ego_obs_list[2] / 5000            # 海拔高度（单位：5公里）
        norm_obs[1] = np.sin(ego_obs_list[3])           # 滚转角正弦值
        norm_obs[2] = np.cos(ego_obs_list[3])           # 滚转角余弦值
        norm_obs[3] = np.sin(ego_obs_list[4])           # 俯仰角正弦值
        norm_obs[4] = np.cos(ego_obs_list[4])           # 俯仰角余弦值
        norm_obs[5] = ego_obs_list[9] / 340             # 机体x轴速度（单位：音速）
        norm_obs[6] = ego_obs_list[10] / 340            # 机体y轴速度（单位：音速）
        norm_obs[7] = ego_obs_list[11] / 340            # 机体z轴速度（单位：音速）
        norm_obs[8] = ego_obs_list[12] / 340            # 真空速（单位：音速）
        # 与敌机的相对信息
        ego_AO, ego_TA, R, side_flag = get2d_AO_TA_R(ego_feature, enm_feature, return_side=True)
        norm_obs[9] = (enm_obs_list[9] - ego_obs_list[9]) / 340
        norm_obs[10] = (enm_obs_list[2] - ego_obs_list[2]) / 1000
        norm_obs[11] = ego_AO
        norm_obs[12] = ego_TA
        norm_obs[13] = R / 10000
        norm_obs[14] = side_flag
        # 与导弹的相对信息
        if len(sim.under_missiles) != 0 and sim.under_missiles[0].is_alive:
            missile_sim = sim.under_missiles[0]
        else:
            missile_sim = None
        if missile_sim is not None:
            missile_feature = np.concatenate((missile_sim.get_position(), missile_sim.get_velocity()))
            ego_AO, ego_TA, R, side_flag = get2d_AO_TA_R(ego_feature, missile_feature, return_side=True)
            norm_obs[15] = (np.linalg.norm(missile_sim.get_velocity()) - ego_obs_list[9]) / 340
            norm_obs[16] = (missile_feature[2] - ego_obs_list[2]) / 1000
            norm_obs[17] = ego_AO
            norm_obs[18] = ego_TA
            norm_obs[19] = R / 10000
            norm_obs[20] = side_flag
        norm_obs = np.expand_dims(norm_obs, axis=0)
        return norm_obs

    def get_action(self, sim: AircraftSimulator):
        """
        根据当前观察值获取导弹躲避连续动作。
        """
        obs = self.get_observation(sim)
        _action, self.rnn_states = self.actor(obs, self.rnn_states)
        action = _action.detach().cpu().numpy().squeeze()
        low = np.array([-1, -1, -1, 0.4], dtype=np.float32)
        high = np.array([1, 1, 1, 0.9], dtype=np.float32)
        return np.clip(action, low, high)

    def reset(self):
        """
        重置代理状态，初始化RNN状态。
        """
        self.rnn_states = np.zeros((1, 1, 128))


