import numpy as np
from gymnasium import spaces
from typing import List, Tuple
from abc import ABC, abstractmethod
from ..core.catalog import Catalog as c


from abc import ABC, abstractmethod
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict, Any
from ..core.catalog import Catalog as c

class BaseTask(ABC):
    """
    基础任务类。
    此类可被继承，用于创建具有自定义观察变量、动作变量、终止条件和奖励函数的任务。
    """

    def __init__(self, config):
        self.config = config
        self.reward_functions = []
        self.termination_conditions = []
        self.load_variables()
        self.load_observation_space()
        self.load_action_space()

    @property
    def num_agents(self):
        return 1

    @abstractmethod
    def load_variables(self):
        """加载状态变量和动作变量"""
        self.state_var = [
            c.position_long_gc_deg,  # 地理坐标经度
            c.position_lat_geod_deg,  # 地理坐标纬度
            c.position_h_sl_m,  # 海拔高度
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,  # 副翼控制命令
            c.fcs_elevator_cmd_norm,  # 升降舵控制命令
            c.fcs_rudder_cmd_norm,  # 方向舵控制命令
            c.fcs_throttle_cmd_norm,  # 油门控制命令
        ]

    @abstractmethod
    def load_observation_space(self):
        """定义观察空间"""
        self.observation_space = spaces.Discrete(5)

    @abstractmethod
    def load_action_space(self):
        """定义动作空间"""
        self.action_space = spaces.Discrete(5)

    def reset(self, env):
        """任务特定的重置方法

        Args:
            env: 环境实例
        """
        for reward_function in self.reward_functions:
            reward_function.reset(self, env)

    def step(self, env):
        """任务特定的步骤方法

        Args:
            env: 环境实例
        """
        pass

    def get_reward(self, env, agent_id, info={}) -> Tuple[float, dict]:
        """
        聚合奖励函数

        Args:
            env: 环境实例
            agent_id: 当前代理的ID
            info: 附加信息

        Returns:
            (tuple):
                reward(float): 当前时间步的总奖励
                info(dict): 附加信息
        """
        from ..reward_functions import RadarLockReward, MissileHitReward
        reward = 0.0
        state_dict = self.get_state_dict(env, agent_id) if hasattr(self, 'get_state_dict') else {}
        for reward_function in self.reward_functions:
            if isinstance(reward_function, (RadarLockReward, MissileHitReward)):
                reward_info = reward_function.get_reward(self, env, agent_id, state_dict)
                reward_value = reward_info[0] if isinstance(reward_info, (tuple, list)) else reward_info
            else:
                reward_info = reward_function.get_reward(self, env, agent_id)
                reward_value = reward_info[0] if isinstance(reward_info, (tuple, list)) else reward_info
            reward += reward_value
        return reward, info

    def get_termination(self, env, agent_id, info={}) -> Tuple[bool, dict]:
        """
        聚合终止条件

        Args:
            env: 环境实例
            agent_id: 当前代理的ID
            info: 附加信息

        Returns:
            (tuple):
                done(bool): 本集是否已结束
                info(dict): 附加信息
        """
        done = False
        success = True
        for condition in self.termination_conditions:
            d, s, info = condition.get_termination(self, env, agent_id, info)
            done = done or d
            success = success and s
            if done:
                break
        return done, info

    def get_obs(self, env, agent_id):
        """从环境中提取特定代理的有用信息。"""
        return np.zeros(2)

    def normalize_action(self, env, agent_id, action):
        """将动作标准化，以符合动作空间。"""
        return np.array(action)


# import logging
# import numpy as np
# from gymnasium import spaces
# from typing import List, Tuple
# from abc import ABC, abstractmethod
# from ..core.catalog import Catalog as c
#
#
# class BaseTask(ABC):
#     """
#     基础任务类。
#     此类可被继承，用于创建具有自定义观察变量、动作变量、终止条件和奖励函数的任务。
#     """
#
#     def __init__(self, config):
#         self.config = config
#         self.reward_functions = []
#         self.termination_conditions = []
#         self.load_variables()
#         self.load_observation_space()
#         self.load_action_space()
#
#     @property
#     def num_agents(self):
#         return 1
#
#     @abstractmethod
#     def load_variables(self):
#         """加载状态变量和动作变量"""
#         self.state_var = [
#             c.position_long_gc_deg,  # 地理坐标经度
#             c.position_lat_geod_deg,  # 地理坐标纬度
#             c.position_h_sl_m,  # 海拔高度
#         ]
#         self.action_var = [
#             c.fcs_aileron_cmd_norm,  # 副翼控制命令
#             c.fcs_elevator_cmd_norm,  # 升降舵控制命令
#             c.fcs_rudder_cmd_norm,  # 方向舵控制命令
#             c.fcs_throttle_cmd_norm,  # 油门控制命令
#         ]
#
#     @abstractmethod
#     def load_observation_space(self):
#         """定义观察空间"""
#         self.observation_space = spaces.Discrete(5)
#
#     @abstractmethod
#     def load_action_space(self):
#         """定义动作空间"""
#         self.action_space = spaces.Discrete(5)
#
#     def reset(self, env):
#         """任务特定的重置方法
#
#         Args:
#             env: 环境实例
#         """
#         for reward_function in self.reward_functions:
#             reward_function.reset(self, env)
#
#     def step(self, env):
#         logging.debug(f"BaseTask.step: env_agents={list(env.agents.keys())}")
#
#         # 动态获取代理数量
#         num_agents = len(env.agents)
#         n_rollout_threads = getattr(env, 'n_rollout_threads', 1)
#
#         # 初始化数组，考虑实际代理数量
#         obs_dim = self.observation_space.shape[0] if hasattr(self.observation_space, 'shape') else 1
#         obs = np.zeros((n_rollout_threads, num_agents, obs_dim), dtype=np.float32)
#         rewards = np.zeros((n_rollout_threads, num_agents, 1), dtype=np.float32)
#         dones = np.zeros((n_rollout_threads, num_agents, 1), dtype=np.bool_)
#         # 初始化 infos，确保包含 current_step
#         infos = [{"current_step": env.current_step} for _ in range(n_rollout_threads)]
#
#         # 收集观察、奖励和终止标志
#         for i, agent_id in enumerate(env.agents):
#             obs[:, i, :] = self.get_obs(env, agent_id)
#             reward, info = self.get_reward(env, agent_id, infos[0])
#             # 确保 reward 是一个数组
#             if np.isscalar(reward):
#                 reward = np.array([[reward]])  # 转换为 (1, 1) 数组
#             elif reward.ndim == 0:
#                 reward = reward.reshape(1, 1)
#             # 确保 reward 形状匹配 (n_rollout_threads, 1)
#             if reward.shape != (n_rollout_threads, 1):
#                 logging.error(f"Reward shape mismatch: got {reward.shape}, expected {(n_rollout_threads, 1)}")
#                 reward = np.full((n_rollout_threads, 1), reward, dtype=np.float32)
#             rewards[:, i, :] = reward
#             done, success, info = self.get_termination(env, agent_id, info)
#             dones[:, i, :] = done
#             # 更新所有线程的 info
#             for j in range(n_rollout_threads):
#                 infos[j].update(info)
#                 infos[j]['current_step'] = env.current_step  # 确保 current_step 不被覆盖
#
#         logging.debug(f"BaseTask.step: obs_shape={obs.shape}, rewards_shape={rewards.shape}, dones_shape={dones.shape}")
#         return obs, rewards, dones, infos
#
#     def get_reward(self, env, agent_id, info={}) -> Tuple[float, dict]:
#         """
#         聚合奖励函数
#
#         Args:
#             env: 环境实例
#             agent_id: 当前代理的ID
#             info: 附加信息
#
#         Returns:
#             (tuple):
#                 reward(float): 当前时间步的总奖励
#                 info(dict): 附加信息
#         """
#         reward = 0.0
#         for reward_function in self.reward_functions:
#             reward += reward_function.get_reward(self, env, agent_id)
#         # 确保 info 中包含 current_step
#         info['current_step'] = env.current_step
#         return reward, info
#
#     def get_termination(self, env, agent_id, info={}) -> Tuple[bool, bool, dict]:
#         """
#         聚合终止条件
#
#         Args:
#             env: 环境实例
#             agent_id: 当前代理的ID
#             info: 附加信息
#
#         Returns:
#             (tuple):
#                 done(bool): 本集是否已结束
#                 success(bool): 本集是否成功
#                 info(dict): 附加信息
#         """
#         done = False
#         success = True
#         # 确保 info 中包含 current_step
#         info['current_step'] = env.current_step
#         for condition in self.termination_conditions:
#             d, s, info = condition.get_termination(self, env, agent_id, info)
#             done = done or d
#             success = success and s
#             info['current_step'] = env.current_step  # 防止被覆盖
#             if done:
#                 break
#         return done, success, info
#
#     def get_obs(self, env, agent_id):
#         """从环境中提取特定代理的有用信息。"""
#         return np.zeros(self.observation_space.shape[0])
#
#     def normalize_action(self, env, agent_id, action):
#         """将动作标准化，以符合动作空间。"""
#         return np.array(action)