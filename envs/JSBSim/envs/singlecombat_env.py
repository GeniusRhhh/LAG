from typing import Tuple
import logging
import numpy as np  # 导入 NumPy 库，用于处理数值数据和数组
from .env_base import BaseEnv  # 从 env_base 模块导入 BaseEnv 类，作为环境基类
from ..tasks import SingleCombatTask, SingleCombatDodgeMissileTask, HierarchicalSingleCombatDodgeMissileTask, \
    HierarchicalSingleCombatShootTask, SingleCombatShootMissileTask, HierarchicalSingleCombatTask  # 导入不同的任务类
import time
class SingleCombatEnv(BaseEnv):
    """
    SingleCombatEnv 是一个一对一的竞技环境。
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)  # 调用父类构造函数进行初始化
        # 环境特定的初始化
        assert len(self.agents.keys()) == 2, f"{self.__class__.__name__} 只支持 1v1 场景!"  # 确保环境中只有两个代理（即 1v1 场景）
        self.init_states = None  # 初始化保存代理初始状态的变量

    def load_task(self):
        """
        加载任务，根据配置文件中指定的任务名称加载不同的任务类型
        """
        taskname = getattr(self.config, 'task', None)  # 从配置文件中获取任务名称
        if taskname == 'singlecombat':  # 如果任务是单一战斗
            self.task = SingleCombatTask(self.config)  # 初始化单一战斗任务
        elif taskname == 'hierarchical_singlecombat':  # 如果任务是层级单一战斗
            self.task = HierarchicalSingleCombatTask(self.config)  # 初始化层级单一战斗任务
        elif taskname == 'singlecombat_dodge_missile':  # 如果任务是单一战斗躲避导弹
            self.task = SingleCombatDodgeMissileTask(self.config)  # 初始化单一战斗躲避导弹任务
        elif taskname == 'singlecombat_shoot':  # 如果任务是单一战斗射击
            self.task = SingleCombatShootMissileTask(self.config)  # 初始化单一战斗射击导弹任务
        elif taskname == 'hierarchical_singlecombat_dodge_missile':  # 如果任务是层级单一战斗躲避导弹
            self.task = HierarchicalSingleCombatDodgeMissileTask(self.config)  # 初始化层级单一战斗躲避导弹任务
        elif taskname == 'hierarchical_singlecombat_shoot':  # 如果任务是层级单一战斗射击
            self.task = HierarchicalSingleCombatShootTask(self.config)  # 初始化层级单一战斗射击任务
        else:
            raise NotImplementedError(f"未知的任务名称: {taskname}")  # 如果任务名称不在已定义的任务中，抛出异常

    def reset(self) -> np.ndarray:
        """
        重置环境，初始化状态并返回代理的观察数据。
        """
        self.current_step = 0  # 将当前步骤初始化为 0
        self.reset_simulators()  # 重置所有模拟器
        self.task.reset(self)  # 重置当前任务
        obs = self.get_obs()  # 获取代理的当前观察数据
        return self._pack(obs)  # 将观察数据打包并返回

    def reset_simulators(self):
        """
        重置所有模拟器的状态，并进行代理状态的切换。
        """
        # 如果 init_states 为 None（即第一次重置），则保存每个代理的初始状态
        if self.init_states is None:
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]  # 获取所有代理的初始状态

        init_states = self.init_states.copy()  # 复制初始状态
        self.np_random.shuffle(init_states)  # 随机打乱初始状态的顺序，以便模拟器的状态交换
        for idx, sim in enumerate(self.agents.values()):  # 遍历所有代理
            sim.reload(init_states[idx])  # 为每个代理加载对应的初始状态
        self._tempsims.clear()  # 清空临时模拟器数据

    # def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    #     self.current_step += 1
    #     info = {"current_step": self.current_step}
    #     action = self._unpack(action)
    #     for agent_id in self.agents.keys():
    #         a_action = self.task.normalize_action(self, agent_id, action[agent_id])
    #         self.agents[agent_id].set_property_values(self.task.action_var, a_action)
    #     for _ in range(self.agent_interaction_steps):
    #         for sim in self._jsbsims.values():
    #             sim.run()
    #         for sim in self._tempsims.values():
    #             sim.run()
    #     self.task.step(self)
    #
    #     obs = self.get_obs()
    #     dones = {}
    #     for agent_id in self.agents.keys():
    #         termination_result = self.task.get_termination(self, agent_id, info)
    #         if len(termination_result) == 2:
    #             done, info = termination_result
    #         elif len(termination_result) >= 2:
    #             done, info = termination_result[0], {**info, **termination_result[1]}
    #         else:
    #             raise ValueError(f"get_termination returned {len(termination_result)} values, expected at least 2")
    #         dones[agent_id] = [done]
    #
    #     rewards = {}
    #     for agent_id in self.agents.keys():
    #         reward, info = self.task.get_reward(self, agent_id, info)
    #         rewards[agent_id] = [reward]
    #
    #     return self._pack(obs), self._pack(rewards), self._pack(dones), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        logging.debug("Entering SingleCombatEnv.step...")
        start_time = time.time()
        self.current_step += 1
        info = {"current_step": self.current_step}
        action = self._unpack(action)
        logging.debug(f"Unpacked action in {time.time() - start_time:.2f}s: {action}")

        start_action = time.time()
        for agent_id in self.agents.keys():
            a_action = self.task.normalize_action(self, agent_id, action[agent_id])
            logging.debug(f"Normalized action for {agent_id}: shape={a_action.shape}")
            for i in range(self.n_rollout_threads):
                logging.debug(f"Setting action for {agent_id}, thread {i}: {a_action[i]}")
                self.agents[agent_id].set_property_values(self.task.action_var, a_action[i])
        logging.debug(f"Actions set in {time.time() - start_action:.2f}s")

        start_sim = time.time()
        for _ in range(self.agent_interaction_steps):
            for sim in self._jsbsims.values():
                sim.run()
            for sim in self._tempsims.values():
                sim.run()
        logging.debug(f"Simulators finished running in {time.time() - start_sim:.2f}s")

        start_task = time.time()
        self.task.step(self)
        logging.debug(f"Task stepped in {time.time() - start_task:.2f}s")

        obs = self.get_obs()
        logging.debug(f"Observations obtained: {obs}")
        dones = {}
        for agent_id in self.agents.keys():
            termination_result = self.task.get_termination(self, agent_id, info)
            if len(termination_result) == 2:
                done, info = termination_result
            elif len(termination_result) >= 2:
                done, info = termination_result[0], {**info, **termination_result[1]}
            else:
                raise ValueError(f"get_termination returned {len(termination_result)} values, expected at least 2")
            dones[agent_id] = [done]
        logging.debug(f"Dones: {dones}")

        rewards = {}
        for agent_id in self.agents.keys():
            reward, info = self.task.get_reward(self, agent_id, info)
            rewards[agent_id] = [reward]
        logging.debug(f"Rewards: {rewards}")

        logging.debug(f"Exiting SingleCombatEnv.step in {time.time() - start_time:.2f}s")
        return self._pack(obs), self._pack(rewards), self._pack(dones), info