import gymnasium
from gymnasium.utils import seeding
import numpy as np
from typing import Dict, Any, Tuple
from ..core.simulatior import AircraftSimulator, BaseSimulator
from ..tasks.task_base import BaseTask
from ..utils.utils import parse_config
import logging


class BaseEnv(gymnasium.Env):
    """
    一个将JSBSim飞行动力学模块（FDM）封装成强化学习环境的类，符合OpenAI Gym Env接口。

    BaseEnv通过与一个具体的任务（Task）结合来模拟航空器的控制任务，任务包括各自特有的观察空间（observation_space）、动作空间（action_space）、奖励计算等。
    """
    metadata = {"render.modes": ["human", "txt"]}

    def __init__(self, config_name: str):
        """
        初始化环境，加载配置文件并设置模拟器参数。

        Args:
            config_name (str): 配置文件的名称，用于加载相应的配置。
        """
        # 解析配置文件并获取基本参数
        self.config = parse_config(config_name)
        self.max_steps = getattr(self.config, 'max_steps', 100)  # 最大步数
        self.sim_freq = getattr(self.config, 'sim_freq', 60)  # 仿真频率
        self.agent_interaction_steps = getattr(self.config, 'agent_interaction_steps', 12)  # 每次交互步数
        self.center_lon, self.center_lat, self.center_alt = \
            getattr(self.config, 'battle_field_center', (120.0, 60.0, 0.0))  # 作战区域的中心坐标
        self._create_records = False  # 标记是否创建记录
        self.load()  # 加载任务和模拟器

    @property
    def num_agents(self) -> int:
        """返回任务中代理的数量"""
        return self.task.num_agents

    @property
    def observation_space(self) -> gymnasium.Space:
        """返回任务的观察空间"""
        return self.task.observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        """返回任务的动作空间"""
        return self.task.action_space

    @property
    def agents(self) -> Dict[str, AircraftSimulator]:
        """返回所有代理的模拟器对象"""
        return self._jsbsims

    @property
    def time_interval(self) -> int:
        """返回每次交互步骤的时间间隔"""
        return self.agent_interaction_steps / self.sim_freq

    def load(self):
        """
        加载任务和模拟器，并设置随机种子。
        """
        self.load_task()  # 加载任务
        self.load_simulator()  # 加载模拟器
        self.seed()  # 设置随机种子

    def load_task(self):
        """加载具体的任务类（BaseTask）"""
        self.task = BaseTask(self.config)

    def load_simulator(self):
        """
        加载飞机模拟器。
        为每个飞机配置模拟器对象，并设置对应的合作伙伴和敌人。
        """
        self._jsbsims = {}  # 存储所有飞机模拟器的字典
        for uid, config in self.config.aircraft_configs.items():
            self._jsbsims[uid] = AircraftSimulator(
                uid=uid,
                color=config.get("color", "Red"),  # 设置飞机颜色
                model=config.get("model", "f16"),  # 设置飞机模型
                init_state=config.get("init_state"),  # 设置飞机初始状态
                origin=getattr(self.config, 'battle_field_center', (120.0, 60.0, 0.0)),  # 设置作战中心位置
                sim_freq=self.sim_freq,  # 仿真频率
                num_missiles=config.get("missile", 0))  # 配置飞机的导弹数量
        # 识别并分类队友和敌人
        _default_team_uid = list(self._jsbsims.keys())[0][0]
        self.ego_ids = [uid for uid in self._jsbsims.keys() if uid[0] == _default_team_uid]  # 队友id列表
        self.enm_ids = [uid for uid in self._jsbsims.keys() if uid[0] != _default_team_uid]  # 敌人id列表

        # 为每个模拟器配置其合作伙伴和敌人
        for key, sim in self._jsbsims.items():
            for k, s in self._jsbsims.items():
                if k == key:
                    pass  # 自己不需要连接
                elif k[0] == key[0]:
                    sim.partners.append(s)  # 同一队伍的其他模拟器为合作伙伴
                else:
                    sim.enemies.append(s)  # 不同队伍的模拟器为敌人

        self._tempsims = {}  # 临时模拟器字典

    def add_temp_simulator(self, sim: BaseSimulator):
        """
        添加临时模拟器（比如测试用或新建的模拟器）

        Args:
            sim (BaseSimulator): 新增的模拟器对象
        """
        self._tempsims[sim.uid] = sim

    def reset(self) -> np.ndarray:
        """
        重置环境状态并返回初始观察。

        Returns:
            np.ndarray: 初始的观察值
        """
        # 重置所有模拟器
        self.current_step = 0
        for sim in self._jsbsims.values():
            sim.reload()  # 重新加载每个模拟器的状态
        self._tempsims.clear()  # 清空临时模拟器

        # 重置任务状态
        self.task.reset(self)

        # 获取初始观察并返回
        obs = self.get_obs()
        return self._pack(obs)

    def _pack(self, data: Dict[str, Any]) -> np.ndarray:
        """Pack seperated key-value dict into grouped np.ndarray"""
        ego_data = np.array([data[uid] for uid in self.ego_ids])
        enm_data = np.array([data[uid] for uid in self.enm_ids])
        if enm_data.shape[0] > 0:
            data = np.concatenate((ego_data, enm_data))  # type: np.ndarray
        else:
            data = ego_data  # type: np.ndarray
        try:
            assert np.isnan(data).sum() == 0
        except AssertionError:
            import pdb
            pdb.set_trace()
        # only return data that belongs to RL agents
        return data[:self.num_agents, ...]

    def _unpack(self, data: np.ndarray) -> Dict[str, Any]:
        """Unpack grouped np.ndarray into seperated key-value dict"""
        assert isinstance(data, (np.ndarray, list, tuple)) and len(data) == self.num_agents
        # unpack data in the same order to packing process
        unpack_data = dict(zip((self.ego_ids + self.enm_ids)[:self.num_agents], data))
        # fill in None for other not-RL agents
        for agent_id in (self.ego_ids + self.enm_ids)[self.num_agents:]:
            unpack_data[agent_id] = None
        return unpack_data

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's observation. Accepts an action and
        returns a tuple (observation, reward_visualize, done, info).

        Args:
            action (np.ndarray): the agents' actions, allow opponent's action input

        Returns:
            (tuple):
                obs: agents' observation of the current environment
                rewards: amount of rewards returned after previous actions
                dones: whether the episode has ended, in which case further step() calls are undefined
                info: auxiliary information
        """
        self.current_step += 1
        info = {"current_step": self.current_step}
        logging.debug(f"Env step: Before action, info={info}")

        # apply actions
        action = self._unpack(action)
        for agent_id in self.agents.keys():
            a_action = self.task.normalize_action(self, agent_id, action[agent_id])
            self.agents[agent_id].set_property_values(self.task.action_var, a_action)

        # run simulation
        for _ in range(self.agent_interaction_steps):
            for sim in self._jsbsims.values():
                sim.run()
            for sim in self._tempsims.values():
                sim.run()

        # 调用 task.step，获取批量结果
        obs, rewards, dones, infos = self.task.step(self)
        logging.debug(f"Env step: After task.step, obs_shape={obs.shape}, infos={infos}")

        # 从 task.step 返回的 obs 中重新获取，确保一致性
        obs_dict = self.get_obs()
        # logging.info(f"Step: obs_dict type = {type(obs_dict)}, obs_dict = {obs_dict}")
        # 处理 dones 和 rewards，确保格式一致
        dones_dict = {}
        rewards_dict = {}
        success_dict = {}
        for agent_id in self.agents.keys():
            done, success, info = self.task.get_termination(self, agent_id, infos[0])
            dones_dict[agent_id] = [done]
            success_dict[agent_id] = [success]
            reward, info = self.task.get_reward(self, agent_id, info)
            rewards_dict[agent_id] = [reward]
            info['current_step'] = self.current_step  # 确保 info 中有 current_step

        logging.debug(f"Env step: Final info={info}")
        obs = self._pack(obs_dict)
        # logging.info(f"Step: packed_obs type = {type(obs)}, packed_obs shape = {obs.shape}, packed_obs = {obs}")
        return self._pack(obs_dict), self._pack(rewards_dict), self._pack(dones_dict), info

    def get_obs(self):
        """
        返回所有代理的观察数据。

        备注：在去中心化执行时，每个代理只应访问它自己的观察。
        """
        return dict([(agent_id, self.task.get_obs(self, agent_id)) for agent_id in self.agents.keys()])

    def get_state(self):
        """
        返回全局状态，通常在去中心化执行时不应使用此函数。

        备注：该函数返回所有代理的观察值的合并。
        """
        state = np.hstack([self.task.get_obs(self, agent_id) for agent_id in self.agents.keys()])
        return dict([(agent_id, state.copy()) for agent_id in self.agents.keys()])

    def close(self):
        """
        清理环境中的所有对象。

        备注：当环境被垃圾回收或程序退出时，会自动调用close()。
        """
        for sim in self._jsbsims.values():
            sim.close()  # 关闭每个模拟器
        for sim in self._tempsims.values():
            sim.close()  # 关闭临时模拟器
        self._jsbsims.clear()  # 清空模拟器字典
        self._tempsims.clear()  # 清空临时模拟器字典

    def render(self, mode="txt", filepath='./JSBSimRecording.txt.acmi'):
        """
        渲染环境。

        每个环境支持的渲染模式可能有所不同（某些环境可能完全不支持渲染）。
        根据约定，如果 mode 参数值为：

        - "human": 在终端打印输出。
        - "txt": 输出到 .txt.acmi 文件中。

        注意：
            确保类的元数据 'render.modes' 键包含支持的渲染模式列表。
            建议在实现此方法时调用 super()，以利用其现有功能。

        参数：
            mode (str): 渲染的模式。
            filepath (str): 当 mode 为 "txt" 时，文件保存路径。
        """
        if mode == "txt":
            # 检查是否需要创建记录文件
            if not self._create_records:
                with open(filepath, mode='w', encoding='utf-8-sig') as f:
                    f.write("FileType=text/acmi/tacview\n")
                    f.write("FileVersion=2.1\n")
                    f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")
                self._create_records = True
            # 写入当前时间步和仿真器日志信息
            with open(filepath, mode='a', encoding='utf-8-sig') as f:
                timestamp = self.current_step * self.time_interval  # 计算时间戳
                f.write(f"#{timestamp:.2f}\n")  # 写入时间戳
                # 遍历所有仿真器对象，记录日志信息
                for sim in self._jsbsims.values():
                    log_msg = sim.log()
                    if log_msg is not None:
                        f.write(log_msg + "\n")
                for sim in self._tempsims.values():
                    log_msg = sim.log()
                    if log_msg is not None:
                        f.write(log_msg + "\n")
        # TODO: 支持实时渲染（例如使用 FlightGear）
        else:
            raise NotImplementedError

    def seed(self, seed=None):
        """
        设置环境的随机数生成器种子。

        注意：
            某些环境可能使用多个伪随机数生成器。
            我们需要捕获所有使用的种子以确保不同生成器之间没有意外的相关性。

        参数：
            seed (int 或 None): 用于设置种子的值。如果为 None，将生成随机种子。

        返回：
            list[int]: 返回环境中所有随机数生成器使用的种子列表。
            列表中的第一个值通常是主要种子值，复现实验时需要传递此值。
        """
        self.np_random, seed = seeding.np_random(seed)  # 初始化随机数生成器
        return [seed]

    def get_agent_ids(self):
        return list(self.agents.keys())