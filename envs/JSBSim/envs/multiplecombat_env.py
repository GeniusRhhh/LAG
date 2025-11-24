# 导入必要的库，用于数值计算、配置文件处理、日志记录和类型提示
import numpy as np  # 用于高效的数值计算
import yaml  # 用于读取 YAML 配置文件
import os  # 用于操作系统相关的功能，如文件路径处理
import logging  # 用于记录日志，便于调试和监控
from typing import Tuple, Dict, Any  # 用于类型提示，提高代码可读性和维护性
from gymnasium import spaces  # Gymnasium 库，用于定义强化学习环境的观测和动作空间
from ..core.simulatior import AircraftSimulator, BaseSimulator  # 导入核心模拟器类
from ..core.catalog import Catalog as c  # 导入 Catalog 类，包含状态和动作的变量定义
from ..tasks import TacticalTemplateTestTask
from ..tasks.multiplecombat_task import (
    HierarchicalMultipleCombatShootTask,
    HierarchicalMultipleCombatTask,
    MultipleCombatTask
)  # 导入多种空战任务类
from ..utils.TacviewRenderer import TacviewRenderer  # 导入 Tacview 可视化工具
from .env_base import BaseEnv  # 导入基础环境类


class MultipleCombatEnv(BaseEnv):
    """2v2 多智能体空战环境，继承自 BaseEnv，支持基于战术模板的时间线流程。"""

    def __init__(self, config_name: str):
        """初始化环境，加载配置文件并设置战术参数、时间线和渲染器。

        Args:
            config_name (str): 配置文件名称，用于加载环境参数
        """
        super().__init__(config_name)  # 调用父类 BaseEnv 的初始化方法
        self._create_records = False  # 是否创建记录，默认为 False
        self.renderer = TacviewRenderer(filepath=None)  # 初始化 Tacview 渲染器，用于可视化仿真

        # 记录初始化日志，包含交互步数和时间间隔
        logging.info(f"Initialized MultipleCombatEnv with agent_interaction_steps={self.agent_interaction_steps}, "
                     f"time_interval={self.time_interval}")

    @property
    def share_observation_space(self) -> spaces.Space:
        """获取共享观测空间。

        Returns:
            spaces.Space: 任务定义的共享观测空间
        """
        return self.task.share_observation_space

    def load_task(self):
        """根据配置文件加载任务类型，并设置战术距离参数。"""
        taskname = getattr(self.config, 'task', None)  # 获取配置文件中的任务名称
        if taskname == 'multiplecombat':
            self.task = MultipleCombatTask(self.config)  # 加载基础多智能体空战任务
        elif taskname == 'hierarchical_multiplecombat':
            self.task = HierarchicalMultipleCombatTask(self.config)  # 加载分层多智能体任务
        elif taskname == 'hierarchical_multiplecombat_shoot':
            self.task = HierarchicalMultipleCombatShootTask(self.config)  # 加载分层射击任务
        elif taskname == 'tactical_template_test':
            self.task = TacticalTemplateTestTask(self.config)
        elif taskname == "pure_maneuver":  # 新增
            from ..tasks import PureManeuverTask
            self.task = PureManeuverTask(self.config)
        else:
            logging.error(f"Unknown task name: {taskname}")  # 记录错误日志
            raise NotImplementedError(f"Unknown taskname: {taskname}")  # 抛出未实现异常

        logging.info(f"Loaded task: {taskname}")

    def load_simulator(self):
        """加载并初始化飞机模拟器，设置编队和敌我关系。"""
        super().load_simulator()  # 调用父类的模拟器加载方法
        self._jsbsims.clear()  # 清空现有的模拟器列表

        # 为每个飞机配置初始化模拟器
        for uid, config in self.config.aircraft_configs.items():
            init_state = config.get("init_state", {})  # 获取初始状态
            self._jsbsims[uid] = AircraftSimulator(
                uid=uid,  # 飞机唯一标识
                color=config.get("color", "Red"),  # 飞机颜色（红方或蓝方）
                model=config.get("model", "f16"),  # 飞机模型，默认为 F-16
                init_state=init_state,  # 初始状态
                origin=getattr(self.config, 'battle_field_center', (120.0, 60.0, 0.0)),  # 战场中心坐标
                sim_freq=self.sim_freq,  # 仿真频率
                num_missiles=config.get("missile", 2)  # 初始导弹数量
            )
            # 获取飞机的初始位置（经度、纬度、高度）
            state = self._jsbsims[uid].get_property_values([
                c.position_long_gc_deg,  # 经度
                c.position_lat_geod_deg,  # 纬度
                c.position_h_sl_ft  # 海拔高度（英尺）
            ])
            logging.debug(f"Initialized {uid} at position: Long={state[0]}, Lat={state[1]}, Alt={state[2]} ft")

        # 设置编队和敌我关系
        self._setup_formation_relationships()

        # 设置长机和僚机角色
        for agent_id, sim in self._jsbsims.items():
            is_leader = agent_id.endswith("100")  # 如果 ID 以 "100" 结尾，则为长机
            sim.set_leader(is_leader)  # 设置长机/僚机状态
            logging.info(f"Initialized Agent {agent_id} as {'Leader' if is_leader else 'Wingman'}")

        self._tempsims = {}  # 初始化临时模拟器字典（如导弹）
        logging.info(f"Loaded simulators: {list(self._jsbsims.keys())}")

    def _setup_formation_relationships(self):
        """设置红方和蓝方的编队关系以及敌我关系。支持1v1和2v2配置"""
        available_agents = list(self._jsbsims.keys())
        logging.info(f"Setting up formation relationships for agents: {available_agents}")

        # 检查是否是1v1配置
        if len(available_agents) == 2 and "A0100" in available_agents and "B0100" in available_agents:
            # 1v1配置
            self._jsbsims["A0100"].partners = []  # 没有队友
            self._jsbsims["A0100"].enemies = [self._jsbsims["B0100"]]
            self._jsbsims["B0100"].partners = []  # 没有队友
            self._jsbsims["B0100"].enemies = [self._jsbsims["A0100"]]
            logging.info("1v1 formation relationships established: A0100 vs B0100")

        # 检查是否是2v2配置
        elif len(available_agents) == 4 and all(
                agent in available_agents for agent in ["A0100", "A0200", "B0100", "B0200"]):
            # 原有的2v2配置
            # 红方编队（A0100 和 A0200）
            self._jsbsims["A0100"].partners = [self._jsbsims["A0200"]]
            self._jsbsims["A0100"].enemies = [self._jsbsims["B0100"], self._jsbsims["B0200"]]
            self._jsbsims["A0200"].partners = [self._jsbsims["A0100"]]
            self._jsbsims["A0200"].enemies = [self._jsbsims["B0100"], self._jsbsims["B0200"]]

            # 蓝方编队（B0100 和 B0200）
            self._jsbsims["B0100"].partners = [self._jsbsims["B0200"]]
            self._jsbsims["B0100"].enemies = [self._jsbsims["A0100"], self._jsbsims["A0200"]]
            self._jsbsims["B0200"].partners = [self._jsbsims["B0100"]]
            self._jsbsims["B0200"].enemies = [self._jsbsims["A0100"], self._jsbsims["A0200"]]
            logging.info("2v2 formation relationships established")

        else:
            # 未知配置，报错
            logging.error(f"Unsupported agent configuration: {available_agents}")
            raise ValueError(
                f"Unsupported agent configuration. Expected 1v1 (A0100, B0100) or 2v2 (A0100, A0200, B0100, B0200), got: {available_agents}")
    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """重置环境状态，包括时间线、战术态势和模拟器。

        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]: 观测值和共享观测值
        """
        self.current_step = 0  # 重置当前步数
        self.agent_interaction_steps = 12  # 设置智能体交互步数

        self.reset_simulators()  # 重置所有模拟器
        self.task.reset(self)  # 重置任务状态
        obs = self.get_obs()  # 获取观测值
        share_obs = self.get_state()  # 获取共享状态

        # 记录重置日志
        logging.info(f"Environment reset, current_step={self.current_step}, "
                     f"agent_interaction_steps={self.agent_interaction_steps}, "
                     f"time_interval={self.time_interval}")
        return self._pack(obs), self._pack(share_obs)  # 打包观测值和共享观测值

    def reset_simulators(self):
        """重置所有飞机模拟器的状态。"""
        for sim in self._jsbsims.values():
            sim.reload()  # 重新加载模拟器状态
            # 获取重置后的位置
            state = sim.get_property_values([
                c.position_long_gc_deg,
                c.position_lat_geod_deg,
                c.position_h_sl_ft
            ])
            logging.debug(f"Reset {sim.uid} at position: Long={state[0]}, Lat={state[1]}, Alt={state[2]} ft")
        self._tempsims.clear()  # 清空临时模拟器（如导弹）
        if hasattr(self, '_finished_missiles'):
            self._finished_missiles.clear()  # 清空已结束的导弹
        if hasattr(self, '_missiles_pending_cleanup'):
            self._missiles_pending_cleanup.clear()  # 清空待清理导弹列表
        logging.info("All simulators reset")

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """执行一步仿真，处理动作、更新时间线和战术态势，并返回观测、奖励等。

        Args:
            action (np.ndarray): 智能体的动作数组

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, dict]: 观测、共享观测、奖励、终止标志和附加信息
        """
        self.current_step += 1  # 增加步数计数
        info = {"current_step": self.current_step, "bloods": {}, "missiles": {}}  # 初始化信息字典

        # 验证动作维度的正确性
        if isinstance(action, np.ndarray):
            logging.debug(f"Step action shape: {action.shape}, n_rollout_threads: {self.n_rollout_threads}")
            if action.ndim == 2 and action.shape[0] == self.num_agents and action.shape[1] == self.action_space.shape[0]:
                if self.n_rollout_threads == 1:
                    action = action[np.newaxis, :, :]  # 为单线程扩展维度
                else:
                    raise ValueError(
                        f"Expected shape (n_rollout_threads, {self.num_agents}, {self.action_space.shape[0]}), got {action.shape}")
            elif action.ndim != 3 or action.shape[1] != self.num_agents:
                raise ValueError(
                    f"Invalid action shape {action.shape}, expected (n_rollout_threads, {self.num_agents}, {self.action_space.shape[0]})")

        action_dict = self._unpack(action)  # 将动作数组解包为字典
        last_positions = {aid: self._jsbsims[aid].get_position() for aid in self._jsbsims}  # 保存当前位置

        # 为每个智能体执行动作
        for agent_id in self._jsbsims.keys():
            agent_action = action_dict[agent_id]
            if agent_action.ndim > 1:
                agent_action = agent_action.squeeze()  # 压缩多余维度

            # 检查动作空间维度
            if hasattr(self.task, 'action_space') and hasattr(self.task.action_space, 'nvec'):
                expected_dim = len(self.task.action_space.nvec)
            else:
                expected_dim = 2  # 默认分层动作空间维度

            if agent_action.shape != (expected_dim,):
                logging.error(f"Agent {agent_id} action shape {agent_action.shape}, expected ({expected_dim},)")
                raise ValueError(f"Agent {agent_id} action shape {agent_action.shape}, expected ({expected_dim},)")

            logging.debug(f"Agent {agent_id} action: {agent_action}")
            norm_action = self.task.normalize_action(self, agent_id, agent_action)  # 归一化动作
            self._jsbsims[agent_id].set_property_values(self.task.action_var, norm_action)  # 设置模拟器动作

        # 执行仿真步骤
        self.agent_interaction_steps = 12  # 设置交互步数
        logging.debug(f"Agent interaction steps: {self.agent_interaction_steps}")

        for _ in range(self.agent_interaction_steps):
            for sim in self._jsbsims.values():
                sim.run()  # 运行飞机模拟器
            for sim in self._tempsims.values():
                sim.run()  # 运行临时模拟器（如导弹）

        # === 导弹清理逻辑 ===
        # 标记已完成的导弹，但延迟一帧清理以确保爆炸效果正确渲染
        if not hasattr(self, '_missiles_pending_cleanup'):
            self._missiles_pending_cleanup = {}
        
        # 检查新完成的导弹
        for uid, sim in self._tempsims.items():
            if hasattr(sim, 'is_done') and sim.is_done:
                if uid not in self._missiles_pending_cleanup:
                    # 标记为待清理，记录完成帧
                    self._missiles_pending_cleanup[uid] = self.current_step
                    logging.debug(f"Missile {uid} completed at step {self.current_step}, marked for delayed cleanup")
        
        # 清理已完成且延迟足够的导弹（延迟2帧确保爆炸效果渲染）
        missiles_to_remove = []
        for uid, completion_step in list(self._missiles_pending_cleanup.items()):
            if self.current_step >= completion_step + 2:  # 延迟2帧
                if uid in self._tempsims:  # 确保导弹还在
                    missiles_to_remove.append(uid)
                    logging.debug(f"Cleaning up missile {uid} (completed at step {completion_step}, current {self.current_step})")
                # 从待清理列表中移除
                del self._missiles_pending_cleanup[uid]
        
        # 移除延迟清理的导弹
        for uid in missiles_to_remove:
            removed_sim = self._tempsims.pop(uid, None)
            if removed_sim:
                logging.info(f"🚀 Removed completed missile: {uid} (delayed cleanup after explosion)")
                # 从发射平台的导弹列表中移除
                if hasattr(removed_sim, 'parent_aircraft') and removed_sim.parent_aircraft:
                    if hasattr(removed_sim.parent_aircraft, 'launch_missiles'):
                        if removed_sim in removed_sim.parent_aircraft.launch_missiles:
                            removed_sim.parent_aircraft.launch_missiles.remove(removed_sim)
                # 从目标的威胁导弹列表中移除
                if hasattr(removed_sim, 'target_aircraft') and removed_sim.target_aircraft:
                    if hasattr(removed_sim.target_aircraft, 'under_missiles'):
                        if removed_sim in removed_sim.target_aircraft.under_missiles:
                            removed_sim.target_aircraft.under_missiles.remove(removed_sim)
        
        if missiles_to_remove:
            logging.info(f"Cleaned {len(missiles_to_remove)} completed missiles from _tempsims")

        # 检查位置跳跃（防止仿真错误）
        for agent_id in self._jsbsims:
            pos = self._jsbsims[agent_id].get_position()
            delta = np.linalg.norm(pos - last_positions[agent_id])
            if delta > 1000:
                logging.warning(f"Step {self.current_step}, Agent {agent_id} jumped: "
                               f"{last_positions[agent_id]} -> {pos}, delta={delta:.2f}m")

        # 执行任务步骤，获取观测、奖励等
        obs, share_obs, rewards, dones, infos = self.task.step(self)

        # 验证观测值，检查是否有 NaN
        for agent_id, o in obs.items():
            if np.any(np.isnan(o)):
                logging.error(f"NaN detected in obs for agent {agent_id}: {o}")

        # 更新信息字典
        info["bloods"] = {agent_id: self._jsbsims[agent_id].bloods for agent_id in self._jsbsims.keys()}  # 飞机生命值
        info["missiles"] = {uid: {"active": sim.is_alive, "success": sim.is_success}
                            for uid, sim in self._tempsims.items()}  # 导弹状态

        agent_ids = list(self._jsbsims.keys())  # 获取所有智能体 ID
        # 将 infos 转换为索引格式：{0: {...}, 1: {...}, ...}
        indexed_infos = {}
        for idx, agent_id in enumerate(agent_ids):
            indexed_infos[idx] = infos.get(agent_id, {"current_phase": "unknown", "reward_details": {}})

        # 添加环境级信息
        indexed_infos.update({
            "bloods": {agent_id: self._jsbsims[agent_id].bloods for agent_id in self._jsbsims.keys()},
            "missiles": {uid: {"active": sim.is_alive, "success": sim.is_success}
                         for uid, sim in self._tempsims.items()}
        })
        # 添加调试日志
        logging.debug(f"After task.step, agents: {list(self._jsbsims.keys())}")
        for agent_id, agent in self._jsbsims.items():
            logging.debug(f"Agent {agent_id} type: {type(agent)}, is_alive: {agent.is_alive}")
        # 返回标准化的观测、共享观测、奖励、终止标志和信息
        return (
            np.stack([obs[agent_id] for agent_id in agent_ids], axis=0),
            np.stack([share_obs[agent_id] for agent_id in agent_ids], axis=0),
            np.stack([rewards[agent_id] for agent_id in agent_ids], axis=0),
            np.stack([dones[agent_id] for agent_id in agent_ids], axis=0),
            indexed_infos
        )

    # 加强时间线管理
    def get_timeline_summary(self):
        """获取时间线总结报告（保留空方法避免外部调用错误）"""
        return {
            "mission_duration": self.current_step * self.time_interval,
            "contact_delay": 0,
            "engagement_duration": None,
            "launch_window_duration": None,
            "terminal_phase_duration": None
        }

    def add_temp_simulator(self, simulator: BaseSimulator):
        """添加临时模拟器（如导弹）。

        Args:
            simulator (BaseSimulator): 要添加的模拟器对象
        """
        self._tempsims[simulator.uid] = simulator
        logging.info(f"Added temporary simulator: {simulator.uid}")

    def render(self, mode="txt", filepath=None, tacview=None):
        """渲染当前环境状态。

        Args:
            mode (str): 渲染模式，默认为文本模式
            filepath (str, optional): 渲染文件路径，默认为 './HierarchySelfplay.txt.acmi'
            tacview (Any, optional): Tacview 相关参数
        """
        if not filepath:
            filepath = './HierarchySelfplay.txt.acmi'
        self.renderer.render(self, mode=mode, filepath=filepath, tacview=tacview)
        logging.debug(f"Rendered step {self.current_step} to {filepath}")

    def close(self):
        """关闭环境，释放所有模拟器和渲染器资源。"""
        for sim in self._jsbsims.values():
            sim.close()  # 关闭飞机模拟器
        for sim in self._tempsims.values():
            sim.close()  # 关闭临时模拟器
        self.renderer.close()  # 关闭渲染器
        logging.info("Environment closed")