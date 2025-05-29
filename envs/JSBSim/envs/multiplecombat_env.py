import numpy as np
import logging
from typing import Tuple, Dict, Any
from .env_base import BaseEnv
from ..core.simulatior import AircraftSimulator, BaseSimulator
from ..tasks.multiplecombat_task import HierarchicalMultipleCombatShootTask, HierarchicalMultipleCombatTask, MultipleCombatTask

class MultipleCombatEnv(BaseEnv):
    """2v2 多智能体空战环境，继承 BaseEnv。"""

    def __init__(self, config_name: str):
        """初始化环境。

        Args:
            config_name: 配置名称。
        """
        super().__init__(config_name)
        self._create_records = False
        logging.info(f"Initializing MultipleCombatEnv, config_name={config_name}")

    @property
    def share_observation_space(self):
        """返回共享观测空间。

        Returns:
            gymnasium.Space: 共享观测空间。
        """
        return self.task.share_observation_space

    def load_task(self):
        """加载任务，根据配置选择具体任务类型。"""
        taskname = getattr(self.config, 'task', None)
        if taskname == 'multiplecombat':
            self.task = MultipleCombatTask(self.config)
        elif taskname == 'hierarchical_multiplecombat':
            self.task = HierarchicalMultipleCombatTask(self.config)
        elif taskname == 'hierarchical_multiplecombat_shoot':
            self.task = HierarchicalMultipleCombatShootTask(self.config)
        else:
            logging.error(f"Unknown task name: {taskname}")
            raise NotImplementedError(f"Unknown taskname: {taskname}")
        logging.info(f"Loaded task: {taskname}")

    def load_simulator(self):
        """初始化模拟器，覆盖 BaseEnv 的方法以设置 2v2 智能体。"""
        super().load_simulator()
        self._jsbsims.clear()
        self._jsbsims.update({
            "A0100": AircraftSimulator(uid="A0100", color="Red", num_missiles=4),
            "A0200": AircraftSimulator(uid="A0200", color="Red", num_missiles=4),
            "B0100": AircraftSimulator(uid="B0100", color="Blue", num_missiles=4),
            "B0200": AircraftSimulator(uid="B0200", color="Blue", num_missiles=4)
        })
        self._jsbsims["A0100"].partners = [self._jsbsims["A0200"]]
        self._jsbsims["A0100"].enemies = [self._jsbsims["B0100"], self._jsbsims["B0200"]]
        self._jsbsims["A0200"].partners = [self._jsbsims["A0100"]]
        self._jsbsims["A0200"].enemies = [self._jsbsims["B0100"], self._jsbsims["B0200"]]
        self._jsbsims["B0100"].partners = [self._jsbsims["B0200"]]
        self._jsbsims["B0100"].enemies = [self._jsbsims["A0100"], self._jsbsims["A0200"]]
        self._jsbsims["B0200"].partners = [self._jsbsims["B0100"]]
        self._jsbsims["B0200"].enemies = [self._jsbsims["A0100"], self._jsbsims["A0200"]]
        for agent_id, sim in self._jsbsims.items():
            is_leader = agent_id.endswith("100")
            sim.set_leader(is_leader)
            logging.info(f"Initialized Agent {agent_id} as {'Leader' if is_leader else 'Wingman'}")
        self._tempsims = {}
        logging.info(f"Loaded simulators: {list(self._jsbsims.keys())}")

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """重置环境，初始化模拟器和任务。

        Returns:
            Tuple[Dict, Dict]: 初始观测和共享观测。
        """
        self.current_step = 0
        self.reset_simulators()
        self.task.reset(self)
        obs = self.get_obs()
        share_obs = self.get_state()
        logging.info(f"Environment reset, current step={self.current_step}")
        return self._pack(obs), self._pack(share_obs)

    def reset_simulators(self):
        """重置所有模拟器状态。"""
        for sim in self._jsbsims.values():
            sim.reload()
        self._tempsims.clear()
        logging.info("All simulators reset")

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """执行一步动作，更新环境状态。

        Args:
            action: 动作数组，形状为 (n_rollout_threads, num_agents, action_dim) 或 (num_agents, action_dim)。

        Returns:
            Tuple: 观测、共享观测、奖励、终止状态和信息。
        """
        self.current_step += 1
        info = {"current_step": self.current_step, "tactical_info": {}}
        # 预处理动作形状
        if isinstance(action, np.ndarray):
            logging.debug(f"Step action shape: {action.shape}, n_rollout_threads: {self.n_rollout_threads}")
            if action.ndim == 2 and action.shape[0] == self.num_agents and action.shape[1] == self.action_space.shape[0]:
                if self.n_rollout_threads == 1:
                    action = action[np.newaxis, :, :]  # 转换为 (1, num_agents, action_dim)
                else:
                    raise ValueError(f"Expected shape (n_rollout_threads, {self.num_agents}, {self.action_space.shape[0]}), got {action.shape}")
        # 解包动作
        action_dict = self._unpack(action)
        # 应用动作
        for agent_id in self._jsbsims.keys():
            # 确保 action_dict[agent_id] 是 (2,)
            agent_action = action_dict[agent_id]
            if agent_action.ndim > 1:
                agent_action = agent_action.squeeze()  # 移除多余维度，如 (1, 2) -> (2,)
            if agent_action.shape != (2,):
                logging.error(f"Agent {agent_id} action shape {agent_action.shape}, expected (2,)")
                raise ValueError(f"Agent {agent_id} action shape {agent_action.shape}, expected (2,)")
            logging.debug(f"Agent {agent_id} action: {agent_action}")
            a_action = self.task.normalize_action(self, agent_id, agent_action)
            self._jsbsims[agent_id].set_property_values(self.task.action_var, a_action)
            # 记录战术信息（如果使用战术模板）
            template_id = agent_action[0] if agent_action.size > 0 else 0
            if template_id > 0 and hasattr(self.task, 'tactical_templates'):
                state = self.task.get_state_dict(self, agent_id)
                tactical_action = self.task.tactical_templates[agent_id].get_action(template_id, state)
                info["tactical_info"][agent_id] = {
                    "maneuver": tactical_action["maneuver"],
                    "phase": tactical_action["phase"],
                    "shoot": self.task._shoot_action.get(agent_id, False),
                    "radar_lock": state["radar_lock"],
                    "has_warning": state["has_warning"],
                    "targets_assigned": len(self.task._target_allocation.get(agent_id, []))
                }
                logging.info(f"Agent {agent_id} tactical info: maneuver={tactical_action['maneuver']}, phase={tactical_action['phase']}, shoot={self.task._shoot_action.get(agent_id, False)}")
        # 运行仿真
        for _ in range(self.agent_interaction_steps):
            for sim in self._jsbsims.values():
                sim.run()
            for sim in self._tempsims.values():
                sim.run()
        self.task.step(self)
        # 获取观测和奖励
        obs = self.get_obs()
        share_obs = self.get_state()
        rewards = {}
        for agent_id in self._jsbsims.keys():
            reward, info = self.task.get_reward(self, agent_id, info)
            rewards[agent_id] = [reward]
        # 平均奖励
        ego_reward = np.mean([rewards[ego_id] for ego_id in self.ego_ids])
        enm_reward = np.mean([rewards[enm_id] for enm_id in self.enm_ids])
        for ego_id in self.ego_ids:
            rewards[ego_id] = [ego_reward]
        for enm_id in self.enm_ids:
            rewards[enm_id] = [enm_reward]
        # 检查终止条件
        dones = {}
        for agent_id in self._jsbsims.keys():
            done, info = self.task.get_termination(self, agent_id, info)
            dones[agent_id] = [done]
        return (
            np.stack([obs[agent_id] for agent_id in self._jsbsims.keys()], axis=0),
            np.stack([share_obs[agent_id] for agent_id in self._jsbsims.keys()], axis=0),
            np.stack([rewards[agent_id] for agent_id in self._jsbsims.keys()], axis=0),
            np.stack([dones[agent_id] for agent_id in self._jsbsims.keys()], axis=0),
            info
        )

    def render(self, mode="txt", filepath='./JSBSimRecording.txt.acmi', tacview=None):
        if mode == "txt":
            if not self._create_records:
                with open(filepath, mode='w', encoding='utf-8-sig') as f:
                    f.write("FileType=text/acmi/tacview\n")
                    f.write("FileVersion=2.1\n")
                    f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")
                self._create_records = True
            with open(filepath, mode='a', encoding='utf-8-sig') as f:
                timestamp = self.current_step * self.time_interval
                f.write(f"#{timestamp:.2f}\n")
                for agent_id, sim in self._jsbsims.items():
                    log_msg = sim.log()
                    if log_msg is not None:
                        state = self.task.get_state_dict(self, agent_id)
                        # 使用动作中的 template_id（从上一步的动作缓存）
                        template_id = self.task._last_action.get(agent_id, [0, 0])[0]  # 默认 [0, 0]
                        tactical_action = self.task.tactical_templates[agent_id].get_action(template_id,
                                                                                            state) if hasattr(self.task,
                                                                                                              'tactical_templates') else {
                            "maneuver": "none", "phase": "none"}
                        logging.debug(f"Agent {agent_id} tactical_action: {tactical_action}")
                        log_msg += f",TacticalInfo=Maneuver:{tactical_action['maneuver']}|Phase:{tactical_action['phase']}|Shoot:{self.task._shoot_action.get(agent_id, False)}|RadarLock:{state['radar_lock']}|RWRWarning:{state['has_warning']}|TargetsAssigned:{len(self.task._target_allocation.get(agent_id, []))}"
                        f.write(log_msg + "\n")
                for sim in self._tempsims.values():
                    log_msg = sim.log()
                    if log_msg is not None:
                        f.write(log_msg + "\n")
        elif mode == "real_time":
            timestamp = self.current_step * self.time_interval
            data = [f"#{timestamp:.2f}\n"]
            for agent_id, sim in self._jsbsims.items():
                log_msg = sim.log()
                if log_msg is not None:
                    state = self.task.get_state_dict(self, agent_id)
                    template_id = self.task._last_action.get(agent_id, [0, 0])[0]  # 默认 [0, 0]
                    tactical_action = self.task.tactical_templates[agent_id].get_action(template_id, state) if hasattr(
                        self.task, 'tactical_templates') else {"maneuver": "none", "phase": "none"}
                    logging.debug(f"Agent {agent_id} tactical_action: {tactical_action}")
                    log_msg += f",TacticalInfo=Maneuver:{tactical_action['maneuver']}|Phase:{tactical_action['phase']}|Shoot:{self.task._shoot_action.get(agent_id, False)}|RadarLock:{state['radar_lock']}|RWRWarning:{state['has_warning']}|TargetsAssigned:{len(self.task._target_allocation.get(agent_id, []))}"
                    data.append(log_msg + "\n")
                for sim in self._tempsims.values():
                    log_msg = sim.log()
                    if log_msg is not None:
                        data.append(log_msg + "\n")
                data_str = "".join(data)
                if tacview:
                    tacview.send_data_to_client(data_str)
        else:
            raise NotImplementedError(f"Unsupported render mode: {mode}")