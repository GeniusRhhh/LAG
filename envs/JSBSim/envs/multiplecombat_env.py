import numpy as np
import yaml
import os
import logging
from typing import Tuple, Dict, Any
from gymnasium import spaces
from ..core.simulatior import AircraftSimulator, BaseSimulator
from ..core.catalog import Catalog as c
from ..tasks.multiplecombat_task import HierarchicalMultipleCombatShootTask, HierarchicalMultipleCombatTask, \
    MultipleCombatTask
from ..utils.TacviewRenderer import TacviewRenderer
from .env_base import BaseEnv


class MultipleCombatEnv(BaseEnv):
    """2v2 多智能体空战环境，继承 BaseEnv。支持基于战术模板的时间线流程。"""

    def __init__(self, config_name: str):
        """初始化环境。"""
        super().__init__(config_name)
        self._create_records = False
        self.renderer = TacviewRenderer(filepath=None)

        # 新增：战术控制距离和时间线管理
        self.tactical_distances = {
            "detection_range": 120000,  # 探测距离 120km
            "engagement_range": 80000,  # 交战距离 80km
            "launch_range": 40000,  # 发射距离 40km
            "mar_range": 15000,  # 最小规避距离 15km
            "wez_range": 25000,  # 武器交战区 25km
            "rmax": 60000,  # 导弹最大射程 60km
            "rmin": 3000  # 导弹最小射程 3km
        }

        # 时间线状态管理
        self.mission_timeline = {
            "mission_start": 0,
            "contact_time": None,
            "engagement_time": None,
            "launch_time": None,
            "impact_time": None,
            "mission_end": None
        }

        # 战术态势评估
        self.tactical_situation = {
            "threat_level": "LOW",  # LOW, MEDIUM, HIGH, CRITICAL
            "engagement_geometry": "NEUTRAL",  # OFFENSIVE, DEFENSIVE, NEUTRAL
            "energy_state": "BALANCED",  # ADVANTAGE, BALANCED, DISADVANTAGE
            "information_quality": "GOOD"  # GOOD, DEGRADED, POOR
        }

        logging.info(f"Initialized MultipleCombatEnv with agent_interaction_steps={self.agent_interaction_steps}, "
                     f"time_interval={self.time_interval}, tactical_distances={self.tactical_distances}")

    @property
    def share_observation_space(self) -> spaces.Space:
        return self.task.share_observation_space

    def load_task(self):
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

        # 检查并传递 tactical_distances
        if hasattr(self, 'tactical_distances'):
            self.task.tactical_distances = self.tactical_distances
        else:
            logging.warning("tactical_distances not found, using default values")
            self.task.tactical_distances = {
                "detection_range": 120000,
                "engagement_range": 80000,
                "launch_range": 40000,
                "mar_range": 15000,
                "wez_range": 25000,
                "rmax": 60000,
                "rmin": 3000
            }
        logging.info(f"Loaded task: {taskname} with tactical distance configuration")

    def load_simulator(self):
        super().load_simulator()
        self._jsbsims.clear()

        # 初始化飞机模拟器
        for uid, config in self.config.aircraft_configs.items():
            init_state = config.get("init_state", {})
            self._jsbsims[uid] = AircraftSimulator(
                uid=uid,
                color=config.get("color", "Red"),
                model=config.get("model", "f16"),
                init_state=init_state,
                origin=getattr(self.config, 'battle_field_center', (120.0, 60.0, 0.0)),
                sim_freq=self.sim_freq,
                num_missiles=config.get("missile", 2)
            )
            state = self._jsbsims[uid].get_property_values([
                c.position_long_gc_deg,
                c.position_lat_geod_deg,
                c.position_h_sl_ft
            ])
            logging.debug(f"Initialized {uid} at position: Long={state[0]}, Lat={state[1]}, Alt={state[2]} ft")

        # 设置编队关系和敌我关系
        self._setup_formation_relationships()

        # 设置长机僚机关系
        for agent_id, sim in self._jsbsims.items():
            is_leader = agent_id.endswith("100")
            sim.set_leader(is_leader)
            logging.info(f"Initialized Agent {agent_id} as {'Leader' if is_leader else 'Wingman'}")

        self._tempsims = {}
        logging.info(f"Loaded simulators: {list(self._jsbsims.keys())}")

    def _setup_formation_relationships(self):
        """设置编队关系"""
        # 红方编队
        self._jsbsims["A0100"].partners = [self._jsbsims["A0200"]]
        self._jsbsims["A0100"].enemies = [self._jsbsims["B0100"], self._jsbsims["B0200"]]
        self._jsbsims["A0200"].partners = [self._jsbsims["A0100"]]
        self._jsbsims["A0200"].enemies = [self._jsbsims["B0100"], self._jsbsims["B0200"]]

        # 蓝方编队
        self._jsbsims["B0100"].partners = [self._jsbsims["B0200"]]
        self._jsbsims["B0100"].enemies = [self._jsbsims["A0100"], self._jsbsims["A0200"]]
        self._jsbsims["B0200"].partners = [self._jsbsims["B0100"]]
        self._jsbsims["B0200"].enemies = [self._jsbsims["A0100"], self._jsbsims["A0200"]]

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        self.current_step = 0
        self.agent_interaction_steps = 12

        # 重置时间线
        self.mission_timeline = {
            "mission_start": 0,
            "contact_time": None,
            "engagement_time": None,
            "launch_time": None,
            "impact_time": None,
            "mission_end": None
        }

        # 重置战术态势
        self.tactical_situation = {
            "threat_level": "LOW",
            "engagement_geometry": "NEUTRAL",
            "energy_state": "BALANCED",
            "information_quality": "GOOD"
        }

        self.reset_simulators()
        self.task.reset(self)
        obs = self.get_obs()
        share_obs = self.get_state()

        logging.info(f"Environment reset, current_step={self.current_step}, "
                     f"agent_interaction_steps={self.agent_interaction_steps}, "
                     f"time_interval={self.time_interval}")
        return self._pack(obs), self._pack(share_obs)

    def reset_simulators(self):
        for sim in self._jsbsims.values():
            sim.reload()
            state = sim.get_property_values([
                c.position_long_gc_deg,
                c.position_lat_geod_deg,
                c.position_h_sl_ft
            ])
            logging.debug(f"Reset {sim.uid} at position: Long={state[0]}, Lat={state[1]}, Alt={state[2]} ft")
        self._tempsims.clear()
        logging.info("All simulators reset")

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        self.current_step += 1
        info = {"current_step": self.current_step, "bloods": {}, "missiles": {}}

        # 验证动作维度
        if isinstance(action, np.ndarray):
            logging.debug(f"Step action shape: {action.shape}, n_rollout_threads: {self.n_rollout_threads}")
            if action.ndim == 2 and action.shape[0] == self.num_agents and action.shape[1] == self.action_space.shape[
                0]:
                if self.n_rollout_threads == 1:
                    action = action[np.newaxis, :, :]
                else:
                    raise ValueError(
                        f"Expected shape (n_rollout_threads, {self.num_agents}, {self.action_space.shape[0]}), got {action.shape}")
            elif action.ndim != 3 or action.shape[1] != self.num_agents:
                raise ValueError(
                    f"Invalid action shape {action.shape}, expected (n_rollout_threads, {self.num_agents}, {self.action_space.shape[0]})")

        # 更新时间线和战术态势
        self._update_mission_timeline()
        self._update_tactical_situation()

        action_dict = self._unpack(action)
        last_positions = {aid: self._jsbsims[aid].get_position() for aid in self._jsbsims}

        # 执行动作
        for agent_id in self._jsbsims.keys():
            agent_action = action_dict[agent_id]
            if agent_action.ndim > 1:
                agent_action = agent_action.squeeze()

            # 根据动作空间类型处理
            if hasattr(self.task, 'action_space') and hasattr(self.task.action_space, 'nvec'):
                expected_dim = len(self.task.action_space.nvec)
            else:
                expected_dim = 2  # 默认为分层动作空间

            if agent_action.shape != (expected_dim,):
                logging.error(f"Agent {agent_id} action shape {agent_action.shape}, expected ({expected_dim},)")
                raise ValueError(f"Agent {agent_id} action shape {agent_action.shape}, expected ({expected_dim},)")

            logging.debug(f"Agent {agent_id} action: {agent_action}")
            norm_action = self.task.normalize_action(self, agent_id, agent_action)
            self._jsbsims[agent_id].set_property_values(self.task.action_var, norm_action)

        # 执行仿真步骤
        self.agent_interaction_steps = 12
        logging.debug(f"Agent interaction steps: {self.agent_interaction_steps}")

        for _ in range(self.agent_interaction_steps):
            for sim in self._jsbsims.values():
                sim.run()
            for sim in self._tempsims.values():
                sim.run()

        # 检查位置跳跃
        for agent_id in self._jsbsims:
            pos = self._jsbsims[agent_id].get_position()
            delta = np.linalg.norm(pos - last_positions[agent_id])
            if delta > 1000:
                logging.warning(f"Step {self.current_step}, Agent {agent_id} jumped: "
                                f"{last_positions[agent_id]} -> {pos}, delta={delta:.2f}m")

        # 执行任务步骤
        obs, share_obs, rewards, dones, infos = self.task.step(self)

        # 验证观测值
        for agent_id, o in obs.items():
            if np.any(np.isnan(o)):
                logging.error(f"NaN detected in obs for agent {agent_id}: {o}")

        # 更新信息
        info["bloods"] = {agent_id: self._jsbsims[agent_id].bloods for agent_id in self._jsbsims.keys()}
        info["missiles"] = {uid: {"active": sim.is_alive, "success": sim.is_success}
                            for uid, sim in self._tempsims.items()}
        info["mission_timeline"] = self.mission_timeline.copy()
        info["tactical_situation"] = self.tactical_situation.copy()

        agent_ids = list(self._jsbsims.keys())
        # 转换infos格式：从 {agent_id: {...}} 到 {0: {...}, 1: {...}, ...}
        indexed_infos = {}
        for idx, agent_id in enumerate(agent_ids):
            indexed_infos[idx] = infos.get(agent_id, {"current_phase": "unknown", "reward_details": {}})

        # 添加环境级别的信息
        indexed_infos.update({
            "bloods": {agent_id: self._jsbsims[agent_id].bloods for agent_id in self._jsbsims.keys()},
            "missiles": {uid: {"active": sim.is_alive, "success": sim.is_success}
                         for uid, sim in self._tempsims.items()},
            "mission_timeline": self.mission_timeline.copy(),
            "tactical_situation": self.tactical_situation.copy()
        })

        return (
            np.stack([obs[agent_id] for agent_id in agent_ids], axis=0),
            np.stack([share_obs[agent_id] for agent_id in agent_ids], axis=0),
            np.stack([rewards[agent_id] for agent_id in agent_ids], axis=0),
            np.stack([dones[agent_id] for agent_id in agent_ids], axis=0),
            indexed_infos
        )

    def _update_mission_timeline(self):
        """更新任务时间线"""
        current_time = self.current_step * self.time_interval

        # 计算最小距离
        min_distance = float('inf')
        for red_id in ['A0100', 'A0200']:
            for blue_id in ['B0100', 'B0200']:
                if (red_id in self._jsbsims and blue_id in self._jsbsims and
                        self._jsbsims[red_id].is_alive and self._jsbsims[blue_id].is_alive):
                    dist = np.linalg.norm(
                        self._jsbsims[red_id].get_position() -
                        self._jsbsims[blue_id].get_position()
                    )
                    min_distance = min(min_distance, dist)

        # 更新时间线节点
        if min_distance < self.tactical_distances["detection_range"] and self.mission_timeline["contact_time"] is None:
            self.mission_timeline["contact_time"] = current_time
            logging.info(f"Contact established at t={current_time:.1f}s, distance={min_distance:.0f}m")

        if min_distance < self.tactical_distances["engagement_range"] and self.mission_timeline[
            "engagement_time"] is None:
            self.mission_timeline["engagement_time"] = current_time
            logging.info(f"Engagement range reached at t={current_time:.1f}s, distance={min_distance:.0f}m")

        if min_distance < self.tactical_distances["launch_range"] and self.mission_timeline["launch_time"] is None:
            self.mission_timeline["launch_time"] = current_time
            logging.info(f"Launch range reached at t={current_time:.1f}s, distance={min_distance:.0f}m")

    def _update_tactical_situation(self):
        """更新战术态势评估"""
        # 计算威胁等级
        min_distance = float('inf')
        missile_threats = 0

        for agent_id, agent in self._jsbsims.items():
            if agent.is_alive:
                # 统计导弹威胁
                if agent.check_missile_warning():
                    missile_threats += 1

                # 计算最小敌距
                for enemy in agent.enemies:
                    if enemy.is_alive:
                        dist = np.linalg.norm(agent.get_position() - enemy.get_position())
                        min_distance = min(min_distance, dist)

        # 威胁等级评估
        if missile_threats > 0 or min_distance < self.tactical_distances["mar_range"]:
            self.tactical_situation["threat_level"] = "CRITICAL"
        elif min_distance < self.tactical_distances["wez_range"]:
            self.tactical_situation["threat_level"] = "HIGH"
        elif min_distance < self.tactical_distances["launch_range"]:
            self.tactical_situation["threat_level"] = "MEDIUM"
        else:
            self.tactical_situation["threat_level"] = "LOW"

        # 交战几何评估
        # 简化实现：基于相对位置和速度
        red_alive = sum(1 for aid in ['A0100', 'A0200'] if self._jsbsims[aid].is_alive)
        blue_alive = sum(1 for aid in ['B0100', 'B0200'] if self._jsbsims[aid].is_alive)

        if red_alive > blue_alive:
            self.tactical_situation["engagement_geometry"] = "OFFENSIVE"
        elif red_alive < blue_alive:
            self.tactical_situation["engagement_geometry"] = "DEFENSIVE"
        else:
            self.tactical_situation["engagement_geometry"] = "NEUTRAL"

    def add_temp_simulator(self, simulator: BaseSimulator):
        """添加临时模拟器（如导弹）"""
        self._tempsims[simulator.uid] = simulator
        logging.info(f"Added temporary simulator: {simulator.uid}")

    def get_tactical_distances(self) -> Dict[str, float]:
        """获取战术距离配置"""
        return self.tactical_distances.copy()

    def get_mission_timeline(self) -> Dict[str, Any]:
        """获取任务时间线"""
        return self.mission_timeline.copy()

    def get_tactical_situation(self) -> Dict[str, str]:
        """获取当前战术态势"""
        return self.tactical_situation.copy()

    def render(self, mode="txt", filepath=None, tacview=None):
        if not filepath:
            filepath = './HierarchySelfplay.txt.acmi'
        self.renderer.render(self, mode=mode, filepath=filepath, tacview=tacview)
        logging.debug(f"Rendered step {self.current_step} to {filepath}")

    def close(self):
        for sim in self._jsbsims.values():
            sim.close()
        for sim in self._tempsims.values():
            sim.close()
        self.renderer.close()
        logging.info("Environment closed")