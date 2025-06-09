import numpy as np
import yaml
import os
import logging
from typing import Tuple, Dict, Any
from gymnasium import spaces
from ..core.simulatior import AircraftSimulator, BaseSimulator
from ..core.catalog import Catalog as c
from ..tasks.multiplecombat_task import HierarchicalMultipleCombatShootTask, HierarchicalMultipleCombatTask, MultipleCombatTask
from ..utils.TacviewRenderer import TacviewRenderer
from .env_base import BaseEnv

class MultipleCombatEnv(BaseEnv):
    """2v2 多智能体空战环境，继承 BaseEnv。"""

    def __init__(self, config_name: str):
        """初始化环境。"""
        super().__init__(config_name)
        self._create_records = False
        self.renderer = TacviewRenderer(filepath=None)
        logging.info(f"Initialized MultipleCombatEnv with agent_interaction_steps={self.agent_interaction_steps}, time_interval={self.time_interval}")

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
        logging.info(f"Loaded task: {taskname}")

    def load_simulator(self):
        super().load_simulator()
        self._jsbsims.clear()
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
        self.current_step = 0
        self.agent_interaction_steps = 12
        self.reset_simulators()
        self.task.reset(self)
        obs = self.get_obs()
        share_obs = self.get_state()
        logging.info(f"Environment reset, current_step={self.current_step}, agent_interaction_steps={self.agent_interaction_steps}, time_interval={self.time_interval}")
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
        if isinstance(action, np.ndarray):
            logging.debug(f"Step action shape: {action.shape}, n_rollout_threads: {self.n_rollout_threads}")
            if action.ndim == 2 and action.shape[0] == self.num_agents and action.shape[1] == self.action_space.shape[0]:
                if self.n_rollout_threads == 1:
                    action = action[np.newaxis, :, :]
                else:
                    raise ValueError(f"Expected shape (n_rollout_threads, {self.num_agents}, {self.action_space.shape[0]}), got {action.shape}")
            elif action.ndim != 3 or action.shape[1] != self.num_agents:
                raise ValueError(f"Invalid action shape {action.shape}, expected (n_rollout_threads, {self.num_agents}, {self.action_space.shape[0]})")
        action_dict = self._unpack(action)
        last_positions = {aid: self._jsbsims[aid].get_position() for aid in self._jsbsims}
        for agent_id in self._jsbsims.keys():
            agent_action = action_dict[agent_id]
            if agent_action.ndim > 1:
                agent_action = agent_action.squeeze()
            if agent_action.shape != (2,):
                logging.error(f"Agent {agent_id} action shape {agent_action.shape}, expected (2,)")
                raise ValueError(f"Agent {agent_id} action shape {agent_action.shape}, expected (2,)")
            logging.debug(f"Agent {agent_id} action: {agent_action}")
            norm_action = self.task.normalize_action(self, agent_id, agent_action)
            self._jsbsims[agent_id].set_property_values(self.task.action_var, norm_action)
        self.agent_interaction_steps = 12
        logging.debug(f"Agent interaction steps: {self.agent_interaction_steps}")
        for _ in range(self.agent_interaction_steps):
            for sim in self._jsbsims.values():
                sim.run()
            for sim in self._tempsims.values():
                sim.run()
        for agent_id in self._jsbsims:
            pos = self._jsbsims[agent_id].get_position()
            delta = np.linalg.norm(pos - last_positions[agent_id])
            if delta > 1000:
                logging.warning(f"Step {self.current_step}, Agent {agent_id} jumped: {last_positions[agent_id]} -> {pos}, delta={delta:.2f}m")
        obs, share_obs, rewards, dones, infos = self.task.step(self)
        for agent_id, o in obs.items():
            if np.any(np.isnan(o)):
                logging.error(f"NaN detected in obs for agent {agent_id}: {o}")
        info["bloods"] = {agent_id: self._jsbsims[agent_id].bloods for agent_id in self._jsbsims.keys()}
        info["missiles"] = {uid: {"active": sim.is_alive, "success": sim.is_success} for uid, sim in self._tempsims.items()}
        agent_ids = list(self._jsbsims.keys())
        return (
            np.stack([obs[agent_id] for agent_id in agent_ids], axis=0),
            np.stack([share_obs[agent_id] for agent_id in agent_ids], axis=0),
            np.stack([rewards[agent_id] for agent_id in agent_ids], axis=0),
            np.stack([dones[agent_id] for agent_id in agent_ids], axis=0),
            info
        )

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