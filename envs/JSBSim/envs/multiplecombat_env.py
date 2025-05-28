import numpy as np
from typing import Tuple, Dict, Any
from .env_base import BaseEnv
from ..tasks.multiplecombat_task import HierarchicalMultipleCombatShootTask, HierarchicalMultipleCombatTask, MultipleCombatTask


class MultipleCombatEnv(BaseEnv):
    """
    MultipleCombatEnv is an multi-player competitive environment.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        self._create_records = False

    @property
    def share_observation_space(self):
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
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Resets the state of the environment and returns an initial observation.

        Returns:
            obs (dict): {agent_id: initial observation}
            share_obs (dict): {agent_id: initial state}
        """
        self.current_step = 0
        self.reset_simulators()
        self.task.reset(self)
        obs = self.get_obs()
        share_obs = self.get_state()
        return self._pack(obs), self._pack(share_obs)

    def reset_simulators(self):
        # Assign new initial condition here!
        for sim in self._jsbsims.values():
            sim.reload()
        self._tempsims.clear()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Run one timestep of the environment's dynamics.

        Args:
            action (np.ndarray): Shape (n_rollout_threads, num_agents, action_dim)

        Returns:
            obs: agents' observation
            share_obs: agents' share observation
            rewards: amount of rewards
            dones: whether the episode has ended
            info: auxiliary information
        """
        self.current_step += 1
        info = {"current_step": self.current_step}

        # 适配向量环境，action 形状为 (n_rollout_threads, num_agents, action_dim)
        # 取第一行（单一线程）
        actions_array = np.zeros((len(self.agents), 4))  # 构造占位符动作，形状 (num_agents, 4)
        agent_ids = list(self.agents.keys())
        action_dict = {agent_id: actions_array[i] for i, agent_id in enumerate(agent_ids)}

        # apply actions
        for agent_id in self.agents.keys():
            a_action = self.task.normalize_action(self, agent_id, action_dict[agent_id])
            self.agents[agent_id].set_property_values(self.task.action_var, a_action)

        # run simulation
        for _ in range(self.agent_interaction_steps):
            for sim in self._jsbsims.values():
                sim.run()
            for sim in self._tempsims.values():
                sim.run()

        self.task.step(self)
        obs = self.get_obs()
        share_obs = self.get_state()

        rewards = {}
        for agent_id in self.agents.keys():
            reward, info = self.task.get_reward(self, agent_id, info)
            rewards[agent_id] = [reward]
        ego_reward = np.mean([rewards[ego_id] for ego_id in self.ego_ids])
        enm_reward = np.mean([rewards[enm_id] for enm_id in self.enm_ids])
        for ego_id in self.ego_ids:
            rewards[ego_id] = [ego_reward]
        for enm_id in self.enm_ids:
            rewards[enm_id] = [enm_reward]

        dones = {}
        for agent_id in self.agents.keys():
            done, info = self.task.get_termination(self, agent_id, info)
            dones[agent_id] = [done]

        return (
            np.stack([obs[agent_id] for agent_id in self.agents.keys()], axis=0),
            np.stack([share_obs[agent_id] for agent_id in self.agents.keys()], axis=0),
            np.stack([rewards[agent_id] for agent_id in self.agents.keys()], axis=0),
            np.stack([dones[agent_id] for agent_id in self.agents.keys()], axis=0),
            info
        )