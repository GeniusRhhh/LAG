from typing import Dict, Any, Tuple

from .env_base import BaseEnv
from ..tasks.heading_task import HeadingTask
import numpy as np
class SingleControlEnv(BaseEnv):
    """
    SingleControlEnv is a fly-control env for a single agent with no enemy fighters.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        assert len(self.agents.keys()) == 1, f"{self.__class__.__name__} only supports 1 aircraft!"
        self.init_states = None
        self.heading_turn_counts = 0  # 初始化计数器

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'heading':
            self.task = HeadingTask(self.config)
        else:
            raise NotImplementedError(f'Unknown taskname: {taskname}')

    def reset(self):
        self.current_step = 0
        self.reset_simulators()
        self.heading_turn_counts = 0  # 每次 reset 时重置
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs)

    def reset_simulators(self):
        if self.init_states is None:
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]
        
        # 根据飞机型号设置不同的速度范围
        agent = list(self.agents.values())[0]
        if agent.model == 'su27sk':
            # SU-27: 更重，需要更高的速度避免失速
            min_velocity = 600.  # ~183 m/s
            max_velocity = 1200. # ~366 m/s
        else:
            # F-16或其他
            min_velocity = 400.
            max_velocity = 1200.
        
        init_heading = self.np_random.uniform(0., 180.)
        init_altitude = self.np_random.uniform(20000., 30000.)  # 提高最低高度
        init_velocities_u = self.np_random.uniform(min_velocity, max_velocity)
        
        for init_state in self.init_states:
            init_state.update({
                'ic_psi_true_deg': init_heading,
                'ic_h_sl_ft': init_altitude,
                'ic_u_fps': init_velocities_u,
                'target_heading_deg': init_heading,
                'target_altitude_ft': init_altitude,
                'target_velocities_u_mps': init_velocities_u * 0.3048,
            })
        for idx, sim in enumerate(self.agents.values()):
            sim.reload(self.init_states[idx])
        self._tempsims.clear()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Override BaseEnv.step to manage heading_turn_counts."""
        self.current_step += 1
        info = {"current_step": self.current_step, "heading_turn_counts": self.heading_turn_counts}

        # Apply actions
        action = self._unpack(action)
        for agent_id in self.agents.keys():
            # action[agent_id]可能是(1, 4)或(4,)，需要确保是(4,)
            agent_action = action[agent_id]
            if isinstance(agent_action, np.ndarray) and agent_action.ndim > 1:
                agent_action = agent_action.squeeze()
            a_action = self.task.normalize_action(self, agent_id, agent_action)
            self.agents[agent_id].set_property_values(self.task.action_var, a_action)

        # Run simulation
        for _ in range(self.agent_interaction_steps):
            for sim in self._jsbsims.values():
                sim.run()
            for sim in self._tempsims.values():
                sim.run()
        self.task.step(self)

        obs = self.get_obs()

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

        rewards = {}
        for agent_id in self.agents.keys():
            reward, info = self.task.get_reward(self, agent_id, info)
            rewards[agent_id] = [reward]

        return self._pack(obs), self._pack(rewards), self._pack(dones), info