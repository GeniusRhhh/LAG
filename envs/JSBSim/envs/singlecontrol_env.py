from .env_base import BaseEnv
from ..tasks.heading_task import HeadingTask
import numpy as np
import logging


class SingleControlEnv(BaseEnv):
    """
    SingleControlEnv is an fly-control env for single agent with no enemy fighters.
    """

    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        assert len(self.agents.keys()) == 1, f"{self.__class__.__name__} only supports 1 aircraft!"
        self.init_states = None

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'heading':
            self.task = HeadingTask(self.config)
        else:
            raise NotImplementedError(f'Unknown taskname: {taskname}')

    def reset(self):
        self.current_step = 0
        self.reset_simulators()
        self.heading_turn_counts = 0
        self.task.reset(self)
        obs = self.get_obs()

        # 添加调试信息，检查 obs 类型和值
        # logging.info(f"Reset: obs type = {type(obs)}, obs = {obs}")

        # # 假设 obs 是一个一维数组 (12,)，索引第 3 个元素 (altitude)
        # if isinstance(obs, np.ndarray) and obs.size >= 4:
        #     initial_altitude = obs[3] * 5000  # 单位：m
        #     if not (14000 <= initial_altitude <= 30000):
        #         logging.warning(f"Initial altitude {initial_altitude:.2f}m out of range [14000, 30000]")
        # else:
        #     logging.warning(f"Unexpected obs format: {obs}, skipping altitude check")
        packed_obs = self._pack(obs)
        # logging.info(
        #     f"Reset: packed_obs type = {type(packed_obs)}, packed_obs shape = {packed_obs.shape}, packed_obs = {packed_obs}")
        return self._pack(obs)

    def reset_simulators(self):
        if self.init_states is None:
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]
        init_heading = self.np_random.uniform(0., 180.)
        init_altitude = self.np_random.uniform(14000., 30000.)
        init_velocities_u = self.np_random.uniform(400., 1200.)
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