from envs.JSBSim.human_task.HumanFreeFlyTask import HumanFreeFlyTask
from .env_base import BaseEnv
from ..tasks.cap_lowlevel_task import CapLowLevelTask
from ..tasks.heading_task import HeadingTask
from ..tasks.recovery_task import RecoveryTask
from ..core.catalog import Catalog as c


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
        elif taskname == 'cap_lowlevel':
            self.task = CapLowLevelTask(self.config)
        elif taskname == 'recovery':
            self.task = RecoveryTask(self.config)
        elif taskname == "HumanFreeFly":
            self.task = HumanFreeFlyTask(self.config)
        else:
            raise NotImplementedError(f'Unknown taskname: {taskname}')

    def reset(self):
        self.current_step = 0
        self.reset_simulators()
        self.heading_turn_counts = 0
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs)

    def step(self, action):
        agent_id = next(iter(self.agents.keys()))
        prev_heading = float(self.agents[agent_id].get_property_value(c.attitude_psi_deg))
        obs, rewards, dones, info = super().step(action)
        current_heading = float(self.agents[agent_id].get_property_value(c.attitude_psi_deg))
        delta_heading = ((current_heading - prev_heading + 180.0) % 360.0) - 180.0
        if abs(delta_heading) >= 3.0:
            self.heading_turn_counts += 1
        info = dict(info)
        info["heading_turn_counts"] = int(self.heading_turn_counts)
        return obs, rewards, dones, info

    def reset_simulators(self):
        if self.init_states is None:
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]
        init_heading = self.np_random.uniform(0., 180.)
        init_altitude = self.np_random.uniform(14000., 30000.)
        init_velocities_u = self.np_random.uniform(400., 1200.)
        for idx, init_state in enumerate(self.init_states):
            agent_id = list(self.agents.keys())[idx]
            init_state.update({
                'ic_psi_true_deg': init_heading,
                'ic_h_sl_ft': init_altitude,
                'ic_u_fps': init_velocities_u,
                'target_heading_deg': init_heading,
                'target_altitude_ft': init_altitude,
                'target_velocities_u_mps': init_velocities_u * 0.3048,
            })
            if hasattr(self.task, 'sample_init_state'):
                init_state = self.task.sample_init_state(self, agent_id, init_state)
            self.init_states[idx] = init_state
        for idx, sim in enumerate(self.agents.values()):
            sim.reload(self.init_states[idx])
        self._tempsims.clear()
