# tasks/heading_task.py
import logging
import numpy as np
from gymnasium import spaces
from .task_base import BaseTask
from ..core.catalog import Catalog as c
from ..reward_functions import AltitudeReward, HeadingReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, UnreachHeading

class HeadingTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)
        self.step_count = 0
        self.reward_functions = [
            HeadingReward(self.config),
            AltitudeReward(self.config),
        ]
        self.termination_conditions = [
            UnreachHeading(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]

    @property
    def num_agents(self):
        return 1

    def load_variables(self):
        self.state_var = [
            c.delta_altitude,
            c.delta_heading,
            c.delta_velocities_u,  # 替换 c.delta_speed
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.velocities_u_mps,
            c.velocities_v_mps,
            c.velocities_w_mps,
            c.velocities_vc_mps,
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,
            c.fcs_elevator_cmd_norm,
            c.fcs_rudder_cmd_norm,
            c.fcs_throttle_cmd_norm,
        ]
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

    def load_action_space(self):
        low = np.array([-1.0, -1.0, -1.0, 0.4], dtype=np.float32)  # 保持 [0.4, 0.9]
        high = np.array([1.0, 1.0, 1.0, 0.9], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def get_obs(self, env, agent_id):
        obs = np.array(env.agents[agent_id].get_property_values(self.state_var))
        norm_obs = np.zeros(12)
        norm_obs[0] = obs[0] / 2000
        norm_obs[1] = obs[1] / 180
        norm_obs[2] = obs[2] / 340  # delta_velocities_u
        norm_obs[3] = obs[3] / 10000
        norm_obs[4] = np.sin(obs[4])
        norm_obs[5] = np.cos(obs[4])
        norm_obs[6] = np.sin(obs[5])
        norm_obs[7] = np.cos(obs[5])
        norm_obs[8] = obs[6] / 340
        norm_obs[9] = obs[7] / 340
        norm_obs[10] = obs[8] / 340
        norm_obs[11] = obs[9] / 340
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)

        self.step_count += 1
        if self.step_count % 500 == 0:
            logging.info(
                f"Agent {agent_id} Obs: delta_altitude={obs[0]:.2f}m, "
                f"delta_heading={obs[1]:.2f}°, delta_velocities_u={obs[2]:.2f}m/s"
            )

        if np.any(np.isnan(norm_obs)) or np.any(np.isinf(norm_obs)):
            logging.warning(f"Agent {agent_id} Invalid obs: {norm_obs}")

        return norm_obs

    def normalize_action(self, env, agent_id, action):
        low = self.action_space.low
        high = self.action_space.high
        norm_act = np.clip(action, low, high)
        if not np.allclose(action, norm_act, atol=1e-5):
            logging.warning(
                f"Agent {agent_id} Action clipped: original={action.tolist()}, clipped={norm_act.tolist()}"
            )
        if np.any(np.isnan(norm_act)) or np.any(np.isinf(norm_act)):
            logging.error(f"Agent {agent_id} Invalid action: {norm_act.tolist()}")
            norm_act = np.clip(np.zeros_like(action), low, high)

        if self.step_count % 500 == 0:
            logging.info(
                f"Agent {agent_id} Action: aileron={action[0]:.4f}, elevator={action[1]:.4f}, "
                f"rudder={action[2]:.4f}, throttle={action[3]:.4f}"
            )

        return norm_act