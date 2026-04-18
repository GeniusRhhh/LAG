import logging

import numpy as np
from gymnasium import spaces
from .task_base import BaseTask
from ..core.catalog import Catalog as c
from ..reward_functions import AltitudeReward, HeadingReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, UnreachHeading


class HeadingTask(BaseTask):
    '''
    Control target heading with discrete action space
    '''
    def __init__(self, config):
        super().__init__(config)
        self.step_count = 0  # 添加step_count属性

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

    def reset(self, env):
        """重置任务状态"""
        super().reset(env)
        self.step_count = 0

    def step(self, env):
        """更新任务状态"""
        super().step(env)
        self.step_count += 1

    def load_variables(self):
        self.state_var = [
            c.delta_altitude,                   # 0. delta_h   (unit: m)
            c.delta_heading,                    # 1. delta_heading  (unit: °)
            c.delta_velocities_u,               # 2. delta_v   (unit: m/s)
            c.position_h_sl_m,                  # 3. altitude  (unit: m)
            c.attitude_roll_rad,                # 4. roll      (unit: rad)
            c.attitude_pitch_rad,               # 5. pitch     (unit: rad)
            c.velocities_u_mps,                 # 6. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 # 7. v_body_y   (unit: m/s)
            c.velocities_w_mps,                 # 8. v_body_z   (unit: m/s)
            c.velocities_vc_mps,                # 9. vc        (unit: m/s)
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,             # [-1., 1.]
            c.fcs_elevator_cmd_norm,            # [-1., 1.]
            c.fcs_rudder_cmd_norm,              # [-1., 1.]
            c.fcs_throttle_cmd_norm,            # [0.4, 0.9]
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
        self.observation_space = spaces.Box(low=-10, high=10., shape=(12,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])

    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space.

        observation(dim 12):
            0. ego delta altitude      (unit: km)
            1. ego delta heading       (unit rad)
            2. ego delta velocities_u  (unit: mh)
            3. ego_altitude            (unit: 5km)
            4. ego_roll_sin
            5. ego_roll_cos
            6. ego_pitch_sin
            7. ego_pitch_cos
            8. ego v_body_x            (unit: mh)
            9. ego v_body_y            (unit: mh)
            10. ego v_body_z           (unit: mh)
            11. ego_vc                 (unit: mh)
        """
        obs = np.array(env.agents[agent_id].get_property_values(self.state_var))
        norm_obs = np.zeros(12)
        norm_obs[0] = obs[0] / 1000         # 0. ego delta altitude (unit: 1km)
        norm_obs[1] = obs[1] / 180 * np.pi  # 1. ego delta heading  (unit rad)
        norm_obs[2] = obs[2] / 340          # 2. ego delta velocities_u (unit: mh)
        norm_obs[3] = obs[3] / 5000         # 3. ego_altitude   (unit: 5km)
        norm_obs[4] = np.sin(obs[4])        # 4. ego_roll_sin
        norm_obs[5] = np.cos(obs[4])        # 5. ego_roll_cos
        norm_obs[6] = np.sin(obs[5])        # 6. ego_pitch_sin
        norm_obs[7] = np.cos(obs[5])        # 7. ego_pitch_cos
        norm_obs[8] = obs[6] / 340          # 8. ego_v_north    (unit: mh)
        norm_obs[9] = obs[7] / 340          # 9. ego_v_east     (unit: mh)
        norm_obs[10] = obs[8] / 340         # 10. ego_v_down    (unit: mh)
        norm_obs[11] = obs[9] / 340         # 11. ego_vc        (unit: mh)
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        """Convert discrete action index into continuous value.
        """
        norm_act = np.zeros(4)
        norm_act[0] = action[0] * 2. / (self.action_space.nvec[0] - 1.) - 1.
        norm_act[1] = action[1] * 2. / (self.action_space.nvec[1] - 1.) - 1.
        norm_act[2] = action[2] * 2. / (self.action_space.nvec[2] - 1.) - 1.
        norm_act[3] = action[3] * 0.5 / (self.action_space.nvec[3] - 1.) + 0.4
        return norm_act

# import logging
# import numpy as np
# from gymnasium import spaces
# from .task_base import BaseTask
# from ..core.catalog import Catalog as c
# from ..reward_functions import AltitudeReward, HeadingReward
# from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, UnreachHeading
#
# class HeadingTask(BaseTask):
#     def __init__(self, config):
#         super().__init__(config)
#         self.step_count = 0
#         self.reward_functions = [
#             HeadingReward(self.config),
#             AltitudeReward(self.config),
#         ]
#         self.termination_conditions = [
#             UnreachHeading(self.config),
#             ExtremeState(self.config),
#             Overload(self.config),
#             LowAltitude(self.config),
#             Timeout(self.config),
#         ]
#
#     @property
#     def num_agents(self):
#         return 1
#
#     def load_variables(self):
#         self.state_var = [
#             c.delta_altitude,
#             c.delta_heading,
#             c.delta_velocities_u,
#             c.position_h_sl_m,
#             c.attitude_roll_rad,
#             c.attitude_pitch_rad,
#             c.velocities_u_mps,
#             c.velocities_v_mps,
#             c.velocities_w_mps,
#             c.velocities_vc_mps,
#         ]
#         self.action_var = [
#             c.fcs_aileron_cmd_norm,
#             c.fcs_elevator_cmd_norm,
#             c.fcs_rudder_cmd_norm,
#             c.fcs_throttle_cmd_norm,
#         ]
#         self.render_var = [
#             c.position_long_gc_deg,
#             c.position_lat_geod_deg,
#             c.position_h_sl_m,
#             c.attitude_roll_rad,
#             c.attitude_pitch_rad,
#             c.attitude_heading_true_rad,
#         ]
#
#     def load_observation_space(self):
#         self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
#
#     def load_action_space(self):
#         low = np.array([-1.0, -1.0, -1.0, 0.4], dtype=np.float32)
#         high = np.array([1.0, 1.0, 1.0, 0.9], dtype=np.float32)
#         self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
#
#     def get_obs(self, env, agent_id):
#         obs = np.array(env.agents[agent_id].get_property_values(self.state_var))
#         norm_obs = np.zeros(12)
#         norm_obs[0] = obs[0] / 2000
#         norm_obs[1] = obs[1] / 180
#         norm_obs[2] = obs[2] / 340
#         norm_obs[3] = obs[3] / 10000
#         norm_obs[4] = np.sin(obs[4])
#         norm_obs[5] = np.cos(obs[4])
#         norm_obs[6] = np.sin(obs[5])
#         norm_obs[7] = np.cos(obs[5])
#         norm_obs[8] = obs[6] / 340
#         norm_obs[9] = obs[7] / 340
#         norm_obs[10] = obs[8] / 340
#         norm_obs[11] = obs[9] / 340
#         norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
#
#         self.step_count += 1
#         if self.step_count % 500 == 0:
#             logging.info(
#                 f"Agent {agent_id} Obs: delta_altitude={obs[0]:.2f}m, "
#                 f"delta_heading={obs[1]:.2f}°, delta_velocities_u={obs[2]:.2f}m/s"
#             )
#
#         if np.any(np.isnan(norm_obs)) or np.any(np.isinf(norm_obs)):
#             logging.warning(f"Agent {agent_id} Invalid obs: {norm_obs}")
#
#         return norm_obs
#
#     def normalize_action(self, env, agent_id, action):
#         low = self.action_space.low
#         high = self.action_space.high
#         norm_act = np.clip(action, low, high)
#         if not np.allclose(action, norm_act, atol=1e-5):
#             logging.warning(
#                 f"Agent {agent_id} Action clipped: original={action.tolist()}, clipped={norm_act.tolist()}"
#             )
#         if np.any(np.isnan(norm_act)) or np.any(np.isinf(norm_act)):
#             logging.error(f"Agent {agent_id} Invalid action: {norm_act.tolist()}")
#             norm_act = np.clip(np.zeros_like(action), low, high)
#
#         if self.step_count % 500 == 0:
#             logging.info(
#                 f"Agent {agent_id} Action: aileron={action[0]:.4f}, elevator={action[1]:.4f}, "
#                 f"rudder={action[2]:.4f}, throttle={action[3]:.4f}"
#             )
#
#         return norm_act
#
#     def get_reward(self, env, agent_id, info={}):
#         reward = 0.0
#         for reward_function in self.reward_functions:
#             r = reward_function.get_reward(self, env, agent_id)
#             reward += r
#             info[f'reward_{reward_function.__class__.__name__}'] = r
#         logging.debug(f"Agent {agent_id} Total Reward: {reward:.4f}, Components: {info}")
#         return reward, info
#
#     def get_termination(self, env, agent_id, info={}):
#         done = False
#         success = True
#         for condition in self.termination_conditions:
#             d, s, term_info = condition.get_termination(self, env, agent_id, info)
#             done = done or d
#             success = success and s
#             info.update(term_info)
#             if done:
#                 info['termination_reason'] = condition.__class__.__name__
#                 logging.debug(f"Agent {agent_id} Termination: {info['termination_reason']}, Info: {info}")
#                 break
#         return done, info