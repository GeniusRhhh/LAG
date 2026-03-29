import numpy as np
from gymnasium import spaces
from typing import Tuple
import torch
from ..tasks import SingleCombatTask
from ..core.catalog import Catalog as c
from ..core.simulatior import MissileSimulator
from ..reward_functions import AltitudeReward, PostureReward, EventDrivenReward, MissilePostureReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, SafeReturn
from ..utils.utils import get_AO_TA_R, LLA2NEU, get_root_dir
from ..model.baseline_actor import BaselineActor
from ..core.radar import RadarModel
class MultipleCombatTask(SingleCombatTask):
    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = [
            AltitudeReward(self.config),
            PostureReward(self.config),
            EventDrivenReward(self.config)
        ]
        self.termination_conditions = [
            SafeReturn(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]
    @property
    def num_agents(self) -> int:
        return 4
    def load_variables(self):
        self.state_var = [
            c.position_long_gc_deg,             # 0. lontitude  (unit: °)
            c.position_lat_geod_deg,            # 1. latitude   (unit: °)
            c.position_h_sl_m,                  # 2. altitude   (unit: m)
            c.attitude_roll_rad,                # 3. roll       (unit: rad)
            c.attitude_pitch_rad,               # 4. pitch      (unit: rad)
            c.attitude_heading_true_rad,        # 5. yaw        (unit: rad)
            c.velocities_v_north_mps,           # 6. v_north    (unit: m/s)
            c.velocities_v_east_mps,            # 7. v_east     (unit: m/s)
            c.velocities_v_down_mps,            # 8. v_down     (unit: m/s)
            c.velocities_u_mps,                 # 9. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 # 10. v_body_y  (unit: m/s)
            c.velocities_w_mps,                 # 11. v_body_z  (unit: m/s)
            c.velocities_vc_mps,                # 12. vc        (unit: m/s)
            c.accelerations_n_pilot_x_norm,     # 13. a_north   (unit: G)
            c.accelerations_n_pilot_y_norm,     # 14. a_east    (unit: G)
            c.accelerations_n_pilot_z_norm,     # 15. a_down    (unit: G)
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
        self.obs_length = 9 + (self.num_agents - 1) * 6
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(self.num_agents * self.obs_length,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])

    def get_obs(self, env, agent_id):
        norm_obs = np.zeros(self.obs_length)
        # (1) ego info normalization
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])
        norm_obs[0] = ego_state[2] / 5000            # 0. ego altitude   (unit: 5km)
        norm_obs[1] = np.sin(ego_state[3])           # 1. ego_roll_sin
        norm_obs[2] = np.cos(ego_state[3])           # 2. ego_roll_cos
        norm_obs[3] = np.sin(ego_state[4])           # 3. ego_pitch_sin
        norm_obs[4] = np.cos(ego_state[4])           # 4. ego_pitch_cos
        norm_obs[5] = ego_state[9] / 340             # 5. ego v_body_x   (unit: mh)
        norm_obs[6] = ego_state[10] / 340            # 6. ego v_body_y   (unit: mh)
        norm_obs[7] = ego_state[11] / 340            # 7. ego v_body_z   (unit: mh)
        norm_obs[8] = ego_state[12] / 340            # 8. ego vc   (unit: mh)(unit: 5G)
        # (2) relative inof w.r.t partner+enemies state
        offset = 8
        for sim in env.agents[agent_id].partners + env.agents[agent_id].enemies:
            state = np.array(sim.get_property_values(self.state_var))
            cur_ned = LLA2NEU(*state[:3], env.center_lon, env.center_lat, env.center_alt)
            feature = np.array([*cur_ned, *(state[6:9])])
            AO, TA, R, side_flag = get_AO_TA_R(ego_feature, feature, return_side=True)
            norm_obs[offset+1] = (state[9] - ego_state[9]) / 340
            norm_obs[offset+2] = (state[2] - ego_state[2]) / 1000
            norm_obs[offset+3] = AO
            norm_obs[offset+4] = TA
            norm_obs[offset+5] = R / 10000
            norm_obs[offset+6] = side_flag
            offset += 6
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

    def get_reward(self, env, agent_id, info: dict = ...) -> Tuple[float, dict]:
        if env.agents[agent_id].is_alive:
            return super().get_reward(env, agent_id, info=info)
        else:
            return 0.0, info

class HierarchicalMultipleCombatTask(MultipleCombatTask):
    
    def __init__(self, config: str):
        super().__init__(config)
        self.lowlevel_policy = BaselineActor()
        self.lowlevel_policy.load_state_dict(torch.load(get_root_dir() + '/model/baseline_model.pt' , map_location=torch.device('cpu') ,weights_only=True))
        self.lowlevel_policy.eval()
        #self.norm_delta_altitude = np.array([0.1, 0, -0.1])
        self.norm_delta_altitude = np.array([-0.1, 0, 0.1]) #改成0下降
        self.norm_delta_heading = np.array([-np.pi / 6, -np.pi / 12, 0, np.pi / 12, np.pi / 6])
        #self.norm_delta_velocity = np.array([0.05, 0, -0.05])
        self.norm_delta_velocity = np.array([-0.05, 0, 0.05]) #改成0减速

    def load_action_space(self):
        self.action_space = spaces.MultiDiscrete([3, 5, 3])

    def normalize_action(self, env, agent_id, action):
        """Convert high-level action into low-level action.
        """
        # generate low-level input_obs
        raw_obs = self.get_obs(env, agent_id)
        input_obs = np.zeros(12)
        # (1) delta altitude/heading/velocity
        input_obs[0] = self.norm_delta_altitude[action[0]]
        input_obs[1] = self.norm_delta_heading[action[1]]
        input_obs[2] = self.norm_delta_velocity[action[2]]
        # (2) ego info
        input_obs[3:12] = raw_obs[:9]
        input_obs = np.expand_dims(input_obs, axis=0)
        # output low-level action
        _action, _rnn_states = self.lowlevel_policy(input_obs, self._inner_rnn_states[agent_id])
        action = _action.detach().cpu().numpy().squeeze(0)
        self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()
        # normalize low-level action
        norm_act = np.zeros(4)
        norm_act[0] = action[0] / 20 - 1.
        norm_act[1] = action[1] / 20 - 1.
        norm_act[2] = action[2] / 20 - 1.
        norm_act[3] = action[3] / 58 + 0.4
        return norm_act

    def reset(self, env):
        """Task-specific reset, include reward function reset.
        """
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        return super().reset(env)



class HierarchicalMultipleCombatShootTask(HierarchicalMultipleCombatTask):
    def __init__(self, config: str):
        super().__init__(config)

        self.radar = RadarModel()

        self.max_attack_angle = getattr(self.config, 'max_attack_angle', 180)
        self.max_attack_distance = getattr(self.config, 'max_attack_distance', np.inf)
        self.min_attack_interval = getattr(self.config, 'min_attack_interval', 125)
        self.reward_functions = [
            PostureReward(self.config),
            MissilePostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config)
        ]

    # def load_observation_space(self):
    #     self.obs_length = 9 + self.num_agents  * 6
    #     self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
    #     self.share_observation_space = spaces.Box(low=-10, high=10., shape=(self.num_agents * self.obs_length,))
    def load_observation_space(self):
        ego_obs_dim = 9  # 自身特征维度
        per_partner_dim = 6  # 每个队友的特征维度（无 target_detected）
        per_enemy_dim = 7  # 每个敌人的特征维度（含 target_detected）
        missile_obs_dim = 6  # 导弹特征数

        num_partners = 1  # 对于2v2，最多1个队友
        num_enemies = 2  # 对于2v2，最多2个敌人

        self.obs_length = ego_obs_dim + num_partners * per_partner_dim + num_enemies * per_enemy_dim + missile_obs_dim
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(self.obs_length,),
            dtype=np.float32
        )
        self.share_observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(self.num_agents * self.obs_length,),
            dtype=np.float32
        )

    def load_action_space(self):
        self.action_space = spaces.MultiDiscrete([3, 5, 3, 2])


    # def get_obs(self, env, agent_id):
    #     norm_obs = np.zeros(self.obs_length)
    #     # (1) ego info normalization
    #     ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
    #     ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
    #     ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])
    #     norm_obs[0] = ego_state[2] / 5000            # 0. ego altitude   (unit: 5km)
    #     norm_obs[1] = np.sin(ego_state[3])           # 1. ego_roll_sin
    #     norm_obs[2] = np.cos(ego_state[3])           # 2. ego_roll_cos
    #     norm_obs[3] = np.sin(ego_state[4])           # 3. ego_pitch_sin
    #     norm_obs[4] = np.cos(ego_state[4])           # 4. ego_pitch_cos
    #     norm_obs[5] = ego_state[9] / 340             # 5. ego v_body_x   (unit: mh)
    #     norm_obs[6] = ego_state[10] / 340            # 6. ego v_body_y   (unit: mh)
    #     norm_obs[7] = ego_state[11] / 340            # 7. ego v_body_z   (unit: mh)
    #     norm_obs[8] = ego_state[12] / 340            # 8. ego vc   (unit: mh)(unit: 5G)
    #     # (2) relative inof w.r.t partner+enemies state
    #     offset = 8
    #     for sim in env.agents[agent_id].partners + env.agents[agent_id].enemies:
    #         state = np.array(sim.get_property_values(self.state_var))
    #         cur_ned = LLA2NEU(*state[:3], env.center_lon, env.center_lat, env.center_alt)
    #         feature = np.array([*cur_ned, *(state[6:9])])
    #         AO, TA, R, side_flag = get_AO_TA_R(ego_feature, feature, return_side=True)
    #         norm_obs[offset+1] = (state[9] - ego_state[9]) / 340
    #         norm_obs[offset+2] = (state[2] - ego_state[2]) / 1000
    #         norm_obs[offset+3] = AO
    #         norm_obs[offset+4] = TA
    #         norm_obs[offset+5] = R / 10000
    #         norm_obs[offset+6] = side_flag
    #         offset += 6
    #     norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
    #     # (3) missile info TODO: multiple missile and parnter's missile?
    #     missile_sim = env.agents[agent_id].check_missile_warning() #
    #     if missile_sim is not None:
    #         missile_feature = np.concatenate((missile_sim.get_position(), missile_sim.get_velocity()))
    #         ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, missile_feature, return_side=True)
    #         norm_obs[offset + 1] = (np.linalg.norm(missile_sim.get_velocity()) - ego_state[9]) / 340
    #         norm_obs[offset + 2] = (missile_feature[2] - ego_state[2]) / 1000
    #         norm_obs[offset + 3] = ego_AO
    #         norm_obs[offset + 4] = ego_TA
    #         norm_obs[offset + 5] = R / 10000
    #         norm_obs[offset + 6] = side_flag
    #
    #     # (4) radar
    #     if len(env.agents[agent_id].enemies) > 0:
    #         enemy = env.agents[agent_id].enemies[0]
    #         is_locked, radar_info = self.radar.detect(
    #             ego_pos=env.agents[agent_id].get_position(),
    #             ego_vel=env.agents[agent_id].get_velocity(),
    #             ego_yaw = env.agents[agent_id].get_property_value(c.attitude_heading_true_rad),
    #             ego_pitch = env.agents[agent_id].get_property_value(c.attitude_pitch_rad),
    #             tar_pos=enemy.get_position(),
    #             tar_vel=enemy.get_velocity()
    #         )
    #         # 挂载到 agent 对象，供 env.step() 拷贝进 infos[agent_id]
    #         env.agents[agent_id].latest_radar_info = radar_info
    #
    #     return norm_obs
    def get_obs(self, env, agent_id):
        # === 初始化观测向量 ===
        norm_obs = np.zeros(self.obs_length)
        # === (1) ego 本机状态提取 + 归一化 ===
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])  # 用于后续 AO/TA/R 计算

        norm_obs[0] = ego_state[2] / 5000  # 高度
        norm_obs[1] = np.sin(ego_state[3])  # 滚转角 sin
        norm_obs[2] = np.cos(ego_state[3])  # 滚转角 cos
        norm_obs[3] = np.sin(ego_state[4])  # 俯仰角 sin
        norm_obs[4] = np.cos(ego_state[4])  # 俯仰角 cos
        norm_obs[5] = ego_state[9] / 340  # v_body_x
        norm_obs[6] = ego_state[10] / 340  # v_body_y
        norm_obs[7] = ego_state[11] / 340  # v_body_z
        norm_obs[8] = ego_state[12] / 340  # vc 总速度

        offset = 8  # 后续写入观测的位置偏移指针

        # === (2a) 队友信息（6维，不使用雷达，始终观测） ===
        for partner in env.agents[agent_id].partners:
            state = np.array(partner.get_property_values(self.state_var))
            cur_ned = LLA2NEU(*state[:3], env.center_lon, env.center_lat, env.center_alt)
            feature = np.array([*cur_ned, *(state[6:9])])
            AO, TA, R, side_flag = get_AO_TA_R(ego_feature, feature, return_side=True)

            norm_obs[offset + 1] = (state[9] - ego_state[9]) / 340
            norm_obs[offset + 2] = (state[2] - ego_state[2]) / 1000
            norm_obs[offset + 3] = AO
            norm_obs[offset + 4] = TA
            norm_obs[offset + 5] = R / 10000
            norm_obs[offset + 6] = side_flag
            offset += 6

        # === (2b) 敌人信息（7维，雷达锁定则填真实，否则填默认 + detected=0）===
        for enemy in env.agents[agent_id].enemies:
            is_locked, radar_info = self.radar.detect(
                ego_pos=env.agents[agent_id].get_position(),
                ego_vel=env.agents[agent_id].get_velocity(),
                ego_yaw=env.agents[agent_id].get_property_value(c.attitude_heading_true_rad),
                ego_pitch=env.agents[agent_id].get_property_value(c.attitude_pitch_rad),
                tar_pos=enemy.get_position(),
                tar_vel=enemy.get_velocity()
            )

            if is_locked:
                # 探测到目标，填入真实信息
                state = np.array(enemy.get_property_values(self.state_var))
                cur_ned = LLA2NEU(*state[:3], env.center_lon, env.center_lat, env.center_alt)
                feature = np.array([*cur_ned, *(state[6:9])])
                AO, TA, R, side_flag = get_AO_TA_R(ego_feature, feature, return_side=True)

                norm_obs[offset + 1] = (state[9] - ego_state[9]) / 340
                norm_obs[offset + 2] = (state[2] - ego_state[2]) / 1000
                norm_obs[offset + 3] = AO
                norm_obs[offset + 4] = TA
                norm_obs[offset + 5] = R / 10000
                norm_obs[offset + 6] = side_flag
                norm_obs[offset + 7] = 1.0  # 被锁定
            else:
                #未探测到目标，填入默认占位值 + detected=0
                norm_obs[offset + 1:offset + 7] = [0.0, 0.0, 0.5, 0.5, 999, 0.5]
                norm_obs[offset + 7] = 0.0  # 未锁定

            #把雷达信息存入 agent，供后续 info_n 拿取
            env.agents[agent_id].latest_radar_info = radar_info
            offset += 7

        # === (3) missile info 来袭导弹信息（固定1个导弹，6维） ===
        missile_sim = env.agents[agent_id].check_missile_warning()
        if missile_sim is not None:
            missile_feature = np.concatenate((missile_sim.get_position(), missile_sim.get_velocity()))
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, missile_feature, return_side=True)

            norm_obs[offset + 1] = (np.linalg.norm(missile_sim.get_velocity()) - ego_state[9]) / 340
            norm_obs[offset + 2] = (missile_feature[2] - ego_state[2]) / 1000
            norm_obs[offset + 3] = ego_AO
            norm_obs[offset + 4] = ego_TA
            norm_obs[offset + 5] = R / 10000
            norm_obs[offset + 6] = side_flag
            # 不管有没有导弹槽，obs_length 中为它预留空间
        # === 最后进行截断归一化处理 ===
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def reset(self, env):
        """Reset fighter blood & missile status
        """
        self._last_shoot_time = {agent_id: -self.min_attack_interval for agent_id in env.agents.keys()}
        self._remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self._shoot_action = {agent_id: False for agent_id in env.agents.keys()}
        return super().reset(env)

    def normalize_action(self, env, agent_id, action):
        self._shoot_action[agent_id] = action[3] > 0
        return super().normalize_action(env, agent_id, action[:3])

    def step(self, env):
        SingleCombatTask.step(self, env)
        for agent_id, agent in env.agents.items():
            # [RL-based missile launch with limited condition]
            # Determine whether can launch missile at the nearest enemy aircraft
            target_list = list(map(lambda x: x.get_position() - agent.get_position(), agent.enemies))
            target_distance = list(map(np.linalg.norm, target_list))
            target_index = np.argmin(target_distance)
            target = target_list[target_index]
            heading = agent.get_velocity()
            distance = target_distance[target_index]
            attack_angle = np.rad2deg(np.arccos(np.clip(np.sum(target * heading) / (distance * np.linalg.norm(heading) + 1e-8), -1, 1)))
            shoot_interval = env.current_step - self._last_shoot_time[agent_id]

            shoot_flag = agent.is_alive and self._shoot_action[agent_id] and self._remaining_missiles[agent_id] > 0 \
                and attack_angle <= self.max_attack_angle and distance <= self.max_attack_distance and shoot_interval >= self.min_attack_interval
            if shoot_flag:
                new_missile_uid = agent_id + str(self._remaining_missiles[agent_id])
                env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=agent.enemies[target_index], uid=new_missile_uid))
                self._remaining_missiles[agent_id] -= 1
                self._last_shoot_time[agent_id] = env.current_step