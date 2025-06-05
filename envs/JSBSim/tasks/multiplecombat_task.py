import logging
import numpy as np
import torch
from gymnasium import spaces
from typing import Tuple, Dict, Any
from ..tasks import SingleCombatTask
from ..core.catalog import Catalog as c
from ..core.simulatior import MissileSimulator
from ..reward_functions import TemplateReward, TacticalReward, AltitudeReward, PostureReward, EventDrivenReward, MissilePostureReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, SafeReturn
from ..utils.utils import get_AO_TA_R, LLA2NEU, get_root_dir
from ..model.baseline_actor import BaselineActor
from ..tasks.TacticalTemplate import TacticalTemplate

class MultipleCombatTask(SingleCombatTask):
    """多智能体空战任务基类，继承单智能体空战任务。"""

    def __init__(self, config):
        """初始化多智能体任务。

        Args:
            config: 配置对象，包含任务参数。
        """
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
        """返回智能体数量。

        Returns:
            int: 智能体数量，固定为 4（2v2）。
        """
        return 4

    def load_variables(self):
        """加载状态、动作和渲染变量。"""
        self.state_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
            c.velocities_v_north_mps,
            c.velocities_v_east_mps,
            c.velocities_v_down_mps,
            c.velocities_u_mps,
            c.velocities_v_mps,
            c.velocities_w_mps,
            c.velocities_vc_mps,
            c.accelerations_n_pilot_x_norm,
            c.accelerations_n_pilot_y_norm,
            c.accelerations_n_pilot_z_norm,
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
        """定义观测空间。"""
        self.obs_length = 9 + (self.num_agents - 1) * 6
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(self.num_agents * self.obs_length,))

    def load_action_space(self):
        """定义动作空间。"""
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])

    def get_obs(self, env, agent_id):
        """获取指定智能体的观测。

        Args:
            env: 环境对象。
            agent_id: 智能体 ID。

        Returns:
            np.ndarray: 归一化的观测向量。
        """
        norm_obs = np.zeros(self.obs_length)
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])
        norm_obs[0] = ego_state[2] / 5000
        norm_obs[1] = np.sin(ego_state[3])
        norm_obs[2] = np.cos(ego_state[3])
        norm_obs[3] = np.sin(ego_state[4])
        norm_obs[4] = np.cos(ego_state[4])
        norm_obs[5] = ego_state[9] / 340
        norm_obs[6] = ego_state[10] / 340
        norm_obs[7] = ego_state[11] / 340
        norm_obs[8] = ego_state[12] / 340
        offset = 9
        for sim in env.agents[agent_id].partners + env.agents[agent_id].enemies:
            state = np.array(sim.get_property_values(self.state_var))
            cur_ned = LLA2NEU(*state[:3], env.center_lon, env.center_lat, env.center_alt)
            feature = np.array([*cur_ned, *(state[6:9])])
            AO, TA, R, side_flag = get_AO_TA_R(ego_feature, feature, return_side=True)
            norm_obs[offset + 0] = (state[9] - ego_state[9]) / 340
            norm_obs[offset + 1] = (state[2] - ego_state[2]) / 1000
            norm_obs[offset + 2] = AO
            norm_obs[offset + 3] = TA
            norm_obs[offset + 4] = R / 10000
            norm_obs[offset + 5] = side_flag
            offset += 6
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        """归一化动作。

        Args:
            env: 环境对象。
            agent_id: 智能体 ID。
            action: 原始动作。

        Returns:
            np.ndarray: 归一化后的动作向量。
        """
        norm_act = np.zeros(4)
        norm_act[0] = action[0] * 2. / (self.action_space.nvec[0] - 1.) - 1.
        norm_act[1] = action[1] * 2. / (self.action_space.nvec[1] - 1.) - 1.
        norm_act[2] = action[2] * 2. / (self.action_space.nvec[2] - 1.) - 1.
        norm_act[3] = action[3] * 0.5 / (self.action_space.nvec[3] - 1.) + 0.4
        return norm_act

    def get_reward(self, env, agent_id, info: dict = ...) -> Tuple[float, dict]:
        """计算奖励。

        Args:
            env: 环境对象。
            agent_id: 智能体 ID。
            info: 附加信息字典。

        Returns:
            Tuple[float, dict]: 奖励值和信息字典。
        """
        if env.agents[agent_id].is_alive:
            return super().get_reward(env, agent_id, info=info)
        else:
            return 0.0, info


class HierarchicalMultipleCombatTask(MultipleCombatTask):
    """分层多智能体空战任务，基于低级策略生成动作。"""

    def __init__(self, config: str):
        """初始化分层任务。

        Args:
            config: 配置对象。
        """
        super().__init__(config)
        self.lowlevel_policy = BaselineActor()
        self.lowlevel_policy.load_state_dict(torch.load(get_root_dir() + '/model/baseline_model.pt', map_location=torch.device('cpu')))
        self.lowlevel_policy.eval()
        self.norm_delta_altitude = np.array([0.1, 0, -0.1])
        self.norm_delta_heading = np.array([-np.pi / 6, -np.pi / 12, 0, np.pi / 12, np.pi / 6])
        self.norm_delta_velocity = np.array([0.05, 0, -0.05])
        self._inner_rnn_states = {}

    def load_action_space(self):
        """定义分层动作空间。"""
        self.action_space = spaces.MultiDiscrete([3, 5, 3])

    def normalize_action(self, env, agent_id, action):
        """归一化分层动作，使用低级策略生成控制命令。

        Args:
            env: 环境对象。
            agent_id: 智能体 ID。
            action: 原始动作。

        Returns:
            np.ndarray: 归一化后的动作向量。
        """
        raw_obs = self.get_obs(env, agent_id)
        input_obs = np.zeros(12)
        input_obs[0] = self.norm_delta_altitude[action[0]]
        input_obs[1] = self.norm_delta_heading[action[1]]
        input_obs[2] = self.norm_delta_velocity[action[2]]
        input_obs[3:12] = raw_obs[:9]
        input_obs = np.expand_dims(input_obs, axis=0)
        _action, _rnn_states = self.lowlevel_policy(input_obs, self._inner_rnn_states[agent_id])
        action = _action.detach().cpu().numpy().squeeze(0)
        self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()
        norm_act = np.zeros(4)
        norm_act[0] = action[0] / 20 - 1.
        norm_act[1] = action[1] / 20 - 1.
        norm_act[2] = action[2] / 20 - 1.
        norm_act[3] = action[3] / 58 + 0.4
        return norm_act

    def reset(self, env):
        """重置任务状态。

        Args:
            env: 环境对象。

        Returns:
            dict: 初始观测。
        """
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        return super().reset(env)


class HierarchicalMultipleCombatShootTask(HierarchicalMultipleCombatTask):
    def __init__(self, config: str):
        super().__init__(config)
        self.max_attack_angle = getattr(self.config, 'max_attack_angle', 75)  # 放宽角度限制
        self.max_attack_distance = getattr(self.config, 'max_attack_distance', 60000)  # 放宽距离
        self.min_attack_interval = getattr(self.config, 'min_attack_interval', 30)  # 缩短间隔
        self.reward_functions = [
            AltitudeReward(self.config),
            PostureReward(self.config),
            MissilePostureReward(self.config),
            EventDrivenReward(self.config),
            TacticalReward(self.config),
            TemplateReward(self.config)
        ]
        self.termination_conditions = [
            SafeReturn(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]
        self.lowlevel_policy = BaselineActor()
        self.lowlevel_policy.load_state_dict(torch.load(get_root_dir() + '/model/baseline_model.pt', map_location=torch.device('cpu')))
        self.lowlevel_policy.eval()
        self.tactical_templates = {}
        self._inner_rnn_states = {}
        self._last_shoot_time = {}
        self._remaining_missiles = {}
        self._shoot_action = {}
        self._last_action = {}
        self._maneuver_history = []
        self._target_allocation = {}
        self.rewards = {agent_id: 0.0 for agent_id in ['A0100', 'A0200', 'B0100', 'B0200']}

    def load_observation_space(self):
        self.obs_length = 14 + self.num_agents * 6 + 6 + 2
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(self.num_agents * self.obs_length,))

    def load_action_space(self):
        self.action_space = spaces.MultiDiscrete([15, 2])

    def get_obs(self, env, agent_id):
        norm_obs = np.zeros(self.obs_length)
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])
        norm_obs[0] = ego_state[2] / 5000
        norm_obs[1] = np.sin(ego_state[3])
        norm_obs[2] = np.cos(ego_state[3])
        norm_obs[3] = np.sin(ego_state[4])
        norm_obs[4] = np.cos(ego_state[4])
        norm_obs[5] = ego_state[9] / 340
        norm_obs[6] = ego_state[10] / 340
        norm_obs[7] = ego_state[11] / 340
        norm_obs[8] = ego_state[12] / 340
        state_dict = self.get_state_dict(env, agent_id)
        radar_state = self.tactical_templates[agent_id].get_radar_state(state_dict)
        norm_obs[9] = 1 if radar_state["radar_lock"] else 0
        norm_obs[10] = 1 if radar_state["has_warning"] else 0
        norm_obs[11] = 1 if env.agents[agent_id].is_leader() else 0
        norm_obs[12] = len(self._target_allocation.get(agent_id, []))
        norm_obs[13] = self.tactical_templates[agent_id].PHASES.index(self.tactical_templates[agent_id].current_phase) / len(self.tactical_templates[agent_id].PHASES)
        offset = 14
        for sim in env.agents[agent_id].partners + env.agents[agent_id].enemies:
            state = np.array(sim.get_property_values(self.state_var))
            cur_ned = LLA2NEU(*state[:3], env.center_lon, env.center_lat, env.center_alt)
            feature = np.array([*cur_ned, *(state[6:9])])
            AO, TA, R, side_flag = get_AO_TA_R(ego_feature, feature, return_side=True)
            norm_obs[offset + 0] = (state[9] - ego_state[9]) / 340
            norm_obs[offset + 1] = (state[2] - ego_state[2]) / 1000
            norm_obs[offset + 2] = AO
            norm_obs[offset + 3] = TA
            norm_obs[offset + 4] = R / 10000
            norm_obs[offset + 5] = side_flag
            offset += 6
        missile_sim = env.agents[agent_id].check_missile_warning()
        if missile_sim is not None:
            missile_feature = np.concatenate((missile_sim.get_position(), missile_sim.get_velocity()))
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, missile_feature, return_side=True)
            norm_obs[offset + 0] = (np.linalg.norm(missile_sim.get_velocity()) - ego_state[9]) / 340
            norm_obs[offset + 1] = (missile_feature[2] - ego_state[2]) / 1000
            norm_obs[offset + 2] = ego_AO
            norm_obs[offset + 3] = ego_TA
            norm_obs[offset + 4] = R / 10000
            norm_obs[offset + 5] = side_flag
        offset += 6
        partner = env.agents[agent_id].partners[0] if env.agents[agent_id].partners else None
        norm_obs[offset] = 1 if partner and partner.is_alive else 0
        norm_obs[offset + 1] = self._remaining_missiles.get(agent_id, 0) / 2
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        template_id, shoot = action[0], action[1] > 0
        self._shoot_action[agent_id] = shoot
        self._last_action[agent_id] = action
        state = self.get_state_dict(env, agent_id)
        if template_id == 0:
            raw_obs = self.get_obs(env, agent_id)
            input_obs = np.zeros(12)
            input_obs[0] = 0.0
            input_obs[1] = 0.0
            input_obs[2] = 0.0
            input_obs[3:12] = raw_obs[:9]
            input_obs = np.expand_dims(input_obs, axis=0)
            _action, _rnn_states = self.lowlevel_policy(input_obs, self._inner_rnn_states[agent_id])
            action = _action.detach().cpu().numpy().squeeze(0)
            self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()
            norm_act = np.zeros(4)
            norm_act[0] = np.clip(action[0] / 20 - 1., -1, 1)
            norm_act[1] = np.clip(action[1] / 20 - 1., -1, 1)
            norm_act[2] = np.clip(action[2] / 20 - 1., -1, 1)
            norm_act[3] = np.clip(action[3] / 58 + 0.4, 0.4, 0.9)
            logging.debug(f"Agent {agent_id} normalize_action: template_id={template_id}, shoot={shoot}, "
                         f"raw_action={action.tolist()}, norm_act={norm_act.tolist()}, lowlevel_policy_output")
        else:
            tactical_action = self.tactical_templates[agent_id].get_action(template_id, state)
            ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
            current_altitude = ego_state[2]
            heading_cmd = tactical_action["heading_cmd"]
            if heading_cmd == "maintain":
                heading_cmd = ego_state[5]
            else:
                heading_cmd = float(heading_cmd)
            altitude_cmd = tactical_action["altitude_cmd"]
            if state["has_warning"] and current_altitude > 2000:  # 提高安全高度
                altitude_cmd = max(altitude_cmd, -500)  # 限制下降率
            action = [
                heading_cmd / np.pi,  # 调整归一化因子
                altitude_cmd / 5000,  # 与 JSBSimController 一致
                0,
                np.clip(tactical_action["velocity_cmd"] / 340, 0.8, 2.0)  # 动态速度
            ]
            norm_act = np.zeros(4)
            norm_act[0] = np.clip(action[0], -1, 1)
            norm_act[1] = np.clip(action[1], -1, 1)
            norm_act[2] = np.clip(action[2], -1, 1)
            norm_act[3] = np.clip(action[3], 0.4, 0.9)
            logging.debug(f"Agent {agent_id} normalize_action: template_id={template_id}, shoot={shoot}, "
                         f"raw_action={action}, norm_act={norm_act.tolist()}, "
                         f"heading_cmd={heading_cmd}, altitude_cmd={altitude_cmd}, velocity_cmd={tactical_action['velocity_cmd']}")
        return norm_act

    def get_state_dict(self, env, agent_id):
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        enemies = env.agents[agent_id].enemies
        enemy_distances = [np.linalg.norm(enemy.get_position() - env.agents[agent_id].get_position()) for enemy in enemies]
        partners = env.agents[agent_id].partners
        partner_angle = np.arctan2(partners[0].get_position()[1] - ego_state[1],
                                   partners[0].get_position()[0] - ego_state[0]) if partners else 0
        missile_sim = env.agents[agent_id].check_missile_warning()
        missile_distance = np.linalg.norm(missile_sim.get_position() - ego_state[:3]) if missile_sim else np.inf
        radar_state = self.tactical_templates[agent_id].get_radar_state({
            "enemy_distance": min(enemy_distances) if enemy_distances else np.inf,
            "enemy_angle_off": self.get_enemy_angle(env, agent_id),
            "missile_distance": missile_distance
        })
        return {
            "current_altitude": ego_state[2],  # 添加当前高度
            "enemy_distance": min(enemy_distances) if enemy_distances else 100000,  # 避免 np.inf
            "enemy_angle_off": self.get_enemy_angle(env, agent_id),
            "missile_distance": missile_distance,
            "radar_lock": radar_state["radar_lock"],
            "has_warning": missile_distance < 60000 or radar_state["has_warning"],  # 放宽警告范围
            "missile_launched": self._shoot_action.get(agent_id, False),
            "missile_active": bool(env.agents[agent_id].launch_missiles),
            "missile_hit": any(missile.is_success for missile in env.agents[agent_id].launch_missiles),
            "is_leader": env.agents[agent_id].is_leader(),
            "partner_angle": np.rad2deg(partner_angle),
            "targets_assigned": bool(self._target_allocation.get(agent_id)),
            "attack_decided": self.tactical_templates[agent_id].current_phase == self.tactical_templates[agent_id].PHASES[5]
        }

    def get_enemy_angle(self, env, agent_id):
        ego_pos = env.agents[agent_id].get_position()
        ego_vel = env.agents[agent_id].get_velocity()
        enemies = env.agents[agent_id].enemies
        if not enemies:
            return 0.0
        enemy_pos = enemies[0].get_position()
        relative_vec = enemy_pos - ego_pos
        angle = np.arccos(np.clip(np.dot(relative_vec, ego_vel) / (np.linalg.norm(relative_vec) * np.linalg.norm(ego_vel) + 1e-8), -1, 1))
        return np.rad2deg(angle)

    def allocate_targets(self, env):
        for agent_id in env.agents.keys():
            if env.agents[agent_id].is_leader():
                state = self.get_state_dict(env, agent_id)
                if state["radar_lock"] and 10000 <= state["enemy_distance"] <= 60000:  # 放宽分配范围
                    enemies = sorted(
                        env.agents[agent_id].enemies,
                        key=lambda e: np.linalg.norm(e.get_velocity()) + 0.1 * np.linalg.norm(e.get_position() - env.agents[agent_id].get_position())
                    )
                    if len(enemies) >= 2:
                        self._target_allocation[agent_id] = [enemies[0]]
                        partner_id = env.agents[agent_id].partners[0].uid if env.agents[agent_id].partners else None
                        if partner_id:
                            self._target_allocation[partner_id] = [enemies[1]]
                            logging.info(f"Target allocation: Leader {agent_id} -> {enemies[0].uid}, Wingman {partner_id} -> {enemies[1].uid}")
                    elif enemies:
                        self._target_allocation[agent_id] = [enemies[0]]
                        partner_id = env.agents[agent_id].partners[0].uid if env.agents[agent_id].partners else None
                        if partner_id:
                            self._target_allocation[partner_id] = [enemies[0]]
                            logging.info(f"Target allocation: Leader {agent_id} -> {enemies[0].uid}, Wingman {partner_id} -> {enemies[0].uid}")
                    else:
                        logging.warning(f"Leader {agent_id} has no enemy targets to allocate")

    def reset(self, env):
        self._last_shoot_time = {agent_id: -self.min_attack_interval for agent_id in env.agents.keys()}
        self._remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self._shoot_action = {agent_id: False for agent_id in env.agents.keys()}
        self._last_action = {agent_id: [0, 0] for agent_id in env.agents.keys()}
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        self._maneuver_history = []
        self._target_allocation = {}
        self.tactical_templates = {
            agent_id: TacticalTemplate(is_enemy=agent_id.startswith('B'), env=env, agent_id=agent_id)
            for agent_id in env.agents.keys()
        }
        self.rewards = {agent_id: 0.0 for agent_id in env.agents.keys()}
        for agent_id in env.agents.keys():
            is_leader = agent_id.endswith("100")
            env.agents[agent_id].set_leader(is_leader)
        return super().reset(env)

    def get_tactical_state(self, agent_id):
        if agent_id in self.tactical_templates:
            return {
                "current_phase": self.tactical_templates[agent_id].current_phase,
                "maneuver_history": self._maneuver_history[-10:] if self._maneuver_history else []
            }
        return {"current_phase": "unknown", "maneuver_history": []}

    def step(self, env):
        super().step(env)
        for agent_id in env.agents.keys():
            state = self.get_state_dict(env, agent_id)
            self.tactical_templates[agent_id].update_phase(state)
            if self.tactical_templates[agent_id].current_phase == self.tactical_templates[agent_id].PHASES[4]:
                self.allocate_targets(env)

        for agent_id, agent in env.agents.items():
            target_list = self._target_allocation.get(agent_id, agent.enemies)
            target_distance = [np.linalg.norm(target.get_position() - agent.get_position()) for target in target_list]
            target_index = np.argmin(target_distance) if target_distance else 0
            target = target_list[target_index].get_position() - agent.get_position() if target_distance else np.zeros(3)
            heading = agent.get_velocity()
            distance = target_distance[target_index] if target_distance else np.inf
            attack_angle = np.rad2deg(
                np.arccos(np.clip(np.sum(target * heading) / (distance * np.linalg.norm(heading) + 1e-8), -1, 1)))
            shoot_interval = env.current_step - self._last_shoot_time[agent_id]
            state = self.get_state_dict(env, agent_id)
            shoot_flag = (
                agent.is_alive and
                self._shoot_action.get(agent_id, False) and
                self._remaining_missiles.get(agent_id, 0) > 0 and
                attack_angle <= self.max_attack_angle and
                distance <= self.max_attack_distance and
                shoot_interval >= self.min_attack_interval and
                self.tactical_templates[agent_id].current_phase == self.tactical_templates[agent_id].PHASES[6] and
                state.get("radar_lock", False)
            )
            if shoot_flag:
                new_missile_uid = f"{agent_id}_missile_{self._remaining_missiles[agent_id]}"
                env.add_temp_simulator(
                    MissileSimulator.create(
                        parent=agent,
                        target=target_list[target_index],
                        uid=new_missile_uid
                    )
                )
                self._remaining_missiles[agent_id] -= 1
                self._last_shoot_time[agent_id] = env.current_step
                logging.info(
                    f"Agent {agent_id} launched missile: target={target_list[target_index].uid}, remaining missiles={self._remaining_missiles[agent_id]}")
                self._maneuver_history.append((agent_id, "missile_launch", env.current_step))
            # 记录战术动作
            if self._last_action.get(agent_id, [0, 0])[0] != 0:
                template_id = self._last_action[agent_id][0]
                maneuver_name = self.tactical_templates[agent_id].get_action(template_id, state)["maneuver"]
                self._maneuver_history.append((agent_id, maneuver_name, env.current_step))

        obs = {agent_id: self.get_obs(env, agent_id) for agent_id in env.agents.keys()}
        all_obs = np.stack([obs[agent_id] for agent_id in env.agents.keys()], axis=0)
        share_obs = np.tile(all_obs.flatten(), (len(env.agents), 1))
        share_obs = {agent_id: share_obs[i] for i, agent_id in enumerate(env.agents.keys())}

        rewards = {}
        dones = {}
        infos = {}
        for agent_id in env.agents.keys():
            reward_sum = 0.0
            reward_details = {}
            for func in self.reward_functions:
                reward_info = func.get_reward(self, env, agent_id)
                if isinstance(reward_info, (tuple, list)) and len(reward_info) > 0:
                    reward_value = reward_info[0]
                    reward_details[func.__class__.__name__] = reward_value
                else:
                    reward_value = reward_info
                    reward_details[func.__class__.__name__] = reward_value
                reward_sum += reward_value
            rewards[agent_id] = np.clip([reward_sum], -10, 10)
            self.rewards[agent_id] = rewards[agent_id][0]
            done, info = self.get_termination(env, agent_id, {})
            dones[agent_id] = [done]
            infos[agent_id] = self.get_tactical_state(agent_id)
            infos[agent_id]["reward_details"] = reward_details
            if env.current_step % 50 == 0:  # 增加日志频率
                state_dict = self.get_state_dict(env, agent_id)
                logging.debug(
                    f"Step {env.current_step} - Agent {agent_id}: "
                    f"Reward={reward_sum:.3f}, Reward Details={reward_details}, "
                    f"Action={self._last_action.get(agent_id, [0, 0])}, "
                    f"Phase={self.tactical_templates[agent_id].current_phase}, "
                    f"Enemy Distance={state_dict['enemy_distance']:.1f}m, "
                    f"Attack Angle={state_dict['enemy_angle_off']:.1f}deg, "
                    f"Radar Lock={state_dict['radar_lock']}, "
                    f"Missile Launched={state_dict['missile_launched']}, "
                    f"Remaining Missiles={self._remaining_missiles.get(agent_id, 0)}, "
                    f"Altitude={state_dict['current_altitude']:.1f}m"
                )
        return obs, share_obs, rewards, dones, infos