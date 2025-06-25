import logging
import numpy as np
import torch
from gymnasium import spaces
from typing import Tuple, Dict, Any, List
from ..tasks import SingleCombatTask
from ..core.catalog import Catalog as c
from ..core.simulatior import BaseSimulator, AircraftSimulator, MissileSimulator
from ..reward_functions import (

    TemplateRewardNew,
    TacticalRewardNew,
    AltitudeRewardNew,
    PostureRewardNew,
    EventDrivenRewardNew,
    MissilePostureRewardNew,
    RadarLockRewardNew,
    MissileHitRewardNew,
    BasicFlightReward,
    RewardScaler
)
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, SafeReturn
from ..utils.utils import get_AO_TA_R, LLA2NEU, get_root_dir
from ..model.baseline_actor import BaselineActor
from ..tasks.TacticalTemplate import EnhancedTacticalTemplate


class MultipleCombatTask(SingleCombatTask):
    """多智能体空战任务基类，继承单智能体空战任务。"""

    def __init__(self, config):
        """初始化多智能体任务。"""
        super().__init__(config)
        self.reward_functions = [
            AltitudeRewardNew(self.config),
            PostureRewardNew(self.config),
            EventDrivenRewardNew(self.config),
            RadarLockRewardNew(self.config),
            MissileHitRewardNew(self.config)
        ]
        self.termination_conditions = [
            SafeReturn(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]
        self.allocation_counter = 0
        self.allocation_frequency = 25
        self.step_count = 0

        # 战术距离配置（将从环境传入）
        self.tactical_distances = getattr(config, 'tactical_distances', {
            "detection_range": 120000,
            "engagement_range": 80000,
            "launch_range": 40000,
            "mar_range": 15000,
            "wez_range": 25000,
            "rmax": 60000,
            "rmin": 3000
        })

        # logging.info(f"MultipleCombatTask initialized: num_agents={self.num_agents}, "
        #              f"allocation_frequency={self.allocation_frequency}")

    @property
    def num_agents(self) -> int:
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
        """获取指定智能体的观测。"""
        norm_obs = np.zeros(self.obs_length)

        # 检查智能体是否存在且存活
        if agent_id not in env.agents:
            logging.error(f"Agent {agent_id} not found in env.agents")
            return norm_obs

        if not env.agents[agent_id].is_alive:
            logging.debug(f"Agent {agent_id} is not alive")
            return norm_obs

        try:
            # 获取自身状态
            ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))

            # 检查并修复NaN和异常值
            if np.any(np.isnan(ego_state)):
                logging.error(f"NaN detected in ego_state for {agent_id}: {ego_state}")
                # 修复各个状态值
                if np.isnan(ego_state[2]):  # 高度
                    ego_state[2] = 5000  # 默认5000米
                if np.isnan(ego_state[9]):  # 速度u
                    ego_state[9] = 200  # 默认200m/s
                if np.isnan(ego_state[10]):  # 速度v
                    ego_state[10] = 0
                if np.isnan(ego_state[11]):  # 速度w
                    ego_state[11] = 0
                # 修复其他NaN值
                ego_state = np.nan_to_num(ego_state, nan=0.0, posinf=1000.0, neginf=-1000.0)

            # 物理限制检查
            if ego_state[2] < 0:  # 高度不能为负
                logging.error(f"Agent {agent_id} below ground: {ego_state[2]}m")
                env.agents[agent_id].crash()
                return np.zeros(self.obs_length)

            # 速度限制检查
            total_speed = np.sqrt(ego_state[9] ** 2 + ego_state[10] ** 2 + ego_state[11] ** 2)
            if total_speed > 700:  # 超过700m/s
                logging.error(f"Agent {agent_id} overspeed: {total_speed}m/s")
                # 限制速度
                speed_factor = 500 / total_speed
                ego_state[9] *= speed_factor
                ego_state[10] *= speed_factor
                ego_state[11] *= speed_factor

            # 计算自身位置
            ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
            ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])

            # 自身状态归一化
            norm_obs[0] = ego_state[2] / 5000  # 0. ego altitude   (unit: 5km)
            norm_obs[1] = np.sin(ego_state[3])  # 1. ego_roll_sin
            norm_obs[2] = np.cos(ego_state[3])  # 2. ego_roll_cos
            norm_obs[3] = np.sin(ego_state[4])  # 3. ego_pitch_sin
            norm_obs[4] = np.cos(ego_state[4])  # 4. ego_pitch_cos
            norm_obs[5] = ego_state[9] / 340  # 5. ego v_body_x   (unit: mh)
            norm_obs[6] = ego_state[10] / 340  # 6. ego v_body_y   (unit: mh)
            norm_obs[7] = ego_state[11] / 340  # 7. ego v_body_z   (unit: mh)
            norm_obs[8] = ego_state[12] / 340  # 8. ego vc   (unit: mh)(unit: 5G)

            # 其他智能体相对状态
            offset = 9
            for sim in env.agents[agent_id].partners + env.agents[agent_id].enemies:
                if not sim.is_alive:
                    # 如果目标不存活，填充默认值
                    norm_obs[offset:offset + 6] = [0, 0, 0, 0, 10, 0]
                    offset += 6
                    continue

                try:
                    state = np.array(sim.get_property_values(self.state_var))

                    # 检查目标状态
                    if np.any(np.isnan(state)):
                        logging.warning(f"NaN in target state for {sim.uid}")
                        state = np.nan_to_num(state, nan=0.0)

                    cur_ned = LLA2NEU(*state[:3], env.center_lon, env.center_lat, env.center_alt)
                    feature = np.array([*cur_ned, *(state[6:9])])

                    AO, TA, R, side_flag = get_AO_TA_R(ego_feature, feature, return_side=True)

                    norm_obs[offset + 0] = (state[9] - ego_state[9]) / 340
                    norm_obs[offset + 1] = (state[2] - ego_state[2]) / 1000
                    norm_obs[offset + 2] = AO
                    norm_obs[offset + 3] = TA
                    norm_obs[offset + 4] = R / 10000
                    norm_obs[offset + 5] = side_flag

                except Exception as e:
                    logging.error(f"Error processing target {sim.uid}: {e}")
                    norm_obs[offset:offset + 6] = [0, 0, 0, 0, 10, 0]

                offset += 6

        except Exception as e:
            logging.error(f"Error in get_obs for {agent_id}: {e}")
            return np.zeros(self.obs_length)

        # 最终检查和限制
        norm_obs = np.nan_to_num(norm_obs, nan=0.0, posinf=10.0, neginf=-10.0)
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)

        return norm_obs

    def normalize_action(self, env, agent_id, action):
        """归一化动作。"""
        norm_act = np.zeros(4)
        norm_act[0] = action[0] * 2. / (self.action_space.nvec[0] - 1.) - 1.
        norm_act[1] = action[1] * 2. / (self.action_space.nvec[1] - 1.) - 1.
        norm_act[2] = action[2] * 2. / (self.action_space.nvec[2] - 1.) - 1.
        norm_act[3] = action[3] * 0.5 / (self.action_space.nvec[3] - 1.) + 0.4
        return norm_act

    def get_reward(self, env, agent_id, info: dict = ...) -> Tuple[float, dict]:
        """计算奖励。"""
        if env.agents[agent_id].is_alive:
            return super().get_reward(env, agent_id, info=info)
        else:
            return 0.0, info

    def allocate_targets(self, env):
        """动态分配目标。"""
        alive_agents = {k: v for k, v in env.agents.items() if v.is_alive}
        red_team = {k: v for k, v in alive_agents.items() if v.color == "Red"}
        blue_team = {k: v for k, v in alive_agents.items() if v.color == "Blue"}

        # 为红方分配目标
        for agent_id, agent in red_team.items():
            if not agent.is_leader():
                continue
            min_threat = float("inf")
            closest_enemy = None
            for enemy_id, enemy in blue_team.items():
                distance = np.linalg.norm(agent.get_position() - enemy.get_position())
                threat_score = distance / (1 + np.linalg.norm(enemy.get_velocity()) / 1000)
                if threat_score < min_threat:
                    min_threat = threat_score
                    closest_enemy = enemy
            if closest_enemy:
                self._target_allocation[agent_id] = [closest_enemy]
                for wingman_id in red_team:
                    if wingman_id != agent_id:
                        self._target_allocation[wingman_id] = [closest_enemy]
                logging.info(f"Red leader {agent_id} allocated target: {closest_enemy.uid}, "
                             f"distance={min_threat:.1f}m")

        # 为蓝方分配目标
        for agent_id, agent in blue_team.items():
            if not agent.is_leader():
                continue
            min_threat = float("inf")
            closest_enemy = None
            for enemy_id, enemy in red_team.items():
                distance = np.linalg.norm(agent.get_position() - enemy.get_position())
                threat_score = distance / (1 + np.linalg.norm(enemy.get_velocity()) / 1000)
                if threat_score < min_threat:
                    min_threat = threat_score
                    closest_enemy = enemy
            if closest_enemy:
                self._target_allocation[agent_id] = [closest_enemy]
                for wingman_id in blue_team:
                    if wingman_id != agent_id:
                        self._target_allocation[wingman_id] = [closest_enemy]
                logging.info(f"Blue leader {agent_id} allocated target: {closest_enemy.uid}, "
                             f"distance={min_threat:.1f}m")

    def get_basic_state_dict(self, env, agent_id):
        """获取基本状态字典。"""
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        enemies = env.agents[agent_id].enemies

        # 计算到敌人的距离
        enemy_distances = []
        if enemies:
            for enemy in enemies:
                if enemy.is_alive:
                    distance = np.linalg.norm(enemy.get_position() - env.agents[agent_id].get_position())
                    enemy_distances.append(distance)

        enemy_distance = min(enemy_distances) if enemy_distances else 100000

        # 计算角度偏移
        enemy_angle_off = 0.0
        if enemies and enemies[0].is_alive:
            ego_pos = env.agents[agent_id].get_position()
            ego_vel = env.agents[agent_id].get_velocity()
            enemy_pos = enemies[0].get_position()
            relative_vec = enemy_pos - ego_pos
            if np.linalg.norm(relative_vec) > 0 and np.linalg.norm(ego_vel) > 0:
                angle = np.arccos(np.clip(
                    np.dot(relative_vec, ego_vel) /
                    (np.linalg.norm(relative_vec) * np.linalg.norm(ego_vel)),
                    -1, 1))
                enemy_angle_off = np.rad2deg(angle)

        # 检查导弹威胁
        missile_sim = env.agents[agent_id].check_missile_warning()
        missile_distance = np.linalg.norm(
            missile_sim.get_position() - env.agents[agent_id].get_position()) if missile_sim else np.inf

        # 模拟雷达锁定
        radar_lock = enemy_distance < 80000 and enemy_angle_off < 45

        state_dict = {
            "current_altitude": ego_state[2],
            "enemy_distance": enemy_distance,
            "enemy_angle_off": enemy_angle_off,
            "missile_distance": missile_distance,
            "radar_lock": radar_lock,
            "has_warning": missile_distance < 60000,
            "missile_launched": False,
            "missile_active": False,
            "missile_hit": False,
            "is_leader": env.agents[agent_id].is_leader(),
            "targets_assigned": bool(getattr(self, '_target_allocation', {}).get(agent_id)),
            "attack_decided": False,
            "shoot_probability": 0.1
        }

        return state_dict

    def reset(self, env):
        """重置任务状态。"""
        self.step_count = 0
        self.allocation_counter = 0
        self._target_allocation = {}

        # 重置奖励函数
        for func in self.reward_functions:
            if hasattr(func, 'reset'):
                func.reset(self, env)

        return super().reset(env)


class HierarchicalMultipleCombatTask(MultipleCombatTask):
    """分层多智能体空战任务，基于低级策略生成动作。"""

    def __init__(self, config: str):
        """初始化分层任务。"""
        super().__init__(config)
        self.lowlevel_policy = BaselineActor()
        self.lowlevel_policy.load_state_dict(
            torch.load(get_root_dir() + '/model/baseline_model.pt', map_location=torch.device('cpu')))
        self.lowlevel_policy.eval()

        # 第二层：高层控制参数（3层架构的第二层）
        self.norm_delta_altitude = np.array([-1000, -500, -200, 0, 200, 500, 1000]) / 1000.0  # 7个高度选项
        self.norm_delta_heading = np.array(
            [-np.pi, -np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2, np.pi])  # 9个航向选项
        self.norm_delta_velocity = np.array([-150, -100, -50, 0, 50, 100, 150]) / 100.0  # 7个速度选项

        self._inner_rnn_states = {}
        # 奖励缩放器
        self.reward_scaler = RewardScaler(scale_factor=0.1)
    def load_action_space(self):
        """定义分层动作空间：第二层高层控制。"""
        self.action_space = spaces.MultiDiscrete([7, 9, 7])  # [altitude_cmd_id, heading_cmd_id, velocity_cmd_id]

    def normalize_action(self, env, agent_id, action):
        """归一化分层动作，使用低级策略生成控制命令。"""
        # 基本检查
        if agent_id not in env.agents or not env.agents[agent_id].is_alive:
            return np.array([0.0, 0.0, 0.0, 0.7])

        raw_obs = self.get_obs(env, agent_id)
        input_obs = np.zeros(12)

        # 将离散动作索引转换为连续指令
        input_obs[0] = self.norm_delta_altitude[action[0]]
        input_obs[1] = self.norm_delta_heading[action[1]]
        input_obs[2] = self.norm_delta_velocity[action[2]]
        input_obs[3:12] = raw_obs[:9]

        # 检查输入
        input_obs = np.nan_to_num(input_obs, nan=0.0, posinf=1.0, neginf=-1.0)
        input_obs = np.expand_dims(input_obs, axis=0)

        # 确保RNN状态正确初始化
        if agent_id not in self._inner_rnn_states:
            self._inner_rnn_states[agent_id] = np.zeros((1, 1, 128))

        # 调用低级策略
        _action, _rnn_states = self.lowlevel_policy(input_obs, self._inner_rnn_states[agent_id])
        action_output = _action.detach().cpu().numpy().squeeze(0)
        self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()

        # 恢复原始的动作映射！！！
        norm_act = np.zeros(4)
        norm_act[0] = action_output[0] / 20 - 1.  # Aileron: [-1, 1]
        norm_act[1] = action_output[1] / 20 - 1.  # Elevator: [-1, 1]
        norm_act[2] = action_output[2] / 20 - 1.  # Rudder: [-1, 1]
        norm_act[3] = action_output[3] / 58 + 0.4  # Throttle: [0.4, 0.9]

        # # 只保留最基本的安全限制
        # norm_act[1] = np.clip(norm_act[1], -0.5, 0.5)  # 恢复到原来的±0.5

        # 只在极端情况下介入
        current_alt = env.agents[agent_id].get_position()[2]
        if current_alt < 500:  # 只在极低高度介入
            norm_act[1] = max(norm_act[1], 0.0)  # 不允许下俯
            norm_act[3] = max(norm_act[3], 0.8)  # 增加推力
            logging.warning(f"Agent {agent_id} emergency altitude: {current_alt:.1f}m")

        return norm_act

    def reset(self, env):
        """重置任务状态。"""
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        self._maneuver_history = []
        self._target_allocation = {}

        # 初始化战术模板系统
        self.tactical_templates = {
            agent_id: EnhancedTacticalTemplate(is_enemy=agent_id.startswith('B'), env=env, agent_id=agent_id)
            for agent_id in env.agents.keys()
        }

        # 重置作战阶段
        self.current_phases = {agent_id: "contact_guidance" for agent_id in env.agents.keys()}

        self.rewards = {agent_id: 0.0 for agent_id in env.agents.keys()}
        self.allocation_counter = 0

        # 设置长机僚机关系
        for agent_id in env.agents.keys():
            is_leader = agent_id.endswith("100")
            env.agents[agent_id].set_leader(is_leader)

        logging.info("HierarchicalMultipleCombatShootTask reset: tactical templates and phases initialized")
        return super().reset(env)




class HierarchicalMultipleCombatShootTask(HierarchicalMultipleCombatTask):
    """分层多智能体空战射击任务，集成14种战术模板的三层架构。"""

    def __init__(self, config: str):
        super().__init__(config)
        self.max_attack_angle = getattr(self.config, 'max_attack_angle', 60)
        self.max_attack_distance = getattr(self.config, 'max_attack_distance', 35000)
        self.min_attack_interval = getattr(self.config, 'min_attack_interval', 30)

        # 增强奖励函数
        self.reward_functions = [
            AltitudeRewardNew(self.config),
            PostureRewardNew(self.config),
            MissilePostureRewardNew(self.config),
            EventDrivenRewardNew(self.config),
            TacticalRewardNew(self.config),
            TemplateRewardNew(self.config),
            RadarLockRewardNew(self.config),
            MissileHitRewardNew(self.config),
            BasicFlightReward(self.config)
        ]

        self.termination_conditions = [
            SafeReturn(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]

        # 战术模板系统
        self._inner_rnn_states = {}
        self._last_shoot_time = {}
        self._remaining_missiles = {}
        self._shoot_action = {}
        self._last_action = {}
        self._maneuver_history = []
        self._target_allocation = {}
        self.rewards = {agent_id: 0.0 for agent_id in ['A0100', 'A0200', 'B0100', 'B0200']}

        # 10个作战阶段管理
        self.combat_phases = [
            "contact_guidance",  # 接敌引导
            "target_search",  # 目标搜索
            "target_identification",  # 目标识别
            "threat_assessment",  # 威胁判断
            "target_allocation",  # 目标分配
            "tactical_decision",  # 战术决策
            "missile_launch",  # 发射导弹
            "mid_guidance_defense",  # 中距弹制导/发射后防御
            "terminal_guidance",  # 中距弹末制导
            "effect_assessment"  # 导弹效果评估
        ]

        self.current_phases = {agent_id: "contact_guidance" for agent_id in ['A0100', 'A0200', 'B0100', 'B0200']}

        logging.info(f"HierarchicalMultipleCombatShootTask initialized with 14 tactical templates and 10 combat phases")

    def load_observation_space(self):
        """增强观测空间：包含战术模板和作战阶段信息。"""
        # 基础观测 + 战术状态 + 其他智能体 + 导弹威胁 + 队友状态 + 作战阶段
        self.obs_length = 14 + (self.num_agents-1) * 6 + 6 + 2 + 10  # 新增10维作战阶段信息
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(self.num_agents * self.obs_length,))

    def load_action_space(self):
        """定义三层架构动作空间：第一层战术模板选择 + 射击决策。"""
        # 第一层：战术模板选择层（策略决策层）
        # 15个模板选择（0-14，其中0为无模板直接RL控制）+ 射击决策（0/1）
        self.action_space = spaces.MultiDiscrete([15, 2])

    def get_obs(self, env, agent_id):
        """获取增强观测，包含战术和阶段信息。"""
        norm_obs = np.zeros(self.obs_length)
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])

        # 基本状态观测 (0-8)
        norm_obs[0] = ego_state[2] / 5000
        norm_obs[1] = np.sin(ego_state[3])
        norm_obs[2] = np.cos(ego_state[3])
        norm_obs[3] = np.sin(ego_state[4])
        norm_obs[4] = np.cos(ego_state[4])
        norm_obs[5] = ego_state[9] / 340
        norm_obs[6] = ego_state[10] / 340
        norm_obs[7] = ego_state[11] / 340
        norm_obs[8] = ego_state[12] / 340

        # 战术状态观测 (9-13)
        state_dict = self.get_state_dict(env, agent_id)
        norm_obs[9] = 1 if state_dict["radar_lock"] else 0
        norm_obs[10] = 1 if state_dict["has_warning"] else 0
        norm_obs[11] = 1 if env.agents[agent_id].is_leader() else 0
        norm_obs[12] = len(self._target_allocation.get(agent_id, []))

        # 当前作战阶段 (13)
        if agent_id in self.tactical_templates:
            phase_idx = self.combat_phases.index(self.current_phases[agent_id])
            norm_obs[13] = phase_idx / len(self.combat_phases)
        else:
            norm_obs[13] = 0

        offset = 14
        # 其他智能体观测 (14-31)
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

        # 导弹威胁观测 (32-37)
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
        else:
            norm_obs[offset:offset + 6] = 0.0
        offset += 6

        # 队友状态 (38-39)
        partner = env.agents[agent_id].partners[0] if env.agents[agent_id].partners else None
        norm_obs[offset] = 1 if partner and partner.is_alive else 0
        norm_obs[offset + 1] = self._remaining_missiles.get(agent_id, 0) / 2
        offset += 2

        # 作战阶段状态向量 (40-49)
        phase_vector = np.zeros(10)
        current_phase_idx = self.combat_phases.index(self.current_phases.get(agent_id, "contact_guidance"))
        phase_vector[current_phase_idx] = 1.0
        norm_obs[offset:offset + 10] = phase_vector

        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        """三层架构动作处理：第一层战术模板选择。"""
        if agent_id not in env.agents or not env.agents[agent_id].is_alive:
            return np.array([0.0, 0.0, 0.0, 0.7])

        template_id, shoot = action[0], action[1] > 0
        self._shoot_action[agent_id] = shoot
        self._last_action[agent_id] = action

        # 获取当前状态用于战术决策
        state = self.get_state_dict(env, agent_id)

        if template_id == 0:
            # 无模板：使用RL直接控制高层指令
            # 生成默认的保守飞行指令
            altitude_cmd_id = 1  # 保持高度
            heading_cmd_id = 2  # 保持航向
            velocity_cmd_id = 1  # 保持速度
        else:
            # 使用战术模板生成高层指令
            tactical_action = self.tactical_templates[agent_id].get_tactical_action(template_id, state)

            # 将战术动作转换为离散指令索引
            altitude_cmd_id = self._convert_altitude_to_index(tactical_action.get("altitude_cmd", 0))
            heading_cmd_id = self._convert_heading_to_index(tactical_action.get("heading_cmd", 0))
            velocity_cmd_id = self._convert_velocity_to_index(tactical_action.get("velocity_cmd", 600))

        # 第二层：转换为连续高层指令
        input_obs = np.zeros(12)
        input_obs[0] = self.norm_delta_altitude[altitude_cmd_id]
        input_obs[1] = self.norm_delta_heading[heading_cmd_id]
        input_obs[2] = self.norm_delta_velocity[velocity_cmd_id]

        # 获取基础观测
        raw_obs = self.get_obs(env, agent_id)
        input_obs[3:12] = raw_obs[:9]

        # 安全检查和修复
        input_obs = np.nan_to_num(input_obs, nan=0.0, posinf=1.0, neginf=-1.0)
        input_obs = np.expand_dims(input_obs, axis=0)

        # 确保RNN状态初始化
        if agent_id not in self._inner_rnn_states:
            self._inner_rnn_states[agent_id] = np.zeros((1, 1, 128))

        # 第三层：使用低级策略生成舵面控制
        try:
            _action, _rnn_states = self.lowlevel_policy(input_obs, self._inner_rnn_states[agent_id])
            action_output = _action.detach().cpu().numpy().squeeze(0)
            self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()
        except Exception as e:
            logging.error(f"Lowlevel policy error for {agent_id}: {e}")
            action_output = np.array([20, 20, 20, 29])  # 安全默认值

        # 转换为JSBSim控制指令
        norm_act = np.zeros(4)
        norm_act[0] = action_output[0] / 20 - 1.  # 副翼 [-1, 1]
        norm_act[1] = action_output[1] / 20 - 1.  # 升降舵 [-1, 1]
        norm_act[2] = action_output[2] / 20 - 1.  # 方向舵 [-1, 1]
        norm_act[3] = action_output[3] / 58 + 0.4  # 油门 [0.4, 0.9]

        # 基础飞行保护
        current_alt = env.agents[agent_id].get_position()[2]
        if current_alt < 1000:  # 低空保护
            norm_act[1] = max(norm_act[1], 0.1)  # 强制拉起
            norm_act[3] = max(norm_act[3], 0.8)  # 增加推力
            logging.warning(f"Agent {agent_id} low altitude protection: {current_alt:.1f}m")

        return norm_act

    def _convert_altitude_to_index(self, altitude_cmd):
        """将高度指令转换为索引"""
        # 找到最接近的索引
        altitude_values = np.array([-1000, -500, -200, 0, 200, 500, 1000])
        distances = np.abs(altitude_values - altitude_cmd)
        return np.argmin(distances)

    def _convert_heading_to_index(self, heading_cmd):
        """将航向指令转换为索引"""
        # 限制在±180度范围内
        heading_cmd = np.clip(heading_cmd, -np.pi, np.pi)
        heading_values = np.array(
            [-np.pi, -np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2, np.pi])
        distances = np.abs(heading_values - heading_cmd)
        return np.argmin(distances)

    def _convert_velocity_to_index(self, velocity_cmd):
        """将速度指令转换为索引"""
        # 转换为相对于600的偏移
        velocity_offset = velocity_cmd - 600
        velocity_values = np.array([-150, -100, -50, 0, 50, 100, 150])
        distances = np.abs(velocity_values - velocity_offset)
        return np.argmin(distances)

    def get_state_dict(self, env, agent_id):
        """获取增强状态字典，支持14种战术模板。"""
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        enemies = env.agents[agent_id].enemies

        # 计算敌方状态
        enemy_distances = [np.linalg.norm(enemy.get_position() - env.agents[agent_id].get_position())
                           for enemy in enemies if enemy.is_alive] if enemies else [100000]
        enemy_velocities = [np.linalg.norm(enemy.get_velocity())
                            for enemy in enemies if enemy.is_alive] if enemies else [340.0]

        # 计算队友状态
        partners = env.agents[agent_id].partners
        partner_angle = 0.0
        if partners and partners[0].is_alive:
            partner_pos = partners[0].get_position()
            ego_pos = env.agents[agent_id].get_position()
            partner_angle = np.arctan2(partner_pos[1] - ego_pos[1], partner_pos[0] - ego_pos[0])
            partner_angle = np.rad2deg(partner_angle)

        # 导弹威胁评估
        missile_sim = env.agents[agent_id].check_missile_warning()
        missile_distance = np.linalg.norm(missile_sim.get_position() - ego_state[:3]) if missile_sim else np.inf

        # 增强雷达状态模拟
        enemy_distance = min(enemy_distances) if enemy_distances else 100000
        enemy_angle_off = self.get_enemy_angle(env, agent_id)

        # 基于战术距离的雷达锁定模拟
        radar_lock = (enemy_distance <= self.tactical_distances.get("engagement_range", 80000) and
                      abs(enemy_angle_off) <= 45 and
                      enemy_distance >= self.tactical_distances.get("rmin", 3000))

        # 射击概率计算
        shoot_probability = 0.1
        if radar_lock and enemy_distance <= self.tactical_distances.get("launch_range", 40000):
            distance_factor = 1 - (enemy_distance / self.tactical_distances.get("launch_range", 40000))
            angle_factor = 1 - (abs(enemy_angle_off) / 45)
            shoot_probability = 0.8 * distance_factor * angle_factor

        state_dict = {
            "current_altitude": ego_state[2],
            "enemy_distance": enemy_distance,
            "enemy_velocity": min(enemy_velocities, default=340.0),
            "enemy_angle_off": enemy_angle_off,
            "missile_distance": missile_distance,
            "radar_lock": radar_lock,
            "has_warning": missile_distance < 60000,
            "missile_launched": self._shoot_action.get(agent_id, False),
            "missile_active": bool(env.agents[agent_id].launch_missiles),
            "missile_hit": any(missile.is_success for missile in env.agents[agent_id].launch_missiles),
            "is_leader": env.agents[agent_id].is_leader(),
            "partner_angle": partner_angle,
            "targets_assigned": bool(self._target_allocation.get(agent_id)),
            "attack_decided": self.current_phases.get(agent_id) in ["tactical_decision", "missile_launch"],
            "shoot_probability": shoot_probability,
            "current_phase": self.current_phases.get(agent_id, "contact_guidance"),
            "tactical_distances": self.tactical_distances
        }

        return state_dict

    def get_enemy_angle(self, env, agent_id):
        """计算敌机角度偏移。"""
        ego_pos = env.agents[agent_id].get_position()
        ego_vel = env.agents[agent_id].get_velocity()
        enemies = env.agents[agent_id].enemies

        if not enemies or not enemies[0].is_alive:
            return 0.0

        enemy_pos = enemies[0].get_position()
        relative_vec = enemy_pos - ego_pos

        if np.linalg.norm(relative_vec) == 0 or np.linalg.norm(ego_vel) == 0:
            return 0.0

        angle = np.arccos(np.clip(np.dot(relative_vec, ego_vel) /
                                  (np.linalg.norm(relative_vec) * np.linalg.norm(ego_vel)), -1, 1))
        return np.rad2deg(angle)

    # 在 tasks/TacticalTemplate.py 中替换 update_combat_phase 方法

    def update_combat_phase(self, env, agent_id):
        """更新作战阶段 - 简化快速转换"""
        state = self.get_state_dict(env, agent_id)
        current_phase = self.current_phases.get(agent_id, "contact_guidance")

        distance = state["enemy_distance"]
        radar_lock = state["radar_lock"]
        has_warning = state["has_warning"]
        missile_launched = state["missile_launched"]
        missile_active = state["missile_active"]
        current_altitude = state["current_altitude"]

        # 简化阶段持续时间要求
        if not hasattr(self, 'phase_timers'):
            self.phase_timers = {}

        phase_key = f"{agent_id}_phase_start"
        current_step = getattr(env, 'current_step', 0)

        if phase_key not in self.phase_timers:
            self.phase_timers[phase_key] = current_step

        phase_duration = current_step - self.phase_timers[phase_key]
        min_phase_duration = 2  # 大幅减少从10到2

        new_phase = current_phase

        # 快速阶段转换逻辑
        if phase_duration >= min_phase_duration:

            # 威胁优先 - 有导弹威胁立即进入防御
            if has_warning and current_phase not in ["mid_guidance_defense", "terminal_guidance"]:
                if distance < 20000:
                    new_phase = "terminal_guidance"
                else:
                    new_phase = "mid_guidance_defense"

            # 正常阶段推进
            elif current_phase == "contact_guidance":
                if distance <= 80000:  # 从72000放宽到80000
                    new_phase = "target_search"

            elif current_phase == "target_search":
                if radar_lock or distance <= 60000:  # 有锁定或足够近
                    new_phase = "target_identification"

            elif current_phase == "target_identification":
                if distance <= 45000:  # 进入威胁评估
                    new_phase = "threat_assessment"

            elif current_phase == "threat_assessment":
                if distance <= 35000:  # 快速进入分配
                    new_phase = "target_allocation"

            elif current_phase == "target_allocation":
                if radar_lock and distance <= 40000:  # 快速进入决策
                    new_phase = "tactical_decision"

            elif current_phase == "tactical_decision":
                if missile_launched:
                    new_phase = "missile_launch"
                elif has_warning:
                    new_phase = "mid_guidance_defense"

            elif current_phase == "missile_launch":
                if missile_active or has_warning:
                    new_phase = "mid_guidance_defense"
                elif not missile_active and distance > 30000:
                    new_phase = "effect_assessment"

            elif current_phase == "mid_guidance_defense":
                if distance < 15000:
                    new_phase = "terminal_guidance"
                elif not has_warning and distance > 40000:
                    new_phase = "target_identification"  # 返回识别阶段

            elif current_phase == "terminal_guidance":
                if not has_warning and distance > 25000:
                    new_phase = "effect_assessment"

            elif current_phase == "effect_assessment":
                if distance > 50000:
                    new_phase = "target_search"  # 重新开始
                elif radar_lock and distance <= 40000:
                    new_phase = "tactical_decision"

        # 更新阶段
        if new_phase != current_phase:
            self.current_phases[agent_id] = new_phase
            self.phase_timers[phase_key] = current_step

            logging.debug(f"Agent {agent_id} phase: {current_phase} -> {new_phase} "
                          f"(d={distance:.0f}m, lock={radar_lock}, warn={has_warning})")

        return new_phase

    def _check_phase_conditions(self, phase: str) -> bool:
        """检查阶段条件，增加稳定性"""
        if not hasattr(self.env, 'agents') or self.agent_id not in self.env.agents:
            return False

        agent = self.env.agents[self.agent_id]
        if not agent.is_alive:
            return False

        enemies = agent.enemies
        if not enemies:
            return False

        enemy_distance = min([np.linalg.norm(enemy.get_position() - agent.get_position())
                              for enemy in enemies if enemy.is_alive], default=np.inf)

        # 使用更宽松的条件检查，避免频繁切换
        buffer_factor = 1.1  # 10%缓冲

        if phase == "contact_guidance":
            return enemy_distance > self.TACTICAL_DISTANCES["detection_range"] * buffer_factor
        elif phase == "target_search":
            return (enemy_distance <= self.TACTICAL_DISTANCES["detection_range"] * buffer_factor and
                    enemy_distance > self.TACTICAL_DISTANCES["engagement_range"] * buffer_factor)
        elif phase == "target_identification":
            return (enemy_distance <= self.TACTICAL_DISTANCES["engagement_range"] * buffer_factor and
                    enemy_distance > self.TACTICAL_DISTANCES["wez_range"] * buffer_factor)
        elif phase == "threat_assessment":
            return (enemy_distance <= self.TACTICAL_DISTANCES["wez_range"] * buffer_factor and
                    enemy_distance > self.TACTICAL_DISTANCES["launch_range"] * buffer_factor)
        elif phase == "target_allocation":
            return (enemy_distance <= self.TACTICAL_DISTANCES["launch_range"] * buffer_factor and
                    enemy_distance > self.TACTICAL_DISTANCES["mar_range"] * 2)
        elif phase == "tactical_decision":
            return enemy_distance <= self.TACTICAL_DISTANCES["launch_range"] * buffer_factor
        elif phase == "missile_launch":
            return enemy_distance <= self.TACTICAL_DISTANCES["launch_range"] * buffer_factor
        elif phase == "mid_guidance_defense":
            return enemy_distance <= self.TACTICAL_DISTANCES["mar_range"] * 2
        elif phase == "terminal_guidance":
            return enemy_distance <= self.TACTICAL_DISTANCES["mar_range"] * buffer_factor
        elif phase == "effect_assessment":
            return True

        return False

    def allocate_targets(self, env):
        """基于战术距离的目标分配。"""
        self.allocation_counter += 1
        if self.allocation_counter < self.allocation_frequency:
            return
        self.allocation_counter = 0
        super().allocate_targets(env)

    def reset(self, env):
        """重置任务状态。"""
        # 检查 env.agents 是否有效
        logging.info(f"Resetting with env.agents: {list(env.agents.keys())}")
        if not env.agents:
            logging.error("env.agents is empty, cannot initialize tactical_templates")
            raise ValueError("env.agents is empty")

        # 初始化特定于 HierarchicalMultipleCombatShootTask 的状态
        self._last_shoot_time = {agent_id: -self.min_attack_interval for agent_id in env.agents.keys()}
        self._remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self._shoot_action = {agent_id: False for agent_id in env.agents.keys()}
        self._last_action = {agent_id: [0, 0] for agent_id in env.agents.keys()}
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}

        # 初始化战术模板
        self.tactical_templates = {
            agent_id: EnhancedTacticalTemplate(is_enemy=agent_id.startswith('B'), env=env, agent_id=agent_id)
            for agent_id in env.agents.keys()
        }

        # 初始化其他状态
        self._maneuver_history = []
        self._target_allocation = {}
        self.current_phases = {agent_id: "contact_guidance" for agent_id in env.agents.keys()}
        self.rewards = {agent_id: 0.0 for agent_id in env.agents.keys()}
        self.allocation_counter = 0

        # 设置长机僚机关系
        for agent_id in env.agents.keys():
            is_leader = agent_id.endswith("100")
            env.agents[agent_id].set_leader(is_leader)

        logging.info("HierarchicalMultipleCombatShootTask reset: tactical templates and phases initialized")
        return super().reset(env)

    def get_tactical_state(self, agent_id):
        """获取战术状态信息。"""
        if agent_id in self.tactical_templates:
            state = {
                "current_phase": self.current_phases.get(agent_id, "contact_guidance"),
                "maneuver_history": self._maneuver_history[-10:] if self._maneuver_history else [],
                "last_template": self._last_action.get(agent_id, [0, 0])[0],
                "template_history": getattr(self.tactical_templates[agent_id], 'template_history', [])[-5:]
            }
            return state
        return {"current_phase": "unknown", "maneuver_history": []}

    # 在 tasks/multiple_combat_task.py 中的 step 方法中替换射击逻辑部分

    def step(self, env):
        """执行一步仿真，集成战术模板和阶段管理"""
        self.step_count += 1

        # 更新所有智能体的作战阶段
        for agent_id in env.agents.keys():
            self.update_combat_phase(env, agent_id)
            if agent_id in self.tactical_templates:
                state = self.get_state_dict(env, agent_id)
                self.tactical_templates[agent_id].update_phase(state)

        # 目标分配
        allocation_agents = [aid for aid in env.agents.keys()
                             if self.current_phases.get(aid) == "target_allocation"]
        if allocation_agents:
            self.allocate_targets(env)

        # 修复的导弹发射逻辑
        for agent_id, agent in env.agents.items():
            if not agent.is_alive:
                continue

            target_list = self._target_allocation.get(agent_id, agent.enemies)
            if not target_list:
                continue

            # 找到最近的存活目标
            alive_targets = [t for t in target_list if t.is_alive]
            if not alive_targets:
                continue

            target = min(alive_targets,
                         key=lambda t: np.linalg.norm(t.get_position() - agent.get_position()))
            distance = np.linalg.norm(target.get_position() - agent.get_position())

            # 计算攻击角度
            target_vec = target.get_position() - agent.get_position()
            heading = agent.get_velocity()
            if np.linalg.norm(heading) > 0 and np.linalg.norm(target_vec) > 0:
                attack_angle = np.rad2deg(np.arccos(np.clip(
                    np.dot(target_vec, heading) / (np.linalg.norm(target_vec) * np.linalg.norm(heading)),
                    -1, 1)))
            else:
                attack_angle = 180

            # 射击间隔检查
            shoot_interval = env.current_step - self._last_shoot_time.get(agent_id, -self.min_attack_interval)
            state = self.get_state_dict(env, agent_id)

            # **修复的射击条件** - 更严格
            shoot_flag = (
                    agent.is_alive and
                    self._shoot_action.get(agent_id, False) and
                    self._remaining_missiles.get(agent_id, 0) > 0 and
                    attack_angle <= 25 and  # 从60度大幅收紧到25度
                    28000 <= distance <= 38000 and  # 收紧射击窗口到28-38km
                    shoot_interval >= self.min_attack_interval and
                    state.get("radar_lock", False) and
                    self.current_phases.get(agent_id) in ["missile_launch", "tactical_decision"] and
                    state["enemy_angle_off"] < 25 and  # 从45度收紧到25度
                    state.get("shoot_probability", 0) > 0.5 and  # 提高概率阈值
                    # 新增条件：确保良好的射击窗口
                    np.linalg.norm(agent.get_velocity()) > 200 and  # 最小速度要求
                    state.get("current_altitude", 0) > 4000   # 最小高度要求
            )

            if shoot_flag:
                # 创建导弹
                new_missile_uid = f"{agent_id}{self._remaining_missiles[agent_id]}"
                missile = MissileSimulator.create(
                    parent=agent,
                    target=target,
                    uid=new_missile_uid
                )
                env.add_temp_simulator(missile)

                self._remaining_missiles[agent_id] -= 1
                self._last_shoot_time[agent_id] = env.current_step

                logging.info(f"Agent {agent_id} launched missile: target={target.uid}, "
                             f"distance={distance:.1f}m, angle={attack_angle:.1f}deg, "
                             f"remaining={self._remaining_missiles[agent_id]}")

                # 记录射击事件
                self._maneuver_history.append((agent_id, "missile_launch", env.current_step))

            # 记录其他战术动作
            if self._last_action.get(agent_id, [0, 0])[0] != 0:
                template_id = self._last_action[agent_id][0]
                if agent_id in self.tactical_templates:
                    state_dict = self.get_state_dict(env, agent_id)
                    tactical_action = self.tactical_templates[agent_id].get_tactical_action(template_id, state_dict)
                    maneuver_name = tactical_action.get("maneuver", f"Template_{template_id}")
                    self._maneuver_history.append((agent_id, maneuver_name, env.current_step))

        # 获取观测
        obs = {agent_id: self.get_obs(env, agent_id) for agent_id in env.agents.keys()}

        # 构建共享观测
        all_obs = np.stack([obs[agent_id] for agent_id in sorted(env.agents.keys())], axis=0)
        share_obs = np.tile(all_obs.flatten(), (len(env.agents.keys()), 1))
        share_obs = {agent_id: share_obs[i] for i, agent_id in enumerate(sorted(env.agents.keys()))}

        # 计算奖励
        rewards = {}
        dones = {}
        infos = {}

        for agent_id in env.agents.keys():
            reward_sum = 0.0
            reward_details = {}
            state_dict = self.get_state_dict(env, agent_id)

            # 计算各个奖励组件
            for func in self.reward_functions:
                try:
                    if isinstance(func, (RadarLockRewardNew,MissileHitRewardNew)):
                        reward_info = func.get_reward(self, env, agent_id, state_dict)
                    else:
                        reward_info = func.get_reward(self, env, agent_id)

                    if isinstance(reward_info, (tuple, list)) and len(reward_info) > 0:
                        reward_value = reward_info[0]
                    else:
                        reward_value = reward_info if isinstance(reward_info, (int, float)) else 0.0

                    reward_details[func.__class__.__name__] = reward_value
                    reward_sum += reward_value
                except Exception as e:
                    logging.error(f"Error calculating reward for {func.__class__.__name__}: {e}")
                    reward_details[func.__class__.__name__] = 0.0


            # 缩放奖励
            original_reward = np.clip(reward_sum, -10.0, 10.0)
            # scaled_reward = self.reward_scaler.scale(original_reward)
            scaled_reward = np.clip(original_reward, -10.0, 10.0)  # 直接使用原始奖励
            rewards[agent_id] = np.array([scaled_reward])
            self.rewards[agent_id] = scaled_reward
            # rewards[agent_id] =  np.clip(reward_sum, -10, 10)

            # 检查终止条件
            done, info = self.get_termination(env, agent_id, {})
            dones[agent_id] = [done]

            # 构建详细信息字典
            infos[agent_id] = self.get_tactical_state(agent_id)
            infos[agent_id]["reward_details"] = reward_details
            infos[agent_id]["current_phase"] = self.current_phases.get(agent_id, "contact_guidance")
            infos[agent_id]["step_count"] = self.step_count

            # 定期详细日志输出
            if env.current_step % 250 == 0:
                logging.info(
                    f"Step {env.current_step} - Agent {agent_id}: "
                    f"Reward={reward_sum:.3f}, Phase={self.current_phases.get(agent_id)}, "
                    f"Action={self._last_action.get(agent_id, [0, 0])}, "
                    f"EnemyDistance={state_dict['enemy_distance']:.1f}m, "
                    f"AttackAngle={state_dict['enemy_angle_off']:.1f}deg, "
                    f"RadarLock={state_dict['radar_lock']}, "
                    f"MissileLaunched={state_dict['missile_launched']}, "
                    f"MissileHit={state_dict['missile_hit']}, "
                    f"RemainingMissiles={self._remaining_missiles.get(agent_id, 0)}, "
                    f"Altitude={state_dict['current_altitude']:.1f}m, "
                    f"RewardDetails={reward_details}"
                )

        return obs, share_obs, rewards, dones, infos
