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

        # 设置不同阵营的初始速度
        for agent_id in env.agents:
            if agent_id.startswith('A'):  # 友方智能体
                # 300m/s = 984.25 fps
                env.agents[agent_id].set_property_value(c.ic_u_fps, 984.25)
            elif agent_id.startswith('B'):  # 敌方智能体
                # 280m/s = 918.64 fps  
                env.agents[agent_id].set_property_value(c.ic_u_fps, 918.64)
        
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

        # 第二层：高层控制参数
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

        # 恢复原始的动作映射
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
        # **添加阶段管理需要的属性**
        self.timeline_events = []
        self.decision_log = []
        self.phase_start_times = {}  # 阶段开始时间跟踪
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
        """观测空间：包含战术模板和作战阶段信息。"""
        # 基础观测 + 战术状态 + 其他智能体 + 导弹威胁 + 队友状态 + 作战阶段
        self.obs_length = 14 + (self.num_agents-1) * 6 + 6 + 2 + 10  # 10维作战阶段信息
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(self.num_agents * self.obs_length,))

    def load_action_space(self):
        """三层架构动作空间：第一层战术模板选择 + 射击决策。"""
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
        """战术模板选择 - 基于实际BVR战术原则"""
        if agent_id not in env.agents or not env.agents[agent_id].is_alive:
            return np.array([0.0, 0.0, 0.0, 0.7])

        template_id, shoot = action[0], action[1] > 0
        self._shoot_action[agent_id] = shoot
        self._last_action[agent_id] = action

        # 基于距离和阶段的智能模板选择
        state = self.get_state_dict(env, agent_id)
        current_phase = self.current_phases.get(agent_id, "contact_guidance")
        distance = state["enemy_distance"]
        has_warning = state.get("has_warning", False)
        radar_lock = state.get("radar_lock", False)

        # 如果选择了无模板（0），进行智能推荐
        if template_id == 0:
            template_id = self._recommend_template_by_situation(state, current_phase, distance, has_warning, radar_lock)
            logging.debug(f"Agent {agent_id} auto-selected template {template_id} for situation")

        # 验证模板适用性，不适用时选择替代方案
        if not self._is_template_valid_for_situation(template_id, state, current_phase):
            original_template = template_id
            template_id = self._get_fallback_template(state, current_phase)
            logging.debug(f"Agent {agent_id} template {original_template} invalid, using fallback {template_id}")

        # 记录决策信息用于分析
        self._log_template_decision(agent_id, template_id, current_phase, state)

        # 使用选定的战术模板生成高层指令
        if template_id == 0:
            # 直接RL控制
            altitude_cmd_id = np.random.choice(len(self.norm_delta_altitude))  # 随机选择高度变化
            heading_cmd_id = np.random.choice(len(self.norm_delta_heading))  # 随机选择航向变化
            velocity_cmd_id = np.random.choice(len(self.norm_delta_velocity))  # 随机选择速度变化
        else:
            # 使用战术模板
            tactical_action = self.tactical_templates[agent_id].get_tactical_action(template_id, state)

            altitude_cmd_id = self._convert_altitude_to_index(tactical_action.get("altitude_cmd", 0))
            heading_cmd_id = self._convert_heading_to_index(tactical_action.get("heading_cmd", 0))
            velocity_cmd_id = self._convert_velocity_to_index(tactical_action.get("velocity_cmd", 600))

        # 转换为连续高层指令并调用低级策略
        input_obs = np.zeros(12)
        input_obs[0] = self.norm_delta_altitude[altitude_cmd_id]
        input_obs[1] = self.norm_delta_heading[heading_cmd_id]
        input_obs[2] = self.norm_delta_velocity[velocity_cmd_id]

        raw_obs = self.get_obs(env, agent_id)
        input_obs[3:12] = raw_obs[:9]

        # 安全检查和RNN状态处理
        input_obs = np.nan_to_num(input_obs, nan=0.0, posinf=1.0, neginf=-1.0)
        input_obs = np.expand_dims(input_obs, axis=0)

        if agent_id not in self._inner_rnn_states:
            self._inner_rnn_states[agent_id] = np.zeros((1, 1, 128))

        # 低级策略生成舵面控制
        try:
            _action, _rnn_states = self.lowlevel_policy(input_obs, self._inner_rnn_states[agent_id])
            action_output = _action.detach().cpu().numpy().squeeze(0)
            self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()
        except Exception as e:
            logging.error(f"Lowlevel policy error for {agent_id}: {e}")
            action_output = np.array([20, 20, 20, 29])

        # 转换为JSBSim控制指令
        norm_act = np.zeros(4)
        norm_act[0] = action_output[0] / 20 - 1.  # 副翼
        norm_act[1] = action_output[1] / 20 - 1.  # 升降舵
        norm_act[2] = action_output[2] / 20 - 1.  # 方向舵
        norm_act[3] = action_output[3] / 58 + 0.4  # 油门

        # 基础安全保护
        current_alt = env.agents[agent_id].get_position()[2]
        if current_alt < 1000:
            norm_act[1] = max(norm_act[1], 0.1)
            norm_act[3] = max(norm_act[3], 0.8)

        return norm_act

    def _recommend_template_by_situation(self, state, phase, distance, has_warning, radar_lock):
        """基于战术情况智能推荐模板"""

        # 威胁响应优先级最高
        if has_warning:
            missile_distance = state.get("missile_distance", np.inf)
            if missile_distance < 20000:
                return 3  # Notch - 近距离威胁用地面杂波
            elif missile_distance < 35000:
                return 2  # Beam - 中距离威胁用横向机动
            else:
                return 1  # Crank - 远距离威胁用斜向机动

        # 基于距离的战术选择
        if distance > 80000:  # 远程BVR
            if phase in ["contact_guidance", "target_search"]:
                return 7  # Simple_F_Pole - 保持锁定
            else:
                return 1  # Crank - 机动接敌

        elif distance > 50000:  # 中程BVR
            if radar_lock:
                if state.get("is_leader", False):
                    return 4  # Skate - 长机执行复杂攻击
                else:
                    return 9  # Pincer - 僚机配合夹击
            else:
                return 8  # Advanced_F_Pole - 先获取锁定

        elif distance > 25000:  # 近程BVR
            if radar_lock and phase == "missile_launch":
                return 6  # Banzai - 发射后决策
            elif state.get("is_leader", False):
                return 5  # Short_Skate - 快速攻击
            else:
                return 12  # Engaging_Trail - 僚机跟随

        else:  # 极近距离
            if has_warning:
                return 14  # Defensive_Sequence - 连续防御
            else:
                return 11  # High_Low - 高低分离

    def _is_template_valid_for_situation(self, template_id, state, phase):
        """检查模板在当前情况下是否有效"""
        distance = state["enemy_distance"]
        has_warning = state.get("has_warning", False)
        radar_lock = state.get("radar_lock", False)

        # 防御模板只在有威胁时使用
        defensive_templates = [2, 3, 14]
        if template_id in defensive_templates and not has_warning:
            return False

        # 攻击模板需要合适的距离和锁定条件
        attack_templates = [4, 5, 6]
        if template_id in attack_templates:
            if distance > 60000 or not radar_lock:
                return False

        # 协同模板需要队友存活
        cooperative_templates = [9, 10, 11, 12, 13]
        if template_id in cooperative_templates:
            if not state.get("is_leader", False) and not self._has_alive_partner(state):
                return False

        # F-Pole模板适合中远距离
        fpole_templates = [7, 8]
        if template_id in fpole_templates and distance < 30000:
            return False

        return True

    def _get_fallback_template(self, state, phase):
        """获取安全的备选模板"""
        distance = state["enemy_distance"]
        has_warning = state.get("has_warning", False)

        # 威胁时优先防御
        if has_warning:
            return 2  # Beam

        # 根据距离选择安全模板
        if distance > 60000:
            return 7  # Simple_F_Pole
        elif distance > 40000:
            return 1  # Crank
        elif distance > 25000:
            return 8  # Advanced_F_Pole
        else:
            return 2  # Beam

    def _has_alive_partner(self, state: Dict[str, Any]) -> bool:
        """检查是否有存活的队友。"""
        agent_id = state.get("agent_id")
        env = state.get("env")
        if not agent_id or not env or agent_id not in env.agents:
            return False
        agent = env.agents[agent_id]
        return any(partner.is_alive for partner in agent.partners)

    def _log_template_decision(self, agent_id, template_id, phase, state):
        """记录模板选择决策用于分析"""
        template_names = ["No_Template", "Crank", "Beam", "Notch", "Skate", "Short_Skate",
                          "Banzai", "Simple_F_Pole", "Advanced_F_Pole", "Pincer",
                          "Defensive_Split", "High_Low", "Engaging_Trail", "Loose_Deuce", "Defensive_Sequence"]

        decision_info = {
            "agent": agent_id,
            "step": getattr(self, 'step_count', 0),
            "phase": phase,
            "template": template_names[template_id] if template_id < len(template_names) else "Unknown",
            "distance": state.get("enemy_distance", 0),
            "radar_lock": state.get("radar_lock", False),
            "has_warning": state.get("has_warning", False),
            "altitude": state.get("current_altitude", 0)
        }

        if not hasattr(self, 'decision_log'):
            self.decision_log = []
        self.decision_log.append(decision_info)

        # 每50步分析一次决策模式
        if len(self.decision_log) % 500 == 0:
            self._analyze_recent_decisions()

    def _analyze_recent_decisions(self):
        """分析最近的决策模式"""
        if not hasattr(self, 'decision_log') or len(self.decision_log) < 20:
            return

        recent = self.decision_log[-20:]  # 最近20个决策

        # 统计模板使用分布
        template_counts = {}
        phase_template_map = {}

        for decision in recent:
            template = decision["template"]
            phase = decision["phase"]

            template_counts[template] = template_counts.get(template, 0) + 1

            if phase not in phase_template_map:
                phase_template_map[phase] = {}
            phase_template_map[phase][template] = phase_template_map[phase].get(template, 0) + 1

        # 检查多样性
        unique_templates = len(template_counts)
        if unique_templates < 3:
            logging.warning(f"Low template diversity: only {unique_templates} templates used in recent decisions")

        # 检查阶段适配性
        for phase, templates in phase_template_map.items():
            dominant_template = max(templates.items(), key=lambda x: x[1])
            if dominant_template[1] > len(recent) * 0.8:  # 超过80%使用同一模板
                logging.warning(f"Phase {phase} over-relies on template {dominant_template[0]}")

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
        try:
            ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
            enemies = env.agents[agent_id].enemies
        except Exception as e:
            logging.warning(f"Error getting state for {agent_id}: {e}")
            # 返回默认状态
            ego_state = np.zeros(len(self.state_var))
            enemies = []

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

    def update_combat_phase(self, env, agent_id):
        """阶段转换条件"""
        if agent_id not in env.agents or not env.agents[agent_id].is_alive:
            return

        state = self.get_state_dict(env, agent_id)
        current_phase = self.current_phases.get(agent_id, "contact_guidance")
        distance = state["enemy_distance"]
        radar_lock = state.get("radar_lock", False)

        new_phase = current_phase
        phase_duration = self._get_phase_duration(agent_id)

        # 阶段转换
        if current_phase == "contact_guidance":
            if distance <= self.tactical_distances["detection_range"]:
                new_phase = "target_search"
                logging.info(f"Agent {agent_id}: Entered detection range ({distance:.0f}m)")

        elif current_phase == "target_search":
            if distance <= self.tactical_distances["engagement_range"]:
                new_phase = "target_identification"
                logging.info(f"Agent {agent_id}: Entered engagement range ({distance:.0f}m)")

        elif current_phase == "target_identification":
            if distance <= 50000 or radar_lock:
                new_phase = "threat_assessment"
                logging.info(f"Agent {agent_id}: Threat assessment ({distance:.0f}m, radar_lock={radar_lock})")

        # 放宽threat_assessment的退出条件
        elif current_phase == "threat_assessment":
            # 多重条件：满足任一即可进入target_allocation
            can_proceed = (
                    distance <= 55000 or  # 放宽到55km
                    radar_lock or  # 有雷达锁定就可以
                    state.get("has_warning", False) or  # 受到威胁时也可以
                    # 或者在该阶段停留足够久（自然推进）
                    self._get_phase_duration(agent_id) > 50  # 50步后自动推进
            )

            if can_proceed:
                new_phase = "target_allocation"
                reason = []
                if distance <= 55000: reason.append(f"distance={distance:.0f}m")
                if radar_lock: reason.append("radar_lock")
                if state.get("has_warning", False): reason.append("has_warning")
                if self._get_phase_duration(agent_id) > 50: reason.append("duration_timeout")

                logging.info(f"Agent {agent_id}: Target allocation ({', '.join(reason)})")

        # 后续阶段
        elif current_phase == "target_allocation":
            # 多重触发条件
            can_proceed = (
                    distance <= 50000 or  # 距离条件
                    radar_lock or  # 雷达锁定
                    self._get_phase_duration(agent_id) > 30  # 时间条件
            )
            if can_proceed:
                new_phase = "tactical_decision"
                logging.info(f"Agent {agent_id}: Tactical decision ({distance:.0f}m)")

        elif current_phase == "tactical_decision":
            # 进入发射窗口的条件
            can_launch = (
                    distance <= self.tactical_distances["launch_range"] or  # 40km
                    (radar_lock and distance <= 45000)  # 有锁定时放宽到45km
            )
            if can_launch:
                new_phase = "missile_launch"
                logging.info(f"Agent {agent_id}: Missile launch window ({distance:.0f}m)")
            elif state.get("has_warning", False):
                new_phase = "mid_guidance_defense"

        elif current_phase == "missile_launch":
            if (state.get("missile_launched", False) or
                    state.get("has_warning", False) or
                    distance <= 30000 or
                    phase_duration > 20):   # 缩短发射阶段超时
                new_phase = "mid_guidance_defense"
                logging.info(f"Agent {agent_id}: Launch -> Defense")

        elif current_phase == "mid_guidance_defense":

            # 防止长期停留在防御阶段
            if distance <= self.tactical_distances["mar_range"]:
                new_phase = "terminal_guidance"
                logging.info(f"Agent {agent_id}: Defense -> Terminal ({distance:.0f}m)")
            elif distance > 70000 and not state.get("has_warning", False) and phase_duration > 40:
                # 距离拉开且无威胁时，重新搜索
                new_phase = "target_search"
                logging.info(f"Agent {agent_id}: Defense -> Search (disengaged)")
            elif not state.get("has_warning", False) and phase_duration > 50:
                # 长时间无威胁，重新进入战术决策
                new_phase = "tactical_decision"
                logging.info(f"Agent {agent_id}: Defense -> Decision (no threat)")

        elif current_phase == "terminal_guidance":
            if (distance > 35000 or
                    state.get("missile_hit", False) or
                    not any(m.is_alive for m in env.agents[agent_id].launch_missiles) or phase_duration > 30):
                new_phase = "effect_assessment"
                logging.info(f"Agent {agent_id}: Terminal -> Assessment")

        elif current_phase == "effect_assessment":
            if distance > 60000:
                new_phase = "target_search"
            elif distance <= 40000:
                new_phase = "tactical_decision"
            elif phase_duration > 15:  # 快速完成评估
                new_phase = "target_search"

        # 更新阶段
        if new_phase != current_phase:
            self.current_phases[agent_id] = new_phase
            current_time = getattr(env, 'current_step', 0) * getattr(env, 'time_interval', 0.2)
            self._record_timeline_event(agent_id, current_phase, new_phase, distance, current_time)
            self._update_phase_recommended_templates(agent_id, new_phase, state)

    # 阶段持续时间跟踪
    def _get_phase_duration(self, agent_id):
        """获取当前阶段持续时间"""
        if not hasattr(self, 'phase_start_times'):
            self.phase_start_times = {}

        current_phase = self.current_phases.get(agent_id, "contact_guidance")
        phase_key = f"{agent_id}_{current_phase}"

        if phase_key not in self.phase_start_times:
            self.phase_start_times[phase_key] = getattr(self, 'step_count', 0)

        return getattr(self, 'step_count', 0) - self.phase_start_times[phase_key]

    # 在阶段变化时重置计时器
    def _record_timeline_event(self, agent_id, old_phase, new_phase, distance, time):
        """记录时间线事件并重置阶段计时器"""
        if not hasattr(self, 'timeline_events'):
            self.timeline_events = []

        event = {
            "agent_id": agent_id,
            "time": time,
            "phase_transition": f"{old_phase} -> {new_phase}",
            "distance": distance,
            "step": getattr(self, 'step_count', 0),
            "phase_duration": self._get_phase_duration(agent_id)  # 记录阶段持续时间
        }
        self.timeline_events.append(event)

        # 重置新阶段的计时器
        if not hasattr(self, 'phase_start_times'):
            self.phase_start_times = {}
        phase_key = f"{agent_id}_{new_phase}"
        self.phase_start_times[phase_key] = getattr(self, 'step_count', 0)

        logging.info(f"Timeline Event - {agent_id}: {old_phase} -> {new_phase} at {distance:.0f}m, "
                     f"t={time:.1f}s, duration={event['phase_duration']}steps")

    def _update_phase_recommended_templates(self, agent_id, phase, state):
        """优化的阶段模板推荐 - 平衡推荐"""
        if not hasattr(self, 'phase_recommended_templates'):
            self.phase_recommended_templates = {}

        # 平衡的模板推荐
        phase_templates = {
            "contact_guidance": [7, 1],  # F-Pole, Crank
            "target_search": [7, 1, 8],  # F-Pole, Crank, Advanced_F_Pole
            "target_identification": [1, 8, 9],  # Crank, Advanced_F_Pole, Pincer
            "threat_assessment": [1, 8, 9, 10],  # 加入更多选择
            "target_allocation": [9, 10, 11, 12, 4],  # 协同 + 攻击战术
            "tactical_decision": [4, 5, 6, 8, 9],  # 攻击战术为主
            "missile_launch": [6, 7, 8],  # Banzai, F-Pole系列
            "mid_guidance_defense": [1, 2, 3, 14],  # 防御机动组
            "terminal_guidance": [2, 3, 14],  # 强防御组
            "effect_assessment": [7, 10, 13]  # 评估和重组
        }

        # 基于威胁等级和距离动态调整
        distance = state["enemy_distance"]
        has_warning = state.get("has_warning", False)
        radar_lock = state.get("radar_lock", False)

        if has_warning:
            # 威胁时强制使用防御模板
            phase_templates[phase] = [2, 3, 14]
        elif radar_lock and distance < 50000:
            # 有锁定且距离合适时偏向攻击
            if phase in ["tactical_decision", "missile_launch"]:
                phase_templates[phase] = [4, 5, 6, 8]
        elif distance > 80000:
            # 远距离时使用远程战术
            phase_templates[phase] = [7, 1, 8]

        self.phase_recommended_templates[agent_id] = phase_templates.get(phase, [7])

        logging.debug(
            f"Agent {agent_id} phase {phase}: recommended templates {self.phase_recommended_templates[agent_id]}")

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

    def step(self, env):
        """执行一步仿真，集成战术模板和阶段管理"""
        try:
            self.step_count += 1

            # 更新所有智能体的作战阶段
            for agent_id in env.agents.keys():
                try:
                    self.update_combat_phase(env, agent_id)
                    if agent_id in self.tactical_templates:
                        state = self.get_state_dict(env, agent_id)
                        self.tactical_templates[agent_id].update_phase(state)
                except Exception as e:
                    logging.warning(f"Error updating combat phase for {agent_id}: {e}")
                    continue

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

                # 射击条件
                shoot_flag = (
                        agent.is_alive and
                        self._shoot_action.get(agent_id, False) and
                        self._remaining_missiles.get(agent_id, 0) > 0 and
                        attack_angle <= 120 and  # 65度
                        25000 <= distance <= 60000 and  # 窗口25-60km
                        shoot_interval >= self.min_attack_interval - 20 and
                        (state.get("radar_lock", False) or distance < 60000) and
                        self.current_phases.get(agent_id) in ["missile_launch", "tactical_decision",
                                                              "target_allocation"] and
                        abs(state["enemy_angle_off"]) < 150  # 150度
                )
                # 强制发射
                if (not shoot_flag and
                        agent.is_alive and
                        self._remaining_missiles.get(agent_id, 0) > 0 and
                        25000 <= distance <= 55000 and  # 缩小强制发射距离范围
                        shoot_interval >= 15 and  # 增加强制发射间隔
                        attack_angle <= 150):  # 增加角度限制
                    shoot_flag = True
                    logging.info(
                        f"Agent {agent_id} FORCED missile launch at distance={distance:.0f}m, angle={attack_angle:.1f}deg")
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
                        if isinstance(func, (TacticalRewardNew, RadarLockRewardNew, MissileHitRewardNew)):
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
                scaled_reward = np.clip(original_reward, -10.0, 10.0)  # 直接使用原始奖励
                rewards[agent_id] = np.array([scaled_reward])
                self.rewards[agent_id] = scaled_reward

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

        except Exception as e:
            logging.error(f"Critical error in MultipleCombatTask.step: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")

            # 返回默认值避免返回None
            obs = {agent_id: np.zeros(32) for agent_id in env.agents.keys()}
            share_obs = obs.copy()
            rewards = {agent_id: np.array([0.0]) for agent_id in env.agents.keys()}
            dones = {agent_id: [False] for agent_id in env.agents.keys()}
            infos = {agent_id: {} for agent_id in env.agents.keys()}
            return obs, share_obs, rewards, dones, infos
