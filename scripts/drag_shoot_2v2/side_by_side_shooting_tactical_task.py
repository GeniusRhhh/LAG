#!/usr/bin/env python3
"""
并排射击战术任务 - 完全基于拖曳射击项目的成功架构
严格照抄拖曳射击的实现，只修改战术逻辑部分
"""

import logging
import numpy as np
import torch
from enum import Enum
from envs.JSBSim.tasks.multiplecombat_task import MultipleCombatTask
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.termination_conditions.termination_condition_base import BaseTerminationCondition
from envs.JSBSim.tasks.pure_maneuvers import BasicManeuvers, CompositeManeuverExecutor


class SideBySideTermination(BaseTerminationCondition):
    """并排射击专用终止条件 - 只有双方全灭才终止"""

    def __init__(self, config):
        super().__init__(config)
        self.altitude_limit = getattr(config, 'altitude_limit', 1000)  # 1000米
        self.max_steps = getattr(config, 'max_steps', 2500)

    def get_termination(self, task, env, agent_id, info={}):
        """
        自定义终止条件：
        1. 高度过低
        2. 时间限制
        3. 极端状态
        4. 过载
        5. 双方全灭（不是单架飞机被击落）
        """
        # 检查高度过低
        current_alt = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        if current_alt <= self.altitude_limit:
            self.log(f"{agent_id} altitude too low: {current_alt:.1f}m")
            return True, False, info

        # 检查时间限制
        if env.current_step >= self.max_steps:
            self.log(f"Time limit reached: {env.current_step} steps")
            return True, False, info

        # 检查极端状态
        if env.agents[agent_id].get_property_value(c.detect_extreme_state):
            self.log(f"{agent_id} extreme state detected")
            return True, False, info

        # 检查过载状态
        if (abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_x_norm)) > 15.0 or
            abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_y_norm)) > 15.0 or
            abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_z_norm) + 1) > 15.0):
            self.log(f"{agent_id} overload detected")
            return True, False, info

        # 检查双方存活情况 - 只有当一方全灭时才终止
        red_alive = [aid for aid in ["A0100", "A0200"] if aid in env.agents and env.agents[aid].is_alive]
        blue_alive = [aid for aid in ["B0100", "B0200"] if aid in env.agents and env.agents[aid].is_alive]

        # 只有当一方全灭时才终止
        if len(red_alive) == 0:
            self.log("Red team eliminated")
            return True, True, {"termination_reason": "red_eliminated", "winner": "blue"}
        elif len(blue_alive) == 0:
            self.log("Blue team eliminated")
            return True, True, {"termination_reason": "blue_eliminated", "winner": "red"}

        # 继续仿真 - 不因为单架飞机被击落而终止
        return False, False, info


class TacticalPhase(Enum):
    """并排射击战术阶段"""
    NLT_MELD = "NLT_MELD"    # 90-81km
    MELD_MTR = "MELD_MTR"    # 81-45km
    MTR_TR = "MTR_TR"        # 45-41km
    TR_DOR = "TR_DOR"        # 41-19.6km
    DOR_DR = "DOR_DR"        # 19.6-14.5km


class SideBySideShootingTacticalTask(MultipleCombatTask):
    """
    并排射击战术任务 - 基于拖曳射击架构模式

    动作空间架构：
    - 定义的动作空间: [41, 41, 41, 30] (继承自MultipleCombatTask，但实际不使用)
    - 实际使用的动作空间: [15, 17, 7] (高层战术指令)
    - 转换机制: 高层指令 → baseline模型 → 底层飞行控制

    主要函数调用链：
    normalize_action() → _process_side_by_side_tactics() → _get_tactical_command_indices()
    → _get_leader_command_indices() / _get_wingman_command_indices() / _get_enemy_command_indices()
    → _use_lowlevel_policy() → baseline模型输出底层控制指令

    平稳飞行指令: [7, 8, 3] = [高度0m变化, 航向0°变化, 速度0m/s变化]
    """

    def __init__(self, config):
        """初始化拖曳射击任务 - 学习pure_maneuver_task的模式"""
        super().__init__(config)

        # 使用自定义终止条件 - 完全替换父类的终止条件
        self.termination_conditions = [
            SideBySideTermination(self.config),
        ]

        # 拖曳射击特定的战术距离 - 长机和僚机时间线差异
        self.tactical_distances = {
            'NLT_MELD_min': 81000,   # 81km
            'MELD_MTR_min': 50000,   # 50km - 扩大范围确保进入MTR_TR阶段
            'MTR_TR_min': 40000,     # 40km - 调整为40km
            'TR_DOR_min': 35000,     # 35km - 由于距离在增加，调整阈值确保能进入DOR_DR阶段
            'DOR_DR_min': 14500,     # 14.5km
        }

        # 僚机时间线滞后设置（掩护长机离开）
        self.wingman_delay = {
            'TR_DOR_delay': 4000,    # 僚机TR_DOR阶段滞后8km
            'DOR_DR_delay': 8000,   # DOR_DR阶段滞后10km，确保长机先完成short_skate
        }

        # 当前战术阶段
        self.current_phase = TacticalPhase.NLT_MELD

        # 导弹发射冷却时间管理（替代一次性发射限制）
        self.last_missile_launch_time = {"A0100": -999, "A0200": -999, "B0100": -999, "B0200": -999}

        # 差异化冷却时间：友方前期积极发射，敌方保持原有节奏
        self.friendly_missile_cooldown = 2.0   # 友方2秒冷却，支持快速连续发射
        self.enemy_missile_cooldown = 10.0     # 敌方10秒冷却，保持原有逻辑

        # 友方连续发射管理
        self.friendly_burst_launch = {"A0100": 0, "A0200": 0}  # 记录连续发射次数

        # 导弹发射状态记录（兼容性）
        self.missile_launched = {"A0100": False, "A0200": False, "B0100": False, "B0200": False}

        # 编队间距控制 - 并排射击特有
        self.formation_spacing = {
            'min_spacing': 1852,      # 1海里最小间距
            'max_spacing': 5556,      # 3海里最大间距
            'target_spacing': 3704,   # 2海里目标间距
        }

        # 雷达状态管理 - 使用统一雷达管理器
        from radar_manager import get_unified_radar_manager
        self.radar_manager = get_unified_radar_manager()
        logging.info("📡 统一雷达管理系统已集成到拖曳射击任务")

        # 初始状态记录 - 学习pure_maneuver_task
        self.initial_heading = {}
        self.initial_altitude = {}

        # short_skate机动状态跟踪
        self.short_skate_states = {}
        self.short_skate_start_time = {}

        # baseline模型 - 学习pure_maneuver_task的模式
        self.my_lowlevel_policy = BaselineActor()
        self._inner_rnn_states = {}

        # 添加 pure_maneuvers 机动执行器
        self.basic_maneuvers = BasicManeuvers()
        self.composite_executor = CompositeManeuverExecutor()
        
        # 机动状态跟踪
        self.active_maneuvers = {}  # 跟踪每个智能体的活跃机动
        self.maneuver_start_times = {}  # 机动开始时间
        
        # 精确机动启用标志
        self.enable_precise_maneuvers = True
        logging.info("🎯 精确机动系统已启用 - 集成 pure_maneuvers 控制")

        # 动作空间定义：[15, 17, 7] - 高层战术指令空间
        # 这些指令通过baseline模型转换为底层飞行控制指令 [41, 41, 41, 30]

        # 高度指令数组 (15个选项，索引0-14)
        self.norm_delta_altitude = np.array([
            -1500, -1000, -750, -500, -300, -150, -50, 0, 50, 150, 300, 500, 750, 1000, 1500
        ]) / 1000.0  # 索引7 = 0m变化（平稳飞行）

        # 航向指令数组 (17个选项，索引0-16)
        self.norm_delta_heading = np.array([
            -np.pi,           # 索引0:  -180°
            -2*np.pi/3,       # 索引1:  -120°
            -np.pi/2,         # 索引2:  -90°
            -5*np.pi/12,      # 索引3:  -75°
            -np.pi/3,         # 索引4:  -60°
            -np.pi/4,         # 索引5:  -45°
            -np.pi/6,         # 索引6:  -30°
            -np.pi/12,        # 索引7:  -15°
            0,                # 索引8:  0° (平稳飞行)
            np.pi/12,         # 索引9:  15°
            np.pi/6,          # 索引10: 30°
            np.pi/4,          # 索引11: 45°
            np.pi/3,          # 索引12: 60°
            5*np.pi/12,       # 索引13: 75°
            np.pi/2,          # 索引14: 90°
            2*np.pi/3,        # 索引15: 120°
            np.pi             # 索引16: 180°
        ])

        # 速度指令数组 (7个选项，索引0-6)
        self.norm_delta_velocity = np.array([-150, -100, -50, 0, 50, 100, 150]) / 100.0  # 索引3 = 0m/s变化（平稳飞行）

        # 加载baseline模型
        self._load_baseline_models()

        logging.info("DragShootTacticalTask initialized")

    def get_termination(self, env, agent_id, info={}):
        """
        完全重写终止条件 - 绕过父类的termination_conditions列表
        只有双方全灭、高度过低、或达到时间限制才终止
        """
        # 检查高度过低
        current_alt = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        if current_alt <= 1000:  # 1000米以下
            logging.info(f"{agent_id} altitude too low: {current_alt:.1f}m - terminating")
            return True, {"termination_reason": "low_altitude"}

        # 检查时间限制
        if env.current_step >= 2500:  # 最大步数
            logging.info(f"Time limit reached: {env.current_step} steps - terminating")
            return True, {"termination_reason": "timeout"}

        # 检查极端状态
        if env.agents[agent_id].get_property_value(c.detect_extreme_state):
            logging.info(f"{agent_id} extreme state detected - terminating")
            return True, {"termination_reason": "extreme_state"}

        # 检查过载状态
        if (abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_x_norm)) > 15.0 or
            abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_y_norm)) > 15.0 or
            abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_z_norm) + 1) > 15.0):
            logging.info(f"{agent_id} overload detected - terminating")
            return True, {"termination_reason": "overload"}

        # 检查双方存活情况 - 只有当一方全灭时才终止
        red_alive = [aid for aid in ["A0100", "A0200"] if aid in env.agents and env.agents[aid].is_alive]
        blue_alive = [aid for aid in ["B0100", "B0200"] if aid in env.agents and env.agents[aid].is_alive]

        # 添加调试日志
        if env.current_step % 50 == 0:  # 每50步输出一次状态
            logging.info(f"TERMINATION CHECK - Step {env.current_step}: Red alive: {red_alive}, Blue alive: {blue_alive}")

        # 只有当一方全灭时才终止
        if len(red_alive) == 0:
            logging.info("Red team eliminated - terminating")
            return True, {"termination_reason": "red_eliminated", "winner": "blue"}
        elif len(blue_alive) == 0:
            logging.info("Blue team eliminated - terminating")
            return True, {"termination_reason": "blue_eliminated", "winner": "red"}

        # 继续仿真 - 不因为单架飞机被击落而终止
        return False, info

    def _load_baseline_models(self):
        """加载baseline模型 - 完全学习pure_maneuver_task的实现"""
        try:
            model_path = get_root_dir() + '/model/baseline_model.pt'
            if torch.cuda.is_available():
                device = torch.device("cuda")
                checkpoint = torch.load(model_path, weights_only=True)
            else:
                device = torch.device("cpu")
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            self.my_lowlevel_policy.load_state_dict(checkpoint)
            self.my_lowlevel_policy.eval()
            logging.info("Successfully loaded baseline model for drag-shoot task")
        except Exception as e:
            logging.error(f"加载baseline模型失败: {e}")
            self.my_lowlevel_policy = None
    
    def normalize_action(self, env, agent_id, action):
        """
        动作归一化 - 拖曳射击战术系统
        注意：action参数在此实现中未使用，因为使用内部战术逻辑生成动作
        但必须保留此参数以符合父类接口要求
        """
        if agent_id not in env.agents or not env.agents[agent_id].is_alive:
            return np.array([0.0, 0.0, 0.0, 0.7])

        current_time = env.current_step * env.time_interval

        # 初始化状态记录 - 学习pure_maneuver_task
        if agent_id not in self.initial_heading:
            current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_rad)
            self.initial_heading[agent_id] = np.rad2deg(current_heading)
            current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
            self.initial_altitude[agent_id] = current_altitude
            logging.info(f"{agent_id} 初始状态: 航向{self.initial_heading[agent_id]:.1f}°, 高度{self.initial_altitude[agent_id]:.1f}m")

        # 处理并排射击战术动作
        result = self._process_side_by_side_tactics(env, agent_id, current_time)

        # 处理导弹发射
        self._handle_missile_launch(env, agent_id, current_time)

        return result

    def _process_side_by_side_tactics(self, env, agent_id, current_time):
        """处理并排射击战术 - 学习拖曳射击的处理模式"""
        try:
            # 更新战术阶段
            self._update_tactical_phase(env)

            # 根据智能体角色和当前阶段生成指令索引
            altitude_cmd_id, heading_cmd_id, velocity_cmd_id = self._get_tactical_command_indices(env, agent_id)

            # 使用baseline模型 - 完全学习pure_maneuver_task的_use_lowlevel_policy
            # 检查是否需要精确滚转控制
            if hasattr(self, 'active_maneuvers') and agent_id in self.active_maneuvers:
                maneuver_data = self.active_maneuvers[agent_id]
                if 'target_roll' in maneuver_data and abs(maneuver_data['target_roll']) > 2.0:
                    return self._use_lowlevel_policy_with_roll(
                        env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id, maneuver_data['target_roll']
                    )
            
            return self._use_lowlevel_policy(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)

        except Exception as e:
            logging.error(f"{agent_id} 拖曳射击战术执行错误: {e}")
            return np.array([0.0, 0.0, 0.0, 0.7])

    def _use_lowlevel_policy(self, env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id):
        """使用低级策略网络 - 完全照抄pure_maneuver_task的实现"""
        if self.my_lowlevel_policy is None:
            return np.array([0.0, 0.0, 0.0, 0.7])

        try:
            raw_obs = self.get_obs(env, agent_id)
            input_obs = np.zeros(12)

            # 安全索引访问，防止越界
            altitude_cmd_id = min(altitude_cmd_id, len(self.norm_delta_altitude) - 1)
            heading_cmd_id = min(heading_cmd_id, len(self.norm_delta_heading) - 1)
            velocity_cmd_id = min(velocity_cmd_id, len(self.norm_delta_velocity) - 1)

            input_obs[0] = self.norm_delta_altitude[altitude_cmd_id]
            input_obs[1] = self.norm_delta_heading[heading_cmd_id]
            input_obs[2] = self.norm_delta_velocity[velocity_cmd_id]
            input_obs[3:12] = raw_obs[:9]
            input_obs = np.nan_to_num(input_obs, nan=0.0)
            input_obs = np.expand_dims(input_obs, axis=0)

            if agent_id not in self._inner_rnn_states:
                self._inner_rnn_states[agent_id] = np.zeros((1, 1, 128))

            _action, _rnn_states = self.my_lowlevel_policy(
                torch.FloatTensor(input_obs),
                torch.FloatTensor(self._inner_rnn_states[agent_id])
            )
            action_output = _action.detach().cpu().numpy().squeeze(0)
            self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()

            norm_act = np.zeros(4)
            norm_act[0] = action_output[0] / 20 - 1.
            norm_act[1] = action_output[1] / 20 - 1.
            norm_act[2] = action_output[2] / 20 - 1.
            norm_act[3] = action_output[3] / 58 + 0.4

            current_alt = env.agents[agent_id].get_position()[2]
            if current_alt < 1000:
                norm_act[1] = max(norm_act[1], 0.0)
                norm_act[3] = max(norm_act[3], 0.8)

            return norm_act
        except Exception as e:
            logging.error(f"低级策略错误: {e}")
            return np.array([0.0, 0.0, 0.0, 0.7])

    def _use_lowlevel_policy_with_roll(self, env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id, target_roll):
        """使用低级策略网络，包含精确滚转控制"""
        if self.my_lowlevel_policy is None:
            return np.array([0.0, 0.0, 0.0, 0.7])

        try:
            # 基础网络处理（与原有相同）
            raw_obs = self.get_obs(env, agent_id)
            input_obs = np.zeros(12)

            # 安全索引访问
            altitude_cmd_id = min(altitude_cmd_id, len(self.norm_delta_altitude) - 1)
            heading_cmd_id = min(heading_cmd_id, len(self.norm_delta_heading) - 1)
            velocity_cmd_id = min(velocity_cmd_id, len(self.norm_delta_velocity) - 1)

            input_obs[0] = self.norm_delta_altitude[altitude_cmd_id]
            input_obs[1] = self.norm_delta_heading[heading_cmd_id]
            input_obs[2] = self.norm_delta_velocity[velocity_cmd_id]
            input_obs[3:12] = raw_obs[:9]
            input_obs = np.nan_to_num(input_obs, nan=0.0)
            input_obs = np.expand_dims(input_obs, axis=0)

            if agent_id not in self._inner_rnn_states:
                self._inner_rnn_states[agent_id] = np.zeros((1, 1, 128))

            # 神经网络处理
            _action, _rnn_states = self.my_lowlevel_policy(
                torch.FloatTensor(input_obs),
                torch.FloatTensor(self._inner_rnn_states[agent_id])
            )
            action_output = _action.detach().cpu().numpy().squeeze(0)
            self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()

            # 基础控制信号
            norm_act = np.zeros(4)
            norm_act[0] = action_output[0] / 20 - 1.
            norm_act[1] = action_output[1] / 20 - 1.
            norm_act[2] = action_output[2] / 20 - 1.
            norm_act[3] = action_output[3] / 58 + 0.4

            # 精确滚转控制 - 这是关键改进！
            current_roll = env.agents[agent_id].get_property_value(c.attitude_phi_rad)
            current_roll_deg = np.rad2deg(current_roll)
            roll_error = target_roll - current_roll_deg
            
            if abs(roll_error) > 2.0:
                # 精确的滚转控制
                roll_cmd = np.clip(roll_error / 45.0, -1.0, 1.0)
                norm_act[0] = roll_cmd  # 覆盖基础的副翼控制
                
                # 调试信息
                if env.current_step % 50 == 0:
                    logging.info(f"{agent_id} 精确滚转: 目标={target_roll:.1f}°, "
                               f"当前={current_roll_deg:.1f}°, 误差={roll_error:.1f}°, 指令={roll_cmd:.2f}")

            # 高度安全检查
            current_alt = env.agents[agent_id].get_position()[2]
            if current_alt < 1000:
                norm_act[1] = max(norm_act[1], 0.0)
                norm_act[3] = max(norm_act[3], 0.8)

            return norm_act
        except Exception as e:
            logging.error(f"精确滚转控制错误: {e}")
            return np.array([0.0, 0.0, 0.0, 0.7])

    def _convert_altitude_to_index(self, altitude_cmd):
        """高度指令转索引 - 完全照抄pure_maneuver_task"""
        altitude_values = np.array([
            -1500, -1000, -750, -500, -300, -150, -50, 0, 50, 150, 300, 500, 750, 1000, 1500
        ])
        distances = np.abs(altitude_values - altitude_cmd)
        return np.argmin(distances)

    def _convert_heading_to_index(self, heading_cmd):
        """航向指令转索引 - 完全照抄pure_maneuver_task"""
        heading_cmd = np.clip(heading_cmd, -np.pi, np.pi)
        heading_values = np.array([
            -np.pi,           # -180°
            -2*np.pi/3,       # -120°
            -np.pi/2,         # -90°
            -5*np.pi/12,      # -75°
            -np.pi/3,         # -60°
            -np.pi/4,         # -45°
            -np.pi/6,         # -30°
            -np.pi/12,        # -15°
            0,                # 0°
            np.pi/12,         # 15°
            np.pi/6,          # 30°
            np.pi/4,          # 45°
            np.pi/3,          # 60°
            5*np.pi/12,       # 75°
            np.pi/2,          # 90°
            2*np.pi/3,        # 120°
            np.pi             # 180°
        ])
        distances = np.abs(heading_values - heading_cmd)
        return np.argmin(distances)

    def _convert_velocity_to_index(self, velocity_offset):
        """速度偏移转索引 - 完全照抄pure_maneuver_task"""
        velocity_values = np.array([-150, -100, -50, 0, 50, 100, 150])
        distances = np.abs(velocity_values - velocity_offset)
        return np.argmin(distances)
    
    def step(self, env):
        """执行拖曳射击战术步骤"""
        # 更新战术阶段
        self._update_tactical_phase(env)

        # 处理导弹发射
        current_time = env.current_step * env.time_interval
        for agent_id in env._jsbsims.keys():
            if env._jsbsims[agent_id].is_alive:
                self._handle_missile_launch(env, agent_id, current_time)

        # 更新统一雷达系统状态
        self.radar_manager.update_friendly_radar_states(env, current_time)
        self.radar_manager.update_enemy_radar_states(env, current_time)

        # 详细状态信息 - 每5秒打印一次
        if env.current_step % 25 == 0:
            self._print_detailed_status(env, current_time)

        # 编队间距监控 - 专门监控30-90秒时间段（crank机动完成后到导弹发射前）
        if (30.0 <= current_time <= 90.0 and
            env.current_step % 15 == 0):  # 每3秒打印一次（15步 * 0.2秒/步 = 3秒）
            self._print_formation_spacing(env, current_time)

        # 为每个智能体生成战术动作
        obs = {}
        share_obs = {}
        rewards = {}
        dones = {}
        infos = {}

        for agent_id in env._jsbsims.keys():
            if not env._jsbsims[agent_id].is_alive:
                obs[agent_id] = np.zeros(self.obs_length)
                share_obs[agent_id] = np.zeros(self.obs_length)
                rewards[agent_id] = [-10.0]
                dones[agent_id] = [True]
                infos[agent_id] = {"agent_id": agent_id, "alive": False}
                continue

            # 获取观测
            agent_obs = self.get_obs(env, agent_id)
            obs[agent_id] = agent_obs
            share_obs[agent_id] = agent_obs

            # 注意：战术动作生成已集成到 normalize_action 方法中
            # 通过 _get_tactical_command_indices 系统处理

            # 计算奖励
            reward = self._calculate_reward(env, agent_id)
            rewards[agent_id] = [reward]

            # 检查终止条件
            done, info = self.get_termination(env, agent_id, {})
            dones[agent_id] = [done]
            infos[agent_id] = {"agent_id": agent_id, "alive": True, "phase": self.current_phase.value}

        return obs, share_obs, rewards, dones, infos
    
    def _update_tactical_phase(self, env):
        """更新战术阶段"""
        # 获取主要对抗双方
        leader_red = env._jsbsims.get("A0100")
        leader_blue = env._jsbsims.get("B0100")

        if not leader_red or not leader_blue or not leader_red.is_alive or not leader_blue.is_alive:
            return

        # 计算距离
        distance = self._calculate_distance(leader_red, leader_blue)

        # 确定当前阶段
        new_phase = self._get_phase_by_distance(distance)

        if new_phase != self.current_phase:
            current_time = env.current_step * env.time_interval
            logging.info(f"Phase transition: {self.current_phase.value} -> {new_phase.value} "
                        f"at t={current_time:.1f}s, distance={distance/1000:.1f}km")
            self.current_phase = new_phase
    
    def _get_phase_by_distance(self, distance: float) -> TacticalPhase:
        """根据距离确定战术阶段"""
        if distance > self.tactical_distances['NLT_MELD_min']:
            return TacticalPhase.NLT_MELD
        elif distance > self.tactical_distances['MELD_MTR_min']:
            return TacticalPhase.MELD_MTR
        elif distance > self.tactical_distances['MTR_TR_min']:
            return TacticalPhase.MTR_TR
        elif distance > self.tactical_distances['TR_DOR_min']:
            return TacticalPhase.TR_DOR
        else:
            return TacticalPhase.DOR_DR
    
    def _calculate_distance(self, aircraft1, aircraft2) -> float:
        """计算两架飞机之间的距离"""
        pos1 = aircraft1.get_position()
        pos2 = aircraft2.get_position()
        return np.linalg.norm(pos1 - pos2)

    def _calculate_formation_spacing(self, env, wingman_id):
        """计算编队间距 - 并排射击特有功能"""
        leader_id = "A0100"
        if (leader_id in env.agents and env.agents[leader_id].is_alive and
            wingman_id in env.agents and env.agents[wingman_id].is_alive):
            return self._calculate_distance(env.agents[leader_id], env.agents[wingman_id])
        return self.formation_spacing['target_spacing']  # 默认目标间距

    def _calculate_enemy_formation_spacing(self, env, enemy_wingman_id):
        """计算敌方编队间距 - 敌方专用功能"""
        enemy_leader_id = "B0100"
        if (enemy_leader_id in env.agents and env.agents[enemy_leader_id].is_alive and
            enemy_wingman_id in env.agents and env.agents[enemy_wingman_id].is_alive):
            return self._calculate_distance(env.agents[enemy_leader_id], env.agents[enemy_wingman_id])
        return self.formation_spacing['target_spacing']  # 默认目标间距
    
    # 注意：_get_tactical_action、_get_leader_action、_get_wingman_action 等函数已被删除
    # 这些是废弃的旧动作系统，实际使用的是 _get_tactical_command_indices 系统
    
    # 注意：_get_enemy_action 函数已被删除，敌方AI现在通过 _get_enemy_command_indices 处理

    # 注意：_get_enemy_action_fallback 函数已被删除，敌方AI现在通过 _get_enemy_command_indices 处理
    
    def _get_tactical_command_indices(self, env, agent_id: str):
        """生成战术指令索引 - 基于拖曳射击逻辑"""
        # 根据智能体角色和当前阶段生成指令
        if agent_id == "A0100":  # 己方长机
            return self._get_leader_command_indices(env, agent_id)
        elif agent_id == "A0200":  # 己方僚机
            return self._get_wingman_command_indices(env, agent_id)
        elif agent_id.startswith("B"):  # 敌方
            return self._get_enemy_command_indices(env, agent_id)
        else:
            # 默认平稳飞行：高度0m变化，航向0°变化，速度0m/s变化
            return 7, 8, 3  # [15,17,7]动作空间中的平稳飞行指令

    def _get_leader_command_indices(self, env, agent_id: str):
        """长机战术指令索引 - 基于并排射击战术"""
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))

        # 条件1：已经开始short_skate机动（防止中断）
        if agent_id in self.short_skate_states:
            current_time = env.current_step * env.time_interval
            return self._execute_short_skate(env, agent_id, current_time)

        if self.current_phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR, TacticalPhase.MTR_TR]:
            # 并排射击特色：长机保持平稳飞行朝北接敌（0°）
            return self._maintain_heading_precise(env, agent_id, 0.0)

        elif self.current_phase == TacticalPhase.TR_DOR:
            # 长机在TR_DOR阶段：检查是否应该执行short_skate机动
            current_time = env.current_step * env.time_interval
            # 如果已经发射过导弹且距离上次发射超过5秒，执行左侧short_skate机动
            last_launch = self.last_missile_launch_time.get(agent_id, -999)
            if last_launch > 0 and (current_time - last_launch) > 5.0:
                # 发射导弹后执行左侧short_skate机动
                return self._execute_short_skate_precise(env, agent_id, current_time, direction="LEFT")
            else:
                # 继续精确的平稳飞行等待发射时机
                return self._maintain_heading_precise(env, agent_id, 0.0)

        elif self.current_phase == TacticalPhase.DOR_DR:
            # DOR_DR阶段：长机执行左侧short_skate机动
            current_time = env.current_step * env.time_interval
            return self._execute_short_skate_precise(env, agent_id, current_time, direction="LEFT")
        else:
            return 7, 8, 3

    def _get_wingman_phase_by_distance(self, distance: float) -> TacticalPhase:
        """僚机独立的战术阶段判断 - 体现时间线滞后"""
        # 僚机使用滞后距离判断阶段（延迟 = 距离减少，更近才执行）
        if distance > self.tactical_distances['NLT_MELD_min']:
            return TacticalPhase.NLT_MELD
        elif distance > self.tactical_distances['MELD_MTR_min']:
            return TacticalPhase.MELD_MTR
        elif distance > self.tactical_distances['MTR_TR_min']:
            return TacticalPhase.MTR_TR
        elif distance > (self.tactical_distances['TR_DOR_min'] - self.wingman_delay['TR_DOR_delay']):
            return TacticalPhase.TR_DOR  # 35km - 8km = 27km
        elif distance > (self.tactical_distances['DOR_DR_min'] - self.wingman_delay['DOR_DR_delay']):
            return TacticalPhase.DOR_DR  # 14.5km - 10km = 4.5km
        else:
            return TacticalPhase.DOR_DR

    def _get_wingman_command_indices(self, env, agent_id: str):
        """僚机战术指令索引 - 基于并排射击战术"""
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))

        # 条件1：已经开始short_skate机动（防止中断）
        if agent_id in self.short_skate_states:
            current_time = env.current_step * env.time_interval
            return self._execute_short_skate_precise(env, agent_id, current_time)

        # 计算编队间距 - 并排射击特有功能
        formation_spacing = self._calculate_formation_spacing(env, agent_id)

        if self.current_phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR, TacticalPhase.MTR_TR]:
            # 并排射击特色：僚机保持与长机的平行航向，但保持适当间距
            if formation_spacing < self.formation_spacing['min_spacing']:
                # 间距过小，右偏增加间距
                logging.info(f"⚠️ {agent_id} 编队间距过小({formation_spacing/1852:.1f}海里)，右偏增加间距")
                return self._maintain_heading_precise(env, agent_id, 15.0)  # 右偏15度
            elif formation_spacing > self.formation_spacing['max_spacing']:
                # 间距过大，左偏减少间距
                logging.info(f"⚠️ {agent_id} 编队间距过大({formation_spacing/1852:.1f}海里)，左偏减少间距")
                return self._maintain_heading_precise(env, agent_id, -15.0)  # 左偏15度
            else:
                # 间距合理，保持平行航向
                logging.info(f"✅ 编队间距合理: {formation_spacing/1852:.1f}海里")
                return self._maintain_heading_precise(env, agent_id, 0.0)

        elif self.current_phase == TacticalPhase.TR_DOR:
            # 僚机在TR_DOR阶段：检查是否应该执行short_skate机动
            current_time = env.current_step * env.time_interval
            # 如果已经发射过导弹且距离上次发射超过5秒，执行右侧short_skate机动
            last_launch = self.last_missile_launch_time.get(agent_id, -999)
            if last_launch > 0 and (current_time - last_launch) > 5.0:
                # 发射导弹后执行右侧short_skate机动
                return self._execute_short_skate_precise(env, agent_id, current_time, direction="RIGHT")
            else:
                # 继续保持编队间距
                return self._maintain_heading_precise(env, agent_id, 0.0)

        elif self.current_phase == TacticalPhase.DOR_DR:
            # DOR_DR阶段：僚机执行精确的右侧short_skate机动
            current_time = env.current_step * env.time_interval
            return self._execute_short_skate_precise(env, agent_id, current_time, direction="RIGHT")
        else:
            return 7, 8, 3

    def _normalize_angle_diff(self, angle_diff):
        """标准化角度差值到[-180, 180]范围"""
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        return angle_diff

    def _init_short_skate(self, agent_id, current_time, direction="LEFT"):
        """初始化short_skate机动状态"""
        # 并排射击特色：长机左转，僚机右转
        if agent_id == "A0100":  # 长机左侧返航
            crank_angle = -40.0
            turn_cold_angle = -100.0
        elif agent_id == "A0200":  # 僚机右侧返航
            crank_angle = 40.0
            turn_cold_angle = 100.0
        else:  # 敌方保持原逻辑
            crank_angle = -40.0 if agent_id.startswith('A') else 40.0
            turn_cold_angle = -100.0 if agent_id.startswith('A') else 100.0

        self.short_skate_states[agent_id] = {
            "phase": "crank",  # crank -> turn_cold -> escape
            "phase_start_time": current_time,
            "total_start_time": current_time,
            "crank_angle": crank_angle,
            "turn_cold_angle": turn_cold_angle,
            "initial_heading": None,
            "initial_altitude": None,
            "direction": direction
        }
        self.short_skate_start_time[agent_id] = current_time

    def _execute_short_skate(self, env, agent_id, current_time):
        """执行short_skate机动 - 参考pure_maneuvers实现"""
        if agent_id not in self.short_skate_states:
            self._init_short_skate(agent_id, current_time)

        state = self.short_skate_states[agent_id]
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))

        if state["initial_heading"] is None:
            state["initial_heading"] = current_heading

        phase_time = current_time - state["phase_start_time"]

        # 修复：僚机short_skate时间延长，体现掩护长机的战术意图
        if agent_id == "A0200":  # 僚机
            crank_duration = 18.0  # 延长到18秒（原来是12秒）
            turn_cold_duration = 30.0  # 延长到35秒（原来是25秒）
            escape_duration = 22.0  # 延长到25秒（原来是20秒）
        else:
            crank_duration = 6.0
            turn_cold_duration = 15.0
            escape_duration = 15.0

        # 阶段1：Crank机动 - 左侧40度
        if state["phase"] == "crank":
            if phase_time < crank_duration:
                target_heading = state["initial_heading"] + state["crank_angle"]
                target_heading = target_heading % 360
                heading_diff = self._normalize_angle_diff(target_heading - current_heading)
                if abs(heading_diff) > 5.0:
                    return 7, 6 if heading_diff < 0 else 10, 3  # 左转或右转
                else:
                    return 7, 8, 3  # 保持航向
            else:
                # 进入turn_cold阶段
                state["phase"] = "turn_cold"
                state["phase_start_time"] = current_time
                state["turn_cold_start_heading"] = current_heading

        # 阶段2：Turn Cold - 快速掉头100度
        elif state["phase"] == "turn_cold":
            if phase_time < turn_cold_duration:
                target_heading = state["turn_cold_start_heading"] + state["turn_cold_angle"]
                target_heading = target_heading % 360
                heading_diff = self._normalize_angle_diff(target_heading - current_heading)
                if abs(heading_diff) > 5.0:
                    return 7, 6 if heading_diff < 0 else 10, 3  # 快速转弯
                else:
                    return 7, 8, 3  # 保持航向
            else:
                # 进入escape阶段
                state["phase"] = "escape"
                state["phase_start_time"] = current_time

        # 阶段3：加速逃离
        elif state["phase"] == "escape":
            if phase_time < escape_duration:
                return 7, 8, 5  # 保持航向，加速
            else:
                # 完成short_skate，返航到初始航向的反方向
                if agent_id.startswith('A'):  # 我方：初始0°（南向），返回180°（南向）
                    target_heading = 180.0  # 我方返回南向
                elif agent_id.startswith('B'):  # 敌方：初始180°（北向），返回0°（北向）
                    target_heading = 0.0  # 敌方返回北向
                else:
                    target_heading = 180.0

                heading_diff = self._normalize_angle_diff(target_heading - current_heading)
                if abs(heading_diff) > 5.0:
                    return 7, 6 if heading_diff < 0 else 10, 3
                else:
                    return 7, 8, 3

        return 7, 8, 3  # 默认保持航向

    def _execute_short_skate_precise(self, env, agent_id, current_time, direction="LEFT"):
        """执行精确的 Short Skate 机动 - 支持左右方向"""
        if agent_id not in self.short_skate_states:
            self._init_short_skate(agent_id, current_time, direction)

        state = self.short_skate_states[agent_id]
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        current_velocity = env.agents[agent_id].get_property_value(c.velocities_u_mps)

        if state["initial_heading"] is None:
            state["initial_heading"] = current_heading
            state["initial_altitude"] = current_altitude

        phase_time = current_time - state["phase_start_time"]

        # 时间参数（保持原有的战术时序）
        if agent_id == "A0200":  # 僚机
            crank_duration = 18.0
            turn_cold_duration = 30.0
            escape_duration = 22.0
        else:
            crank_duration = 6.0
            turn_cold_duration = 15.0
            escape_duration = 15.0

        # 阶段1：Crank机动 - 使用 pure_maneuvers 精确转弯
        if state["phase"] == "crank":
            if phase_time < crank_duration:
                # 使用 BasicManeuvers.turn 进行精确的 Crank 转弯
                turn_angle = state["crank_angle"]  # -40.0 或 40.0
                turn_rate = 4.0  # 适中的转弯率
                
                result = self.basic_maneuvers.turn(
                    phase_time,
                    state["initial_heading"], 
                    turn_angle,
                    turn_rate
                )
                
                phase, target_heading, target_altitude, velocity_offset, target_roll = result
                
                if phase is None:
                    return 7, 8, 3  # 机动完成，保持状态
                
                # 存储精确控制数据
                if target_roll is not None:
                    self.active_maneuvers[agent_id] = {'target_roll': target_roll}
                
                # 转换为索引（保持与原系统兼容）
                return self._convert_maneuver_result_to_indices(
                    env, agent_id, target_heading, target_altitude, velocity_offset, target_roll,
                    state["initial_heading"], state["initial_altitude"]
                )
            else:
                # 进入 turn_cold 阶段
                state["phase"] = "turn_cold"
                state["phase_start_time"] = current_time
                state["turn_cold_start_heading"] = current_heading

        # 阶段2：Turn Cold - 精确的快速掉头
        elif state["phase"] == "turn_cold":
            if phase_time < turn_cold_duration:
                # 使用 BasicManeuvers.turn 进行精确的快速转向
                remaining_angle = state["turn_cold_angle"]  # -100.0 或 100.0
                turn_rate = 6.0  # 更快的转弯率，体现 "快速脱离"
                
                result = self.basic_maneuvers.turn(
                    phase_time,
                    state["turn_cold_start_heading"],
                    remaining_angle,
                    turn_rate
                )
                
                phase, target_heading, target_altitude, velocity_offset, target_roll = result
                
                if phase is None:
                    return 7, 8, 3
                
                # 存储精确控制数据
                if target_roll is not None:
                    self.active_maneuvers[agent_id] = {'target_roll': target_roll}
                
                return self._convert_maneuver_result_to_indices(
                    env, agent_id, target_heading, target_altitude, velocity_offset, target_roll,
                    state["turn_cold_start_heading"], current_altitude
                )
            else:
                # 进入逃离阶段
                state["phase"] = "escape"
                state["phase_start_time"] = current_time

        # 阶段3：加速逃离 - 使用 pure_maneuvers 的加速机动
        elif state["phase"] == "escape":
            if phase_time < escape_duration:
                # 使用 BasicManeuvers.accelerate_escape 进行精确的逃离
                escape_heading = current_heading  # 保持当前航向
                
                result = self.basic_maneuvers.accelerate_escape(
                    phase_time,
                    escape_heading,
                    escape_duration,
                    50.0  # 加速50m/s
                )
                
                phase, target_heading, target_altitude, velocity_offset, target_roll = result
                
                # 存储精确控制数据（如果有）
                if target_roll is not None:
                    self.active_maneuvers[agent_id] = {'target_roll': target_roll}
                
                return self._convert_maneuver_result_to_indices(
                    env, agent_id, target_heading, target_altitude, velocity_offset, target_roll,
                    escape_heading, current_altitude
                )
            else:
                # Short Skate 完成，清除精确控制数据
                if agent_id in self.active_maneuvers:
                    del self.active_maneuvers[agent_id]
                
                # 返航
                if agent_id.startswith('A'):
                    target_heading = 180.0  # 我方返回南向
                else:
                    target_heading = 0.0    # 敌方返回北向
                
                heading_diff = self._normalize_angle_diff(target_heading - current_heading)
                if abs(heading_diff) > 5.0:
                    return 7, 6 if heading_diff < 0 else 10, 3
                else:
                    return 7, 8, 3

        return 7, 8, 3  # 默认保持航向

    def _convert_maneuver_result_to_indices(self, env, agent_id, target_heading, target_altitude, 
                                           velocity_offset, target_roll, initial_heading, initial_altitude):
        """将 pure_maneuvers 的结果转换为拖曳射击兼容的索引"""
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        current_velocity = env.agents[agent_id].get_property_value(c.velocities_u_mps)
        
        # 默认索引
        altitude_cmd_id = 7  # 保持高度
        heading_cmd_id = 8   # 保持航向
        velocity_cmd_id = 3  # 保持速度
        
        # 高度控制
        if target_altitude is not None:
            altitude_diff = target_altitude - current_altitude
            if abs(altitude_diff) > 5.0:
                altitude_cmd_id = self._convert_altitude_to_index(altitude_diff)
        else:
            # 保持初始高度
            if initial_altitude is not None:
                altitude_diff = initial_altitude - current_altitude
                if abs(altitude_diff) > 10.0:  # 只有偏离较大时才纠正
                    altitude_cmd_id = self._convert_altitude_to_index(altitude_diff)
        
        # 航向控制 - 这是关键！
        if target_heading is not None:
            heading_diff = target_heading - current_heading
            while heading_diff > 180: heading_diff -= 360
            while heading_diff < -180: heading_diff += 360
            
            # 使用更精确的控制阈值
            if abs(heading_diff) > 1.0:  # 1度精度
                heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))
        
        # 速度控制
        if velocity_offset is not None and abs(velocity_offset) > 2.0:
            velocity_cmd_id = self._convert_velocity_to_index(velocity_offset)
        
        # 如果有滚转角要求，使用专门的滚转控制
        if target_roll is not None and abs(target_roll) > 2.0:
            # 使用精确滚转控制，但仍然返回索引值（不使用低级策略）
            logging.debug(f"{agent_id} 精确滚转控制: 目标滚转角={target_roll:.1f}°")
            # 这里只返回索引，滚转控制将在环境中通过 normalize_action 处理
            return altitude_cmd_id, heading_cmd_id, velocity_cmd_id
        else:
            # 返回索引值，保持与原有系统兼容
            return altitude_cmd_id, heading_cmd_id, velocity_cmd_id

    def _maintain_heading_precise(self, env, agent_id, target_heading, duration=10.0):
        """精确的航向保持 - 优化长机航向控制精度"""
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)

        # 计算航向差值
        heading_diff = target_heading - current_heading
        while heading_diff > 180: heading_diff -= 360
        while heading_diff < -180: heading_diff += 360

        # 默认索引：[15,17,7]动作空间中的平稳飞行
        altitude_cmd_id = 7  # 高度索引7 = 0m变化（保持高度）
        heading_cmd_id = 8   # 航向索引8 = 0°变化（保持航向）
        velocity_cmd_id = 3  # 速度索引3 = 0m/s变化（保持速度）

        # 针对长机A0100的超精确航向控制（目标0°）
        if agent_id == "A0100" and target_heading == 0.0:
            # 超精确控制：0.5度精度，确保长机保持正北向
            if abs(heading_diff) > 0.5:
                heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))
                if env.current_step % 50 == 0:  # 增加日志频率用于调试
                    logging.debug(f"{agent_id} 超精确航向控制: 目标={target_heading:.1f}°, "
                                 f"当前={current_heading:.1f}°, 差值={heading_diff:.1f}°")
        else:
            # 其他飞机的标准精确控制：1度精度
            if abs(heading_diff) > 1.0:
                heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))
                if env.current_step % 100 == 0:  # 减少日志频率
                    logging.debug(f"{agent_id} 精确航向: 目标={target_heading:.1f}°, "
                                 f"当前={current_heading:.1f}°, 差值={heading_diff:.1f}°")

        return altitude_cmd_id, heading_cmd_id, velocity_cmd_id

    def _get_min_distance_to_enemy(self, env, agent_id: str):
        """获取到最近敌机的距离"""
        if not env.agents[agent_id].is_alive:
            return float('inf')

        my_pos = np.array([
            env.agents[agent_id].get_property_value(c.position_long_gc_deg),
            env.agents[agent_id].get_property_value(c.position_lat_gc_deg),
            env.agents[agent_id].get_property_value(c.position_h_sl_m)
        ])

        min_distance = float('inf')
        for enemy_id in env.agents:
            if enemy_id.startswith('A') and agent_id.startswith('A'):
                continue  # 同队
            if enemy_id.startswith('B') and agent_id.startswith('B'):
                continue  # 同队
            if not env.agents[enemy_id].is_alive:
                continue

            enemy_pos = np.array([
                env.agents[enemy_id].get_property_value(c.position_long_gc_deg),
                env.agents[enemy_id].get_property_value(c.position_lat_gc_deg),
                env.agents[enemy_id].get_property_value(c.position_h_sl_m)
            ])

            # 计算距离（简化为欧几里得距离）
            distance = np.linalg.norm(my_pos - enemy_pos) * 111000  # 转换为米
            min_distance = min(min_distance, distance)

        return min_distance

    def _get_enemy_command_indices(self, env, agent_id: str):
        """敌方战术指令索引 - 使用统一敌方AI系统"""
        current_time = env.current_step * env.time_interval

        try:
            # 检查是否有集成的统一敌方AI系统
            if hasattr(self, 'unified_enemy_ai'):
                altitude_cmd, heading_cmd, velocity_cmd = self.unified_enemy_ai.get_enemy_action(
                    env, agent_id, current_time
                )
                logging.debug(f"✅ {agent_id} 统一敌方AI指令: ({altitude_cmd}, {heading_cmd}, {velocity_cmd})")
                return altitude_cmd, heading_cmd, velocity_cmd
            else:
                logging.warning(f"⚠️ {agent_id} 未找到统一敌方AI系统，使用备用方案")
                return self._get_enemy_command_indices_fallback(env, agent_id, current_time)

        except Exception as e:
            logging.error(f"统一敌方AI执行错误: {e}")
            return self._get_enemy_command_indices_fallback(env, agent_id, current_time)

    def _get_enemy_command_indices_fallback(self, env, agent_id: str, current_time: float):
        """敌方战术指令索引备用方案 - 原有逻辑"""
        # 根据敌方角色分配不同的战术逻辑
        if agent_id == "B0100":  # 敌方长机
            return self._get_enemy_leader_command_indices(env, agent_id)
        elif agent_id == "B0200":  # 敌方僚机
            return self._get_enemy_wingman_command_indices(env, agent_id)
        else:
            # 默认平稳飞行
            return 7, 8, 3

    def _get_enemy_leader_command_indices(self, env, agent_id: str):
        """敌方长机战术指令索引 - 完全复制友方长机逻辑"""
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))

        # 条件1：已经开始short_skate机动（防止中断）
        if agent_id in self.short_skate_states:
            current_time = env.current_step * env.time_interval
            return self._execute_short_skate(env, agent_id, current_time)

        if self.current_phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR, TacticalPhase.MTR_TR]:
            # 敌方长机保持南向接敌（180°）
            return self._maintain_heading_precise(env, agent_id, 180.0)

        elif self.current_phase == TacticalPhase.TR_DOR:
            # 敌方长机在TR_DOR阶段：检查是否应该执行short_skate机动
            current_time = env.current_step * env.time_interval
            # 如果已经发射过导弹且距离上次发射超过5秒，执行右侧short_skate机动
            last_launch = self.last_missile_launch_time.get(agent_id, -999)
            if last_launch > 0 and (current_time - last_launch) > 5.0:
                # 发射导弹后执行右侧short_skate机动（与友方相反）
                return self._execute_short_skate_precise(env, agent_id, current_time, direction="RIGHT")
            else:
                # 继续精确的平稳飞行等待发射时机
                return self._maintain_heading_precise(env, agent_id, 180.0)

        elif self.current_phase == TacticalPhase.DOR_DR:
            # DOR_DR阶段：敌方长机执行右侧short_skate机动
            current_time = env.current_step * env.time_interval
            return self._execute_short_skate_precise(env, agent_id, current_time, direction="RIGHT")
        else:
            return 7, 8, 3

    def _get_enemy_wingman_command_indices(self, env, agent_id: str):
        """敌方僚机战术指令索引 - 完全复制友方僚机逻辑"""
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))

        # 条件1：已经开始short_skate机动（防止中断）
        if agent_id in self.short_skate_states:
            current_time = env.current_step * env.time_interval
            return self._execute_short_skate_precise(env, agent_id, current_time)

        # 计算敌方编队间距
        enemy_formation_spacing = self._calculate_enemy_formation_spacing(env, agent_id)

        if self.current_phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR, TacticalPhase.MTR_TR]:
            # 敌方僚机保持与敌方长机的平行航向，但保持适当间距
            if enemy_formation_spacing < self.formation_spacing['min_spacing']:
                # 间距过小，左偏增加间距
                logging.info(f"⚠️ {agent_id} 敌方编队间距过小({enemy_formation_spacing/1852:.1f}海里)，左偏增加间距")
                return self._maintain_heading_precise(env, agent_id, 165.0)  # 左偏15度
            elif enemy_formation_spacing > self.formation_spacing['max_spacing']:
                # 间距过大，右偏减少间距
                logging.info(f"⚠️ {agent_id} 敌方编队间距过大({enemy_formation_spacing/1852:.1f}海里)，右偏减少间距")
                return self._maintain_heading_precise(env, agent_id, 195.0)  # 右偏15度
            else:
                # 间距合理，保持平行航向
                logging.info(f"✅ 敌方编队间距合理: {enemy_formation_spacing/1852:.1f}海里")
                return self._maintain_heading_precise(env, agent_id, 180.0)

        elif self.current_phase == TacticalPhase.TR_DOR:
            # 敌方僚机在TR_DOR阶段：检查是否应该执行short_skate机动
            current_time = env.current_step * env.time_interval
            # 如果已经发射过导弹且距离上次发射超过5秒，执行左侧short_skate机动
            last_launch = self.last_missile_launch_time.get(agent_id, -999)
            if last_launch > 0 and (current_time - last_launch) > 5.0:
                # 发射导弹后执行左侧short_skate机动
                return self._execute_short_skate_precise(env, agent_id, current_time, direction="LEFT")
            else:
                # 继续保持编队间距
                return self._maintain_heading_precise(env, agent_id, 180.0)

        elif self.current_phase == TacticalPhase.DOR_DR:
            # DOR_DR阶段：敌方僚机执行左侧short_skate机动
            current_time = env.current_step * env.time_interval
            return self._execute_short_skate_precise(env, agent_id, current_time, direction="LEFT")
        else:
            return 7, 8, 3



    def reset(self, env):
        """重置任务状态 - 学习pure_maneuver_task的reset模式"""
        super().reset(env)
        self.current_phase = TacticalPhase.NLT_MELD
        self.last_missile_launch_time = {"A0100": -999, "A0200": -999, "B0100": -999, "B0200": -999}
        self.friendly_burst_launch = {"A0100": 0, "A0200": 0}
        self.initial_heading.clear()
        self.initial_altitude.clear()
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        logging.info("DragShootTacticalTask reset completed")
        return super().reset(env)

    def _calculate_reward(self, env, agent_id: str):
        """计算奖励 - 基于父类实现"""
        if not env.agents[agent_id].is_alive:
            return -10.0

        # 基本存活奖励
        reward = 1.0

        # 高度奖励
        altitude = env.agents[agent_id].get_position()[2]
        if altitude < 1000:
            reward -= 5.0  # 低高度惩罚
        elif 5000 <= altitude <= 8000:
            reward += 1.0  # 合适高度奖励

        return reward

    def _handle_missile_launch(self, env, agent_id: str, current_time: float):
        """
        处理导弹发射 - 友方前期积极发射策略

        友方发射策略：
        - MTR_TR和TR_DOR阶段：积极发射，2秒冷却，支持连续发射
        - DOR_DR阶段（返航）：完全禁止发射
        - 前期可一次性发射2枚导弹（距离40-48km时）

        敌方发射策略：
        - 保持原有逻辑：10秒冷却时间
        - 不受返航限制影响
        """
        if not env.agents[agent_id].is_alive:
            return

        # 检查导弹数量
        if env.agents[agent_id].num_missiles <= 0:
            return

        # 友方返航期间禁止发射导弹
        if agent_id.startswith('A') and self.current_phase == TacticalPhase.DOR_DR:
            logging.info(f"{agent_id} 返航期间禁止发射导弹")
            return

        # 差异化冷却时间检查
        last_launch = self.last_missile_launch_time.get(agent_id, -999)
        if agent_id.startswith('A'):  # 友方使用短冷却时间
            cooldown = self.friendly_missile_cooldown
        else:  # 敌方使用长冷却时间
            cooldown = self.enemy_missile_cooldown

        if current_time - last_launch < cooldown:
            return

        # 找到目标
        target = self._find_target(env, agent_id)
        if not target:
            return

        distance = self._calculate_distance(env.agents[agent_id], target)
        # 新增：检查目标是否即将被击落
        target_under_threat = self._check_target_under_threat(env, target, current_time)

        # 根据拖曳射击战术确定发射条件
        should_launch = False

        if agent_id == "A0100":  # 己方长机积极发射策略
            # 扩展发射阶段：MTR_TR和TR_DOR阶段都可以发射
            in_launch_phase = self.current_phase in [TacticalPhase.MTR_TR, TacticalPhase.TR_DOR]
            # 扩展发射距离：40-50km范围内都可以发射
            in_launch_range = 40000 <= distance <= 50000

            should_launch = in_launch_phase and in_launch_range

            if should_launch:
                logging.info(f"A0100长机发射条件满足: 阶段={self.current_phase.value}, 距离={distance/1000:.1f}km")
        elif agent_id == "A0200":  # 己方僚机积极发射策略
            # 扩展发射阶段：MTR_TR和TR_DOR阶段都可以发射
            in_launch_phase = self.current_phase in [TacticalPhase.MTR_TR, TacticalPhase.TR_DOR]
            # 僚机发射距离稍近一些：35-45km范围
            in_launch_range = 35000 <= distance <= 45000

            # 添加速度检查，确保发射时飞机速度正常
            aircraft_speed = np.linalg.norm(env.agents[agent_id].get_velocity())
            speed_ok = aircraft_speed > 150

            # 避免向即将被击落的目标发射
            if target_under_threat:
                logging.info(f"A0200: 目标{target.uid}即将被击落，取消发射")
                should_launch = False
            else:
                should_launch = in_launch_phase and in_launch_range and speed_ok

                if should_launch:
                    logging.info(f"A0200僚机发射条件满足: 阶段={self.current_phase.value}, 距离={distance/1000:.1f}km")

        elif agent_id == "B0100":  # 敌方长机 - 智能发射逻辑
            # 使用敌方AI的发射判断
            should_launch = self._enemy_should_launch_missile(env, agent_id, target, distance, current_time)
            
            # 更宽松的条件：在任何阶段，只要距离合适就发射
            if not should_launch and distance <= 60000:  # 移除missile_launched限制
                logging.info(f"B0100宽松条件发射: 阶段={self.current_phase.value}, 距离={distance/1000:.1f}km")
                should_launch = True
        elif agent_id == "B0200":  # 敌方僚机 - 智能发射逻辑
            # 使用敌方AI的发射判断
            should_launch = self._enemy_should_launch_missile(env, agent_id, target, distance, current_time)

        # 敌方第二轮发射逻辑 - 在DOR_DR阶段且距离较近时
        if agent_id.startswith('B') and self.current_phase == TacticalPhase.DOR_DR:
            if 14000 <= distance <= 18000:  # 更近的距离进行第二轮发射
                # 检查是否还有导弹
                if env.agents[agent_id].num_missiles > 0:
                    # 重置发射状态，允许第二轮发射
                    if not hasattr(self, 'second_launch_done'):
                        self.second_launch_done = {}
                    if agent_id not in self.second_launch_done:
                        self.second_launch_done[agent_id] = False

                    should_launch = should_launch or not self.second_launch_done.get(agent_id, False)

        if should_launch:
            # 友方支持连续发射机制
            if agent_id.startswith('A'):
                self._launch_friendly_missiles(env, agent_id, target, current_time)
            else:
                self._launch_missile(env, agent_id, target, current_time)

    def _launch_friendly_missiles(self, env, agent_id: str, target, current_time: float):
        """友方连续发射机制 - 支持在合适条件下一次性发射多枚导弹"""
        aircraft = env.agents[agent_id]
        distance = self._calculate_distance(aircraft, target)

        # 确定发射数量
        missiles_to_launch = 1  # 默认发射1枚

        # 在前期阶段且距离合适时，考虑发射2枚导弹
        if (self.current_phase in [TacticalPhase.MTR_TR, TacticalPhase.TR_DOR] and
            aircraft.num_missiles >= 2 and
            40000 <= distance <= 48000):  # 最佳发射距离

            # 检查是否已经进行过连续发射
            burst_count = self.friendly_burst_launch.get(agent_id, 0)
            if burst_count == 0:  # 第一次连续发射机会
                missiles_to_launch = 2
                self.friendly_burst_launch[agent_id] = 1
                logging.info(f"{agent_id} 前期积极发射策略：一次性发射2枚导弹")

        # 执行发射
        for i in range(missiles_to_launch):
            if aircraft.num_missiles > 0:
                self._launch_missile(env, agent_id, target, current_time)
                if i < missiles_to_launch - 1:  # 不是最后一枚导弹
                    # 短暂延迟，模拟连续发射
                    current_time += 0.5  # 0.5秒间隔

    def _enemy_should_launch_missile(self, env, agent_id: str, target, distance: float, current_time: float) -> bool:
        """敌方智能导弹发射判断"""
        # 基本条件检查
        if env.agents[agent_id].num_missiles <= 0:
            return False

        # 敌方使用长冷却时间
        last_launch = self.last_missile_launch_time.get(agent_id, -999)
        if current_time - last_launch < self.enemy_missile_cooldown:
            return False

        # 使用简化的威胁评估逻辑，避免旧AI模块错误
        try:
            # 基于距离的简化威胁评估
            if distance < 25000:  # 25km内为高威胁
                threat_level = "HIGH"
            elif distance < 40000:  # 40km内为中威胁
                threat_level = "MEDIUM"
            else:
                threat_level = "LOW"

            # 基于威胁等级和距离决定发射
            if threat_level in ["HIGH", "MEDIUM"]:
                # 在威胁下，更积极地发射
                if 25000 <= distance <= 60000:
                    logging.info(f"{agent_id} 威胁发射: 威胁等级={threat_level}, 距离={distance/1000:.1f}km")
                    return True

            # 正常发射条件
            if agent_id == "B0100":
                # 长机：在TR_DOR阶段或更早发射
                if self.current_phase in [TacticalPhase.MTR_TR, TacticalPhase.TR_DOR]:
                    if 35000 <= distance <= 55000:
                        logging.info(f"{agent_id} 正常发射: 阶段={self.current_phase.value}, 距离={distance/1000:.1f}km")
                        return True

            elif agent_id == "B0200":
                # 僚机：稍晚发射，距离更近
                if self.current_phase in [TacticalPhase.TR_DOR, TacticalPhase.DOR_DR]:
                    if 30000 <= distance <= 50000:
                        logging.info(f"{agent_id} 正常发射: 阶段={self.current_phase.value}, 距离={distance/1000:.1f}km")
                        return True

            return False

        except Exception as e:
            logging.error(f"发射决策错误: {e}")
            # 备用发射逻辑
            if 30000 <= distance <= 50000:
                return True
            return False

    def _check_target_under_threat(self, env, target, current_time):
        """检查目标是否即将被击落"""
        target_id = target.uid

        # 检查是否有导弹正在攻击该目标
        for missile_id, missile in env._tempsims.items():
            if hasattr(missile, 'target_aircraft') and missile.target_aircraft:
                if missile.target_aircraft.uid == target_id:
                    # 计算导弹到目标的距离
                    missile_pos = missile.get_position()
                    target_pos = target.get_position()
                    missile_distance = np.linalg.norm(missile_pos - target_pos)

                    # 如果导弹距离目标小于15km且速度正常，认为目标即将被击落
                    if missile_distance < 15000 and np.linalg.norm(missile.get_velocity()) > 500:
                        # logging.info(f"目标{target_id}即将被导弹{missile_id}击落，距离{missile_distance / 1000:.1f}km")
                        return True

        return False
    def _find_target(self, env, agent_id: str):
        """找到目标"""
        for enemy_id, enemy in env.agents.items():
            if self._is_enemy(agent_id, enemy_id) and enemy.is_alive:
                return enemy
        return None

    def _is_enemy(self, agent_id1: str, agent_id2: str) -> bool:
        """判断是否为敌方"""
        return (agent_id1.startswith('A') and agent_id2.startswith('B')) or \
               (agent_id1.startswith('B') and agent_id2.startswith('A'))

    def _launch_missile(self, env, agent_id: str, target, current_time: float):
        """发射导弹 - 根据发射平台选择导弹类型"""
        try:
            aircraft = env.agents[agent_id]

            # 创建导弹ID - 使用正确的格式 A0100 → A1001, A1002
            missile_count = 2 - aircraft.num_missiles + 1  # 第1枚或第2枚导弹
            # A0100 → A100, B0100 → B100
            base_id = agent_id[0] + agent_id[2:]  # A0100 → A100
            missile_uid = f"{base_id}{missile_count}"  # A100 → A1001

            # 根据发射平台选择导弹类型
            if agent_id.startswith('A'):  # 我方飞机 - 使用AIM-120C7
                from envs.JSBSim.core.simulatior import MissileSimulator
                missile = MissileSimulator.create(
                    parent=aircraft,
                    target=target,
                    uid=missile_uid
                )
                missile_type = "AIM-120C-7"
            else:  # 敌方飞机 - 使用R-27ER
                from r27er_missile import R27ERMissileSimulator
                missile = R27ERMissileSimulator.create(
                    parent=aircraft,
                    target=target,
                    uid=missile_uid
                )
                missile_type = "R-27ER"

            # 添加到环境的临时模拟器
            env.add_temp_simulator(missile)

            # 初始化导弹记录系统
            if not hasattr(env, '_missile_records'):
                env._missile_records = {}
            
            # 记录导弹信息
            env._missile_records[missile_uid] = {
                'launcher': agent_id,
                'target': target.uid,
                'type': missile_type,
                'status': 'LAUNCHED',
                'launch_time': current_time,
                'launch_position': aircraft.get_position().copy(),
                'launch_velocity': aircraft.get_velocity().copy()
            }

            # 更新状态
            aircraft.num_missiles -= 1

            # 更新发射时间记录（替代missile_launched机制）
            self.last_missile_launch_time[agent_id] = current_time

            # 保留第二轮发射标记（仅用于敌方特殊逻辑）
            if self.current_phase == TacticalPhase.DOR_DR and agent_id.startswith('B'):
                if not hasattr(self, 'second_launch_done'):
                    self.second_launch_done = {}
                self.second_launch_done[agent_id] = True

            # 详细的导弹发射时间线日志
            distance_km = self._calculate_distance(aircraft, target)/1000
            logging.info(f"🚀 MISSILE LAUNCH: {agent_id} -> {target.uid} at t={current_time:.1f}s, "
                        f"distance={distance_km:.1f}km, missile_id={missile_uid}, remaining_missiles={aircraft.num_missiles}")

            # 记录发射时间用于协调分析
            if not hasattr(self, 'missile_launch_timeline'):
                self.missile_launch_timeline = {}
            self.missile_launch_timeline[agent_id] = {
                'launch_time': current_time,
                'phase': self.current_phase.value,
                'distance': distance_km,
                'target': target.uid,
                'missile_type': missile_type
            }

            # 分析发射协调
            logging.info(f"📊 导弹发射时间线分析:")
            if agent_id == "A0100":
                logging.info(f"   长机A0100发射: {current_time:.1f}s, 阶段={self.current_phase.value}, 距离={distance_km:.1f}km")
            elif agent_id == "A0200":
                logging.info(f"   僚机A0200发射: {current_time:.1f}s, 阶段={self.current_phase.value}, 距离={distance_km:.1f}km")
                # 检查与长机的发射时间差
                if "A0100" in self.missile_launch_timeline:
                    leader_launch = self.missile_launch_timeline["A0100"]
                    time_diff = current_time - leader_launch['launch_time']
                    phase_diff = f"{leader_launch['phase']} -> {self.current_phase.value}"
                    logging.info(f"   僚机发射延迟: {time_diff:.1f}s (相对于长机)")
                    logging.info(f"   阶段变化: {phase_diff}")

                    if time_diff > 30:
                        logging.warning(f"⚠️  僚机发射延迟过大: {time_diff:.1f}s > 30s，可能影响掩护效果")
                    elif time_diff < 0:
                        logging.info(f"✅  僚机提前发射: {abs(time_diff):.1f}s，有效的战术协调")
                    else:
                        logging.info(f"✅  僚机发射时机合理: {time_diff:.1f}s延迟")

            # 敌方发射分析
            elif agent_id.startswith("B"):
                logging.info(f"   敌方{agent_id}发射: {current_time:.1f}s, 阶段={self.current_phase.value}, 距离={distance_km:.1f}km")
                # 分析敌方发射时机相对于我方的情况
                our_launches = [k for k in self.missile_launch_timeline.keys() if k.startswith("A")]
                if our_launches:
                    earliest_our_launch = min([self.missile_launch_timeline[k]['launch_time'] for k in our_launches])
                    enemy_delay = current_time - earliest_our_launch
                    logging.info(f"   敌方发射延迟: {enemy_delay:.1f}s (相对于我方最早发射)")

            # 打印当前所有发射记录
            logging.info(f"📊 当前发射状态汇总:")
            for launcher, data in self.missile_launch_timeline.items():
                logging.info(f"   {launcher}: {data['launch_time']:.1f}s, {data['phase']}, {data['distance']:.1f}km -> {data['target']}")

        except Exception as e:
            logging.error(f"Failed to launch missile from {agent_id}: {e}")
            import traceback
            traceback.print_exc()

    def _print_detailed_status(self, env, current_time: float):
        """打印详细的战术状态信息"""
        try:
            # 获取主要对抗双方
            leader_red = env._jsbsims.get("A0100")
            leader_blue = env._jsbsims.get("B0100")

            if not leader_red or not leader_blue:
                return

            distance = self._calculate_distance(leader_red, leader_blue)

            logging.info(f"\n{'='*60}")
            logging.info(f"📊 TACTICAL STATUS at t={current_time:.1f}s")
            logging.info(f"Phase: {self.current_phase.value} | Distance: {distance/1000:.1f}km")
            logging.info(f"{'='*60}")

            # 飞机状态和机动信息
            for agent_id, aircraft in env._jsbsims.items():
                if aircraft.is_alive:
                    pos = aircraft.get_position()
                    heading = np.rad2deg(aircraft.get_property_value(c.attitude_psi_rad))
                    altitude = aircraft.get_property_value(c.position_h_sl_m)
                    velocity = np.linalg.norm(aircraft.get_velocity())

                    # 获取当前机动指令
                    if agent_id in ["A0100", "A0200"]:
                        if agent_id == "A0100":
                            cmd_indices = self._get_leader_command_indices(env, agent_id)
                        else:
                            cmd_indices = self._get_wingman_command_indices(env, agent_id)
                        maneuver_desc = self._get_maneuver_description(cmd_indices)
                    else:
                        cmd_indices = self._get_enemy_command_indices(env, agent_id)
                        maneuver_desc = self._get_enemy_maneuver_description(cmd_indices)

                    logging.info(f"✈️  {agent_id}: pos=({pos[0]/1000:.1f}, {pos[1]/1000:.1f}, {pos[2]/1000:.1f})km, "
                               f"hdg={heading:.1f}°, alt={altitude:.0f}m, vel={velocity:.1f}m/s, "
                               f"missiles={aircraft.num_missiles}, maneuver={maneuver_desc}")

            # 导弹状态
            missile_count = len(env._tempsims)
            if missile_count > 0:
                logging.info(f"🚀 Active missiles: {missile_count}")
                for missile_id, missile in env._tempsims.items():
                    if hasattr(missile, 'get_position'):
                        m_pos = missile.get_position()
                        m_vel = np.linalg.norm(missile.get_velocity())
                        logging.info(f"   {missile_id}: pos=({m_pos[0]/1000:.1f}, {m_pos[1]/1000:.1f}, {m_pos[2]/1000:.1f})km, "
                                   f"vel={m_vel:.1f}m/s")

            # 发射状态
            launched_status = [f"{k}:{v}" for k, v in self.missile_launched.items()]
            logging.info(f"🎯 Launch status: {', '.join(launched_status)}")

            logging.info(f"{'='*60}\n")

        except Exception as e:
            logging.warning(f"Failed to print detailed status: {e}")

    def _get_maneuver_description(self, cmd_indices):
        """获取机动动作的描述"""
        try:
            alt_cmd, heading_cmd, speed_cmd = cmd_indices

            # 航向动作描述
            if heading_cmd == 6:
                heading_desc = "左转"
            elif heading_cmd == 8:
                heading_desc = "直飞"
            elif heading_cmd == 10:
                heading_desc = "右转"
            else:
                heading_desc = f"未知({heading_cmd})"

            # 高度动作描述
            if alt_cmd == 7:
                alt_desc = "保持高度"
            else:
                alt_desc = f"高度({alt_cmd})"

            # 速度动作描述
            if speed_cmd == 3:
                speed_desc = "保持速度"
            else:
                speed_desc = f"速度({speed_cmd})"

            return f"{heading_desc}+{alt_desc}+{speed_desc}"
        except:
            return "未知机动"

    def _get_enemy_maneuver_description(self, cmd_indices):
        """获取敌方机动动作的描述"""
        try:
            alt_cmd, heading_cmd, speed_cmd = cmd_indices

            # 敌方航向动作描述（更详细）
            heading_map = {
                0: "急左转", 2: "大幅左转", 4: "中等左转", 6: "轻微左转",
                8: "直飞", 10: "轻微右转", 12: "中等右转", 14: "大幅右转", 16: "急右转"
            }
            heading_desc = heading_map.get(heading_cmd, f"转向({heading_cmd})")

            # 敌方高度动作描述
            alt_map = {
                0: "急俯冲", 2: "大幅俯冲", 3: "俯冲", 4: "轻微俯冲", 5: "轻微俯冲",
                6: "轻微爬升", 7: "保持高度", 8: "轻微爬升", 9: "爬升", 10: "大幅爬升"
            }
            alt_desc = alt_map.get(alt_cmd, f"高度({alt_cmd})")

            # 敌方速度动作描述
            speed_map = {
                0: "大幅减速", 1: "减速", 2: "轻微减速", 3: "保持速度",
                4: "加速", 5: "大幅加速", 6: "最大加速"
            }
            speed_desc = speed_map.get(speed_cmd, f"速度({speed_cmd})")

            return f"敌方AI:{heading_desc}+{alt_desc}+{speed_desc}"
        except:
            return "敌方AI:未知机动"

    def _print_formation_spacing(self, env, current_time: float):
        """打印编队间距信息 - 专门监控30-90秒时间段（crank机动完成后）"""
        try:
            # 获取我方长机和僚机
            leader = env._jsbsims.get("A0100")
            wingman = env._jsbsims.get("A0200")

            if not leader or not wingman or not leader.is_alive or not wingman.is_alive:
                return

            # 计算编队间距
            formation_distance = self._calculate_distance(leader, wingman)

            # 转换为海里 (1海里 = 1852米)
            distance_nm = formation_distance / 1852.0
            distance_km = formation_distance / 1000.0

            # 判断是否在标准范围内 (5-10海里)
            in_range = 5.0 <= distance_nm <= 10.0
            status_icon = "✅" if in_range else "⚠️"

            # 获取飞机航向信息
            leader_heading = np.rad2deg(leader.get_property_value(c.attitude_psi_rad))
            wingman_heading = np.rad2deg(wingman.get_property_value(c.attitude_psi_rad))

            # 打印编队间距信息（包含更多细节）
            logging.info(f"🔍 CRANK机动后编队间距监控:")
            logging.info(f"{status_icon} [{self.current_phase.value}] 时间: {current_time:.1f}s, "
                        f"编队间距: {distance_km:.1f}km ({distance_nm:.1f}海里)")
            logging.info(f"   长机A0100航向: {leader_heading:.1f}°, 僚机A0200航向: {wingman_heading:.1f}°")

            # 如果超出标准范围，给出提示
            if not in_range:
                if distance_nm < 5.0:
                    logging.warning(f"⚠️  编队间距过近！当前{distance_nm:.1f}海里 < 标准5海里")
                else:
                    logging.warning(f"⚠️  编队间距过远！当前{distance_nm:.1f}海里 > 标准10海里")
            else:
                logging.info(f"✅  编队间距符合标准！{distance_nm:.1f}海里在5-10海里范围内")

        except Exception as e:
            logging.error(f"编队间距监控失败: {e}")



    def get_radar_states(self):
        """获取雷达状态 - 使用统一雷达管理器"""
        try:
            radar_summary = self.radar_manager.get_radar_performance_summary()
            # 转换为兼容格式
            states = {}
            for agent_id, radar_info in radar_summary["friendly_radars"].items():
                states[agent_id] = radar_info["status"]
            for agent_id, radar_info in radar_summary["enemy_radars"].items():
                states[agent_id] = radar_info["status"]
            return states
        except Exception as e:
            logging.error(f"❌ 获取雷达状态错误: {e}")
            return {"A0100": "SEARCH", "A0200": "SEARCH", "B0100": "SEARCH", "B0200": "SEARCH"}
