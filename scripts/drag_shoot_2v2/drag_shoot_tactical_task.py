#!/usr/bin/env python3
"""
拖曳射击战术任务 - 完全基于pure_maneuver_task架构模式
严格照抄pure_maneuver_task的实现，只修改战术逻辑部分
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


class DragShootTermination(BaseTerminationCondition):
    """拖曳射击专用终止条件 - 只有双方全灭才终止"""

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
    """拖曳射击战术阶段"""
    NLT_MELD = "NLT_MELD"    # 90-81km
    MELD_MTR = "MELD_MTR"    # 81-45km
    MTR_TR = "MTR_TR"        # 45-41km
    TR_DOR = "TR_DOR"        # 41-19.6km
    DOR_DR = "DOR_DR"        # 19.6-14.5km


class DragShootTacticalTask(MultipleCombatTask):
    """拖曳射击战术任务 - 基于pure_maneuver_task架构模式"""

    def __init__(self, config):
        """初始化拖曳射击任务 - 学习pure_maneuver_task的模式"""
        super().__init__(config)

        # 使用自定义终止条件 - 完全替换父类的终止条件
        self.termination_conditions = [
            DragShootTermination(self.config),
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
            'TR_DOR_delay': 8000,    # 僚机TR_DOR阶段滞后8km
            'DOR_DR_delay': 10000,   # DOR_DR阶段滞后10km，确保长机先完成short_skate
        }

        # 当前战术阶段
        self.current_phase = TacticalPhase.NLT_MELD
        self.missile_launched = {"A0100": False, "A0200": False, "B0100": False, "B0200": False}

        # 雷达状态管理
        self.radar_states = {"A0100": "SEARCH", "A0200": "SEARCH", "B0100": "SEARCH", "B0200": "SEARCH"}

        # 初始状态记录 - 学习pure_maneuver_task
        self.initial_heading = {}
        self.initial_altitude = {}

        # short_skate机动状态跟踪
        self.short_skate_states = {}
        self.short_skate_start_time = {}

        # baseline模型 - 学习pure_maneuver_task的模式
        self.my_lowlevel_policy = BaselineActor()
        self._inner_rnn_states = {}

        # 指令数组 - 完全照抄pure_maneuver_task的定义
        self.norm_delta_altitude = np.array([
            -1500, -1000, -750, -500, -300, -150, -50, 0, 50, 150, 300, 500, 750, 1000, 1500
        ]) / 1000.0
        self.norm_delta_heading = np.array([
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
        self.norm_delta_velocity = np.array([-150, -100, -50, 0, 50, 100, 150]) / 100.0

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
        """动作归一化 - 学习pure_maneuver_task的模式"""
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

        # 处理拖曳射击战术动作
        result = self._process_drag_shoot_tactics(env, agent_id, current_time)

        # 处理导弹发射
        self._handle_missile_launch(env, agent_id, current_time)

        # 更新雷达状态
        self._update_radar_state(env, agent_id, current_time)

        return result

    def _process_drag_shoot_tactics(self, env, agent_id, current_time):
        """处理拖曳射击战术 - 学习pure_maneuver_task的处理模式"""
        try:
            # 更新战术阶段
            self._update_tactical_phase(env)

            # 根据智能体角色和当前阶段生成指令索引
            altitude_cmd_id, heading_cmd_id, velocity_cmd_id = self._get_tactical_command_indices(env, agent_id)

            # 使用baseline模型 - 完全学习pure_maneuver_task的_use_lowlevel_policy
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

        # 详细状态信息 - 每5秒打印一次
        if env.current_step % 25 == 0:
            self._print_detailed_status(env, current_time)

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

            # 生成战术动作
            tactical_action = self._get_tactical_action(env, agent_id)

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
    
    def _get_tactical_action(self, env, agent_id: str):
        """生成战术动作 - 基于拖曳射击逻辑"""
        # 根据智能体角色和当前阶段生成动作
        if agent_id == "A0100":  # 己方长机
            return self._get_leader_action(env, agent_id)
        elif agent_id == "A0200":  # 己方僚机
            return self._get_wingman_action(env, agent_id)
        elif agent_id.startswith("B"):  # 敌方
            return self._get_enemy_action(env, agent_id)
        else:
            # 默认平稳飞行
            return np.array([3, 4, 3])  # 中性指令
    
    def _get_leader_action(self, env, agent_id: str):
        """长机战术动作"""
        if self.current_phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR, TacticalPhase.MTR_TR]:
            # 平稳飞行 (航向180°)
            return np.array([3, 4, 3])  # 保持高度、航向、速度
        elif self.current_phase == TacticalPhase.TR_DOR:
            # 左侧short_skate (turn_angle=-45.0)
            return np.array([3, 2, 3])  # 保持高度、左转、保持速度
        elif self.current_phase == TacticalPhase.DOR_DR:
            # 返航 (航向0°)
            return np.array([3, 0, 3])  # 保持高度、大幅左转、保持速度
        else:
            return np.array([3, 4, 3])
    
    def _get_wingman_action(self, env, agent_id: str):
        """僚机战术动作"""
        if self.current_phase == TacticalPhase.NLT_MELD:
            # 右侧crank (turn_angle=30.0)
            return np.array([3, 6, 3])  # 保持高度、右转、保持速度
        elif self.current_phase == TacticalPhase.MELD_MTR:
            # 左侧crank (turn_angle=-30.0, 调整至180°)
            return np.array([3, 2, 3])  # 保持高度、左转、保持速度
        elif self.current_phase in [TacticalPhase.MTR_TR, TacticalPhase.TR_DOR]:
            # 平稳飞行 (航向180°)
            return np.array([3, 4, 3])  # 保持高度、航向、速度
        elif self.current_phase == TacticalPhase.DOR_DR:
            # 左侧short_skate后返航
            return np.array([3, 2, 3])  # 保持高度、左转、保持速度
        else:
            return np.array([3, 4, 3])
    
    def _get_enemy_action(self, env, agent_id: str):
        """敌方战术动作 - CAP任务short_skate返航逻辑"""
        # 检查是否应该返航
        should_return = False

        # 条件1：队友被击落
        if agent_id == "B0200":
            if "B0100" not in env.agents or not env.agents["B0100"].is_alive:
                should_return = True
        elif agent_id == "B0100":
            if "B0200" not in env.agents or not env.agents["B0200"].is_alive:
                should_return = True

        # 条件2：距离过近（进入危险区域）
        current_pos = np.array([
            env.agents[agent_id].get_property_value(c.position_long_gc_deg),
            env.agents[agent_id].get_property_value(c.position_lat_gc_deg)
        ])

        # 计算与我方的最近距离
        min_distance = float('inf')
        for friendly_id in ["A0100", "A0200"]:
            if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                friendly_pos = np.array([
                    env.agents[friendly_id].get_property_value(c.position_long_gc_deg),
                    env.agents[friendly_id].get_property_value(c.position_lat_gc_deg)
                ])
                distance = np.linalg.norm((current_pos - friendly_pos) * 111000)  # 转换为米
                min_distance = min(min_distance, distance)

        # 条件3：进入DOR_DR阶段，与我方同步返航
        if self.current_phase == TacticalPhase.DOR_DR:
            should_return = True

        # 如果距离小于30km，返航
        if min_distance < 30000:
            should_return = True

        if should_return:
            # 执行short_skate返航机动
            current_time = env.current_step * env.time_interval
            action = self._execute_short_skate(env, agent_id, current_time)
            # 转换为敌方动作格式 - 确保正确的动作空间转换
            alt_action = max(0, min(6, action[0] - 4))  # 高度动作：7->3, 范围[0,6]
            hdg_action = max(0, min(8, action[1] - 4))  # 航向动作：8->4, 范围[0,8]
            vel_action = max(0, min(4, action[2]))      # 速度动作：保持原值, 范围[0,4]
            return np.array([alt_action, hdg_action, vel_action])
        else:
            # 正常CAP巡逻：平稳飞行 (航向0°)
            return np.array([3, 4, 3])  # 保持高度、航向、速度
    
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
            # 默认平稳飞行
            return 7, 8, 3  # 中性指令

    def _get_leader_command_indices(self, env, agent_id: str):
        """长机战术指令索引 - 基于拖曳射击战术"""
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))

        if self.current_phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR, TacticalPhase.MTR_TR]:
            # 平稳飞行 - 朝北接敌（0°）
            target_heading = 0.0
            if abs(current_heading - target_heading) > 5.0:  # 如果偏离超过5°，修正航向
                heading_diff = self._normalize_angle_diff(target_heading - current_heading)
                if heading_diff > 0:
                    return 7, 10, 3  # 保持高度、右转、保持速度
                else:
                    return 7, 6, 3   # 保持高度、左转、保持速度
            else:
                return 7, 8, 3  # 保持高度、直飞、保持速度

        elif self.current_phase == TacticalPhase.TR_DOR:
            # 长机在TR_DOR阶段：发射导弹后执行左侧short_skate机动
            if self.missile_launched.get(agent_id, False):
                # 已发射导弹，执行完整的short_skate机动
                current_time = env.current_step * env.time_interval
                return self._execute_short_skate(env, agent_id, current_time)
            else:
                # 未发射导弹，继续平稳飞行等待发射时机
                target_heading = 0.0
                heading_diff = self._normalize_angle_diff(target_heading - current_heading)
                if abs(heading_diff) > 5.0:
                    if heading_diff > 0:
                        return 7, 10, 3  # 右转
                    else:
                        return 7, 6, 3   # 左转
                else:
                    return 7, 8, 3  # 保持航向

        elif self.current_phase == TacticalPhase.DOR_DR:
            # DOR_DR阶段：长机执行short_skate机动
            current_time = env.current_step * env.time_interval
            return self._execute_short_skate(env, agent_id, current_time)
        else:
            return 7, 8, 3

    def _get_wingman_command_indices(self, env, agent_id: str):
        """僚机战术指令索引 - 基于拖曳射击战术"""
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))

        if self.current_phase == TacticalPhase.NLT_MELD:
            # 右侧crank: 航向从0°调整至30°（右偏30°）
            target_heading = 30.0
            heading_diff = self._normalize_angle_diff(target_heading - current_heading)
            if abs(heading_diff) > 5.0:
                if heading_diff > 0:
                    return 7, 10, 3  # 右转
                else:
                    return 7, 6, 3   # 左转
            else:
                return 7, 8, 3  # 保持航向

        elif self.current_phase == TacticalPhase.MELD_MTR:
            # 左侧crank: 航向从30°调整回0°（左转30°）
            target_heading = 0.0
            heading_diff = self._normalize_angle_diff(target_heading - current_heading)
            if abs(heading_diff) > 5.0:
                if heading_diff > 0:
                    return 7, 10, 3  # 右转
                else:
                    return 7, 6, 3   # 左转
            else:
                return 7, 8, 3  # 保持航向

        elif self.current_phase == TacticalPhase.MTR_TR:
            # 平稳飞行 - 保持航向0°
            target_heading = 0.0
            heading_diff = self._normalize_angle_diff(target_heading - current_heading)
            if abs(heading_diff) > 5.0:
                if heading_diff > 0:
                    return 7, 10, 3  # 右转
                else:
                    return 7, 6, 3   # 左转
            else:
                return 7, 8, 3  # 保持航向

        elif self.current_phase == TacticalPhase.TR_DOR:
            # 僚机在TR_DOR阶段：发射导弹后做小角度左侧crank
            if not self.missile_launched.get(agent_id, False):
                # 未发射导弹，执行左侧小crank指向敌机（小角度左转约10°）
                target_heading = 350.0  # 从0°左转10°到350°，小角度crank
                heading_diff = self._normalize_angle_diff(target_heading - current_heading)
                if abs(heading_diff) > 2.0:  # 更小的容差，精确控制
                    if heading_diff > 0:
                        return 7, 10, 3  # 右转
                    else:
                        return 7, 6, 3   # 左转
                else:
                    return 7, 8, 3  # 保持航向，等待发射时机
            else:
                # 已发射导弹，继续做小角度左侧crank（不是short_skate）
                target_heading = 340.0  # 继续左转到340°，保持小角度crank
                heading_diff = self._normalize_angle_diff(target_heading - current_heading)
                if abs(heading_diff) > 2.0:
                    if heading_diff > 0:
                        return 7, 10, 3  # 右转
                    else:
                        return 7, 6, 3   # 左转
                else:
                    return 7, 8, 3  # 保持航向

        elif self.current_phase == TacticalPhase.DOR_DR:
            # DOR_DR阶段：僚机执行完整的左侧short_skate机动
            current_time = env.current_step * env.time_interval
            return self._execute_short_skate(env, agent_id, current_time)
        else:
            return 7, 8, 3

    def _normalize_angle_diff(self, angle_diff):
        """标准化角度差值到[-180, 180]范围"""
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        return angle_diff

    def _init_short_skate(self, agent_id, current_time):
        """初始化short_skate机动状态"""
        self.short_skate_states[agent_id] = {
            "phase": "crank",  # crank -> turn_cold -> escape
            "phase_start_time": current_time,
            "total_start_time": current_time,
            "crank_angle": -40.0 if agent_id.startswith('A') else 40.0,  # 我方左侧，敌方右侧
            "turn_cold_angle": -100.0 if agent_id.startswith('A') else 100.0,  # 我方左侧，敌方右侧
            "initial_heading": None
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

        # 阶段1：Crank机动 - 左侧40度 (12秒)
        if state["phase"] == "crank":
            if phase_time < 12.0:
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

        # 阶段2：Turn Cold - 快速掉头100度 (25秒)
        elif state["phase"] == "turn_cold":
            if phase_time < 25.0:
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

        # 阶段3：加速逃离 (20秒)
        elif state["phase"] == "escape":
            if phase_time < 20.0:
                return 7, 8, 5  # 保持航向，加速
            else:
                # 完成short_skate，返航到初始航向的反方向
                if agent_id.startswith('A'):  # 我方：初始180°（南向），返回180°（南向）
                    target_heading = 180.0  # 我方返回南向
                elif agent_id.startswith('B'):  # 敌方：初始0°（北向），返回0°（北向）
                    target_heading = 0.0  # 敌方返回北向
                else:
                    target_heading = 180.0

                heading_diff = self._normalize_angle_diff(target_heading - current_heading)
                if abs(heading_diff) > 5.0:
                    return 7, 6 if heading_diff < 0 else 10, 3
                else:
                    return 7, 8, 3

        return 7, 8, 3  # 默认保持航向

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
        """敌方战术指令索引 - 朝南接敌，特定条件下执行short_skate"""
        current_time = env.current_step * env.time_interval

        # 检查是否应该执行short_skate
        should_return = False

        # 条件1：队友被击落
        if agent_id == "B0200":
            if "B0100" not in env.agents or not env.agents["B0100"].is_alive:
                should_return = True
        elif agent_id == "B0100":
            if "B0200" not in env.agents or not env.agents["B0200"].is_alive:
                should_return = True

        # 条件2：DOR_DR阶段
        if self.current_phase == TacticalPhase.DOR_DR:
            should_return = True

        if should_return:
            # 执行short_skate
            action = self._execute_short_skate(env, agent_id, current_time)
            return int(action[0]), int(action[1]), int(action[2])
        else:
            # 正常朝南接敌
            current_heading = np.rad2deg(env._jsbsims[agent_id].get_property_value(c.attitude_psi_rad))
            target_heading = 180.0
            heading_diff = self._normalize_angle_diff(target_heading - current_heading)
            if abs(heading_diff) > 5.0:
                if heading_diff > 0:
                    return 7, 10, 3  # 右转
                else:
                    return 7, 6, 3  # 左转
            else:
                return 7, 8, 3  # 保持航向

    def reset(self, env):
        """重置任务状态 - 学习pure_maneuver_task的reset模式"""
        super().reset(env)
        self.current_phase = TacticalPhase.NLT_MELD
        self.missile_launched = {"A0100": False, "A0200": False, "B0100": False, "B0200": False}
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
        """处理导弹发射 - 严格按照拖曳射击战术需求"""
        if not env.agents[agent_id].is_alive:
            return

        # 检查导弹数量
        if env.agents[agent_id].num_missiles <= 0:
            return

        # 找到目标
        target = self._find_target(env, agent_id)
        if not target:
            return

        distance = self._calculate_distance(env.agents[agent_id], target)

        # 详细调试信息
        if env.current_step % 50 == 0:  # 每10秒打印一次
            logging.info(f"MISSILE CHECK: {agent_id} -> target distance={distance/1000:.1f}km, "
                        f"phase={self.current_phase.value}, missiles={env.agents[agent_id].num_missiles}, "
                        f"launched={self.missile_launched.get(agent_id, False)}")

        # 根据拖曳射击战术确定发射条件
        should_launch = False

        if agent_id == "A0100":  # 己方长机45km发射
            should_launch = (self.current_phase == TacticalPhase.MTR_TR and
                           44000 <= distance <= 47000 and not self.missile_launched.get(agent_id, False))
        elif agent_id == "A0200":  # 己方僚机滞后发射（体现时间线滞后）
            # 僚机发射距离更近，体现滞后时间线
            wingman_launch_min = 40000 - self.wingman_delay['TR_DOR_delay']  # 32km
            wingman_launch_max = 43000 - self.wingman_delay['TR_DOR_delay']  # 35km
            should_launch = (self.current_phase == TacticalPhase.TR_DOR and
                           wingman_launch_min <= distance <= wingman_launch_max and
                           not self.missile_launched.get(agent_id, False))
        elif agent_id == "B0100":  # 敌方长机 - 暂时禁用导弹发射
            should_launch = False  # 禁用敌方导弹，观察我方完整机动流程
        elif agent_id == "B0200":  # 敌方僚机 - 暂时禁用导弹发射
            should_launch = False  # 禁用敌方导弹，观察我方完整机动流程

        # 敌方第二轮发射 (19.6km) - 需要重置发射状态
        if agent_id.startswith('B') and self.current_phase == TacticalPhase.DOR_DR:
            if 19000 <= distance <= 21000:
                # 重置发射状态，允许第二轮发射
                if not hasattr(self, 'second_launch_done'):
                    self.second_launch_done = {}
                if agent_id not in self.second_launch_done:
                    self.second_launch_done[agent_id] = False

                should_launch = should_launch or not self.second_launch_done.get(agent_id, False)

        if should_launch:
            self._launch_missile(env, agent_id, target, current_time)

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
        """发射导弹 - 创建真实的导弹模拟器"""
        try:
            from envs.JSBSim.core.simulatior import MissileSimulator

            aircraft = env.agents[agent_id]

            # 创建导弹ID - 使用正确的格式 A0100 → A1001, A1002
            missile_count = 2 - aircraft.num_missiles + 1  # 第1枚或第2枚导弹
            # A0100 → A100, B0100 → B100
            base_id = agent_id[0] + agent_id[2:]  # A0100 → A100
            missile_uid = f"{base_id}{missile_count}"  # A100 → A1001

            # 创建真实的导弹模拟器
            missile = MissileSimulator.create(
                parent=aircraft,
                target=target,
                uid=missile_uid
            )

            # 添加到环境的临时模拟器
            env.add_temp_simulator(missile)

            # 更新状态
            aircraft.num_missiles -= 1

            # 标记发射状态
            if self.current_phase == TacticalPhase.DOR_DR and agent_id.startswith('B'):
                # 第二轮发射
                if not hasattr(self, 'second_launch_done'):
                    self.second_launch_done = {}
                self.second_launch_done[agent_id] = True
            else:
                # 第一轮发射
                self.missile_launched[agent_id] = True

            logging.info(f"🚀 MISSILE LAUNCH: {agent_id} -> {target.uid} at t={current_time:.1f}s, "
                        f"distance={self._calculate_distance(aircraft, target)/1000:.1f}km, "
                        f"missile_id={missile_uid}, remaining_missiles={aircraft.num_missiles}")

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
                        maneuver_desc = "Enemy maneuver"

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



    def _update_radar_state(self, env, agent_id: str, current_time: float):
        """更新雷达状态 - 基于拖曳射击战术需求"""
        if not env.agents[agent_id].is_alive:
            return

        # 找到最近的敌机
        target = self._find_target(env, agent_id)
        if not target:
            self.radar_states[agent_id] = "SEARCH"
            return

        distance = self._calculate_distance(env.agents[agent_id], target)

        # 根据距离和战术阶段确定雷达状态
        if distance > 90000:  # 90km
            self.radar_states[agent_id] = "SEARCH"
        elif distance > 81000:  # 81km
            self.radar_states[agent_id] = "SEARCH"
        elif distance > 45000:  # 45km
            self.radar_states[agent_id] = "TRACK"
        else:  # < 45km
            self.radar_states[agent_id] = "LOCK"

        # 返航阶段重新搜索
        if self.current_phase == TacticalPhase.DOR_DR and agent_id.startswith('A'):
            self.radar_states[agent_id] = "SEARCH"
