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

        # 拖曳射击特定的战术距离
        self.tactical_distances = {
            'NLT_MELD_min': 81000,   # 81km
            'MELD_MTR_min': 45000,   # 45km
            'MTR_TR_min': 41000,     # 41km
            'TR_DOR_min': 19600,     # 19.6km
            'DOR_DR_min': 14500,     # 14.5km
        }

        # 当前战术阶段
        self.current_phase = TacticalPhase.NLT_MELD
        self.missile_launched = {"A0100": False, "A0200": False, "B0100": False, "B0200": False}

        # 初始状态记录 - 学习pure_maneuver_task
        self.initial_heading = {}
        self.initial_altitude = {}

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
        return self._process_drag_shoot_tactics(env, agent_id, current_time)

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
        """敌方战术动作 - 平稳飞行"""
        # 敌方始终平稳飞行 (航向0°)
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
            # 平稳飞行 - level_flight模式
            return 7, 8, 3  # 保持高度、航向、速度
        elif self.current_phase == TacticalPhase.TR_DOR:
            # 左侧short_skate (turn_angle=-45.0)
            target_heading = 180.0 - 45.0  # 135°
            heading_diff = target_heading - current_heading
            while heading_diff > 180: heading_diff -= 360
            while heading_diff < -180: heading_diff += 360
            heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))
            return 7, heading_cmd_id, 3  # 保持高度、左转45°、保持速度
        elif self.current_phase == TacticalPhase.DOR_DR:
            # 返航 (航向0°)
            target_heading = 0.0
            heading_diff = target_heading - current_heading
            while heading_diff > 180: heading_diff -= 360
            while heading_diff < -180: heading_diff += 360
            heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))
            return 7, heading_cmd_id, 3  # 保持高度、转向0°、保持速度
        else:
            return 7, 8, 3

    def _get_wingman_command_indices(self, env, agent_id: str):
        """僚机战术指令索引 - 基于拖曳射击战术"""
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))

        if self.current_phase == TacticalPhase.NLT_MELD:
            # 右侧crank (turn_angle=30.0)
            target_heading = 180.0 + 30.0  # 210°
            heading_diff = target_heading - current_heading
            while heading_diff > 180: heading_diff -= 360
            while heading_diff < -180: heading_diff += 360
            heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))
            return 7, heading_cmd_id, 3  # 保持高度、右转30°、保持速度
        elif self.current_phase == TacticalPhase.MELD_MTR:
            # 左侧crank (turn_angle=-30.0, 调整至180°)
            target_heading = 180.0
            heading_diff = target_heading - current_heading
            while heading_diff > 180: heading_diff -= 360
            while heading_diff < -180: heading_diff += 360
            heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))
            return 7, heading_cmd_id, 3  # 保持高度、调整至180°、保持速度
        elif self.current_phase in [TacticalPhase.MTR_TR, TacticalPhase.TR_DOR]:
            # 平稳飞行 (航向180°)
            return 7, 8, 3  # 保持高度、航向、速度
        elif self.current_phase == TacticalPhase.DOR_DR:
            # 左侧short_skate后返航
            target_heading = 0.0
            heading_diff = target_heading - current_heading
            while heading_diff > 180: heading_diff -= 360
            while heading_diff < -180: heading_diff += 360
            heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))
            return 7, heading_cmd_id, 3  # 保持高度、转向0°、保持速度
        else:
            return 7, 8, 3

    def _get_enemy_command_indices(self, env, agent_id: str):
        """敌方战术指令索引 - 平稳飞行"""
        # 敌方始终平稳飞行 - level_flight模式
        return 7, 8, 3  # 保持高度、航向、速度

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
