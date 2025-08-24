#!/usr/bin/env python3
"""
上下夹击战术任务 (High-Low Split Attack Tactical Task)

严格基于拖曳射击战术的成功架构重构，实现上下夹击战术：
- 通过垂直分离创造立体攻击覆盖区域
- 僚机建立高度优势（1200m高度差）
- 时间线差异化执行（参考wingman_delay机制）
- 分时发射策略
- 分离脱离机动
"""

import os
import sys
import logging
import numpy as np
from enum import Enum
from typing import Dict, Any, Optional, Tuple

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# 导入基础模块
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.tasks.multiplecombat_task import MultipleCombatTask
from envs.JSBSim.model.baseline_actor import BaselineActor

# 导入战术阶段枚举 - 与拖曳射击完全一致
class TacticalPhase(Enum):
    """战术阶段枚举 - 与拖曳射击保持一致"""
    NLT_MELD = "NLT_MELD"     # No Later Than Merge (81km+)
    MELD_MTR = "MELD_MTR"     # Merge to Missile Threat Range (50-81km)
    MTR_TR = "MTR_TR"         # Missile Threat Range to Threat Range (40-50km)
    TR_DOR = "TR_DOR"         # Threat Range to Defensive Offensive Range (19.6-40km)
    DOR_DR = "DOR_DR"         # Defensive Offensive Range to Defensive Range (14.5-19.6km)


class HighLowSplitTacticalTask(MultipleCombatTask):
    """
    上下夹击战术任务 - 严格基于拖曳射击架构
    
    动作空间架构（完全复制拖曳射击）：
    - 定义的动作空间: [41, 41, 41, 30] (继承自MultipleCombatTask，但实际不使用)
    - 实际使用的动作空间: [15, 17, 7] (高层战术指令)
    - 转换机制: 高层指令 → baseline模型 → 底层飞行控制
    
    主要函数调用链：
    normalize_action() → _process_high_low_split_tactics() → _get_tactical_command_indices()
    → _get_leader_command_indices() / _get_wingman_command_indices() / _get_enemy_command_indices()
    → _use_lowlevel_policy() → baseline模型输出底层控制指令
    
    平稳飞行指令: [7, 8, 3] = [高度0m变化, 航向0°变化, 速度0m/s变化]
    """

    def __init__(self, config):
        """初始化上下夹击任务 - 完全学习拖曳射击的模式"""
        super().__init__(config)
        
        # === 完全复制拖曳射击的核心架构 ===
        
        # 战术距离配置 - 与拖曳射击完全一致
        self.tactical_distances = {
            'NLT_MELD_min': 81000,       # 81km
            'MELD_MTR_min': 50000,       # 50km
            'MTR_TR_min': 40000,         # 40km
            'TR_DOR_min': 35000,         # 35km
            'DOR_DR_min': 14500,         # 14.5km
        }

        # 上下夹击特有：僚机高度差时间线滞后设置
        self.wingman_delay = {
            'altitude_establishment': 15.0,  # 僚机高度建立滞后15秒
            'TR_DOR_delay': 4000,           # 僚机TR_DOR阶段滞后4km
            'DOR_DR_delay': 8000,           # DOR_DR阶段滞后8km
        }

        # 当前战术阶段
        self.current_phase = TacticalPhase.NLT_MELD

        # 导弹发射冷却时间管理
        self.last_missile_launch_time = {"A0100": -999, "A0200": -999, "B0100": -999, "B0200": -999}

        # 上下夹击特有：高度管理状态 - 完全基于拖曳射击的成功策略
        self.high_low_states = {
            "A0100": {  # 长机 - 完全复制拖曳射击的高度策略
                "target_altitude": 6096.0,  # 保持初始高度，与拖曳射击一致
                "altitude_established": True,  # 已经建立
                "rtb_direction": "left"
            },
            "A0200": {  # 僚机 - 也保持初始高度，先确保系统稳定
                "target_altitude": 6096.0,  # 暂时与长机相同，确保系统稳定
                "altitude_established": True,  # 已经建立
                "rtb_direction": "right"
            }
        }

        # 初始状态记录 - 学习拖曳射击
        self.initial_heading = {}
        self.initial_altitude = {}
        self._inner_rnn_states = {}

        # === 完全复制拖曳射击的动作空间定义 ===
        
        # 高层战术指令空间 [15, 17, 7] - 与拖曳射击完全一致
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
        self.norm_delta_velocity = np.array([
            -100, -50, -20, 0, 20, 50, 100
        ]) / 340.0  # 索引3 = 0m/s变化（平稳飞行）

        # 加载baseline模型 - 完全复制拖曳射击的方法
        self._load_baseline_models()

        logging.info("上下夹击战术系统初始化完成")

    def _load_baseline_models(self):
        """加载baseline模型 - 完全复制拖曳射击的实现"""
        try:
            import torch
            from envs.JSBSim.utils.utils import get_root_dir
            from envs.JSBSim.model.baseline_actor import BaselineActor

            # 创建baseline actor
            self.my_lowlevel_policy = BaselineActor()

            model_path = get_root_dir() + '/model/baseline_model.pt'
            if torch.cuda.is_available():
                device = torch.device("cuda")
                checkpoint = torch.load(model_path, weights_only=True)
            else:
                device = torch.device("cpu")
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            self.my_lowlevel_policy.load_state_dict(checkpoint)
            self.my_lowlevel_policy.eval()
            logging.info("Successfully loaded baseline model for high-low split task")
        except Exception as e:
            logging.error(f"加载baseline模型失败: {e}")
            self.my_lowlevel_policy = None

    def reset(self, env):
        """重置任务状态 - 学习拖曳射击的重置模式"""
        super().reset(env)
        
        # 重置战术阶段
        self.current_phase = TacticalPhase.NLT_MELD
        
        # 重置导弹发射时间
        self.last_missile_launch_time = {"A0100": -999, "A0200": -999, "B0100": -999, "B0200": -999}
        
        # 重置初始状态记录
        self.initial_heading.clear()
        self.initial_altitude.clear()
        self._inner_rnn_states.clear()
        
        # 重置高度管理状态
        for agent_id in self.high_low_states:
            self.high_low_states[agent_id]["altitude_established"] = False
        
        logging.info("上下夹击战术状态已重置")

    def normalize_action(self, env, agent_id, action):
        """
        动作归一化 - 上下夹击战术系统
        完全复制拖曳射击的架构
        """
        if agent_id not in env.agents or not env.agents[agent_id].is_alive:
            return np.array([0.0, 0.0, 0.0, 0.7])

        current_time = env.current_step * env.time_interval

        # 初始化状态记录 - 学习拖曳射击
        if agent_id not in self.initial_heading:
            current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_rad)
            self.initial_heading[agent_id] = np.rad2deg(current_heading)
            current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
            self.initial_altitude[agent_id] = current_altitude
            logging.info(f"{agent_id} 初始状态: 航向{self.initial_heading[agent_id]:.1f}°, 高度{self.initial_altitude[agent_id]:.1f}m")

        # 处理上下夹击战术动作
        result = self._process_high_low_split_tactics(env, agent_id, current_time)

        # 处理导弹发射
        self._handle_missile_launch(env, agent_id, current_time)

        return result

    def _process_high_low_split_tactics(self, env, agent_id, current_time):
        """处理上下夹击战术 - 学习拖曳射击的处理模式"""
        try:
            # 更新战术阶段
            self._update_tactical_phase(env)

            # 获取战术指令索引
            altitude_cmd_id, heading_cmd_id, velocity_cmd_id = self._get_tactical_command_indices(env, agent_id, current_time)

            # 调试信息 - 每5秒打印一次指令
            if env.current_step % 25 == 0:
                logging.info(f"{agent_id} 指令索引: 高度={altitude_cmd_id}, 航向={heading_cmd_id}, 速度={velocity_cmd_id}")

            # 使用底层策略转换为实际控制指令
            action_output = self._use_lowlevel_policy(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)

            # 调试信息 - 每5秒打印一次最终动作
            if env.current_step % 25 == 0:
                logging.info(f"{agent_id} 最终动作: {action_output}")

            return action_output

        except Exception as e:
            logging.error(f"上下夹击战术处理失败 {agent_id}: {e}")
            return np.array([0.0, 0.0, 0.0, 0.7])

    def _update_tactical_phase(self, env):
        """更新战术阶段 - 基于敌我距离"""
        try:
            # 计算最小敌我距离
            min_distance = float('inf')
            for red_id in ["A0100", "A0200"]:
                if red_id in env.agents and env.agents[red_id].is_alive:
                    red_pos = env.agents[red_id].get_position()
                    for blue_id in ["B0100", "B0200"]:
                        if blue_id in env.agents and env.agents[blue_id].is_alive:
                            blue_pos = env.agents[blue_id].get_position()
                            distance = np.linalg.norm(np.array(red_pos) - np.array(blue_pos))
                            min_distance = min(min_distance, distance)

            # 根据距离确定战术阶段 - 与拖曳射击完全一致
            new_phase = self._determine_phase_by_distance(min_distance)

            if new_phase != self.current_phase:
                logging.info(f"战术阶段切换: {self.current_phase.value} → {new_phase.value} (距离: {min_distance/1000:.1f}km)")
                self.current_phase = new_phase

        except Exception as e:
            logging.error(f"战术阶段更新失败: {e}")

    def _determine_phase_by_distance(self, distance):
        """根据距离确定战术阶段 - 与拖曳射击完全一致"""
        if distance > self.tactical_distances['NLT_MELD_min']:
            return TacticalPhase.NLT_MELD
        elif distance > self.tactical_distances['MELD_MTR_min']:
            return TacticalPhase.MELD_MTR
        elif distance > self.tactical_distances['MTR_TR_min']:
            return TacticalPhase.MTR_TR
        elif distance > (self.tactical_distances['TR_DOR_min'] - self.wingman_delay['TR_DOR_delay']):
            return TacticalPhase.TR_DOR  # 35km - 4km = 31km
        elif distance > (self.tactical_distances['DOR_DR_min'] - self.wingman_delay['DOR_DR_delay']):
            return TacticalPhase.DOR_DR  # 14.5km - 8km = 6.5km
        else:
            return TacticalPhase.DOR_DR

    def _get_tactical_command_indices(self, env, agent_id: str, current_time: float):
        """获取战术指令索引 - 分发给具体角色"""
        if agent_id == "A0100":  # 友方长机
            return self._get_leader_command_indices(env, agent_id)
        elif agent_id == "A0200":  # 友方僚机
            return self._get_wingman_command_indices(env, agent_id)
        else:  # 敌方
            return self._get_enemy_command_indices(env, agent_id)

    def _get_leader_command_indices(self, env, agent_id: str):
        """长机战术指令索引 - 完全复制拖曳射击的成功策略"""
        try:
            if self.current_phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR, TacticalPhase.MTR_TR]:
                # 长机平稳飞行 - 朝北接敌（0°）- 完全复制拖曳射击
                return self._maintain_heading_precise(env, agent_id, 0.0)

            elif self.current_phase == TacticalPhase.TR_DOR:
                # 长机继续平稳飞行等待发射时机 - 复制拖曳射击
                return self._maintain_heading_precise(env, agent_id, 0.0)

            else:  # DOR_DR
                # 长机平稳飞行 - 复制拖曳射击
                return self._maintain_heading_precise(env, agent_id, 0.0)

        except Exception as e:
            logging.error(f"长机指令生成失败 {agent_id}: {e}")
            return 7, 8, 3  # 平稳飞行

    def _get_wingman_command_indices(self, env, agent_id: str):
        """僚机战术指令索引 - 完全复制拖曳射击的成功策略"""
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))

        # 计算僚机与敌机的距离
        leader_blue = env._jsbsims.get("B0100") or env._jsbsims.get("B0200")
        if leader_blue and leader_blue.is_alive:
            distance = self._calculate_distance(env.agents[agent_id], leader_blue)
            # 使用僚机独立的阶段判断
            wingman_phase = self._get_wingman_phase_by_distance(distance)
        else:
            wingman_phase = self.current_phase  # 如果无法计算距离，使用全局阶段

        if wingman_phase == TacticalPhase.NLT_MELD:
            # 右侧crank: 精确航向从0°调整至68°（右偏68°）- 优化编队间距至5.5海里
            return self._maintain_heading_precise(env, agent_id, 68.0)

        elif wingman_phase == TacticalPhase.MELD_MTR:
            # 左侧crank: 精确航向从68°调整回0°（左转68°）- 回归编队
            return self._maintain_heading_precise(env, agent_id, 0.0)

        elif wingman_phase == TacticalPhase.MTR_TR:
            # 平稳飞行 - 精确保持航向0°
            return self._maintain_heading_precise(env, agent_id, 0.0)

        elif wingman_phase == TacticalPhase.TR_DOR:
            # 未发射导弹，执行精确的左侧小crank指向敌机（小角度左转约10°）
            return self._maintain_heading_precise(env, agent_id, 350.0)

        elif wingman_phase == TacticalPhase.DOR_DR:
            # DOR_DR阶段：僚机执行精确的平稳飞行
            return self._maintain_heading_precise(env, agent_id, 0.0)
        else:
            return 7, 8, 3

    def _get_enemy_command_indices(self, env, agent_id: str):
        """敌方指令索引 - 简化版本"""
        try:
            # 简化的敌方行为
            if agent_id == "B0100":
                # 敌方长机：保持高度，右转
                return 7, 10, 3  # 保持高度，右转30°，保持速度
            elif agent_id == "B0200":
                # 敌方僚机：保持高度，左转
                return 7, 6, 3   # 保持高度，左转30°，保持速度
            else:
                return 7, 8, 3   # 平稳飞行
        except Exception as e:
            logging.error(f"敌方指令生成失败 {agent_id}: {e}")
            return 7, 8, 3  # 平稳飞行

    def _use_lowlevel_policy(self, env, agent_id: str, altitude_cmd_id: int, heading_cmd_id: int, velocity_cmd_id: int):
        """使用底层策略转换指令 - 完全复制拖曳射击的实现"""
        if self.my_lowlevel_policy is None:
            return np.array([0.0, 0.0, 0.0, 0.7])

        try:
            import torch

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

            # 完全按照拖曳射击的PyTorch调用方式
            _action, _rnn_states = self.my_lowlevel_policy(
                torch.FloatTensor(input_obs),
                torch.FloatTensor(self._inner_rnn_states[agent_id])
            )
            action_output = _action.detach().cpu().numpy().squeeze(0)
            self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()

            return action_output

        except Exception as e:
            logging.error(f"底层策略调用失败 {agent_id}: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            return np.array([0.0, 0.0, 0.0, 0.7])

    def _handle_missile_launch(self, env, agent_id: str, current_time: float):
        """处理导弹发射 - 基于拖曳射击逻辑"""
        try:
            # 发射冷却时间检查
            if current_time - self.last_missile_launch_time[agent_id] < 10.0:
                return

            # 根据战术阶段和角色决定发射时机
            should_launch = False

            if agent_id == "A0100":  # 长机
                # 长机在MTR-TR阶段末期发射
                if self.current_phase == TacticalPhase.MTR_TR:
                    should_launch = True
            elif agent_id == "A0200":  # 僚机
                # 僚机在TR-DOR阶段末期发射
                if self.current_phase == TacticalPhase.TR_DOR and self.high_low_states[agent_id]["altitude_established"]:
                    should_launch = True

            if should_launch:
                # 尝试发射导弹
                if hasattr(env.agents[agent_id], 'launch_missile'):
                    env.agents[agent_id].launch_missile()
                    self.last_missile_launch_time[agent_id] = current_time
                    logging.info(f"{agent_id} 发射导弹 (阶段: {self.current_phase.value})")

        except Exception as e:
            logging.error(f"导弹发射处理失败 {agent_id}: {e}")

    def step(self, env):
        """执行上下夹击战术步骤 - 完全基于拖曳射击模式"""
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

        # 为每个智能体生成观测、奖励等
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

            # 计算奖励
            reward = self._calculate_reward(env, agent_id)
            rewards[agent_id] = [reward]

            # 检查终止条件
            done, info = self.get_termination(env, agent_id, {})
            dones[agent_id] = [done]
            infos[agent_id] = {"agent_id": agent_id, "alive": True, "phase": self.current_phase.value}

        return obs, share_obs, rewards, dones, infos

    def _print_detailed_status(self, env, current_time):
        """打印详细状态信息 - 基于拖曳射击"""
        try:
            logging.info(f"\n=== 上下夹击战术状态 t={current_time:.1f}s ===")
            logging.info(f"当前阶段: {self.current_phase.value}")

            for agent_id in ["A0100", "A0200"]:
                if agent_id in env._jsbsims and env._jsbsims[agent_id].is_alive:
                    agent = env._jsbsims[agent_id]
                    pos = agent.get_position()
                    altitude = agent.get_property_value(c.position_h_sl_m)
                    heading = np.rad2deg(agent.get_property_value(c.attitude_psi_rad))
                    velocity = agent.get_property_value(c.velocities_u_fps) * 0.3048

                    # 获取高度状态
                    if agent_id in self.high_low_states:
                        state = self.high_low_states[agent_id]
                        target_alt = state["target_altitude"]
                        alt_established = "✓" if state["altitude_established"] else "✗"
                        alt_diff = altitude - target_alt

                        logging.info(f"{agent_id}: 位置=({pos[0]/1000:.1f}, {pos[1]/1000:.1f})km, "
                                   f"高度={altitude:.0f}m (目标{target_alt:.0f}m, 差值{alt_diff:+.0f}m, 建立{alt_established}), "
                                   f"航向={heading:.1f}°, 速度={velocity:.1f}m/s")

            # 计算高度差
            if ("A0100" in env._jsbsims and env._jsbsims["A0100"].is_alive and
                "A0200" in env._jsbsims and env._jsbsims["A0200"].is_alive):
                alt_a0100 = env._jsbsims["A0100"].get_property_value(c.position_h_sl_m)
                alt_a0200 = env._jsbsims["A0200"].get_property_value(c.position_h_sl_m)
                alt_diff = alt_a0200 - alt_a0100
                logging.info(f"高度差: {alt_diff:+.0f}m (目标: +1200m)")

        except Exception as e:
            logging.error(f"状态打印失败: {e}")

    def _calculate_reward(self, env, agent_id):
        """计算奖励 - 基于拖曳射击"""
        try:
            if not env._jsbsims[agent_id].is_alive:
                return -10.0

            reward = 0.0

            # 高度控制奖励
            if agent_id in self.high_low_states:
                current_altitude = env._jsbsims[agent_id].get_property_value(c.position_h_sl_m)
                target_altitude = self.high_low_states[agent_id]["target_altitude"]
                altitude_diff = abs(current_altitude - target_altitude)

                if altitude_diff < 50:  # 50m容差内
                    reward += 1.0
                    self.high_low_states[agent_id]["altitude_established"] = True
                elif altitude_diff < 200:  # 200m容差内
                    reward += 0.5
                else:
                    reward -= 0.1

            # 存活奖励
            reward += 0.1

            return reward

        except Exception as e:
            logging.error(f"奖励计算失败 {agent_id}: {e}")
            return 0.0

    def _maintain_altitude_and_heading_precise(self, env, agent_id, target_altitude, target_heading):
        """精确的高度和航向控制 - 基于拖曳射击的成功方法"""
        try:
            current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
            current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))

            # 高度控制 - 使用更温和的控制
            altitude_diff = target_altitude - current_altitude
            if abs(altitude_diff) > 100:  # 100m容差
                if altitude_diff > 0:
                    # 爬升控制 - 使用更温和的指令
                    if altitude_diff > 2000:
                        altitude_cmd_id = 10  # +300m 中等爬升
                    elif altitude_diff > 1000:
                        altitude_cmd_id = 9   # +150m 缓慢爬升
                    elif altitude_diff > 500:
                        altitude_cmd_id = 8   # +50m 非常缓慢爬升
                    else:
                        altitude_cmd_id = 8   # +50m 微调爬升
                else:
                    # 下降控制 - 使用更温和的指令
                    if altitude_diff < -1000:
                        altitude_cmd_id = 6   # -50m 缓慢下降
                    else:
                        altitude_cmd_id = 6   # -50m 微调下降
            else:
                altitude_cmd_id = 7  # 保持高度

            # 调试信息
            if env.current_step % 25 == 0:  # 每5秒打印一次
                logging.info(f"{agent_id} 高度控制: 当前={current_altitude:.0f}m, 目标={target_altitude:.0f}m, "
                           f"差值={altitude_diff:+.0f}m, 指令ID={altitude_cmd_id}")

            # 航向控制 - 精确控制
            heading_diff = target_heading - current_heading
            while heading_diff > 180: heading_diff -= 360
            while heading_diff < -180: heading_diff += 360

            # 默认保持航向
            heading_cmd_id = 8  # 0°变化

            # 精确航向控制：2度精度
            if abs(heading_diff) > 2.0:
                heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))

            # 速度控制 - 爬升时加速
            if altitude_diff > 200:
                velocity_cmd_id = 4  # +20m/s 加速爬升
            elif altitude_diff < -200:
                velocity_cmd_id = 2  # -20m/s 减速下降
            else:
                velocity_cmd_id = 3  # 保持速度

            return altitude_cmd_id, heading_cmd_id, velocity_cmd_id

        except Exception as e:
            logging.error(f"精确控制失败 {agent_id}: {e}")
            return 7, 8, 3

    def _convert_heading_to_index(self, heading_diff_rad):
        """将航向差值转换为索引 - 基于拖曳射击"""
        try:
            # 找到最接近的航向指令索引
            distances = np.abs(self.norm_delta_heading - heading_diff_rad)
            return np.argmin(distances)
        except:
            return 8  # 默认保持航向

    def _maintain_heading_precise(self, env, agent_id, target_heading, duration=10.0):
        """精确的航向保持 - 完全复制拖曳射击的成功实现"""
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

    def _maintain_heading_with_climb(self, env, agent_id, target_heading, target_altitude):
        """航向保持 + 温和爬升 - 专为僚机设计，基于拖曳射击的成功模式"""
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)

        # 计算航向差值
        heading_diff = target_heading - current_heading
        while heading_diff > 180: heading_diff -= 360
        while heading_diff < -180: heading_diff += 360

        # 高度控制 - 非常温和的爬升，避免过度机动
        altitude_diff = target_altitude - current_altitude
        if altitude_diff > 200:  # 需要明显爬升
            altitude_cmd_id = 8  # +50m 温和爬升
        elif altitude_diff > 50:   # 需要微调爬升
            altitude_cmd_id = 8  # +50m 温和爬升
        elif altitude_diff < -200: # 需要明显下降
            altitude_cmd_id = 6  # -50m 温和下降
        elif altitude_diff < -50:  # 需要微调下降
            altitude_cmd_id = 6  # -50m 温和下降
        else:
            altitude_cmd_id = 7  # 保持高度

        # 航向控制 - 完全复制拖曳射击的精确控制
        heading_cmd_id = 8   # 默认保持航向
        if abs(heading_diff) > 1.0:
            heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))

        # 速度控制 - 保持稳定，避免过度加速
        velocity_cmd_id = 3  # 保持速度，与拖曳射击一致

        return altitude_cmd_id, heading_cmd_id, velocity_cmd_id

    def _get_wingman_phase_by_distance(self, distance):
        """根据距离判断僚机的战术阶段 - 复制拖曳射击"""
        distance_km = distance / 1000.0

        if distance_km > 80:
            return TacticalPhase.NLT_MELD
        elif distance_km > 60:
            return TacticalPhase.MELD_MTR
        elif distance_km > 40:
            return TacticalPhase.MTR_TR
        elif distance_km > 20:
            return TacticalPhase.TR_DOR
        else:
            return TacticalPhase.DOR_DR

    def _calculate_distance(self, agent1, agent2):
        """计算两个智能体之间的距离 - 复制拖曳射击"""
        try:
            pos1 = agent1.get_position()
            pos2 = agent2.get_position()
            return np.linalg.norm(np.array(pos1) - np.array(pos2))
        except:
            return 100000.0  # 默认距离
