#!/usr/bin/env python3
"""
工作的钳形夹击战术任务 - 基于现有DragShootTacticalTask的架构
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


class TacticalPhase(Enum):
    """钳形夹击战术阶段"""
    NLT_MELD = "NLT_MELD"    # 90-81km: 钳形展开
    MELD_MTR = "MELD_MTR"    # 81-45km: 钳形收拢
    MTR_TR = "MTR_TR"        # 45-41km: 分层攻击
    TR_DOR = "TR_DOR"        # 41-19.6km: 脱离机动
    DOR_DR = "DOR_DR"        # 19.6-14.5km: 返航


class PincerAttackTacticalTask(MultipleCombatTask):
    """钳形夹击战术任务"""
    
    def __init__(self, config):
        """初始化钳形夹击战术任务"""
        super().__init__(config)
        
        # 钳形夹击特有配置
        self.tactical_distances = {
            'NLT_MELD_min': 81000,   # 81km
            'MELD_MTR_min': 45000,   # 45km
            'MTR_TR_min': 41000,     # 41km
            'TR_DOR_min': 19600,     # 19.6km
            'DOR_DR_min': 14500,     # 14.5km
        }
        
        # 钳形夹击参数
        self.pincer_config = {
            'crank_angle': 45.0,           # Crank角度
            'max_off_boresight': 60.0,     # 最大侧向发射角度
            'formation_spacing': 8000,      # 编队间距 (8km)
            'leader_priority': True,        # 长机优先发射
            'wingman_delay': 5.0,          # 僚机发射延迟 (秒)
        }
        
        # 当前战术阶段
        self.current_phase = TacticalPhase.NLT_MELD
        
        # 底层策略
        self.my_lowlevel_policy = BaselineActor()
        self._inner_rnn_states = {}
        
        # 动作空间定义：[15, 17, 7] - 高层战术指令空间
        self.norm_delta_altitude = np.array([
            -1500, -1000, -750, -500, -300, -150, -50, 0, 50, 150, 300, 500, 750, 1000, 1500
        ]) / 1000.0  # 索引7 = 0m变化（平稳飞行）
        
        self.norm_delta_heading = np.array([
            -60, -45, -30, -20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 30, 45, 60
        ]) * np.pi / 180.0  # 索引8 = 0°变化（保持航向）
        
        self.norm_delta_velocity = np.array([
            -100, -50, -20, 0, 20, 50, 100
        ])  # 索引3 = 0m/s变化（保持速度）
        
        logging.info("钳形夹击战术任务初始化完成")
    
    def normalize_action(self, env, agent_id, action):
        """钳形夹击战术的normalize_action实现"""
        if agent_id not in env.agents or not env.agents[agent_id].is_alive:
            return np.array([0.0, 0.0, 0.0, 0.7])
        
        try:
            current_time = env.current_step * env.time_interval
            
            # 获取战术指令索引
            altitude_cmd_id, heading_cmd_id, velocity_cmd_id = self._get_tactical_command_indices(env, agent_id)
            
            # 使用底层策略
            return self._use_lowlevel_policy(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)
            
        except Exception as e:
            logging.error(f"钳形夹击战术逻辑失败 {agent_id}: {e}")
            return np.array([0.0, 0.0, 0.0, 0.7])
    
    def _get_tactical_command_indices(self, env, agent_id):
        """获取钳形夹击战术指令索引"""
        # 根据当前阶段和角色确定战术行为
        if agent_id.startswith('A'):  # 友方
            return self._get_friendly_command_indices(env, agent_id)
        else:  # 敌方
            return self._get_enemy_command_indices(env, agent_id)
    
    def _get_friendly_command_indices(self, env, agent_id):
        """友方钳形夹击指令"""
        if self.current_phase == TacticalPhase.NLT_MELD:
            # 阶段1: 钳形展开
            return self._execute_pincer_spread(env, agent_id)
            
        elif self.current_phase == TacticalPhase.MELD_MTR:
            # 阶段2: 钳形收拢
            return self._execute_pincer_converge(env, agent_id)
            
        elif self.current_phase == TacticalPhase.MTR_TR:
            # 阶段3: 分层攻击
            return self._execute_layered_attack(env, agent_id)
            
        elif self.current_phase == TacticalPhase.TR_DOR:
            # 阶段4: 脱离机动
            return self._execute_pincer_escape(env, agent_id)
            
        else:  # DOR_DR
            # 阶段5: 返航
            return self._execute_return_to_base(env, agent_id)
    
    def _execute_pincer_spread(self, env, agent_id):
        """执行钳形展开机动"""
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        
        # 计算目标Crank角度
        if agent_id == "A0100":  # 长机向右Crank
            target_heading = (current_heading + self.pincer_config['crank_angle']) % 360
            print(f"[钳形展开] {agent_id} (长机): 当前航向{current_heading:.1f}° -> 目标航向{target_heading:.1f}° (右Crank)")
        else:  # 僚机向左Crank
            target_heading = (current_heading - self.pincer_config['crank_angle']) % 360
            print(f"[钳形展开] {agent_id} (僚机): 当前航向{current_heading:.1f}° -> 目标航向{target_heading:.1f}° (左Crank)")
        
        # 计算航向差值
        heading_diff = target_heading - current_heading
        while heading_diff > 180: heading_diff -= 360
        while heading_diff < -180: heading_diff += 360
        
        # 转换为指令索引
        if abs(heading_diff) > 2.0:
            heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))
        else:
            heading_cmd_id = 8  # 保持航向
        
        return 7, heading_cmd_id, 3  # 保持高度，调整航向，保持速度
    
    def _execute_pincer_converge(self, env, agent_id):
        """执行钳形收拢机动"""
        # 获取敌机位置
        enemy_pos = self._get_primary_enemy_position(env)
        if enemy_pos is None:
            return 7, 8, 3
        
        # 计算指向敌机的航向
        my_pos = env.agents[agent_id].get_position()
        target_heading = self._calculate_bearing_to_target(my_pos, enemy_pos)
        
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        heading_diff = target_heading - current_heading
        while heading_diff > 180: heading_diff -= 360
        while heading_diff < -180: heading_diff += 360
        
        print(f"[钳形收拢] {agent_id}: 当前航向{current_heading:.1f}° -> 目标航向{target_heading:.1f}°")
        
        # 转换为指令索引
        if abs(heading_diff) > 2.0:
            heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))
        else:
            heading_cmd_id = 8
        
        return 7, heading_cmd_id, 3
    
    def _execute_layered_attack(self, env, agent_id):
        """执行分层攻击"""
        if agent_id == "A0100":  # 长机
            print(f"[分层攻击] {agent_id} (长机): 积极攻击")
            return self._leader_attack_behavior(env, agent_id)
        else:  # 僚机
            print(f"[分层攻击] {agent_id} (僚机): 保持滞后")
            return self._wingman_support_behavior(env, agent_id)
    
    def _execute_pincer_escape(self, env, agent_id):
        """执行钳形脱离机动"""
        print(f"[脱离机动] {agent_id}: 执行Short Skate")
        # 转向180度返航
        target_heading = 180.0
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        
        heading_diff = target_heading - current_heading
        while heading_diff > 180: heading_diff -= 360
        while heading_diff < -180: heading_diff += 360
        
        heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff)) if abs(heading_diff) > 2.0 else 8
        
        return 7, heading_cmd_id, 4  # 保持高度，转向返航，加速
    
    def _execute_return_to_base(self, env, agent_id):
        """执行返航"""
        print(f"[返航] {agent_id}: 返回基地")
        return 7, 8, 4  # 保持高度，保持航向，加速返航
    
    def _leader_attack_behavior(self, env, agent_id):
        """长机攻击行为"""
        return self._execute_pincer_converge(env, agent_id)  # 继续指向敌机
    
    def _wingman_support_behavior(self, env, agent_id):
        """僚机支援行为"""
        return 7, 8, 3  # 保持当前状态
    
    def _get_enemy_command_indices(self, env, agent_id):
        """敌方指令 - 简单对抗逻辑"""
        # 敌方保持朝向友方
        friendly_pos = self._get_primary_friendly_position(env)
        if friendly_pos is None:
            return 7, 8, 3
        
        my_pos = env.agents[agent_id].get_position()
        target_heading = self._calculate_bearing_to_target(my_pos, friendly_pos)
        
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        heading_diff = target_heading - current_heading
        while heading_diff > 180: heading_diff -= 360
        while heading_diff < -180: heading_diff += 360
        
        heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff)) if abs(heading_diff) > 2.0 else 8
        
        return 7, heading_cmd_id, 3
    
    def _get_primary_enemy_position(self, env):
        """获取主要敌机位置"""
        for agent_id, agent in env.agents.items():
            if agent_id.startswith('B') and agent.is_alive:
                return agent.get_position()
        return None
    
    def _get_primary_friendly_position(self, env):
        """获取主要友机位置"""
        for agent_id, agent in env.agents.items():
            if agent_id.startswith('A') and agent.is_alive:
                return agent.get_position()
        return None
    
    def _calculate_bearing_to_target(self, my_pos, target_pos):
        """计算指向目标的方位角"""
        dx = target_pos[0] - my_pos[0]
        dy = target_pos[1] - my_pos[1]
        bearing = np.rad2deg(np.arctan2(dx, dy))
        return bearing % 360
    
    def _convert_heading_to_index(self, heading_diff_rad):
        """将航向差值转换为指令索引"""
        diff_deg = np.rad2deg(heading_diff_rad)
        heading_options = [-60, -45, -30, -20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 30, 45, 60]
        
        closest_idx = 0
        min_diff = abs(diff_deg - heading_options[0])
        
        for i, option in enumerate(heading_options):
            if abs(diff_deg - option) < min_diff:
                min_diff = abs(diff_deg - option)
                closest_idx = i
        
        return closest_idx
    
    def _use_lowlevel_policy(self, env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id):
        """使用底层策略"""
        try:
            # 构建高层指令
            high_level_action = np.array([
                self.norm_delta_altitude[altitude_cmd_id],
                self.norm_delta_heading[heading_cmd_id],
                self.norm_delta_velocity[velocity_cmd_id]
            ])
            
            # 获取或初始化RNN状态
            if agent_id not in self._inner_rnn_states:
                self._inner_rnn_states[agent_id] = np.zeros((1, 128))
            
            # 获取观测
            obs = self.get_obs(env, agent_id)
            
            # 使用baseline模型
            action, self._inner_rnn_states[agent_id] = self.my_lowlevel_policy.predict(
                obs, self._inner_rnn_states[agent_id], deterministic=True
            )
            
            return action
            
        except Exception as e:
            logging.error(f"底层策略执行失败 {agent_id}: {e}")
            return np.array([0.0, 0.0, 0.0, 0.7])
    
    def step(self, env):
        """执行钳形夹击战术步骤 - 基于DragShootTacticalTask的step方法"""
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
            
            # 计算奖励
            rewards[agent_id] = [0.0]  # 简化奖励
            
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
    
    def _get_phase_by_distance(self, distance):
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
    
    def _calculate_distance(self, aircraft1, aircraft2):
        """计算两架飞机之间的距离"""
        pos1 = aircraft1.get_position()
        pos2 = aircraft2.get_position()
        return np.linalg.norm(pos1 - pos2)
