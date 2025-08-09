#!/usr/bin/env python3
"""
基础战术任务抽象类 - 为不同战术类型提供统一接口
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, Tuple
import math

# 导入现有模块
from envs.JSBSim.tasks.multiplecombat_task import MultipleCombatTask
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.core.simulatior import AircraftSimulator
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.tasks.pure_maneuvers import BasicManeuvers, CompositeManeuverExecutor


class TacticalType(Enum):
    """战术类型枚举"""
    DRAG_SHOOT = "DRAG_SHOOT"        # 拖曳射击
    PINCER_ATTACK = "PINCER_ATTACK"  # 钳形夹击
    # 未来可扩展其他战术类型


class TacticalPhase(Enum):
    """通用战术阶段 - 基于距离的标准阶段划分"""
    NLT_MELD = "NLT_MELD"    # 90-81km: 非致命时间-中距离交战线
    MELD_MTR = "MELD_MTR"    # 81-45km: 中距离交战线-导弹目标范围
    MTR_TR = "MTR_TR"        # 45-41km: 导弹目标范围-目标范围
    TR_DOR = "TR_DOR"        # 41-19.6km: 目标范围-动态攻击范围
    DOR_DR = "DOR_DR"        # 19.6-14.5km: 动态攻击范围-防御范围


class BaseTacticalTask(MultipleCombatTask, ABC):
    """
    基础战术任务抽象类
    
    提供所有战术类型的通用基础设施：
    - 阶段管理
    - 基础机动
    - 雷达管理
    - 导弹控制
    - 编队协调
    """
    
    def __init__(self, config):
        """初始化基础战术任务"""
        super().__init__(config)
        
        # 战术类型标识
        self.tactical_type = self.get_tactical_type()
        
        # 通用战术状态
        self.current_phase = TacticalPhase.NLT_MELD
        self.phase_start_time = 0.0
        self.phase_start_step = 0
        
        # 通用距离配置 - 可被子类覆盖
        self.tactical_distances = {
            'NLT_MELD_min': 81000,   # 81km
            'MELD_MTR_min': 45000,   # 45km
            'MTR_TR_min': 41000,     # 41km
            'TR_DOR_min': 19600,     # 19.6km
            'DOR_DR_min': 14500,     # 14.5km
        }
        
        # 基础机动系统
        self.my_lowlevel_policy = BaselineActor()
        self._inner_rnn_states = {}
        self.basic_maneuvers = BasicManeuvers()
        self.composite_executor = CompositeManeuverExecutor()
        
        # 机动状态跟踪
        self.active_maneuvers = {}
        self.maneuver_start_times = {}
        
        # 导弹发射管理
        self.last_missile_launch_time = {}
        self.friendly_missile_cooldown = 2.0  # 友方2秒冷却
        self.enemy_missile_cooldown = 10.0    # 敌方10秒冷却
        
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
        
        logging.info(f"{self.tactical_type.value} 战术任务初始化完成")
    
    @abstractmethod
    def get_tactical_type(self) -> TacticalType:
        """获取战术类型 - 子类必须实现"""
        pass
    
    @abstractmethod
    def _get_tactical_command_indices(self, env, agent_id: str) -> Tuple[int, int, int]:
        """获取战术指令索引 - 子类必须实现具体战术逻辑"""
        pass
    
    def normalize_action(self, env, agent_id, action):
        """
        动作归一化 - 通用框架
        """
        if agent_id not in env.agents or not env.agents[agent_id].is_alive:
            return np.array([0.0, 0.0, 0.0, 0.7])
        
        current_time = env.current_step * env.time_interval
        
        # 处理战术逻辑
        return self._process_tactical_logic(env, agent_id, current_time)
    
    def _process_tactical_logic(self, env, agent_id: str, current_time: float):
        """处理战术逻辑 - 通用框架"""
        try:
            # 更新战术阶段
            self._update_tactical_phase(env)
            
            # 获取战术指令索引 - 调用子类实现
            altitude_cmd_id, heading_cmd_id, velocity_cmd_id = self._get_tactical_command_indices(env, agent_id)
            
            # 使用底层策略
            return self._use_lowlevel_policy(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)
            
        except Exception as e:
            logging.error(f"战术逻辑处理失败 {agent_id}: {e}")
            return self._use_lowlevel_policy(env, agent_id, 7, 8, 3)  # 平稳飞行
    
    def _update_tactical_phase(self, env):
        """更新战术阶段 - 通用实现"""
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
            self.phase_start_time = current_time
            self.phase_start_step = env.current_step
    
    def _get_phase_by_distance(self, distance: float) -> TacticalPhase:
        """根据距离确定阶段 - 通用实现"""
        if distance >= self.tactical_distances['NLT_MELD_min']:
            return TacticalPhase.NLT_MELD
        elif distance >= self.tactical_distances['MELD_MTR_min']:
            return TacticalPhase.MELD_MTR
        elif distance >= self.tactical_distances['MTR_TR_min']:
            return TacticalPhase.MTR_TR
        elif distance >= self.tactical_distances['TR_DOR_min']:
            return TacticalPhase.TR_DOR
        else:
            return TacticalPhase.DOR_DR
    
    def _calculate_distance(self, aircraft1: AircraftSimulator, aircraft2: AircraftSimulator) -> float:
        """计算两架飞机之间的距离 - 通用实现"""
        pos1 = aircraft1.get_position()
        pos2 = aircraft2.get_position()
        return np.linalg.norm(pos1 - pos2)
    
    def _use_lowlevel_policy(self, env, agent_id: str, altitude_cmd_id: int, heading_cmd_id: int, velocity_cmd_id: int):
        """使用底层策略 - 通用实现"""
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
            obs = env.get_obs(agent_id)
            
            # 使用baseline模型
            action, self._inner_rnn_states[agent_id] = self.my_lowlevel_policy.predict(
                obs, self._inner_rnn_states[agent_id], deterministic=True
            )
            
            return action
            
        except Exception as e:
            logging.error(f"底层策略执行失败 {agent_id}: {e}")
            return np.array([0.0, 0.0, 0.0, 0.7])
    
    def step(self, env):
        """执行战术步骤 - 通用框架"""
        # 更新战术阶段
        self._update_tactical_phase(env)
        
        # 处理导弹发射
        current_time = env.current_step * env.time_interval
        for agent_id in env._jsbsims.keys():
            if env._jsbsims[agent_id].is_alive:
                self._handle_missile_launch(env, agent_id, current_time)
        
        # 更新雷达状态
        self._update_radar_states(env, current_time)
        
        # 调用父类step方法
        return super().step(env)
    
    def _handle_missile_launch(self, env, agent_id: str, current_time: float):
        """处理导弹发射 - 基础实现，子类可覆盖"""
        # 基础发射逻辑 - 子类可以覆盖实现特定战术的发射策略
        pass
    
    def _update_radar_states(self, env, current_time: float):
        """更新雷达状态 - 通用实现"""
        try:
            from radar_manager import update_all_radars
            update_all_radars(env, current_time)
        except ImportError:
            logging.warning("雷达管理器模块未找到")
        except Exception as e:
            logging.error(f"雷达状态更新失败: {e}")
