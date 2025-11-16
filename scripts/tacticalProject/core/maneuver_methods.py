#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
机动方法模块（Maneuver Methods Module）

包含TacticsExecutor的基础机动执行方法

作者：TacticalProject重构组
版本：2.0.0
日期：2024-11-13
"""

from typing import Tuple, Optional, Dict
import numpy as np
from envs.JSBSim.core.catalog import Catalog as c
import logging


class ManeuverMethods:
    """
    机动方法集合类
    包含所有基础机动的执行方法
    """
    
    def __init__(self):
        """初始化机动方法所需的状态变量"""
        # Short Skate机动专用状态
        if not hasattr(self, 'short_skate_states'):
            self.short_skate_states = {}
    
    # ==================== 基础控制方法 ====================
    
    def _maintain_heading_precise(self, env, agent_id: str, target_heading: float) -> Tuple[int, int, int]:
        """
        精确保持指定航向飞行
        
        Args:
            env: 环境对象
            agent_id: 智能体ID
            target_heading: 目标航向（度）
        
        Returns:
            (alt_cmd, hdg_cmd, vel_cmd): 控制指令三元组
        """
        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
        heading_diff = self._normalize_angle(target_heading - current_heading)
        
        # 计算航向指令
        hdg_cmd = 8  # 默认保持
        if abs(heading_diff) > 5:
            hdg_cmd = self._calculate_heading_command(heading_diff)
        
        return 7, hdg_cmd, 3  # 保持高度、调整航向、保持速度
    
    def _turn_to_heading(self, env, agent_id: str, target_heading: float) -> Tuple[int, int, int]:
        """
        转向到指定航向（更积极的转向）
        
        Args:
            env: 环境对象
            agent_id: 智能体ID
            target_heading: 目标航向（度）
        
        Returns:
            (alt_cmd, hdg_cmd, vel_cmd): 控制指令三元组
        """
        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
        heading_diff = self._normalize_angle(target_heading - current_heading)
        
        # 直接计算航向指令
        hdg_idx = self._convert_heading_to_index(np.deg2rad(heading_diff))
        
        return 7, hdg_idx, 3
    
    def _calculate_heading_command(self, heading_diff: float) -> int:
        """
        计算航向控制指令
        
        Args:
            heading_diff: 航向差（度）
        
        Returns:
            航向指令索引 [0-16]
        """
        if heading_diff > 90:
            return 14  # 右转90°
        elif heading_diff > 60:
            return 12  # 右转60°
        elif heading_diff > 30:
            return 10  # 右转30°
        elif heading_diff > 15:
            return 9   # 右转15°
        elif heading_diff < -90:
            return 2   # 左转90°
        elif heading_diff < -60:
            return 4   # 左转60°
        elif heading_diff < -30:
            return 6   # 左转30°
        elif heading_diff < -15:
            return 7   # 左转15°
        else:
            return 8   # 保持
    
    def _convert_heading_to_index(self, heading_rad: float) -> int:
        """
        将航向变化（弧度）转换为指令索引
        
        Args:
            heading_rad: 航向变化（弧度）
        
        Returns:
            航向指令索引 [0-16]
        """
        # 映射表：弧度 -> 指令索引
        heading_map = [
            (-np.pi, 0),      # -180°
            (-3*np.pi/4, 1),  # -135°
            (-np.pi/2, 2),    # -90°
            (-5*np.pi/12, 3), # -75°
            (-np.pi/3, 4),    # -60°
            (-np.pi/4, 5),    # -45°
            (-np.pi/6, 6),    # -30°
            (-np.pi/12, 7),   # -15°
            (0, 8),           # 0°
            (np.pi/12, 9),    # 15°
            (np.pi/6, 10),    # 30°
            (np.pi/4, 11),    # 45°
            (np.pi/3, 12),    # 60°
            (5*np.pi/12, 13), # 75°
            (np.pi/2, 14),    # 90°
            (3*np.pi/4, 15),  # 135°
            (np.pi, 16),      # 180°
        ]
        
        # 找最接近的值
        min_diff = float('inf')
        best_idx = 8
        for angle, idx in heading_map:
            diff = abs(heading_rad - angle)
            if diff < min_diff:
                min_diff = diff
                best_idx = idx
        
        return best_idx
    
    def _calculate_altitude_command(self, alt_diff: float) -> int:
        """
        计算高度控制指令
        
        Args:
            alt_diff: 高度差（米）
        
        Returns:
            高度指令索引 [0-14]
        """
        if alt_diff > 1000:
            return 13  # 爬升1000米
        elif alt_diff > 500:
            return 11  # 爬升500米
        elif alt_diff > 300:
            return 10  # 爬升300米
        elif alt_diff > 200:
            return 9   # 爬升200米
        elif alt_diff < -1000:
            return 1   # 下降1000米
        elif alt_diff < -500:
            return 3   # 下降500米
        elif alt_diff < -300:
            return 4   # 下降300米
        elif alt_diff < -200:
            return 5   # 下降200米
        else:
            return 7   # 保持
    
    def _normalize_angle(self, angle: float) -> float:
        """
        角度归一化到[-180, 180]
        
        Args:
            angle: 原始角度（度）
        
        Returns:
            归一化后的角度
        """
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
    
    # ==================== 战术机动方法 ====================
    
    def _execute_tactical_crank(self, env, agent_id: str, direction: str = 'left', 
                               climb: bool = False) -> Tuple[int, int, int]:
        """
        执行战术Crank机动
        飞机做crank机动的同时可选择爬升或下降
        
        Args:
            env: 环境对象
            agent_id: 智能体ID
            direction: 方向（'left' 或 'right'）
            climb: 是否爬升
        
        Returns:
            (alt_cmd, hdg_cmd, vel_cmd): 控制指令三元组
        """
        current_time = env.current_step * env.time_interval
        
        # 初始化状态
        if agent_id not in self.maneuver_states or \
           self.maneuver_states[agent_id].get('type') != 'tactical_crank':
            self.maneuver_states[agent_id] = {
                'type': 'tactical_crank',
                'phase': 'crank',
                'start_time': current_time,
                'phase_start_time': current_time,
                'direction': direction,
                'climb': climb
            }
            action = "爬升" if climb else "平飞"
            self.logger.info(f"🔄 [{agent_id}] 开始{direction}侧Crank+{action}机动")
        
        state = self.maneuver_states[agent_id]
        phase_time = current_time - state['phase_start_time']
        
        # 执行Crank机动
        if phase_time < self.maneuver_params['crank']['duration']:
            alt_cmd = 11 if state['climb'] else 7  # 爬升或保持
            hdg_cmd = 4 if state['direction'] == 'left' else 12  # 左转或右转60°
            return alt_cmd, hdg_cmd, 3
        else:
            # 机动完成
            del self.maneuver_states[agent_id]
            return 7, 8, 3
    
    def _execute_short_skate_precise(self, env, agent_id: str, current_time: float, 
                                    direction: str = 'auto') -> Tuple[int, int, int]:
        """
        执行精确的Short Skate机动（S形规避）
        
        阶段：
            1. Crank机动：偏转60°
            2. Turn Cold：反向转150°
            3. Escape：加速逃离
        
        Args:
            env: 环境对象
            agent_id: 智能体ID
            current_time: 当前时间
            direction: 方向（'auto'自动选择，'left'左转，'right'右转）
        
        Returns:
            (alt_cmd, hdg_cmd, vel_cmd): 控制指令三元组
        """
        if agent_id not in self.short_skate_states:
            self._init_short_skate(agent_id, current_time, direction)
        
        state = self.short_skate_states[agent_id]
        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
        
        if state["initial_heading"] is None:
            state["initial_heading"] = current_heading
        
        phase_time = current_time - state["phase_start_time"]
        
        # 时间参数（僚机时间缩短）
        if agent_id == "A0200":
            crank_duration = 12.0     # 缩短：18→12秒
            turn_cold_duration = 18.0  # 缩短：30→18秒
            escape_duration = 15.0     # 缩短：22→15秒
        else:
            crank_duration = 6.0
            turn_cold_duration = 15.0
            escape_duration = 15.0
        
        # 阶段1：Crank机动
        if state["phase"] == "crank":
            if phase_time < crank_duration:
                target_heading = state["initial_heading"] + state["crank_angle"]
                target_heading = target_heading % 360
                heading_diff = self._normalize_angle(target_heading - current_heading)
                
                if abs(heading_diff) > 5.0:
                    hdg_cmd = 6 if heading_diff < 0 else 10
                    return 7, hdg_cmd, 3
                else:
                    return 7, 8, 3
            else:
                state["phase"] = "turn_cold"
                state["phase_start_time"] = current_time
                state["turn_cold_start_heading"] = current_heading
        
        # 阶段2：Turn Cold
        elif state["phase"] == "turn_cold":
            if phase_time < turn_cold_duration:
                target_heading = state["turn_cold_start_heading"] + state["turn_cold_angle"]
                target_heading = target_heading % 360
                heading_diff = self._normalize_angle(target_heading - current_heading)
                
                if abs(heading_diff) > 5.0:
                    hdg_cmd = 6 if heading_diff < 0 else 10
                    return 7, hdg_cmd, 3
                else:
                    return 7, 8, 3
            else:
                state["phase"] = "escape"
                state["phase_start_time"] = current_time
        
        # 阶段3：加速逃离
        elif state["phase"] == "escape":
            if phase_time < escape_duration:
                return 7, 8, 5  # 保持航向+加速
            else:
                # 完成，返航
                if agent_id.startswith('A'):
                    target_heading = 180.0
                else:
                    target_heading = 0.0
                
                heading_diff = self._normalize_angle(target_heading - current_heading)
                if abs(heading_diff) > 5.0:
                    hdg_cmd = 6 if heading_diff < 0 else 10
                    return 7, hdg_cmd, 3
                else:
                    # 机动完成，清理状态
                    del self.short_skate_states[agent_id]
                    return 7, 8, 3
        
        return 7, 8, 3
    
    def _init_short_skate(self, agent_id: str, current_time: float, direction: str = 'auto'):
        """
        初始化Short Skate机动状态
        
        Args:
            agent_id: 智能体ID
            current_time: 当前时间
            direction: 方向选择
        """
        # 决定方向
        if direction == 'auto':
            if agent_id.startswith('A'):
                # 我方：长机左转，僚机右转
                direction = 'left' if agent_id.endswith('100') else 'right'
            else:
                # 敌方：相反
                direction = 'right' if agent_id.endswith('100') else 'left'
        
        # 设置角度
        crank_angle = 60.0 if direction == 'left' else -60.0
        turn_cold_angle = -150.0 if direction == 'left' else 150.0
        
        self.short_skate_states[agent_id] = {
            "phase": "crank",
            "phase_start_time": current_time,
            "initial_heading": None,
            "crank_angle": crank_angle,
            "turn_cold_angle": turn_cold_angle,
            "turn_cold_start_heading": None,
            "direction": direction
        }
        
        self.logger.info(f"🔄 [{agent_id}] 初始化Short Skate机动 ({direction}侧)")
    
    def _execute_beam(self, env, agent_id: str) -> Tuple[int, int, int]:
        """
        执行Beam机动（垂直方向飞行）
        
        Args:
            env: 环境对象
            agent_id: 智能体ID
        
        Returns:
            (alt_cmd, hdg_cmd, vel_cmd): 控制指令三元组
        """
        # Beam机动：转向垂直于威胁方向
        # 简化实现：左转或右转90°
        is_lead = agent_id.endswith('100')
        hdg_cmd = 2 if is_lead else 14  # 长机左转90°，僚机右转90°
        
        return 7, hdg_cmd, 3
    
    def _execute_notch_back(self, env, agent_id: str) -> Tuple[int, int, int]:
        """
        执行Notch Back机动
        完整的陷波机动，包括转向和下降
        
        Args:
            env: 环境对象
            agent_id: 智能体ID
        
        Returns:
            (alt_cmd, hdg_cmd, vel_cmd): 控制指令三元组
        """
        current_time = env.current_step * env.time_interval
        
        # 初始化状态
        if agent_id not in self.maneuver_states or \
           self.maneuver_states[agent_id].get('type') != 'notch_back':
            self.maneuver_states[agent_id] = {
                'type': 'notch_back',
                'phase': 'turn',
                'start_time': current_time,
                'phase_start_time': current_time
            }
            self.logger.info(f"🛡️ [{agent_id}] 开始Notch Back机动")
        
        state = self.maneuver_states[agent_id]
        phase_time = current_time - state['phase_start_time']
        
        # 阶段1：转向120°
        if state['phase'] == 'turn':
            if phase_time < 5.0:
                # 转向+下降
                return 3, 4, 3  # 下降500m + 左转60°
            else:
                state['phase'] = 'maintain'
                state['phase_start_time'] = current_time
        
        # 阶段2：保持
        elif state['phase'] == 'maintain':
            if phase_time < 5.0:
                return 7, 8, 3  # 保持
            else:
                # 机动完成
                del self.maneuver_states[agent_id]
                return 7, 8, 3
        
        return 7, 8, 3
    
    # ==================== 编队控制方法 ====================
    
    def _establish_trail_formation(self, env, agent_id: str, target_distance: float) -> Tuple[int, int, int]:
        """
        建立纵向编队（前后队形）
        
        Args:
            env: 环境对象
            agent_id: 智能体ID（僚机）
            target_distance: 目标间距（米）
        
        Returns:
            (alt_cmd, hdg_cmd, vel_cmd): 控制指令三元组
        """
        # 获取长机和僚机位置
        lead_id = 'A0100'
        if lead_id not in env.agents or not env.agents[lead_id].is_alive:
            return 7, 8, 3  # 长机不存在，保持
        
        lead_pos = env.agents[lead_id].get_position()
        wing_pos = env.agents[agent_id].get_position()
        
        # 计算纵向距离（X方向）
        dx = wing_pos[0] - lead_pos[0]  # 北向为正
        dy = wing_pos[1] - lead_pos[1]  # 东向为正
        
        # 目标：僚机在长机后方
        longitudinal_error = dx + target_distance  # 负值表示在后方
        lateral_error = dy  # 横向偏差
        
        # 速度调整
        vel_cmd = 3
        if abs(longitudinal_error) > 1000:
            if longitudinal_error > 0:
                vel_cmd = 2  # 减速，拉开距离
            else:
                vel_cmd = 4  # 加速，追上
        
        # 航向调整（保持对齐）
        hdg_cmd = 8
        if abs(lateral_error) > 500:
            if lateral_error > 0:
                hdg_cmd = 7  # 左转15°
            else:
                hdg_cmd = 9  # 右转15°
        
        return 7, hdg_cmd, vel_cmd
    
    def _maintain_trail_formation(self, env, agent_id: str, target_distance: float) -> Tuple[int, int, int]:
        """
        保持纵向编队
        
        Args:
            env: 环境对象
            agent_id: 智能体ID（僚机）
            target_distance: 目标间距（米）
        
        Returns:
            (alt_cmd, hdg_cmd, vel_cmd): 控制指令三元组
        """
        # 与建立编队类似，但参数更保守
        return self._establish_trail_formation(env, agent_id, target_distance)
    
    # ==================== 辅助方法 ====================
    
    def _calculate_distance_to_target(self, env, agent_id: str, target_id: str) -> float:
        """
        计算到目标的距离
        
        Args:
            env: 环境对象
            agent_id: 智能体ID
            target_id: 目标ID
        
        Returns:
            距离（米）
        """
        if target_id not in env.agents or not env.agents[target_id].is_alive:
            return float('inf')
        
        my_pos = env.agents[agent_id].get_position()
        target_pos = env.agents[target_id].get_position()
        
        distance = np.linalg.norm(np.array(target_pos) - np.array(my_pos))
        return distance
    
    def _get_phase_by_distance(self, distance: float) -> str:
        """
        根据距离判断战术阶段
        
        Args:
            distance: 距离（米）
        
        Returns:
            阶段名称
        """
        if distance > 120000:
            return 'NLT_MELD'
        elif distance > 100000:
            return 'NLT_MELD'
        elif distance > 80000:
            return 'MELD_MTR'
        elif distance > 78000:
            return 'MTR_LR'
        elif distance > 75000:
            return 'LR_TR'
        elif distance > 70000:
            return 'TR_DOR'
        elif distance > 65000:
            return 'DOR_DR'
        elif distance > 40000:
            return 'DR_MAR'
        else:
            return 'BEYOND_MAR'
    
    def _check_missile_launch(self, agent_id: str):
        """
        检查并标记导弹发射
        
        Args:
            agent_id: 智能体ID
        """
        if agent_id not in self.missile_launched:
            self.missile_launched[agent_id] = False
        
        if not self.missile_launched[agent_id]:
            self.missile_launched[agent_id] = True
            self.logger.info(f"🚀 [{agent_id}] 标记发射导弹")
