"""
DOR节点：Decision on Re-attack
核心任务：规避敌导弹 + 预决策下一战术
"""
from typing import Dict
from .base_node import BaseNode
import numpy as np


class DORNode(BaseNode):
    """DOR节点"""
    
    def __init__(self):
        super().__init__('DOR')
    
    def execute(self, aircraft_state: Dict, enemy_formation: list,
                decision: Dict, tactical_context: Dict) -> Dict:
        """
        执行DOR节点核心任务
        
        核心任务：
        1. 根据威胁值选择规避机动
        2. 预决策下一阶段战术（已在decision中）
        3. 调整规避机动参数，使其有助于下一战术队形形成
        
        Args:
            aircraft_state: 飞机状态
            enemy_formation: 敌方编队
            decision: 决策结果（包含next_tactic和next_roles）
            tactical_context: 战术上下文
        
        Returns:
            机动指令
        """
        threat = decision.get('threat', {})
        threat_level = threat.get('total', 0.5)
        
        # 1. 根据威胁值选择规避机动
        if threat_level > 0.8:
            # 高威胁：Short Skate
            maneuver = self._short_skate_maneuver(aircraft_state)
        elif threat_level > 0.5:
            # 中威胁：Notch Back
            maneuver = self._notch_back_maneuver(aircraft_state)
        else:
            # 低威胁：Beam机动
            maneuver = self._beam_maneuver(aircraft_state)
        
        # 2. 调整规避参数以适应下一战术
        next_tactic = decision.get('next_tactic', 5)
        next_role = decision.get('next_roles', {}).get(
            tactical_context.get('role', 'lead'), 'left'
        )
        
        maneuver = self._adjust_for_next_formation(
            maneuver, next_tactic, next_role, aircraft_state
        )
        
        maneuver['next_tactic'] = next_tactic
        maneuver['next_role'] = next_role
        
        return maneuver
    
    def _short_skate_maneuver(self, aircraft_state: Dict):
        """Short Skate机动：180度快速回转"""
        current_pos = aircraft_state['position']
        current_heading = aircraft_state['heading']
        
        target_heading = (current_heading + 180.0) % 360.0
        
        return {
            'type': 'short_skate',
            'target_position': current_pos,
            'target_heading': target_heading,
            'target_altitude': current_pos[2],
            'turn_angle': 180.0,
            'description': 'Short Skate规避'
        }
    
    def _notch_back_maneuver(self, aircraft_state: Dict):
        """Notch Back机动：180度回转+下降"""
        current_pos = aircraft_state['position']
        current_heading = aircraft_state['heading']
        
        target_heading = (current_heading + 180.0) % 360.0
        target_altitude = max(current_pos[2] - 500.0, 1000.0)
        
        return {
            'type': 'notch_back',
            'target_position': current_pos,
            'target_heading': target_heading,
            'target_altitude': target_altitude,
            'turn_angle': 180.0,
            'description': 'Notch Back规避'
        }
    
    def _beam_maneuver(self, aircraft_state: Dict):
        """Beam机动：90度侧对"""
        current_pos = aircraft_state['position']
        current_heading = aircraft_state['heading']
        
        target_heading = (current_heading + 90.0) % 360.0
        
        return {
            'type': 'beam',
            'target_position': current_pos,
            'target_heading': target_heading,
            'target_altitude': current_pos[2],
            'turn_angle': 90.0,
            'description': 'Beam机动规避'
        }
    
    def _adjust_for_next_formation(self, maneuver: Dict, next_tactic: int,
                                   next_role: str, aircraft_state: Dict):
        """
        调整规避机动参数，使其有助于下一战术队形形成
        
        Args:
            maneuver: 原始机动指令
            next_tactic: 下一战术编号
            next_role: 下一战术中的角色
            aircraft_state: 飞机状态
        
        Returns:
            调整后的机动指令
        """
        current_heading = aircraft_state['heading']
        
        # 根据下一战术调整转向方向
        if next_tactic == 2:  # 钳形攻势
            if next_role == 'left':
                # 下一战术需要向左，调整转向方向
                adjustment = -10.0
            else:
                # 下一战术需要向右
                adjustment = 10.0
        elif next_tactic == 3:  # 上下夹击
            if next_role == 'high':
                # 下一战术需要爬升
                maneuver['target_altitude'] = aircraft_state['position'][2] + 500.0
                adjustment = 0.0
            else:
                adjustment = 0.0
        else:
            # 其他战术：不调整
            adjustment = 0.0
        
        # 应用调整
        if adjustment != 0.0:
            maneuver['target_heading'] = (maneuver['target_heading'] + adjustment) % 360.0
            maneuver['description'] += f' (调整{adjustment:.1f}度以适应下一战术)'
        
        return maneuver
    
    def get_core_task(self) -> str:
        return "规避敌导弹，预决策下一战术"
