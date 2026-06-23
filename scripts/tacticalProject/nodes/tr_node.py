"""
TR节点：Turn Range
核心任务：中制导结束，规避敌方攻击
"""
from typing import Dict
from .base_node import BaseNode
import numpy as np


class TRNode(BaseNode):
    """TR节点"""
    
    def __init__(self):
        super().__init__('TR')
    
    def execute(self, aircraft_state: Dict, enemy_formation: list,
                decision: Dict, tactical_context: Dict) -> Dict:
        """
        执行TR节点核心任务
        
        核心任务：
        1. 导弹转为主动导引，中制导结束
        2. 判断是否脱离
        3. 如果脱离：执行脱离机动（Short Skate/Notch Back）
        4. 如果继续：执行规避机动
        
        Args:
            aircraft_state: 飞机状态
            enemy_formation: 敌方编队
            decision: 决策结果
            tactical_context: 战术上下文
        
        Returns:
            机动指令
        """
        action = decision.get('action', 'continue')
        threat = decision.get('threat', {})
        
        # 中制导结束
        midcourse_end = {'midcourse_guidance': False}
        
        if action == 'retreat':
            # 脱离：执行脱离机动
            maneuver = self._disengage_maneuver(aircraft_state, threat)
        elif action == 'evasion':
            # 规避：执行规避机动
            maneuver = self._evasion_maneuver(aircraft_state, threat)
        else:
            # 继续：执行规避机动（TR节点核心任务就是规避）
            maneuver = self._evasion_maneuver(aircraft_state, threat)
        
        maneuver.update(midcourse_end)
        return maneuver
    
    def _disengage_maneuver(self, aircraft_state: Dict, threat: Dict):
        """
        脱离机动
        
        在DOR前脱离，可在DR节点重新转热
        """
        current_pos = aircraft_state['position']
        current_heading = aircraft_state['heading']
        threat_level = threat.get('total', 0.5)
        
        if threat_level > 0.8:
            # 高威胁：Short Skate（180度快速回转）
            maneuver_type = 'short_skate'
            turn_angle = 180.0
            description = 'Short Skate脱离'
        else:
            # 中威胁：Notch Back（180度回转+下降）
            maneuver_type = 'notch_back'
            turn_angle = 180.0
            description = 'Notch Back脱离'
        
        target_heading = (current_heading + turn_angle) % 360.0
        
        # 如果是Notch Back，下降500m
        if maneuver_type == 'notch_back':
            target_altitude = max(current_pos[2] - 500.0, 1000.0)
        else:
            target_altitude = current_pos[2]
        
        return {
            'type': maneuver_type,
            'target_position': current_pos,
            'target_heading': target_heading,
            'target_altitude': target_altitude,
            'turn_angle': turn_angle,
            'description': description,
            'can_reengage': True  # 标记可以在DR节点重新转热
        }
    
    def _evasion_maneuver(self, aircraft_state: Dict, threat: Dict):
        """
        规避机动
        
        继续进攻，但执行规避动作
        """
        current_pos = aircraft_state['position']
        current_heading = aircraft_state['heading']
        threat_level = threat.get('total', 0.5)
        
        if threat_level > 0.6:
            # 较高威胁：Beam机动（90度侧对）
            turn_angle = 90.0
            description = 'Beam机动（规避）'
        else:
            # 较低威胁：小角度规避（30度）
            turn_angle = 30.0
            description = '小角度规避'
        
        target_heading = current_heading + turn_angle
        
        return {
            'type': 'evasion',
            'target_position': current_pos,
            'target_heading': target_heading,
            'target_altitude': current_pos[2],
            'turn_angle': turn_angle,
            'description': description
        }
    
    def get_core_task(self) -> str:
        return "中制导结束，规避敌方攻击"
