"""
MELD节点：Merge Entry Launch Decision
核心任务：战术决策 + 目标分配 + 队形调整
"""
from typing import Dict
from .base_node import BaseNode
import numpy as np


class MELDNode(BaseNode):
    """MELD节点"""
    
    def __init__(self):
        super().__init__('MELD')
    
    def execute(self, aircraft_state: Dict, enemy_formation: list,
                decision: Dict, tactical_context: Dict) -> Dict:
        """
        执行MELD节点核心任务
        
        核心任务：
        1. 战术决策（已在decision_manager中完成）
        2. 目标分配（已在decision_manager中完成）
        3. 根据选定战术调整队形
        
        Args:
            aircraft_state: 飞机状态
            enemy_formation: 敌方编队
            decision: 决策结果（包含战术和角色）
            tactical_context: 战术上下文
        
        Returns:
            机动指令
        """
        tactic = decision.get('tactic', 5)  # 默认并排射击
        role = decision.get('roles', {}).get(tactical_context.get('role', 'lead'), 'left')
        
        current_pos = aircraft_state['position']
        current_heading = aircraft_state['heading']
        
        # 根据战术和角色调整队形
        if tactic == 1:  # 拖曳射击
            maneuver = self._drag_shoot_formation(current_pos, current_heading, role)
        elif tactic == 2:  # 钳形攻势
            maneuver = self._pincer_formation(current_pos, current_heading, role)
        elif tactic == 3:  # 上下夹击
            maneuver = self._high_low_formation(current_pos, current_heading, role)
        elif tactic == 4:  # 前后攻击
            maneuver = self._sequential_formation(current_pos, current_heading, role)
        else:  # 并排射击
            maneuver = self._side_by_side_formation(current_pos, current_heading, role)
        
        maneuver['tactic'] = tactic
        maneuver['role'] = role
        
        return maneuver
    
    def _drag_shoot_formation(self, pos, heading, role):
        """拖曳射击队形"""
        if role == 'drag':
            # 拖曳机：前出2km
            offset = np.array([2000.0, 0.0, 0.0])
        else:
            # 射击机：后退2km
            offset = np.array([-2000.0, 0.0, 0.0])
        
        target_pos = pos + self._rotate_vector(offset, heading)
        
        return {
            'type': 'tactical_formation',
            'target_position': target_pos,
            'target_heading': heading,
            'target_altitude': pos[2],
            'description': f'拖曳射击队形 - {role}'
        }
    
    def _pincer_formation(self, pos, heading, role):
        """钳形攻势队形"""
        if role == 'left':
            # 左机：向左偏45度
            heading_offset = -45.0
        else:
            # 右机：向右偏45度
            heading_offset = 45.0
        
        target_heading = heading + heading_offset
        
        return {
            'type': 'tactical_formation',
            'target_position': pos,
            'target_heading': target_heading,
            'target_altitude': pos[2],
            'description': f'钳形攻势队形 - {role}'
        }
    
    def _high_low_formation(self, pos, heading, role):
        """上下夹击队形"""
        if role == 'high':
            # 高机：爬升1000m
            altitude_offset = 1000.0
        else:
            # 低机：保持高度
            altitude_offset = 0.0
        
        target_altitude = pos[2] + altitude_offset
        
        return {
            'type': 'tactical_formation',
            'target_position': np.array([pos[0], pos[1], target_altitude]),
            'target_heading': heading,
            'target_altitude': target_altitude,
            'description': f'上下夹击队形 - {role}'
        }
    
    def _sequential_formation(self, pos, heading, role):
        """前后攻击队形"""
        if role == 'front':
            # 前机：前出3km
            offset = np.array([3000.0, 0.0, 0.0])
        else:
            # 后机：后退3km
            offset = np.array([-3000.0, 0.0, 0.0])
        
        target_pos = pos + self._rotate_vector(offset, heading)
        
        return {
            'type': 'tactical_formation',
            'target_position': target_pos,
            'target_heading': heading,
            'target_altitude': pos[2],
            'description': f'前后攻击队形 - {role}'
        }
    
    def _side_by_side_formation(self, pos, heading, role):
        """并排射击队形"""
        if role == 'left':
            # 左机：向左偏移1km
            lateral_offset = -1000.0
        else:
            # 右机：向右偏移1km
            lateral_offset = 1000.0
        
        heading_rad = np.radians(heading)
        lateral_vector = np.array([
            -np.sin(heading_rad) * lateral_offset,
            np.cos(heading_rad) * lateral_offset,
            0.0
        ])
        
        target_pos = pos + lateral_vector
        
        return {
            'type': 'tactical_formation',
            'target_position': target_pos,
            'target_heading': heading,
            'target_altitude': pos[2],
            'description': f'并排射击队形 - {role}'
        }
    
    def _rotate_vector(self, vector, heading):
        """根据航向旋转向量"""
        heading_rad = np.radians(heading)
        rotation_matrix = np.array([
            [np.cos(heading_rad), -np.sin(heading_rad), 0],
            [np.sin(heading_rad), np.cos(heading_rad), 0],
            [0, 0, 1]
        ])
        return rotation_matrix @ vector
    
    def get_core_task(self) -> str:
        return "战术决策、目标分配、队形调整"
