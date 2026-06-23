"""
NLT节点：No Later Than
核心任务：从巡逻编队转变为并排射击队形
"""
from typing import Dict
from .base_node import BaseNode
import numpy as np


class NLTNode(BaseNode):
    """NLT节点"""
    
    def __init__(self):
        super().__init__('NLT')
    
    def execute(self, aircraft_state: Dict, enemy_formation: list,
                decision: Dict, tactical_context: Dict) -> Dict:
        """
        执行NLT节点核心任务
        
        核心任务：
        1. 雷达从搜索模式切换为跟踪模式
        2. 从巡逻编队转变为并排射击队形
        
        Args:
            aircraft_state: 飞机状态
            enemy_formation: 敌方编队
            decision: 决策结果
            tactical_context: 战术上下文
        
        Returns:
            机动指令
        """
        # 1. 雷达切换为跟踪模式
        radar_mode = 'track'
        
        # 2. 计算并排射击队形位置
        # 长机在左，僚机在右，间隔2km
        aircraft_role = tactical_context.get('role', 'lead')
        
        if aircraft_role == 'lead':
            # 长机：向左偏移1km
            lateral_offset = -1000.0  # m
        else:
            # 僚机：向右偏移1km
            lateral_offset = 1000.0  # m
        
        # 3. 计算目标位置
        current_pos = aircraft_state['position']
        current_heading = aircraft_state['heading']
        
        # 计算侧向偏移向量
        heading_rad = np.radians(current_heading)
        lateral_vector = np.array([
            -np.sin(heading_rad) * lateral_offset,
            np.cos(heading_rad) * lateral_offset,
            0.0
        ])
        
        target_pos = current_pos + lateral_vector
        
        # 4. 生成机动指令
        maneuver = {
            'type': 'formation_adjustment',
            'target_position': target_pos,
            'target_heading': current_heading,
            'target_altitude': current_pos[2],
            'radar_mode': radar_mode,
            'description': '转变为并排射击队形'
        }
        
        return maneuver
    
    def get_core_task(self) -> str:
        return "从巡逻编队转变为并排射击队形，雷达切换为跟踪模式"
