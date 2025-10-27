"""
MAR节点：Minimum Abort Range
核心任务：执行脱离机动，规避攻击
"""
from typing import Dict
from .base_node import BaseNode


class MARNode(BaseNode):
    """MAR节点"""
    
    def __init__(self):
        super().__init__('MAR')
    
    def execute(self, aircraft_state: Dict, enemy_formation: list,
                decision: Dict, tactical_context: Dict) -> Dict:
        """
        执行MAR节点核心任务
        
        核心任务：
        1. 在MAR前撤离战场
        2. 执行脱离机动
        
        Args:
            aircraft_state: 飞机状态
            enemy_formation: 敌方编队
            decision: 决策结果
            tactical_context: 战术上下文
        
        Returns:
            机动指令
        """
        current_pos = aircraft_state['position']
        current_heading = aircraft_state['heading']
        
        # 执行脱离机动
        # 180度快速回转，加速撤离
        target_heading = (current_heading + 180.0) % 360.0
        
        return {
            'type': 'abort',
            'target_position': current_pos,
            'target_heading': target_heading,
            'target_altitude': current_pos[2],
            'turn_angle': 180.0,
            'accelerate': True,  # 加速撤离
            'description': 'MAR脱离机动',
            'mission_complete': True
        }
    
    def get_core_task(self) -> str:
        return "执行脱离机动，规避攻击"
