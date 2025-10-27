"""
MTR节点：Missile Target Range
核心任务：前往攻击占位点
"""
from typing import Dict
from .base_node import BaseNode
import numpy as np
from utils.geometry import calculate_distance


class MTRNode(BaseNode):
    """MTR节点"""
    
    def __init__(self):
        super().__init__('MTR')
    
    def execute(self, aircraft_state: Dict, enemy_formation: list,
                decision: Dict, tactical_context: Dict) -> Dict:
        """
        执行MTR节点核心任务
        
        核心任务：
        1. 判断是否继续进攻（已在decision中）
        2. 如果继续：前往攻击占位点
        3. 如果撤退：执行返航机动
        
        Args:
            aircraft_state: 飞机状态
            enemy_formation: 敌方编队
            decision: 决策结果
            tactical_context: 战术上下文
        
        Returns:
            机动指令
        """
        action = decision.get('action', 'continue')
        
        if action == 'retreat':
            # 撤退：返航
            return self._retreat_maneuver(aircraft_state)
        
        # 继续进攻：计算攻击占位点
        assigned_target_idx = tactical_context.get('assigned_target_idx', 0)
        if assigned_target_idx < len(enemy_formation):
            target_state = enemy_formation[assigned_target_idx]
        else:
            target_state = enemy_formation[0] if enemy_formation else None
        
        if target_state is None:
            return self._retreat_maneuver(aircraft_state)
        
        # 计算攻击占位点
        attack_position = self._calculate_attack_position(
            aircraft_state, target_state, tactical_context
        )
        
        return {
            'type': 'move_to_attack_position',
            'target_position': attack_position,
            'target_heading': self._calculate_heading_to_target(
                attack_position, target_state['position']
            ),
            'target_altitude': attack_position[2],
            'description': '前往攻击占位点'
        }
    
    def _calculate_attack_position(self, aircraft_state: Dict, 
                                   target_state: Dict, tactical_context: Dict):
        """
        计算攻击占位点
        
        攻击占位点：在目标前方，距离约78km（LR距离）
        """
        target_pos = target_state['position']
        target_vel = target_state['velocity']
        
        # 预测目标未来位置（10秒后）
        predicted_target_pos = target_pos + target_vel * 10.0
        
        # 计算从目标指向我机的方向
        my_pos = aircraft_state['position']
        direction = my_pos - predicted_target_pos
        direction_norm = np.linalg.norm(direction[:2])
        
        if direction_norm < 1e-6:
            direction_unit = np.array([1.0, 0.0, 0.0])
        else:
            direction_unit = direction / direction_norm
            direction_unit = np.append(direction_unit[:2], 0.0)
        
        # 攻击占位点：在目标前方78km
        attack_distance = 78000.0  # m
        attack_position = predicted_target_pos + direction_unit * attack_distance
        
        # 保持高度
        attack_position[2] = my_pos[2]
        
        return attack_position
    
    def _calculate_heading_to_target(self, from_pos, to_pos):
        """计算指向目标的航向"""
        direction = to_pos[:2] - from_pos[:2]
        heading = np.arctan2(direction[1], direction[0]) * 180 / np.pi
        return heading
    
    def _retreat_maneuver(self, aircraft_state: Dict):
        """撤退机动"""
        current_pos = aircraft_state['position']
        current_heading = aircraft_state['heading']
        
        # 180度转向，返航
        retreat_heading = (current_heading + 180.0) % 360.0
        
        return {
            'type': 'retreat',
            'target_position': current_pos,
            'target_heading': retreat_heading,
            'target_altitude': current_pos[2],
            'description': '撤退返航'
        }
    
    def get_core_task(self) -> str:
        return "前往攻击占位点"
