"""
LR节点：Launch Range
核心任务：发射导弹 + 完成中制导 + 微偏置
"""
from typing import Dict
from .base_node import BaseNode
import numpy as np
from utils.constants import CRANK_ADJUSTMENT_RANGE, RADAR_EFFECTIVE_ANGLE


class LRNode(BaseNode):
    """LR节点"""
    
    def __init__(self):
        super().__init__('LR')
        self.missile_launched = False
    
    def execute(self, aircraft_state: Dict, enemy_formation: list,
                decision: Dict, tactical_context: Dict) -> Dict:
        """
        执行LR节点核心任务
        
        核心任务：
        1. 发射导弹（对分配的敌机）
        2. 判断是否继续中制导
        3. 如果继续：执行Crank机动或平飞，保持雷达照射
        4. 中制导微偏置：如果雷达照射角度不够，执行小角度调整
        
        Args:
            aircraft_state: 飞机状态
            enemy_formation: 敌方编队
            decision: 决策结果
            tactical_context: 战术上下文
        
        Returns:
            机动指令
        """
        action = decision.get('action', 'continue')
        
        # 1. 发射导弹
        if not self.missile_launched:
            self.missile_launched = True
            launch_command = {
                'launch_missile': True,
                'target_idx': tactical_context.get('assigned_target_idx', 0)
            }
        else:
            launch_command = {'launch_missile': False}
        
        # 2. 判断行动
        if action == 'retreat':
            # 撤退：终止中制导
            maneuver = self._retreat_maneuver(aircraft_state)
            maneuver.update(launch_command)
            return maneuver
        
        # 3. 继续中制导
        assigned_target_idx = tactical_context.get('assigned_target_idx', 0)
        if assigned_target_idx < len(enemy_formation):
            target_state = enemy_formation[assigned_target_idx]
        else:
            target_state = enemy_formation[0] if enemy_formation else None
        
        if target_state is None:
            maneuver = self._retreat_maneuver(aircraft_state)
            maneuver.update(launch_command)
            return maneuver
        
        # 4. 检查雷达照射角度
        radar_angle = self._calculate_radar_angle(aircraft_state, target_state)
        
        if abs(radar_angle) > RADAR_EFFECTIVE_ANGLE:
            # 雷达照射角度不够，需要微偏置
            maneuver = self._crank_adjustment(aircraft_state, target_state, radar_angle)
        elif action == 'evasion':
            # 规避但保持雷达照射
            maneuver = self._evasion_with_radar(aircraft_state, target_state)
        else:
            # 正常中制导：Crank机动或平飞
            maneuver = self._midcourse_guidance(aircraft_state, target_state)
        
        maneuver.update(launch_command)
        return maneuver
    
    def _calculate_radar_angle(self, aircraft_state: Dict, target_state: Dict):
        """
        计算雷达照射角度
        
        Returns:
            雷达照射角度（度），0度表示正对目标
        """
        my_pos = aircraft_state['position']
        my_heading = aircraft_state['heading']
        target_pos = target_state['position']
        
        # 计算目标方向
        direction = target_pos[:2] - my_pos[:2]
        target_angle = np.arctan2(direction[1], direction[0]) * 180 / np.pi
        
        # 计算雷达照射角度（我机航向与目标方向的夹角）
        radar_angle = target_angle - my_heading
        
        # 归一化到[-180, 180]
        while radar_angle > 180:
            radar_angle -= 360
        while radar_angle < -180:
            radar_angle += 360
        
        return radar_angle
    
    def _crank_adjustment(self, aircraft_state: Dict, target_state: Dict, radar_angle: float):
        """
        中制导微偏置
        
        如果雷达照射角度不够，执行小角度Crank调整（Δψ∈[-15°,+15°]）
        """
        current_heading = aircraft_state['heading']
        
        # 计算调整角度
        if radar_angle > 0:
            # 目标在右侧，向右偏
            adjustment = min(radar_angle - RADAR_EFFECTIVE_ANGLE, CRANK_ADJUSTMENT_RANGE[1])
        else:
            # 目标在左侧，向左偏
            adjustment = max(radar_angle + RADAR_EFFECTIVE_ANGLE, CRANK_ADJUSTMENT_RANGE[0])
        
        target_heading = current_heading + adjustment
        
        return {
            'type': 'crank_adjustment',
            'target_position': aircraft_state['position'],
            'target_heading': target_heading,
            'target_altitude': aircraft_state['position'][2],
            'adjustment_angle': adjustment,
            'description': f'中制导微偏置 (调整{adjustment:.1f}度)'
        }
    
    def _evasion_with_radar(self, aircraft_state: Dict, target_state: Dict):
        """规避但保持雷达照射"""
        current_heading = aircraft_state['heading']
        
        # 执行Crank机动（偏15度）
        crank_angle = 15.0
        target_heading = current_heading + crank_angle
        
        return {
            'type': 'evasion_crank',
            'target_position': aircraft_state['position'],
            'target_heading': target_heading,
            'target_altitude': aircraft_state['position'][2],
            'description': '规避机动（保持雷达照射）'
        }
    
    def _midcourse_guidance(self, aircraft_state: Dict, target_state: Dict):
        """正常中制导：Crank机动或平飞"""
        current_pos = aircraft_state['position']
        current_heading = aircraft_state['heading']
        
        # 计算到目标的距离
        target_pos = target_state['position']
        distance = np.linalg.norm(target_pos[:2] - current_pos[:2]) / 1000.0
        
        if distance > 76.0:
            # 距离较远，执行Crank机动（偏10度）
            crank_angle = 10.0
            target_heading = current_heading + crank_angle
            maneuver_type = 'crank'
            description = 'Crank机动（中制导）'
        else:
            # 距离较近，平飞保持雷达照射
            target_heading = current_heading
            maneuver_type = 'maintain_heading'
            description = '平飞（中制导）'
        
        return {
            'type': maneuver_type,
            'target_position': current_pos,
            'target_heading': target_heading,
            'target_altitude': current_pos[2],
            'description': description
        }
    
    def _retreat_maneuver(self, aircraft_state: Dict):
        """撤退机动"""
        current_pos = aircraft_state['position']
        current_heading = aircraft_state['heading']
        
        retreat_heading = (current_heading + 180.0) % 360.0
        
        return {
            'type': 'retreat',
            'target_position': current_pos,
            'target_heading': retreat_heading,
            'target_altitude': current_pos[2],
            'description': '撤退（终止中制导）'
        }
    
    def get_core_task(self) -> str:
        return "发射导弹、完成中制导、微偏置调整"
