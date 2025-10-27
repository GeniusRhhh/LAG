"""
DR节点：Disengagement Range
核心任务：Beam机动转侧对，20s内决策下一阶段战术
"""
from typing import Dict
from .base_node import BaseNode
import numpy as np


class DRNode(BaseNode):
    """DR节点"""
    
    def __init__(self):
        super().__init__('DR')
        self.beam_start_time = None
    
    def execute(self, aircraft_state: Dict, enemy_formation: list,
                decision: Dict, tactical_context: Dict) -> Dict:
        """
        执行DR节点核心任务
        
        核心任务：
        1. 执行Beam机动，转侧对敌方（三九线）
        2. 保持侧对状态，持续约20秒
        3. 在20秒内决策下一阶段战术（已在decision中）
        4. 判断：重新转热发起进攻 or 返航
        
        Args:
            aircraft_state: 飞机状态
            enemy_formation: 敌方编队
            decision: 决策结果（包含should_reengage）
            tactical_context: 战术上下文
        
        Returns:
            机动指令
        """
        current_time = tactical_context.get('current_time', 0.0)
        
        # 记录Beam开始时间
        if self.beam_start_time is None:
            self.beam_start_time = current_time
        
        # 检查是否已经过了20秒
        elapsed_time = current_time - self.beam_start_time
        
        if elapsed_time < 20.0:
            # 前20秒：保持Beam机动
            maneuver = self._beam_maneuver(aircraft_state, enemy_formation)
            maneuver['elapsed_time'] = elapsed_time
            maneuver['waiting_for_decision'] = True
        else:
            # 20秒后：根据决策执行
            should_reengage = decision.get('should_reengage', False)
            
            if should_reengage:
                # 重新转热，发起第二轮进攻
                maneuver = self._reengage_maneuver(aircraft_state, decision, tactical_context)
            else:
                # 返航
                maneuver = self._return_to_base(aircraft_state)
        
        return maneuver
    
    def _beam_maneuver(self, aircraft_state: Dict, enemy_formation: list):
        """
        Beam机动：转侧对敌方
        
        三九线：与敌机航向垂直
        """
        current_pos = aircraft_state['position']
        current_heading = aircraft_state['heading']
        
        if enemy_formation:
            # 计算敌机平均航向
            enemy_headings = [e.get('heading', 0.0) for e in enemy_formation]
            avg_enemy_heading = np.mean(enemy_headings)
            
            # 计算垂直于敌机航向的方向（三九线）
            # 选择更接近当前航向的垂直方向
            perpendicular1 = (avg_enemy_heading + 90.0) % 360.0
            perpendicular2 = (avg_enemy_heading - 90.0) % 360.0
            
            diff1 = abs(perpendicular1 - current_heading)
            diff2 = abs(perpendicular2 - current_heading)
            
            if diff1 < diff2:
                target_heading = perpendicular1
            else:
                target_heading = perpendicular2
        else:
            # 没有敌机，保持当前航向
            target_heading = current_heading
        
        return {
            'type': 'beam',
            'target_position': current_pos,
            'target_heading': target_heading,
            'target_altitude': current_pos[2],
            'description': 'Beam机动（侧对敌方）'
        }
    
    def _reengage_maneuver(self, aircraft_state: Dict, decision: Dict, 
                          tactical_context: Dict):
        """
        重新转热机动
        
        执行第二次战术回转（180° → 0°），发起第二轮进攻
        """
        current_pos = aircraft_state['position']
        current_heading = aircraft_state['heading']
        
        # 180度回转，重新转热
        target_heading = (current_heading + 180.0) % 360.0
        
        # 获取下一战术和角色
        next_tactic = decision.get('next_tactic', 5)
        next_role = decision.get('next_roles', {}).get(
            tactical_context.get('role', 'lead'), 'left'
        )
        
        return {
            'type': 'reengage',
            'target_position': current_pos,
            'target_heading': target_heading,
            'target_altitude': current_pos[2],
            'turn_angle': 180.0,
            'next_tactic': next_tactic,
            'next_role': next_role,
            'description': '重新转热（第二轮进攻）',
            'enter_second_phase': True
        }
    
    def _return_to_base(self, aircraft_state: Dict):
        """返航机动"""
        current_pos = aircraft_state['position']
        current_heading = aircraft_state['heading']
        
        # 假设基地在后方，180度转向
        return_heading = (current_heading + 180.0) % 360.0
        
        return {
            'type': 'return_to_base',
            'target_position': current_pos,
            'target_heading': return_heading,
            'target_altitude': current_pos[2],
            'description': '返航',
            'mission_complete': True
        }
    
    def reset(self):
        """重置DR节点"""
        self.beam_start_time = None
    
    def get_core_task(self) -> str:
        return "Beam转侧对，20s内决策下一阶段战术"
