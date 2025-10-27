"""
战术回转
180度回转机动，用于脱离和重新进攻
"""
from typing import Dict, List
from .base_tactic import BaseTactic


class TacticalTurn(BaseTactic):
    """战术回转"""
    
    def __init__(self):
        super().__init__(7, '战术回转')
        self.turn_phase = None  # 'first_turn' or 'second_turn'
        self.turn_start_time = None
        self.turn_complete = False
    
    def get_formation(self, aircraft_role: str) -> Dict:
        """
        战术回转没有固定队形
        """
        return {
            'longitudinal_offset': 0.0,
            'lateral_offset': 0.0,
            'altitude_offset': 0.0,
            'description': '战术回转'
        }
    
    def execute_first_turn(self, aircraft_state: Dict) -> Dict:
        """
        第一次回转：0° → 180°
        
        目的：脱离敌导弹威胁
        """
        self.turn_phase = 'first_turn'
        self.turn_complete = False
        
        current_heading = aircraft_state['heading']
        target_heading = (current_heading + 180.0) % 360.0
        
        return {
            'type': 'tactical_turn_first',
            'target_heading': target_heading,
            'turn_angle': 180.0,
            'description': '第一次战术回转（脱离）'
        }
    
    def execute_second_turn(self, aircraft_state: Dict) -> Dict:
        """
        第二次回转：180° → 0°
        
        目的：重新转热，发起第二轮进攻
        """
        self.turn_phase = 'second_turn'
        self.turn_complete = False
        
        current_heading = aircraft_state['heading']
        target_heading = (current_heading + 180.0) % 360.0
        
        return {
            'type': 'tactical_turn_second',
            'target_heading': target_heading,
            'turn_angle': 180.0,
            'description': '第二次战术回转（重新进攻）'
        }
    
    def check_special_cases(self, aircraft_state: Dict, enemy_formation: List,
                           control_range: str, current_time: float,
                           missiles_remaining: int) -> Dict:
        """
        检查战术回转的7种特殊情况
        
        Returns:
            特殊情况处理结果
        """
        result = {'has_special_case': False, 'action': None}
        
        # 1. 回转未完成就触发DR → 直接撤离
        if self.turn_phase == 'first_turn' and not self.turn_complete and control_range == 'DR':
            result['has_special_case'] = True
            result['action'] = 'abort_mission'
            result['description'] = '回转未完成触发DR，直接撤离'
            return result
        
        # 2. 到达MAR → 加速撤离
        if control_range == 'MAR':
            result['has_special_case'] = True
            result['action'] = 'accelerate_retreat'
            result['description'] = '到达MAR，加速撤离'
            return result
        
        # 3. 回转后敌机脱离 → 加力开始第二战术
        if self.turn_phase == 'first_turn' and self.turn_complete:
            if self._check_enemy_disengaged(enemy_formation):
                result['has_special_case'] = True
                result['action'] = 'pursue_enemy'
                result['description'] = '敌机脱离，加力追击'
                return result
        
        # 4. 回转后追击成功 → 不用管
        # （由其他系统处理）
        
        # 5. 长时间未进入下一距离 → 战术回转，结束作战
        if self.turn_start_time and current_time - self.turn_start_time > 30.0:
            result['has_special_case'] = True
            result['action'] = 'end_mission'
            result['description'] = '长时间未进入下一距离，结束作战'
            return result
        
        # 6. 导弹打光 → 交战结束
        if missiles_remaining == 0:
            result['has_special_case'] = True
            result['action'] = 'return_to_base'
            result['description'] = '导弹打光，返航'
            return result
        
        # 7. 敌机被击落 → 任务完成
        if not enemy_formation or all(e.get('destroyed', False) for e in enemy_formation):
            result['has_special_case'] = True
            result['action'] = 'mission_complete'
            result['description'] = '敌机被击落，任务完成'
            return result
        
        return result
    
    def _check_enemy_disengaged(self, enemy_formation: List) -> bool:
        """检查敌机是否脱离"""
        if not enemy_formation:
            return True
        
        # 简化判断：检查敌机是否远离（距离>150km）
        for enemy in enemy_formation:
            if enemy.get('distance', 0) < 150.0:
                return False
        
        return True
    
    def mark_turn_complete(self):
        """标记回转完成"""
        self.turn_complete = True
    
    def start_turn(self, current_time: float):
        """开始回转"""
        self.turn_start_time = current_time
    
    def get_node_decision(self, node_name: str, aircraft_role: str,
                         aircraft_state: Dict, enemy_formation: List,
                         decision: Dict) -> Dict:
        """
        战术回转主要在TR和DR节点执行
        """
        base_node = node_name.rstrip("12'")
        
        adjustments = {}
        
        if base_node == 'TR':
            # TR节点：可能执行第一次回转
            if decision.get('action') == 'retreat':
                adjustments['execute_first_turn'] = True
        
        elif base_node == 'DR':
            # DR节点：可能执行第二次回转
            if decision.get('should_reengage', False):
                adjustments['execute_second_turn'] = True
        
        return adjustments
    
    def get_description(self) -> str:
        return "战术回转：180度回转机动，用于脱离和重新进攻"
