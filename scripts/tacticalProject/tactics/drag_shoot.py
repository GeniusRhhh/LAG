"""
拖曳射击战术
拖曳机在前吸引敌机火力，射击机在后承担主要射击任务
"""
from typing import Dict, List
import logging
from core import TacticalDecisionManager
from .base_tactic import BaseTactic


class DragShootDecisionManager(TacticalDecisionManager):
    """强制拖曳射击战术的决策管理器"""
    
    def _execute_meld_decision(self, aircraft_id: str, my_state: dict,
                               enemy_formation: list) -> dict:
        """MELD节点：强制选择拖曳射击战术"""
        self.current_tactic = 1
        self.current_roles = {'lead': 'drag', 'wingman': 'shooter'}
        
        logging.info(f"[MELD] 强制选择战术: 拖曳射击")
        
        self.lead_decision.set_assigned_target(0)
        self.wingman_decision.set_assigned_target(1)
        
        # 返回基本决策
        threats = self._calculate_formation_threats(my_state, enemy_formation)
        return {
            'node': 'MELD',
            'action': 'continue',
            'tactic': 1,
            'role': self.current_roles.get('lead' if aircraft_id.endswith('100') else 'wingman'),
            'target': 0 if aircraft_id.endswith('100') else 1,
            'threat': threats.get(aircraft_id, 0.5)
        }


# 保留原有的战术类以兼容
class DragShootTactic(BaseTactic):
    """拖曳射击战术"""
    
    def __init__(self):
        super().__init__(1, '拖曳射击')
        self.drag_offset = 2000.0  # 拖曳机前出距离 (m)
        self.shooter_offset = -2000.0  # 射击机后退距离 (m)
    
    def get_formation(self, aircraft_role: str) -> Dict:
        """
        获取战术队形
        
        拖曳机在前，射击机在后，纵向间隔4km
        """
        tactical_role = self.roles.get(aircraft_role, 'shooter')
        
        if tactical_role == 'drag':
            # 拖曳机：前出
            offset = self.drag_offset
            description = '拖曳机（前出）'
        else:
            # 射击机：后退
            offset = self.shooter_offset
            description = '射击机（后退）'
        
        return {
            'longitudinal_offset': offset,
            'lateral_offset': 0.0,
            'altitude_offset': 0.0,
            'description': description
        }
    
    def get_node_decision(self, node_name: str, aircraft_role: str,
                         aircraft_state: Dict, enemy_formation: List,
                         decision: Dict) -> Dict:
        """
        获取各节点的战术特定决策
        
        拖曳射击的特点：
        - 拖曳机承受更高威胁，但不轻易撤退
        - 射击机在后掩护，承担主要射击任务
        """
        tactical_role = self.roles.get(aircraft_role, 'shooter')
        base_node = node_name.rstrip("12'")
        
        adjustments = {}
        
        if base_node == 'MTR':
            # MTR节点：拖曳机前出，射击机保持距离
            if tactical_role == 'drag':
                adjustments['priority'] = 'attract_fire'
                adjustments['aggressive'] = True
            else:
                adjustments['priority'] = 'prepare_attack'
                adjustments['aggressive'] = False
        
        elif base_node == 'LR':
            # LR节点：射击机承担主要射击任务
            if tactical_role == 'shooter':
                adjustments['fire_priority'] = 'high'
                adjustments['missile_count'] = 2  # 发射2枚导弹
            else:
                adjustments['fire_priority'] = 'low'
                adjustments['missile_count'] = 1  # 发射1枚导弹
        
        elif base_node == 'TR':
            # TR节点：拖曳机可能需要脱离，射击机继续
            if tactical_role == 'drag' and decision.get('action') == 'retreat':
                adjustments['allow_retreat'] = True
            else:
                adjustments['continue_mission'] = True
        
        elif base_node == 'DOR':
            # DOR节点：根据角色调整规避强度
            if tactical_role == 'drag':
                adjustments['evasion_intensity'] = 'high'
            else:
                adjustments['evasion_intensity'] = 'medium'
        
        return adjustments
    
    def get_description(self) -> str:
        return "拖曳射击：拖曳机前出吸引火力，射击机在后承担主要射击任务"
