"""
并排射击战术
双机并排保持同步接敌、发射
"""
from typing import Dict, List
import logging
from core import TacticalDecisionManager
from .base_tactic import BaseTactic


class SideBySideDecisionManager(TacticalDecisionManager):
    """强制并排射击战术的决策管理器"""
    
    def _execute_meld_decision(self, aircraft_id: str, my_state: dict,
                               enemy_formation: list) -> dict:
        """MELD节点：强制选择并排射击战术"""
        self.current_tactic = 5
        self.current_roles = {'lead': 'left', 'wingman': 'right'}
        
        logging.info(f"[MELD] 强制选择战术: 并排射击")
        
        self.lead_decision.set_assigned_target(0)
        self.wingman_decision.set_assigned_target(1)
        
        threats = self._calculate_formation_threats(my_state, enemy_formation)
        return {
            'node': 'MELD',
            'action': 'continue',
            'tactic': 5,
            'role': self.current_roles.get('lead' if aircraft_id.endswith('100') else 'wingman'),
            'target': 0 if aircraft_id.endswith('100') else 1,
            'threat': threats.get(aircraft_id, 0.5)
        }


# 保留原有的战术类
class SideBySideTactic(BaseTactic):
    """并排射击战术"""
    
    def __init__(self):
        super().__init__(5, '并排射击')
        self.lateral_separation = 1000.0  # 横向间隔 (m)
    
    def get_formation(self, aircraft_role: str) -> Dict:
        """
        获取战术队形
        
        左机和右机并排，横向间隔2km
        """
        tactical_role = self.roles.get(aircraft_role, 'left')
        
        if tactical_role == 'left':
            # 左机：向左偏移
            lateral_offset = -self.lateral_separation
            description = '左机（并排）'
        else:
            # 右机：向右偏移
            lateral_offset = self.lateral_separation
            description = '右机（并排）'
        
        return {
            'longitudinal_offset': 0.0,
            'lateral_offset': lateral_offset,
            'altitude_offset': 0.0,
            'description': description
        }
    
    def get_node_decision(self, node_name: str, aircraft_role: str,
                         aircraft_state: Dict, enemy_formation: List,
                         decision: Dict) -> Dict:
        """
        获取各节点的战术特定决策
        
        并排射击的特点：
        - 双机同步接敌
        - 同时发射导弹
        - 保持横向间隔
        - 时间窗口一致（MTR1=MTR2, LR1=LR2）
        """
        base_node = node_name.rstrip("12'")
        
        adjustments = {}
        
        if base_node == 'MTR':
            # MTR节点：同步到达
            adjustments['synchronized_arrival'] = True
            adjustments['maintain_lateral_separation'] = True
        
        elif base_node == 'LR':
            # LR节点：同时发射
            adjustments['synchronized_launch'] = True
            adjustments['missile_count'] = 2
            adjustments['launch_timing'] = 'immediate'
        
        elif base_node == 'TR':
            # TR节点：同步规避
            adjustments['synchronized_evasion'] = True
        
        elif base_node == 'DOR':
            # DOR节点：保持横向间隔
            adjustments['maintain_lateral_separation'] = True
        
        elif base_node == 'DR':
            # DR节点：同步决策
            adjustments['synchronized_decision'] = True
        
        return adjustments
    
    def get_description(self) -> str:
        return "并排射击：双机并排同步接敌、发射，保持横向间隔"
