"""
钳形攻势战术
我方两架战机呈左右包夹态势对敌机射击
"""
from typing import Dict, List
import logging
from core import TacticalDecisionManager
from .base_tactic import BaseTactic


class PincerAttackDecisionManager(TacticalDecisionManager):
    """强制钳形攻势战术的决策管理器"""
    
    def _execute_meld_decision(self, aircraft_id: str, my_state: dict,
                               enemy_formation: list) -> dict:
        """MELD节点：强制选择钳形攻势战术"""
        self.current_tactic = 2
        self.current_roles = {'lead': 'left', 'wingman': 'right'}
        
        logging.info(f"[MELD] 强制选择战术: 钳形攻势")
        
        self.lead_decision.set_assigned_target(0)
        self.wingman_decision.set_assigned_target(1)
        
        threats = self._calculate_formation_threats(my_state, enemy_formation)
        return {
            'node': 'MELD',
            'action': 'continue',
            'tactic': 2,
            'role': self.current_roles.get('lead' if aircraft_id.endswith('100') else 'wingman'),
            'target': 0 if aircraft_id.endswith('100') else 1,
            'threat': threats.get(aircraft_id, 0.5)
        }


# 保留原有的战术类
class PincerAttackTactic(BaseTactic):
    """钳形攻势战术"""
    
    def __init__(self):
        super().__init__(2, '钳形攻势')
        self.pincer_angle = 45.0  # 钳形角度 (度)
    
    def get_formation(self, aircraft_role: str) -> Dict:
        """
        获取战术队形
        
        左机向左偏45度，右机向右偏45度，形成钳形包夹
        """
        tactical_role = self.roles.get(aircraft_role, 'left')
        
        if tactical_role == 'left':
            # 左机：向左偏
            heading_offset = -self.pincer_angle
            description = '左翼（钳形）'
        else:
            # 右机：向右偏
            heading_offset = self.pincer_angle
            description = '右翼（钳形）'
        
        return {
            'longitudinal_offset': 0.0,
            'lateral_offset': 0.0,
            'altitude_offset': 0.0,
            'heading_offset': heading_offset,
            'description': description
        }
    
    def get_node_decision(self, node_name: str, aircraft_role: str,
                         aircraft_state: Dict, enemy_formation: List,
                         decision: Dict) -> Dict:
        """
        获取各节点的战术特定决策
        
        钳形攻势的特点：
        - 双机从左右两侧同时接敌
        - 分散敌机注意力和火力
        - 增大敌机的角度威胁
        """
        tactical_role = self.roles.get(aircraft_role, 'left')
        base_node = node_name.rstrip("12'")
        
        adjustments = {}
        
        if base_node == 'MTR':
            # MTR节点：保持钳形角度
            adjustments['maintain_pincer_angle'] = True
            adjustments['target_angle'] = self.pincer_angle if tactical_role == 'right' else -self.pincer_angle
        
        elif base_node == 'LR':
            # LR节点：同时发射，最大化火力威胁
            adjustments['synchronized_launch'] = True
            adjustments['missile_count'] = 2
        
        elif base_node == 'TR':
            # TR节点：保持钳形态势
            if decision.get('action') == 'evasion':
                adjustments['maintain_lateral_separation'] = True
        
        elif base_node == 'DOR':
            # DOR节点：规避时保持左右分离
            if tactical_role == 'left':
                adjustments['evasion_direction'] = 'left'
            else:
                adjustments['evasion_direction'] = 'right'
        
        return adjustments
    
    def get_description(self) -> str:
        return "钳形攻势：双机左右包夹，分散敌机火力"
