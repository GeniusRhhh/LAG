"""
上下夹击战术
我方两架战机呈上下夹击态势对敌机射击
"""
from typing import Dict, List
import logging
from core import TacticalDecisionManager
from .base_tactic import BaseTactic


class HighLowAttackDecisionManager(TacticalDecisionManager):
    """强制上下夹击战术的决策管理器"""
    
    def _execute_meld_decision(self, aircraft_id: str, my_state: dict,
                               enemy_formation: list) -> dict:
        """MELD节点：强制选择上下夹击战术"""
        self.current_tactic = 3
        self.current_roles = {'lead': 'high', 'wingman': 'low'}
        
        logging.info(f"[MELD] 强制选择战术: 上下夹击")
        
        self.lead_decision.set_assigned_target(0)
        self.wingman_decision.set_assigned_target(1)
        
        threats = self._calculate_formation_threats(my_state, enemy_formation)
        return {
            'node': 'MELD',
            'action': 'continue',
            'tactic': 3,
            'role': self.current_roles.get('lead' if aircraft_id.endswith('100') else 'wingman'),
            'target': 0 if aircraft_id.endswith('100') else 1,
            'threat': threats.get(aircraft_id, 0.5)
        }


# 保留原有的战术类
class HighLowAttackTactic(BaseTactic):
    """上下夹击战术"""
    
    def __init__(self):
        super().__init__(3, '上下夹击')
        self.altitude_separation = 1000.0  # 高度间隔 (m)
    
    def get_formation(self, aircraft_role: str) -> Dict:
        """
        获取战术队形
        
        高机爬升至高空，低机保持低空，形成上下夹击
        """
        tactical_role = self.roles.get(aircraft_role, 'low')
        
        if tactical_role == 'high':
            # 高机：爬升
            altitude_offset = self.altitude_separation
            description = '高空机（上下夹击）'
        else:
            # 低机：保持低空
            altitude_offset = 0.0
            description = '低空机（上下夹击）'
        
        return {
            'longitudinal_offset': 0.0,
            'lateral_offset': 0.0,
            'altitude_offset': altitude_offset,
            'description': description
        }
    
    def get_node_decision(self, node_name: str, aircraft_role: str,
                         aircraft_state: Dict, enemy_formation: List,
                         decision: Dict) -> Dict:
        """
        获取各节点的战术特定决策
        
        上下夹击的特点：
        - 高机从高空俯冲攻击，具有能量优势
        - 低机从低空攻击，增大敌机高度威胁
        - 敌机难以同时应对上下两个方向
        """
        tactical_role = self.roles.get(aircraft_role, 'low')
        base_node = node_name.rstrip("12'")
        
        adjustments = {}
        
        if base_node == 'MTR':
            # MTR节点：高机保持高度优势
            if tactical_role == 'high':
                adjustments['maintain_altitude_advantage'] = True
                adjustments['dive_angle'] = 10.0  # 轻微俯冲
            else:
                adjustments['maintain_low_altitude'] = True
        
        elif base_node == 'LR':
            # LR节点：高机承担主要打击任务
            if tactical_role == 'high':
                adjustments['fire_priority'] = 'high'
                adjustments['missile_count'] = 2
                adjustments['attack_angle'] = 'dive'  # 俯冲攻击
            else:
                adjustments['fire_priority'] = 'medium'
                adjustments['missile_count'] = 1
        
        elif base_node == 'TR':
            # TR节点：保持高度分离
            adjustments['maintain_altitude_separation'] = True
        
        elif base_node == 'DOR':
            # DOR节点：规避时保持高度差
            if tactical_role == 'high':
                adjustments['evasion_altitude'] = 'maintain_high'
            else:
                adjustments['evasion_altitude'] = 'maintain_low'
        
        return adjustments
    
    def get_description(self) -> str:
        return "上下夹击：高机俯冲攻击，低机低空攻击，增大敌机高度威胁"
