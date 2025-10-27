"""
战术规避
只包含1-2个简单机动的防御战术
"""
from typing import Dict, List
from .base_tactic import BaseTactic


class TacticalEvasion(BaseTactic):
    """战术规避"""
    
    def __init__(self):
        super().__init__(6, '战术规避')
    
    def get_formation(self, aircraft_role: str) -> Dict:
        """
        战术规避没有固定队形，根据威胁动态调整
        """
        return {
            'longitudinal_offset': 0.0,
            'lateral_offset': 0.0,
            'altitude_offset': 0.0,
            'description': '战术规避（动态）'
        }
    
    def get_node_decision(self, node_name: str, aircraft_role: str,
                         aircraft_state: Dict, enemy_formation: List,
                         decision: Dict) -> Dict:
        """
        战术规避主要在DOR节点执行
        """
        base_node = node_name.rstrip("12'")
        threat = decision.get('threat', {})
        threat_level = threat.get('total', 0.5)
        
        adjustments = {}
        
        if base_node == 'DOR':
            # DOR节点：根据威胁值选择规避机动
            if threat_level > 0.8:
                adjustments['evasion_type'] = 'short_skate'
                adjustments['turn_angle'] = 180.0
                adjustments['description'] = 'Short Skate规避'
            elif threat_level > 0.5:
                adjustments['evasion_type'] = 'notch_back'
                adjustments['turn_angle'] = 180.0
                adjustments['altitude_change'] = -500.0
                adjustments['description'] = 'Notch Back规避'
            else:
                adjustments['evasion_type'] = 'beam'
                adjustments['turn_angle'] = 90.0
                adjustments['description'] = 'Beam机动规避'
        
        return adjustments
    
    def get_description(self) -> str:
        return "战术规避：根据威胁值动态选择规避机动"
