"""
前后攻击战术
前机吸引火力，后机隐蔽接敌，依次顺序打击
"""
from typing import Dict, List
from .base_tactic import BaseTactic


class SequentialAttackTactic(BaseTactic):
    """前后攻击战术"""
    
    def __init__(self):
        super().__init__(4, '前后攻击')
        self.front_offset = 3000.0  # 前机前出距离 (m)
        self.rear_offset = -3000.0  # 后机后退距离 (m)
    
    def get_formation(self, aircraft_role: str) -> Dict:
        """
        获取战术队形
        
        前机前出，后机后退，形成前后攻击态势
        """
        tactical_role = self.roles.get(aircraft_role, 'front')
        
        if tactical_role == 'front':
            # 前机：前出
            offset = self.front_offset
            description = '前机（前后攻击）'
        else:
            # 后机：后退
            offset = self.rear_offset
            description = '后机（前后攻击）'
        
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
        
        前后攻击的特点：
        - 前机先接敌，吸引火力
        - 后机隐蔽接敌，伺机攻击
        - 依次顺序打击，分散敌机防御
        """
        tactical_role = self.roles.get(aircraft_role, 'front')
        base_node = node_name.rstrip("12'")
        
        adjustments = {}
        
        if base_node == 'MTR':
            # MTR节点：前机先到达，后机延迟
            if tactical_role == 'front':
                adjustments['arrival_priority'] = 'first'
                adjustments['attract_attention'] = True
            else:
                adjustments['arrival_priority'] = 'second'
                adjustments['stay_hidden'] = True
        
        elif base_node == 'LR':
            # LR节点：前机先发射，后机延迟发射
            if tactical_role == 'front':
                adjustments['launch_timing'] = 'immediate'
                adjustments['missile_count'] = 1
            else:
                adjustments['launch_timing'] = 'delayed'
                adjustments['launch_delay'] = 5.0  # 延迟5秒
                adjustments['missile_count'] = 2
        
        elif base_node == 'TR':
            # TR节点：前机可能需要脱离，后机继续
            if tactical_role == 'front' and decision.get('action') == 'retreat':
                adjustments['allow_retreat'] = True
            else:
                adjustments['continue_mission'] = True
        
        elif base_node == 'DOR':
            # DOR节点：保持前后间隔
            adjustments['maintain_longitudinal_separation'] = True
        
        return adjustments
    
    def get_description(self) -> str:
        return "前后攻击：前机吸引火力，后机隐蔽接敌，依次打击"
