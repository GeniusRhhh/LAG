"""
战术选择系统
"""
from typing import Dict, Tuple
from utils.constants import (
    TACTICS, DRAG_SHOOT_HIGH_THREAT, DRAG_SHOOT_LOW_THREAT,
    PINCER_ANGLE_THREAT, HIGH_LOW_ALTITUDE_THREAT,
    SEQUENTIAL_THREAT_SUM, SEQUENTIAL_THREAT_RATIO,
    SIDE_BY_SIDE_THREAT_SUM, SIDE_BY_SIDE_THREAT_RATIO
)


class TacticSelector:
    """战术选择器"""
    
    def __init__(self):
        """初始化战术选择器"""
        pass
    
    def select_tactic(self, blue_threats: Dict, red_threats: Dict) -> Tuple[int, Dict]:
        """
        根据威胁值选择战术
        
        Args:
            blue_threats: 我方双机的威胁值
                {
                    'lead': {'total': 0.6, 'angle': 0.5, 'altitude': 0.3, ...},
                    'wingman': {'total': 0.4, 'angle': 0.3, 'altitude': 0.2, ...}
                }
            red_threats: 敌方双机对我方的威胁值
                {
                    'enemy1_to_lead': {'total': 0.7, ...},
                    'enemy2_to_lead': {'total': 0.5, ...},
                    'enemy1_to_wingman': {'total': 0.6, ...},
                    'enemy2_to_wingman': {'total': 0.4, ...}
                }
        
        Returns:
            (战术编号, 角色分配字典)
        """
        # 提取威胁值
        lead_threat = blue_threats.get('lead', {}).get('total', 0.0)
        wingman_threat = blue_threats.get('wingman', {}).get('total', 0.0)
        
        lead_angle_threat = blue_threats.get('lead', {}).get('angle', 0.0)
        wingman_angle_threat = blue_threats.get('wingman', {}).get('angle', 0.0)
        
        lead_altitude_threat = blue_threats.get('lead', {}).get('altitude', 0.0)
        wingman_altitude_threat = blue_threats.get('wingman', {}).get('altitude', 0.0)
        
        # 计算敌方威胁和
        enemy_threats = [
            red_threats.get('enemy1_to_lead', {}).get('total', 0.0),
            red_threats.get('enemy2_to_lead', {}).get('total', 0.0),
        ]
        threat_sum = sum(enemy_threats)
        threat_max = max(enemy_threats) if enemy_threats else 0.0
        threat_min = min(enemy_threats) if enemy_threats else 0.0
        threat_ratio = threat_max / threat_min if threat_min > 0.01 else 999
        
        # 1. 检查拖曳射击
        if self._check_drag_shoot(lead_threat, wingman_threat):
            return self._select_drag_shoot(lead_threat, wingman_threat)
        
        # 2. 检查钳形攻势
        if self._check_pincer_attack(lead_angle_threat, wingman_angle_threat):
            return self._select_pincer_attack(lead_angle_threat, wingman_angle_threat)
        
        # 3. 检查上下夹击
        if self._check_high_low_attack(lead_altitude_threat, wingman_altitude_threat):
            return self._select_high_low_attack(lead_altitude_threat, wingman_altitude_threat)
        
        # 4. 检查前后攻击
        if self._check_sequential_attack(threat_sum, threat_ratio):
            return self._select_sequential_attack(lead_threat, wingman_threat)
        
        # 5. 默认：并排射击
        return self._select_side_by_side()
    
    def _check_drag_shoot(self, lead_threat: float, wingman_threat: float) -> bool:
        """检查是否适用拖曳射击"""
        return ((lead_threat > DRAG_SHOOT_HIGH_THREAT and wingman_threat <= DRAG_SHOOT_LOW_THREAT) or
                (wingman_threat > DRAG_SHOOT_HIGH_THREAT and lead_threat <= DRAG_SHOOT_LOW_THREAT))
    
    def _select_drag_shoot(self, lead_threat: float, wingman_threat: float) -> Tuple[int, Dict]:
        """选择拖曳射击，分配角色"""
        if lead_threat > wingman_threat:
            # 长机威胁大，僚机前出作为拖曳机
            roles = {'lead': 'shooter', 'wingman': 'drag'}
        else:
            # 僚机威胁大，长机前出作为拖曳机
            roles = {'lead': 'drag', 'wingman': 'shooter'}
        
        return TACTICS['DRAG_SHOOT'], roles
    
    def _check_pincer_attack(self, lead_angle_threat: float, wingman_angle_threat: float) -> bool:
        """检查是否适用钳形攻势"""
        return (lead_angle_threat > PINCER_ANGLE_THREAT or
                wingman_angle_threat > PINCER_ANGLE_THREAT)
    
    def _select_pincer_attack(self, lead_angle_threat: float, wingman_angle_threat: float) -> Tuple[int, Dict]:
        """选择钳形攻势，分配角色"""
        roles = {'lead': 'left', 'wingman': 'right'}
        return TACTICS['PINCER_ATTACK'], roles
    
    def _check_high_low_attack(self, lead_altitude_threat: float, wingman_altitude_threat: float) -> bool:
        """检查是否适用上下夹击"""
        return (lead_altitude_threat > HIGH_LOW_ALTITUDE_THREAT or
                wingman_altitude_threat > HIGH_LOW_ALTITUDE_THREAT)
    
    def _select_high_low_attack(self, lead_altitude_threat: float, wingman_altitude_threat: float) -> Tuple[int, Dict]:
        """选择上下夹击，分配角色"""
        if lead_altitude_threat > wingman_altitude_threat:
            # 长机高度威胁大，僚机爬升至高空
            roles = {'lead': 'low', 'wingman': 'high'}
        else:
            # 僚机高度威胁大，长机爬升至高空
            roles = {'lead': 'high', 'wingman': 'low'}
        
        return TACTICS['HIGH_LOW_ATTACK'], roles
    
    def _check_sequential_attack(self, threat_sum: float, threat_ratio: float) -> bool:
        """检查是否适用前后攻击"""
        return (threat_sum > SEQUENTIAL_THREAT_SUM and
                threat_ratio < SEQUENTIAL_THREAT_RATIO)
    
    def _select_sequential_attack(self, lead_threat: float, wingman_threat: float) -> Tuple[int, Dict]:
        """选择前后攻击，分配角色"""
        if lead_threat < wingman_threat:
            # 长机威胁小，长机前出
            roles = {'lead': 'front', 'wingman': 'rear'}
        else:
            # 僚机威胁小，僚机前出
            roles = {'lead': 'rear', 'wingman': 'front'}
        
        return TACTICS['SEQUENTIAL_ATTACK'], roles
    
    def _select_side_by_side(self) -> Tuple[int, Dict]:
        """选择并排射击"""
        roles = {'lead': 'left', 'wingman': 'right'}
        return TACTICS['SIDE_BY_SIDE'], roles
