"""
威胁值计算系统
"""
import numpy as np
from typing import Dict, Tuple
from utils.geometry import calculate_relative_geometry, calculate_speed
from utils.constants import (
    THREAT_THRESHOLDS, RETREAT_TOTAL_THREAT, RETREAT_SINGLE_THREAT,
    EVASION_SINGLE_THREAT, DECISION_TYPE, ENEMY_INTENT
)


class ThreatCalculator:
    """威胁值计算器"""
    
    def __init__(self):
        """初始化威胁值计算器"""
        self.max_distance = 120.0  # km
        self.max_altitude_diff = 5000.0  # m
        self.max_speed = 600.0  # m/s
    
    def calculate_threat(self, my_state: Dict, enemy_state: Dict) -> Dict:
        """
        计算敌方对我方的威胁值
        
        Args:
            my_state: 我机状态
            enemy_state: 敌机状态
        
        Returns:
            威胁值字典
                - 'distance': 距离威胁值 (0~1)
                - 'angle': 角度威胁值 (0~1)
                - 'altitude': 高度威胁值 (0~1)
                - 'speed': 速度威胁值 (0~1)
                - 'total': 总威胁值 (0~1)
        """
        # 计算相对几何关系
        geometry = calculate_relative_geometry(my_state, enemy_state)
        
        # 计算各项威胁值
        distance_threat = self._calculate_distance_threat(geometry['distance'])
        angle_threat = self._calculate_angle_threat(geometry['aspect_angle'], geometry['angle_off'])
        altitude_threat = self._calculate_altitude_threat(geometry['altitude_diff'])
        speed_threat = self._calculate_speed_threat(my_state['velocity'], enemy_state['velocity'])
        
        # 计算总威胁值（加权平均）
        total_threat = (
            distance_threat * 0.3 +
            angle_threat * 0.3 +
            altitude_threat * 0.2 +
            speed_threat * 0.2
        )
        
        return {
            'distance': distance_threat,
            'angle': angle_threat,
            'altitude': altitude_threat,
            'speed': speed_threat,
            'total': total_threat,
        }
    
    def _calculate_distance_threat(self, distance: float) -> float:
        """
        计算距离威胁值
        
        Args:
            distance: 距离 (km)
        
        Returns:
            距离威胁值 (0~1)，距离越近威胁越大
        """
        if distance >= self.max_distance:
            return 0.0
        
        # 非线性映射：距离越近，威胁增长越快
        normalized_distance = distance / self.max_distance
        threat = 1.0 - normalized_distance ** 0.5
        
        return np.clip(threat, 0.0, 1.0)
    
    def _calculate_angle_threat(self, aspect_angle: float, angle_off: float) -> float:
        """
        计算角度威胁值
        
        Args:
            aspect_angle: 敌机进入角 (度)
            angle_off: 我机离轴角 (度)
        
        Returns:
            角度威胁值 (0~1)，敌机越正对我机且我机越侧对敌机，威胁越大
        """
        # 敌机进入角：0度（正对）威胁最大，180度（背离）威胁最小
        aspect_threat = 1.0 - aspect_angle / 180.0
        
        # 我机离轴角：90度（侧对）威胁最大，0度（正对）威胁最小
        angle_off_threat = abs(90.0 - angle_off) / 90.0
        angle_off_threat = 1.0 - angle_off_threat
        
        # 综合角度威胁
        threat = aspect_threat * 0.6 + angle_off_threat * 0.4
        
        return np.clip(threat, 0.0, 1.0)
    
    def _calculate_altitude_threat(self, altitude_diff: float) -> float:
        """
        计算高度威胁值
        
        Args:
            altitude_diff: 高度差 (m)，正值表示我机更高
        
        Returns:
            高度威胁值 (0~1)，敌机高度优势越大威胁越大
        """
        # 敌机高度优势（负的altitude_diff）
        enemy_altitude_advantage = -altitude_diff
        
        if enemy_altitude_advantage <= 0:
            # 我机更高或同高度，威胁较小
            threat = 0.0
        else:
            # 敌机更高，威胁增大
            threat = min(enemy_altitude_advantage / self.max_altitude_diff, 1.0)
        
        return np.clip(threat, 0.0, 1.0)
    
    def _calculate_speed_threat(self, my_vel: np.ndarray, enemy_vel: np.ndarray) -> float:
        """
        计算速度威胁值
        
        Args:
            my_vel: 我机速度 (m/s)
            enemy_vel: 敌机速度 (m/s)
        
        Returns:
            速度威胁值 (0~1)，敌机速度优势越大威胁越大
        """
        my_speed = calculate_speed(my_vel)
        enemy_speed = calculate_speed(enemy_vel)
        
        # 敌机速度优势
        speed_advantage = enemy_speed - my_speed
        
        if speed_advantage <= 0:
            # 我机更快或同速度，威胁较小
            threat = 0.0
        else:
            # 敌机更快，威胁增大
            threat = min(speed_advantage / (self.max_speed * 0.3), 1.0)
        
        return np.clip(threat, 0.0, 1.0)
    
    def calculate_threat_degree(self, threat: Dict, enemy_intent: str) -> str:
        """
        根据威胁值和敌方意图，判断威胁程度
        
        Args:
            threat: 威胁值字典
            enemy_intent: 敌方意图
        
        Returns:
            决策结果：'retreat' / 'evasion' / 'continue'
        """
        # 1. 检查撤退条件
        if self._should_retreat(threat, enemy_intent):
            return DECISION_TYPE['RETREAT']
        
        # 2. 检查规避条件
        if self._should_evade(threat, enemy_intent):
            return DECISION_TYPE['EVASION']
        
        # 3. 继续执行核心任务
        return DECISION_TYPE['CONTINUE']
    
    def _should_retreat(self, threat: Dict, enemy_intent: str) -> bool:
        """
        判断是否应该撤退
        
        条件：敌方意图为"进攻" 且
             （总威胁值>0.8 或 单项威胁值有≥2个>0.8）
        """
        if enemy_intent != ENEMY_INTENT['ATTACK']:
            return False
        
        # 检查总威胁值
        if threat['total'] > RETREAT_TOTAL_THREAT:
            return True
        
        # 检查单项威胁值
        high_threat_count = sum([
            threat['distance'] > RETREAT_SINGLE_THREAT,
            threat['angle'] > RETREAT_SINGLE_THREAT,
            threat['altitude'] > RETREAT_SINGLE_THREAT,
            threat['speed'] > RETREAT_SINGLE_THREAT,
        ])
        
        if high_threat_count >= 2:
            return True
        
        return False
    
    def _should_evade(self, threat: Dict, enemy_intent: str) -> bool:
        """
        判断是否应该规避
        
        条件：敌方意图为"中立"或"进攻" 且
             单项威胁值有1-2个>0.8
        """
        if enemy_intent not in [ENEMY_INTENT['ATTACK'], ENEMY_INTENT['NEUTRAL']]:
            return False
        
        # 检查单项威胁值
        high_threat_count = sum([
            threat['distance'] > EVASION_SINGLE_THREAT,
            threat['angle'] > EVASION_SINGLE_THREAT,
            threat['altitude'] > EVASION_SINGLE_THREAT,
            threat['speed'] > EVASION_SINGLE_THREAT,
        ])
        
        if 1 <= high_threat_count <= 2:
            return True
        
        return False
    
    def calculate_formation_threat(self, my_state: Dict, enemy_formation: list) -> Dict:
        """
        计算敌方编队对我方单机的威胁值
        
        Args:
            my_state: 我机状态
            enemy_formation: 敌方编队状态列表
        
        Returns:
            威胁值字典
                - 'threats': 各敌机的威胁值列表
                - 'max_threat': 最大威胁值
                - 'total_threat': 总威胁值（所有敌机威胁值之和）
        """
        threats = []
        for enemy_state in enemy_formation:
            threat = self.calculate_threat(my_state, enemy_state)
            threats.append(threat)
        
        if not threats:
            return {
                'threats': [],
                'max_threat': {'total': 0.0},
                'total_threat': 0.0,
            }
        
        # 找到最大威胁
        max_threat = max(threats, key=lambda t: t['total'])
        
        # 计算总威胁（所有敌机威胁值之和，但不超过1.0）
        total_threat = min(sum(t['total'] for t in threats), 1.0)
        
        return {
            'threats': threats,
            'max_threat': max_threat,
            'total_threat': total_threat,
        }
