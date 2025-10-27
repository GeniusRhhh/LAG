"""
敌方意图识别（简化版）
"""
from utils.constants import ENEMY_INTENT, ATTACK_ASPECT_ANGLE, DISENGAGE_ASPECT_ANGLE
from utils.geometry import calculate_relative_geometry


class EnemyIntentRecognizer:
    """敌方意图识别器（简化版）"""
    
    def __init__(self):
        """初始化敌方意图识别器"""
        pass
    
    def recognize(self, my_state: dict, enemy_state: dict) -> str:
        """
        识别敌方意图（简化版）
        
        Args:
            my_state: 我机状态
            enemy_state: 敌机状态
        
        Returns:
            敌方意图：'attack' / 'neutral' / 'disengage'
        """
        # 计算相对几何关系
        geometry = calculate_relative_geometry(my_state, enemy_state)
        
        # 判断进攻意图
        if self._is_attacking(geometry):
            return ENEMY_INTENT['ATTACK']
        
        # 判断逃逸意图
        if self._is_disengaging(geometry):
            return ENEMY_INTENT['DISENGAGE']
        
        # 中立意图
        return ENEMY_INTENT['NEUTRAL']
    
    def _is_attacking(self, geometry: dict) -> bool:
        """
        判断是否为进攻意图
        
        条件：敌机航向指向我机（进入角<45°）且正在接近
        """
        return (geometry['aspect_angle'] < ATTACK_ASPECT_ANGLE and
                geometry['is_closing'])
    
    def _is_disengaging(self, geometry: dict) -> bool:
        """
        判断是否为逃逸意图
        
        条件：敌机航向背离我机（进入角>135°）
        """
        return geometry['aspect_angle'] > DISENGAGE_ASPECT_ANGLE
