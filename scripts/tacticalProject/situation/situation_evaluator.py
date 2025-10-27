"""
态势评估（简化版）
"""
from utils.geometry import calculate_relative_geometry


class SituationEvaluator:
    """态势评估器（简化版）"""
    
    def __init__(self):
        """初始化态势评估器"""
        pass
    
    def evaluate(self, my_state: dict, enemy_state: dict) -> str:
        """
        评估战场态势（简化版）
        
        只考虑距离和角度两项：
        - 我方占优：距离近 且 角度好（我机正对敌机）
        - 敌方占优：距离远 且 角度差（敌机正对我机）
        - 均势：其他情况
        
        Args:
            my_state: 我机状态
            enemy_state: 敌机状态
        
        Returns:
            态势评估结果：'advantage' / 'parity' / 'disadvantage'
        """
        geometry = calculate_relative_geometry(my_state, enemy_state)
        
        # 距离优势：距离越近越好
        distance_advantage = self._evaluate_distance(geometry['distance'])
        
        # 角度优势：我机正对敌机且敌机侧对我机最好
        angle_advantage = self._evaluate_angle(geometry['aspect_angle'], geometry['angle_off'])
        
        # 综合评分
        score = distance_advantage + angle_advantage
        
        if score > 0.3:
            return 'advantage'
        elif score < -0.3:
            return 'disadvantage'
        else:
            return 'parity'
    
    def _evaluate_distance(self, distance: float) -> float:
        """
        评估距离优势
        
        Args:
            distance: 距离 (km)
        
        Returns:
            距离优势评分 (-1~1)，正值表示我方占优
        """
        # 最佳距离：70-80km（在攻击区内但不太近）
        optimal_distance = 75.0
        
        if distance < 40:
            # 太近，危险
            return -0.5
        elif 70 <= distance <= 80:
            # 最佳距离
            return 1.0
        elif distance > 120:
            # 太远
            return -0.5
        else:
            # 其他距离
            return 0.0
    
    def _evaluate_angle(self, aspect_angle: float, angle_off: float) -> float:
        """
        评估角度优势
        
        Args:
            aspect_angle: 敌机进入角 (度)
            angle_off: 我机离轴角 (度)
        
        Returns:
            角度优势评分 (-1~1)，正值表示我方占优
        """
        # 理想情况：我机正对敌机（angle_off<30°）且敌机侧对我机（aspect_angle>90°）
        my_pointing = 1.0 - angle_off / 180.0  # 我机正对程度
        enemy_not_pointing = aspect_angle / 180.0  # 敌机不正对程度
        
        score = my_pointing * 0.6 + enemy_not_pointing * 0.4
        
        # 归一化到[-1, 1]
        return score * 2 - 1
