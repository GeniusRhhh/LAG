"""
意图预测模块
预测敌机意图：攻击、佯攻、协同、干扰、侦察、规避/防御、逃逸
"""
import numpy as np
import logging


class IntentPredictor:
    """意图预测器"""
    
    def __init__(self):
        self.intent_history = {}  # 历史意图记录
    
    def predict_enemy_intent(self, enemy_aircraft, my_aircraft, env):
        """
        预测敌机意图
        
        基于：
        - 敌机航向
        - 敌机速度变化
        - 敌机高度变化
        - 敌机与我机的相对位置
        
        Returns:
            str: 'ATTACK', 'FEINT', 'COORDINATE', 'JAM', 'RECON', 'EVADE', 'ESCAPE'
        """
        try:
            if not enemy_aircraft or not enemy_aircraft.is_alive:
                return 'ESCAPE'
            
            # 获取敌机状态
            enemy_pos = np.array(enemy_aircraft.get_position())
            my_pos = np.array(my_aircraft.get_position())
            enemy_vel = np.array(enemy_aircraft.get_velocity())
            enemy_heading = enemy_aircraft.get_property_value('attitude/psi-rad')
            
            # 计算相对位置
            rel_pos = my_pos - enemy_pos
            distance = np.linalg.norm(rel_pos)
            
            # 计算敌机航向与我机方位的夹角
            angle_to_me = np.arctan2(rel_pos[1], rel_pos[0])
            heading_diff = abs(self._normalize_angle(angle_to_me - enemy_heading))
            
            # 计算敌机速度
            enemy_speed = np.linalg.norm(enemy_vel)
            
            # 意图判断逻辑
            
            # 1. 逃逸：敌机背向我机且距离增加
            if heading_diff > np.pi * 0.75 and distance > 80000:
                return 'ESCAPE'
            
            # 2. 攻击：敌机指向我机且距离接近
            if heading_diff < np.pi * 0.25 and distance < 100000:
                return 'ATTACK'
            
            # 3. 规避/防御：敌机侧向我机
            if np.pi * 0.4 < heading_diff < np.pi * 0.6:
                return 'EVADE'
            
            # 4. 佯攻：敌机指向我机但距离较远
            if heading_diff < np.pi * 0.25 and distance > 100000:
                return 'FEINT'
            
            # 5. 默认：侦察
            return 'RECON'
            
        except Exception as e:
            logging.error(f"预测敌机意图错误: {e}")
            return 'ATTACK'  # 默认假设攻击
    
    def classify_intent_type(self, intent):
        """
        分类意图类型
        
        Returns:
            str: 'ATTACK_TYPE', 'NEUTRAL_TYPE', 'ESCAPE_TYPE'
        """
        if intent in ['ATTACK', 'FEINT', 'COORDINATE']:
            return 'ATTACK_TYPE'
        elif intent in ['JAM', 'RECON', 'EVADE']:
            return 'NEUTRAL_TYPE'
        else:  # ESCAPE
            return 'ESCAPE_TYPE'
    
    def _normalize_angle(self, angle):
        """标准化角度到[-π, π]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
