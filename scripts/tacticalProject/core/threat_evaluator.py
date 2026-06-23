"""
威胁值评估模块
计算总威胁度、角度威胁值、高度威胁值
"""
import numpy as np
import logging


class ThreatEvaluator:
    """威胁值评估器"""
    
    def __init__(self):
        self.threat_cache = {}  # 缓存威胁值计算结果
    
    def calculate_total_threat(self, my_aircraft, enemy_aircraft, env):
        """
        计算总威胁度
        
        考虑因素：
        - 距离威胁
        - 速度威胁
        - 角度威胁
        - 高度威胁
        
        Returns:
            float: 0-1之间的威胁值
        """
        try:
            # 获取位置和速度
            my_pos = np.array(my_aircraft.get_position())
            enemy_pos = np.array(enemy_aircraft.get_position())
            my_vel = np.array(my_aircraft.get_velocity())
            enemy_vel = np.array(enemy_aircraft.get_velocity())
            
            # 1. 距离威胁 (越近威胁越大)
            distance = np.linalg.norm(my_pos - enemy_pos)
            distance_threat = self._calculate_distance_threat(distance)
            
            # 2. 速度威胁 (敌机速度越快威胁越大)
            my_speed = np.linalg.norm(my_vel)
            enemy_speed = np.linalg.norm(enemy_vel)
            speed_threat = self._calculate_speed_threat(my_speed, enemy_speed)
            
            # 3. 角度威胁
            angle_threat = self.calculate_angle_threat(my_aircraft, enemy_aircraft)
            
            # 4. 高度威胁
            altitude_threat = self.calculate_altitude_threat(my_aircraft, enemy_aircraft)
            
            # 综合威胁值 (加权平均)
            total_threat = (
                0.3 * distance_threat +
                0.2 * speed_threat +
                0.3 * angle_threat +
                0.2 * altitude_threat
            )
            
            return min(1.0, max(0.0, total_threat))
            
        except Exception as e:
            logging.error(f"计算总威胁度错误: {e}")
            return 0.5
    
    def calculate_angle_threat(self, my_aircraft, enemy_aircraft):
        """
        计算角度威胁值
        
        考虑：
        - 敌机是否在我机前方
        - 我机是否在敌机雷达照射范围内
        - 相对角度优势
        
        Returns:
            float: 0-1之间的威胁值
        """
        try:
            my_pos = np.array(my_aircraft.get_position())
            enemy_pos = np.array(enemy_aircraft.get_position())
            
            # 获取航向（使用catalog常量）
            from envs.JSBSim.core.catalog import Catalog as c
            my_heading = my_aircraft.get_property_value(c.attitude_psi_rad)
            enemy_heading = enemy_aircraft.get_property_value(c.attitude_psi_rad)
            
            # 计算相对位置向量
            rel_pos = enemy_pos - my_pos
            rel_pos_2d = rel_pos[:2]  # 只考虑水平面
            
            # 计算敌机相对于我机的方位角
            angle_to_enemy = np.arctan2(rel_pos_2d[1], rel_pos_2d[0])
            
            # 计算我机航向与敌机方位的夹角
            my_angle_diff = abs(self._normalize_angle(angle_to_enemy - my_heading))
            
            # 计算敌机航向与我机方位的夹角
            angle_to_me = np.arctan2(-rel_pos_2d[1], -rel_pos_2d[0])
            enemy_angle_diff = abs(self._normalize_angle(angle_to_me - enemy_heading))
            
            # 角度威胁：敌机在我前方且我在敌机前方时威胁最大
            # 我机角度差越小越好（敌机在我正前方）
            # 敌机角度差越小威胁越大（我在敌机正前方）
            my_angle_threat = 1.0 - (my_angle_diff / np.pi)  # 0度=1.0, 180度=0.0
            enemy_angle_threat = 1.0 - (enemy_angle_diff / np.pi)
            
            # 综合角度威胁
            angle_threat = 0.4 * my_angle_threat + 0.6 * enemy_angle_threat
            
            return min(1.0, max(0.0, angle_threat))
            
        except Exception as e:
            logging.error(f"计算角度威胁值错误: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            logging.error(f"my_heading_val type: {type(my_heading_val)}, value: {my_heading_val}")
            logging.error(f"enemy_heading_val type: {type(enemy_heading_val)}, value: {enemy_heading_val}")
            return 0.5
    
    def calculate_altitude_threat(self, my_aircraft, enemy_aircraft):
        """
        计算高度威胁值
        
        考虑：
        - 高度差
        - 敌机是否占据高度优势
        
        Returns:
            float: 0-1之间的威胁值
        """
        try:
            my_alt = my_aircraft.get_position()[2]
            enemy_alt = enemy_aircraft.get_position()[2]
            
            # 高度差
            alt_diff = enemy_alt - my_alt
            
            # 高度威胁：敌机高度越高威胁越大
            # 高度差在-2000m到+2000m之间映射到0-1
            altitude_threat = (alt_diff + 2000) / 4000
            
            return min(1.0, max(0.0, altitude_threat))
            
        except Exception as e:
            logging.error(f"计算高度威胁值错误: {e}")
            return 0.5
    
    def _calculate_distance_threat(self, distance):
        """计算距离威胁 (距离越近威胁越大)"""
        # 距离在40km-120km之间映射到1.0-0.0
        if distance < 40000:
            return 1.0
        elif distance > 120000:
            return 0.0
        else:
            return 1.0 - (distance - 40000) / 80000
    
    def _calculate_speed_threat(self, my_speed, enemy_speed):
        """计算速度威胁 (敌机速度优势越大威胁越大)"""
        speed_diff = enemy_speed - my_speed
        # 速度差在-200到+200之间映射到0-1
        speed_threat = (speed_diff + 200) / 400
        return min(1.0, max(0.0, speed_threat))
    
    def _normalize_angle(self, angle):
        """标准化角度到[-π, π]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def evaluate_situation(self, env):
        """
        评估整体态势
        
        Returns:
            dict: 包含各机威胁值和态势评估的字典
        """
        try:
            # 我方飞机
            my_aircraft = [env._jsbsims.get("A0100"), env._jsbsims.get("A0200")]
            enemy_aircraft = [env._jsbsims.get("B0100"), env._jsbsims.get("B0200")]
            
            # 计算长机威胁（敌机对A0100的威胁）
            lead_threats = []
            for enemy_ac in enemy_aircraft:
                if enemy_ac and enemy_ac.is_alive and my_aircraft[0] and my_aircraft[0].is_alive:
                    threat = self.calculate_total_threat(my_aircraft[0], enemy_ac, env)
                    lead_threats.append(threat)
            
            # 计算僚机威胁（敌机对A0200的威胁）
            wingman_threats = []
            for enemy_ac in enemy_aircraft:
                if enemy_ac and enemy_ac.is_alive and my_aircraft[1] and my_aircraft[1].is_alive:
                    threat = self.calculate_total_threat(my_aircraft[1], enemy_ac, env)
                    wingman_threats.append(threat)
            
            # 计算平均威胁
            lead_threat = max(lead_threats) if lead_threats else 0.5
            wingman_threat = max(wingman_threats) if wingman_threats else 0.5
            avg_threat = (lead_threat + wingman_threat) / 2
            
            # 态势判断
            situation = 'ADVANTAGE' if avg_threat < 0.5 else 'DISADVANTAGE'
            
            return {
                'lead_threat': lead_threat,
                'wingman_threat': wingman_threat,
                'avg_threat': avg_threat,
                'situation': situation
            }
                
        except Exception as e:
            logging.error(f"评估态势错误: {e}")
            return {
                'lead_threat': 0.5,
                'wingman_threat': 0.5,
                'avg_threat': 0.5,
                'situation': 'ADVANTAGE'
            }
