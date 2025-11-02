"""
意图识别系统
识别敌方意图：进攻（攻击、协同、佯攻）、中立（干扰、侦察、防御）、逃逸
"""

import numpy as np
import logging
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
from envs.JSBSim.core.catalog import Catalog as c


class EnemyIntent(Enum):
    """敌方意图类型"""
    # 攻击类型
    ATTACK = "ATTACK"           # 攻击
    COORDINATED = "COORDINATED"  # 协同攻击
    FEINT = "FEINT"             # 佯攻
    
    # 中立类型
    JAMMING = "JAMMING"         # 干扰
    RECONNAISSANCE = "RECONNAISSANCE"  # 侦察
    DEFENSIVE = "DEFENSIVE"     # 防御/规避
    
    # 脱离类型
    ESCAPE = "ESCAPE"           # 逃逸


class FriendlyIntent(Enum):
    """我方意图类型"""
    AGGRESSIVE_CLEAR = "AGGRESSIVE_CLEAR"      # 激进肃清
    CONSERVATIVE_CLEAR = "CONSERVATIVE_CLEAR"  # 保守肃清
    DEFENSIVE = "DEFENSIVE"                    # 防御意图


@dataclass
class IntentFeatures:
    """意图特征"""
    heading_angle: float      # 航向角（相对于我机）
    closing_rate: float       # 接近速率
    altitude_change: float    # 高度变化率
    speed_change: float       # 速度变化率
    formation_type: str       # 编队类型
    radar_status: str         # 雷达状态
    distance: float           # 距离


class IntentRecognizer:
    """
    意图识别器
    基于敌机行为特征识别其作战意图
    """
    
    def __init__(self):
        """初始化意图识别器"""
        # 历史数据存储（用于趋势分析）
        self.intent_history = {}  # agent_id -> List[EnemyIntent]
        self.feature_history = {}  # agent_id -> List[IntentFeatures]
        self.history_length = 5   # 保留最近5个历史记录
        
        logging.info("✅ 意图识别系统初始化完成")
    
    def recognize_enemy_intent(
        self,
        env,
        enemy_id: str,
        my_aircraft_list: List
    ) -> EnemyIntent:
        """
        识别敌方意图
        
        Args:
            env: 环境对象
            enemy_id: 敌方飞机ID
            my_aircraft_list: 我方飞机列表
            
        Returns:
            EnemyIntent: 识别的敌方意图
        """
        try:
            if enemy_id not in env.agents or not env.agents[enemy_id].is_alive:
                return EnemyIntent.ESCAPE
            
            # 提取特征
            features = self._extract_features(env, enemy_id, my_aircraft_list)
            
            # 存储历史
            if enemy_id not in self.feature_history:
                self.feature_history[enemy_id] = []
            self.feature_history[enemy_id].append(features)
            if len(self.feature_history[enemy_id]) > self.history_length:
                self.feature_history[enemy_id].pop(0)
            
            # 基于规则的意图识别
            intent = self._classify_intent(features, enemy_id)
            
            # 存储意图历史
            if enemy_id not in self.intent_history:
                self.intent_history[enemy_id] = []
            self.intent_history[enemy_id].append(intent)
            if len(self.intent_history[enemy_id]) > self.history_length:
                self.intent_history[enemy_id].pop(0)
            
            return intent
            
        except Exception as e:
            logging.error(f"意图识别失败 {enemy_id}: {e}")
            return EnemyIntent.ATTACK  # 默认认为是攻击
    
    def _extract_features(
        self,
        env,
        enemy_id: str,
        my_aircraft_list: List
    ) -> IntentFeatures:
        """提取敌机行为特征"""
        enemy = env.agents[enemy_id]
        enemy_pos = enemy.get_position()
        enemy_heading = np.rad2deg(enemy.get_property_value(c.attitude_psi_rad))
        enemy_speed = enemy.get_property_value(c.velocities_u_fps) * 0.3048
        enemy_alt = enemy.get_property_value(c.position_h_sl_m)
        
        # 找到最近的我方飞机
        min_distance = float('inf')
        closest_my_aircraft = None
        for my_aircraft in my_aircraft_list:
            if my_aircraft.is_alive:
                my_pos = my_aircraft.get_position()
                dist = np.linalg.norm(np.array(enemy_pos) - np.array(my_pos))
                if dist < min_distance:
                    min_distance = dist
                    closest_my_aircraft = my_aircraft
        
        if closest_my_aircraft is None:
            return IntentFeatures(0, 0, 0, 0, "UNKNOWN", "UNKNOWN", 100000)
        
        # 计算相对参数
        my_pos = closest_my_aircraft.get_position()
        
        # 航向角（敌机航向相对于我机方位）
        dx = my_pos[0] - enemy_pos[0]
        dy = my_pos[1] - enemy_pos[1]
        bearing_to_me = np.rad2deg(np.arctan2(dy, dx))
        bearing_to_me = (90 - bearing_to_me) % 360
        heading_angle = abs(self._normalize_angle(enemy_heading - bearing_to_me))
        
        # 接近速率（简化：基于距离变化）
        closing_rate = 0
        if enemy_id in self.feature_history and len(self.feature_history[enemy_id]) > 0:
            prev_distance = self.feature_history[enemy_id][-1].distance
            time_interval = env.time_interval
            closing_rate = (prev_distance - min_distance) / (time_interval * env.agent_interaction_steps)
        
        # 高度变化率
        altitude_change = 0
        if enemy_id in self.feature_history and len(self.feature_history[enemy_id]) > 0:
            prev_alt = self.feature_history[enemy_id][-1].altitude_change  # 实际存的是高度
            time_interval = env.time_interval
            altitude_change = (enemy_alt - prev_alt) / (time_interval * env.agent_interaction_steps) if prev_alt != 0 else 0
        
        # 速度变化率
        speed_change = 0
        if enemy_id in self.feature_history and len(self.feature_history[enemy_id]) > 0:
            prev_speed = self.feature_history[enemy_id][-1].speed_change  # 实际存的是速度
            time_interval = env.time_interval
            speed_change = (enemy_speed - prev_speed) / (time_interval * env.agent_interaction_steps) if prev_speed != 0 else 0
        
        # 编队类型（简化）
        formation_type = self._detect_formation(env, enemy_id)
        
        # 雷达状态（简化）
        radar_status = "ACTIVE"  # 假设总是主动
        
        return IntentFeatures(
            heading_angle=heading_angle,
            closing_rate=closing_rate,
            altitude_change=enemy_alt if altitude_change == 0 else altitude_change,  # 第一次存高度
            speed_change=enemy_speed if speed_change == 0 else speed_change,  # 第一次存速度
            formation_type=formation_type,
            radar_status=radar_status,
            distance=min_distance
        )
    
    def _classify_intent(self, features: IntentFeatures, enemy_id: str) -> EnemyIntent:
        """
        基于规则分类意图
        
        规则设计：
        1. 逃逸：航向背离（>135°）且距离增加
        2. 攻击：航向对准（<45°）且距离减小且速度快
        3. 协同：编队攻击
        4. 佯攻：接近后转向
        5. 防御/规避：侧向机动（60-120°）
        6. 侦察：保持距离，低速
        7. 干扰：不明显特征时的默认
        """
        heading = features.heading_angle
        closing = features.closing_rate
        distance_km = features.distance / 1000.0
        
        # 1. 逃逸判断
        if heading > 135 and closing < -50:  # 背离且远离
            return EnemyIntent.ESCAPE
        
        # 2. 攻击判断
        if heading < 45 and closing > 100 and distance_km < 100:
            # 检查是否为协同攻击
            if features.formation_type == "COORDINATED":
                return EnemyIntent.COORDINATED
            return EnemyIntent.ATTACK
        
        # 3. 佯攻判断（需要历史数据）
        if enemy_id in self.intent_history and len(self.intent_history[enemy_id]) >= 3:
            recent_intents = self.intent_history[enemy_id][-3:]
            # 如果之前是攻击，现在转向
            if EnemyIntent.ATTACK in recent_intents and heading > 60:
                return EnemyIntent.FEINT
        
        # 4. 防御/规避判断
        if 60 <= heading <= 120 and abs(features.altitude_change) > 50:
            return EnemyIntent.DEFENSIVE
        
        # 5. 侦察判断
        if distance_km > 80 and abs(closing) < 50:
            return EnemyIntent.RECONNAISSANCE
        
        # 6. 干扰（默认中立类型）
        if 45 <= heading <= 135:
            return EnemyIntent.JAMMING
        
        # 默认：攻击
        return EnemyIntent.ATTACK
    
    def _detect_formation(self, env, enemy_id: str) -> str:
        """检测敌机编队类型"""
        try:
            enemy_pos = env.agents[enemy_id].get_position()
            
            # 查找队友
            teammate_id = None
            if enemy_id == "B0100":
                teammate_id = "B0200"
            elif enemy_id == "B0200":
                teammate_id = "B0100"
            
            if teammate_id and teammate_id in env.agents and env.agents[teammate_id].is_alive:
                teammate_pos = env.agents[teammate_id].get_position()
                distance = np.linalg.norm(np.array(teammate_pos) - np.array(enemy_pos))
                
                # 简单判断：距离<10km认为是协同编队
                if distance < 10000:
                    return "COORDINATED"
            
            return "SINGLE"
            
        except:
            return "UNKNOWN"
    
    def _normalize_angle(self, angle: float) -> float:
        """将角度规范化到[-180, 180]"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
    
    def get_intent_confidence(self, enemy_id: str) -> float:
        """
        获取意图识别的置信度
        基于历史一致性
        
        Returns:
            float: 置信度 [0, 1]
        """
        if enemy_id not in self.intent_history or len(self.intent_history[enemy_id]) < 2:
            return 0.5
        
        recent = self.intent_history[enemy_id][-3:]
        most_common = max(set(recent), key=recent.count)
        confidence = recent.count(most_common) / len(recent)
        
        return confidence
