"""
态势评估系统
根据文档实现5维态势评估：角度、距离、高度、速度、探测概率（RCS）
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from envs.JSBSim.core.catalog import Catalog as c


class TacticalPhase(Enum):
    """战术阶段枚举"""
    NLT_MELD = "NLT_MELD"
    MELD_MTR = "MELD_MTR"
    MTR_LR = "MTR_LR"
    LR_TR = "LR_TR"
    TR_DOR = "TR_DOR"
    DOR_DR = "DOR_DR"
    DR_MAR = "DR_MAR"
    BEYOND_MAR = "BEYOND_MAR"


@dataclass
class SituationScore:
    """态势得分"""
    angle: float      # 角度项 [0, 1]
    distance: float   # 距离项 [0, 1]
    altitude: float   # 高度项 [0, 1]
    speed: float      # 速度项 [0, 1]
    detection: float  # 探测概率项 [0, 1]
    total: float      # 总态势值 [0, 1]


class SituationEvaluator:
    """
    态势评估器
    实现5维态势评估模型
    """
    
    def __init__(self):
        """初始化态势评估器"""
        # 阶段权重配置（根据不同阶段调整各项权重）
        self.phase_weights = {
            TacticalPhase.NLT_MELD: {
                'angle': 0.35,
                'distance': 0.35,
                'altitude': 0.10,
                'speed': 0.10,
                'detection': 0.10
            },
            TacticalPhase.MELD_MTR: {
                'angle': 0.30,
                'distance': 0.30,
                'altitude': 0.15,
                'speed': 0.15,
                'detection': 0.10
            },
            TacticalPhase.MTR_LR: {
                'angle': 0.25,
                'distance': 0.25,
                'altitude': 0.20,
                'speed': 0.20,
                'detection': 0.10
            },
            TacticalPhase.LR_TR: {
                'angle': 0.25,
                'distance': 0.25,
                'altitude': 0.25,
                'speed': 0.25,
                'detection': 0.0  # 文档要求：后续交战过程去掉探测概率项
            },
            TacticalPhase.TR_DOR: {
                'angle': 0.25,
                'distance': 0.25,
                'altitude': 0.25,
                'speed': 0.25,
                'detection': 0.0  # 文档要求：后续交战过程去掉探测概率项
            },
            TacticalPhase.DOR_DR: {
                'angle': 0.30,
                'distance': 0.20,
                'altitude': 0.25,
                'speed': 0.25,
                'detection': 0.0  # 文档要求：后续交战过程去掉探测概率项
            },
            TacticalPhase.DR_MAR: {
                'angle': 0.35,
                'distance': 0.20,
                'altitude': 0.25,
                'speed': 0.20,
                'detection': 0.0  # 文档要求：后续交战过程去掉探测概率项
            },
            TacticalPhase.BEYOND_MAR: {
                'angle': 0.40,
                'distance': 0.15,
                'altitude': 0.25,
                'speed': 0.20,
                'detection': 0.0  # 文档要求：后续交战过程去掉探测概率项
            }
        }
        
        # 🔥 注释掉初始化日志，减少输出噪音
        # logging.info("✅ 态势评估系统初始化完成")
    
    def evaluate_situation(
        self,
        my_aircraft,
        enemy_aircraft,
        phase: TacticalPhase
    ) -> SituationScore:
        """
        评估我机相对于敌机的态势
        
        Args:
            my_aircraft: 我方飞机
            enemy_aircraft: 敌方飞机
            phase: 当前战术阶段
            
        Returns:
            SituationScore: 态势得分
        """
        # 计算各项态势值
        angle_score = self._evaluate_angle(my_aircraft, enemy_aircraft)
        distance_score = self._evaluate_distance(my_aircraft, enemy_aircraft, phase)
        altitude_score = self._evaluate_altitude(my_aircraft, enemy_aircraft)
        speed_score = self._evaluate_speed(my_aircraft, enemy_aircraft, phase)
        detection_score = self._evaluate_detection(my_aircraft, enemy_aircraft)
        
        # 获取当前阶段的权重
        weights = self.phase_weights.get(phase, self.phase_weights[TacticalPhase.NLT_MELD])
        
        # 计算加权总态势值
        total = (
            weights['angle'] * angle_score +
            weights['distance'] * distance_score +
            weights['altitude'] * altitude_score +
            weights['speed'] * speed_score +
            weights['detection'] * detection_score
        )
        
        return SituationScore(
            angle=angle_score,
            distance=distance_score,
            altitude=altitude_score,
            speed=speed_score,
            detection=detection_score,
            total=total
        )
    
    def _evaluate_angle(self, my_aircraft, enemy_aircraft) -> float:
        """
        评估角度优势
        考虑我机的偏离角和敌机的进入角
        
        返回值范围：[0, 1]，1表示最优角度态势
        """
        try:
            # 获取位置和航向
            my_pos = my_aircraft.get_position()
            enemy_pos = enemy_aircraft.get_position()
            my_heading = np.rad2deg(my_aircraft.get_property_value(c.attitude_psi_rad))
            enemy_heading = np.rad2deg(enemy_aircraft.get_property_value(c.attitude_psi_rad))
            
            # 计算相对位置向量
            dx = enemy_pos[0] - my_pos[0]
            dy = enemy_pos[1] - my_pos[1]
            bearing_to_enemy = np.rad2deg(np.arctan2(dy, dx))
            bearing_to_enemy = (90 - bearing_to_enemy) % 360
            
            # 计算我机偏离角（我机航向与目标方位的夹角）
            my_aspect = abs(self._normalize_angle(my_heading - bearing_to_enemy))
            
            # 计算敌机进入角（敌机航向与我机方位的夹角）
            bearing_to_me = (bearing_to_enemy + 180) % 360
            enemy_aspect = abs(self._normalize_angle(enemy_heading - bearing_to_me))
            
            # 角度评分：
            # 我机偏离角越小越好（正对敌机），敌机进入角越大越好（侧对或尾对我机）
            # 最优：我机0°偏离，敌机180°进入角
            my_aspect_score = 1.0 - (my_aspect / 180.0)
            enemy_aspect_score = enemy_aspect / 180.0
            
            # 综合角度得分
            angle_score = 0.6 * my_aspect_score + 0.4 * enemy_aspect_score
            
            return np.clip(angle_score, 0.0, 1.0)
            
        except Exception as e:
            logging.error(f"角度评估失败: {e}")
            return 0.5
    
    def _evaluate_distance(self, my_aircraft, enemy_aircraft, phase: TacticalPhase) -> float:
        """
        评估距离优势
        根据当前阶段设置理想距离
        
        返回值范围：[0, 1]，1表示处于最佳攻击距离
        """
        try:
            my_pos = my_aircraft.get_position()
            enemy_pos = enemy_aircraft.get_position()
            distance = np.linalg.norm(np.array(enemy_pos) - np.array(my_pos))
            distance_km = distance / 1000.0
            
            # 根据阶段设置理想距离范围
            if phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR]:
                # 远距接敌阶段：希望快速接近，理想距离80-120km
                # 修复：初始120km不应触发威胁
                ideal_distance = 100000
                tolerance = 30000
            elif phase in [TacticalPhase.MTR_LR, TacticalPhase.LR_TR]:
                # 发射准备阶段：理想距离78km左右
                ideal_distance = 78000
                tolerance = 5000
            elif phase == TacticalPhase.TR_DOR:
                # 中制导结束阶段：理想距离70-75km
                ideal_distance = 72000
                tolerance = 8000
            elif phase == TacticalPhase.DOR_DR:
                # 规避阶段：理想距离65-70km
                ideal_distance = 67000
                tolerance = 10000
            else:
                # 其他阶段：理想距离50-65km
                ideal_distance = 57000
                tolerance = 15000
            
            # 计算距离得分（高斯函数）
            distance_diff = abs(distance - ideal_distance)
            distance_score = np.exp(-(distance_diff ** 2) / (2 * tolerance ** 2))
            
            return np.clip(distance_score, 0.0, 1.0)
            
        except Exception as e:
            logging.error(f"距离评估失败: {e}")
            return 0.5
    
    def _evaluate_altitude(self, my_aircraft, enemy_aircraft) -> float:
        """
        评估高度优势
        适度的高度优势有利于导弹攻击
        
        返回值范围：[0, 1]，1表示最佳高度优势
        """
        try:
            my_alt = my_aircraft.get_property_value(c.position_h_sl_m)
            enemy_alt = enemy_aircraft.get_property_value(c.position_h_sl_m)
            alt_diff = my_alt - enemy_alt
            
            # 理想高度优势：高出敌机1000-3000米
            ideal_alt_diff = 2000
            tolerance = 2000
            
            # 高度优势评分
            if alt_diff > 0:
                # 有高度优势
                if alt_diff <= ideal_alt_diff + tolerance:
                    # 适度优势
                    alt_score = 0.5 + 0.5 * (alt_diff / (ideal_alt_diff + tolerance))
                else:
                    # 过高（不利）
                    excess = alt_diff - (ideal_alt_diff + tolerance)
                    alt_score = 1.0 - 0.3 * (excess / 5000)  # 超过5000m以上严重扣分
            else:
                # 无高度优势或劣势
                alt_score = 0.5 * (1.0 + alt_diff / 3000)  # 低3000m以上严重扣分
            
            return np.clip(alt_score, 0.0, 1.0)
            
        except Exception as e:
            logging.error(f"高度评估失败: {e}")
            return 0.5
    
    def _evaluate_speed(self, my_aircraft, enemy_aircraft, phase: TacticalPhase) -> float:
        """
        评估速度优势
        根据阶段调整理想速度
        
        返回值范围：[0, 1]，1表示最佳速度态势
        """
        try:
            my_speed = my_aircraft.get_property_value(c.velocities_u_fps) * 0.3048  # fps转m/s
            enemy_speed = enemy_aircraft.get_property_value(c.velocities_u_fps) * 0.3048
            
            # 根据阶段调整理想速度策略
            if phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR]:
                # 远距接敌：希望快速接近，高速为优
                ideal_speed = 350  # m/s
                speed_score = my_speed / ideal_speed
            elif phase in [TacticalPhase.MTR_LR, TacticalPhase.LR_TR]:
                # 发射准备：适中速度，便于机动
                ideal_speed = 300  # m/s
                speed_diff = abs(my_speed - ideal_speed)
                speed_score = 1.0 - (speed_diff / 100)
            elif phase in [TacticalPhase.TR_DOR, TacticalPhase.DOR_DR]:
                # 规避阶段：速度与敌机相近为优，便于保持态势
                speed_ratio = my_speed / (enemy_speed + 1e-6)
                speed_score = 1.0 - abs(speed_ratio - 1.0) * 0.5
            else:
                # 其他阶段：保持灵活性
                ideal_speed = 320
                speed_diff = abs(my_speed - ideal_speed)
                speed_score = 1.0 - (speed_diff / 100)
            
            return np.clip(speed_score, 0.0, 1.0)
            
        except Exception as e:
            logging.error(f"速度评估失败: {e}")
            return 0.5
    
    def _evaluate_detection(self, my_aircraft, enemy_aircraft) -> float:
        """
        评估探测概率（基于RCS）
        考虑雷达探测能力和目标RCS
        
        返回值范围：[0, 1]，1表示最佳探测态势
        """
        try:
            # 获取相对几何关系
            my_pos = my_aircraft.get_position()
            enemy_pos = enemy_aircraft.get_position()
            distance = np.linalg.norm(np.array(enemy_pos) - np.array(my_pos))
            
            my_heading = np.rad2deg(my_aircraft.get_property_value(c.attitude_psi_rad))
            enemy_heading = np.rad2deg(enemy_aircraft.get_property_value(c.attitude_psi_rad))
            
            # 计算雷达照射角
            dx = enemy_pos[0] - my_pos[0]
            dy = enemy_pos[1] - my_pos[1]
            bearing_to_enemy = np.rad2deg(np.arctan2(dy, dx))
            bearing_to_enemy = (90 - bearing_to_enemy) % 360
            
            # 目标RCS主要取决于“目标相对雷达的视角”，即敌机机头与雷达视线夹角
            # 角度α=0表示敌机机头正对雷达（正面RCS小），α=90侧视RCS大，α=180尾视RCS中等。
            # 这里实现一个接近你文档(3-9)思想的简化分段模型（只做工程近似，保持稳定性）。
            alpha = abs(self._normalize_angle(enemy_heading - (bearing_to_enemy + 180) % 360))  # 敌机机头相对雷达视线
            if alpha <= 90.0:
                # 0→90：逐步增大
                rcs_factor = 0.45 + 0.65 * np.sin(np.deg2rad(alpha))  # ~[0.45,1.10]
            else:
                # 90→180：逐步减小（尾视略小于侧视）
                rcs_factor = 0.85 - 0.35 * np.sin(np.deg2rad(alpha - 90.0))  # ~[0.85,0.50]
            rcs_factor = float(np.clip(rcs_factor, 0.35, 1.15))
            
            # 距离因素（雷达探测能力随距离衰减）
            distance_km = distance / 1000.0
            max_detection_range = 150.0  # km
            # 用平滑衰减而不是线性（远距离更合理）
            distance_factor = float(np.exp(- (distance_km / max_detection_range) ** 2))
            
            # 综合探测概率
            detection_score = float(rcs_factor) * float(distance_factor)
            
            return np.clip(detection_score, 0.0, 1.0)
            
        except Exception as e:
            logging.error(f"探测概率评估失败: {e}")
            return 0.5
    
    def _normalize_angle(self, angle: float) -> float:
        """将角度规范化到[-180, 180]"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
    
    def calculate_threat(self, situation: SituationScore) -> float:
        """
        根据态势计算威胁值
        威胁值与态势值负相关
        
        Args:
            situation: 我机对敌机的态势得分
            
        Returns:
            float: 威胁值 [0, 1]，1表示最大威胁
        """
        # 简单反转：态势高则威胁低
        threat = 1.0 - situation.total
        return np.clip(threat, 0.0, 1.0)
