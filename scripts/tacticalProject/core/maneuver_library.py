"""
机动动作库
实现所有战术机动动作
"""

import numpy as np
import logging
from typing import Tuple
from enum import Enum
from envs.JSBSim.core.catalog import Catalog as c


class ManeuverType(Enum):
    """机动类型"""
    TACTICAL_CRANK = "TACTICAL_CRANK"          # 战术Crank
    CRANK = "CRANK"                            # Crank
    LEVEL_FLIGHT = "LEVEL_FLIGHT"              # 平飞
    ACCELERATE = "ACCELERATE"                  # 加速
    DECELERATE = "DECELERATE"                  # 减速
    CLIMB = "CLIMB"                            # 爬升
    DESCEND = "DESCEND"                        # 下降
    TACTICAL_CLIMB = "TACTICAL_CLIMB"          # 战术爬升
    TACTICAL_DESCEND = "TACTICAL_DESCEND"      # 战术下降
    NOTCH_BACK = "NOTCH_BACK"                  # Notch Back
    SHORT_SKATE = "SHORT_SKATE"                # Short Skate
    BEAM = "BEAM"                              # Beam机动


class ManeuverLibrary:
    """
    机动动作库
    提供所有基本和战术机动动作
    """
    
    def __init__(self):
        """初始化机动动作库"""
        logging.info("✅ 机动动作库初始化完成")
    
    def execute_tactical_climb(
        self,
        env,
        agent_id: str,
        target_altitude_gain: float = 500.0,
        turn_angle: float = -90.0,
        accelerate: bool = True
    ) -> Tuple[int, int, int]:
        """
        战术爬升
        
        特点：爬升 + 转向 + 加速的组合机动
        用于在DOR阶段快速获得高度优势同时规避
        
        Args:
            env: 环境
            agent_id: 飞机ID
            target_altitude_gain: 目标爬升高度(m)
            turn_angle: 转向角度（正为右转，负为左转）
            accelerate: 是否加速
            
        Returns:
            (alt_cmd, hdg_cmd, vel_cmd)
        """
        # 高度指令：大幅爬升
        if target_altitude_gain >= 1000:
            alt_cmd = 14  # +1500m
        elif target_altitude_gain >= 500:
            alt_cmd = 11  # +500m
        elif target_altitude_gain >= 200:
            alt_cmd = 8   # +200m
        else:
            alt_cmd = 7   # 保持
        
        # 航向指令：根据转向角度
        if turn_angle <= -60:
            hdg_cmd = 2   # 左转90°
        elif turn_angle <= -30:
            hdg_cmd = 4   # 左转60°
        elif turn_angle <= -15:
            hdg_cmd = 6   # 左转30°
        elif turn_angle >= 60:
            hdg_cmd = 14  # 右转90°
        elif turn_angle >= 30:
            hdg_cmd = 12  # 右转60°
        elif turn_angle >= 15:
            hdg_cmd = 10  # 右转30°
        else:
            hdg_cmd = 8   # 保持航向
        
        # 速度指令
        if accelerate:
            vel_cmd = 5   # 大幅加速+100m/s
        else:
            vel_cmd = 4   # 加速+50m/s
        
        return alt_cmd, hdg_cmd, vel_cmd
    
    def execute_tactical_descend(
        self,
        env,
        agent_id: str,
        target_altitude_loss: float = 500.0,
        turn_angle: float = 90.0,
        accelerate: bool = True
    ) -> Tuple[int, int, int]:
        """
        战术下降
        
        特点：下降 + 转向 + 加速的组合机动
        用于快速脱离高威胁区域
        
        Args:
            env: 环境
            agent_id: 飞机ID
            target_altitude_loss: 目标下降高度(m)
            turn_angle: 转向角度
            accelerate: 是否加速
            
        Returns:
            (alt_cmd, hdg_cmd, vel_cmd)
        """
        # 高度指令：大幅下降
        if target_altitude_loss >= 500:
            alt_cmd = 1   # -1500m
        elif target_altitude_loss >= 300:
            alt_cmd = 4   # -500m
        elif target_altitude_loss >= 150:
            alt_cmd = 5   # -150m
        else:
            alt_cmd = 7   # 保持
        
        # 航向指令
        if turn_angle <= -60:
            hdg_cmd = 2   # 左转90°
        elif turn_angle <= -30:
            hdg_cmd = 4   # 左转60°
        elif turn_angle <= -15:
            hdg_cmd = 6   # 左转30°
        elif turn_angle >= 60:
            hdg_cmd = 14  # 右转90°
        elif turn_angle >= 30:
            hdg_cmd = 12  # 右转60°
        elif turn_angle >= 15:
            hdg_cmd = 10  # 右转30°
        else:
            hdg_cmd = 8   # 保持航向
        
        # 速度指令
        if accelerate:
            vel_cmd = 5   # 大幅加速
        else:
            vel_cmd = 4   # 加速
        
        return alt_cmd, hdg_cmd, vel_cmd
    
    def execute_notch_back(
        self,
        env,
        agent_id: str,
        direction: str = "left"
    ) -> Tuple[int, int, int]:
        """
        Notch Back机动
        
        雷达反制机动：快速转向使敌机雷达失锁
        核心：90°转向 + 下降 + 加速
        
        Args:
            env: 环境
            agent_id: 飞机ID
            direction: 转向方向 "left" 或 "right"
            
        Returns:
            (alt_cmd, hdg_cmd, vel_cmd)
        """
        # 高度指令：下降以利用地杂波
        alt_cmd = 4  # 下降-500m
        
        # 航向指令：90°急转
        if direction == "left":
            hdg_cmd = 2  # 左转90°
        else:
            hdg_cmd = 14  # 右转90°
        
        # 速度指令：大幅加速
        vel_cmd = 5  # 加速+100m/s
        
        return alt_cmd, hdg_cmd, vel_cmd
    
    def execute_beam_maneuver(
        self,
        env,
        agent_id: str,
        enemy_bearing: float
    ) -> Tuple[int, int, int]:
        """
        Beam机动（三九线机动）
        
        目标：将敌机置于自身3/9点方向（侧向90°）
        减少雷达截面积，规避导弹
        
        Args:
            env: 环境
            agent_id: 飞机ID
            enemy_bearing: 敌机方位角
            
        Returns:
            (alt_cmd, hdg_cmd, vel_cmd)
        """
        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
        
        # 计算目标航向：使敌机位于3点或9点方向
        beam_left = (enemy_bearing - 90) % 360   # 3点方向
        beam_right = (enemy_bearing + 90) % 360  # 9点方向
        
        # 选择较近的目标航向
        diff_left = abs(self._normalize_angle(beam_left - current_heading))
        diff_right = abs(self._normalize_angle(beam_right - current_heading))
        
        if diff_left < diff_right:
            target_heading = beam_left
            turn_direction = "left" if self._normalize_angle(beam_left - current_heading) < 0 else "right"
        else:
            target_heading = beam_right
            turn_direction = "right" if self._normalize_angle(beam_right - current_heading) > 0 else "left"
        
        # 计算转向角度
        heading_diff = abs(self._normalize_angle(target_heading - current_heading))
        
        # 航向指令
        if heading_diff > 75:
            hdg_cmd = 2 if turn_direction == "left" else 14  # 90°转
        elif heading_diff > 45:
            hdg_cmd = 4 if turn_direction == "left" else 12  # 60°转
        elif heading_diff > 20:
            hdg_cmd = 6 if turn_direction == "left" else 10  # 30°转
        elif heading_diff > 10:
            hdg_cmd = 7 if turn_direction == "left" else 9   # 15°转
        else:
            hdg_cmd = 8  # 保持
        
        # 高度：保持
        alt_cmd = 7
        
        # 速度：中速保持机动性
        vel_cmd = 3  # 保持速度
        
        return alt_cmd, hdg_cmd, vel_cmd
    
    def execute_crank(
        self,
        env,
        agent_id: str,
        crank_angle: float = 30.0,
        direction: str = "auto"
    ) -> Tuple[int, int, int]:
        """
        Crank机动
        
        在保持雷达照射的同时侧向偏离，为规避做准备
        
        Args:
            env: 环境
            agent_id: 飞机ID
            crank_angle: Crank角度（度）
            direction: "left", "right", 或 "auto"
            
        Returns:
            (alt_cmd, hdg_cmd, vel_cmd)
        """
        # 自动选择方向（基于战术需要）
        if direction == "auto":
            # 简化：默认右侧Crank
            direction = "right"
        
        # 航向指令
        if crank_angle >= 45:
            hdg_cmd = 12 if direction == "right" else 4  # 60°
        elif crank_angle >= 20:
            hdg_cmd = 10 if direction == "right" else 6  # 30°
        else:
            hdg_cmd = 9 if direction == "right" else 7   # 15°
        
        # 高度：保持
        alt_cmd = 7
        
        # 速度：保持
        vel_cmd = 3
        
        return alt_cmd, hdg_cmd, vel_cmd
    
    def execute_short_skate(
        self,
        env,
        agent_id: str,
        direction: str = "left",
        current_time: float = 0
    ) -> Tuple[int, int, int]:
        """
        Short Skate机动
        
        两段式机动：Crank转向 + Turn Cold返航
        
        Args:
            env: 环境
            agent_id: 飞机ID
            direction: 转向方向
            current_time: 当前时间
            
        Returns:
            (alt_cmd, hdg_cmd, vel_cmd)
        """
        # 简化实现：直接执行大角度转向
        if direction == "left":
            hdg_cmd = 2  # 左转90°
        else:
            hdg_cmd = 14  # 右转90°
        
        alt_cmd = 7  # 保持高度
        vel_cmd = 5  # 加速
        
        return alt_cmd, hdg_cmd, vel_cmd
    
    def _normalize_angle(self, angle: float) -> float:
        """将角度规范化到[-180, 180]"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
