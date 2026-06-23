"""
战术辅助函数 - 从tacticalTemplateProject移植
包含经过验证的机动控制函数
"""
import logging
import numpy as np
from envs.JSBSim.core.catalog import Catalog as c


def normalize_angle_diff(angle_diff):
    """标准化角度差值到[-180, 180]范围"""
    while angle_diff > 180:
        angle_diff -= 360
    while angle_diff < -180:
        angle_diff += 360
    return angle_diff


def maintain_heading_precise(env, agent_id, target_heading, tolerance=5.0):
    """
    精确保持航向 - 从原战术模板移植
    
    Args:
        env: 环境
        agent_id: 飞机ID
        target_heading: 目标航向（度）
        tolerance: 容差（度）
    
    Returns:
        (altitude_cmd, heading_cmd, velocity_cmd)
    """
    current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
    heading_diff = normalize_angle_diff(target_heading - current_heading)
    
    if abs(heading_diff) > tolerance:
        # 需要转向
        if heading_diff < 0:
            return 7, 6, 3  # 保持高度 + 左转30° + 保持速度
        else:
            return 7, 10, 3  # 保持高度 + 右转30° + 保持速度
    else:
        # 保持航向
        return 7, 8, 3  # 保持高度 + 保持航向 + 保持速度


def get_wingman_phase_by_distance_old_style(distance, tactical_distances, wingman_delay):
    """
    僚机独立的战术阶段判断 - 体现时间线滞后
    从原战术模板移植，适配5阶段系统到8阶段系统
    
    原来的5阶段：NLT_MELD, MELD_MTR, MTR_TR, TR_DOR, DOR_DR
    现在的8阶段：NLT_MELD, MELD_MTR, MTR_LR, LR_TR, TR_DOR, DOR_DR, DR_MAR, BEYOND_MAR
    
    映射关系：
    - NLT_MELD -> NLT_MELD
    - MELD_MTR -> MELD_MTR
    - MTR_TR -> MTR_LR + LR_TR (合并)
    - TR_DOR -> TR_DOR
    - DOR_DR -> DOR_DR + DR_MAR (合并)
    """
    from enum import Enum
    
    class TacticalPhase(Enum):
        NLT_MELD = "NLT_MELD"
        MELD_MTR = "MELD_MTR"
        MTR_LR = "MTR_LR"
        LR_TR = "LR_TR"
        TR_DOR = "TR_DOR"
        DOR_DR = "DOR_DR"
        DR_MAR = "DR_MAR"
        BEYOND_MAR = "BEYOND_MAR"
    
    # 原来的距离阈值（5阶段系统）
    # NLT_MELD_min: 81km
    # MELD_MTR_min: 50km
    # MTR_TR_min: 40km
    # TR_DOR_min: 35km
    # DOR_DR_min: 14.5km
    
    # 现在的距离阈值（8阶段系统）
    # MELD: 100km
    # MTR: 80km
    # LR: 78km
    # TR: 75km
    # DOR: 70km
    # DR: 65km
    # MAR: 40km
    
    # 僚机使用滞后距离判断阶段
    if distance > tactical_distances.get('MELD', 100000):
        return TacticalPhase.NLT_MELD
    elif distance > tactical_distances.get('MTR', 80000):
        return TacticalPhase.MELD_MTR
    elif distance > tactical_distances.get('LR', 78000):
        return TacticalPhase.MTR_LR
    elif distance > tactical_distances.get('TR', 75000):
        return TacticalPhase.LR_TR
    elif distance > (tactical_distances.get('DOR', 70000) - wingman_delay.get('TR_DOR_delay', 4000)):
        return TacticalPhase.TR_DOR  # 70km - 4km = 66km
    elif distance > (tactical_distances.get('DR', 65000) - wingman_delay.get('DOR_DR_delay', 8000)):
        return TacticalPhase.DOR_DR  # 65km - 8km = 57km
    elif distance > tactical_distances.get('MAR', 40000):
        return TacticalPhase.DR_MAR
    else:
        return TacticalPhase.BEYOND_MAR


def calculate_formation_spacing(env, agent_id):
    """
    计算编队间距 - 并排射击特有
    
    Returns:
        间距（米）
    """
    try:
        if agent_id == "A0200":  # 僚机
            leader_id = "A0100"
        elif agent_id == "B0200":  # 敌方僚机
            leader_id = "B0100"
        else:
            return 3704.0  # 默认2海里
        
        if leader_id not in env.agents or not env.agents[leader_id].is_alive:
            return 3704.0
        
        pos1 = env.agents[agent_id].get_position()
        pos2 = env.agents[leader_id].get_position()
        spacing = np.linalg.norm(pos1 - pos2)
        
        return spacing
    except Exception as e:
        logging.warning(f"计算编队间距失败: {e}")
        return 3704.0
