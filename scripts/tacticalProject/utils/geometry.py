"""
几何计算工具函数
"""
import numpy as np
from typing import Tuple, Dict


def calculate_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """
    计算两点之间的欧氏距离
    
    Args:
        pos1: 位置1 [x, y, z] (m)
        pos2: 位置2 [x, y, z] (m)
    
    Returns:
        距离 (km)
    """
    return np.linalg.norm(pos1 - pos2) / 1000.0


def calculate_horizontal_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """
    计算水平距离（忽略高度）
    
    Args:
        pos1: 位置1 [x, y, z] (m)
        pos2: 位置2 [x, y, z] (m)
    
    Returns:
        水平距离 (km)
    """
    return np.linalg.norm(pos1[:2] - pos2[:2]) / 1000.0


def calculate_aspect_angle(my_pos: np.ndarray, my_heading: float,
                           target_pos: np.ndarray, target_heading: float) -> float:
    """
    计算目标相对于我机的进入角（Aspect Angle）
    
    Args:
        my_pos: 我机位置 [x, y, z] (m)
        my_heading: 我机航向 (度)
        target_pos: 目标位置 [x, y, z] (m)
        target_heading: 目标航向 (度)
    
    Returns:
        进入角 (度)，0度表示目标正对我机，180度表示目标背离我机
    """
    # 计算目标到我机的方向向量
    direction = my_pos[:2] - target_pos[:2]
    direction_angle = np.arctan2(direction[1], direction[0]) * 180 / np.pi
    
    # 计算进入角（目标航向与目标到我机方向的夹角）
    aspect_angle = abs(normalize_angle(target_heading - direction_angle))
    
    return aspect_angle


def calculate_angle_off(my_pos: np.ndarray, my_heading: float,
                        target_pos: np.ndarray) -> float:
    """
    计算我机相对于目标的离轴角（Angle Off）
    
    Args:
        my_pos: 我机位置 [x, y, z] (m)
        my_heading: 我机航向 (度)
        target_pos: 目标位置 [x, y, z] (m)
    
    Returns:
        离轴角 (度)，0度表示我机正对目标
    """
    # 计算我机到目标的方向向量
    direction = target_pos[:2] - my_pos[:2]
    direction_angle = np.arctan2(direction[1], direction[0]) * 180 / np.pi
    
    # 计算离轴角
    angle_off = abs(normalize_angle(my_heading - direction_angle))
    
    return angle_off


def normalize_angle(angle: float) -> float:
    """
    将角度归一化到[-180, 180]范围
    
    Args:
        angle: 角度 (度)
    
    Returns:
        归一化后的角度 (度)
    """
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def calculate_closure_rate(my_pos: np.ndarray, my_vel: np.ndarray,
                           target_pos: np.ndarray, target_vel: np.ndarray) -> float:
    """
    计算接近率（Closure Rate）
    
    Args:
        my_pos: 我机位置 [x, y, z] (m)
        my_vel: 我机速度 [vx, vy, vz] (m/s)
        target_pos: 目标位置 [x, y, z] (m)
        target_vel: 目标速度 [vx, vy, vz] (m/s)
    
    Returns:
        接近率 (m/s)，正值表示接近，负值表示远离
    """
    # 计算相对位置向量
    rel_pos = target_pos - my_pos
    rel_pos_norm = np.linalg.norm(rel_pos)
    
    if rel_pos_norm < 1e-6:
        return 0.0
    
    # 计算相对速度向量
    rel_vel = target_vel - my_vel
    
    # 接近率 = 相对速度在相对位置方向上的投影
    closure_rate = -np.dot(rel_vel, rel_pos) / rel_pos_norm
    
    return closure_rate


def calculate_altitude_difference(my_pos: np.ndarray, target_pos: np.ndarray) -> float:
    """
    计算高度差
    
    Args:
        my_pos: 我机位置 [x, y, z] (m)
        target_pos: 目标位置 [x, y, z] (m)
    
    Returns:
        高度差 (m)，正值表示我机更高
    """
    return my_pos[2] - target_pos[2]


def calculate_speed(vel: np.ndarray) -> float:
    """
    计算速度大小
    
    Args:
        vel: 速度向量 [vx, vy, vz] (m/s)
    
    Returns:
        速度大小 (m/s)
    """
    return np.linalg.norm(vel)


def calculate_heading_from_velocity(vel: np.ndarray) -> float:
    """
    从速度向量计算航向
    
    Args:
        vel: 速度向量 [vx, vy, vz] (m/s)
    
    Returns:
        航向 (度)，0度为北，90度为东
    """
    return np.arctan2(vel[1], vel[0]) * 180 / np.pi


def is_closing(my_pos: np.ndarray, my_vel: np.ndarray,
               target_pos: np.ndarray, target_vel: np.ndarray) -> bool:
    """
    判断是否正在接近目标
    
    Args:
        my_pos: 我机位置 [x, y, z] (m)
        my_vel: 我机速度 [vx, vy, vz] (m/s)
        target_pos: 目标位置 [x, y, z] (m)
        target_vel: 目标速度 [vx, vy, vz] (m/s)
    
    Returns:
        True表示正在接近
    """
    return calculate_closure_rate(my_pos, my_vel, target_pos, target_vel) > 0


def calculate_relative_geometry(my_state: Dict, target_state: Dict) -> Dict:
    """
    计算相对几何关系
    
    Args:
        my_state: 我机状态字典
            - 'position': np.ndarray [x, y, z] (m)
            - 'velocity': np.ndarray [vx, vy, vz] (m/s)
            - 'heading': float (度)
        target_state: 目标状态字典
            - 'position': np.ndarray [x, y, z] (m)
            - 'velocity': np.ndarray [vx, vy, vz] (m/s)
            - 'heading': float (度)
    
    Returns:
        相对几何关系字典
            - 'distance': float (km)
            - 'horizontal_distance': float (km)
            - 'altitude_diff': float (m)
            - 'aspect_angle': float (度)
            - 'angle_off': float (度)
            - 'closure_rate': float (m/s)
            - 'is_closing': bool
    """
    return {
        'distance': calculate_distance(my_state['position'], target_state['position']),
        'horizontal_distance': calculate_horizontal_distance(my_state['position'], target_state['position']),
        'altitude_diff': calculate_altitude_difference(my_state['position'], target_state['position']),
        'aspect_angle': calculate_aspect_angle(my_state['position'], my_state['heading'],
                                               target_state['position'], target_state['heading']),
        'angle_off': calculate_angle_off(my_state['position'], my_state['heading'],
                                         target_state['position']),
        'closure_rate': calculate_closure_rate(my_state['position'], my_state['velocity'],
                                               target_state['position'], target_state['velocity']),
        'is_closing': is_closing(my_state['position'], my_state['velocity'],
                                target_state['position'], target_state['velocity']),
    }
