# envs/JSBSim/tasks/pure_maneuvers.py
import numpy as np
import math
from typing import Tuple, Dict, Any
from dataclasses import dataclass
import logging


@dataclass
class ManeuverState:
    """机动状态"""
    time: float = 0.0
    phase: str = "init"
    target_heading: float = 0.0
    initial_heading: float = 0.0
    crank_angle: float = 45.0
    turn_rate: float = 3.0
    hold_time: float = 20.0


class PureManeuvers:
    """纯机动函数库 - 老师要求的函数封装"""

    # envs/JSBSim/tasks/pure_maneuvers.py

    @staticmethod
    def crank_maneuver(time_sec: float,
                       initial_heading_deg: float = 0.0,
                       crank_angle_deg: float = 45.0,
                       turn_rate_deg_per_sec: float = 3.0,
                       hold_time_sec: float = 20.0):
        """
        Crank机动 - 纯函数实现，返回phase信息

        Args:
            time_sec: 当前时间 (秒)
            initial_heading_deg: 初始航向角 (度)
            crank_angle_deg: Crank角度 (度)
            turn_rate_deg_per_sec: 转弯率 (度/秒)
            hold_time_sec: 保持时间 (秒)

        Returns:
            (phase, target_heading, target_roll) - 字符串, 度, 度
        """

        # 计算转弯时间
        turn_time = abs(crank_angle_deg) / turn_rate_deg_per_sec

        if time_sec <= turn_time:
            # 阶段1：转弯到Crank角度
            phase = "TURN_TO_CRANK"
            progress = time_sec / turn_time
            target_heading = initial_heading_deg + crank_angle_deg * progress
            target_roll = 0.0  # 保持平飞

        elif time_sec <= turn_time + hold_time_sec:
            # 阶段2：保持Crank角度
            phase = "HOLD_CRANK"
            target_heading = initial_heading_deg + crank_angle_deg
            target_roll = 0.0  # 保持平飞

        else:
            # 阶段3：返回初始航向
            phase = "RETURN_TO_INITIAL"
            return_time = time_sec - turn_time - hold_time_sec
            if return_time <= turn_time:
                progress = return_time / turn_time
                target_heading = (initial_heading_deg + crank_angle_deg) - crank_angle_deg * progress
            else:
                target_heading = initial_heading_deg
            target_roll = 0.0

        # 保持航向角在0-360度范围内
        target_heading = target_heading % 360

        return phase, target_heading, target_roll

    @staticmethod
    def beam_maneuver(time_sec: float,
                      initial_heading_deg: float = 0.0,
                      beam_angle_deg: float = 90.0,
                      turn_rate_deg_per_sec: float = 5.0,
                      hold_time_sec: float = 15.0) -> Tuple[float, float, float]:
        """
        Beam机动 - 垂直于威胁方向
        """
        turn_time = abs(beam_angle_deg) / turn_rate_deg_per_sec

        if time_sec <= turn_time:
            progress = time_sec / turn_time
            target_heading = initial_heading_deg + beam_angle_deg * progress
        else:
            target_heading = initial_heading_deg + beam_angle_deg

        target_heading = target_heading % 360
        target_velocity = 850.0  # 高速机动
        target_altitude = 6096.0

        return target_heading, target_velocity, target_altitude

    @staticmethod
    def notch_maneuver(time_sec: float,
                       initial_heading_deg: float = 0.0,
                       notch_angle_deg: float = 120.0,
                       turn_rate_deg_per_sec: float = 4.0,
                       hold_time_sec: float = 10.0) -> Tuple[float, float, float]:
        """
        Notch机动 - 逃离机动
        """
        turn_time = abs(notch_angle_deg) / turn_rate_deg_per_sec

        if time_sec <= turn_time:
            progress = time_sec / turn_time
            target_heading = initial_heading_deg + notch_angle_deg * progress
        else:
            target_heading = initial_heading_deg + notch_angle_deg

        target_heading = target_heading % 360
        target_velocity = 900.0  # 最高速度逃离
        target_altitude = 6096.0

        return target_heading, target_velocity, target_altitude

    @staticmethod
    def get_maneuver_description(maneuver_name: str) -> str:
        """获取机动描述"""
        descriptions = {
            "crank": "Crank机动 - 斜向机动，保持雷达锁定同时避开威胁",
            "beam": "Beam机动 - 90度横向机动，最大化多普勒效应",
            "notch": "Notch机动 - 逃离机动，利用地面杂波隐蔽"
        }
        return descriptions.get(maneuver_name, "未知机动")