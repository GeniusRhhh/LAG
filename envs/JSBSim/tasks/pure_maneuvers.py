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
    """纯机动函数库"""

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
                      hold_time_sec: float = 15.0):
        """
        Beam机动 - 横向态势机动，消耗导弹动能

        特点：
        - 90度横向转弯，形成3/9线态势
        - 保持高度和速度，专注于横向机动
        - 使相对径向速度接近0，雷达多普勒速度为0
        - 迫使导弹大角度转弯，消耗其动能

        Args:
            time_sec: 当前时间 (秒)
            initial_heading_deg: 初始航向角 (度)
            beam_angle_deg: Beam角度 (度，通常为90度)
            turn_rate_deg_per_sec: 转弯率 (度/秒)
            hold_time_sec: 保持横向飞行时间 (秒)

        Returns:
            (phase, target_heading, target_roll) - 字符串, 度, 度
        """

        # 计算转弯时间
        turn_time = abs(beam_angle_deg) / turn_rate_deg_per_sec

        if time_sec <= turn_time:
            # 阶段1：快速转向横向态势
            phase = "TURN_TO_BEAM"
            progress = time_sec / turn_time
            target_heading = initial_heading_deg + beam_angle_deg * progress
            target_roll = 0.0

        elif time_sec <= turn_time + hold_time_sec:
            # 阶段2：保持横向态势，消耗导弹能量
            phase = "HOLD_BEAM"
            target_heading = initial_heading_deg + beam_angle_deg
            target_roll = 0.0

        else:
            # 阶段3：恢复原航向（可选）
            phase = "RETURN_TO_INITIAL"
            return_time = time_sec - turn_time - hold_time_sec
            if return_time <= turn_time:
                progress = return_time / turn_time
                target_heading = (initial_heading_deg + beam_angle_deg) - beam_angle_deg * progress
            else:
                target_heading = initial_heading_deg
            target_roll = 0.0

        # 保持航向角在0-360度范围内
        target_heading = target_heading % 360

        return phase, target_heading, target_roll

    @staticmethod
    def notch_maneuver(time_sec: float,
                       initial_heading_deg: float = 0.0,
                       initial_altitude_ft: float = 20000.0,
                       notch_angle_deg: float = 90.0,
                       turn_rate_deg_per_sec: float = 4.0,
                       descent_rate_ft_per_sec: float = 60.0,
                       descent_time_sec: float = 5.0,
                       hold_time_sec: float = 15.0):
        """
        Notch机动 - 地面杂波隐蔽机动，打断雷达锁定

        特点：
        - 温和下降到较低高度（模拟进入地面杂波区）
        - 然后横向转弯
        - 目标是打断敌方雷达锁定
        - 注意：采用安全的下降率，避免失控

        Returns:
            (phase, target_heading, target_altitude, target_roll) - 4个参数
        """

        # 计算转弯时间
        turn_time = abs(notch_angle_deg) / turn_rate_deg_per_sec

        # 计算目标下降高度
        max_descent = descent_rate_ft_per_sec * descent_time_sec
        target_low_altitude = initial_altitude_ft - max_descent

        # 不低于安全高度
        min_safe_altitude = 9000.0
        if target_low_altitude < min_safe_altitude:
            target_low_altitude = min_safe_altitude
            # 重新计算实际下降率
            actual_descent = initial_altitude_ft - target_low_altitude
            if descent_time_sec > 0:
                descent_rate_ft_per_sec = actual_descent / descent_time_sec

        if time_sec <= descent_time_sec:
            # 阶段1：温和下降，模拟进入地面杂波区
            phase = "DESCENT_TO_CLUTTER"
            target_heading = initial_heading_deg  # 保持原航向

            # 线性插值，平滑下降
            progress = time_sec / descent_time_sec
            target_altitude = initial_altitude_ft - (descent_rate_ft_per_sec * time_sec)
            target_altitude = max(target_altitude, min_safe_altitude)  # 双重保险
            target_roll = 0.0

        elif time_sec <= descent_time_sec + turn_time:
            # 阶段2：在较低空进行横向转弯
            phase = "TURN_IN_CLUTTER"
            turn_progress = (time_sec - descent_time_sec) / turn_time
            target_heading = initial_heading_deg + notch_angle_deg * turn_progress
            target_altitude = target_low_altitude  # 保持下降后的高度
            target_roll = 0.0

        elif time_sec <= descent_time_sec + turn_time + hold_time_sec:
            # 阶段3：在较低空保持横向飞行
            phase = "HOLD_IN_CLUTTER"
            target_heading = initial_heading_deg + notch_angle_deg
            target_altitude = target_low_altitude  # 保持下降后的高度
            target_roll = 0.0

        else:
            # 阶段4：温和爬升并返回
            phase = "CLIMB_AND_RETURN"
            return_time = time_sec - descent_time_sec - turn_time - hold_time_sec

            # 温和的爬升回到原高度
            climb_time = 10.0  # 给10秒时间爬升回去
            if return_time <= climb_time:
                progress = return_time / climb_time
                # 同时回转航向和爬升高度
                if return_time <= turn_time:
                    heading_progress = return_time / turn_time
                    target_heading = (initial_heading_deg + notch_angle_deg) - notch_angle_deg * heading_progress
                else:
                    target_heading = initial_heading_deg

                # 温和爬升
                altitude_recovery = (initial_altitude_ft - target_low_altitude) * progress
                target_altitude = target_low_altitude + altitude_recovery
            else:
                target_heading = initial_heading_deg
                target_altitude = initial_altitude_ft

            target_roll = 0.0

        # 保持航向角在0-360度范围内
        target_heading = target_heading % 360

        # 最终安全检查
        target_altitude = max(target_altitude, min_safe_altitude)

        return phase, target_heading, target_altitude, target_roll

    @staticmethod
    def get_maneuver_description(maneuver_name: str) -> str:
        """获取机动描述"""
        descriptions = {
            "crank": "Crank机动 - 斜向机动，保持雷达锁定同时避开威胁",
            "beam": "Beam机动 - 90度横向机动，最大化多普勒效应",
            "notch": "Notch机动 - 逃离机动，利用地面杂波隐蔽"
        }
        return descriptions.get(maneuver_name, "未知机动")