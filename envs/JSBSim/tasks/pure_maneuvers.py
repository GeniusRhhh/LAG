# envs/JSBSim/tasks/pure_maneuvers.py
import numpy as np
import math
from typing import Tuple, Dict, Any, List, Optional
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


@dataclass
class BasicManeuver:
    """基础机动定义"""
    name: str
    duration: float
    target_heading: float
    target_altitude: float
    target_velocity: float
    target_roll: float
    description: str


@dataclass
class CompositeManeuver:
    """组合机动定义"""
    name: str
    maneuvers: List[BasicManeuver]
    total_duration: float
    description: str


class BasicManeuvers:
    """基础机动库 - 您要求的小机动动作"""

    @staticmethod
    def level_flight(time_sec: float, duration: float = 10.0):
        """平飞 - 保持当前高度、航向、速度"""
        if time_sec <= duration:
            return "LEVEL_FLIGHT", None, None, None, 0.0
        return None, None, None, None, 0.0

    @staticmethod
    def accelerate(time_sec: float, duration: float = 5.0, velocity_increase: float = 50.0):
        """加速 - 增加速度"""
        if time_sec <= duration:
            return "ACCELERATE", None, None, velocity_increase, 0.0
        return None, None, None, None, 0.0

    @staticmethod
    def decelerate(time_sec: float, duration: float = 5.0, velocity_decrease: float = 50.0):
        """减速 - 减少速度"""
        if time_sec <= duration:
            return "DECELERATE", None, None, -velocity_decrease, 0.0
        return None, None, None, None, 0.0

    @staticmethod
    def turn(time_sec: float,
             initial_heading: float,
             turn_angle: float = 45.0,
             turn_rate: float = 3.0):
        """转弯 - 改变航向"""
        turn_time = abs(turn_angle) / turn_rate

        if time_sec <= turn_time:
            progress = time_sec / turn_time
            target_heading = initial_heading + turn_angle * progress
            return "TURN", target_heading, None, None, 0.0
        return None, None, None, None, 0.0

    @staticmethod
    def pull_up(time_sec: float,
                duration: float = 8.0,
                altitude_gain: float = 1000.0):
        """拉起 - 爬升"""
        if time_sec <= duration:
            return "PULL_UP", None, altitude_gain, None, 0.0
        return None, None, None, None, 0.0

    @staticmethod
    def dive(time_sec: float,
             duration: float = 8.0,
             altitude_loss: float = 1000.0):
        """俯冲 - 下降"""
        if time_sec <= duration:
            return "DIVE", None, -altitude_loss, None, 0.0
        return None, None, None, None, 0.0

    @staticmethod
    def diagonal_flight(time_sec: float,
                        duration: float = 10.0,
                        heading_change: float = 30.0,
                        altitude_change: float = 500.0):
        """斜直飞 - 同时改变航向和高度"""
        if time_sec <= duration:
            progress = time_sec / duration
            target_heading = heading_change * progress
            target_altitude = altitude_change * progress
            return "DIAGONAL_FLIGHT", target_heading, target_altitude, None, 0.0
        return None, None, None, None, 0.0

    @staticmethod
    def roll(time_sec: float,
             duration: float = 3.0,
             roll_angle: float = 45.0):
        """滚转 - 改变滚转角"""
        if time_sec <= duration:
            return "ROLL", None, None, None, roll_angle
        return None, None, None, None, 0.0

    @staticmethod
    def turn_pull_up(time_sec: float,
                     initial_heading: float,
                     turn_angle: float = 45.0,
                     turn_rate: float = 3.0,
                     altitude_gain: float = 1000.0):
        """转弯拉起 - 同时转弯和爬升"""
        turn_time = abs(turn_angle) / turn_rate

        if time_sec <= turn_time:
            # 同时进行转弯和爬升
            progress = time_sec / turn_time
            target_heading = initial_heading + turn_angle * progress
            target_altitude = altitude_gain * progress
            return "TURN_PULL_UP", target_heading, target_altitude, None, 0.0
        return None, None, None, None, 0.0

    @staticmethod
    def turn_dive(time_sec: float,
                  initial_heading: float,
                  turn_angle: float = 45.0,
                  turn_rate: float = 3.0,
                  altitude_loss: float = 1000.0):
        """转弯俯冲 - 同时转弯和下降"""
        turn_time = abs(turn_angle) / turn_rate

        if time_sec <= turn_time:
            # 同时进行转弯和下降
            progress = time_sec / turn_time
            target_heading = initial_heading + turn_angle * progress
            target_altitude = -altitude_loss * progress
            return "TURN_DIVE", target_heading, target_altitude, None, 0.0
        return None, None, None, None, 0.0


class ManeuverComposer:
    """机动组合器 - 将基础机动组合成复杂战术"""

    def __init__(self):
        self.basic_maneuvers = BasicManeuvers()
        self.composite_maneuvers = {}
        self._setup_tactical_templates()

    def _setup_tactical_templates(self):
        """设置战术模板"""
        # 脱离机动模板
        self.composite_maneuvers["escape"] = CompositeManeuver(
            name="脱离机动",
            maneuvers=[
                BasicManeuver("急转弯", 5.0, 90.0, 0.0, 0.0, 0.0, "快速转向"),
                BasicManeuver("加速", 3.0, 0.0, 0.0, 100.0, 0.0, "加速脱离"),
                BasicManeuver("平飞", 10.0, 0.0, 0.0, 0.0, 0.0, "稳定飞行")
            ],
            total_duration=18.0,
            description="快速脱离威胁区域"
        )

        # 侧跃升拉起俯冲攻击模板
        self.composite_maneuvers["attack"] = CompositeManeuver(
            name="侧跃升拉起俯冲攻击",
            maneuvers=[
                BasicManeuver("侧跃升", 8.0, 45.0, 2000.0, 0.0, 0.0, "侧向爬升"),
                BasicManeuver("保持高度", 5.0, 0.0, 0.0, 0.0, 0.0, "保持高度"),
                BasicManeuver("俯冲攻击", 10.0, -45.0, -2000.0, 50.0, 0.0, "俯冲攻击")
            ],
            total_duration=23.0,
            description="经典的攻击机动"
        )

        # 防御机动模板
        self.composite_maneuvers["defense"] = CompositeManeuver(
            name="防御机动",
            maneuvers=[
                BasicManeuver("滚转", 2.0, 0.0, 0.0, 0.0, 45.0, "快速滚转"),
                BasicManeuver("俯冲", 5.0, 0.0, -1000.0, 0.0, 0.0, "俯冲躲避"),
                BasicManeuver("转弯", 8.0, 90.0, 0.0, 0.0, 0.0, "转向脱离"),
                BasicManeuver("爬升", 6.0, 0.0, 1000.0, 0.0, 0.0, "恢复高度")
            ],
            total_duration=21.0,
            description="综合防御机动"
        )

    def execute_composite_maneuver(self,
                                   maneuver_name: str,
                                   time_sec: float,
                                   initial_heading: float = 0.0,
                                   initial_altitude: float = 20000.0):
        """执行组合机动"""
        if maneuver_name not in self.composite_maneuvers:
            return None, None, None, None, 0.0

        composite = self.composite_maneuvers[maneuver_name]

        if time_sec > composite.total_duration:
            return None, None, None, None, 0.0

        # 找到当前应该执行的机动
        current_time = 0.0
        for maneuver in composite.maneuvers:
            if time_sec <= current_time + maneuver.duration:
                # 执行这个基础机动
                local_time = time_sec - current_time

                if maneuver.name == "急转弯":
                    return self.basic_maneuvers.turn(local_time, initial_heading, maneuver.target_heading, 5.0)
                elif maneuver.name == "加速":
                    return self.basic_maneuvers.accelerate(local_time, maneuver.duration, maneuver.target_velocity)
                elif maneuver.name == "平飞":
                    return self.basic_maneuvers.level_flight(local_time, maneuver.duration)
                elif maneuver.name == "侧跃升":
                    return self.basic_maneuvers.turn_pull_up(local_time, initial_heading, maneuver.target_heading, 3.0,
                                                             maneuver.target_altitude)
                elif maneuver.name == "保持高度":
                    return self.basic_maneuvers.level_flight(local_time, maneuver.duration)
                elif maneuver.name == "俯冲攻击":
                    return self.basic_maneuvers.turn_dive(local_time, initial_heading, maneuver.target_heading, 3.0,
                                                          abs(maneuver.target_altitude))
                elif maneuver.name == "滚转":
                    return self.basic_maneuvers.roll(local_time, maneuver.duration, maneuver.target_roll)
                elif maneuver.name == "俯冲":
                    return self.basic_maneuvers.dive(local_time, maneuver.duration, abs(maneuver.target_altitude))
                elif maneuver.name == "转弯":
                    return self.basic_maneuvers.turn(local_time, initial_heading, maneuver.target_heading, 3.0)
                elif maneuver.name == "爬升":
                    return self.basic_maneuvers.pull_up(local_time, maneuver.duration, maneuver.target_altitude)

            current_time += maneuver.duration

        return None, None, None, None, 0.0

    def create_custom_maneuver(self, maneuvers: List[BasicManeuver], name: str, description: str = ""):
        """创建自定义组合机动"""
        total_duration = sum(m.duration for m in maneuvers)
        composite = CompositeManeuver(name, maneuvers, total_duration, description)
        self.composite_maneuvers[name] = composite
        return composite


class PureManeuvers:
    """纯机动函数库 - 现有的大机动"""

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
            "notch": "Notch机动 - 逃离机动，利用地面杂波隐蔽",
            # 基础机动
            "level_flight": "平飞 - 保持当前高度、航向、速度",
            "accelerate": "加速 - 增加速度",
            "decelerate": "减速 - 减少速度",
            "turn": "转弯 - 改变航向",
            "pull_up": "拉起 - 爬升",
            "dive": "俯冲 - 下降",
            "diagonal_flight": "斜直飞 - 同时改变航向和高度",
            "roll": "滚转 - 改变滚转角",
            "turn_pull_up": "转弯拉起 - 同时转弯和爬升",
            "turn_dive": "转弯俯冲 - 同时转弯和下降",
            # 组合机动
            "escape": "脱离机动 - 快速脱离威胁区域",
            "attack": "侧跃升拉起俯冲攻击 - 经典的攻击机动",
            "defense": "防御机动 - 综合防御机动"
        }
        return descriptions.get(maneuver_name, "未知机动")