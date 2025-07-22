import json
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
    """基础机动库"""

    @staticmethod
    def level_flight(time_sec: float, duration: float = 10.0, current_altitude: float = 0.0,
                     current_velocity: float = 0.0):
        """平飞 - 保持当前高度、航向、速度"""
        if time_sec <= duration:
            return "LEVEL_FLIGHT", None, current_altitude, 0.0, 0.0
        return None, None, None, None, 0.0

    @staticmethod
    def accelerate(time_sec: float, current_velocity: float, duration: float = 5.0, velocity_increase: float = 50.0,
                   max_velocity: float = 350.0):
        """加速 - 在duration时间内完成velocity_increase的速度增加"""
        if time_sec <= duration:
            progress = time_sec / duration
            smooth_progress = 3 * progress ** 2 - 2 * progress ** 3
            velocity_offset = velocity_increase * smooth_progress

            # 确保不超过最大速度
            if current_velocity + velocity_offset > max_velocity:
                velocity_offset = max_velocity - current_velocity

            return "ACCELERATE", None, None, velocity_offset, 0.0
        return None, None, None, None, 0.0

    @staticmethod
    def decelerate(time_sec: float, current_velocity: float, duration: float = 5.0, velocity_decrease: float = 50.0,
                   min_velocity: float = 150.0):
        """减速 - 在duration时间内完成velocity_decrease的速度减少，不改变航向"""
        if time_sec <= duration:
            progress = time_sec / duration
            smooth_progress = 3 * progress ** 2 - 2 * progress ** 3
            velocity_offset = -velocity_decrease * smooth_progress

            # 确保不低于最小速度
            if current_velocity + velocity_offset < min_velocity:
                velocity_offset = min_velocity - current_velocity

            # 修复：减速时不改变航向和高度
            return "DECELERATE", None, None, velocity_offset, 0.0
        return None, None, None, None, 0.0

    @staticmethod
    def turn(time_sec: float, initial_heading: float, turn_angle: float = 45.0, turn_rate: float = 3.0):
        """转弯 - 根据turn_rate计算时间，在该时间内完成turn_angle的转弯"""
        turn_time = abs(turn_angle) / turn_rate
        if time_sec <= turn_time:
            progress = time_sec / turn_time
            # 使用更平滑的曲线确保精确到达目标角度
            smooth_progress = progress
            target_heading = initial_heading + turn_angle * smooth_progress

            # 计算滚转角度
            target_roll = min(30.0, abs(turn_angle) / 3) * (1 if turn_angle > 0 else -1) * math.sin(progress * math.pi)
            return "TURN", target_heading, None, None, target_roll
        else:
            # 确保精确到达目标角度
            final_heading = initial_heading + turn_angle
            return "TURN_COMPLETE", final_heading, None, None, 0.0

    @staticmethod
    def pull_up(time_sec: float, initial_altitude: float, duration: float = 8.0, altitude_gain: float = 1000.0):
        """拉起 - 严格在duration时间内完成altitude_gain的高度增加"""
        if time_sec <= duration:
            progress = time_sec / duration
            smooth_progress = 3 * progress ** 2 - 2 * progress ** 3
            target_altitude = initial_altitude + altitude_gain * smooth_progress

            # 爬升时速度补偿
            current_climb_rate = altitude_gain / duration
            velocity_compensation = max(0, -current_climb_rate * 0.1)

            return "PULL_UP", None, target_altitude, velocity_compensation, 0.0
        else:
            return "PULL_UP_COMPLETE", None, initial_altitude + altitude_gain, 0.0, 0.0

    @staticmethod
    def dive(time_sec: float, initial_altitude: float, duration: float = 8.0, altitude_loss: float = 1000.0,
             min_altitude: float = 3000.0):
        """俯冲 - 严格在duration时间内完成altitude_loss的高度减少"""
        target_final_altitude = max(initial_altitude - altitude_loss, min_altitude)
        actual_altitude_loss = initial_altitude - target_final_altitude

        if time_sec <= duration:
            progress = time_sec / duration
            smooth_progress = 3 * progress ** 2 - 2 * progress ** 3
            target_altitude = initial_altitude - actual_altitude_loss * smooth_progress

            # 俯冲时速度补偿（增加速度）
            current_descent_rate = actual_altitude_loss / duration
            velocity_compensation = current_descent_rate * 0.15

            return "DIVE", None, target_altitude, velocity_compensation, 0.0
        else:
            return "DIVE_COMPLETE", None, target_final_altitude, 0.0, 0.0

    @staticmethod
    def diagonal_flight(time_sec: float, initial_heading: float, initial_altitude: float,
                        duration: float = 10.0, heading_change: float = 30.0, altitude_change: float = 500.0,
                        min_altitude: float = 3000.0):
        """斜向飞行 - 严格在duration时间内同时完成heading_change和altitude_change"""
        target_final_altitude = max(initial_altitude + altitude_change, min_altitude)
        actual_altitude_change = target_final_altitude - initial_altitude

        if time_sec <= duration:
            progress = time_sec / duration
            smooth_progress = 3 * progress ** 2 - 2 * progress ** 3

            target_heading = initial_heading + heading_change * smooth_progress
            target_altitude = initial_altitude + actual_altitude_change * smooth_progress

            target_roll = min(20.0, abs(heading_change) / 4) * (1 if heading_change > 0 else -1) * math.sin(
                progress * math.pi)

            if actual_altitude_change > 0:
                velocity_compensation = -abs(actual_altitude_change) * 0.02 * progress
            else:
                velocity_compensation = abs(actual_altitude_change) * 0.02 * progress

            return "DIAGONAL_FLIGHT", target_heading, target_altitude, velocity_compensation, target_roll
        else:
            return "DIAGONAL_FLIGHT_COMPLETE", initial_heading + heading_change, target_final_altitude, 0.0, 0.0

    @staticmethod
    def circle(time_sec: float, initial_heading: float, duration: float = 20.0, radius: float = 1000.0,
               direction: str = "clockwise", turn_rate: float = 3.0):
        """盘旋 - 修复版本：真正的持续转弯"""
        if time_sec <= duration:
            # 持续以turn_rate的速度转弯
            total_degrees_turned = turn_rate * time_sec  # 总共转过的角度

            # 方向控制
            if direction == "counterclockwise":
                total_degrees_turned = -total_degrees_turned

            # 目标航向 = 初始航向 + 总转角
            target_heading = initial_heading + total_degrees_turned

            # 持续的坡度角
            bank_angle = min(30.0, turn_rate * 8)  # 根据转弯率计算坡度角
            target_roll = bank_angle if direction == "clockwise" else -bank_angle

            # 盘旋时轻微减速
            velocity_compensation = -20.0

            return "CIRCLE", target_heading, None, velocity_compensation, target_roll
        else:
            return "CIRCLE_COMPLETE", initial_heading, None, 0.0, 0.0

    @staticmethod
    def barrel_roll(time_sec: float, initial_heading: float, duration: float = 8.0,
                    roll_revolutions: float = 1.0, direction: str = "right"):
        """桶滚 - 修复版本：真正的桶滚机动"""
        if time_sec <= duration:
            progress = time_sec / duration

            # 1. 航向摆动（模拟螺旋路径）
            oscillation_freq = roll_revolutions  # 摆动频率
            heading_amplitude = 15.0  # 摆动幅度
            heading_oscillation = heading_amplitude * math.sin(2 * math.pi * oscillation_freq * progress)
            target_heading = initial_heading + heading_oscillation

            # 2. 滚转角度 - 连续滚转
            total_roll_degrees = 360.0 * roll_revolutions * progress
            if direction == "left":
                total_roll_degrees = -total_roll_degrees

            # 标准化滚转角度到[-180, 180]
            target_roll = total_roll_degrees % 360.0
            if target_roll > 180.0:
                target_roll -= 360.0

            # 3. 高度轻微变化（桶滚的上下起伏特征）
            altitude_amplitude = 80.0  # 高度变化幅度
            altitude_oscillation = altitude_amplitude * math.sin(2 * math.pi * oscillation_freq * progress)

            # 4. 桶滚时略微加速
            velocity_compensation = 15.0

            return "BARREL_ROLL", target_heading, altitude_oscillation, velocity_compensation, target_roll
        else:
            return "BARREL_ROLL_COMPLETE", initial_heading, 0.0, 0.0, 0.0


class ManeuverComposer:
    """机动组合器 - 增强版本，支持更多组合机动"""

    def __init__(self):
        self.basic_maneuvers = BasicManeuvers()
        self.composite_maneuvers = {}
        self.maneuver_states = {}  # 存储每个组合机动的状态
        self._setup_tactical_templates()

    def _setup_tactical_templates(self, custom_params=None):
        """设置战术模板 - 扩展更多组合机动"""
        if custom_params is None:
            custom_params = {}

        # 转弯拉起组合
        turn_pull_up_params = custom_params.get("turn_pull_up", {})
        turn_pull_up_maneuvers = [
            BasicManeuver(
                name="转弯",
                duration=turn_pull_up_params.get("turn_duration", 15.0),  # 增加转弯时间
                target_heading=turn_pull_up_params.get("turn_angle", 45.0),
                target_altitude=0.0,
                target_velocity=0.0,
                target_roll=0.0,
                description="转弯阶段"
            ),
            BasicManeuver(
                name="拉起",
                duration=turn_pull_up_params.get("pull_up_duration", 8.0),
                target_heading=0.0,
                target_altitude=turn_pull_up_params.get("altitude_gain", 1000.0),
                target_velocity=0.0,
                target_roll=0.0,
                description="爬升阶段"
            )
        ]
        self.composite_maneuvers["turn_pull_up"] = CompositeManeuver(
            name="转弯拉起组合",
            maneuvers=turn_pull_up_maneuvers,
            total_duration=sum(m.duration for m in turn_pull_up_maneuvers),
            description="先转弯后拉起的组合机动"
        )

        # 转弯俯冲组合
        turn_dive_params = custom_params.get("turn_dive", {})
        turn_dive_maneuvers = [
            BasicManeuver(
                name="转弯",
                duration=turn_dive_params.get("turn_duration", 15.0),  # 增加转弯时间
                target_heading=turn_dive_params.get("turn_angle", 45.0),
                target_altitude=0.0,
                target_velocity=0.0,
                target_roll=0.0,
                description="转弯阶段"
            ),
            BasicManeuver(
                name="俯冲",
                duration=turn_dive_params.get("dive_duration", 8.0),
                target_heading=0.0,
                target_altitude=turn_dive_params.get("altitude_loss", 1000.0),  # 俯冲高度损失
                target_velocity=0.0,
                target_roll=0.0,
                description="俯冲阶段"
            )
        ]
        self.composite_maneuvers["turn_dive"] = CompositeManeuver(
            name="转弯俯冲组合",
            maneuvers=turn_dive_maneuvers,
            total_duration=sum(m.duration for m in turn_dive_maneuvers),
            description="先转弯后俯冲的组合机动"
        )

        # 双转弯组合（S型机动）
        s_turn_params = custom_params.get("s_turn", {})
        s_turn_maneuvers = [
            BasicManeuver(
                name="转弯",
                duration=s_turn_params.get("first_turn_duration", 10.0),
                target_heading=s_turn_params.get("first_turn_angle", 45.0),
                target_altitude=0.0,
                target_velocity=0.0,
                target_roll=0.0,
                description="第一次转弯"
            ),
            BasicManeuver(
                name="转弯",
                duration=s_turn_params.get("second_turn_duration", 10.0),
                target_heading=s_turn_params.get("second_turn_angle", -90.0),  # 相反方向
                target_altitude=0.0,
                target_velocity=0.0,
                target_roll=0.0,
                description="第二次转弯"
            )
        ]
        self.composite_maneuvers["s_turn"] = CompositeManeuver(
            name="S型转弯组合",
            maneuvers=s_turn_maneuvers,
            total_duration=sum(m.duration for m in s_turn_maneuvers),
            description="S型连续转弯机动"
        )

        # 攻击逃离组合
        attack_escape_params = custom_params.get("attack_escape", {})
        attack_escape_maneuvers = [
            BasicManeuver(
                name="加速",
                duration=attack_escape_params.get("accelerate_duration", 5.0),
                target_heading=0.0,
                target_altitude=0.0,
                target_velocity=attack_escape_params.get("velocity_increase", 100.0),
                target_roll=0.0,
                description="加速接近"
            ),
            BasicManeuver(
                name="转弯",
                duration=attack_escape_params.get("turn_duration", 8.0),
                target_heading=attack_escape_params.get("escape_angle", 120.0),
                target_altitude=0.0,
                target_velocity=0.0,
                target_roll=0.0,
                description="转弯逃离"
            ),
            BasicManeuver(
                name="俯冲",
                duration=attack_escape_params.get("dive_duration", 6.0),
                target_heading=0.0,
                target_altitude=attack_escape_params.get("dive_altitude", 800.0),
                target_velocity=0.0,
                target_roll=0.0,
                description="俯冲加速"
            )
        ]
        self.composite_maneuvers["attack_escape"] = CompositeManeuver(
            name="攻击逃离组合",
            maneuvers=attack_escape_maneuvers,
            total_duration=sum(m.duration for m in attack_escape_maneuvers),
            description="攻击后的逃离机动"
        )

    def execute_composite_maneuver(self, maneuver_name: str, time_sec: float, initial_heading: float,
                                   initial_altitude: float, params: Dict[str, Any] = None):
        """执行组合机动 - 修复版本：正确的阶段切换和状态保持"""
        if params is None:
            params = {}

        if maneuver_name not in self.composite_maneuvers:
            logging.warning(f"组合机动 {maneuver_name} 不存在")
            return None, None, None, None, 0.0

        composite = self.composite_maneuvers[maneuver_name]

        # 获取或初始化机动状态
        state_key = f"{maneuver_name}_state"
        if state_key not in self.maneuver_states:
            self.maneuver_states[state_key] = {
                "current_phase": 0,
                "phase_start_time": 0.0,
                "total_heading_change": 0.0,
                "phase_initial_heading": initial_heading,
                "phase_initial_altitude": initial_altitude
            }

        state = self.maneuver_states[state_key]

        # 计算当前应该在哪个阶段
        elapsed_time = 0.0
        current_phase = -1

        for i, maneuver in enumerate(composite.maneuvers):
            phase_end_time = elapsed_time + maneuver.duration
            if time_sec <= phase_end_time:
                current_phase = i
                break
            elapsed_time += maneuver.duration

        # 检查是否需要切换阶段
        if current_phase != state["current_phase"] and current_phase >= 0:
            # 切换到新阶段
            state["current_phase"] = current_phase
            state["phase_start_time"] = elapsed_time

            # 更新阶段初始状态
            if current_phase == 0:
                state["phase_initial_heading"] = initial_heading
                state["phase_initial_altitude"] = initial_altitude
                state["total_heading_change"] = 0.0
            else:
                # 继承前一阶段的结果
                prev_maneuver = composite.maneuvers[current_phase - 1]
                if prev_maneuver.name == "转弯":
                    state["total_heading_change"] += prev_maneuver.target_heading
                    state["phase_initial_heading"] = initial_heading + state["total_heading_change"]
                elif prev_maneuver.name == "拉起":
                    state["phase_initial_altitude"] += prev_maneuver.target_altitude
                elif prev_maneuver.name == "俯冲":
                    state["phase_initial_altitude"] -= prev_maneuver.target_altitude

        if current_phase < 0:
            # 所有机动都完成了
            logging.info(f"✅ 组合机动 {maneuver_name} 完成")
            # 清除状态
            if state_key in self.maneuver_states:
                del self.maneuver_states[state_key]
            return None, None, None, None, 0.0

        # 执行当前阶段的机动
        maneuver = composite.maneuvers[current_phase]
        maneuver_time = time_sec - state["phase_start_time"]

        # 调试日志
        if int(time_sec * 5) % 25 == 0:  # 每0.2秒打印一次
            logging.info(f"🔄 组合机动 {maneuver_name} 阶段{current_phase + 1}/{len(composite.maneuvers)}: {maneuver.name}, "
                         f"总时间={time_sec:.1f}s, 阶段时间={maneuver_time:.1f}s/{maneuver.duration:.1f}s")

        # 根据机动名称调用对应的基础机动函数
        if maneuver.name == "转弯":
            turn_angle = maneuver.target_heading
            turn_rate = params.get("turn_rate", 3.0)
            result = self.basic_maneuvers.turn(
                maneuver_time,
                state["phase_initial_heading"],
                turn_angle,
                turn_rate
            )
            return result

        elif maneuver.name == "拉起":
            altitude_gain = maneuver.target_altitude
            result = self.basic_maneuvers.pull_up(
                maneuver_time,
                state["phase_initial_altitude"],
                maneuver.duration,
                altitude_gain
            )
            return result

        elif maneuver.name == "俯冲":
            altitude_loss = maneuver.target_altitude  # 存储的是绝对值
            min_altitude = params.get("min_altitude", 3000.0)
            result = self.basic_maneuvers.dive(
                maneuver_time,
                state["phase_initial_altitude"],
                maneuver.duration,
                altitude_loss,
                min_altitude
            )
            return result

        elif maneuver.name == "加速":
            velocity_increase = maneuver.target_velocity
            current_velocity = params.get("current_velocity", 250.0)
            result = self.basic_maneuvers.accelerate(
                maneuver_time,
                current_velocity,
                maneuver.duration,
                velocity_increase
            )
            return result

        elif maneuver.name == "减速":
            velocity_decrease = maneuver.target_velocity
            current_velocity = params.get("current_velocity", 300.0)
            result = self.basic_maneuvers.decelerate(
                maneuver_time,
                current_velocity,
                maneuver.duration,
                velocity_decrease
            )
            return result

        else:
            logging.warning(f"未知的机动阶段: {maneuver.name}")
            return None, None, None, None, 0.0

    def reset_composite_maneuver(self, maneuver_name: str):
        """重置组合机动状态"""
        state_key = f"{maneuver_name}_state"
        if state_key in self.maneuver_states:
            del self.maneuver_states[state_key]
            logging.info(f"重置组合机动 {maneuver_name} 状态")

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
        """Crank机动 - 纯函数实现，返回phase信息"""
        turn_time = abs(crank_angle_deg) / turn_rate_deg_per_sec

        if time_sec <= turn_time:
            phase = "TURN_TO_CRANK"
            progress = time_sec / turn_time
            target_heading = initial_heading_deg + crank_angle_deg * progress
            target_roll = 0.0
        elif time_sec <= turn_time + hold_time_sec:
            phase = "HOLD_CRANK"
            target_heading = initial_heading_deg + crank_angle_deg
            target_roll = 0.0
        else:
            phase = "RETURN_TO_INITIAL"
            return_time = time_sec - turn_time - hold_time_sec
            if return_time <= turn_time:
                progress = return_time / turn_time
                target_heading = (initial_heading_deg + crank_angle_deg) - crank_angle_deg * progress
            else:
                target_heading = initial_heading_deg
            target_roll = 0.0

        target_heading = target_heading % 360
        return phase, target_heading, target_roll

    @staticmethod
    def beam_maneuver(time_sec: float,
                      initial_heading_deg: float = 0.0,
                      beam_angle_deg: float = 90.0,
                      turn_rate_deg_per_sec: float = 5.0,
                      hold_time_sec: float = 15.0):
        """Beam机动 - 横向态势机动，消耗导弹动能"""
        turn_time = abs(beam_angle_deg) / turn_rate_deg_per_sec

        if time_sec <= turn_time:
            phase = "TURN_TO_BEAM"
            progress = time_sec / turn_time
            target_heading = initial_heading_deg + beam_angle_deg * progress
            target_roll = 0.0
        elif time_sec <= turn_time + hold_time_sec:
            phase = "HOLD_BEAM"
            target_heading = initial_heading_deg + beam_angle_deg
            target_roll = 0.0
        else:
            phase = "RETURN_TO_INITIAL"
            return_time = time_sec - turn_time - hold_time_sec
            if return_time <= turn_time:
                progress = return_time / turn_time
                target_heading = (initial_heading_deg + beam_angle_deg) - beam_angle_deg * progress
            else:
                target_heading = initial_heading_deg
            target_roll = 0.0

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
        """Notch机动 - 地面杂波隐蔽机动，打断雷达锁定"""
        turn_time = abs(notch_angle_deg) / turn_rate_deg_per_sec
        max_descent = descent_rate_ft_per_sec * descent_time_sec
        target_low_altitude = initial_altitude_ft - max_descent
        min_safe_altitude = 9000.0

        if target_low_altitude < min_safe_altitude:
            target_low_altitude = min_safe_altitude
            if descent_time_sec > 0:
                descent_rate_ft_per_sec = (initial_altitude_ft - target_low_altitude) / descent_time_sec

        if time_sec <= descent_time_sec:
            phase = "DESCENT_TO_CLUTTER"
            target_heading = initial_heading_deg
            target_altitude = initial_altitude_ft - (descent_rate_ft_per_sec * time_sec)
            target_altitude = max(target_altitude, min_safe_altitude)
            target_roll = 0.0
        elif time_sec <= descent_time_sec + turn_time:
            phase = "TURN_IN_CLUTTER"
            turn_progress = (time_sec - descent_time_sec) / turn_time
            target_heading = initial_heading_deg + notch_angle_deg * turn_progress
            target_altitude = target_low_altitude
            target_roll = 0.0
        elif time_sec <= descent_time_sec + turn_time + hold_time_sec:
            phase = "HOLD_IN_CLUTTER"
            target_heading = initial_heading_deg + notch_angle_deg
            target_altitude = target_low_altitude
            target_roll = 0.0
        else:
            phase = "CLIMB_AND_RETURN"
            return_time = time_sec - descent_time_sec - turn_time - hold_time_sec
            climb_time = 10.0
            if return_time <= climb_time:
                progress = return_time / climb_time
                if return_time <= turn_time:
                    heading_progress = return_time / turn_time
                    target_heading = (initial_heading_deg + notch_angle_deg) - notch_angle_deg * heading_progress
                else:
                    target_heading = initial_heading_deg
                altitude_recovery = (initial_altitude_ft - target_low_altitude) * progress
                target_altitude = target_low_altitude + altitude_recovery
            else:
                target_heading = initial_heading_deg
                target_altitude = initial_altitude_ft
            target_roll = 0.0

        target_heading = target_heading % 360
        target_altitude = max(target_altitude, min_safe_altitude)
        return phase, target_heading, target_altitude, target_roll

    @staticmethod
    def get_maneuver_description(maneuver_name: str) -> str:
        """获取机动描述"""
        descriptions = {
            "crank": "Crank机动 - 斜向机动，保持雷达锁定同时避开威胁",
            "beam": "Beam机动 - 90度横向机动，最大化多普勒效应",
            "notch": "Notch机动 - 逃离机动，利用地面杂波隐蔽",
            "level_flight": "平飞 - 保持当前高度、航向、速度",
            "accelerate": "加速 - 在指定时间内增加指定速度",
            "decelerate": "减速 - 在指定时间内减少指定速度",
            "turn": "转弯 - 在计算时间内改变指定角度",
            "pull_up": "拉起 - 在指定时间内爬升指定高度",
            "dive": "俯冲 - 在指定时间内下降指定高度",
            "diagonal_flight": "斜直飞 - 在指定时间内同时改变航向和高度",
            "circle": "盘旋 - 持续转弯的圆形路径飞行",
            "barrel_roll": "桶滚 - 连续滚转的螺旋机动",
            "turn_pull_up": "转弯拉起组合 - 先转弯后拉起",
            "turn_dive": "转弯俯冲组合 - 先转弯后俯冲",
            "s_turn": "S型转弯组合 - 连续相反方向转弯",
            "attack_escape": "攻击逃离组合 - 加速攻击后转弯俯冲逃离"
        }
        return descriptions.get(maneuver_name, "未知机动")