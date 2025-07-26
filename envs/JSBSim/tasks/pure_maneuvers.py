import json
import numpy as np
import math
from typing import Tuple, Dict, Any, List, Optional
from dataclasses import dataclass
import logging


def normalize_heading(heading_deg: float) -> float:
    """规范化航向到[0, 360)范围"""
    while heading_deg >= 360.0:
        heading_deg -= 360.0
    while heading_deg < 0.0:
        heading_deg += 360.0
    return heading_deg


class BasicManeuvers:
    """基础机动库"""

    @staticmethod
    def level_flight(time_sec: float, duration: float = 10.0, current_altitude: float = 0.0,
                     current_velocity: float = 0.0):
        """平飞 - 保持当前状态"""
        if time_sec <= duration:
            return "LEVEL_FLIGHT", None, None, None, None
        return None, None, None, None, None

    @staticmethod
    def accelerate(time_sec: float, current_velocity: float, duration: float = 5.0, velocity_increase: float = 50.0,
                   max_velocity: float = 350.0):
        """加速 - 仅改变速度"""
        if time_sec <= duration:
            progress = time_sec / duration
            smooth_progress = 3 * progress ** 2 - 2 * progress ** 3
            velocity_offset = velocity_increase * smooth_progress

            if current_velocity + velocity_offset > max_velocity:
                velocity_offset = max_velocity - current_velocity

            return "ACCELERATE", None, None, velocity_offset, None
        return None, None, None, None, None

    @staticmethod
    def decelerate(time_sec: float, current_velocity: float, duration: float = 5.0, velocity_decrease: float = 50.0,
                   min_velocity: float = 150.0):
        """减速 - 仅改变速度"""
        if time_sec <= duration:
            progress = time_sec / duration
            smooth_progress = 3 * progress ** 2 - 2 * progress ** 3
            velocity_offset = -velocity_decrease * smooth_progress

            if current_velocity + velocity_offset < min_velocity:
                velocity_offset = min_velocity - current_velocity

            return "DECELERATE", None, None, velocity_offset, None
        return None, None, None, None, None

    @staticmethod
    def turn(time_sec: float, initial_heading: float, turn_angle: float = 45.0, turn_rate: float = 3.0):
        """转弯 - 仅改变航向，支持任意角度"""
        turn_time = abs(turn_angle) / turn_rate
        if time_sec <= turn_time:
            progress = time_sec / turn_time
            target_heading = normalize_heading(initial_heading + turn_angle * progress)
            required_roll = abs(turn_rate) * 12.0
            roll_magnitude = min(required_roll, 50.0)
            target_roll = roll_magnitude * (1 if turn_angle > 0 else -1) * math.sin(progress * math.pi)
            return "TURN", target_heading, None, None, target_roll
        else:
            final_heading = normalize_heading(initial_heading + turn_angle)
            return "TURN_COMPLETE", final_heading, None, None, None

    @staticmethod
    def pull_up(time_sec: float, initial_altitude: float, duration: float = 8.0, altitude_gain: float = 1000.0):
        """拉起 - 仅改变高度"""
        if time_sec <= duration:
            progress = time_sec / duration
            smooth_progress = 3 * progress ** 2 - 2 * progress ** 3
            target_altitude = initial_altitude + altitude_gain * smooth_progress
            velocity_compensation = -altitude_gain * 0.02 / duration
            return "PULL_UP", None, target_altitude, velocity_compensation, None
        else:
            return "PULL_UP_COMPLETE", None, initial_altitude + altitude_gain, None, None

    @staticmethod
    def dive(time_sec: float, initial_altitude: float, duration: float = 8.0, altitude_loss: float = 1000.0,
             min_altitude: float = 3000.0):
        """俯冲 - 仅改变高度"""
        target_final_altitude = max(initial_altitude - altitude_loss, min_altitude)
        actual_altitude_loss = initial_altitude - target_final_altitude

        if time_sec <= duration:
            progress = time_sec / duration
            smooth_progress = 3 * progress ** 2 - 2 * progress ** 3
            target_altitude = initial_altitude - actual_altitude_loss * smooth_progress
            velocity_compensation = actual_altitude_loss * 0.02 / duration
            return "DIVE", None, target_altitude, velocity_compensation, None
        else:
            return "DIVE_COMPLETE", None, target_final_altitude, None, None

    @staticmethod
    def diagonal_flight(time_sec: float, initial_heading: float, initial_altitude: float,
                        duration: float = 10.0, heading_change: float = 30.0, altitude_change: float = 500.0,
                        min_altitude: float = 3000.0):
        """斜向飞行 - 同时改变航向和高度"""
        target_final_altitude = max(initial_altitude + altitude_change, min_altitude)
        actual_altitude_change = target_final_altitude - initial_altitude

        if time_sec <= duration:
            progress = time_sec / duration
            smooth_progress = 3 * progress ** 2 - 2 * progress ** 3
            target_heading = normalize_heading(initial_heading + heading_change * smooth_progress)
            target_altitude = initial_altitude + actual_altitude_change * smooth_progress
            target_roll = min(20.0, abs(heading_change) / 4) * (1 if heading_change > 0 else -1) * math.sin(
                progress * math.pi)
            velocity_compensation = -abs(actual_altitude_change) * 0.01 / duration if actual_altitude_change > 0 else abs(
                actual_altitude_change) * 0.01 / duration
            return "DIAGONAL_FLIGHT", target_heading, target_altitude, velocity_compensation, target_roll
        else:
            return "DIAGONAL_FLIGHT_COMPLETE", normalize_heading(
                initial_heading + heading_change), target_final_altitude, None, None


@dataclass
class ManeuverStep:
    """单个机动步骤定义"""
    name: str
    params: Dict[str, Any]
    duration: float


class CompositeManeuverExecutor:
    """组合机动执行器"""

    def __init__(self):
        self.basic_maneuvers = BasicManeuvers()
        self.maneuver_definitions = {}
        self.active_states = {}
        self._setup_predefined_maneuvers()

    def _setup_predefined_maneuvers(self):
        """设置预定义的组合机动"""
        self.maneuver_definitions["turn_pull_up"] = [
            ManeuverStep("turn", {"turn_angle": 90.0, "turn_rate": 3.0}, 35.0),
            ManeuverStep("pull_up", {"altitude_gain": 2000.0}, 20.0)
        ]
        self.maneuver_definitions["turn_dive"] = [
            ManeuverStep("turn", {"turn_angle": 90.0, "turn_rate": 3.0}, 35.0),
            ManeuverStep("dive", {"altitude_loss": 1500.0, "min_altitude": 2000.0}, 20.0)
        ]
        self.maneuver_definitions["spiral_climb"] = [
            ManeuverStep("turn", {"turn_angle": 360.0, "turn_rate": 2.0}, 180.0),
            ManeuverStep("pull_up", {"altitude_gain": 1500.0}, 25.0)
        ]

    def update_maneuver_params(self, maneuver_name: str, custom_params: Dict[str, Any]):
        """更新组合机动的参数"""
        if maneuver_name not in self.maneuver_definitions:
            logging.warning(f"未知的组合机动: {maneuver_name}")
            return

        if maneuver_name == "turn_pull_up" and "turn_pull_up" in custom_params:
            params = custom_params["turn_pull_up"]
            self.maneuver_definitions[maneuver_name][0].params.update({
                "turn_angle": params.get("turn_angle", 90.0),
                "turn_rate": params.get("turn_rate", 3.0)
            })
            self.maneuver_definitions[maneuver_name][0].duration = params.get("turn_duration", 30.0)
            self.maneuver_definitions[maneuver_name][1].params.update({
                "altitude_gain": params.get("altitude_gain", 2000.0)
            })
            self.maneuver_definitions[maneuver_name][1].duration = params.get("pull_up_duration", 15.0)

        elif maneuver_name == "turn_dive" and "turn_dive" in custom_params:
            params = custom_params["turn_dive"]
            self.maneuver_definitions[maneuver_name][0].params.update({
                "turn_angle": params.get("turn_angle", 90.0),
                "turn_rate": params.get("turn_rate", 3.0)
            })
            self.maneuver_definitions[maneuver_name][0].duration = params.get("turn_duration", 30.0)
            self.maneuver_definitions[maneuver_name][1].params.update({
                "altitude_loss": params.get("altitude_loss", 1500.0),
                "min_altitude": params.get("min_altitude", 2000.0)
            })
            self.maneuver_definitions[maneuver_name][1].duration = params.get("dive_duration", 15.0)

        logging.info(f"组合机动 {maneuver_name} 参数更新完成")
        for i, step in enumerate(self.maneuver_definitions[maneuver_name]):
            logging.info(f"步骤{i + 1}: {step.name} {step.params} 持续{step.duration}s")

    def execute_composite_maneuver(self, maneuver_name: str, total_time: float,
                                   initial_heading: float, initial_altitude: float,
                                   current_velocity: float, current_altitude: float) -> Tuple:
        """执行组合机动"""
        if maneuver_name not in self.maneuver_definitions:
            logging.warning(f"未知的组合机动: {maneuver_name}")
            return None, None, None, None, None

        steps = self.maneuver_definitions[maneuver_name]

        if maneuver_name not in self.active_states:
            self.active_states[maneuver_name] = {
                "current_step": 0,
                "step_start_time": 0.0,
                "cumulative_heading": initial_heading,
                "cumulative_altitude": initial_altitude,
                "last_completed_step": -1
            }

        state = self.active_states[maneuver_name]
        elapsed_time = 0.0
        current_step_index = -1

        for i, step in enumerate(steps):
            if total_time <= elapsed_time + step.duration:
                current_step_index = i
                break
            elapsed_time += step.duration

        if current_step_index == -1:
            if maneuver_name in self.active_states:
                del self.active_states[maneuver_name]
            return None, None, None, None, None

        if current_step_index != state["current_step"]:
            if current_step_index > 0:
                prev_step = steps[current_step_index - 1]
                if prev_step.name == "turn":
                    state["cumulative_heading"] += prev_step.params["turn_angle"]
                    state["cumulative_heading"] = normalize_heading(state["cumulative_heading"])
                elif prev_step.name == "pull_up":
                    state["cumulative_altitude"] += prev_step.params["altitude_gain"]
                elif prev_step.name == "dive":
                    state["cumulative_altitude"] -= prev_step.params["altitude_loss"]
                    state["cumulative_altitude"] = max(state["cumulative_altitude"],
                                                       prev_step.params.get("min_altitude", 2000.0))

            state["current_step"] = current_step_index
            state["step_start_time"] = elapsed_time

            logging.info(f"组合机动 {maneuver_name} 切换到步骤 {current_step_index + 1}: {steps[current_step_index].name}")
            logging.info(f"累积航向: {state['cumulative_heading']:.1f}°")
            logging.info(f"累积高度: {state['cumulative_altitude']:.1f}m")

        current_step = steps[current_step_index]
        step_time = total_time - state["step_start_time"]

        if int(total_time * 5) % 25 == 0:
            logging.info(f"组合机动 {maneuver_name} 步骤{current_step_index + 1}/{len(steps)}: {current_step.name}")
            logging.info(f"总时间: {total_time:.1f}s, 步骤时间: {step_time:.1f}s/{current_step.duration:.1f}s")

        if current_step.name == "turn":
            return self.basic_maneuvers.turn(
                step_time,
                state["cumulative_heading"],
                current_step.params["turn_angle"],
                current_step.params["turn_rate"]
            )
        elif current_step.name == "pull_up":
            return self.basic_maneuvers.pull_up(
                step_time,
                state["cumulative_altitude"],
                current_step.duration,
                current_step.params["altitude_gain"]
            )
        elif current_step.name == "dive":
            return self.basic_maneuvers.dive(
                step_time,
                state["cumulative_altitude"],
                current_step.duration,
                current_step.params["altitude_loss"],
                current_step.params.get("min_altitude", 3000.0)
            )
        elif current_step.name == "accelerate":
            return self.basic_maneuvers.accelerate(
                step_time,
                current_velocity,
                current_step.duration,
                current_step.params["velocity_increase"]
            )
        elif current_step.name == "decelerate":
            return self.basic_maneuvers.decelerate(
                step_time,
                current_velocity,
                current_step.duration,
                current_step.params["velocity_decrease"]
            )
        else:
            logging.warning(f"未知的基础机动: {current_step.name}")
            return None, None, None, None, None

    def reset_maneuver_state(self, maneuver_name: str):
        """重置组合机动状态"""
        if maneuver_name in self.active_states:
            del self.active_states[maneuver_name]
            logging.info(f"重置组合机动 {maneuver_name} 状态")

    def get_available_maneuvers(self) -> List[str]:
        """获取可用的组合机动列表"""
        return list(self.maneuver_definitions.keys())

    def get_maneuver_info(self, maneuver_name: str) -> Dict[str, Any]:
        """获取组合机动信息"""
        if maneuver_name not in self.maneuver_definitions:
            return {}

        steps = self.maneuver_definitions[maneuver_name]
        total_duration = sum(step.duration for step in steps)
        return {
            "name": maneuver_name,
            "steps": [(step.name, step.params, step.duration) for step in steps],
            "total_duration": total_duration,
            "description": f"组合机动包含{len(steps)}个步骤，总时长{total_duration:.1f}秒"
        }


class ManeuverComposer:
    """兼容性包装器"""

    def __init__(self):
        self.executor = CompositeManeuverExecutor()
        self.composite_maneuvers = {}
        self.maneuver_states = {}

    def _setup_tactical_templates(self, custom_params=None):
        """设置战术模板"""
        if custom_params:
            for maneuver_name in ["turn_pull_up", "turn_dive"]:
                if maneuver_name in custom_params or any(key in custom_params for key in [maneuver_name]):
                    self.executor.update_maneuver_params(maneuver_name, custom_params)

    def execute_composite_maneuver(self, maneuver_name: str, time_sec: float,
                                   initial_heading: float, initial_altitude: float,
                                   params: Dict[str, Any] = None):
        """执行组合机动"""
        current_velocity = params.get("current_velocity", 250.0) if params else 250.0
        current_altitude = params.get("current_altitude", initial_altitude) if params else initial_altitude
        return self.executor.execute_composite_maneuver(
            maneuver_name, time_sec, initial_heading, initial_altitude,
            current_velocity, current_altitude
        )

    def reset_composite_maneuver(self, maneuver_name: str):
        """重置组合机动状态"""
        self.executor.reset_maneuver_state(maneuver_name)


class PureManeuvers:
    """纯机动函数库"""

    @staticmethod
    def crank_maneuver(time_sec: float,
                       initial_heading_deg: float = 0.0,
                       crank_angle_deg: float = 45.0,
                       turn_rate_deg_per_sec: float = 3.0,
                       hold_time_sec: float = 20.0):
        """Crank机动"""
        turn_time = abs(crank_angle_deg) / turn_rate_deg_per_sec
        if time_sec <= turn_time:
            phase = "TURN_TO_CRANK"
            progress = time_sec / turn_time
            target_heading = normalize_heading(initial_heading_deg + crank_angle_deg * progress)
            target_roll = 0.0
        elif time_sec <= turn_time + hold_time_sec:
            phase = "HOLD_CRANK"
            target_heading = normalize_heading(initial_heading_deg + crank_angle_deg)
            target_roll = 0.0
        else:
            phase = "RETURN_TO_INITIAL"
            return_time = time_sec - turn_time - hold_time_sec
            if return_time <= turn_time:
                progress = return_time / turn_time
                target_heading = normalize_heading((initial_heading_deg + crank_angle_deg) - crank_angle_deg * progress)
            else:
                target_heading = normalize_heading(initial_heading_deg)
            target_roll = 0.0
        return phase, target_heading, target_roll

    @staticmethod
    def beam_maneuver(time_sec: float,
                      initial_heading_deg: float = 0.0,
                      beam_angle_deg: float = 90.0,
                      turn_rate_deg_per_sec: float = 5.0,
                      hold_time_sec: float = 15.0):
        """Beam机动"""
        turn_time = abs(beam_angle_deg) / turn_rate_deg_per_sec
        if time_sec <= turn_time:
            phase = "TURN_TO_BEAM"
            progress = time_sec / turn_time
            target_heading = normalize_heading(initial_heading_deg + beam_angle_deg * progress)
            target_roll = 0.0
        elif time_sec <= turn_time + hold_time_sec:
            phase = "HOLD_BEAM"
            target_heading = normalize_heading(initial_heading_deg + beam_angle_deg)
            target_roll = 0.0
        else:
            phase = "RETURN_TO_INITIAL"
            return_time = time_sec - turn_time - hold_time_sec
            if return_time <= turn_time:
                progress = return_time / turn_time
                target_heading = normalize_heading((initial_heading_deg + beam_angle_deg) - beam_angle_deg * progress)
            else:
                target_heading = normalize_heading(initial_heading_deg)
            target_roll = 0.0
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
        """Notch机动"""
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
            target_heading = normalize_heading(initial_heading_deg)
            target_altitude = initial_altitude_ft - (descent_rate_ft_per_sec * time_sec)
            target_altitude = max(target_altitude, min_safe_altitude)
            target_roll = 0.0
        elif time_sec <= descent_time_sec + turn_time:
            phase = "TURN_IN_CLUTTER"
            turn_progress = (time_sec - descent_time_sec) / turn_time
            target_heading = normalize_heading(initial_heading_deg + notch_angle_deg * turn_progress)
            target_altitude = target_low_altitude
            target_roll = 0.0
        elif time_sec <= descent_time_sec + turn_time + hold_time_sec:
            phase = "HOLD_IN_CLUTTER"
            target_heading = normalize_heading(initial_heading_deg + notch_angle_deg)
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
                    target_heading = normalize_heading(
                        (initial_heading_deg + notch_angle_deg) - notch_angle_deg * heading_progress)
                else:
                    target_heading = normalize_heading(initial_heading_deg)
                altitude_recovery = (initial_altitude_ft - target_low_altitude) * progress
                target_altitude = target_low_altitude + altitude_recovery
            else:
                target_heading = normalize_heading(initial_heading_deg)
                target_altitude = initial_altitude_ft
            target_roll = 0.0
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
            "turn": "转弯 - 在计算时间内改变指定角度，支持大角度和左转",
            "pull_up": "拉起 - 在指定时间内爬升指定高度",
            "dive": "俯冲 - 在指定时间内下降指定高度",
            "diagonal_flight": "斜直飞 - 在指定时间内同时改变航向和高度",
            "turn_pull_up": "转弯拉起组合 - 先转弯后拉起",
            "turn_dive": "转弯俯冲组合 - 先转弯后俯冲",
            "s_turn": "S转弯组合 - 左转+右转+左转",
            "spiral_climb": "盘旋爬升组合 - 360度转弯+拉起",
            "dive_turn": "俯冲转弯组合 - 先俯冲后转弯",
            "accelerate_turn": "加速转弯组合 - 先加速后转弯"
        }
        return descriptions.get(maneuver_name, "未知机动")