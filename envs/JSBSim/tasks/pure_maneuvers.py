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
    """基础机动库 - 您要求的小机动动作"""

    @staticmethod
    def level_flight(time_sec: float, duration: float = 10.0, current_altitude: float = 0.0, current_velocity: float = 0.0):
        """平飞 - 保持当前高度、航向、速度（完善：返回当前值作为target，确保稳定）"""
        if time_sec <= duration:
            return "LEVEL_FLIGHT", None, current_altitude, current_velocity, 0.0
        return None, None, None, None, 0.0

    @staticmethod
    def accelerate(time_sec: float, duration: float = 5.0, velocity_increase: float = 50.0, max_velocity: float = 300.0):
        """加速 - 增加速度（完善：添加速度上限，避免不现实加速）"""
        if time_sec <= duration:
            target_velocity = velocity_increase * (time_sec / duration)  # 平滑线性增加
            target_velocity = min(target_velocity, max_velocity)  # 安全上限
            return "ACCELERATE", None, None, target_velocity, 0.0
        return None, None, None, None, 0.0

    @staticmethod
    def decelerate(time_sec: float, duration: float = 5.0, velocity_decrease: float = 50.0, min_velocity: float = 100.0):
        """减速 - 减少速度（完善：添加最小速度阈值，防止失速）"""
        if time_sec <= duration:
            target_velocity = -velocity_decrease * (time_sec / duration)  # 平滑线性减少
            target_velocity = max(target_velocity, min_velocity)  # 安全下限
            return "DECELERATE", None, None, target_velocity, 0.0
        return None, None, None, None, 0.0

    @staticmethod
    def turn(time_sec: float, initial_heading: float, turn_angle: float = 45.0, turn_rate: float = 3.0):
        """转弯 - 改变航向（完善：添加滚转辅助，模拟协调转弯）"""
        turn_time = abs(turn_angle) / turn_rate
        if time_sec <= turn_time:
            progress = time_sec / turn_time
            target_heading = initial_heading + turn_angle * math.sin(progress * math.pi)  # 平滑sin曲线
            target_roll = min(30.0, abs(turn_angle) / 3) * (1 if turn_angle > 0 else -1)  # 辅助滚转
            return "TURN", target_heading, None, None, target_roll
        return None, None, None, None, 0.0

    @staticmethod
    def pull_up(time_sec: float, duration: float = 8.0, altitude_gain: float = 1000.0):
        """拉起 - 爬升（完善：整合速度调整，更真实）"""
        if time_sec <= duration:
            target_altitude = altitude_gain * (time_sec / duration)
            target_velocity = -0.1 * altitude_gain * (time_sec / duration)  # 爬升时速度略降
            return "PULL_UP", None, target_altitude, target_velocity, 0.0
        return None, None, None, None, 0.0

    @staticmethod
    def dive(time_sec: float, duration: float = 8.0, altitude_loss: float = 1000.0, min_altitude: float = 9000.0):
        """俯冲 - 下降（完善：添加min_altitude，防止撞地）"""
        if time_sec <= duration:
            target_altitude = -altitude_loss * (time_sec / duration)
            target_altitude = max(target_altitude, min_altitude)  # 安全下限
            return "DIVE", None, target_altitude, None, 0.0
        return None, None, None, None, 0.0

    @staticmethod
    def diagonal_flight(time_sec: float, duration: float = 10.0, heading_change: float = 30.0, altitude_change: float = 500.0):
        """斜直飞 - 同时改变航向和高度（完善：添加速度补偿）"""
        if time_sec <= duration:
            progress = time_sec / duration
            target_heading = heading_change * math.sin(progress * math.pi)  # 平滑
            target_altitude = altitude_change * math.sin(progress * math.pi)
            target_velocity = -0.05 * abs(altitude_change) * progress  # 能量补偿
            return "DIAGONAL_FLIGHT", target_heading, target_altitude, target_velocity, 0.0
        return None, None, None, None, 0.0

    @staticmethod
    def roll(time_sec: float, duration: float = 3.0, roll_angle: float = 45.0, roll_rate: float = 90.0):
        """滚转 - 改变滚转角（完善：添加roll_rate，实现渐进滚转）"""
        roll_time = abs(roll_angle) / roll_rate
        if time_sec <= roll_time:
            progress = time_sec / roll_time
            target_roll = roll_angle * math.sin(progress * math.pi)  # 平滑渐进
            return "ROLL", None, None, None, target_roll
        return None, None, None, None, 0.0

    @staticmethod
    def turn_pull_up(time_sec: float, initial_heading: float, turn_angle: float = 45.0, turn_rate: float = 3.0, altitude_gain: float = 1000.0):
        """转弯拉起 - 同时转弯和爬升（完善：添加滚转模拟倾斜）"""
        turn_time = abs(turn_angle) / turn_rate
        if time_sec <= turn_time:
            progress = time_sec / turn_time
            target_heading = initial_heading + turn_angle * math.sin(progress * math.pi)
            target_altitude = altitude_gain * math.sin(progress * math.pi)
            target_roll = min(30.0, abs(turn_angle) / 3) * (1 if turn_angle > 0 else -1)
            return "TURN_PULL_UP", target_heading, target_altitude, None, target_roll
        return None, None, None, None, 0.0

    @staticmethod
    def turn_dive(time_sec: float, initial_heading: float, turn_angle: float = 45.0, turn_rate: float = 3.0, altitude_loss: float = 1000.0, min_altitude: float = 9000.0):
        """转弯俯冲 - 同时转弯和下降（完善：添加min_altitude和滚转）"""
        turn_time = abs(turn_angle) / turn_rate
        if time_sec <= turn_time:
            progress = time_sec / turn_time
            target_heading = initial_heading + turn_angle * math.sin(progress * math.pi)
            target_altitude = -altitude_loss * math.sin(progress * math.pi)
            target_altitude = max(target_altitude, min_altitude)
            target_roll = min(30.0, abs(turn_angle) / 3) * (1 if turn_angle > 0 else -1)
            return "TURN_DIVE", target_heading, target_altitude, None, target_roll
        return None, None, None, None, 0.0

    @staticmethod
    def circle(time_sec: float, duration: float = 20.0, radius: float = 1000.0, direction: str = "clockwise"):
        """盘旋 - 圆形路径（新扩展）"""
        if time_sec <= duration:
            angle = (time_sec / duration) * 360.0 * (1 if direction == "clockwise" else -1)
            return "CIRCLE", angle, None, None, 0.0
        return None, None, None, None, 0.0

    @staticmethod
    def barrel_roll(time_sec: float, duration: float = 5.0, roll_angle: float = 360.0):
        """桶滚 - 完整滚转（新扩展）"""
        if time_sec <= duration:
            target_roll = roll_angle * (time_sec / duration)
            return "BARREL_ROLL", None, None, None, target_roll
        return None, None, None, None, 0.0


class ManeuverComposer:
    """机动组合器 - 将基础机动组合成复杂战术"""

    def __init__(self):
        self.basic_maneuvers = BasicManeuvers()
        self.composite_maneuvers = {}
        self._setup_tactical_templates()

    def _setup_tactical_templates(self, custom_params=None):
        """设置战术模板 - 支持自定义参数覆盖"""
        if custom_params is None:
            custom_params = {}

        # 脱离机动模板
        escape_params = custom_params.get("escape", {})
        escape_maneuvers = [
            BasicManeuver("急转弯", escape_params.get("turn_duration", 5.0), escape_params.get("turn_angle", 90.0), 0.0,
                          0.0, 0.0, "快速转向"),
            BasicManeuver("加速", escape_params.get("accel_duration", 3.0), 0.0, 0.0,
                          escape_params.get("accel_velocity", 100.0), 0.0, "加速脱离"),
            BasicManeuver("平飞", escape_params.get("level_duration", 10.0), 0.0, 0.0, 0.0, 0.0, "稳定飞行")
        ]
        self.composite_maneuvers["escape"] = CompositeManeuver(
            name="脱离机动",
            maneuvers=escape_maneuvers,
            total_duration=sum(m.duration for m in escape_maneuvers),
            description="快速脱离威胁区域"
        )

        # 攻击机动模板
        attack_params = custom_params.get("attack", {})
        attack_maneuvers = [
            BasicManeuver("侧跃升", attack_params.get("side_climb_duration", 8.0),
                          attack_params.get("side_climb_angle", 45.0), attack_params.get("side_climb_alt", 2000.0), 0.0,
                          0.0, "侧向爬升"),
            BasicManeuver("保持高度", attack_params.get("hold_duration", 5.0), 0.0, 0.0, 0.0, 0.0, "保持高度"),
            BasicManeuver("俯冲攻击", attack_params.get("dive_duration", 10.0), attack_params.get("dive_angle", -45.0),
                          attack_params.get("dive_alt", -2000.0), attack_params.get("dive_velocity", 50.0), 0.0, "俯冲攻击")
        ]
        self.composite_maneuvers["attack"] = CompositeManeuver(
            name="侧跃升拉起俯冲攻击",
            maneuvers=attack_maneuvers,
            total_duration=sum(m.duration for m in attack_maneuvers),
            description="经典的攻击机动"
        )

        # 防御机动模板
        defense_params = custom_params.get("defense", {})
        defense_maneuvers = [
            BasicManeuver("滚转", defense_params.get("roll_duration", 2.0), 0.0, 0.0, 0.0,
                          defense_params.get("roll_angle", 45.0), "快速滚转"),
            BasicManeuver("俯冲", defense_params.get("dive_duration", 5.0), 0.0, defense_params.get("dive_alt", -1000.0),
                          0.0, 0.0, "俯冲躲避"),
            BasicManeuver("转弯", defense_params.get("turn_duration", 8.0), defense_params.get("turn_angle", 90.0), 0.0,
                          0.0, 0.0, "转向脱离"),
            BasicManeuver("爬升", defense_params.get("climb_duration", 6.0), 0.0, defense_params.get("climb_alt", 1000.0),
                          0.0, 0.0, "恢复高度")
        ]
        self.composite_maneuvers["defense"] = CompositeManeuver(
            name="防御机动",
            maneuvers=defense_maneuvers,
            total_duration=sum(m.duration for m in defense_maneuvers),
            description="综合防御机动"
        )

        # 剪刀机动模板
        scissor_params = custom_params.get("scissor", {})
        scissor_maneuvers = [
            BasicManeuver("转弯", scissor_params.get("turn_duration", 4.0), scissor_params.get("turn_angle", 45.0), 0.0,
                          0.0, 0.0, "左转"),
            BasicManeuver("转弯", scissor_params.get("turn_duration", 4.0), scissor_params.get("turn_angle", -45.0), 0.0,
                          0.0, 0.0, "右转"),
            BasicManeuver("加速", scissor_params.get("accel_duration", 3.0), 0.0, 0.0,
                          scissor_params.get("accel_velocity", 50.0), 0.0, "加速")
        ]
        self.composite_maneuvers["scissor"] = CompositeManeuver(
            name="剪刀机动",
            maneuvers=scissor_maneuvers,
            total_duration=sum(m.duration for m in scissor_maneuvers),
            description="交替转弯防御"
        )

        # 高YOYO攻击模板
        yoyo_params = custom_params.get("high_yoyo", {})
        yoyo_maneuvers = [
            BasicManeuver("爬升", yoyo_params.get("climb_duration", 5.0), 0.0, yoyo_params.get("climb_alt", 1500.0), 0.0,
                          0.0, "高位爬升"),
            BasicManeuver("转弯", yoyo_params.get("turn_duration", 6.0), yoyo_params.get("turn_angle", 60.0), 0.0, 0.0,
                          0.0, "转弯定位"),
            BasicManeuver("俯冲", yoyo_params.get("dive_duration", 5.0), 0.0, yoyo_params.get("dive_alt", -1500.0), 0.0,
                          0.0, "俯冲攻击")
        ]
        self.composite_maneuvers["high_yoyo"] = CompositeManeuver(
            name="高YOYO攻击",
            maneuvers=yoyo_maneuvers,
            total_duration=sum(m.duration for m in yoyo_maneuvers),
            description="高位YOYO攻击机动"
        )

    def execute_composite_maneuver(self, maneuver_name: str, time_sec: float, initial_heading: float, initial_altitude: float, params: Dict[str, Any] = None):
        """执行组合机动"""
        if params is None:
            params = {}

        if maneuver_name not in self.composite_maneuvers:
            logging.warning(f"组合机动 {maneuver_name} 不存在")
            return None, None, None, None, 0.0

        composite = self.composite_maneuvers[maneuver_name]
        current_time = time_sec
        accumulated_time = 0.0

        for maneuver in composite.maneuvers:
            if current_time <= maneuver.duration:
                # 根据机动名称动态调用基础机动函数，并应用自定义参数
                maneuver_params = params.get(maneuver.name, {})
                try:
                    if maneuver.name in ["急转弯", "转弯"]:
                        result = self.basic_maneuvers.turn(
                            current_time,
                            initial_heading,
                            maneuver_params.get("turn_angle", maneuver.target_heading),
                            maneuver_params.get("turn_rate", 3.0)
                        )
                    elif maneuver.name == "加速":
                        result = self.basic_maneuvers.accelerate(
                            current_time,
                            maneuver_params.get("duration", maneuver.duration),
                            maneuver_params.get("accel_velocity", maneuver.target_velocity),
                            maneuver_params.get("max_velocity", 300.0)
                        )
                    elif maneuver.name == "平飞":
                        result = self.basic_maneuvers.level_flight(
                            current_time,
                            maneuver_params.get("duration", maneuver.duration),
                            current_altitude=initial_altitude,
                            current_velocity=maneuver_params.get("current_velocity", 0.0)
                        )
                    elif maneuver.name in ["侧跃升", "转弯拉起"]:
                        result = self.basic_maneuvers.turn_pull_up(
                            current_time,
                            initial_heading,
                            maneuver_params.get("side_climb_angle", maneuver.target_heading),
                            maneuver_params.get("turn_rate", 3.0),
                            maneuver_params.get("side_climb_alt", maneuver.target_altitude)
                        )
                    elif maneuver.name == "保持高度":
                        result = self.basic_maneuvers.level_flight(
                            current_time,
                            maneuver_params.get("duration", maneuver.duration),
                            current_altitude=initial_altitude,
                            current_velocity=maneuver_params.get("current_velocity", 0.0)
                        )
                    elif maneuver.name in ["俯冲攻击", "俯冲"]:
                        result = self.basic_maneuvers.dive(
                            current_time,
                            maneuver_params.get("duration", maneuver.duration),
                            maneuver_params.get("dive_alt", maneuver.target_altitude),
                            maneuver_params.get("min_altitude", 9000.0)
                        )
                    elif maneuver.name == "滚转":
                        result = self.basic_maneuvers.roll(
                            current_time,
                            maneuver_params.get("duration", maneuver.duration),
                            maneuver_params.get("roll_angle", maneuver.target_roll),
                            maneuver_params.get("roll_rate", 90.0)
                        )
                    elif maneuver.name == "爬升":
                        result = self.basic_maneuvers.pull_up(
                            current_time,
                            maneuver_params.get("duration", maneuver.duration),
                            maneuver_params.get("climb_alt", maneuver.target_altitude)
                        )
                    else:
                        logging.warning(f"未知子动作: {maneuver.name}")
                        result = (None, None, None, None, 0.0)

                    return result
                except Exception as e:
                    logging.error(f"执行子动作 {maneuver.name} 失败: {e}")
                    return None, None, None, None, 0.0

            current_time -= maneuver.duration
            accumulated_time += maneuver.duration

        # logging.info(f"组合机动 {maneuver_name} 完成")
        return None, None, None, None, 0.0

    def add_maneuver_to_template(self, template_name: str, basic_maneuver: BasicManeuver, insert_position: int = -1):
        """动态添加子动作到模板"""
        if template_name not in self.composite_maneuvers:
            logging.warning(f"模板 {template_name} 不存在")
            return
        composite = self.composite_maneuvers[template_name]
        if insert_position == -1:
            composite.maneuvers.append(basic_maneuver)
        else:
            composite.maneuvers.insert(insert_position, basic_maneuver)
        composite.total_duration = sum(m.duration for m in composite.maneuvers)
        logging.info(f"添加 {basic_maneuver.name} 到 {template_name}")

    def save_template(self, template_name: str, filepath: str):
        """保存模板到JSON"""
        if template_name not in self.composite_maneuvers:
            return
        composite = self.composite_maneuvers[template_name]
        data = {
            "name": composite.name,
            "maneuvers": [vars(m) for m in composite.maneuvers],  # 转换为dict
            "total_duration": composite.total_duration,
            "description": composite.description
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info(f"保存模板 {template_name} 到 {filepath}")

    def load_template(self, filepath: str, template_name: str):
        """从JSON加载模板"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        maneuvers = [BasicManeuver(**m) for m in data["maneuvers"]]
        self.composite_maneuvers[template_name] = CompositeManeuver(
            name=data["name"],
            maneuvers=maneuvers,
            total_duration=data["total_duration"],
            description=data["description"]
        )
        logging.info(f"加载模板 {template_name} 从 {filepath}")

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
        """
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
        """
        Beam机动 - 横向态势机动，消耗导弹动能
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
        """
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
            "accelerate": "加速 - 增加速度",
            "decelerate": "减速 - 减少速度",
            "turn": "转弯 - 改变航向",
            "pull_up": "拉起 - 爬升",
            "dive": "俯冲 - 下降",
            "diagonal_flight": "斜直飞 - 同时改变航向和高度",
            "roll": "滚转 - 改变滚转角",
            "turn_pull_up": "转弯拉起 - 同时转弯和爬升",
            "turn_dive": "转弯俯冲 - 同时转弯和下降",
            "escape": "脱离机动 - 快速脱离威胁区域",
            "attack": "侧跃升拉起俯冲攻击 - 经典的攻击机动",
            "defense": "防御机动 - 综合防御机动"
        }
        return descriptions.get(maneuver_name, "未知机动")