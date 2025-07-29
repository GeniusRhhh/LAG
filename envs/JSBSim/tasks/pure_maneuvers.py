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
    """基础机动库 - 保持不变"""

    @staticmethod
    def level_flight(time_sec: float, duration: float = 10.0, current_altitude: float = 0.0,
                     current_velocity: float = 0.0):
        """平飞 - 完全不改变任何控制量"""
        if time_sec <= duration:
            return "LEVEL_FLIGHT", None, None, None, None
        return None, None, None, None, None

    @staticmethod
    def accelerate(time_sec: float, current_velocity: float, duration: float = 5.0, velocity_increase: float = 50.0,
                   max_velocity: float = 350.0):
        """加速 - 只改变速度，绝对不改变航向和高度"""
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
        """减速 - 只改变速度，绝对不改变航向和高度"""
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
        """转弯 - 基于目标角度的精确控制版本"""
        # 计算目标航向
        target_final_heading = normalize_heading(initial_heading + turn_angle)

        # 基础转弯时间 + 足够的调整时间
        base_turn_time = abs(turn_angle) / turn_rate
        max_turn_time = base_turn_time + 15.0  # 给足够时间确保达到目标

        if time_sec <= max_turn_time:
            # 转弯阶段：直接返回目标角度，让控制系统处理实时反馈
            progress = min(time_sec / base_turn_time, 1.0)

            # 使用平滑的进度曲线
            smooth_progress = 3 * progress ** 2 - 2 * progress ** 3
            current_target = initial_heading + turn_angle * smooth_progress
            current_target = normalize_heading(current_target)

            # 计算滚转角 - 减小以降低高度变化
            if progress < 1.0:
                required_roll = abs(turn_rate) * 6.0  # 降低滚转角强度
                roll_magnitude = min(required_roll, 25.0)  # 限制最大滚转角
                target_roll = roll_magnitude * (1 if turn_angle > 0 else -1) * math.sin(progress * math.pi)
            else:
                target_roll = 0.0  # 转弯完成后归零滚转角

            phase = "TURNING" if progress < 0.95 else "TURN_ADJUSTING"
            return phase, current_target, None, None, target_roll
        else:
            # 转弯完成
            return "TURN_FINISHED", target_final_heading, None, None, 0.0

    @staticmethod
    def turn_level(time_sec: float, initial_heading: float, initial_altitude: float,
                   turn_angle: float = 45.0, turn_rate: float = 3.0):
        """保持高度的转弯 - 专门抑制高度变化"""
        # 计算目标航向
        target_final_heading = normalize_heading(initial_heading + turn_angle)

        # 基础转弯时间 + 调整时间
        base_turn_time = abs(turn_angle) / turn_rate
        max_turn_time = base_turn_time + 10.0

        if time_sec <= max_turn_time:
            progress = min(time_sec / base_turn_time, 1.0)
            smooth_progress = 3 * progress ** 2 - 2 * progress ** 3
            current_target = initial_heading + turn_angle * smooth_progress
            current_target = normalize_heading(current_target)

            # 极小的滚转角，最大限度减少高度变化
            if progress < 1.0:
                required_roll = abs(turn_rate) * 4.0  # 进一步降低滚转角
                roll_magnitude = min(required_roll, 20.0)  # 限制最大滚转角到20度
                target_roll = roll_magnitude * (1 if turn_angle > 0 else -1) * math.sin(progress * math.pi)
            else:
                target_roll = 0.0

            # 强制保持初始高度
            target_altitude = initial_altitude

            phase = "TURNING_LEVEL" if progress < 0.95 else "TURN_LEVEL_ADJUSTING"
            return phase, current_target, target_altitude, 0.0, target_roll
        else:
            return "TURN_LEVEL_FINISHED", target_final_heading, initial_altitude, 0.0, 0.0

    @staticmethod
    def pull_up(time_sec: float, initial_altitude: float, duration: float = 8.0, altitude_gain: float = 1000.0):
        """拉起 - 只改变高度，不改变航向"""
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
        """俯冲 - 抑制高度补偿版本，防止JSBSim自动升力补偿"""
        target_final_altitude = max(initial_altitude - altitude_loss, min_altitude)
        actual_altitude_loss = initial_altitude - target_final_altitude

        if time_sec <= duration:
            progress = time_sec / duration
            # 使用线性下降，避免过于激进
            target_altitude = initial_altitude - actual_altitude_loss * progress

            # 确保不低于最小高度
            target_altitude = max(target_altitude, min_altitude)

            # 完全消除速度补偿，避免JSBSim的升力补偿
            # 不给任何速度补偿，让飞机自然下降
            return "DIVING", None, target_altitude, 0.0, None
        else:
            # 俯冲完成，确保达到目标高度
            return "DIVE_FINISHED", None, target_final_altitude, 0.0, None

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

            velocity_compensation = -abs(
                actual_altitude_change) * 0.01 / duration if actual_altitude_change > 0 else abs(
                actual_altitude_change) * 0.01 / duration

            return "DIAGONAL_FLIGHT", target_heading, target_altitude, velocity_compensation, target_roll
        else:
            return "DIAGONAL_FLIGHT_COMPLETE", normalize_heading(
                initial_heading + heading_change), target_final_altitude, None, None

    @staticmethod
    def maintain_heading_flight(time_sec: float, target_heading: float, duration: float = 10.0):
        """保持指定航向的平飞 - 用于战术机动"""
        if time_sec <= duration:
            return "MAINTAINING_HEADING", target_heading, None, None, None
        return "MAINTAIN_COMPLETE", target_heading, None, None, None

    @staticmethod
    def accelerate_escape(time_sec: float, target_heading: float, duration: float = 15.0,
                         acceleration: float = 50.0):
        """加速逃离 - 保持航向并加速"""
        if time_sec <= duration:
            # 计算速度增量，前半段加速，后半段保持
            if time_sec <= duration / 2:
                # 前半段：逐渐加速
                progress = time_sec / (duration / 2)
                velocity_offset = acceleration * progress
            else:
                # 后半段：保持最大加速
                velocity_offset = acceleration

            return "ACCELERATING_ESCAPE", target_heading, None, velocity_offset, None
        return "ESCAPE_COMPLETE", target_heading, None, acceleration, None



    @staticmethod
    def high_g_turn(time_sec: float, initial_heading: float, turn_angle: float = 180.0,
                   g_force: float = 7.0, turn_rate: float = 8.0, initial_roll: float = 0.0):
        """高G力转弯 - 在当前滚转姿态下进行水平转弯"""
        turn_duration = abs(turn_angle) / turn_rate

        if time_sec <= turn_duration:
            # 高G转弯：快速改变航向，保持高度
            progress = time_sec / turn_duration
            smooth_progress = 3 * progress ** 2 - 2 * progress ** 3

            current_turn = turn_angle * smooth_progress
            current_heading = normalize_heading(initial_heading + current_turn)

            # 关键：在初始滚转角基础上增加转弯滚转
            # 如果已经是倒飞（135度），就在倒飞状态下转弯
            turn_roll = 30.0 * math.sin(progress * math.pi)  # 转弯时的额外滚转
            current_roll = initial_roll + turn_roll

            # 高度控制：如果是倒飞状态，需要补偿
            if abs(initial_roll) > 90.0:
                altitude_offset = 100.0 * math.sin(progress * math.pi)  # 倒飞时的高度补偿
            else:
                altitude_offset = 0.0

            return "HIGH_G_TURNING", current_heading, altitude_offset, None, current_roll
        else:
            final_heading = normalize_heading(initial_heading + turn_angle)
            # 转弯完成后，保持初始滚转角
            return "HIGH_G_TURN_COMPLETE", final_heading, None, None, initial_roll

    @staticmethod
    def vertical_loop(time_sec: float, initial_heading: float, loop_type: str = "full",
                      g_force: float = 6.0, initial_roll: float = 0.0):
        """垂直回旋 - 在当前滚转姿态下进行垂直机动"""
        if loop_type == "full":
            # 完整垂直回旋：上升-倒飞-下降-恢复
            loop_duration = 20.0  # 完整回旋20秒
        else:
            # 半回旋：上升-倒飞-恢复
            loop_duration = 12.0  # 半回旋12秒

        if time_sec <= loop_duration:
            progress = time_sec / loop_duration

            if loop_type == "full":
                # 完整360度垂直回旋
                loop_progress = progress * 2 * math.pi  # 0到2π
            else:
                # 180度半回旋
                loop_progress = progress * math.pi  # 0到π

            # 大幅减少高度变化，避免坠毁
            max_altitude_change = 200.0  # 从800米减少到200米
            altitude_offset = max_altitude_change * math.sin(loop_progress) * 0.3  # 再减少到30%

            # 减少俯仰角，避免过度机动
            max_pitch = 20.0  # 从60度减少到20度
            pitch_angle = max_pitch * math.sin(loop_progress)

            # 关键修复：保持初始滚转角，不改变！
            # 如果是倒飞状态（135度），就保持倒飞进行垂直机动
            current_roll = initial_roll

            # 航向在垂直回旋中的变化
            if loop_type == "half" and progress > 0.5:
                # 半回旋后期，航向开始反向
                heading_change = 180.0 * (progress - 0.5) * 2  # 后半段逐渐转向180度
                current_heading = normalize_heading(initial_heading + heading_change)
            else:
                current_heading = initial_heading

            return "VERTICAL_LOOPING", current_heading, altitude_offset, None, current_roll
        else:
            # 垂直回旋完成，恢复水平飞行
            if loop_type == "half":
                # 半回旋完成后，航向反向180度
                final_heading = normalize_heading(initial_heading + 180.0)
            else:
                # 完整回旋后，航向回到原方向
                final_heading = initial_heading
            # 保持最终的滚转角，不恢复到0度
            return "VERTICAL_LOOP_COMPLETE", final_heading, None, None, initial_roll

    @staticmethod
    def adaptive_crank(time_sec: float, initial_heading: float, crank_angle: float = 45.0,
                      turn_rate: float = 4.0, enemy_heading: float = 180.0):
        """自适应Crank机动 - 根据敌机方向选择Crank方向"""
        # 计算到敌机的角度差
        angle_to_enemy = normalize_heading(enemy_heading - initial_heading)

        # 选择Crank方向：选择较短的转弯路径
        if angle_to_enemy <= 180.0:
            # 敌机在右侧，向右Crank
            crank_direction = 1.0
        else:
            # 敌机在左侧，向左Crank
            crank_direction = -1.0

        actual_crank_angle = crank_angle * crank_direction

        # 执行转弯
        turn_duration = abs(actual_crank_angle) / turn_rate

        if time_sec <= turn_duration:
            progress = time_sec / turn_duration
            current_turn = actual_crank_angle * progress
            current_heading = normalize_heading(initial_heading + current_turn)
            return "ADAPTIVE_CRANKING", current_heading, None, None, None
        else:
            final_heading = normalize_heading(initial_heading + actual_crank_angle)
            return "ADAPTIVE_CRANK_COMPLETE", final_heading, None, None, None

    @staticmethod
    def adaptive_turn_to_enemy(time_sec: float, initial_heading: float, turn_rate: float = 5.0,
                              enemy_heading: float = 180.0):
        """自适应转向敌机 - 根据敌机方向选择最短路径转向"""
        # 计算需要转向的角度
        angle_to_enemy = normalize_heading(enemy_heading - initial_heading)

        # 选择最短转弯路径
        if angle_to_enemy > 180.0:
            turn_angle = angle_to_enemy - 360.0  # 向左转
        else:
            turn_angle = angle_to_enemy  # 向右转

        # 执行转弯
        turn_duration = abs(turn_angle) / turn_rate

        if time_sec <= turn_duration:
            progress = time_sec / turn_duration
            current_turn = turn_angle * progress
            current_heading = normalize_heading(initial_heading + current_turn)
            return "ADAPTIVE_TURNING", current_heading, None, None, None
        else:
            # 转向完成，正对敌机
            return "ADAPTIVE_TURN_COMPLETE", enemy_heading, None, None, None

    @staticmethod
    def sliceback_vertical_loop(time_sec: float, initial_heading: float, loop_type: str = "half",
                               g_force: float = 6.0, inverted: bool = True):
        """Sliceback专用垂直回旋 - 直接设置为倒飞状态进行拉回"""
        # 直接调用vertical_loop，但设置为倒飞状态
        return BasicManeuvers.vertical_loop(time_sec, initial_heading, loop_type, g_force, 135.0)

@dataclass
class ManeuverStep:
    """单个机动步骤定义"""
    name: str  # 机动名称
    params: Dict[str, Any]  # 机动参数
    duration: float  # 持续时间


class CompositeManeuverExecutor:
    """全新的组合机动执行器 - 简单直接的拼装方式"""

    def __init__(self):
        self.basic_maneuvers = BasicManeuvers()
        self.maneuver_definitions = {}
        self.active_states = {}
        self._setup_predefined_maneuvers()

    def _setup_predefined_maneuvers(self):
        """设置预定义的组合机动"""

        # 转弯拉起：先转弯，再拉起 - 增加时间确保完成
        self.maneuver_definitions["turn_pull_up"] = [
            ManeuverStep("turn", {"turn_angle": 90.0, "turn_rate": 3.0}, 35.0),  # 增加到35秒
            ManeuverStep("pull_up", {"altitude_gain": 2000.0}, 20.0)  # 增加到20秒
        ]

        # 转弯俯冲：先转弯，再俯冲 - 增加时间确保完成
        self.maneuver_definitions["turn_dive"] = [
            ManeuverStep("turn", {"turn_angle": 90.0, "turn_rate": 3.0}, 35.0),  # 增加到35秒
            ManeuverStep("dive", {"altitude_loss": 1500.0, "min_altitude": 2000.0}, 20.0)  # 增加到20秒
        ]

        # 盘旋爬升：360度转弯 + 拉起 - 确保有足够时间
        self.maneuver_definitions["spiral_climb"] = [
            ManeuverStep("turn", {"turn_angle": 360.0, "turn_rate": 2.0}, 180.0),  # 360度需要180秒
            ManeuverStep("pull_up", {"altitude_gain": 1500.0}, 25.0)  # 增加拉起时间
        ]
        # 1. Crank机动：转70度 + 保持新航向（保持高度版）
        self.maneuver_definitions["crank_tactical"] = [
            ManeuverStep("turn_level", {
                "turn_angle": 30.0,
                "turn_rate": 4.0  # 降低转弯率以减少高度变化
            }, 20.0),  # 给足够时间确保角度精确
            ManeuverStep("maintain_heading_flight", {}, 30.0)
        ]

        # 2. Beam机动：转90度 + 保持横向态势（保持高度版）
        self.maneuver_definitions["beam_tactical"] = [
            ManeuverStep("turn_level", {
                "turn_angle": 90.0,
                "turn_rate": 4.0  # 降低转弯率减少高度变化
            }, 25.0),  # 给足够时间确保90度精确
            ManeuverStep("maintain_heading_flight", {}, 35.0)
        ]

        # 3. Notch机动：下降 + 稳定 + 转90度 + 保持（保持高度版）
        self.maneuver_definitions["notch_tactical"] = [
            ManeuverStep("dive", {
                "altitude_loss": 1800.0,
                "min_altitude": 2500.0
            }, 10.0),  # 充分的下降时间
            ManeuverStep("maintain_heading_flight", {}, 3.0),  # 稳定在低高度
            ManeuverStep("turn_level", {  # 使用turn_level保持低高度
                "turn_angle": 90.0,
                "turn_rate": 4.0
            }, 25.0),
            ManeuverStep("maintain_heading_flight", {}, 16.0)  # 保持低高度新航向
        ]

        # 4. Short Skate机动：激进的脱离式火力投送战术
        # Short Skate: 发射 → Crank维持锁定 → Turn Cold快速脱离 → 加速逃离
        self.maneuver_definitions["short_skate_tactical"] = [
            # 阶段1：保持航向准备发射（模拟发射阶段）
            ManeuverStep("maintain_heading_flight", {}, 5.0),  # 保持5秒准备发射

            # 阶段2：Crank机动 - 偏离40度维持雷达锁定
            ManeuverStep("turn_level", {
                "turn_angle": 40.0,  # Crank角度40度
                "turn_rate": 5.0     # 稍快的转弯率
            }, 12.0),  # 缩短Crank转弯时间

            # 阶段3：短暂保持Crank角度
            ManeuverStep("maintain_heading_flight", {}, 6.0),  # 保持Crank角度6秒

            # 阶段4：Turn Cold - 快速掉头脱离
            ManeuverStep("turn_level", {
                "turn_angle": 100.0,  # 100度掉头（总共140度）
                "turn_rate": 6.0      # 更快的转弯率用于快速脱离
            }, 25.0),  # 缩短转弯时间

            # 阶段5：加速逃离 - 保持逃逸航向并加速
            ManeuverStep("accelerate_escape", {
                "acceleration": 50.0  # 加速50m/s
            }, 20.0)  # 加速逃离20秒
        ]

        # 5. Banzai机动：发射后决策战术（Launch & Decide）
        # 在MAR外发射，然后Crank，敌机仍存活则交汇格斗
        self.maneuver_definitions["banzai_tactical"] = [
            # 阶段1：保持航向准备发射
            ManeuverStep("maintain_heading_flight", {}, 8.0),  # 保持8秒准备发射

            # 阶段2：发射后Crank机动 - 根据敌机方向选择Crank方向
            ManeuverStep("adaptive_crank", {
                "crank_angle": 45.0,  # Crank角度45度（方向自适应）
                "turn_rate": 4.0      # 标准转弯率
            }, 18.0),  # Crank转弯阶段

            # 阶段3：保持Crank态势观察敌机
            ManeuverStep("maintain_heading_flight", {}, 12.0),  # 保持Crank角度12秒

            # 阶段4：决策阶段 - 转向迎敌准备格斗
            ManeuverStep("adaptive_turn_to_enemy", {
                "turn_rate": 5.0  # 较快转弯准备格斗
            }, 30.0),  # 转向迎敌阶段，增加时间确保完成转弯

            # 阶段5：保持迎敌航向准备交汇格斗
            ManeuverStep("maintain_heading_flight", {}, 15.0)  # 保持迎敌航向
        ]

        # 6. Sliceback机动：直接使用vertical_loop实现倒飞拉回
        # 简化设计，只保留有效的机动
        self.maneuver_definitions["sliceback_tactical"] = [
            # 直接使用vertical_loop，设置为倒飞状态
            ManeuverStep("sliceback_vertical_loop", {
                "loop_type": "half",  # 半回旋（180度垂直）
                "g_force": 6.0,       # 高G力6G
                "inverted": True      # 标记为倒飞拉回
            }, 15.0),  # 倒飞拉回阶段

            # 恢复平飞
            ManeuverStep("maintain_heading_flight", {}, 10.0)  # 恢复平飞
        ]

    def update_maneuver_params(self, maneuver_name: str, custom_params: Dict[str, Any]):
        """更新组合机动的参数"""
        if maneuver_name not in self.maneuver_definitions:
            logging.warning(f"❌ 未知的组合机动: {maneuver_name}")
            return

        # 根据传入的参数更新机动定义
        if maneuver_name == "turn_pull_up" and "turn_pull_up" in custom_params:
            params = custom_params["turn_pull_up"]
            # 更新转弯步骤
            self.maneuver_definitions[maneuver_name][0].params.update({
                "turn_angle": params.get("turn_angle", 90.0),
                "turn_rate": params.get("turn_rate", 3.0)
            })
            self.maneuver_definitions[maneuver_name][0].duration = params.get("turn_duration", 30.0)

            # 更新拉起步骤
            self.maneuver_definitions[maneuver_name][1].params.update({
                "altitude_gain": params.get("altitude_gain", 2000.0)
            })
            self.maneuver_definitions[maneuver_name][1].duration = params.get("pull_up_duration", 15.0)

        elif maneuver_name == "turn_dive" and "turn_dive" in custom_params:
            params = custom_params["turn_dive"]
            # 更新转弯步骤
            self.maneuver_definitions[maneuver_name][0].params.update({
                "turn_angle": params.get("turn_angle", 90.0),
                "turn_rate": params.get("turn_rate", 3.0)
            })
            self.maneuver_definitions[maneuver_name][0].duration = params.get("turn_duration", 30.0)

            # 更新俯冲步骤
            self.maneuver_definitions[maneuver_name][1].params.update({
                "altitude_loss": params.get("altitude_loss", 1500.0),
                "min_altitude": params.get("min_altitude", 2000.0)
            })
            self.maneuver_definitions[maneuver_name][1].duration = params.get("dive_duration", 15.0)

        logging.info(f"✅ 组合机动 {maneuver_name} 参数更新完成")
        for i, step in enumerate(self.maneuver_definitions[maneuver_name]):
            logging.info(f"   步骤{i + 1}: {step.name} {step.params} 持续{step.duration}s")

    def execute_composite_maneuver(self, maneuver_name: str, total_time: float,
                                   initial_heading: float, initial_altitude: float,
                                   current_velocity: float, current_altitude: float) -> Tuple:
        """执行组合机动 - 精确控制版本"""

        if maneuver_name not in self.maneuver_definitions:
            logging.warning(f"❌ 未知的组合机动: {maneuver_name}")
            return None, None, None, None, None

        steps = self.maneuver_definitions[maneuver_name]

        # 初始化状态
        if maneuver_name not in self.active_states:
            self.active_states[maneuver_name] = {
                "current_step": -1,
                "step_start_time": 0.0,
                "step_initial_heading": initial_heading,
                "step_initial_altitude": initial_altitude,
                "final_heading": initial_heading,
                "final_altitude": initial_altitude
            }

        state = self.active_states[maneuver_name]

        # 计算当前步骤
        elapsed_time = 0.0
        current_step_index = -1

        for i, step in enumerate(steps):
            if total_time <= elapsed_time + step.duration:
                current_step_index = i
                break
            elapsed_time += step.duration

        if current_step_index == -1:
            # 机动完成后，保持最终状态而不是返回None
            if maneuver_name in self.active_states:
                state = self.active_states[maneuver_name]

                # 优先使用实际的最终状态
                if "final_heading" in state and state["final_heading"] is not None:
                    final_heading = state["final_heading"]
                else:
                    # 备用方案：计算理论上的最终航向
                    theoretical_final_heading = initial_heading
                    for step in steps:
                        if step.name in ["turn", "turn_level"]:
                            theoretical_final_heading = normalize_heading(
                                theoretical_final_heading + step.params["turn_angle"]
                            )
                    final_heading = theoretical_final_heading

                final_altitude = state.get("final_altitude", initial_altitude)

                # 调试信息
                logging.debug(f"{maneuver_name} 完成: 最终航向={final_heading:.1f}°")

                # 保持最终状态，继续平稳飞行
                return "MANEUVER_COMPLETED", final_heading, final_altitude, 0.0, 0.0
            else:
                # 如果没有状态记录，保持初始状态
                return "MANEUVER_COMPLETED", initial_heading, initial_altitude, 0.0, 0.0

        current_step = steps[current_step_index]

        # 步骤切换处理（简化版）
        if current_step_index != state["current_step"]:
            # 计算累积状态
            step_initial_heading = state["step_initial_heading"]
            step_initial_altitude = state["step_initial_altitude"]

            # 使用实际的最终状态而不是累积计算
            if current_step_index > 0:
                # 优先使用实际的最终状态
                if "final_heading" in state and state["final_heading"] is not None:
                    step_initial_heading = state["final_heading"]
                    logging.debug(f"使用实际最终航向: {step_initial_heading:.1f}°")
                else:
                    # 备用方案：累积计算
                    for i in range(current_step_index):
                        prev_step = steps[i]
                        if prev_step.name in ["turn", "turn_level"]:
                            old_heading = step_initial_heading
                            step_initial_heading = normalize_heading(
                                step_initial_heading + prev_step.params["turn_angle"]
                            )
                            logging.debug(f"累积计算 步骤{i+1} {prev_step.name}: {old_heading:.1f}° + {prev_step.params['turn_angle']:.1f}° = {step_initial_heading:.1f}°")

                # 高度处理
                if "final_altitude" in state and state["final_altitude"] is not None:
                    step_initial_altitude = state["final_altitude"]
                else:
                    # 备用方案：累积计算
                    for i in range(current_step_index):
                        prev_step = steps[i]
                        if prev_step.name == "dive":
                            step_initial_altitude -= prev_step.params["altitude_loss"]
                            step_initial_altitude = max(step_initial_altitude,
                                                        prev_step.params.get("min_altitude", 2000.0))
                        elif prev_step.name == "pull_up":
                            step_initial_altitude += prev_step.params["altitude_gain"]

            # 更新状态
            state["current_step"] = current_step_index
            state["step_start_time"] = elapsed_time
            state["step_initial_heading"] = step_initial_heading
            state["step_initial_altitude"] = step_initial_altitude

            logging.info(f"🔄 组合机动 {maneuver_name} 步骤 {current_step_index + 1}: {current_step.name}")
            logging.info(f"   起始: 航向={step_initial_heading:.1f}°, 高度={step_initial_altitude:.1f}m")

        # 执行当前步骤
        step_time = total_time - state["step_start_time"]

        # 调用相应的基础机动
        if current_step.name == "turn":
            result = self.basic_maneuvers.turn(
                step_time,
                state["step_initial_heading"],
                current_step.params["turn_angle"],
                current_step.params["turn_rate"]
            )
            # 更新最终状态
            if result[0] in ["TURN_FINISHED", "TURN_ADJUSTING"]:
                state["final_heading"] = result[1] if result[1] is not None else state["final_heading"]
            return result
        elif current_step.name == "turn_level":
            # 对于turn_level，如果有实际的最终高度，使用实际高度而不是累积计算的高度
            if "final_altitude" in state and state["final_altitude"] is not None:
                target_altitude = state["final_altitude"]
            else:
                target_altitude = state["step_initial_altitude"]

            result = self.basic_maneuvers.turn_level(
                step_time,
                state["step_initial_heading"],
                target_altitude,  # 使用实际高度
                current_step.params["turn_angle"],
                current_step.params["turn_rate"]
            )
            # 更新最终状态
            if result[0] in ["TURN_LEVEL_FINISHED", "TURN_LEVEL_ADJUSTING"]:
                state["final_heading"] = result[1] if result[1] is not None else state["final_heading"]
                state["final_altitude"] = result[2] if result[2] is not None else state["final_altitude"]
            return result
        elif current_step.name == "maintain_heading_flight":
            # 保持航向：使用最终状态的航向，如果没有则使用累积航向
            if "final_heading" in state and state["final_heading"] is not None:
                target_heading = state["final_heading"]
            else:
                target_heading = state["step_initial_heading"]
            result = self.basic_maneuvers.maintain_heading_flight(
                step_time,
                target_heading,
                current_step.duration
            )
            # 更新最终状态
            if result[0] in ["MAINTAINING_HEADING", "MAINTAIN_COMPLETE"]:
                state["final_heading"] = result[1] if result[1] is not None else state["final_heading"]
            return result
        elif current_step.name == "dive":
            result = self.basic_maneuvers.dive(
                step_time,
                state["step_initial_altitude"],
                current_step.duration,
                current_step.params["altitude_loss"],
                current_step.params.get("min_altitude", 3000.0)
            )
            # 更新最终状态
            if result[0] in ["DIVING", "DIVE_FINISHED"]:
                state["final_altitude"] = result[2] if result[2] is not None else state["final_altitude"]
            return result
        elif current_step.name == "pull_up":
            return self.basic_maneuvers.pull_up(
                step_time,
                state["step_initial_altitude"],
                current_step.duration,
                current_step.params["altitude_gain"]
            )
        elif current_step.name == "accelerate_escape":
            # 加速逃离：使用最终状态的航向
            if "final_heading" in state and state["final_heading"] is not None:
                target_heading = state["final_heading"]
            else:
                target_heading = state["step_initial_heading"]
            result = self.basic_maneuvers.accelerate_escape(
                step_time,
                target_heading,
                current_step.duration,
                current_step.params.get("acceleration", 50.0)
            )
            # 更新最终状态
            if result[0] in ["ACCELERATING_ESCAPE", "ESCAPE_COMPLETE"]:
                state["final_heading"] = result[1] if result[1] is not None else state["final_heading"]
            return result

        elif current_step.name == "high_g_turn":
            # 获取当前的滚转角状态
            current_roll = state.get("current_roll", 0.0)

            result = self.basic_maneuvers.high_g_turn(
                step_time,
                state["step_initial_heading"],
                current_step.params.get("turn_angle", 180.0),
                current_step.params.get("g_force", 7.0),
                current_step.params.get("turn_rate", 8.0),
                current_roll  # 传递当前滚转角
            )
            # 更新最终状态
            if result[0] in ["HIGH_G_TURNING", "HIGH_G_TURN_COMPLETE"]:
                state["final_heading"] = result[1] if result[1] is not None else state["final_heading"]
                # 保存滚转角状态
                if len(result) > 4 and result[4] is not None:
                    state["current_roll"] = result[4]
            return result
        elif current_step.name == "vertical_loop":
            # 获取当前的滚转角状态（来自前一个机动）
            current_roll = state.get("current_roll", 0.0)

            result = self.basic_maneuvers.vertical_loop(
                step_time,
                state["step_initial_heading"],
                current_step.params.get("loop_type", "half"),
                current_step.params.get("g_force", 6.0),
                current_roll  # 传递当前滚转角
            )
            # 更新最终状态
            if result[0] in ["VERTICAL_LOOPING", "VERTICAL_LOOP_COMPLETE"]:
                state["final_heading"] = result[1] if result[1] is not None else state["final_heading"]
                # 保存滚转角状态
                if len(result) > 4 and result[4] is not None:
                    state["current_roll"] = result[4]
            return result
        elif current_step.name == "adaptive_crank":
            # 简化为固定的Crank机动，避免复杂的敌机检测
            # 根据敌机通常在180度方向，选择合适的Crank方向
            initial_heading = state["step_initial_heading"]
            if initial_heading <= 90 or initial_heading >= 270:
                # 朝北或接近北，向右Crank
                crank_angle = 45.0
            else:
                # 朝南或接近南，向左Crank
                crank_angle = -45.0

            result = self.basic_maneuvers.turn_level(
                step_time,
                initial_heading,
                state.get("step_initial_altitude", 6096.0),
                crank_angle,
                current_step.params.get("turn_rate", 4.0)
            )
            # 更新最终状态
            if result[0] in ["TURN_LEVEL_FINISHED", "TURN_LEVEL_ADJUSTING"]:
                state["final_heading"] = result[1] if result[1] is not None else state["final_heading"]
            return result
        elif current_step.name == "adaptive_turn_to_enemy":
            # 攻击性转向：从Crank位置转向敌机（180度方向）
            initial_heading = state["step_initial_heading"]

            # 计算从当前位置到敌机（180度）的转向角度
            target_heading = 180.0
            angle_diff = normalize_heading(target_heading - initial_heading)

            # 选择最短路径转向敌机
            if angle_diff > 180.0:
                turn_angle = angle_diff - 360.0  # 向左转
            else:
                turn_angle = angle_diff  # 向右转

            logging.debug(f"攻击转向: 从{initial_heading:.1f}°转向敌机{target_heading:.1f}°, 转角{turn_angle:.1f}°")

            result = self.basic_maneuvers.turn_level(
                step_time,
                initial_heading,
                state.get("step_initial_altitude", 6096.0),
                turn_angle,
                current_step.params.get("turn_rate", 5.0)
            )
            # 更新最终状态
            if result[0] in ["TURN_LEVEL_FINISHED", "TURN_LEVEL_ADJUSTING"]:
                state["final_heading"] = result[1] if result[1] is not None else state["final_heading"]
            return result

        elif current_step.name == "hold_altitude_and_heading":
            return self.basic_maneuvers.hold_altitude_and_heading(
                step_time,
                state["final_heading"],
                state["final_altitude"],
                current_step.duration
            )
        elif current_step.name == "enter_inverted_flight":
            result = self.basic_maneuvers.enter_inverted_flight(
                step_time,
                state["step_initial_heading"],
                current_step.params.get("duration", 5.0)
            )
            # 更新最终状态
            if result[0] in ["ENTERING_INVERTED", "INVERTED_FLIGHT_READY"]:
                state["final_heading"] = result[1] if result[1] is not None else state["final_heading"]
                # 保存滚转角状态
                if len(result) > 4 and result[4] is not None:
                    state["current_roll"] = result[4]
                    logging.debug(f"进入倒飞状态，保存滚转角: {result[4]:.1f}°")
            return result
        elif current_step.name == "sliceback_vertical_loop":
            result = self.basic_maneuvers.sliceback_vertical_loop(
                step_time,
                state["step_initial_heading"],
                current_step.params.get("loop_type", "half"),
                current_step.params.get("g_force", 6.0),
                current_step.params.get("inverted", True)
            )
            # 更新最终状态
            if result[0] in ["VERTICAL_LOOPING", "VERTICAL_LOOP_COMPLETE"]:
                state["final_heading"] = result[1] if result[1] is not None else state["final_heading"]
            return result
        else:
            logging.warning(f"❌ 未知的基础机动: {current_step.name}")
            return None, None, None, None, None

    def reset_maneuver_state(self, maneuver_name: str):
        """重置组合机动状态"""
        if maneuver_name in self.active_states:
            del self.active_states[maneuver_name]
            logging.info(f"🔄 重置组合机动 {maneuver_name} 状态")

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


# 保持旧的接口兼容性
class ManeuverComposer:
    """兼容性包装器"""

    def __init__(self):
        self.executor = CompositeManeuverExecutor()
        self.composite_maneuvers = {}
        self.maneuver_states = {}

    def _setup_tactical_templates(self, custom_params=None):
        """设置战术模板 - 使用新的执行器"""
        if custom_params:
            for maneuver_name in ["turn_pull_up", "turn_dive"]:
                if maneuver_name in custom_params or any(key in custom_params for key in [maneuver_name]):
                    self.executor.update_maneuver_params(maneuver_name, custom_params)

    def execute_composite_maneuver(self, maneuver_name: str, time_sec: float,
                                   initial_heading: float, initial_altitude: float,
                                   params: Dict[str, Any] = None):
        """执行组合机动 - 使用新的执行器"""
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
    """纯机动函数库 - 保持不变"""

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
            "turn_dive": "转弯俯冲组合 - 先转弯后俯冲"
        }
        return descriptions.get(maneuver_name, "未知机动")