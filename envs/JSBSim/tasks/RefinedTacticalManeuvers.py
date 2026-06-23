import numpy as np
import logging
from typing import Dict, Any, Optional


class RefinedTacticalManeuvers:

    def __init__(self):
        self.maneuver_states = {}

    def get_maneuver_state(self, maneuver_name: str, agent_id: str):
        """Get or initialize maneuver state"""
        key = f"{agent_id}_{maneuver_name}"
        if key not in self.maneuver_states:
            self.maneuver_states[key] = {
                "phase": "init",
                "timer": 0,
                "target_heading": 0.0,
                "current_heading": 0.0,
                "target_altitude": 0.0,
                "initial_altitude": 0.0,
                "threat_direction": 0.0,
                "entry_conditions": {},
                "completed": False
            }
        return self.maneuver_states[key]

    def reset_maneuver_state(self, agent_id: str, maneuver_name: Optional[str] = None):
        """Reset state for a specific maneuver or all maneuvers for an agent."""
        if maneuver_name:
            key_prefix = f"{agent_id}_{maneuver_name}"
            keys_to_delete = [k for k in self.maneuver_states if k.startswith(key_prefix)]
        else:
            key_prefix = f"{agent_id}_"
            keys_to_delete = [k for k in self.maneuver_states if k.startswith(key_prefix)]

        for k in keys_to_delete:
            del self.maneuver_states[k]
        logging.debug(f"Reset maneuver states for key prefix: {key_prefix}")


    def _calculate_threat_direction(self, state: Dict[str, Any]) -> float:
        """Calculate true threat direction"""
        missiles_incoming = state.get("missiles_incoming", [])
        enemy_angle_off = state.get("enemy_angle_off", 0)
        if missiles_incoming:
            missile_bearing = np.radians(enemy_angle_off)
            logging.debug(f"Missile threat direction: {np.degrees(missile_bearing):.1f}°")
            return missile_bearing
        return np.radians(enemy_angle_off)

    def _smooth_heading_change(self, current: float, target: float, rate: float = 0.05) -> float:
        """Smooth heading change"""
        diff = target - current
        if diff > np.pi:
            diff -= 2 * np.pi
        elif diff < -np.pi:
            diff += 2 * np.pi
        max_change = rate
        if abs(diff) <= max_change:
            return target
        return current + np.sign(diff) * max_change

    def _smooth_altitude_change(self, current: float, target: float, rate: float = 50.0) -> float:
        """Smooth altitude change"""
        diff = target - current
        max_change = rate
        if abs(diff) <= max_change:
            return target
        return current + np.sign(diff) * max_change

    # =================================================================================
    # NEW PARAMETERIZED MANEUVER FUNCTIONS
    # These functions are designed for isolated testing and demonstration.
    # They are driven by explicit parameters rather than a real-time state dict.
    # =================================================================================

    def execute_crank(self, agent_id: str, crank_angle_deg: float, velocity_kts: float, duration_steps: int) -> Dict[str, Any]:
        """
        Executes a Crank maneuver based on specific parameters.

        :param agent_id: ID of the agent performing the maneuver.
        :param crank_angle_deg: The desired angle for the crank, in degrees.
        :param velocity_kts: The desired velocity during the maneuver, in knots.
        :param duration_steps: The number of simulation steps to hold the crank.
        :return: A dictionary of control commands.
        """
        maneuver_state = self.get_maneuver_state("execute_crank", agent_id)
        maneuver_state["timer"] += 1

        if maneuver_state["timer"] > duration_steps:
            # Maneuver complete, return to neutral
            maneuver_state["completed"] = True
            logging.info(f"Crank for {agent_id} completed after {duration_steps} steps.")
            return {"heading_cmd": 0, "altitude_cmd": 0, "velocity_cmd": 340, "shoot": False, "maneuver_active": False}

        # Command a constant turn to the desired angle
        heading_cmd = np.radians(crank_angle_deg)
        # Convert knots to m/s for velocity command
        velocity_cmd = velocity_kts * 0.514444

        return {
            "heading_cmd": heading_cmd,
            "altitude_cmd": 0,  # Maintain altitude
            "velocity_cmd": velocity_cmd,
            "shoot": False,
            "maneuver_active": True
        }

    def execute_beam(self, agent_id: str, direction: str, velocity_kts: float, duration_steps: int) -> Dict[str, Any]:
        """
        Executes a Beam maneuver (90-degree turn).

        :param agent_id: ID of the agent.
        :param direction: 'left' or 'right'.
        :param velocity_kts: Desired velocity in knots.
        :param duration_steps: Duration in simulation steps.
        :return: Control commands.
        """
        maneuver_state = self.get_maneuver_state("execute_beam", agent_id)
        maneuver_state["timer"] += 1

        if maneuver_state["timer"] > duration_steps:
            maneuver_state["completed"] = True
            return {"heading_cmd": 0, "altitude_cmd": 0, "velocity_cmd": 340, "shoot": False, "maneuver_active": False}

        heading_cmd_deg = 90 if direction == 'left' else -90
        heading_cmd = np.radians(heading_cmd_deg)
        velocity_cmd = velocity_kts * 0.514444

        return {
            "heading_cmd": heading_cmd,
            "altitude_cmd": 0,
            "velocity_cmd": velocity_cmd,
            "shoot": False,
            "maneuver_active": True
        }

    def execute_notch(self, agent_id: str, direction: str, descent_rate_mps: float, velocity_kts: float, duration_steps: int) -> Dict[str, Any]:
        """
        Executes a Notch maneuver (Beam + descent).

        :param agent_id: ID of the agent.
        :param direction: 'left' or 'right' for the beam component.
        :param descent_rate_mps: Vertical speed in meters per second (should be negative).
        :param velocity_kts: Desired velocity in knots.
        :param duration_steps: Duration in simulation steps.
        :return: Control commands.
        """
        maneuver_state = self.get_maneuver_state("execute_notch", agent_id)
        maneuver_state["timer"] += 1

        if maneuver_state["timer"] > duration_steps:
            maneuver_state["completed"] = True
            return {"heading_cmd": 0, "altitude_cmd": 0, "velocity_cmd": 340, "shoot": False, "maneuver_active": False}

        heading_cmd_deg = 90 if direction == 'left' else -90
        heading_cmd = np.radians(heading_cmd_deg)
        velocity_cmd = velocity_kts * 0.514444

        return {
            "heading_cmd": heading_cmd,
            "altitude_cmd": descent_rate_mps, # This is a rate command
            "velocity_cmd": velocity_cmd,
            "shoot": False,
            "maneuver_active": True
        }

    def execute_skate(self, agent_id: str, target_state: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a multi-phase Skate maneuver.
        This maneuver logic is stateful and depends on a simulated target state.

        :param agent_id: ID of the agent.
        :param target_state: A dictionary with simulated enemy/missile info, e.g., {'distance_m': 50000, 'is_alive': True}.
        :param params: A dictionary of parameters for the maneuver, e.g.,
                       {'launch_range_m': 45000, 'turn_cold_duration_s': 20, 're_engage_range_m': 60000}.
        :return: Control commands.
        """
        maneuver_state = self.get_maneuver_state("execute_skate", agent_id)
        maneuver_state["timer"] += 1
        distance_m = target_state.get("distance_m", 100000)

        # Phase transitions are now based on parameters and simulated state
        if maneuver_state["phase"] == "init":
            if distance_m < params.get('launch_range_m', 45000):
                maneuver_state["phase"] = "launch"
                maneuver_state["timer"] = 0
                logging.info(f"Skate for {agent_id}: entering 'launch' phase.")
                return {"heading_cmd": 0, "altitude_cmd": 0, "velocity_cmd": 340, "shoot": True, "maneuver_active": True}

        elif maneuver_state["phase"] == "launch":
             # Wait for a few steps to simulate launch, then turn cold
            if maneuver_state["timer"] > 50: # 5 seconds at 10Hz
                maneuver_state["phase"] = "turn_cold"
                maneuver_state["timer"] = 0
                logging.info(f"Skate for {agent_id}: entering 'turn_cold' phase.")

        elif maneuver_state["phase"] == "turn_cold":
            if maneuver_state["timer"] > params.get('turn_cold_duration_steps', 200):
                maneuver_state["phase"] = "re_engage"
                maneuver_state["timer"] = 0
                logging.info(f"Skate for {agent_id}: entering 're_engage' phase.")
            # Turn 180 degrees away
            return {"heading_cmd": np.pi, "altitude_cmd": 0, "velocity_cmd": 400, "shoot": False, "maneuver_active": True}

        elif maneuver_state["phase"] == "re_engage":
            # Turn back towards the target
            if distance_m < params.get('re_engage_range_m', 60000):
                 # Simple logic to turn back, could be improved with target bearing
                maneuver_state["completed"] = True # End here for simplicity
                logging.info(f"Skate for {agent_id}: completed.")
                return {"heading_cmd": 0, "altitude_cmd": 0, "velocity_cmd": 340, "shoot": False, "maneuver_active": False}

        # Default action if no phase matches
        return {"heading_cmd": 0, "altitude_cmd": 0, "velocity_cmd": 340, "shoot": False, "maneuver_active": False}


    # =================================================================================
    # ORIGINAL REAL-TIME LOGIC (UNCHANGED)
    # =================================================================================

    def get_maneuver_action(self, template_id: int, state: Dict[str, Any]) -> Dict[str, Any]:
        """Get maneuver action based on template ID"""
        if template_id == 1:
            return self._crank_logic(state)
        elif template_id == 2:
            return self._beam_logic(state)
        elif template_id == 3:
            return self._notch_logic(state)
        elif template_id == 4:
            return self._skate_logic(state)
        elif template_id == 5:
            return self._short_skate_logic(state)
        elif template_id == 6:
            return self._banzai_logic(state)
        elif template_id == 7:
            return self._simple_f_pole_logic(state)
        elif template_id == 8:
            return self._advanced_f_pole_logic(state)
        elif template_id == 9:
            return self._pincer_logic(state)
        elif template_id == 10:
            return self._defensive_split_logic(state)
        elif template_id == 11:
            return self._high_low_logic(state)
        elif template_id == 12:
            return self._engaging_trail_logic(state)
        elif template_id == 13:
            return self._loose_deuce_logic(state)
        elif template_id == 14:
            return self._defensive_sequence_logic(state)
        else:
            logging.warning(f"Unknown template_id: {template_id}, using default action")
            return {"heading_cmd": 0, "altitude_cmd": 0, "velocity_cmd": 600, "shoot": False}

    def _crank_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Crank机动逻辑 - 多阶段状态机"""

        agent_id = state.get("agent_id", "A0100")
        maneuver_state = self.get_maneuver_state("crank", agent_id)

        enemy_distance = state.get("enemy_distance", 50000)
        enemy_angle_off = state.get("enemy_angle_off", 0)
        radar_lock = state.get("radar_lock", False)
        missile_distance = state.get("missile_distance", np.inf)
        has_missile_threat = len(
            [m for m in state.get("missiles_incoming", []) if hasattr(m, 'is_alive') and m.is_alive]) > 0
        current_altitude = state.get("current_altitude", 5000)

        # 触发条件检查
        should_crank = (
                enemy_distance < 65000 or
                missile_distance < 40000 or
                has_missile_threat
        )

        if not should_crank:
            return {"heading_cmd": 0, "altitude_cmd": 0, "velocity_cmd": 600, "shoot": False, "maneuver_active": False}

        maneuver_state["timer"] += 1

        # 阶段1：初始化和威胁评估
        if maneuver_state["phase"] == "init":
            # 计算威胁方向
            threat_direction = self._calculate_threat_direction(state)
            maneuver_state["threat_direction"] = threat_direction
            maneuver_state["current_heading"] = 0.0  # 当前相对航向

            # 决定Crank方向 - 避开威胁方向
            if has_missile_threat or missile_distance < 30000:
                # 高威胁：大角度规避
                crank_angle = np.radians(50)
                velocity_boost = 75
                threat_level = "HIGH"
            elif enemy_distance < 35000:
                # 中威胁：中等角度
                crank_angle = np.radians(40)
                velocity_boost = 50
                threat_level = "MEDIUM"
            else:
                # 低威胁：小角度
                crank_angle = np.radians(30)
                velocity_boost = 25
                threat_level = "LOW"

            # 选择规避方向：远离威胁方向
            if threat_direction > 0:
                maneuver_state["target_heading"] = -crank_angle  # 向左规避
            else:
                maneuver_state["target_heading"] = crank_angle  # 向右规避

            maneuver_state["velocity_boost"] = velocity_boost
            maneuver_state["threat_level"] = threat_level
            maneuver_state["phase"] = "turning"
            maneuver_state["timer"] = 0

            logging.info(f"Crank启动: 威胁={threat_level}, 目标角度={np.degrees(maneuver_state['target_heading']):.1f}°")

        # 阶段2：渐进式转向到目标角度
        elif maneuver_state["phase"] == "turning":
            target_heading = maneuver_state["target_heading"]
            current_heading = maneuver_state["current_heading"]

            # 平滑转向，避免跳跃
            new_heading = self._smooth_heading_change(current_heading, target_heading, rate=0.08)
            maneuver_state["current_heading"] = new_heading

            # 检查是否到达目标角度
            if abs(new_heading - target_heading) < np.radians(3):  # 3度容差
                maneuver_state["phase"] = "maintaining"
                maneuver_state["timer"] = 0
                logging.debug(f"Crank到达目标角度: {np.degrees(new_heading):.1f}°")

            # 超时保护
            elif maneuver_state["timer"] > 150:
                maneuver_state["phase"] = "maintaining"
                logging.warning(f"Crank转向超时，强制进入保持阶段")

        # 阶段3：保持Crank角度并监控威胁
        elif maneuver_state["phase"] == "maintaining":
            current_heading = maneuver_state["current_heading"]

            # 重新评估威胁
            current_threat_direction = self._calculate_threat_direction(state)

            # 如果威胁方向大幅变化，调整Crank角度
            threat_change = abs(current_threat_direction - maneuver_state["threat_direction"])
            if threat_change > np.radians(30):  # 威胁方向变化>30度
                maneuver_state["phase"] = "adjusting"
                maneuver_state["threat_direction"] = current_threat_direction
                logging.debug(f"威胁方向变化{np.degrees(threat_change):.1f}°，调整Crank")

            # 威胁解除条件
            elif (missile_distance > 50000 and not has_missile_threat and
                  enemy_distance > 70000 and maneuver_state["timer"] > 100):
                maneuver_state["phase"] = "recovering"
                maneuver_state["timer"] = 0
                logging.debug(f"威胁解除，Crank开始恢复")

        # 阶段4：调整Crank角度
        elif maneuver_state["phase"] == "adjusting":
            # 重新计算目标角度
            threat_direction = maneuver_state["threat_direction"]
            if has_missile_threat or missile_distance < 25000:
                crank_angle = np.radians(55)  # 增大角度
            else:
                crank_angle = np.radians(35)

            if threat_direction > 0:
                new_target = -crank_angle
            else:
                new_target = crank_angle

            maneuver_state["target_heading"] = new_target
            maneuver_state["phase"] = "turning"
            maneuver_state["timer"] = 0

        # 阶段5：恢复正常姿态
        elif maneuver_state["phase"] == "recovering":
            current_heading = maneuver_state["current_heading"]

            # 渐进式恢复到直飞
            new_heading = self._smooth_heading_change(current_heading, 0.0, rate=0.06)
            maneuver_state["current_heading"] = new_heading

            if abs(new_heading) < np.radians(5):  # 5度容差
                maneuver_state["completed"] = True
                maneuver_state["phase"] = "init"  # 重置状态
                logging.debug(f"Crank恢复完成")

        # 计算输出
        heading_cmd = maneuver_state["current_heading"]
        velocity_cmd = 600 + maneuver_state.get("velocity_boost", 0)

        # 射击条件：保持在雷达锁定扇区内
        shoot_ok = (
                radar_lock and
                30000 < enemy_distance < 50000 and
                abs(heading_cmd) < np.radians(55) and  # 仍在锁定扇区
                not has_missile_threat
        )

        return {
            "heading_cmd": heading_cmd,
            "altitude_cmd": 0,
            "velocity_cmd": velocity_cmd,
            "maintain_lock": True,
            "shoot": shoot_ok,
            "maneuver_active": True,
            "crank_phase": maneuver_state["phase"],
            "threat_level": maneuver_state.get("threat_level", "UNKNOWN")
        }

    def _beam_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Beam机动逻辑 - 90度横向机动消耗导弹动能"""

        enemy_angle_off = state.get("enemy_angle_off", 0)
        missile_distance = state.get("missile_distance", np.inf)
        enemy_distance = state.get("enemy_distance", 50000)
        has_missile_threat = len(
            [m for m in state.get("missiles_incoming", []) if hasattr(m, 'is_alive') and m.is_alive]) > 0

        # 触发条件
        should_beam = (
                missile_distance < 30000 or
                has_missile_threat or
                (enemy_distance < 25000 and abs(enemy_angle_off) < 45)  # 近距离正面威胁
        )

        if not should_beam:
            return {"heading_cmd": 0, "altitude_cmd": 0, "velocity_cmd": 600, "shoot": False, "maneuver_active": False}

        # 计算90度横向机动
        target_angle = 90.0  # 目标90度
        current_angle = abs(enemy_angle_off)

        if current_angle < 85:  # 还没到90度
            angle_deficit = target_angle - current_angle
            if enemy_angle_off >= 0:
                heading_cmd = np.radians(min(angle_deficit, 60))  # 向左转，最大60度
            else:
                heading_cmd = np.radians(-min(angle_deficit, 60))  # 向右转
        else:
            heading_cmd = 0  # 已接近垂直，保持

        # 威胁等级调整
        if missile_distance < 15000:
            velocity_cmd = 700  # 最高速度规避
            altitude_cmd = 150  # 轻微爬升
            beam_intensity = "MAXIMUM"
        elif missile_distance < 25000:
            velocity_cmd = 650  # 高速规避
            altitude_cmd = 100
            beam_intensity = "HIGH"
        else:
            velocity_cmd = 620
            altitude_cmd = 0
            beam_intensity = "STANDARD"

        logging.debug(
            f"Beam执行: 目标角度=90°, 当前={enemy_angle_off:.1f}°, 转向={np.degrees(heading_cmd):.1f}°, 强度={beam_intensity}")

        return {
            "heading_cmd": heading_cmd,
            "altitude_cmd": altitude_cmd,
            "velocity_cmd": velocity_cmd,
            "doppler_minimize": True,
            "shoot": False,  # Beam时不射击
            "maneuver_active": True,
            "beam_angle": abs(enemy_angle_off),
            "beam_intensity": beam_intensity
        }
    def _notch_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Notch机动逻辑 - 地面杂波遮蔽"""

        current_alt = state.get("current_altitude", 5000)
        missile_distance = state.get("missile_distance", np.inf)
        enemy_distance = state.get("enemy_distance", 50000)
        has_missile_threat = len(
            [m for m in state.get("missiles_incoming", []) if hasattr(m, 'is_alive') and m.is_alive]) > 0

        # 触发条件
        should_notch = (
                missile_distance < 35000 or
                has_missile_threat or
                (enemy_distance < 20000)  # 近距离必须用Notch
        )

        if not should_notch:
            return {"heading_cmd": 0, "altitude_cmd": 0, "velocity_cmd": 600, "shoot": False, "maneuver_active": False}

        # 首先执行Beam机动
        beam_action = self._beam_logic(state)

        # 高度管理 - Notch的关键
        if missile_distance < 25000:
            # 紧急情况：快速下降到杂波区
            if current_alt > 2500:
                target_altitude = 1500  # 目标1500米
                altitude_cmd = max(-1000, target_altitude - current_alt)
                notch_phase = "emergency_descent"
            else:
                altitude_cmd = -300  # 已在低空，继续下降
                notch_phase = "in_clutter"
        elif missile_distance < 40000:
            # 预防性下降
            if current_alt > 3500:
                target_altitude = 2500
                altitude_cmd = max(-600, target_altitude - current_alt)
                notch_phase = "preventive_descent"
            else:
                altitude_cmd = -200
                notch_phase = "clutter_level"
        else:
            altitude_cmd = 0
            notch_phase = "normal"

        # 速度管理
        if current_alt < 2000:
            velocity_cmd = 580  # 低空减速，保持控制
        elif altitude_cmd < -500:
            velocity_cmd = 650  # 下降时可以加速
        else:
            velocity_cmd = 620

        # 计算杂波效果
        clutter_effectiveness = 0.0
        if current_alt < 1800:
            clutter_effectiveness = 0.85  # 85%杂波遮蔽
        elif current_alt < 2500:
            clutter_effectiveness = 0.65  # 65%效果
        elif current_alt < 3500:
            clutter_effectiveness = 0.35  # 35%效果

        logging.debug(
            f"Notch执行: 高度={current_alt:.0f}m, 下降={altitude_cmd:.0f}m, 杂波效果={clutter_effectiveness:.2f}, 阶段={notch_phase}")

        # 继承Beam的横向机动，添加高度控制
        notch_action = beam_action.copy()
        notch_action.update({
            "altitude_cmd": altitude_cmd,
            "velocity_cmd": velocity_cmd,
            "ground_clutter": True,
            "clutter_effectiveness": clutter_effectiveness,
            "notch_phase": notch_phase,
            "target_altitude": 1500 if missile_distance < 25000 else current_alt
        })

        return notch_action
    def _simple_f_pole_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Simple F-Pole maneuver logic"""
        agent_id = state.get("agent_id", "A0100")
        maneuver_state = self.get_maneuver_state("f_pole", agent_id)

        enemy_distance = state.get("enemy_distance", 50000)
        radar_lock = state.get("radar_lock", False)
        missile_distance = state.get("missile_distance", np.inf)
        enemy_angle_off = state.get("enemy_angle_off", 0)

        should_f_pole = enemy_distance > 25000 and radar_lock

        if not should_f_pole:
            return {"heading_cmd": 0, "altitude_cmd": 0, "velocity_cmd": 600, "shoot": False, "maneuver_active": False}

        maneuver_state["timer"] += 1
        if maneuver_state["phase"] == "init":
            maneuver_state["phase"] = "optimizing"

        if enemy_distance > 60000:
            heading_cmd = 0
            velocity_cmd = 660
            f_pole_strategy = "approach"
        elif enemy_distance > 40000:
            heading_cmd = np.radians(-enemy_angle_off * 0.3) if abs(enemy_angle_off) > 10 else 0
            velocity_cmd = 630
            f_pole_strategy = "maintain"
        else:
            heading_cmd = np.radians(abs(enemy_angle_off) * 0.2)
            velocity_cmd = 600
            f_pole_strategy = "extend"

        shoot_ok = radar_lock and 35000 < enemy_distance < 55000 and abs(enemy_angle_off) < 25 and missile_distance > 40000

        return {
            "heading_cmd": heading_cmd,
            "altitude_cmd": 0,
            "velocity_cmd": velocity_cmd,
            "maintain_lock": True,
            "shoot": shoot_ok,
            "maneuver_active": True,
            "f_pole_strategy": f_pole_strategy,
            "optimal_f_pole": True
        }

    def _advanced_f_pole_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced F-Pole maneuver logic"""
        base_action = self._simple_f_pole_logic(state)
        if not base_action["maneuver_active"]:
            return base_action

        enemy_distance = state.get("enemy_distance", 50000)
        has_warning = state.get("has_warning", False)
        missile_distance = state.get("missile_distance", np.inf)

        if has_warning or missile_distance < 35000:
            crank_action = self._crank_logic(state)
            if crank_action.get("maneuver_active", False):
                heading_cmd = crank_action["heading_cmd"] * 0.6
                velocity_cmd = max(base_action["velocity_cmd"], crank_action["velocity_cmd"])
                advanced_strategy = "threat_aware_f_pole"
            else:
                heading_cmd = base_action["heading_cmd"]
                velocity_cmd = base_action["velocity_cmd"]
                advanced_strategy = "standard_f_pole"
        else:
            heading_cmd = base_action["heading_cmd"]
            velocity_cmd = base_action["velocity_cmd"]
            advanced_strategy = "standard_f_pole"

        base_action.update({
            "heading_cmd": heading_cmd,
            "velocity_cmd": velocity_cmd,
            "advanced_strategy": advanced_strategy,
            "threat_adaptive": has_warning
        })
        return base_action

    def _skate_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Skate maneuver logic"""
        agent_id = state.get("agent_id", "A0100")
        maneuver_state = self.get_maneuver_state("skate", agent_id)

        enemy_distance = state.get("enemy_distance", 50000)
        radar_lock = state.get("radar_lock", False)
        missile_distance = state.get("missile_distance", np.inf)

        maneuver_state["timer"] += 1

        if maneuver_state["phase"] == "init" and enemy_distance > 45000 and radar_lock:
            maneuver_state["phase"] = "first_launch"
            maneuver_state["missiles_fired"] = 1
            return {
                "heading_cmd": 0,
                "altitude_cmd": 0,
                "velocity_cmd": 640,
                "shoot": True,
                "maneuver_active": True,
                "skate_phase": "first_launch"
            }

        elif maneuver_state["phase"] == "first_launch" and maneuver_state["timer"] > 80:
            maneuver_state["phase"] = "post_launch_crank"
            maneuver_state["timer"] = 0
            return {
                "heading_cmd": 0,
                "altitude_cmd": 0,
                "velocity_cmd": 620,
                "shoot": False,
                "maneuver_active": True,
                "skate_phase": "waiting"
            }

        elif maneuver_state["phase"] == "post_launch_crank":
            if enemy_distance > 20000 and maneuver_state["timer"] < 200:
                crank_action = self._crank_logic(state)
                crank_action["skate_phase"] = "crank"
                return crank_action
            maneuver_state["phase"] = "turn_cold"
            maneuver_state["timer"] = 0

        elif maneuver_state["phase"] == "turn_cold" and maneuver_state["timer"] < 150:
            return {
                "heading_cmd": np.pi,
                "altitude_cmd": 0,
                "velocity_cmd": 720,
                "shoot": False,
                "maneuver_active": True,
                "skate_phase": "turn_cold"
            }

        elif maneuver_state["phase"] == "turn_cold" and maneuver_state["timer"] >= 150:
            maneuver_state["phase"] = "re_engage_check"
            maneuver_state["timer"] = 0

        elif maneuver_state["phase"] == "re_engage_check":
            if missile_distance > 50000 and enemy_distance < 60000:
                maneuver_state["phase"] = "re_engage"
            elif maneuver_state["timer"] > 100:
                maneuver_state["completed"] = True
                maneuver_state["phase"] = "init"

        elif maneuver_state["phase"] == "re_engage":
            if radar_lock and 35000 < enemy_distance < 50000:
                maneuver_state["phase"] = "second_launch"
                maneuver_state["missiles_fired"] = 2
                return {
                    "heading_cmd": 0,
                    "altitude_cmd": 0,
                    "velocity_cmd": 640,
                    "shoot": True,
                    "maneuver_active": True,
                    "skate_phase": "second_launch"
                }
            return {
                "heading_cmd": 0,
                "altitude_cmd": 0,
                "velocity_cmd": 650,
                "shoot": False,
                "maneuver_active": True,
                "skate_phase": "re_engage"
            }

        elif maneuver_state["phase"] == "second_launch":
            maneuver_state["completed"] = True
            maneuver_state["phase"] = "init"
            return {
                "heading_cmd": np.pi,
                "altitude_cmd": 200,
                "velocity_cmd": 700,
                "shoot": False,
                "maneuver_active": True,
                "skate_phase": "final_escape"
            }

        return {
            "heading_cmd": 0,
            "altitude_cmd": 0,
            "velocity_cmd": 620,
            "shoot": False,
            "maneuver_active": True,
            "skate_phase": maneuver_state.get("phase", "approach")
        }

    def _short_skate_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Short Skate maneuver logic"""
        agent_id = state.get("agent_id", "A0100")
        maneuver_state = self.get_maneuver_state("short_skate", agent_id)

        enemy_distance = state.get("enemy_distance", 50000)
        radar_lock = state.get("radar_lock", False)
        missile_distance = state.get("missile_distance", np.inf)

        maneuver_state["timer"] += 1

        if maneuver_state["phase"] == "init" and enemy_distance > 40000 and radar_lock:
            maneuver_state["phase"] = "launch"
            maneuver_state["fired"] = True
            return {
                "heading_cmd": 0,
                "altitude_cmd": 0,
                "velocity_cmd": 640,
                "shoot": True,
                "maneuver_active": True,
                "short_skate_phase": "launch"
            }

        elif maneuver_state["phase"] == "launch" and maneuver_state["timer"] > 40:
            maneuver_state["phase"] = "escape"
            maneuver_state["timer"] = 0

        elif maneuver_state["phase"] == "escape":
            if missile_distance < 30000:
                escape_intensity = "EMERGENCY"
                heading_cmd = np.pi
                velocity_cmd = 750
            else:
                escape_intensity = "STANDARD"
                heading_cmd = np.pi * 0.9
                velocity_cmd = 700
            if maneuver_state["timer"] > 200:
                maneuver_state["completed"] = True
                maneuver_state["phase"] = "init"
            return {
                "heading_cmd": heading_cmd,
                "altitude_cmd": 0,
                "velocity_cmd": velocity_cmd,
                "shoot": False,
                "maneuver_active": True,
                "short_skate_phase": "escape",
                "escape_intensity": escape_intensity
            }

        return {
            "heading_cmd": 0,
            "altitude_cmd": 0,
            "velocity_cmd": 630,
            "shoot": False,
            "maneuver_active": True,
            "short_skate_phase": "approach"
        }

    def _banzai_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Banzai maneuver logic"""
        agent_id = state.get("agent_id", "A0100")
        maneuver_state = self.get_maneuver_state("banzai", agent_id)

        enemy_distance = state.get("enemy_distance", 50000)
        radar_lock = state.get("radar_lock", False)
        missile_distance = state.get("missile_distance", np.inf)

        maneuver_state["timer"] += 1

        if maneuver_state["phase"] == "init" and enemy_distance > 25000 and radar_lock:
            maneuver_state["phase"] = "launch"
            maneuver_state["fired"] = True
            return {
                "heading_cmd": 0,
                "altitude_cmd": 0,
                "velocity_cmd": 620,
                "shoot": True,
                "maneuver_active": True,
                "banzai_phase": "launch"
            }

        elif maneuver_state["phase"] == "launch" and maneuver_state["timer"] > 50:
            maneuver_state["phase"] = "post_launch_crank"
            maneuver_state["timer"] = 0

        elif maneuver_state["phase"] == "post_launch_crank":
            if enemy_distance > 20000:
                crank_action = self._crank_logic(state)
                crank_action["banzai_phase"] = "post_launch_crank"
                return crank_action
            maneuver_state["phase"] = "decision_point"

        elif maneuver_state["phase"] == "decision_point":
            enemy_alive = True  # Simplified assumption
            if enemy_alive:
                maneuver_state["phase"] = "merge_commit"
                return {
                    "heading_cmd": 0,
                    "altitude_cmd": 0,
                    "velocity_cmd": 680,
                    "shoot": False,
                    "maneuver_active": True,
                    "banzai_phase": "merge_commit"
                }
            maneuver_state["completed"] = True
            maneuver_state["phase"] = "init"

        elif maneuver_state["phase"] == "merge_commit":
            if enemy_distance < 10000:
                maneuver_state["completed"] = True
                maneuver_state["phase"] = "init"
            return {
                "heading_cmd": 0,
                "altitude_cmd": 0,
                "velocity_cmd": 700,
                "shoot": False,
                "maneuver_active": True,
                "banzai_phase": "merge"
            }

        return {
            "heading_cmd": 0,
            "altitude_cmd": 0,
            "velocity_cmd": 620,
            "shoot": False,
            "maneuver_active": True,
            "banzai_phase": "approach"
        }

    def _pincer_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Pincer maneuver logic"""
        is_leader = state.get("is_leader", False)
        enemy_distance = state.get("enemy_distance", 50000)
        radar_lock = state.get("radar_lock", False)

        should_pincer = 40000 < enemy_distance < 80000 and radar_lock

        if not should_pincer:
            return {"heading_cmd": 0, "altitude_cmd": 0, "velocity_cmd": 600, "shoot": False, "maneuver_active": False}

        target_separation_angle = np.radians(45)
        heading_cmd = np.radians(22.5) if is_leader else np.radians(-22.5)
        role = "primary_attacker" if is_leader else "secondary_attacker"

        if enemy_distance > 65000:
            velocity_cmd = 680
            pincer_phase = "approach"
        elif enemy_distance > 45000:
            velocity_cmd = 640
            pincer_phase = "execute"
        else:
            velocity_cmd = 600
            pincer_phase = "close_range"

        shoot_ok = radar_lock and 45000 < enemy_distance < 70000 and pincer_phase == "execute"

        return {
            "heading_cmd": heading_cmd,
            "altitude_cmd": 0,
            "velocity_cmd": velocity_cmd,
            "coordinate_attack": True,
            "shoot": shoot_ok,
            "maneuver_active": True,
            "pincer_role": role,
            "pincer_phase": pincer_phase,
            "separation_angle": target_separation_angle
        }

    def _defensive_split_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Defensive Split maneuver logic"""
        is_leader = state.get("is_leader", False)
        enemy_distance = state.get("enemy_distance", 50000)
        has_warning = state.get("has_warning", False)
        missile_distance = state.get("missile_distance", np.inf)

        should_split = has_warning or missile_distance < 40000 or enemy_distance < 30000

        if not should_split:
            return {"heading_cmd": 0, "altitude_cmd": 0, "velocity_cmd": 600, "shoot": False, "maneuver_active": False}

        if missile_distance < 20000:
            separation_angle = np.radians(60)
            velocity_cmd = 720
            split_intensity = "EMERGENCY"
        elif missile_distance < 35000:
            separation_angle = np.radians(45)
            velocity_cmd = 680
            split_intensity = "STANDARD"
        else:
            separation_angle = np.radians(30)
            velocity_cmd = 640
            split_intensity = "PREVENTIVE"

        heading_cmd = separation_angle if is_leader else -separation_angle
        altitude_cmd = 200 if is_leader else -200
        split_role = "primary_evader" if is_leader else "secondary_evader"

        return {
            "heading_cmd": heading_cmd,
            "altitude_cmd": altitude_cmd,
            "velocity_cmd": velocity_cmd,
            "separation_maneuver": True,
            "shoot": False,
            "maneuver_active": True,
            "split_role": split_role,
            "split_intensity": split_intensity,
            "coordination_required": True
        }

    def _high_low_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """High-Low maneuver logic"""
        is_leader = state.get("is_leader", False)
        current_alt = state.get("current_altitude", 5000)
        enemy_distance = state.get("enemy_distance", 50000)
        radar_lock = state.get("radar_lock", False)

        should_high_low = enemy_distance > 35000 and radar_lock

        if not should_high_low:
            return {"heading_cmd": 0, "altitude_cmd": 0, "velocity_cmd": 600, "shoot": False, "maneuver_active": False}

        target_altitude = 9000 if is_leader else 3000
        role = "high_fighter" if is_leader else "low_fighter"
        tactical_advantage = "altitude_energy" if is_leader else "stealth_approach"

        altitude_diff = target_altitude - current_alt
        if abs(altitude_diff) > 500:
            altitude_cmd = np.clip(altitude_diff, -800, 800)
            high_low_phase = "positioning"
        else:
            altitude_cmd = 0
            high_low_phase = "maintained"

        if high_low_phase == "positioning":
            velocity_cmd = 620
            heading_cmd = 0
        else:
            heading_cmd = np.radians(10) if is_leader else np.radians(-10)
            velocity_cmd = 640 if is_leader else 660

        shoot_ok = radar_lock and (45000 < enemy_distance < 65000 if is_leader else 35000 < enemy_distance < 55000)

        return {
            "heading_cmd": heading_cmd,
            "altitude_cmd": altitude_cmd,
            "velocity_cmd": velocity_cmd,
            "altitude_optimization": True,
            "shoot": shoot_ok,
            "maneuver_active": True,
            "high_low_role": role,
            "high_low_phase": high_low_phase,
            "tactical_advantage": tactical_advantage
        }

    def _engaging_trail_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Engaging Trail maneuver logic"""
        is_leader = state.get("is_leader", False)
        enemy_distance = state.get("enemy_distance", 50000)
        radar_lock = state.get("radar_lock", False)

        should_trail = enemy_distance > 25000 and radar_lock

        if not should_trail:
            return {"heading_cmd": 0, "altitude_cmd": 0, "velocity_cmd": 600, "shoot": False, "maneuver_active": False}

        if is_leader:
            heading_cmd = 0
            if enemy_distance > 40000:
                velocity_cmd = 660
                trail_phase = "approach"
                shoot_ok = False
            elif enemy_distance > 30000:
                velocity_cmd = 640
                trail_phase = "attack"
                shoot_ok = radar_lock
            else:
                velocity_cmd = 580
                trail_phase = "disengage"
                shoot_ok = False
            role = "lead_attacker"
        else:
            ideal_trail_distance = 8000
            current_trail_distance = enemy_distance - 8000
            if current_trail_distance < 6000:
                velocity_cmd = 580
                heading_cmd = np.radians(5)
            elif current_trail_distance > 10000:
                velocity_cmd = 680
                heading_cmd = 0
            else:
                velocity_cmd = 620
                heading_cmd = 0
            trail_phase = "wingman_attack" if enemy_distance < 35000 else "following"
            shoot_ok = radar_lock and enemy_distance < 35000
            role = "trail_attacker"

        return {
            "heading_cmd": heading_cmd,
            "altitude_cmd": 0,
            "velocity_cmd": velocity_cmd,
            "shoot": shoot_ok,
            "maneuver_active": True,
            "trail_role": role,
            "trail_phase": trail_phase,
            "trail_distance": 8000 if not is_leader else None
        }

    def _loose_deuce_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Loose Deuce maneuver logic"""
        agent_id = state.get("agent_id", "A0100")
        maneuver_state = self.get_maneuver_state("loose_deuce", agent_id)
        is_leader = state.get("is_leader", False)
        enemy_distance = state.get("enemy_distance", 50000)
        radar_lock = state.get("radar_lock", False)
        current_alt = state.get("current_altitude", 5000)

        should_loose_deuce = enemy_distance > 20000

        if not should_loose_deuce:
            return {"heading_cmd": 0, "altitude_cmd": 0, "velocity_cmd": 600, "shoot": False, "maneuver_active": False}

        if "angle" not in maneuver_state:
            maneuver_state["angle"] = 0.0

        maneuver_state["timer"] += 1
        maneuver_state["angle"] += 0.02
        if maneuver_state["angle"] > 2 * np.pi:
            maneuver_state["angle"] = 0.0

        target_altitude = 7000 if is_leader else 4000
        role = "high_orbit" if is_leader else "low_orbit"

        altitude_diff = target_altitude - current_alt
        altitude_cmd = np.clip(altitude_diff, -400, 400)

        orbit_angle = maneuver_state["angle"]
        heading_cmd = np.sin(orbit_angle) * 0.3 if is_leader else np.sin(orbit_angle + np.pi) * 0.3

        if enemy_distance > 50000:
            velocity_cmd = 640
        elif enemy_distance > 30000:
            velocity_cmd = 620
        else:
            velocity_cmd = 660

        shoot_ok = radar_lock and 30000 < enemy_distance < 60000 and abs(heading_cmd) < np.radians(20)

        return {
            "heading_cmd": heading_cmd,
            "altitude_cmd": altitude_cmd,
            "velocity_cmd": velocity_cmd,
            "orbit_pattern": True,
            "shoot": shoot_ok,
            "maneuver_active": True,
            "orbit_role": role,
            "orbit_angle": orbit_angle,
            "mutual_support": True
        }

    def _defensive_sequence_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Defensive Sequence maneuver logic"""
        agent_id = state.get("agent_id", "A0100")
        maneuver_state = self.get_maneuver_state("defense_seq", agent_id)

        missile_distance = state.get("missile_distance", np.inf)
        has_warning = state.get("has_warning", False)

        should_defend = has_warning or missile_distance < 45000

        if not should_defend:
            return {"heading_cmd": 0, "altitude_cmd": 0, "velocity_cmd": 600, "shoot": False, "maneuver_active": False}

        maneuver_state["timer"] += 1

        if maneuver_state["phase"] == "init":
            maneuver_state["phase"] = "beam"
            maneuver_state["timer"] = 0

        if maneuver_state["phase"] == "beam":
            if maneuver_state["timer"] > 120 or missile_distance < 30000:
                maneuver_state["phase"] = "notch"
                maneuver_state["timer"] = 0
            beam_action = self._beam_logic(state)
            beam_action.update({"defense_stage": 1, "defense_phase": "beam"})
            return beam_action

        elif maneuver_state["phase"] == "notch":
            if maneuver_state["timer"] > 180 or missile_distance < 25000:
                maneuver_state["phase"] = "f_pole"
                maneuver_state["timer"] = 0
            notch_action = self._notch_logic(state)
            notch_action.update({"defense_stage": 2, "defense_phase": "notch"})
            return notch_action

        elif maneuver_state["phase"] == "f_pole":
            if maneuver_state["timer"] > 100 or missile_distance < 20000:
                maneuver_state["phase"] = "crank"
                maneuver_state["timer"] = 0
            return {
                "heading_cmd": np.radians(180),
                "altitude_cmd": 100,
                "velocity_cmd": 700,
                "shoot": False,
                "maneuver_active": True,
                "defense_stage": 3,
                "defense_phase": "f_pole_extend"
            }

        elif maneuver_state["phase"] == "crank":
            if maneuver_state["timer"] > 150 or missile_distance < 15000:
                maneuver_state["phase"] = "turn_cold"
                maneuver_state["timer"] = 0
            crank_action = self._crank_logic(state)
            crank_action.update({"defense_stage": 4, "defense_phase": "crank"})
            return crank_action

        elif maneuver_state["phase"] == "turn_cold":
            if maneuver_state["timer"] > 200 or missile_distance > 40000:
                maneuver_state["phase"] = "split"
                maneuver_state["timer"] = 0
            return {
                "heading_cmd": np.pi,
                "altitude_cmd": 0,
                "velocity_cmd": 750,
                "shoot": False,
                "maneuver_active": True,
                "defense_stage": 5,
                "defense_phase": "turn_cold"
            }

        elif maneuver_state["phase"] == "split":
            if missile_distance > 50000:
                maneuver_state["completed"] = True
                maneuver_state["phase"] = "init"
                return {
                    "heading_cmd": 0,
                    "altitude_cmd": 200,
                    "velocity_cmd": 650,
                    "shoot": state.get("radar_lock", False),
                    "maneuver_active": True,
                    "defense_stage": 6,
                    "defense_phase": "counter_attack"
                }
            if maneuver_state["timer"] > 200:
                maneuver_state["completed"] = True
                maneuver_state["phase"] = "init"
            return {
                "heading_cmd": np.radians(135),
                "altitude_cmd": -200,
                "velocity_cmd": 720,
                "shoot": False,
                "maneuver_active": True,
                "defense_stage": 6,
                "defense_phase": "drag_split"
            }

        return {
            "heading_cmd": 0,
            "altitude_cmd": 0,
            "velocity_cmd": 600,
            "shoot": False,
            "maneuver_active": False
        }