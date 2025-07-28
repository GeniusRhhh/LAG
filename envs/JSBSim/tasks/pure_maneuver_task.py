import logging
import numpy as np
import torch
from typing import Dict, Any
from .multiplecombat_task import MultipleCombatTask
from .pure_maneuvers import PureManeuvers, BasicManeuvers, CompositeManeuverExecutor
from ..core.catalog import Catalog as c
from ..model.baseline_actor import BaselineActor
from ..utils.utils import get_root_dir


class PureManeuverTask(MultipleCombatTask):
    """纯机动测试任务 - 支持基础机动和组合机动"""

    def __init__(self, config):
        super().__init__(config)
        self.maneuver_type = "crank"
        self.maneuver_params = {
            "crank_angle_deg": 60.0,
            "turn_rate_deg_per_sec": 3.0,
            "hold_time_sec": 20.0
        }
        self.basic_maneuver_params = {
            "turn_angle": 45.0,
            "turn_rate": 3.0,
            "altitude_change": 1000.0,
            "velocity_change": 50.0,
            "duration": 10.0
        }
        self.composite_maneuver_name = "turn_pull_up"
        self.composite_maneuver_params = {}
        self.test_agent_id = "A0100"
        self.observer_agent_id = "B0100"
        self.test_start_time = 0.0
        self.initial_heading = {}
        self.initial_altitude = {}
        self.trajectory_data = {}
        self.my_lowlevel_policy = BaselineActor()
        self.enemy_baseline_policy = BaselineActor()
        self._inner_rnn_states = {}
        self._enemy_rnn_states = {}
        # 扩充控制映射数组，与索引转换函数保持一致
        self.norm_delta_altitude = np.array([
            -1500, -1000, -750, -500, -300, -150, -50, 0, 50, 150, 300, 500, 750, 1000, 1500
        ]) / 1000.0
        self.norm_delta_heading = np.array([
            -np.pi,           # -180°
            -2*np.pi/3,       # -120°
            -np.pi/2,         # -90°
            -5*np.pi/12,      # -75°
            -np.pi/3,         # -60°
            -np.pi/4,         # -45°
            -np.pi/6,         # -30°
            -np.pi/12,        # -15°
            0,                # 0°
            np.pi/12,         # 15°
            np.pi/6,          # 30°
            np.pi/4,          # 45°
            np.pi/3,          # 60°
            5*np.pi/12,       # 75°
            np.pi/2,          # 90°
            2*np.pi/3,        # 120°
            np.pi             # 180°
        ])
        self.norm_delta_velocity = np.array([-150, -100, -50, 0, 50, 100, 150]) / 100.0
        self.composite_executor = CompositeManeuverExecutor()
        self.maneuver_composer = self.composite_executor
        self._load_baseline_models()
        logging.info("PureManeuverTask初始化完成 - 支持基础机动和组合机动")

    def _load_baseline_models(self):
        """加载baseline模型"""
        try:
            model_path = get_root_dir() + '/model/baseline_model.pt'
            if torch.cuda.is_available():
                device = torch.device("cuda")
                checkpoint = torch.load(model_path)
            else:
                device = torch.device("cpu")
                checkpoint = torch.load(model_path, map_location=device)
            self.my_lowlevel_policy.load_state_dict(checkpoint)
            self.my_lowlevel_policy.eval()
            self.enemy_baseline_policy.load_state_dict(checkpoint)
            self.enemy_baseline_policy.eval()
        except Exception as e:
            logging.error(f"加载baseline模型失败: {e}")
            self.my_lowlevel_policy = None
            self.enemy_baseline_policy = None

    def reset(self, env):
        """重置任务状态"""
        from .task_base import BaseTask
        BaseTask.reset(self, env)
        self.step_count = 0
        self.test_start_time = 0.0
        self.initial_heading.clear()
        self.initial_altitude.clear()
        self.trajectory_data.clear()
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        self._enemy_rnn_states = {agent_id: torch.zeros(1, 1, 128) for agent_id in env.agents.keys()}
        if self.composite_maneuver_params:
            self.composite_executor.update_maneuver_params(self.composite_maneuver_name, self.composite_maneuver_params)
        if hasattr(self.composite_executor, 'active_states'):
            self.composite_executor.active_states.clear()
        logging.info(f"PureManeuverTask重置完成")
        logging.info(f"机动类型: {self.maneuver_type}")
        if self.maneuver_type == "basic":
            basic_name = self.maneuver_params.get("basic_maneuver_name", "未设置")
            logging.info(f"基础机动: {basic_name}")
            logging.info(f"机动参数: {self.basic_maneuver_params}")
        elif self.maneuver_type == "composite":
            logging.info(f"组合机动: {self.composite_maneuver_name}")
            logging.info(f"机动参数: {self.composite_maneuver_params}")
        else:
            logging.info(f"大机动参数: {self.maneuver_params}")

    def set_basic_maneuver(self, maneuver_name, **params):
        """设置基础机动"""
        self.maneuver_type = "basic"
        self.maneuver_params["basic_maneuver_name"] = maneuver_name
        self.basic_maneuver_params.clear()
        if maneuver_name == "level_flight":
            default_params = {"duration": 30.0}
        elif maneuver_name in ["accelerate", "decelerate"]:
            default_params = {"velocity_change": 50.0, "duration": 20.0}
        elif maneuver_name == "turn":
            default_params = {"turn_angle": 45.0, "turn_rate": 3.0}
        elif maneuver_name in ["pull_up", "dive"]:
            default_params = {"altitude_change": 1500.0, "duration": 15.0}
            if maneuver_name == "dive":
                default_params["min_altitude"] = 2000.0
        elif maneuver_name == "diagonal_flight":
            default_params = {"turn_angle": 45.0, "altitude_change": 1000.0, "duration": 15.0}
        else:
            default_params = {"duration": 20.0}
        self.basic_maneuver_params.update(default_params)
        self.basic_maneuver_params.update(params)
        logging.info(f"基础机动设置完成")
        logging.info(f"机动名称: {maneuver_name}")
        logging.info(f"用户参数: {params}")
        logging.info(f"最终参数: {self.basic_maneuver_params}")

    def set_composite_maneuver(self, maneuver_name="turn_pull_up", custom_params=None):
        """设置组合机动"""
        self.maneuver_type = "composite"
        self.composite_maneuver_name = maneuver_name
        self.composite_maneuver_params = custom_params or {}
        logging.info(f"组合机动设置完成")
        logging.info(f"机动名称: {maneuver_name}")
        logging.info(f"机动参数: {self.composite_maneuver_params}")
        if custom_params:
            self.composite_executor.update_maneuver_params(maneuver_name, custom_params)
        self.composite_executor.reset_maneuver_state(maneuver_name)
        info = self.composite_executor.get_maneuver_info(maneuver_name)
        if info:
            logging.info(f"机动描述: {info['description']}")
            for i, (step_name, step_params, step_duration) in enumerate(info['steps']):
                logging.info(f"步骤{i + 1}: {step_name} {step_params} 持续{step_duration}s")

    def normalize_action(self, env, agent_id, action):
        """动作归一化"""
        if agent_id not in env.agents or not env.agents[agent_id].is_alive:
            return np.array([0.0, 0.0, 0.0, 0.7])
        current_time = env.current_step * env.time_interval
        if agent_id not in self.initial_heading:
            current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_rad)
            self.initial_heading[agent_id] = np.rad2deg(current_heading)
            current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
            self.initial_altitude[agent_id] = current_altitude
            logging.info(f"{agent_id} 初始状态:")
            logging.info(f"航向: {self.initial_heading[agent_id]:.1f}°")
            logging.info(f"高度: {self.initial_altitude[agent_id]:.1f}m")
        if agent_id == self.test_agent_id:
            return self._process_test_maneuver_action(env, agent_id, current_time)
        else:
            return self._process_observer_behavior(env, agent_id)

    def _process_test_maneuver_action(self, env, agent_id, current_time):
        """处理测试飞机的机动动作"""
        try:
            initial_heading = self.initial_heading[agent_id]
            initial_altitude = self.initial_altitude[agent_id]
            if self.maneuver_type in ["crank", "beam", "notch"]:
                return self._process_large_maneuver(env, agent_id, current_time, initial_heading, initial_altitude)
            elif self.maneuver_type == "basic":
                return self._process_basic_maneuver(env, agent_id, current_time, initial_heading, initial_altitude)
            elif self.maneuver_type == "composite":
                return self._process_composite_maneuver(env, agent_id, current_time, initial_heading, initial_altitude)
            else:
                return self._use_lowlevel_policy(env, agent_id, 3, 4, 3)
        except Exception as e:
            logging.error(f"{agent_id} 机动执行错误: {e}", exc_info=True)
            return self._direct_control_mapping(env, agent_id, 3, 4, 3)

    def _process_basic_maneuver(self, env, agent_id, current_time, initial_heading, initial_altitude):
        """处理基础机动"""
        basic_maneuver_name = self.maneuver_params.get("basic_maneuver_name", "level_flight")
        current_velocity = env.agents[agent_id].get_property_value(c.velocities_u_mps)
        current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        result = self._call_basic_maneuver_function(basic_maneuver_name, current_time,
                                                    initial_heading, initial_altitude,
                                                    current_velocity, current_altitude)
        phase, target_heading, target_altitude, velocity_offset, target_roll = result
        if phase is None:
            return self._use_lowlevel_policy(env, agent_id, 3, 4, 3)
        # 动态日志频率控制
        log_interval = 25 if current_time <= 80.0 else 100  # 前80秒每25步，后面每100步
        if env.current_step % log_interval == 0:
            log_msg = f"{agent_id} 执行 {basic_maneuver_name} [{phase}] at t={current_time:.1f}s | "
            if target_heading is not None:
                log_msg += f"Hdg: {current_heading:.1f}° -> {target_heading:.1f}° | "
            if target_altitude is not None:
                log_msg += f"Alt: {current_altitude:.1f}m -> {target_altitude:.1f}m | "
            if velocity_offset is not None:
                log_msg += f"Vel: {current_velocity:.1f}m/s (Δ{velocity_offset:+.1f}) |"
            logging.info(log_msg)
        # 使用新数组的中间索引作为默认值
        altitude_cmd_id = 7  # 15个值的中间索引，对应0米变化
        heading_cmd_id = 8   # 17个值的中间索引，对应0度变化
        velocity_cmd_id = 3  # 7个值的中间索引，对应0m/s变化
        # 高度控制 - 增强稳定性，抑制JSBSim的自动升力补偿
        if basic_maneuver_name in ["turn", "turn_level", "accelerate", "decelerate", "level_flight"]:
            # 对于转弯等机动，强制保持初始高度，抑制高度上升
            altitude_diff = initial_altitude - current_altitude
            if abs(altitude_diff) > 5.0:  # 进一步降低阈值到5米，更敏感地控制高度
                # 如果高度上升，给予更强的下降指令
                if current_altitude > initial_altitude + 30.0:  # 高度上升超过30米就强制下降
                    altitude_diff = altitude_diff * 2.0  # 增强下降控制到200%
                elif current_altitude > initial_altitude + 15.0:  # 高度上升超过15米
                    altitude_diff = altitude_diff * 1.5  # 增强下降控制到150%
                altitude_cmd_id = self._convert_altitude_to_index(altitude_diff)
        elif target_altitude is not None:
            altitude_diff = target_altitude - current_altitude
            if abs(altitude_diff) > 5.0:  # 降低阈值到5米
                altitude_cmd_id = self._convert_altitude_to_index(altitude_diff)

        # 角度控制 - 使用扩充的离散控制
        if target_heading is not None:
            heading_diff = target_heading - current_heading
            while heading_diff > 180: heading_diff -= 360
            while heading_diff < -180: heading_diff += 360

            # 强制角度控制：如果角度差异超过1度就进行控制
            if abs(heading_diff) > 1.0:
                heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))

        if velocity_offset is not None and basic_maneuver_name in ["accelerate", "decelerate"]:
            if abs(velocity_offset) > 2.0:
                velocity_cmd_id = self._convert_velocity_to_index(velocity_offset)
        if target_roll is not None and abs(target_roll) > 1.0:
            return self._use_lowlevel_policy_with_roll(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id,
                                                       target_roll)
        else:
            return self._use_lowlevel_policy(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)


    def _process_composite_maneuver(self, env, agent_id, current_time, initial_heading, initial_altitude):
        """处理组合机动 - 完全修复版本"""
        current_velocity = env.agents[agent_id].get_property_value(c.velocities_u_mps)
        current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))

        result = self.composite_executor.execute_composite_maneuver(
            self.composite_maneuver_name, current_time, initial_heading, initial_altitude,
            current_velocity, current_altitude
        )
        phase, target_heading, target_altitude, velocity_offset, target_roll = result

        if phase is None:
            # 机动完成后保持平稳飞行，使用新数组的中间索引
            return self._use_lowlevel_policy(env, agent_id, 7, 8, 3)

        # 动态日志频率控制
        log_interval = 25 if current_time <= 80.0 else 100  # 前80秒每25步，后面每100步
        if env.current_step % log_interval == 0:
            log_msg = f"{agent_id} 组合机动 {self.composite_maneuver_name} [{phase}] at t={current_time:.1f}s | "
            if target_heading is not None:
                log_msg += f"Hdg: {current_heading:.1f}° -> {target_heading:.1f}° | "
            if target_altitude is not None:
                log_msg += f"Alt: {current_altitude:.1f}m -> {target_altitude:.1f}m | "
            if velocity_offset is not None:
                log_msg += f"Vel: Δ{velocity_offset:+.1f}m/s |"
            logging.info(log_msg)

        # 使用新数组的中间索引作为默认值
        altitude_cmd_id, heading_cmd_id, velocity_cmd_id = 7, 8, 3

        # 高度控制逻辑 - 增强稳定性控制
        if phase in ["TURNING", "TURNING_LEVEL", "TURN_LEVEL_ADJUSTING"]:
            # 转弯状态：强制保持目标高度或初始高度，更严格的控制
            if target_altitude is not None:
                altitude_diff = target_altitude - current_altitude
            else:
                altitude_diff = initial_altitude - current_altitude

            # 更严格的高度控制，降低阈值并增强控制强度
            if abs(altitude_diff) > 5.0:  # 进一步降低阈值到5米
                # 对于大的高度偏差，增强控制强度
                if abs(altitude_diff) > 100.0:
                    altitude_diff = altitude_diff * 1.5  # 增强50%控制强度
                altitude_cmd_id = self._convert_altitude_to_index(altitude_diff)

        elif phase in ["DIVING", "DIVE_FINISHED"]:
            # 俯冲状态：按照目标高度调整
            if target_altitude is not None:
                altitude_diff = target_altitude - current_altitude
                if abs(altitude_diff) > 5.0:  # 降低阈值到5米
                    altitude_cmd_id = self._convert_altitude_to_index(altitude_diff)

        elif phase in ["MANEUVER_COMPLETED"]:
            # 机动完成状态：保持目标高度和航向，确保平稳飞行
            if target_altitude is not None:
                altitude_diff = target_altitude - current_altitude
                if abs(altitude_diff) > 5.0:  # 保持精确高度控制
                    altitude_cmd_id = self._convert_altitude_to_index(altitude_diff)

        elif target_altitude is not None:
            # 其他状态：按照目标高度调整
            altitude_diff = target_altitude - current_altitude
            if abs(altitude_diff) > 5.0:  # 降低阈值到5米
                altitude_cmd_id = self._convert_altitude_to_index(altitude_diff)

        if target_heading is not None:
            heading_diff = target_heading - current_heading
            while heading_diff > 180: heading_diff -= 360
            while heading_diff < -180: heading_diff += 360

            # 根据阶段调整精度
            if phase in ["TURN_ADJUSTING", "TURN_LEVEL_ADJUSTING"]:
                threshold = 1.0  # 转弯调整阶段：1度精度
            elif phase in ["MAINTAINING_HEADING", "MANEUVER_COMPLETED"]:
                threshold = 1.5  # 保持航向和机动完成：1.5度精度
            else:
                threshold = 1.0  # 其他阶段：1度精度

            if abs(heading_diff) > threshold:
                heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))

        # 速度控制
        if velocity_offset is not None and abs(velocity_offset) > 2.0:
            velocity_cmd_id = self._convert_velocity_to_index(velocity_offset)

        # 滚转控制
        if target_roll is not None and abs(target_roll) > 1.0:
            return self._use_lowlevel_policy_with_roll(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id,
                                                       target_roll)
        else:
            return self._use_lowlevel_policy(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)

    def _process_large_maneuver(self, env, agent_id, current_time, initial_heading, initial_altitude):
        """处理大机动"""
        if self.maneuver_type == "crank":
            maneuver_result = PureManeuvers.crank_maneuver(
                current_time,
                initial_heading,
                self.maneuver_params["crank_angle_deg"],
                self.maneuver_params["turn_rate_deg_per_sec"],
                self.maneuver_params["hold_time_sec"]
            )
        elif self.maneuver_type == "beam":
            maneuver_result = PureManeuvers.beam_maneuver(
                current_time,
                initial_heading,
                self.maneuver_params.get("beam_angle_deg", 90.0),
                self.maneuver_params.get("turn_rate_deg_per_sec", 5.0),
                self.maneuver_params.get("hold_time_sec", 15.0)
            )
        elif self.maneuver_type == "notch":
            maneuver_result = PureManeuvers.notch_maneuver(
                current_time,
                initial_heading,
                initial_altitude,
                self.maneuver_params.get("notch_angle_deg", 90.0),
                self.maneuver_params.get("turn_rate_deg_per_sec", 4.0),
                self.maneuver_params.get("descent_rate_ft_per_sec", 60.0),
                self.maneuver_params.get("descent_time_sec", 5.0),
                self.maneuver_params.get("hold_time_sec", 15.0)
            )
        else:
            maneuver_result = None
        if maneuver_result is None:
            return self._use_lowlevel_policy(env, agent_id, 7, 8, 3)  # 使用新数组的中间索引
        if len(maneuver_result) == 3:
            phase, target_heading, target_roll = maneuver_result
            target_altitude = None
        else:
            phase, target_heading, target_altitude, target_roll = maneuver_result
        if env.current_step % 100 == 0:
            logging.info(f"{agent_id} {self.maneuver_type.upper()} {phase}: 目标={target_heading:.1f}°")
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        heading_diff = target_heading - current_heading
        while heading_diff > 180:
            heading_diff -= 360
        while heading_diff < -180:
            heading_diff += 360
        if abs(heading_diff) < 2.0:
            heading_cmd_id = 8  # 使用新数组的中间索引
        else:
            heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))
        if target_altitude is not None:
            current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
            altitude_diff = target_altitude - current_altitude
            altitude_cmd_id = self._convert_altitude_to_index(altitude_diff)
        else:
            altitude_cmd_id = 7  # 使用新数组的中间索引
        velocity_cmd_id = 3
        return self._use_lowlevel_policy(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)

    def _call_basic_maneuver_function(self, basic_maneuver_name, current_time, initial_heading, initial_altitude,
                                      current_velocity, current_altitude):
        """调用基础机动函数"""
        params = self.basic_maneuver_params
        if current_time < 1.0:
            logging.info(f"调用基础机动函数 {basic_maneuver_name}")
            logging.info(f"使用参数: {params}")
        if basic_maneuver_name == "level_flight":
            return BasicManeuvers.level_flight(
                current_time,
                params.get("duration", 20.0),
                current_altitude=current_altitude,
                current_velocity=current_velocity
            )
        elif basic_maneuver_name == "accelerate":
            return BasicManeuvers.accelerate(
                current_time,
                current_velocity,
                params.get("duration", 20.0),
                params.get("velocity_change", 50.0)
            )
        elif basic_maneuver_name == "decelerate":
            return BasicManeuvers.decelerate(
                current_time,
                current_velocity,
                params.get("duration", 20.0),
                params.get("velocity_change", 50.0)
            )
        elif basic_maneuver_name == "turn":
            turn_angle = params.get("turn_angle", 45.0)
            turn_rate = params.get("turn_rate", 3.0)
            return BasicManeuvers.turn(
                current_time,
                initial_heading,
                turn_angle,
                turn_rate
            )
        elif basic_maneuver_name == "turn_level":
            turn_angle = params.get("turn_angle", 45.0)
            turn_rate = params.get("turn_rate", 3.0)
            return BasicManeuvers.turn_level(
                current_time,
                initial_heading,
                initial_altitude,
                turn_angle,
                turn_rate
            )
        elif basic_maneuver_name == "pull_up":
            return BasicManeuvers.pull_up(
                current_time,
                initial_altitude,
                params.get("duration", 15.0),
                params.get("altitude_change", 1500.0)
            )
        elif basic_maneuver_name == "dive":
            return BasicManeuvers.dive(
                current_time,
                initial_altitude,
                params.get("duration", 15.0),
                params.get("altitude_change", 1500.0),
                params.get("min_altitude", 2000.0)
            )
        elif basic_maneuver_name == "diagonal_flight":
            return BasicManeuvers.diagonal_flight(
                current_time,
                initial_heading,
                initial_altitude,
                params.get("duration", 15.0),
                params.get("turn_angle", 45.0),
                params.get("altitude_change", 1000.0),
                params.get("min_altitude", 3000.0)
            )
        elif basic_maneuver_name == "maintain_heading_flight":
            return BasicManeuvers.maintain_heading_flight(
                current_time,
                params.get("target_heading", initial_heading),
                params.get("duration", 15.0)
            )

        else:
            return (None, None, None, None, None)

    def _process_observer_behavior(self, env, agent_id):
        """处理观察飞机的行为 - 修复索引以保持平稳飞行"""
        try:
            # 使用新数组的中间索引来保持平稳飞行
            # 高度：15个值的中间是索引7 (对应0米变化)
            # 航向：17个值的中间是索引8 (对应0度变化)
            # 速度：7个值的中间是索引3 (对应0m/s变化)
            altitude_cmd_id = 7  # 对应0米高度变化
            heading_cmd_id = 8   # 对应0度航向变化
            velocity_cmd_id = 3  # 对应0m/s速度变化
            return self._use_lowlevel_policy(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)
        except Exception as e:
            logging.error(f"{agent_id} 观察行为错误: {e}")
            return np.array([0.0, 0.0, 0.0, 0.7])

    def step(self, env):
        """执行一步纯机动仿真"""
        self.step_count += 1
        current_time = env.current_step * env.time_interval
        obs = {}
        share_obs = {}
        rewards = {}
        dones = {}
        infos = {}
        for agent_id in env.agents.keys():
            if not env.agents[agent_id].is_alive:
                obs[agent_id] = np.zeros(self.obs_length)
                share_obs[agent_id] = np.zeros(self.obs_length)
                rewards[agent_id] = [-10.0]
                dones[agent_id] = [True]
                infos[agent_id] = {"agent_id": agent_id, "alive": False}
                continue
            agent_obs = self.get_obs(env, agent_id)
            obs[agent_id] = agent_obs
            share_obs[agent_id] = agent_obs
            reward = self._calculate_maneuver_reward(env, agent_id, current_time)
            rewards[agent_id] = [reward]
            done, info = self.get_termination(env, agent_id, {})
            dones[agent_id] = [done]
            agent_info = {
                "agent_id": agent_id,
                "time": current_time,
                "step": self.step_count,
                "maneuver_type": self.maneuver_type,
                "altitude": env.agents[agent_id].get_property_value(c.position_h_sl_m),
                "heading": np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad)),
                "velocity": env.agents[agent_id].get_property_value(c.velocities_u_mps),
                "reward": reward,
                "alive": True
            }
            infos[agent_id] = agent_info
            self._record_trajectory_data(env, agent_id, current_time)

            # 动态日志记录频率：前80秒详细记录，后面间隔久一点
            if current_time <= 80.0:
                # 前80秒：每20步（约3.3秒）记录一次
                if env.current_step % 20 == 0:
                    logging.info(f"[详细] 步骤 {env.current_step} (t={current_time:.1f}s) - {agent_id}: "
                                 f"高度={agent_info['altitude']:.1f}m, "
                                 f"航向={agent_info['heading']:.1f}°, "
                                 f"速度={agent_info['velocity']:.1f}m/s, "
                                 f"奖励={reward:.3f}")
            else:
                # 80秒后：每200步（约33秒）记录一次
                if env.current_step % 200 == 0:
                    logging.info(f"[概要] 步骤 {env.current_step} (t={current_time:.1f}s) - {agent_id}: "
                                 f"高度={agent_info['altitude']:.1f}m, "
                                 f"航向={agent_info['heading']:.1f}°, "
                                 f"速度={agent_info['velocity']:.1f}m/s, "
                                 f"奖励={reward:.3f}")
        return obs, share_obs, rewards, dones, infos

    def _calculate_maneuver_reward(self, env, agent_id, current_time) -> float:
        """计算机动奖励"""
        reward = 1.0
        altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        if altitude < 3000:
            reward -= 2.0
        elif altitude > 4000:
            reward += 0.1
        if agent_id == self.test_agent_id:
            reward += self._evaluate_maneuver_performance(env, agent_id, current_time)
        return np.clip(reward, -10, 10)

    def _evaluate_maneuver_performance(self, env, agent_id, current_time) -> float:
        """评估机动执行效果"""
        try:
            if self.maneuver_type in ["crank", "beam", "notch"]:
                return self._evaluate_large_maneuver_performance(env, agent_id, current_time)
            elif self.maneuver_type == "basic":
                return self._evaluate_basic_maneuver_performance(env, agent_id, current_time)
            elif self.maneuver_type == "composite":
                return self._evaluate_composite_maneuver_performance(env, agent_id, current_time)
            return 0.0
        except Exception as e:
            logging.error(f"机动性能评估错误: {e}")
            return 0.0

    def _evaluate_basic_maneuver_performance(self, env, agent_id, current_time) -> float:
        """评估基础机动性能"""
        try:
            basic_maneuver_name = self.maneuver_params.get("basic_maneuver_name", "level_flight")
            current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
            current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
            current_velocity = env.agents[agent_id].get_property_value(c.velocities_u_mps)
            if basic_maneuver_name in ["turn"]:
                initial_heading = self.initial_heading[agent_id]
                expected_heading = initial_heading + self.basic_maneuver_params.get("turn_angle", 90.0)
                heading_error = abs(current_heading - expected_heading)
                if heading_error > 180:
                    heading_error = 360 - heading_error
                if heading_error < 10:
                    return 0.3
                elif heading_error < 30:
                    return 0.1
                else:
                    return -0.05
            elif basic_maneuver_name in ["pull_up", "dive"]:
                initial_altitude = self.initial_altitude[agent_id]
                if basic_maneuver_name == "pull_up":
                    expected_altitude = initial_altitude + self.basic_maneuver_params.get("altitude_change", 1500.0)
                else:
                    expected_altitude = initial_altitude - self.basic_maneuver_params.get("altitude_change", 1500.0)
                altitude_error = abs(current_altitude - expected_altitude)
                if altitude_error < 200:
                    return 0.3
                elif altitude_error < 500:
                    return 0.1
                else:
                    return -0.05
            return 0.0
        except Exception as e:
            logging.error(f"基础机动性能评估错误: {e}")
            return 0.0

    def _evaluate_large_maneuver_performance(self, env, agent_id, current_time) -> float:
        """评估大机动性能"""
        try:
            if self.maneuver_type == "crank":
                initial_heading = self.initial_heading[agent_id]
                maneuver_result = PureManeuvers.crank_maneuver(
                    current_time,
                    initial_heading,
                    self.maneuver_params["crank_angle_deg"],
                    self.maneuver_params["turn_rate_deg_per_sec"],
                    self.maneuver_params["hold_time_sec"]
                )
                if maneuver_result:
                    phase, target_heading, target_roll = maneuver_result
                    current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
                    heading_error = abs(current_heading - target_heading)
                    if heading_error > 180:
                        heading_error = 360 - heading_error
                    if heading_error < 5:
                        return 0.5
                    elif heading_error < 15:
                        return 0.2
                    else:
                        return -0.1
            return 0.0
        except Exception as e:
            logging.error(f"大机动性能评估错误: {e}")
            return 0.0

    def _evaluate_composite_maneuver_performance(self, env, agent_id, current_time) -> float:
        """评估组合机动性能"""
        try:
            current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
            current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
            current_velocity = env.agents[agent_id].get_property_value(c.velocities_u_mps)
            reward = 0.0
            if 3000 <= current_altitude <= 8000:
                reward += 0.1
            if 200 <= current_velocity <= 350:
                reward += 0.1
            if self.composite_maneuver_name in ["turn_pull_up", "turn_dive"]:
                initial_heading = self.initial_heading[agent_id]
                initial_altitude = self.initial_altitude[agent_id]
                heading_change = abs(current_heading - initial_heading)
                altitude_change = abs(current_altitude - initial_altitude)
                if heading_change > 20 and altitude_change > 200:
                    reward += 0.2
            return reward
        except Exception as e:
            logging.error(f"组合机动性能评估错误: {e}")
            return 0.0

    def _record_trajectory_data(self, env, agent_id, current_time):
        """记录轨迹数据"""
        if agent_id not in self.trajectory_data:
            self.trajectory_data[agent_id] = []
        agent = env.agents[agent_id]
        state = {
            "time": current_time,
            "altitude": agent.get_property_value(c.position_h_sl_m),
            "heading": np.rad2deg(agent.get_property_value(c.attitude_psi_rad)),
            "roll": np.rad2deg(agent.get_property_value(c.attitude_phi_rad)),
            "pitch": np.rad2deg(agent.get_property_value(c.attitude_theta_rad)),
            "velocity": agent.get_property_value(c.velocities_u_mps),
            "longitude": agent.get_property_value(c.position_long_gc_deg),
            "latitude": agent.get_property_value(c.position_lat_geod_deg)
        }
        self.trajectory_data[agent_id].append(state)

    def _convert_altitude_to_index(self, altitude_cmd):
        """高度指令转索引 - 扩充精度版本"""
        # 扩充高度控制精度
        # 原来：[-1000, -500, -200, 0, 200, 500, 1000]
        # 现在：[-1500, -1000, -750, -500, -300, -150, -50, 0, 50, 150, 300, 500, 750, 1000, 1500]
        altitude_values = np.array([
            -1500, -1000, -750, -500, -300, -150, -50, 0, 50, 150, 300, 500, 750, 1000, 1500
        ])
        distances = np.abs(altitude_values - altitude_cmd)
        return np.argmin(distances)

    def _convert_heading_to_index(self, heading_cmd):
        """航向指令转索引 - 扩充精度版本"""
        heading_cmd = np.clip(heading_cmd, -np.pi, np.pi)
        # 扩充到更多离散值，提高精度
        # 原来：[-180°, -90°, -60°, -30°, 0°, 30°, 60°, 90°, 180°]
        # 现在：[-180°, -120°, -90°, -75°, -60°, -45°, -30°, -15°, 0°, 15°, 30°, 45°, 60°, 75°, 90°, 120°, 180°]
        heading_values = np.array([
            -np.pi,           # -180°
            -2*np.pi/3,       # -120°
            -np.pi/2,         # -90°
            -5*np.pi/12,      # -75°
            -np.pi/3,         # -60°
            -np.pi/4,         # -45°
            -np.pi/6,         # -30°
            -np.pi/12,        # -15°
            0,                # 0°
            np.pi/12,         # 15°
            np.pi/6,          # 30°
            np.pi/4,          # 45°
            np.pi/3,          # 60°
            5*np.pi/12,       # 75°
            np.pi/2,          # 90°
            2*np.pi/3,        # 120°
            np.pi             # 180°
        ])
        distances = np.abs(heading_values - heading_cmd)
        return np.argmin(distances)

    def _convert_velocity_to_index(self, velocity_offset):
        """速度偏移转索引"""
        velocity_values = np.array([-150, -100, -50, 0, 50, 100, 150])
        distances = np.abs(velocity_values - velocity_offset)
        return np.argmin(distances)



    def _use_lowlevel_policy(self, env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id):
        """使用低级策略网络"""
        if self.my_lowlevel_policy is None:
            return self._direct_control_mapping(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)
        try:
            raw_obs = self.get_obs(env, agent_id)
            input_obs = np.zeros(12)

            # 安全索引访问，防止越界
            altitude_cmd_id = min(altitude_cmd_id, len(self.norm_delta_altitude) - 1)
            heading_cmd_id = min(heading_cmd_id, len(self.norm_delta_heading) - 1)
            velocity_cmd_id = min(velocity_cmd_id, len(self.norm_delta_velocity) - 1)

            input_obs[0] = self.norm_delta_altitude[altitude_cmd_id]
            input_obs[1] = self.norm_delta_heading[heading_cmd_id]
            input_obs[2] = self.norm_delta_velocity[velocity_cmd_id]
            input_obs[3:12] = raw_obs[:9]
            input_obs = np.nan_to_num(input_obs, nan=0.0)
            input_obs = np.expand_dims(input_obs, axis=0)
            if agent_id not in self._inner_rnn_states:
                self._inner_rnn_states[agent_id] = np.zeros((1, 1, 128))
            _action, _rnn_states = self.my_lowlevel_policy(
                torch.FloatTensor(input_obs),
                torch.FloatTensor(self._inner_rnn_states[agent_id])
            )
            action_output = _action.detach().cpu().numpy().squeeze(0)
            self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()
            norm_act = np.zeros(4)
            norm_act[0] = action_output[0] / 20 - 1.
            norm_act[1] = action_output[1] / 20 - 1.
            norm_act[2] = action_output[2] / 20 - 1.
            norm_act[3] = action_output[3] / 58 + 0.4
            current_alt = env.agents[agent_id].get_position()[2]
            if current_alt < 1000:
                norm_act[1] = max(norm_act[1], 0.0)
                norm_act[3] = max(norm_act[3], 0.8)
            return norm_act
        except Exception as e:
            logging.error(f"低级策略错误: {e}")
            return self._direct_control_mapping(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)

    def _use_lowlevel_policy_with_roll(self, env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id,
                                       target_roll):
        """使用低级策略网络，包含滚转控制"""
        if self.my_lowlevel_policy is None:
            return self._direct_control_mapping_with_roll(env, agent_id, altitude_cmd_id, heading_cmd_id,
                                                          velocity_cmd_id, target_roll)
        try:
            raw_obs = self.get_obs(env, agent_id)
            input_obs = np.zeros(12)

            # 安全索引访问，防止越界
            altitude_cmd_id = min(altitude_cmd_id, len(self.norm_delta_altitude) - 1)
            heading_cmd_id = min(heading_cmd_id, len(self.norm_delta_heading) - 1)
            velocity_cmd_id = min(velocity_cmd_id, len(self.norm_delta_velocity) - 1)

            input_obs[0] = self.norm_delta_altitude[altitude_cmd_id]
            input_obs[1] = self.norm_delta_heading[heading_cmd_id]
            input_obs[2] = self.norm_delta_velocity[velocity_cmd_id]
            input_obs[3:12] = raw_obs[:9]
            input_obs = np.nan_to_num(input_obs, nan=0.0)
            input_obs = np.expand_dims(input_obs, axis=0)
            if agent_id not in self._inner_rnn_states:
                self._inner_rnn_states[agent_id] = np.zeros((1, 1, 128))
            _action, _rnn_states = self.my_lowlevel_policy(
                torch.FloatTensor(input_obs),
                torch.FloatTensor(self._inner_rnn_states[agent_id])
            )
            action_output = _action.detach().cpu().numpy().squeeze(0)
            self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()
            norm_act = np.zeros(4)
            norm_act[0] = action_output[0] / 20 - 1.
            norm_act[1] = action_output[1] / 20 - 1.
            norm_act[2] = action_output[2] / 20 - 1.
            norm_act[3] = action_output[3] / 58 + 0.4
            current_roll = env.agents[agent_id].get_property_value(c.attitude_phi_rad)
            current_roll_deg = np.rad2deg(current_roll)
            roll_error = target_roll - current_roll_deg
            if abs(roll_error) > 2.0:
                roll_cmd = np.clip(roll_error / 45.0, -1.0, 1.0)
                norm_act[0] = roll_cmd
                if env.current_step % 50 == 0:
                    logging.info(f"{agent_id} 滚转控制: 目标={target_roll:.1f}°, "
                                 f"当前={current_roll_deg:.1f}°, 误差={roll_error:.1f}°, 指令={roll_cmd:.2f}")
            current_alt = env.agents[agent_id].get_position()[2]
            if current_alt < 1000:
                norm_act[1] = max(norm_act[1], 0.0)
                norm_act[3] = max(norm_act[3], 0.8)
            return norm_act
        except Exception as e:
            logging.error(f"带滚转的低级策略错误: {e}")
            return self._direct_control_mapping_with_roll(env, agent_id, altitude_cmd_id, heading_cmd_id,
                                                          velocity_cmd_id, target_roll)

    def _direct_control_mapping(self, env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id):
        """直接控制映射"""
        try:
            current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
            current_altitude = env.agents[agent_id].get_position()[2]
            aileron = 0.0
            elevator = 0.0
            rudder = 0.0
            throttle = 0.7
            # 安全索引访问，防止越界
            altitude_cmd_id = min(altitude_cmd_id, len(self.norm_delta_altitude) - 1)
            heading_cmd_id = min(heading_cmd_id, len(self.norm_delta_heading) - 1)

            target_altitude_change = self.norm_delta_altitude[altitude_cmd_id] * 1000.0  # 转换回米
            if target_altitude_change > 0:
                elevator = 0.1
                throttle = 0.8
            elif target_altitude_change < 0:
                elevator = -0.1
                throttle = 0.6
            target_heading_change = np.rad2deg(self.norm_delta_heading[heading_cmd_id])
            if target_heading_change > 5:
                aileron = 0.2
                rudder = 0.1
            elif target_heading_change < -5:
                aileron = -0.2
                rudder = -0.1
            if current_altitude < 1000:
                elevator = max(elevator, 0.1)
                throttle = max(throttle, 0.8)
            return np.array([aileron, elevator, rudder, throttle])
        except Exception as e:
            logging.error(f"直接控制映射错误: {e}")
            return np.array([0.0, 0.0, 0.0, 0.7])

    def _direct_control_mapping_with_roll(self, env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id,
                                          target_roll):
        """直接控制映射，包含滚转控制"""
        try:
            base_action = self._direct_control_mapping(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)
            current_roll = env.agents[agent_id].get_property_value(c.attitude_phi_rad)
            current_roll_deg = np.rad2deg(current_roll)
            roll_error = target_roll - current_roll_deg
            if abs(roll_error) > 2.0:
                roll_cmd = np.clip(roll_error / 45.0, -1.0, 1.0)
                base_action[0] = roll_cmd
            return base_action
        except Exception as e:
            logging.error(f"带滚转的直接控制映射错误: {e}")
            return np.array([0.0, 0.0, 0.0, 0.7])

    def set_crank_params(self, angle_deg=45.0, turn_rate_deg_per_sec=3.0, hold_time_sec=20.0):
        """设置Crank机动参数"""
        self.maneuver_params = {
            "crank_angle_deg": angle_deg,
            "turn_rate_deg_per_sec": turn_rate_deg_per_sec,
            "hold_time_sec": hold_time_sec
        }
        logging.info(f"Crank参数更新: 角度={angle_deg}°, 转弯率={turn_rate_deg_per_sec}°/s, 保持时间={hold_time_sec}s")

    def set_beam_params(self, angle_deg=90.0, turn_rate_deg_per_sec=5.0, hold_time_sec=15.0):
        """设置Beam机动参数"""
        self.maneuver_params = {
            "beam_angle_deg": angle_deg,
            "turn_rate_deg_per_sec": turn_rate_deg_per_sec,
            "hold_time_sec": hold_time_sec
        }
        logging.info(f"Beam参数更新: 角度={angle_deg}°, 转弯率={turn_rate_deg_per_sec}°/s, 保持时间={hold_time_sec}s")

    def set_notch_params(self, angle_deg=90.0, turn_rate_deg_per_sec=4.0,
                         descent_rate_ft_per_sec=60.0, descent_time_sec=5.0, hold_time_sec=15.0):
        """设置Notch机动参数"""
        self.maneuver_params = {
            "notch_angle_deg": angle_deg,
            "turn_rate_deg_per_sec": turn_rate_deg_per_sec,
            "descent_rate_ft_per_sec": descent_rate_ft_per_sec,
            "descent_time_sec": descent_time_sec,
            "hold_time_sec": hold_time_sec
        }
        logging.info(f"Notch参数更新: 角度={angle_deg}°, 转弯率={turn_rate_deg_per_sec}°/s, "
                     f"下降率={descent_rate_ft_per_sec}ft/s, 下降时间={descent_time_sec}s, 保持时间={hold_time_sec}s")

    def set_maneuver_type(self, maneuver_type="crank"):
        """设置机动类型"""
        self.maneuver_type = maneuver_type
        logging.info(f"机动类型设置为: {maneuver_type}")

    def set_short_skate_maneuver(self, custom_params=None):
        """设置Short Skate机动 - 发射后快速脱离战术"""
        self.set_composite_maneuver("short_skate_tactical", custom_params)
        logging.info("Short Skate机动设置完成")
        logging.info("机动描述: 发射后快速脱离战术")
        logging.info("阶段: 发射准备 -> Crank机动 -> Turn Cold逃逸")

    def get_available_maneuvers(self):
        """获取可用的机动列表"""
        maneuvers = {
            "large_maneuvers": ["crank", "beam", "notch"],
            "basic_maneuvers": [
                "level_flight", "accelerate", "decelerate", "turn",
                "pull_up", "dive", "diagonal_flight"
            ],
            "composite_maneuvers": self.composite_executor.get_available_maneuvers()
        }
        return maneuvers

    def export_trajectory_data(self, filepath=None):
        """导出轨迹数据"""
        if not self.trajectory_data:
            logging.warning("没有轨迹数据可导出")
            return None
        if filepath is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"trajectory_data_{self.maneuver_type}_{timestamp}.json"
        try:
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.trajectory_data, f, indent=2, ensure_ascii=False)
            logging.info(f"轨迹数据已导出到: {filepath}")
            return filepath
        except Exception as e:
            logging.error(f"轨迹数据导出失败: {e}")
            return None