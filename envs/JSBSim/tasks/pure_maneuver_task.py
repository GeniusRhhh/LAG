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
        # F16模型使用use_mlp_actlayer=False训练
        self.my_lowlevel_policy = BaselineActor(use_mlp_actlayer=False)
        self.enemy_baseline_policy = BaselineActor(use_mlp_actlayer=False)
        self._inner_rnn_states = {}
        self._enemy_rnn_states = {}

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
        # 修复速度控制：更精确的速度映射，确保加速机动有明显效果
        self.norm_delta_velocity = np.array([-200, -150, -100, 0, 50, 100, 200]) / 5.0  # 改为除以5，进一步增强控制力度
        self.composite_executor = CompositeManeuverExecutor()
        self.maneuver_composer = self.composite_executor
        self._load_baseline_models()
        logging.info("PureManeuverTask初始化完成 - 支持基础机动和组合机动")

    def _load_baseline_models(self):
        """加载baseline模型 - 强制使用F16 baseline_model.pt"""
        import os
        try:
            root_dir = get_root_dir()
            # 强制使用F16 baseline_model.pt，不使用SU27模型
            model_path = os.path.join(root_dir, 'model', 'baseline_model.pt')
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"未找到F16 baseline模型文件: {model_path}")
            
            logging.info(f"✅ 强制加载F16 baseline模型: {model_path}")
            
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
            logging.info(f"✅ Baseline模型加载成功")
        except Exception as e:
            logging.error(f"❌ 加载baseline模型失败: {e}")
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
        elif maneuver_name == "accelerate":
            default_params = {"velocity_increase": 100.0, "duration": 20.0}
        elif maneuver_name == "decelerate":
            default_params = {"velocity_decrease": 100.0, "duration": 20.0}
        elif maneuver_name == "turn":
            default_params = {"turn_angle": 45.0, "turn_rate": 3.0}
        elif maneuver_name in ["pull_up", "dive"]:
            default_params = {"altitude_change": 1500.0, "duration": 15.0}
            if maneuver_name == "dive":
                default_params["min_altitude"] = 2000.0
        elif maneuver_name == "diagonal_flight":
            default_params = {"turn_angle": 45.0, "altitude_change": 1000.0, "duration": 15.0}
        elif maneuver_name == "notch_back":
            default_params = {"duration": 25.0, "altitude_loss": 1500.0, "turn_angle": 90.0, "min_altitude": 2500.0}
        elif maneuver_name == "short_skate":
            default_params = {"duration": 35.0, "crank_angle": 45.0, "hold_time": 5.0, "turn_back_angle": 180.0, "acceleration": 50.0}

        elif maneuver_name == "accelerate_escape":
            default_params = {"acceleration": 50.0, "duration": 20.0}
        elif maneuver_name == "vertical_loop":
            default_params = {"loop_type": "half", "g_force": 6.0, "duration": 15.0}
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
            # 只记录A0100的初始状态，注释掉B0100敌方信息
            if agent_id == "A0100":
                logging.info(f"{agent_id} 初始状态:")
                logging.info(f"航向: {self.initial_heading[agent_id]:.1f}°")
                logging.info(f"高度: {self.initial_altitude[agent_id]:.1f}m")
            # 注释掉B0100的初始状态信息
            # elif agent_id == "B0100":
            #     pass  # 不输出B0100的初始状态
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
        
        # 记录初始速度和目标增量（用于后续速度稳定控制）
        if not hasattr(self, '_initial_velocity'):
            self._initial_velocity = {}
        if not hasattr(self, '_target_velocity_increase'):
            self._target_velocity_increase = {}
            
        if agent_id not in self._initial_velocity:
            self._initial_velocity[agent_id] = current_velocity
            if basic_maneuver_name == "accelerate":
                params = self.maneuver_params.get("params", {})
                target_increase = params.get("velocity_change", params.get("velocity_increase", 50.0))
                self._target_velocity_increase[agent_id] = target_increase
                logging.info(f"[机动开始] {agent_id} accelerate: 初始速度={current_velocity:.1f}m/s, 目标增量={target_increase:.1f}m/s")
            elif basic_maneuver_name == "decelerate":
                params = self.maneuver_params.get("params", {})
                target_decrease = params.get("velocity_change", params.get("velocity_decrease", 40.0))
                self._target_velocity_increase[agent_id] = -target_decrease  # 负值表示减速
                logging.info(f"[机动开始] {agent_id} decelerate: 初始速度={current_velocity:.1f}m/s, 目标减量={target_decrease:.1f}m/s")
        current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        result = self._call_basic_maneuver_function(basic_maneuver_name, current_time,
                                                    initial_heading, initial_altitude,
                                                    current_velocity, current_altitude, current_heading)
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
        if basic_maneuver_name in ["turn", "turn_level", "accelerate", "decelerate", "level_flight", "crank"]:
            # 对于水平飞行机动，强制严格保持初始高度
            altitude_diff = initial_altitude - current_altitude
            if abs(altitude_diff) > 2.0:
                if basic_maneuver_name in ["accelerate", "decelerate"]:
                    if current_altitude > initial_altitude + 15.0:
                        altitude_diff = altitude_diff * 3.5
                    elif current_altitude > initial_altitude + 8.0:
                        altitude_diff = altitude_diff * 2.5
                    elif abs(altitude_diff) > 4.0:
                        altitude_diff = altitude_diff * 2.0
                elif basic_maneuver_name == "turn_level":
                    if current_altitude > initial_altitude + 20.0:
                        altitude_diff = altitude_diff * 3.0
                    elif current_altitude > initial_altitude + 10.0:
                        altitude_diff = altitude_diff * 2.5
                    elif abs(altitude_diff) > 5.0:
                        altitude_diff = altitude_diff * 2.0
                else:
                    if current_altitude > initial_altitude + 25.0:
                        altitude_diff = altitude_diff * 2.5
                    elif current_altitude > initial_altitude + 12.0:
                        altitude_diff = altitude_diff * 2.0
                altitude_cmd_id = self._convert_altitude_to_index(altitude_diff)
        elif target_altitude is not None:
            altitude_diff = target_altitude - current_altitude
            if abs(altitude_diff) > 2.0:
                if basic_maneuver_name == "dive":
                    # 修复：dive机动根据phase调整控制强度
                    if phase == "DIVE_FINISHED":
                        # 俯冲完成后，强制保持目标高度，使用强控制
                        if abs(altitude_diff) > 100.0:
                            altitude_diff = altitude_diff * 4.0  # 强力拉升
                        elif abs(altitude_diff) > 50.0:
                            altitude_diff = altitude_diff * 3.5
                        else:
                            altitude_diff = altitude_diff * 3.0
                    else:
                        # 俯冲过程中使用温和控制
                        if abs(altitude_diff) > 200.0:
                            altitude_diff = altitude_diff * 2.5
                        elif abs(altitude_diff) > 100.0:
                            altitude_diff = altitude_diff * 2.0
                        elif abs(altitude_diff) > 50.0:
                            altitude_diff = altitude_diff * 1.8
                        else:
                            altitude_diff = altitude_diff * 1.5
                elif basic_maneuver_name == "pull_up":
                    # pull_up保持原有控制强度
                    if abs(altitude_diff) > 100.0:
                        altitude_diff = altitude_diff * 3.5
                    elif abs(altitude_diff) > 50.0:
                        altitude_diff = altitude_diff * 3.0
                    elif abs(altitude_diff) > 20.0:
                        altitude_diff = altitude_diff * 2.5
                    else:
                        altitude_diff = altitude_diff * 2.0
                altitude_cmd_id = self._convert_altitude_to_index(altitude_diff)

        # 角度控制 - 使用扩充的离散控制
        if target_heading is not None:
            heading_diff = target_heading - current_heading
            while heading_diff > 180: heading_diff -= 360
            while heading_diff < -180: heading_diff += 360

            # 强制角度控制：如果角度差异超过1度就进行控制
            if abs(heading_diff) > 1.0:
                heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))

        # 修复加速/减速动作的速度控制
        if velocity_offset is not None:
            if basic_maneuver_name == "level_flight" and abs(velocity_offset) < 0.1:
                velocity_cmd_id = 3
            elif abs(velocity_offset) > 1.0:
                if basic_maneuver_name == "decelerate":
                    # 修复：减速使用标准系数，避免过度控制
                    velocity_offset = velocity_offset * 1.0  # 减速使用标准系数
                elif basic_maneuver_name == "accelerate":
                    velocity_offset = velocity_offset * 1.2  # 加速使用1.2倍轻微增强
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
                if phase == "DIVE_FINISHED":
                    # 俯冲完成：强制保持目标高度，降低阈值到2米
                    if abs(altitude_diff) > 2.0:
                        # 增强控制强度，确保稳定在目标高度
                        if altitude_diff < 0:  # 需要下降
                            altitude_diff = altitude_diff * 2.5
                        else:  # 需要拉升
                            altitude_diff = altitude_diff * 3.5
                        altitude_cmd_id = self._convert_altitude_to_index(altitude_diff)
                elif abs(altitude_diff) > 5.0:  # DIVING阶段：5米阈值
                    altitude_cmd_id = self._convert_altitude_to_index(altitude_diff)

        elif phase in ["MANEUVER_COMPLETED", "ACCELERATING_ESCAPE", "ESCAPE_COMPLETE",
                       "VERTICAL_LOOPING", "VERTICAL_LOOP_COMPLETE"]:
            # 机动完成状态和特殊机动状态：保持目标高度和航向，确保平稳飞行
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
            elif phase in ["MAINTAINING_HEADING", "MANEUVER_COMPLETED", "ESCAPE_COMPLETE", "VERTICAL_LOOP_COMPLETE"]:
                threshold = 1.5  # 保持航向和机动完成：1.5度精度
            else:
                threshold = 1.0  # 其他阶段：1度精度

            if abs(heading_diff) > threshold:
                heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))

        # 速度控制
        # 关键修复：机动完成后必须强制维持速度稳定，不能继续加减速
        if velocity_offset is not None:
            if phase in ["DECELERATION_COMPLETE", "ACCELERATION_COMPLETE", "DIVE_FINISHED"]:
                # 机动完成后，需要主动稳定速度，抵消惯性影响
                # 使用双重检查：速度变化率 + 目标速度偏差
                
                # 记录目标速度（如果还没有记录）
                if not hasattr(self, '_target_velocity'):
                    self._target_velocity = {}
                if agent_id not in self._target_velocity:
                    # 根据机动类型设置目标速度
                    if phase == "ACCELERATION_COMPLETE":
                        # 加速完成，目标速度应该是初始速度+目标增量
                        if hasattr(self, '_initial_velocity') and agent_id in self._initial_velocity:
                            target_increase = getattr(self, '_target_velocity_increase', {}).get(agent_id, 50.0)
                            self._target_velocity[agent_id] = self._initial_velocity[agent_id] + target_increase
                        else:
                            self._target_velocity[agent_id] = current_velocity  # 保持当前速度
                    else:
                        self._target_velocity[agent_id] = current_velocity  # 保持当前速度
                
                target_vel = self._target_velocity[agent_id]
                velocity_error = current_velocity - target_vel
                
                # 检查速度变化率
                velocity_change_rate = 0.0
                if hasattr(self, '_last_velocity') and agent_id in self._last_velocity:
                    velocity_change_rate = current_velocity - self._last_velocity[agent_id]
                
                logging.info(f"[速度稳定] {phase}: 当前={current_velocity:.1f}m/s, 目标={target_vel:.1f}m/s, 误差={velocity_error:.1f}m/s, 变化率={velocity_change_rate:.1f}m/s")
                
                # 主动速度稳定控制
                if phase == "ACCELERATION_COMPLETE":
                    if velocity_error > 10.0 or velocity_change_rate > 0.5:  # 速度超出目标太多或还在增加
                        velocity_cmd_id = 1  # 强减速（对应-150/5=-30m/s）
                        logging.info(f"🚨 加速超调严重(误差={velocity_error:.1f}m/s, 变化率={velocity_change_rate:.1f}m/s)，强减速: velocity_cmd_id=1 (-30m/s)")
                    elif velocity_error > 5.0:  # 轻微超调
                        velocity_cmd_id = 2  # 轻减速（对应-100/5=-20m/s）
                        logging.info(f"⚠️ 加速轻微超调(误差={velocity_error:.1f}m/s)，轻减速: velocity_cmd_id=2 (-20m/s)")
                    else:
                        velocity_cmd_id = 3  # 中性控制
                        logging.info(f"✅ 加速已稳定(误差={velocity_error:.1f}m/s)，中性控制: velocity_cmd_id=3 (0m/s)")
                elif phase == "DECELERATION_COMPLETE":
                    if velocity_error < -10.0 or velocity_change_rate < -0.5:  # 减速过度或还在减少
                        velocity_cmd_id = 5  # 强加速（对应100/5=20m/s）
                        logging.info(f"🚨 减速过度严重(误差={velocity_error:.1f}m/s, 变化率={velocity_change_rate:.1f}m/s)，强加速: velocity_cmd_id=5 (+20m/s)")
                    elif velocity_error < -5.0:  # 轻微过度
                        velocity_cmd_id = 4  # 轻加速（对应50/5=10m/s）
                        logging.info(f"⚠️ 减速轻微过度(误差={velocity_error:.1f}m/s)，轻加速: velocity_cmd_id=4 (+10m/s)")
                    else:
                        velocity_cmd_id = 3  # 中性控制
                        logging.info(f"✅ 减速已稳定(误差={velocity_error:.1f}m/s)，中性控制: velocity_cmd_id=3 (0m/s)")
                else:
                    velocity_cmd_id = 3  # 其他情况使用中性控制
                    
                # 记录当前速度用于下次比较
                if not hasattr(self, '_last_velocity'):
                    self._last_velocity = {}
                self._last_velocity[agent_id] = current_velocity
                    
            elif abs(velocity_offset) > 2.0:
                velocity_cmd_id = self._convert_velocity_to_index(velocity_offset)
            else:
                # 对于小的速度偏差，也使用中性控制
                velocity_cmd_id = 3  # 修复：使用索引3（对应0m/s）

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
                                      current_velocity, current_altitude, current_heading=None):
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
            # 修复：使用正确的参数顺序调用新版本的accelerate函数
            velocity_increase = params.get("velocity_change", params.get("velocity_increase", 50.0))
            return BasicManeuvers.accelerate(
                current_time,
                current_velocity,
                initial_altitude,  # 传递初始高度
                params.get("duration", 15.0),  # 持续时间
                velocity_increase,  # 速度增量
                350.0,  # 最大速度限制
                initial_heading  # 初始航向
            )
        elif basic_maneuver_name == "decelerate":
            # 修复：使用正确的参数顺序调用新版本的decelerate函数
            velocity_decrease = params.get("velocity_change", params.get("velocity_decrease", 40.0))
            return BasicManeuvers.decelerate(
                current_time,
                current_velocity,
                initial_altitude,  # 传递初始高度
                params.get("duration", 25.0),  # 持续时间
                velocity_decrease,  # 速度减量
                200.0,  # 最小速度限制
                initial_heading  # 初始航向
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
            # 支持新的参数名称 altitude_gain，同时保持向后兼容
            altitude_change = params.get("altitude_gain", params.get("altitude_change", 1500.0))
            return BasicManeuvers.pull_up(
                current_time,
                initial_altitude,
                params.get("duration", 15.0),
                altitude_change
            )
        elif basic_maneuver_name == "dive":
            # 支持新的参数名称 altitude_loss，同时保持向后兼容
            altitude_change = params.get("altitude_loss", params.get("altitude_change", 1500.0))
            return BasicManeuvers.dive(
                current_time,
                initial_altitude,
                params.get("duration", 15.0),
                altitude_change,
                params.get("min_altitude", 2000.0),
                initial_heading  # 传入initial_heading保持航向稳定
            )
        elif basic_maneuver_name == "crank":
            # 战术偏置转向机动 - 传递当前航向进行精确控制
            crank_angle = params.get("crank_angle", params.get("turn_angle", 45.0))
            turn_rate = params.get("turn_rate", 3.0)
            duration = params.get("duration", None)  # 如果None则自动计算
            return BasicManeuvers.crank(
                current_time,
                initial_heading,
                initial_altitude,
                crank_angle,
                turn_rate,
                duration,
                current_heading
            )
        elif basic_maneuver_name == "diagonal_flight":
            # 支持组合机动的参数名称，同时保持向后兼容
            turn_angle = params.get("turn_angle", 45.0)

            # 正确处理高度变化参数
            if "altitude_gain" in params:
                altitude_change = abs(params.get("altitude_gain", 1000.0))  # 爬升为正值
            elif "altitude_loss" in params:
                altitude_change = -abs(params.get("altitude_loss", 1000.0))  # 俯冲为负值
            else:
                altitude_change = params.get("altitude_change", 1000.0)  # 向后兼容

            return BasicManeuvers.diagonal_flight(
                current_time,
                initial_heading,
                initial_altitude,
                params.get("duration", 15.0),
                turn_angle,
                altitude_change,
                params.get("min_altitude", 3000.0)
            )
        elif basic_maneuver_name == "maintain_heading_flight":
            return BasicManeuvers.maintain_heading_flight(
                current_time,
                params.get("target_heading", initial_heading),
                params.get("duration", 15.0)
            )

        elif basic_maneuver_name == "accelerate_escape":
            return BasicManeuvers.accelerate_escape(
                current_time,
                initial_heading,
                params.get("duration", 20.0),
                params.get("acceleration", 50.0)
            )
        elif basic_maneuver_name == "vertical_loop":
            return BasicManeuvers.vertical_loop(
                current_time,
                initial_heading,
                params.get("loop_type", "half"),
                params.get("g_force", 6.0)
            )
        elif basic_maneuver_name == "notch_back":
            return BasicManeuvers.notch_back(
                current_time,
                initial_heading,
                initial_altitude,
                params.get("duration", 25.0),
                params.get("altitude_loss", 1500.0),
                params.get("turn_angle", 90.0),
                params.get("min_altitude", 2500.0)
            )
        elif basic_maneuver_name == "short_skate":
            return BasicManeuvers.short_skate(
                current_time,
                initial_heading,
                initial_altitude,
                params.get("duration", 35.0),
                params.get("crank_angle", 45.0),
                params.get("hold_time", 5.0),
                params.get("turn_back_angle", 180.0),
                params.get("acceleration", 50.0)
            )
        else:
            return (None, None, None, None, None)

    def _process_observer_behavior(self, env, agent_id):
        """处理观察飞机的行为 - 使用F16 baseline保持平飞"""
        try:
            # 使用新数组的中间索引来保持平稳飞行
            # 高度：15个值的中间是索引7 (对应0米变化)
            # 航向：17个值的中间是索引8 (对应0度变化)
            # 速度：7个值的中间是索引3 (对应0m/s速度变化)
            altitude_cmd_id = 7  # 对应0米高度变化
            heading_cmd_id = 8   # 对应0度航向变化
            velocity_cmd_id = 3  # 对应0m/s速度变化
            # 使用F16 baseline策略
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
            # 只记录A0100的信息，注释掉B0100敌方信息
            if agent_id == "A0100":
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
            # 注释掉B0100的调试信息
            # elif agent_id == "B0100":
            #     pass  # 不输出B0100的调试信息
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
        """速度偏移转索引 - 修复：更精确的速度映射"""
        # 修复：使用更细粒度的速度映射，确保小幅加速也能生效
        velocity_values = np.array([-200, -150, -100, 0, 50, 100, 200])  # 添加50m/s档位

        # 平衡的速度控制映射，避免过于激进
        if velocity_offset >= 50.0:  # 50m/s及以上使用最大索引6
            return 6
        elif velocity_offset >= 30.0:  # 30m/s及以上使用索引5
            return 5
        elif velocity_offset >= 15.0:  # 15m/s及以上的加速使用索引4
            return 4
        elif velocity_offset <= -40.0:  # 大幅减速 - 降低阈值
            return 0
        elif velocity_offset <= -20.0:  # 中等减速 - 降低阈值
            return 1
        elif velocity_offset <= -10.0:  # 轻微减速 - 降低阈值
            return 2
        else:  # 其他情况使用最接近的值
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
            # 叠加高度控制：无论是否有速度控制，都根据高度指令微调升降舵与油门
            alt_cmd_norm = float(self.norm_delta_altitude[min(altitude_cmd_id, len(self.norm_delta_altitude)-1)])
            elev_add = np.clip(alt_cmd_norm * 0.25, -0.25, 0.25)
            norm_act[1] = np.clip(norm_act[1] + elev_add, -1.0, 1.0)
            if alt_cmd_norm > 0:
                norm_act[3] = max(norm_act[3], 0.85)
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
            velocity_cmd_id = min(velocity_cmd_id, len(self.norm_delta_velocity) - 1)

            # 修复速度控制：根据velocity_cmd_id调整推力（超级增强版本，匹配新的归一化值）
            target_velocity_change = self.norm_delta_velocity[velocity_cmd_id]
            if target_velocity_change > 30.0:  # 极大幅加速 (200/5=40.0)
                throttle = 1.0  # 最大推力
                elevator = 0.0  # 保持水平，专注加速
            elif target_velocity_change > 15.0:  # 大幅加速 (100/5=20.0)
                throttle = 1.0  # 最大推力
                elevator = 0.0
            elif target_velocity_change > 8.0:  # 中等加速 (50/5=10.0)
                throttle = 0.98  # 提高推力
                elevator = 0.0
            elif target_velocity_change > 0.0:  # 轻微加速
                throttle = 0.90  # 提高推力
                elevator = 0.0
            elif target_velocity_change < -30.0:  # 极大幅减速 (-200/5=-40.0)
                throttle = 0.55  # 保持足够油门维持升力
                elevator = 0.12  # 微抬机头增加阻力减速
            elif target_velocity_change < -20.0:  # 大幅减速 (-150/5=-30.0)
                throttle = 0.60
                elevator = 0.10  # 微抬机头增加阻力减速
            elif target_velocity_change < -15.0:  # 中等减速 (-100/5=-20.0)
                throttle = 0.65
                elevator = 0.08  # 微抬机头增加阻力减速
            elif target_velocity_change < 0.0:  # 轻微减速
                throttle = 0.68
                elevator = 0.05  # 微抬机头增加阻力减速
            else:  # 保持速度
                throttle = 0.7
                elevator = 0.0

            # 高度控制：根据机动类型决定是否启用
            target_altitude_change = self.norm_delta_altitude[altitude_cmd_id] * 1000.0  # 转换回米
            
            # 关键修复：加速机动期间禁用高度控制，避免控制冲突
            basic_maneuver_name = getattr(self, 'maneuver_params', {}).get("basic_maneuver_name", "")
            is_accelerating = target_velocity_change > 5.0
            is_decelerating = target_velocity_change < -5.0
            
            if basic_maneuver_name == "accelerate" and is_accelerating:
                # 加速机动期间：完全禁用高度控制，专注速度控制
                elev_add = 0.0
                logging.debug(f"加速机动期间禁用高度控制: target_vel_change={target_velocity_change:.1f}")
            elif is_decelerating:
                # 减速时维持高度控制，但限制范围避免过度
                elev_add = np.clip(target_altitude_change / 5000.0, -0.12, 0.12)
            else:
                # 其他情况正常高度控制
                elev_add = np.clip(target_altitude_change / 4000.0, -0.25, 0.25)
            
            elevator = np.clip(elevator + elev_add, -1.0, 1.0)
            if target_altitude_change > 0:
                throttle = max(throttle, 0.85)  # 爬升时确保足够推力
            target_heading_change = np.rad2deg(self.norm_delta_heading[heading_cmd_id])
            
            # 关键修复：加速机动期间使用温和的航向控制
            if basic_maneuver_name == "accelerate" and is_accelerating:
                # 加速期间：温和的航向控制，避免过度干预
                if target_heading_change > 10:  # 提高阈值，减少干预
                    aileron = 0.1  # 减小控制量
                    rudder = 0.05
                elif target_heading_change < -10:
                    aileron = -0.1
                    rudder = -0.05
                logging.debug(f"加速机动期间温和航向控制: target_hdg_change={target_heading_change:.1f}")
            elif is_decelerating:
                # 减速时加强航向控制，保持稳定
                if target_heading_change > 5:
                    aileron = 0.15  # 增强控制
                    rudder = 0.08
                elif target_heading_change < -5:
                    aileron = -0.15
                    rudder = -0.08
            else:
                # 其他情况正常航向控制
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

    def set_banzai_maneuver(self, custom_params=None):
        """设置Banzai机动 - 发射后决策战术"""
        self.set_composite_maneuver("banzai_tactical", custom_params)
        logging.info("Banzai机动设置完成")
        logging.info("机动描述: 发射后决策战术(Launch & Decide)")
        logging.info("阶段: 发射准备 -> Crank防御 -> 决策转向 -> 迎敌格斗")

    def set_sliceback_maneuver(self, custom_params=None):
        """设置Sliceback机动 - 水平滚转+高G回旋"""
        self.set_composite_maneuver("sliceback_tactical", custom_params)
        logging.info("Sliceback机动设置完成")
        logging.info("机动描述: 水平滚转135度 + 高G力拉杆回旋")
        logging.info("阶段: 桶滚135度 -> 高G回旋180度 -> 恢复平飞")

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