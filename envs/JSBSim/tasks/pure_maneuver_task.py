# envs/JSBSim/tasks/pure_maneuver_task.py
import logging
import numpy as np
import torch
from typing import Dict, Any
from .multiplecombat_task import MultipleCombatTask
from .pure_maneuvers import PureManeuvers, BasicManeuvers, ManeuverComposer
from ..core.catalog import Catalog as c
from ..model.baseline_actor import BaselineActor
from ..utils.utils import get_root_dir


class PureManeuverTask(MultipleCombatTask):
    """纯机动测试任务 - 支持基础机动和组合机动"""

    def __init__(self, config):
        super().__init__(config)

        # 机动参数暴露，可以修改
        self.maneuver_type = "crank"  # 支持: crank, beam, notch, basic, composite
        self.maneuver_params = {
            "crank_angle_deg": 60.0,
            "turn_rate_deg_per_sec": 3.0,
            "hold_time_sec": 20.0
        }

        # 基础机动参数
        self.basic_maneuver_params = {
            "turn_angle": 45.0,
            "turn_rate": 3.0,
            "altitude_change": 1000.0,
            "velocity_change": 50.0,
            "duration": 10.0
        }

        # 组合机动参数
        self.composite_maneuver_name = "escape"  # escape, attack, defense

        # 测试配置
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

        self.norm_delta_altitude = np.array([-1000, -500, -200, 0, 200, 500, 1000]) / 1000.0
        self.norm_delta_heading = np.array(
            [-np.pi, -np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2, np.pi])
        self.norm_delta_velocity = np.array([-150, -100, -50, 0, 50, 100, 150]) / 100.0

        # 初始化机动组合器
        self.maneuver_composer = ManeuverComposer()

        # 加载模型
        self._load_baseline_models()

        logging.info("PureManeuverTask initialized - 支持基础机动和组合机动")

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

            # 加载主策略网络
            self.my_lowlevel_policy.load_state_dict(checkpoint)
            self.my_lowlevel_policy.eval()

            # 敌方也使用相同的策略
            self.enemy_baseline_policy.load_state_dict(checkpoint)
            self.enemy_baseline_policy.eval()


        except Exception as e:
            logging.error(f"加载baseline模型失败: {e}")
            self.my_lowlevel_policy = None
            self.enemy_baseline_policy = None

    def reset(self, env):
        """重置任务状态"""
        # 调用基类重置
        from .task_base import BaseTask
        BaseTask.reset(self, env)

        # 重置测试状态
        self.step_count = 0
        self.test_start_time = 0.0
        self.initial_heading.clear()
        self.initial_altitude.clear()
        self.trajectory_data.clear()

        # 重置RNN状态
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        self._enemy_rnn_states = {agent_id: torch.zeros(1, 1, 128) for agent_id in env.agents.keys()}

        logging.info(f"PureManeuverTask reset - testing: {self.maneuver_type}")
        logging.info(f"Maneuver params: {self.maneuver_params}")

    def normalize_action(self, env, agent_id, action):
        """动作归一化 """
        if agent_id not in env.agents or not env.agents[agent_id].is_alive:
            return np.array([0.0, 0.0, 0.0, 0.7])

        current_time = env.current_step * env.time_interval

        # 记录初始状态（只记录一次）
        if agent_id not in self.initial_heading:
            current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_rad)
            self.initial_heading[agent_id] = np.rad2deg(current_heading)

            current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
            self.initial_altitude[agent_id] = current_altitude

            logging.info(f"{agent_id} initial heading: {self.initial_heading[agent_id]:.1f}°")
            logging.info(f"{agent_id} initial altitude: {self.initial_altitude[agent_id]:.1f}m")

        # 区分测试飞机和观察飞机
        if agent_id == self.test_agent_id:
            # 测试飞机：执行机动
            return self._process_test_maneuver_action(env, agent_id, current_time)
        else:
            # 观察飞机：使用智能baseline行为
            return self._process_observer_behavior(env, agent_id)

    def _process_test_maneuver_action(self, env, agent_id, current_time):
        """处理测试飞机的机动动作 - 支持多种机动类型"""
        try:
            initial_heading = self.initial_heading[agent_id]
            initial_altitude = self.initial_altitude[agent_id]

            if self.maneuver_type in ["crank", "beam", "notch"]:
                # 现有的大机动
                return self._process_large_maneuver(env, agent_id, current_time, initial_heading, initial_altitude)

            elif self.maneuver_type == "basic":
                # 基础机动
                return self._process_basic_maneuver(env, agent_id, current_time, initial_heading, initial_altitude)

            elif self.maneuver_type == "composite":
                # 组合机动
                return self._process_composite_maneuver(env, agent_id, current_time, initial_heading, initial_altitude)

            else:
                # 默认保持稳定飞行
                return self._use_lowlevel_policy(env, agent_id, 3, 4, 3)

        except Exception as e:
            logging.error(f"❌ {agent_id} maneuver execution error: {e}")
            return self._direct_control_mapping(env, agent_id, 3, 4, 3)

    def _process_large_maneuver(self, env, agent_id, current_time, initial_heading, initial_altitude):
        """处理大机动 """
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
            # 机动结束，保持稳定飞行
            altitude_cmd_id = 3  # 保持高度
            heading_cmd_id = 4  # 保持航向
            velocity_cmd_id = 3  # 保持速度
        else:
            if len(maneuver_result) == 3:
                phase, target_heading, target_roll = maneuver_result
                target_altitude = None
            else:
                phase, target_heading, target_altitude, target_roll = maneuver_result

            # 调试输出
            if env.current_step % 100 == 0:
                logging.info(
                    f" {agent_id} {self.maneuver_type.title()} {phase}: Target={target_heading:.1f}°")

            # 转换为导航指令
            current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
            heading_diff = target_heading - current_heading

            # 规范化角度差到 [-180, 180] 范围
            while heading_diff > 180:
                heading_diff -= 360
            while heading_diff < -180:
                heading_diff += 360

            # 如果航向差很小，就保持当前航向
            if abs(heading_diff) < 2.0:
                heading_cmd_id = 4  # 保持航向
            else:
                heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))

            # 高度控制
            if target_altitude is not None:
                current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
                altitude_diff = target_altitude - current_altitude
                altitude_cmd_id = self._convert_altitude_to_index(altitude_diff)
            else:
                altitude_cmd_id = 3  # 保持高度

            velocity_cmd_id = 3  # 保持速度

        return self._use_lowlevel_policy(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)

    def _process_basic_maneuver(self, env, agent_id, current_time, initial_heading, initial_altitude):
        """处理基础机动"""
        basic_maneuver_name = self.maneuver_params.get("basic_maneuver_name", "level_flight")

        # 调用基础机动函数
        if basic_maneuver_name == "level_flight":
            result = BasicManeuvers.level_flight(current_time, self.basic_maneuver_params["duration"])
        elif basic_maneuver_name == "accelerate":
            result = BasicManeuvers.accelerate(current_time, self.basic_maneuver_params["duration"],
                                               self.basic_maneuver_params["velocity_change"])
        elif basic_maneuver_name == "decelerate":
            result = BasicManeuvers.decelerate(current_time, self.basic_maneuver_params["duration"],
                                               self.basic_maneuver_params["velocity_change"])
        elif basic_maneuver_name == "turn":
            result = BasicManeuvers.turn(current_time, initial_heading,
                                         self.basic_maneuver_params["turn_angle"],
                                         self.basic_maneuver_params["turn_rate"])
        elif basic_maneuver_name == "pull_up":
            result = BasicManeuvers.pull_up(current_time, self.basic_maneuver_params["duration"],
                                            self.basic_maneuver_params["altitude_change"])
        elif basic_maneuver_name == "dive":
            result = BasicManeuvers.dive(current_time, self.basic_maneuver_params["duration"],
                                         self.basic_maneuver_params["altitude_change"])
        elif basic_maneuver_name == "diagonal_flight":
            result = BasicManeuvers.diagonal_flight(current_time, self.basic_maneuver_params["duration"],
                                                    self.basic_maneuver_params["turn_angle"],
                                                    self.basic_maneuver_params["altitude_change"])
        elif basic_maneuver_name == "roll":
            result = BasicManeuvers.roll(current_time, self.basic_maneuver_params["duration"], 45.0)
        elif basic_maneuver_name == "turn_pull_up":
            result = BasicManeuvers.turn_pull_up(current_time, initial_heading,
                                                 self.basic_maneuver_params["turn_angle"],
                                                 self.basic_maneuver_params["turn_rate"],
                                                 self.basic_maneuver_params["altitude_change"])
        elif basic_maneuver_name == "turn_dive":
            result = BasicManeuvers.turn_dive(current_time, initial_heading,
                                              self.basic_maneuver_params["turn_angle"],
                                              self.basic_maneuver_params["turn_rate"],
                                              self.basic_maneuver_params["altitude_change"])
        else:
            result = (None, None, None, None, 0.0)

        phase, target_heading, target_altitude, target_velocity, target_roll = result

        if phase is None:
            # 机动结束，保持稳定飞行
            return self._use_lowlevel_policy(env, agent_id, 3, 4, 3)

        # 调试输出
        if env.current_step % 100 == 0:
            logging.info(f" {agent_id} {basic_maneuver_name} {phase}")

        # 转换为导航指令
        altitude_cmd_id = 3  # 默认保持高度
        heading_cmd_id = 4  # 默认保持航向
        velocity_cmd_id = 3  # 默认保持速度

        # 航向控制
        if target_heading is not None:
            current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
            heading_diff = target_heading - current_heading

            # 规范化角度差
            while heading_diff > 180:
                heading_diff -= 360
            while heading_diff < -180:
                heading_diff += 360

            if abs(heading_diff) < 2.0:
                heading_cmd_id = 4  # 保持航向
            else:
                heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))

        # 高度控制
        if target_altitude is not None:
            current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
            altitude_diff = target_altitude - current_altitude
            altitude_cmd_id = self._convert_altitude_to_index(altitude_diff)

        # 速度控制
        if target_velocity is not None:
            velocity_cmd_id = self._convert_velocity_to_index(target_velocity)

        return self._use_lowlevel_policy(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)

    def _process_composite_maneuver(self, env, agent_id, current_time, initial_heading, initial_altitude):
        """处理组合机动"""
        result = self.maneuver_composer.execute_composite_maneuver(
            self.composite_maneuver_name,
            current_time,
            initial_heading,
            initial_altitude
        )

        phase, target_heading, target_altitude, target_velocity, target_roll = result

        if phase is None:
            # 机动结束，保持稳定飞行
            return self._use_lowlevel_policy(env, agent_id, 3, 4, 3)

        # 调试输出
        if env.current_step % 100 == 0:
            logging.info(f" {agent_id} {self.composite_maneuver_name} {phase}")

        # 转换为导航指令
        altitude_cmd_id = 3
        heading_cmd_id = 4
        velocity_cmd_id = 3

        # 航向控制
        if target_heading is not None:
            current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
            heading_diff = target_heading - current_heading

            while heading_diff > 180:
                heading_diff -= 360
            while heading_diff < -180:
                heading_diff += 360

            if abs(heading_diff) < 2.0:
                heading_cmd_id = 4
            else:
                heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))

        # 高度控制
        if target_altitude is not None:
            current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
            altitude_diff = target_altitude - current_altitude
            altitude_cmd_id = self._convert_altitude_to_index(altitude_diff)

        # 速度控制
        if target_velocity is not None:
            velocity_cmd_id = self._convert_velocity_to_index(target_velocity)

        return self._use_lowlevel_policy(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)

    def _process_observer_behavior(self, env, agent_id):
        """处理观察飞机的行为 """
        try:
            # 生成简单的高层指令保持稳定飞行
            altitude_cmd_id = 3  # 保持高度
            heading_cmd_id = 4  # 保持航向
            velocity_cmd_id = 3  # 保持速度

            return self._use_lowlevel_policy(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)

        except Exception as e:
            logging.error(f"{agent_id} observer behavior error: {e}")
            return np.array([0.0, 0.0, 0.0, 0.7])

    def step(self, env):
        """执行一步纯机动仿真"""
        self.step_count += 1
        current_time = env.current_step * env.time_interval

        # 准备返回数据
        obs = {}
        share_obs = {}
        rewards = {}
        dones = {}
        infos = {}

        for agent_id in env.agents.keys():
            if not env.agents[agent_id].is_alive:
                # 死亡智能体的默认值
                obs[agent_id] = np.zeros(self.obs_length)
                share_obs[agent_id] = np.zeros(self.obs_length)
                rewards[agent_id] = [-10.0]
                dones[agent_id] = [True]
                infos[agent_id] = {"agent_id": agent_id, "alive": False}
                continue

            # 1. 计算观测
            agent_obs = self.get_obs(env, agent_id)
            obs[agent_id] = agent_obs
            share_obs[agent_id] = agent_obs

            # 2. 计算奖励
            reward = self._calculate_maneuver_reward(env, agent_id, current_time)
            rewards[agent_id] = [reward]

            # 3. 检查终止条件
            done, info = self.get_termination(env, agent_id, {})
            dones[agent_id] = [done]

            # 4. 构建信息字典
            agent_info = {
                "agent_id": agent_id,
                "time": current_time,
                "step": self.step_count,
                "maneuver_type": self.maneuver_type,
                "maneuver_params": self.maneuver_params.copy(),
                "altitude": env.agents[agent_id].get_property_value(c.position_h_sl_m),
                "heading": np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad)),
                "velocity": env.agents[agent_id].get_property_value(c.velocities_u_mps),
                "reward": reward,
                "alive": True
            }
            infos[agent_id] = agent_info

            # 记录轨迹
            self._record_trajectory_data(env, agent_id, current_time)

            # 定期输出状态信息
            if env.current_step %100 == 0:
                logging.info(f"Step {env.current_step} - {agent_id}: "
                             f"Alt={agent_info['altitude']:.1f}m, "
                             f"Hdg={agent_info['heading']:.1f}°, "
                             f"Vel={agent_info['velocity']:.1f}m/s, "
                             f"Reward={reward:.3f}")

        return obs, share_obs, rewards, dones, infos

    def _calculate_maneuver_reward(self, env, agent_id, current_time) -> float:
        """计算机动奖励"""
        reward = 1.0  # 基础存活奖励

        # 高度安全检查
        altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        if altitude < 3000:
            reward -= 2.0  # 高度过低惩罚
        elif altitude > 4000:
            reward += 0.1  # 安全高度奖励

        # 如果是测试飞机，检查机动执行效果
        if agent_id == self.test_agent_id:
            reward += self._evaluate_maneuver_performance(env, agent_id, current_time)

        return np.clip(reward, -10, 10)

    def _evaluate_maneuver_performance(self, env, agent_id, current_time) -> float:
        """评估机动执行效果"""
        try:
            if self.maneuver_type in ["crank", "beam", "notch"]:
                # 大机动评估
                return self._evaluate_large_maneuver_performance(env, agent_id, current_time)
            elif self.maneuver_type == "basic":
                # 基础机动评估
                return self._evaluate_basic_maneuver_performance(env, agent_id, current_time)
            elif self.maneuver_type == "composite":
                # 组合机动评估
                return self._evaluate_composite_maneuver_performance(env, agent_id, current_time)

            return 0.0

        except Exception as e:
            logging.error(f"机动性能评估错误: {e}")
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

                    # 航向跟踪奖励
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
            logging.error(f"❌ 大机动性能评估错误: {e}")
            return 0.0

    def _evaluate_basic_maneuver_performance(self, env, agent_id, current_time) -> float:
        """评估基础机动性能"""
        try:
            basic_maneuver_name = self.maneuver_params.get("basic_maneuver_name", "level_flight")

            # 根据基础机动类型评估
            if basic_maneuver_name in ["turn", "turn_pull_up", "turn_dive"]:
                # 转弯类机动：评估航向跟踪
                initial_heading = self.initial_heading[agent_id]
                if basic_maneuver_name == "turn":
                    result = BasicManeuvers.turn(current_time, initial_heading,
                                                 self.basic_maneuver_params["turn_angle"],
                                                 self.basic_maneuver_params["turn_rate"])
                elif basic_maneuver_name == "turn_pull_up":
                    result = BasicManeuvers.turn_pull_up(current_time, initial_heading,
                                                         self.basic_maneuver_params["turn_angle"],
                                                         self.basic_maneuver_params["turn_rate"],
                                                         self.basic_maneuver_params["altitude_change"])
                elif basic_maneuver_name == "turn_dive":
                    result = BasicManeuvers.turn_dive(current_time, initial_heading,
                                                      self.basic_maneuver_params["turn_angle"],
                                                      self.basic_maneuver_params["turn_rate"],
                                                      self.basic_maneuver_params["altitude_change"])

                if result[0] is not None:
                    phase, target_heading, target_altitude, target_velocity, target_roll = result
                    current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))

                    heading_error = abs(current_heading - target_heading)
                    if heading_error > 180:
                        heading_error = 360 - heading_error

                    if heading_error < 5:
                        return 0.3
                    elif heading_error < 15:
                        return 0.1
                    else:
                        return -0.05

            elif basic_maneuver_name in ["pull_up", "dive"]:
                # 高度变化机动：评估高度跟踪
                current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
                initial_altitude = self.initial_altitude[agent_id]

                if basic_maneuver_name == "pull_up":
                    expected_altitude = initial_altitude + self.basic_maneuver_params["altitude_change"]
                else:
                    expected_altitude = initial_altitude - self.basic_maneuver_params["altitude_change"]

                altitude_error = abs(current_altitude - expected_altitude)
                if altitude_error < 100:
                    return 0.3
                elif altitude_error < 300:
                    return 0.1
                else:
                    return -0.05

            elif basic_maneuver_name in ["accelerate", "decelerate"]:
                # 速度变化机动：评估速度跟踪
                current_velocity = env.agents[agent_id].get_property_value(c.velocities_u_mps)
                initial_velocity = 800.0  # 假设初始速度

                if basic_maneuver_name == "accelerate":
                    expected_velocity = initial_velocity + self.basic_maneuver_params["velocity_change"]
                else:
                    expected_velocity = initial_velocity - self.basic_maneuver_params["velocity_change"]

                velocity_error = abs(current_velocity - expected_velocity)
                if velocity_error < 20:
                    return 0.3
                elif velocity_error < 50:
                    return 0.1
                else:
                    return -0.05

            return 0.0

        except Exception as e:
            logging.error(f" 基础机动性能评估错误: {e}")
            return 0.0

    def _evaluate_composite_maneuver_performance(self, env, agent_id, current_time) -> float:
        """评估组合机动性能"""
        try:
            # 组合机动评估：综合评估航向、高度、速度跟踪
            current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
            current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
            current_velocity = env.agents[agent_id].get_property_value(c.velocities_u_mps)

            # 基础奖励
            reward = 0.0

            # 高度安全奖励
            if 3000 <= current_altitude <= 4000:
                reward += 0.1

            # 速度稳定奖励
            if 700 <= current_velocity <= 900:
                reward += 0.1

            return reward

        except Exception as e:
            logging.error(f" 组合机动性能评估错误: {e}")
            return 0.0

    def _record_trajectory_data(self, env, agent_id, current_time):
        """记录轨迹数据"""
        if agent_id not in self.trajectory_data:
            self.trajectory_data[agent_id] = []

        # 获取当前状态
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
        """高度指令转索引 - 复用你的成功实现"""
        altitude_values = np.array([-1000, -500, -200, 0, 200, 500, 1000])
        distances = np.abs(altitude_values - altitude_cmd)
        return np.argmin(distances)

    def _convert_heading_to_index(self, heading_cmd):
        """航向指令转索引 - 复用你的成功实现"""
        heading_cmd = np.clip(heading_cmd, -np.pi, np.pi)
        heading_values = np.array(
            [-np.pi, -np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2, np.pi])
        distances = np.abs(heading_values - heading_cmd)
        return np.argmin(distances)

    def _convert_velocity_to_index(self, velocity_offset):
        """速度偏移转索引 - 复用你的成功实现"""
        velocity_values = np.array([-150, -100, -50, 0, 50, 100, 150])
        distances = np.abs(velocity_values - velocity_offset)
        return np.argmin(distances)

    def _use_lowlevel_policy(self, env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id):
        """使用低级策略网络 """

        if self.my_lowlevel_policy is None:
            return self._direct_control_mapping(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)

        try:
            # 构建输入 - 完全按照你的成功格式
            raw_obs = self.get_obs(env, agent_id)
            input_obs = np.zeros(12)

            input_obs[0] = self.norm_delta_altitude[altitude_cmd_id]
            input_obs[1] = self.norm_delta_heading[heading_cmd_id]
            input_obs[2] = self.norm_delta_velocity[velocity_cmd_id]
            input_obs[3:12] = raw_obs[:9]

            input_obs = np.nan_to_num(input_obs, nan=0.0)
            input_obs = np.expand_dims(input_obs, axis=0)

            if agent_id not in self._inner_rnn_states:
                self._inner_rnn_states[agent_id] = np.zeros((1, 1, 128))

            # 调用策略
            _action, _rnn_states = self.my_lowlevel_policy(
                torch.FloatTensor(input_obs),
                torch.FloatTensor(self._inner_rnn_states[agent_id])
            )
            action_output = _action.detach().cpu().numpy().squeeze(0)
            self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()

            # 转换为控制指令
            norm_act = np.zeros(4)
            norm_act[0] = action_output[0] / 20 - 1.
            norm_act[1] = action_output[1] / 20 - 1.
            norm_act[2] = action_output[2] / 20 - 1.
            norm_act[3] = action_output[3] / 58 + 0.4

            # 安全限制
            current_alt = env.agents[agent_id].get_position()[2]
            if current_alt < 1000:
                norm_act[1] = max(norm_act[1], 0.0)
                norm_act[3] = max(norm_act[3], 0.8)

            return norm_act

        except Exception as e:
            logging.error(f"Lowlevel policy error: {e}")
            return self._direct_control_mapping(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)

    def _direct_control_mapping(self, env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id):
        """直接控制映射 - 应急备用"""
        try:
            # 简单的PID风格控制
            current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
            current_altitude = env.agents[agent_id].get_position()[2]

            # 基础控制
            aileron = 0.0
            elevator = 0.0
            rudder = 0.0
            throttle = 0.7

            # 高度控制
            target_altitude_change = [-1000, -500, -200, 0, 200, 500, 1000][altitude_cmd_id]
            if target_altitude_change > 0:
                elevator = 0.1
                throttle = 0.8
            elif target_altitude_change < 0:
                elevator = -0.1
                throttle = 0.6

            # 航向控制
            target_heading_change = np.rad2deg(self.norm_delta_heading[heading_cmd_id])
            if target_heading_change > 5:
                aileron = 0.2
                rudder = 0.1
            elif target_heading_change < -5:
                aileron = -0.2
                rudder = -0.1

            # 安全限制
            if current_altitude < 1000:
                elevator = max(elevator, 0.1)
                throttle = max(throttle, 0.8)

            return np.array([aileron, elevator, rudder, throttle])

        except Exception as e:
            logging.error(f"Direct control mapping error: {e}")
            return np.array([0.0, 0.0, 0.0, 0.7])

    def set_crank_params(self, angle_deg=45.0, turn_rate_deg_per_sec=3.0, hold_time_sec=20.0):
        """设置Crank机动参数"""
        self.maneuver_params = {
            "crank_angle_deg": angle_deg,
            "turn_rate_deg_per_sec": turn_rate_deg_per_sec,
            "hold_time_sec": hold_time_sec
        }
        logging.info(f" Crank参数更新: 角度={angle_deg}°, 转弯率={turn_rate_deg_per_sec}°/s, 保持时间={hold_time_sec}s")

    def set_beam_params(self, angle_deg=90.0, turn_rate_deg_per_sec=5.0, hold_time_sec=15.0):
        """设置Beam机动参数"""
        self.maneuver_params = {
            "beam_angle_deg": angle_deg,
            "turn_rate_deg_per_sec": turn_rate_deg_per_sec,
            "hold_time_sec": hold_time_sec
        }
        logging.info(f" Beam参数更新: 角度={angle_deg}°, 转弯率={turn_rate_deg_per_sec}°/s, 保持时间={hold_time_sec}s")

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
        logging.info(f" Notch参数更新: 角度={angle_deg}°, 转弯率={turn_rate_deg_per_sec}°/s, "
                     f"下降率={descent_rate_ft_per_sec}ft/s, 下降时间={descent_time_sec}s, 保持时间={hold_time_sec}s")

    def set_basic_maneuver(self, maneuver_name="level_flight", **kwargs):
        """设置基础机动"""
        self.maneuver_type = "basic"
        self.maneuver_params["basic_maneuver_name"] = maneuver_name

        # 更新基础机动参数
        for key, value in kwargs.items():
            if key in self.basic_maneuver_params:
                self.basic_maneuver_params[key] = value

        logging.info(f" 基础机动设置: {maneuver_name}, 参数: {self.basic_maneuver_params}")

    def set_composite_maneuver(self, maneuver_name="escape"):
        """设置组合机动"""
        self.maneuver_type = "composite"
        self.composite_maneuver_name = maneuver_name

        if maneuver_name in self.maneuver_composer.composite_maneuvers:
            composite = self.maneuver_composer.composite_maneuvers[maneuver_name]
            logging.info(f" 组合机动设置: {maneuver_name} - {composite.description}")
        else:
            logging.warning(f"️ 未知的组合机动: {maneuver_name}")

    def set_maneuver_type(self, maneuver_type="crank"):
        """设置机动类型"""
        self.maneuver_type = maneuver_type
        logging.info(f"机动类型设置为: {maneuver_type}")

    def get_available_maneuvers(self):
        """获取可用的机动列表"""
        maneuvers = {
            "large_maneuvers": ["crank", "beam", "notch"],
            "basic_maneuvers": [
                "level_flight", "accelerate", "decelerate", "turn",
                "pull_up", "dive", "diagonal_flight", "roll",
                "turn_pull_up", "turn_dive"
            ],
            "composite_maneuvers": list(self.maneuver_composer.composite_maneuvers.keys())
        }
        return maneuvers

    def export_trajectory_data(self, filepath=None):
        """导出轨迹数据"""
        if not self.trajectory_data:
            logging.warning("️ 没有轨迹数据可导出")
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
            logging.error(f" 轨迹数据导出失败: {e}")
            return None