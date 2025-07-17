# envs/JSBSim/tasks/pure_maneuver_task.py
import logging
import numpy as np
import torch
from typing import Dict, Any

from gymnasium import spaces

from .multiplecombat_task import MultipleCombatTask
from .pure_maneuvers import PureManeuvers
from ..core.catalog import Catalog as c
from ..model.baseline_actor import BaselineActor
from ..utils.utils import get_root_dir


class PureManeuverTask(MultipleCombatTask):
    """机动测试任务"""

    def __init__(self, config):
        super().__init__(config)
        self.config = config  # 保存配置引用
        #机动参数暴露
        self.maneuver_type = "crank"
        self.maneuver_params = {
            # Crank机动参数
            "crank_angle_deg": 45.0,
            "turn_rate_deg_per_sec": 3.0,
            "hold_time_sec": 40.0,

            # Beam机动参数 - 横向态势
            "beam_angle_deg": 90.0,  # 90度横向
            "beam_turn_rate_deg_per_sec": 5.0,  # 快速转向
            "beam_hold_time_sec": 40.0,

            # Notch机动参数 - 地面杂波隐蔽
            "notch_angle_deg": 90.0,  # 横向转弯角度
            "notch_turn_rate_deg_per_sec": 4.0,
            "notch_descent_rate_ft_per_sec": 100.0,  # 下降率
            "notch_descent_time_sec": 20.0,  # 下降时间
            "notch_hold_time_sec": 40.0
        }

        # 测试配置
        self.test_agent_id = "A0100"
        self.observer_agent_id = "B0100"
        self.test_start_time = 0.0
        self.initial_heading = {}
        self.trajectory_data = {}

        # 三层架构
        self.my_lowlevel_policy = BaselineActor()
        self.enemy_baseline_policy = BaselineActor()
        self._inner_rnn_states = {}
        self._enemy_rnn_states = {}

        # 参数映射
        self.norm_delta_altitude = np.array([-1000, -500, -200, 0, 200, 500, 1000]) / 1000.0
        self.norm_delta_heading = np.array(
            [-np.pi, -np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2, np.pi])
        self.norm_delta_velocity = np.array([-150, -100, -50, 0, 50, 100, 150]) / 100.0

        # 加载模型
        self._load_baseline_models()

    @property
    def num_agents(self) -> int:
        return len(self.config.aircraft_configs)

    def load_action_space(self):
        """第二层高层控制"""
        self.action_space = spaces.MultiDiscrete([7, 9, 7])  # [altitude_cmd_id, heading_cmd_id, velocity_cmd_id]

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

            # 敌方使用相同的策略
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

        # 重置测试状态
        self.step_count = 0
        self.test_start_time = 0.0
        self.initial_heading.clear()
        self.trajectory_data.clear()

        # 重置RNN状态
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        self._enemy_rnn_states = {agent_id: torch.zeros(1, 1, 128) for agent_id in env.agents.keys()}

    def normalize_action(self, env, agent_id, action):
        """动作归一化"""
        if agent_id not in env.agents or not env.agents[agent_id].is_alive:
            return np.array([0.0, 0.0, 0.0, 0.7])

        current_time = env.current_step * env.time_interval

        # 记录初始航向
        if agent_id not in self.initial_heading:
            current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_rad)
            self.initial_heading[agent_id] = np.rad2deg(current_heading)
            logging.info(f"{agent_id} initial heading: {self.initial_heading[agent_id]:.1f}°")

        # 区分测试飞机和观察飞机
        if agent_id == self.test_agent_id:
            # 测试飞机：执行纯机动
            return self._process_test_maneuver_action(env, agent_id, current_time)
        else:
            # 观察飞机：使用baseline行为
            return self._process_observer_behavior(env, agent_id)

    def _process_test_maneuver_action(self, env, agent_id, current_time):
        """处理测试飞机的机动动作 - 使用纯机动函数"""
        try:
            # 第一层：高层机动函数调用
            initial_heading = self.initial_heading[agent_id]

            if self.maneuver_type == "crank":
                maneuver_result = PureManeuvers.crank_maneuver(
                    current_time,
                    initial_heading,
                    self.maneuver_params["crank_angle_deg"],
                    self.maneuver_params["turn_rate_deg_per_sec"],
                    self.maneuver_params["hold_time_sec"]
                )
                if maneuver_result:
                    phase, target_heading, target_roll = maneuver_result
                    target_altitude = None  # 保持当前高度

            elif self.maneuver_type == "beam":
                maneuver_result = PureManeuvers.beam_maneuver(
                    current_time,
                    initial_heading,
                    self.maneuver_params["beam_angle_deg"],
                    self.maneuver_params["beam_turn_rate_deg_per_sec"],
                    self.maneuver_params["beam_hold_time_sec"]
                )
                if maneuver_result:
                    phase, target_heading, target_roll = maneuver_result
                    target_altitude = None  # 保持当前高度

            elif self.maneuver_type == "notch":
                # 获取当前高度
                current_altitude_ft = env.agents[agent_id].get_property_value(c.position_h_sl_ft)

                maneuver_result = PureManeuvers.notch_maneuver(
                    current_time,
                    initial_heading,
                    current_altitude_ft,
                    self.maneuver_params["notch_angle_deg"],
                    self.maneuver_params["notch_turn_rate_deg_per_sec"],
                    self.maneuver_params["notch_descent_rate_ft_per_sec"],
                    self.maneuver_params["notch_descent_time_sec"],
                    self.maneuver_params["notch_hold_time_sec"]
                )
                # Notch返回4个值，包含高度
                if maneuver_result:
                    phase, target_heading, target_altitude, target_roll = maneuver_result
            else:
                maneuver_result = None

            if maneuver_result is None:
                altitude_cmd_id = 3  # 保持高度
                heading_cmd_id = 4  # 保持航向
                velocity_cmd_id = 3  # 保持速度
            else:
                # 调试输出
                if env.current_step % 100 == 0:
                    alt_info = f", Alt={target_altitude:.0f}ft" if target_altitude else ""
                    logging.info(
                        f" {agent_id} {self.maneuver_type.upper()} {phase}: Target={target_heading:.1f}°{alt_info}")

                # 处理航向指令
                current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
                heading_diff = target_heading - current_heading
                while heading_diff > 180:
                    heading_diff -= 360
                while heading_diff < -180:
                    heading_diff += 360

                if abs(heading_diff) < 2.0:
                    heading_cmd_id = 4  # 保持航向
                else:
                    heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))

                # 处理高度指令
                if target_altitude is None:
                    altitude_cmd_id = 3  # 保持当前高度
                else:
                    current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_ft)
                    altitude_diff = target_altitude - current_altitude
                    altitude_cmd_id = self._convert_altitude_to_index(altitude_diff)

                velocity_cmd_id = 3  # 保持速度

            return self._use_lowlevel_policy(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)

        except Exception as e:
            logging.error(f" {agent_id} maneuver execution error: {e}")
            return self._direct_control_mapping(env, agent_id, 3, 4, 3)

    def _process_observer_behavior(self, env, agent_id):
        """处理观察飞机的行为 - 敌方baseline策略"""
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
            if env.current_step % 50 == 0:
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
            logging.error(f"机动性能评估错误: {e}")
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


    def _convert_altitude_to_index(self, altitude_diff_ft):
        """将高度差转换为指令索引"""
        altitude_diff_m = altitude_diff_ft * 0.3048  # 转换为米

        # 映射到规范化的高度指令
        if altitude_diff_m > 500:
            return 6  # 大幅上升
        elif altitude_diff_m > 200:
            return 5  # 中等上升
        elif altitude_diff_m > 50:
            return 4  # 小幅上升
        elif altitude_diff_m > -50:
            return 3  # 保持高度
        elif altitude_diff_m > -200:
            return 2  # 小幅下降
        elif altitude_diff_m > -500:
            return 1  # 中等下降
        else:
            return 0  # 大幅下降

    def _convert_heading_to_index(self, heading_cmd):
        """航向指令转索引"""
        heading_cmd = np.clip(heading_cmd, -np.pi, np.pi)
        heading_values = np.array(
            [-np.pi, -np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2, np.pi])
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


    def set_crank_params(self, angle_deg=60.0, turn_rate_deg_per_sec=3.0, hold_time_sec=30.0):
        """设置Crank机动参数"""
        self.maneuver_params = {
            "crank_angle_deg": angle_deg,
            "turn_rate_deg_per_sec": turn_rate_deg_per_sec,
            "hold_time_sec": hold_time_sec
        }
        logging.info(f"Crank参数更新: 角度={angle_deg}°, 转弯率={turn_rate_deg_per_sec}°/s, 保持时间={hold_time_sec}s")

    def set_beam_params(self, angle_deg=90.0, turn_rate_deg_per_sec=5.0, hold_time_sec=40.0):
        """设置Beam机动参数"""
        self.maneuver_params["beam_angle_deg"] = angle_deg
        self.maneuver_params["beam_turn_rate_deg_per_sec"] = turn_rate_deg_per_sec
        self.maneuver_params["beam_hold_time_sec"] = hold_time_sec
        logging.info(f"Beam参数设置: {angle_deg}°横向, {turn_rate_deg_per_sec}°/s, {hold_time_sec}s")

    def set_notch_params(self, angle_deg=90.0, turn_rate_deg_per_sec=4.0,
                        descent_rate_ft_per_sec=70.0,
                        descent_time_sec=5.0,
                        hold_time_sec=20.0):
        """设置Notch机动参数"""
        self.maneuver_params["notch_angle_deg"] = angle_deg
        self.maneuver_params["notch_turn_rate_deg_per_sec"] = turn_rate_deg_per_sec
        self.maneuver_params["notch_descent_rate_ft_per_sec"] = descent_rate_ft_per_sec
        self.maneuver_params["notch_descent_time_sec"] = descent_time_sec
        self.maneuver_params["notch_hold_time_sec"] = hold_time_sec
        logging.info(f"Notch参数设置: {angle_deg}°转向, 安全下降{descent_rate_ft_per_sec}ft/s x {descent_time_sec}s, 保持{hold_time_sec}s")




    def set_maneuver_type(self, maneuver_type="crank"):
        """设置机动类型"""
        self.maneuver_type = maneuver_type
        logging.info(f"机动类型设置为: {maneuver_type}")