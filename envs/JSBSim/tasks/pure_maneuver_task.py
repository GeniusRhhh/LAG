# envs/JSBSim/tasks/pure_maneuver_task.py
import logging
import numpy as np
import torch
from typing import Dict, Any
from .multiplecombat_task import MultipleCombatTask
from .pure_maneuvers import PureManeuvers
from ..core.catalog import Catalog as c
from ..model.baseline_actor import BaselineActor
from ..utils.utils import get_root_dir


class PureManeuverTask(MultipleCombatTask):
    """纯机动测试任务 - 老师要求的函数封装"""

    def __init__(self, config):
        super().__init__(config)

        # 【老师要求】：机动参数暴露，可以轻松修改
        self.maneuver_type = "crank"
        self.maneuver_params = {
            "crank_angle_deg": 60.0,  # 可改为60度等
            "turn_rate_deg_per_sec": 3.0,  # 转弯率
            "hold_time_sec": 20.0  # 保持时间
        }

        # 测试配置
        self.test_agent_id = "A0100"  # 执行机动的飞机
        self.observer_agent_id = "B0100"  # 观察飞机
        self.test_start_time = 0.0
        self.initial_heading = {}
        self.trajectory_data = {}

        # 【核心】：完全复用你的成功三层架构
        self.my_lowlevel_policy = BaselineActor()
        self.enemy_baseline_policy = BaselineActor()
        self._inner_rnn_states = {}
        self._enemy_rnn_states = {}

        # 【关键】：完全复用你的成功参数映射
        self.norm_delta_altitude = np.array([-1000, -500, -200, 0, 200, 500, 1000]) / 1000.0
        self.norm_delta_heading = np.array(
            [-np.pi, -np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2, np.pi])
        self.norm_delta_velocity = np.array([-150, -100, -50, 0, 50, 100, 150]) / 100.0

        # 加载模型
        self._load_baseline_models()

        logging.info("✅ PureManeuverTask initialized - 老师要求的纯函数机动系统")

    def _load_baseline_models(self):
        """加载你的成功baseline模型"""
        try:
            # 【修正】：使用正确的模型路径
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

            logging.info("✅ 成功加载你的baseline模型")

        except Exception as e:
            logging.error(f"❌ 加载baseline模型失败: {e}")
            self.my_lowlevel_policy = None
            self.enemy_baseline_policy = None

    def reset(self, env):
        """重置任务状态"""
        # 调用基类重置（跳过MultipleCombatTask的复杂重置）
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

        logging.info(f"📍 PureManeuverTask reset - testing: {self.maneuver_type}")
        logging.info(f"🎯 Maneuver params: {self.maneuver_params}")

    def normalize_action(self, env, agent_id, action):
        """动作归一化 - 完全复用你的成功架构"""
        if agent_id not in env.agents or not env.agents[agent_id].is_alive:
            return np.array([0.0, 0.0, 0.0, 0.7])

        current_time = env.current_step * env.time_interval

        # 记录初始航向（只记录一次）
        if agent_id not in self.initial_heading:
            current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_rad)
            self.initial_heading[agent_id] = np.rad2deg(current_heading)
            logging.info(f"📍 {agent_id} initial heading: {self.initial_heading[agent_id]:.1f}°")

        # 区分测试飞机和观察飞机
        if agent_id == self.test_agent_id:
            # 测试飞机：执行纯机动
            return self._process_test_maneuver_action(env, agent_id, current_time)
        else:
            # 观察飞机：使用智能baseline行为
            return self._process_observer_behavior(env, agent_id)

    # envs/JSBSim/tasks/pure_maneuver_task.py

    def _process_test_maneuver_action(self, env, agent_id, current_time):
        """处理测试飞机的机动动作 - 使用纯机动函数"""
        try:
            # 【第一层：高层机动函数调用】
            initial_heading = self.initial_heading[agent_id]

            if self.maneuver_type == "crank":
                maneuver_result = PureManeuvers.crank_maneuver(
                    current_time,
                    initial_heading,
                    self.maneuver_params["crank_angle_deg"],
                    self.maneuver_params["turn_rate_deg_per_sec"],
                    self.maneuver_params["hold_time_sec"]
                )
            else:
                maneuver_result = None

            if maneuver_result is None:
                # 机动结束，保持稳定飞行
                altitude_cmd_id = 3  # 保持高度
                heading_cmd_id = 4  # 保持航向
                velocity_cmd_id = 3  # 保持速度
            else:
                phase, target_heading, target_roll = maneuver_result

                # 调试输出
                if env.current_step % 25 == 0:
                    logging.info(
                        f"🎯 {agent_id} Crank {phase}: Target={target_heading:.1f}°, Current={np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad)):.1f}°")

                # 【第二层：转换为中层导航指令】
                current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))

                # 计算航向差 - 修正角度计算
                heading_diff = target_heading - current_heading

                # 规范化角度差到 [-180, 180] 范围
                while heading_diff > 180:
                    heading_diff -= 360
                while heading_diff < -180:
                    heading_diff += 360

                # 如果航向差很小，就保持当前航向
                if abs(heading_diff) < 2.0:  # 2度容差
                    heading_cmd_id = 4  # 保持航向
                else:
                    # 转换为导航指令索引
                    heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))

                altitude_cmd_id = 3  # 保持高度
                velocity_cmd_id = 3  # 保持速度

            # 【第三层：调用你的成功底层策略网络】
            return self._use_lowlevel_policy(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)

        except Exception as e:
            logging.error(f"❌ {agent_id} maneuver execution error: {e}")
            return self._direct_control_mapping(env, agent_id, 3, 4, 3)

    def _process_observer_behavior(self, env, agent_id):
        """处理观察飞机的行为 - 使用你的成功敌方baseline策略"""
        try:
            # 生成简单的高层指令保持稳定飞行
            altitude_cmd_id = 3  # 保持高度
            heading_cmd_id = 4  # 保持航向
            velocity_cmd_id = 3  # 保持速度

            return self._use_lowlevel_policy(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)

        except Exception as e:
            logging.error(f"❌ {agent_id} observer behavior error: {e}")
            return np.array([0.0, 0.0, 0.0, 0.7])

    # ================================================================================
    # 【关键】：重写step方法 - 这是问题的根源！
    # ================================================================================

    def step(self, env):
        """执行一步纯机动仿真 - 简化版不需要复杂的战术模板"""
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
            logging.error(f"❌ 机动性能评估错误: {e}")
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

    # ================================================================================
    # 【关键方法】：完全复用你的成功实现
    # ================================================================================

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
        """使用低级策略网络 - 完全复用你的成功实现"""

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

            # 转换为控制指令 - 完全按照你的成功格式
            norm_act = np.zeros(4)
            norm_act[0] = action_output[0] / 20 - 1.
            norm_act[1] = action_output[1] / 20 - 1.
            norm_act[2] = action_output[2] / 20 - 1.
            norm_act[3] = action_output[3] / 58 + 0.4

            # 安全限制 - 复用你的成功逻辑
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

    # ================================================================================
    # 【老师要求】：参数控制接口
    # ================================================================================

    def set_crank_params(self, angle_deg=45.0, turn_rate_deg_per_sec=3.0, hold_time_sec=20.0):
        """设置Crank机动参数 - 老师要求的参数暴露"""
        self.maneuver_params = {
            "crank_angle_deg": angle_deg,
            "turn_rate_deg_per_sec": turn_rate_deg_per_sec,
            "hold_time_sec": hold_time_sec
        }
        logging.info(f"🎯 Crank参数更新: 角度={angle_deg}°, 转弯率={turn_rate_deg_per_sec}°/s, 保持时间={hold_time_sec}s")

    def set_maneuver_type(self, maneuver_type="crank"):
        """设置机动类型"""
        self.maneuver_type = maneuver_type
        logging.info(f"🎯 机动类型设置为: {maneuver_type}")