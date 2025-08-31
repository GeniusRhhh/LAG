#!/usr/bin/env python3
"""
基础动作数据生成框架
基于现有的基础动作系统，生成大量标注数据用于机器学习训练
支持双重输出：ACMI文件 + CSV数据表
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import random
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.tasks.pure_maneuver_task import PureManeuverTask
from envs.JSBSim.tasks.pure_maneuvers import BasicManeuvers
from envs.JSBSim.core.catalog import Catalog as c


@dataclass
class ActionConfig:
    """基础动作配置"""
    name: str                    # 动作名称
    function_name: str          # 对应的函数名
    param_ranges: Dict[str, Tuple[float, float]]  # 参数范围
    duration_range: Tuple[float, float]           # 持续时间范围
    samples_count: int = 100    # 生成样本数量


class BasicActionDataGenerator:
    """基础动作数据生成器"""
    
    def __init__(self, output_dir: str = "scripts/drag_shoot_2v2/basic_action_data"):
        self.output_dir = output_dir
        self.setup_logging()
        self.setup_output_directories()
        
        # 11种标准基础动作配置
        self.action_configs = self._setup_standard_actions()
        
        # 数据记录
        self.trajectory_data = []
        self.current_action_type = ""
        
    def setup_logging(self):
        """设置日志"""
        log_file = os.path.join(self.output_dir, f"data_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        os.makedirs(self.output_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
    def setup_output_directories(self):
        """设置输出目录 - 为每种基础动作创建独立子目录"""
        # 创建基础输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 为每种基础动作创建独立的子目录
        self.action_dirs = {}
        action_names = [
            "level_flight", "accelerate", "decelerate", "climb", "dive",
            "turn", "Crank", "tactical_crank", "tactical_climb",
            "tactical_dive", "notch_back", "short_skate"
        ]

        for action_name in action_names:
            action_dir = os.path.join(self.output_dir, action_name)
            os.makedirs(action_dir, exist_ok=True)
            self.action_dirs[action_name] = action_dir

        logging.info(f"创建了{len(action_names)}个动作子目录")
        
    def _setup_standard_actions(self) -> Dict[str, ActionConfig]:
        """设置11种标准基础动作配置"""
        configs = {}

        # 1. 平飞 (level_flight)
        configs["level_flight"] = ActionConfig(
            name="level_flight",
            function_name="level_flight",
            param_ranges={
                "duration": (15.0, 25.0),  # 基础动作：15-25秒
                "current_velocity": (200.0, 300.0)
            },
            duration_range=(15.0, 25.0),
            samples_count=10  # 每种动作10个样本
        )

        # 2. 加速 (accelerate)
        configs["accelerate"] = ActionConfig(
            name="accelerate",
            function_name="accelerate",
            param_ranges={
                "velocity_increase": (40.0, 100.0),
                "duration": (15.0, 25.0)  # 基础动作：15-25秒
            },
            duration_range=(15.0, 25.0),
            samples_count=10  # 每种动作10个样本
        )

        # 3. 减速 (decelerate)
        configs["decelerate"] = ActionConfig(
            name="decelerate",
            function_name="decelerate",
            param_ranges={
                "velocity_decrease": (40.0, 100.0),
                "duration": (15.0, 25.0)  # 基础动作：15-25秒
            },
            duration_range=(15.0, 25.0),
            samples_count=10  # 每种动作10个样本
        )

        # 4. 爬升 (climb/pull_up)
        configs["climb"] = ActionConfig(
            name="climb",
            function_name="pull_up",
            param_ranges={
                "altitude_gain": (1000.0, 2500.0),
                "duration": (15.0, 25.0)  # 基础动作：15-25秒
            },
            duration_range=(15.0, 25.0),
            samples_count=10  # 每种动作10个样本
        )

        # 5. 下降 (dive)
        configs["dive"] = ActionConfig(
            name="dive",
            function_name="dive",
            param_ranges={
                "altitude_loss": (1000.0, 2500.0),
                "duration": (15.0, 25.0),  # 基础动作：15-25秒
                "min_altitude": (2000.0, 3500.0)
            },
            duration_range=(15.0, 25.0),
            samples_count=10  # 每种动作10个样本
        )

        # 6. 转弯 (turn) - 包含左转和右转
        configs["turn"] = ActionConfig(
            name="turn",
            function_name="turn",
            param_ranges={
                "turn_angle": (-90.0, 90.0),  # 负值左转，正值右转
                "turn_rate": (2.5, 6.0)
            },
            duration_range=(15.0, 25.0),  # 基础动作：15-25秒
            samples_count=10  # 每种动作10个样本
        )

        # 7. Crank (标准规避机动)
        configs["Crank"] = ActionConfig(
            name="Crank",
            function_name="turn",
            param_ranges={
                "turn_angle": (-70.0, 70.0),   # 包含左右Crank
                "turn_rate": (2.5, 6.0)
            },
            duration_range=(20.0, 35.0),  # 战术动作：20-35秒
            samples_count=10  # 每种动作10个样本
        )

        # 8. 战术Crank (tactical_crank)
        configs["tactical_crank"] = ActionConfig(
            name="tactical_crank",
            function_name="turn",  # 使用基础turn函数
            param_ranges={
                "turn_angle": (-80.0, 80.0),   # 战术Crank更大角度
                "turn_rate": (3.0, 6.0)
            },
            duration_range=(20.0, 35.0),  # 战术动作：20-35秒
            samples_count=10  # 每种动作10个样本
        )

        # 9. 战术爬升 (tactical_climb)
        configs["tactical_climb"] = ActionConfig(
            name="tactical_climb",
            function_name="pull_up",
            param_ranges={
                "altitude_gain": (1500.0, 3000.0),  # 战术爬升更大高度
                "duration": (20.0, 35.0)  # 战术动作：20-35秒
            },
            duration_range=(20.0, 35.0),
            samples_count=10  # 每种动作10个样本
        )

        # 10. 战术下降 (tactical_dive)
        configs["tactical_dive"] = ActionConfig(
            name="tactical_dive",
            function_name="dive",
            param_ranges={
                "altitude_loss": (1500.0, 3000.0),  # 战术下降更大高度
                "duration": (20.0, 35.0),  # 战术动作：20-35秒
                "min_altitude": (2000.0, 3500.0)
            },
            duration_range=(20.0, 35.0),
            samples_count=10  # 每种动作10个样本
        )

        # 11. Notch back (后撤规避)
        configs["notch_back"] = ActionConfig(
            name="notch_back",
            function_name="turn",  # 使用基础turn函数
            param_ranges={
                "turn_angle": (-110.0, 110.0),  # 包含左右Notch
                "turn_rate": (2.0, 5.0),
                "duration": (20.0, 35.0)  # 战术动作：20-35秒
            },
            duration_range=(20.0, 35.0),
            samples_count=10  # 每种动作10个样本
        )

        # 12. Short skate (短距离规避)
        configs["short_skate"] = ActionConfig(
            name="short_skate",
            function_name="turn",  # 使用基础turn函数
            param_ranges={
                "turn_angle": (-45.0, 45.0),  # 短距离规避角度较小
                "turn_rate": (3.0, 7.0),  # 转弯速率较快
                "duration": (20.0, 35.0)  # 战术动作：20-35秒
            },
            duration_range=(20.0, 35.0),
            samples_count=10  # 每种动作10个样本
        )

        return configs
        
    def generate_random_params(self, config: ActionConfig) -> Dict[str, float]:
        """为指定动作生成随机参数"""
        params = {}
        for param_name, (min_val, max_val) in config.param_ranges.items():
            params[param_name] = random.uniform(min_val, max_val)
        return params
        
    def generate_single_action_data(self, action_name: str, sample_index: int) -> Tuple[str, str]:
        """生成单个动作的数据，返回(ACMI文件路径, CSV文件路径)"""
        config = self.action_configs[action_name]
        params = self.generate_random_params(config)

        # 确定动作标注名称（用于CSV数据标注）
        actual_action_name = self._determine_action_label(action_name, params)

        # 生成参数化文件名
        date_str = datetime.now().strftime("%m%d")  # 简化时间戳为MMDD格式
        param_str = self._format_filename_parameters(action_name, params)

        # 新的参数化文件命名格式
        base_filename = f"basic_maneuver_{action_name}_{param_str}_{sample_index:03d}_{date_str}"
        acmi_filename = f"{base_filename}.acmi"
        csv_filename = f"{base_filename}.csv"

        # 使用动作专用目录
        action_dir = self.action_dirs[action_name]
        acmi_filepath = os.path.join(action_dir, acmi_filename)
        csv_filepath = os.path.join(action_dir, csv_filename)

        logging.info(f"生成 {action_name} 样本 {sample_index}/{config.samples_count}")
        logging.info(f"参数: {params}")
        logging.info(f"动作标注: {actual_action_name}")
        logging.info(f"文件名: {base_filename}")

        # 保存当前动作参数供状态检测使用
        self.current_action_params = params

        # 执行仿真
        success = self._run_simulation(actual_action_name, config.function_name, params, acmi_filepath, csv_filepath)

        if success:
            return acmi_filepath, csv_filepath
        else:
            return None, None

    def _determine_action_label(self, action_name: str, params: Dict[str, Any]) -> str:
        """确定动作标注名称（用于CSV数据标注）"""
        # 对于转弯类动作，根据turn_angle确定方向性标注
        if action_name in ["Crank", "turn", "tactical_crank"] and "turn_angle" in params:
            if params["turn_angle"] < 0:
                return "左转"
            else:
                return "右转"
        elif action_name == "tactical_crank" and "crank_angle" in params:
            if params["crank_angle"] < 0:
                return "左转"
            else:
                return "右转"
        elif action_name == "notch_back" and "turn_angle" in params:
            if params["turn_angle"] < 0:
                return "左转"
            else:
                return "右转"
        else:
            # 其他动作使用原始名称
            return action_name

    def _format_filename_parameters(self, action_name: str, params: Dict[str, Any]) -> str:
        """根据动作类型和参数生成格式化的文件名参数部分"""
        param_parts = []

        if action_name in ["Crank", "turn"]:
            # 转弯类动作：{direction}_{angle}deg_{rate}dps
            direction = "left" if params.get("turn_angle", 0) < 0 else "right"
            angle = int(abs(params.get("turn_angle", 0)))
            rate = round(params.get("turn_rate", 0), 1)
            param_parts = [direction, f"{angle}deg", f"{rate}dps"]

        elif action_name == "tactical_crank":
            # 战术Crank：{direction}_{angle}deg_{rate}dps
            direction = "left" if params.get("crank_angle", 0) < 0 else "right"
            angle = int(abs(params.get("crank_angle", 0)))
            rate = round(params.get("turn_rate", 0), 1)
            param_parts = [direction, f"{angle}deg", f"{rate}dps"]

        elif action_name in ["climb", "tactical_climb"]:
            # 爬升类动作：{altitude}m_{duration}s
            altitude = int(params.get("altitude_gain", 0))
            duration = int(params.get("duration", 0))
            param_parts = [f"{altitude}m", f"{duration}s"]

        elif action_name in ["dive", "tactical_dive"]:
            # 下降类动作：{altitude}m_{duration}s
            altitude = int(params.get("altitude_loss", 0))
            duration = int(params.get("duration", 0))
            param_parts = [f"{altitude}m", f"{duration}s"]

        elif action_name == "accelerate":
            # 加速动作：{speed}mps_{duration}s
            speed = int(params.get("velocity_increase", 0))
            duration = int(params.get("duration", 0))
            param_parts = [f"{speed}mps", f"{duration}s"]

        elif action_name == "decelerate":
            # 减速动作：{speed}mps_{duration}s
            speed = int(params.get("velocity_decrease", 0))
            duration = int(params.get("duration", 0))
            param_parts = [f"{speed}mps", f"{duration}s"]

        elif action_name == "level_flight":
            # 平飞动作：{speed}mps_{duration}s
            speed = int(params.get("current_velocity", 250))
            duration = int(params.get("duration", 0))
            param_parts = [f"{speed}mps", f"{duration}s"]

        elif action_name == "notch_back":
            # Notch back：{direction}_{angle}deg_{altitude}m
            direction = "left" if params.get("turn_angle", 0) < 0 else "right"
            angle = int(abs(params.get("turn_angle", 0)))
            altitude = int(params.get("altitude_loss", 0))
            param_parts = [direction, f"{angle}deg", f"{altitude}m"]

        else:
            # 默认格式：使用第一个数值参数
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    if "angle" in key.lower():
                        param_parts.append(f"{int(abs(value))}deg")
                    elif "duration" in key.lower() or "time" in key.lower():
                        param_parts.append(f"{int(value)}s")
                    elif "altitude" in key.lower():
                        param_parts.append(f"{int(abs(value))}m")
                    elif "velocity" in key.lower() or "speed" in key.lower():
                        param_parts.append(f"{int(abs(value))}mps")
                    break

        return "_".join(param_parts) if param_parts else "default"

    def _run_simulation(self, action_name: str, function_name: str, params: Dict[str, float],
                       acmi_path: str, csv_path: str) -> bool:
        """运行单次仿真 - 添加精确时间范围控制"""
        try:
            # 创建环境
            config_name = "simple_maneuver_config"
            env = MultipleCombatEnv(config_name)
            task = env.task

            # 设置基础动作
            task.set_basic_maneuver(function_name, **params)
            self.current_action_type = action_name

            # 获取动作持续时间
            action_duration = params.get('duration', 20.0)  # 默认20秒

            # 计算仿真总时长（动作时间 + 缓冲时间）
            buffer_time = 18.0  # 18秒缓冲时间，确保复杂机动动作完全执行
            simulation_duration = action_duration + buffer_time

            logging.info(f"动作 {action_name} 持续时间: {action_duration:.1f}秒")
            logging.info(f"仿真总时长: {simulation_duration:.1f}秒 (包含{buffer_time:.1f}秒缓冲)")

            # 初始化ACMI文件
            self._initialize_acmi_file(acmi_path)

            # 重置轨迹数据
            self.trajectory_data = []

            # 运行仿真
            obs, share_obs = env.reset()
            step = 0
            done = False

            # 计算最大步数（基于仿真总时长）
            max_steps = int(simulation_duration / env.time_interval) + 50  # 额外50步保险

            # 分离ACMI渲染和CSV数据记录逻辑
            action_start_time = 0.0
            action_start_step = int(action_start_time / env.time_interval)

            # CSV数据记录范围：基于动作duration参数
            csv_end_time = action_duration  # 使用动作参数的duration时间
            csv_end_step = int(csv_end_time / env.time_interval)

            logging.info(f"ACMI渲染范围: 整个仿真时长 {simulation_duration:.1f}s")
            logging.info(f"CSV数据记录范围: 步数 {action_start_step} - {csv_end_step} (时间 {action_start_time:.1f}s - {csv_end_time:.1f}s)")
            logging.info(f"仿真最大步数: {max_steps} (基于{simulation_duration:.1f}秒总时长)")

            while not done and step < max_steps:
                # 执行动作
                action_dim = env.action_space.shape[0]
                actions = np.zeros((env.n_rollout_threads, env.num_agents, action_dim), dtype=np.float32)
                obs, share_obs, rewards, dones, infos = env.step(actions)

                current_time = step * env.time_interval

                # ACMI渲染：整个仿真过程都记录
                if step % 2 == 0:  # 每2步记录一次ACMI
                    self._write_acmi_frame(acmi_path, env)

                # CSV数据记录：仅在动作执行期间记录
                if action_start_step <= step <= csv_end_step:
                    if step % 2 == 0:  # 每2步记录一次CSV数据
                        # 调整时间为相对于动作开始的时间
                        relative_time = current_time - action_start_time
                        self._record_trajectory_data(env, relative_time)

                step += 1
                done = np.any(dones)

                # 仿真正常结束检查
                if step >= max_steps:
                    logging.info(f"仿真完成，总步数: {step}")
                    break

            # 保存CSV数据
            self._save_csv_data(csv_path)

            env.close()

            # 验证数据时间范围
            if self.trajectory_data:
                actual_start_time = min(data['Time_s'] for data in self.trajectory_data)
                actual_end_time = max(data['Time_s'] for data in self.trajectory_data)
                actual_duration = actual_end_time - actual_start_time

                # 验证CSV数据时间范围是否与duration参数一致
                duration_match = abs(actual_duration - action_duration) < 1.0  # 允许1秒误差

                logging.info(f"✅ {action_name} 仿真完成:")
                logging.info(f"  ACMI文件: 完整仿真时长 {simulation_duration:.1f}s")
                logging.info(f"  CSV数据: 时间范围 {actual_start_time:.1f}s - {actual_end_time:.1f}s")
                logging.info(f"  数据持续时间: {actual_duration:.1f}s (目标: {action_duration:.1f}s)")

                if duration_match:
                    logging.info(f"  ✅ 数据时间范围与duration参数匹配")
                else:
                    logging.warning(f"  ⚠️ 数据时间范围与duration参数不匹配 (差异: {abs(actual_duration - action_duration):.1f}s)")
            else:
                logging.warning(f"⚠️ {action_name} 没有记录到轨迹数据")

            return True
            
        except Exception as e:
            logging.error(f"❌ {action_name} 仿真失败: {e}")
            return False

    def _detect_action_completion_status(self, env, function_name: str) -> str:
        """检测动作完成状态 - 简化版本基于时间和参数"""
        try:
            current_time = env.current_step * env.time_interval

            # 获取动作参数
            if hasattr(self, 'current_action_params'):
                params = self.current_action_params
            else:
                return "UNKNOWN"

            # 基于动作类型和参数估算完成状态
            if function_name == "turn":
                # 转弯动作：基于转弯角度和转弯速率计算预期时间
                turn_angle = abs(params.get('turn_angle', 45.0))
                turn_rate = params.get('turn_rate', 3.0)
                expected_time = turn_angle / turn_rate

                if current_time >= expected_time * 0.9:  # 90%完成度
                    return "TURN_FINISHED"
                elif current_time >= expected_time * 0.5:
                    return "TURN_ADJUSTING"
                else:
                    return "TURNING"

            elif function_name == "accelerate":
                # 加速动作：基于duration参数
                duration = params.get('duration', 10.0)
                if current_time >= duration * 0.95:  # 95%完成度
                    return "ACCELERATION_FINISHED"
                else:
                    return "ACCELERATING"

            elif function_name == "level_flight":
                # 平飞动作：基于duration参数
                duration = params.get('duration', 20.0)
                if current_time >= duration * 0.95:  # 95%完成度
                    return "LEVEL_FLIGHT_FINISHED"
                else:
                    return "LEVEL_FLIGHT"

            return "UNKNOWN"

        except Exception as e:
            logging.debug(f"状态检测失败: {e}")
            return "UNKNOWN"

    def _is_action_completed(self, action_status: str, function_name: str) -> bool:
        """判断动作是否完成"""
        if action_status == "UNKNOWN":
            return False

        # 根据不同动作类型判断完成状态
        if function_name == "turn":
            return action_status == "TURN_FINISHED"
        elif function_name == "accelerate":
            return action_status in ["ACCELERATION_FINISHED", "ACCELERATE_FINISHED"]
        elif function_name == "level_flight":
            # level_flight通常没有明确的完成状态，使用时间判断
            return action_status in ["LEVEL_FLIGHT_FINISHED", "FINISHED"]
        elif function_name == "climb":
            return action_status in ["CLIMB_FINISHED", "FINISHED"]
        elif function_name == "dive":
            return action_status in ["DIVE_FINISHED", "FINISHED"]
        else:
            # 通用完成状态检测
            return "FINISHED" in action_status.upper()

    def _get_action_duration_from_status(self, env, function_name: str) -> float:
        """从动作状态获取实际执行时间"""
        try:
            # 这是一个备用方法，用于在无法检测状态时估算时间
            current_time = env.current_step * env.time_interval
            return current_time
        except:
            return 20.0  # 默认时间

    def _initialize_acmi_file(self, filepath: str):
        """初始化ACMI文件 - 添加F-16模型定义"""
        try:
            with open(filepath, mode='w', encoding='utf-8-sig') as f:
                f.write("FileType=text/acmi/tacview\n")
                f.write("FileVersion=2.1\n")
                f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")

                # 添加飞机模型定义 - 确保显示为F-16而非几何形状
                f.write("0,A0100,Name=F16,Color=Red\n")
                f.write("0,B0100,Name=F16,Color=Blue\n")
        except Exception as e:
            logging.error(f"ACMI文件初始化失败: {e}")

    def _write_acmi_frame(self, filepath: str, env):
        """写入ACMI帧数据 - 使用与现有战术仿真一致的格式"""
        try:
            with open(filepath, mode='a', encoding='utf-8-sig') as f:
                current_time = env.current_step * env.time_interval
                f.write(f"#{current_time:.2f}\n")

                for agent_id, agent in env.agents.items():
                    if agent.is_alive:
                        # 使用agent.log()方法获取正确的ACMI格式数据
                        log_msg = agent.log()
                        if log_msg:
                            f.write(log_msg + "\n")
        except Exception as e:
            logging.error(f"ACMI帧写入失败: {e}")

    def _record_trajectory_data(self, env, current_time: float):
        """记录轨迹数据用于CSV输出"""
        for agent_id, agent in env.agents.items():
            if agent.is_alive:
                # 获取位置数据
                pos_x = agent.get_property_value(c.position_long_gc_deg) * 111320  # 转换为米
                pos_y = agent.get_property_value(c.position_lat_geod_deg) * 111320  # 转换为米
                pos_z = agent.get_property_value(c.position_h_sl_m)

                # 获取速度和姿态数据
                velocity = agent.get_property_value(c.velocities_u_mps)
                heading = np.rad2deg(agent.get_property_value(c.attitude_psi_rad))
                pitch = np.rad2deg(agent.get_property_value(c.attitude_theta_rad))
                roll = np.rad2deg(agent.get_property_value(c.attitude_phi_rad))

                # 记录数据点
                data_point = {
                    'Time_s': current_time,
                    'Agent_ID': agent_id,
                    'X_m': pos_x,
                    'Y_m': pos_y,
                    'Z_m': pos_z,
                    'Velocity_m_s': velocity,
                    'Heading_deg': heading,
                    'Pitch_deg': pitch,
                    'Roll_deg': roll,
                    'Action_Type': self.current_action_type
                }

                self.trajectory_data.append(data_point)

    def _save_csv_data(self, csv_path: str):
        """保存CSV数据"""
        try:
            if not self.trajectory_data:
                logging.warning("没有轨迹数据可保存")
                return

            df = pd.DataFrame(self.trajectory_data)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            logging.info(f"CSV数据已保存: {csv_path} ({len(df)} 行)")

        except Exception as e:
            logging.error(f"CSV数据保存失败: {e}")

    def generate_batch_data(self, action_names: List[str] = None, samples_per_action: int = None):
        """批量生成数据"""
        if action_names is None:
            action_names = list(self.action_configs.keys())

        total_samples = 0
        successful_samples = 0

        logging.info(f"🚀 开始批量生成基础动作数据")
        logging.info(f"动作类型: {action_names}")
        logging.info(f"输出目录: {self.output_dir}")

        for action_name in action_names:
            if action_name not in self.action_configs:
                logging.warning(f"未知动作类型: {action_name}")
                continue

            config = self.action_configs[action_name]
            samples_count = samples_per_action or config.samples_count

            logging.info(f"\n📊 生成 {action_name} 数据 ({samples_count} 样本)")

            for i in range(1, samples_count + 1):
                total_samples += 1
                acmi_path, csv_path = self.generate_single_action_data(action_name, i)

                if acmi_path and csv_path:
                    successful_samples += 1
                    logging.info(f"✅ 样本 {i}/{samples_count} 完成")
                else:
                    logging.error(f"❌ 样本 {i}/{samples_count} 失败")

        logging.info(f"\n🎉 批量生成完成!")
        logging.info(f"总样本数: {total_samples}")
        logging.info(f"成功样本数: {successful_samples}")
        logging.info(f"成功率: {successful_samples/total_samples*100:.1f}%")

        return successful_samples, total_samples

    def generate_test_samples(self):
        """生成测试样本 - 每种动作生成5个样本"""
        logging.info("🧪 生成测试样本 (每种动作5个样本)")
        return self.generate_batch_data(samples_per_action=5)

    def get_available_actions(self) -> List[str]:
        """获取可用的动作列表"""
        return list(self.action_configs.keys())


def main():
    """主函数"""
    print("🎯 基础动作数据生成框架")
    print("=" * 50)

    # 创建生成器
    generator = BasicActionDataGenerator()

    print("可用的11种标准基础动作:")
    for i, action in enumerate(generator.get_available_actions(), 1):
        print(f"  {i:2d}. {action}")

    print("\n选择生成模式:")
    print("  1. 测试模式 (每种动作5个样本)")
    print("  2. 完整模式 (每种动作100个样本)")
    print("  3. 自定义模式")

    try:
        choice = int(input("\n请选择模式 (1-3): ").strip())

        if choice == 1:
            print("\n🧪 开始测试模式生成...")
            generator.generate_test_samples()

        elif choice == 2:
            print("\n🚀 开始完整模式生成...")
            generator.generate_batch_data()

        elif choice == 3:
            print("\n🔧 自定义模式")
            samples = int(input("每种动作生成样本数: ").strip())
            generator.generate_batch_data(samples_per_action=samples)

        else:
            print("❌ 无效选择")

    except KeyboardInterrupt:
        print("\n\n⏹️ 用户中断生成")
    except Exception as e:
        print(f"\n❌ 生成失败: {e}")


if __name__ == "__main__":
    main()
