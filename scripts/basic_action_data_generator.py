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
import argparse
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
    
    def __init__(self, output_dir: str = "scripts/tacticalTemplateProject/basic_action_data"):
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
            "level_flight", "accelerate", "decelerate", "turn_left", "turn_right",
            "climb", "climb_left", "climb_right", "dive", "dive_left", "dive_right"
        ]

        for action_name in action_names:
            action_dir = os.path.join(self.output_dir, action_name)
            os.makedirs(action_dir, exist_ok=True)
            self.action_dirs[action_name] = action_dir

        logging.info(f"创建了{len(action_names)}个动作子目录")
        
    def _setup_standard_actions(self) -> Dict[str, ActionConfig]:
        """设置11种标准基础动作配置"""
        configs = {}

        # 1. 平飞 (level_flight) - 基础参考动作
        configs["level_flight"] = ActionConfig(
            name="level_flight",
            function_name="level_flight",
            param_ranges={
                "duration": (20.0, 30.0),  # 平飞持续时间
            },
            duration_range=(20.0, 30.0),
            samples_count=50  # 扩大到50个高质量样本
        )

        # 2. 加速 (accelerate) - 修复函数映射
        configs["accelerate"] = ActionConfig(
            name="accelerate",
            function_name="accelerate",  # 修复：使用正确的accelerate函数
            param_ranges={
                "velocity_increase": (60.0, 120.0),  # 修复：加速幅度提高到60-120 m/s，确保有效控制
                "duration": (10.0, 35.0)  # 加速持续时间：10-35秒
            },
            duration_range=(10.0, 35.0),
            samples_count=50  # 扩大到50个高质量样本
        )

        # 3. 减速 (decelerate) - 修复函数映射
        configs["decelerate"] = ActionConfig(
            name="decelerate",
            function_name="decelerate",  # 修复：使用正确的decelerate函数
            param_ranges={
                "velocity_decrease": (60.0, 120.0),  # 修复：减速幅度提高到60-120 m/s，确保有效控制
                "duration": (10.0, 35.0)  # 减速持续时间：10-35秒
            },
            duration_range=(10.0, 35.0),
            samples_count=50  # 扩大到50个高质量样本
        )

        # 4. 左转 (turn_left) - 小角度范围修复
        configs["turn_left"] = ActionConfig(
            name="turn_left",
            function_name="turn",
            param_ranges={
                "turn_angle": (-90.0, -15.0),  # 修复为小角度：15-90度左转
                "turn_rate": (2.0, 6.0),  # 合理转弯速率：2.0-6.0度/秒
            },
            duration_range=(15.0, 25.0),  # 小角度转弯时间范围
            samples_count=50  # 扩大到50个高质量样本
        )

        # 5. 右转 (turn_right) - 小角度范围修复
        configs["turn_right"] = ActionConfig(
            name="turn_right",
            function_name="turn",
            param_ranges={
                "turn_angle": (15.0, 90.0),  # 修复为小角度：15-90度右转
                "turn_rate": (2.0, 6.0),  # 合理转弯速率：2.0-6.0度/秒
            },
            duration_range=(15.0, 25.0),  # 小角度转弯时间范围
            samples_count=50  # 扩大到50个高质量样本
        )

        # 6. 爬升 (climb/pull_up) - 物理可行的参数差异化
        configs["climb"] = ActionConfig(
            name="climb",
            function_name="pull_up",
            param_ranges={
                "altitude_gain": (1000.0, 4000.0),  # 物理可行：1000-4000米爬升（初始高度10000m）
                "duration": (15.0, 25.0)  # 爬升持续时间
            },
            duration_range=(15.0, 25.0),
            samples_count=50  # 扩大到50个高质量样本
        )

        # 7. 俯冲 (dive) - 物理可行的参数差异化
        configs["dive"] = ActionConfig(
            name="dive",
            function_name="dive",
            param_ranges={
                "altitude_loss": (1000.0, 7000.0),  # 物理可行：1000-7000米俯冲（初始高度10000m，最低3000m）
                "duration": (15.0, 25.0),  # 俯冲持续时间
                "min_altitude": (2000.0, 3000.0)  # 安全最小高度范围
            },
            duration_range=(15.0, 25.0),
            samples_count=50  # 扩大到50个高质量样本
        )

        # 8. 左爬升 (climb_left) - 小角度组合机动修复
        configs["climb_left"] = ActionConfig(
            name="climb_left",
            function_name="diagonal_flight",  # 使用组合机动函数
            param_ranges={
                "altitude_gain": (1000.0, 3000.0),  # 物理可行：1000-3000米爬升
                "turn_angle": (-90.0, -15.0),  # 修复为小角度：15-90度左转
                "turn_rate": (2.0, 6.0),  # 合理转弯率：2-6度/秒
                "duration": (20.0, 30.0)  # 组合机动：20-30秒
            },
            duration_range=(20.0, 30.0),
            samples_count=50  # 扩大到50个高质量样本
        )

        # 9. 右爬升 (climb_right) - 小角度组合机动修复
        configs["climb_right"] = ActionConfig(
            name="climb_right",
            function_name="diagonal_flight",  # 使用组合机动函数
            param_ranges={
                "altitude_gain": (1000.0, 3000.0),  # 物理可行：1000-3000米爬升
                "turn_angle": (15.0, 90.0),  # 修复为小角度：15-90度右转
                "turn_rate": (2.0, 6.0),  # 合理转弯率：2-6度/秒
                "duration": (20.0, 30.0)  # 组合机动：20-30秒
            },
            duration_range=(20.0, 30.0),
            samples_count=50  # 扩大到50个高质量样本
        )



        # 10. 左俯冲 (dive_left) - 小角度组合机动修复
        configs["dive_left"] = ActionConfig(
            name="dive_left",
            function_name="diagonal_flight",  # 使用组合机动函数
            param_ranges={
                "altitude_loss": (1500.0, 4000.0),  # 物理可行：1500-4000米俯冲
                "turn_angle": (-90.0, -15.0),  # 修复为小角度：15-90度左转
                "turn_rate": (2.0, 6.0),  # 合理转弯率：2-6度/秒
                "min_altitude": (2000.0, 3000.0),  # 安全最小高度范围
                "duration": (20.0, 30.0)  # 组合机动：20-30秒
            },
            duration_range=(20.0, 30.0),
            samples_count=50  # 扩大到50个高质量样本
        )

        # 11. 右俯冲 (dive_right) - 小角度组合机动修复
        configs["dive_right"] = ActionConfig(
            name="dive_right",
            function_name="diagonal_flight",  # 使用组合机动函数
            param_ranges={
                "altitude_loss": (1500.0, 4000.0),  # 物理可行：1500-4000米俯冲
                "turn_angle": (15.0, 90.0),  # 修复为小角度：15-90度右转
                "turn_rate": (2.0, 6.0),  # 合理转弯率：2-6度/秒
                "min_altitude": (2000.0, 3000.0),  # 安全最小高度范围
                "duration": (20.0, 30.0)  # 组合机动：20-30秒
            },
            duration_range=(20.0, 30.0),
            samples_count=50  # 扩大到50个高质量样本
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

        # 动态调整初始高度以支持大幅度俯冲
        initial_altitude = self._calculate_dynamic_initial_altitude(action_name, params)

        # 确定动作标注名称（用于CSV数据标注）
        actual_action_name = self._determine_action_label(action_name, params)

        # 生成规范化文件名（移除冗余前缀和时间戳）
        param_str = self._format_filename_parameters(action_name, params)

        # 新的简洁文件命名格式：{动作类型}_{关键参数值}_{样本编号}
        base_filename = f"{action_name}_{param_str}_{sample_index:03d}"
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
        """根据动作类型和参数生成格式化的文件名参数部分 - 新规范"""
        param_parts = []

        # 基础动作命名规范
        if action_name in ["turn_left", "turn_right"]:
            # 转弯动作：{angle}deg
            angle = int(abs(params.get("turn_angle", 0)))
            param_parts = [f"{angle}deg"]

        elif action_name == "climb":
            # 爬升动作：{altitude}m
            altitude = int(params.get("altitude_gain", 0))
            param_parts = [f"{altitude}m"]

        elif action_name == "dive":
            # 俯冲动作：{altitude}m
            altitude = int(params.get("altitude_loss", 0))
            param_parts = [f"{altitude}m"]

        elif action_name == "accelerate":
            # 加速动作（现在是平飞）：{duration}s
            duration = int(params.get("duration", 20))
            param_parts = [f"{duration}s"]

        elif action_name == "decelerate":
            # 减速动作（现在是平飞）：{duration}s
            duration = int(params.get("duration", 20))
            param_parts = [f"{duration}s"]

        elif action_name == "level_flight":
            # 平飞动作：{speed}mps
            speed = int(params.get("current_velocity", 250))
            param_parts = [f"{speed}mps"]

        # 组合动作命名规范
        elif action_name in ["climb_left", "climb_right", "dive_left", "dive_right"]:
            # 组合动作：{高度参数}_{转弯参数}
            if "climb" in action_name:
                altitude = int(params.get("altitude_gain", 0))
                param_parts.append(f"{altitude}m")
            else:  # dive
                altitude = int(params.get("altitude_loss", 0))
                param_parts.append(f"{altitude}m")

            angle = int(abs(params.get("turn_angle", 0)))
            param_parts.append(f"{angle}deg")

        else:
            # 默认格式：自动识别关键参数
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    if "angle" in key.lower():
                        param_parts.append(f"{int(abs(value))}deg")
                    elif "altitude" in key.lower():
                        param_parts.append(f"{int(abs(value))}m")
                    elif "velocity" in key.lower() or "speed" in key.lower():
                        param_parts.append(f"{int(abs(value))}mps")
                    if len(param_parts) >= 2:  # 最多两个关键参数
                        break

        return "_".join(param_parts) if param_parts else "default"

    def _calculate_dynamic_initial_altitude(self, action_name: str, params: Dict[str, Any]) -> float:
        """根据动作类型和参数动态计算初始高度，确保有足够空间完成机动"""
        base_altitude = 10000.0  # 基础高度10000米

        # 俯冲相关动作需要更高的初始高度
        if action_name in ["dive", "dive_left", "dive_right"]:
            altitude_loss = params.get("altitude_loss", 0.0)
            if altitude_loss > 5000.0:
                # 大幅度俯冲：设置更高初始高度
                required_altitude = altitude_loss + 5000.0  # 保留5000米安全高度
                dynamic_altitude = max(base_altitude, required_altitude)
                logging.info(f"🏔️ 大幅度俯冲动作 {action_name}: 俯冲{altitude_loss:.0f}m, 动态初始高度: {dynamic_altitude:.0f}m")
                return dynamic_altitude
            elif altitude_loss > 3000.0:
                # 中等俯冲：适度提升初始高度
                dynamic_altitude = base_altitude + 2000.0  # 12000米
                logging.info(f"🏔️ 中等俯冲动作 {action_name}: 俯冲{altitude_loss:.0f}m, 动态初始高度: {dynamic_altitude:.0f}m")
                return dynamic_altitude

        # 其他动作使用标准高度
        return base_altitude

    def _update_config_conditions(self, action_name: str, altitude_meters: float):
        """动态更新配置文件中的初始条件（高度和速度）"""
        import yaml

        config_path = "envs/JSBSim/configs/simple_maneuver_config.yaml"
        altitude_feet = altitude_meters * 3.28084  # 转换为英尺

        # 根据动作类型设置不同的初始速度
        if action_name == "accelerate":
            velocity_fps = 820.0  # 加速动作使用较低初始速度 (250 m/s)
            logging.info(f"🚀 加速动作使用低初始速度: 820 fps (250 m/s)")
        else:
            velocity_fps = 1200.0  # 其他动作使用标准初始速度 (365 m/s)
            logging.info(f"✈️ {action_name}动作使用标准初始速度: 1200 fps (365 m/s)")

        try:
            # 读取配置文件
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # 更新所有飞机的初始条件
            for aircraft_id in config['aircraft_configs']:
                config['aircraft_configs'][aircraft_id]['init_state']['ic_h_sl_ft'] = altitude_feet
                config['aircraft_configs'][aircraft_id]['init_state']['ic_u_fps'] = velocity_fps

            # 写回配置文件
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

            logging.info(f"🏔️ 动态更新初始条件: 高度{altitude_meters:.0f}m ({altitude_feet:.0f}ft), 速度{velocity_fps:.0f}fps")

        except Exception as e:
            logging.warning(f"⚠️ 无法更新配置文件: {e}")

    def _run_simulation(self, action_name: str, function_name: str, params: Dict[str, float],
                       acmi_path: str, csv_path: str) -> bool:
        """运行单次仿真 - 添加精确时间范围控制"""
        try:
            # 动态调整初始条件配置（高度和速度）
            dynamic_altitude = self._calculate_dynamic_initial_altitude(action_name, params)
            self._update_config_conditions(action_name, dynamic_altitude)

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

            # 重置坐标参考点（每次仿真重新初始化相对坐标系）
            if hasattr(self, 'reference_lon'):
                delattr(self, 'reference_lon')
            if hasattr(self, 'reference_lat'):
                delattr(self, 'reference_lat')
            if hasattr(self, 'reference_z'):
                delattr(self, 'reference_z')

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

                # CSV数据记录：仅在动作执行期间记录（修复：每步都记录，确保0.2s间隔）
                if action_start_step <= step <= csv_end_step:
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
        """记录轨迹数据用于CSV输出 - 只记录A0100代理数据，使用拖曳射击项目完全相同的坐标系统"""
        for agent_id, agent in env.agents.items():
            # 只记录A0100代理数据，排除B0100（敌方）数据
            if agent.is_alive and agent_id == "A0100":
                # 使用与拖曳射击项目完全相同的坐标计算方法
                # 直接使用agent.get_position()，这已经是基于battle_field_center的LLA2NEU转换后的NEU坐标
                # 现在battle_field_center已设置为[120.0, 60.4, 0.0]，与拖曳射击项目完全一致
                pos = agent.get_position()  # 这返回基于战场中心的NEU坐标系[N, E, U]

                # 获取速度和姿态数据
                velocity = agent.get_property_value(c.velocities_u_mps)
                heading = np.rad2deg(agent.get_property_value(c.attitude_psi_rad))
                pitch = np.rad2deg(agent.get_property_value(c.attitude_theta_rad))
                roll = np.rad2deg(agent.get_property_value(c.attitude_phi_rad))

                # 记录数据点（使用拖曳射击项目完全相同的坐标格式）
                # 现在坐标应该与拖曳射击项目一致：A0100在X≈-44535m，Z≈5940m
                data_point = {
                    'Time_s': current_time,
                    'Agent_ID': agent_id,
                    'X_m': pos[0],  # NEU坐标系的North分量（基于战场中心）
                    'Y_m': pos[1],  # NEU坐标系的East分量（基于战场中心）
                    'Z_m': pos[2],  # NEU坐标系的Up分量（基于战场中心）
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
    parser = argparse.ArgumentParser(description='基础动作数据生成框架')
    parser.add_argument('--action', type=str, default='all',
                       help='生成的动作类型 (all, accelerate, decelerate, 等)')
    parser.add_argument('--mode', type=str, default='test',
                       choices=['test', 'production', 'custom'],
                       help='生成模式: test(5个样本), production(配置文件中的样本数), custom(自定义)')
    parser.add_argument('--samples', type=int, default=None,
                       help='自定义模式下每种动作的样本数')

    args = parser.parse_args()

    print("🎯 基础动作数据生成框架")
    print("=" * 50)

    # 创建生成器
    generator = BasicActionDataGenerator()

    print("可用的11种标准基础动作:")
    for i, action in enumerate(generator.get_available_actions(), 1):
        print(f"  {i:2d}. {action}")

    print(f"\n选择的模式: {args.mode}")
    if args.action != 'all':
        print(f"指定动作: {args.action}")

    try:
        if args.mode == 'test':
            print("\n🧪 开始测试模式生成...")
            if args.action == 'all':
                generator.generate_test_samples()
            else:
                generator.generate_batch_data([args.action], samples_per_action=5)

        elif args.mode == 'production':
            print("\n🚀 开始生产模式生成...")
            if args.action == 'all':
                generator.generate_batch_data()
            else:
                generator.generate_batch_data([args.action])

        elif args.mode == 'custom':
            samples = args.samples or int(input("每种动作生成样本数: ").strip())
            print(f"\n🔧 自定义模式 ({samples}个样本)")
            if args.action == 'all':
                generator.generate_batch_data(samples_per_action=samples)
            else:
                generator.generate_batch_data([args.action], samples_per_action=samples)

        else:
            print("❌ 无效模式")

    except KeyboardInterrupt:
        print("\n\n⏹️ 用户中断生成")
    except Exception as e:
        print(f"\n❌ 生成失败: {e}")


if __name__ == "__main__":
    main()
