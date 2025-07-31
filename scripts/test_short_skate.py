#!/usr/bin/env python3
"""
Short Skate机动测试脚本
测试新实现的Short Skate机动动作
"""

import os
import sys
import logging
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('short_skate_test.log', encoding='utf-8')
    ]
)

def test_short_skate_maneuver():
    """测试Short Skate机动"""
    try:
        from envs.JSBSim.envs.env_base import BaseEnv
        from envs.JSBSim.tasks.pure_maneuver_task import PureManeuverTask
        from envs.JSBSim.utils.utils import get_root_dir
        
        logging.info("=" * 60)
        logging.info("Short Skate机动测试开始")
        logging.info("=" * 60)
        
        # 创建环境配置
        config_path = os.path.join(get_root_dir(), "envs", "JSBSim", "configs", "1", "heading.yaml")
        
        # 创建环境
        env = BaseEnv(config_path)
        
        # 创建任务
        task = PureManeuverTask(env.config)
        env.task = task
        
        # 设置Short Skate机动
        task.set_short_skate_maneuver()
        
        # 重置环境
        obs = env.reset()
        
        logging.info("环境初始化完成")
        logging.info(f"智能体数量: {env.num_agents}")
        logging.info(f"观察空间维度: {env.observation_space.shape}")
        logging.info(f"动作空间维度: {env.action_space.shape}")
        
        # 运行仿真
        max_steps = 1000  # 约166秒的仿真时间
        step = 0
        done = False
        
        logging.info("开始Short Skate机动仿真...")
        
        while not done and step < max_steps:
            # 使用零动作，让任务自己控制机动
            action_dim = env.action_space.shape[0]
            actions = np.zeros((env.n_rollout_threads, env.num_agents, action_dim), dtype=np.float32)
            
            # 执行步骤
            obs, share_obs, rewards, dones, infos = env.step(actions)
            
            step += 1
            done = np.any(dones)
            
            # 记录关键信息
            current_time = env.current_step * env.time_interval
            
            # 每30步记录一次状态（前80秒详细，后面简略）
            if current_time <= 80.0:
                if step % 30 == 0:
                    for agent_id in env.agents.keys():
                        if env.agents[agent_id].is_alive:
                            altitude = infos[agent_id]['altitude']
                            heading = infos[agent_id]['heading']
                            velocity = infos[agent_id]['velocity']
                            logging.info(f"[详细] t={current_time:.1f}s - {agent_id}: "
                                       f"高度={altitude:.1f}m, 航向={heading:.1f}°, 速度={velocity:.1f}m/s")
            else:
                if step % 100 == 0:
                    for agent_id in env.agents.keys():
                        if env.agents[agent_id].is_alive:
                            altitude = infos[agent_id]['altitude']
                            heading = infos[agent_id]['heading']
                            velocity = infos[agent_id]['velocity']
                            logging.info(f"[概要] t={current_time:.1f}s - {agent_id}: "
                                       f"高度={altitude:.1f}m, 航向={heading:.1f}°, 速度={velocity:.1f}m/s")
        
        # 关闭环境
        env.close()
        
        logging.info("=" * 60)
        logging.info("Short Skate机动测试完成")
        logging.info(f"总步数: {step}")
        logging.info(f"仿真时间: {step * env.time_interval:.1f}秒")
        logging.info("=" * 60)
        
        return True
        
    except Exception as e:
        logging.error(f"Short Skate机动测试失败: {e}", exc_info=True)
        return False

def test_enhanced_angle_control():
    """测试增强的角度控制"""
    try:
        from envs.JSBSim.tasks.pure_maneuvers import BasicManeuvers

        logging.info("=" * 60)
        logging.info("增强角度控制测试")
        logging.info("=" * 60)

        # 测试不同角度的转弯
        test_angles = [30, 45, 60, 70, 90]

        for angle in test_angles:
            logging.info(f"测试转弯角度: {angle}度")

            # 测试普通转弯
            logging.info("  普通转弯:")
            initial_heading = 0.0
            turn_rate = 3.0
            duration = 0.0

            while duration < 30.0:  # 最多30秒
                result = BasicManeuvers.turn(duration, initial_heading, angle, turn_rate)
                phase, target_heading, _, _, target_roll = result

                if phase is None:
                    break

                if duration % 5.0 < 0.1:  # 每5秒记录一次
                    logging.info(f"    t={duration:.1f}s: 阶段={phase}, "
                               f"目标航向={target_heading:.1f}°, 滚转角={target_roll or 0:.1f}°")

                duration += 0.1

            final_heading = target_heading if target_heading is not None else initial_heading
            actual_turn = final_heading - initial_heading
            logging.info(f"    最终转弯角度: {actual_turn:.1f}° (目标: {angle}°)")
            logging.info(f"    角度误差: {abs(actual_turn - angle):.1f}°")

            # 测试保持高度转弯
            logging.info("  保持高度转弯:")
            initial_altitude = 6000.0
            duration = 0.0

            while duration < 30.0:  # 最多30秒
                result = BasicManeuvers.turn_level(duration, initial_heading, initial_altitude, angle, turn_rate)
                phase, target_heading, target_altitude, _, target_roll = result

                if phase is None:
                    break

                if duration % 5.0 < 0.1:  # 每5秒记录一次
                    logging.info(f"    t={duration:.1f}s: 阶段={phase}, "
                               f"目标航向={target_heading:.1f}°, 目标高度={target_altitude:.1f}m, "
                               f"滚转角={target_roll or 0:.1f}°")

                duration += 0.1

            final_heading = target_heading if target_heading is not None else initial_heading
            actual_turn = final_heading - initial_heading
            logging.info(f"    最终转弯角度: {actual_turn:.1f}° (目标: {angle}°)")
            logging.info(f"    角度误差: {abs(actual_turn - angle):.1f}°")
            logging.info(f"    目标高度: {target_altitude:.1f}m (初始: {initial_altitude:.1f}m)")
            logging.info("")

        logging.info("增强角度控制测试完成")
        return True

    except Exception as e:
        logging.error(f"角度控制测试失败: {e}", exc_info=True)
        return False

def test_discrete_control_precision():
    """测试扩充离散控制精度"""
    logging.info("=" * 60)
    logging.info("扩充离散控制精度测试")
    logging.info("=" * 60)

    try:
        from envs.JSBSim.tasks.pure_maneuver_task import PureManeuverTask

        # 创建任务实例来测试转换函数
        task = PureManeuverTask({})

        # 测试航向控制精度
        logging.info("航向控制精度测试:")
        test_angles = [15, 30, 40, 45, 60, 70, 75, 90, 120]
        for angle in test_angles:
            angle_rad = np.radians(angle)
            index = task._convert_heading_to_index(angle_rad)
            # 反向计算实际角度
            actual_angle = np.degrees(task.norm_delta_heading[index])
            error = abs(actual_angle - angle)
            logging.info(f"  目标: {angle:3.0f}° -> 索引: {index:2d} -> 实际: {actual_angle:5.1f}° (误差: {error:4.1f}°)")

        # 测试高度控制精度
        logging.info("\n高度控制精度测试:")
        test_altitudes = [50, 150, 300, 500, 750, 1000, 1200]
        for altitude in test_altitudes:
            index = task._convert_altitude_to_index(altitude)
            # 反向计算实际高度
            actual_altitude = task.norm_delta_altitude[index] * 1000.0  # 转换回米
            error = abs(actual_altitude - altitude)
            logging.info(f"  目标: {altitude:4.0f}m -> 索引: {index:2d} -> 实际: {actual_altitude:6.0f}m (误差: {error:4.0f}m)")

        # 测试索引范围
        logging.info(f"\n索引范围测试:")
        logging.info(f"  航向数组长度: {len(task.norm_delta_heading)} (索引范围: 0-{len(task.norm_delta_heading)-1})")
        logging.info(f"  高度数组长度: {len(task.norm_delta_altitude)} (索引范围: 0-{len(task.norm_delta_altitude)-1})")
        logging.info(f"  速度数组长度: {len(task.norm_delta_velocity)} (索引范围: 0-{len(task.norm_delta_velocity)-1})")

        logging.info("扩充离散控制精度测试完成")
        return True

    except Exception as e:
        logging.error(f"离散控制精度测试失败: {e}", exc_info=True)
        return False

def test_precise_angle_control():
    """测试精确角度控制修复"""
    logging.info("=" * 60)
    logging.info("精确角度控制修复测试")
    logging.info("=" * 60)

    test_cases = [
        ("crank_tactical", "Crank机动 - 目标70度"),
        ("beam_tactical", "Beam机动 - 目标90度"),
        ("short_skate_tactical", "Short Skate机动 - 目标40度+180度")
    ]

    for maneuver_name, description in test_cases:
        logging.info(f"测试: {description}")
        try:
            from envs.JSBSim.envs.env_base import BaseEnv
            from envs.JSBSim.tasks.pure_maneuver_task import PureManeuverTask
            from envs.JSBSim.utils.utils import get_root_dir

            # 创建环境配置
            config_path = os.path.join(get_root_dir(), "envs", "JSBSim", "configs", "1", "heading.yaml")

            # 创建环境
            env = BaseEnv(config_path)

            # 创建任务
            task = PureManeuverTask(env.config)
            env.task = task

            # 设置机动
            task.set_composite_maneuver(maneuver_name)

            # 重置环境
            obs = env.reset()

            initial_heading = None
            initial_altitude = None

            # 运行仿真
            max_steps = 400  # 约80秒
            step = 0
            done = False

            while not done and step < max_steps:
                action_dim = env.action_space.shape[0]
                actions = np.zeros((env.n_rollout_threads, env.num_agents, action_dim), dtype=np.float32)

                obs, share_obs, rewards, dones, infos = env.step(actions)

                step += 1
                done = np.any(dones)
                current_time = env.current_step * env.time_interval

                # 记录初始状态
                if step == 1:
                    for agent_id in env.agents.keys():
                        if env.agents[agent_id].is_alive:
                            initial_heading = infos[agent_id]['heading']
                            initial_altitude = infos[agent_id]['altitude']
                            logging.info(f"初始状态 - 航向: {initial_heading:.1f}°, 高度: {initial_altitude:.1f}m")

                # 记录关键时刻的状态
                if step % 100 == 0 or current_time >= 80.0:
                    for agent_id in env.agents.keys():
                        if env.agents[agent_id].is_alive:
                            current_heading = infos[agent_id]['heading']
                            current_altitude = infos[agent_id]['altitude']
                            heading_change = current_heading - initial_heading
                            altitude_change = current_altitude - initial_altitude

                            logging.info(f"t={current_time:.1f}s - 航向: {current_heading:.1f}° (变化: {heading_change:+.1f}°), "
                                       f"高度: {current_altitude:.1f}m (变化: {altitude_change:+.1f}m)")

                    if current_time >= 80.0:
                        break

            # 关闭环境
            env.close()

            logging.info(f"{description} 测试完成\n")

        except Exception as e:
            logging.error(f"{description} 测试失败: {e}")

def test_maneuver_completion_stability():
    """测试机动完成后的稳定性"""
    logging.info("=" * 60)
    logging.info("机动完成后稳定性测试")
    logging.info("=" * 60)

    test_cases = [
        ("crank_tactical", "Crank机动完成后稳定性", 50),  # 50秒，机动应该在50秒完成
        ("beam_tactical", "Beam机动完成后稳定性", 51),   # 51秒
        ("notch_tactical", "Notch机动完成后稳定性", 54), # 54秒
        ("short_skate_tactical", "Short Skate机动完成后稳定性", 93) # 93秒
    ]

    for maneuver_name, description, expected_duration in test_cases:
        logging.info(f"\n{description} (预期时长: {expected_duration}秒)")
        try:
            from envs.JSBSim.envs.env_base import BaseEnv
            from envs.JSBSim.tasks.pure_maneuver_task import PureManeuverTask
            from envs.JSBSim.utils.utils import get_root_dir

            config_path = os.path.join(get_root_dir(), "envs", "JSBSim", "configs", "1", "heading.yaml")
            env = BaseEnv(config_path)
            task = PureManeuverTask(env.config)
            env.task = task
            task.set_composite_maneuver(maneuver_name)
            obs = env.reset()

            # 记录初始状态
            initial_states = {}
            for agent_id in env.agents.keys():
                if env.agents[agent_id].is_alive:
                    initial_states[agent_id] = {
                        'heading': env.agents[agent_id].get_property_value(env.agents[agent_id].catalog.attitude_psi_rad) * 180/np.pi,
                        'altitude': env.agents[agent_id].get_property_value(env.agents[agent_id].catalog.position_h_sl_m)
                    }

            # 运行到机动完成后再多20秒
            max_steps = int((expected_duration + 20) * 5)  # 5步/秒
            step = 0
            done = False
            maneuver_completed_time = None

            while not done and step < max_steps:
                action_dim = env.action_space.shape[0]
                actions = np.zeros((env.n_rollout_threads, env.num_agents, action_dim), dtype=np.float32)
                obs, share_obs, rewards, dones, infos = env.step(actions)
                step += 1
                done = np.any(dones)
                current_time = env.current_step * env.time_interval

                # 检测机动是否完成
                if current_time >= expected_duration and maneuver_completed_time is None:
                    maneuver_completed_time = current_time
                    logging.info(f"  机动应该在 {expected_duration}秒 完成，当前时间: {current_time:.1f}秒")

                # 记录关键时刻的状态
                if step % 50 == 0 or (maneuver_completed_time and current_time <= maneuver_completed_time + 10):
                    for agent_id in env.agents.keys():
                        if env.agents[agent_id].is_alive:
                            current_heading = infos[agent_id]['heading']
                            current_altitude = infos[agent_id]['altitude']
                            initial_heading = initial_states[agent_id]['heading']
                            initial_altitude = initial_states[agent_id]['altitude']

                            heading_change = current_heading - initial_heading
                            altitude_change = current_altitude - initial_altitude

                            status = "机动中" if current_time < expected_duration else "完成后"
                            logging.info(f"  {agent_id} t={current_time:.1f}s [{status}]: "
                                       f"航向={current_heading:.1f}° (Δ{heading_change:+.1f}°), "
                                       f"高度={current_altitude:.1f}m (Δ{altitude_change:+.1f}m)")

                # 如果机动完成后10秒还在运行，检查稳定性
                if maneuver_completed_time and current_time >= maneuver_completed_time + 10:
                    break

            env.close()

        except Exception as e:
            logging.error(f"{description} 失败: {e}", exc_info=True)

def test_notch_altitude_control():
    """测试Notch机动的高度控制修复"""
    logging.info("=" * 60)
    logging.info("Notch机动高度控制测试")
    logging.info("=" * 60)

    try:
        from envs.JSBSim.envs.env_base import BaseEnv
        from envs.JSBSim.tasks.pure_maneuver_task import PureManeuverTask
        from envs.JSBSim.utils.utils import get_root_dir

        config_path = os.path.join(get_root_dir(), "envs", "JSBSim", "configs", "1", "heading.yaml")
        env = BaseEnv(config_path)
        task = PureManeuverTask(env.config)
        env.task = task
        task.set_composite_maneuver("notch_tactical")
        obs = env.reset()

        initial_altitude = None
        dive_altitude = None
        max_altitude_during_turn = 0

        max_steps = 300  # 60秒
        step = 0
        done = False

        while not done and step < max_steps:
            action_dim = env.action_space.shape[0]
            actions = np.zeros((env.n_rollout_threads, env.num_agents, action_dim), dtype=np.float32)
            obs, share_obs, rewards, dones, infos = env.step(actions)
            step += 1
            done = np.any(dones)
            current_time = env.current_step * env.time_interval

            for agent_id in env.agents.keys():
                if env.agents[agent_id].is_alive:
                    current_altitude = infos[agent_id]['altitude']

                    if step == 1:
                        initial_altitude = current_altitude
                        logging.info(f"初始高度: {initial_altitude:.1f}m")

                    # 记录俯冲后的最低高度
                    if 10 <= current_time <= 15 and (dive_altitude is None or current_altitude < dive_altitude):
                        dive_altitude = current_altitude

                    # 记录转弯期间的最高高度
                    if 15 <= current_time <= 30:
                        max_altitude_during_turn = max(max_altitude_during_turn, current_altitude)

                    # 每10秒记录一次状态
                    if step % 50 == 0:
                        current_heading = infos[agent_id]['heading']
                        logging.info(f"t={current_time:.1f}s: 高度={current_altitude:.1f}m, 航向={current_heading:.1f}°")

        env.close()

        if initial_altitude and dive_altitude:
            altitude_loss = initial_altitude - dive_altitude
            altitude_gain_during_turn = max_altitude_during_turn - dive_altitude
            logging.info(f"\n结果分析:")
            logging.info(f"初始高度: {initial_altitude:.1f}m")
            logging.info(f"俯冲后最低高度: {dive_altitude:.1f}m (下降 {altitude_loss:.1f}m)")
            logging.info(f"转弯期间最高高度: {max_altitude_during_turn:.1f}m")
            logging.info(f"转弯期间高度上升: {altitude_gain_during_turn:.1f}m")

            if max_altitude_during_turn > initial_altitude:
                logging.warning(f"⚠️  转弯时高度超过初始高度 {max_altitude_during_turn - initial_altitude:.1f}m")
            else:
                logging.info(f"✅ 转弯时高度控制良好，未超过初始高度")

    except Exception as e:
        logging.error(f"Notch高度控制测试失败: {e}", exc_info=True)

def test_short_skate_heading_control():
    """测试Short Skate的航向保持修复"""
    logging.info("=" * 60)
    logging.info("Short Skate航向保持测试")
    logging.info("=" * 60)

    try:
        from envs.JSBSim.envs.env_base import BaseEnv
        from envs.JSBSim.tasks.pure_maneuver_task import PureManeuverTask
        from envs.JSBSim.utils.utils import get_root_dir

        config_path = os.path.join(get_root_dir(), "envs", "JSBSim", "configs", "1", "heading.yaml")
        env = BaseEnv(config_path)
        task = PureManeuverTask(env.config)
        env.task = task
        task.set_composite_maneuver("short_skate_tactical")
        obs = env.reset()

        phase_headings = {}

        max_steps = 500  # 100秒
        step = 0
        done = False

        while not done and step < max_steps:
            action_dim = env.action_space.shape[0]
            actions = np.zeros((env.n_rollout_threads, env.num_agents, action_dim), dtype=np.float32)
            obs, share_obs, rewards, dones, infos = env.step(actions)
            step += 1
            done = np.any(dones)
            current_time = env.current_step * env.time_interval

            for agent_id in env.agents.keys():
                if env.agents[agent_id].is_alive:
                    current_heading = infos[agent_id]['heading']

                    # 记录各阶段的航向
                    if 18 <= current_time <= 22:  # 第一次转弯完成
                        phase_headings["after_first_turn"] = current_heading
                    elif 76 <= current_time <= 80:  # 第二次转弯完成
                        phase_headings["after_second_turn"] = current_heading
                    elif 90 <= current_time <= 95:  # 最终保持阶段
                        phase_headings["final_maintain"] = current_heading

                    # 每20秒记录一次状态
                    if step % 100 == 0:
                        logging.info(f"t={current_time:.1f}s: 航向={current_heading:.1f}°")

        env.close()

        logging.info(f"\n结果分析:")
        if "after_first_turn" in phase_headings:
            logging.info(f"第一次转弯后航向: {phase_headings['after_first_turn']:.1f}° (目标: ~40°)")
        if "after_second_turn" in phase_headings:
            logging.info(f"第二次转弯后航向: {phase_headings['after_second_turn']:.1f}° (目标: ~140°)")
        if "final_maintain" in phase_headings:
            final_heading = phase_headings["final_maintain"]
            second_turn_heading = phase_headings.get("after_second_turn", 140)
            heading_drift = abs(final_heading - second_turn_heading)
            logging.info(f"最终保持航向: {final_heading:.1f}°")
            logging.info(f"航向漂移: {heading_drift:.1f}°")

            if heading_drift < 10:
                logging.info(f"✅ 航向保持良好，漂移小于10度")
            else:
                logging.warning(f"⚠️  航向漂移过大: {heading_drift:.1f}度")

    except Exception as e:
        logging.error(f"Short Skate航向保持测试失败: {e}", exc_info=True)

if __name__ == "__main__":
    logging.info("Notch和Short Skate修复测试系统")

    # 测试Notch高度控制
    logging.info("1. 测试Notch机动高度控制修复...")
    test_notch_altitude_control()

    # 测试Short Skate航向保持
    logging.info("2. 测试Short Skate航向保持修复...")
    test_short_skate_heading_control()

    logging.info("✅ 测试完成")
