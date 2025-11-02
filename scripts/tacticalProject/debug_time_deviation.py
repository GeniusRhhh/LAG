#!/usr/bin/env python3
"""
机动时间偏差调试工具
分析预期时间vs实际时间的差异，并输出详细的调试信息
"""

import os
import sys
import json
import numpy as np
import logging
from datetime import datetime

# 设置路径
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repo_root)
tactical_project_dir = os.path.dirname(os.path.abspath(__file__))

from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.tasks.pure_maneuvers import BasicManeuvers, CompositeManeuverExecutor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_expected_duration(kind, name, params):
    """获取预期持续时间"""
    if kind == "basic":
        if name == "level_flight":
            return float(params.get("duration", 20.0))
        elif name in ("accelerate", "decelerate"):
            return float(params.get("duration", 20.0))
        elif name == "pull_up":
            return float(params.get("duration", 15.0))
        elif name == "dive":
            return float(params.get("duration", 15.0))
        elif name in ("turn", "turn_level"):
            # 转弯时间 = 角度/转弯率 + 额外调整时间
            angle = abs(float(params.get("turn_angle", params.get("heading_change", 30.0))))
            turn_rate = float(params.get("turn_rate", 3.0))
            base_time = angle / max(turn_rate, 1e-3)
            extra_time = min(20.0, angle / 15.0 + 8.0)
            return base_time + extra_time
        elif name == "diagonal_flight":
            ang = abs(float(params.get("turn_angle", params.get("heading_change", 30.0))))
            tr = float(params.get("turn_rate", 3.0))
            altc = abs(float(params.get("altitude_change", 1000.0)))
            vr = float(params.get("vertical_rate", 50.0))
            rec = max(ang / max(tr, 1e-3) + 6.0, altc / max(vr, 1e-3) + 6.0)
            return float(max(params.get("duration", 15.0), rec))
    return 60.0

def debug_single_maneuver(kind, name, params):
    """调试单个机动的时间控制"""
    print(f"\n{'='*60}")
    print(f"调试机动: {kind}/{name}")
    print(f"参数: {params}")
    
    # 计算预期时间
    expected_duration = get_expected_duration(kind, name, params)
    print(f"预期持续时间: {expected_duration:.1f}s")
    
    # 初始化环境
    env_config = {
        "scenario": "1v1/NoWeapon/Selfplay",
        "num_agents": 1,
        "render_mode": None,
        "num_envs": 1,
        "use_baseline": False,
        "use_selfplay": False,
        "use_wandb": False,
        "wandb_project": "debug",
        "wandb_run_name": f"debug_{name}",
        "seed": 0,
        "max_episode_steps": 2000,
        "time_interval": 0.2,
        "use_acmi": False,
    }
    
    env = MultipleCombatEnv(env_config)
    obs, share_obs = env.reset()
    
    # 获取智能体
    agent = list(env.agents.values())[0]
    init_heading = np.rad2deg(agent.get_property_value(c.attitude_psi_rad))
    init_altitude = agent.get_property_value(c.position_h_sl_m)
    init_velocity = agent.get_property_value(c.velocities_u_mps)
    
    print(f"初始状态: 航向={init_heading:.1f}°, 高度={init_altitude:.1f}m, 速度={init_velocity:.1f}m/s")
    
    # 记录时间和状态
    times = []
    phases = []
    maneuver_start_time = None
    maneuver_complete_time = None
    stable_duration = 5.0
    
    step = 0
    max_steps = int((expected_duration * 2 + 20) / env.time_interval)
    
    while step < max_steps:
        actions = np.zeros((env.n_rollout_threads, env.num_agents, 3), dtype=np.float32)
        obs, share_obs, rewards, dones, infos = env.step(actions)
        
        tsec = env.current_step * env.time_interval
        times.append(tsec)
        
        # 执行机动
        if kind == "basic":
            current_heading = np.rad2deg(agent.get_property_value(c.attitude_psi_rad))
            if name == "level_flight":
                result = BasicManeuvers.level_flight(tsec, params.get("duration", 20.0), init_altitude, init_velocity)
            elif name == "accelerate":
                current_vel = agent.get_property_value(c.velocities_u_mps)
                result = BasicManeuvers.accelerate(tsec, current_vel, init_altitude, 
                                                 params.get("duration", 20.0), 
                                                 params.get("velocity_change", 50.0))
            elif name == "decelerate":
                current_vel = agent.get_property_value(c.velocities_u_mps)
                result = BasicManeuvers.decelerate(tsec, current_vel, init_altitude,
                                                 params.get("duration", 25.0),
                                                 params.get("velocity_change", 40.0))
            elif name == "pull_up":
                result = BasicManeuvers.pull_up(tsec, init_altitude, 
                                               params.get("duration", 15.0),
                                               params.get("altitude_gain", 1500.0))
            elif name == "dive":
                current_alt = agent.get_property_value(c.position_h_sl_m)
                result = BasicManeuvers.dive(tsec, current_alt,
                                           params.get("duration", 15.0),
                                           params.get("altitude_loss", 1500.0),
                                           params.get("min_altitude", 3000.0))
            elif name in ("turn", "turn_level"):
                result = BasicManeuvers.crank(tsec, init_heading, init_altitude,
                                            params.get("turn_angle", 80.0),
                                            params.get("turn_rate", 3.0),
                                            None, current_heading)
            else:
                result = (None, None, None, None, None)
            
            phase = result[0] if result[0] else "UNKNOWN"
            phases.append(phase)
            
            # 检测机动开始
            if maneuver_start_time is None and phase != "UNKNOWN":
                maneuver_start_time = tsec
                print(f"机动开始: {phase} at t={tsec:.1f}s")
            
            # 检测机动完成
            if phase in ("ACCELERATION_COMPLETE", "DECELERATION_COMPLETE", "DIVE_FINISHED"):
                if maneuver_complete_time is None:
                    maneuver_complete_time = tsec
                    actual_maneuver_duration = tsec - (maneuver_start_time or 0)
                    print(f"机动完成: {phase} at t={tsec:.1f}s")
                    print(f"实际机动时间: {actual_maneuver_duration:.1f}s")
                    print(f"时间偏差: {actual_maneuver_duration - expected_duration:.1f}s ({((actual_maneuver_duration - expected_duration) / expected_duration * 100):+.1f}%)")
                elif tsec >= maneuver_complete_time + stable_duration:
                    total_duration = tsec
                    print(f"总持续时间（含稳定飞行）: {total_duration:.1f}s")
                    print(f"总时间偏差: {total_duration - expected_duration:.1f}s ({((total_duration - expected_duration) / expected_duration * 100):+.1f}%)")
                    break
        
        # 每5秒输出一次状态
        if step % 25 == 0:  # 0.2s * 25 = 5s
            current_heading = np.rad2deg(agent.get_property_value(c.attitude_psi_rad))
            current_altitude = agent.get_property_value(c.position_h_sl_m)
            current_velocity = agent.get_property_value(c.velocities_u_mps)
            
            hdg_change = current_heading - init_heading
            while hdg_change > 180: hdg_change -= 360
            while hdg_change < -180: hdg_change += 360
            
            alt_change = current_altitude - init_altitude
            vel_change = current_velocity - init_velocity
            
            print(f"[t={tsec:5.1f}s] 航向变化={hdg_change:+6.1f}° 高度变化={alt_change:+7.1f}m 速度变化={vel_change:+6.1f}m/s 阶段={phase}")
        
        step += 1
        
        # 检查是否超时
        if tsec > expected_duration * 2 + 20:
            print(f"超时退出 at t={tsec:.1f}s")
            break
    
    env.close()
    
    # 总结
    final_time = times[-1] if times else 0
    print(f"\n总结:")
    print(f"预期时间: {expected_duration:.1f}s")
    print(f"实际时间: {final_time:.1f}s")
    print(f"时间偏差: {final_time - expected_duration:.1f}s ({((final_time - expected_duration) / expected_duration * 100):+.1f}%)")
    
    return {
        "name": name,
        "expected_duration": expected_duration,
        "actual_duration": final_time,
        "deviation": final_time - expected_duration,
        "deviation_percent": (final_time - expected_duration) / expected_duration * 100
    }

def main():
    """主函数 - 调试所有机动类型"""
    print("机动时间偏差调试工具")
    print("="*60)
    
    # 测试用例
    test_cases = [
        {"kind": "basic", "name": "level_flight", "params": {"duration": 30.0}},
        {"kind": "basic", "name": "turn_level", "params": {"turn_angle": 80.0, "turn_rate": 3.0}},
        {"kind": "basic", "name": "pull_up", "params": {"altitude_gain": 1500.0, "duration": 20.0}},
        {"kind": "basic", "name": "dive", "params": {"altitude_loss": 1500.0, "duration": 20.0, "min_altitude": 2000.0}},
        {"kind": "basic", "name": "accelerate", "params": {"velocity_change": 50.0, "duration": 20.0}},
        {"kind": "basic", "name": "decelerate", "params": {"velocity_change": 50.0, "duration": 20.0}},
    ]
    
    results = []
    
    for test_case in test_cases:
        try:
            result = debug_single_maneuver(test_case["kind"], test_case["name"], test_case["params"])
            results.append(result)
        except Exception as e:
            print(f"错误: {e}")
            continue
    
    # 输出汇总表格
    print(f"\n{'='*80}")
    print("时间偏差汇总表")
    print(f"{'='*80}")
    print(f"{'机动类型':<15} {'预期时间(s)':<12} {'实际时间(s)':<12} {'偏差(s)':<10} {'偏差(%)':<10}")
    print(f"{'-'*80}")
    
    for result in results:
        print(f"{result['name']:<15} {result['expected_duration']:<12.1f} {result['actual_duration']:<12.1f} "
              f"{result['deviation']:<10.1f} {result['deviation_percent']:<10.1f}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(tactical_project_dir, f"time_deviation_debug_{timestamp}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n调试结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
