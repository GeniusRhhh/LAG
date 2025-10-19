#!/usr/bin/env python3
"""
回转射击战术仿真运行脚本 - 基于钳形夹击架构
双机在DOR前做short skate，到达安全回转距离后回转重新交战，类钳形包夹
"""

import os
import sys
import logging
import numpy as np
import time
import pandas as pd
from datetime import datetime

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# 导入必要模块
from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.core.catalog import Catalog as c
from turn_around_shooting_tactical_task import TurnAroundShootingTacticalTask
from turn_around_enemy_ai_adapter import create_turn_around_enemy_ai_integration


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'turn_around_simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def record_simulation_data(env, current_time, trajectory_data, radar_data, missile_data):
    """记录仿真数据 - 使用统一数据记录器"""
    from unified_data_recorder import UnifiedDataRecorder

    # 创建临时记录器实例
    temp_recorder = UnifiedDataRecorder("turn_around_shooting")

    # 获取战术任务以支持行动注释
    tactical_task = getattr(env, 'task', None)

    # 记录所有数据
    temp_recorder.record_aircraft_trajectory(env, current_time, tactical_task)
    temp_recorder.record_radar_data(env, current_time)
    temp_recorder.record_missile_data(env, current_time)

    # 将数据添加到现有列表中
    trajectory_data.extend(temp_recorder.trajectory_data)
    radar_data.extend(temp_recorder.radar_data)
    missile_data.extend(temp_recorder.missile_data)


def save_csv_data(output_dir, timestamp, trajectory_data, radar_data, missile_data, simulation_log=None):
    """保存CSV数据文件"""
    from unified_data_recorder import UnifiedDataRecorder

    # 创建统一数据记录器
    recorder = UnifiedDataRecorder("turn_around_shooting")

    # 将数据添加到记录器
    recorder.trajectory_data = trajectory_data
    recorder.radar_data = radar_data
    recorder.missile_data = missile_data

    # 使用统一格式保存文件
    saved_files = recorder.save_csv_files(output_dir, timestamp, simulation_log)

    print("[CHECK] 回转射击战术轨迹数据生成完成")

    return saved_files


def generate_missile_analysis(missile_df, output_dir, timestamp):
    """生成导弹轨迹分析数据"""
    if missile_df.empty:
        print("没有导弹数据，跳过轨迹分析")
        return

    missile_analysis = []

    # 按导弹ID分组分析
    for missile_id in missile_df['Missile_ID'].unique():
        missile_traj = missile_df[missile_df['Missile_ID'] == missile_id]

        if missile_traj.empty:
            continue

        # 计算导弹轨迹统计
        launch_time = missile_traj['Time_s'].min()
        final_time = missile_traj['Time_s'].max()
        flight_duration = final_time - launch_time

        # 计算飞行距离
        positions = missile_traj[['X_m', 'Y_m', 'Z_m']].values
        if len(positions) > 1:
            distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            total_distance = np.sum(distances)
        else:
            total_distance = 0.0

        # 平均速度
        avg_velocity = missile_traj['Velocity_m_s'].mean()
        max_velocity = missile_traj['Velocity_m_s'].max()

        # 最终状态
        final_status = missile_traj['Status'].iloc[-1]

        missile_analysis.append({
            'Missile_ID': missile_id,
            'Launcher_ID': missile_traj['Launcher_ID'].iloc[0],
            'Target_ID': missile_traj['Target_ID'].iloc[0],
            'Launch_Time_s': launch_time,
            'Final_Time_s': final_time,
            'Flight_Duration_s': flight_duration,
            'Total_Distance_m': total_distance,
            'Average_Velocity_m_s': avg_velocity,
            'Max_Velocity_m_s': max_velocity,
            'Final_Status': final_status
        })

    # 保存导弹轨迹分析
    if missile_analysis:
        analysis_df = pd.DataFrame(missile_analysis)
        analysis_file = os.path.join(output_dir, f"turn_around_missile_analysis_{timestamp}.csv")
        analysis_df.to_csv(analysis_file, index=False)
        print(f"回转射击导弹轨迹分析已保存: {analysis_file}")


def print_tactical_info():
    """打印回转射击战术信息"""
    print("=" * 80)
    print("[TRIDENT] 双机回转射击战术仿真")
    print("=" * 80)
    print("战术描述: 双机在DOR前做short skate，到达安全回转距离后回转重新交战")
    print("")
    print("战术阶段:")
    print("  1. NLT-MELD (90-81km): 保持间距，平稳飞行")
    print("  2. MELD-MTR (81-45km): 保持间距，平稳飞行")
    print("  3. MTR-TR (45-41km): 保持间距，平稳飞行，快结束时发射导弹")
    print("  4. TR-DOR (41-19.6km): 第一次脱离+回转交战")
    print("     - 平稳飞行一段后进行short skate（长机朝左，僚机朝右）-> 朝180度")
    print("     - 到达安全距离后回转重新交战（长机朝右，僚机朝左）-> 朝0度")
    print("  5. DOR-DR (19.6-14.5km): 第二次脱离返航")
    print("     - 再次做short skate（长机朝右，僚机朝左）-> 朝180度")
    print("     - 平稳飞行返航")
    print("")
    print("配置文件: turn_around_shooting_tactical.yaml")
    print("=" * 80)


def print_simulation_status(env, current_time, step_count):
    """打印仿真状态"""
    try:
        # 获取战术阶段
        task = env.task
        if hasattr(task, 'current_phase'):
            phase = task.current_phase.value
        else:
            phase = "UNKNOWN"
        
        # 计算双方距离
        leader_red = env._jsbsims.get("A0100")
        leader_blue = env._jsbsims.get("B0100")
        
        if leader_red and leader_blue and leader_red.is_alive and leader_blue.is_alive:
            pos1 = leader_red.get_position()
            pos2 = leader_blue.get_position()
            distance = np.linalg.norm(np.array(pos1) - np.array(pos2)) / 1000.0
        else:
            distance = 0.0
        
        # 统计存活飞机和导弹
        red_alive = sum(1 for aid in ["A0100", "A0200"] 
                       if aid in env.agents and env.agents[aid].is_alive)
        blue_alive = sum(1 for aid in ["B0100", "B0200"] 
                        if aid in env.agents and env.agents[aid].is_alive)
        missiles = len(env._tempsims) if hasattr(env, '_tempsims') else 0
        
        print(f"{current_time:6.1f}s | {step_count:4d} | {phase:10s} | {distance:6.1f}km | "
              f"红{red_alive}蓝{blue_alive} | 导弹{missiles}")
        
    except Exception as e:
        logging.warning(f"打印状态失败: {e}")


def run_turn_around_simulation():
    """运行回转射击战术仿真"""
    setup_logging()
    print_tactical_info()

    # 创建输出目录
    output_dir = os.path.join(current_dir, "turn_around_shooting_results")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 创建环境
    logging.info("创建回转射击战术环境...")
    config_name = "turn_around_shooting_tactical"
    
    # 使用配置名称创建环境
    env = MultipleCombatEnv(config_name)
    
    # 强制设置1500步（300秒）
    env.max_steps = 1500
    
    # 创建回转射击战术任务
    logging.info("创建回转射击战术任务...")
    tactical_task = TurnAroundShootingTacticalTask(env.config)
    
    # 集成统一敌方AI系统
    logging.info("集成统一敌方AI系统...")
    enemy_ai_adapter = create_turn_around_enemy_ai_integration(tactical_task)
    if enemy_ai_adapter is None:
        logging.warning("⚠️ 统一敌方AI系统集成失败，将使用基础敌方行为")
    
    # 替换环境的任务
    env.task = tactical_task
    logging.info("回转射击战术任务设置完成")

    # 数据记录
    trajectory_data = []
    radar_data = []
    missile_data = []

    # 准备ACMI文件路径
    acmi_filepath = os.path.join(output_dir, f"turn_around_2v2_{timestamp}.acmi")

    try:
        # 重置环境
        logging.info("重置环境...")
        obs = env.reset()
        
        logging.info("开始仿真...")
        start_time = time.time()
        
        # 仿真参数
        max_steps = 1500
        time_interval = 0.2
        step_count = 0
        
        print("\n开始回转射击仿真...")
        print("=" * 100)
        print("时间(s) | 步数 | 战术阶段 | 距离(km) | 飞机状态")
        print("-" * 100)
        
        # 仿真循环
        while step_count < max_steps:
            # 创建虚拟动作
            num_agents = len(env._jsbsims)
            actions = np.zeros((1, num_agents, 4))
            
            # 执行步骤
            obs, share_obs, rewards, dones, infos = env.step(actions)
            
            step_count += 1
            current_time = step_count * time_interval
            
            # 每步都渲染ACMI - 这是关键！
            try:
                env.render(mode="txt", filepath=acmi_filepath)
            except Exception as e:
                logging.warning(f"Failed to render step {step_count}: {e}")
            
            # 记录数据
            record_simulation_data(env, current_time, trajectory_data, radar_data, missile_data)
            
            # 打印进度
            if step_count % 25 == 0:
                print_simulation_status(env, current_time, step_count)
            
            # 检查是否结束
            if isinstance(dones, dict):
                all_done = all(dones.values())
            else:
                all_done = all(dones)
            
            if all_done:
                logging.info(f"仿真提前结束 at step {step_count}")
                break
        
        # 仿真完成
        elapsed_time = time.time() - start_time
        
        print("=" * 100)
        print("[SUCCESS] 回转射击仿真完成!")
        print(f"仿真时间: {current_time:.1f}秒")
        print(f"总步数: {step_count}")
        print(f"计算耗时: {elapsed_time:.2f}秒")
        print("=" * 100)
        
        # 保存数据
        logging.info("保存仿真数据...")
        
        # 转换为DataFrame
        trajectory_df = pd.DataFrame(trajectory_data)
        radar_df = pd.DataFrame(radar_data)
        missile_df = pd.DataFrame(missile_data)
        
        # 保存CSV文件
        simulation_log = None
        save_csv_data(output_dir, timestamp, trajectory_data, radar_data, missile_data, simulation_log)
        
        # 生成导弹分析
        if not missile_df.empty:
            generate_missile_analysis(missile_df, output_dir, timestamp)
        
        # ACMI文件已在每步生成
        logging.info(f"ACMI文件已保存: {acmi_filepath}")
        
        print(f"\n[FILE] 数据文件生成完成:")
        print(f"  ACMI文件: {acmi_filepath}")
        print(f"  CSV数据目录: {output_dir}")
        print(f"  轨迹数据: turn_around_trajectory_{timestamp}.csv")
        print(f"  雷达数据: turn_around_radar_status_{timestamp}.csv")
        if not missile_df.empty:
            print(f"  导弹数据: turn_around_missile_trajectory_{timestamp}.csv")
            print(f"  导弹分析: turn_around_missile_analysis_{timestamp}.csv")
        
        print("\n[TARGET] 使用说明:")
        print("1. 使用TacView打开ACMI文件查看3D回放")
        print("2. 使用Excel或Python分析CSV数据文件")
        print("=" * 100)
        
    except Exception as e:
        logging.error(f"仿真过程中发生错误: {e}")
        import traceback
        logging.error(traceback.format_exc())
    
    finally:
        # 清理资源
        env.close()
        logging.info("环境已关闭")


if __name__ == "__main__":
    run_turn_around_simulation()

