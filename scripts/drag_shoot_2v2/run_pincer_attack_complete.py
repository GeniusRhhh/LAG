#!/usr/bin/env python3
"""
钳形夹击战术仿真运行脚本 - 完全基于run_drag_shoot_simulation.py架构
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
from pincer_attack_tactical_task_complete import PincerAttackTacticalTask


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'pincer_attack_simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def record_pincer_simulation_data(env, current_time, trajectory_data, radar_data, missile_data):
    """记录钳形夹击仿真数据到CSV格式"""
    # 记录飞机轨迹数据
    for agent_id, aircraft in env._jsbsims.items():
        if aircraft.is_alive:
            pos = aircraft.get_position()
            heading = np.rad2deg(aircraft.get_property_value(c.attitude_psi_rad))
            pitch = np.rad2deg(aircraft.get_property_value(c.attitude_pitch_rad))
            velocity_vector = aircraft.get_velocity()
            velocity = np.linalg.norm(velocity_vector)
            trajectory_data.append({
                'Time_s': current_time,
                'Agent_ID': agent_id,
                'Type': 'F-16',
                'X_m': pos[0],
                'Y_m': pos[1],
                'Z_m': pos[2],
                'Heading_deg': heading,
                'Pitch_deg': pitch,
                'Velocity_m_s': velocity,
                'Tactical_Phase': env.task.current_phase.value if hasattr(env.task, 'current_phase') else 'Unknown'
            })

    # 使用雷达管理器记录雷达数据
    try:
        from radar_manager import record_radar_data
        radar_data.extend(record_radar_data(env, current_time))
    except ImportError:
        # 如果没有雷达管理器，记录基本雷达状态
        for agent_id, aircraft in env._jsbsims.items():
            if aircraft.is_alive:
                radar_data.append({
                    'Time_s': current_time,
                    'Agent_ID': agent_id,
                    'Radar_Status': 'Active',
                    'Detection_Range_km': 100.0,  # 默认探测距离
                    'Lock_Status': 'Searching'
                })

    # 记录导弹数据（如果有）
    if hasattr(env, '_tempsims'):
        for missile_id, missile_sim in env._tempsims.items():
            if missile_sim.is_alive:
                missile_pos = missile_sim.get_position()
                missile_velocity = np.linalg.norm(missile_sim.get_velocity())

                missile_data.append({
                    'Time_s': current_time,
                    'Missile_ID': missile_id,
                    'Launcher_ID': getattr(missile_sim, 'launcher_id', 'Unknown'),
                    'Target_ID': getattr(missile_sim, 'target_id', 'Unknown'),
                    'X_m': missile_pos[0],
                    'Y_m': missile_pos[1],
                    'Z_m': missile_pos[2],
                    'Velocity_m_s': missile_velocity,
                    'Status': 'Active',
                    'Data_Type': 'Trajectory'
                })


def save_pincer_csv_data(output_dir, timestamp, trajectory_data, radar_data, missile_data):
    """保存钳形夹击CSV数据文件"""

    # 保存轨迹数据
    if trajectory_data:
        trajectory_df = pd.DataFrame(trajectory_data)
        trajectory_file = os.path.join(output_dir, f"pincer_trajectory_{timestamp}.csv")
        trajectory_df.to_csv(trajectory_file, index=False)
        print(f"钳形夹击轨迹数据已保存: {trajectory_file}")

    # 保存雷达数据
    if radar_data:
        radar_df = pd.DataFrame(radar_data)
        radar_file = os.path.join(output_dir, f"pincer_radar_status_{timestamp}.csv")
        radar_df.to_csv(radar_file, index=False)
        print(f"钳形夹击雷达数据已保存: {radar_file}")

    # 保存导弹数据
    if missile_data:
        missile_df = pd.DataFrame(missile_data)

        # 分离状态数据和轨迹数据
        trajectory_df = missile_df[missile_df.get('Data_Type', '') == 'Trajectory']

        # 保存导弹轨迹数据
        if not trajectory_df.empty:
            missile_trajectory_file = os.path.join(output_dir, f"pincer_missile_trajectory_{timestamp}.csv")
            trajectory_df.to_csv(missile_trajectory_file, index=False)
            print(f"钳形夹击导弹轨迹数据已保存: {missile_trajectory_file}")

            # 生成导弹轨迹分析数据
            generate_pincer_missile_analysis(trajectory_df, output_dir, timestamp)
        else:
            print("没有钳形夹击导弹轨迹数据")
    else:
        print("没有钳形夹击导弹数据")


def generate_pincer_missile_analysis(missile_df, output_dir, timestamp):
    """生成钳形夹击导弹轨迹分析数据"""
    if missile_df.empty:
        print("没有钳形夹击导弹数据，跳过轨迹分析")
        return

    missile_analysis = []

    # 按导弹ID分组分析
    for missile_id in missile_df['Missile_ID'].unique():
        missile_traj = missile_df[missile_df['Missile_ID'] == missile_id]

        if missile_traj.empty:
            continue

        # 计算导弹轨迹统计
        launch_time = missile_traj['Time_s'].min()
        end_time = missile_traj['Time_s'].max()
        flight_duration = end_time - launch_time

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

        missile_analysis.append({
            'Missile_ID': missile_id,
            'Launcher_ID': missile_traj['Launcher_ID'].iloc[0],
            'Target_ID': missile_traj['Target_ID'].iloc[0],
            'Launch_Time_s': launch_time,
            'End_Time_s': end_time,
            'Flight_Duration_s': flight_duration,
            'Total_Distance_m': total_distance,
            'Average_Velocity_m_s': avg_velocity,
            'Max_Velocity_m_s': max_velocity,
            'Final_Status': missile_traj['Status'].iloc[-1],
            'Tactical_Context': 'Pincer_Attack'
        })

    # 保存导弹轨迹分析
    if missile_analysis:
        analysis_df = pd.DataFrame(missile_analysis)
        analysis_file = os.path.join(output_dir, f"pincer_missile_analysis_{timestamp}.csv")
        analysis_df.to_csv(analysis_file, index=False)
        print(f"钳形夹击导弹轨迹分析已保存: {analysis_file}")

        # 生成导弹轨迹摘要报告
        generate_pincer_missile_summary(analysis_df, output_dir, timestamp)


def generate_pincer_missile_summary(analysis_df, output_dir, timestamp):
    """生成钳形夹击导弹摘要报告"""
    summary_file = os.path.join(output_dir, f"pincer_missile_summary_{timestamp}.txt")

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("钳形夹击战术导弹摘要报告\n")
        f.write("=" * 40 + "\n")
        f.write(f"生成时间: {datetime.now()}\n")
        f.write(f"总导弹数量: {len(analysis_df)}\n\n")

        if not analysis_df.empty:
            f.write("导弹统计:\n")
            f.write(f"  平均飞行时间: {analysis_df['Flight_Duration_s'].mean():.2f}秒\n")
            f.write(f"  平均飞行距离: {analysis_df['Total_Distance_m'].mean()/1000:.2f}km\n")
            f.write(f"  平均速度: {analysis_df['Average_Velocity_m_s'].mean():.2f}m/s\n")
            f.write(f"  最大速度: {analysis_df['Max_Velocity_m_s'].max():.2f}m/s\n\n")

            f.write("各导弹详情:\n")
            for _, missile in analysis_df.iterrows():
                f.write(f"  {missile['Missile_ID']}: {missile['Launcher_ID']} -> {missile['Target_ID']}, "
                       f"飞行{missile['Flight_Duration_s']:.1f}s, "
                       f"距离{missile['Total_Distance_m']/1000:.1f}km\n")

    print(f"钳形夹击导弹摘要报告已保存: {summary_file}")


def print_tactical_info():
    """打印钳形夹击战术信息"""
    print("=" * 80)
    print("🔱 双机钳形夹击战术仿真")
    print("=" * 80)
    print("战术名称: 钳形夹击 (Pincer Attack)")
    print("战术描述: 双机向两侧展开形成钳形包夹，然后收拢攻击")
    print("战术阶段: 钳形展开 → 钳形收拢 → 分层攻击 → 脱离机动 → 返航")
    print("关键特性: 双机Crank展开, 包夹收拢, 分层攻击, 内侧脱离")
    print("适用场景: 2v2对抗, 包夹战术, 双方钳形对抗")
    print("=" * 80)


def run_pincer_attack_simulation():
    """运行钳形夹击仿真 - 包含完整的ACMI和CSV数据生成"""
    setup_logging()
    print_tactical_info()

    try:
        # 创建环境 - 使用现有的拖曳射击配置
        logging.info("创建仿真环境...")
        env = MultipleCombatEnv("drag_shoot_tactical")

        # 创建钳形夹击战术任务
        logging.info("创建钳形夹击战术任务...")
        tactical_task = PincerAttackTacticalTask(env.config)

        # 替换环境的任务
        env.task = tactical_task
        logging.info("钳形夹击战术任务设置完成")

        # 重置环境
        logging.info("重置环境...")
        obs = env.reset()

        # 创建输出目录
        output_dir = "pincer_attack_results"
        os.makedirs(output_dir, exist_ok=True)

        # 仿真参数
        max_steps = 1500  # 300秒仿真 (5分钟)
        time_interval = 0.2

        logging.info(f"开始钳形夹击仿真 (最大步数: {max_steps}, 时间间隔: {time_interval}s)")

        # 准备ACMI文件路径和数据记录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        acmi_filepath = os.path.join(output_dir, f"pincer_attack_2v2_{timestamp}.acmi")

        # 初始化CSV数据记录
        trajectory_data = []
        radar_data = []
        missile_data = []

        # 仿真循环
        step_count = 0
        start_time = datetime.now()

        print("\n开始钳形夹击仿真...")
        print("=" * 100)
        print("时间(s) | 步数 | 战术阶段 | 距离(km) | 飞机 | 导弹")
        print("-" * 100)

        while step_count < max_steps:
            # 创建虚拟动作
            num_agents = len(env._jsbsims)
            actions = np.zeros((1, num_agents, 4))

            # 执行一步
            obs, share_obs, rewards, dones, infos = env.step(actions)

            step_count += 1
            current_time = step_count * time_interval

            # 每步都渲染ACMI - 这是关键！
            try:
                env.render(mode="txt", filepath=acmi_filepath)
            except Exception as e:
                logging.warning(f"Failed to render step {step_count}: {e}")

            # 记录数据
            record_pincer_simulation_data(env, current_time, trajectory_data, radar_data, missile_data)

            # 每25步打印一次状态 (5秒)
            if step_count % 25 == 0:
                print_simulation_status(env, current_time, step_count)

            # 检查终止条件
            if isinstance(dones, dict):
                all_done = all(dones.values())
            else:
                all_done = all(dones)

            if all_done:
                logging.info(f"仿真提前结束 at step {step_count}")
                break

        # 仿真完成
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("=" * 100)
        print("🎉 钳形夹击仿真完成!")
        print(f"仿真时间: {current_time:.1f}秒")
        print(f"总步数: {step_count}")
        print(f"计算耗时: {duration:.2f}秒")
        print("=" * 100)

        # 打印最终状态
        print_final_status(env)

        # 保存CSV数据
        save_pincer_csv_data(output_dir, timestamp, trajectory_data, radar_data, missile_data)

        # 保存基本结果摘要
        save_simulation_results(env, step_count, current_time)

        # ACMI文件已在每步生成
        print(f"\n📁 数据文件生成完成:")
        print(f"  ACMI文件: {acmi_filepath}")
        print(f"  CSV数据目录: {output_dir}")
        print(f"  轨迹数据: pincer_trajectory_{timestamp}.csv")
        print(f"  雷达数据: pincer_radar_status_{timestamp}.csv")
        if missile_data:
            print(f"  导弹数据: pincer_missile_trajectory_{timestamp}.csv")
            print(f"  导弹分析: pincer_missile_analysis_{timestamp}.csv")

        print("\n🎯 使用说明:")
        print("1. 使用TacView打开ACMI文件查看3D回放")
        print("2. 使用Excel或Python分析CSV数据文件")
        print("3. 查看生成的分析报告了解战术效果")

        return True

    except Exception as e:
        logging.error(f"钳形夹击仿真失败: {e}")
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_simulation_status(env, current_time: float, step_count: int):
    """打印仿真状态"""
    try:
        # 获取战术阶段
        phase = env.task.current_phase.value if hasattr(env.task, 'current_phase') else "Unknown"
        
        # 统计存活飞机
        alive_aircraft = sum(1 for agent in env.agents.values() if agent.is_alive)
        
        # 统计活跃导弹
        active_missiles = len(env._tempsims) if hasattr(env, '_tempsims') else 0
        
        print(f"t={current_time:6.1f}s | step={step_count:4d} | phase={phase:8s} | aircraft={alive_aircraft} | missiles={active_missiles}")
        
        # 打印飞机状态
        for agent_id, agent in env.agents.items():
            if agent.is_alive:
                pos = agent.get_position()
                heading = np.rad2deg(agent.get_property_value(c.attitude_psi_rad))
                altitude = pos[2]
                velocity = np.linalg.norm(agent.get_velocity())
                missiles = getattr(agent, 'num_missiles', 0)
                
                print(f"  {agent_id}: pos=({pos[0]/1000:5.1f}, {pos[1]/1000:5.1f}, {altitude/1000:4.1f})km, "
                      f"hdg={heading:5.1f}°, vel={velocity:5.1f}m/s, missiles={missiles}")
        
        # 计算双方距离
        red_pos = None
        blue_pos = None
        for agent_id, agent in env.agents.items():
            if agent.is_alive:
                if agent_id.startswith('A') and red_pos is None:
                    red_pos = agent.get_position()
                elif agent_id.startswith('B') and blue_pos is None:
                    blue_pos = agent.get_position()
        
        if red_pos is not None and blue_pos is not None:
            distance = np.linalg.norm(blue_pos - red_pos)
            print(f"  双方距离: {distance/1000:.1f}km")
        
    except Exception as e:
        logging.warning(f"状态打印失败: {e}")


def print_final_status(env):
    """打印最终状态"""
    print("\n最终飞机状态:")
    red_alive = 0
    blue_alive = 0
    
    for agent_id, agent in env.agents.items():
        if agent.is_alive:
            pos = agent.get_position()
            missiles = getattr(agent, 'num_missiles', 0)
            print(f"  {agent_id}: 存活, 位置({pos[0]/1000:5.1f}, {pos[1]/1000:5.1f}, {pos[2]/1000:4.1f})km, 剩余导弹{missiles}")
            if agent_id.startswith('A'):
                red_alive += 1
            else:
                blue_alive += 1
        else:
            print(f"  {agent_id}: 被击落")
    
    print(f"\n战斗结果:")
    print(f"  红方存活: {red_alive}/2")
    print(f"  蓝方存活: {blue_alive}/2")
    
    if red_alive > blue_alive:
        print("  🏆 红方获胜!")
    elif blue_alive > red_alive:
        print("  🏆 蓝方获胜!")
    else:
        print("  🤝 平局!")


def save_simulation_results(env, step_count: int, simulation_time: float):
    """保存仿真结果"""
    try:
        # 创建结果目录
        results_dir = "pincer_attack_results"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存摘要报告
        summary_file = os.path.join(results_dir, f"pincer_summary_{timestamp}.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("钳形夹击战术仿真摘要报告\n")
            f.write("=" * 40 + "\n")
            f.write(f"仿真时间: {simulation_time:.1f}秒\n")
            f.write(f"总步数: {step_count}\n")
            f.write(f"战术类型: 钳形夹击\n")
            f.write(f"生成时间: {datetime.now()}\n\n")
            
            f.write("最终飞机状态:\n")
            red_alive = 0
            blue_alive = 0
            for agent_id, agent in env.agents.items():
                if agent.is_alive:
                    pos = agent.get_position()
                    f.write(f"  {agent_id}: 存活, 位置({pos[0]/1000:.1f}, {pos[1]/1000:.1f}, {pos[2]/1000:.1f})km\n")
                    if agent_id.startswith('A'):
                        red_alive += 1
                    else:
                        blue_alive += 1
                else:
                    f.write(f"  {agent_id}: 被击落\n")
            
            f.write(f"\n战斗结果:\n")
            f.write(f"  红方存活: {red_alive}/2\n")
            f.write(f"  蓝方存活: {blue_alive}/2\n")
            
            if red_alive > blue_alive:
                f.write("  结果: 红方获胜\n")
            elif blue_alive > red_alive:
                f.write("  结果: 蓝方获胜\n")
            else:
                f.write("  结果: 平局\n")
        
        logging.info(f"仿真结果已保存到 {results_dir}")
        
    except Exception as e:
        logging.error(f"保存结果失败: {e}")


if __name__ == "__main__":
    print("=" * 80)
    print("🔱 钳形夹击战术仿真系统")
    print("=" * 80)
    print("功能: 生成完整的ACMI文件和CSV数据报告")
    print("输出: ACMI文件 + 轨迹CSV + 雷达CSV + 导弹CSV + 分析报告")
    print("=" * 80)

    success = run_pincer_attack_simulation()

    if success:
        print("\n🎉 钳形夹击仿真成功完成!")
        print("下一步可以:")
        print("1. 查看生成的CSV数据文件")
        print("2. 使用TacView查看ACMI文件")
        print("3. 运行数据分析脚本生成图表")
        print("4. 对比拖曳射击和钳形夹击的战术效果")
    else:
        print("\n❌ 仿真失败，请检查配置和日志")

    exit(0 if success else 1)
