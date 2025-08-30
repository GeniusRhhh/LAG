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


def _get_missile_status(missile_sim) -> str:
    """获取导弹的正确状态"""
    try:
        # 检查导弹是否存活
        if hasattr(missile_sim, 'is_alive') and not missile_sim.is_alive:
            # 导弹已经爆炸，检查是否击中
            if hasattr(missile_sim, 'is_success') and missile_sim.is_success:
                return 'HIT'
            else:
                return 'MISS'

        # 导弹仍在飞行，检查飞行阶段
        if hasattr(missile_sim, '_phase'):
            phase = missile_sim._phase
            if hasattr(missile_sim, 'BOOST_PHASE') and phase == missile_sim.BOOST_PHASE:
                return 'BOOST'
            elif hasattr(missile_sim, 'MIDCOURSE_PHASE') and phase == missile_sim.MIDCOURSE_PHASE:
                return 'MIDCOURSE'
            elif hasattr(missile_sim, 'TERMINAL_PHASE') and phase == missile_sim.TERMINAL_PHASE:
                return 'TERMINAL'

        # 检查导弹状态属性
        if hasattr(missile_sim, '_R27ERMissileSimulator__status'):
            status = missile_sim._R27ERMissileSimulator__status
            if hasattr(missile_sim, 'LAUNCHED') and status == missile_sim.LAUNCHED:
                return 'LAUNCHED'
            elif hasattr(missile_sim, 'HIT') and status == missile_sim.HIT:
                return 'HIT'
            elif hasattr(missile_sim, 'MISS') and status == missile_sim.MISS:
                return 'MISS'

        # 默认状态
        return 'ACTIVE'

    except Exception as e:
        logging.warning(f"获取导弹状态失败: {e}")
        return 'UNKNOWN'


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
    """记录钳形夹击仿真数据 - 使用统一数据记录器"""
    # 使用统一数据记录器
    from unified_data_recorder import UnifiedDataRecorder

    # 创建临时记录器实例
    temp_recorder = UnifiedDataRecorder("pincer_attack")

    # 记录所有数据
    temp_recorder.record_aircraft_trajectory(env, current_time)
    temp_recorder.record_radar_data(env, current_time)
    temp_recorder.record_missile_data(env, current_time)

    # 将数据添加到现有列表中
    trajectory_data.extend(temp_recorder.trajectory_data)
    radar_data.extend(temp_recorder.radar_data)
    missile_data.extend(temp_recorder.missile_data)


def save_pincer_csv_data(output_dir, timestamp, trajectory_data, radar_data, missile_data, simulation_log=None, tactical_task=None):
    """保存钳形夹击CSV数据文件 - 纯净轨迹数据"""
    from unified_data_recorder import UnifiedDataRecorder
    # from pincer_tactical_action_extractor import PincerTacticalActionExtractor  # 已禁用动作标注系统

    # 创建统一数据记录器
    recorder = UnifiedDataRecorder("pincer_attack")

    # 将数据添加到记录器
    recorder.trajectory_data = trajectory_data
    recorder.radar_data = radar_data
    recorder.missile_data = missile_data

    # 使用统一格式保存文件 - 传入仿真日志
    saved_files = recorder.save_csv_files(output_dir, timestamp, simulation_log)

    # 动作标注系统已禁用 - 生成纯净轨迹数据
    print("✅ 钳形攻击纯净轨迹数据生成完成")

    return saved_files


def generate_pincer_action_analysis_report(trajectory_df, output_dir, timestamp):
    """生成钳形攻击动作标注分析报告 - 已禁用"""
    # 动作分析报告已禁用
    return


def get_missile_final_status_and_time(missile_traj):
    """获取导弹的真正最终状态和结束时间"""
    # 按时间排序确保顺序正确
    missile_traj = missile_traj.sort_values('Time_s')

    # 查找HIT状态
    hit_records = missile_traj[missile_traj['Status'] == 'HIT']
    if not hit_records.empty:
        # 如果有HIT记录，使用第一个HIT记录的时间
        hit_time = hit_records['Time_s'].iloc[0]
        return 'HIT', hit_time

    # 查找MISS状态
    miss_records = missile_traj[missile_traj['Status'] == 'MISS']
    if not miss_records.empty:
        # 如果有MISS记录，使用第一个MISS记录的时间
        miss_time = miss_records['Time_s'].iloc[0]
        return 'MISS', miss_time

    # 查找其他终止状态
    terminal_states = ['TIMEOUT', 'DESTROYED', 'LOST']
    for state in terminal_states:
        state_records = missile_traj[missile_traj['Status'] == state]
        if not state_records.empty:
            state_time = state_records['Time_s'].iloc[0]
            return state, state_time

    # 如果没有找到明确的终止状态，检查是否有状态变化
    # 从活跃状态（BOOST, MIDCOURSE, TERMINAL, ACTIVE）变为非活跃状态
    active_states = ['BOOST', 'MIDCOURSE', 'TERMINAL', 'ACTIVE']

    # 找到最后一个活跃状态的记录
    active_records = missile_traj[missile_traj['Status'].isin(active_states)]
    if not active_records.empty:
        last_active_time = active_records['Time_s'].iloc[-1]
        last_active_status = active_records['Status'].iloc[-1]

        # 检查在最后活跃时间之后是否有其他记录
        after_active = missile_traj[missile_traj['Time_s'] > last_active_time]
        if not after_active.empty:
            # 有后续记录，使用后续记录的第一个状态和时间
            next_status = after_active['Status'].iloc[0]
            next_time = after_active['Time_s'].iloc[0]
            return next_status, next_time
        else:
            # 没有后续记录，导弹可能仍在飞行，使用最后记录
            return last_active_status, last_active_time

    # 如果都没有，使用最后一条记录
    final_status = missile_traj['Status'].iloc[-1]
    final_time = missile_traj['Time_s'].iloc[-1]
    return final_status, final_time


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

        # 获取真正的最终状态和结束时间
        final_status, final_time = get_missile_final_status_and_time(missile_traj)
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

        # 计算最小距离到目标
        if 'Range_to_Target_km' in missile_traj.columns:
            min_distance_to_target = missile_traj['Range_to_Target_km'].min()
        else:
            min_distance_to_target = 0.0

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
            'Min_Distance_to_Target_km': min_distance_to_target,
            'Final_Status': final_status
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
            # 统计击中情况
            hit_missiles = len(analysis_df[analysis_df['Final_Status'] == 'HIT'])
            miss_missiles = len(analysis_df[analysis_df['Final_Status'] == 'MISS'])
            active_missiles = len(analysis_df[analysis_df['Final_Status'] == 'ACTIVE'])
            other_missiles = len(analysis_df) - hit_missiles - miss_missiles - active_missiles

            f.write("导弹击中统计:\n")
            f.write(f"  导弹击中目标: {hit_missiles}\n")
            f.write(f"  导弹未击中目标: {miss_missiles}\n")
            f.write(f"  仍在飞行导弹: {active_missiles}\n")
            if other_missiles > 0:
                f.write(f"  其他状态导弹: {other_missiles}\n")

            # 计算命中率
            if hit_missiles + miss_missiles > 0:
                hit_rate = hit_missiles / (hit_missiles + miss_missiles) * 100
                f.write(f"  导弹命中率: {hit_rate:.1f}% ({hit_missiles}/{hit_missiles + miss_missiles})\n")
            f.write("\n")

            f.write("导弹性能统计:\n")
            f.write(f"  平均飞行时间: {analysis_df['Flight_Duration_s'].mean():.2f}秒\n")
            f.write(f"  平均飞行距离: {analysis_df['Total_Distance_m'].mean()/1000:.2f}km\n")
            f.write(f"  平均速度: {analysis_df['Average_Velocity_m_s'].mean():.2f}m/s\n")
            f.write(f"  最大速度: {analysis_df['Max_Velocity_m_s'].max():.2f}m/s\n\n")

            f.write("各导弹详情:\n")
            for _, missile in analysis_df.iterrows():
                f.write(f"  {missile['Missile_ID']}: {missile['Launcher_ID']} -> {missile['Target_ID']}, "
                       f"飞行{missile['Flight_Duration_s']:.1f}s, "
                       f"距离{missile['Total_Distance_m']/1000:.1f}km, "
                       f"状态: {missile['Final_Status']}\n")

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
    import io
    import sys

    setup_logging()
    print_tactical_info()

    # 捕获仿真输出
    captured_output = io.StringIO()
    original_stdout = sys.stdout

    try:
        # 重定向输出到捕获器
        sys.stdout = captured_output
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

        # 创建输出目录 - 修正问题1：使用正确的目录结构
        # 使用脚本所在目录的绝对路径，避免相对路径问题
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "pincer_attack_results")
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

        # 恢复标准输出并获取捕获的内容
        sys.stdout = original_stdout
        simulation_log = captured_output.getvalue()

        # 保存CSV数据 - 传入仿真日志和战术任务
        save_pincer_csv_data(output_dir, timestamp, trajectory_data, radar_data, missile_data, simulation_log, tactical_task)

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
        sys.stdout = original_stdout  # 恢复输出
        logging.error(f"钳形夹击仿真失败: {e}")
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 确保输出总是被恢复
        sys.stdout = original_stdout


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
        # 创建结果目录 - 修正问题1：使用正确的目录结构
        # 使用脚本所在目录的绝对路径，避免相对路径问题
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, "pincer_attack_results")
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
