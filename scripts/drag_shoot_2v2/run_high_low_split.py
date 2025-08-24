#!/usr/bin/env python3
"""
上下夹击战术仿真运行器 - 基于拖曳射击成功架构
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# 导入必要模块
from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.core.catalog import Catalog as c
from high_low_split_tactical_task import HighLowSplitTacticalTask


def run_high_low_split_simulation():
    """运行上下夹击仿真 - 基于拖曳射击架构"""
    print("=" * 60)
    print("上下夹击战术仿真 (High-Low Split Attack)")
    print("=" * 60)
    print("战术概述: 通过垂直分离创造立体攻击覆盖")
    print("核心特点: 僚机高度优势1200m, 时间线差异化, 分时发射, 分离脱离")
    print("=" * 60)
    
    try:
        # 创建环境 - 完全复制拖曳射击的方式
        print("创建仿真环境...")
        env = MultipleCombatEnv("drag_shoot_tactical")

        # 强制设置1500步（300秒）- 复制拖曳射击
        env.max_steps = 1500

        # 替换任务为上下夹击任务 - 完全复制拖曳射击的方式
        env.task = HighLowSplitTacticalTask(env.config)

        # 重置环境
        obs = env.reset()

        print("仿真环境初始化完成")
        
        # 仿真参数
        max_steps = 1500
        step_count = 0
        
        # 数据记录
        trajectory_data = []
        
        print(f"开始仿真，最大步数: {max_steps}")
        print()
        
        # 仿真循环 - 完全基于拖曳射击的成功模式
        while step_count < max_steps:
            # 使用虚拟动作，实际由任务类处理 - 与拖曳射击完全一致
            num_agents = len(env._jsbsims)
            dummy_actions = np.zeros((1, num_agents, 4))
            
            try:
                obs, share_obs, rewards, dones, info = env.step(dummy_actions)
            except Exception as e:
                print(f"步骤{step_count}执行失败: {e}")
                import traceback
                traceback.print_exc()
                break
            
            # 记录数据
            current_time = step_count * env.time_interval
            
            # 记录飞机轨迹数据
            for agent_id in env.agents:
                if env.agents[agent_id].is_alive:
                    try:
                        pos = env.agents[agent_id].get_position()
                        altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
                        velocity = env.agents[agent_id].get_property_value(c.velocities_u_fps) * 0.3048
                        heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
                        
                        trajectory_data.append({
                            'Time_s': current_time,
                            'Agent_ID': agent_id,
                            'X_m': pos[0],
                            'Y_m': pos[1], 
                            'Z_m': pos[2],
                            'Altitude_m': altitude,
                            'Velocity_m_s': velocity,
                            'Heading_deg': heading,
                            'Status': 'ALIVE'
                        })
                    except Exception as e:
                        print(f"数据记录失败 {agent_id}: {e}")
            
            # 状态输出
            if step_count % 250 == 0:  # 每50秒输出一次
                alive_agents = [aid for aid in env.agents if env.agents[aid].is_alive]
                print(f"t={current_time:6.1f}s | step={step_count:4d} | 存活: {len(alive_agents)}")
                
                # 输出高度信息和战术状态
                for agent_id in ["A0100", "A0200"]:
                    if agent_id in env.agents and env.agents[agent_id].is_alive:
                        try:
                            altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
                            pos = env.agents[agent_id].get_position()
                            
                            # 获取战术状态
                            tactical_info = ""
                            if hasattr(env.task, 'high_low_states') and agent_id in env.task.high_low_states:
                                state = env.task.high_low_states[agent_id]
                                target_alt = state["target_altitude"]
                                alt_established = "✓" if state["altitude_established"] else "✗"
                                tactical_info = f", 目标高度={target_alt:.0f}m, 高度建立={alt_established}"
                            
                            print(f"  {agent_id}: 高度={altitude:.0f}m, 位置=({pos[0]/1000:.1f}, {pos[1]/1000:.1f})km{tactical_info}")
                        except Exception as e:
                            print(f"  {agent_id}: 状态获取失败: {e}")
                
                # 输出当前战术阶段
                if hasattr(env.task, 'current_phase'):
                    print(f"  当前战术阶段: {env.task.current_phase.name}")
                print()
            
            # 检查终止条件
            alive_red = sum(1 for aid in ["A0100", "A0200"] if aid in env.agents and env.agents[aid].is_alive)
            alive_blue = sum(1 for aid in ["B0100", "B0200"] if aid in env.agents and env.agents[aid].is_alive)
            
            if alive_red == 0 or alive_blue == 0:
                print(f"仿真结束: 红方存活{alive_red}, 蓝方存活{alive_blue}")
                break
            
            step_count += 1
        
        # 保存结果
        if trajectory_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = "high_low_split_results"
            os.makedirs(results_dir, exist_ok=True)
            
            trajectory_file = os.path.join(results_dir, f"high_low_split_trajectory_{timestamp}.csv")
            df_trajectory = pd.DataFrame(trajectory_data)
            df_trajectory.to_csv(trajectory_file, index=False)
            print(f"轨迹数据已保存: {trajectory_file}")
        
        # 最终状态报告
        final_time = step_count * env.time_interval
        print(f"\n仿真完成!")
        print(f"  仿真时间: {final_time:.1f}秒")
        print(f"  总步数: {step_count}")
        
        # 最终飞机状态
        for agent_id in env.agents:
            if env.agents[agent_id].is_alive:
                try:
                    pos = env.agents[agent_id].get_position()
                    altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
                    print(f"  {agent_id}: 存活, 位置({pos[0]/1000:6.1f}, {pos[1]/1000:6.1f})km, 高度{altitude:.0f}m")
                except:
                    print(f"  {agent_id}: 存活")
            else:
                print(f"  {agent_id}: 被击落")
        
        # 战术效果分析
        analyze_tactical_effectiveness(trajectory_data)
        
        env.close()
        return True
        
    except Exception as e:
        print(f"仿真运行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_tactical_effectiveness(trajectory_data):
    """分析战术效果"""
    try:
        if not trajectory_data:
            return
        
        df = pd.DataFrame(trajectory_data)
        
        # 分析高度差维持情况
        red_agents = df[df['Agent_ID'].isin(['A0100', 'A0200'])]
        
        if len(red_agents) > 0:
            print("\n战术效果分析:")
            
            # 分析各时间段的高度差
            time_points = [50, 100, 150, 200, 250]
            for t in time_points:
                time_data = red_agents[abs(red_agents['Time_s'] - t) < 5]  # ±5秒范围
                if len(time_data) >= 2:
                    a0100_data = time_data[time_data['Agent_ID'] == 'A0100']
                    a0200_data = time_data[time_data['Agent_ID'] == 'A0200']
                    
                    if len(a0100_data) > 0 and len(a0200_data) > 0:
                        a0100_alt = a0100_data['Altitude_m'].mean()
                        a0200_alt = a0200_data['Altitude_m'].mean()
                        
                        if not (np.isnan(a0100_alt) or np.isnan(a0200_alt)):
                            alt_diff = a0200_alt - a0100_alt
                            print(f"  t={t}s: 长机{a0100_alt:.0f}m, 僚机{a0200_alt:.0f}m, 高度差{alt_diff:+.0f}m")
            
            # 分析最终高度差
            final_data = red_agents[red_agents['Time_s'] == red_agents['Time_s'].max()]
            a0100_final = final_data[final_data['Agent_ID'] == 'A0100']
            a0200_final = final_data[final_data['Agent_ID'] == 'A0200']
            
            if len(a0100_final) > 0 and len(a0200_final) > 0:
                final_a0100 = a0100_final['Altitude_m'].iloc[0]
                final_a0200 = a0200_final['Altitude_m'].iloc[0]
                final_diff = final_a0200 - final_a0100
                print(f"  最终高度差: {final_diff:+.0f}m")
                
                if abs(final_diff - 1200) < 200:  # 允许200m误差
                    print("  ✓ 高度差维持良好")
                else:
                    print("  ✗ 高度差偏离目标")
        
    except Exception as e:
        print(f"战术效果分析失败: {e}")


if __name__ == "__main__":
    try:
        success = run_high_low_split_simulation()
        if success:
            print("\n上下夹击战术仿真成功完成！")
        else:
            print("\n上下夹击战术仿真失败")
            sys.exit(1)
    except Exception as e:
        print(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
