#!/usr/bin/env python3
"""
上下夹击战术仿真运行器V2 - 完全基于拖曳射击的成功架构
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
from high_low_split_v2_tactical_task import HighLowSplitV2TacticalTask


def run_high_low_split_v2_simulation():
    """运行上下夹击V2仿真 - 完全基于拖曳射击架构"""
    print("=" * 60)
    print("上下夹击战术仿真V2 (High-Low Split Attack V2)")
    print("=" * 60)
    print("基于拖曳射击的成功架构，确保系统稳定运行")
    print("=" * 60)
    
    try:
        # 创建环境 - 完全复制拖曳射击的方式
        print("创建仿真环境...")
        env = MultipleCombatEnv("drag_shoot_tactical")
        
        # 强制设置1500步（300秒）- 复制拖曳射击
        env.max_steps = 1500

        # 替换任务为上下夹击任务V2 - 完全复制拖曳射击的方式
        env.task = HighLowSplitV2TacticalTask(env.config)
        
        # 重置环境
        obs = env.reset()
        
        print("仿真环境初始化完成")
        
        # 仿真参数
        max_steps = 1500
        step_count = 0

        # 准备ACMI文件路径和数据记录 - 完全复制拖曳射击
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "high_low_split_v2_results"
        os.makedirs(output_dir, exist_ok=True)
        acmi_filepath = os.path.join(output_dir, f"high_low_split_v2_{timestamp}.acmi")

        # 初始化CSV数据记录 - 使用统一数据记录器
        trajectory_data = []
        radar_data = []
        missile_data = []
        
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

            # 每步都渲染ACMI - 完全复制拖曳射击
            try:
                env.render(mode="txt", filepath=acmi_filepath)
            except Exception as e:
                logging.warning(f"Failed to render step {step_count}: {e}")

            # 记录数据 - 使用统一数据记录器
            try:
                from unified_data_recorder import UnifiedDataRecorder
                temp_recorder = UnifiedDataRecorder("high_low_split_v2")
                temp_recorder.record_aircraft_trajectory(env, current_time)
                temp_recorder.record_radar_data(env, current_time)
                temp_recorder.record_missile_data(env, current_time)

                # 将数据添加到现有列表中
                trajectory_data.extend(temp_recorder.trajectory_data)
                radar_data.extend(temp_recorder.radar_data)
                missile_data.extend(temp_recorder.missile_data)
            except Exception as e:
                logging.warning(f"Failed to record data at step {step_count}: {e}")
            
            # 数据记录已在上面的统一数据记录器中完成
            
            # 状态输出
            if step_count % 250 == 0:  # 每50秒输出一次
                alive_agents = [aid for aid in env.agents if env.agents[aid].is_alive]
                print(f"t={current_time:6.1f}s | step={step_count:4d} | 存活: {len(alive_agents)}")
                
                # 输出高度信息
                for agent_id in ["A0100", "A0200"]:
                    if agent_id in env.agents and env.agents[agent_id].is_alive:
                        try:
                            altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
                            pos = env.agents[agent_id].get_position()
                            print(f"  {agent_id}: 高度={altitude:.0f}m, 位置=({pos[0]/1000:.1f}, {pos[1]/1000:.1f})km")
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
        
        # 保存结果 - 使用完整的数据保存系统
        try:
            from unified_data_recorder import UnifiedDataRecorder

            # 创建统一数据记录器
            recorder = UnifiedDataRecorder("high_low_split_v2")

            # 将数据添加到记录器
            recorder.trajectory_data = trajectory_data
            recorder.radar_data = radar_data
            recorder.missile_data = missile_data

            # 创建仿真日志
            simulation_log = []
            simulation_log.append(f"上下夹击战术仿真V2完成")
            simulation_log.append(f"仿真时间: {step_count * env.time_interval:.1f}秒")
            simulation_log.append(f"总步数: {step_count}")

            # 使用统一格式保存文件
            saved_files = recorder.save_csv_files(output_dir, timestamp, simulation_log)

            print(f"数据文件已保存:")
            for file_path in saved_files:
                print(f"  {file_path}")

        except Exception as e:
            print(f"数据保存失败: {e}")
            # 备用保存方法
            if trajectory_data:
                trajectory_file = os.path.join(output_dir, f"high_low_split_v2_trajectory_{timestamp}.csv")
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

        # ACMI文件已在每步生成
        print(f"\nACMI文件已生成: {acmi_filepath}")
        print("CSV数据文件已保存")
        print("=" * 60)
        print("仿真成功完成!")
        print(f"数据文件保存在: {output_dir}")
        print(f"ACMI文件: {acmi_filepath}")
        print("可以使用TacView查看ACMI文件")
        print("=" * 60)

        env.close()
        return True
        
    except Exception as e:
        print(f"仿真运行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = run_high_low_split_v2_simulation()
        if success:
            print("\n上下夹击战术仿真V2成功完成！")
        else:
            print("\n上下夹击战术仿真V2失败")
            sys.exit(1)
    except Exception as e:
        print(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
