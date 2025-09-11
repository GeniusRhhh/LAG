#!/usr/bin/env python3
"""
上下夹击战术仿真运行器 - 基于拖曳射击的成功架构
完全复制拖曳射击的运行逻辑，只替换战术任务
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import io

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# 导入必要模块
from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.core.catalog import Catalog as c
from scripts.drag_shoot_2v2.high_low_attack_fixed import HighLowAttackTacticalTask


def record_simulation_data(env, current_time, trajectory_data, radar_data, missile_data, tactical_task=None):
    """记录仿真数据 - 完全复制拖曳射击"""
    from unified_data_recorder import UnifiedDataRecorder

    # 创建临时记录器实例
    temp_recorder = UnifiedDataRecorder("high_low_attack")

    # 记录所有数据，传递tactical_task参数以获取Action_Intent数据
    temp_recorder.record_all_data(env, current_time, tactical_task)

    # 将数据添加到现有列表中
    trajectory_data.extend(temp_recorder.trajectory_data)
    radar_data.extend(temp_recorder.radar_data)
    missile_data.extend(temp_recorder.missile_data)


def save_csv_data(output_dir, timestamp, trajectory_data, radar_data, missile_data, simulation_log=None):
    """保存CSV数据文件 - 完全复制拖曳射击"""
    from unified_data_recorder import UnifiedDataRecorder

    # 创建统一数据记录器
    recorder = UnifiedDataRecorder("high_low_attack")

    # 将数据添加到记录器
    recorder.trajectory_data = trajectory_data
    recorder.radar_data = radar_data
    recorder.missile_data = missile_data

    # 使用统一格式保存文件
    saved_files = recorder.save_csv_files(output_dir, timestamp, simulation_log)

    return saved_files


def run_high_low_attack_simulation():
    """运行上下夹击战术仿真 - 完全基于拖曳射击的成功架构"""
    # 捕获仿真日志
    captured_output = io.StringIO()
    original_stdout = sys.stdout
    
    try:
        # 重定向输出到捕获器
        sys.stdout = captured_output
        
        # 创建环境 - 完全复制拖曳射击
        logging.info("创建仿真环境...")
        env = MultipleCombatEnv("drag_shoot_tactical")
        
        # 替换任务为上下夹击战术任务
        logging.info("创建上下夹击战术任务...")
        tactical_task = HighLowAttackTacticalTask(env.config)
        env.task = tactical_task

        # 集成统一敌方AI系统
        logging.info("集成统一敌方AI系统...")
        from high_low_enemy_ai_adapter import create_high_low_enemy_ai_integration
        enemy_ai_adapter = create_high_low_enemy_ai_integration(tactical_task)
        if enemy_ai_adapter:
            logging.info("✅ 上下夹击统一敌方AI系统集成成功")
        else:
            logging.warning("⚠️ 上下夹击统一敌方AI系统集成失败，将使用默认敌方行为")
        logging.info("上下夹击战术任务设置完成")
        
        # 重置环境
        logging.info("重置环境...")
        obs = env.reset()
        
        # 创建输出目录 - 修复路径问题，使用绝对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "high_low_attack_results")
        os.makedirs(output_dir, exist_ok=True)
        
        # 仿真参数
        max_steps = 2500  # 500秒仿真
        time_interval = 0.2
        
        logging.info(f"开始上下夹击仿真 (最大步数: {max_steps}, 时间间隔: {time_interval}s)")
        
        # 准备ACMI文件路径和数据记录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        acmi_filepath = os.path.join(output_dir, f"high_low_attack_fixed_{timestamp}.acmi")
        
        # 初始化CSV数据记录
        trajectory_data = []
        radar_data = []
        missile_data = []
        
        # 获取智能体列表
        agents = list(env.agents.keys())
        
        # 恢复标准输出用于显示进度
        sys.stdout = original_stdout
        print("仿真环境初始化完成")
        print(f"智能体: {agents}")
        print(f"开始仿真，最大步数: {max_steps}")
        print("-" * 60)
        
        # 重新捕获输出
        sys.stdout = captured_output
        
        step_count = 0
        
        # 仿真循环 - 完全复制拖曳射击的成功模式
        while step_count < max_steps:
            current_time = step_count * time_interval
            
            # 使用虚拟动作，实际由任务类处理 - 与拖曳射击完全一致
            num_agents = len(env._jsbsims)
            dummy_actions = np.zeros((1, num_agents, 4))
            
            try:
                obs, share_obs, rewards, dones, info = env.step(dummy_actions)
            except Exception as e:
                logging.error(f"步骤{step_count}执行失败: {e}")
                import traceback
                traceback.print_exc()
                break

            step_count += 1

            # 每步都渲染ACMI - 这是关键！
            try:
                env.render(mode="txt", filepath=acmi_filepath)
            except Exception as e:
                logging.warning(f"Failed to render step {step_count}: {e}")

            # 记录数据
            try:
                record_simulation_data(env, current_time, trajectory_data, radar_data, missile_data, env.task)
            except Exception as e:
                logging.warning(f"Failed to record data at step {step_count}: {e}")

            # 每50步打印一次状态
            if step_count % 50 == 0:
                # 临时恢复输出显示进度
                sys.stdout = original_stdout
                alive_count = sum(1 for agent_id in agents if env.agents[agent_id].is_alive)
                missile_count = len(env._tempsims)
                phase = getattr(env.task, 'current_phase', 'UNKNOWN')
                print(f"步数: {step_count}, 时间: {current_time:6.1f}s, 存活: {alive_count}, 导弹: {missile_count}, 阶段: {phase}")
                sys.stdout = captured_output

            # 检查终止条件
            if isinstance(dones, dict):
                all_done = all(dones.values())
            else:
                all_done = all(dones)

            if all_done:
                logging.info(f"仿真提前结束 at step {step_count}")
                break
        
        # 恢复标准输出并获取捕获的内容
        sys.stdout = original_stdout
        simulation_log = captured_output.getvalue()
        
        print(f"仿真在第{step_count}步终止")
        print("-" * 60)
        print("仿真完成")
        
        # 保存CSV数据
        try:
            saved_files = save_csv_data(output_dir, timestamp, trajectory_data, radar_data, missile_data, simulation_log)
            print("CSV数据文件已保存:")
            for file_type, file_path in saved_files.items():
                print(f"  {file_type}: {file_path}")

            # 动作标注系统已禁用 - 生成纯净轨迹数据
            print("✅ 上下夹击纯净轨迹数据生成完成")

        except Exception as e:
            print(f"CSV数据保存失败: {e}")
        
        # ACMI文件已在每步生成
        print(f"ACMI文件已保存: {acmi_filepath}")
        
        # 分析结果
        analyze_high_low_attack_results(trajectory_data)
        
        return True
        
    except KeyboardInterrupt:
        sys.stdout = original_stdout  # 恢复输出
        print("\n仿真被用户中断")
        return False
        
    except Exception as e:
        sys.stdout = original_stdout  # 恢复输出
        print(f"仿真运行失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 确保输出总是被恢复
        sys.stdout = original_stdout


def analyze_high_low_attack_results(trajectory_data):
    """分析上下夹击战术结果"""
    if not trajectory_data:
        print("⚠️ 没有轨迹数据可供分析")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(trajectory_data)
    
    # 计算平均高度
    leader_data = df[df['Agent_ID'] == 'A0100']
    wingman_data = df[df['Agent_ID'] == 'A0200']
    
    if not leader_data.empty and not wingman_data.empty:
        leader_avg_alt = leader_data['Z_m'].mean()
        wingman_avg_alt = wingman_data['Z_m'].mean()
        altitude_diff = wingman_avg_alt - leader_avg_alt
        
        print("=" * 60)
        print("上下夹击战术分析")
        print("=" * 60)
        print(f"长机平均高度: {leader_avg_alt:.0f}m")
        print(f"僚机平均高度: {wingman_avg_alt:.0f}m")
        print(f"高度差: {altitude_diff:.0f}m")
        
        if altitude_diff > 1500:
            print("✅ 高度优势建立成功")
        elif altitude_diff > 800:
            print("⚠️ 高度优势建立部分成功")
        else:
            print("❌ 高度优势建立失败")
    
    print("\n仿真分析完成")


if __name__ == "__main__":
    print("=" * 60)
    print("上下夹击战术仿真 (基于拖曳射击架构)")
    print("=" * 60)
    
    success = run_high_low_attack_simulation()
    
    if success:
        print("✅ 仿真成功完成")
    else:
        print("❌ 仿真失败")
        sys.exit(1)
