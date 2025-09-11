#!/usr/bin/env python3
"""
Action_Intent数据记录测试脚本
运行100步测试，验证CSV文件是否包含Action_Intent列
"""

import os
import sys
import time
import logging
import numpy as np
from datetime import datetime

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)

from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from side_by_side_shooting_tactical_task import SideBySideShootingTacticalTask

# 设置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_action_intent_recording():
    """测试Action_Intent数据记录"""
    print("🧪 开始Action_Intent数据记录测试...")
    
    try:
        # 创建环境
        print("📝 创建仿真环境...")
        config_name = "side_by_side_shooting_tactical"
        env = MultipleCombatEnv(config_name)
        
        # 设置测试步数
        test_steps = 100
        env.max_steps = test_steps
        
        # 创建战术任务
        print("📝 创建并排射击战术任务...")
        tactical_task = SideBySideShootingTacticalTask(env.config)
        env.task = tactical_task
        
        # 集成统一敌方AI系统
        print("📝 集成统一敌方AI系统...")
        from side_by_side_enemy_ai_adapter import create_side_by_side_enemy_ai_integration
        enemy_ai_adapter = create_side_by_side_enemy_ai_integration(tactical_task)
        if enemy_ai_adapter:
            print("✅ 统一敌方AI系统集成成功")
            
            # 测试适配器的注释功能
            print("🧪 测试适配器注释功能...")
            for agent_id in ["B0100", "B0200"]:
                try:
                    annotation = enemy_ai_adapter.get_action_annotation_for_csv(agent_id)
                    print(f"  {agent_id}: {annotation}")
                except Exception as e:
                    print(f"  {agent_id}: 错误 - {e}")
        else:
            print("❌ 统一敌方AI系统集成失败")
            return False
        
        # 重置环境
        print("📝 重置环境...")
        obs = env.reset()
        
        # 创建输出目录
        output_dir = os.path.join(current_dir, "action_intent_test_results")
        os.makedirs(output_dir, exist_ok=True)

        # 创建数据记录器
        from unified_data_recorder import UnifiedDataRecorder
        data_recorder = UnifiedDataRecorder("action_intent_test")
        
        print(f"🚀 开始运行{test_steps}步测试...")
        
        for step in range(test_steps):
            current_time = step * env.time_interval
            
            # 每10步打印状态
            if step % 10 == 0:
                print(f"  步骤 {step}/{test_steps} (时间: {current_time:.1f}s)")
                
                # 测试敌方AI注释
                for agent_id in ["B0100", "B0200"]:
                    if agent_id in env.agents and env.agents[agent_id].is_alive:
                        try:
                            if hasattr(tactical_task, 'unified_enemy_ai'):
                                annotation = tactical_task.unified_enemy_ai.get_action_annotation_for_csv(agent_id)
                                print(f"    {agent_id} 注释: {annotation}")
                            else:
                                print(f"    {agent_id}: 未找到统一敌方AI")
                        except Exception as e:
                            print(f"    {agent_id} 注释错误: {e}")
            
            # 生成虚拟动作
            num_agents = len(env._jsbsims)
            dummy_actions = np.zeros((1, num_agents, 4))
            
            try:
                obs, share_obs, rewards, dones, info = env.step(dummy_actions)
            except Exception as e:
                print(f"❌ 步骤{step}执行失败: {e}")
                break
            
            # 记录数据
            try:
                data_recorder.record_all_data(env, current_time, tactical_task)
            except Exception as e:
                print(f"⚠️ 数据记录失败 步骤{step}: {e}")
        
        print("📊 保存CSV数据...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存CSV文件
        try:
            saved_files = data_recorder.save_csv_files(output_dir, timestamp)
            print(f"✅ CSV文件已保存: {saved_files}")

            # 检查轨迹CSV文件
            trajectory_file = None
            for file_path in saved_files.values():
                if 'trajectory' in file_path.lower():
                    trajectory_file = file_path
                    break
            
            if trajectory_file and os.path.exists(trajectory_file):
                print(f"🔍 检查轨迹CSV文件: {trajectory_file}")
                
                # 读取CSV文件头部
                with open(trajectory_file, 'r', encoding='utf-8') as f:
                    header = f.readline().strip()
                    print(f"  CSV头部: {header}")
                    
                    # 检查是否包含Action_Intent列
                    if 'Action_Intent' in header:
                        print("✅ CSV文件包含Action_Intent列")
                        
                        # 读取几行数据查看内容
                        print("📋 前几行数据:")
                        for i, line in enumerate(f):
                            if i >= 5:  # 只显示前5行数据
                                break
                            print(f"  行{i+2}: {line.strip()}")
                    else:
                        print("❌ CSV文件缺少Action_Intent列")
                        print(f"  实际列: {header.split(',')}")
            else:
                print("❌ 未找到轨迹CSV文件")
                
        except Exception as e:
            print(f"❌ CSV保存失败: {e}")
            import traceback
            traceback.print_exc()
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_action_intent_recording()
    if success:
        print("🎉 测试完成")
    else:
        print("💥 测试失败")
