#!/usr/bin/env python3
"""
前后攻击战术简单测试 - 最小化版本
"""

import os
import sys
import logging
import numpy as np

# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

def test_simple():
    """简单测试"""
    print("=" * 60)
    print("前后攻击战术简单测试")
    print("=" * 60)
    
    try:
        # 导入必要模块
        from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
        from envs.JSBSim.core.catalog import Catalog as c
        print("✅ 模块导入成功")
        
        # 创建环境
        print("创建仿真环境...")
        env = MultipleCombatEnv("drag_shoot_tactical")
        print("✅ 环境创建成功")
        
        # 导入前后攻击任务
        from front_back_attack_tactical_task_v2 import FrontBackAttackTacticalTaskV2
        print("✅ 前后攻击任务导入成功")
        
        # 创建任务
        print("创建前后攻击任务...")
        task = FrontBackAttackTacticalTaskV2(env.config)
        print("✅ 任务创建成功")
        
        # 替换任务
        env.task = task
        print("✅ 任务替换成功")
        
        # 重置环境
        print("重置环境...")
        obs = env.reset()
        print("✅ 环境重置成功")
        
        # 运行几步测试
        print("运行测试步骤...")
        for step in range(10):
            actions = {}
            for agent_id in env.agents.keys():
                if env.agents[agent_id].is_alive:
                    try:
                        # 使用战术任务的normalize_action
                        action = env.task.normalize_action(env, agent_id, None)
                        actions[agent_id] = action
                        print(f"✅ {agent_id} 动作生成成功: {action}")
                    except Exception as e:
                        print(f"❌ {agent_id} 动作生成失败: {e}")
                        actions[agent_id] = np.array([0.0, 0.0, 0.0, 0.7])
                else:
                    actions[agent_id] = np.array([0.0, 0.0, 0.0, 0.0])
            
            # 环境步进
            try:
                obs, rewards, dones, infos = env.step(actions)
                current_time = step * env.time_interval
                alive_count = sum(1 for agent in env.agents.values() if agent.is_alive)
                print(f"步骤 {step}: t={current_time:.1f}s, 存活={alive_count}/4, 阶段={env.task.current_phase.value}")
            except Exception as e:
                print(f"❌ 步骤 {step} 执行失败: {e}")
                break
        
        env.close()
        print("✅ 简单测试完成")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple()
