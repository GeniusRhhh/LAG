#!/usr/bin/env python3
"""
基线测试 - 测试原始拖曳射击任务是否正常工作
"""

import os
import sys
import logging
import numpy as np
from datetime import datetime
import traceback

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

def test_baseline():
    """测试基线拖曳射击任务"""
    print("测试基线拖曳射击任务...")
    
    try:
        from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
        
        # 使用现有的拖曳射击配置
        env = MultipleCombatEnv("drag_shoot_tactical")
        print("✓ 环境创建成功")
        
        # 获取任务
        task = env.task
        print(f"✓ 任务类型: {type(task).__name__}")
        
        # 重置环境
        print("重置环境...")
        obs = env.reset()
        print("✓ 环境重置成功")
        
        # 运行仿真
        max_steps = 50  # 运行50步测试
        
        print("\n开始基线仿真...")
        print("=" * 60)
        
        for step in range(max_steps):
            print(f"\n--- 步骤 {step} ---")
            
            try:
                # 创建虚拟动作
                num_agents = len(env._jsbsims)
                actions = np.zeros((1, num_agents, 4))
                print(f"创建动作数组: shape={actions.shape}")
                
                # 直接调用env.step
                print("调用 env.step...")
                step_result = env.step(actions)
                print(f"env.step 返回: {type(step_result)}")
                
                if step_result is None:
                    print("ERROR: env.step 返回 None!")
                    break
                
                # 检查返回值的结构
                if isinstance(step_result, tuple):
                    print(f"返回元组长度: {len(step_result)}")
                    obs, share_obs, rewards, dones, infos = step_result
                    print("✓ 成功解包返回值")
                else:
                    print(f"ERROR: 返回值不是元组: {type(step_result)}")
                    break
                
                # 打印状态
                alive_count = sum(1 for agent in env.agents.values() if agent.is_alive)
                print(f"存活飞机: {alive_count}")
                
                # 打印飞机位置
                if step % 10 == 0:
                    for agent_id, agent in env.agents.items():
                        if agent.is_alive:
                            pos = agent.get_position()
                            from envs.JSBSim.core.catalog import Catalog as c
                            heading = np.rad2deg(agent.get_property_value(c.attitude_psi_rad))
                            print(f"  {agent_id}: pos=({pos[0]/1000:6.1f}, {pos[1]/1000:6.1f})km, hdg={heading:5.1f}°")
                
                # 检查终止条件
                if all(dones.values()):
                    print(f"仿真在第 {step} 步结束")
                    break
                    
            except Exception as e:
                print(f"步骤 {step} 执行失败: {e}")
                traceback.print_exc()
                break
        
        print("\n=" * 60)
        print("✓ 基线仿真完成")
        
        return True
        
    except Exception as e:
        print(f"✗ 基线测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("基线测试 - 原始拖曳射击任务")
    print("=" * 60)
    
    success = test_baseline()
    
    if success:
        print("\n🎉 基线测试成功！原始任务工作正常。")
    else:
        print("\n❌ 基线测试失败，原始任务有问题。")

if __name__ == "__main__":
    main()
