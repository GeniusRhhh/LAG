#!/usr/bin/env python3
"""
基础钳形夹击测试 - 使用现有环境和任务
"""

import os
import sys
import logging
import numpy as np
from datetime import datetime

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

def test_basic_simulation():
    """测试基础仿真"""
    print("测试基础钳形夹击仿真...")
    
    try:
        from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
        
        # 使用现有的配置文件
        env = MultipleCombatEnv("pincer_attack_tactical")
        print("✓ 环境创建成功")
        
        # 重置环境
        obs = env.reset()
        print("✓ 环境重置成功")
        
        # 运行几步测试
        max_steps = 50  # 只运行50步测试
        
        for step in range(max_steps):
            # 创建虚拟动作
            num_agents = len(env._jsbsims)
            actions = np.zeros((1, num_agents, 4))
            
            # 执行一步
            obs, share_obs, rewards, dones, infos = env.step(actions)
            
            # 打印状态
            if step % 10 == 0:
                print(f"步骤 {step}: 存活飞机 {sum(1 for agent in env.agents.values() if agent.is_alive)}")
            
            # 检查终止条件
            if all(dones.values()):
                print(f"仿真在第 {step} 步结束")
                break
        
        print("✓ 基础仿真测试成功")
        return True
        
    except Exception as e:
        print(f"✗ 基础仿真测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tactical_task_integration():
    """测试战术任务集成"""
    print("测试战术任务集成...")
    
    try:
        from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
        from tactical_factory import create_tactical_task, TacticalType
        
        # 创建环境
        env = MultipleCombatEnv("pincer_attack_tactical")
        
        # 创建钳形夹击任务
        config = {
            'scenario': 'test_integration',
            'pincer_config': {
                'crank_angle': 45.0,
                'max_off_boresight': 60.0,
            }
        }
        
        tactical_task = create_tactical_task(TacticalType.PINCER_ATTACK, config)
        print("✓ 钳形夹击任务创建成功")
        
        # 测试normalize_action方法
        env.reset()
        
        # 获取一个智能体ID
        agent_id = list(env.agents.keys())[0]
        
        # 测试normalize_action
        dummy_action = np.array([0.0, 0.0, 0.0, 0.7])
        normalized_action = tactical_task.normalize_action(env, agent_id, dummy_action)
        
        print(f"✓ normalize_action测试成功: {normalized_action.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 战术任务集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("钳形夹击基础测试")
    print("=" * 50)
    
    tests = [
        ("基础仿真", test_basic_simulation),
        ("战术任务集成", test_tactical_task_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ 测试异常: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("测试结果:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n通过率: {passed}/{len(results)} ({passed/len(results)*100:.1f}%)")
    
    if passed == len(results):
        print("所有测试通过！")
    else:
        print("部分测试失败，需要修复问题。")

if __name__ == "__main__":
    main()
