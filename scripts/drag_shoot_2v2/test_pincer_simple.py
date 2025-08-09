#!/usr/bin/env python3
"""
简化的钳形夹击测试脚本
"""

import os
import sys
import logging

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

def test_basic_imports():
    """测试基本导入"""
    print("测试基本导入...")
    
    try:
        from tactical_factory import create_tactical_task, TacticalType
        print("✓ 战术工厂导入成功")
        
        from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
        print("✓ 环境导入成功")
        
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_config_file():
    """测试配置文件"""
    print("测试配置文件...")
    
    config_path = "../../envs/JSBSim/configs/pincer_attack_tactical.yaml"
    if os.path.exists(config_path):
        print("✓ 配置文件存在")
        return True
    else:
        print(f"✗ 配置文件不存在: {config_path}")
        return False

def test_tactical_task_creation():
    """测试战术任务创建"""
    print("测试战术任务创建...")
    
    try:
        from tactical_factory import create_tactical_task, TacticalType
        
        config = {
            'scenario': 'test_pincer',
            'pincer_config': {
                'crank_angle': 45.0,
                'max_off_boresight': 60.0,
            }
        }
        
        task = create_tactical_task(TacticalType.PINCER_ATTACK, config)
        print("✓ 钳形夹击任务创建成功")
        print(f"  任务类型: {task.get_tactical_type().value}")
        
        return True
    except Exception as e:
        print(f"✗ 任务创建失败: {e}")
        return False

def test_environment_creation():
    """测试环境创建"""
    print("测试环境创建...")
    
    try:
        from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv

        env = MultipleCombatEnv("pincer_attack_tactical")
        print("✓ 环境创建成功")
        
        return True
    except Exception as e:
        print(f"✗ 环境创建失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("钳形夹击简化测试")
    print("=" * 50)
    
    tests = [
        ("基本导入", test_basic_imports),
        ("配置文件", test_config_file),
        ("战术任务创建", test_tactical_task_creation),
        ("环境创建", test_environment_creation),
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
        print("所有测试通过！可以尝试运行完整仿真。")
    else:
        print("部分测试失败，需要修复问题。")

if __name__ == "__main__":
    main()
