#!/usr/bin/env python3
"""
战术多样性测试脚本
测试是否所有5种战术都能被选择
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.complete_tactical_system import CompleteTacticalSystem
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_tactical_variety():
    """测试战术选择的多样性"""
    
    print("🎯 测试战术选择多样性")
    print("=" * 60)
    
    # 测试不同的我方意图
    intents = ['AGGRESSIVE_CLEAR', 'CONSERVATIVE_CLEAR', 'DEFENSIVE']
    
    # 测试不同的控制距离
    distances = ['NLT', 'MELD', 'MTR']
    
    results = {}
    
    for intent in intents:
        print(f"\n📋 测试我方意图: {intent}")
        print("-" * 40)
        
        system = CompleteTacticalSystem(my_intent=intent)
        
        for distance in distances:
            print(f"\n  测试距离: {distance}")
            
            # 模拟飞机列表（简化）
            my_aircraft_list = [type('Aircraft', (), {'is_alive': True, 'get_position': lambda: [0, 0, 6000]})() for _ in range(2)]
            enemy_aircraft_list = [type('Aircraft', (), {'is_alive': True, 'get_position': lambda: [80000, 0, 6000]})() for _ in range(2)]
            
            # 模拟环境（简化）
            env = type('Env', (), {'current_step': 0, 'time_interval': 0.1})()
            
            # 进行多次测试（不同随机种子）
            tactics_chosen = set()
            
            for test_num in range(10):
                try:
                    tactic, roles = system.select_tactic(
                        control_distance=distance,
                        my_aircraft_list=my_aircraft_list,
                        enemy_aircraft_list=enemy_aircraft_list,
                        env=env
                    )
                    tactics_chosen.add(tactic)
                    print(f"    测试{test_num+1}: {tactic}")
                    
                    # 强制清空当前战术以允许重新选择
                    system.current_tactic = None
                    
                except Exception as e:
                    print(f"    测试{test_num+1}: ERROR - {e}")
            
            results[f"{intent}_{distance}"] = tactics_chosen
            print(f"    总共选择了 {len(tactics_chosen)} 种战术: {tactics_chosen}")
    
    # 统计结果
    print("\n" + "=" * 60)
    print("📊 统计结果")
    print("=" * 60)
    
    all_tactics = set()
    for key, tactics in results.items():
        all_tactics.update(tactics)
        print(f"{key}: {len(tactics)} 种战术 - {tactics}")
    
    print(f"\n总体统计：")
    print(f"- 总共出现了 {len(all_tactics)} 种不同战术")
    print(f"- 战术类型: {sorted(all_tactics)}")
    
    # 检查是否所有5种基础战术都出现了
    expected_tactics = {'DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'}
    missing_tactics = expected_tactics - all_tactics
    
    if missing_tactics:
        print(f"❌ 缺失的战术: {missing_tactics}")
        print("可能原因：决策表配置或战术评分问题")
    else:
        print(f"✅ 所有基础战术都出现了")
    
    return results, all_tactics

if __name__ == "__main__":
    test_tactical_variety()