#!/usr/bin/env python3
"""
测试意图识别数据生成
验证17动作类型到4意图标签的映射是否正确
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_enemy_tactical_ai import UnifiedEnemyTacticalAI, ActionType

def test_intent_mapping():
    """测试意图映射是否正确"""
    print("=" * 60)
    print("测试意图识别映射表")
    print("=" * 60)
    
    # 创建AI实例
    ai = UnifiedEnemyTacticalAI()
    
    # 预期的映射关系
    expected_mapping = {
        ActionType.MAINTAIN_HEADING: "RECONNAISSANCE",
        ActionType.TURN_LEFT: "DEFENSE",
        ActionType.TURN_RIGHT: "DEFENSE",
        ActionType.CLIMB: "DEFENSE",
        ActionType.DESCEND: "DEFENSE",
        ActionType.ACCELERATE: "ATTACK",
        ActionType.DECELERATE: "DEFENSE",
        ActionType.CRANK_LEFT: "ATTACK",
        ActionType.CRANK_RIGHT: "ATTACK",
        ActionType.NOTCH_MANEUVER: "DEFENSE",
        ActionType.BEAM_MANEUVER: "DEFENSE",
        ActionType.DIVE_ESCAPE: "DEFENSE",
        ActionType.CHAFF_FLARE_MANEUVER: "DEFENSE",
        ActionType.SPIRAL_DIVE: "DEFENSE",
        ActionType.SHORT_SKATE: "DEFENSE",
        ActionType.DEFENSIVE_SPLIT: "DEFENSE",
        ActionType.AGGRESSIVE_APPROACH: "ATTACK",
        ActionType.RETURN_TO_BASE: "RETREAT"
    }
    
    # 验证映射
    all_correct = True
    print(f"\n{'序号':<4} {'动作类型':<30} {'意图标签':<15} {'状态':<10}")
    print("-" * 60)
    
    for idx, (action_type, expected_intent) in enumerate(expected_mapping.items(), 1):
        actual_intent = ai.action_to_intent_mapping.get(action_type, "MISSING")
        status = "✅" if actual_intent == expected_intent else "❌"
        
        if actual_intent != expected_intent:
            all_correct = False
        
        print(f"{idx:<4} {action_type.value:<30} {actual_intent:<15} {status:<10}")
    
    print("-" * 60)
    
    # 统计各意图标签的动作数量
    intent_counts = {}
    for intent in ai.action_to_intent_mapping.values():
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    print(f"\n意图标签统计:")
    for intent, count in sorted(intent_counts.items()):
        print(f"  {intent}: {count}个动作")
    
    print(f"\n总计: {len(ai.action_to_intent_mapping)}个动作类型")
    
    if all_correct:
        print("\n✅ 所有映射正确!")
    else:
        print("\n❌ 存在映射错误!")
    
    return all_correct

def test_get_current_intent():
    """测试获取当前意图的方法"""
    print("\n" + "=" * 60)
    print("测试获取当前意图方法")
    print("=" * 60)
    
    ai = UnifiedEnemyTacticalAI()
    
    # 模拟设置不同的动作
    test_cases = [
        (ActionType.CRANK_LEFT, "ATTACK"),
        (ActionType.TURN_LEFT, "DEFENSE"),
        (ActionType.SHORT_SKATE, "DEFENSE"),
        (ActionType.MAINTAIN_HEADING, "RECONNAISSANCE"),
        (ActionType.RETURN_TO_BASE, "RETREAT")
    ]
    
    print(f"\n{'动作类型':<30} {'预期意图':<15} {'实际意图':<15} {'状态':<10}")
    print("-" * 70)
    
    all_correct = True
    for action_type, expected_intent in test_cases:
        # 设置当前动作
        ai.current_action['B0100'] = action_type
        
        # 获取意图
        actual_intent = ai.get_current_intent('B0100')
        
        status = "✅" if actual_intent == expected_intent else "❌"
        if actual_intent != expected_intent:
            all_correct = False
        
        print(f"{action_type.value:<30} {expected_intent:<15} {actual_intent:<15} {status:<10}")
    
    print("-" * 70)
    
    if all_correct:
        print("\n✅ 意图获取方法正确!")
    else:
        print("\n❌ 意图获取方法存在错误!")
    
    return all_correct

def main():
    """主测试函数"""
    print("\n🎯 意图识别数据生成测试\n")
    
    # 测试1: 映射表
    test1_passed = test_intent_mapping()
    
    # 测试2: 获取意图方法
    test2_passed = test_get_current_intent()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"映射表测试: {'✅ 通过' if test1_passed else '❌ 失败'}")
    print(f"意图获取测试: {'✅ 通过' if test2_passed else '❌ 失败'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 所有测试通过!")
        return 0
    else:
        print("\n❌ 部分测试失败!")
        return 1

if __name__ == "__main__":
    exit(main())
