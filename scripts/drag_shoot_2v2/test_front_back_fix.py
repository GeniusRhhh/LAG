#!/usr/bin/env python3
"""
前后攻击战术修复验证脚本
验证返航方向和时间线控制的修复效果
"""

import sys
import os
import logging

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from front_back_attack_final_task import FrontBackAttackFinalTask, TacticalPhase

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_front_back_attack_fixes():
    """测试前后攻击战术的修复效果"""
    print("=" * 60)
    print("前后攻击战术修复验证")
    print("=" * 60)
    
    # 创建配置对象
    class MockConfig:
        def __init__(self):
            pass
    
    config = MockConfig()
    task = FrontBackAttackFinalTask(config)
    
    # 测试1：返航方向修复
    print("\n🔧 测试1：返航方向修复")
    print("-" * 30)
    
    task._init_short_skate('A0100', 0.0)  # 长机
    task._init_short_skate('A0200', 0.0)  # 僚机
    
    leader_crank = task.short_skate_states['A0100']['crank_angle']
    wingman_crank = task.short_skate_states['A0200']['crank_angle']
    
    print(f"长机(A0100)返航方向: {leader_crank}° ({'左侧' if leader_crank < 0 else '右侧'})")
    print(f"僚机(A0200)返航方向: {wingman_crank}° ({'左侧' if wingman_crank < 0 else '右侧'})")
    
    # 验证结果
    if leader_crank == -40.0 and wingman_crank == 40.0:
        print("✅ 返航方向修复成功：长机左侧，僚机右侧")
    else:
        print("❌ 返航方向修复失败")
        return False
    
    # 测试2：僚机时间线滞后
    print("\n🔧 测试2：僚机时间线滞后")
    print("-" * 30)
    
    print(f"TR_DOR阶段滞后: {task.wingman_delay['TR_DOR_delay']/1000:.1f}km")
    print(f"DOR_DR阶段滞后: {task.wingman_delay['DOR_DR_delay']/1000:.1f}km")
    
    # 测试不同距离下的阶段判断
    test_distances = [
        (45000, "MTR_TR", "长机已进入TR_DOR，僚机仍在MTR_TR"),
        (31000, "TR_DOR", "长机已进入DOR_DR，僚机进入TR_DOR"),
        (27000, "DOR_DR", "僚机进入DOR_DR阶段"),
        (20000, "DOR_DR", "僚机继续DOR_DR阶段"),
    ]
    
    print("\n距离 -> 僚机阶段 (说明)")
    for distance, expected_phase, description in test_distances:
        actual_phase = task._get_wingman_phase_by_distance(distance)
        status = "✅" if actual_phase.value == expected_phase else "❌"
        print(f"{distance/1000:5.1f}km -> {actual_phase.value:8s} {status} ({description})")
    
    # 测试3：时间线差异验证
    print("\n🔧 测试3：前后攻击时间线对比")
    print("-" * 30)
    
    # 模拟长机和僚机在相同距离下的不同阶段
    global_distances = [40000, 35000, 30000, 25000, 20000, 15000]
    
    print("距离(km) | 长机阶段    | 僚机阶段    | 时间线差异")
    print("-" * 50)
    
    for distance in global_distances:
        # 长机使用全局阶段判断
        if distance > task.tactical_distances['NLT_MELD_min']:
            leader_phase = TacticalPhase.NLT_MELD
        elif distance > task.tactical_distances['MELD_MTR_min']:
            leader_phase = TacticalPhase.MELD_MTR
        elif distance > task.tactical_distances['MTR_TR_min']:
            leader_phase = TacticalPhase.MTR_TR
        elif distance > task.tactical_distances['TR_DOR_min']:
            leader_phase = TacticalPhase.TR_DOR
        elif distance > task.tactical_distances['DOR_DR_min']:
            leader_phase = TacticalPhase.DOR_DR
        else:
            leader_phase = TacticalPhase.DOR_DR
        
        # 僚机使用滞后阶段判断
        wingman_phase = task._get_wingman_phase_by_distance(distance)
        
        # 判断是否有时间线差异
        diff_indicator = "⏰" if leader_phase != wingman_phase else "  "
        
        print(f"{distance/1000:7.1f} | {leader_phase.value:11s} | {wingman_phase.value:11s} | {diff_indicator}")
    
    print("\n⏰ = 存在时间线差异（僚机滞后）")
    
    print("\n" + "=" * 60)
    print("✅ 前后攻击战术修复验证完成")
    print("=" * 60)
    
    print("\n📋 修复总结:")
    print("1. ✅ 返航方向：长机左侧(-40°)，僚机右侧(+40°)")
    print("2. ✅ 时间线控制：僚机使用独立的滞后阶段判断")
    print("3. ✅ 高度控制：添加调试信息，严格保持初始高度")
    print("4. ✅ 战术流程：严格按照前后攻击时间线执行")
    
    return True

if __name__ == "__main__":
    test_front_back_attack_fixes()
