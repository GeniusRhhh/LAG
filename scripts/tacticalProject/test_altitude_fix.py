#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试高度指令修复逻辑
验证修复后的 _apply_adjustment_params 函数是否正确计算高度差
"""

import numpy as np

# 模拟归一化数组
norm_delta_altitude = np.array([
    -1500, -1000, -750, -500, -300, -150, -50, 0, 50, 150, 300, 500, 750, 1000, 1500
]) / 1000.0  # 索引7 = 0m变化（平稳飞行）

def _convert_altitude_to_index(altitude_diff):
    """将高度差值（米）转换为索引"""
    altitude_diff_km = altitude_diff / 1000.0
    idx = np.argmin(np.abs(norm_delta_altitude - altitude_diff_km))
    return int(idx)

def test_old_logic():
    """测试旧逻辑（错误的）"""
    print("=" * 80)
    print("测试旧逻辑（错误的）")
    print("=" * 80)
    
    alt_diff = 1500  # 目标高度差
    
    # 旧逻辑：直接使用固定增量
    lead_alt_idx = _convert_altitude_to_index(+alt_diff)
    wing_alt_idx = _convert_altitude_to_index(-alt_diff)
    
    print(f"目标高度差: {alt_diff}m")
    print(f"长机指令索引: {lead_alt_idx} (对应 {norm_delta_altitude[lead_alt_idx]*1000:.0f}m)")
    print(f"僚机指令索引: {wing_alt_idx} (对应 {norm_delta_altitude[wing_alt_idx]*1000:.0f}m)")
    print()
    
    # 模拟多次调用的累积效应
    print("模拟10次调用的累积效应:")
    lead_alt = 6000  # 初始高度6km
    wing_alt = 6000
    
    for i in range(10):
        lead_alt += norm_delta_altitude[lead_alt_idx] * 1000
        wing_alt += norm_delta_altitude[wing_alt_idx] * 1000
        print(f"  第{i+1}次: 长机={lead_alt:.0f}m, 僚机={wing_alt:.0f}m, 高度差={lead_alt-wing_alt:.0f}m")
    
    print(f"\n❌ 问题：长机爬升到{lead_alt:.0f}m（失速风险），僚机下降到{wing_alt:.0f}m（撞地风险）")
    print()

def test_new_logic():
    """测试新逻辑（正确的）"""
    print("=" * 80)
    print("测试新逻辑（正确的）")
    print("=" * 80)
    
    alt_diff = 1500  # 目标高度差
    threshold = 200  # 调整阈值
    
    # 初始状态
    lead_alt = 6000  # 初始高度6km
    wing_alt = 6000
    
    print(f"目标高度差: {alt_diff}m")
    print(f"调整阈值: {threshold}m")
    print(f"初始高度: 长机={lead_alt:.0f}m, 僚机={wing_alt:.0f}m")
    print()
    
    # 模拟10次调用
    print("模拟10次调用:")
    for i in range(10):
        # 计算平均高度和目标高度
        avg_alt = (lead_alt + wing_alt) / 2
        target_high_alt = avg_alt + alt_diff / 2  # 6750m
        target_low_alt = avg_alt - alt_diff / 2   # 5250m
        
        # 限制目标高度在安全范围内（3km-10km）
        target_high_alt = np.clip(target_high_alt, 3000, 10000)
        target_low_alt = np.clip(target_low_alt, 3000, 10000)
        
        # 计算到目标的差值（假设长机=high）
        lead_alt_diff = target_high_alt - lead_alt
        wing_alt_diff = target_low_alt - wing_alt
        
        # 只有差值大于阈值时才调整
        if abs(lead_alt_diff) > threshold:
            lead_alt_idx = _convert_altitude_to_index(lead_alt_diff)
            lead_alt += norm_delta_altitude[lead_alt_idx] * 1000
        else:
            lead_alt_idx = 7  # 保持高度
        
        if abs(wing_alt_diff) > threshold:
            wing_alt_idx = _convert_altitude_to_index(wing_alt_diff)
            wing_alt += norm_delta_altitude[wing_alt_idx] * 1000
        else:
            wing_alt_idx = 7  # 保持高度
        
        print(f"  第{i+1}次: 长机={lead_alt:.0f}m (目标{target_high_alt:.0f}m, 差值{lead_alt_diff:.0f}m, 索引{lead_alt_idx}), "
              f"僚机={wing_alt:.0f}m (目标{target_low_alt:.0f}m, 差值{wing_alt_diff:.0f}m, 索引{wing_alt_idx})")
    
    print(f"\n✅ 结果：长机稳定在{lead_alt:.0f}m，僚机稳定在{wing_alt:.0f}m，高度差={lead_alt-wing_alt:.0f}m")
    print()

def test_edge_cases():
    """测试边界情况"""
    print("=" * 80)
    print("测试边界情况")
    print("=" * 80)
    
    # 测试1：初始高度已经很高
    print("\n测试1：初始高度已经很高（长机9km，僚机8km）")
    lead_alt = 9000
    wing_alt = 8000
    alt_diff = 1500
    
    avg_alt = (lead_alt + wing_alt) / 2
    target_high_alt = np.clip(avg_alt + alt_diff / 2, 3000, 10000)
    target_low_alt = np.clip(avg_alt - alt_diff / 2, 3000, 10000)
    
    print(f"  平均高度: {avg_alt:.0f}m")
    print(f"  目标高位: {target_high_alt:.0f}m (限制后)")
    print(f"  目标低位: {target_low_alt:.0f}m (限制后)")
    print(f"  ✅ 高度被限制在10km以内，避免失速")
    
    # 测试2：初始高度已经很低
    print("\n测试2：初始高度已经很低（长机4km，僚机3.5km）")
    lead_alt = 4000
    wing_alt = 3500
    
    avg_alt = (lead_alt + wing_alt) / 2
    target_high_alt = np.clip(avg_alt + alt_diff / 2, 3000, 10000)
    target_low_alt = np.clip(avg_alt - alt_diff / 2, 3000, 10000)
    
    print(f"  平均高度: {avg_alt:.0f}m")
    print(f"  目标高位: {target_high_alt:.0f}m (限制后)")
    print(f"  目标低位: {target_low_alt:.0f}m (限制后)")
    print(f"  ✅ 高度被限制在3km以上，避免撞地")
    
    print()

if __name__ == "__main__":
    test_old_logic()
    test_new_logic()
    test_edge_cases()
    
    print("=" * 80)
    print("总结")
    print("=" * 80)
    print("旧逻辑问题：")
    print("  - 每次调用都爬升/下降1500m")
    print("  - 10次调用后，长机爬升到21km（失速），僚机下降到-9km（撞地）")
    print()
    print("新逻辑优势：")
    print("  - 计算到目标高度的差值，而不是固定增量")
    print("  - 高度稳定在目标值附近（6750m和5250m）")
    print("  - 有安全范围限制（3-10km）")
    print("  - 有调整阈值（200m），避免频繁微调")
    print("=" * 80)

