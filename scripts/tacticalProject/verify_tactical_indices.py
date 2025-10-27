#!/usr/bin/env python3
"""
战术指令索引验证脚本
验证所有战术的指令索引是否正确映射到期望的机动动作
"""

import numpy as np

# 复制tactical_task.py中的指令数组定义
norm_delta_altitude = np.array([
    -1500, -1000, -750, -500, -300, -150, -50, 0, 50, 150, 300, 500, 750, 1000, 1500
]) / 1000.0  # 索引7 = 0m变化（平稳飞行）

norm_delta_heading = np.array([
    -np.pi,           # 索引0:  -180°
    -2*np.pi/3,       # 索引1:  -120°
    -np.pi/2,         # 索引2:  -90°
    -5*np.pi/12,      # 索引3:  -75°
    -np.pi/3,         # 索引4:  -60°
    -np.pi/4,         # 索引5:  -45°
    -np.pi/6,         # 索引6:  -30°
    -np.pi/12,        # 索引7:  -15°
    0,                # 索引8:  0° (平稳飞行)
    np.pi/12,         # 索引9:  15°
    np.pi/6,          # 索引10: 30°
    np.pi/4,          # 索引11: 45°
    np.pi/3,          # 索引12: 60°
    5*np.pi/12,       # 索引13: 75°
    np.pi/2,          # 索引14: 90°
    2*np.pi/3,        # 索引15: 120°
    np.pi             # 索引16: 180°
])

norm_delta_velocity = np.array([
    -150, -100, -50, 0, 50, 100, 150
]) / 100.0  # 索引3 = 0m/s变化（平稳飞行）


def verify_index(name, altitude_idx, heading_idx, velocity_idx):
    """验证指令索引并打印结果"""
    alt_change = norm_delta_altitude[altitude_idx] * 1000  # 转换回米
    heading_change = np.rad2deg(norm_delta_heading[heading_idx])  # 转换为度
    vel_change = norm_delta_velocity[velocity_idx] * 100  # 转换回m/s
    
    print(f"{name:30s} -> 高度:{alt_change:+7.0f}m, 航向:{heading_change:+6.1f}°, 速度:{vel_change:+6.0f}m/s")
    
    return alt_change, heading_change, vel_change


print("=" * 80)
print("战术指令索引验证")
print("=" * 80)

print("\n【基础指令验证】")
verify_index("平稳飞行", 7, 8, 3)
verify_index("左转15°", 7, 7, 3)
verify_index("右转15°", 7, 9, 3)
verify_index("左转30°", 7, 6, 3)
verify_index("右转30°", 7, 10, 3)
verify_index("左转45°", 7, 5, 3)
verify_index("右转45°", 7, 11, 3)
verify_index("左转60°", 7, 4, 3)
verify_index("右转60°", 7, 12, 3)
verify_index("左转90°", 7, 2, 3)
verify_index("右转90°", 7, 14, 3)
verify_index("爬升+500m", 11, 8, 3)
verify_index("俯冲-500m", 3, 8, 3)
verify_index("加速+50m/s", 7, 8, 4)
verify_index("加速+100m/s", 7, 8, 5)

print("\n【战术1: 拖曳射击 (Drag Shoot)】")
print("NLT-MELD阶段:")
verify_index("  长机平稳", 7, 8, 3)
verify_index("  僚机右侧Crank 60°", 7, 12, 3)

print("MELD-MTR阶段:")
verify_index("  长机平稳", 7, 8, 3)
verify_index("  僚机左侧Crank -45°", 7, 5, 3)

print("\n【战术2: 钳形攻势 (Pincer Attack)】")
print("NLT-MELD阶段:")
verify_index("  长机左侧Crank -45°", 7, 5, 3)
verify_index("  僚机右侧Crank +45°", 7, 11, 3)

print("MELD-MTR阶段:")
verify_index("  双机回正", 7, 8, 3)

print("\n【战术3: 上下夹击 (High-Low Attack)】")
print("MELD-MTR阶段:")
verify_index("  长机平稳", 7, 8, 3)
verify_index("  僚机爬升+500m", 11, 8, 3)

print("LR-TR阶段:")
verify_index("  僚机俯冲-500m", 3, 8, 3)

print("\n【战术4: 前后攻击 (Sequential Attack)】")
print("MELD-MTR阶段:")
verify_index("  长机前出+50m/s", 7, 8, 4)
verify_index("  僚机左侧Crank -45°", 7, 5, 3)

print("\n【战术5: 并排射击 (Side-by-Side)】")
print("全阶段:")
verify_index("  双机平稳", 7, 8, 3)

print("\n【Short Skate机动】")
print("Crank阶段:")
verify_index("  左侧Crank 30°", 7, 6, 3)
verify_index("  右侧Crank 30°", 7, 10, 3)

print("Turn Cold阶段:")
verify_index("  左转90°", 7, 2, 3)
verify_index("  右转90°", 7, 14, 3)

print("Escape阶段:")
verify_index("  加速+100m/s", 7, 8, 5)

print("\n" + "=" * 80)
print("验证完成！")
print("=" * 80)

# 检查潜在问题
print("\n【潜在问题检查】")
issues = []

# 检查索引9是否真的是15°
if abs(np.rad2deg(norm_delta_heading[9]) - 15.0) > 0.1:
    issues.append(f"⚠️ 索引9不是15°，而是{np.rad2deg(norm_delta_heading[9]):.1f}°")

# 检查索引8是否真的是0°
if abs(norm_delta_heading[8]) > 0.001:
    issues.append(f"⚠️ 索引8不是0°，而是{np.rad2deg(norm_delta_heading[8]):.1f}°")

# 检查索引7是否真的是0m
if abs(norm_delta_altitude[7]) > 0.001:
    issues.append(f"⚠️ 索引7不是0m，而是{norm_delta_altitude[7]*1000:.1f}m")

# 检查索引3是否真的是0m/s
if abs(norm_delta_velocity[3]) > 0.001:
    issues.append(f"⚠️ 索引3不是0m/s，而是{norm_delta_velocity[3]*100:.1f}m/s")

if issues:
    for issue in issues:
        print(issue)
else:
    print("✅ 所有基础索引验证通过！")
