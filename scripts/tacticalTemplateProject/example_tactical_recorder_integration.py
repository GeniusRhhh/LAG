#!/usr/bin/env python3
"""
战术态势记录器集成示例
展示如何在仿真中使用TacticalSituationRecorder
"""

import logging
from tactical_situation_recorder import TacticalSituationRecorder, get_tactical_recorder


def simulation_loop_example():
    """仿真循环集成示例（伪代码）"""
    
    # ==================== 1. 初始化阶段 ====================
    print("=" * 60)
    print("步骤1：初始化战术态势记录器")
    print("=" * 60)
    
    # 获取全局记录器实例
    recorder = get_tactical_recorder()
    
    # 或者创建独立实例
    # recorder = TacticalSituationRecorder()
    
    
    # ==================== 2. 仿真循环中记录 ====================
    print("\n" + "=" * 60)
    print("步骤2：在仿真循环中记录数据")
    print("=" * 60)
    
    print("""
# 伪代码示例：
def simulation_step(env, current_time):
    '''每个仿真步（0.2秒）调用一次'''
    
    # 获取敌方机动意图（从AI决策系统）
    action_intent_b0100 = get_enemy_action_intent(env, 'B0100')  # 例如："攻击性机动"
    action_intent_b0200 = get_enemy_action_intent(env, 'B0200')  # 例如："防御性规避"
    
    # 记录当前帧数据
    frame_data = recorder.record_frame(
        env=env,
        current_time=current_time,
        action_intent_b0100=action_intent_b0100,
        action_intent_b0200=action_intent_b0200
    )
    
    # 可选：实时打印关键信息
    if int(current_time * 10) % 10 == 0:  # 每2秒打印一次
        print(f"⏱️  时间: {current_time:.1f}s | "
              f"距离d1: {frame_data['d1_A0100_B0100_km']:.2f}km | "
              f"A0100干扰: {frame_data['jammed_A0100']}")
    """)
    
    
    # ==================== 3. 仿真结束后保存 ====================
    print("\n" + "=" * 60)
    print("步骤3：仿真结束后保存数据")
    print("=" * 60)
    
    print("""
# 仿真结束后
def simulation_end():
    # 保存到CSV
    recorder.save_to_csv('output/tactical_situation.csv')
    
    # 或者获取DataFrame进行分析
    df = recorder.get_dataframe()
    
    # 示例：筛选受干扰的帧
    jammed_frames = df[(df['jammed_A0100'] == 1) | (df['jammed_A0200'] == 1)]
    print(f"受干扰帧数: {len(jammed_frames)}")
    
    # 示例：分析距离变化
    import matplotlib.pyplot as plt
    plt.plot(df['Time_s'], df['d1_A0100_B0100_km'])
    plt.xlabel('时间 (s)')
    plt.ylabel('距离 (km)')
    plt.title('A0100与B0100距离变化')
    plt.savefig('distance_plot.png')
    """)


def complete_integration_example():
    """完整集成代码示例"""
    
    print("\n" + "=" * 60)
    print("完整集成代码示例")
    print("=" * 60)
    
    code = '''
# ========== 完整仿真脚本 ==========

from tactical_situation_recorder import get_tactical_recorder

# 1. 初始化
recorder = get_tactical_recorder()

# 2. 仿真主循环
for step in range(num_steps):
    current_time = step * 0.2  # 0.2秒/步
    
    # 执行仿真步
    env.step(actions)
    
    # 获取敌方机动意图（从trajectory或AI系统）
    # 方法1：从trajectory数据库查询
    action_b0100 = trajectory_db.get_action_intent('B0100', current_time)
    action_b0200 = trajectory_db.get_action_intent('B0200', current_time)
    
    # 方法2：从AI决策模块获取
    # action_b0100 = enemy_ai_b0100.get_current_maneuver()
    # action_b0200 = enemy_ai_b0200.get_current_maneuver()
    
    # 方法3：如果无法获取，使用默认值
    if action_b0100 is None:
        action_b0100 = "Unknown"
    if action_b0200 is None:
        action_b0200 = "Unknown"
    
    # 记录数据
    recorder.record_frame(
        env=env,
        current_time=current_time,
        action_intent_b0100=action_b0100,
        action_intent_b0200=action_b0200
    )

# 3. 保存数据
recorder.save_to_csv('results/tactical_situation.csv')

print("✅ 仿真完成，数据已保存")
'''
    
    print(code)


def data_analysis_example():
    """数据分析示例"""
    
    print("\n" + "=" * 60)
    print("数据分析示例")
    print("=" * 60)
    
    code = '''
# ========== 数据分析脚本 ==========

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('tactical_situation.csv')

# ===== 分析1: 距离变化趋势 =====
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# d1: A0100 vs B0100
axes[0, 0].plot(df['Time_s'], df['d1_A0100_B0100_km'])
axes[0, 0].set_title('A0100 vs B0100 距离')
axes[0, 0].set_xlabel('时间 (s)')
axes[0, 0].set_ylabel('距离 (km)')

# d2: A0100 vs B0200
axes[0, 1].plot(df['Time_s'], df['d2_A0100_B0200_km'])
axes[0, 1].set_title('A0100 vs B0200 距离')
axes[0, 1].set_xlabel('时间 (s)')
axes[0, 1].set_ylabel('距离 (km)')

# d3: A0200 vs B0100
axes[1, 0].plot(df['Time_s'], df['d3_A0200_B0100_km'])
axes[1, 0].set_title('A0200 vs B0100 距离')
axes[1, 0].set_xlabel('时间 (s)')
axes[1, 0].set_ylabel('距离 (km)')

# d4: A0200 vs B0200
axes[1, 1].plot(df['Time_s'], df['d4_A0200_B0200_km'])
axes[1, 1].set_title('A0200 vs B0200 距离')
axes[1, 1].set_xlabel('时间 (s)')
axes[1, 1].set_ylabel('距离 (km)')

plt.tight_layout()
plt.savefig('distance_analysis.png')

# ===== 分析2: 高度对比 =====
plt.figure(figsize=(12, 6))
plt.plot(df['Time_s'], df['h1_A0100_m'], label='A0100', linewidth=2)
plt.plot(df['Time_s'], df['h2_A0200_m'], label='A0200', linewidth=2)
plt.plot(df['Time_s'], df['h3_B0100_m'], label='B0100', linewidth=2, linestyle='--')
plt.plot(df['Time_s'], df['h4_B0200_m'], label='B0200', linewidth=2, linestyle='--')
plt.xlabel('时间 (s)')
plt.ylabel('高度 (m)')
plt.title('敌我高度对比')
plt.legend()
plt.grid(True)
plt.savefig('altitude_comparison.png')

# ===== 分析3: 速度对比 =====
plt.figure(figsize=(12, 6))
plt.plot(df['Time_s'], df['v1_A0100_ms'], label='A0100')
plt.plot(df['Time_s'], df['v2_A0200_ms'], label='A0200')
plt.plot(df['Time_s'], df['v3_B0100_ms'], label='B0100', linestyle='--')
plt.plot(df['Time_s'], df['v4_B0200_ms'], label='B0200', linestyle='--')
plt.xlabel('时间 (s)')
plt.ylabel('速度 (m/s)')
plt.title('敌我速度对比')
plt.legend()
plt.grid(True)
plt.savefig('velocity_comparison.png')

# ===== 分析4: 干扰状态时间线 =====
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# A0100干扰状态
axes[0].fill_between(df['Time_s'], 0, df['jammed_A0100'], 
                     alpha=0.5, color='red', label='受干扰')
axes[0].set_ylabel('A0100 干扰状态')
axes[0].set_ylim(-0.1, 1.1)
axes[0].legend()

# A0200干扰状态
axes[1].fill_between(df['Time_s'], 0, df['jammed_A0200'], 
                     alpha=0.5, color='red', label='受干扰')
axes[1].set_ylabel('A0200 干扰状态')
axes[1].set_ylim(-0.1, 1.1)
axes[1].set_xlabel('时间 (s)')
axes[1].legend()

plt.tight_layout()
plt.savefig('jamming_timeline.png')

# ===== 分析5: 敌方机动逻辑统计 =====
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# B0100机动分布
action_counts_b0100 = df['action_B0100'].value_counts()
axes[0].bar(range(len(action_counts_b0100)), action_counts_b0100.values)
axes[0].set_xticks(range(len(action_counts_b0100)))
axes[0].set_xticklabels(action_counts_b0100.index, rotation=45, ha='right')
axes[0].set_ylabel('帧数')
axes[0].set_title('B0100 机动逻辑分布')

# B0200机动分布
action_counts_b0200 = df['action_B0200'].value_counts()
axes[1].bar(range(len(action_counts_b0200)), action_counts_b0200.values)
axes[1].set_xticks(range(len(action_counts_b0200)))
axes[1].set_xticklabels(action_counts_b0200.index, rotation=45, ha='right')
axes[1].set_ylabel('帧数')
axes[1].set_title('B0200 机动逻辑分布')

plt.tight_layout()
plt.savefig('maneuver_distribution.png')

# ===== 分析6: 关键事件识别 =====
# 识别最近接近距离
min_distance_d1 = df['d1_A0100_B0100_km'].min()
min_distance_time_d1 = df.loc[df['d1_A0100_B0100_km'].idxmin(), 'Time_s']
print(f"A0100与B0100最近距离: {min_distance_d1:.2f}km @ {min_distance_time_d1:.1f}s")

# 识别受干扰时段
jammed_periods = df[df['jammed_A0100'] == 1]
if len(jammed_periods) > 0:
    print(f"A0100受干扰时长: {len(jammed_periods) * 0.2:.1f}s")
    print(f"受干扰比例: {len(jammed_periods)/len(df)*100:.1f}%")

# 识别高速段（速度 > 250 m/s）
high_speed_a0100 = df[df['v1_A0100_ms'] > 250]
print(f"A0100高速飞行时长: {len(high_speed_a0100) * 0.2:.1f}s")

print("\\n✅ 分析完成，图表已保存")
'''
    
    print(code)


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 运行示例
    simulation_loop_example()
    complete_integration_example()
    data_analysis_example()
    
    print("\n" + "=" * 60)
    print("✅ 集成示例已展示完成")
    print("=" * 60)
    print("\n💡 提示：")
    print("  - 将代码复制到你的仿真脚本中")
    print("  - 确保在每个仿真步调用 record_frame()")
    print("  - 仿真结束后调用 save_to_csv()")
    print("  - 使用pandas和matplotlib分析数据")
