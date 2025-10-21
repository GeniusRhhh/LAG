#!/usr/bin/env python3
"""
战术态势数据记录器集成示例
展示如何在现有的战术仿真中集成TacticalDataLogger
"""

import logging
from tactical_data_logger import TacticalDataLogger


def integrate_logger_example():
    """
    集成示例：在run_pincer_attack.py中添加数据记录
    """
    
    # ==================== 在仿真开始前 ====================
    # 创建数据记录器
    tactical_logger = TacticalDataLogger()
    logging.info("✅ 战术态势数据记录器已初始化")
    
    # ==================== 在仿真循环中 ====================
    # 假设你的仿真循环如下：
    """
    for step in range(max_steps):
        # 1. 获取友方动作
        A0100_action = get_friendly_action(...)
        A0200_action = get_friendly_action(...)
        
        # 2. 获取敌方动作
        B0100_action = enemy_ai.get_enemy_command(env, 'B0100', current_time)
        B0200_action = enemy_ai.get_enemy_command(env, 'B0200', current_time)
        
        # 3. 执行一步仿真
        env.step({...})
        
        # 4. 【新增】记录态势数据
        tactical_logger.log_step(env, step, current_time, enemy_ai)
        
        # 5. 检查终止条件
        if done:
            break
    """
    
    # ==================== 在仿真结束后 ====================
    # 保存数据到CSV
    output_file = "results/tactical_situation.csv"
    tactical_logger.save_to_csv(output_file)
    
    # 获取并打印摘要
    summary = tactical_logger.get_summary()
    print("\n" + "="*80)
    print("📊 态势数据摘要")
    print("="*80)
    print(f"总步数: {summary.get('total_steps', 0)}")
    print(f"仿真时间: {summary.get('sim_time_range', (0, 0))} 秒")
    print(f"最小交战距离: {summary.get('min_distance', 0):.0f} 米")
    print(f"我方最大高度: {summary.get('max_altitude_f', 0):.0f} 米")
    print(f"敌方最大高度: {summary.get('max_altitude_s', 0):.0f} 米")
    print(f"我方平均速度: {summary.get('avg_speed_f', 0):.1f} m/s")
    print(f"敌方平均速度: {summary.get('avg_speed_s', 0):.1f} m/s")
    print("="*80)


def modify_existing_simulation():
    """
    具体修改步骤：在run_pincer_attack.py中的修改位置
    """
    modifications = """
    # ========================================
    # 修改位置1: 导入模块（文件开头）
    # ========================================
    from tactical_data_logger import TacticalDataLogger
    
    
    # ========================================
    # 修改位置2: 初始化记录器（main函数或run_simulation函数开始）
    # ========================================
    def run_simulation(env, ...):
        # ... 现有代码 ...
        
        # 【新增】创建态势数据记录器
        tactical_logger = TacticalDataLogger()
        
        # ... 现有代码 ...
    
    
    # ========================================
    # 修改位置3: 记录数据（仿真循环内，每一步）
    # ========================================
    while not done:
        # ... 获取动作、执行仿真等现有代码 ...
        
        # 【新增】记录当前步的态势数据
        tactical_logger.log_step(env, current_step, current_time, enemy_ai)
        
        # ... 检查终止条件等现有代码 ...
    
    
    # ========================================
    # 修改位置4: 保存数据（仿真结束后）
    # ========================================
    # ... 现有的结果保存代码 ...
    
    # 【新增】保存态势数据
    situation_file = os.path.join(output_dir, "tactical_situation.csv")
    tactical_logger.save_to_csv(situation_file)
    
    # 【可选】打印摘要
    summary = tactical_logger.get_summary()
    logging.info(f"态势数据摘要: {summary}")
    """
    
    print(modifications)


def data_analysis_example():
    """
    数据分析示例：如何使用保存的CSV数据
    """
    analysis_code = """
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # 1. 读取数据
    df = pd.read_csv('tactical_situation.csv')
    
    # 2. 基础统计
    print("数据概览:")
    print(df.describe())
    
    # 3. 绘制距离变化图
    plt.figure(figsize=(12, 6))
    plt.plot(df['sim_time_s'], df['dist_f1_s1_m'], label='F1-S1')
    plt.plot(df['sim_time_s'], df['dist_f1_s2_m'], label='F1-S2')
    plt.plot(df['sim_time_s'], df['dist_f2_s1_m'], label='F2-S1')
    plt.plot(df['sim_time_s'], df['dist_f2_s2_m'], label='F2-S2')
    plt.xlabel('时间 (s)')
    plt.ylabel('距离 (m)')
    plt.title('双方距离变化')
    plt.legend()
    plt.grid(True)
    plt.savefig('distance_plot.png')
    
    # 4. 绘制高度变化图
    plt.figure(figsize=(12, 6))
    plt.plot(df['sim_time_s'], df['alt_f1_m'], label='F1', linestyle='-')
    plt.plot(df['sim_time_s'], df['alt_f2_m'], label='F2', linestyle='-')
    plt.plot(df['sim_time_s'], df['alt_s1_m'], label='S1', linestyle='--')
    plt.plot(df['sim_time_s'], df['alt_s2_m'], label='S2', linestyle='--')
    plt.xlabel('时间 (s)')
    plt.ylabel('高度 (m)')
    plt.title('双方高度变化')
    plt.legend()
    plt.grid(True)
    plt.savefig('altitude_plot.png')
    
    # 5. 按敌方机动逻辑分组统计
    action_stats = df.groupby('action_s1').agg({
        'dist_f1_s1_m': 'mean',
        'alt_s1_m': 'mean',
        'vel_s1_ms': 'mean'
    })
    print("\\n敌方S1不同机动下的平均态势:")
    print(action_stats)
    
    # 6. 找出关键事件
    min_dist = df['dist_f1_s1_m'].min()
    min_dist_time = df.loc[df['dist_f1_s1_m'].idxmin(), 'sim_time_s']
    print(f"\\n最小交战距离: {min_dist:.0f}m，发生在 {min_dist_time:.1f}s")
    
    # 7. 导出特定时刻的快照
    critical_moments = df[df['dist_f1_s1_m'] < 30000]  # 距离小于30km的时刻
    critical_moments.to_csv('critical_moments.csv', index=False)
    """
    
    print("数据分析示例代码:")
    print(analysis_code)


if __name__ == "__main__":
    print("="*80)
    print("战术态势数据记录器 - 集成指南")
    print("="*80)
    print()
    
    print("📋 第一步：了解集成方法")
    print("-" * 80)
    integrate_logger_example()
    
    print("\n\n📝 第二步：查看具体修改位置")
    print("-" * 80)
    modify_existing_simulation()
    
    print("\n\n📊 第三步：数据分析示例")
    print("-" * 80)
    data_analysis_example()
    
    print("\n\n✅ 集成完成后，你将得到:")
    print("  1. tactical_situation.csv - 完整的态势数据表")
    print("  2. 28列详细数据，包括距离、高度、速度、方位角、进入角等")
    print("  3. 可直接用于Excel分析或Python绘图")
    print("  4. 与现有trajectory表完美兼容")
