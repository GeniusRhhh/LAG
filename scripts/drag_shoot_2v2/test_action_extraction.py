#!/usr/bin/env python3
"""
测试动作提取器的修正效果
验证钳形攻击Crank机动识别和Short Skate阶段分类
"""

import pandas as pd
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pincer_tactical_action_extractor import PincerTacticalActionExtractor

def test_crank_detection():
    """测试Crank机动识别"""
    print("🔍 测试Crank机动识别逻辑")
    
    # 创建动作提取器
    extractor = PincerTacticalActionExtractor()
    
    # 测试早期阶段的Crank识别
    test_cases = [
        # (agent_id, current_time, heading, heading_change, expected_result)
        ("A0100", 10.0, 359.5, -2.5, "应该识别为Crank左转"),
        ("A0100", 20.0, 357.0, -1.5, "应该识别为Crank左转"),
        ("A0200", 10.0, 0.2, 1.8, "应该识别为Crank右转"),
        ("A0200", 20.0, 2.0, 0.8, "应该识别为Crank右转"),
        ("A0100", 50.0, 356.0, -0.5, "应该识别为收拢"),
        ("A0100", 80.0, 355.0, -0.2, "应该识别为平飞"),
    ]
    
    for agent_id, current_time, heading, heading_change, expected in test_cases:
        current_state = {'heading': heading}
        
        # 测试是否在Crank阶段
        is_crank_phase = extractor._is_in_pincer_crank_phase(agent_id, current_time, current_state)
        
        if is_crank_phase:
            # 测试Crank机动识别
            crank_type, direction = extractor._identify_pincer_crank_maneuver(
                agent_id, current_time, heading_change, current_state
            )
            result = f"{crank_type} - {direction}"
        else:
            result = "不在Crank阶段"
        
        print(f"  {agent_id} @ {current_time}s: {result} ({expected})")

def test_short_skate_detection():
    """测试Short Skate阶段分类"""
    print("\n🔍 测试Short Skate阶段分类逻辑")
    
    extractor = PincerTacticalActionExtractor()
    
    test_cases = [
        # (agent_id, current_time, heading, heading_change, expected_result)
        ("A0100", 76.0, 180.0, -15.0, "应该识别为偏转阶段"),
        ("A0100", 77.0, 165.0, -8.0, "应该识别为偏转阶段"),
        ("A0100", 81.0, 160.0, -0.5, "应该识别为返航阶段"),
        ("A0100", 90.0, 160.0, 0.0, "应该识别为返航阶段"),
        ("A0200", 84.0, 90.0, 12.0, "应该识别为偏转阶段"),
        ("A0200", 89.0, 95.0, 0.2, "应该识别为返航阶段"),
    ]
    
    for agent_id, current_time, heading, heading_change, expected in test_cases:
        current_state = {'heading': heading}
        
        # 测试Short Skate阶段分析
        action_type, direction = extractor._analyze_short_skate_phase(
            agent_id, 0, heading_change, 0, current_time, current_state
        )
        
        result = f"{action_type} - {direction}"
        print(f"  {agent_id} @ {current_time}s: {result} ({expected})")

def analyze_actual_data():
    """分析实际数据中的问题"""
    print("\n🔍 分析实际轨迹数据")
    
    # 读取最新的轨迹数据
    trajectory_file = "scripts/drag_shoot_2v2/pincer_attack_results/pincer_attack_trajectory_20250828_122249.csv"
    
    if not os.path.exists(trajectory_file):
        print(f"❌ 轨迹文件不存在: {trajectory_file}")
        return
    
    df = pd.read_csv(trajectory_file)
    extractor = PincerTacticalActionExtractor()
    
    # 分析早期阶段数据
    print("\n早期阶段分析 (0-40秒):")
    early_data = df[(df['Time_s'] >= 0) & (df['Time_s'] <= 40)]
    
    for agent_id in ['A0100', 'A0200']:
        agent_data = early_data[early_data['Agent_ID'] == agent_id].sort_values('Time_s')
        if len(agent_data) > 5:
            print(f"\n{agent_id}:")
            
            # 测试前几个时间点
            for i in range(min(5, len(agent_data))):
                row = agent_data.iloc[i]
                current_time = row['Time_s']
                current_heading = row['Heading_deg']
                
                # 计算航向变化
                if i > 0:
                    prev_heading = agent_data.iloc[i-1]['Heading_deg']
                    heading_change = extractor._calculate_heading_change(prev_heading, current_heading)
                else:
                    heading_change = 0
                
                current_state = {'heading': current_heading}
                
                # 测试Crank识别
                is_crank_phase = extractor._is_in_pincer_crank_phase(agent_id, current_time, current_state)
                
                if is_crank_phase:
                    crank_type, direction = extractor._identify_pincer_crank_maneuver(
                        agent_id, current_time, heading_change, current_state
                    )
                    predicted_action = f"{crank_type} - {direction}"
                else:
                    predicted_action = "平飞"
                
                actual_action = row['Action_Type']
                
                print(f"  时间{current_time}s: 航向{current_heading:.1f}° 变化{heading_change:.2f}° -> 预测:{predicted_action} 实际:{actual_action}")

if __name__ == "__main__":
    print("🧪 动作提取器修正效果测试")
    print("=" * 50)
    
    test_crank_detection()
    test_short_skate_detection()
    analyze_actual_data()
    
    print("\n" + "=" * 50)
    print("✅ 测试完成")
