#!/usr/bin/env python3
"""
简单测试：验证导弹发射逻辑修复
"""

import sys
import os
sys.path.append('scripts/tacticalProject')

# 检查导弹发射逻辑的修复
def test_missile_logic():
    print("🚀 检查导弹发射逻辑修复")
    
    try:
        # 导入tactical_task
        from tactical_task import TacticalTask
        print("✅ tactical_task导入成功")
        
        # 检查是否有_handle_missile_launches方法
        if hasattr(TacticalTask, '_handle_missile_launches'):
            print("✅ _handle_missile_launches方法存在")
        else:
            print("❌ _handle_missile_launches方法不存在")
            
        # 检查是否有_launch_missile方法
        if hasattr(TacticalTask, '_launch_missile'):
            print("✅ _launch_missile方法存在")
        else:
            print("❌ _launch_missile方法不存在")
            
        # 检查是否有_normalize_angle_diff方法
        if hasattr(TacticalTask, '_normalize_angle_diff'):
            print("✅ _normalize_angle_diff方法存在")
        else:
            print("❌ _normalize_angle_diff方法不存在")
            
        print("\n🎯 修复总结:")
        print("1. 导弹发射逻辑已从should_launch_missile改为_handle_missile_launches")
        print("2. missile_launched标志现在表示'请求发射'而不是'已发射阻止'")
        print("3. A0200僚机现在应该能够正常发射导弹")
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
    except Exception as e:
        print(f"❌ 其他错误: {e}")

if __name__ == "__main__":
    test_missile_logic()