#!/usr/bin/env python3
"""
验证重构版环境设置
"""

import os
import sys

def verify_files():
    """验证所有必需文件是否存在"""
    print("🔍 验证文件...")
    
    required_files = [
        'base_tactical_task_v3.py',
        'drag_shoot_tactical_task_v3.py',
        'pincer_attack_tactical_task_v3.py',
        'front_back_attack_v3.py',
        'high_low_attack_v3.py',
        'side_by_side_shooting_v3.py',
        'turn_around_shooting_v3.py',
        'radar_manager.py',
        'unified_enemy_tactical_ai.py',
        'unified_data_recorder.py',
        'run_drag_shoot_test.py',
    ]
    
    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - 缺失！")
            missing.append(file)
    
    if missing:
        print(f"\n❌ 缺少 {len(missing)} 个文件")
        return False
    else:
        print(f"\n✅ 所有 {len(required_files)} 个必需文件都存在")
        return True

def verify_imports():
    """验证导入是否正常"""
    print("\n🔍 验证导入...")
    
    # 添加项目根目录到路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    sys.path.insert(0, project_root)
    
    try:
        print("  测试基类导入...")
        from base_tactical_task_v3 import BaseTacticalTask
        print("  ✅ 基类导入成功")
        
        print("  测试拖曳射击导入...")
        from drag_shoot_tactical_task_v3 import DragShootTacticalTask
        print("  ✅ 拖曳射击导入成功")
        
        print("  测试雷达管理器导入...")
        from radar_manager import get_unified_radar_manager
        print("  ✅ 雷达管理器导入成功")
        
        print("  测试数据记录器导入...")
        from unified_data_recorder import UnifiedDataRecorder
        print("  ✅ 数据记录器导入成功")
        
        print("\n✅ 所有导入测试通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_class_structure():
    """验证类结构"""
    print("\n🔍 验证类结构...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    sys.path.insert(0, project_root)
    
    try:
        from base_tactical_task_v3 import BaseTacticalTask
        from drag_shoot_tactical_task_v3 import DragShootTacticalTask
        
        # 检查继承关系
        if issubclass(DragShootTacticalTask, BaseTacticalTask):
            print("  ✅ DragShootTacticalTask 正确继承 BaseTacticalTask")
        else:
            print("  ❌ 继承关系错误")
            return False
        
        # 检查基类方法
        base_methods = ['_init_radar_and_rwr', '_load_baseline_model', '_update_radar_and_rwr', '_use_lowlevel_policy']
        for method in base_methods:
            if hasattr(BaseTacticalTask, method):
                print(f"  ✅ 基类方法 {method} 存在")
            else:
                print(f"  ❌ 基类方法 {method} 缺失")
                return False
        
        print("\n✅ 类结构验证通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 类结构验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 80)
    print("🚀 重构版V3环境验证")
    print("=" * 80)
    
    results = []
    
    # 验证文件
    results.append(("文件检查", verify_files()))
    
    # 验证导入
    results.append(("导入测试", verify_imports()))
    
    # 验证类结构
    results.append(("类结构验证", verify_class_structure()))
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 验证结果")
    print("=" * 80)
    
    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 所有验证通过！环境配置正确。")
        print("\n可以运行测试脚本：")
        print("  python run_drag_shoot_test.py")
    else:
        print("❌ 部分验证失败，请检查配置。")
    print("=" * 80)

if __name__ == "__main__":
    main()
