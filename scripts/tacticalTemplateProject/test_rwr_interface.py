#!/usr/bin/env python3
"""
RWR接口功能验证脚本
用于确认所有RWR功能已正确实现
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_rwr_imports():
    """测试1: 验证RWR接口可以正确导入"""
    print("\n" + "="*60)
    print("测试1: 验证RWR接口导入")
    print("="*60)
    
    try:
        from radar_manager import (
            get_rwr_threat_level,
            get_rwr_threat_sources,
            get_rwr_max_threat_bearing,
            get_unified_radar_manager
        )
        print("✅ 所有RWR接口函数导入成功")
        print(f"   - get_rwr_threat_level: {get_rwr_threat_level}")
        print(f"   - get_rwr_threat_sources: {get_rwr_threat_sources}")
        print(f"   - get_rwr_max_threat_bearing: {get_rwr_max_threat_bearing}")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_rwr_manager_methods():
    """测试2: 验证UnifiedRadarManager类中的RWR方法"""
    print("\n" + "="*60)
    print("测试2: 验证UnifiedRadarManager类方法")
    print("="*60)
    
    try:
        from radar_manager import get_unified_radar_manager
        
        radar_manager = get_unified_radar_manager()
        
        # 检查类方法是否存在
        required_methods = [
            'get_rwr_threat_level',
            'get_rwr_threat_sources',
            'get_rwr_max_threat_bearing',
            '_update_rwr_states'
        ]
        
        all_exist = True
        for method_name in required_methods:
            if hasattr(radar_manager, method_name):
                print(f"✅ 方法存在: {method_name}")
            else:
                print(f"❌ 方法缺失: {method_name}")
                all_exist = False
        
        return all_exist
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_rwr_data_structure():
    """测试3: 验证RWR数据结构"""
    print("\n" + "="*60)
    print("测试3: 验证RWR数据结构")
    print("="*60)
    
    try:
        from radar_manager import get_unified_radar_manager
        
        radar_manager = get_unified_radar_manager()
        
        # 检查rwr_states数据结构
        if hasattr(radar_manager, 'rwr_states'):
            print("✅ rwr_states 数据结构存在")
            print(f"   包含的飞机: {list(radar_manager.rwr_states.keys())}")
            
            # 检查数据结构内容
            for agent_id, rwr_data in radar_manager.rwr_states.items():
                required_keys = ['threat_level', 'threat_sources', 'last_warning']
                has_all_keys = all(key in rwr_data for key in required_keys)
                
                if has_all_keys:
                    print(f"✅ {agent_id} RWR数据结构完整: {list(rwr_data.keys())}")
                else:
                    print(f"❌ {agent_id} RWR数据结构不完整")
                    return False
            
            return True
        else:
            print("❌ rwr_states 数据结构不存在")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_rwr_function_calls():
    """测试4: 验证RWR函数可以正常调用"""
    print("\n" + "="*60)
    print("测试4: 验证RWR函数调用")
    print("="*60)
    
    try:
        from radar_manager import (
            get_rwr_threat_level,
            get_rwr_threat_sources,
            get_rwr_max_threat_bearing
        )
        
        test_agent_id = "A0100"
        
        # 测试get_rwr_threat_level
        threat_level = get_rwr_threat_level(test_agent_id)
        print(f"✅ get_rwr_threat_level('{test_agent_id}') = {threat_level}")
        
        # 测试get_rwr_threat_sources
        threat_sources = get_rwr_threat_sources(test_agent_id)
        print(f"✅ get_rwr_threat_sources('{test_agent_id}') = {threat_sources}")
        
        # 测试get_rwr_max_threat_bearing
        max_bearing = get_rwr_max_threat_bearing(test_agent_id)
        print(f"✅ get_rwr_max_threat_bearing('{test_agent_id}') = {max_bearing}")
        
        return True
        
    except Exception as e:
        print(f"❌ 函数调用失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_drag_shoot_integration():
    """测试5: 验证拖曳射击战术中的RWR集成"""
    print("\n" + "="*60)
    print("测试5: 验证拖曳射击战术RWR集成")
    print("="*60)
    
    try:
        # 检查文件是否存在
        import os
        file_path = "C:\\Users\\ZRF\\PycharmProjects\\LAG\\scripts\\tacticalTemplateProject\\drag_shoot_tactical_task.py"
        
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            return False
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查关键代码是否存在
        checks = {
            "导入RWR接口": "from radar_manager import get_unified_radar_manager, get_rwr_threat_level, get_rwr_threat_sources",
            "保存RWR函数引用": "self.get_rwr_threat_level = get_rwr_threat_level",
            "RWR检测方法定义": "def _check_rwr_threats(self, env):",
            "RWR威胁等级获取": "threat_level = self.get_rwr_threat_level(agent_id)",
            "RWR威胁源获取": "threat_sources = self.get_rwr_threat_sources(agent_id)"
        }
        
        all_passed = True
        for check_name, check_code in checks.items():
            if check_code in content:
                print(f"✅ {check_name}: 已实现")
            else:
                print(f"❌ {check_name}: 未找到")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_missile_launch_check():
    """测试6: 验证导弹发射检查接口"""
    print("\n" + "="*60)
    print("测试6: 验证导弹发射检查接口")
    print("="*60)
    
    try:
        from radar_manager import check_missile_launch_conditions
        
        print("✅ check_missile_launch_conditions 函数导入成功")
        print(f"   函数签名: {check_missile_launch_conditions.__doc__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("🧪 RWR接口功能验证测试套件")
    print("="*60)
    
    tests = [
        ("RWR接口导入", test_rwr_imports),
        ("UnifiedRadarManager方法", test_rwr_manager_methods),
        ("RWR数据结构", test_rwr_data_structure),
        ("RWR函数调用", test_rwr_function_calls),
        ("拖曳射击集成", test_drag_shoot_integration),
        ("导弹发射检查", test_missile_launch_check)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ 测试 '{test_name}' 执行异常: {e}")
            results.append((test_name, False))
    
    # 打印总结
    print("\n" + "="*60)
    print("📊 测试结果总结")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {test_name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！RWR接口已完全实现并集成。")
        return True
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查实现。")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
