#!/usr/bin/env python3
"""
PyCharm运行环境诊断脚本
诊断可能导致文件生成失败的环境问题
"""

import os
import sys
import logging
from pathlib import Path

def diagnose_environment():
    """诊断PyCharm运行环境"""
    print("🔍 PyCharm运行环境诊断")
    print("=" * 60)
    
    # 1. 工作目录检查
    print("\n📁 工作目录信息:")
    current_dir = os.getcwd()
    print(f"当前工作目录: {current_dir}")
    print(f"脚本所在目录: {os.path.dirname(os.path.abspath(__file__))}")
    
    # 2. 项目根目录检查
    print("\n🏠 项目根目录检查:")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"推断的项目根目录: {project_root}")
    print(f"项目根目录是否存在: {os.path.exists(project_root)}")
    
    # 3. 输出目录检查
    print("\n📂 输出目录检查:")
    output_dir = "scripts/drag_shoot_2v2/basic_action_data"
    
    # 相对路径检查
    relative_path = os.path.join(current_dir, output_dir)
    print(f"相对路径: {relative_path}")
    print(f"相对路径存在: {os.path.exists(relative_path)}")
    
    # 绝对路径检查
    absolute_path = os.path.join(project_root, output_dir)
    print(f"绝对路径: {absolute_path}")
    print(f"绝对路径存在: {os.path.exists(absolute_path)}")
    
    # 4. 权限检查
    print("\n🔐 权限检查:")
    test_dirs = [current_dir, project_root, relative_path, absolute_path]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            readable = os.access(test_dir, os.R_OK)
            writable = os.access(test_dir, os.W_OK)
            executable = os.access(test_dir, os.X_OK)
            print(f"{test_dir}:")
            print(f"  可读: {readable}, 可写: {writable}, 可执行: {executable}")
    
    # 5. 创建测试文件
    print("\n📝 文件创建测试:")
    test_paths = [
        os.path.join(current_dir, "test_file.txt"),
        os.path.join(relative_path, "test_file.txt") if os.path.exists(relative_path) else None,
        os.path.join(absolute_path, "test_file.txt") if os.path.exists(absolute_path) else None
    ]
    
    for i, test_path in enumerate(test_paths):
        if test_path is None:
            continue
            
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(test_path), exist_ok=True)
            
            # 尝试创建文件
            with open(test_path, 'w', encoding='utf-8') as f:
                f.write("测试文件内容")
            
            # 检查文件是否真的存在
            if os.path.exists(test_path):
                print(f"✅ 测试文件创建成功: {test_path}")
                # 清理测试文件
                os.remove(test_path)
            else:
                print(f"❌ 测试文件创建失败: {test_path} (文件不存在)")
                
        except Exception as e:
            print(f"❌ 测试文件创建失败: {test_path} - {e}")
    
    # 6. Python路径检查
    print("\n🐍 Python路径检查:")
    print(f"Python可执行文件: {sys.executable}")
    print(f"Python版本: {sys.version}")
    print(f"Python路径:")
    for path in sys.path[:5]:  # 只显示前5个路径
        print(f"  {path}")
    
    # 7. 环境变量检查
    print("\n🌍 相关环境变量:")
    env_vars = ['PYTHONPATH', 'PATH', 'TEMP', 'TMP', 'HOME', 'USERPROFILE']
    for var in env_vars:
        value = os.environ.get(var, '未设置')
        if len(value) > 100:
            value = value[:100] + "..."
        print(f"{var}: {value}")
    
    # 8. 推荐解决方案
    print("\n💡 推荐解决方案:")
    print("1. 确保PyCharm的工作目录设置为项目根目录")
    print("2. 检查PyCharm的运行配置中的工作目录设置")
    print("3. 尝试使用绝对路径而不是相对路径")
    print("4. 检查文件权限和磁盘空间")
    print("5. 尝试在PyCharm终端中运行脚本")

def create_pycharm_run_config():
    """创建PyCharm运行配置建议"""
    print("\n⚙️ PyCharm运行配置建议:")
    print("=" * 40)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("在PyCharm中创建运行配置时，请设置:")
    print(f"- 脚本路径: {os.path.join(project_root, 'scripts', 'basic_action_data_generator.py')}")
    print(f"- 工作目录: {project_root}")
    print("- 参数: --action accelerate --mode test")
    print()
    print("或者在PyCharm终端中运行:")
    print(f"cd {project_root}")
    print("python scripts/basic_action_data_generator.py --action accelerate --mode test")

if __name__ == "__main__":
    diagnose_environment()
    create_pycharm_run_config()
