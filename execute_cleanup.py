#!/usr/bin/env python3
"""
代码清理执行脚本
根据分析报告，提供交互式的文件删除功能
"""

import os
import shutil
import glob
from pathlib import Path
import json

def load_cleanup_report():
    """加载清理建议报告"""
    try:
        with open('final_cleanup_recommendation.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ 未找到清理建议报告，请先运行 final_cleanup_recommendation.py")
        return None

def confirm_deletion(item_type, items):
    """确认删除操作"""
    print(f"\n准备删除 {item_type}:")
    for item in items[:10]:  # 只显示前10个
        print(f"  🗑️ {item}")
    if len(items) > 10:
        print(f"  ... 还有 {len(items) - 10} 个项目")
    
    response = input(f"\n确认删除这 {len(items)} 个{item_type}吗？(y/N): ").strip().lower()
    return response in ['y', 'yes']

def delete_files(file_list):
    """删除文件列表"""
    deleted_count = 0
    failed_count = 0
    
    for file_path in file_list:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"  ✅ 已删除: {file_path}")
                deleted_count += 1
            else:
                print(f"  ⚠️ 文件不存在: {file_path}")
        except Exception as e:
            print(f"  ❌ 删除失败: {file_path} - {e}")
            failed_count += 1
    
    return deleted_count, failed_count

def delete_directories(dir_list):
    """删除目录列表"""
    deleted_count = 0
    failed_count = 0
    
    for dir_path in dir_list:
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                print(f"  ✅ 已删除目录: {dir_path}")
                deleted_count += 1
            else:
                print(f"  ⚠️ 目录不存在: {dir_path}")
        except Exception as e:
            print(f"  ❌ 删除失败: {dir_path} - {e}")
            failed_count += 1
    
    return deleted_count, failed_count

def delete_log_files():
    """删除日志文件"""
    log_patterns = [
        "pincer_attack_simulation_*.log",
        "test_fixes.log"
    ]
    
    deleted_count = 0
    for pattern in log_patterns:
        files = glob.glob(pattern)
        for file_path in files:
            try:
                os.remove(file_path)
                print(f"  ✅ 已删除日志: {file_path}")
                deleted_count += 1
            except Exception as e:
                print(f"  ❌ 删除失败: {file_path} - {e}")
    
    return deleted_count

def interactive_cleanup():
    """交互式清理"""
    report = load_cleanup_report()
    if not report:
        return
    
    print("="*80)
    print("🧹 交互式代码清理工具")
    print("="*80)
    
    total_deleted_files = 0
    total_deleted_dirs = 0
    
    # 1. 清理调试和测试文件
    print("\n1️⃣ 清理调试和测试文件")
    debug_files = []
    for category, files in report['safe_to_delete'].items():
        if "调试" in category or "测试" in category:
            debug_files.extend(files)
    
    if debug_files and confirm_deletion("调试和测试文件", debug_files):
        deleted, failed = delete_files(debug_files)
        total_deleted_files += deleted
        print(f"✅ 删除了 {deleted} 个调试和测试文件")
    
    # 2. 清理验证和分析文件
    print("\n2️⃣ 清理验证和分析文件")
    analysis_files = []
    for category, files in report['safe_to_delete'].items():
        if "验证" in category or "分析" in category:
            analysis_files.extend(files)
    
    if analysis_files and confirm_deletion("验证和分析文件", analysis_files):
        deleted, failed = delete_files(analysis_files)
        total_deleted_files += deleted
        print(f"✅ 删除了 {deleted} 个验证和分析文件")
    
    # 3. 清理废弃的AI系统文件
    print("\n3️⃣ 清理废弃的AI系统文件")
    deprecated_ai_files = []
    for category, files in report['safe_to_delete'].items():
        if "废弃" in category:
            deprecated_ai_files.extend(files)
    
    if deprecated_ai_files and confirm_deletion("废弃的AI系统文件", deprecated_ai_files):
        deleted, failed = delete_files(deprecated_ai_files)
        total_deleted_files += deleted
        print(f"✅ 删除了 {deleted} 个废弃的AI系统文件")
    
    # 4. 清理未使用的提取器
    print("\n4️⃣ 清理未使用的提取器文件")
    extractor_files = []
    for category, files in report['safe_to_delete'].items():
        if "提取器" in category:
            extractor_files.extend(files)
    
    if extractor_files and confirm_deletion("未使用的提取器文件", extractor_files):
        deleted, failed = delete_files(extractor_files)
        total_deleted_files += deleted
        print(f"✅ 删除了 {deleted} 个未使用的提取器文件")
    
    # 5. 清理历史文件目录
    print("\n5️⃣ 清理历史文件目录")
    historical_dirs = report['historical_directories']
    if historical_dirs and confirm_deletion("历史文件目录", historical_dirs):
        deleted, failed = delete_directories(historical_dirs)
        total_deleted_dirs += deleted
        print(f"✅ 删除了 {deleted} 个历史文件目录")
    
    # 6. 清理大型结果目录
    print("\n6️⃣ 清理大型结果目录")
    large_dirs = []
    for category, dirs in report['large_directories_cleanup'].items():
        large_dirs.extend(dirs)
    
    if large_dirs and confirm_deletion("大型结果目录", large_dirs):
        deleted, failed = delete_directories(large_dirs)
        total_deleted_dirs += deleted
        print(f"✅ 删除了 {deleted} 个大型结果目录")
    
    # 7. 清理根目录文件
    print("\n7️⃣ 清理根目录分析脚本")
    root_files = []
    for category, files in report['root_level_cleanup'].items():
        if "分析" in category or "生成" in category:
            root_files.extend(files)
    
    if root_files and confirm_deletion("根目录分析脚本", root_files):
        deleted, failed = delete_files(root_files)
        total_deleted_files += deleted
        print(f"✅ 删除了 {deleted} 个根目录分析脚本")
    
    # 8. 清理日志文件
    print("\n8️⃣ 清理日志文件")
    if confirm_deletion("日志文件", ["所有匹配的日志文件"]):
        deleted = delete_log_files()
        total_deleted_files += deleted
        print(f"✅ 删除了 {deleted} 个日志文件")
    
    # 9. 清理分析报告文件
    print("\n9️⃣ 清理分析报告文件")
    report_files = []
    for category, files in report['root_level_cleanup'].items():
        if "报告" in category:
            report_files.extend(files)
    
    if report_files and confirm_deletion("分析报告文件", report_files):
        deleted, failed = delete_files(report_files)
        total_deleted_files += deleted
        print(f"✅ 删除了 {deleted} 个分析报告文件")
    
    # 总结
    print("\n" + "="*80)
    print("🎉 清理完成！")
    print("="*80)
    print(f"📊 总计删除:")
    print(f"  文件: {total_deleted_files} 个")
    print(f"  目录: {total_deleted_dirs} 个")
    print(f"\n✅ 核心战术仿真文件已保留，系统应该可以正常运行")
    print(f"⚠️ 建议运行主要功能测试以确认系统正常")

def main():
    """主函数"""
    print("🧹 代码清理执行脚本")
    print("本脚本将根据分析报告执行交互式清理")
    print("⚠️ 请确保已备份重要数据！")
    
    response = input("\n继续执行清理吗？(y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ 清理已取消")
        return
    
    interactive_cleanup()

if __name__ == "__main__":
    main()
