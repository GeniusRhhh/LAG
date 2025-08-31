#!/usr/bin/env python3
"""
CSV文件编码修复脚本
确保所有CSV文件都使用正确的UTF-8编码，修复可能的乱码问题
"""

import pandas as pd
import os
import glob
from typing import List, Dict


def detect_file_encoding(file_path: str) -> str:
    """检测文件编码 - 简化版本"""
    # 尝试常见编码
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read(100)  # 尝试读取前100个字符
                return encoding
        except:
            continue

    return 'utf-8'  # 默认返回utf-8


def fix_garbled_text(text: str) -> str:
    """修复常见的乱码文本"""
    if pd.isna(text) or not isinstance(text, str):
        return text
        
    # 常见乱码映射
    garbled_mapping = {
        '骞抽': '平飞',
        '鏃?': '无',
        '鏃犮': '无',
        '鎴樻湳crank': '战术crank',
        '鎴樻湳': '战术',
        '宸﹁浆': '左转',
        '鍙宠浆': '右转',
        '鐩撮': '直飞',
        '鐖?': '爬升',
        '淇?': '俯冲',
        '鍔犻': '加速',
        '鍑忛': '减速',
        'Crank': 'Crank',
        'crank': 'crank'
    }
    
    fixed_text = text
    for garbled, correct in garbled_mapping.items():
        fixed_text = fixed_text.replace(garbled, correct)
    
    return fixed_text


def fix_csv_file_encoding(file_path: str) -> bool:
    """修复单个CSV文件的编码问题"""
    try:
        print(f"处理文件: {os.path.basename(file_path)}")
        
        # 检测原始编码
        original_encoding = detect_file_encoding(file_path)
        print(f"  检测到编码: {original_encoding}")
        
        # 尝试用不同编码读取文件
        df = None
        for encoding in [original_encoding, 'utf-8', 'gbk', 'gb2312', 'utf-8-sig']:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"  成功用 {encoding} 读取")
                break
            except Exception as e:
                continue
        
        if df is None:
            print(f"  ❌ 无法读取文件")
            return False
        
        # 检查是否有Action_Type列需要修复
        if 'Action_Type' in df.columns:
            print(f"  发现Action_Type列，检查乱码...")
            
            # 修复Action_Type列的乱码
            original_actions = df['Action_Type'].tolist()
            df['Action_Type'] = df['Action_Type'].apply(fix_garbled_text)
            
            # 修复Direction列的乱码（如果存在）
            if 'Direction' in df.columns:
                df['Direction'] = df['Direction'].apply(fix_garbled_text)
            
            # 检查是否有修复
            fixed_actions = df['Action_Type'].tolist()
            changes_made = any(orig != fixed for orig, fixed in zip(original_actions, fixed_actions))
            
            if changes_made:
                print(f"  ✅ 发现并修复了乱码")
                # 显示修复的样本
                for i, (orig, fixed) in enumerate(zip(original_actions[:5], fixed_actions[:5])):
                    if orig != fixed:
                        print(f"    修复: '{orig}' -> '{fixed}'")
            else:
                print(f"  ✅ 未发现乱码，文件正常")
        
        # 保存文件，确保使用UTF-8编码
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"  ✅ 文件已保存为UTF-8编码")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 处理失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("CSV文件编码修复工具")
    print("=" * 60)
    
    # 搜索所有包含Action_Type的CSV文件
    search_patterns = [
        'scripts/drag_shoot_2v2/air_combat_results/*trajectory*.csv',
        'scripts/drag_shoot_2v2/pincer_attack_results/*trajectory*.csv',
        'scripts/drag_shoot_2v2/front_back_attack_results/*trajectory*.csv',
        'scripts/drag_shoot_2v2/high_low_attack_results/*trajectory*.csv'
    ]
    
    all_files = []
    for pattern in search_patterns:
        all_files.extend(glob.glob(pattern))
    
    print(f"找到 {len(all_files)} 个轨迹文件")
    
    if not all_files:
        print("未找到任何轨迹文件")
        return
    
    # 处理每个文件
    success_count = 0
    for file_path in all_files:
        if fix_csv_file_encoding(file_path):
            success_count += 1
        print()
    
    print("=" * 60)
    print(f"处理完成: {success_count}/{len(all_files)} 个文件成功处理")
    print("=" * 60)


if __name__ == "__main__":
    main()
