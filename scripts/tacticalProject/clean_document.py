# -*- coding: utf-8 -*-
"""
文档清理脚本：删除重复的第5章和第6章内容
"""

def clean_document():
    input_file = r"d:\Pycharm\LAG\scripts\tacticalProject\项目说明.md"
    output_file = r"d:\Pycharm\LAG\scripts\tacticalProject\项目说明_cleaned.md"
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 找到关键行号
    # 第一个错误的5.3节开始于行937（索引936）
    # 第二个正确的5.3节开始于行1338（索引1337）
    
    # 保留行1到936（包括第5.1, 5.2, 5.3拖曳射击, 5.4钳形夹击）
    # 删除行937到1337（重复的第6章和决策算法）
    # 保留行1338到结束（正确的5.3框架定义和第6章）
    
    cleaned_lines = []
    
    # 保留前936行
    cleaned_lines.extend(lines[:936])
    
    # 跳过行937-1337（索引936-1336）
    # 保留从行1338开始的内容（索引1337开始）
    cleaned_lines.extend(lines[1337:])
    
    # 写入清理后的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)
    
    print(f"文档清理完成！")
    print(f"原文件：{input_file}")
    print(f"清理后文件：{output_file}")
    print(f"删除了 {1337-936} 行重复内容")

if __name__ == "__main__":
    clean_document()
