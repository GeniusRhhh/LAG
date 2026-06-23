#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tacview战术结果快速查看器
用于快速打开和分析战术仿真的Tacview文件
Created by GitHub Copilot
"""
import os
import subprocess
import sys
from pathlib import Path

class TacviewViewer:
    def __init__(self):
        self.results_dir = Path("D:/Pycharm/LAG/scripts/tacticalProject/tactical_simulation_results")
        # 常见的Tacview安装路径
        self.tacview_paths = [
            "C:/Program Files/Tacview/Tacview.exe",
            "C:/Program Files (x86)/Tacview/Tacview.exe", 
            "D:/Program Files/Tacview/Tacview.exe",
            "C:/Users/%USERNAME%/AppData/Local/Tacview/Tacview.exe"
        ]
        
    def find_tacview_exe(self):
        """查找Tacview可执行文件"""
        for path in self.tacview_paths:
            expanded_path = os.path.expandvars(path)
            if os.path.exists(expanded_path):
                return expanded_path
        return None
        
    def get_latest_acmi_files(self, limit=10):
        """获取最新的ACMI文件列表"""
        acmi_files = []
        
        for acmi_file in self.results_dir.glob("*.acmi"):
            acmi_files.append(acmi_file)
            
        # 按修改时间排序，最新的在前
        acmi_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return acmi_files[:limit]
        
    def identify_tactic_from_filename(self, filename):
        """从文件名识别战术类型"""
        tactics = {
            'pincer': 'PINCER_ATTACK (钳形攻势)',
            'drag': 'DRAG_SHOOT (拖拽射击)', 
            'high_low': 'HIGH_LOW_ATTACK (高低攻击)',
            'sequential': 'SEQUENTIAL_ATTACK (顺序攻击)',
            'side': 'SIDE_BY_SIDE (并肩作战)'
        }
        
        filename_lower = filename.lower()
        for key, value in tactics.items():
            if key in filename_lower:
                return value
                
        return "未知战术"
        
    def analyze_acmi_file(self, acmi_file):
        """简单分析ACMI文件信息"""
        try:
            with open(acmi_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            lines = content.split('\n')
            duration = 0
            entities = set()
            
            for line in lines:
                if line.startswith('#'):
                    try:
                        timestamp = float(line[1:])
                        duration = max(duration, timestamp)
                    except ValueError:
                        continue
                elif ',' in line and not line.startswith('FileType'):
                    parts = line.split(',')
                    if len(parts) > 0:
                        entity_id = parts[0]
                        if entity_id and not entity_id.startswith('FileType'):
                            entities.add(entity_id)
                            
            return {
                'duration': duration,
                'entity_count': len(entities),
                'file_size': acmi_file.stat().st_size
            }
        except Exception as e:
            return {'error': str(e)}
            
    def display_available_files(self):
        """显示可用的ACMI文件列表"""
        acmi_files = self.get_latest_acmi_files()
        
        if not acmi_files:
            print("❌ 未找到任何ACMI文件")
            return []
            
        print("📋 可用的战术仿真文件:")
        print("=" * 80)
        
        for i, acmi_file in enumerate(acmi_files, 1):
            tactic = self.identify_tactic_from_filename(acmi_file.name)
            analysis = self.analyze_acmi_file(acmi_file)
            
            file_time = acmi_file.stat().st_mtime
            import datetime
            time_str = datetime.datetime.fromtimestamp(file_time).strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"{i:2d}. {acmi_file.name}")
            print(f"    🎯 战术: {tactic}")
            print(f"    ⏰ 时间: {time_str}")
            
            if 'error' not in analysis:
                print(f"    📊 时长: {analysis['duration']:.1f}秒")
                print(f"    ✈️  实体: {analysis['entity_count']}个")
                print(f"    📁 大小: {analysis['file_size']/1024:.1f}KB")
            else:
                print(f"    ⚠️  分析失败: {analysis['error']}")
                
            print()
            
        return acmi_files
        
    def open_in_tacview(self, acmi_file):
        """在Tacview中打开ACMI文件"""
        tacview_exe = self.find_tacview_exe()
        
        if not tacview_exe:
            print("❌ 未找到Tacview安装，请确保Tacview已正确安装")
            print("💡 Tacview官网: https://www.tacview.net/")
            return False
            
        try:
            print(f"🚀 正在打开Tacview...")
            print(f"📁 文件: {acmi_file}")
            
            # 启动Tacview并打开文件
            subprocess.Popen([tacview_exe, str(acmi_file)])
            print(f"✅ Tacview已启动")
            return True
            
        except Exception as e:
            print(f"❌ 启动Tacview失败: {str(e)}")
            return False
            
    def open_file_explorer(self, acmi_file):
        """在文件资源管理器中显示文件"""
        try:
            subprocess.Popen(['explorer', '/select,', str(acmi_file)])
            print(f"📁 已在文件资源管理器中显示: {acmi_file.name}")
        except Exception as e:
            print(f"❌ 打开文件资源管理器失败: {str(e)}")
            
    def interactive_viewer(self):
        """交互式查看器"""
        print("🎮 Tacview战术结果查看器")
        print("=" * 50)
        
        while True:
            acmi_files = self.display_available_files()
            
            if not acmi_files:
                break
                
            print("操作选项:")
            print("  输入数字 - 在Tacview中打开对应文件")
            print("  f + 数字 - 在文件资源管理器中显示文件")
            print("  r - 刷新列表")
            print("  q - 退出")
            
            try:
                choice = input("\n请输入选择: ").strip().lower()
                
                if choice == 'q':
                    break
                elif choice == 'r':
                    continue
                elif choice.startswith('f'):
                    try:
                        index = int(choice[1:]) - 1
                        if 0 <= index < len(acmi_files):
                            self.open_file_explorer(acmi_files[index])
                    except (ValueError, IndexError):
                        print("❌ 无效选择")
                else:
                    try:
                        index = int(choice) - 1
                        if 0 <= index < len(acmi_files):
                            self.open_in_tacview(acmi_files[index])
                        else:
                            print("❌ 无效选择")
                    except ValueError:
                        print("❌ 无效输入")
                        
            except KeyboardInterrupt:
                print("\n👋 再见!")
                break
                
        print("\n✅ 查看器已退出")

def main():
    """主函数"""
    viewer = TacviewViewer()
    
    print("🎯 检查最新的仿真结果...")
    acmi_files = viewer.get_latest_acmi_files(3)
    
    if acmi_files:
        print(f"\n📊 找到 {len(acmi_files)} 个最新的ACMI文件")
        
        # 显示最新的文件
        latest_file = acmi_files[0]
        tactic = viewer.identify_tactic_from_filename(latest_file.name)
        
        print(f"\n🎯 最新仿真: {latest_file.name}")
        print(f"📋 战术类型: {tactic}")
        
        # 询问是否直接打开最新文件
        response = input(f"\n是否直接在Tacview中打开最新文件? (y/n): ").strip().lower()
        if response == 'y':
            if viewer.open_in_tacview(latest_file):
                print("💡 提示: 在Tacview中可以:")
                print("  - 使用3D视图观察战术执行过程")
                print("  - 调节播放速度查看关键时刻")
                print("  - 切换视角跟随不同飞机")
                print("  - 分析导弹轨迹和命中情况")
        else:
            # 进入交互模式
            viewer.interactive_viewer()
    else:
        print("❌ 未找到任何ACMI文件，请先运行战术仿真")

if __name__ == "__main__":
    main()