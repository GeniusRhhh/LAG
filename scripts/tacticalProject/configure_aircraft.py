#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
战术项目飞机模型配置管理器
提供简单的命令行界面来选择F16或SU27模型进行战术仿真

使用方法:
1. 快速配置: python configure_aircraft.py --preset f16_vs_f16
2. 自定义配置: python configure_aircraft.py --my-aircraft f16 --enemy-aircraft su27sk
3. 查看当前配置: python configure_aircraft.py --status

作者：Assistant
日期：2025-01-09
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入配置模块
from core.aircraft_model_config import (
    get_aircraft_config, 
    set_aircraft_config,
    quick_setup_f16_vs_f16,
    quick_setup_su27_vs_su27, 
    quick_setup_f16_vs_su27,
    quick_setup_su27_vs_f16,
    AircraftType,
    ModelType
)
from config.tactical_config import get_config, update_config


class TacticalAircraftConfigManager:
    """战术项目飞机配置管理器"""
    
    def __init__(self):
        self.config_file = Path(current_dir) / "config" / "current_aircraft_config.json"
        self.aircraft_config = get_aircraft_config()
        
        # 确保配置目录存在
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
    
    def save_config(self):
        """保存当前配置到文件"""
        config_data = {
            "my_aircraft": self.aircraft_config.current_my_aircraft.value,
            "enemy_aircraft": self.aircraft_config.current_enemy_aircraft.value,
            "my_model": self.aircraft_config.current_my_model.value,
            "enemy_model": self.aircraft_config.current_enemy_model.value,
            "timestamp": "2025-01-09"
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        # 更新战术配置
        tactical_config = get_config('aircraft_model')
        tactical_config.update({
            'my_aircraft': self.aircraft_config.current_my_aircraft.value,
            'enemy_aircraft': self.aircraft_config.current_enemy_aircraft.value,
            'my_model': self.aircraft_config.current_my_model.value,
            'enemy_model': self.aircraft_config.current_enemy_model.value,
        })
        update_config('aircraft_model', tactical_config)
        
        logging.info(f"✅ 配置已保存到: {self.config_file}")
    
    def load_config(self):
        """从文件加载配置"""
        if not self.config_file.exists():
            logging.info("⚠️ 配置文件不存在，使用默认配置")
            return
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 应用配置
            set_aircraft_config(
                my_aircraft=config_data.get('my_aircraft', 'f16'),
                enemy_aircraft=config_data.get('enemy_aircraft', 'f16'),
                my_model=config_data.get('my_model', 'baseline'),
                enemy_model=config_data.get('enemy_model', 'baseline')
            )
            
            logging.info(f"✅ 配置已从文件加载: {self.config_file}")
            
        except Exception as e:
            logging.error(f"❌ 加载配置文件失败: {e}")
            
    def print_status(self):
        """打印当前配置状态"""
        print("\n" + "="*50)
        print("当前飞机模型配置")
        print("="*50)
        print(f"我方飞机: {self.aircraft_config.current_my_aircraft.value}")
        print(f"我方模型: {self.aircraft_config.current_my_model.value}")
        print("-" * 30)
        print(f"敌方飞机: {self.aircraft_config.current_enemy_aircraft.value}")
        print(f"敌方模型: {self.aircraft_config.current_enemy_model.value}")
        print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description='战术项目飞机模型配置工具')
    
    # 预设模式
    parser.add_argument('--preset', type=str, 
                      choices=['f16_vs_f16', 'su27_vs_su27', 'f16_vs_su27', 'su27_vs_f16'],
                      help='使用预设配置')
    
    # 自定义模式
    parser.add_argument('--my-aircraft', type=str, choices=['f16', 'su27sk'], help='我方飞机型号')
    parser.add_argument('--enemy-aircraft', type=str, choices=['f16', 'su27sk'], help='敌方飞机型号')
    parser.add_argument('--my-model', type=str, choices=['baseline', 'ppo', 'su27_baseline'], help='我方模型类型')
    parser.add_argument('--enemy-model', type=str, choices=['baseline', 'ppo', 'su27_baseline'], help='敌方模型类型')
    
    # 查看状态
    parser.add_argument('--status', action='store_true', help='查看当前配置')
    
    args = parser.parse_args()
    
    manager = TacticalAircraftConfigManager()
    
    if args.status:
        manager.load_config()
        manager.print_status()
        return

    if args.preset:
        if args.preset == 'f16_vs_f16':
            quick_setup_f16_vs_f16()
        elif args.preset == 'su27_vs_su27':
            quick_setup_su27_vs_su27()
        elif args.preset == 'f16_vs_su27':
            quick_setup_f16_vs_su27()
        elif args.preset == 'su27_vs_f16':
            quick_setup_su27_vs_f16()
        print(f"✅ 已应用预设: {args.preset}")
        manager.save_config()
        manager.print_status()
        return
        
    if args.my_aircraft or args.enemy_aircraft:
        # 加载现有配置作为基础
        manager.load_config()
        
        # 更新配置
        set_aircraft_config(
            my_aircraft=args.my_aircraft or manager.aircraft_config.current_my_aircraft.value,
            enemy_aircraft=args.enemy_aircraft or manager.aircraft_config.current_enemy_aircraft.value,
            my_model=args.my_model or manager.aircraft_config.current_my_model.value,
            enemy_model=args.enemy_model or manager.aircraft_config.current_enemy_model.value
        )
        manager.save_config()
        manager.print_status()
        return
        
    # 如果没有参数，显示帮助
    parser.print_help()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
