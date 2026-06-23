"""
飞机模型配置管理模块
用于管理仿真中使用的飞机型号（F16/SU27）和模型类型（Baseline/PPO）
"""
from enum import Enum
from typing import Dict, Any, Optional
import logging

class AircraftType(Enum):
    """飞机型号枚举"""
    F16 = "f16"
    SU27 = "su27sk"

class ModelType(Enum):
    """AI模型类型枚举"""
    BASELINE = "baseline"
    PPO = "ppo"
    SU27_BASELINE = "su27_baseline"

class AircraftModelConfig:
    """飞机模型配置类"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AircraftModelConfig, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # 默认配置
        self.current_my_aircraft = AircraftType.F16
        self.current_enemy_aircraft = AircraftType.F16
        self.current_my_model = ModelType.BASELINE
        self.current_enemy_model = ModelType.BASELINE
        
        self._initialized = True
        logging.info("✅ 飞机模型配置初始化完成")

    def set_config(self, my_aircraft: str = None, enemy_aircraft: str = None, 
                  my_model: str = None, enemy_model: str = None):
        """设置配置"""
        if my_aircraft:
            self.current_my_aircraft = AircraftType(my_aircraft)
        if enemy_aircraft:
            self.current_enemy_aircraft = AircraftType(enemy_aircraft)
        if my_model:
            self.current_my_model = ModelType(my_model)
        if enemy_model:
            self.current_enemy_model = ModelType(enemy_model)
            
        logging.info(f"🔄 更新飞机配置: 我方={self.current_my_aircraft.value}, 敌方={self.current_enemy_aircraft.value}")

# 全局单例
_config = AircraftModelConfig()

def get_aircraft_config():
    """获取配置单例"""
    return _config

def set_aircraft_config(my_aircraft=None, enemy_aircraft=None, my_model=None, enemy_model=None):
    """设置飞机配置"""
    _config.set_config(my_aircraft, enemy_aircraft, my_model, enemy_model)

# 快捷设置函数
def quick_setup_f16_vs_f16():
    set_aircraft_config('f16', 'f16', 'baseline', 'baseline')

def quick_setup_su27_vs_su27():
    set_aircraft_config('su27sk', 'su27sk', 'su27_baseline', 'su27_baseline')

def quick_setup_f16_vs_su27():
    set_aircraft_config('f16', 'su27sk', 'baseline', 'su27_baseline')

def quick_setup_su27_vs_f16():
    set_aircraft_config('su27sk', 'f16', 'su27_baseline', 'baseline')
