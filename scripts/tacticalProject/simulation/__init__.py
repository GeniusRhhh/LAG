"""
仿真模块
包含雷达、导弹等仿真组件
"""
from .radar_types import RadarStatus, ECMType, RadarTarget, APG68RadarModel, N001VERadarModel
from .radar_manager import UnifiedRadarManager
from .r27er_missile import R27ERMissileSimulator

# 创建适配类以兼容新的战术适配器
class FriendlyRadarManager:
    """友方雷达管理器适配类"""
    def __init__(self, aircraft_id: str, radar_type: str = 'APG68'):
        self.aircraft_id = aircraft_id
        self.radar_type = radar_type
        self.unified_manager = UnifiedRadarManager()
    
    def update(self, agent, targets, current_time: float):
        """更新雷达状态"""
        # 使用统一管理器的方法
        pass
    
    def get_status(self):
        """获取雷达状态"""
        return self.unified_manager.get_friendly_radar_state(self.aircraft_id)


class EnemyRadarManager:
    """敌方雷达管理器适配类"""
    def __init__(self, aircraft_id: str, radar_type: str = 'N001VE'):
        self.aircraft_id = aircraft_id
        self.radar_type = radar_type
        self.unified_manager = UnifiedRadarManager()
    
    def update(self, agent, targets, current_time: float):
        """更新雷达状态"""
        # 使用统一管理器的方法
        pass
    
    def get_status(self):
        """获取雷达状态"""
        return self.unified_manager.get_enemy_radar_state(self.aircraft_id)
