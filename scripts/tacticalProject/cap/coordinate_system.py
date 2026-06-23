"""
坐标转换系统
以A0100长机为基准，支持任意航向的战场配置
"""
import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class BattlefieldConfig:
    """战场配置（基准参数）"""
    a0100_lon: float = 120.6757    # A0100经度
    a0100_lat: float = 60.0        # A0100纬度
    a0100_heading: float = 0.0     # A0100航向（度，0=北）
    a0100_x: float = 75.0          # A0100在战场相对坐标的X
    a0100_y: float = 0.0           # A0100在战场相对坐标的Y
    deg_to_km: float = 111.0       # 1度≈111km


class CoordinateSystem:
    """坐标转换器"""
    
    def __init__(self, config: BattlefieldConfig = None):
        self.config = config or BattlefieldConfig()
    
    def battlefield_to_geodetic(self, x_rel: float, y_rel: float) -> Tuple[float, float]:
        """战场相对坐标 → 地球坐标(经度, 纬度)"""
        cfg = self.config
        
        # 相对A0100的偏移
        dx = x_rel - cfg.a0100_x
        dy = y_rel - cfg.a0100_y
        
        # 旋转到地球坐标系（航向角：北=0，顺时针为正）
        theta = np.radians(cfg.a0100_heading)
        dx_earth = dx * np.cos(theta) + dy * np.sin(theta)
        dy_earth = -dx * np.sin(theta) + dy * np.cos(theta)
        
        # 纬度缩放修正 (Equirectangular Projection)
        scale = np.cos(np.radians(cfg.a0100_lat))
        
        # 转换为经纬度
        lon = cfg.a0100_lon + dx_earth / (cfg.deg_to_km * scale)
        lat = cfg.a0100_lat + dy_earth / cfg.deg_to_km
        return lon, lat
    
    def geodetic_to_battlefield(self, lon: float, lat: float) -> Tuple[float, float]:
        """地球坐标 → 战场相对坐标"""
        cfg = self.config
        
        # 纬度缩放修正
        scale = np.cos(np.radians(cfg.a0100_lat))
        
        # 相对A0100的经纬度偏移转为km
        dx_earth = (lon - cfg.a0100_lon) * (cfg.deg_to_km * scale)
        dy_earth = (lat - cfg.a0100_lat) * cfg.deg_to_km
        
        # 逆旋转
        theta = np.radians(-cfg.a0100_heading)
        dx = dx_earth * np.cos(theta) + dy_earth * np.sin(theta)
        dy = -dx_earth * np.sin(theta) + dy_earth * np.cos(theta)
        
        return cfg.a0100_x + dx, cfg.a0100_y + dy
    
    def convert_heading_to_earth(self, heading_rel: float) -> float:
        """战场相对航向 → 地球航向"""
        return (heading_rel + self.config.a0100_heading) % 360
    
    def convert_heading_to_battlefield(self, heading_earth: float) -> float:
        """地球航向 → 战场相对航向"""
        return (heading_earth - self.config.a0100_heading) % 360
