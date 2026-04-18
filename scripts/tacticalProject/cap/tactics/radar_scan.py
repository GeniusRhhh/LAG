"""
雷达扫描参数 - 基于需求文档表2
"""
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class RadarMode(Enum):
    """雷达扫描模式"""
    NARROW = "NARROW"    # ±10°
    MEDIUM = "MEDIUM"    # ±30°
    WIDE = "WIDE"        # ±60°


@dataclass(frozen=True)
class RadarScanParams:
    """雷达扫描参数（基于需求文档表2）"""
    # 方位角范围 (度)
    AZIMUTH = {RadarMode.NARROW: 10, RadarMode.MEDIUM: 30, RadarMode.WIDE: 60}
    
    # 扫描时间(秒): [俯仰1行, 俯仰2行, 俯仰4行]
    SCAN_TIME = {
        RadarMode.NARROW: [2, 4, 8],
        RadarMode.MEDIUM: [5, 10, 20],
        RadarMode.WIDE: [10, 20, 40],
    }
    
    detection_range: float = 200.0   # 探测距离(km)
    max_tracks: int = 6              # 最大跟踪目标数
    
    @staticmethod
    def get_scan_time(mode: RadarMode, elevation_bars: int = 2) -> float:
        """获取扫描时间(秒)"""
        bar_idx = {1: 0, 2: 1, 4: 2}.get(elevation_bars, 1)
        return RadarScanParams.SCAN_TIME[mode][bar_idx]
    
    @staticmethod
    def get_azimuth(mode: RadarMode) -> float:
        """获取方位角范围(度)"""
        return RadarScanParams.AZIMUTH[mode]


# 默认参数
DEFAULT_RADAR = RadarScanParams()
