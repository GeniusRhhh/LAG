"""
FAOR责任区管理
管理风险区划分、靶眼计算
"""
import numpy as np
from enum import Enum
from typing import Tuple
from dataclasses import dataclass


class RiskZone(Enum):
    """风险区枚举"""
    HIGH = "HIGH"         # 高风险区 (y: 0-100km)
    MEDIUM = "MEDIUM"     # 中风险区 (y: 100-200km)
    LOW = "LOW"           # 低风险区 (y: 200-300km)
    OUTSIDE = "OUTSIDE"   # 责任区外


@dataclass
class FAORBoundary:
    """FAOR边界"""
    x_min: float = 0.0
    x_max: float = 200.0
    y_min: float = 0.0
    y_max: float = 300.0


class FAORManager:
    """FAOR责任区管理器"""
    
    def __init__(self, width: float = 200.0, length: float = 300.0):
        self.width = width
        self.length = length
        self.boundary = FAORBoundary(0, width, 0, length)
        
        # 风险区Y边界（从南到北：高→中→低）
        self.high_y_max = length / 3       # 100km
        self.medium_y_max = length * 2 / 3  # 200km
        
        # 靶眼位置（北边界中心）
        self.bullseye = (width / 2, length)  # (100, 300)
    
    def get_risk_zone(self, x: float, y: float) -> RiskZone:
        """判断位置所在风险区"""
        # 检查是否在FAOR内
        if not (0 <= x <= self.width and 0 <= y <= self.length):
            return RiskZone.OUTSIDE
        
        if y < self.high_y_max:
            return RiskZone.HIGH
        elif y < self.medium_y_max:
            return RiskZone.MEDIUM
        else:
            return RiskZone.LOW

    def get_bullseye(self) -> Tuple[float, float]:
        """返回靶眼位置（战场相对坐标，km）。"""
        return self.bullseye
    
    def get_bullseye_relative(self, x: float, y: float) -> Tuple[float, float]:
        """计算相对靶眼的方位和距离
        
        Returns:
            (bearing, distance): 方位角(度，北=0)，距离(km)
        """
        dx = x - self.bullseye[0]
        dy = y - self.bullseye[1]
        distance = np.sqrt(dx**2 + dy**2)
        bearing = np.degrees(np.arctan2(dx, dy)) % 360
        return bearing, distance
    
    def is_inside(self, x: float, y: float) -> bool:
        """检查是否在FAOR内"""
        return 0 <= x <= self.width and 0 <= y <= self.length

    # 兼容文档/旧命名
    def is_inside_faor(self, x: float, y: float) -> bool:
        return self.is_inside(x, y)
