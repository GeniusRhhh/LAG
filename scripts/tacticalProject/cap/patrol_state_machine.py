"""
巡逻状态机
管理单架飞机的跑马圈巡逻状态
"""
from enum import Enum
from typing import Tuple
from dataclasses import dataclass


class PatrolState(Enum):
    """巡逻状态"""
    HOT_NORTH = "HOT_NORTH"       # 热段朝北
    TRANS_WEST = "TRANS_WEST"     # 过渡朝西
    COLD_SOUTH = "COLD_SOUTH"     # 冷段朝南
    TRANS_EAST = "TRANS_EAST"     # 过渡朝东
    INTERCEPT = "INTERCEPT"       # 拦截模式


@dataclass
class PatrolBox:
    """巡逻区域"""
    x_min: float  # 最小X（僚机X）
    x_max: float  # 最大X（长机X）
    y_min: float  # 最小Y（0）
    y_max: float  # 最大Y（僚机Y）


class PatrolStateMachine:
    """巡逻状态机（逆时针，全部左转）"""
    
    # 状态对应的目标航向（战场相对坐标，北=0）
    STATE_HEADINGS = {
        PatrolState.HOT_NORTH: 0,      # 朝北
        PatrolState.TRANS_WEST: 270,   # 朝西
        PatrolState.COLD_SOUTH: 180,   # 朝南
        PatrolState.TRANS_EAST: 90,    # 朝东
    }
    
    def __init__(self, patrol_box: PatrolBox, initial_hot: bool = True):
        """
        Args:
            patrol_box: 巡逻区域
            initial_hot: 初始是否为热段
        """
        self.box = patrol_box
        self.state = PatrolState.HOT_NORTH if initial_hot else PatrolState.COLD_SOUTH
        self.tolerance = 2.0  # km
    
    def update(self, x: float, y: float) -> Tuple[PatrolState, float]:
        """更新状态，返回(新状态, 目标航向)
        
        Args:
            x, y: 当前战场相对坐标
            
        Returns:
            (state, target_heading): 当前状态和目标航向（战场相对）
        """
        if self.state == PatrolState.INTERCEPT:
            return self.state, 0  # 拦截模式由外部控制
        
        b = self.box
        t = self.tolerance
        
        # 状态转换逻辑（逆时针）
        if self.state == PatrolState.HOT_NORTH:
            if y >= b.y_max - t:
                self.state = PatrolState.TRANS_WEST
        
        elif self.state == PatrolState.TRANS_WEST:
            if x <= b.x_min + t:
                self.state = PatrolState.COLD_SOUTH
        
        elif self.state == PatrolState.COLD_SOUTH:
            if y <= b.y_min + t:
                self.state = PatrolState.TRANS_EAST
        
        elif self.state == PatrolState.TRANS_EAST:
            if x >= b.x_max - t:
                self.state = PatrolState.HOT_NORTH
        
        return self.state, self.STATE_HEADINGS.get(self.state, 0)
    
    def enter_intercept(self):
        """进入拦截模式"""
        self.state = PatrolState.INTERCEPT
    
    def exit_intercept(self, resume_hot: bool = True):
        """退出拦截模式"""
        self.state = PatrolState.HOT_NORTH if resume_hot else PatrolState.COLD_SOUTH
    
    def is_hot(self) -> bool:
        """是否处于热段"""
        return self.state == PatrolState.HOT_NORTH
    
    def is_cold(self) -> bool:
        """是否处于冷段"""
        return self.state == PatrolState.COLD_SOUTH
