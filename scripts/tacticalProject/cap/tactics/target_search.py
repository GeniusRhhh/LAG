"""
目标丢失搜索算法模块
- 简单版：超时切换搜索模式（已在cooperative_detection.py实现）
- 专业版：基于预测的定向搜索
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class SearchMode(Enum):
    """搜索模式"""
    NONE = "NONE"           # 无需搜索
    DIRECTED = "DIRECTED"   # 定向搜索（基于预测位置）
    SECTOR = "SECTOR"       # 扇区搜索（上次位置周围）
    FULL = "FULL"           # 全域搜索


@dataclass
class SearchTask:
    """搜索任务"""
    target_id: str
    mode: SearchMode
    predicted_position: Optional[Tuple[float, float]]  # 预测位置
    search_bearing: float  # 搜索方位
    search_range: float   # 搜索范围(度)
    priority: float       # 优先级


class SimpleTargetSearch:
    """简单目标搜索 - 超时全域搜索"""
    
    def __init__(self, lost_timeout: float = 20.0):
        self.lost_timeout = lost_timeout
        self._lost_times: Dict[str, float] = {}  # 目标丢失时刻
        self._last_positions: Dict[str, Tuple[float, float]] = {}  # 最后已知位置
    
    def update_track(self, target_id: str, position: Tuple[float, float], current_time: float):
        """更新目标位置（目标在跟踪中）"""
        self._last_positions[target_id] = position
        if target_id in self._lost_times:
            del self._lost_times[target_id]
    
    def mark_lost(self, target_id: str, current_time: float):
        """标记目标丢失"""
        if target_id not in self._lost_times:
            self._lost_times[target_id] = current_time
    
    def get_search_tasks(self, current_time: float) -> List[SearchTask]:
        """获取搜索任务"""
        tasks = []
        for tid, lost_time in self._lost_times.items():
            elapsed = current_time - lost_time
            if elapsed > self.lost_timeout:
                # 超时 → 全域搜索
                last_pos = self._last_positions.get(tid)
                tasks.append(SearchTask(
                    target_id=tid,
                    mode=SearchMode.FULL,
                    predicted_position=last_pos,
                    search_bearing=0,
                    search_range=120,  # ±60°
                    priority=1.0
                ))
        return tasks
    
    def is_target_lost(self, target_id: str, current_time: float) -> bool:
        """检查目标是否丢失超时"""
        if target_id not in self._lost_times:
            return False
        return current_time - self._lost_times[target_id] > self.lost_timeout


class PredictiveTargetSearch:
    """预测搜索 - 基于运动模型预测目标位置
    
    当目标丢失时：
    1. 使用最后已知状态预测当前位置
    2. 计算搜索方位和范围
    3. 优先在预测位置周围搜索
    """
    
    def __init__(self, lost_timeout: float = 20.0, prediction_horizon: float = 30.0):
        self.lost_timeout = lost_timeout
        self.prediction_horizon = prediction_horizon
        self._lost_times: Dict[str, float] = {}
        self._last_states: Dict[str, dict] = {}  # 包含位置、速度、航向
    
    def update_track(self, target_id: str, 
                     position: Tuple[float, float],
                     velocity: Tuple[float, float],
                     heading: float,
                     current_time: float):
        """更新目标状态"""
        self._last_states[target_id] = {
            'position': position,
            'velocity': velocity,
            'heading': heading,
            'time': current_time
        }
        if target_id in self._lost_times:
            del self._lost_times[target_id]
    
    def mark_lost(self, target_id: str, current_time: float):
        """标记目标丢失"""
        if target_id not in self._lost_times:
            self._lost_times[target_id] = current_time
    
    def predict_position(self, target_id: str, current_time: float) -> Optional[Tuple[float, float]]:
        """预测目标当前位置"""
        if target_id not in self._last_states:
            return None
        
        state = self._last_states[target_id]
        dt = current_time - state['time']
        
        # 限制预测时间
        dt = min(dt, self.prediction_horizon)
        
        # 线性预测
        x = state['position'][0] + state['velocity'][0] * dt / 1000  # m/s to km/s
        y = state['position'][1] + state['velocity'][1] * dt / 1000
        
        return (x, y)
    
    def get_search_tasks(self, current_time: float, 
                         my_position: Tuple[float, float] = (75, 50)) -> List[SearchTask]:
        """获取搜索任务"""
        tasks = []
        
        for tid, lost_time in self._lost_times.items():
            elapsed = current_time - lost_time
            
            if elapsed <= self.lost_timeout:
                # 未超时 → 定向搜索
                pred_pos = self.predict_position(tid, current_time)
                if pred_pos:
                    # 计算相对方位
                    dx = pred_pos[0] - my_position[0]
                    dy = pred_pos[1] - my_position[1]
                    bearing = np.degrees(np.arctan2(dx, dy)) % 360
                    
                    # 搜索范围随时间增大
                    search_range = 10 + elapsed  # 10° + 1°/秒
                    
                    tasks.append(SearchTask(
                        target_id=tid,
                        mode=SearchMode.DIRECTED,
                        predicted_position=pred_pos,
                        search_bearing=bearing,
                        search_range=min(30, search_range),
                        priority=1.0 / (elapsed + 1)
                    ))
            else:
                # 超时 → 扇区搜索
                last_state = self._last_states.get(tid)
                if last_state:
                    # 以最后位置为中心的扇区搜索
                    dx = last_state['position'][0] - my_position[0]
                    dy = last_state['position'][1] - my_position[1]
                    bearing = np.degrees(np.arctan2(dx, dy)) % 360
                    
                    tasks.append(SearchTask(
                        target_id=tid,
                        mode=SearchMode.SECTOR,
                        predicted_position=last_state['position'],
                        search_bearing=bearing,
                        search_range=60,  # 较宽扇区
                        priority=0.5
                    ))
                else:
                    # 无历史数据 → 全域搜索
                    tasks.append(SearchTask(
                        target_id=tid,
                        mode=SearchMode.FULL,
                        predicted_position=None,
                        search_bearing=0,
                        search_range=120,
                        priority=0.3
                    ))
        
        return tasks
    
    def is_target_lost(self, target_id: str, current_time: float) -> bool:
        """检查目标是否丢失"""
        return target_id in self._lost_times


def create_target_search(use_prediction: bool = False, **kwargs):
    """工厂函数：创建目标搜索器"""
    if use_prediction:
        return PredictiveTargetSearch(**kwargs)
    else:
        return SimpleTargetSearch(**kwargs)
