"""
协同探测V4 - 多目标管理器
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from .detection_types import TargetInfo, TargetState


class MultiTargetManager:
    """多目标管理器"""
    
    def __init__(self):
        self._targets: Dict[str, TargetInfo] = {}
        self._lost_time: Dict[str, float] = {}
        self._last_known: Dict[str, TargetState] = {}
    
    def update_from_awacs(self, targets: Dict[str, TargetInfo], current_time: float):
        """从预警机更新目标"""
        for tid, info in targets.items():
            if tid in self._targets:
                # 更新现有目标
                self._targets[tid].x = info.x
                self._targets[tid].y = info.y
                self._targets[tid].vx = info.vx
                self._targets[tid].vy = info.vy
                self._targets[tid].timestamp = current_time
                self._targets[tid].source = 'awacs'
            else:
                # 新目标
                info.timestamp = current_time
                self._targets[tid] = info
            
            # 记录最后已知状态
            self._last_known[tid] = TargetState(
                x=info.x, y=info.y, vx=info.vx, vy=info.vy, timestamp=current_time
            )
            
            # 清除丢失状态
            if tid in self._lost_time:
                del self._lost_time[tid]
    
    def update_from_radar(self, detections: Dict[str, List[Tuple[str, float, float]]], 
                          current_time: float):
        """从雷达更新目标"""
        for fighter_id, targets in detections.items():
            for tid, x, y in targets:
                if tid in self._targets:
                    self._targets[tid].x = x
                    self._targets[tid].y = y
                    self._targets[tid].last_radar_contact = current_time
                    self._targets[tid].is_tracked = True
                    self._targets[tid].source = 'radar'
                    if fighter_id not in self._targets[tid].tracker_ids:
                        self._targets[tid].tracker_ids.append(fighter_id)
                    
                    # 更新最后已知状态
                    self._last_known[tid] = TargetState(
                        x=x, y=y, 
                        vx=self._targets[tid].vx, 
                        vy=self._targets[tid].vy,
                        timestamp=current_time
                    )
                
                # 清除丢失状态
                if tid in self._lost_time:
                    del self._lost_time[tid]
    
    def update_target(self, tid: str, x: float, y: float, vx: float, vy: float,
                      timestamp: float, source: str = 'awacs', error_radius: float = 2.5):
        """更新或创建目标"""
        if tid in self._targets:
            self._targets[tid].x = x
            self._targets[tid].y = y
            self._targets[tid].vx = vx
            self._targets[tid].vy = vy
            self._targets[tid].timestamp = timestamp
            self._targets[tid].source = source
            self._targets[tid].error_radius = error_radius
        else:
            self._targets[tid] = TargetInfo(
                target_id=tid, x=x, y=y, vx=vx, vy=vy,
                timestamp=timestamp, source=source, error_radius=error_radius
            )
        
        # 更新最后已知状态
        self._last_known[tid] = TargetState(
            x=x, y=y, vx=vx, vy=vy, timestamp=timestamp
        )
        
        # 清除丢失状态
        if tid in self._lost_time:
            del self._lost_time[tid]
    
    def update_estimated_state(self, tid: str, state: TargetState):
        """更新估计状态"""
        if tid in self._targets:
            self._targets[tid].x = state.x
            self._targets[tid].y = state.y
            self._targets[tid].vx = state.vx
            self._targets[tid].vy = state.vy
            self._targets[tid].covariance = state.covariance
            self._targets[tid].timestamp = state.timestamp
    
    def check_lost_targets(self, current_time: float, timeout: float) -> List[str]:
        """检查丢失目标"""
        lost = []
        for tid, target in self._targets.items():
            age = current_time - target.timestamp
            if age > timeout:
                if tid not in self._lost_time:
                    self._lost_time[tid] = current_time
                lost.append(tid)
        return lost
    
    def get_lost_targets(self, current_time: float, timeout: float) -> List[str]:
        """获取已丢失超过阈值的目标"""
        return [tid for tid, lost_time in self._lost_time.items()
                if current_time - lost_time >= 0]
    
    def get_all_targets(self) -> Dict[str, TargetInfo]:
        return self._targets.copy()
    
    def get_target(self, tid: str) -> Optional[TargetInfo]:
        return self._targets.get(tid)
    
    def get_last_known_state(self, tid: str) -> Optional[TargetState]:
        return self._last_known.get(tid)
    
    def clear(self):
        self._targets.clear()
        self._lost_time.clear()
        self._last_known.clear()
