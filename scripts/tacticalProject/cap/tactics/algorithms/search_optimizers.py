"""
协同探测V4 - 搜索策略优化器
简化版: 均匀分配搜索区域
专业版: 信息熵最大化
"""
import numpy as np
from typing import Dict, List, Optional
from ..detection_types import FighterInfo, SearchRegion


class UniformSearchOptimizer:
    """均匀分配搜索区域（简化版）"""
    
    def optimize(self, search_regions: Dict[str, SearchRegion],
                 fighters: List[FighterInfo],
                 probability_map: Optional[np.ndarray] = None) -> Dict[str, SearchRegion]:
        """均匀分配"""
        if not search_regions or not fighters:
            return {}
        
        assignments = {}
        regions = list(search_regions.values())
        
        # 简单轮询分配
        for i, fighter in enumerate(fighters):
            if i < len(regions):
                assignments[fighter.fighter_id] = regions[i]
            else:
                # 战机多于区域时，分配最近的区域
                nearest = min(regions, key=lambda r: 
                    np.sqrt((fighter.x - r.center_x)**2 + (fighter.y - r.center_y)**2))
                assignments[fighter.fighter_id] = nearest
        
        return assignments


class InfoEntropyOptimizer:
    """信息熵最大化搜索优化器（专业版）"""
    
    def __init__(self, grid_resolution: float = 1.0):
        self.grid_resolution = grid_resolution
    
    def optimize(self, search_regions: Dict[str, SearchRegion],
                 fighters: List[FighterInfo],
                 probability_map: Optional[np.ndarray] = None) -> Dict[str, SearchRegion]:
        """最大化信息增益分配"""
        if not search_regions or not fighters:
            return {}
        
        # 无概率图时回退到均匀分配
        if probability_map is None:
            return UniformSearchOptimizer().optimize(search_regions, fighters, None)
        
        assignments = {}
        available_fighters = list(fighters)
        assigned_regions = set()
        
        # 计算每个区域的信息增益
        region_gains = {}
        for tid, region in search_regions.items():
            gain = self._calc_info_gain(region, probability_map)
            region_gains[tid] = gain
        
        # 按信息增益排序
        sorted_regions = sorted(region_gains.items(), key=lambda x: x[1], reverse=True)
        
        # 贪心分配：优先分配高增益区域
        for tid, gain in sorted_regions:
            if not available_fighters:
                break
            if tid in assigned_regions:
                continue
            
            region = search_regions[tid]
            
            # 选择最近的可用战机
            best_fighter = min(available_fighters,
                key=lambda f: np.sqrt((f.x - region.center_x)**2 + 
                                     (f.y - region.center_y)**2))
            
            # 调整优先级
            adjusted_region = SearchRegion(
                center_x=region.center_x,
                center_y=region.center_y,
                radius=region.radius,
                scan_range=region.scan_range,
                priority=gain,
                bearing=region.bearing,
                particles=region.particles
            )
            
            assignments[best_fighter.fighter_id] = adjusted_region
            available_fighters.remove(best_fighter)
            assigned_regions.add(tid)
        
        # 剩余战机分配到次优区域
        remaining_regions = [r for tid, r in search_regions.items() 
                           if tid not in assigned_regions]
        for fighter in available_fighters:
            if remaining_regions:
                nearest = min(remaining_regions, key=lambda r:
                    np.sqrt((fighter.x - r.center_x)**2 + (fighter.y - r.center_y)**2))
                assignments[fighter.fighter_id] = nearest
                remaining_regions.remove(nearest)
        
        return assignments
    
    def _calc_info_gain(self, region: SearchRegion, 
                        probability_map: np.ndarray) -> float:
        """计算区域的信息增益"""
        # 获取概率图尺寸
        nx, ny = probability_map.shape
        
        # 假设概率图覆盖 [-100, 100] x [0, 300] km
        x_min, x_max = -100.0, 100.0
        y_min, y_max = 0.0, 300.0
        
        # 区域中心转换为网格索引
        cx = int((region.center_x - x_min) / (x_max - x_min) * nx)
        cy = int((region.center_y - y_min) / (y_max - y_min) * ny)
        r = int(region.radius / ((x_max - x_min) / nx))
        
        # 边界检查
        cx = max(0, min(nx-1, cx))
        cy = max(0, min(ny-1, cy))
        r = max(1, r)
        
        # 计算区域内的概率和（信息增益近似）
        prob_sum = 0.0
        count = 0
        for ix in range(max(0, cx-r), min(nx, cx+r+1)):
            for iy in range(max(0, cy-r), min(ny, cy+r+1)):
                dist = np.sqrt((ix-cx)**2 + (iy-cy)**2)
                if dist <= r:
                    prob_sum += probability_map[ix, iy]
                    count += 1
        
        if count == 0:
            return 0.0
        
        # 信息增益 = 区域概率 × 搜索效率
        avg_prob = prob_sum / count
        efficiency = 1.0 / (region.radius + 1)  # 小区域效率高
        
        return avg_prob * efficiency * 100  # 放大以便排序
