"""
协同探测V4 - 概率图管理器
简化版: 简单概率图
专业版: 贝叶斯概率图
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from ..detection_types import SearchRegion


class SimpleProbabilityMap:
    """简单概率图（简化版）"""
    
    def __init__(self, width: float = 200.0, length: float = 300.0,
                 resolution: float = 2.0):
        self.width = width
        self.length = length
        self.resolution = resolution
        
        self.nx = int(width / resolution)
        self.ny = int(length / resolution)
        
        # 初始化均匀概率
        self._map = np.ones((self.nx, self.ny)) / (self.nx * self.ny)
    
    def initialize_uniform(self):
        """初始化为均匀分布"""
        self._map = np.ones((self.nx, self.ny)) / (self.nx * self.ny)
    
    def initialize_from_region(self, region: SearchRegion):
        """从搜索区域初始化"""
        self._map = np.zeros((self.nx, self.ny))
        
        # 区域中心转换为网格索引
        cx = int((region.center_x + self.width/2) / self.resolution)
        cy = int(region.center_y / self.resolution)
        r = int(region.radius / self.resolution)
        
        # 高斯分布
        for ix in range(max(0, cx-r*2), min(self.nx, cx+r*2+1)):
            for iy in range(max(0, cy-r*2), min(self.ny, cy+r*2+1)):
                dist = np.sqrt((ix-cx)**2 + (iy-cy)**2) * self.resolution
                self._map[ix, iy] = np.exp(-0.5 * (dist / region.radius)**2)
        
        # 归一化
        total = np.sum(self._map)
        if total > 1e-10:
            self._map /= total
    
    def update_after_search(self, searched_regions: Dict[str, SearchRegion],
                           found_targets: List[str]):
        """搜索后更新（简化版：直接降低搜索区域概率）"""
        if found_targets:
            # 发现目标，重置概率图
            self.initialize_uniform()
            return
        
        detection_prob = 0.7  # 探测概率
        
        for fid, region in searched_regions.items():
            cx = int((region.center_x + self.width/2) / self.resolution)
            cy = int(region.center_y / self.resolution)
            r = int(region.radius / self.resolution)
            
            for ix in range(max(0, cx-r), min(self.nx, cx+r+1)):
                for iy in range(max(0, cy-r), min(self.ny, cy+r+1)):
                    dist = np.sqrt((ix-cx)**2 + (iy-cy)**2)
                    if dist <= r:
                        # 未发现：降低概率
                        self._map[ix, iy] *= (1 - detection_prob)
        
        # 归一化
        total = np.sum(self._map)
        if total > 1e-10:
            self._map /= total
    
    def get_map(self) -> np.ndarray:
        return self._map.copy()
    
    def get_high_prob_regions(self, threshold: float = 0.01) -> List[Tuple[float, float, float]]:
        """获取高概率区域 [(x, y, prob), ...]"""
        regions = []
        for ix in range(self.nx):
            for iy in range(self.ny):
                if self._map[ix, iy] > threshold:
                    x = ix * self.resolution - self.width/2
                    y = iy * self.resolution
                    regions.append((x, y, self._map[ix, iy]))
        return sorted(regions, key=lambda r: r[2], reverse=True)


class BayesianProbabilityMap:
    """贝叶斯概率图（专业版）"""
    
    def __init__(self, width: float = 200.0, length: float = 300.0,
                 resolution: float = 1.0):
        self.width = width
        self.length = length
        self.resolution = resolution
        
        self.nx = int(width / resolution)
        self.ny = int(length / resolution)
        
        # 初始化均匀概率
        self._map = np.ones((self.nx, self.ny)) / (self.nx * self.ny)
        
        # 探测概率模型参数
        self.base_detection_prob = 0.8
        self.detection_decay = 0.1  # 距离衰减
    
    def initialize_uniform(self):
        """初始化为均匀分布"""
        self._map = np.ones((self.nx, self.ny)) / (self.nx * self.ny)
    
    def initialize_from_particles(self, particles: np.ndarray):
        """从粒子初始化概率图"""
        self._map = np.zeros((self.nx, self.ny))
        
        for p in particles:
            ix = int((p[0] + self.width/2) / self.resolution)
            iy = int(p[1] / self.resolution)
            
            if 0 <= ix < self.nx and 0 <= iy < self.ny:
                self._map[ix, iy] += 1
        
        # 高斯平滑
        self._map = self._gaussian_smooth(self._map, sigma=2.0)
        
        # 归一化
        total = np.sum(self._map)
        if total > 1e-10:
            self._map /= total
    
    def _gaussian_smooth(self, data: np.ndarray, sigma: float) -> np.ndarray:
        """高斯平滑"""
        from scipy.ndimage import gaussian_filter
        try:
            return gaussian_filter(data, sigma=sigma)
        except ImportError:
            # 简单平滑
            kernel_size = int(sigma * 3)
            result = data.copy()
            for _ in range(kernel_size):
                padded = np.pad(result, 1, mode='edge')
                result = (padded[:-2, 1:-1] + padded[2:, 1:-1] + 
                         padded[1:-1, :-2] + padded[1:-1, 2:] + 
                         4 * padded[1:-1, 1:-1]) / 8
            return result
    
    def update_after_search(self, searched_regions: Dict[str, SearchRegion],
                           found_targets: List[str]):
        """贝叶斯更新"""
        if found_targets:
            # 发现目标，重置
            self.initialize_uniform()
            return
        
        for fid, region in searched_regions.items():
            # 计算搜索区域的网格索引
            cx = int((region.center_x + self.width/2) / self.resolution)
            cy = int(region.center_y / self.resolution)
            r = int(region.radius / self.resolution)
            
            for ix in range(max(0, cx-r), min(self.nx, cx+r+1)):
                for iy in range(max(0, cy-r), min(self.ny, cy+r+1)):
                    dist = np.sqrt((ix-cx)**2 + (iy-cy)**2) * self.resolution
                    if dist <= region.radius:
                        # 计算该位置的探测概率
                        detection_prob = self.base_detection_prob * \
                            np.exp(-self.detection_decay * dist / region.radius)
                        
                        # 贝叶斯更新
                        # P(target|not_detected) ∝ P(not_detected|target) × P(target)
                        # = (1 - detection_prob) × P(target)
                        self._map[ix, iy] *= (1 - detection_prob)
        
        # 归一化
        total = np.sum(self._map)
        if total > 1e-10:
            self._map /= total
        else:
            # 概率过低，重置
            self.initialize_uniform()
    
    def update_with_detection(self, detected_pos: Tuple[float, float],
                              detection_noise: float = 1.0):
        """检测到目标后更新"""
        # 在检测位置附近增加概率
        cx = int((detected_pos[0] + self.width/2) / self.resolution)
        cy = int(detected_pos[1] / self.resolution)
        r = int(detection_noise * 3 / self.resolution)
        
        for ix in range(max(0, cx-r), min(self.nx, cx+r+1)):
            for iy in range(max(0, cy-r), min(self.ny, cy+r+1)):
                dist = np.sqrt((ix-cx)**2 + (iy-cy)**2) * self.resolution
                # 高斯增加
                self._map[ix, iy] += np.exp(-0.5 * (dist / detection_noise)**2)
        
        # 归一化
        total = np.sum(self._map)
        if total > 1e-10:
            self._map /= total
    
    def get_map(self) -> np.ndarray:
        return self._map.copy()
    
    def get_entropy(self) -> float:
        """计算当前概率图的熵"""
        # 避免log(0)
        p = self._map.flatten()
        p = p[p > 1e-10]
        return -np.sum(p * np.log(p))
    
    def get_max_prob_location(self) -> Tuple[float, float, float]:
        """获取最大概率位置"""
        idx = np.argmax(self._map)
        ix, iy = np.unravel_index(idx, self._map.shape)
        x = ix * self.resolution - self.width/2
        y = iy * self.resolution
        return x, y, self._map[ix, iy]
    
    def get_high_prob_regions(self, threshold: float = 0.001,
                              max_regions: int = 5) -> List[Tuple[float, float, float]]:
        """获取高概率区域"""
        regions = []
        for ix in range(self.nx):
            for iy in range(self.ny):
                if self._map[ix, iy] > threshold:
                    x = ix * self.resolution - self.width/2
                    y = iy * self.resolution
                    regions.append((x, y, self._map[ix, iy]))
        
        # 按概率排序，返回前N个
        regions.sort(key=lambda r: r[2], reverse=True)
        return regions[:max_regions]
