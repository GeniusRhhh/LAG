"""
协同探测V4 - 丢失目标预测器
简化版: 运动学预测
专业版: 粒子滤波预测
"""
import numpy as np
from typing import Dict, Optional
from ..detection_types import TargetState, SearchRegion


class KinematicPredictor:
    """运动学预测器（简化版）"""
    
    def predict(self, last_state: TargetState, lost_duration: float) -> SearchRegion:
        """位置 + 速度 × 时间"""
        # 预测中心
        pred_x = last_state.x + last_state.vx * lost_duration
        pred_y = last_state.y + last_state.vy * lost_duration
        
        # 搜索半径：考虑可能的转向
        max_turn_rate = 3.0  # 度/秒
        max_turn = max_turn_rate * lost_duration
        speed = np.sqrt(last_state.vx**2 + last_state.vy**2)
        
        # 横向偏移
        lateral_offset = speed * lost_duration * np.sin(np.radians(min(max_turn, 90)))
        
        # 搜索半径
        search_radius = max(5.0, lateral_offset + 2.0)
        
        # 计算扫描角度
        scan_range = np.degrees(np.arctan2(search_radius, 100)) * 2
        scan_range = max(10.0, min(30.0, scan_range))
        
        return SearchRegion(
            center_x=pred_x,
            center_y=pred_y,
            radius=search_radius,
            scan_range=scan_range,
            priority=1.0,
            bearing=np.degrees(np.arctan2(pred_x, pred_y)) % 360
        )


class ParticleFilterPredictor:
    """粒子滤波预测器（专业版）"""
    
    def __init__(self, n_particles: int = 1000):
        self.n_particles = n_particles
        self._particles: Dict[str, np.ndarray] = {}
    
    def predict(self, last_state: TargetState, lost_duration: float,
                target_id: str = None) -> SearchRegion:
        """粒子滤波预测"""
        # 初始化粒子 [x, y, vx, vy]
        particles = np.zeros((self.n_particles, 4))
        particles[:, 0] = last_state.x + np.random.normal(0, 1.0, self.n_particles)
        particles[:, 1] = last_state.y + np.random.normal(0, 1.0, self.n_particles)
        particles[:, 2] = last_state.vx + np.random.normal(0, 0.05, self.n_particles)
        particles[:, 3] = last_state.vy + np.random.normal(0, 0.05, self.n_particles)
        
        # 传播粒子（随机机动模型）
        dt = 0.5  # 传播步长
        steps = max(1, int(lost_duration / dt))
        
        for _ in range(steps):
            # 随机加速度（模拟机动）
            ax = np.random.normal(0, 0.01, self.n_particles)
            ay = np.random.normal(0, 0.01, self.n_particles)
            
            # 更新速度
            particles[:, 2] += ax * dt
            particles[:, 3] += ay * dt
            
            # 限制速度
            speed = np.sqrt(particles[:, 2]**2 + particles[:, 3]**2)
            max_speed = 0.4  # km/s
            mask = speed > max_speed
            if np.any(mask):
                particles[mask, 2] *= max_speed / speed[mask]
                particles[mask, 3] *= max_speed / speed[mask]
            
            # 更新位置
            particles[:, 0] += particles[:, 2] * dt
            particles[:, 1] += particles[:, 3] * dt
        
        # 计算搜索区域
        center_x = np.mean(particles[:, 0])
        center_y = np.mean(particles[:, 1])
        
        # 99%置信区间
        std_x = np.std(particles[:, 0])
        std_y = np.std(particles[:, 1])
        radius = 2.576 * max(std_x, std_y)  # 99%置信
        radius = max(5.0, min(50.0, radius))
        
        # 计算扫描角度
        scan_range = np.degrees(np.arctan2(radius, 100)) * 2
        scan_range = max(10.0, min(40.0, scan_range))
        
        # 保存粒子
        if target_id:
            self._particles[target_id] = particles.copy()
        
        return SearchRegion(
            center_x=center_x,
            center_y=center_y,
            radius=radius,
            scan_range=scan_range,
            priority=1.0,
            bearing=np.degrees(np.arctan2(center_x, center_y)) % 360,
            particles=particles
        )
    
    def get_particles(self, target_id: str) -> Optional[np.ndarray]:
        """获取目标的粒子分布"""
        return self._particles.get(target_id)
    
    def update_with_detection(self, target_id: str, detected_pos: tuple,
                              detection_noise: float = 1.0):
        """检测到目标后更新粒子"""
        if target_id not in self._particles:
            return
        
        particles = self._particles[target_id]
        
        # 计算权重（基于与检测位置的距离）
        dx = particles[:, 0] - detected_pos[0]
        dy = particles[:, 1] - detected_pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        weights = np.exp(-0.5 * (dist / detection_noise)**2)
        weights /= np.sum(weights)
        
        # 重采样
        indices = np.random.choice(self.n_particles, self.n_particles, p=weights)
        self._particles[target_id] = particles[indices]
