"""
高级航迹融合算法模块
- 简单版：加权平均融合（已在track_fusion.py实现）
- 专业版：联邦滤波融合
"""
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .picture import Track, FusedTrack, TrackSource


@dataclass
class FusionResult:
    """融合结果"""
    position: np.ndarray
    velocity: np.ndarray
    heading: float
    confidence: float
    covariance: Optional[np.ndarray] = None
    source_count: int = 1


class SimpleFusion:
    """简单融合 - 加权平均
    
    与track_fusion.py中的实现一致
    """
    
    def __init__(self, awacs_weight: float = 0.3, radar_weight: float = 0.7):
        self.awacs_weight = awacs_weight
        self.radar_weight = radar_weight
    
    def fuse(self, awacs_track: Optional[Track], radar_track: Optional[Track]) -> FusionResult:
        """加权平均融合"""
        if awacs_track is None and radar_track is None:
            raise ValueError("At least one track required")
        
        if awacs_track is None:
            return FusionResult(
                position=radar_track.position.copy(),
                velocity=radar_track.velocity.copy(),
                heading=radar_track.heading,
                confidence=radar_track.confidence,
                source_count=1
            )
        
        if radar_track is None:
            return FusionResult(
                position=awacs_track.position.copy(),
                velocity=awacs_track.velocity.copy(),
                heading=awacs_track.heading,
                confidence=awacs_track.confidence,
                source_count=1
            )
        
        # 双源融合
        w_a, w_r = self.awacs_weight, self.radar_weight
        
        fused_pos = w_a * awacs_track.position + w_r * radar_track.position
        fused_vel = w_a * awacs_track.velocity + w_r * radar_track.velocity
        fused_hdg = self._fuse_heading(awacs_track.heading, radar_track.heading, w_a, w_r)
        fused_conf = min(1.0, w_a * awacs_track.confidence + w_r * radar_track.confidence)
        
        return FusionResult(
            position=fused_pos,
            velocity=fused_vel,
            heading=fused_hdg,
            confidence=fused_conf,
            source_count=2
        )
    
    def _fuse_heading(self, h1: float, h2: float, w1: float, w2: float) -> float:
        """融合航向（处理角度环绕）"""
        r1, r2 = np.radians(h1), np.radians(h2)
        x = w1 * np.cos(r1) + w2 * np.cos(r2)
        y = w1 * np.sin(r1) + w2 * np.sin(r2)
        return np.degrees(np.arctan2(y, x)) % 360


class FederatedFusion:
    """联邦滤波融合 - 专业版
    
    原理：
    1. 每个传感器独立维护局部估计
    2. 根据各传感器协方差进行信息加权融合
    3. 协方差小（精度高）的传感器权重大
    """
    
    def __init__(self, awacs_cov: float = 2.5**2, radar_cov: float = 0.1**2):
        """
        Args:
            awacs_cov: 预警机位置方差 (km^2)
            radar_cov: 雷达位置方差 (km^2)
        """
        self.awacs_cov = awacs_cov
        self.radar_cov = radar_cov
    
    def fuse(self, awacs_track: Optional[Track], radar_track: Optional[Track]) -> FusionResult:
        """联邦滤波融合 - 基于协方差的信息加权"""
        if awacs_track is None and radar_track is None:
            raise ValueError("At least one track required")
        
        if awacs_track is None:
            P = np.eye(3) * self.radar_cov
            return FusionResult(
                position=radar_track.position.copy(),
                velocity=radar_track.velocity.copy(),
                heading=radar_track.heading,
                confidence=radar_track.confidence,
                covariance=P,
                source_count=1
            )
        
        if radar_track is None:
            P = np.eye(3) * self.awacs_cov
            return FusionResult(
                position=awacs_track.position.copy(),
                velocity=awacs_track.velocity.copy(),
                heading=awacs_track.heading,
                confidence=awacs_track.confidence,
                covariance=P,
                source_count=1
            )
        
        # 联邦融合：信息加权
        # P_fused^-1 = P_awacs^-1 + P_radar^-1
        # x_fused = P_fused * (P_awacs^-1 * x_awacs + P_radar^-1 * x_radar)
        
        P_a = np.eye(3) * self.awacs_cov
        P_r = np.eye(3) * self.radar_cov
        
        P_a_inv = np.linalg.inv(P_a)
        P_r_inv = np.linalg.inv(P_r)
        
        P_fused_inv = P_a_inv + P_r_inv
        P_fused = np.linalg.inv(P_fused_inv)
        
        x_fused = P_fused @ (P_a_inv @ awacs_track.position + P_r_inv @ radar_track.position)
        v_fused = P_fused @ (P_a_inv @ awacs_track.velocity + P_r_inv @ radar_track.velocity)
        
        # 航向融合使用简单加权（因为是角度）
        w_a = self.radar_cov / (self.awacs_cov + self.radar_cov)  # 反向权重
        w_r = self.awacs_cov / (self.awacs_cov + self.radar_cov)
        fused_hdg = self._fuse_heading(awacs_track.heading, radar_track.heading, w_a, w_r)
        
        # 置信度取最高
        fused_conf = max(awacs_track.confidence, radar_track.confidence)
        
        return FusionResult(
            position=x_fused,
            velocity=v_fused,
            heading=fused_hdg,
            confidence=fused_conf,
            covariance=P_fused,
            source_count=2
        )
    
    def _fuse_heading(self, h1: float, h2: float, w1: float, w2: float) -> float:
        """融合航向"""
        r1, r2 = np.radians(h1), np.radians(h2)
        x = w1 * np.cos(r1) + w2 * np.cos(r2)
        y = w1 * np.sin(r1) + w2 * np.sin(r2)
        return np.degrees(np.arctan2(y, x)) % 360


def create_fusion_algorithm(use_federated: bool = False, **kwargs):
    """工厂函数：创建融合算法"""
    if use_federated:
        return FederatedFusion(**kwargs)
    else:
        return SimpleFusion(**kwargs)
