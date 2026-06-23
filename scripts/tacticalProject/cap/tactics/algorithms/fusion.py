"""
协同探测V4 - 航迹融合
简化版: 加权平均
专业版: Track-to-Track Fusion + Covariance Intersection
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from ..detection_types import TargetInfo
from ..multi_target_manager import MultiTargetManager


class WeightedAverageFusion:
    """加权平均融合（简化版）"""
    
    def fuse(self, target_manager: MultiTargetManager,
             radar_detections: Dict[str, List[Tuple[str, float, float]]],
             current_time: float) -> Dict[str, TargetInfo]:
        """按距离/SNR加权平均"""
        fused = {}
        
        # 收集每个目标的所有测量
        target_measurements: Dict[str, List[Tuple[float, float, float]]] = {}
        
        for fid, detections in radar_detections.items():
            for tid, x, y in detections:
                if tid not in target_measurements:
                    target_measurements[tid] = []
                target_measurements[tid].append((x, y, 1.0))  # 权重=1
        
        for tid, measurements in target_measurements.items():
            if not measurements:
                continue
            
            # 加权平均
            total_weight = sum(m[2] for m in measurements)
            if total_weight < 1e-10:
                continue
            
            fused_x = sum(m[0] * m[2] for m in measurements) / total_weight
            fused_y = sum(m[1] * m[2] for m in measurements) / total_weight
            
            # 获取原始目标信息
            orig = target_manager.get_target(tid)
            
            fused[tid] = TargetInfo(
                target_id=tid,
                x=fused_x, y=fused_y,
                vx=orig.vx if orig else 0.0,
                vy=orig.vy if orig else -0.3,
                source='fused',
                timestamp=current_time,
                is_tracked=True,
                track_quality=min(1.0, len(measurements) * 0.3)
            )
        
        return fused


class T2TFusion:
    """Track-to-Track Fusion + Covariance Intersection（专业版）"""
    
    def __init__(self):
        self._track_history: Dict[str, List[Tuple[float, float, float]]] = {}
    
    def fuse(self, target_manager: MultiTargetManager,
             radar_detections: Dict[str, List[Tuple[str, float, float]]],
             current_time: float) -> Dict[str, TargetInfo]:
        """T2TF融合"""
        fused = {}
        
        # 收集每个目标的航迹
        target_tracks: Dict[str, List[Dict]] = {}
        
        for fid, detections in radar_detections.items():
            for tid, x, y in detections:
                if tid not in target_tracks:
                    target_tracks[tid] = []
                
                # 假设每个雷达有自己的协方差
                cov = np.eye(2) * 0.1
                target_tracks[tid].append({
                    'state': np.array([x, y]),
                    'cov': cov,
                    'fighter_id': fid
                })
        
        for tid, tracks in target_tracks.items():
            if len(tracks) >= 2:
                # Covariance Intersection
                fused_state, fused_cov = self._covariance_intersection(tracks)
                quality = min(1.0, len(tracks) * 0.4)
            elif len(tracks) == 1:
                fused_state = tracks[0]['state']
                fused_cov = tracks[0]['cov']
                quality = 0.5
            else:
                continue
            
            # 获取原始目标信息
            orig = target_manager.get_target(tid)
            
            fused[tid] = TargetInfo(
                target_id=tid,
                x=fused_state[0], y=fused_state[1],
                vx=orig.vx if orig else 0.0,
                vy=orig.vy if orig else -0.3,
                covariance=self._expand_cov(fused_cov),
                source='fused',
                timestamp=current_time,
                is_tracked=True,
                track_quality=quality,
                tracker_ids=[t['fighter_id'] for t in tracks]
            )
        
        return fused
    
    def _covariance_intersection(self, tracks: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """协方差交叉融合"""
        n = len(tracks)
        
        # 优化omega（简化：使用均匀权重）
        omega = 1.0 / n
        
        # 融合协方差
        fused_cov_inv = np.zeros((2, 2))
        fused_state_weighted = np.zeros(2)
        
        for track in tracks:
            cov = track['cov']
            # 确保正定
            cov = (cov + cov.T) / 2
            min_eig = np.min(np.linalg.eigvalsh(cov))
            if min_eig < 0.001:
                cov += np.eye(2) * (0.001 - min_eig)
            
            cov_inv = np.linalg.inv(cov)
            fused_cov_inv += omega * cov_inv
            fused_state_weighted += omega * cov_inv @ track['state']
        
        # 确保可逆
        min_eig = np.min(np.linalg.eigvalsh(fused_cov_inv))
        if min_eig < 0.001:
            fused_cov_inv += np.eye(2) * (0.001 - min_eig)
        
        fused_cov = np.linalg.inv(fused_cov_inv)
        fused_state = fused_cov @ fused_state_weighted
        
        return fused_state, fused_cov
    
    def _expand_cov(self, cov_2x2: np.ndarray) -> np.ndarray:
        """扩展2x2协方差到4x4"""
        cov_4x4 = np.eye(4) * 0.1
        cov_4x4[:2, :2] = cov_2x2
        return cov_4x4
