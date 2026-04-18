"""预警机数据源 - MockAwacsDataSource

模拟预警机探测，含误差和目标丢失。

说明：为保证验证对比(算法/无算法)可复现，支持通过seed固定随机序列。

⚠️ 重要：仅靠“固定seed”并不总能保证 proposed/baseline 的 AWACS 丢失时间点完全一致。
原因是：若 AWACS 的随机数消耗与“哪些目标被处理/是否continue”有关（例如距离门控、
扫描周期continue、目标数量变化），则即使 seed 相同也可能出现随机序列偏移。

为此提供可选的确定性模式（用于验证脚本）：
- CAP_AWACS_DETERMINISTIC=1：对每个目标/每个扫描slot 使用稳定哈希生成伪随机数，避免序列漂移。
- CAP_AWACS_FIXED_CENTER_KM="x,y,z"：使用固定参考中心计算距离门控，避免我方机动改变门控。

默认不启用，保持原有随机行为不变。
"""
import os
import numpy as np
import hashlib
from typing import Dict, List, Optional
from dataclasses import dataclass
from .picture import Track, TrackSource
import logging

log = logging.getLogger(__name__)


@dataclass
class AwacsConfig:
    """预警机配置"""
    detection_range: float = 400.0    # 探测距离(km)
    position_error: float = 2.5       # 位置误差(km) 2-3km取中值
    altitude_error: float = 0.5       # 高度误差(km)
    heading_error: float = 10.0       # 航向误差(度)
    max_lost_duration: float = 20.0   # 最大丢失时长(秒)
    lost_probability: float = 0.02    # 每次更新丢失概率


class MockAwacsDataSource:
    """模拟预警机数据源"""
    
    def __init__(self, config: Optional[AwacsConfig] = None, seed: Optional[int] = None):
        self.config = config or AwacsConfig()
        self._tracks: Dict[str, Track] = {}
        self._lost_targets: Dict[str, float] = {}  # track_id -> 丢失开始时间
        if seed is None:
            raw = os.environ.get('CAP_AWACS_SEED', '').strip()
            if raw:
                try:
                    seed = int(raw)
                except Exception:
                    seed = None
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        det = os.environ.get('CAP_AWACS_DETERMINISTIC', '').strip().lower()
        self._deterministic = det in ('1', 'true', 'yes', 'y', 'on')

        # 可选：固定参考中心（用于距离门控，避免我方机动影响AWACS随机消耗）
        self._fixed_center = None
        raw_center = os.environ.get('CAP_AWACS_FIXED_CENTER_KM', '').strip()
        if raw_center:
            try:
                parts = [float(x) for x in raw_center.replace(';', ',').split(',') if x.strip()]
                if len(parts) >= 2:
                    x, y = parts[0], parts[1]
                    z = parts[2] if len(parts) >= 3 else 9.0
                    self._fixed_center = np.array([x, y, z], dtype=float)
            except Exception:
                self._fixed_center = None

    def _u01(self, key: str) -> float:
        """Stable pseudo-random U[0,1) from (seed,key).

        Used only when deterministic mode is enabled.
        """
        seed = 0 if self._seed is None else int(self._seed)
        data = f"{seed}|{key}".encode('utf-8', errors='ignore')
        digest = hashlib.blake2b(data, digest_size=8).digest()
        val = int.from_bytes(digest, 'little', signed=False)
        return (val & ((1 << 64) - 1)) / float(1 << 64)
    
    def set_ground_truth(self, targets: List[Dict]):
        """设置真实目标数据 (来自仿真环境)
        
        targets格式: [{
            'id': str, 'position': [x,y,z] km, 
            'velocity': [vx,vy,vz] m/s, 'heading': float
        }]
        """
        self._ground_truth = {t['id']: t for t in targets}
    
    def update(self, current_time: float, friendly_positions: List[np.ndarray] = None) -> Dict[str, Track]:
        """更新探测数据，返回当前可见航迹
        
        Args:
            current_time: 当前仿真时间(秒)
            friendly_positions: 我方飞机位置列表(km)，用于计算探测距离
                              预警机探测距离是基于我方战机位置计算的敌我相对距离
        """
        cfg = self.config
        tracks = {}
        
        # 计算参考中心（用于距离判断）
        # - 默认：使用我方编队中心
        # - 验证/公平性：可通过 CAP_AWACS_FIXED_CENTER_KM 强制固定
        if self._fixed_center is not None:
            friendly_center = self._fixed_center
        elif friendly_positions and len(friendly_positions) > 0:
            friendly_center = np.mean(friendly_positions, axis=0)
        else:
            # 默认使用我方编队中心位置 (X=100km, Y=0km)
            friendly_center = np.array([100.0, 0.0, 9.0])
        
        for tid, gt in getattr(self, '_ground_truth', {}).items():
            pos = np.array(gt['position'])
            
            # 距离检查：基于我方编队中心到敌机的距离
            # 用户需求："预警机探测距离是400km，这个探测距离是基于我方战机的位置计算的敌我相对距离"
            dist = np.linalg.norm(pos - friendly_center)
            if dist > cfg.detection_range:
                continue
            
            # === E-3预警机功能级建模 ===
            # 1. 10秒扫描周期：只有当目标方向被扫描到时才更新
            # 简化：每10秒对所有方向更新一次（模拟360°扫描周期）
            last_scan_key = f'_last_scan_{tid}'
            last_scan_time = getattr(self, last_scan_key, -100.0)
            if current_time - last_scan_time < 10.0:
                # 使用上次扫描结果（如果有的话）
                if tid in self._tracks:
                    tracks[tid] = self._tracks[tid]
                continue
            setattr(self, last_scan_key, current_time)

            scan_slot = int(current_time // 10.0)  # 与10s扫描周期对齐的slot
            
            # 2. 基于目标高度的探测概率（预警机对低空目标探测困难）
            target_alt_km = pos[2] if len(pos) > 2 else 8.0  # 默认中空
            
            if self._handle_lost_with_altitude(tid, current_time, target_alt_km, scan_slot=scan_slot):
                continue
            
            # 添加误差（需保证误差在2-3km之间，尽量接近3km）
            if self._deterministic:
                u1 = self._u01(f"posdist:{tid}:{scan_slot}")
                u2 = self._u01(f"postheta:{tid}:{scan_slot}")
                u3 = self._u01(f"alt:{tid}:{scan_slot}")
                u4 = self._u01(f"hdg:{tid}:{scan_slot}")
                offset_dist = 2.0 + u1 * 1.0
                theta = u2 * 2 * np.pi
                alt_noise = (u3 * 2.0 - 1.0) * cfg.altitude_error
                hdg_noise = (u4 * 2.0 - 1.0) * cfg.heading_error
            else:
                offset_dist = 2.0 + self._rng.random() * 1.0  # [2.0, 3.0] km
                theta = self._rng.random() * 2 * np.pi
                alt_noise = self._rng.uniform(-cfg.altitude_error, cfg.altitude_error)
                hdg_noise = self._rng.uniform(-cfg.heading_error, cfg.heading_error)
            offset_x = offset_dist * np.cos(theta)
            offset_y = offset_dist * np.sin(theta)
            
            noisy_pos = pos.copy()
            noisy_pos[0] += offset_x
            noisy_pos[1] += offset_y
            noisy_pos[2] = max(0, pos[2] + alt_noise)
            noisy_heading = gt['heading'] + hdg_noise
            
            tracks[tid] = Track(
                track_id=tid,
                source=TrackSource.AWACS,
                position=noisy_pos,
                velocity=np.array(gt['velocity']),
                heading=noisy_heading % 360,
                timestamp=current_time,
                confidence=0.7,  # 预警机数据置信度较低
                is_hostile=True
            )
            
            # 首次探测日志
            if not hasattr(self, '_ever_detected'):
                self._ever_detected = set()
            if tid not in self._ever_detected:
                self._ever_detected.add(tid)
                log.info(f"[预警机] 首次发现目标{tid} (距离{dist:.1f}km)")
        
        self._tracks = tracks
        return tracks
    
    def _handle_lost(self, track_id: str, current_time: float, scan_slot: Optional[int] = None) -> bool:
        """处理目标丢失逻辑，返回True表示当前丢失"""
        cfg = self.config
        
        # 已丢失状态检查
        if track_id in self._lost_targets:
            # 兼容旧版本只存时间的情况
            data = self._lost_targets[track_id]
            if isinstance(data, tuple):
                lost_time, duration = data
            else:
                lost_time, duration = data, cfg.max_lost_duration
                
            if current_time - lost_time < duration:
                return True  # 仍在丢失中
            else:
                del self._lost_targets[track_id]  # 恢复
                return False
        
        # 随机触发丢失
        if self._deterministic:
            slot = 0 if scan_slot is None else int(scan_slot)
            u_loss = self._u01(f"lost:{track_id}:{slot}")
            u_dur = self._u01(f"dur:{track_id}:{slot}")
            trigger = u_loss < cfg.lost_probability
            duration = 5.0 + u_dur * max(0.0, (cfg.max_lost_duration - 5.0))
        else:
            trigger = self._rng.random() < cfg.lost_probability
            duration = self._rng.uniform(5.0, cfg.max_lost_duration)

        if trigger:
            self._lost_targets[track_id] = (current_time, duration)
            log.info(f"[预警机] 目标{track_id}随机丢失 (时长{duration:.1f}s)")
            return True
        
        return False
    
    def _handle_lost_with_altitude(self, track_id: str, current_time: float,
                                   target_alt_km: float, scan_slot: Optional[int] = None) -> bool:
        """基于目标高度的丢失处理 (预警机功能级建模)
        
        E-3预警机对低空目标探测困难:
        - 高空 (>7km): 正常探测 (2%丢失概率)
        - 中空 (5-7km): 轻微困难 (5%丢失概率)
        - 低空 (3.5-5km): 显著困难 (30%丢失概率)
        - 超低空 (1.5-3.5km): 严重困难 (70%丢失概率) ⚠️ 关键区间
        - 极低空 (<1.5km): 几乎无法探测 (95%丢失概率)
        
        Args:
            track_id: 目标ID
            current_time: 当前时间
            target_alt_km: 目标高度(km)
        """
        cfg = self.config
        
        # 已丢失状态检查(与原逻辑相同)
        if track_id in self._lost_targets:
            data = self._lost_targets[track_id]
            if isinstance(data, tuple):
                lost_time, duration = data
            else:
                lost_time, duration = data, cfg.max_lost_duration
                
            if current_time - lost_time < duration:
                return True
            else:
                del self._lost_targets[track_id]
                return False
        
        # 基于高度的丢失概率 (调整为更严格的低空探测)
        if target_alt_km > 7.0:
            loss_prob = cfg.lost_probability  # 2% (正常)
        elif target_alt_km > 5.0:
            loss_prob = 0.05  # 5% (中空)
        elif target_alt_km > 3.5:
            loss_prob = 0.30  # 30% (低空)
        elif target_alt_km > 1.5:
            loss_prob = 0.70  # 70% (超低空 - 大幅提高丢失率)
        else:
            loss_prob = 0.95  # 95% (极低空)
        
        # 触发丢失
        if self._deterministic:
            slot = 0 if scan_slot is None else int(scan_slot)
            u_loss = self._u01(f"lost_alt:{track_id}:{slot}")
            u_dur = self._u01(f"dur_alt:{track_id}:{slot}")
            trigger = u_loss < loss_prob
            duration = 5.0 + u_dur * 10.0  # 5~15
        else:
            trigger = self._rng.random() < loss_prob
            duration = self._rng.uniform(5.0, 15.0)

        if trigger:
            # 低空目标丢失时间更长
            if target_alt_km < 3.5:
                duration = min(duration * 1.8, 35.0)  # 增加丢失时长
                
            self._lost_targets[track_id] = (current_time, duration)
            if (self._u01(f"lostlog:{track_id}:{slot}") if self._deterministic else self._rng.random()) < 0.15: # 提高日志频率以便观察
                log.info(f"[预警机] 目标{track_id}丢失 (高度{target_alt_km:.1f}km, 概率{loss_prob:.0%}, 时长{duration:.1f}s)")
            return True
        
        return False
    
    def get_tracks(self) -> Dict[str, Track]:
        """获取当前航迹"""
        return self._tracks.copy()
    
    def is_target_lost(self, track_id: str) -> bool:
        """检查目标是否丢失"""
        return track_id in self._lost_targets
