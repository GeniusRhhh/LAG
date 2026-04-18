"""
协同探测V4 - 数据类型定义
"""
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


class DetectionPhase(Enum):
    """探测阶段 - V5优化版
    
    V5流程：
    PATROL(>400km) → BLIND_INTERCEPT(400~200km) → ACQUIRE(200~180km) 
    → FUSE(180~150km) → EVALUATE(150~120km) → HANDOVER(<120km)
    """
    PATROL = "PATROL"               # >400km 默认巡逻
    BLIND_INTERCEPT = "BLIND_INTERCEPT"  # 400~200km 编队前出（雷达关机）
    ACQUIRE = "ACQUIRE"             # 200km → NLT(180km) 发现目标
    FUSE = "FUSE"                   # NLT → MELD(150km) 雷达融合
    EVALUATE = "EVALUATE"           # MELD → MTR(120km) 跟踪评估
    HANDOVER = "HANDOVER"           # <MTR(120km) 移交协同跟踪


class DetectionMode(Enum):
    """探测模式"""
    SWEEP = "SWEEP"         # 推磨扫描（无预警信息）
    DIRECTED = "DIRECTED"   # 定向扫描（有预警信息）
    SEARCH = "SEARCH"       # 搜索模式（目标丢失）


@dataclass
class TargetState:
    """目标状态（估计器输出）"""
    x: float
    y: float
    vx: float = 0.0
    vy: float = -0.3
    covariance: Optional[np.ndarray] = None
    timestamp: float = 0.0


@dataclass
class TargetInfo:
    """目标信息"""
    target_id: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = -0.3
    altitude: float = 10.0
    heading: float = 180.0
    error_radius: float = 2.5
    timestamp: float = 0.0
    confidence: float = 1.0
    source: str = 'awacs'  # 'awacs' | 'radar' | 'fused'
    
    # 跟踪状态
    is_tracked: bool = False
    tracker_ids: List[str] = field(default_factory=list)
    track_quality: float = 0.0
    last_radar_contact: float = 0.0
    covariance: Optional[np.ndarray] = None
    
    def distance_to(self, pos: Tuple[float, float]) -> float:
        return np.sqrt((self.x - pos[0])**2 + (self.y - pos[1])**2)


@dataclass
class FighterInfo:
    """战机信息"""
    fighter_id: str
    x: float
    y: float
    heading: float
    is_hot: bool
    radar_available: bool = True
    current_target: Optional[str] = None
    track_quality: float = 0.0
    
    def distance_to(self, target: TargetInfo) -> float:
        return np.sqrt((self.x - target.x)**2 + (self.y - target.y)**2)
    
    def bearing_to(self, target: TargetInfo) -> float:
        dx = target.x - self.x
        dy = target.y - self.y
        return np.degrees(np.arctan2(dx, dy)) % 360


@dataclass
class AwacsInfoV4:
    """预警机信息（多目标）"""
    targets: Dict[str, TargetInfo]
    timestamp: float
    is_available: bool = True
    
    def get_nearest(self, ref_pos: Tuple[float, float]) -> Optional[TargetInfo]:
        if not self.targets:
            return None
        return min(self.targets.values(), key=lambda t: t.distance_to(ref_pos))
    
    def get_sorted_by_distance(self, ref_pos: Tuple[float, float]) -> List[TargetInfo]:
        return sorted(self.targets.values(), key=lambda t: t.distance_to(ref_pos))


@dataclass
class ScanAssignment:
    """扫描分配"""
    fighter_id: str
    azimuth_center: float     # 扫描中心方位(度)
    azimuth_range: float      # 扫描范围(±度)
    elevation_bars: int = 2
    scan_time: float = 4.0
    priority: float = 1.0
    target_id: Optional[str] = None


@dataclass
class SearchRegion:
    """搜索区域"""
    center_x: float
    center_y: float
    radius: float
    scan_range: float
    priority: float = 1.0
    bearing: float = 0.0
    particles: Optional[np.ndarray] = None


@dataclass
class DetectionResultV4:
    """探测结果"""
    phase: DetectionPhase
    mode: DetectionMode
    scan_assignments: Dict[str, ScanAssignment]
    target_assignments: Dict[str, List[str]]
    target_states: Dict[str, TargetInfo]
    fused_tracks: Dict[str, TargetInfo] = field(default_factory=dict)
    search_regions: Dict[str, SearchRegion] = field(default_factory=dict)
    probability_map: Optional[np.ndarray] = None
    handover_ready: bool = False
    handover_targets: List[str] = field(default_factory=list)
    phase_changed: bool = False
    mode_changed: bool = False
    track_qualities: Dict[str, float] = field(default_factory=dict)
    ready_for_launch: List[str] = field(default_factory=list)
    timestamp: float = 0.0
