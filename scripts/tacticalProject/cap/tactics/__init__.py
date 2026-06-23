# 战术模块
from .cooperative_detection import CooperativeDetection, DetectionMode, ScanAssignment
from .radar_scan import RadarScanParams, RadarMode
from .cooperative_engagement import (
    CooperativeEngagement, 
    EngagementAssignment, 
    EngagementState,
    TrackingStatus,
    GuidanceStatus
)
from .formation_guidance import FormationGuidance, GuidanceTarget
from .velocity_coordination import VelocityCoordination, SpeedCommand

# 探测子系统
from .detection_config import DetectionRangeConfig
from .detection_types import (
    DetectionPhase, DetectionMode as DetectionModeV4,
    TargetInfo, FighterInfo, AwacsInfoV4,
    ScanAssignment as ScanAssignmentV4,
    SearchRegion, DetectionResultV4, TargetState
)
from .multi_target_manager import MultiTargetManager

__all__ = [
    # 核心探测模块
    'CooperativeDetection', 'DetectionMode', 'ScanAssignment',
    'RadarScanParams', 'RadarMode', 
    'CooperativeEngagement', 'EngagementAssignment', 'EngagementState',
    'TrackingStatus', 'GuidanceStatus',
    # 编队引导与速度协调
    'FormationGuidance', 'GuidanceTarget',
    'VelocityCoordination', 'SpeedCommand',
    # 探测子系统
    'DetectionRangeConfig', 'DetectionPhase', 'DetectionModeV4',
    'TargetInfo', 'FighterInfo', 'AwacsInfoV4',
    'ScanAssignmentV4', 'SearchRegion', 'DetectionResultV4', 'TargetState',
    'MultiTargetManager'
]
