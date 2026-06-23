"""
战术选择器 - 基于风险区的战术选择
根据敌机所在风险区和态势选择最优战术
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import logging

from .picture import Picture, FusedTrack, RiskZone
from .control_ranges import ControlRanges, DEFAULT_RANGES

log = logging.getLogger(__name__)


class TacticType(Enum):
    """战术类型"""
    # 低风险区战术 (200-300km)
    T_DS = "drag_shoot"        # 拖刀战术
    T_PA = "pincer_attack"     # 钳形攻击
    T_HL = "high_low"          # 高低攻击
    T_SBS = "side_by_side"     # 并列攻击
    
    # 中风险区战术 (100-200km)
    T_FB = "front_back"        # 前后攻击
    
    # 高风险区战术 (0-100km)
    T_TE = "tactical_evasion"  # 战术规避
    T_TT = "tactical_turn"     # 战术转弯


@dataclass
class TacticAssignment:
    """战术分配"""
    tactic: TacticType
    target_id: str
    shooter_id: str
    support_id: Optional[str] = None
    priority: int = 0
    stage_reason: str = ""
    tactic_reason: str = ""
    maneuver_reason: str = ""
    parameter_reason: str = ""
    decision_snapshot: Optional[Dict[str, object]] = None


class TacticSelector:
    """战术选择器 - 基于风险区选择最优战术"""
    
    # 风险区对应的可用战术
    ZONE_TACTICS = {
        RiskZone.LOW: [TacticType.T_DS, TacticType.T_PA, TacticType.T_HL, TacticType.T_SBS],
        RiskZone.MEDIUM: [TacticType.T_FB, TacticType.T_TE, TacticType.T_TT],
        RiskZone.HIGH: [TacticType.T_TE, TacticType.T_TT],
    }
    
    def __init__(self, ranges: Optional[ControlRanges] = None):
        self.ranges = ranges or DEFAULT_RANGES
        self._current_tactic: Optional[TacticType] = None
        self._current_zone: Optional[RiskZone] = None
        self._tactic_start_time: float = 0.0
        self._min_tactic_duration: float = 60.0  # 最小战术持续时间60秒
        self._last_log_time: float = 0.0
        
    def select_tactic(
        self,
        picture: Picture,
        friendly_positions: Dict[str, np.ndarray],
        current_time: float,
        threats: Optional[List[FusedTrack]] = None,
    ) -> Optional[TacticAssignment]:
        # 正常战术选择逻辑
        threats = threats if threats is not None else picture.get_all_threats()
        if not threats:
            return None
        
        # 获取最近威胁
        center = self._get_formation_center(friendly_positions)
        nearest = self._get_nearest_threat(threats, center)
        if not nearest:
            return None
        
        # 确定风险区
        zone = picture.get_risk_zone(nearest.y)
        if zone == RiskZone.OUTSIDE:
            return None
        
        # 选择战术
        tactic = self._select_tactic_for_zone(zone, nearest, friendly_positions, current_time)
        
        # 选择最优射手
        shooter, support = self._select_best_shooter(nearest, friendly_positions)
        
        return TacticAssignment(
            tactic=tactic,
            target_id=nearest.track_id,
            shooter_id=shooter,
            support_id=support,
            priority=self._get_priority(zone)
        )
    
    def _select_tactic_for_zone(self, zone: RiskZone, threat: FusedTrack,
                                 friendly_positions: Dict[str, np.ndarray],
                                 current_time: float) -> TacticType:
        """根据风险区选择战术"""
        available = self.ZONE_TACTICS.get(zone, [TacticType.T_TE])
        
        # 检查是否需要保持当前战术（优先保持稳定）
        if self._should_keep_tactic(current_time, zone):
            return self._current_tactic
        
        # 根据态势选择最优战术
        if zone == RiskZone.LOW:
            tactic = self._select_low_zone_tactic(threat, friendly_positions)
        elif zone == RiskZone.MEDIUM:
            tactic = self._select_medium_zone_tactic(threat, friendly_positions)
        else:  # HIGH
            tactic = self._select_high_zone_tactic(threat, friendly_positions)
        
        # 更新当前战术（只在真正切换时）
        if tactic != self._current_tactic:
            log.info(f"🎯 [战术] {self._current_tactic.value if self._current_tactic else 'None'} → {tactic.value} (区:{zone.value})")
            self._current_tactic = tactic
            self._current_zone = zone
            self._tactic_start_time = current_time
        
        return tactic
    
    def _select_low_zone_tactic(self, threat: FusedTrack,
                                 friendly_positions: Dict[str, np.ndarray]) -> TacticType:
        """低风险区战术选择"""
        # 根据敌机航向判断
        # 敌机朝向我方(航向接近180°) -> 拖刀战术
        # 敌机横向移动 -> 钳形攻击
        # 敌机高度差大 -> 高低攻击
        # 默认 -> 并列攻击
        
        heading_diff = abs(threat.heading - 180.0)
        if heading_diff < 30:  # 敌机正对我方
            return TacticType.T_DS
        elif heading_diff > 60:  # 敌机横向移动
            return TacticType.T_PA
        else:
            # 检查高度差
            if friendly_positions:
                avg_alt = np.mean([p[2] for p in friendly_positions.values()])
                alt_diff = abs(threat.altitude - avg_alt)
                if alt_diff > 2.0:  # 高度差>2km
                    return TacticType.T_HL
            return TacticType.T_SBS
    
    def _select_medium_zone_tactic(self, threat: FusedTrack,
                                    friendly_positions: Dict[str, np.ndarray]) -> TacticType:
        """中风险区战术选择"""
        heading_diff = abs(threat.heading - 180.0)
        # 中风险区默认前后攻击；仅在明确高机动/高闭合态时规避
        if heading_diff > 100:
            return TacticType.T_TT
        nearest_dist = float('inf')
        for pos in friendly_positions.values():
            nearest_dist = min(nearest_dist, float(np.linalg.norm(pos[:2] - np.array([threat.x, threat.y]))))
        if threat.speed > 380 and heading_diff < 30 and nearest_dist < 90:
            return TacticType.T_TE
        return TacticType.T_FB
    
    def _select_high_zone_tactic(self, threat: FusedTrack,
                                  friendly_positions: Dict[str, np.ndarray]) -> TacticType:
        """高风险区战术选择"""
        # 高风险区优先规避
        # 如果敌机速度较慢或正在转弯，可以尝试战术转弯
        if threat.speed < 250:  # 速度<250m/s
            return TacticType.T_TT
        return TacticType.T_TE
    
    def _select_best_shooter(self, threat: FusedTrack,
                              friendly_positions: Dict[str, np.ndarray]) -> Tuple[str, Optional[str]]:
        """
        选择最优射手和支援机
        
        选择标准:
        1. 距离目标最近
        2. 处于有利攻击角度
        """
        if not friendly_positions:
            return 'A0100', None
        
        scores = {}
        threat_pos = np.array([threat.x, threat.y])
        
        for aid, pos in friendly_positions.items():
            # 距离分数 (越近越好)
            dist = np.linalg.norm(pos[:2] - threat_pos)
            dist_score = max(0, 300 - dist) / 300  # 归一化到0-1
            
            # 角度分数 (正对目标更好)
            # 简化：假设我方朝北(0°)，敌机朝南(180°)
            angle_score = 0.5  # 默认中等
            
            scores[aid] = dist_score * 0.7 + angle_score * 0.3
        
        # 排序选择
        sorted_agents = sorted(scores.keys(), key=lambda a: scores[a], reverse=True)
        shooter = sorted_agents[0]
        support = sorted_agents[1] if len(sorted_agents) > 1 else None
        
        return shooter, support
    
    def _get_formation_center(self, friendly_positions: Dict[str, np.ndarray]) -> np.ndarray:
        """获取编队中心"""
        if not friendly_positions:
            return np.array([100.0, 50.0, 8.0])
        positions = list(friendly_positions.values())
        return np.mean(positions, axis=0)
    
    def _get_nearest_threat(self, threats: List[FusedTrack], 
                            center: np.ndarray) -> Optional[FusedTrack]:
        """获取最近威胁"""
        if not threats:
            return None
        return min(threats, key=lambda t: np.linalg.norm(
            np.array([t.x, t.y]) - center[:2]))
    
    def _should_keep_tactic(self, current_time: float, new_zone: RiskZone) -> bool:
        """检查是否应保持当前战术"""
        if self._current_tactic is None:
            return False
        
        # HIGH区威胁必须立即响应
        if new_zone == RiskZone.HIGH and self._current_zone != RiskZone.HIGH:
            return False
        
        # 在最小持续时间内保持战术
        elapsed = current_time - self._tactic_start_time
        if elapsed < self._min_tactic_duration:
            return True
        
        # 超过持续时间但区域相同，继续保持（避免频繁切换）
        if new_zone == self._current_zone:
            return True
        
        return False
    
    def _get_priority(self, zone: RiskZone) -> int:
        """获取优先级"""
        priorities = {
            RiskZone.HIGH: 3,
            RiskZone.MEDIUM: 2,
            RiskZone.LOW: 1,
            RiskZone.OUTSIDE: 0
        }
        return priorities.get(zone, 0)
    
    def get_current_tactic(self) -> Optional[TacticType]:
        """获取当前战术"""
        return self._current_tactic
