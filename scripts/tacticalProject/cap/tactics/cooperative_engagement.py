"""
协同交战模块 - 目标分配、协同跟踪、接力制导
功能：威胁排序、双机协同跟踪、目标分配、接力制导
"""
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from ..picture import FusedTrack

log = logging.getLogger(__name__)


class EngagementState(Enum):
    """交战状态"""
    TWS = "TWS"          # 边扫边跟
    TRACKING = "TRACKING" # 协同跟踪（发射前10秒）
    GUIDING = "GUIDING"   # 制导中
    RELAY = "RELAY"       # 接力制导


@dataclass
class EngagementAssignment:
    """交战分配"""
    shooter_id: str
    target_id: str
    state: EngagementState = EngagementState.TWS
    score: float = 0.0
    backup_tracker: Optional[str] = None  # 协同跟踪的备份机
    is_primary: bool = True  # 是否为主跟踪机


@dataclass
class TrackingStatus:
    """跟踪状态"""
    target_id: str
    primary_tracker: str
    secondary_tracker: Optional[str] = None
    quality: float = 0.0
    tracking_start_time: float = 0.0
    is_cooperative: bool = False  # 是否双机协同跟踪


@dataclass 
class GuidanceStatus:
    """制导状态"""
    missile_id: str
    target_id: str
    guiding_agent: str
    launch_time: float = 0.0
    relay_candidate: Optional[str] = None


class CooperativeEngagement:
    """协同交战管理器 - 支持双机协同跟踪和接力制导"""
    
    # 配置参数
    COOPERATIVE_TRACK_TIME = 10.0  # 发射前协同跟踪时间(秒)
    TRACK_QUALITY_THRESHOLD = 0.6  # 跟踪质量告警阈值
    ASSIGNMENT_STICKY_TIME = 5.0   # 分配粘性时间(秒)，避免频繁切换
    
    def __init__(self):
        self.assignments: Dict[str, EngagementAssignment] = {}
        self.tracking_status: Dict[str, TrackingStatus] = {}  # target_id -> status
        self.guidance_status: Dict[str, GuidanceStatus] = {}  # missile_id -> status
        self._last_assignment_time: Dict[str, float] = {}
        self._track_quality_alerts: Set[str] = set()
    
    def update(
        self,
        friendlies: List[str],
        picture,
        my_positions: Dict[str, np.ndarray],
        track_qualities: Dict[str, Dict[str, float]] = None,
        current_time: float = 0.0,
        distance_to_launch: float = float('inf'),
        threats: Optional[List["FusedTrack"]] = None,
    ) -> Dict[str, EngagementAssignment]:
        """更新交战分配
        
        Args:
            friendlies: 可用飞机列表
            picture: 态势图
            my_positions: 我方位置 {id: [x,y,z]}
            track_qualities: 跟踪质量 {agent_id: {target_id: quality}}
            current_time: 当前时间
            distance_to_launch: 距发射距离(km)，用于判断是否启动协同跟踪
        """
        threats = threats if threats is not None else picture.get_all_threats()
        if not threats:
            self.assignments.clear()
            self.tracking_status.clear()
            self._last_assignment_time.clear()
            self._track_quality_alerts.clear()
            return {}

        valid_target_ids = {
            str(getattr(threat, "track_id", "") or "")
            for threat in threats
            if getattr(threat, "track_id", None)
        }
        valid_friendlies = {str(fid) for fid in friendlies}
        if valid_target_ids:
            self.assignments = {
                aid: asgn
                for aid, asgn in self.assignments.items()
                if str(aid) in valid_friendlies and str(getattr(asgn, "target_id", "") or "") in valid_target_ids
            }
            self.tracking_status = {
                tid: status
                for tid, status in self.tracking_status.items()
                if str(tid) in valid_target_ids
            }
            self._last_assignment_time = {
                tid: ts
                for tid, ts in self._last_assignment_time.items()
                if str(tid) in valid_target_ids
            }
            self._track_quality_alerts = {
                key
                for key in self._track_quality_alerts
                if key.split(":", 1)[-1] in valid_target_ids
            }
        
        # 🔥 Debug: 打印输入信息 (已注释 - 目标分配成功后关闭)
        # if current_time % 60 < 0.2:
        #     print(f"\n[协同交战] 输入: friendlies={len(friendlies)} threats={len(threats)}")
        #     for t in threats:
        #         print(f"  威胁: {t.track_id} pos=({t.x:.1f}, {t.y:.1f})")
        
        # 计算编队中心
        center = np.mean(list(my_positions.values()), axis=0) if my_positions else np.zeros(3)
        
        # 威胁排序（距离+航向威胁度）
        def threat_score(t):
            dist = np.linalg.norm(t.position[:2] - center[:2])
            # 航向朝向我方的威胁更高
            heading_factor = 1.0 if abs(t.heading - 180) < 45 else 0.7
            return dist / heading_factor
        
        sorted_threats = sorted(threats, key=threat_score)
        
        # 判断是否需要协同跟踪（距发射距离<20km时启动）
        need_cooperative = distance_to_launch < 20.0
        
        # ===== 修复：优先确保所有目标都有分配，再考虑协同跟踪 =====
        # 第一轮：每个目标分配一个最优射手（空间对应优先）
        new_assignments = {}
        available = set(friendlies)
        
        for threat in sorted_threats:
            if not available:
                break
            
            tid = threat.track_id
            
            # 检查是否保持现有分配（粘性）
            existing = self._get_existing_assignment_for_target(tid)
            if existing and existing.shooter_id in available:
                if current_time - self._last_assignment_time.get(tid, 0) < self.ASSIGNMENT_STICKY_TIME:
                    new_assignments[existing.shooter_id] = existing
                    available.remove(existing.shooter_id)
                    continue
            
            # 选择最优射手（使用空间对应算法）
            best = self._select_best_shooter(available, threat.position, my_positions, track_qualities, tid)
            if not best:
                continue
            
            # 创建分配（暂不分配备份机）
            state = EngagementState.TRACKING if need_cooperative else EngagementState.TWS
            asgn = EngagementAssignment(
                shooter_id=best,
                target_id=tid,
                state=state,
                score=np.linalg.norm(my_positions[best][:2] - threat.position[:2]),
                is_primary=True
            )
            
            new_assignments[best] = asgn
            available.remove(best)
            self._last_assignment_time[tid] = current_time
        
        # 第二轮：如果还有剩余飞机且需要协同跟踪，分配备份跟踪机
        if need_cooperative and available:
            for threat in sorted_threats:
                if not available:
                    break
                tid = threat.track_id
                # 找到该目标的主跟踪机
                primary = None
                for aid, asgn in new_assignments.items():
                    if asgn.target_id == tid and asgn.is_primary:
                        primary = asgn
                        break
                if not primary:
                    continue
                    
                # 分配备份跟踪机
                backup = self._select_best_shooter(available, threat.position, my_positions, track_qualities, tid)
                if backup:
                    primary.backup_tracker = backup
                    backup_asgn = EngagementAssignment(
                        shooter_id=backup,
                        target_id=tid,
                        state=EngagementState.TRACKING,
                        score=np.linalg.norm(my_positions[backup][:2] - threat.position[:2]),
                        is_primary=False
                    )
                    new_assignments[backup] = backup_asgn
                    available.remove(backup)
                    
                    self.tracking_status[tid] = TrackingStatus(
                        target_id=tid,
                        primary_tracker=primary.shooter_id,
                        secondary_tracker=backup,
                        is_cooperative=True,
                        tracking_start_time=current_time
                    )
        
        # 检查跟踪质量告警
        if track_qualities:
            self._check_track_quality(track_qualities, new_assignments)
        
        self.assignments = new_assignments
        
        # 🔥 Debug: 打印最终分配结果 (已注释 - 目标分配成功后关闭)
        # if current_time % 60 < 0.2:
        #     print(f"\n[协同交战] 输出: 分配数={len(new_assignments)}")
        #     for aid, asgn in new_assignments.items():
        #         print(f"  {aid} -> {asgn.target_id} (primary={asgn.is_primary}, score={asgn.score:.1f})")
        
        return new_assignments
    
    def _select_best_shooter(self, available: Set[str], target_pos: np.ndarray,
                            my_positions: Dict[str, np.ndarray],
                            track_qualities: Dict[str, Dict[str, float]] = None,
                            target_id: str = None) -> Optional[str]:
        """选择最优射手（空间对应 + 距离 + 跟踪质量）
        
        空间对应原则：
        - 我方左编队(A0100/A0200, x<100) 优先攻击敌方右侧目标(x>0)
        - 我方右编队(A0300/A0400, x>100) 优先攻击敌方左侧目标(x<0)
        这样确保双编队分别对抗不同目标群，避免集火问题
        """
        best, best_score = None, float('inf')
        
        # 🔥 修复：使用编队ID判断左右，而不是X坐标
        # 左编队: A0100/A0200 (ID中包含01或02)
        # 右编队: A0300/A0400 (ID中包含03或04)
        # 敌方左编队: B0100/B0200
        # 敌方右编队: B0300/B0400
        
        # 根据目标ID判断左右（更可靠）
        if target_id:
            # B0100/B0200 = 左侧，B0300/B0400 = 右侧
            target_on_left = target_id in ['B0100', 'B0200']
        else:
            # 回退到X坐标判断
            center_x = 100.0
            target_on_left = target_pos[0] < center_x
        
        for sid in available:
            pos = my_positions.get(sid)
            if pos is None:
                continue
            
            dist = np.linalg.norm(pos[:2] - target_pos[:2])
            
            # 🔥 修复：根据飞机ID判断左右编队
            # A0100/A0200 = 左编队，A0300/A0400 = 右编队
            shooter_on_left = sid in ['A0100', 'A0200']
            
            # 🔥 修复：同侧对抗原则 - 左编队打左目标，右编队打右目标
            # 我方左编队(A0100/A0200, x<center) vs 敌方左侧(B0100/B0200, x<center)
            # 我方右编队(A0300/A0400, x>center) vs 敌方右侧(B0300/B0400, x>center)
            if shooter_on_left == target_on_left:
                # 同侧对抗 = 理想配对，给予奖励（负值减少总分）
                spatial_penalty = -30.0
            else:
                # 异侧对抗 = 不理想，增加惩罚（正值增加总分）
                spatial_penalty = 100.0  # 大幅惩罚确保优先同侧配对
            
            # 跟踪质量加成
            quality_bonus = 0
            if track_qualities and sid in track_qualities and target_id:
                q = track_qualities[sid].get(target_id, 0)
                quality_bonus = (1 - q) * 20  # 质量越高，惩罚越小
            
            score = dist + spatial_penalty + quality_bonus
            
            # 🔥 Debug: 打印评分细节（已注释 - 目标分配成功后关闭）
            # if not hasattr(self, '_score_debug_counter'):
            #     self._score_debug_counter = 0
            # self._score_debug_counter += 1
            # 
            # # 每200次打印一次，显示详细的空间对应信息
            # if self._score_debug_counter % 200 == 0:
            #     side_info = f"射手{'左' if shooter_on_left else '右'}侧 vs 目标{'左' if target_on_left else '右'}侧"
            #     match_info = "✓匹配" if shooter_on_left == target_on_left else "✗不匹配"
            #     print(f"[目标分配] {target_id} -> {sid}: {side_info} {match_info} | 距离={dist:.1f}km 空间={spatial_penalty:+.1f} 质量={quality_bonus:.1f} 总分={score:.1f}")
            
            if score < best_score:
                best_score = score
                best = sid
        
        return best
    
    def _get_existing_assignment_for_target(self, target_id: str) -> Optional[EngagementAssignment]:
        """获取目标的现有分配"""
        for asgn in self.assignments.values():
            if asgn.target_id == target_id and asgn.is_primary:
                return asgn
        return None
    
    def _check_track_quality(self, track_qualities: Dict[str, Dict[str, float]],
                            assignments: Dict[str, EngagementAssignment]):
        """检查跟踪质量，低于阈值时告警（每个组合只告警一次）"""
        for aid, asgn in assignments.items():
            if aid in track_qualities:
                q = track_qualities[aid].get(asgn.target_id, 0)
                if q < self.TRACK_QUALITY_THRESHOLD:
                    key = f"{aid}:{asgn.target_id}"
                    if key not in self._track_quality_alerts:
                        self._track_quality_alerts.add(key)
                        log.warning(f"⚠️ [跟踪质量] {aid}对{asgn.target_id}跟踪质量低: {q:.2f}")
    
    # === 接力制导 ===
    
    def start_guidance(self, missile_id: str, target_id: str, guiding_agent: str, 
                      launch_time: float, friendlies: List[str], 
                      my_positions: Dict[str, np.ndarray]) -> GuidanceStatus:
        """开始制导，并预选接力候选机"""
        # 选择接力候选（除制导机外最近的友机）
        candidate = None
        min_dist = float('inf')
        
        target_asgn = self._get_existing_assignment_for_target(target_id)
        target_pos = my_positions.get(guiding_agent, np.zeros(3))
        
        for fid in friendlies:
            if fid == guiding_agent:
                continue
            pos = my_positions.get(fid)
            if pos is not None:
                d = np.linalg.norm(pos[:2] - target_pos[:2])
                if d < min_dist:
                    min_dist = d
                    candidate = fid
        
        status = GuidanceStatus(
            missile_id=missile_id,
            target_id=target_id,
            guiding_agent=guiding_agent,
            launch_time=launch_time,
            relay_candidate=candidate
        )
        self.guidance_status[missile_id] = status
        return status
    
    def request_relay(self, missile_id: str, reason: str = "evasion") -> Optional[str]:
        """请求接力制导，返回新的制导机ID"""
        status = self.guidance_status.get(missile_id)
        if not status or not status.relay_candidate:
            log.warning(f"⚠️ [接力制导] {missile_id}无可用接力机")
            return None
        
        old_agent = status.guiding_agent
        new_agent = status.relay_candidate
        status.guiding_agent = new_agent
        status.relay_candidate = None  # 清除候选
        
        # 更新分配状态
        if old_agent in self.assignments:
            self.assignments[old_agent].state = EngagementState.TWS
        if new_agent in self.assignments:
            self.assignments[new_agent].state = EngagementState.RELAY
        
        log.info(f"🔄 [接力制导] {missile_id}: {old_agent} → {new_agent} (原因: {reason})")
        return new_agent
    
    def get_assignment(self, agent_id: str) -> Optional[EngagementAssignment]:
        return self.assignments.get(agent_id)
    
    def get_cooperative_tracking_targets(self) -> List[str]:
        """获取正在协同跟踪的目标列表"""
        return [tid for tid, s in self.tracking_status.items() if s.is_cooperative]
    
    def is_track_quality_alert(self, agent_id: str, target_id: str) -> bool:
        """检查是否有跟踪质量告警"""
        return f"{agent_id}:{target_id}" in self._track_quality_alerts
