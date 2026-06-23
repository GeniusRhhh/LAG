"""
CAP专用雷达系统 - 支持协同探测、跟踪质量计算、多目标限制
实现需求：推磨扫描、定向扫描、全域扫描、最多6目标高精度跟踪
"""
import logging
import os
import numpy as np
import random
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


_CAP_DEBUG_PRINT = os.environ.get('CAP_DEBUG_PRINT') == '1'


def _cap_print(*args, **kwargs):
    if _CAP_DEBUG_PRINT:
        print(*args, **kwargs)

# 前向视场限制（机载火控雷达前向可扫描的最大方位半角）
# 与任务定义一致：扫描状态仅允许 ±10 / ±30 / ±60，因此最大前向视场取 ±60°。
MAX_FORWARD_AZIMUTH_DEG = 60.0


class RadarMode(Enum):
    """雷达工作模式"""
    SEARCH = "SEARCH"        # 搜索
    TRACK = "TRACK"          # TWS跟踪
    LOCK = "LOCK"            # STT锁定
    DIRECTED = "DIRECTED"    # 定向扫描
    SEARCH_ALL = "SEARCH_ALL" # 全域扫描


@dataclass
class RadarConfig:
    """雷达参数配置"""
    name: str
    max_detection_range: float = 200.0  # km
    max_track_range: float = 160.0      # km
    max_lock_range: float = 120.0       # km
    max_tracks: int = 6                 # 最大高精度跟踪数
    
    # 扫描时间表 {俯仰条数: {方位角范围: 扫描时间}}
    scan_params: Dict[int, Dict[int, float]] = field(default_factory=lambda: {
        1: {10: 2.0, 30: 5.0, 60: 10.0},
        2: {10: 4.0, 30: 10.0, 60: 20.0},
        4: {10: 8.0, 30: 20.0, 60: 40.0}
    })
    
    base_detection_prob: float = 0.85
    track_loss_prob: float = 0.02


RADAR_CAP_STD = RadarConfig(name="CAP_STD")


@dataclass
class RadarTrack:
    """雷达航迹"""
    track_id: str
    distance: float         # km
    bearing: float          # 相对方位(度)
    elevation: float        # 仰角(度)
    velocity: float         # 径向速度(m/s)
    rcs: float = 5.0
    detection_prob: float = 0.0
    track_quality: float = 0.0
    last_update: float = 0.0
    in_notch: bool = False
    update_count: int = 0   # 连续更新次数，用于质量计算


class CAPRadarManager:
    """CAP雷达管理器 - 支持跟踪质量计算和多目标限制"""
    
    TRACK_EXPIRE_TIME = 30.0  # 航迹过期时间(秒)
    NOTCH_VELOCITY = 50.0     # 多普勒盲区速度阈值(m/s)
    
    def __init__(self):
        self._configs: Dict[str, RadarConfig] = {}
        self._radar_modes: Dict[str, RadarMode] = {}
        self._radar_tracks: Dict[str, Dict[str, RadarTrack]] = {}
        self._scan_assignments: Dict[str, Tuple[float, float, int]] = {}
        self._last_scan_time: Dict[str, float] = {}
        self._lock_targets: Dict[str, Optional[str]] = {}
        
    def init_agents(self, agent_ids: List[str]):
        """初始化Agent雷达"""
        for aid in agent_ids:
            self._configs[aid] = RADAR_CAP_STD
            self._radar_modes[aid] = RadarMode.SEARCH
            self._radar_tracks[aid] = {}
            self._scan_assignments[aid] = (0.0, 60.0, 2)
            self._last_scan_time[aid] = 0.0
            self._lock_targets[aid] = None
    
    def set_scan_assignment(self, agent_id: str, center: float, azimuth_range: float, bars: int = 2):
        """设置扫描分配"""
        # 归一化到 (-180, 180]
        try:
            center = float(center)
        except Exception:
            center = 0.0
        center = (center + 180.0) % 360.0 - 180.0

        # 约束扫描半角到 [0, MAX_FORWARD_AZIMUTH_DEG]
        try:
            azimuth_range = float(azimuth_range)
        except Exception:
            azimuth_range = 0.0
        azimuth_range = max(0.0, min(MAX_FORWARD_AZIMUTH_DEG, azimuth_range))

        # 前向视场限制：确保扫描扇区完全落在 [-MAX_FORWARD_AZIMUTH_DEG, +MAX_FORWARD_AZIMUTH_DEG]
        # 即 center ± azimuth_range 必须在前向视场内。
        if azimuth_range >= MAX_FORWARD_AZIMUTH_DEG:
            center = 0.0
            azimuth_range = MAX_FORWARD_AZIMUTH_DEG
        else:
            lo = -MAX_FORWARD_AZIMUTH_DEG + azimuth_range
            hi = MAX_FORWARD_AZIMUTH_DEG - azimuth_range
            center = max(lo, min(hi, center))

        self._scan_assignments[agent_id] = (center, azimuth_range, int(bars))
        if azimuth_range <= 10:
            self._radar_modes[agent_id] = RadarMode.DIRECTED
        elif azimuth_range >= 60:
            self._radar_modes[agent_id] = RadarMode.SEARCH_ALL
        else:
            self._radar_modes[agent_id] = RadarMode.SEARCH
    
    def set_radar_off(self, agent_id: str):
        """关闭雷达（用于冷段飞机）"""
        self._radar_modes[agent_id] = None  # None表示关机
        self._scan_assignments[agent_id] = (0.0, 0.0, 0)  # 清空扫描分配
    
    def update(self, env, current_time: float) -> Dict[str, Dict[str, RadarTrack]]:
        """更新所有雷达 - 连续波束扫描模型
        
        改进：波束在扫描周期内连续扫过不同方位角
        目标在波束扫到其方位时被探测，不同目标有不同探测时刻
        """
        # 🔥 诊断计数器
        if not hasattr(self, '_radar_debug_counter'):
            self._radar_debug_counter = 0
        self._radar_debug_counter += 1
        
        for agent_id in list(self._radar_modes.keys()):
            if agent_id not in env.agents or not env.agents[agent_id].is_alive:
                continue
            
            # ✅ 修复：检查雷达是否关闭（冷段飞机）
            if self._radar_modes.get(agent_id) is None:
                # 雷达关机，清空航迹并跳过扫描
                self._radar_tracks[agent_id] = {}
                continue
            
            config = self._configs.get(agent_id, RADAR_CAP_STD)
            center, az_range, bars = self._scan_assignments.get(agent_id, (0, 60, 2))
            
            # 查表：az_range是半幅（±度），直接用于查表
            # 表2定义：±10°→10, ±30°→30, ±60°→60
            az_key = min([10, 30, 60], key=lambda x: abs(x - az_range))
            scan_period = config.scan_params.get(bars, {}).get(az_key, 4.0)
            
            # 🔥 优化：智能波束指向 - 优先扫描中心区域
            last_scan = self._last_scan_time.get(agent_id, current_time)
            elapsed = current_time - last_scan
            
            if elapsed >= scan_period:
                # 完成一个完整扫描周期，重置
                self._last_scan_time[agent_id] = current_time
                elapsed = 0
            
            # 🔥 中心优先扫描：优先扫描中心区域，确保快速探测
            scan_progress = elapsed / scan_period  # 0~1
            left_edge = center - az_range
            right_edge = center + az_range
            
            # 中心优先扫描：中→左→中→右
            if scan_progress < 0.5:
                # 前半周期：中心 → 左边界
                current_beam_az = center - (center - left_edge) * (scan_progress * 2)
            else:
                # 后半周期：中心 → 右边界
                current_beam_az = center + (right_edge - center) * ((scan_progress - 0.5) * 2)
            
            # 🔥 诊断日志：打印波束扫描详情（每300次=60秒）
            if self._radar_debug_counter % 300 == 1 and agent_id == 'A0100':
                _cap_print(f"\n[雷达波束] {agent_id} | 时间:{current_time:.1f}s")
                _cap_print(f"  扫描分配: center={center:.1f}° range=±{az_range:.1f}° bars={bars}")
                _cap_print(f"  扫描周期: {scan_period:.1f}s (az_key={az_key})")
                _cap_print(f"  波束位置: {current_beam_az:.1f}° (进度:{scan_progress:.1%})")
                _cap_print(f"  扫描范围: {left_edge:.1f}° ~ {right_edge:.1f}° (总宽度:{right_edge-left_edge:.1f}°)")
            
            # 扫描当前波束位置附近的目标
            beam_width = 3.0  # 波束宽度约3度
            self._scan_at_beam_position(env, agent_id, current_time, current_beam_az, beam_width, config)
            
            self._update_tracks(agent_id, current_time, config)
            self._update_mode(agent_id)
        
        return self._radar_tracks
    
    def _scan_at_beam_position(self, env, agent_id: str, current_time: float, 
                                beam_az: float, beam_width: float, config: RadarConfig):
        """在当前波束位置扫描目标
        
        只有目标方位在波束宽度内时才能被探测
        """
        agent = env.agents[agent_id]
        
        # 获取本机航向
        try:
            from envs.JSBSim.core.catalog import Catalog as c
            heading = np.degrees(agent.get_property_value(c.attitude_heading_true_rad)) % 360
        except:
            heading = 0.0
        
        enemy_prefix = "B" if agent_id.startswith("A") else "A"
        
        # 🔥 诊断：收集所有目标信息
        target_info = []
        
        for eid in env.agents:
            if not eid.startswith(enemy_prefix) or not env.agents[eid].is_alive:
                continue
            
            target = env.agents[eid]
            dist_km, abs_bearing, elev = self._calc_geometry(agent, target)
            
            # 距离检查
            if dist_km > config.max_detection_range:
                target_info.append(f"{eid}:超距({dist_km:.0f}km)")
                continue
            
            # 相对机头方位
            rel_bearing = (abs_bearing - heading + 180) % 360 - 180

            # ✅ 前向视场限制：仅允许探测/跟踪前向 ±60° 内目标
            if abs(rel_bearing) > MAX_FORWARD_AZIMUTH_DEG:
                target_info.append(f"{eid}:{rel_bearing:+.1f}°(✗超前向视场)")
                continue
            
            # 检查目标是否在当前波束位置
            beam_offset = rel_bearing - beam_az
            if beam_offset > 180:
                beam_offset -= 360
            elif beam_offset < -180:
                beam_offset += 360
            
            # 目标是否在波束宽度内
            in_beam = abs(beam_offset) <= beam_width
            
            # 🔥 诊断：记录目标状态
            status = "✓探测" if in_beam else f"✗偏离{beam_offset:+.1f}°"
            target_info.append(f"{eid}:{rel_bearing:+.1f}°({status})")
            
            if not in_beam:
                continue
            
            # 探测概率检查
            det_prob, in_notch = self._calc_detection_prob(agent, target, dist_km, config)
            
            # 已跟踪的目标提高探测概率
            if eid in self._radar_tracks[agent_id]:
                det_prob = min(1.0, det_prob + 0.3)
            
            if random.random() > det_prob:
                continue
            
            # 速度计算
            try:
                vel = target.get_velocity()
                radial_vel = np.linalg.norm(vel)
            except:
                radial_vel = 250.0
            
            # 创建或更新航迹
            if eid in self._radar_tracks[agent_id]:
                track = self._radar_tracks[agent_id][eid]
                track.distance = dist_km
                track.bearing = rel_bearing
                track.elevation = elev
                track.velocity = radial_vel
                track.detection_prob = det_prob
                track.last_update = current_time
                track.in_notch = in_notch
                track.update_count += 1
            else:
                self._radar_tracks[agent_id][eid] = RadarTrack(
                    track_id=eid,
                    distance=dist_km,
                    bearing=rel_bearing,
                    elevation=elev,
                    velocity=radial_vel,
                    detection_prob=det_prob,
                    last_update=current_time,
                    in_notch=in_notch,
                    update_count=1
                )
                # 🔥 首次探测日志
                _cap_print(f"[首次探测] {agent_id} -> {eid} | 方位:{rel_bearing:+.1f}° 距离:{dist_km:.0f}km 波束:{beam_az:.1f}°")
        
        # 🔥 诊断日志：打印目标扫描详情（每300次=60秒）
        if hasattr(self, '_radar_debug_counter') and self._radar_debug_counter % 300 == 1 and agent_id == 'A0100':
            _cap_print(f"  目标状态: {' | '.join(target_info) if target_info else '无目标'}")
    
    def _scan_radar(self, env, agent_id: str, current_time: float):
        """执行雷达扫描
        
        修复：波束检查逻辑改进
        - 使用正确的方位角归一化
        - 支持全域扫描模式（az_range >= 60时放宽限制）
        - 热段飞机朝北时可探测北方敌机
        """
        agent = env.agents[agent_id]
        config = self._configs.get(agent_id, RADAR_CAP_STD)
        center, az_range, bars = self._scan_assignments.get(agent_id, (0, 60, 2))
        
        # 获取本机航向（真北为0°）
        try:
            from envs.JSBSim.core.catalog import Catalog as c
            heading = np.degrees(agent.get_property_value(c.attitude_heading_true_rad)) % 360
        except:
            heading = 0.0
        
        enemy_prefix = "B" if agent_id.startswith("A") else "A"
        
        for eid in env.agents:
            if not eid.startswith(enemy_prefix) or not env.agents[eid].is_alive:
                continue
            
            target = env.agents[eid]
            dist_km, abs_bearing, elev = self._calc_geometry(agent, target)
            
            # 距离检查
            if dist_km > config.max_detection_range:
                if eid in self._radar_tracks[agent_id]:
                    del self._radar_tracks[agent_id][eid]
                continue
            
            # 相对机头方位（-180° ~ +180°）
            rel_bearing = (abs_bearing - heading + 180) % 360 - 180

            # ✅ 前向视场限制：仅允许在前向 ±60° 内扫描/跟踪
            if abs(rel_bearing) > MAX_FORWARD_AZIMUTH_DEG:
                if eid in self._radar_tracks[agent_id]:
                    del self._radar_tracks[agent_id][eid]
                continue
            
            # 波束检查（改进版）
            is_locked = (eid == self._lock_targets.get(agent_id))
            
            # 计算波束偏移（考虑角度环绕）
            beam_offset = rel_bearing - center
            if beam_offset > 180:
                beam_offset -= 360
            elif beam_offset < -180:
                beam_offset += 360
            
            # 全域扫描模式（az_range >= 60）放宽到前半球
            # 注意：本项目按定义最大扫描仅 ±60，因此不再放宽到 ±90。
            effective_az_range = min(az_range, MAX_FORWARD_AZIMUTH_DEG)
            
            if is_locked:
                # 锁定目标：仍需在前向视场内
                if abs(rel_bearing) > MAX_FORWARD_AZIMUTH_DEG:
                    continue
            elif abs(beam_offset) > effective_az_range:
                continue
            
            # 探测概率
            det_prob, in_notch = self._calc_detection_prob(agent, target, dist_km, config)
            if is_locked:
                det_prob = max(det_prob, 0.95)
                in_notch = False
            
            # 探测判定
            if random.random() < det_prob:
                vel = self._calc_radial_velocity(agent, target)
                
                if eid in self._radar_tracks[agent_id]:
                    t = self._radar_tracks[agent_id][eid]
                    t.distance, t.bearing, t.elevation = dist_km, rel_bearing, elev
                    t.velocity, t.detection_prob = vel, det_prob
                    t.last_update, t.in_notch = current_time, in_notch
                    t.update_count += 1
                    # 降低日志频率，每5秒或概率输出
                    if t.update_count % 50 == 0:
                        log.debug(f"[雷达] {agent_id}更新{eid}: 距离{dist_km:.1f}km 概率{det_prob:.2f}")
                else:
                    # 检查跟踪数量限制
                    if len(self._radar_tracks[agent_id]) >= config.max_tracks:
                        removed = self._remove_farthest_track(agent_id)
                        if removed:
                            log.info(f"[雷达] {agent_id}移除最远航迹{removed}以跟踪{eid}")
                    
                    self._radar_tracks[agent_id][eid] = RadarTrack(
                        track_id=eid, distance=dist_km, bearing=rel_bearing,
                        elevation=elev, velocity=vel, detection_prob=det_prob,
                        last_update=current_time, in_notch=in_notch, update_count=1
                    )
                    log.info(f"🔒 [雷达] {agent_id}首次锁定{eid}: 距离{dist_km:.1f}km 方位{rel_bearing:.1f}° 高差{(target.get_position()[2]-agent.get_position()[2])/1000:.1f}km 概率{det_prob:.2f}")
            else:
                # 记录未探测到的原因（低频日志）
                if current_time % 5 < 0.2 and random.random() < 0.5:
                     reason = []
                     if in_notch: reason.append("Notch盲区")
                     if det_prob < 0.3: reason.append(f"概率低({det_prob:.2f})")
                     if reason:
                        log.debug(f"[雷达] {agent_id}未探测到{eid}: {' '.join(reason)}")

                if eid in self._radar_tracks[agent_id]:
                    self._radar_tracks[agent_id][eid].update_count = max(0, 
                        self._radar_tracks[agent_id][eid].update_count - 2)
    
    def _remove_farthest_track(self, agent_id: str):
        """移除最远的非锁定航迹"""
        tracks = self._radar_tracks[agent_id]
        lock_tid = self._lock_targets.get(agent_id)
        
        farthest_tid, max_dist = None, 0
        for tid, t in tracks.items():
            if tid != lock_tid and t.distance > max_dist:
                max_dist = t.distance
                farthest_tid = tid
        
        if farthest_tid:
            del tracks[farthest_tid]
            return farthest_tid
        return None
    
    def _calc_geometry(self, agent, target) -> Tuple[float, float, float]:
        """计算几何参数
        
        返回：(距离km, 绝对方位角°, 仰角°)
        方位角：真北为0°，顺时针增加
        """
        pos1 = np.array(agent.get_position())  # NED: (north, east, down)
        pos2 = np.array(target.get_position())
        
        diff = pos2 - pos1  # (delta_north, delta_east, delta_down)
        dist_m = np.linalg.norm(diff)
        horiz_dist = np.linalg.norm(diff[:2])
        
        # 方位角：arctan2(east, north) 得到真北为0°的方位角
        # NED坐标系：diff[0]=north, diff[1]=east
        abs_bearing = np.degrees(np.arctan2(diff[1], diff[0])) % 360
        elev = np.degrees(np.arctan2(-diff[2], horiz_dist)) if horiz_dist > 0 else 0
        
        return dist_m / 1000, abs_bearing, elev
    
    def _calc_radial_velocity(self, agent, target) -> float:
        """计算径向速度"""
        try:
            pos1, pos2 = np.array(agent.get_position()), np.array(target.get_position())
            vel2 = np.array(target.get_velocity())
            
            los = pos2 - pos1
            los_norm = los / (np.linalg.norm(los) + 1e-6)
            return float(np.dot(vel2, los_norm))
        except:
            return 0.0
    
    def _calc_detection_prob(self, agent, target, dist_km: float, 
                            config: RadarConfig) -> Tuple[float, bool]:
        """计算探测概率 (功能级雷达建模)
        
        包含：
        1. 距离衰减
        2. 多普勒盲区 (Notch)
        3. 低空探测概率衰减 (地面杂波)
        4. 俯视角探测衰减 (Look-down)
        5. 最小仰角限制
        """
        if dist_km > config.max_detection_range:
            return 0.0, False
        
        # === 基础探测概率 ===
        # 距离因子（雷达方程简化：1/R^2）
        range_factor = 1.0 - (dist_km / config.max_detection_range) ** 2
        base = config.base_detection_prob * max(0.3, range_factor)
        
        # === 多普勒盲区 (Notch) ===
        radial_vel = self._calc_radial_velocity(agent, target)
        in_notch = abs(radial_vel) < self.NOTCH_VELOCITY
        notch_factor = 0.15 if in_notch else min(1.2, 1.0 + abs(radial_vel) / 400)
        
        # === 低空探测概率衰减 (地面杂波) ===
        # 目标高度越低，地面杂波干扰越强
        try:
            target_alt_m = target.get_position()[2]  # NED: up is negative
            if target_alt_m < 0:  # NED坐标系中高度向上为负
                target_alt_m = -target_alt_m
            target_alt_km = target_alt_m / 1000.0
        except:
            target_alt_km = 8.0  # 默认中空
        
        if target_alt_km > 5.0:
            altitude_factor = 1.0        # 高空：正常探测
        elif target_alt_km > 3.0:
            altitude_factor = 0.9        # 中空偏低：轻微衰减
        elif target_alt_km > 1.0:
            altitude_factor = 0.7        # 低空：显著衰减（地面杂波增强）
        else:
            altitude_factor = 0.4        # 超低空：严重衰减（地形遮挡+杂波）
        
        # === 俯视角探测衰减 (Look-down) ===
        # 当本机俯视目标时，地面杂波更强
        try:
            agent_alt_m = agent.get_position()[2]
            if agent_alt_m < 0:
                agent_alt_m = -agent_alt_m
            agent_alt_km = agent_alt_m / 1000.0
            
            # 俯视角估算（简化：高度差/水平距离）
            alt_diff = agent_alt_km - target_alt_km
            if dist_km > 0:
                look_down_angle = np.degrees(np.arctan2(alt_diff * 1000, dist_km * 1000))
            else:
                look_down_angle = 0.0
        except:
            look_down_angle = 0.0
            agent_alt_km = 8.0
        
        # 俯视角 < -30° 时杂波干扰增强
        if look_down_angle < -30:
            look_down_factor = 0.6  # 大俯视角：地面杂波严重
        elif look_down_angle < -15:
            look_down_factor = 0.8  # 中等俯视角
        else:
            look_down_factor = 1.0  # 平视或仰视：正常
        
        # === 最终概率 ===
        final_prob = base * notch_factor * altitude_factor * look_down_factor
        
        return max(0.0, min(1.0, final_prob)), in_notch
    
    def _count_trackers(self, target_id: str) -> int:
        """统计跟踪指定目标的战机数量
        
        Args:
            target_id: 目标ID
            
        Returns:
            跟踪该目标的战机数量
        """
        count = 0
        for agent_id, tracks in self._radar_tracks.items():
            if target_id in tracks:
                count += 1
        return count
    
    def _update_tracks(self, agent_id: str, current_time: float, config: RadarConfig):
        """更新航迹状态和质量"""
        expired = []
        for tid, track in self._radar_tracks[agent_id].items():
            age = current_time - track.last_update
            
            if age > self.TRACK_EXPIRE_TIME:
                expired.append(tid)
            else:
                # 跟踪质量：更新频率 × 年龄因子 × 距离因子 × 融合度因子
                # 更新因子：连续更新5次达到满分
                update_factor = min(1.0, track.update_count / 5)
                # 年龄因子：刚更新为1.0，30秒未更新为0
                age_factor = max(0, 1.0 - age / self.TRACK_EXPIRE_TIME)
                # 距离因子：使用探测距离而非跟踪距离，确保远距离也有质量
                # 在探测范围内时，距离因子从1.0（近）到0.3（远）
                dist_factor = max(0.3, 1.0 - 0.7 * track.distance / config.max_detection_range)
                # 融合度因子：统计跟踪该目标的战机数量，最优为2机
                tracker_count = self._count_trackers(tid)
                fusion_factor = min(1.0, tracker_count / 2.0)
                
                track.track_quality = update_factor * age_factor * dist_factor * fusion_factor
        
        for tid in expired:
            del self._radar_tracks[agent_id][tid]
    
    def _update_mode(self, agent_id: str):
        """更新雷达模式"""
        tracks = self._radar_tracks[agent_id]
        lock_tid = self._lock_targets.get(agent_id)
        
        if lock_tid:
            if lock_tid in tracks:
                self._radar_modes[agent_id] = RadarMode.LOCK
                return
            else:
                self._lock_targets[agent_id] = None
        
        if tracks:
            self._radar_modes[agent_id] = RadarMode.TRACK
        elif self._radar_modes[agent_id] not in (RadarMode.DIRECTED, RadarMode.SEARCH_ALL):
            self._radar_modes[agent_id] = RadarMode.SEARCH
                
    def lock_target(self, agent_id: str, target_id: Optional[str]):
        """设置锁定目标"""
        self._lock_targets[agent_id] = target_id
        if target_id and target_id in self._radar_tracks.get(agent_id, {}):
            self._radar_modes[agent_id] = RadarMode.LOCK
    
    # === 外部接口 ===
    
    def get_mode(self, agent_id: str) -> RadarMode:
        return self._radar_modes.get(agent_id, RadarMode.SEARCH)
    
    def get_tracks(self, agent_id: str) -> Dict[str, RadarTrack]:
        return self._radar_tracks.get(agent_id, {})
    
    def get_lock_target(self, agent_id: str) -> Optional[str]:
        return self._lock_targets.get(agent_id)
    
    def get_nearest_target(self, agent_id: str) -> Optional[RadarTrack]:
        tracks = self._radar_tracks.get(agent_id, {})
        return min(tracks.values(), key=lambda t: t.distance) if tracks else None
    
    def is_target_locked(self, agent_id: str) -> bool:
        return self._lock_targets.get(agent_id) is not None
    
    def get_track_quality(self, agent_id: str, target_id: str) -> float:
        """获取指定目标的跟踪质量"""
        tracks = self._radar_tracks.get(agent_id, {})
        if target_id in tracks:
            return tracks[target_id].track_quality
        return 0.0
    
    def get_all_track_qualities(self) -> Dict[str, Dict[str, float]]:
        """获取所有跟踪质量 {agent_id: {target_id: quality}}"""
        result = {}
        for aid, tracks in self._radar_tracks.items():
            result[aid] = {tid: t.track_quality for tid, t in tracks.items()}
        return result
