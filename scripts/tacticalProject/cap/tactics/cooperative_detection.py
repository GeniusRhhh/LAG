"""
协同探测模块 - 4机雷达协同扫描
三种模式：推磨扫描、定向扫描、全域扫描
优化：基于飞机实际位置分配扫描扇区
集成：IMM-EKF状态估计 + 置信域预测
"""
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .radar_scan import RadarMode, RadarScanParams, DEFAULT_RADAR

log = logging.getLogger(__name__)

# 卡方分布阈值 (用于置信域计算)
CHI2_GAMMA = {
    0.90: 4.605,
    0.95: 5.991,
    0.99: 9.210,
}


class DetectionMode(Enum):
    """探测模式"""
    SWEEP = "SWEEP"       # 推磨扫描（无预警信息）
    DIRECTED = "DIRECTED" # 定向扫描（有预警信息）
    SEARCH = "SEARCH"     # 全域扫描（目标丢失）


@dataclass
class ScanAssignment:
    """扫描任务分配"""
    agent_id: str
    azimuth_center: float    # 扫描中心方位(度)
    azimuth_range: float     # 扫描范围(±度)
    elevation_bars: int      # 俯仰行数
    radar_mode: RadarMode
    scan_time: float         # 扫描周期(秒)


class CooperativeDetection:
    """协同探测管理器 - 基于飞机位置智能分配扫描扇区
    
    增强功能：
    1. IMM-EKF状态估计集成
    2. 置信域预测（算法2.5）
    3. 动态扫描范围计算（公式3.3-3.4）
    """
    
    TARGET_LOST_TIMEOUT = 20.0  # 目标丢失超时(秒)
    
    def __init__(self, faor_width: float = 200.0, faor_length: float = 300.0, enable_3d: bool = True):
        self.faor_width = faor_width
        self.faor_length = faor_length
        self.enable_3d = enable_3d  # 3D模式默认启用
        self._mode = DetectionMode.SWEEP
        self._assignments: Dict[str, ScanAssignment] = {}
        self._last_update = 0.0
        self._target_lost_time: Optional[float] = None
        self._last_track_time: Dict[str, float] = {}  # 目标最后跟踪时间
        # 智能搜索：存储最后已知目标信息
        self._last_known_targets: Dict[str, Dict] = {}  # {tid: {pos, vel, heading, time}}
        
        # IMM状态估计器（延迟初始化）
        self._imm_estimators: Dict[str, 'IMMFilter'] = {}
        self._target_covariances: Dict[str, np.ndarray] = {}
        
        # 敌方编队估计
        self._enemy_center: Optional[Tuple[float, float]] = None
        self._confidence_radius: float = 10.0  # 默认置信半径 km
        
        # 3D扩展：高度协调（默认启用）
        self._altitude_coordinator = None
        if enable_3d:
            try:
                from .altitude_coordination import AltitudeCoordination
                self._altitude_coordinator = AltitudeCoordination()
                log.info("[3D高度协调] 模块已启用")
            except ImportError:
                log.warning("[3D高度协调] 无法导入高度协调模块，3D功能禁用")
                self.enable_3d = False

    
    @property
    def mode(self) -> DetectionMode:
        return self._mode
    
    def update(self, agents: List[str], has_awacs_info: bool, 
               target_lost: bool, awacs_bearing: Optional[float],
               current_time: float,
               agent_positions: Dict[str, Tuple[float, float]] = None,
               all_target_bearings: List[float] = None) -> Dict[str, ScanAssignment]:
        """更新扫描分配
        
        Args:
            agents: 可用飞机列表（仅热段飞机）
            has_awacs_info: 是否有预警机信息
            target_lost: 目标是否丢失
            awacs_bearing: 预警机提供的最近目标方位(度)
            current_time: 当前时间
            agent_positions: 飞机位置 {id: (x_km, y_km)}，用于智能分配
            all_target_bearings: 所有目标的方位列表(度)，用于自适应覆盖
        """
        # 探测模式决策（立即响应）
        if target_lost:
            # 目标丢失立即进入SEARCH模式（不等待20秒）
            # 20秒是最坏情况的持续时间，不是触发阈值
            self._mode = DetectionMode.SEARCH
            if self._target_lost_time is None:
                self._target_lost_time = current_time
                log.info("[协同] 目标丢失，立即进入SEARCH模式")
        elif has_awacs_info:
            self._mode = DetectionMode.DIRECTED
            self._target_lost_time = None
        else:
            self._mode = DetectionMode.SWEEP
            self._target_lost_time = None
        
        # 分配扫描任务
        if self._mode == DetectionMode.SWEEP:
            self._assign_sweep_scan(agents, agent_positions)
        elif self._mode == DetectionMode.DIRECTED:
            # 🔥 Debug: 打印传入参数（每300秒一次）
            if current_time % 300 < 0.3:
                print(f"[协同探测update] agents={agents} (共{len(agents)}架)")
                print(f"  agent_positions keys: {list(agent_positions.keys()) if agent_positions else 'None'}")
                print(f"  all_target_bearings: {len(all_target_bearings) if all_target_bearings else 0}个")
            
            self._assign_directed_scan(agents, awacs_bearing or 0.0, all_target_bearings, agent_positions)
        else:
            self._assign_search_scan(agents, agent_positions)
        
        self._last_update = current_time
        
        # 详细扫描分配日志 (低频采样：每300秒一次)
        # if current_time % 300 < 0.3:
        #     assign_str = []
        #     for aid, asgn in self._assignments.items():
        #         az_str = f"{asgn.azimuth_center-asgn.azimuth_range/2:.0f}~{asgn.azimuth_center+asgn.azimuth_range/2:.0f}"
        #         assign_str.append(f"{aid}:{az_str}")
        #     print(f"[扫描] {self._mode.value} | {' '.join(assign_str)}")
            
        return self._assignments
    
    def _assign_sweep_scan(self, agents: List[str], positions: Dict[str, Tuple[float, float]] = None):
        """推磨扫描：基于飞机X位置分配扫描扇区
        
        使用MEDIUM模式（±30°，2行，10s）进行推磨扫描
        """
        n = len(agents)
        if n == 0:
            return
        
        # 按X位置排序飞机（从左到右）
        if positions:
            sorted_agents = sorted(agents, key=lambda a: positions.get(a, (0, 0))[0])
        else:
            sorted_agents = agents
        
        # 每机负责一个扇区，总覆盖±60°
        total_angle = 120.0  # 总覆盖角度 ±60°
        sector_angle = total_angle / n
        
        for i, aid in enumerate(sorted_agents):
            # 从左到右分配扇区
            azimuth_center = -60 + sector_angle * (i + 0.5)
            
            self._assignments[aid] = ScanAssignment(
                agent_id=aid,
                azimuth_center=azimuth_center,
                azimuth_range=30.0,  # 固定±30°（MEDIUM模式）
                elevation_bars=2,
                radar_mode=RadarMode.MEDIUM,
                scan_time=10.0  # MEDIUM + 2行 = 10s
            )
    
    def _assign_directed_scan(self, agents: List[str], bearing: float, 
                               all_target_bearings: List[float] = None,
                               agent_positions: Dict[str, Tuple[float, float]] = None):
        """定向扫描：多机协同扫描目标区域（略微分散提高覆盖可靠性）
        
        算法来源：《多机协同探测算法设计与验证报告》第4.2.3节
        
        设计原理：
        1. 计算所有目标的方位跨度和中心
        2. 各战机扫描中心略微分散（Δθ_spread=5°）
        3. 既保证目标区域完整覆盖，又避免完全重叠
        4. 任何飞机探测到目标立即共享给全队
        
        分散公式：
        δ_i = (i - (N-1)/2) × Δθ_spread
        
        优点：
        - 提高覆盖可靠性（避免所有雷达同时指向同一点）
        - 扩大有效覆盖范围（处理预警机误差）
        - 保持快速截获能力（仍使用NARROW模式）
        - 充分利用数据链（任何飞机探测到立即共享）
        """
        # 初始化调试计数器
        if not hasattr(self, '_scan_debug_counter'):
            self._scan_debug_counter = 0
        self._scan_debug_counter += 1
        
        n = len(agents)
        if n == 0:
            return
        
        # 如果没有多目标信息，回退到单一方位模式
        if not all_target_bearings or len(all_target_bearings) == 0:
            all_target_bearings = [bearing]
        
        # 计算目标区域中心和跨度
        min_bearing = min(all_target_bearings)
        max_bearing = max(all_target_bearings)
        bearing_span = max_bearing - min_bearing
        
        # 方位跨度过大时的处理 (可能跨越±180度边界)
        if bearing_span > 180:
            # 目标可能分布在-170°和+170°，实际跨度是20°
            adjusted_bearings = [(b + 360 if b < 0 else b) for b in all_target_bearings]
            min_bearing = min(adjusted_bearings)
            max_bearing = max(adjusted_bearings)
            bearing_span = max_bearing - min_bearing
        
        center_bearing = (min_bearing + max_bearing) / 2
        # 转换回-180~180范围
        if center_bearing > 180:
            center_bearing -= 360
        
        # 🔥 Debug: 打印目标分布情况（每600次=120秒一次）
        # if self._scan_debug_counter % 600 == 1:
        #     target_list = ', '.join([f"{b:.1f}°" for b in all_target_bearings])
        #     print(f"[扇区分配] 目标数:{len(all_target_bearings)} 方位:[{target_list}] 跨度:{bearing_span:.1f}° ({min_bearing:.1f}~{max_bearing:.1f})")
        
        # === 定向扫描模式（智能分散策略）===
        # 
        # 策略选择逻辑：
        # 1. 如果目标跨度小（<15°）：所有飞机指向中心（集中火力）
        # 2. 如果目标跨度大（≥15°）：略微分散（扩大覆盖）
        #
        # 雷达硬属性约束（不可违反）：
        # - NARROW: ±10° / 2s
        # - MEDIUM: ±30° / 5s
        # - WIDE: ±60° / 10s
        
        # 判断是否需要分散 + 自适应扫描范围
        #
        # 过去这里固定使用 NARROW(±10°) 会导致“边缘目标长期不在任何扇区内”，
        # 在 V03（宽正面 + 低空 + 折线/回正 + 预警信息间歇）中会表现为 B0400 永久漏探。
        #
        # 修复策略：根据目标方位跨度选择 NARROW/MEDIUM/WIDE，确保覆盖完整性。
        # - 小跨度：NARROW 快速首探
        # - 中等跨度：MEDIUM 平衡覆盖与刷新
        # - 大跨度：WIDE 保底全探（牺牲刷新速度，但避免漏探）
        use_spread = bearing_span >= 15.0

        if bearing_span >= 50.0:
            scan_range = 60.0
            radar_mode = RadarMode.WIDE
            scan_time = 10.0
            spread_angle = 20.0 if use_spread else 0.0
        elif bearing_span >= 20.0:
            scan_range = 30.0
            radar_mode = RadarMode.MEDIUM
            scan_time = 5.0
            spread_angle = 10.0 if use_spread else 0.0
        else:
            scan_range = 10.0
            radar_mode = RadarMode.NARROW
            scan_time = 2.0
            spread_angle = 5.0 if use_spread else 0.0
        
        # 为各飞机分配扫描任务
        for i, aid in enumerate(agents):
            # 计算偏移量：δ_i = (i - (N-1)/2) × Δθ_spread
            offset = (i - (n - 1) / 2) * spread_angle
            scan_center = center_bearing + offset
            
            self._assignments[aid] = ScanAssignment(
                agent_id=aid,
                azimuth_center=scan_center,
                azimuth_range=scan_range,
                elevation_bars=1,  # 单行扫描（最快）
                radar_mode=radar_mode,
                scan_time=scan_time
            )
        
        # 🔥 Debug: 打印定向扫描分配（每600次=120秒一次）
        # if self._scan_debug_counter % 600 == 1:
        #     print(f"[扇区分配] ✅ 模式:定向扫描（智能分散）")
        #     print(f"  目标区域: 中心{center_bearing:.1f}° 跨度{bearing_span:.1f}°")
        #     print(f"  扫描范围: ±{scan_range:.1f}° ({radar_mode.value})")
        #     print(f"  分散策略: {spread_angle:.1f}°间隔 ({'启用' if use_spread else '禁用'})")
        #     for aid in agents:
        #         asgn = self._assignments[aid]
        #         az_min = asgn.azimuth_center - asgn.azimuth_range
        #         az_max = asgn.azimuth_center + asgn.azimuth_range
        #         print(f"  {aid}: 中心{asgn.azimuth_center:+.1f}° 范围±{asgn.azimuth_range:.1f}° 覆盖[{az_min:+.1f}°, {az_max:+.1f}°]")
    
    def _assign_search_scan(self, agents: List[str], positions: Dict[str, Tuple[float, float]] = None):
        """智能搜索：基于目标最后已知信息预测位置并聚焦搜索
        
        改进：不是全域扫描，而是：
        1. 根据目标最后位置、速度、航向预测当前位置
        2. 所有战机协同指向预测区域
        3. 扩大搜索范围应对预测误差
        """
        n = len(agents)
        if n == 0:
            return
        
        current_time = self._last_update

        # 编队中心（用于把预测位置映射为相对方位）
        if positions:
            pos_list = [positions[a] for a in agents if a in positions]
            if pos_list:
                center_x = float(np.mean([p[0] for p in pos_list]))
                center_y = float(np.mean([p[1] for p in pos_list]))
            else:
                center_x, center_y = 100.0, 0.0
        else:
            center_x, center_y = 100.0, 0.0
        
        # 计算预测的目标方位
        predicted_bearings = []
        for tid, info in self._last_known_targets.items():
            if info is None:
                continue
            # 预测目标位置
            dt = current_time - info.get('time', current_time)
            pos = info.get('pos', (0, 100))
            vel = info.get('vel', (0, -0.3))  # km/s
            
            # 简单线性外推
            pred_x = pos[0] + vel[0] * dt
            pred_y = pos[1] + vel[1] * dt
            
            # 计算相对编队中心的方位（真北=0°，顺时针为正），再转为(-180,180]
            bearing = np.degrees(np.arctan2(pred_x - center_x, pred_y - center_y))
            bearing = (bearing + 180) % 360 - 180
            predicted_bearings.append(float(bearing))
        
        # 如果有预测目标，聚焦该方向；否则回退到全域扫描
        if predicted_bearings:
            avg_bearing = np.mean(predicted_bearings)
            # 按报告：SEARCH 采用 WIDE（±60°）覆盖预测区域
            for aid in agents:
                self._assignments[aid] = ScanAssignment(
                    agent_id=aid,
                    azimuth_center=float(avg_bearing),
                    azimuth_range=60.0,
                    elevation_bars=4,
                    radar_mode=RadarMode.WIDE,
                    scan_time=RadarScanParams.get_scan_time(RadarMode.WIDE, 4)
                )
            
            # 限制日志频率
            if current_time % 10 < 0.3:
                log.info(f"[协同] 智能搜索: 预测方位={avg_bearing:.1f}°, 覆盖范围=±60°")
        else:
            # 无预测信息，回退到均匀全域扫描
            total_angle = 120.0
            sector_angle = total_angle / n
            
            sorted_agents = agents
            if positions:
                sorted_agents = sorted(agents, key=lambda a: positions.get(a, (0, 0))[0])
            
            for i, aid in enumerate(sorted_agents):
                azimuth_center = -60 + sector_angle * (i + 0.5)
                self._assignments[aid] = ScanAssignment(
                    agent_id=aid,
                    azimuth_center=azimuth_center,
                    azimuth_range=60.0,
                    elevation_bars=4,
                    radar_mode=RadarMode.WIDE,
                    scan_time=RadarScanParams.get_scan_time(RadarMode.WIDE, 4)
                )
    
    def update_track_time(self, target_id: str, current_time: float):
        """更新目标最后跟踪时间（用于丢失检测/σ_man传播）

        约定：current_time 表示“该目标最后一次获得新量测/可信更新”的时间戳。
        为避免每步无量测也刷新导致 τ≈0，本函数只允许时间戳单调前进。
        """
        last = self._last_track_time.get(target_id)
        if last is None or current_time > last:
            self._last_track_time[target_id] = current_time
    
    def update_target_info(self, target_id: str, position: Tuple[float, float], 
                           velocity: Tuple[float, float], heading: float, 
                           current_time: float):
        """更新目标最后已知信息 (用于智能搜索预测)
        
        Args:
            target_id: 目标ID
            position: (x, y) km
            velocity: (vx, vy) km/s
            heading: 航向(度)
            current_time: 当前时间(秒)
        """
        # 仅当时间戳前进时才刷新（防止无量测每步覆盖“最后已知时刻”）
        prev = self._last_known_targets.get(target_id)
        prev_time = None if not prev else prev.get('time')
        if prev_time is None or current_time > prev_time:
            self._last_known_targets[target_id] = {
                'pos': position,
                'vel': velocity,
                'heading': heading,
                'time': current_time
            }
            self.update_track_time(target_id, current_time)
    
    def is_target_lost(self, target_id: str, current_time: float) -> bool:
        """检查目标是否丢失（超过20秒未更新）"""
        last = self._last_track_time.get(target_id)
        if last is None:
            return False
        return current_time - last > self.TARGET_LOST_TIMEOUT

    def any_target_lost(self, current_time: float) -> bool:
        """检查是否存在任一已知目标丢失。

        重要：覆盖“掉出当前输入列表”的目标（仍保留在_last_track_time里）。
        """
        if not self._last_track_time:
            return False
        return any((current_time - t) > self.TARGET_LOST_TIMEOUT for t in self._last_track_time.values())

    def get_last_track_time(self, target_id: str) -> Optional[float]:
        """获取用于IMM估计器的“最后一次新量测时间戳”。

        说明：
        - 协同探测层可能在无IMM更新时也记录了_last_track_time（用于模式切换/丢失触发）。
        - 拦截链路的IMM初始化应以“估计器是否已有状态”为准。
        因此：若该目标尚未创建IMM估计器，则返回None，确保首次量测能触发初始化更新。
        """
        if target_id not in self._imm_estimators:
            return None

        # 关键：不要直接返回 _last_track_time。
        # _last_track_time 既被IMM更新推进，也可能被update_target_info/雷达航迹推进，
        # 若用它门控IMM更新，会导致“量测已记录但IMM未真正update”从而长期卡在首次量测。
        # 因此这里返回“IMM估计器内部已吸收量测的时间戳”。
        try:
            state = self._imm_estimators[target_id].get_state(target_id)
            if state is not None and getattr(state, 'timestamp', None) is not None:
                return float(state.timestamp)
        except Exception:
            pass

        # 兜底：若无法从估计器取时间戳，再退回_last_track_time
        return self._last_track_time.get(target_id)

    def predict_target_state(self, target_id: str, current_time: float) -> Tuple[Tuple[float, float], np.ndarray]:
        """无新量测时给出目标在current_time的预测位置/协方差。

        注意：该预测不应改变滤波器内部状态（用于规划/显示与σ_man传播）。
        """
        # 没有估计器：退回到最后已知信息或0
        if target_id not in self._imm_estimators:
            info = self._last_known_targets.get(target_id, None)
            if info and 'pos' in info:
                pos = info['pos']
            else:
                pos = (0.0, 0.0)
            return (float(pos[0]), float(pos[1])), (self._target_covariances.get(target_id) or (np.eye(4) * 10.0))

        last_time = self._last_track_time.get(target_id, current_time)
        tau = max(0.0, current_time - last_time)
        estimator = self._imm_estimators[target_id]
        try:
            x_pred, P_pred = estimator.predict(tau)
            self._target_covariances[target_id] = P_pred
            return (float(x_pred[0]), float(x_pred[1])), P_pred
        except Exception:
            state = None
            try:
                state = estimator.get_state(target_id)
            except Exception:
                state = None
            if state is not None:
                self._target_covariances[target_id] = state.covariance
                return (float(state.x), float(state.y)), state.covariance
            return (0.0, 0.0), (self._target_covariances.get(target_id) or (np.eye(4) * 10.0))
    
    def get_assignment(self, agent_id: str) -> Optional[ScanAssignment]:
        return self._assignments.get(agent_id)
    
    def get_hot_agents(self, patrol_states: Dict[str, str], cap_state: str = None) -> List[str]:
        """获取参与协同探测的飞机列表
        
        改进：在ENGAGE/INTERCEPT阶段，所有存活飞机都参与协同探测
        只有在PATROL阶段才区分热段/冷段
        
        Args:
            patrol_states: 各飞机巡逻状态
            cap_state: 当前CAP总体状态 (PATROL/ENGAGE/INTERCEPT等)
        """
        # 在交战阶段，所有飞机都参与协同探测
        # 🔥 修复：兼容 "INTERCEPT" 和 "CAPState.INTERCEPT" 两种格式
        if cap_state:
            if 'ENGAGE' in cap_state or 'INTERCEPT' in cap_state:
                return list(patrol_states.keys())
        
        # 巡逻阶段：只有热段飞机参与
        return [aid for aid, state in patrol_states.items() 
                if 'HOT' in state or 'NORTH' in state]
    
    def get_coverage_report(self) -> Dict[str, float]:
        """获取扫描覆盖报告"""
        if not self._assignments:
            return {"total_coverage": 0, "overlap": 0}
        
        # 计算总覆盖角度和重叠
        ranges = [(a.azimuth_center - a.azimuth_range, a.azimuth_center + a.azimuth_range) 
                  for a in self._assignments.values()]
        ranges.sort()
        
        total = sum(r[1] - r[0] for r in ranges)
        # 简化重叠计算
        merged = 0
        prev_end = -180
        for start, end in ranges:
            if start < prev_end:
                merged += prev_end - start
            prev_end = max(prev_end, end)
        
        return {"total_coverage": total, "overlap": merged}
    
    # ============ 算法2.5: 置信域预测 ============
    
    def compute_confidence_radius(self, target_id: str, gamma: float = 0.99) -> float:
        """计算目标置信半径（算法2.5）
        
        根据目标状态协方差矩阵计算置信椭圆长轴
        r_target = sqrt(chi2_gamma * lambda_max(P_pos))
        
        Args:
            target_id: 目标ID
            gamma: 置信度 (0.90, 0.95, 0.99)
        
        Returns:
            置信半径 (km)
        """
        if target_id not in self._target_covariances:
            return self._confidence_radius  # 返回默认值
        
        P = self._target_covariances[target_id]
        if P is None or P.shape[0] < 2:
            return self._confidence_radius
        
        # 提取位置协方差子矩阵 (2x2)
        P_pos = P[:2, :2]
        
        # 特征值分解
        try:
            eigenvalues = np.linalg.eigvalsh(P_pos)
            lambda_max = max(eigenvalues)
        except np.linalg.LinAlgError:
            return self._confidence_radius
        
        # 卡方阈值
        chi2 = CHI2_GAMMA.get(gamma, 9.210)
        
        # 置信半径
        r = np.sqrt(chi2 * lambda_max)
        return max(r, 2.5)  # 最小2.5km（预警机精度）
    
    def update_target_state_imm(
        self, 
        target_id: str, 
        position: Tuple[float, float],
        current_time: float
    ) -> Tuple[Tuple[float, float], np.ndarray]:
        """使用IMM-EKF更新目标状态估计（增强版 - 算法1.1完整实现）
        
        Args:
            target_id: 目标ID
            position: 观测位置 (x, y) km
            current_time: 当前时间(秒)
        
        Returns:
            (估计位置, 协方差矩阵)
        """
        # 延迟导入IMM模块（使用增强版）
        try:
            from .algorithms.estimators import IMMEKFEstimator
        except ImportError:
            # 回退到简单估计
            self._target_covariances[target_id] = np.eye(4) * 10.0
            return position, self._target_covariances[target_id]
        
        # 初始化或获取估计器
        if target_id not in self._imm_estimators:
            self._imm_estimators[target_id] = IMMEKFEstimator()

        # 若时间戳未前进，视为“无新量测”，不重复update（避免dt≈0反复量测更新）
        # 注意：只有在估计器已存在有效状态时才跳过；否则允许首次量测初始化。
        existing_state = None
        try:
            existing_state = self._imm_estimators[target_id].get_state(target_id)
        except Exception:
            existing_state = None

        # 使用“估计器内部时间戳”作为是否有新量测的判据
        last_meas = None
        if existing_state is not None:
            try:
                last_meas = float(getattr(existing_state, 'timestamp', None))
            except Exception:
                last_meas = None

        if existing_state is not None and last_meas is not None and current_time <= last_meas + 1e-6:
            # 返回预测/当前状态，但不刷新_last_track_time
            return self.predict_target_state(target_id, max(current_time, last_meas))
        
        # 计算时间间隔（不需要，estimator内部处理）
        # IMM-EKF更新
        estimator = self._imm_estimators[target_id]
        measurement_noise = 2.5  # AWACS error (km)
        
        state = estimator.update(
            target_id=target_id,
            measurement=position,
            measurement_noise=measurement_noise,
            timestamp=current_time
        )
        
        # 存储协方差
        self._target_covariances[target_id] = state.covariance
        self.update_track_time(target_id, current_time)
        
        # 更新置信半径
        self._confidence_radius = self.compute_confidence_radius(target_id)
        
        return (state.x, state.y), state.covariance
    
    def estimate_enemy_center(
        self, 
        target_positions: Dict[str, Tuple[float, float]]
    ) -> Tuple[Tuple[float, float], float]:
        """估计敌方编队中心和分布范围
        
        Args:
            target_positions: 各目标位置 {tid: (x, y)}
        
        Returns:
            (中心位置, 分布半径)
        """
        if not target_positions:
            return self._enemy_center or (100, 200), self._confidence_radius
        
        positions = list(target_positions.values())
        center_x = np.mean([p[0] for p in positions])
        center_y = np.mean([p[1] for p in positions])
        
        # 计算分布半径（各目标到中心的最大距离）
        max_dist = 0.0
        for p in positions:
            d = np.sqrt((p[0] - center_x)**2 + (p[1] - center_y)**2)
            max_dist = max(max_dist, d)
        
        # 置信半径 = 分布半径 + 单目标不确定性
        total_radius = max_dist + self._confidence_radius
        
        self._enemy_center = (center_x, center_y)
        return (center_x, center_y), total_radius
    
    def compute_dynamic_scan_range(
        self, 
        target_id: str,
        distance: float,
        gamma: float = 0.99
    ) -> float:
        """计算动态扫描范围（公式3.3-3.4）
        
        基于协方差椭圆长轴和战机到目标距离计算扫描角度
        theta_scan = 2 * arctan(a / d)
        
        Args:
            target_id: 目标ID
            distance: 战机到目标距离 (km)
            gamma: 置信度
        
        Returns:
            扫描半宽 (度)，限制在[10, 60]范围内
        """
        if distance < 1.0:
            distance = 1.0
        
        # 获取置信半径
        r = self.compute_confidence_radius(target_id, gamma)
        
        # 计算扫描角度
        theta_scan = np.degrees(2 * np.arctan(r / distance))
        
        # 限制在雷达体制允许范围内
        theta_scan = max(10.0, min(60.0, theta_scan))
        
        return theta_scan
    
    @property
    def enemy_center(self) -> Optional[Tuple[float, float]]:
        """获取敌方编队中心估计"""
        return self._enemy_center
    
    @property
    def confidence_radius(self) -> float:
        """获取当前置信半径"""
        return self._confidence_radius
    
    # ============ 算法2.5完整: σ_man机动不确定性传播 ============
    
    def propagate_sigma_man(
        self,
        current_time: float,
        gamma: float = 0.99,
        target_ids: Optional['List[str]'] = None,
        max_tau: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """算法2.5完整实现：机动不确定性传播与置信域预测
        
        在无新量测时间段内,用IMM预测方程将各目标位置协方差向前传播,
        得到t时刻位置不确定性,按置信度γ构造外接圆半径r_j(t),
        取所有目标上界作为σ_man(t)
        
        Args:
            current_time: 当前时间
            gamma: 置信度
        
        Returns:
            (sigma_man, {target_id: r_j(t)})
        """
        chi2 = CHI2_GAMMA.get(gamma, 9.210)
        r_targets: Dict[str, float] = {}

        # 关键修复：
        # - 只对“当前仍在用/仍在输入列表中的目标”传播不确定性。
        #   否则掉线目标(>20s无量测)会把 sigma_man 拉到几百/上千km，
        #   进而让 R_target 爆炸，导致编队引导目标点跳变、早期出现大量无用转弯。
        ids = list(target_ids) if target_ids else list(self._imm_estimators.keys())

        # 预测时域上限：若未提供则使用丢失超时，避免极端dropout导致数值失控
        tau_cap = float(max_tau) if max_tau is not None else float(getattr(self, 'TARGET_LOST_TIMEOUT', 20.0))
        tau_cap = max(0.0, tau_cap)

        for tid in ids:
            estimator = self._imm_estimators.get(tid)
            if estimator is None:
                continue

            last_time = self._last_track_time.get(tid, current_time)
            tau = current_time - last_time
            if tau_cap > 0.0:
                tau = min(tau, tau_cap)
            
            if tau < 0.1:
                # 刚更新，直接用当前协方差
                r = self.compute_confidence_radius(tid, gamma)
            else:
                # 无量测预测：用IMM predict向前传播协方差
                try:
                    x_pred, P_pred = estimator.predict(tau, target_id=tid)
                    P_pos = P_pred[:2, :2]
                    eigenvalues = np.linalg.eigvalsh(P_pos)
                    lambda_max = max(eigenvalues)
                    r = np.sqrt(chi2 * lambda_max)
                    r = max(r, 2.5)
                except Exception:
                    r = self._confidence_radius
            
            r_targets[tid] = r
        
        # σ_man(t) = max_j r_j(t)  (式2.4d)
        sigma_man = max(r_targets.values()) if r_targets else self._confidence_radius
        return sigma_man, r_targets
    
    # ============ 公式2.4: 敌方分布范围估计 ============
    
    def compute_distribution_range(
        self,
        target_positions: Dict[str, Tuple[float, float]],
        current_time: float,
        sigma_awacs: float = 2.5,
        k_sigma: float = 3.0
    ) -> Tuple[float, float]:
        """计算敌方分布范围σ_enemy和目标区域半径R_target（公式2.4/2.5）
        
        σ_enemy(t) = max(σ_spatial, σ_awacs) + σ_man(t)
        R_target = k_σ · σ_enemy
        
        Args:
            target_positions: 各目标位置
            current_time: 当前时间
            sigma_awacs: 预警机位置误差(km)
            k_sigma: 覆盖系数(默认3倍)
        
        Returns:
            (sigma_enemy, R_target)
        """
        M = len(target_positions)
        
        # σ_spatial: 空间分散度
        if M <= 1:
            sigma_spatial = 5.0  # 默认值
        else:
            positions = list(target_positions.values())
            cx = np.mean([p[0] for p in positions])
            cy = np.mean([p[1] for p in positions])
            sigma_spatial = np.sqrt(
                sum((p[0]-cx)**2 + (p[1]-cy)**2 for p in positions) / M
            )
        
        # σ_man: 机动不确定性增量（避免把AWACS下限重复叠加）
        sigma_man_total, _ = self.propagate_sigma_man(
            current_time=current_time,
            gamma=0.99,
            target_ids=list(target_positions.keys()),
        )
        sigma_man_inc = max(0.0, sigma_man_total - sigma_awacs)

        # σ_enemy = max(σ_spatial, σ_awacs) + σ_man_inc  (式2.4)
        sigma_enemy = max(sigma_spatial, sigma_awacs) + sigma_man_inc
        
        # R_target = k_σ · σ_enemy  (式2.5)
        R_target = k_sigma * sigma_enemy
        
        return sigma_enemy, R_target
    
    # ============ 3D扩展: 高度协调接口 ============
    
    def compute_altitude_commands_3d(
        self,
        target_positions_3d: Dict[str, Tuple[float, float, float]],
        fighter_positions_2d: Dict[str, Tuple[float, float]],
        fighter_altitudes: Dict[str, float],
        horizontal_assignments: Dict[str, List[str]] = None
    ) -> Dict[str, 'AltitudeCommand']:
        """计算3D高度指令（如果启用3D模式）
        
        Args:
            target_positions_3d: 目标3D位置 {tid: (x, y, z)}
            fighter_positions_2d: 战机2D位置 {fid: (x, y)}
            fighter_altitudes: 战机当前高度 {fid: altitude_km}
            horizontal_assignments: 水平面分配结果（可选）
        
        Returns:
            高度指令字典，如果3D未启用则返回空字典
        """
        if not self.enable_3d or self._altitude_coordinator is None:
            return {}
        
        return self._altitude_coordinator.compute_altitude_commands(
            target_positions_3d=target_positions_3d,
            fighter_positions_2d=fighter_positions_2d,
            fighter_altitudes=fighter_altitudes,
            horizontal_assignments=horizontal_assignments
        )
    
    def get_altitude_command(self, fighter_id: str) -> Optional['AltitudeCommand']:
        """获取指定战机的高度指令"""
        if not self.enable_3d or self._altitude_coordinator is None:
            return None
        return self._altitude_coordinator.get_command(fighter_id)
    
    # ============ 算法2.10: 动态路径调整 ============
    
    def dynamic_path_adjustment(
        self,
        current_headings: Dict[str, float],
        current_speeds: Dict[str, float],
        target_headings: Dict[str, float],
        target_speeds: Dict[str, float],
        event_awacs_update: bool = False,
        event_maneuver_detected: bool = False,
        event_awacs_lost: bool = False,
        event_target_lost: bool = False,
        alpha: float = 0.7
    ) -> Tuple[Dict[str, float], Dict[str, float], str]:
        """算法2.10：动态路径调整
        
        根据四类事件决定是否重新计算引导目标:
        1. 预警更新 -> 重算中心+置信域+引导
        2. 敌方机动 -> 放大置信域+重算引导
        3. 预警丢失 -> 飞行层继续, 雷达切SEARCH
        4. 目标丢失 -> 切SEARCH
        
        最后对航向和速度做一阶平滑(系数α), 避免指令突变
        
        Args:
            current_headings: 各机当前航向
            current_speeds: 各机当前速度
            target_headings: 算法2.6输出的目标航向
            target_speeds: 算法2.7输出的目标速度
            event_*: 四类事件标志
            alpha: 一阶平滑系数(0-1, 越大越平滑)
        
        Returns:
            (new_headings, new_speeds, action_taken)
        """
        action_taken = "NONE"
        
        if event_awacs_update:
            action_taken = "AWACS_UPDATE"
            # 正常使用算法2.6/2.7的结果
        elif event_maneuver_detected:
            action_taken = "MANEUVER_DETECTED"
            # 正常使用算法2.6/2.7的结果(置信域已放大)
        elif event_awacs_lost:
            action_taken = "AWACS_LOST"
            # 飞行层保持引导, 雷达层需切SEARCH
        elif event_target_lost:
            action_taken = "TARGET_LOST"
            # 切SEARCH模式
        
        # 一阶平滑：θ_new = α·θ_current + (1-α)·θ_target
        new_headings = {}
        new_speeds = {}
        
        for fid in current_headings:
            theta_cur = current_headings[fid]
            theta_tgt = target_headings.get(fid, theta_cur)
            v_cur = current_speeds.get(fid, 250.0)
            v_tgt = target_speeds.get(fid, v_cur)
            
            # 航向平滑（处理360°环绕）
            diff = theta_tgt - theta_cur
            if diff > 180: diff -= 360
            elif diff < -180: diff += 360
            theta_new = (theta_cur + (1 - alpha) * diff) % 360
            
            # 速度平滑
            v_new = alpha * v_cur + (1 - alpha) * v_tgt
            
            new_headings[fid] = theta_new
            new_speeds[fid] = v_new
        
        return new_headings, new_speeds, action_taken
