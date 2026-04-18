"""
高度协调模块 - 3D扩展
处理垂直维度的目标分配和高度指令生成

设计原则：
1. 独立于2D水平面算法
2. 基于能量管理原则分配高度
3. 输出高度指令，与2D指令在执行层融合
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class AltitudeLayer(Enum):
    """高度层分类"""
    LOW = "LOW"      # <8km
    MID = "MID"      # 8-10km
    HIGH = "HIGH"    # >10km


@dataclass
class AltitudeCommand:
    """高度指令"""
    fighter_id: str
    current_altitude: float      # 当前高度 (km)
    target_altitude: float       # 目标高度 (km)
    climb_rate: float            # 爬升率 (m/s)，正为爬升，负为俯冲
    energy_advantage: bool       # 是否有能量优势
    assigned_targets: List[str]  # 分配的目标ID列表


class AltitudeCoordination:
    """高度协调管理器
    
    功能：
    1. 目标高度分层
    2. 战机-目标3D匹配（考虑能量优势）
    3. 生成高度指令
    """
    
    # 高度层边界
    LOW_CEILING = 8.0   # km
    MID_CEILING = 10.0  # km
    
    # 能量管理参数
    ENERGY_ADVANTAGE_FACTOR = 0.5   # 高打低的代价折扣
    ENERGY_DISADVANTAGE_FACTOR = 1.5  # 低打高的代价惩罚
    
    # 爬升性能参数（基于F-16/Su-27典型性能）
    MAX_CLIMB_RATE = 250.0  # m/s (约50,000 ft/min)
    MAX_DIVE_RATE = -150.0  # m/s
    CRUISE_CLIMB_RATE = 50.0  # m/s (巡航爬升)
    
    def __init__(self):
        self._last_commands: Dict[str, AltitudeCommand] = {}
    
    def compute_altitude_commands(
        self,
        target_positions_3d: Dict[str, Tuple[float, float, float]],
        fighter_positions_2d: Dict[str, Tuple[float, float]],
        fighter_altitudes: Dict[str, float],
        horizontal_assignments: Dict[str, List[str]] = None
    ) -> Dict[str, AltitudeCommand]:
        """计算高度协调指令
        
        Args:
            target_positions_3d: 目标3D位置 {tid: (x, y, z)}
            fighter_positions_2d: 战机2D位置 {fid: (x, y)}
            fighter_altitudes: 战机当前高度 {fid: altitude_km}
            horizontal_assignments: 水平面分配结果 {fid: [tid1, tid2, ...]}
                                   如果为None，则自动计算3D分配
        
        Returns:
            高度指令 {fid: AltitudeCommand}
        """
        if not target_positions_3d:
            return {}
        
        # 步骤1：目标高度分层
        target_layers = self._classify_targets_by_altitude(target_positions_3d)
        
        # 步骤2：战机高度分类
        fighter_layers = self._classify_fighters_by_altitude(fighter_altitudes)
        
        # 步骤3：3D目标分配（如果没有提供水平分配）
        if horizontal_assignments is None:
            assignments = self._allocate_targets_3d(
                target_positions_3d,
                fighter_positions_2d,
                fighter_altitudes,
                target_layers,
                fighter_layers
            )
        else:
            assignments = horizontal_assignments
        
        # 步骤4：为每个战机生成高度指令
        commands = {}
        for fid, assigned_tids in assignments.items():
            if not assigned_tids:
                continue
            
            current_alt = fighter_altitudes.get(fid, 9.0)
            
            # 计算目标高度（分配目标的平均高度）
            target_alts = [target_positions_3d[tid][2] for tid in assigned_tids 
                          if tid in target_positions_3d]
            if not target_alts:
                continue
            
            target_alt = np.mean(target_alts)
            
            # 判断能量优势
            energy_advantage = current_alt > target_alt
            
            # 计算爬升率
            alt_diff = target_alt - current_alt
            if abs(alt_diff) < 0.5:  # 高度差<500m，保持当前高度
                climb_rate = 0.0
            elif alt_diff > 0:  # 需要爬升
                climb_rate = min(self.CRUISE_CLIMB_RATE, abs(alt_diff) * 10)
            else:  # 需要俯冲
                climb_rate = max(self.MAX_DIVE_RATE, -abs(alt_diff) * 10)
            
            commands[fid] = AltitudeCommand(
                fighter_id=fid,
                current_altitude=current_alt,
                target_altitude=target_alt,
                climb_rate=climb_rate,
                energy_advantage=energy_advantage,
                assigned_targets=assigned_tids
            )
        
        self._last_commands = commands
        return commands
    
    def _classify_targets_by_altitude(
        self, 
        target_positions_3d: Dict[str, Tuple[float, float, float]]
    ) -> Dict[str, AltitudeLayer]:
        """目标高度分层"""
        layers = {}
        for tid, (x, y, z) in target_positions_3d.items():
            if z < self.LOW_CEILING:
                layers[tid] = AltitudeLayer.LOW
            elif z <= self.MID_CEILING:
                layers[tid] = AltitudeLayer.MID
            else:
                layers[tid] = AltitudeLayer.HIGH
        return layers
    
    def _classify_fighters_by_altitude(
        self, 
        fighter_altitudes: Dict[str, float]
    ) -> Dict[str, AltitudeLayer]:
        """战机高度分类"""
        layers = {}
        for fid, alt in fighter_altitudes.items():
            if alt < self.LOW_CEILING:
                layers[fid] = AltitudeLayer.LOW
            elif alt <= self.MID_CEILING:
                layers[fid] = AltitudeLayer.MID
            else:
                layers[fid] = AltitudeLayer.HIGH
        return layers
    
    def _allocate_targets_3d(
        self,
        target_positions_3d: Dict[str, Tuple[float, float, float]],
        fighter_positions_2d: Dict[str, Tuple[float, float]],
        fighter_altitudes: Dict[str, float],
        target_layers: Dict[str, AltitudeLayer],
        fighter_layers: Dict[str, AltitudeLayer]
    ) -> Dict[str, List[str]]:
        """3D目标分配算法
        
        策略：
        1. 优先同层分配
        2. 高打低有优势（能量优势）
        3. 低打高劣势（需要爬升）
        4. 综合考虑水平距离和高度差
        """
        assignments = {fid: [] for fid in fighter_positions_2d}
        
        # 为每个目标找最佳战机
        for tid, (tx, ty, tz) in target_positions_3d.items():
            min_cost = float('inf')
            best_fighter = None
            
            for fid in fighter_positions_2d:
                fx, fy = fighter_positions_2d[fid]
                fz = fighter_altitudes[fid]
                
                # 水平距离
                h_dist = np.sqrt((tx - fx)**2 + (ty - fy)**2)
                
                # 高度差
                v_dist = abs(tz - fz)
                
                # 能量管理：高打低有优势
                if fz > tz:
                    v_penalty = v_dist * self.ENERGY_ADVANTAGE_FACTOR
                else:
                    v_penalty = v_dist * self.ENERGY_DISADVANTAGE_FACTOR
                
                # 综合代价
                cost = h_dist + v_penalty
                
                if cost < min_cost:
                    min_cost = cost
                    best_fighter = fid
            
            if best_fighter:
                assignments[best_fighter].append(tid)
        
        return assignments
    
    def get_command(self, fighter_id: str) -> Optional[AltitudeCommand]:
        """获取指定战机的高度指令"""
        return self._last_commands.get(fighter_id)
    
    def get_all_commands(self) -> Dict[str, AltitudeCommand]:
        """获取所有高度指令"""
        return self._last_commands.copy()
